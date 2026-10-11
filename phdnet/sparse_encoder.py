"""M1 稀疏编码器 —— 对应初级感觉皮层的稀疏放电与抑制性侧抑制（k-WTA 竞争）。

架构对应（文档 §2 M1）：
    u = W_enc·x + b          全层电位（感受野投影）
    s_i = norm(u_i) 若 i ∈ top-k，否则 0   —— 胜者全取得到 SDR 稀疏分布式表示

P61（fhz 2026-09-29「迭代默认 fp32，模型默认 bf16」）：**迭代量全程 fp32**
（u / top-k 归一 / 学习增量），模型权重可按 dtype 存储。

⚠ 诚实边界（numpy/BLAS 路径的物理限制）：**bf16/fp16 存储 + fp32 迭代必然
每次付一次上采样转换**（W 读 8.4 MB + 写 16.8 MB + 再读 16.8 MB）——比 fp32
直接 GEMV 更慢。因此 M1 默认 `fp32`（访存真减半、零转换、舍入 ~1e-7）；
需要严格 bf16 语义时显式传 `dtype="bf16"`（走 ml_dtypes，代价是上面的转换，
仅在 torch/NPU 路径上才有意义——读出侧 bf16 已由 `--readout-dtype` 默认启用）。
"""

import os
import platform as _platform

import numpy as np

try:
    from numba import njit, prange
    _NUMBA_ENC = True
except ImportError:                                     # pragma: no cover
    _NUMBA_ENC = False

    def njit(*a, **kw):
        def wrap(fn):
            return fn
        return wrap if not a or not callable(a[0]) else a[0]

    prange = range

try:                                        # bf16 支持（numpy 扩展）
    import ml_dtypes
    _BF16 = np.dtype(ml_dtypes.bfloat16)
except Exception:                           # pragma: no cover
    _BF16 = None

_MODEL_DTYPES = {
    "bf16": _BF16,
    "fp16": np.dtype(np.float16),
    "fp32": np.dtype(np.float32),
    "fp64": np.dtype(np.float64),
}


def resolve_model_dtype(name: str) -> np.dtype:
    """模型存储 dtype → numpy dtype（bf16 不可用时回落 fp32 并说明）。"""
    dt = _MODEL_DTYPES.get(str(name).lower())
    if dt is None:                          # bf16 缺 ml_dtypes
        if str(name).lower() == "bf16":
            return _MODEL_DTYPES["fp32"]
        raise ValueError(f"未知模型 dtype: {name!r}；可用 {sorted(_MODEL_DTYPES)}")
    return dt


@njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _gemv_rows(W, x, b, out):
    """out[r] = W[r,:]·x + b[r] —— 自写 GEMV 替代 BLAS。

    动机（P77）：服务器 12:50/13:40 日志 M1_encode = 57-58 ms/tok（本机 x86
    同规模 0.5-1 ms）——**aarch64 上 numpy 对该形状的 GEMV 走了病态路径**
    （fp32/fp64 都慢，与 dtype 无关）。自写核行间 prange、行内顺序累加，
    跨平台性能可控。**逐位说明**：行内按列序累加，与 BLAS 的归约顺序在
    数学上同为 Σ，但浮点结合顺序可能差 1-2 ulp（与 P52 融合核同级）。
    """
    n = W.shape[0]
    for r in prange(n):
        acc = 0.0
        for c in range(x.shape[0]):
            acc += W[r, c] * x[c]
        out[r] = acc + b[r]


@njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _gemv_rows_ilp(W, x, b, out):
    """**P165**：`_gemv_rows` + 行内 4 路 ILP 累加器。

    为什么加 ILP：原核行内是**单条串行依赖链**（`acc += ...`），CPU 的
    乘加单元每个时钟只能等上一次加法完成 → 实测只有 10.5 GB/s，而
    x86 的 BLAS（同规模）能到 56.9 GB/s。拆成 4 个独立累加器后 4路并行，
    打破依赖链 → 实测 **13.1 GB/s（1.25×，fp32）**。

    ⚠ **不是逐位等价**：`acc` 的结合顺序变了（先分 4 段再合并）。
    实测 fp32 **relerr = 0.0**（因为 4 路的分段恰好对齐且加法结合满足
    交换），fp64 **relerr = 2.6e-15**（约 1 ulp，与 P52 融合核同级）。
    → 故这是**数值路径**，不是 bit-exact 路径。若需要逐位，见 `_gemv_rows`。

    为什么是 4 路不是 8 路：实测 8 路反而慢（fp32 642 vs 611 µs，
    fp64 776 vs 650 µs）—— 4 路已能填满发射端口，再多只增寄存器压力。
    """
    n = W.shape[0]
    n4 = x.shape[0] // 4
    for r in prange(n):
        a0 = 0.0
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        for c in range(n4):
            a0 += W[r, c] * x[c]
            a1 += W[r, n4 + c] * x[n4 + c]
            a2 += W[r, 2 * n4 + c] * x[2 * n4 + c]
            a3 += W[r, 3 * n4 + c] * x[3 * n4 + c]
        # 余数列（x.shape[0] % 4 != 0）：4 路分段只覆盖前 4*n4 列，**必须补算**，
        # 否则尾部列被静默丢弃（实测 n=6 max|Δ|=4.62、n=7 max|Δ|=15.3、
        # n=5 max|Δ|=6.33 —— 训练照跑、只是结果错）。n%4==0 时本循环不执行，
        # **既有路径逐位不变**。生产 n_input=2*n_sdr 恰为 4 倍数才一直没暴露。
        for c in range(4 * n4, x.shape[0]):
            a3 += W[r, c] * x[c]
        s01 = a0 + a1
        s23 = a2 + a3
        out[r] = (s01 + s23) + b[r]


# ══════════════════════════════════════════════════════════════════════════
# **P165：GEMV 路径改「运行时探测」，不再用平台名硬编码**
# ══════════════════════════════════════════════════════════════════════════
# 原判据（P77）：`_platform.machine().startswith("aarch64")` → aarch64 用
# numba、x86 用 BLAS。**这是用「平台名」代理「BLAS 快慢」** —— 两个缺陷：
#   ① aarch64 上 BLAS 慢是**那次构建的 OpenBLAS 问题**，不是架构的必然；
#      新版 OpenBLAS/Numpy 可能已修 → 判据过时且无法感知。
#   ② x86 上 BLAS 快也不是必然 —— 换 CPU/换 BLAS 实现可能翻转。
# P165 实测（本机 x86，1b 档 1024×2048，best-of-9）：
#   | 实现 | fp32 | fp64 |
#   |---|---|---|
#   | numba prange（当前 aarch64 路径）| 762 µs / 10.5 GB/s | 714 µs |
#   | numba prange + 4路 ILP（P165 新） | **611 µs / 13.1 GB/s** | 650 µs |
#   | numpy BLAS | **141 µs / 56.9 GB/s** | 563 µs |
#   → **BLAS 在 fp32 下快 5.4×**（AVX 达到 57 GB/s ≈ 饱和），而 fp64 只快 1.27×。
# → 正确做法：**进程级实测一次两者耗时**，谁快用谁。加缓存避免每步探测。
_BLAS_GEMV_OK: dict = {}


_GEMV_ILP = bool(int(os.environ.get("PHD_GEMV_ILP", "1")))


def _prefer_blas_gemv(dtype: np.dtype) -> bool:
    """返回 True 表示「本进程实测 BLAS 的该 dtype GEMV 更快」。

    只测一次（`_BLAS_GEMV_OK` 缓存）。探测用 64×128 的小矩阵，
    形状比真实小很多 → 结论**可能不适用于真实形状**，故刻意用
    「同 dtype、同 row-major 结构」的最小尺寸；宁可误判为「用 BLAS」
    （BLAS 在fp32 明显更好），也不要误判为「用 numba」。
    """
    key = str(dtype)
    if key in _BLAS_GEMV_OK:
        return _BLAS_GEMV_OK[key]
    _BLAS_GEMV_OK[key] = True          # 默认走 BLAS（fp32 实测快 5.4×）
    if not _NUMBA_ENC:
        return True
    try:
        import time as _t
        n_r, n_c = 64, 128
        W = np.zeros((n_r, n_c), dtype=dtype)
        x = np.zeros(n_c, dtype=dtype)
        b = np.zeros(n_r, dtype=dtype)
        out = np.empty(n_r, dtype=dtype)
        # 预热（含 JIT，若走 numba）
        W @ x
        _gemv_rows(W, x, b, out)
        t_blas = 1e9
        t_nb = 1e9
        for _ in range(3):
            t0 = _t.perf_counter()
            for _ in range(10):
                np.dot(W, x, out=out)
            t_blas = min(t_blas, (_t.perf_counter() - t0) / 10)
            t0 = _t.perf_counter()
            for _ in range(10):
                _gemv_rows(W, x, b, out)
            t_nb = min(t_nb, (_t.perf_counter() - t0) / 10)
        _BLAS_GEMV_OK[key] = bool(t_blas <= t_nb)
    except Exception:                                    # noqa: BLE001
        _BLAS_GEMV_OK[key] = True       # 探测失败 → 走 BLAS（生产默认更快）
    return _BLAS_GEMV_OK[key]


class SparseEncoder:
    def __init__(self, n_input: int, n_sdr: int, k: int, rng: np.random.Generator,
                 dtype: str = "fp32"):
        self.n_input, self.n_sdr, self.k = n_input, n_sdr, k
        # 固定随机投影（生产中可由 Oja 规则慢调，这里保持冻结以聚焦高层学习）
        # P61：默认 fp32（访存比旧 fp64 减半，且无上采样转换开销——bf16 存储
        # 在 numpy 路径反而更慢，见模块 docstring）。**迭代量恒为 fp32**。
        self.dtype = resolve_model_dtype(dtype)
        self._w32 = None                      # 仅低精度存储时的 fp32 工作副本
        self.W = (rng.normal(0.0, 1.0 / np.sqrt(n_input), size=(n_sdr, n_input))
                  .astype(self.dtype))
        self.b = rng.uniform(0.0, 0.1, size=n_sdr)          # 偏置保持 fp32（小）
        # T2.2 稳态目标：各行权重的初始 L2 范数（Foldiak 学习后拉回，防塌缩）
        self._hn = np.linalg.norm(np.asarray(self.W, dtype=np.float64), axis=1)

    def _w_fp32(self) -> np.ndarray:
        """fp32 权重视图：fp32/fp64 存储直接返回（零拷贝）；低精度则用缓存副本。"""
        if self.W.dtype in (np.dtype(np.float32), np.dtype(np.float64)):
            return self.W
        if self._w32 is None or self._w32.shape != self.W.shape:
            self._w32 = np.asarray(self.W, dtype=np.float32)
        return self._w32

    def encode(self, x: np.ndarray):
        """输入 x -> (s, idx)。s: SDR 稀疏向量；idx: 胜者索引（供稀疏快速通路）。"""
        # P61/P76：**输入 dtype 必须跟随权重 dtype**，否则混合 dtype（如
        # fp64 权重 @ fp32 输入——P75 把默认改回 fp64 后踩中）会让 numpy 脱离
        # BLAS 走逐元素慢路径：本机 x86 慢 5.9×，昇腾 aarch64 上 M1_encode
        # 0.85 → 62-67 ms/tok（约 70×）。输入只有 n_in 个元素，转换代价可忽略。
        W32 = self._w_fp32()
        x_cast = np.asarray(x, dtype=W32.dtype)
        # P77 原来是「平台名硬编码」：aarch64 → numba、x86 → BLAS。
        # ⚠⚠ **P165 改为运行时探测**（BLAS 快慢不是架构的必然属性，
        #   是那次构建的 BLAS 实现的问题；换 CPU/换实现会翻转）→ 实测。
        #   P165 本机 x86 实测：fp32 下 BLAS **快 5.4×**（141 vs 762 µs，
        #   56.9 vs 10.5 GB/s）→ x86 走 BLAS 正确，**且不依赖平台名**。
        use_numba = (_NUMBA_ENC
                     and not _prefer_blas_gemv(W32.dtype)
                     and W32.dtype in (np.float64, np.float32))
        if use_numba:
            # P77：某些平台上 numpy 对该形状 GEMV 病态慢（57-58 ms/tok），
            # 自写 numba 核跨平台一致（行内顺序累加，容差 1-2 ulp）
            # ⚠ P107b 曾把 W/x/b 转 fp16 进核——**aarch64 numba 不支持
            # float16 数组**（NotImplementedError: float16，数据模型缺失），
            # 已回滚为 fp32 进核。fp16 收益改由**存储/检查点侧**拿（P84）。
            # P165：改用 ILP 核（行内 4 路累加器，比单链快 1.25×）。
            # ⚠ **注意 dtype 混用会让 BLAS 掉出快路径**（P75/MEMORY）：
            #   `_gemv_rows_ilp` 内部acc 是 float64 累加，**必须在 W.dtype 上
            #   原样算**，不能 fp32 W 配 fp64 累加器（那是混合 dtype）。
            #   故这里按 W32.dtype 分派，不再强制转 fp64。
            _kern = _gemv_rows_ilp if _GEMV_ILP else _gemv_rows
            u = np.empty(self.n_sdr, dtype=W32.dtype)
            _kern(W32, x_cast, self.b.astype(W32.dtype), u)
        else:
            # P188：偏置 b 须跟随 W.dtype——BLAS/numba 两路原本就不一致：
            # numba 路径有 `self.b.astype(W32.dtype)`，BLAS 路径漏了 →
            # fp64 的 b 把 u 拉回 float64，s0 连带 float64 一路漏到
            # Rust 校验层（m2_add_outer 炸「v1 必须是 fp32」），
            # 且 M1 输出走 fp64 违背 P107 编码器 fp32 定案。
            u = W32 @ x_cast + self.b.astype(W32.dtype)
        idx = np.argpartition(-u, self.k - 1)[: self.k]      # k-WTA 竞争（侧抑制的抽象）
        s = np.zeros_like(u)
        win = u[idx]
        s[idx] = (win - win.min()) / (win.max() - win.min() + 1e-9) + 0.1  # 归一到 [0.1, 1.1]
        return s, np.sort(idx)

    def learn(self, x: np.ndarray, s: np.ndarray, eta: float) -> None:
        """T2.2 可学习稀疏词典（Foldiak 局部规则，默认关闭）。

        仅活跃（胜者）行更新：ΔW_i = η·s_i·(x − s_i·W_i) —— Hebb 项学习
        输入相关感受野，逐行 Oja 归一化项防塌缩；随后把每行范数拉回初始值
        （稳态双约束之二）。k-WTA 竞争（encode 内）即 Foldiak 的侧抑制分权，
        与稳态缩放共同构成路线图要求的「k-WTA + 稳态双约束」。

        局部性：只有 s_i>0 的行被触碰，成本 O(k·n_input)；无全局非局部梯度。
        """
        act = np.nonzero(s > 0.0)[0]
        if act.size == 0:
            return
        W = self._w_fp32()                                 # 迭代 fp32（P61）
        u = W @ x                                          # 当前全层电位（感受野投影）
        sa = s[act]
        W[act] += eta * (sa[:, None] * (x[None, :] - sa[:, None] * W[act]))
        # 稳态突触缩放：行范数拉回初始值（保留相对结构，防 Oja/Hebb 漂移塌缩）
        nrm = np.maximum(np.linalg.norm(W[act], axis=1), 1e-12)
        W[act] *= (self._hn[act] / nrm)[:, None]
        if W is not self.W:                                # 低精度存储：写回
            self.W[act] = W[act].astype(self.W.dtype)
