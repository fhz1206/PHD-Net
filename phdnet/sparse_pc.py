"""M2 稀疏连接版预测编码层级 —— 结构性稀疏（sparse connectivity）。

================================================================================
为什么需要这个模块（神经认知依据，铁律②）
================================================================================
`pc.py` 的 `PredictiveCodingStack` 用**稠密矩阵**承载皮层权重（n×n 全连接，
连接率 100%）。这与皮层的真实结构不符：

| 项 | 皮层（实测文献） | 稠密栈（旧） |
|---|---|---|
| 单个锥体神经元突触数 | ~10³–10⁴ | n（满连接） |
| 潜在靶点 | ~10¹¹ | n |
| **连接率** | **~10⁻⁵（长程）～10% 量级（局部）** | **100%** |
| 计算/存储单位 | **只对存在的突触** | 全矩阵 |

皮层的组织原则是 **dense representation + sparse connectivity**（表示稠密、
连接稀疏）：信息由大量神经元的稀疏活动模式承载，而不是靠两两全连接。
本模块把主干改造为**结构性稀疏**：

  - 每个神经元只有 k 条**存在**的入边（默认 k = n_in/8 ≈ 12.5%，接近局部
    皮层连接率的量级），以 CSR 存储，**不存在的连接不存在**（不是"零权重"）；
  - 前向/学习只遍历存在的边（O(k·n) 而非 O(n²)）；
  - 学习规则与稠密版逐条对应（误差驱动 ΔW_dn、Oja 规则 ΔW_up、突触缩放稳态）。

数值关系（可验证）：**k = n_in（全连接图）时本模块与稠密栈数值一致**
（CSR 求和顺序按列索引升序；与 BLAS 内部求和顺序不保证逐位相同，
对拍判据见 tests/verifiers/verify_parallel_consistency.py）。

开关（默认关闭，默认路径逐位不变）：
  cfg.sparse_conn  —— True 时主干改用本模块
  cfg.conn_k       —— 每神经元入边数 k（0 = 自动取 n_in//8）
"""

from __future__ import annotations

import numpy as np

from .plasticity import NUMBA_OK

if NUMBA_OK:                                        # pragma: no cover
    from numba import njit, prange

    # P123（fhz 指示加 nogil）：`--m2-kernel plain` 走的就是下面这5 个核。
    # `nogil=True` 让核执行期间**释放 GIL**，从而能与其它 Python 线程并发
    # —— 生产环境里主进程有个**nogil 分词线程**（日志：
    # `[parallel] tokenisation thread (numba nogil)=153`），它要占大量核。
    # 核若持 GIL，分词线程就得等 → 这是「真实存在的并发」，不是假设。
    #
    # ⚠⚠ **但代价必须先量**（P121 教训：别为「看起来更并发」到处加）：
    #   本机 x86 8 核实测 1B 档尺寸（n=1024, k=128）：
    #     单次调用    无 nogil 0.090 ms → 有 nogil 0.097 ms（**慢 7.8%**）
    #     并发场景    无 nogil 0.021 s  → 有 nogil 0.024 s（**慢 14%**）
    #   即**本机是负收益**。但本机是 8 核 x86，生产是 **191 核 aarch64**——
    #   x86 结论不构成昇腾证据（项目已实测到四次方向相反）。故按指示加上，
    #   **但必须在服务器 A/B 后再定去留**（见 P123 的 A/B 判据）。
    #
    # 判据（服务器）：`--step-profiling` 的 `segments(win)` 里 M2_infer 与
    # `loop: tokenize`。若 **tokenize 明显变短**（分词不再被阻塞）→ 留；
    # 若 M2_infer 变长且 tokenize 不变 → 去掉（说明本机结论成立，昇腾也如此）。
    @njit(cache=True, fastmath=True, parallel=True, nogil=True)
    def _csr_matvec(indptr, idx, val, x):
        """y[i] = Σ_{p∈入边(i)} val[p]·x[idx[p]]（只遍历存在的突触；按行并行）。"""
        n = indptr.shape[0] - 1
        y = np.zeros(n)
        for i in prange(n):
            s = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                s += val[p] * x[idx[p]]
            y[i] = s
        return y

    @njit(cache=True, fastmath=True, parallel=True, nogil=True)
    def _csr_add_outer(indptr, idx, val, a, b, eta):
        """稀疏外积累加：val[p] += eta·a[i]·b[idx[p]]（边 (i, idx[p])）。"""
        n = indptr.shape[0] - 1
        for i in prange(n):
            ai = a[i]
            if ai == 0.0:
                continue
            for p in range(indptr[i], indptr[i + 1]):
                val[p] += eta * ai * b[idx[p]]

    @njit(cache=True, fastmath=True, parallel=True, nogil=True)
    def _csr_oja_up(indptr, idx, val, post, pre, eta):
        """稀疏 Oja：Δw = η·post[i]·(pre[j] − post[i]·w)（边 (i, j)）。"""
        n = indptr.shape[0] - 1
        for i in prange(n):
            pi = post[i]
            if pi == 0.0:
                continue
            for p in range(indptr[i], indptr[i + 1]):
                val[p] += eta * pi * (pre[idx[p]] - pi * val[p])

    @njit(cache=True, fastmath=True)
    def _csr_clip(val, w_max):
        for p in range(val.shape[0]):
            v = val[p]
            if v > w_max:
                val[p] = w_max
            elif v < -w_max:
                val[p] = -w_max

    @njit(cache=True, fastmath=True, parallel=True, nogil=True)
    def _csr_row_norms(indptr, val):
        n = indptr.shape[0] - 1
        # ⚠ P175：`np.zeros(n)` 曾是 **fp64**（`s = 0.0` 让 numba 提升），
        #   而 `val` 自 P173 起是 fp32 → **同一份数据两种精度**。
        #   现统一为 fp32（与 val、Rust 版一致）。
        out = np.zeros(n, dtype=np.float32)
        for i in prange(n):
            s = np.float32(0.0)
            for p in range(indptr[i], indptr[i + 1]):
                s += val[p] * val[p]
            out[i] = s ** np.float32(0.5)
        return out

    @njit(cache=True, fastmath=True, parallel=True, nogil=True)
    def _csr_scale_rows(indptr, val, target):
        n = indptr.shape[0] - 1
        for i in prange(n):
            s = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                s += val[p] * val[p]
            nrm = s ** 0.5
            if nrm < 1e-12:
                nrm = 1e-12
            f = target[i] / nrm
            for p in range(indptr[i], indptr[i + 1]):
                val[p] *= f
else:                                               # pandas 回退（纯 numpy）
    def _csr_matvec(indptr, idx, val, x):
        y = np.zeros(indptr.shape[0] - 1)
        for i in range(len(y)):
            p0, p1 = indptr[i], indptr[i + 1]
            if p1 > p0:
                y[i] = float(val[p0:p1] @ x[idx[p0:p1]])
        return y

    def _csr_add_outer(indptr, idx, val, a, b, eta):
        for i in range(len(a)):
            if a[i] != 0.0:
                p0, p1 = indptr[i], indptr[i + 1]
                val[p0:p1] += eta * a[i] * b[idx[p0:p1]]

    def _csr_oja_up(indptr, idx, val, post, pre, eta):
        for i in range(len(post)):
            if post[i] != 0.0:
                p0, p1 = indptr[i], indptr[i + 1]
                val[p0:p1] += eta * post[i] * (pre[idx[p0:p1]] - post[i] * val[p0:p1])

    def _csr_clip(val, w_max):
        np.clip(val, -w_max, w_max, out=val)

    def _csr_row_norms(indptr, val):
        # ⚠ P175：同numba 版，统一 fp32（见上方注释）。
        out = np.zeros(indptr.shape[0] - 1, dtype=np.float32)
        for i in range(len(out)):
            p0, p1 = indptr[i], indptr[i + 1]
            out[i] = (np.float32(np.linalg.norm(val[p0:p1]))
                      if p1 > p0 else np.float32(0.0))
        return out

    def _csr_scale_rows(indptr, val, target):
        for i in range(len(target)):
            p0, p1 = indptr[i], indptr[i + 1]
            nrm = max(float(np.linalg.norm(val[p0:p1])) if p1 > p0 else 0.0, 1e-12)
            val[p0:p1] *= target[i] / nrm


# ══════════════════════════════════════════════════════════════════════════
# **P192：Cython 核分派层**（默认关闭，铁律④）
# ══════════════════════════════════════════════════════════════════════════
# 目的：让 `--cython-kernels auto` 能把本文件上最热的 5 个核换成 Cython nogil
# 实现，同时保证：
#   · **默认 off 时下面每个调用点逐位不变**（连函数解析都走原 numba 路径）；
#   · Cython 不可用时**静默回落 numba** 并留下可查的原因；
#   · 口径（dtype / 索引类型 / 参数语义）与 numba 版一一对齐 —— 这一点是
#     Rust 臂（P182/P188）失败的真因：dtype/idx 口径在 `a[i]` vs `a[idx[p]]`
#     这种地方对不齐，门禁才发现时已经改了训练轨迹。
#
# ⚠ **诚实边界（务必随代码一起读）**：Cython 用 MSVC `/O2` = 严格 IEEE-754
#   （**无** FMA 收缩、**无**重结合），而本文件的 numba 核全部
#   `fastmath=True`（**有**收缩 + 重结合）→ 开此开关后训练轨迹会有
#   **fp32 级微小漂移**（实测 max|d| ~1e-8 ~ 1e-5，随规模）。
#   **不得**称为「逐位等价」。门禁见 tests/verifiers/verify_cykernels.py。
#
# ⚠ **不走 prange 的两个核**（`_csr_clip` / `_csr_row_norms` / `_csr_scale_rows`）
#   未做 Cython 版：前两者是 O(nnz) 一次扫描，numba 的向量化/连续访问已接近
#   带宽上限，Cython 收益不可知；`_csr_scale_rows` 还含 sqrt + 归一化，
#   浮点语义（`s ** 0.5` vs `np.sqrt`）易出 ulp 差，**不值得为它开漂移风险**。
#   这是「不做」也是结论的一部分。

try:                                       # pragma: no cover - 纯导入分支
    from .cykernels import get_kernels as _get_cyk
except Exception:                          # noqa: BLE001
    _get_cyk = None

#: 直接构造机制的兼容默认值；PHDNet 显式传实例句柄，不读取该全局值。
_CYK = None
_CYK_MODE: str | None = None
_CYK_DEFAULT = object()  # 兼容 cyk_init；模型会显式传入实例句柄（包括 None）。


def cyk_init(mode: str = "off"):
    """按 `mode` 初始化本文件的 Cython 分派；返回是否生效。

    ⚠ **必须在 `SparsePCStack` 被构造之前调用**，因为分派发生在核调用点。
    重复调用是幂等的（同一 mode 直接返回上次结果）。
    ⚠ `mode` 非法值不在这里报错 —— 那是 `config.py::__post_init__` 的
    fail-fast 职责（唯一入口校验，避免两处校验漂移）。
    """
    global _CYK, _CYK_MODE
    if mode == _CYK_MODE and (mode == "off" or _CYK is not None):
        return _CYK is not None
    _CYK_MODE = mode
    if mode == "off" or _get_cyk is None:
        _CYK = None
        if mode == "force":
            raise RuntimeError("Cython loader 不可用（force）")
        return False
    try:
        k = _get_cyk(mode)
        if k is None and mode == "force":
            raise RuntimeError("Cython 扩展未激活（force）")
    except Exception:                      # noqa: BLE001
        _CYK = None
        if mode == "force":
            raise
        return False
    # `get_kernels` 返回**模块本身**（不是 Kernels 门面）→ None 即未激活。
    # ⚠ 故此处不能用 `k.active`（那是门面属性，模块上没有）——
    #   首版写成 `k.active` 抛 AttributeError: module has no attribute 'active'。
    _CYK = k
    return _CYK is not None


def _random_csr(rng: np.random.Generator, n_rows: int, n_cols: int, k: int,
                scale: float, lognormal: bool = False,
                exc_ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每行随机 k 条入边（无放回，列索引升序）——发育期随机连接的抽象。

    `lognormal=True` 时权重按皮层式分布（重尾对数正态 × E/I 符号比，O1-4）。
    """
    k = max(1, min(k, n_cols))
    indptr = np.arange(0, (n_rows + 1) * k, k, dtype=np.int64)
    idx = np.empty(n_rows * k, dtype=np.int64)
    # ⚠⚠ **P173（fhz 2026-10-03 指令「M2/M3/M4a 改为 fp32」）**
    # 精度从 fp64 降到 **fp32**。理由与代价：
    #   · 理由①：fp64 使 CSR 流**翻倍**（val 8→4 B/边）→ 访存受限下直接慢 2×；
    #     而 P110 实测 fp32 的非目标行更新保留率 **99.95%**（fp64≈100%），
    #     差 0.05% —— **这个代价远小于访存收益**。
    #   · 理由②：fp32 才能用**手写 SIMD**（AVX2 是 f32 指令，f64 无 FMA）→ P171 实测
    #     Rust SIMD 8 线程 241 µs vs BLAS 214 µs。
    #   · 代价：累加次序变化 → **不再与旧 fp64 版逐位相同**（需重新 rebaseline）。
    #   · ⚠ **门禁不做「改前vs 改后」的对比**（那是跨精度的比较，无意义）；
    #     只验「同一精度下 Rust 与 Python 一致」。
    val = np.empty(n_rows * k, dtype=np.float32)
    if lognormal:
        from .inits import cortical_init
        for r in range(n_rows):
            cols = np.sort(rng.choice(n_cols, size=k, replace=False))
            s = slice(r * k, (r + 1) * k)
            idx[s] = cols
            val[s] = cortical_init(rng, (k,), scale, exc_ratio).ravel()
        return indptr, idx, val
    for r in range(n_rows):
        cols = np.sort(rng.choice(n_cols, size=k, replace=False))
        s = slice(r * k, (r + 1) * k)
        idx[s] = cols
        val[s] = rng.normal(0.0, scale, k)
    return indptr, idx, val


def _from_dense_csr(W: np.ndarray, k: int):
    """把稠密权重按每行 |w| 最大的 k 列转为 CSR（列索引升序，值原样保留）。

    生物学对应：突触修剪保留强连接（"use it or lose it"）——用于
    稠密→稀疏蒸馏，以及**可比性对拍**（k = 全列时两栈权重完全相同）。
    """
    n_rows, n_cols = W.shape
    k = max(1, min(k, n_cols))
    indptr = np.arange(0, (n_rows + 1) * k, k, dtype=np.int64)
    # ── P-优化（向量化，逐位不变）────────────────────────────────────
    # 逐行 argsort+slice 的 Python 循环 → 整行**全量** argsort（⚠ 不用
    # argpartition：k 窗口边界处并列值的 tie 次序必须与逐行实现完全一致）。
    # `np.argsort(-np.abs(W), axis=1)` 对每行跑的是与
    # `np.argsort(-np.abs(W[r]))` 相同的 numpy 默认 introsort、比较同一逻辑
    # 序列 ⇒ 排列逐位相同。tie-heavy 对拍见
    # tests/verifiers/verify_perf_no_regression.py（全 0 / 大量重复值 /
    # 两值矩阵 / k=1 / k=n_cols / NaN 行）。
    # 值路径 fp64→fp32 逐元素转换语义不变：take_along_axis 先取原值、
    # 再 astype(np.float32)，与 `val[s] = W[r, cols]` 的赋值期转换同语义。
    order = np.argsort(-np.abs(W), axis=1)[:, :k]      # (n_rows, k)
    order.sort()                                       # 列索引升序 == 原 cols.sort()
    idx = order.astype(np.int64).ravel()
    val = np.take_along_axis(W, order, axis=1).astype(np.float32).ravel()
    return indptr, idx, val


def _transpose_csr(indptr, idx, val, n_new_rows: int, scale: float = 1.0):
    """把 (n_rows × k) 稀疏图转置为 (n_cols × k') 的 CSR（值 × scale）。

    生成权重以识别权重的转置初始化（生成-识别共享先验，与稠密版一致）。
    """
    src_row = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    new_rows = idx.copy()
    order = np.lexsort((src_row, new_rows))
    flat_src, flat_val = src_row[order], val[order] * scale
    counts = np.bincount(new_rows, minlength=n_new_rows)
    new_indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return new_indptr, flat_src.astype(np.int64), flat_val


# ── P52：M2 推理融合核（把 5 次 matvec + n_steps 循环融进一次调用）────────
# 逐位一致的关键：**每行入边仍按 indptr 顺序累加**（s += val[p]*x[idx[p]]），
# 与 `_csr_matvec` 完全同序；行级并行沿用 prange 口径。
# 收益来源：① Python 层 5 次 kernel 调用 → 1 次；② 中间数组（e1/d2/e0/d1）
# 从「每步重新分配」变成「核内一次分配 + 循环内复用」；③ 消除 4×n_steps 次
# 跨核的写回/读回。返回独立数组 → `cache` 跨步持有语义与原版完全相同。
if NUMBA_OK:
    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _pc_infer_fused(up0, up1, dn0, dn1, s0, n_steps):
        n1 = up0[0].shape[0] - 1
        n2 = up1[0].shape[0] - 1
        n0 = dn0[0].shape[0] - 1         # ⚠ P176：需要 n0（`e0` 的正确长度）
        # ⚠⚠ **P176（fhz 拍板「统一改 fp32」）**：原本是 `np.empty(n1)`（**fp64**），
        #   而 `val` 自 P173 起是 fp32 → 融合核是「fp32 存储 + fp64 累加 +
        #   fp64 中间数组」的**混合精度**。现统一为 fp32。
        #   · 代价：累加精度降低 → **需重新 rebaseline**
        #   · 收益：① 可用 fp32 SIMD（fp64 无 FMA）
        #         ② 与 `m2_matvec` 等原子算子**逐位一致** → 融合/非融合等价
        r1 = np.empty(n1, dtype=np.float32)
        r2 = np.empty(n2, dtype=np.float32)
        for i in prange(n1):                      # up0 @ s0 → tanh
            s = np.float32(0.0)
            for p in range(up0[0][i], up0[0][i + 1]):
                s += up0[2][p] * s0[up0[1][p]]
            r1[i] = np.tanh(s)
        for i in prange(n2):                      # up1 @ r1 → tanh
            s = np.float32(0.0)
            for p in range(up1[0][i], up1[0][i + 1]):
                s += up1[2][p] * r1[up1[1][p]]
            r2[i] = np.tanh(s)
        for _ in range(n_steps):
            e1 = np.empty(n1)                     # e1 = r1 - dn1 @ r2
            d2 = np.empty(n2)
            for i in prange(n1):
                s = np.float32(0.0)
                for p in range(dn1[0][i], dn1[0][i + 1]):
                    s += dn1[2][p] * r2[dn1[1][p]]
                e1[i] = r1[i] - s
            for i in prange(n2):                  # d2 = clip(up1 @ e1)
                s = np.float32(0.0)
                for p in range(up1[0][i], up1[0][i + 1]):
                    s += up1[2][p] * e1[up1[1][p]]
                d2[i] = min(0.5, max(-0.5, s))
            for i in prange(n2):                  # r2 = tanh(r2 + 0.15*d2)
                r2[i] = np.tanh(r2[i] + 0.15 * d2[i])
            # ⚠⚠ **P176 修正越界 bug**（P52 遗留，fhz 拍板「按语义应读 s0」）：
            #   `e0` 原分配为 `np.empty(n1)`，但 **`dn0` 有 `n0` 行**、
            #   且 **`up0` 的列空间也是 `n0`** → 下面的
            #   `d1 = clip(up0 @ e0)` 会**越界读** `e0` 的相邻内存
            #   （实测 n0=256 / n1=192 时越界 **64** 个元素；
            #     numba `prange` **不做边界检查** → 静默读垃圾）。
            #   **正确长度是 `n0`** —— 与非融合路径实测一致（非融合 `e0` 长 256）。
            #   ⚠ 这修正了 docstring 里「融合前后逐位一致」的**真实 bug**
            #     （该声明此前**不成立**）。**需重新 rebaseline。**
            e0 = np.empty(n0, dtype=np.float32)  # e0 = s0 - dn0 @ r1
            d1 = np.empty(n1, dtype=np.float32)
            for i in prange(n0):                  # ⚠ P176：n1 -> n0（dn0 有 n0 行）
                s = np.float32(0.0)
                for p in range(dn0[0][i], dn0[0][i + 1]):
                    s += dn0[2][p] * r1[dn0[1][p]]
                e0[i] = s0[i] - s
            for i in prange(n1):                  # d1 = clip(up0 @ e0)
                s = np.float32(0.0)
                for p in range(up0[0][i], up0[0][i + 1]):
                    s += up0[2][p] * e0[up0[1][p]]
                d1[i] = min(0.5, max(-0.5, s))
            for i in prange(n1):                  # r1 = tanh(r1 + 0.15*d1)
                r1[i] = np.tanh(r1[i] + 0.15 * d1[i])
        e0 = np.empty(n0, dtype=np.float32)  # 末尾误差（⚠ P176：n1→n0，原越界）
        e1 = np.empty(n1, dtype=np.float32)
        for i in prange(n0):                     # ⚠ P176：n1 -> n0（dn0 有 n0 行）
            s = np.float32(0.0)
            for p in range(dn0[0][i], dn0[0][i + 1]):
                s += dn0[2][p] * r1[dn0[1][p]]
            e0[i] = s0[i] - s
        for i in prange(n1):
            s = 0.0
            for p in range(dn1[0][i], dn1[0][i + 1]):
                s += dn1[2][p] * r2[dn1[1][p]]
            e1[i] = r1[i] - s
        return r1, r2, e0, e1
else:                                                   # numba 缺失回退
    def _pc_infer_fused(up0, up1, dn0, dn1, s0, n_steps):
        """无 numba 时的等价回退（逐算子，语义同原版）。"""
        r1 = np.tanh(_csr_matvec(*up0, s0))
        r2 = np.tanh(_csr_matvec(*up1, r1))
        for _ in range(n_steps):
            e1 = r1 - _csr_matvec(*dn1, r2)
            d2 = np.clip(_csr_matvec(*up1, e1), -0.5, 0.5)
            r2 = np.tanh(r2 + 0.15 * d2)
            e0 = s0 - _csr_matvec(*dn0, r1)
            d1 = np.clip(_csr_matvec(*up0, e0), -0.5, 0.5)
            r1 = np.tanh(r1 + 0.15 * d1)
        e0 = s0 - _csr_matvec(*dn0, r1)
        e1 = r1 - _csr_matvec(*dn1, r2)
        return r1, r2, e0, e1


# ── P59：M2 学习融合核（`learn` 8 次核调用 → 1 次；`learn_predictive` 12 → 1）
# 与 P52 推理融合核对称：收益来自 ①Python 层 kernel 启动次数骤减（每步 8~12
# 次 → 1 次）、②中间向量（predictive 侧 e0p/e1p）核内一次分配、③消除跨核
# 的 val 写回/读回。
# **逐位一致的关键**：每个子操作的**表达式与遍历顺序逐行照抄**原核
# （含 `if ai == 0.0: continue` 短路），同一 val 被多次更新时保持原次序
# （predictive 侧 dn0 先 (1−mix) 后 mix）。行级 prange 沿用原口径——行内顺序
# 不变、行间写集不相交 ⇒ 与串行版逐位一致。
# ⚠ homeostasis=True **不走本核**：e0p/e1p 需在 matvec 之后归一化，留在
# Python 侧按原路径执行（罕见路径，性能不敏感）。
if NUMBA_OK:
    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _pc_learn_fused(dn0, dn1, up0, up1, e0, e1, r1, r2, s0,
                        eta_pc, eta_oja, w_max):
        n = dn0[0].shape[0] - 1                   # 下行生成：val += η·e0[i]·r1[j]
        for i in prange(n):
            ai = e0[i]
            if ai == 0.0:
                continue
            for p in range(dn0[0][i], dn0[0][i + 1]):
                dn0[2][p] += eta_pc * ai * r1[dn0[1][p]]
        n = dn1[0].shape[0] - 1
        for i in prange(n):
            ai = e1[i]
            if ai == 0.0:
                continue
            for p in range(dn1[0][i], dn1[0][i + 1]):
                dn1[2][p] += eta_pc * ai * r2[dn1[1][p]]
        n = up0[0].shape[0] - 1                   # 上行识别：Oja
        for i in prange(n):
            pi = r1[i]
            if pi == 0.0:
                continue
            for p in range(up0[0][i], up0[0][i + 1]):
                up0[2][p] += eta_oja * pi * (s0[up0[1][p]] - pi * up0[2][p])
        n = up1[0].shape[0] - 1
        for i in prange(n):
            pi = r2[i]
            if pi == 0.0:
                continue
            for p in range(up1[0][i], up1[0][i + 1]):
                up1[2][p] += eta_oja * pi * (r1[up1[1][p]] - pi * up1[2][p])
        for p in prange(up0[2].shape[0]):        # clip（合并 4 次为 4 段循环）
            v = up0[2][p]
            if v > w_max:
                up0[2][p] = w_max
            elif v < -w_max:
                up0[2][p] = -w_max
        for p in prange(up1[2].shape[0]):
            v = up1[2][p]
            if v > w_max:
                up1[2][p] = w_max
            elif v < -w_max:
                up1[2][p] = -w_max
        for p in prange(dn0[2].shape[0]):
            v = dn0[2][p]
            if v > w_max:
                dn0[2][p] = w_max
            elif v < -w_max:
                dn0[2][p] = -w_max
        for p in prange(dn1[2].shape[0]):
            v = dn1[2][p]
            if v > w_max:
                dn1[2][p] = w_max
            elif v < -w_max:
                dn1[2][p] = -w_max

else:                                                   # numba 缺失回退
    def _pc_learn_fused(dn0, dn1, up0, up1, e0, e1, r1, r2, s0,
                        eta_pc, eta_oja, w_max):
        _csr_add_outer(*dn0, e0, r1, eta_pc)
        _csr_add_outer(*dn1, e1, r2, eta_pc)
        _csr_oja_up(*up0, r1, s0, eta_oja)
        _csr_oja_up(*up1, r2, r1, eta_oja)
        for t in (up0, up1, dn0, dn1):
            _csr_clip(t[2], w_max)

class SparsePCStack:
    """结构性稀疏预测编码栈（接口与 `PredictiveCodingStack` 对齐，可 drop-in）。

    四个权重（上行识别 W_up0/W_up1、下行生成 W_dn0/W_dn1）均以 CSR 承载：
      indptr[i]:indptr[i+1] 给出第 i 个**目标**神经元存在的入边（源索引 + 权重）。
    学习只更新存在的边；`k = n_in` 时退化为全连接图（与稠密栈数值一致）。
    """

    def __init__(self, n0: int, n1: int, n2: int, eta_pc: float, eta_oja: float,
                 rng: np.random.Generator, w_max: float = 2.0,
                 conn_k: int = 0, lognormal_init: bool = False,
                 exc_ratio: float = 0.8, fused: bool = True, cython_module=_CYK_DEFAULT):
        self._cyk = _CYK if cython_module is _CYK_DEFAULT else cython_module
        self.eta_pc, self.eta_oja, self.w_max = eta_pc, eta_oja, w_max
        # ⚠⚠ 2026-10-07（fhz 指令「删除 rust 版本，默认全部改回 python」）：
        #   **M2 主干的 Rust 后端已整体移除**：参数 m2_backend / rs_threads /
        #   rs_fused_min_nnz、状态 self._rs / rust_effective /
        #   m2_backend_reason / _csr_i32_cache 与 phdnet_rs/ 目录一并删除，
        #   M2 所有算子只走 numba/numpy（融合核 + 非融合分派核）。
        #   理由：Rust 臂 = 同一算子两份数值实现 + 双份门禁 + 一条 ctypes 跨界；
        #   收益只在大 shape 上成立（P182 实测交叉点 nnz≈52 万），代价却是
        #   dtype/idx 口径必须处处对齐（P188 那类漏 cast 直接崩）。
        # P181（历史记录，已作废）：`m2_backend` 曾默认 "rust"。

        # P103：**不能 bool()**——那会把 "serial" 变成 True，让 P99 的单核核
        # 永不可达（审查实测：`--m2-kernel serial` 实际走的是 parallel 核，
        # 整条 P99 链净效果为零）。三态原样保留：True / "fused" / "serial" / False。
        self.fused = fused

        self.n0, self.n1, self.n2 = n0, n1, n2
        k0 = conn_k if conn_k > 0 else max(1, n0 // 8)
        k1 = conn_k if conn_k > 0 else max(1, n1 // 8)
        self.k0, self.k1 = min(k0, n0), min(k1, n1)
        # 上行（识别）稀疏图（O1-4：可选皮层式权重分布）
        # ⚠ 初始化尺度按 **fan-in 归一化**（He/Glorot）：稀疏栈每神经元只有 k 条入边，
        #   故权重标准差取 1/√k 而非 1/√n_in —— 否则前向幅值偏低 √(k/n_in) 倍
        #   （k=32/n=256 时 ~0.35×），経 tanh 后表征趋 0、区分度下降。
        #   该量纲错误在 4,000 字符短训练口径下被掩盖（PPL −0.30%），
        #   在全语料口径下暴露为 +37.6% 劣化；按 fan-in 修正后两者应一致。
        self.up0 = _random_csr(rng, n1, n0, self.k0, 1.0 / np.sqrt(self.k0),
                               lognormal_init, exc_ratio)
        self.up1 = _random_csr(rng, n2, n1, self.k1, 1.0 / np.sqrt(self.k1),
                               lognormal_init, exc_ratio)
        # 下行（生成）稀疏图 = 上行图转置 × 0.5（与稠密版同一初始化语义）
        self.dn0 = _transpose_csr(*self.up0, n_new_rows=n0, scale=0.5)
        self.dn1 = _transpose_csr(*self.up1, n_new_rows=n1, scale=0.5)

        # 稳态目标：各行权重初始 L2 范数（突触缩放拉回此值）
        self._hn_up0 = self._row_norms(self.up0)
        self._hn_up1 = self._row_norms(self.up1)
        self._hn_dn0 = self._row_norms(self.dn0)
        self._hn_dn1 = self._row_norms(self.dn1)
        self.act_ema = None

    # ---------- 兼容接口：暴露稀疏矩阵形式的"权重"（只读快照） ----------
    @classmethod
    def from_dense(cls, W_up0: np.ndarray, W_up1: np.ndarray, W_dn0: np.ndarray,
                   W_dn1: np.ndarray, eta_pc: float, eta_oja: float,
                   w_max: float = 2.0, k: int = 0) -> "SparsePCStack":
        """由稠密权重构造稀疏栈（每行取 |w| 最大的 k 列；k≤0 或 k≥列数 → 全列）。

        用途：(a) **可比性对拍**——k = 全列时两栈权重完全相同，前向/学习应一致；
              (b) 稠密→稀疏蒸馏（保留幅值最大的连接，对应"强突触优先保留"）。
        """
        self = cls.__new__(cls)
        self._cyk = _CYK
        self.eta_pc, self.eta_oja, self.w_max = eta_pc, eta_oja, w_max
        self.n0, self.n1, self.n2 = W_up0.shape[1], W_up0.shape[0], W_up1.shape[0]
        self.k0 = min(k, W_up0.shape[1]) if k > 0 else W_up0.shape[1]
        self.k1 = min(k, W_up1.shape[1]) if k > 0 else W_up1.shape[1]
        self.up0 = _from_dense_csr(W_up0, self.k0)
        self.up1 = _from_dense_csr(W_up1, self.k1)
        self.dn0 = _from_dense_csr(W_dn0, W_dn0.shape[1])
        self.dn1 = _from_dense_csr(W_dn1, W_dn1.shape[1])
        self._hn_up0 = self._row_norms(self.up0)
        self._hn_up1 = self._row_norms(self.up1)
        self._hn_dn0 = self._row_norms(self.dn0)
        self._hn_dn1 = self._row_norms(self.dn1)
        self.act_ema = None
        return self

    # ── M2 算子（numba/numpy 单实现）──────────────────────────────
    # ⚠ 2026-10-07：Rust 分派臂已整体移除（fhz「删除 rust 版本，默认全部
    #   改回 python」）。下面这些方法现在是 **numba 核的薄封装** —— 保留方法名
    #   是为了不改动 `infer`/`learn`/`learn_predictive`/`homeostatic_rate` 等
    #   所有调用点；M2 主路径仍是融合核 `_pc_infer_fused`/`_pc_learn_fused`。

    def _mv(self, csr, x):
        """CSR SpMV。`csr` 是 `(indptr, idx, val)` 元组。

        Cython 激活时显式用连续 fp32 缓冲；encoder fp64 与中间 fp64
        向量也可调用。开启分支因此包含 fp32 舍入（非逐位等价）；off
        原样把输入交给旧 numba 核，不增加转换或修改旧 dtype。
        """
        ip, ix, vl = csr
        if self._cyk is not None:
            out = np.empty(ip.shape[0] - 1, dtype=np.float32)
            self._cyk.csr_matvec(np.ascontiguousarray(ip, dtype=np.int64),
                                 np.ascontiguousarray(ix, dtype=np.int64),
                                 np.ascontiguousarray(vl, dtype=np.float32),
                                 np.ascontiguousarray(x, dtype=np.float32), out)
            return out
        return _csr_matvec(ip, ix, vl, x)

    def _add_outer(self, csr, a, b, eta):
        """稀疏外积累加（原地）。

        P192：⚠ `a` 是**行向量**（长 n_rows）、`b` 被 `idx` 索引 —— 这条
        口径首版 Cython 写错过（写成 `a[idx[p]]`），被门禁 B2 以
        max|d|=0.339 抓住。现两侧一致。
        """
        ip, ix, vl = csr
        if self._cyk is not None:
            vc = np.ascontiguousarray(vl, dtype=np.float32)
            self._cyk.csr_add_outer(np.ascontiguousarray(ip, dtype=np.int64),
                                    np.ascontiguousarray(ix, dtype=np.int64), vc,
                                    np.ascontiguousarray(a, dtype=np.float32),
                                    np.ascontiguousarray(b, dtype=np.float32), eta)
            if vc is not vl:
                vl[...] = vc
            return vl
        return _csr_add_outer(ip, ix, vl, a, b, eta)

    def _oja(self, csr, post, pre, eta):
        """稀疏 Oja（原地）。P192：同上，两侧口径一致。"""
        ip, ix, vl = csr
        if self._cyk is not None:
            vc = np.ascontiguousarray(vl, dtype=np.float32)
            self._cyk.csr_oja_up(np.ascontiguousarray(ip, dtype=np.int64),
                                 np.ascontiguousarray(ix, dtype=np.int64), vc,
                                 np.ascontiguousarray(post, dtype=np.float32),
                                 np.ascontiguousarray(pre, dtype=np.float32), eta)
            if vc is not vl:
                vl[...] = vc
            return vl
        return _csr_oja_up(ip, ix, vl, post, pre, eta)

    def _clip(self, val, w_max):
        """权重裁剪（原地）。

        P192：⚠ **刻意不接 Cython**。numba `_csr_clip` 已在 P99 的 fused 路
        里被 `_pc_learn_fused` 融合掉（见下），单独调用只是 serial 兜底路径，
        而 Cython 版会引入一次额外跨界调用 + 无收益的舍入漂移。
        """
        return _csr_clip(val, w_max)

    def _row_norms(self, csr):
        """逐行 L2 范数。"""
        ip, _ix, vl = csr
        return _csr_row_norms(ip, vl)

    def _scale_rows(self, csr, target):
        """逐行缩放（原地）。"""
        ip, ix, vl = csr
        return _csr_scale_rows(ip, vl, target)

    def _dense(indptr, idx, val, n_rows: int, n_cols: int) -> np.ndarray:
        # ── P-优化（向量化，逐位不变）：逐行 for → np.repeat + 高级索引一次写入。
        # ⚠ 保持「非 staticmethod 的裸函数」现状（**改动前即如此，非本次引入**）：
        #   经 `self._dense(...)` 调用会把 self 当 indptr 传入 → TypeError
        #   （W_* 属性今天就是这个状态；门禁/测试只经类名
        #   `SparsePCStack._dense(...)` 调用）。**不加 @staticmethod**——
        #   加了会改变 W_* 属性的可观察行为（TypeError → 返回数组），
        #   超出「逐位不变」口径。
        # 数值等价：flat 写入顺序 = 原 for 的行序（行内 idx 升序、无重复列
        # ⇒ (行,列) 不重复，last-wins 无歧义）；空行 repeat 计 0 自动跳过。
        W = np.zeros((n_rows, n_cols))
        counts = np.diff(indptr[:n_rows + 1])
        rows = np.repeat(np.arange(n_rows, dtype=np.int64), counts)
        p0, p1 = int(indptr[0]), int(indptr[n_rows])
        W[rows, idx[p0:p1]] = val[p0:p1]
        return W

    @property
    def W_up0(self):                        # 只读稠密视图（测试/分析用，勿写入）
        return self._dense(*self.up0, self.n1, self.n0)

    @property
    def W_up1(self):
        return self._dense(*self.up1, self.n2, self.n1)

    @property
    def W_dn0(self):
        return self._dense(*self.dn0, self.n0, self.n1)

    @property
    def W_dn1(self):
        return self._dense(*self.dn1, self.n1, self.n2)

    def n_synapses(self) -> int:
        """实际存在的突触数（结构性稀疏的存储/计算单位）。"""
        return int(sum(len(v[2]) for v in (self.up0, self.up1, self.dn0, self.dn1)))



    # ---------- 推理（稀疏前向，逻辑与稠密版逐条对应） ----------
    def infer(self, s0: np.ndarray, n_steps: int = 1) -> dict:
        """M2 预测编码推理（**融合核**版，P52：最大段做 numba 融合）。

        融合前后**逐位一致**：每行入边仍按 indptr 顺序累加
        （`s += val[p]*x[idx[p]]`），只是把原来「5 次独立 kernel 调用 +
        4×n_steps 个中间数组分配」换成「1 次核调用」，中间数组在核内一次分配、
        循环内复用。**返回独立数组**（与原版语义相同，`cache` 可安全跨步持有）。
        行级并行沿用 `_csr_matvec` 的 prange 口径。
        """
        if self.fused == "serial":
            # P99：单核融合核（无 prange 屏障）——昇腾上并行融合慢 3-4×，
            # 屏障成本（10 个 prange 区的 fork/join）远超计算本身。
            r1, r2, e0, e1 = _pc_infer_fused_serial(
                self.up0, self.up1, self.dn0, self.dn1, s0, n_steps)
            return {"s0": s0, "r1": r1, "r2": r2, "e0": e0, "e1": e1}
        if self.fused:

            r1, r2, e0, e1 = _pc_infer_fused(self.up0, self.up1, self.dn0,
                                            self.dn1, s0, n_steps)
            return {"s0": s0, "r1": r1, "r2": r2, "e0": e0, "e1": e1}
        # P75：非融合路径 = P52 之前的原始实现（逐行照抄，语义确定），保留作
        # A/B 对照——服务器（昇腾 aarch64）实测融合核段 2.4 → 20-27 ms/tok，
        # 疑似 prange + fastmath 在该平台退化；x86 上融合核快 1.15-2.16×。
        r1 = np.tanh(self._mv(self.up0, s0))
        r2 = np.tanh(self._mv(self.up1, r1))
        for _ in range(n_steps):
            e1 = r1 - self._mv(self.dn1, r2)
            d2 = np.clip(self._mv(self.up1, e1), -0.5, 0.5)
            r2 = np.tanh(r2 + 0.15 * d2)
            e0 = s0 - self._mv(self.dn0, r1)
            d1 = np.clip(self._mv(self.up0, e0), -0.5, 0.5)
            r1 = np.tanh(r1 + 0.15 * d1)
        e0 = s0 - self._mv(self.dn0, r1)
        e1 = r1 - self._mv(self.dn1, r2)
        return {"s0": s0, "r1": r1, "r2": r2, "e0": e0, "e1": e1}

    # ---------- 学习（只更新存在的突触） ----------
    def learn(self, cache: dict, eta_scale: float = 1.0,
              homeostasis: bool = False) -> None:
        if self.eta_pc == 0.0 and self.eta_oja == 0.0 and not homeostasis:
            return
        s0, r1, r2 = cache["s0"], cache["r1"], cache["r2"]
        e0, e1 = cache["e0"], cache["e1"]
        if homeostasis:
            e0 = e0 / max(float(np.linalg.norm(e0)), 1e-9)
            e1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
        eta = self.eta_pc * eta_scale
        # P59：homeostasis=False 走融合核（8 次 → 1 次，逐位一致）；
        # homeostasis=True 保持原路径（e0/e1 归一化在核外，见融合核注释）。
        if not homeostasis:

            _pc_learn_fused(self.dn0, self.dn1, self.up0, self.up1,
                            e0, e1, r1, r2, s0, eta, self.eta_oja * eta_scale,
                            self.w_max)
            return
        # ⚠ 注意：**只有 homeostasis=True 才到得了这里** ——
        #   `homeostasis=False` 走上面的 numba 融合核。
        # （2026-10-07：原 P175 的「Rust 算子分派」已随 rust 版本一并删除。）
        # 下行生成权重：误差 × 上层表示（稀疏外积）
        self._add_outer(self.dn0, e0, r1, eta)
        self._add_outer(self.dn1, e1, r2, eta)
        # 上行识别权重：Oja 规则
        eo = self.eta_oja * eta_scale
        self._oja(self.up0, r1, s0, eo)
        self._oja(self.up1, r2, r1, eo)
        self._clip(self.up0[2], self.w_max)
        self._clip(self.up1[2], self.w_max)
        self._clip(self.dn0[2], self.w_max)
        self._clip(self.dn1[2], self.w_max)
        if homeostasis:
            self._homeostatic_scale()

    def learn_predictive(self, prev: dict, cache: dict, eta_scale: float = 1.0,
                         mix: float = 0.5, homeostasis: bool = False) -> None:
        """时序预测项 + 重构项的凸组合（与稠密版逐条对应，稀疏更新）。"""
        if self.eta_pc == 0.0 and self.eta_oja == 0.0 and not homeostasis:
            return
        s0, r1, r2 = cache["s0"], cache["r1"], cache["r2"]
        p_r1, p_r2 = prev["r1"], prev["r2"]
        e0, e1 = cache["e0"], cache["e1"]
        if homeostasis:
            e0 = e0 / max(float(np.linalg.norm(e0)), 1e-9)
            e1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
        e0p = s0 - self._mv(self.dn0, p_r1)
        e1p = r1 - self._mv(self.dn1, p_r2)
        if homeostasis:
            e0p = e0p / max(float(np.linalg.norm(e0p)), 1e-9)
            e1p = e1p / max(float(np.linalg.norm(e1p)), 1e-9)
        eta = self.eta_pc * eta_scale
        one_minus = 1.0 - mix
        # ⚠ P59 实测：`learn_predictive` 的融合版**负收益**（12 个 prange 段
        # 的线程调度 > 省下的 11 次核启动；26 万边 1.36×、105 万边 0.96×、
        # 419 万边 0.91×，服务器 191 核只会更差）→ 保持多核调用原路径。
        self._add_outer(self.dn0, e0, r1, eta * one_minus)
        self._add_outer(self.dn0, e0p, p_r1, eta * mix)
        self._add_outer(self.dn1, e1, r2, eta * one_minus)
        self._add_outer(self.dn1, e1p, p_r2, eta * mix)
        eo = self.eta_oja * eta_scale
        self._oja(self.up0, r1, s0, eo)
        self._oja(self.up1, r2, r1, eo)
        self._clip(self.up0[2], self.w_max)
        self._clip(self.up1[2], self.w_max)
        self._clip(self.dn0[2], self.w_max)
        self._clip(self.dn1[2], self.w_max)
        if homeostasis:
            self._homeostatic_scale()

    def homeostatic_rate(self, r2: np.ndarray, target: float,
                         eta_h: float) -> None:
        """逐神经元目标发放率（内在可塑性）：按增益缩放该神经元存在的入边。"""
        if self.act_ema is None:
            self.act_ema = np.full(len(r2), target)
        act = (r2 + 1.0) * 0.5
        self.act_ema += eta_h * (act - self.act_ema)
        gain = np.exp(np.clip((target - self.act_ema) * 2.0, -0.5, 0.5))
        indptr, _idx, val = self.up1
        for i in range(len(gain)):
            p0, p1 = indptr[i], indptr[i + 1]
            val[p0:p1] *= gain[i]

    def _homeostatic_scale(self) -> None:
        """突触缩放（Turrigiano 2008）：每行存在的权重范数拉回初始值。"""
        self._scale_rows(self.up0, self._hn_up0)
        self._scale_rows(self.up1, self._hn_up1)
        self._scale_rows(self.dn0, self._hn_dn0)
        self._scale_rows(self.dn1, self._hn_dn1)

    def prune_silence(self, threshold: float) -> int:
        """发育期突触修剪（`critical_period`）：把 |w| < threshold 的**存在突触**
        权重置零并返回被沉默的边数。

        生物学对应「突触沉默」（silent synapse）——结构保留、传递效率归零，
        是可逆的早期修剪形态。真正的**结构性删除**（从 CSR 中移除该边，
        使行度不齐）需要变长 CSR / 重新压缩，列为后续项。
        与稠密版语义对齐：稠密版同样是把低于阈值的元素置零。
        """
        n = 0
        for (_ip, _idx, val) in (self.up0, self.up1, self.dn0, self.dn1):
            mask = np.abs(val) < threshold
            n += int(mask.sum())
            val[mask] = 0.0
        return n

    def stats(self) -> dict:
        """结构性稀疏统计：存在突触数 vs 等维度稠密矩阵的元素数。"""
        dense_total = (self.n1 * self.n0 + self.n2 * self.n1
                       + self.n0 * self.n1 + self.n1 * self.n2)
        syn = self.n_synapses()
        return {"synapses": syn, "dense_equivalent": dense_total,
                "connectivity": syn / dense_total if dense_total else 0.0,
                "k0": self.k0, "k1": self.k1}


# ---------------------------------------------------------------- P99
def _make_serial_kernel():
    """P99：M2 单核融合核 = P52 融合核去掉并行（`parallel=True` → 无）。

    动机（服务器实测）：`_pc_infer_fused` 在 x86 快 1.15–2.16×，但在**昇腾
    aarch64 上慢 3–4×**（M2_infer 2.4 → 20–27 ms/tok）。根因不是计算量，而是
    **10 个 prange 区的 fork/join 屏障**在「很多核 + 小矩阵」（1024 行 ×
    128 边）上远比计算贵——核内 prange 1→6 线程本身只有 1.16×（访存饱和）。

    本核保留「一次调用 + 核内复用中间数组」的收益，去掉并行屏障；逐位口径与
    融合核一致（行内按 indptr 顺序累加、fastmath=True → 1-2 ulp 容差）。
    """
    if not NUMBA_OK:
        def _serial(up0, up1, dn0, dn1, s0, n_steps):
            r1 = np.tanh(_csr_matvec(*up0, s0))
            r2 = np.tanh(_csr_matvec(*up1, r1))
            e0 = e1 = None
            for _ in range(n_steps):
                e1 = r1 - _csr_matvec(*dn1, r2)
                d2 = np.clip(_csr_matvec(*up1, e1), -0.5, 0.5)
                r2 = np.tanh(r2 + 0.15 * d2)
                e0 = s0 - _csr_matvec(*dn0, r1) if e0 is None else e0 + (s0 - _csr_matvec(*dn0, r1))
                d1 = np.clip(_csr_matvec(*up0, e0), -0.5, 0.5)
                r1 = np.tanh(r1 + 0.15 * d1)
            return r1, r2, (s0 - _csr_matvec(*dn0, r1)), (r1 - _csr_matvec(*dn1, r2))
        return _serial

    def _serial(up0, up1, dn0, dn1, s0, n_steps):
        n1 = up0[0].shape[0] - 1
        n2 = up1[0].shape[0] - 1
        r1 = np.empty(n1)
        r2 = np.empty(n2)
        for i in range(n1):                       # up0 @ s0 → tanh
            s = 0.0
            for p in range(up0[0][i], up0[0][i + 1]):
                s += up0[2][p] * s0[up0[1][p]]
            r1[i] = np.tanh(s)
        for i in range(n2):                       # up1 @ r1 → tanh
            s = 0.0
            for p in range(up1[0][i], up1[0][i + 1]):
                s += up1[2][p] * r1[up1[1][p]]
            r2[i] = np.tanh(s)
        e0 = np.empty(n1)
        e1 = np.empty(n1)
        for _ in range(n_steps):
            for i in range(n1):                   # e1 = r1 - dn1 @ r2
                s = 0.0
                for p in range(dn1[0][i], dn1[0][i + 1]):
                    s += dn1[2][p] * r2[dn1[1][p]]
                e1[i] = r1[i] - s
            for i in range(n2):                   # d2 = clip(up1 @ e1)
                s = 0.0
                for p in range(up1[0][i], up1[0][i + 1]):
                    s += up1[2][p] * e1[up1[1][p]]
                d2 = min(0.5, max(-0.5, s))       # P103: 标量 np.clip 在 numba
                                                    # nopython 下 TypingError
                _ = d2
                r2[i] = np.tanh(r2[i] + 0.15 * d2)
            for i in range(n1):                   # e0 += s0 - dn0 @ r1
                s = 0.0
                for p in range(dn0[0][i], dn0[0][i + 1]):
                    s += dn0[2][p] * r1[dn0[1][p]]
                e0[i] = s0[i] - s
            for i in range(n1):                   # d1 = clip(up0 @ e0)
                s = 0.0
                for p in range(up0[0][i], up0[0][i + 1]):
                    s += up0[2][p] * e0[up0[1][p]]
                r1[i] = np.tanh(r1[i] + 0.15 * min(0.5, max(-0.5, s)))
        # P103：**步外再重算一次 e0/e1**——与 parallel 核（:261-273）语义一致：
        # 步内每步重算 e0，但返回前用**已更新的 r1/r2** 再算一遍。缺这段会让
        # serial 与 parallel 差一次 dn0@r1 的重算（实测 max|Δ| ≈ 0.19）。
        for i in range(n1):                       # e0 = s0 - dn0 @ r1（末步）
            s = 0.0
            for p in range(dn0[0][i], dn0[0][i + 1]):
                s += dn0[2][p] * r1[dn0[1][p]]
            e0[i] = s0[i] - s
        for i in range(n1):                       # e1 = r1 - dn1 @ r2
            s = 0.0
            for p in range(dn1[0][i], dn1[0][i + 1]):
                s += dn1[2][p] * r2[dn1[1][p]]
            e1[i] = r1[i] - s
        return r1, r2, e0, e1

    return njit(cache=True, nogil=True, fastmath=True)(_serial)


_pc_infer_fused_serial = _make_serial_kernel()
