"""M6 读出头 —— 对应 IT → 前额叶/前运动皮层的决策读出。

监督信号仅存在于最末端（局部感知器规则）；自 model.py 拆分。

O1-3（2026-09-23）：**结构性稀疏读出**（`conn_k > 0`，默认关闭）。
皮层→输出投射同样是稀疏的（每个下游细胞只接收部分上游输入），而此前读出是
稠密矩阵（n_out×n_in，连接率 100%）。启用后读出权重以 CSR 承载（每输出单元
k 条**存在**的入边，不存在的连接不存储、不计算），复用 `sparse_pc` 的 numba 核。
学习规则**不变**（仍是 softmax 交叉熵的末端梯度）——变化的是连接的存在性结构。
"""

import numpy as np

from .sparse_pc import (_csr_add_outer, _csr_clip, _csr_matvec, _csr_oja_up,
                        _random_csr)
from .plasticity import NUMBA_OK

# ---------------------------------------------------------------------------
# P7 性能（2026-09-24）：稠密读出的**融合并行更新核**。
#
# 实测（tools/_prof_step.py，256 维栈 / k_sparse=32 / 词级 LM）：读出学习占
# **78.9%**（8.97 ms/token，总计 11.37）。根因不是 FLOPs 而是**内存流量**：
# 旧路径 `tmp = outer(dp,h)`（分配 ~12.6 MB）→ `tmp *= eta` → `W -= tmp`，
# 每 token 触达约 3×12.6 MB 读 + 2×12.6 MB 写 ≈ 75 MB，单核带宽被吃满。
#
# 本核把三步融合为**一次对 W 的遍历**（读+写 25 MB），并按输出行并行：
#   - 元素级独立（`W[i,j] -= (dp[i]·h[j])·eta`），无跨行归约 → prange 安全；
#   - 运算顺序与 numpy 路径**逐位相同**（先 dp[i]*h[j]，再乘 eta，最后减）；
#   - 行梯度为 0 时整行短路（减 0 不改变任何值，含 ±0.0）。
# 因此是**纯性能改动、数值逐位等价**，不引入行为开关（同 P6 的处置）。
# ---------------------------------------------------------------------------
if NUMBA_OK:                                        # pragma: no cover
    from numba import njit, prange

    @njit(cache=True, parallel=True, fastmath=False)
    def _ro_dense_update(W, dp, h, eta):
        """W -= (dp ⊗ h) · eta —— 融合、按行并行；与 numpy 三步路径逐位等价。"""
        n_out, n_in = W.shape
        for i in prange(n_out):
            e = dp[i]
            if e == 0.0:
                continue
            for j in range(n_in):
                W[i, j] -= (e * h[j]) * eta

    # 注：`y = W·h` **没有**换成 numba 并行核——实测它比 OpenBLAS 的 dgemv
    # 慢 0.68×（1540×768）且**不逐位等价**（BLAS 的分块求和顺序不同，最大偏差
    # 1.4e-14）。故前向仍走 BLAS，融合核只用于更新（ci/verifiers/verify_readout_fused.py）。
else:                                               # pragma: no cover
    _ro_dense_update = None


# P9：量化码本的融合核（fp16/bf16/fp8/fp4）——**位算法量化**（非二分搜索：
# 搜索版每元素 ~15 次比较使更新核计算受限，实测 fp16 慢 fp32 33×）。
# 各格式舍入约定（与 numpy 参考/quantize_to 一致）：
#   fp16 = IEEE RNE（numpy astype 同源）；bf16 = RNE 位截断；
#   fp8/fp4 = 最近格点、并列取小幅值（与格点 searchsorted 同则）。
# 越界一律 clamp 到最大有限码（防 Inf/NaN 毒化）。
if NUMBA_OK:                                        # pragma: no cover
    from numba import get_thread_id

    def _q_scratch():
        import numba as _nb
        return np.empty((_nb.get_num_threads(), 1), dtype=np.float32)

    @njit(inline="always")
    def _bits_u32(scr, w):
        """f32 标量 → uint32 位型（f32 scratch 的位型 view）。

        ⚠ scr 必须是 float32 数组（uint32 数组会发生数值截断而非位型重解释）。
        """
        t = get_thread_id()
        scr[t, 0] = w
        return scr.view(np.uint32)[t, 0]

    @njit(inline="always")
    def _q_bf16(scr, w):
        a = int(_bits_u32(scr, w))
        s = (a >> 16) & 0x8000
        r = a & 0x7FFFFFFF
        c = (r + 0x7FFF + ((r >> 16) & 1)) >> 16
        if c >= 0x7F80:
            c = 0x7F7F                              # clamp 最大有限
        return np.uint16(c | s)

    @njit(inline="always")
    def _q_fp16(scr, w):
        a = int(_bits_u32(scr, w))
        s = (a >> 16) & 0x8000
        r = a & 0x7FFFFFFF
        if r >= 0x477FF000:
            c = 0x7BFF                              # 溢出/Inf/NaN → 最大有限
        elif r < 0x38800000:
            # < 2^-14 → 次正规或 0（RNE）
            e32 = r >> 23
            if e32 == 0:
                c = 0
            else:
                man = (r & 0x7FFFFF) | 0x800000
                shift = 126 - e32                   # value/2^-24 的定点移量（≥ 1）
                hm = man >> shift
                rem2 = (man - (hm << shift)) * 2
                if rem2 > (1 << shift):
                    hm += 1
                elif rem2 == (1 << shift) and (hm & 1):
                    hm += 1                         # RNE：并列取偶
                if hm >= 0x400:
                    c = 0x3C00                      # 进位到最小正规
                else:
                    c = hm
        else:
            r2 = r + 0x0FFF + ((r >> 13) & 1)
            c = (r2 >> 13) - 0x1C000                # 指数重置偏置 127→15
            if c >= 0x7C00:
                c = 0x7BFF                          # 溢出 clamp
        return np.uint16(c | s)

    @njit(inline="always")
    def _q_fp8(scr, w):
        a = _bits_u32(scr, w)
        s = int((a >> 24) & np.uint32(0x80))
        r = int(a & np.uint32(0x7FFFFFFF))
        if r >= 0x7F800000:
            c = 0x7E                                # Inf/NaN → 最大有限 448
        else:
            e32 = r >> 23
            if e32 <= 108:
                c = 0                               # < 2^-10（最小次正规之半）→ 0
            elif e32 <= 120:
                # 次正规：u = (1+m/2^23)·2^(e32-121)，m ∈ [0,7]
                man = (r & 0x7FFFFF) | 0x800000
                shift = 141 - e32                   # ∈ [21, 32]
                u = man >> shift
                rem = man - (u << shift)
                if rem * 2 > (1 << shift):
                    u += 1                          # 并列取小幅值（不进位）
                c = u if u <= 7 else 8              # 次正规舍入进位 → 最小正规码
            else:
                # 正规：e8 = e32-120 ∈ [1,15]，3 位尾数在 bit 20
                e8 = e32 - 120
                man3 = (r >> 20) & 7
                rem = r & 0xFFFFF
                if rem * 2 > 0x100000:              # 严格过半 → 进位（并列取小）
                    man3 += 1
                    if man3 == 8:
                        man3 = 0
                        e8 += 1
                if e8 >= 16 or (e8 == 15 and man3 == 7):
                    c = 0x7E                        # 溢出 / e4m3fn 的 NaN 槽 → 448
                else:
                    c = (e8 << 3) | man3
        return np.uint8(c | s)

    @njit(inline="always")
    def _q_fp4(scr, w, wscale):
        v = w / wscale
        if v < 0.0:
            v = -v
            if v <= 0.25: q = 8
            elif v <= 0.75: q = 9
            elif v <= 1.25: q = 10
            elif v <= 1.75: q = 11
            elif v <= 2.5: q = 12
            elif v <= 3.5: q = 13
            elif v <= 5.0: q = 14
            else: q = 15
        else:
            if v <= 0.25: q = 0
            elif v <= 0.75: q = 1
            elif v <= 1.25: q = 2
            elif v <= 1.75: q = 3
            elif v <= 2.5: q = 4
            elif v <= 3.5: q = 5
            elif v <= 5.0: q = 6
            else: q = 7
        return np.uint8(q)

# P9：量化码本的更新核（fp16/bf16/fp8/fp4）——LUT 反量化 + 位算法重量化。
if NUMBA_OK:                                        # pragma: no cover
    @njit(cache=True, parallel=True, fastmath=False)
    def _ro_q_update_fp16(codes, scr, dp, h, eta, n_in):
        """fp16 码本：LUT 反量化 → fp32 更新 → RNE 位算法重量化（按行并行）。"""
        n_out = codes.shape[0] // n_in
        for i in prange(n_out):
            e = dp[i]
            if e == 0.0:
                continue
            base = i * n_in
            for j in range(n_in):
                o = base + j
                w = lut_fp16[codes[o]] - (e * h[j]) * eta
                codes[o] = _q_fp16(scr, w)

    @njit(cache=True, parallel=True, fastmath=False)
    def _ro_q_update_bf16(codes, scr, dp, h, eta, n_in):
        """bf16 码本：同上（RNE 位截断）。"""
        n_out = codes.shape[0] // n_in
        for i in prange(n_out):
            e = dp[i]
            if e == 0.0:
                continue
            base = i * n_in
            for j in range(n_in):
                o = base + j
                w = lut_bf16[codes[o]] - (e * h[j]) * eta
                codes[o] = _q_bf16(scr, w)

    @njit(cache=True, parallel=True, fastmath=False)
    def _ro_q_update_fp8(codes, scr, dp, h, eta, n_in):
        """fp8 e4m3 码本：同上（最近格点、并列取小幅值）。"""
        n_out = codes.shape[0] // n_in
        for i in prange(n_out):
            e = dp[i]
            if e == 0.0:
                continue
            base = i * n_in
            for j in range(n_in):
                o = base + j
                w = lut_fp8[codes[o]] - (e * h[j]) * eta
                codes[o] = _q_fp8(scr, w)

    @njit(cache=True, parallel=True, fastmath=False)
    def _ro_q_update_fp4(codes, scr, dp, h, eta, n_in, wscale):
        """fp4 e2m1：半字节打包 + 逐张量缩放（低 4 位 = 偶下标，高 4 位 = 奇下标）。"""
        n_out = codes.shape[0] // ((n_in + 1) // 2)
        for i in prange(n_out):
            e = dp[i]
            if e == 0.0:
                continue
            row = i * ((n_in + 1) // 2)
            for j in range(n_in):
                o = row + (j >> 1)
                c = (codes[o] >> 4) if (j & 1) else (codes[o] & np.uint8(0xF))
                w = lut_fp4[c] * wscale - (e * h[j]) * eta
                q = _q_fp4(scr, w, wscale)
                if j & 1:
                    codes[o] = (codes[o] & np.uint8(0x0F)) | np.uint8(q << 4)
                else:
                    codes[o] = (codes[o] & np.uint8(0xF0)) | np.uint8(q)

    @njit(cache=True, parallel=True, fastmath=False)
    def _ro_q_matvec_u16(codes, lut, h, n_in, wscale):
        n_out = codes.shape[0] // n_in
        y = np.empty(n_out, dtype=np.float64)
        for i in prange(n_out):
            s = 0.0
            base = i * n_in
            for j in range(n_in):
                s += lut[codes[base + j]] * wscale * h[j]
            y[i] = s
        return y

    @njit(cache=True, parallel=True, fastmath=False)
    def _ro_q_matvec_u8(codes, lut, h, n_in, wscale):
        n_out = codes.shape[0] // n_in
        y = np.empty(n_out, dtype=np.float64)
        for i in prange(n_out):
            s = 0.0
            base = i * n_in
            for j in range(n_in):
                s += lut[codes[base + j]] * wscale * h[j]
            y[i] = s
        return y

    @njit(cache=True, parallel=True, fastmath=False)
    def _ro_q_matvec_fp4(codes, lut, h, n_in, wscale):
        n_out = codes.shape[0] // ((n_in + 1) // 2)
        y = np.empty(n_out, dtype=np.float64)
        for i in prange(n_out):
            s = 0.0
            row = i * ((n_in + 1) // 2)
            for j in range(n_in):
                c = (codes[row + (j >> 1)] >> 4) if (j & 1) \
                    else (codes[row + (j >> 1)] & np.uint8(0xF))
                s += lut[c] * wscale * h[j]
            y[i] = s
        return y
else:                                               # pragma: no cover
    (_ro_q_update_fp16, _ro_q_update_bf16, _ro_q_update_fp8,
     _ro_q_update_fp4) = (None, None, None, None)
    _ro_q_matvec_u16 = _ro_q_matvec_u8 = _ro_q_matvec_fp4 = None

_RO_STATE = {"ok": None}      # None=未探测 / True=可用 / False=回退 numpy


# ---------------------------------------------------------------------------
# P9 读出精度体系（2026-09-26，fhz 指令：**停止 fp64 支持**；新增 fp16/bf16/
# fp8/fp4；**默认 fp32**）。
#
# 设计：低精度格式**按原生位型存储为码本**，计算时查表（LUT）反量化到 fp32、
# 更新后重新量化写回；融合核内联查表/二分量化 → 内存流量 ∝ 存储位宽：
#   fp16（2B，IEEE 半精度）/ bf16（2B，截尾 fp32 指数）/ fp8（1B，e4m3fn）/
#   fp4（0.5B，e2m1 半字节打包）——流量分别为 fp32 的 1/2、1/2、1/4、1/8。
# 量化 = 到格式正格点集的最近邻（二分搜索，numba 纯算术实现，无位技巧依赖）；
# 越界截断到格点端点、NaN 归 +0（防权重投毒扩散）。
# 质量口径：softmax/NLL 一律在 fp64 上计算（y 反量化后升精度），仅存储与
#   梯度更新走低精度——这是低精度训练的标准「高精度主回路」结构。
# ---------------------------------------------------------------------------

RO_DTYPES = ("fp32", "fp16", "bf16", "fp8", "fp4")


def _build_lut(fmt: str) -> np.ndarray:
    """格式 → (反量化 LUT[f32], 正格点 lat[f32 升序·含 +0], 符号位)。

    lat = lut[0:有限正上界]——码 0 = +0.0，格点下标即存储码（正数域）。
    """
    if fmt == "fp16":
        # 位型重解释（⚠ 不是数值转换——码 c 的 LUT 值 = 以 c 为 fp16 位型的浮点值）
        lut = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
        b, sb = 0x7C00, 15                          # 排除 Inf/NaN；含 +0 与次正规
    elif fmt == "bf16":
        u32 = np.arange(65536, dtype=np.uint16).astype(np.uint32) << 16
        lut = np.frombuffer(u32.tobytes(), dtype=np.float32).copy()
        b, sb = 0x7F80, 15
    elif fmt == "fp8":                              # e4m3fn：无 Inf，0x7F=NaN
        codes = np.arange(256, dtype=np.uint8)
        s = np.where(codes & 0x80, -1.0, 1.0)
        e = ((codes >> 3) & 0xF).astype(np.int32)
        m = (codes & 7).astype(np.float64)
        val = np.where(e == 0, (m / 8.0) * 2.0 ** -6,
                       (1.0 + m / 8.0) * 2.0 ** (e - 7)).astype(np.float32)
        val = np.where((e == 15) & (m == 7), np.float32(np.nan), val) * s
        lut = val.astype(np.float32)
        b, sb = 0x7F, 7
    elif fmt == "fp4":                              # e2m1：±{0,.5,1,1.5,2,3,4,6}
        pos = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
        lut = np.concatenate([pos, -pos]).astype(np.float32)       # 全 16 码（15 = -0）
        b, sb = 8, 3
    else:
        raise ValueError(f"未知精度格式: {fmt}")
    lut = np.ascontiguousarray(lut)
    lat = np.ascontiguousarray(lut[:b])
    return lut, lat, sb


_Q_TABLES: dict = {}
# 全局 LUT 单例（numba 内核按全局名冻结引用；构建为一次性 ~ms 级开销）
lut_fp16, lat_fp16, sb_fp16 = _build_lut("fp16")
lut_bf16, lat_bf16, sb_bf16 = _build_lut("bf16")
lut_fp8, lat_fp8, sb_fp8 = _build_lut("fp8")
lut_fp4, lat_fp4, sb_fp4 = _build_lut("fp4")


def _q_tables(fmt: str):
    if fmt not in _Q_TABLES:
        lut, lat, sb = _build_lut(fmt)
        _Q_TABLES[fmt] = (lut, lat, sb)
    return _Q_TABLES[fmt]


def quantize_to(fmt: str, x: np.ndarray) -> np.ndarray:
    """fp32 数组 → 指定格式的码本（numpy 向量化；最近格点、越界截断、NaN→最大格点）。"""
    lut, lat, sb = _q_tables(fmt)
    x = np.ascontiguousarray(x, dtype=np.float32).ravel()
    if fmt == "fp32":
        return x
    ax = np.abs(x)
    idx = np.searchsorted(lat, ax)
    idx = np.clip(idx, 0, len(lat) - 1)
    left = np.maximum(idx - 1, 0)
    pick_left = (ax - lat[left]) <= (lat[idx] - ax)
    idx = np.where(pick_left, left, idx).astype(np.uint32)
    sign = np.where(x < 0, np.uint32(1 << sb), np.uint32(0)).astype(np.uint32)
    codes = idx | sign
    if fmt == "fp4":
        codes = codes.astype(np.uint8).reshape(-1, 2)
        return (codes[:, 0] | (codes[:, 1] << 4)).astype(np.uint8)
    return codes.astype(np.uint16 if fmt in ("fp16", "bf16") else np.uint8)


def dequantize_from(fmt: str, codes: np.ndarray, n_elem: int | None = None) -> np.ndarray:
    """码本 → fp32（numpy 向量化）。fp4 半字节展开。"""
    lut, lat, sb = _q_tables(fmt)
    if fmt == "fp32":
        return codes
    if fmt == "fp4":
        lo = (codes & 0xF).astype(np.int32)
        hi = (codes >> 4).astype(np.int32)
        vals = np.empty(codes.size * 2, dtype=np.float32)
        vals[0::2] = lut[lo]
        vals[1::2] = lut[hi]
        return vals[:n_elem] if n_elem is not None else vals
    idx = codes.astype(np.int32)
    return lut[idx]


class Readout:
    """M6 读出头 —— 对应 IT→前运动皮层；感知器式局部规则（监督仅在此末端）。

    连接结构可选：稠密（默认，`conn_k=0`）或结构性稀疏 CSR（`conn_k=k>0`）。
    """

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator,
                 w_clip: float = 0.0, dtype: str = "fp32", conn_k: int = 0,
                 lognormal_init: bool = False, exc_ratio: float = 0.8,
                 hidden: int = 0, hid_k: int = 0, eta_hid: float = 0.002):
        self._n_in, self._n_out = int(n_in), int(n_out)
        self.conn_k = int(conn_k) if conn_k and conn_k > 0 else 0
        # A4（2026-09-23）：**两级群体读出**（`hidden > 0`，默认 0 = 单层线性）。
        # 脑对应：皮层→输出的投射是**多级（多突触中继）**且靠**群体编码**，
        # 不是单层线性分类器。第一级：h → 隐藏群体（稀疏投射 + k-WTA 侧抑制竞争，
        # 局部无监督 Oja 学习）；第二级：群体 → 输出（稀疏投射 + 任务监督）。
        self.hidden = int(hidden) if hidden and hidden > 0 else 0
        self.hid_k = int(hid_k)
        self.eta_hid = float(eta_hid)
        # P9 精度体系（2026-09-26，fhz 指令）：**停止 fp64 支持**；可选
        # fp32（默认）/ fp16 / bf16 / fp8 / fp4。低精度 = 原生位型码本存储 +
        # 查表反量化计算 + 重新量化写回（P9 融合核），softmax/NLL 保持 fp64。
        # 注：结构性稀疏模式暂用 fp32 CSR（与量化码本不叠加——组合另行立项）。
        if dtype not in RO_DTYPES:
            raise ValueError(f"readout dtype 须为 {RO_DTYPES} 之一，实际: {dtype!r}"
                             "（fp64 已按 fhz 指令停止支持）")
        self.dtype_name = dtype if self.conn_k == 0 else "fp32"
        self.qfmt = self.dtype_name if self.dtype_name != "fp32" else None
        self.dtype = np.float32
        # C8 修复：可选权重范数上限，防止长跑/高学习率下读出权重溢出
        # （默认 0.0 = 关闭，保持旧行为逐位不变）
        self.w_clip = float(w_clip)
        # B7（2026-09-22）：minibatch 梯度累积缓冲（accumulate=1 时永不启用）
        self._grad_acc: np.ndarray | None = None
        self._acc_n = 0

        if self.hidden > 0:                     # A4：两级群体读出
            k1 = self.hid_k if self.hid_k > 0 else max(1, n_in // 8)
            k2 = self.conn_k if self.conn_k > 0 else max(1, self.hidden // 8)
            self.conn_k = 0
            # ⚠ fan-in 补偿：稀疏投射的权重尺度 ×√(fan_in/k)，使其输出幅值与
            #   稠密版（尺度 0.05、fan_in 全量）可比；否则幅值偏低 √(k/fan_in) 倍。
            self._W1 = _random_csr(rng, self.hidden, n_in, k1,
                                   0.05 * np.sqrt(n_in / k1),
                                   lognormal_init, exc_ratio)
            self._W2 = _random_csr(rng, n_out, self.hidden, k2,
                                   0.05 * np.sqrt(self.hidden / k2),
                                   lognormal_init, exc_ratio)
            self._hid_keep = max(1, self.hidden // 8)    # k-WTA 保留数（群体竞争）
            self._W = None
            self._csr = None
            self._codes = None
            self._wscale = 1.0
            self._lut = self._lat = None
            self._sbit = 0
        elif self.conn_k > 0:                   # O1-3：结构性稀疏读出（单级）
            self.conn_k = max(1, min(self.conn_k, n_in))
            self._csr = _random_csr(rng, n_out, n_in, self.conn_k,
                                    0.05 * np.sqrt(n_in / self.conn_k),   # fan-in 补偿
                                    lognormal_init, exc_ratio)
            self._W = None
            self._codes = None
            self._wscale = 1.0
            self._lut = self._lat = None
            self._sbit = 0
        elif lognormal_init:                    # O1-4：皮层式（重尾 + E/I 比）
            from .inits import cortical_init
            self._codes = None
            self._set_dense(cortical_init(rng, (n_out, n_in), 0.05, exc_ratio))
        else:
            self._csr = None
            self._set_dense(rng.normal(0, 0.05, (n_out, n_in)))

    def _set_dense(self, W: np.ndarray) -> None:
        """按当前精度格式存放稠密权重（fp32 直存；低精度量化为码本）。

        fp4 追加逐张量缩放（microscaling）：max|W| 映射到格点上限 6，
        否则小尺度权重会整体落入 e2m1 的 0 格点。
        """
        if self.qfmt is None:
            self._W = np.ascontiguousarray(W, dtype=np.float32)
            self._codes = None
            self._lut = self._lat = None
            self._sbit = 0
            self._wscale = 1.0
        else:
            W = np.ascontiguousarray(W, dtype=np.float32)
            self._lut, self._lat, self._sbit = _q_tables(self.qfmt)
            if self.qfmt == "fp4":
                m = float(np.abs(W).max()) if W.size else 0.0
                self._wscale = (m / 6.0) if m > 0 else 1.0
                self._codes = quantize_to("fp4", W / self._wscale)
            else:
                self._wscale = 1.0
                self._codes = quantize_to(self.qfmt, W)
            if self.qfmt == "fp4" and self._codes.size % 2:
                self._codes = np.concatenate([self._codes,
                                              np.zeros(1, dtype=np.uint8)])

    def _deq_w(self) -> np.ndarray:
        """码本 → fp32 稠密权重（兼容视图；统计/保存用）。"""
        n = self._n_out * self._n_in
        return (dequantize_from(self.qfmt, self._codes, n)
                * self._wscale).reshape(self._n_out, self._n_in)

    # ---------- 权重访问（稀疏模式返回只读稠密视图，兼容既有代码/统计） ----------
    @classmethod
    def from_dense(cls, W: np.ndarray, w_clip: float = 0.0, k: int = 0,
                   dtype: str = "fp32") -> "Readout":
        """由稠密权重构造（每输出单元取 |w| 最大的 k 列；k≤0 或 k≥列数 → 全列）。

        用途：(a) 可比性对拍——k = 全列时与稠密读出**权重完全相同**；
              (b) 稠密→稀疏蒸馏（保留最强连接）。
        """
        from .sparse_pc import _from_dense_csr
        n_out, n_in = W.shape
        self = cls.__new__(cls)
        self._n_in, self._n_out = n_in, n_out
        self.hidden = 0
        self.conn_k = min(k, n_in) if k > 0 else n_in
        self.dtype_name = dtype if self.conn_k == 0 else "fp32"
        self.qfmt = self.dtype_name if self.dtype_name != "fp32" else None
        self.dtype = np.float32
        self.w_clip = float(w_clip)
        self._grad_acc, self._acc_n = None, 0
        self._csr = _from_dense_csr(W, self.conn_k)
        self._codes = None
        self._wscale = 1.0
        self._lut = self._lat = None
        self._sbit = 0
        if self.conn_k >= n_in or k <= 0:       # 全列 = 稠密 → 走精度存储路径
            self._csr = None
            self.conn_k = 0
            self._set_dense(np.asarray(W, dtype=np.float32))
        else:
            self._W = None
        return self

    @property
    def W(self) -> np.ndarray:
        """输出层权重的稠密视图（量化模式 = 反量化快照 fp32；两级模式返回 W2）。"""
        if self.hidden > 0:
            ip, idx, val = self._W2
            W = np.zeros((self._n_out, self.hidden), dtype=val.dtype)
            for i in range(self._n_out):
                W[i, idx[ip[i]:ip[i + 1]]] = val[ip[i]:ip[i + 1]]
            return W
        if self.conn_k > 0:
            ip, idx, val = self._csr
            W = np.zeros((self._n_out, self._n_in), dtype=val.dtype)
            for i in range(self._n_out):
                W[i, idx[ip[i]:ip[i + 1]]] = val[ip[i]:ip[i + 1]]
            return W
        return self._W if self.qfmt is None else self._deq_w()

    @W.setter
    def W(self, v) -> None:
        if self.qfmt is None:
            self._W = np.ascontiguousarray(v, dtype=np.float32)
        else:
            self._set_dense(v)

    def n_synapses(self) -> int:
        """实际存在的连接数（两级/稀疏模式 = CSR 条目；稠密模式 = 矩阵元素数）。"""
        if self.hidden > 0:
            return int(len(self._W1[2]) + len(self._W2[2]))
        return int(len(self._csr[2])) if self.conn_k > 0 else int(self._n_out * self._n_in)

    def _kwta(self, u: np.ndarray) -> np.ndarray:
        """群体竞争（侧抑制）：仅保留响应最强的 `hidden/8` 个单元并压缩幅值。

        脑对应：皮层群体编码的稀疏化竞争（抑制性中间神经元介导的侧抑制）。
        """
        k = min(self._hid_keep, len(u))
        idx = np.argpartition(-u, k - 1)[:k]
        a = np.zeros_like(u)
        a[idx] = np.tanh(u[idx])
        return a

    def stats(self) -> dict:
        dense = self._n_out * self._n_in
        n = self.n_synapses()
        out = {"synapses": n, "dense_equivalent": dense,
               "connectivity": n / dense if dense else 1.0, "k": self.conn_k,
               "dtype": self.dtype_name,
               "storage_MB": (self._codes.nbytes if self._codes is not None
                              else (self._W.nbytes if self._W is not None else 0)) / 1e6}
        if self.hidden > 0:
            out.update({"levels": 2, "hidden": self.hidden,
                        "synapses_w1": int(len(self._W1[2])),
                        "synapses_w2": int(len(self._W2[2]))})
        return out

    # ---------- 前向 ----------
    def __call__(self, h: np.ndarray) -> np.ndarray:
        if self.hidden > 0:                     # 两级群体读出（稀疏投射 + 群体竞争）
            a = self._kwta(_csr_matvec(*self._W1, h))
            return _csr_matvec(*self._W2, a)
        if self.conn_k > 0:                     # 稀疏：只遍历存在的边
            return _csr_matvec(*self._csr, h)
        if self.qfmt is not None:               # P9：量化码本内联反量化 matvec
            K = {"fp16": _ro_q_matvec_u16, "bf16": _ro_q_matvec_u16,
                 "fp8": _ro_q_matvec_u8, "fp4": _ro_q_matvec_fp4}[self.qfmt]
            return K(self._codes, self._lut, h.astype(np.float32, copy=False),
                     self._n_in, self._wscale)
        if self._W.dtype == np.float32:         # fp32：BLAS sgemv（升精度返回）
            return (self._W @ h.astype(np.float32, copy=False)).astype(np.float64,
                                                                       copy=False)
        return self._W @ h

    def _apply_update(self, dp: np.ndarray, h: np.ndarray, eta: float) -> bool:
        """融合核派发：按精度格式选择内核；返回 False = 调用方走 numpy 回退。"""
        if _ro_dense_update is None:
            return False
        dp32 = np.asarray(dp, dtype=np.float32)
        h32 = np.asarray(h, dtype=np.float32)
        e32 = np.float32(eta)
        if self.qfmt is None:
            if eta == 0.0:
                return True
            _ro_dense_update(self._W, dp32, h32, e32)
            return True
        if eta == 0.0:
            return True
        scr = _q_scratch()
        n = self._n_in
        if self.qfmt == "fp16":
            _ro_q_update_fp16(self._codes, scr, dp32, h32, e32, n)
        elif self.qfmt == "bf16":
            _ro_q_update_bf16(self._codes, scr, dp32, h32, e32, n)
        elif self.qfmt == "fp8":
            _ro_q_update_fp8(self._codes, scr, dp32, h32, e32, n)
        else:
            _ro_q_update_fp4(self._codes, scr, dp32, h32, e32, n, self._wscale)
        return True

    def _clip(self) -> None:
        if self.w_clip > 0.0:
            if self.hidden > 0:
                _csr_clip(self._W1[2], self.w_clip)
                _csr_clip(self._W2[2], self.w_clip)
            elif self.conn_k > 0:
                _csr_clip(self._csr[2], self.w_clip)
            elif self.qfmt is None:
                np.clip(self._W, -self.w_clip, self.w_clip, out=self._W)
            else:
                self._set_dense(np.clip(self._deq_w(), -self.w_clip, self.w_clip))

    def _check_contract(self, h: np.ndarray, target: np.ndarray | None) -> None:
        """维度契约校验（fail-fast）。

        此前维度不符时只抛 numpy 的模糊广播错误（如
        `operands could not be broadcast together with shapes (64,) (32,)`），
        调用方难以定位。注意 `step(x, target=...)` 的 target 维度是**读出输出
        维度**（`cfg.n_readout`，默认 = `cfg.n_input`），**不是** `n_top`。
        """
        if h.shape[0] != self._n_in:
            raise ValueError(
                f"Readout 输入维度不符：期望 {self._n_in}（h 的维度），实际 {h.shape[0]}")
        if target is not None and target.shape[0] != self._n_out:
            raise ValueError(
                f"Readout 目标维度不符：期望 {self._n_out}"
                f"（= cfg.n_readout，0 时取 cfg.n_input），实际 {target.shape[0]}。"
                "注意 PHDNet.step(x, target=...) 的 target 是**读出输出维度**，不是 n_top；"
                "可用 net.n_out 查询。")

    # ---------- 学习（感知器 / softmax） ----------
    def learn(self, h: np.ndarray, target: np.ndarray, eta: float) -> None:
        self._check_contract(h, target)
        if self.hidden > 0:                     # 两级：输出监督 + 中间层局部无监督
            a = self._kwta(_csr_matvec(*self._W1, h))
            y = _csr_matvec(*self._W2, a)
            _csr_add_outer(*self._W2, target - y, a, eta)
            if self.eta_hid > 0.0:
                _csr_oja_up(*self._W1, a, h, self.eta_hid)   # Hebbian/Oja（局部）
            self._clip()
            return
        y = self.__call__(h)
        if self.conn_k > 0:                     # ΔW = η·(t − y) ⊗ h（只更新存在的边）
            _csr_add_outer(*self._csr, target - y, h, eta)
        elif not self._apply_update(y - target, h, eta):
            self.W = self.W - eta * np.outer(target - y, h)   # numpy 回退（含量化模式）
        self._clip()

    def learn_softmax(self, h: np.ndarray, target: np.ndarray, eta: float,
                      y_pre: np.ndarray | None = None,
                      accumulate: int = 1) -> float:
        """softmax 感知器（交叉熵的局部梯度 ∂L/∂y = p − t，仅作用于末端）。

        返回当前样本的 −log p(correct)，供困惑度统计。

        P6 性能修复（2026-09-21，逐位等价）：
          ① y_pre：调用方（model.step）前向已算得 W@h，传入可省一次重复
             矩阵-向量乘。与内部重算逐位一致（同一 W、同一 h）；None 时
             自行计算（旧行为）。y_pre 本体不被修改（y = y_pre − max 新数组）。
          ② 梯度更新改为「外积后原地缩放」：tmp = outer(dp,h); tmp *= eta;
             W -= tmp —— 与 eta * outer(dp,h) 逐位一致（IEEE 乘法交换律），
             但少分配一个 (n_out, n_in) 临时数组（词级读出下 ~9.5 MB），
             消除训练热路径的主要内存流量。

        B7 minibatch（2026-09-22，默认关闭）：`accumulate=N>1` 时累积 N 步
        梯度、按**平均梯度**更新一次（ΔW = −η·mean(g)，标准 minibatch 语义）。

        O1-3（2026-09-23，默认关闭）：`conn_k>0` 时更新只触达存在的连接
        （学习规则与稠密版相同，仅连接结构不同）。
        """
        if target is not None:
            self._check_contract(h, target)
        if y_pre is not None:
            y = y_pre - y_pre.max()      # 新数组，不修改调用方持有的 y_pre
        else:
            y = self.__call__(h)
            y = y - y.max()
        p = np.exp(y)
        p /= p.sum()
        correct = int(np.argmax(target))
        nll = float(-np.log(p[correct] + 1e-12))

        if accumulate > 1:                              # B7：minibatch 累积
            if self.hidden > 0:                         # 两级：按中间激活累积
                a_acc = self._kwta(_csr_matvec(*self._W1, h))
                g = np.outer(p - target, a_acc)
            else:
                a_acc = None
                g = np.outer(p - target, h)
            self._grad_acc = g if self._grad_acc is None else self._grad_acc + g
            self._acc_n += 1
            if self._acc_n >= accumulate:
                c = eta / float(accumulate)
                if self.hidden > 0:                     # 稀疏按行写回 W2 + W1 局部
                    ip, idx, val = self._W2
                    for i in range(self._n_out):
                        s = slice(ip[i], ip[i + 1])
                        val[s] -= c * self._grad_acc[i, idx[s]]
                    if self.eta_hid > 0.0:
                        _csr_oja_up(*self._W1, a_acc, h, self.eta_hid)
                elif self.conn_k > 0:                   # 稀疏按行写回累积梯度
                    ip, idx, val = self._csr
                    for i in range(self._n_out):
                        s = slice(ip[i], ip[i + 1])
                        val[s] -= c * self._grad_acc[i, idx[s]]
                else:
                    self.W = self.W - c * self._grad_acc     # 兼容量化模式（setter 重量化）
                self._grad_acc = None
                self._acc_n = 0
                self._clip()
            return nll

        if self.hidden > 0:                             # 两级：W2 监督 + W1 局部无监督
            a = self._kwta(_csr_matvec(*self._W1, h))
            _csr_add_outer(*self._W2, p - target, a, -eta)
            if self.eta_hid > 0.0:
                _csr_oja_up(*self._W1, a, h, self.eta_hid)
        elif self.conn_k > 0:                           # 稀疏梯度下降（存在的边）
            _csr_add_outer(*self._csr, p - target, h, -eta)
        else:
            # P7/P9：融合核（fp32 原生；fp16/bf16/fp8/fp4 = 码本内联查表 + 重新量化）
            if not self._apply_update(p - target, h, eta):
                self.W = self.W - eta * np.outer(p - target, h)   # numpy 回退（含量化）
        self._clip()
        return nll
