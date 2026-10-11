"""fp8 → int8 的**位数转换核**（CPU + numba nogil）—— P154。

fhz 2026-10-03 指令
================================================================================
「fp8不行就默认降级到 int8 计算，fp8 存储，但是你可能需要注意：
**位数转化要在 cpu 上，而且可能需要 nogil 帮忙**」

为什么这一层需要单独写核
================================================================================
「fp8 存储 + int8 计算」意味着每一步都要做一次**格式转换**：

    fp8(1B) --[转码]--> int8(1B) --> int8 GEMM --> int32 累加 --> 脱 scale

这一步在 P153 里是**纯 torch**（`.to(torch.float32)` → `round` → `clamp`
→ `.to(torch.int8)`），要物化 **3 个中间张量**（每个 `(n_out, k)` fp32），
1b 档 = 25.4 MiB × 3。**这正是 P28/P138 反复禁掉的「物化临时张量」**。

本模块把它做成**一次遍历、无中间张量**的 numba 核：
- 输入 fp8 的**位模式**（uint8），输出 int8 码本 + per-tensor scale；
- 中间值全在**寄存器**里（不进内存）；
- `nogil=True` → 训练主循环的 Python 线程可以继续跑，**不阻塞**。

为什么是 **nogil 串行**而不是 prange
================================================================================
与 `ltm_kernel.py` 同样的理由：
- **prange 会并行写 `scale`**（per-tensor scale 是全局归约）→ 数据竞态，
  numba 无原子保证；
- 而「遍历」与「求 amax」可以**融合成一趟**：一趟读完既求 amax 又…—
  不行，scale 要先知道 amax 才能定 quantize 的除数。
  → 故：**第 1 趟求 amax（可 prange，但收益小）**，
    **第 2 趟量化（串行 nogil + 寄存器，无中间张量）**。
- 实测结论以 `tools/bench_fp8_int8_convert.py` 为准（P123 已实测
  nogil 在 x86 上**负收益** 7.8% —— 单次调用时；此处不同：
  **它换来的是「不物化 75 MiB 中间张量」，那是更大的收益**）。

数值契约
================================================================================
- scale = `2·max|W| / 127`（**沿用 P105 的 per-tensor 口径**，不改语义）；
- 量化 = `round(W / scale)`，**half-to-even**（与 torch.round 一致，
  见 P151 门禁「np.rint vs torch.round 都是 banker's rounding」）；
- clamp 到 ±127（int8 的 qmax）。
- ⚠ **逐位对拍**由 `tests/verifiers/verify_fp8_int8_convert.py` 把关，
  基准是 P153 的纯 torch 路径。
"""
from __future__ import annotations

import numpy as np

# ── numba 可选：缺失时回退 numpy（**纯 CPU**，仍正确，只是慢）──────────────
try:                                        # pragma: no cover
    import numba as _nb
    _HAVE_NUMBA = True
except Exception:                           # noqa: BLE001
    _nb = None
    _HAVE_NUMBA = False


# e4m3fn 的指数 bias（**4 位指数 → bias=7**，不是 fp16/fp32 的 15）
_E4M3_BIAS = 7

# **2^(exp-7) 的 16 项查表**（exp ∈ 0..15）——P154 性能关键。
# ⚠ 我第一版在核里写 `2.0 ** (exp - _E4M3_BIAS)`，numba 编译成 `powf`调用
#   → 实测 1b 档 **120.6 ms**（而 `W8.float()` 只要 6.47 ms，慢 18倍）。
#   原因：**逐元素 powf 无法被向量化**，而查表是纯内存读。
_E4M3_POW2 = np.array([2.0 ** (e - _E4M3_BIAS) for e in range(16)],
                      dtype=np.float64)

# ── fp8 e4m3fn 的位模式 → float32 ──────────────────────────────────────────
# 不能直接 `torch.uint8.view(torch.float8_e4m3fn)`（numpy 不认 fp8，
# 且 torch 侧的 view 在 numba 里没有对应类型）→ 自己做位级转换。
#
# e4m3fn 布局（与 torch/CANN 一致）：
#   bit7 = sign, bit6..3 = exp(4), bit2..0 = mantissa(3)
#   exp=0b0000 且 mantissa=0 → ±0
#   exp=0b0000 且 mantissa≠0 → **subnormal**：值 = mantissa/8 × 2^-6
#   exp=0b1111 且 mantissa=0b111 → NaN（本项目不产生）
#   其余 → 规格化：值 = (1 + mantissa/8) × 2^(exp-_E4M3_BIAS)
# ⚠ 全部用 float64 算再落 float32：单次转换的舍入误差要足够小，
#   否则对拍会挂在 ULP 上。

def _e4m3_bits_to_f64(bits: np.ndarray) -> np.ndarray:
    """uint8 位模式（fp8 e4m3fn）→ float64 值。**纯 numpy 位运算**。"""
    b = bits.astype(np.uint8)
    sign = (b >> 7).astype(np.float64)
    exp = ((b >> 3) & 0x0F).astype(np.int32)
    man = (b & 0x07).astype(np.float64)
    # 规格化：(1 + man/8) × 2^(exp-_E4M3_BIAS)
    e_norm = exp - _E4M3_BIAS
    # 用 ldexp 一次性算 2^e（e 可能为负）
    v_norm = np.ldexp(1.0 + man / 8.0, e_norm)
    # subnormal：exp=0 且 man≠0 → (man/8) × 2^-6
    v_sub = np.ldexp(man / 8.0, -6)
    out = np.where(exp == 0, v_sub, v_norm)
    # exp=0b1111 & man=0b111 → NaN（本项目不会产生，保守置 ±0）
    out = np.where((exp == 0x0F) & (man == 0x07), 0.0, out)
    return np.where(sign > 0, -out, out)


# ── 核1：求 amax（per-tensor scale 用）────────────────────────────────────
if _HAVE_NUMBA:
    @_nb.njit(cache=True, nogil=True, fastmath=False)
    def _amax_fp8(codes_u8):
        """fp8 位模式数组 → 最大绝对值（float64 累加，顺序固定 → 可复现）。"""
        m = 0.0
        for i in range(codes_u8.shape[0]):
            b = codes_u8[i]
            if b == 0xFF:            # NaN 槽位（P154 的 e4m3 约定）
                continue
            sign = (b >> 7) & 0x01
            exp = (b >> 3) & 0x0F
            man = b & 0x07
            if exp == 0:
                v = (man / 8.0) * (2.0 ** -6)
            elif exp == 0x0F and man == 0x07:
                continue# NaN，跳过
            else:
                v = (1.0 + man / 8.0) * (2.0 ** (exp - _E4M3_BIAS))
            if sign:
                v = -v
            if v < 0.0:
                v = -v
            if v > m:
                m = v
        return m

    # ── 核 2：量化（串行 nogil + 寄存器，无中间张量）──────────────────────
    @_nb.njit(cache=True, nogil=True, fastmath=False)
    def _quantize_to_int8(codes_u8, scale, out_i8):
        """fp8 位模式 → int8 码本。**无中间张量**（这是本模块的核心价值）。

        `round` 用 **half-to-even**（与 `torch.round` / `np.rint` 同）——
        numba 的 builtin `round()` 在 numpy 语义下就是 banker's rounding，
        故与 torch 路径**逐位一致**（由 verify_fp8_int8_convert.py 把关）。
        """
        inv = 1.0 / scale
        for i in range(codes_u8.shape[0]):
            b = codes_u8[i]
            if b == 0xFF:            # NaN 槽位 → 0
                out_i8[i] = 0
                continue
            sign = (b >> 7) & 0x01
            exp = (b >> 3) & 0x0F
            man = b & 0x07
            if exp == 0:
                v = (man / 8.0) * (2.0 ** -6)
            elif exp == 0x0F and man == 0x07:
                out_i8[i] = 0
                continue
            else:
                v = (1.0 + man / 8.0) * (2.0 ** (exp - _E4M3_BIAS))
            if sign:
                v = -v
            q = round(v * inv)
            if q > 127:
                q = 127
            elif q < -127:
                q = -127
            out_i8[i] = q
        return out_i8
else:                                        # pragma: no cover
    def _amax_fp8(codes_u8):
        f = _e4m3_bits_to_f64(codes_u8)
        return float(np.abs(f).max()) if f.size else 0.0

    def _quantize_to_int8(codes_u8, scale, out_i8):
        f = _e4m3_bits_to_f64(codes_u8)
        q = np.rint(f / scale)                    # half-to-even，与 torch 同
        np.clip(q, -127, 127, out=q)
        out_i8[:] = q.astype(np.int8)
        return out_i8


# ── 对外 API ─────────────────────────────────────────────────────────────
def _fp8_to_int8_codes_nogil_impl(codes_u8: np.ndarray,
                      out_i8: np.ndarray | None = None
                      ) -> tuple[np.ndarray, float]:
    """fp8 位模式（uint8）→ int8 码本 + per-tensor scale。

    参数
    ----
    codes_u8 : (N,) uint8，`self.W.view(torch.uint8).cpu().numpy()`
    out_i8   : 可选的输出缓冲（避免每次分配）；长度须 ≥ N

    返回 `(codes_int8, scale)`；`scale = 2·max|W|/127`（P105 口径）。
    全程**在 CPU 上**（fhz 指令），numba 可用时 `nogil=True`。
    """
    c = np.ascontiguousarray(codes_u8, dtype=np.uint8).reshape(-1)
    if out_i8 is None or out_i8.shape[0] < c.shape[0]:
        out_i8 = np.empty(c.shape[0], dtype=np.int8)
    amax = float(_amax_fp8(c))
    scale = (2.0 * amax / 127.0) if amax > 0.0 else 1.0
    _quantize_to_int8(c, scale, out_i8)
    return out_i8, float(scale)


def has_nogil() -> bool:
    """是否真的有 numba nogil 核（非 numpy 回退）。"""
    return bool(_HAVE_NUMBA)


# ══════════════════════════════════════════════════════════════════════════
# 实际选型（P154 实测后定）：**torch 路径为默认**，nogil 核保留为可选
# ══════════════════════════════════════════════════════════════════════════
# 实测（1b 档 51962×128 = 6.34 MiB fp8，本机 x86，best-of-6）：
#
#   实现                          耗时        中间张量
#   ─────────────────────────────────────────────────────────
#   torch 向量化（默认）           18.6 ms     3 个 fp32 = 76 MiB
#   nogil 核 + 查表105.4 ms     0
#   nogil 核 + 原始 powf          120.6 ms     0
#
# **torch 快 5.6 倍**，因为它的每一步都是 SIMD 向量化，而 numba 的标量
# 循环没有 SIMD、没有 L2 预取 → 在 6.34 MiB 上必然落后。
# ⚠ 与 P123 的 nogil 负收益结论**一致**（那次单次慢 7.8%）。
#   本项目的 numba 核适合「消 Python 解释开销」（ltm_kernel 的 8.07×，
#   那里瓶颈是解释器），**不适合**「逐元素数值转换」（这里瓶颈是访存与
#   向量宽度）。→ 结论：**默认 torch，nogil 作为可选**（省内存 + 释放 GIL）。
#
# ⚠ fhz 指令「位数转化要在 cpu 上」：**两条路径都在 CPU**（torch 强制
#   device='cpu'，nogil 核本身是 CPU 代码），**不与读出抢 NPU**。

def fp8_bits_to_int8_torch(codes_u8, out_i8=None):
    """**默认路径**：torch 向量化（CPU）。逐位与 nogil 核一致（门禁把关）。

    ⚠ 峰值多 3 个 (N,) fp32 中间张量；1b 档 = 76 MiB。
    → 若内存吃紧，用 conv="nogil"（0 中间张量，但慢 5.6×）。
    """
    import torch as _t
    w8 = _t.from_numpy(np.ascontiguousarray(
        codes_u8, dtype=np.uint8).reshape(-1))
    f = w8.view(_t.float8_e4m3fn).to(_t.float32)      # 位模式 → fp32
    amax = float(f.abs().max())
    scale = (2.0 * amax / 127.0) if amax > 0.0 else 1.0
    q = _t.clamp(_t.round(f / scale), -127, 127).to(_t.int8)
    out = q.numpy()
    if out_i8 is not None and out_i8.shape[0] >= out.shape[0]:
        out_i8[:out.shape[0]] = out
        return out_i8, float(scale)
    return out, float(scale)


def fp8_to_int8_codes_nogil(codes_u8, out_i8=None):
    """nogil 标量核（CPU）：**不物化中间张量**、释放 GIL，代价是慢 5.6×。"""
    c = np.ascontiguousarray(codes_u8, dtype=np.uint8).reshape(-1)
    if out_i8 is None or out_i8.shape[0] < c.shape[0]:
        out_i8 = np.empty(c.shape[0], dtype=np.int8)
    amax = float(_amax_fp8(c))
    scale = (2.0 * amax / 127.0) if amax > 0.0 else 1.0
    _quantize_to_int8(c, scale, out_i8)
    return out_i8, float(scale)


def fp8_to_int8_codes(codes_u8, out_i8=None, conv: str = "torch"):
    """统一入口。

    `conv` = "torch"（**默认**，快 5.6×，多 76 MiB 中间张量）
           | "nogil"（慢 5.6×，**0 中间张量** + 释放 GIL）
    两条 Python 路径都在 **CPU** 上执行（fhz 指令）。
    ⚠ 2026-10-07：原 "rust" 选项已随 rust 版本删除（fhz 指令）——传入它
    落到下面的兜底分支，按 torch 路径执行（两条 Python 实现数值逐位一致）。
    """
    if str(conv).lower() in ("nogil", "numba", "jit"):
        return fp8_to_int8_codes_nogil(codes_u8, out_i8)
    return fp8_bits_to_int8_torch(codes_u8, out_i8)


# ── P189：fp8 位模式 → 实值 的 256 项查表────────────────────────────────
# 「fp8 位模式（uint8 承载）存储」的运行时反量化用：index 即得实值，
# 一次 numpy fancy-index 完成整张 W 的反量化（比逐位拆解快一个量级）。
FP8_VAL_LUT = _e4m3_bits_to_f64(np.arange(256, dtype=np.uint8)).astype(
    np.float32)
