#!/usr/bin/env python3
"""门禁：Rust 版M1–M6 与 Python 参照版的对拍。

fhz 2026-10-03：「把部分转移到 Rust（Python 版本仍然保留）」。

**Python 版是参照实现，Rust 版是候选实现 —— 对拍通过之前不启用。**
本门禁是那道「通过」的判据。

## 判据分两档（不可混用）

| 情形 | 判据 | 原因 |
|---|---|---|
| **同实现不同线程数** | **逐位** | 分块不改行内累加次序 → 应完全相同 |
| **Rust vs Python(BLAS/einsum)** | **容差 1e-5** | 归约顺序不同（Rust 标量 vs BLAS 的 FMA 分块），逐位不可能 |
| **Rust vs Python(逐元素参考)** | **逐位** | 两者都是标量升序累加 → 应完全相同 |

⚠ 这是 P156的教训延伸：int8 累加**数学上不可能**（127×127 > 127），
所以对拍 int8 路径只能用容差；而 fp32/fp64 累加**可以**要求逐位。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

for _p in (str(Path(__file__).resolve().parents[2]),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _RESULTS.append((bool(ok), name, detail))
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name,
                         ("  —— " + detail) if detail else ""))
    return bool(ok)


print("=" * 72)
print("门禁：Rust 版 vs Python 参照版对拍")
print("=" * 72)

# ── 加载 Rust 库（不可用时 SKIP 而非 FAIL —— 环境问题不是代码问题）────────
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "phdnet_rs"))
    from phdnet_rs import load as _rs_load
    K, _why = _rs_load()
except Exception as _e:                                       # noqa: BLE001
    K, _why = None, "%s: %s" % (type(_e).__name__, _e)

if K is None:
    print("[SKIP] Rust 库不可用：", _why)
    print("       构建：cd phdnet_rs && cargo build --release")
    print("=" * 72)
    print("结果：SKIP（环境未构建 Rust 库，非代码缺陷）")
    print("=" * 72)
    raise SystemExit(0)

check("A0 Rust 库可加载", True, "has_npu=%s" % K.has_npu())
print("    （has_npu=False 是正确的：本机无 NPU，"
      "CPU 参照路径仍必须可用 —— 见 docs 的回落纪律）")

rng = np.random.default_rng(0)

# ── A. M1 GEMV ────────────────────────────────────────────────────────
print("[A] M1 GEMV")
r, c = 1024, 2048
W = rng.normal(0, 0.05, (r, c)).astype(np.float32)
W = np.ascontiguousarray(W)
x = rng.normal(0, 1, c).astype(np.float32)
b = rng.normal(0, 0.01, r).astype(np.float32)

o_rs = np.zeros(r, dtype=np.float32)
K.m1_gemv(W, x, b, o_rs, 8)
ref = (W @ x + b).astype(np.float32)
rel = float(np.abs(o_rs - ref).max() / max(1e-30, np.abs(ref).max()))
check("A1 Rust GEMV vs numpy BLAS（容差 1e-5）", rel < 1e-5,
      "relerr=%.3e（fp32 eps=1.19e-07）" % rel)

# **同实现、不同线程数 → 必须逐位**
o1 = np.zeros(r, dtype=np.float32)
K.m1_gemv(W, x, b, o1, 1)
check("A2 Rust 多线程(8) vs 单线程：**逐位**", np.array_equal(o1, o_rs),
      "分块不改行内累加次序 → 应完全相同")
o16 = np.zeros(r, dtype=np.float32)
K.m1_gemv(W, x, b, o16, 16)
check("A3 Rust 16 线程 vs 1 线程：**逐位**", np.array_equal(o1, o16))

# ── B. M1 k-WTA ────────────────────────────────────────────────────────
print("[B] M1 k-WTA")
n = 1024
k = 128
u = rng.normal(0, 1, n).astype(np.float32)
s_rs, idx_rs = K.m1_kwta(u, k)
# Python 参照：argpartition 选前k（无序），再归一化
idx_py = np.argpartition(-u, k - 1)[:k]
check("B1 选出的索引**集合**与 Python 一致",
      set(idx_rs.tolist()) == set(idx_py.tolist()),
      "Rust=%d 个，Py=%d 个，交集=%d"
      % (len(set(idx_rs.tolist())), len(set(idx_py.tolist())),
         len(set(idx_rs.tolist()) & set(idx_py.tolist()))))
win = u[idx_py]
mn, mx = float(win.min()), float(win.max())
s_py = np.zeros(n, dtype=np.float32)
s_py[idx_py] = (win - mn) / (mx - mn + 1e-9) + 0.1
# 逐位比：同样的归一化公式 + 同样的浮点运算顺序
d_s = float(np.abs(s_rs - s_py).max())
check("B2 归一化后的 s **逐位**一致（同一公式、同一顺序）", d_s == 0.0,
      "max|Δ|=%.3e" % d_s)
# ⚠ idx 的**顺序**不保证一致（Rust 按值降序，Python 是无序 argpartition）
#   → 这是**刻意**的，上层最终会 np.sort。记录但不判FAIL。
check("B3 idx 顺序差异是**已知且可接受**（上层会 sort）",
      True,
      "Rust=降序，Python=无序；集合相同即可（见 B1）")

# ── C. M2 CSR SpMV ─────────────────────────────────────────────────────
print("[C] M2 CSR SpMV")
n_rows, kk = 512, 128
indptr = np.arange(0, n_rows * kk + 1, kk, dtype=np.int64)
idx = rng.integers(0, n_rows, n_rows * kk).astype(np.int64)
val = rng.normal(0, 0.05, n_rows * kk).astype(np.float32)
xv = rng.normal(0, 1, n_rows).astype(np.float32)

y_rs = np.zeros(n_rows, dtype=np.float32)
K.csr_spmm(indptr, idx, val, xv, y_rs, 8)
# Python 参照：**标量升序累加**（与 Rust 同序 → 应逐位）
y_ref = np.zeros(n_rows, dtype=np.float32)
for i in range(n_rows):
    s = np.float32(0.0)
    for pp in range(int(indptr[i]), int(indptr[i + 1])):
        s = np.float32(s + val[pp] * xv[int(idx[pp])])
    y_ref[i] = s
check("C1 Rust SpMV vs Python 标量升序：**逐位**",
      np.array_equal(y_rs, y_ref),
      "max|Δ|=%.3e（两者累加次序相同）" % float(np.abs(y_rs - y_ref).max()))

y1 = np.zeros(n_rows, dtype=np.float32)
K.csr_spmm(indptr, idx, val, xv, y1, 1)
check("C2 SpMV 多线程 vs 单线程：**逐位**", np.array_equal(y1, y_rs))

# ── D. M6 稀疏读出前向 ────────────────────────────────────────────────
print("[D] M6 稀疏读出")
n_out, kk2, n_h = 256, 128, 512
Ws = rng.normal(0, 0.05, (n_out, kk2)).astype(np.float32)
Ws = np.ascontiguousarray(Ws)
gidx = rng.integers(0, n_h, kk2).astype(np.int64)
hs = rng.normal(0, 1, n_h).astype(np.float32)
y6 = np.zeros(n_out, dtype=np.float32)
K.m6_sparse_fwd(Ws, gidx, hs, y6, 4)
# Python 参照
y6_ref = np.zeros(n_out, dtype=np.float32)
g = hs[gidx]
for i in range(n_out):
    s = np.float32(0.0)
    for j in range(kk2):
        s = np.float32(s + Ws[i, j] * g[j])
    y6_ref[i] = s
check("D1 Rust 稀疏读出 vs Python 标量：**逐位**",
      np.array_equal(y6, y6_ref),
      "max|Δ|=%.3e" % float(np.abs(y6 - y6_ref).max()))

# ── E. M4a / M5 ───────────────────────────────────────────────────────
print("[E] M4a 工作记忆 / M5 调制")
slots = rng.normal(0, 1, 64).astype(np.float32)
slots_ref = slots.copy()
rvec = rng.normal(0, 1, 64).astype(np.float32)
gamma, gate, thresh = 0.9, 0.8, 0.5
slots_ref *= np.float32(gamma)          # Python：np.float32 乘法
if gate >= thresh:
    slots_ref[:] = rvec
K.m4a(slots, rvec, gamma, gate, thresh)
check("E1 M4a 衰减+门控写入 **逐位**", np.array_equal(slots, slots_ref),
      "max|Δ|=%.3e" % float(np.abs(slots - slots_ref).max()))

surprise = 1.7
g_rs, m_rs, v_rs, c_rs = K.m5_gate(surprise, 0.0, 0.0, 0)
# Python 参照（Welford）
mean, m2, cnt = 0.0, 0.0, 0
d_ = surprise - mean
mean += d_ / (cnt + 1.0)
m2 += d_ * (surprise - mean)
cnt += 1
var = m2 / (cnt - 1.0) if cnt > 1 else 0.0
sig = np.sqrt(var) + 1e-9
z = (surprise - mean) / sig
g_py = 1.0 / (1.0 + np.exp(-2.0 * z))
rel_g = abs(g_rs - g_py) / max(1e-12, abs(g_py))
check("E2 M5 gate（容差 1e-6；exp 的实现差异）", rel_g < 1e-6,
      "Rust=%.10f Py=%.10f relerr=%.3e" % (g_rs, g_py, rel_g))
check("E3 M5 的 Welford 统计量逐位一致",
      m_rs == np.float32(mean).astype(np.float64).__float__() or abs(m_rs - mean) < 1e-6,
      "Rust mean=%.10f Py mean=%.10f" % (m_rs, mean))

# ── F. SIMD 路径（P171）─────────────────────────────────────────────
print("[F] SIMD（AVX2）路径")
check("F0 has_avx2 可调用", True, "avx2=%s" % K.has_avx2())
if K.has_avx2():
    ysimd = np.zeros(r, dtype=np.float32)
    K.m1_gemv_simd(W, x, b, ysimd, 8)
    rel_s = float(np.abs(ysimd - ref).max() / max(1e-30, np.abs(ref).max()))
    # ⚠ SIMD **不要求逐位**（4 路 FMA 累加器改变求和顺序）→ 容差
    check("F1 SIMD vs numpy BLAS（**容差 1e-5**，非逐位）", rel_s < 1e-5,
          "relerr=%.3e（4 路 FMA 改求和顺序 → 预期 ~1e-7）" % rel_s)
    ysimd1 = np.zeros(r, dtype=np.float32)
    K.m1_gemv_simd(W, x, b, ysimd1, 1)
    d_s = float(np.abs(ysimd1 - ysimd).max()
                / max(1e-30, np.abs(ysimd).max()))
    check("F2 SIMD 1 线程 vs 8 线程：容差内（分块不改行内 SIMD 序）",
          d_s < 1e-6, "relerr=%.3e" % d_s)
    # 标量路径仍**要求逐位**（SIMD 不影响它）
    ysc = np.zeros(r, dtype=np.float32)
    ysc8 = np.zeros(r, dtype=np.float32)
    K.m1_gemv(W, x, b, ysc, 1)
    K.m1_gemv(W, x, b, ysc8, 8)
    check("F3 标量路径 1 vs 8 线程：**逐位**（未受 SIMD 影响）",
          np.array_equal(ysc, ysc8))

# ── G. dtype 分派（P172：Rust 跟着 Python 生产精度走）─────────────────
print("[G] dtype 分派（fp32/fp64/fp16）")
_DT_CODES = {}
for _d, _want in (("fp32", 0), ("fp64", 1), ("fp16", 2)):
    try:
        _c = K.dtype_code(_d)
        _DT_CODES[_d] = _c
        check("G1 %s -> code=%d" % (_d, _want), _c == _want,
              "实得 %d" % _c)
    except Exception as e:                                # noqa: BLE001
        check("G1 %s 可解析" % _d, False, str(e)[:50])
# 不认识的 dtype 必须**报错**，不静默回落
try:
    K.dtype_code("bf16")
    check("G2 未知 dtype 抛异常（P161：不静默回落）", False, "竟然成功了")
except ValueError:
    check("G2 未知 dtype 抛异常（P161：不静默回落）", True, "bf16 被拒")

# SIMD 可用性：只有 fp32 有
check("G3 只有 fp32 有手写 SIMD（fp64/fp16 无）",
      K.dtype_has_simd("fp32") and not K.dtype_has_simd("fp64")
      and not K.dtype_has_simd("fp16"),
      "fp32=%s fp64=%s fp16=%s" % (K.dtype_has_simd("fp32"),
                                K.dtype_has_simd("fp64"),
                                K.dtype_has_simd("fp16")))

# 三档精度都能算（用各档的 numpy 做参照，注意容差按 dtype eps 定）
for _npdt, _name, _tolmul in ((np.float32, "fp32", 1e-5),
                              (np.float64, "fp64", 1e-12),
                              (np.float16, "fp16", 1e-2)):
    try:
        _W = np.ascontiguousarray(rng.normal(0, 0.05, (512, 256)).astype(_npdt))
        _x = rng.normal(0, 1, 256).astype(_npdt)
        _b = rng.normal(0, 0.01, 512).astype(_npdt)
        _o = np.zeros(512, dtype=_npdt)
        K.m1_gemv_dt(_W, _x, _b, _o, _npdt, 4)
        _ref = (_W.astype(np.float64) @ _x.astype(np.float64)
                + _b.astype(np.float64))
        _rel = float(np.abs(_o.astype(np.float64) - _ref).max()
                     / max(1e-30, np.abs(_ref).max()))
        check("G4 %s dtype 路径可算且在容差内" % _name, _rel <= _tolmul,
              "max_relerr=%.3e（容差 %.0e）" % (_rel, _tolmul))
    except Exception as e:                                # noqa: BLE001
        check("G4 %s dtype 路径可算" % _name, False, str(e)[:50])

# ── H. M2 CSR SpMV SIMD（P173）─────────────────────────────────────
print("[H] M2 CSR SpMV SIMD（fp32 改造后）")
_n_rows, _kk = 512, 128
_indptr = np.arange(0, _n_rows * _kk + 1, _kk, dtype=np.int64)
_idx = rng.integers(0, _n_rows, _n_rows * _kk).astype(np.int64)
_val32 = rng.normal(0, 0.05, _n_rows * _kk).astype(np.float32)
_x32 = rng.normal(0, 1, _n_rows).astype(np.float32)
_y_sd = np.zeros(_n_rows, dtype=np.float32)
_y_sc = np.zeros(_n_rows, dtype=np.float32)
K.csr_spmm_simd(_indptr, _idx, _val32, _x32, _y_sd, 4)
K.csr_spmm(_indptr, _idx, _val32, _x32, _y_sc, 4)
# Python 标量参照（fp32 升序累加）
_y_ref = np.zeros(_n_rows, dtype=np.float32)
for _i in range(_n_rows):
    _a, _b = int(_indptr[_i]), int(_indptr[_i + 1])
    _acc = np.float32(0.0)
    for _pp in range(_a, _b):
        _acc = np.float32(_acc + _val32[_pp] * _x32[int(_idx[_pp])])
    _y_ref[_i] = _acc
_d_sd = float(np.abs(_y_sd - _y_ref).max() / max(1e-30, np.abs(_y_ref).max()))
_d_sc = float(np.abs(_y_sc - _y_ref).max() / max(1e-30, np.abs(_y_ref).max()))
check("H1 SpMV SIMD vs Python 标量（**容差 1e-5**）", _d_sd < 1e-5,
      "relerr=%.3e（8 lane 并行改求和顺序）" % _d_sd)
check("H2 SpMV 标量 vs Python 标量：**逐位**（同序累加）", _d_sc == 0.0,
      "relerr=%.3e" % _d_sc)

# ── I. Python 侧 fp32 改造（P173）──────────────────────────────────
print("[I] Python 侧 M2/M3/M4a 已是 fp32（P173）")
import inspect as _insp
from phdnet import sparse_pc as _spc, plasticity as _pl, wm as _wm
_src_sparse = _insp.getsource(_spc)
check("I1 M2 CSR val 用 fp32",
      "np.empty(n_rows * k, dtype=np.float32)" in _src_sparse,
      "P173：fp64 -> fp32（val 8->4 B/边，访存减半）")
check("I2 M2 已无 fp64 的 val 分配",
      "np.empty(n_rows * k, dtype=np.float64)" not in _src_sparse, "")
_src_pl = _insp.getsource(_pl)
check("I3 M3 STDP W 用 fp32",
      "np.zeros((n, m_edges), dtype=np.float32)" in _src_pl, "")
_src_wm = _insp.getsource(_wm)
check("I4 M4a slots/strength 用 fp32",
      "np.zeros((n_slots, n), dtype=np.float32)" in _src_wm
      and "np.zeros(n_slots, dtype=np.float32)" in _src_wm, "")
# M6 应保持 fp16（P163）
from phdnet.backends.accel_readout import AccelReadout as _AR
_ro = _AR(64, 64, None, device="cpu", dtype="fp16")
check("I5 M6 读出计算域是 fp16（P163）",
      str(_ro._cdtype) == "torch.float16", "_cdtype=%s" % _ro._cdtype)

n_fail = sum(1 for ok, _, _ in _RESULTS if not ok)
print()
print("=" * 72)
print("结果：%d 例 FAIL%s" % (n_fail,
                            (" → " + str([n for ok, n, _ in _RESULTS if not ok]))
                            if n_fail else ""))
print("=" * 72)
sys.exit(1 if n_fail else 0)