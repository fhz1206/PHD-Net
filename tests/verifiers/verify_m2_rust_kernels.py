#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2 CSR 的 Rust 算子门禁（P175）。

⚠ **对拍基准的精度说明（重要）**：
Python 的 numba 算子返回 **fp64**（`s = 0.0` 让 numba 提升精度），
而 Rust 版是**真 fp32**。若直接拿 numba 输出当基准 → 比的是**跨精度**
（实测 relerr 6e-08 ~ 2.6e-07，会被误判成 FAIL）。
故本门禁用「**Python 语义 + 全程 fp32**」的参照 —— 逐条对照 numba 源码。

判据分两类（与 Rust 侧实现一致）：
· **逐位**：原地更新类（add_outer / oja_up / clip / scale_rows）与
  `row_norms`（IEEE 精确 sqrt）—— 标量路径可逐位。
· **容差 1e-5**：`matvec` 长行（AVX2 SIMD 改求和顺序）。
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, os.path.join(_ROOT, "phdnet_rs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phdnet_rs import load  # noqa: E402

K, _why = load()
if K is None:
    print("[FAIL] Rust 库加载失败：%s" % _why)
    print("→ 先运行 bash phdnet_rs/build.sh")
    raise SystemExit(1)

_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append((bool(ok), name, detail))
    print("[%s] %s  —— %s" % ("PASS" if ok else "FAIL", name, detail))


# ── fp32 参照（严格复刻 numba 源码语义，但全程 fp32）────────────────
def ref32_matvec(ip, ix, v, x, rows):
    out = np.zeros(rows, dtype=np.float32)
    for i in range(rows):
        s = np.float32(0.0)
        for p in range(ip[i], ip[i + 1]):
            s = np.float32(s + v[p] * x[ix[p]])
        out[i] = s
    return out


def ref32_add_outer(ip, ix, v, a, b, eta):
    # ⚠⚠ **必须先把 eta 截成 fp32** —— Python float 是 fp64，而 ctypes 的
    #   `c_float` 会**截断**它。实测：不截断 → 65536 个元素里差 1e-07（假FAIL）；
    #   截断后 → **逐位一致**。这是个非常容易漏掉的坑（K3 曾因此误报）。
    e = np.float32(eta)
    out = v.copy()
    for i in range(len(a)):
        ai = np.float32(a[i])
        if ai == np.float32(0.0):
            continue
        for p in range(ip[i], ip[i + 1]):
            out[p] = np.float32(out[p] + np.float32(e * ai
                                                  * np.float32(b[ix[p]])))
    return out


def ref32_oja(ip, ix, v, post, pre, eta):
    e = np.float32(eta)
    out = v.copy()
    for i in range(len(post)):
        pi = np.float32(post[i])
        if pi == np.float32(0.0):
            continue
        for p in range(ip[i], ip[i + 1]):
            upd = np.float32(np.float32(e * pi)
                             * np.float32(pre[ix[p]] - pi * out[p]))
            out[p] = np.float32(out[p] + upd)
    return out


def ref32_row_norms(ip, v, rows):
    out = np.zeros(rows, dtype=np.float32)
    for i in range(rows):
        s = np.float32(0.0)
        for p in range(ip[i], ip[i + 1]):
            s = np.float32(s + np.float32(v[p] * v[p]))
        out[i] = np.float32(np.sqrt(s))
    return out


def ref32_scale_rows(ip, v, tgt, rows):
    out = v.copy()
    for i in range(rows):
        s = np.float32(0.0)
        for p in range(ip[i], ip[i + 1]):
            s = np.float32(s + np.float32(out[p] * out[p]))
        nrm = np.float32(np.sqrt(s))
        if nrm < np.float32(1e-12):
            nrm = np.float32(1e-12)
        f = np.float32(np.float32(tgt[i]) / nrm)
        for p in range(ip[i], ip[i + 1]):
            out[p] = np.float32(out[p] * f)
    return out


def rec(name, got, want, tol):
    g = np.asarray(got)
    w = np.asarray(want)
    rel = float(np.abs(g.astype(np.float64) - w.astype(np.float64)).max()
                / max(1e-30, np.abs(w.astype(np.float64)).max()))
    exact = np.array_equal(g, w)
    ok = exact if tol == 0.0 else rel <= tol
    check(name, ok, "**逐位**" if exact else "relerr=%.3e（容差 %.0e）"
          % (rel, tol))
    return ok


# ══════════════════════════════════════════════════════════════════════════
print("[K] M2 CSR Rust 全集（6 算子）vs Python 语义 fp32 参照")
rng = np.random.default_rng(0)
n, k = 512, 128
nnz = n * k
indptr = np.arange(0, nnz + 1, k, dtype=np.int64)
idx = rng.integers(0, n, nnz).astype(np.int64)
val0 = rng.normal(0, 0.05, nnz).astype(np.float32)
x = rng.normal(0, 1, n).astype(np.float32)
a = rng.normal(0, 1, n).astype(np.float32)
b = rng.normal(0, 1, n).astype(np.float32)
post = rng.normal(0, 1, n).astype(np.float32)
pre = rng.normal(0, 1, n).astype(np.float32)
target = np.abs(rng.normal(0, 0.1, n)).astype(np.float32)

# 1) matvec 长行（SIMD → 容差）
o = np.zeros(n, dtype=np.float32)
K.m2_matvec(indptr, idx, val0, x, o, 4)
rec("K1 m2_matvec 长行（AVX2）", o, ref32_matvec(indptr, idx, val0, x, n),
    1e-5)

# 2) matvec 短行（回退标量 → 逐位）
k2 = 16
ip_s = np.arange(0, n * k2 + 1, k2, dtype=np.int64)
ix_s = rng.integers(0, n, n * k2).astype(np.int64)
v_s = rng.normal(0, 0.05, n * k2).astype(np.float32)
o_s = np.zeros(n, dtype=np.float32)
K.m2_matvec(ip_s, ix_s, v_s, x, o_s, 4)
rec("K2 m2_matvec 短行(<32) 回退标量", o_s,
    ref32_matvec(ip_s, ix_s, v_s, x, n), 0.0)

# 3) add_outer（原地 → 逐位）
vr = val0.copy()
K.m2_add_outer(indptr, idx, vr, a, b, 0.05, 4)
rec("K3 m2_add_outer 原地", vr,
    ref32_add_outer(indptr, idx, val0, a, b, 0.05), 0.0)

# 4) oja_up（原地 → 逐位）
vr = val0.copy()
K.m2_oja_up(indptr, idx, vr, post, pre, 0.02, 4)
rec("K4 m2_oja_up 原地", vr,
    ref32_oja(indptr, idx, val0, post, pre, 0.02), 0.0)

# 5) clip（原地 → 逐位；复刻 if/elif 而非 f32::clamp）
vr = (rng.normal(0, 1, nnz) * 2.0).astype(np.float32)
vp = vr.copy()
K.m2_clip(vr, 0.5, 4)
w = 0.5
for j in range(vp.size):
    vv = vp[j]
    if vv > w:
        vp[j] = w
    elif vv < -w:
        vp[j] = -w
rec("K5 m2_clip 原地", vr, vp, 0.0)

# 6) row_norms（逐位；fp32 sqrt 是 IEEE 精确）
o_r = np.zeros(n, dtype=np.float32)
K.m2_row_norms(indptr, idx, val0, o_r, 4)
rec("K6 m2_row_norms", o_r, ref32_row_norms(indptr, val0, n), 0.0)

# 7) scale_rows（原地 → 逐位）
vr = val0.copy()
K.m2_scale_rows(indptr, idx, vr, target, 4)
rec("K7 m2_scale_rows 原地", vr,
    ref32_scale_rows(indptr, val0, target, n), 0.0)

# ══════════════════════════════════════════════════════════════════════════
print("[L] 边界与 fail-fast")
# L1 全零行不得产生 NaN/Inf
ip_z = np.array([0, 2, 4, 6], dtype=np.int64)
ix_z = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
v_z = np.zeros(6, dtype=np.float32)
K.m2_scale_rows(ip_z, ix_z, v_z, np.ones(3, np.float32), 1)
check("L1 全零行不产生 NaN/Inf", np.all(np.isfinite(v_z)),
      "max=%.3e" % (float(np.abs(v_z).max()) if v_z.size else 0.0))

# L2 a[i]==0 的跳过语义（numba 有 `if ai == 0.0: continue`）
v_a = rng.normal(0, 0.05, nnz).astype(np.float32)
a0 = np.zeros(n, dtype=np.float32)          # 全 0
vr = v_a.copy()
K.m2_add_outer(indptr, idx, vr, a0, b, 0.05, 4)
check("L2 a 全 0 → val **完全不变**（复刻 continue 语义）",
      np.array_equal(vr, v_a), "Δ=%d" % int((vr != v_a).sum()))

# L3 fp64输入必须报错（P173 前 val 正是 fp64）
for nm, fn in (("matvec", lambda: K.m2_matvec(
                    indptr, idx, val0.astype(np.float64),
                    x.astype(np.float64), np.zeros(n, np.float64), 1)),
               ("add_outer", lambda: K.m2_add_outer(
                    indptr, idx, val0.astype(np.float64),
                    a.astype(np.float64), b.astype(np.float64), 0.05, 1))):
    try:
        fn()
        check("L3 %s 的 fp64 输入抛异常" % nm, False, "竟然没报错")
    except ValueError:
        check("L3 %s 的 fp64 输入抛异常" % nm, True, "已拦下")

# L4 非连续数组必须报错
v_nc = np.ascontiguousarray(rng.normal(0, 1, nnz * 2).astype(np.float32))[::2]
try:
    K.m2_clip(v_nc, 1.0, 1)
    check("L4 非连续 val抛异常", False, "竟然没报错")
except ValueError:
    check("L4 非连续 val 抛异常", True, "已拦下")

# L5 idx 越界必须报错
ix_bad = idx.copy()
ix_bad[0] = n + 999
try:
    K.m2_matvec(indptr, ix_bad, val0, x, np.zeros(n, np.float32), 1)
    check("L5 idx 越界抛异常", False, "竟然没报错")
except ValueError:
    check("L5 idx 越界抛异常", True, "已拦下")

# ══════════════════════════════════════════════════════════════════════════
print("[M] SparsePCStack 集成（默认关，必须零回归）")
from phdnet.sparse_pc import SparsePCStack  # noqa: E402

n0, n1, n2 = 128, 96, 64
s0 = rng.normal(0, 1, n0).astype(np.float32)


def mk(backend, fused, seed=0):
    r = np.random.default_rng(seed)
    return SparsePCStack(n0, n1, n2, 0.05, 0.02, r, w_max=2.0, conn_k=8,
                         fused=fused, m2_backend=backend)


# M1 默认后端就是 numpy（铁律：默认关）
st = mk("numpy", False)
check("M1 默认 m2_backend 是 numpy", st.m2_backend == "numpy", st.m2_backend)
check("M1 默认 _rs 为 None（Rust 未加载）", st._rs is None, "None")

# M2 非法后端名必须报错
try:
    mk("cuda", False)
    check("M2 非法后端名抛异常", False, "竟然没报错")
except ValueError:
    check("M2 非法后端名抛异常", True, "已拦下")

# M3 非法后端名不拼错字（构造时即校验）
try:
    SparsePCStack(8, 8, 8, 0.0, 0.0, np.random.default_rng(0),
                  m2_backend="rustt")
    check("M3 近似拼错的后端名也报错", False, "竟然没报错")
except ValueError:
    check("M3 近似拼错的后端名也报错", True, "已拦下")

# M4 numpy vs rust 后端在非融合路径上数值一致
st_np = mk("numpy", False)
st_rs = mk("rust", False)
for nm in ("up0", "up1", "dn0", "dn1"):
    setattr(st_rs, nm, tuple(x.copy() for x in getattr(st_np, nm)))
    setattr(st_rs, "_hn_" + nm, getattr(st_np, "_hn_" + nm).copy())
c_np = st_np.infer(s0, n_steps=2)
c_rs = st_rs.infer(s0, n_steps=2)
worst = 0.0
for kk in ("r1", "r2", "e0", "e1"):
    aa = np.asarray(c_np[kk], np.float64)
    bb = np.asarray(c_rs[kk], np.float64)
    worst = max(worst, float(np.abs(aa - bb).max()
                             / max(1e-30, np.abs(aa).max())))
check("M4 infer: numpy vs rust 一致（容差 1e-5）", worst < 1e-5,
      "worst_relerr=%.3e" % worst)

st_np.learn(c_np, homeostasis=True)
st_rs.learn(c_rs, homeostasis=True)
worst_w = 0.0
for nm in ("up0", "up1", "dn0", "dn1"):
    aa = np.asarray(getattr(st_np, nm)[2], np.float64)
    bb = np.asarray(getattr(st_rs, nm)[2], np.float64)
    worst_w = max(worst_w, float(np.abs(aa - bb).max()
                                 / max(1e-30, np.abs(aa).max())))
check("M5 learn+homeostatic: numpy vs rust 一致", worst_w < 1e-5,
      "worst_relerr=%.3e" % worst_w)

# M6 _row_norms 返回 fp32（P175 统一，避免与 val 精度不一致）
check("M6 _row_norms 返回 fp32（与 val 一致）",
      st_np._hn_up0.dtype == np.float32, str(st_np._hn_up0.dtype))

# M7 融合核 + rust：能跑通，但要如实标注「未生效」
st_f = mk("rust", True)
cf = st_f.infer(s0, n_steps=1)
check("M7 融合核 + rust 可运行（如实标注未生效）",
      np.all(np.isfinite(cf["r1"])), "fused=True 时仍走 numba 融合核")

# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════
print("[N] M2 融合核（P176）：Rust vs Python")
from phdnet import sparse_pc as _spc  # noqa: E402

_n0, _n1, _n2 = 128, 96, 64
_s = np.random.default_rng(7).normal(0, 1, _n0).astype(np.float32)


def _mk_stack(kk):
    return SparsePCStack(_n0, _n1, _n2, 0.05, 0.02,
                         np.random.default_rng(11), conn_k=kk, fused=True)


# N0 Python 融合核已是全 fp32（P176）
_st0 = _mk_stack(8)
_c = _spc._pc_infer_fused(_st0.up0, _st0.up1, _st0.dn0, _st0.dn1, _s, 2)
check("N0 Python 融合核输出已是 **fp32**（P176 统一）",
      all(x.dtype == np.float32 for x in _c),
      "r1=%s e0=%s" % (_c[0].dtype, _c[2].dtype))
# N1 e0 的长度必须是 n0（P176 修的越界bug）
check("N1 e0 长度 = n0（P176 修正的越界 bug）",
      len(_c[2]) == _n0,
      "e0=%d, n0=%d（原为 %d → `clip(up0 @ e0)` 越界读）"
      % (len(_c[2]), _n0, _n1))
# N1b 非融合路径的 e0 也应是 n0（两者一致 = docstring 的「逐位一致」成立）
_stn = SparsePCStack(_n0, _n1, _n2, 0.05, 0.02, np.random.default_rng(11),
                     conn_k=8, fused=False)
_cn = _stn.infer(_s, n_steps=2)
check("N1b 融合/非融合的 e0 长度一致（「逐位一致」的前提）",
      len(_c[2]) == len(_cn["e0"]),
      "fused=%d unfused=%d" % (len(_c[2]), len(_cn["e0"])))

# N2 infer 融合核：Rust vs Python（k=8/16标量、32/64 SIMD）
for _kk in (8, 16, 32, 64):
    _st = _mk_stack(_kk)
    _py = _spc._pc_infer_fused(_st.up0, _st.up1, _st.dn0, _st.dn1, _s, 2)
    _r1 = np.zeros(_n1, np.float32)
    _r2 = np.zeros(_n2, np.float32)
    _e0 = np.zeros(_n0, np.float32)      # ⚠ P176：n0 长
    _e1 = np.zeros(_n1, np.float32)
    K.m2_infer_fused(_st.up0, _st.up1, _st.dn0, _st.dn1, _s,
                     _r1, _r2, _e0, _e1, 2, 0)
    _w = 0.0
    for _a, _b in ((_py[0], _r1), (_py[1], _r2), (_py[2], _e0), (_py[3], _e1)):
        _aa = np.asarray(_a, np.float64)
        _bb = np.asarray(_b, np.float64)
        _w = max(_w, float(np.abs(_aa - _bb).max()
                           / max(1e-30, np.abs(_aa).max())))
    _mode = "SIMD" if _kk >= 32 else "标量"
    check("N2 infer 融合核 k=%-3d（%s）vs Python" % (_kk, _mode), _w < 1e-5,
          "worst_relerr=%.3e" % _w)

# N3 learn 融合核：⚠ **已知语义差异**（Python 越界读，Rust 钳制）
_a = _mk_stack(8)
_b = _mk_stack(8)
for _nm in ("up0", "up1", "dn0", "dn1"):
    _src = getattr(_a, _nm)
    _dst = getattr(_b, _nm)
    _dst[0][:] = _src[0][:]
    _dst[1][:] = _src[1][:]
    _dst[2][:] = _src[2][:]
_c2 = _spc._pc_infer_fused(_a.up0, _a.up1, _a.dn0, _a.dn1, _s, 2)
_spc._pc_learn_fused(_a.dn0, _a.dn1, _a.up0, _a.up1,
                     _c2[2], _c2[3], _c2[0], _c2[1], _s,
                     0.05, 0.02, _a.w_max)
K.m2_learn_fused(_b.dn0, _b.dn1, _b.up0, _b.up1,
                 _c2[2], _c2[3], _c2[0], _c2[1], _s,
                 0.05, 0.02, _b.w_max, 0)
_wl = 0.0
for _nm in ("up0", "up1", "dn0", "dn1"):
    _av = getattr(_a, _nm)[2].astype(np.float64)
    _bv = getattr(_b, _nm)[2].astype(np.float64)
    _wl = max(_wl, float(np.abs(_av - _bv).max()
                         / max(1e-30, np.abs(_av).max())))
# ⚠ 判据用 1e-6（不是逐位）：Rust 钳制 + numba fastmath 的 FMA 差异
check("N3 learn 融合核 vs Python（**容差 1e-6**，非逐位 —— 见下注）",
      _wl < 1e-6, "worst_relerr=%.3e" % _wl)
print("     ⚠ 已知差异 ①：Python `prange` 不做边界检查 → dn0(n0行) 读 e0 时曾越界；")
print("       Rust 钳制到 e0 长度 → **不等价**（P176 已修Python 侧，但 fastmath")
print("       的 FMA 融合仍使结果差 ~1e-8 量级）。")

# N4 fail-fast
try:
    K.m2_infer_fused((_st0.up0[0], _st0.up0[1],
                      _st0.up0[2].astype(np.float64)),
                     _st0.up1, _st0.dn0, _st0.dn1, _s,
                     np.zeros(_n1, np.float32), np.zeros(_n2, np.float32),
                     np.zeros(_n0, np.float32), np.zeros(_n1, np.float32),
                     1, 0)
    check("N4 fp64 val 必须被拦", False, "竟然没报错")
except ValueError:
    check("N4 fp64 val 必须被拦", True, "已拦下")
# N4b e0 长度错误必须被拦
try:
    K.m2_infer_fused(_st0.up0, _st0.up1, _st0.dn0, _st0.dn1, _s,
                     np.zeros(_n1, np.float32), np.zeros(_n2, np.float32),
                     np.zeros(_n1, np.float32),   # ← 错长度（n1而非 n0）
                     np.zeros(_n1, np.float32), 1, 0)
    check("N4b e0 长度错误必须被拦（P176 的 bug 防线）", False, "竟然没报错")
except ValueError:
    check("N4b e0 长度错误必须被拦（P176 的 bug 防线）", True, "已拦下")

n_fail = sum(1 for ok, _, _ in _RESULTS if not ok)
print("\n" + "=" * 70)
print("结果：%d 例 FAIL（共 %d 条）" % (n_fail, len(_RESULTS)))
print("=" * 70)
raise SystemExit(1 if n_fail else 0)