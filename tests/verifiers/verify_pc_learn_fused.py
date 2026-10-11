"""M2 学习侧融合核（P59）对拍 + 计时——与 P52 推理融合核对称。

**逐位一致**（硬门槛）：融合核 vs 原「多核调用」路径，四个 CSR 权重的 `val`
数组必须**逐位相等**（`np.array_equal`，不是 allclose）。理由：行内遍历顺序、
`if ai == 0.0` 短路、同一 val 多次更新的次序全部照抄原核，行级 prange 的
写集不相交 ⇒ 应为逐位一致。

**计时**：best-of-3（本机单次噪声可达 3×，P15 教训）。CSR 规模取生产同量级
（n=8192、conn_k=16）。

运行：``python tests/verifiers/verify_pc_learn_fused.py``（退出码 0 = PASS）
"""
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from phdnet.sparse_pc import (NUMBA_OK, SparsePCStack,                # noqa: E402
                             _csr_add_outer, _csr_oja_up, _csr_clip,
                             _csr_matvec)

# ⚠ P15 教训：逐位对拍的基线**必须是 numba 串行版**。用纯 Python matvec 做
# 基线会因 fastmath 重结合/FMA 决策差 1 ulp，制造假阳性（本次实测：纯 Python
# 基线报 dn0/dn1 不一致，换 numba 基线后 max|Δ| = 0）。

CASES: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    CASES.append((name, bool(cond)))
    print(f"  {'✓' if cond else '✗'} {name}")


def _reference_learn(st, cache, eta_scale, homeostasis=False):
    """原「多核调用」路径（P59 之前的实现，逐行照抄）。"""
    s0, r1, r2 = cache["s0"], cache["r1"], cache["r2"]
    e0, e1 = cache["e0"], cache["e1"]
    if homeostasis:
        e0 = e0 / max(float(np.linalg.norm(e0)), 1e-9)
        e1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
    eta = st.eta_pc * eta_scale
    _csr_add_outer(*st.dn0, e0, r1, eta)
    _csr_add_outer(*st.dn1, e1, r2, eta)
    eo = st.eta_oja * eta_scale
    _csr_oja_up(*st.up0, r1, s0, eo)
    _csr_oja_up(*st.up1, r2, r1, eo)
    for t in (st.up0, st.up1, st.dn0, st.dn1):
        _csr_clip(t[2], st.w_max)
    if homeostasis:                       # 与 SparsePCStack.learn 同流程
        st._homeostatic_scale()


def _reference_learn_pred(st, prev, cache, eta_scale, mix, homeostasis=False):
    s0, r1, r2 = cache["s0"], cache["r1"], cache["r2"]
    p_r1, p_r2 = prev["r1"], prev["r2"]
    e0, e1 = cache["e0"], cache["e1"]
    if homeostasis:
        e0 = e0 / max(float(np.linalg.norm(e0)), 1e-9)
        e1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
    e0p = s0 - _csr_matvec_ref(*st.dn0, p_r1)
    e1p = r1 - _csr_matvec_ref(*st.dn1, p_r2)
    if homeostasis:
        e0p = e0p / max(float(np.linalg.norm(e0p)), 1e-9)
        e1p = e1p / max(float(np.linalg.norm(e1p)), 1e-9)
    eta = st.eta_pc * eta_scale
    om = 1.0 - mix
    _csr_add_outer(*st.dn0, e0, r1, eta * om)
    _csr_add_outer(*st.dn0, e0p, p_r1, eta * mix)
    _csr_add_outer(*st.dn1, e1, r2, eta * om)
    _csr_add_outer(*st.dn1, e1p, p_r2, eta * mix)
    eo = st.eta_oja * eta_scale
    _csr_oja_up(*st.up0, r1, s0, eo)
    _csr_oja_up(*st.up1, r2, r1, eo)
    for t in (st.up0, st.up1, st.dn0, st.dn1):
        _csr_clip(t[2], st.w_max)
    if homeostasis:                       # 与 learn_predictive 同流程
        st._homeostatic_scale()


def _csr_matvec_ref(indptr, idx, val, x):
    return _csr_matvec(indptr, idx, val, x)


def _mk(n0, n1, n2, seed=0, conn_k=16):
    rng = np.random.default_rng(seed)
    # 2026-10-07：rust 后端已整体删除（M2 只剩 numba/numpy 一套实现），
    #   本文件「融合核 vs 原多核调用路径 逐位一致」的契约自然成立。
    return SparsePCStack(n0, n1, n2, eta_pc=0.05, eta_oja=0.01, rng=rng,
                         w_max=2.0, conn_k=conn_k)


def _cache(st, seed=1):
    rng = np.random.default_rng(seed)
    n1 = st.up0[0].shape[0] - 1
    n2 = st.up1[0].shape[0] - 1
    n0 = st.dn0[0].shape[0] - 1
    return {"s0": rng.normal(0, 1, n0), "r1": rng.normal(0, 1, n1),
            "r2": rng.normal(0, 1, n2), "e0": rng.normal(0, 1, n0),
            "e1": rng.normal(0, 1, n1)}


def _vals(st):
    return [t[2].copy() for t in (st.up0, st.up1, st.dn0, st.dn1)]


def _best_of(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _main() -> int:
    print("P59：M2 学习侧融合核 —— 逐位对拍 + 计时")
    if not NUMBA_OK:
        print("  numba 不可用：融合核走 Python 回退，跳过数值/性能验证")
        return 0

    n0, n1, n2 = 1024, 4096, 4096

    # ── 1. `learn` 逐位一致（含 homeostasis=False 走融合核的判定）────────
    st_a, st_b = _mk(n0, n1, n2, seed=0), _mk(n0, n1, n2, seed=0)
    cache = _cache(st_a, seed=1)
    st_a.learn(cache, eta_scale=1.0)              # 融合核
    _reference_learn(st_b, cache, 1.0)            # 原路径
    for va, vb in zip(_vals(st_a), _vals(st_b)):
        check("learn 逐位一致: up0/up1/dn0/dn1", np.array_equal(va, vb))

    # ── 2. homeostasis=True 仍走原路径（融合核不参与）──────────────────────
    st_c, st_d = _mk(n0, n1, n2, seed=2), _mk(n0, n1, n2, seed=2)
    cache2 = _cache(st_c, seed=3)
    st_c.learn(cache2, eta_scale=1.0, homeostasis=True)
    _reference_learn(st_d, cache2, 1.0, homeostasis=True)
    for va, vb in zip(_vals(st_c), _vals(st_d)):
        check("learn(homeostasis) 逐位一致: up0/up1/dn0/dn1", np.array_equal(va, vb))

    # ── 3. `learn_predictive` 回归护栏（P59 判定其融合版负收益 → 保持原路径，
    #        这里验证它与参考实现仍逐位一致，防将来误改）────────────────────
    st_e, st_f = _mk(n0, n1, n2, seed=4), _mk(n0, n1, n2, seed=4)
    cache3 = _cache(st_e, seed=5)
    prev = {"r1": np.random.default_rng(6).normal(0, 1, n1),
            "r2": np.random.default_rng(7).normal(0, 1, n2)}
    st_e.learn_predictive(prev, cache3, eta_scale=1.0, mix=0.5)
    _reference_learn_pred(st_f, prev, cache3, 1.0, 0.5)
    for va, vb in zip(_vals(st_e), _vals(st_f)):
        check("learn_predictive 逐位一致: up0/up1/dn0/dn1", np.array_equal(va, vb))

    # ── 4. 计时（best-of-3，生产同量级 CSR）──────────────────────────────
    big = _mk(8192, 8192, 8192, seed=8)
    cbig = _cache(big, seed=9)
    t_ref = _best_of(lambda: _reference_learn(big, cbig, 1.0))
    big2 = _mk(8192, 8192, 8192, seed=8)
    t_fus = _best_of(lambda: big2.learn(cbig, eta_scale=1.0))
    # 计时需从同一初始状态起（learn 会改权重）：用两份同 seed 的栈即可
    print(f"    learn           8 核 {t_ref*1e3:7.3f} ms | 融合 1 核 "
          f"{t_fus*1e3:7.3f} ms | {t_ref/max(t_fus,1e-12):.2f}×")

    n_ok = sum(1 for _, ok in CASES if ok)
    print(f"\n通过 {n_ok}/{len(CASES)}（另附 learn 多规模计时）")
    return 0 if n_ok == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
