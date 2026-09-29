"""P68：M4b 的 numba 核逐位对拍 + 计时（`encode` 去重 / `recall` 反投影）。

fhz 2026-09-29：「给可以用 numba 来加速的都用上」。本脚本验证 `phdnet/
ltm_kernel.py` 的两个 nogil 核与原 Python 实现**逐位一致**（不是 allclose），
并给出服务器形态的计时。

关键约束（P68 设计）：
- `recall` 的输出维度被多个 big_i 共享 → 核必须**串行**（prange 会竞态），
  收益来自消除解释器开销 + 释放 GIL；
- `rev` 是静态的 → 预 CSR 化零维护成本；越界 big_i 跳过 = 原 `.get((), )`。

运行：`python tests/verifiers/verify_ltm_kernels.py`（退出码 0 = PASS）
"""
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import phdnet.bigltm as bigltm_mod                                # noqa: E402
from phdnet.bigltm import SparseLTM                               # noqa: E402

CASES: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    CASES.append((name, bool(cond)))
    print(f"  {'✓' if cond else '✗'} {name}")


def _mk_ltm(n_dim=1024, k_hash=4, n_neurons=1 << 20, seed=0):
    return SparseLTM(n_dim=n_dim, n_neurons=n_neurons, m_out=72,
                     k_hash=k_hash, lam=0.7, eta=0.08, seed=seed)


def _py_encode(ltm: SparseLTM, rate):
    """原 Python 实现（基线）。"""
    dims = np.nonzero(rate > 0.0)[0]
    if dims.size == 0:
        return []
    out, seen = [], set()
    for j in dims.tolist():
        for i in ltm.idx[j].tolist():
            if i not in seen:
                seen.add(i)
                out.append(i)
    return out


def _py_recall_project(ltm: SparseLTM, scores):
    """原 Python 反投影（基线）。"""
    out = np.zeros(ltm.n_dim)
    for big_i, s in scores.items():
        for j in ltm.rev.get(big_i, ()):
            out[j] += s
    return out


def _best(fn, reps=5):
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b


def _main() -> int:
    print("P68/P70：M4b numba 核 —— 逐位对拍 + 计时（recall 路径全 numba；encode 已回滚）")
    rng = np.random.default_rng(0)
    ltm = _mk_ltm()

    # ── 1. rev 的 CSR 化与 dict 版等价（逐键逐维）─────────────────────────
    ok = True
    for big_i, js in ltm.rev.items():
        p0 = ltm._rev_indptr[big_i]
        p1 = ltm._rev_indptr[big_i + 1]
        if list(ltm._rev_indices[p0:p1]) != list(js):
            ok = False
            break
    check("rev CSR 化与 dict 逐维一致（顺序也一致）", ok)

    # ── 2. encode 逐位一致（多种稀疏度）───────────────────────────────────
    for dims_n in (1, 8, 64, 256):
        rate = np.zeros(ltm.n_dim)
        rate[rng.choice(ltm.n_dim, size=dims_n, replace=False)] = rng.random(dims_n)
        got = ltm.encode(rate)
        exp = _py_encode(ltm, rate)
        check(f"encode 逐位一致（激活 {dims_n} 维）", got == exp)

    # ── 3. recall 反投影逐位一致（含越界 big_i 与空绑定）──────────────────
    for n_scores, with_oor in ((10, False), (200, False), (50, True)):
        keys = rng.choice(1 << 20, size=n_scores, replace=False)
        if with_oor:                       # 混入几乎肯定没有绑定的超大索引
            keys = np.concatenate([keys, np.array([1 << 19, (1 << 20) - 1])])
        scores = {int(k): float(rng.random()) for k in keys}
        out_num = np.zeros(ltm.n_dim)
        ks = np.fromiter(scores.keys(), dtype=np.int64, count=len(scores))
        ws = np.fromiter(scores.values(), dtype=np.float64, count=len(scores))
        bigltm_mod._recall_project(ks, ws, out_num, ltm._rev_indptr,
                                   ltm._rev_indices)
        out_py = _py_recall_project(ltm, scores)
        check(f"recall 反投影逐位一致（{n_scores} 项, 越界={with_oor}）",
              np.array_equal(out_num, out_py))

    # ── 3b. P70：整条 recall 逐位一致（predict_arr + 核 vs 原 dict 路径）──
    from phdnet.sparse_table import OnlineCSRTable
    for int8 in (False, True):
        tbl = OnlineCSRTable(n_neurons=1 << 20, m_out=72, lam=0.9, eta=0.01,
                             w_max=1.0, seed=0, prune=0, growth_guidance=False,
                             int8_store=int8)
        r2 = np.random.default_rng(5)
        rows = [int(x) for x in r2.choice(1 << 20, size=120, replace=False)]
        act = [int(x) for x in r2.choice(1 << 20, size=40, replace=False)]
        for i in rows:                      # 造出「行接近满 + 键随机」的真实状态
            tbl._new_row(i)
            ks = r2.choice(1 << 20, size=min(tbl.m_out - 1, 71), replace=False)
            ks = list(ks); r2.shuffle(ks)
            for k in ks:
                tbl._append(i, int(k), float(r2.random()))
        # 原路径：dict predict + Python 双循环
        scores = tbl.predict(act)
        ref = np.zeros(ltm.n_dim)
        for big_i, s in scores.items():
            for j in ltm.rev.get(big_i, ()):
                ref[j] += s
        # 新路径：predict_arr + numba 核
        out = np.zeros(ltm.n_dim)
        kk, wv = tbl.predict_arr(act)
        bigltm_mod._recall_project(kk, wv, out, ltm._rev_indptr, ltm._rev_indices)
        check(f"P70 recall 全链路逐位一致（int8={int8}，行满+键随机）",
              np.array_equal(out, ref))

    # ── 4. 计时（服务器形态：活跃 200 维 → 千级索引；recall 2000 项）──────
    rate = np.zeros(ltm.n_dim)
    rate[rng.choice(ltm.n_dim, size=200, replace=False)] = 1.0
    a = _best(lambda: _py_encode(ltm, rate))
    print(f"    encode(200 dims -> {len(ltm.encode(rate))} idx): "
          f"python {a*1e6:7.1f} us（**生产走这条**；numba 版实测 0.84–1.75× "
          f"波动且 O(n²) 去重，已回滚）")

    keys = rng.choice(1 << 20, size=2000, replace=False)
    scores = {int(k): float(rng.random()) for k in keys}
    ks = np.fromiter(scores.keys(), dtype=np.int64, count=len(scores))
    ws = np.fromiter(scores.values(), dtype=np.float64, count=len(scores))
    buf = np.zeros(ltm.n_dim)

    def py_rec():
        o = np.zeros(ltm.n_dim)
        for big_i, s in scores.items():
            for j in ltm.rev.get(big_i, ()):
                o[j] += s
        return o

    def nb_rec():
        o = np.zeros(ltm.n_dim)
        bigltm_mod._recall_project(ks, ws, o, ltm._rev_indptr, ltm._rev_indices)
        return o

    a2 = _best(py_rec, 3)
    b2 = _best(nb_rec, 3)
    print(f"    recall(2000 scores -> {sum(len(ltm.rev.get(int(k), ())) for k in ks)} "
          f"bindings): python {a2*1e6:7.1f} us -> numba {b2*1e6:7.1f} us | "
          f"{a2/max(b2,1e-12):.2f}x")

    # 5. predict_arr 计时（服务器实测形态：active=256 行 × 约 72 槽 ≈ 1.8 万元素）
    tbl2 = OnlineCSRTable(n_neurons=1 << 20, m_out=72, lam=0.9, eta=0.01,
                          w_max=1.0, seed=0, prune=0, growth_guidance=False)
    r3 = np.random.default_rng(6)
    act2 = [int(x) for x in r3.choice(1 << 20, size=256, replace=False)]
    for i in act2:
        tbl2._new_row(i)
        ks2 = list(r3.choice(1 << 20, size=71, replace=False))
        r3.shuffle(ks2)
        for k in ks2:
            tbl2._append(i, int(k), float(r3.random()))
    a3 = _best(lambda: tbl2.predict(act2), 3)
    b3 = _best(lambda: tbl2.predict_arr(act2), 3)
    print(f"    predict(active=256 rows x 72 slots): dict {a3*1e3:7.3f} ms -> "
          f"predict_arr {b3*1e3:7.3f} ms | {a3/max(b3,1e-12):.2f}x")

    n_ok = sum(1 for _, ok in CASES if ok)
    print(f"\n通过 {n_ok}/{len(CASES)}（另附计时）")
    return 0 if n_ok == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
