"""P78：`OnlineCSRTable.learn` 批量多核路径逐位对拍 + 服务器形态计时。

**动机**（服务器实测，aarch64）：一次 imprint 的 Python `learn` 要 ~20 s
（|prev|×|cur| ≈ 65k 组合 × 每次 `_find_slot` 线性扫描 + dict 访问，
aarch64 的 Python 解释器与 dict 访问比 x86 慢 10–20×），摊薄成
`M4b_ltm` 40+ ms/tok 且只占 1 个核。

**测试要求**（P67 的教训全部继承）：
1. 行已填至接近 `m_out` 且键**打乱**（否则槽位查找总命中首项，测出最优而非平均）；
2. 覆盖 fp64 / int8 / growth_guidance；
3. **多步序列**（生长 + 量化 + 迹衰减交织）必须逐位——P67 就是死在这里；
4. 批量路径与 Python 路径用**同一份 CSR 数据**（共享元组）避免采样偏差。

运行：`python tests/verifiers/verify_ltm_learn_batch.py`（退出码 0 = PASS）
"""
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from phdnet.sparse_table import OnlineCSRTable                    # noqa: E402

CASES: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    CASES.append((name, bool(cond)))
    print(f"  {'✓' if cond else '✗'} {name}")


def _py_only(m_out=72, int8=False):
    t = OnlineCSRTable(n_neurons=1 << 22, m_out=m_out, lam=0.9, eta=0.01,
                       w_max=1.0, seed=0, prune=0, growth_guidance=False,
                       int8_store=int8)
    t.BATCH_MIN_COMBOS = 10 ** 18          # 禁用批量路径（原 Python 路径）
    return t


def _batched(m_out=72, int8=False, guidance=False):
    t = OnlineCSRTable(n_neurons=1 << 22, m_out=m_out, lam=0.9, eta=0.01,
                       w_max=1.0, seed=0, prune=0, growth_guidance=guidance,
                       int8_store=int8)
    t.BATCH_MIN_COMBOS = 1                 # 强制批量路径
    return t


def _seed_state(tbl, prev, cur, rng, fill=None):
    """长跑后的真实状态：迹非零（短路不生效）+ 行接近满 + 键打乱。"""
    tbl._touch(cur, tbl.t_pre, tbl.stamp_pre)
    tbl._touch(cur, tbl.t_post, tbl.stamp_post)
    shuffled = list(cur)
    rng.shuffle(shuffled)
    fill = fill if fill is not None else max(1, tbl.m_out - 1)
    for i in prev:
        tbl._new_row(int(i))
        for k in shuffled[:fill]:
            tbl._append(int(i), int(k), 0.5)


def _snapshot(tbl):
    return [(i, tbl.keys[i][:tbl.size[i]].copy(), tbl.vals[i][:tbl.size[i]].copy())
            for i in sorted(tbl.keys.keys())]


def _same(a, b) -> bool:
    if len(a) != len(b):
        return False
    for (ia, ka, va), (ib, kb, vb) in zip(a, b):
        if ia != ib or not np.array_equal(ka, kb) or not np.array_equal(va, vb):
            return False
    return True


def _best(fn, reps=3):
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b


def _main() -> int:
    print("P78：learn 批量多核路径 —— 逐位对拍 + 服务器形态计时")
    rng = np.random.default_rng(0)

    # ── 1. 单步逐位（fp64/int8 × guidance）───────────────────────────────
    for int8 in (False, True):
        for guid in (False, True):
            n_act = 128                     # 128×128=16384 ≥ 4096 阈值
            prev = [int(x) for x in rng.choice(1 << 22, size=n_act, replace=False)]
            cur = [int(x) for x in rng.choice(1 << 22, size=n_act, replace=False)]
            tp, bp = _py_only(int8=int8), _batched(int8=int8, guidance=guid)
            tp.growth_guidance = bp.growth_guidance = guid
            _seed_state(tp, prev, cur, np.random.default_rng(1))
            _seed_state(bp, prev, cur, np.random.default_rng(1))
            tp.learn(prev, cur)
            bp.learn(prev, cur)
            check(f"批量 vs Python 逐位一致（int8={int8}, guidance={guid}）",
                  _same(_snapshot(tp), _snapshot(bp)))

    # ── 2. 多步序列（P67 就死在这里）─────────────────────────────────────
    n_act = 200
    pv = [int(x) for x in rng.choice(1 << 22, size=n_act, replace=False)]
    tp, tf = _py_only(int8=True), _batched(int8=True)
    _seed_state(tp, pv[:60], pv[:60], np.random.default_rng(2), fill=30)
    _seed_state(tf, pv[:60], pv[:60], np.random.default_rng(2), fill=30)
    ok = True
    for step in range(6):
        cv = [int(x) for x in rng.choice(1 << 22, size=n_act, replace=False)]
        tp.learn(pv, cv)
        tf.learn(pv, cv)
        pv = cv
        if not _same(_snapshot(tp), _snapshot(tf)):
            ok = False
            print(f"    ✗ step {step} 开始发散")
            break
    check("6 步序列逐表内容逐位一致（含生长/量化/迹衰减）", ok)

    # ── 3. 服务器形态计时（active 256×256，即 [ltm-diag] 实测规模）────────
    for n_act in (128, 256):
        prev = [int(x) for x in rng.choice(1 << 22, size=n_act, replace=False)]
        cur = [int(x) for x in rng.choice(1 << 22, size=n_act, replace=False)]
        tp, tf = _py_only(), _batched()
        _seed_state(tp, prev, cur, np.random.default_rng(3))
        _seed_state(tf, prev, cur, np.random.default_rng(3))
        a = _best(lambda: tp.learn(prev, cur))
        b = _best(lambda: tf.learn(prev, cur))
        print(f"    active {n_act}x{n_act} ({n_act*n_act} combos): "
              f"python {a*1e3:8.2f} ms -> prange {b*1e3:8.2f} ms | "
              f"{a/max(b,1e-12):.2f}x")

    n_ok = sum(1 for _, ok in CASES if ok)
    print(f"\n通过 {n_ok}/{len(CASES)}（另附服务器形态计时）")
    return 0 if n_ok == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
