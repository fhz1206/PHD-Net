"""O5-opt 实验：长程整段复现率（exact）的机制优化。

上轮结论：分段校验在现有配置下**未触发**（误差是"自洽地漂移"）。
本轮换机制：**双通路交叉验证（vote）** —— 候选得分取两条检索通路
（单步前向 W·c 与二次迭代 W·clean(W·c)）分数的**乘积**（AND 逻辑），
假候选难以同时通过两条通路 → 抑制自洽漂移型误差。

同时做 κ（再入强度）网格——上轮所有配置固定 κ=0.4，未单独扫描。

公平性：每个评估配置前把记忆恢复到训练后快照（评估内含在线 observe，
否则先评估的配置会因状态累积而占优/吃亏）。

口径：demo_copy（T6 延迟复制）——训练 n=8 × 120 段；评估 n=4/8 各 20 段；
      指标 exact（整段完全复现率，O5 目标指标）与 pos（位置准确率）。

用法：python tools/experiment_o5_opt.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests"))

from demo_copy import ALPHA, K_SDR, N_DIM, gen_sequences, sdrs  # noqa: E402
from phdnet.context_memory import ContextMemory  # noqa: E402

RHO, ETA, FORGET, SEP = 0.5, 0.2, 0.95, 1.0
N_TRAIN_SEQ, N_EVAL_SEQ = 120, 20
KAPPAS = [0.0, 0.2, 0.4, 0.8]

EVAL_CFGS = [
    ("base(cs1)", dict(clean_steps=1)),
    ("vote", dict(clean_steps=1, vote=True)),
    ("cs2", dict(clean_steps=2)),
    ("vote+cs2", dict(clean_steps=2, vote=True)),
    ("cs3", dict(clean_steps=3)),
    ("vote+cs3", dict(clean_steps=3, vote=True)),
    ("vote+seg4", dict(clean_steps=1, vote=True, segment=4, seg_thresh=0.5)),
]


def train_mem(codes, kappa):
    mem = ContextMemory(N_DIM, rho=RHO, eta=ETA, k_clean=K_SDR, forget=FORGET,
                        sep=SEP, kappa=kappa)
    seqs = gen_sequences(8, N_TRAIN_SEQ, seed=1)
    ep = 0
    for seq in seqs:
        mem.begin_episode(ep)
        for ch in seq:
            mem.observe(codes[ch])
        mem.end_episode()
        ep += 1
    return mem, ep


def evaluate(mem, codes, n, ep_start, *, clean_steps=1, vote=False,
             segment=0, seg_thresh=0.5):
    seqs = gen_sequences(n, N_EVAL_SEQ, seed=99)
    pos_hit = pos_tot = exact = 0
    ep = ep_start
    mem.bidir, mem.bidir_w = False, 1.0
    for seq in seqs:
        mem.begin_episode(ep)
        for ch in seq:
            mem.observe(codes[ch])
        got = mem.recall_scored(n, codes, clean_steps=clean_steps,
                                segment=segment, seg_thresh=seg_thresh, vote=vote)
        for g, t in zip(got, seq):
            pos_tot += 1
            pos_hit += int(g == t)
        exact += int(list(got) == list(seq))
        ep += 1
    return {"pos": pos_hit / max(1, pos_tot), "exact": exact / max(1, len(seqs))}


def main() -> None:
    codes = sdrs(ALPHA)
    print(f"O5-opt | SDR {N_DIM}/{K_SDR} | 训练 n=8 × {N_TRAIN_SEQ} 段 | 评估 n=4/8 × {N_EVAL_SEQ} 段")
    print(f"随机基线 {100 / len(ALPHA):.1f}% | 上轮 baseline exact(n=4) = 5.0%")
    print("=" * 92)
    best = None
    for kappa in KAPPAS:
        t0 = time.perf_counter()
        mem, ep = train_mem(codes, kappa)
        snap = (mem.W.copy(), mem.W_ic.copy(), mem.slot.copy(), mem.ctx.copy())
        print(f"\n--- κ={kappa} （训练 {time.perf_counter() - t0:.0f}s）---")
        for name, kw in EVAL_CFGS:
            mem.W[:] = snap[0]
            mem.W_ic[:] = snap[1]
            mem.slot[:] = snap[2]
            mem.ctx[:] = snap[3]                       # 恢复训练后快照（公平）
            cur = ep
            res = {}
            for n in (4, 8):
                res[n] = evaluate(mem, codes, n, cur, **kw)
                cur += N_EVAL_SEQ
            line = "  ".join(f"n={n}: {res[n]['pos'] * 100:5.1f}% / exact {res[n]['exact'] * 100:4.1f}%"
                             for n in (4, 8))
            print(f"  {name:<12} {line}", flush=True)
            key = res[4]["exact"] * 100 + res[8]["exact"] * 50
            if best is None or key > best[0]:
                best = (key, f"κ={kappa} {name}", dict(res))
    print("\n" + "=" * 92)
    print(f"最优：{best[1]}")
    for n, r in best[2].items():
        print(f"  n={n}: pos {r['pos'] * 100:.1f}% / exact {r['exact'] * 100:.1f}%")
    print("（对照：上轮 baseline n=4 exact 5.0% / n=8 0.0%）")


if __name__ == "__main__":
    main()
