"""O5 实验：长程整段复制的级联误差治理（分段校验）。

口径沿用 tests/demo_copy.py（T6 延迟复制）：
  训练 n=8 × 150 段；评估 n = 4 / 8 / 16（各 40 段）；
  指标 = 位置准确率 pos 与**整段完全复现率 exact**（O5 的核心目标指标）。

对比：
  - baseline：demo_copy 网格较优配置（sep=1.0, forget=0.95, rho=0.5, kappa=0.4）
  - bidir：双向校验（既有机制，对照）
  - segment=N：O5 新增分段校验（N=2/4/8，阈值 0.5/0.7）
  - segment + bidir 组合

用法：python tools/experiment_o5_segment.py
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

RHO, ETA, FORGET, SEP, KAPPA = 0.5, 0.2, 0.95, 1.0, 0.4
N_TRAIN_SEQ = 150
N_EVAL_SEQ = 40


def train_mem(codes) -> tuple[ContextMemory, int]:
    mem = ContextMemory(N_DIM, rho=RHO, eta=ETA, k_clean=K_SDR, forget=FORGET,
                        sep=SEP, kappa=KAPPA)
    seqs = gen_sequences(8, N_TRAIN_SEQ, seed=1)
    ep = 0
    for seq in seqs:
        mem.begin_episode(ep)
        for ch in seq:
            mem.observe(codes[ch])
        mem.end_episode()
        ep += 1
    return mem, ep


def evaluate(mem, codes, n, ep_start, *, bidir=False, segment=0, seg_thresh=0.5):
    seqs = gen_sequences(n, N_EVAL_SEQ, seed=99)
    pos_hit = pos_tot = exact = 0
    ep = ep_start
    mem.bidir, mem.bidir_w = bidir, 1.0
    for seq in seqs:
        mem.begin_episode(ep)
        for ch in seq:
            mem.observe(codes[ch])
        got = mem.recall_scored(n, codes, clean_steps=1,
                                segment=segment, seg_thresh=seg_thresh)
        for g, t in zip(got, seq):
            pos_tot += 1
            pos_hit += int(g == t)
        exact += int(list(got) == list(seq))
        ep += 1
    return {"pos": pos_hit / max(1, pos_tot), "exact": exact / max(1, len(seqs))}


CONFIGS = [
    ("baseline", dict(bidir=False, segment=0, seg_thresh=0.5)),
    ("bidir", dict(bidir=True, segment=0, seg_thresh=0.5)),
    ("seg2_t0.5", dict(bidir=False, segment=2, seg_thresh=0.5)),
    ("seg4_t0.5", dict(bidir=False, segment=4, seg_thresh=0.5)),
    ("seg4_t0.7", dict(bidir=False, segment=4, seg_thresh=0.7)),
    ("seg8_t0.5", dict(bidir=False, segment=8, seg_thresh=0.5)),
    ("seg4+bidir", dict(bidir=True, segment=4, seg_thresh=0.5)),
]


def main() -> None:
    codes = sdrs(ALPHA)
    print(f"T6 长程复制（O5 分段校验）| SDR {N_DIM} 维 / {K_SDR} 活跃 | 训练 n=8 × {N_TRAIN_SEQ} 段")
    print(f"随机基线 = {100 / len(ALPHA):.1f}%    既有实测：默认栈 ~10%（位置）")
    print("=" * 88)
    print(f"{'配置':<14}{'n=4 pos/exact':>22}{'n=8 pos/exact':>22}{'n=16 pos/exact':>22}")
    rows = []
    for name, kw in CONFIGS:
        t0 = time.perf_counter()
        mem, ep = train_mem(codes)
        cur = ep
        res = {}
        for n in (4, 8, 16):
            res[n] = evaluate(mem, codes, n, cur, **kw)
            cur += N_EVAL_SEQ
        dt = time.perf_counter() - t0
        rows.append((name, res, dt))
        cells = "".join(f"{res[n]['pos'] * 100:>10.1f}%/{res[n]['exact'] * 100:>8.1f}%"
                        for n in (4, 8, 16))
        print(f"{name:<14}{cells}   ({dt:.0f}s)", flush=True)

    print("=" * 88)
    b = rows[0][1]
    print("相对基线的 exact 变化（百分点）：")
    for name, res, dt in rows[1:]:
        d4 = (res[4]["exact"] - b[4]["exact"]) * 100
        d8 = (res[8]["exact"] - b[8]["exact"]) * 100
        d16 = (res[16]["exact"] - b[16]["exact"]) * 100
        print(f"  {name:<14} n=4 {d4:+.1f}pp   n=8 {d8:+.1f}pp   n=16 {d16:+.1f}pp")
    print("=" * 88)
    print("注：exact = 整段序列完全复现率（O5 唯一目标指标）；pos 为位置准确率（护栏）。")


if __name__ == "__main__":
    main()
