"""P6b-2 实验：读出学习率（eta_readout）网格——fp32 实验的副产物发现。

副产物：`experiment_fp32_readout.py` 中 fp32 本身无速度收益（−5%）且 PPL 与 fp64
完全相同（97.2596），但"学习率补偿"变体给出 −1.5% / −3.7% 的 PPL 增益 →
说明增益来自 **eta_readout 0.05 → 0.06/0.08**，与 fp32 无关。本脚本在 fp64
口径下（排除 fp32 干扰）做纯学习率网格。

口径：冻结语料，训练段前 4,000 字符，评估段后 20%（与 ablation 同口径，基线 97.2596）。

用法：python tools/experiment_readout_lr.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.config import PHDNetConfig          # noqa: E402
from phdnet.word_lm import PHDWordLM            # noqa: E402

CORPUS = _ROOT / "eval_corpus" / "internal_corpus.txt"
BASE = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
            eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11, pred_in_readout=True)
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)
TRAIN_CHARS = 4000
LR_GRID = [0.05, 0.06, 0.08, 0.10, 0.12, 0.15]


def main() -> None:
    text = CORPUS.read_text(encoding="utf-8")
    split = int(len(text) * 0.8)
    train_txt = text[:split][:TRAIN_CHARS]
    eval_txt = text[split:]
    print(f"语料 {len(text)} | 训练段 {len(train_txt)} | 评估段 {len(eval_txt)}")
    print("=" * 76)
    rows = []
    for lr in LR_GRID:
        cfg = PHDNetConfig(**{**BASE, "eta_readout": lr})
        lm = PHDWordLM(text, cfg, seg_kwargs=SEG)
        t0 = time.perf_counter()
        lm.train_stream(train_txt)
        m = lm.evaluate(eval_txt)
        dt = time.perf_counter() - t0
        rows.append((lr, float(m["ppl_char"]), float(m["bpc"]), dt))
        print(f"eta_readout={lr:<5} ppl_char={m['ppl_char']:>9.4f}  bpc={m['bpc']:.4f}  ({dt:.0f}s)",
              flush=True)

    base_ppl = rows[0][1]
    print("=" * 76)
    print(f"{'eta_readout':>12}{'ppl_char':>11}{'ΔPPL%':>9}")
    for lr, ppl, bpc, dt in rows:
        print(f"{lr:>12}{ppl:>11.4f}{(ppl - base_ppl) / base_ppl * 100:>+9.2f}")
    best = min(rows, key=lambda r: r[1])
    print("=" * 76)
    print(f"最优: eta_readout={best[0]} → ppl_char {best[1]:.4f}"
          f"（相对默认 0.05 变化 {(best[1] - base_ppl) / base_ppl * 100:+.2f}%）")
    print("注：eta_readout 默认值 0.05 属既有默认行为；若要改为更优值需 fhz 确认")
    print("    （改默认值会改变所有历史基线的数值口径）。")


if __name__ == "__main__":
    main()
