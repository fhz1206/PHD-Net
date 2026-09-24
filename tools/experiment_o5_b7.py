"""O5（长程级联误差治理：再入连接 + 分段校验）与 B7（minibatch 统计效率）实验。

口径：冻结语料 datasets/eval/internal_corpus.txt（23,504 字符）；
      训练段 = 前 80% 的前 4,000 字符（与 ablation_modules / scaling_curve 同口径，
      便于交叉对照：本脚本 baseline 应与 ablation baseline 逐位一致 = 97.2596）；
      评估段 = 后 20%（4,701 字符，全量）。

默认路径验证：baseline 配置不含任何新开关 → 若与 ablation 基线一致，
即证明 O5/B7 的改动未影响默认数值路径。

用法：python tools/experiment_o5_b7.py
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

CORPUS = _ROOT / "datasets" / "eval" / "internal_corpus.txt"
BASE = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
            eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
            pred_in_readout=True)
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)
TRAIN_CHARS = 4000

CONFIGS = [
    ("baseline", {}, "无新开关（应与 ablation 基线 97.2596 逐位一致）"),
    ("O5_recur0.5", dict(readout_recurrence=True, recur_gain=0.5), "再入连接 gain=0.5"),
    ("O5_recur1.0", dict(readout_recurrence=True, recur_gain=1.0), "再入连接 gain=1.0（全量）"),
    ("O5_segcheck50", dict(segment_check=50), "分段校验 N=50（低置信段读出降权）"),
    ("O5_both", dict(readout_recurrence=True, recur_gain=0.5, segment_check=50),
     "再入 + 分段校验组合"),
    ("B7_minibatch4", dict(minibatch_size=4), "读出 minibatch=4（平均梯度更新）"),
    ("B7_minibatch8", dict(minibatch_size=8), "读出 minibatch=8"),
    ("B7_mb4_anneal", dict(minibatch_size=4, eta_readout=0.1),
     "minibatch=4 + 学习率补偿(0.1)"),
]


def main() -> None:
    text = CORPUS.read_text(encoding="utf-8")
    split = int(len(text) * 0.8)
    train_txt = text[:split][:TRAIN_CHARS]
    eval_txt = text[split:]
    print(f"语料 {len(text)} 字符 | 训练段 {len(train_txt)} | 评估段 {len(eval_txt)}")
    print("=" * 76)

    rows = []
    for name, ov, note in CONFIGS:
        t0 = time.perf_counter()
        cfg = PHDNetConfig(**{**BASE, **ov})
        lm = PHDWordLM(text, cfg, seg_kwargs=SEG)
        lm.train_stream(train_txt)
        m = lm.evaluate(eval_txt)
        dt = time.perf_counter() - t0
        rows.append((name, m["ppl_char"], m["bpc"], m["n_tok"], dt, note))
        print(f"{name:<18} ppl_char={m['ppl_char']:>9.4f}  bpc={m['bpc']:.4f}  "
              f"({dt:.1f}s)  {note}", flush=True)

    base = rows[0][1]
    print("=" * 76)
    print(f"{'配置':<18}{'ppl_char':>10}{'变化率%':>10}{'耗时':>8}")
    for name, ppl, bpc, ntok, dt, note in rows:
        chg = (ppl - base) / base * 100.0
        print(f"{name:<18}{ppl:>10.4f}{chg:>+10.2f}{dt:>7.1f}s")
    print("=" * 76)
    print("判读：变化率 <0 = 优于基线（PPL 下降）；O5 的目标是长程整段复制（见 demo_copy 口径），")
    print("      LM PPL 仅作「不损害主任务」的护栏检查。")


if __name__ == "__main__":
    main()
