"""P6b 验证：读出 float32 化（`readout_fp32`）的性能与数值影响。

口径：冻结语料，训练段前 4,000 字符（与 ablation/scaling 同口径），评估段后 20%。
  - baseline 应与 97.2596 逐位一致（证明默认路径未被改动）；
  - fp32 会改变数值（精度下降）→ 报告 PPL 变化率与 ms/token 加速比；
  - 附一个"学习率补偿"变体（精度损失可用更大学习率部分抵消）。

用法：python tools/experiment_fp32_readout.py
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

CONFIGS = [
    ("baseline_fp64", {}, "默认路径（权重 float64）"),
    ("fp32_readout", dict(readout_fp32=True), "读出 fp32（带宽减半）"),
    ("fp32_lr0.06", dict(readout_fp32=True, eta_readout=0.06), "fp32 + 学习率补偿 0.06"),
    ("fp32_lr0.08", dict(readout_fp32=True, eta_readout=0.08), "fp32 + 学习率补偿 0.08"),
]


def main() -> None:
    text = CORPUS.read_text(encoding="utf-8")
    split = int(len(text) * 0.8)
    train_txt = text[:split][:TRAIN_CHARS]
    eval_txt = text[split:]
    print(f"语料 {len(text)} | 训练段 {len(train_txt)} | 评估段 {len(eval_txt)}")
    print("=" * 84)
    rows = []
    for name, ov, note in CONFIGS:
        cfg = PHDNetConfig(**{**BASE, **ov})
        lm = PHDWordLM(text, cfg, seg_kwargs=SEG)
        n_tok = len(lm.tokenize(train_txt))
        t0 = time.perf_counter()
        lm.train_stream(train_txt)
        dt = time.perf_counter() - t0
        m = lm.evaluate(eval_txt)
        ms = dt / max(1, n_tok) * 1000
        rows.append((name, float(m["ppl_char"]), ms, note, lm.net.readout.W.dtype))
        print(f"{name:<16} ppl_char={m['ppl_char']:>9.4f}  {ms:>6.3f} ms/token  "
              f"W.dtype={lm.net.readout.W.dtype}  {note}", flush=True)

    base_ppl, base_ms = rows[0][1], rows[0][2]
    print("=" * 84)
    print(f"{'配置':<16}{'ppl_char':>10}{'ΔPPL%':>9}{'ms/token':>11}{'加速':>8}{'W dtype':>11}")
    for name, ppl, ms, note, dt_ in rows:
        print(f"{name:<16}{ppl:>10.4f}{(ppl - base_ppl) / base_ppl * 100:>+9.2f}"
              f"{ms:>11.3f}{base_ms / ms:>7.2f}×{str(dt_):>11}")
    print("=" * 84)
    print("判读：默认路径须与 97.2596 一致；fp32 属数值变化（默认关闭），"
          "以 PPL 小幅退化换取训练吞吐提升。")


if __name__ == "__main__":
    main()
