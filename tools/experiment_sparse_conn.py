"""O1-2 实验：主干结构性稀疏连接（sparse_conn）的端到端效果。

口径：冻结语料，训练段前 4,000 字符（与 ablation 同口径，稠密基线 97.2596），
      评估段后 20% 全量。报告 ppl_char / ms·token⁻¹ / 可塑参数（稀疏口径按存在突触）。

配置：稠密基线 → conn_k = 8 / 16 / 32 / 64（连接率 3.8% / 7.5% / 15% / 30%）。

用法：python tools/experiment_sparse_conn.py
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
from phdnet.model import count_params           # noqa: E402
from phdnet.word_lm import PHDWordLM            # noqa: E402

CORPUS = _ROOT / "datasets" / "eval" / "internal_corpus.txt"
BASE = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
            eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11, pred_in_readout=True)
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)
TRAIN_CHARS = 4000

CONFIGS = [
    ("dense", {}, "稠密主干（基线 97.2596，连接率 100%）"),
    ("sparse_k8", dict(sparse_conn=True, conn_k=8), "结构性稀疏 k=8（连接率 3.8%）"),
    ("sparse_k16", dict(sparse_conn=True, conn_k=16), "结构性稀疏 k=16（7.5%）"),
    ("sparse_k32", dict(sparse_conn=True, conn_k=32), "结构性稀疏 k=32（15%）"),
    ("sparse_k64", dict(sparse_conn=True, conn_k=64), "结构性稀疏 k=64（30%）"),
]


def main() -> None:
    text = CORPUS.read_text(encoding="utf-8")
    split = int(len(text) * 0.8)
    train_txt = text[:split][:TRAIN_CHARS]
    eval_txt = text[split:]
    print(f"语料 {len(text)} | 训练段 {len(train_txt)} | 评估段 {len(eval_txt)}")
    print("=" * 92)
    rows = []
    for name, ov, note in CONFIGS:
        cfg = PHDNetConfig(**{**BASE, **ov})
        lm = PHDWordLM(text, cfg, seg_kwargs=SEG)
        n_par = count_params(lm.net)
        n_tok = len(lm.tokenize(train_txt))
        t0 = time.perf_counter()
        lm.train_stream(train_txt)
        dt = time.perf_counter() - t0
        m = lm.evaluate(eval_txt)
        ms = dt / max(1, n_tok) * 1000
        extra = ""
        if hasattr(lm.net.pc, "stats"):
            st = lm.net.pc.stats()
            extra = (f"存在突触 {st['synapses']:,} / 稠密等价 {st['dense_equivalent']:,}"
                     f"（连接率 {st['connectivity'] * 100:.1f}%）")
        rows.append((name, float(m["ppl_char"]), ms, n_par, dt, note))
        print(f"{name:<12} ppl_char={m['ppl_char']:>9.4f}  {ms:>6.3f} ms/tok  "
              f"params={n_par:>9,}  {extra}", flush=True)

    base_ppl, base_ms = rows[0][1], rows[0][2]
    print("=" * 92)
    print(f"{'配置':<12}{'ppl_char':>10}{'ΔPPL%':>9}{'ms/token':>11}{'加速':>8}{'参数量':>11}  说明")
    for name, ppl, ms, n_par, dt, note in rows:
        print(f"{name:<12}{ppl:>10.4f}{(ppl - base_ppl) / base_ppl * 100:>+9.2f}"
              f"{ms:>11.3f}{base_ms / ms:>7.2f}×{n_par:>11,}  {note}")
    print("=" * 92)
    print("判读：ΔPPL% 与加速并看——结构性稀疏以更少突触换取（近似）同等 PPL 时才算净收益。")


if __name__ == "__main__":
    main()
