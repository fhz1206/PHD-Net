"""A4 实测：读出头两级群体读出（`readout_hidden`）vs 单层线性 vs 单级稀疏。

脑对应：皮层→输出的投射是**多级中继 + 群体编码**，而非单层线性分类器。
本脚本实测两级读出（h → 隐藏群体〔稀疏投射 + k-WTA 侧抑制 + 局部 Oja〕→ 输出）
的精度/速度/参数量代价。

口径：冻结语料，训练段前 4,000 字符（基线 ppl_char = 97.2596），评估段后 20%。

用法：python tools/experiment_two_level_readout.py
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
    ("baseline_single", {}, "单层线性读出（基线）"),
    ("sparse_single_k8", dict(readout_conn_k=8), "单级稀疏读出 k=8（对照）"),
    ("two_lvl_h256", dict(readout_hidden=256), "两级群体读出 隐藏 256"),
    ("two_lvl_h512", dict(readout_hidden=512), "两级群体读出 隐藏 512"),
    ("two_lvl_h1024", dict(readout_hidden=1024), "两级群体读出 隐藏 1024"),
]


def main() -> None:
    text = CORPUS.read_text(encoding="utf-8")
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split][:TRAIN_CHARS], text[split:]
    print(f"语料 {len(text)} | 训练段 {len(train_txt)} | 评估段 {len(eval_txt)} | 基线 97.2596")
    print("=" * 96)
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
        st = lm.net.readout.stats()
        extra = (f"{st.get('levels', 1)} 级 / 隐藏 {st.get('hidden', '-')} / "
                 f"连接率 {st['connectivity'] * 100:.1f}%")
        rows.append((name, float(m["ppl_char"]), ms, n_par, extra, note))
        print(f"{name:<18} ppl={m['ppl_char']:>9.4f}  {ms:>6.3f} ms/tok  "
              f"params={n_par:>9,}  {extra}", flush=True)

    base = rows[0][1]
    base_ms = rows[0][2]
    print("=" * 96)
    print(f"{'配置':<18}{'ppl_char':>10}{'ΔPPL%':>9}{'ms/token':>11}{'加速':>8}{'参数量':>11}  说明")
    for name, ppl, ms, n_par, extra, note in rows:
        print(f"{name:<18}{ppl:>10.4f}{(ppl - base) / base * 100:>+9.2f}"
              f"{ms:>11.3f}{base_ms / ms:>7.2f}×{n_par:>11,}  {note}")
    print("=" * 96)
    print("判读：两级读出为**吞吐换精度**（速度显著提升、参数大减，但 PPL +10~13%）；"
          "默认关闭，作为类脑读出的可选实现。")


if __name__ == "__main__":
    main()
