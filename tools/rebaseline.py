"""基线复测（口径锚点）——两个口径 + 吞吐 + 主干连接率。

口径与历史基线完全一致：
  · 全语料：tests/eval_tasks_lm.py 的 task1_lm（80/20 划分，冻结语料 23,504 字符）
  · 4,000 字符：同一划分，训练段只取前 4,000 字符
  · 配置 tests/eval_common.py 的 BASE（n_sdr=256 / k_sparse=32 / eta_pc=0 / pred_in_readout=True）

历史锚点（fan-in 修正后、稀疏默认开启）：
  4,000 字符 96.7241 / 7.207 ms/token；全语料 77.5261 / 9.516 ms/token；连接率 12.5%
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import numpy as np

from eval_common import BASE, DOC, SEG
from phdnet.config import PHDNetConfig
from phdnet.word_lm import PHDWordLM

ANCHOR = {4000: 90.2480, "full": 73.1166}   # fhz 2026-09-27 拍板：eta_readout 0.05→0.15 后的新锚点（旧锚点见 git 历史）


def run(limit: int | None = None) -> dict:
    with open(DOC, encoding="utf-8") as f:
        text = f.read()
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split], text[split:]
    seg = train_txt[:limit] if limit else train_txt
    cfg = PHDNetConfig(**BASE)
    lm = PHDWordLM(text, cfg, seg_kwargs=SEG)
    t0 = time.perf_counter()
    lm.train_stream(seg)
    dt = time.perf_counter() - t0
    m = lm.evaluate(eval_txt)
    n_tok = len(lm.tokenize(seg))
    conn = 1.0
    if getattr(lm.net.pc, "n_synapses", None):
        st = lm.net.pc.stats()
        conn = st.get("connectivity", 1.0)
    return {"ppl_char": m["ppl_char"], "bpc": m["bpc"],
            "ms_per_token": dt / max(1, n_tok) * 1000,
            "connectivity": conn, "tokens": n_tok}


if __name__ == "__main__":
    print("=" * 78)
    print("基线复测（冻结语料 / BASE 口径）—— 验证默认路径未回归")
    print("=" * 78)
    for label, lim in (("4,000 字符", 4000), ("全语料", None)):
        r = run(lim)
        a = ANCHOR[lim if lim else "full"]
        d = (r["ppl_char"] - a) / a * 100
        flag = "✅ 一致" if abs(d) < 0.01 else f"⚠ 偏差 {d:+.2f}%"
        print(f"{label:<10} ppl_char = {r['ppl_char']:.4f}  bpc {r['bpc']:.3f}  "
              f"{r['ms_per_token']:.3f} ms/token  连接率 {r['connectivity']*100:.1f}%  "
              f"锚点 {a}  {flag}", flush=True)
