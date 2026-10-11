"""冻结语料基线比较工具：不更新 ANCHOR，不自动判定数值偏移的原因。

80/20 划分 eval_corpus/internal_corpus.txt（27,034 字符）；训练段取前
4,000 字符或全训练段。配置来自 tests/eval_common.py::BASE。
默认显式使用历史 fp32 读出口径；--readout-dtype fp8 可测当前库默认精度，
但不与历史 fp32 锚点判一致。--json 输出含配置、语料哈希及版本的原始记录。

P192 STDP 学习语义修正后可能偏离历史锚点；偏移需归因，不能自动视为回归。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
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

ANCHOR = {4000: 394.4687, "full": 359.2603}   # 历史 fp32 口径，仅比较，永不由本工具更新


def run(limit: int | None = None, readout_dtype: str = "fp32") -> dict:
    with open(DOC, encoding="utf-8") as f:
        text = f.read()
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split], text[split:]
    seg = train_txt[:limit] if limit else train_txt
    cfg = PHDNetConfig(**{**BASE, "readout_dtype": readout_dtype})
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
    with open(DOC, "rb") as f:
        corpus_sha256 = hashlib.sha256(f.read()).hexdigest()
    return {"ppl_char": m["ppl_char"], "bpc": m["bpc"],
            "ms_per_token": dt / max(1, n_tok) * 1000,
            "connectivity": conn, "tokens": n_tok,
            "readout_dtype": readout_dtype, "cython_kernels": cfg.cython_kernels,
            "config_overrides": {**BASE, "readout_dtype": readout_dtype},
            "seg_kwargs": SEG, "train_chars": len(seg), "eval_chars": len(eval_txt),
            "corpus_chars": len(text), "corpus_sha256": corpus_sha256,
            "python": platform.python_version(), "platform": platform.platform(),
            "historical_fp32_anchor": ANCHOR[limit if limit else "full"],
            "anchor_comparable_dtype": readout_dtype == "fp32"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readout-dtype", choices=("fp32", "fp8"), default="fp32")
    ap.add_argument("--json", action="store_true", help="最后输出 JSON 记录（不写文件）")
    args = ap.parse_args()
    print("=" * 78)
    print(f"基线比较（冻结语料 / BASE / {args.readout_dtype}）—— 偏移须另行归因")
    print("=" * 78)
    records = []
    for label, lim in (("4,000 字符", 4000), ("全语料", None)):
        r = run(lim, args.readout_dtype)
        a = ANCHOR[lim if lim else "full"]
        d = (r["ppl_char"] - a) / a * 100
        r["label"] = label
        r["historical_delta_pct"] = d if args.readout_dtype == "fp32" else None
        records.append(r)
        flag = ("不同精度口径，不判一致" if args.readout_dtype != "fp32" else
                ("✅ 与历史数值一致" if abs(d) < 0.01 else f"⚠ 偏移 {d:+.2f}%（非自动回归结论）"))
        # ⚠ **标签澄清**（P126）：这里的「连接率」取自 `lm.net.pc.stats()`，
        # 是 **M2 预测编码主干**的连接率（BASE 栈 256 维 → conn_k=32 → 12.5%），
        # **不是 M6 读出的**。M6 读出连接率 = `readout_conn_k / n_readout_input`
        # （生产口径 128/3072 = **4.2%**，或幂律 P124 下按 alpha 倾斜）。
        # 两者差3 倍，打印在一起极易误读 → 故显式标注。
        print(f"{label:<10} ppl_char = {r['ppl_char']:.4f}  bpc {r['bpc']:.3f}  "
              f"{r['ms_per_token']:.3f} ms/token  "
              f"[M2主干] 连接率 {r['connectivity']*100:.1f}%  "
              f"历史 fp32 锚点 {a}  {flag}", flush=True)
    if args.json:
        cp = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True)
        head = cp.stdout.decode("ascii", "replace").strip() if cp.returncode == 0 else None
        status = subprocess.run(["git", "status", "--porcelain"], cwd=_ROOT, capture_output=True)
        dirty = bool(status.stdout) if status.returncode == 0 else None
        print("RESULT_JSON=" + json.dumps({"head": head, "dirty": dirty,
              "anchor_policy": "comparison-only; unchanged historical fp32 values",
              "results": records}, ensure_ascii=False))
