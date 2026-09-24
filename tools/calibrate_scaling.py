"""O7 校准脚本（临时）：实测不同宽度档位的参数规模 + 单遍耗时，用于确定三点配置。

不修改 phdnet/ / tests/。仅测量 count_params 与小规模耗时，不做完整实验。
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from phdnet.config import PHDNetConfig
from phdnet.word_lm import PHDWordLM
from phdnet.model import count_params

DOC = str(ROOT / "datasets" / "eval" / "internal_corpus.txt")
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)
BASE = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
            eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
            pred_in_readout=True)

text = open(DOC, encoding="utf-8").read()
split = int(len(text) * 0.8)
train_txt, eval_txt = text[:split], text[split:]
N_CHARS_TRAIN = 4000  # 截断口径（与正式实验一致）
train_trunc = train_txt[:N_CHARS_TRAIN]

print(f"corpus={len(text)} train80%={len(train_txt)} eval={len(eval_txt)} "
      f"train_trunc={len(train_trunc)}", flush=True)

# 词表大小（由全语料决定，三点共用）
probe = PHDWordLM(text, PHDNetConfig(**BASE), seg_kwargs=SEG)
V = len(probe.tok)
print(f"vocab_size(V)={V}", flush=True)

def measure(n_sdr, n_mid, n_top, k_sparse, time_train=False):
    cfg = {**BASE, "n_sdr": n_sdr, "n_mid": n_mid, "n_top": n_top, "k_sparse": k_sparse}
    lm = PHDWordLM(text, PHDNetConfig(**cfg), seg_kwargs=SEG)
    npar = count_params(lm.net)
    ms = None
    if time_train:
        t0 = time.perf_counter()
        lm.train_stream(train_trunc)
        dt = time.perf_counter() - t0
        ntok = len(lm.tokenize(train_trunc))
        ms = dt / max(1, ntok) * 1000
        m = lm.evaluate(eval_txt)
        print(f"  -> ppl_char={m['ppl_char']:.3f} bpc={m['bpc']:.4f} n_tok={m['n_tok']} oov={m['oov']}", flush=True)
    print(f"D=({n_sdr},{n_mid},{n_top}) k={k_sparse} params={npar:,} "
          f"ms/tok={ms if ms is None else round(ms,3)}", flush=True)
    return npar

print("=== 参数规模扫描（不训练）===", flush=True)
for D in (256, 600, 900, 1200, 1500, 2000, 3000, 4350, 5000):
    measure(D, D, D, max(1, D // 8))

print("=== 计时校准：BASE 在 4000 字符上的单遍耗时 ===", flush=True)
measure(256, 256, 256, 32, time_train=True)
