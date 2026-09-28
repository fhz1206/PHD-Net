"""全量架构文档 80/20 评估（v0.0.0 规范口径，无截断）。

仅评测 datasets/eval/internal_corpus.txt 全量（当前 18692 字符），按版本治理要求每轮重跑；
不再附加 333K 字符 SFT 语料（其 O(n^2) 分词器成本在本机 CPU 不可行）。
口径：ppl_char = exp(总 token NLL / 评估段字符数)；bpc = ppl_char 的 log2。
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from phdnet.config import PHDNetConfig
from phdnet.word_lm import PHDWordLM


def load(f):
    with open(f, encoding="utf-8") as fh:
        return fh.read()


def main():
    doc = load("datasets/eval/internal_corpus.txt")
    n = len(doc)
    cut = int(n * 0.8)
    train_text, eval_text = doc[:cut], doc[cut:]
    print(f"架构文档全量字符={n}  训练={cut}  评估={n-cut}", flush=True)
    cfg = PHDNetConfig(n_sdr=128, k_sparse=16, n_mid=128, n_top=128,
                       eta_pc=0.0, eta_oja=0.0, eta_stdp=0.03, seed=11,
                       pred_in_readout=True)
    seg = dict(max_len=4, min_count=8, min_entropy=1.0)
    t0 = time.time()
    lm = PHDWordLM(train_text, cfg, seg_kwargs=seg)
    t1 = time.time()
    print(f"词表构建 {t1-t0:.1f}s  词表大小={len(lm.tok)}", flush=True)
    hist = lm.train_stream(train_text, log_every=100000, sleep_every=0)
    t2 = time.time()
    print(f"训练 {t2-t1:.1f}s  末段NLL={hist[-1] if hist else float('nan'):.3f}", flush=True)
    m = lm.evaluate(eval_text)
    t3 = time.time()
    print(f"评估 {t3-t2:.1f}s", flush=True)
    print(f"ppl_token={m['ppl_token']:.2f}  ppl_char={m['ppl_char']:.2f}  "
          f"bpc={m['bpc']:.3f}  n_tok={m['n_tok']}  oov_rate={m['oov_rate']*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
