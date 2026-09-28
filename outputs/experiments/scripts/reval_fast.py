"""快速重评估报告（截断子样本，因纯 Python 分词器在 CPU 上全量过慢）。

说明：完整 13.8K 字架构文档的全量训练在本机 CPU 上 >8min 未能完成（已实测中断）。
本脚本改用**截断子样本**以在当前会话内给出可复现的代表性数字，并逐阶段计时，
以暴露瓶颈。口径与 outputs/reval.py 完全一致（同小网络配置 + 同分词参数）。

配置：n_sdr=128,k=16,n_mid=128,n_top=128,eta_pc=0,eta_oja=0,eta_stdp=0.03,
      seed=11,pred_in_readout=True；分词 max_len=4,min_count=8,min_entropy=1.0。
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


def bench(train_text, eval_text, label):
    cfg = PHDNetConfig(n_sdr=128, k_sparse=16, n_mid=128, n_top=128,
                       eta_pc=0.0, eta_oja=0.0, eta_stdp=0.03, seed=11,
                       pred_in_readout=True)
    seg = dict(max_len=4, min_count=8, min_entropy=1.0)
    t0 = time.time()
    lm = PHDWordLM(train_text, cfg, seg_kwargs=seg)
    t1 = time.time()
    print(f"  [{label}] 词表构建 {t1-t0:.1f}s  词表大小={len(lm.tok)}", flush=True)
    hist = lm.train_stream(train_text, log_every=100000, sleep_every=0)
    t2 = time.time()
    print(f"  [{label}] 训练 {t2-t1:.1f}s  末段NLL={hist[-1] if hist else float('nan'):.3f}", flush=True)
    m = lm.evaluate(eval_text)
    t3 = time.time()
    print(f"  [{label}] 评估 {t3-t2:.1f}s", flush=True)
    print(f"    ppl_token={m['ppl_token']:.2f}  ppl_char={m['ppl_char']:.2f}  "
          f"bpc={m['bpc']:.3f}  n_tok={m['n_tok']}  oov_rate={m['oov_rate']*100:.1f}%", flush=True)


if __name__ == "__main__":
    doc = load("datasets/eval/internal_corpus.txt")
    print(f"架构文档总字符={len(doc)}", flush=True)

    # 截断：前 6000 训练 / 随后 2000 评估（代表性子样本）
    bench(doc[:6000], doc[6000:8000], "架构文档(截断 6000/2000)")

    if os.path.exists("data/tool_train.txt"):
        t = load("data/tool_train.txt")
        print(f"新 SFT 语料 tool_train.txt 总字符={len(t)}", flush=True)
        bench(t[:3000], t[3000:4000], "tool_train.txt(截断 3000/1000)")
