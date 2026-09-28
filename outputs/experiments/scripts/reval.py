"""重新评估：重测当前 v0.0.0 词级 LM 指标 + 新 SFT 语料可学习性。

口径严格对齐 memory 中的"跨粒度统一字符归一 PPL"约定：
    ppl_char = exp(总 token NLL / 评估段字符数)；bpc = ppl_char 的 log2。
语料 = datasets/eval/internal_corpus.txt 当前版（80/20 划分），按版本治理要求每轮重跑。
另外在 data/tool_train.txt（新 SFT 语料）上做可学习性 sanity check。

配置对齐 prepare_r1sft / outputs/train_r1sft.py 的小网络默认值：
    n_sdr=128, k=16, n_mid=128, n_top=128, eta_pc=0, eta_oja=0, eta_stdp=0.03,
    seed=11, pred_in_readout=True；分词 max_len=4, min_count=8, min_entropy=1.0。
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


def build_and_eval(train_text, eval_text, label):
    cfg = PHDNetConfig(n_sdr=128, k_sparse=16, n_mid=128, n_top=128,
                       eta_pc=0.0, eta_oja=0.0, eta_stdp=0.03, seed=11,
                       pred_in_readout=True)
    seg = dict(max_len=4, min_count=8, min_entropy=1.0)
    t0 = time.time()
    lm = PHDWordLM(train_text, cfg, seg_kwargs=seg)
    t1 = time.time()
    hist = lm.train_stream(train_text, log_every=100000, sleep_every=0)
    t2 = time.time()
    m = lm.evaluate(eval_text)
    t3 = time.time()
    print(f"[{label}]")
    print(f"  vocab={len(lm.tok)}  构建={t1-t0:.1f}s  训练={t2-t1:.1f}s  评估={t3-t2:.1f}s")
    print(f"  末段NLL={hist[-1] if hist else float('nan'):.3f}")
    print(f"  ppl_token={m['ppl_token']:.2f}  ppl_char={m['ppl_char']:.2f}  "
          f"bpc={m['bpc']:.3f}  n_tok={m['n_tok']}  oov_rate={m['oov_rate']*100:.1f}%")


if __name__ == "__main__":
    doc = load("datasets/eval/internal_corpus.txt")
    n = len(doc)
    cut = int(n * 0.8)
    build_and_eval(doc[:cut], doc[cut:], f"架构文档 80/20（{n} 字符）")

    if os.path.exists("data/tool_train.txt"):
        t = load("data/tool_train.txt")
        nc = len(t)
        c2 = int(nc * 0.95)
        build_and_eval(t[:c2], t[c2:], f"新 SFT 语料 tool_train.txt 95/5（{nc} 字符）")
