"""P206：复现并定位「学习期 PPL 大幅上升」。

三种可能（本脚本分别检验）：
  A 域切换（良性）：min 极低 + median 低 + max 高；换域后回落
  B 真退化（恶性）：median 与 max 同步抬升且不回落
  C 探针/口径问题：sliding PPL 变了但固定探针集没变

判据：同一模型同时记录 sliding PPL 与**固定探针集** PPL（同语料）。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "train")
import numpy as np

from config_1b import build_cfg
from ckpt_1b import load_vocab_file
from phdnet.corpus import load_cc_traces
from phdnet.tokenizer import U64, mix64
from phdnet.word_encoder import WordSegmenter, WordTokenizer
from phdnet.word_lm import PHDWordLM

VOCAB = Path("outputs/models/vocab_smoke_cc_traces.json")
words, segv, wmax = load_vocab_file(VOCAB)
seg_kw = dict(max_len=wmax, min_count=2, min_entropy=0.5)
tokens = sorted(set(words))


def make_tok(cfg):
    t = WordTokenizer.__new__(WordTokenizer)
    t.seg = WordSegmenter("seed", **seg_kw)
    t.seg.vocab = set(segv)
    t.seg.max_len = wmax
    t.tokens = tokens
    t.stoi = {w: i for i, w in enumerate(tokens)}
    t.n_sdr, t.n_active, t.seed = cfg.n_sdr, cfg.k_sparse, cfg.seed
    t._enc_buf = np.zeros(2 * cfg.n_sdr, dtype=np.float32)
    t._oh_buf = np.zeros(len(tokens), dtype=np.float16)
    t._oh_last = -1
    t._sdrs = {}
    for i, w in enumerate(tokens):
        x = U64(i) * U64(2654435761) + np.arange(cfg.k_sparse, dtype=np.uint64)
        idx = (mix64(x + U64(cfg.seed)) % U64(cfg.n_sdr)).astype(np.int64)
        s = np.zeros(cfg.n_sdr)
        s[idx] = 1.0
        t._sdrs[w] = s
    return t


def run(text, n_tokens, tag, cfg=None):
    """训练 n_tokens，返回 (sliding PPL 序列, 固定探针 PPL 序列)。"""
    cfg = cfg or build_cfg("smoke")
    cfg.step_profiling = False
    lm = PHDWordLM("seed", cfg, seg_kwargs=seg_kw, tokenizer=make_tok(cfg))
    stoi = lm.tok.stoi
    toks = [w for w in lm.tok.seg.tokenize(text) if w in stoi]
    if len(toks) < 200:
        print(f"  [{tag}] 语料太短（{len(toks)} tok），跳过")
        return [], []
    # 固定探针集 = 语料**中段**（避开训练段与首尾）
    probe = toks[len(toks) // 3: len(toks) // 3 + 1200]
    nlls = []
    W = 500
    for i in range(min(n_tokens, len(toks) - 1)):
        a, b = toks[i], toks[i + 1]
        prev = toks[i - 1] if i > 0 else None
        x = lm.tok.encode_composite(a, prev)
        d = lm.net.step(x, target=lm.tok.onehot(stoi[b]), learn=True,
                       target_idx=stoi[b])
        nlls.append(float(d["nll"]))
        if (i + 1) % W == 0:
            # 固定探针 PPL（只读前向，不改状态）
            # ⚠ P206 修正：探针必须传 target_idx（训练同款契约）。原脚本只传
            #   target onehot；readonly 分支用 `argmax(target)`——而 onehot
            #   的维度是 **n_input 拼接域**（不是词表域），argmax 恒指向
            #   onehot 的活跃位 → 探针 nll 恒 1.0（实测）。训练路径传的是
            #   target_idx=stoi[b]（词表索引），探针必须同款。
            pn = []
            for j in range(len(probe) - 1):
                p0 = probe[j - 1] if j > 0 else None
                xx = lm.tok.encode_composite(probe[j], p0)
                dd = lm.net.step(xx, target=lm.tok.onehot(stoi[probe[j + 1]]),
                                 learn=False, readonly=True,
                                 target_idx=stoi[probe[j + 1]])
                pn.append(float(dd["nll"]))
            sp = float(np.exp(np.mean(nlls[-W:])))
            pp = float(np.exp(np.mean(pn)))
            q = np.quantile(nlls[-W:], [0, 0.5, 1.0])
            print(f"  [{tag}] step {i+1:>6}  sliding {sp:>10.1f}  "
                  f"probe {pp:>8.1f}  "
                  f"nll[min/med/max] {np.exp(q[0]):.0f}/{np.exp(q[1]):.0f}/{np.exp(q[2]):.0f}",
                  flush=True)
    return nlls, []


print("=" * 78)
print("P206 ① 单域语料：PPL 是否单调下降（真退化的判据）")
print("=" * 78)
TEXT = load_cc_traces(Path("datasets/cc_traces/cc_000.parquet"), limit_chars=300000)
run(TEXT, 20000, "single-domain")

print()
print("=" * 78)
print("P206 ② 域切换语料：拼接两段不同域，检验「换域 → PPL 冲高 → 回落」")
print("=" * 78)
half = len(TEXT) // 2
TEXT2 = TEXT[:half] + "\n\n" + TEXT[half:]
run(TEXT2, 20000, "two-domain")

print()
print("=" * 78)
print("P206 ③ 学习率退火与 η_readout 的影响（退化是否来自「恒定学习率」）")
print("=" * 78)
for anneal in (0.0, 0.9997):
    cfg = build_cfg("smoke")
    cfg.eta_readout_anneal = anneal
    print(f"  --- eta_readout_anneal={anneal} ---")
    run(TEXT, 20000, f"anneal{anneal}", cfg)
