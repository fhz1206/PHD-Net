"""P206-2：replay 拉锯验证——replay_every 4 vs 16 vs 关，在「文档边界」下的 PPL 摆动。"""
import sys
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


TEXT = load_cc_traces(Path("datasets/cc_traces/cc_000.parquet"), limit_chars=300000)


def run(replay_every, n=14000):
    cfg = build_cfg("smoke")
    cfg.step_profiling = False
    cfg.replay_every = replay_every
    lm = PHDWordLM("seed", cfg, seg_kwargs=seg_kw, tokenizer=make_tok(cfg))
    stoi = lm.tok.stoi
    toks = [w for w in lm.tok.seg.tokenize(TEXT) if w in stoi]
    # 固定探针集（中段，避开训练段）
    probe = toks[len(toks) // 3: len(toks) // 3 + 1000]
    nlls = []
    probes = []
    for i in range(min(n, len(toks) - 1)):
        a, b = toks[i], toks[i + 1]
        prev = toks[i - 1] if i > 0 else None
        x = lm.tok.encode_composite(a, prev)
        d = lm.net.step(x, target=lm.tok.onehot(stoi[b]), learn=True,
                        target_idx=stoi[b])
        nlls.append(float(d["nll"]))
        if (i + 1) % 2000 == 0:
            pn = []
            for j in range(len(probe) - 1):
                p0 = probe[j - 1] if j > 0 else None
                xx = lm.tok.encode_composite(probe[j], p0)
                dd = lm.net.step(xx, target=lm.tok.onehot(stoi[probe[j + 1]]),
                                 learn=False, readonly=True,
                                 target_idx=stoi[probe[j + 1]])
                pn.append(float(dd["nll"]))
            probes.append(float(np.exp(np.mean(pn))))
    arr = np.asarray(nlls)
    # 每 2000 步窗口的 sliding PPL + 摆动幅度
    wins = [float(np.exp(arr[k:k + 2000].mean()))
            for k in range(0, len(arr) - 2000, 2000)]
    swing = max(wins) / max(1e-9, min(wins))
    return wins, swing, probes, float(np.exp(arr[-2000:].mean()))


print(f"{'replay_every':>13} {'窗口 PPL 序列':>44} {'摆动倍数':>9} {'探针末值':>10}")
for re_ in (4, 16):
    wins, swing, probes, tail = run(re_)
    ws = " → ".join(f"{w:.0f}" for w in wins[:6])
    print(f"{re_:>13} {ws:>44} {swing:>8.1f}x {probes[-1]:>10.1f}")
