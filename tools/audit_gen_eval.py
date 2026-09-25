"""第四轮审计 · 泛化能力实测（2026-09-25，配 docs/性能评估与迭代方案 §5.5）。

协议
----
- 模型：train_1b 在 sft 中文对话语料上流式训练（检查点 = 训练至第 N 个流 token）；
- 域内 held-out：同一语料流的**后续段**（skip N tokens 后取 M tokens）——模型从未见过；
- 近域：`datasets/eval/internal_corpus.txt`（中文架构文档，23,504 字符）；
- 远域：`datasets/eval/ood_wiki.txt`（中文维基探针）；
- 口径：字符归一 PPL = exp(Σ token NLL / 评估段字符数)（OOV 步跳过）；
  ppl_pen = OOV 步按词表均匀 −log(1/V) 补记 NLL 的悲观口径；
- 参照：词级 2-gram（回退 1-gram → 均匀；词表=训练段 token 集，评估段未见表词折叠 UNK；
  仅用训练段构建，无泄漏）。
- 评估全程 readonly（不改变模型状态），可复现。
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT / "train_1b"), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from infer import load_from_ckpt                               # noqa: E402
from corpus_stream import StreamingTokenizer, char_chunks      # noqa: E402


def eval_tokens(lm, tokens, V: int):
    """对一段 token 流做 readonly 评测，返回统计 dict。"""
    stoi = lm.tok.stoi
    nll_sum = 0.0
    n_steps = 0
    n_oov = 0
    chars = 0
    p2 = None
    it = iter(tokens)
    p1 = next(it, None)
    t0 = next(it, None) if p1 is not None else None
    while t0 is not None:
        chars += len(p1) if p1 else 0
        p2_in = p2 is None or p2 in stoi
        if p2_in and p1 in stoi and t0 in stoi:
            x = lm.tok.encode_composite(p1, p2)
            tgt = lm.tok.onehot(stoi[t0])
            d = lm.net.step(x, target=tgt, learn=False, readonly=True)
            nll_sum += float(d["nll"])
            n_steps += 1
        else:
            n_oov += 1
        p2, p1 = p1, t0
        t0 = next(it, None)
    chars += len(p1) if p1 else 0
    ppl = math.exp(nll_sum / max(1, chars))
    ppl_pen = math.exp((nll_sum + n_oov * math.log(V)) / max(1, chars))
    return dict(chars=chars, tokens=n_steps + n_oov, oov=n_oov,
                oov_rate=n_oov / max(1, n_steps + n_oov),
                ppl=ppl, ppl_pen=ppl_pen)


def build_bigram(train_tokens: list[str]):
    """词级 2-gram（回退 1-gram → 均匀），UNK 折叠；仅用训练段，无泄漏。"""
    uni = Counter(train_tokens)
    bi = Counter(zip(train_tokens[:-1], train_tokens[1:]))
    V = len(uni)

    def nll(prev: str | None, t: str) -> float:
        p_unk = 1.0 / V
        pu = (uni.get(t, 0) + 1) / (len(train_tokens) + V)   # add-1 平滑
        if prev is not None:
            c_prev = uni.get(prev, 0)
            if c_prev > 0:
                c_bi = bi.get((prev, t), 0)
                pb = c_bi / c_prev
                if pb > 0:
                    return -math.log(pb)
        return -math.log(pu)

    return nll, V


def eval_bigram(nll_fn, tokens) -> dict:
    nll_sum = 0.0
    chars = 0
    n = 0
    p2 = None
    it = iter(tokens)
    p1 = next(it, None)
    t0 = next(it, None) if p1 is not None else None
    while t0 is not None:
        chars += len(p1) if p1 else 0
        nll_sum += nll_fn(p2, t0)
        n += 1
        p2, p1 = p1, t0
        t0 = next(it, None)
    chars += len(p1) if p1 else 0
    return dict(chars=chars, tokens=n, ppl=math.exp(nll_sum / max(1, chars)))


def main() -> None:
    ap = argparse.ArgumentParser(description="第四轮审计 · 泛化能力实测")
    ap.add_argument("--ckpt", default="models/phdnet1b_smoke_sft.npz")
    ap.add_argument("--skip-tokens", type=int, default=150000,
                    help="域内 held-out 从训练流第 N 个 token 开始（=训练 token 数）")
    ap.add_argument("--indomain-tokens", type=int, default=30000)
    ap.add_argument("--sft", default="datasets/sft/sft_000.*.parquet")
    args = ap.parse_args()

    lm = load_from_ckpt(Path(args.ckpt))
    import json
    with np.load(args.ckpt, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"][0]))
    V = len(lm.tok)
    print(f"检查点: {args.ckpt}（done={int(meta['done']):,}）| 词表 {V:,}")
    if hasattr(lm.net.ltm, "table"):
        st = lm.net.ltm.table.stats()
        print(f"大空间表: 已生长 {st['grown_synapses']:,} / {st['capacity']:,}"
              f"（利用率 {st['utilization']:.4%}）")

    segs: dict[str, list[str]] = {}

    # 1) 域内 held-out：重放训练流，skip 后取段（同时收集训练段 token 供 2-gram）
    stream = StreamingTokenizer(lm.tok.seg, char_chunks(_ROOT / args.sft))
    train_toks: list[str] = []
    for i, t in enumerate(stream):
        if i < args.skip_tokens:
            train_toks.append(t)
        else:
            segs["in_domain_heldout"] = [t]
            break
    for i, t in enumerate(stream):
        segs["in_domain_heldout"].append(t)
        if i + 1 >= args.indomain_tokens:
            break
    print(f"域内段: skip {len(train_toks):,} + eval {len(segs['in_domain_heldout']):,} tokens")

    # 2) 近域 / 3) 远域
    segs["near_domain_doc"] = lm.tok.seg.tokenize(
        (_ROOT / "datasets/eval/internal_corpus.txt").read_text(encoding="utf-8"))
    segs["far_domain_wiki"] = lm.tok.seg.tokenize(
        (_ROOT / "datasets/eval/ood_wiki.txt").read_text(encoding="utf-8"))

    nll_fn, _ = build_bigram(train_toks)

    print("-" * 96)
    print(f"{'目标':20s} {'字符':>9s} {'tokens':>8s} {'OOV':>7s} "
          f"{'ppl_char':>10s} {'ppl_pen':>10s} {'2gram_ppl':>10s} {'PHD/2gram':>9s}")
    base_pen = None
    for name, toks in segs.items():
        r = eval_tokens(lm, toks, V)
        b = eval_bigram(nll_fn, toks)
        ratio = r["ppl_pen"] / b["ppl"] if b["ppl"] > 0 else float("nan")
        if base_pen is None:
            base_pen = r["ppl_pen"]
        infl = r["ppl_pen"] / base_pen
        print(f"{name:20s} {r['chars']:>9,} {r['tokens']:>8,} {r['oov_rate']:>6.2%} "
              f"{r['ppl']:>10.2f} {r['ppl_pen']:>10.2f} {b['ppl']:>10.2f} "
              f"{ratio:>9.3f}  inflation={infl:.2f}x")
    print("-" * 96)
    print("口径注记：ppl_char 跳过 OOV 步（覆盖墙下表观偏低）；ppl_pen 为 OOV 补罚悲观口径；"
          "inflation 以域内 held-out ppl_pen 为 1.00×；2-gram 词表=训练段（UNK 折叠），"
          "覆盖全部 token，与 ppl_pen 可比。")


if __name__ == "__main__":
    main()
