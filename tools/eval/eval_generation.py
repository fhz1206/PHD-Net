"""生成能力评估：加载检查点 → 自回归采样 → 客观指标（P198）。

用法：
  # 基本（从检查点 + 词表快照）
  py -3.14 tools/eval/eval_generation.py \
      --ckpt outputs/models/phdnet1b_smoke_cc_traces.npz \
      --vocab outputs/models/vocab_smoke_cc_traces.json

  # 指定生成参数
  py -3.14 tools/eval/eval_generation.py --tau 0.8 --topk 40 --topp 0.9

  # 用本地 cc_traces parquet 做 bigram 基准（默认前 20 条 trace）
  py -3.14 tools/eval/eval_generation.py --ref datasets/cc_traces/cc_000.parquet

指标（全部客观、可复现）：
  · bigram 合法率：生成的相邻 token 对在训练语料出现过的比例
  · distinct：去重 token 占比（模式坍缩检测）
  · 5 次采样开头去重：多样性
  · 未训练对照：同 seed 同参数的随机初始化模型输出

⚠ **诚实边界**：这些指标能证明「没学会」，但不能单独证明「有创造力」。
bigram 合法率高只说明记住了局部共现，不等于能生成合理代码。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "train")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config_1b import build_cfg                        # noqa: E402
from phdnet.config import PHDNetConfig                 # noqa: E402
from phdnet.corpus import load_cc_traces               # noqa: E402
from phdnet.tokenizer import U64, mix64                # noqa: E402
from phdnet.word_encoder import WordSegmenter, WordTokenizer  # noqa: E402
from phdnet.word_lm import PHDWordLM                   # noqa: E402


def _make_tok(cfg, words, segv, wmax):
    """按词表快照手工构造 WordTokenizer（不经重分词，逐字一致）。"""
    seg_kw = dict(max_len=wmax, min_count=2, min_entropy=0.5)
    tokens = sorted(set(words))
    t = WordTokenizer.__new__(WordTokenizer)
    t.seg = WordSegmenter("seed", **seg_kw)
    t.seg.vocab = set(segv)
    t.seg.max_len = wmax
    t.tokens = tokens
    t.stoi = {w: i for i, w in enumerate(tokens)}
    t.n_sdr = cfg.n_sdr
    t.n_active = cfg.k_sparse
    t.seed = cfg.seed
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


def main() -> None:
    ap = argparse.ArgumentParser(description="PHD-Net 生成能力评估")
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--vocab", required=True, type=Path)
    ap.add_argument("--ref", type=Path, default=None,
                    help="bigram 基准语料（parquet 或 .txt；默认前 20 条本地 trace）")
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--n-tokens", type=int, default=60)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=0, help="0=关闭")
    ap.add_argument("--topp", type=float, default=0.0, help="0=关闭")
    ap.add_argument("--n-samples", type=int, default=5)
    args = ap.parse_args()

    # ── 1. cfg 来自检查点（保证 conn_k / dtype 等全部一致）──────────
    with np.load(args.ckpt, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"][0]))
    cfg = PHDNetConfig.from_ckpt(meta["cfg"])

    # ── 2. 词表快照 → tok → LM → load_model ───────────────────────
    from ckpt_1b import load_model, load_vocab_file
    words, segv, wmax = load_vocab_file(args.vocab)
    tok = _make_tok(cfg, words, segv, wmax)
    lm = PHDWordLM("seed", cfg, seg_kwargs=dict(max_len=wmax, min_count=2,
                                                min_entropy=0.5), tokenizer=tok)
    load_model(args.ckpt, lm)
    print(f"[model] step_count={lm.net.step_count} vocab={len(lm.tok):,} "
          f"conn_k={cfg.readout_conn_k} dtype={cfg.readout_dtype}")

    # ── 3. 基准语料 ───────────────────────────────────────────────
    if args.ref is None:
        _d = _ROOT / "datasets" / "cc_traces" / "raw"
        if _d.is_dir():
            _fs = sorted(_d.glob("*.jsonl"))[:20]
            ref_text = "\n\n".join(f.read_text(encoding="utf-8") for f in _fs)
        else:
            ref_text = ""
    elif args.ref.suffix.lower() == ".parquet":
        ref_text = load_cc_traces(args.ref, limit_chars=400000)
    else:
        ref_text = args.ref.read_text(encoding="utf-8")
    train_toks = lm.tok.seg.tokenize(ref_text) if ref_text else []
    print(f"[ref] {len(ref_text):,} chars → {len(train_toks):,} tokens")

    if not train_toks:
        print("[ref] 无基准语料 → 只报 distinct / 多样性（bigram 跳过）")

    # ── 4. 未训练对照 ─────────────────────────────────────────────
    cfg0 = PHDNetConfig.from_ckpt(meta["cfg"])
    lm0 = PHDWordLM("seed", cfg0, seg_kwargs=dict(max_len=wmax, min_count=2,
                                                  min_entropy=0.5),
                    tokenizer=_make_tok(cfg0, words, segv, wmax))

    seed = train_toks[0] if train_toks else lm.tok.tokens[0]
    kw = dict(tau=args.tau, topk=args.topk, topp=args.topp)
    print(f"\n[gen] τ={args.tau} topk={args.topk} topp={args.topp} "
          f"n={args.n_tokens} seed={seed!r}")
    print("=" * 72)

    rng = np.random.default_rng(42)
    out0 = lm0.generate(seed, n_tokens=args.n_tokens, rng=rng, **kw)
    if train_toks:
        bv0 = lm0.bigram_validity(out0, train_toks)
        print(f"未训练对照: bigram={bv0:.1%}")
        print("   ", "".join(out0)[:200])
        print()

    for i in range(args.n_samples):
        rng = np.random.default_rng(100 + i)
        out = lm.generate(seed, n_tokens=args.n_tokens, rng=rng, **kw)
        bv = (lm.bigram_validity(out, train_toks) if train_toks else float("nan"))
        distinct = len(set(out)) / len(out)
        print(f"sample {i + 1}: bigram={bv:.1%} distinct={distinct:.1%}")
        print("   ", "".join(out)[:220])
        print()

    # ── 5. 汇总 ───────────────────────────────────────────────────
    rng = np.random.default_rng(7)
    outs = [lm.generate(seed, n_tokens=40, rng=rng, **kw)
            for _ in range(args.n_samples)]
    sigs = {"".join(o)[:40] for o in outs}
    bvs = ([lm.bigram_validity(o, train_toks) for o in outs]
           if train_toks else [])
    print("=" * 72)
    if bvs:
        print(f"bigram 合法率: mean={np.mean(bvs):.1%} max={np.max(bvs):.1%}"
              f"（未训练对照 {bv0:.1%}）")
    print(f"采样多样性: {len(sigs)}/{args.n_samples} 个不同开头"
          f"（5/5=好，1/5=模式坍缩）")


if __name__ == "__main__":
    main()
