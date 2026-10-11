"""PHD-Net 强化学习训练入口（P27）。

在 SFT/预训练权重之上做 REINFORCE（策略梯度直接复用 P6 读出感知器更新）。

示例：
    # 1) 数据集内联 reward（JSONL 每行 {"prompt": ..., "reward": 1.0}）
    python tools/train_rl.py --init-from outputs/models/phdnet1b_1b_sft_final.npz \\
        --prompts data/rl/prompts.jsonl --iters 200 --rl-lr 0.02

    # 2) 自定义奖励函数（--reward-fn 指向 module:function）
    python tools/train_rl.py --init-from ... --prompts p.jsonl \\
        --reward-fn my_reward:score_completion

数据格式：JSONL，每行至少 {"prompt": "..."}；可选 "reward"（数字或
{"value": x}）。无内联 reward 时必须给 --reward-fn（签名
`(prompt, completion, tokens, logprobs, **item) -> float`）。

奖励函数示例（写进文件即可）：
    def score_completion(prompt, completion, **kw):
        return 1.0 if "正确答案" in completion else 0.0
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))
os.environ.setdefault(  # P39：numba 缓存持久化（不被 __pycache__ 清理波及）
    "NUMBA_CACHE_DIR", str(_ROOT / "outputs" / "numba_cache"))
sys.path.insert(0, str(_ROOT / "train"))

import numpy as np                                          # noqa: E402


def load_reward_fn(spec: str):
    """`module:function` → callable。"""
    mod_name, _, fn_name = spec.partition(":")
    if not fn_name:
        raise ValueError(f"--reward-fn 需为 'module:function'（收到 {spec!r}）")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise AttributeError(f"{mod_name} 中没有 {fn_name}")
    return fn


def load_lm(init_from: Path, accel: str = "auto"):
    """从检查点重建 LM（词表/分词器/状态全部自包含）。"""
    import json

    from ckpt_1b import _rebuild_sdrs, load_model
    from config_1b import SEG_KWARGS
    from phdnet.config import PHDNetConfig
    from phdnet.word_encoder import WordSegmenter, WordTokenizer
    from phdnet.word_lm import PHDWordLM

    with np.load(str(init_from), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"][0]))
        vocab = {str(w) for w in z["tok_vocab"]}
        tokens = [str(t) for t in z["tok_tokens"]]
        max_len = int(z["tok_max_len"][0]) if "tok_max_len" in z.files else 6
    cfg = PHDNetConfig.from_ckpt(meta["cfg"])
    cfg.accel_readout = accel                       # P19：读出设备
    seg = WordSegmenter.__new__(WordSegmenter)
    seg.vocab = vocab
    seg.max_len = max_len
    tok = WordTokenizer.__new__(WordTokenizer)
    tok.seg = seg
    tok.tokens = tokens
    tok.stoi = {t: i for i, t in enumerate(tokens)}
    tok.n_sdr, tok.n_active, tok.seed = cfg.n_sdr, cfg.k_sparse, cfg.seed
    _rebuild_sdrs(tok, tokens)
    lm = PHDWordLM("", cfg, tokenizer=tok)
    load_model(init_from, lm)
    print(f"[rl] 模型已加载：{init_from} | 词表 {len(tokens):,} | "
          f"读出后端 {getattr(lm.net, '_readout_backend', 'numba-cpu')}")
    return lm


def main() -> int:
    ap = argparse.ArgumentParser(description="PHD-Net RL（REINFORCE）训练")
    ap.add_argument("--init-from", type=Path, required=True,
                    help="初始化检查点（通常是 SFT/预训练后的 npz）")
    ap.add_argument("--prompts", type=Path, required=True,
                    help="JSONL 提示集（每行 {\"prompt\": ..., \"reward\": ...}）")
    ap.add_argument("--reward-fn", type=str, default=None,
                    help="自定义奖励函数 module:function（无内联 reward 时必填）")
    ap.add_argument("--iters", type=int, default=100, help="训练轮数")
    ap.add_argument("--batch-prompts", type=int, default=0,
                    help="每轮采样条数（0=全部）")
    ap.add_argument("--rl-lr", type=float, default=0.01,
                    help="读出更新的 η 基数（实际 η = rl-lr × advantage）")
    ap.add_argument("--n-tokens", type=int, default=32, help="每条轨迹生成 token 数")
    ap.add_argument("--tau", type=float, default=1.0, help="采样温度")
    ap.add_argument("--topk", type=int, default=0, help="top-k 截断（0=全分布）")
    ap.add_argument("--baseline-decay", type=float, default=0.9,
                    help="基线滑动平均衰减（降方差）")
    ap.add_argument("--no-normalize-adv", action="store_true",
                    help="不归一化优势（默认归一化，但奖励无方差时自动跳过）")
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help="KL 正则系数（>0 时冻结初始读出作参考，额外占一份 W 内存）")
    ap.add_argument("--accel", type=str, default="auto",
                    help="读出设备（P19：auto 有 cuda/cann/rocm 就用）")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=Path, default=None, help="训练后保存到此 npz")
    args = ap.parse_args()

    if not args.init_from.exists():
        print(f"[rl] 找不到 {args.init_from}（如实报告，退出）")
        return 2
    if not args.prompts.exists():
        print(f"[rl] 找不到提示集 {args.prompts}（如实报告，退出）")
        return 2

    from phdnet.rl import REINFORCETrainer, load_prompts

    items = load_prompts(args.prompts)
    have_inline = all(("reward" in it) for it in items)
    reward_fn = load_reward_fn(args.reward_fn) if args.reward_fn else None
    if not have_inline and reward_fn is None:
        print("[rl] 提示集没有内联 reward，且未给 --reward-fn —— 无法打分。"
              "请提供其一（数据集 reward 字段或 module:function）。")
        return 2
    print(f"[rl] 提示集 {len(items):,} 条 | 奖励来源："
          f"{'数据集内联' if have_inline else args.reward_fn}")

    lm = load_lm(args.init_from, accel=args.accel)
    tr = REINFORCETrainer(
        lm, lr=args.rl_lr, baseline_decay=args.baseline_decay,
        normalize_adv=not args.no_normalize_adv, kl_coef=args.kl_coef,
        reward_fn=reward_fn, free_stream=True)
    if args.kl_coef > 0:
        tr.pin_reference()
        print(f"[rl] 已冻结参考策略（KL 系数 {args.kl_coef}，额外占一份读出内存）")

    pool = items
    if args.batch_prompts and args.batch_prompts < len(items):
        rng0 = np.random.default_rng(args.seed)
        pool = [items[i] for i in rng0.choice(len(items),
                                             size=args.batch_prompts,
                                             replace=False)]

    t0 = time.perf_counter()
    zero_adv_rounds = 0
    for it in range(1, args.iters + 1):
        rs = tr.collect(pool, n_tokens=args.n_tokens, tau=args.tau,
                        topk=args.topk, seed=args.seed + it)
        st = tr.update(rs)
        if st.get("steps", 0) == 0:
            zero_adv_rounds += 1
        elif zero_adv_rounds:
            zero_adv_rounds = 0
        if zero_adv_rounds == max(3, args.log_every):
            print("[rl] 提示：连续多轮「优势恒为 0」——采样到的轨迹奖励完全相同，"
                  "没有梯度信号。常见原因与对策：\n"
                  "      · 奖励太稀疏（命中才算）→ 改用**稠密奖励**"
                  "（如目标字符覆盖率）或分段奖励；\n"
                  "      · 采样太短/温度太低 → 增大 --n-tokens 或 --tau；\n"
                  "      · 策略太弱（刚 SFT 完）→ 先做几轮有监督微调再 RL。", flush=True)
        if it % args.log_every == 0 or it == 1:
            el = time.perf_counter() - t0
            print(f"  iter {it:>5d}/{args.iters}  reward {st['reward']:+.4f}"
                  f"  baseline {st['baseline']:+.4f}"
                  f"  |adv| {st['adv_abs']:.3f}  nll {st['nll']:.3f}"
                  + (f"  kl {st['kl']:.4f}" if "kl" in st else "")
                  + f"  |  {el:.0f}s", flush=True)
    print(f"[rl] 完成：{tr.stats()}")
    if args.save:
        from ckpt_1b import save_model
        save_model(args.save, lm, lm.cfg, 0,
                   extra={"rl_iters": args.iters, "rl_stats": tr.stats()})
        print(f"[rl] 已保存：{args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
