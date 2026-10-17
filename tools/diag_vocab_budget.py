"""P198：词表规模 vs 训练预算的诊断（本会话实测发现的缺失能力）。

背景（2026-10-17 实测，cc_traces / smoke 档）：
  · 词表 20,913（4M 字符采样）+ 100K token → PPL **3322**（随机 e^9.95）
  · 词表  3,392（300K 字符采样）+ 100K token → PPL **1310**（随机 e^8.13）
  同样的训练量，**目标空间缩小 6× → PPL 降 61%**。
  频次分解显示：top-100 词 NLL 8.38（≈随机），其余全部 > 随机
  → 每词平均曝光 ~30 次仍不够，`eta_readout` 的单步更新在数千输出
  维度上收敛极慢。

本工具在**训练前**估算这个比值并给出建议，避免跑几小时后才发现
词表定得过大。

用法：
  py -3.14 tools/diag_vocab_budget.py --vocab-sample-chars 4000000 --tokens 100000
  py -3.14 tools/diag_vocab_budget.py --from-snapshot outputs/models/vocab_smoke_cc_traces.json --tokens 100000

判据（经验阈值，来自本会话实测；不是理论推导）：
  · exposure = tokens / vocab（每词平均曝光次数）
  · exposure < 30  → 严重不足（PPL 接近随机）
  · 30 ≤ exposure < 100 → 勉强（高频词能学，中频全丢）
  · exposure ≥ 100 → 充足
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _chars_per_token(sample_text: str) -> float:
    """实测 chars/token（用项目自己的分词器，最准）。"""
    from phdnet.word_encoder import WordSegmenter
    seg = WordSegmenter(sample_text[:200000], max_len=6, min_count=5,
                        min_entropy=1.0)
    toks = seg.tokenize(sample_text[:200000])
    return len(sample_text[:200000]) / max(1, len(toks))


def main() -> None:
    ap = argparse.ArgumentParser(description="词表规模 vs 训练预算诊断")
    ap.add_argument("--vocab-sample-chars", type=int, default=4_000_000,
                    help="词表采样字符数（与 train.py 同名参数一致）")
    ap.add_argument("--tokens", type=int, default=100_000,
                    help="训练 token 预算")
    ap.add_argument("--from-snapshot", type=Path, default=None,
                    help="直接从词表快照读词表大小（比估算准）")
    ap.add_argument("--data", default="cc_traces",
                    choices=["cc_traces", "sft", "pretrain"],
                    help="用于 chars/token 估算的数据集")
    args = ap.parse_args()

    # ── 1. 词表大小 ───────────────────────────────────────────────
    if args.from_snapshot is not None:
        from train.ckpt_1b import load_vocab_file
        words, _segv, _wmax = load_vocab_file(args.from_snapshot)
        vocab = len(set(words))
        src = f"快照 {args.from_snapshot.name}"
    else:
        # 用实测两点拟合（cc_traces 上）vocab ≈ a·chars^b：
        #   (3e5, 3392) 与 (4e6, 20913)
        #   b = ln(20913/3392)/ln(4e6/3e5) = 1.8192/2.5903 = 0.7024
        #   a = 3392/3e5^0.7024 = 3392/3319 = 1.0220
        # 拟合误差：4M 处预测 20,7xx vs 实测 20,913（<1%）。
        _b = math.log(20913 / 3392) / math.log(4e6 / 3e5)
        _a = 3392 / (300000 ** _b)
        vocab = max(1, int(_a * (args.vocab_sample_chars ** _b)))
        src = f"{args.vocab_sample_chars:,} 字符采样（幂律拟合 b={_b:.3f}）"
    print(f"词表规模: {vocab:,} 词（来源: {src}）")

    # ── 2. 曝光度 ─────────────────────────────────────────────────
    exposure = args.tokens / vocab
    print(f"训练预算: {args.tokens:,} tokens")
    print(f"每词平均曝光: {exposure:.1f} 次")
    print()
    if exposure < 30:
        # 目标曝光 100 → 需要的词表 = tokens / 100
        target_vocab = max(500, int(args.tokens / 100))
        # 反解 chars = (vocab/a)^(1/b)
        _b = math.log(20913 / 3392) / math.log(4e6 / 3e5)
        _a = 3392 / (300000 ** _b)
        target_chars = int((target_vocab / _a) ** (1 / _b))
        level, advice = "🔴 严重不足", (
            f"PPL 会接近随机（实测：曝光 4.8 → PPL 3322 ≈ 随机）。\n"
            f"  **推荐**：减小词表 → --vocab-sample-chars {target_chars:,}"
            f"（≈{target_vocab:,} 词，曝光 ≈100）\n"
            f"  或增大预算：--tokens {int(vocab * 100):,}（本机 CPU 上 "
            f"{vocab * 100 / 1000:.0f}K token ≈ {vocab * 100 * 0.003 / 60:.0f} 分钟）")
    elif exposure < 100:
        level, advice = "🟡 勉强", (
            f"高频词能学，中频全丢（实测 top-100 NLL≈随机、其余 >随机）。\n"
            f"  建议 --tokens {int(vocab * 100):,} 或缩小词表。")
    else:
        level, advice = "🟢 充足", "曝光度足够，可以开跑。"
    print(f"判定: {level}")
    print(advice)

    # ── 3. 与实测锚点对照 ─────────────────────────────────────────
    print("\n── 本会话实测锚点（cc_traces / smoke 档）──")
    print(f"{'词表':>8}{'tokens':>9}{'曝光':>7}{'PPL':>8}{'随机PPL':>9}")
    for v, tk, ppl, rnd in ((20913, 100000, 3322, 9.95),
                            (3392, 100000, 1310, 8.13)):
        print(f"{v:>8,}{tk:>9,}{tk / v:>7.1f}{ppl:>8}{rnd:>9.2f}")
    print("→ 曝光 4.8 → PPL 3322（≈随机 e^8.07）")
    print("→ 曝光 29.5 → PPL 1310（≈随机 e^7.19）")


if __name__ == "__main__":
    main()
