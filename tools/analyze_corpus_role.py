"""语料角色判定：给定语料更适合 **SFT** 还是 **预训练（pretrain）**。

================================================================================
为什么不用「按空行切样本」的老办法
================================================================================
初版按 `\\n\\n` 切块统计，结果失真：SFT 语料内部也有空行（instruction\\n\\ninput\\
\\noutput），切出来的"样本"其实是碎片（Magpie 被切成 36 万块、中位长 126 字符，
与真实的 40,619 条 / 平均 7,661 字符严重不符）。故改为**不依赖样本边界**的
结构统计——只看文本自身的形态特征。

================================================================================
判据（全部可复现，不是主观印象）
================================================================================
| 指标 | SFT 倾向 | 预训练倾向 |
|---|---|---|
| `instruct` 指令句行占比 | 高（含"请/如何/什么/为什么/写/生成…"） | 低 |
| `dialog` 对话标记密度（用户：/助手：/问：/答：） | 高 | 低 |
| `question` 疑问句行占比 | 高 | 低 |
| `blank` 空行率 | 高（问答分段） | 低（连续叙述） |
| `mean_line` 平均行长 | 中短 | 长（段落体） |
| `entropy` 字符熵 bits/char | 中 | 高（语言/知识密度） |
| `self_repeat` 8-gram 自重复 | 低 | 中高（叙述套式） |

用法：python tools/analyze_corpus_role.py [文件 ...]
"""
from __future__ import annotations

import argparse
import math
import os
from collections import Counter

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)

INSTRUCT_HEAD = ("请", "如何", "为什么", "什么", "写", "生成", "解释", "介绍",
                 "描述", "列举", "列出", "翻译", "计算", "帮", "给我", "我需要",
                 "能否", "是否", "有哪些", "怎", "推荐", "总结", "分析", "比较",
                 "write", "how", "what", "why", "explain", "generate", "list",
                 "solve", "design", "implement", "given", "please", "create")
DIALOG_MARK = ("用户：", "助手：", "问：", "答：", "User:", "Assistant:",
               "Human:", "AI:", "Q:", "A:")


def _read(path: str, limit_mb: float = 40.0) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read(int(limit_mb * 1_000_000))


def entropy_bits(s: str) -> float:
    c = Counter(s)
    n = len(s)
    if n < 100:
        return 0.0
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def self_repeat(s: str, n: int = 8) -> float:
    s = s[:2_000_000]
    if len(s) < 3 * n:
        return 0.0
    grams = [s[i:i + n] for i in range(0, len(s) - n + 1, 3)]
    return 1.0 - len(set(grams)) / max(1, len(grams))


def analyze(path: str, limit_mb: float = 40.0) -> dict:
    data = _read(path, limit_mb)
    lines = [l for l in data.split("\n")]
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return {"path": path, "error": "空文件"}
    n = len(lines)
    instructs = sum(1 for l in non_empty if l.strip().startswith(INSTRUCT_HEAD))
    dialog = sum(1 for l in non_empty if any(m in l for m in DIALOG_MARK))
    ques = sum(1 for l in non_empty if ("？" in l or "?" in l))
    return {
        "path": os.path.relpath(path, _ROOT),
        "mb": os.path.getsize(path) / 1e6,
        "lines": n,
        "mean_line": float(np.mean([len(l) for l in non_empty])),
        "blank": float(sum(1 for l in lines if not l.strip()) / n),
        "instruct": instructs / len(non_empty),
        "dialog": dialog / len(non_empty),
        "question": ques / len(non_empty),
        "entropy": entropy_bits(data[:2_000_000]),
        "self_repeat": self_repeat(data),
    }


def verdict(r: dict) -> tuple[str, int, int]:
    sft = pre = 0
    sft += 2 if r["instruct"] > 0.10 else (1 if r["instruct"] > 0.04 else 0)
    sft += 2 if r["dialog"] > 0.05 else (1 if r["dialog"] > 0.01 else 0)
    sft += 1 if r["question"] > 0.15 else 0
    sft += 1 if r["blank"] > 0.10 else 0
    pre += 2 if r["entropy"] > 4.0 else (1 if r["entropy"] > 3.5 else 0)
    pre += 2 if r["mean_line"] > 120 else (1 if r["mean_line"] > 60 else 0)
    pre += 1 if r["self_repeat"] > 0.05 else 0
    pre += 1 if r["blank"] < 0.03 else 0
    pre += 1 if r["instruct"] < 0.02 else 0
    if sft - pre >= 4:
        return "SFT", sft, pre
    if pre - sft >= 4:
        return "预训练", sft, pre
    return "两者皆可", sft, pre


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--limit-mb", type=float, default=40.0)
    a = ap.parse_args()
    files = a.files or [
        "datasets/sft/deepctrl_sft_zh.txt",
        "datasets/sft/magpie_reasoning_r1.txt",
        "datasets/sft/ultrainteract_sft.txt",
        "datasets/pretrain/infinity_m7core.txt",
    ]
    print("=" * 104)
    print("语料角色判定（SFT vs 预训练）——不依赖样本边界的结构统计")
    print("=" * 104)
    print(f"{'语料':<32}{'MB':>7}{'平均行长':>9}{'空行率':>8}{'指令句':>8}"
          f"{'对话标记':>9}{'疑问句':>8}{'熵':>7}{'自重复':>8}  判定")
    print("-" * 104)
    rows = []
    for f in files:
        if not os.path.exists(f):
            print(f"{f:<32}  (不存在)")
            continue
        r = analyze(f, a.limit_mb)
        if r.get("error"):
            print(f"{r['path']:<32}  {r['error']}")
            continue
        v, s, p = verdict(r)
        rows.append((r, v, s, p))
        print(f"{r['path'][:31]:<32}{r['mb']:>7.0f}{r['mean_line']:>9.0f}"
              f"{r['blank']*100:>7.1f}%{r['instruct']*100:>7.1f}%"
              f"{r['dialog']*100:>8.1f}%{r['question']*100:>7.1f}%"
              f"{r['entropy']:>7.2f}{r['self_repeat']*100:>7.1f}%  "
              f"{v}（SFT {s} vs PT {p}）")
    print("-" * 104)
    print("读法：指令句/对话标记/疑问句高 → 问答配对结构 → SFT；")
    print("      平均行长 + 字符熵高、空行率低 → 连续叙述、语言与知识密度大 → 预训练。")


if __name__ == "__main__":
    main()
