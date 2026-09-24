"""n-gram 统计基线族 —— 语言模型的对照下界（字符级 / 词级）。

自 lm.py（字符级 NGramBaseline）与 word_lm.py（词级 WordNGram）归并；
两者原路径仍可用（lm.py / word_lm.py 均再导出）。"""

from math import log

import numpy as np


class WordNGram:
    """token 级 n-gram 基线（加 1 平滑），供词级 LM 对照。"""

    def __init__(self, train_tokens: list[str], n: int = 2):
        self.n = n
        vocab = sorted(set(train_tokens))
        self.stoi = {t: i for i, t in enumerate(vocab)}
        # 2026-09-18 修复：显式 OOV 索引（原实现用 -1，会与合法 id 混用、
        # 且分母未把 OOV 这一类计入，平滑口径不一致）
        self.unk = len(self.stoi)
        self.counts: dict[tuple, dict[int, int]] = {}
        for i in range(len(train_tokens) - n):
            ctx = tuple(self.stoi[t] for t in train_tokens[i:i + n])
            nxt = self.stoi[train_tokens[i + n]]
            d = self.counts.setdefault(ctx, {})
            d[nxt] = d.get(nxt, 0) + 1

    def nll(self, eval_tokens: list[str]) -> float:
        """加 1 平滑的 token 平均 NLL；OOV 计入独立一类，保证概率归一。"""
        V = len(self.stoi) + 1                       # +1 = OOV 类
        total, cnt = 0.0, 0
        unk = self.unk
        for i in range(len(eval_tokens) - self.n):
            ctx = tuple(self.stoi.get(t, unk) for t in eval_tokens[i:i + self.n])
            nxt = self.stoi.get(eval_tokens[i + self.n], unk)
            cc = self.counts.get(ctx, {})
            total += -log((cc.get(nxt, 0) + 1) / (sum(cc.values()) + V))
            cnt += 1
        return total / max(1, cnt)


class NGramBaseline:
    """字符级 n-gram 基线（加 1 平滑）。字符表由全量文本构建，计数仅用训练段。"""

    def __init__(self, train_text: str, vocab_text: str, n: int = 3):
        self.n = n
        self.chars = sorted(set(vocab_text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.counts: dict[tuple, dict[int, int]] = {}
        for t in range(len(train_text) - n):
            ctx = tuple(self.stoi[c] for c in train_text[t:t + n])
            nxt = self.stoi[train_text[t + n]]
            self.counts.setdefault(ctx, {})
            self.counts[ctx][nxt] = self.counts[ctx].get(nxt, 0) + 1

    def evaluate(self, text: str) -> float:
        V = len(self.chars)
        total, cnt = 0.0, 0
        for t in range(len(text) - self.n):
            ctx = tuple(self.stoi[c] for c in text[t:t + self.n])
            nxt = self.stoi[text[t + self.n]]
            cc = self.counts.get(ctx, {})
            total += -np.log((cc.get(nxt, 0) + 1) / (sum(cc.values()) + V))
            cnt += 1
        return float(np.exp(total / max(1, cnt)))
