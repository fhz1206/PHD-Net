"""O4 验收：WordSegmenter 向量化改造的正确性对拍 + 加速比实测。

用法：python tests/verifiers/verify_seg_equiv.py
  - 正确性对拍：在 3 组超参下，对比「原实现（内联参考）」与「新实现」的
      vocab 集合（set==set）与 tokenize(text) 输出列表（逐元素 ==）。
      语料 = eval_corpus/internal_corpus.txt 前 6000 字符。
  - 加速比：对全文 23504 字符各构建 1 次，报告 ms 与倍数。
  - 退出码 0 = 全部一致；非 0 = 不一致。
"""

import sys
import time

sys.path.insert(0, "D:/AiModel/train")

from collections import Counter
from math import log

import numpy as np

from phdnet.word_encoder import WordSegmenter as NewWordSegmenter


# ───────────────────────────────────────────────────────────────────────────
# 原实现（逐字内联，作为参考函数，不参与任何优化）
# ───────────────────────────────────────────────────────────────────────────
class RefWordSegmenter:
    """原 O(候选×n) 纯 Python 实现的逐字副本（O4 前的基线）。"""

    def __init__(self, text, max_len=6, min_count=5, min_entropy=1.0):
        n = len(text)
        self.max_len = max_len
        self.vocab = set()
        for L in range(2, max_len + 1):
            grams = Counter(text[i:i + L] for i in range(n - L + 1))
            cand = [w for w, c in grams.items() if c >= min_count]
            cand.sort(key=lambda w: -grams[w])
            for w in cand:
                if w in self.vocab:
                    continue
                pos = [i for i in range(n - L + 1) if text[i:i + L] == w]
                if len(pos) < min_count:
                    continue
                left = Counter(text[i - 1] for i in pos if i > 0)
                right = Counter(text[i + L] for i in pos if i + L < n)
                h_l = self._entropy(left)
                h_r = self._entropy(right)
                if min(h_l, h_r) >= min_entropy:
                    self.vocab.add(w)

    @staticmethod
    def _entropy(counter):
        total = sum(counter.values())
        if total == 0:
            return 0.0
        return -sum((c / total) * log(c / total) for c in counter.values())

    def tokenize(self, text):
        out = []
        i, n = 0, len(text)
        while i < n:
            for L in range(min(self.max_len, n - i), 1, -1):
                if text[i:i + L] in self.vocab:
                    out.append(text[i:i + L])
                    i += L
                    break
            else:
                out.append(text[i])
                i += 1
        return out


# ───────────────────────────────────────────────────────────────────────────
# 参数与语料
# ───────────────────────────────────────────────────────────────────────────
CORPUS = "D:/AiModel/train/eval_corpus/internal_corpus.txt"
PARAMS = [
    dict(max_len=6, min_count=5, min_entropy=1.0),   # 默认组
    dict(max_len=4, min_count=3, min_entropy=0.5),
    dict(max_len=6, min_count=8, min_entropy=1.5),
]

with open(CORPUS, encoding="utf-8") as f:
    text_full = f.read()
text_sub = text_full[:6000]


def main():
    all_ok = True
    lines = []
    lines.append("=== O4 正确性对拍（语料 = 前 6000 字符）===")
    for p in PARAMS:
        ref = RefWordSegmenter(text_sub, **p)
        new = NewWordSegmenter(text_sub, **p)
        vocab_ok = (ref.vocab == new.vocab)
        tok_ref = ref.tokenize(text_sub)
        tok_new = new.tokenize(text_sub)
        tok_ok = (tok_ref == tok_new)
        ok = vocab_ok and tok_ok
        all_ok = all_ok and ok
        lines.append(
            f"[{'OK ' if ok else 'FAIL'}] params={p} "
            f"| vocab: ref={len(ref.vocab)} new={len(new.vocab)} equal={vocab_ok} "
            f"| tokenize: len(ref)={len(tok_ref)} len(new)={len(tok_new)} equal={tok_ok}"
        )
        if not vocab_ok:
            diff_ref = ref.vocab - new.vocab
            diff_new = new.vocab - ref.vocab
            lines.append(f"    vocab 仅旧有: {sorted(diff_ref)[:10]}")
            lines.append(f"    vocab 仅新有: {sorted(diff_new)[:10]}")
        if not tok_ok:
            for idx in range(min(len(tok_ref), len(tok_new))):
                if tok_ref[idx] != tok_new[idx]:
                    lines.append(f"    tokenize 首差 @ {idx}: "
                                 f"ref={tok_ref[idx]!r} new={tok_new[idx]!r}")
                    break

    # ── 加速比实测（全文 23504 字符，各 1 次）──
    p_time = PARAMS[0]
    lines.append("")
    lines.append("=== O4 加速比实测（语料 = 全文 23504 字符，各 1 次）===")
    t0 = time.perf_counter()
    _ = RefWordSegmenter(text_full, **p_time)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = NewWordSegmenter(text_full, **p_time)
    t_new = time.perf_counter() - t0
    speedup = t_old / t_new if t_new > 0 else float("inf")
    lines.append(f"旧实现构建耗时: {t_old * 1000:.1f} ms")
    lines.append(f"新实现构建耗时: {t_new * 1000:.1f} ms")
    lines.append(f"加速比: {speedup:.1f}×  （目标 ≥ 5×）")
    if speedup < 5.0:
        lines.append("  [WARN] 加速比未达 5×")

    lines.append("")
    lines.append("对拍结论: " + ("全部一致 ✅" if all_ok else "存在不一致 ❌"))

    report = "\n".join(lines)
    print(report)

    # 同时写证据日志
    with open("D:/AiModel/train/outputs/test/verify_seg_equiv.log", "w", encoding="utf-8") as f:
        f.write(report + "\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
