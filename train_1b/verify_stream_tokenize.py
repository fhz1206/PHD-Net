"""对拍：StreamingTokenizer（流式）vs WordSegmenter.tokenize（全量）逐位一致。

验证声明（corpus_stream.py）：
  贪心最长匹配的在线实现与全量实现在同一词表、同一语料流定义
  （concat(样本_i + SEP)）下产出**逐位相同**的 token 序列，
  且与分块方式无关（样本可长可短——任意长度语料不截断的正确性基础）。

用例：
  1. 真实语料（eval 内置语料）整体流式 vs 全量 tokenize；
  2. 人为把语料切成极小样本（1–200 字符，强制频繁跨界）再流式 vs 全量；
  3. 词长边界压力：构造恰好 max_len 与 max_len+1 的跨块候选词。
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_HERE), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config_1b import SEG_KWARGS
from corpus_stream import SEP, StreamingTokenizer, char_chunks
from phdnet.word_encoder import WordSegmenter

ok = True


def check(tag: str, text: str, chunks: list[str], seg: WordSegmenter) -> bool:
    """chunks 流式分词 vs text 全量分词，逐位比对。"""
    full = list(seg.tokenize(text))
    stream = list(StreamingTokenizer(seg, iter(chunks)))
    same = full == stream
    global ok
    ok &= same
    n_diff = next((i for i, (a, b) in enumerate(zip(full, stream)) if a != b),
                  None) if not same else None
    print(f"  {tag}: 全量 {len(full):,} tokens vs 流式 {len(stream):,} tokens → "
          f"{'PASS ✓' if same else f'FAIL ✗（首个差异 @{n_diff}: {full[n_diff]!r} vs {stream[n_diff]!r}）'}")
    return same


def main() -> None:
    seg = WordSegmenter(" ".join(char_chunks(
        _ROOT / "datasets" / "eval" / "internal_corpus.txt")), **SEG_KWARGS)
    print(f"词表 {len(seg.vocab):,} 涌现词 | max_len={seg.max_len}")

    # ── 用例 1：真实语料，按既有样本分块（流式定义）──
    chunks = list(char_chunks(_ROOT / "datasets" / "eval" / "internal_corpus.txt"))
    text1 = "".join(chunks)
    check("用例1 真实语料（样本分块）", text1, chunks, seg)

    # ── 用例 2：极小样本（1–200 字符随机切，强制频繁跨界）──
    # 注意：切块后每块注入 SEP → 对照文本必须用拼接后的结果（同一文本，两种分词方式）
    import random
    rng = random.Random(7)
    pieces = [text1[:20000]][0]
    chunks2, i = [], 0
    while i < len(pieces):
        k = rng.randint(1, 200)
        chunks2.append(pieces[i:i + k] + SEP)
        i += k
    text2 = "".join(chunks2)
    check("用例2 极小样本切分（1–200 字符）", text2, chunks2, seg)

    # ── 用例 3：词长边界压力——单块内构造 max_len 恰好跨界的样本 ──
    # 从词表取最长词，构造「词前半 | 词后半」跨越样本边界的用例
    longest = max(seg.vocab, key=len) if seg.vocab else "测测试试"
    L = len(longest)
    cut = max(1, L // 2)
    chunks3 = [longest[:cut] + SEP, longest[cut:] + SEP, "收尾文本" + SEP]
    text3 = "".join(chunks3)
    check(f"用例3 词长边界（{longest!r} 切于 {cut}）", text3, chunks3, seg)

    # ── 用例 4：单字符块（最极端跨界频率）──
    chunks4 = [c + SEP for c in text2[:3000]]
    text4 = "".join(chunks4)
    check("用例4 单字符块", text4, chunks4, seg)

    print("=" * 60)
    print("流式分词逐位等价: " + ("PASS ✓（全部用例一致）" if ok else "FAIL ✗"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
