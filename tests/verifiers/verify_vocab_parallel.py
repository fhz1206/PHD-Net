"""多核词表构建 / 全量扫描 的逐位等价对拍（fhz 2026-09-25 多核指令的验收门）。

用例
----
A  词涌现并行（按 L） vs 串行 —— 词集合完全一致（合成文本 + eval 内置语料）
A2 head 模式 token 收集并行 vs 串行 —— token 集合完全一致
B  全量扫描并行（锚点链） vs 串行贪心 —— token 流逐位一致；
   B2 在词表中**注入跨样本 token**（如 "\\n的"）强制走锚点链解析路径，
   多种切批粒度 / 进程数下仍逐位一致（锚点链机制的关键压力测试）
C  eval 内置语料端到端：并行流 ≡ 串行 StreamingTokenizer 流

任何一例 FAIL 即退出码非 0。运行：python train_1b/verify_vocab_parallel.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_HERE), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config_1b import SEG_KWARGS                                        # noqa: E402
from corpus_stream import PrefetchChars, SEP, StreamingTokenizer       # noqa: E402
from corpus_stream import char_chunks                                   # noqa: E402
from phdnet.word_encoder import WordSegmenter                           # noqa: E402
from vocab_parallel import build_segmenter_parallel                     # noqa: E402
from vocab_parallel import iter_tokens_parallel, parallel_head_tokens   # noqa: E402
from vocab_parallel import scan_vocab_parallel                          # noqa: E402

EVAL = _ROOT / "eval_corpus" / "internal_corpus.txt"

_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" —— {detail}" if detail else ""), flush=True)
    if not ok:
        _FAILURES.append(name)


class _StubSeg:
    """借 WordSegmenter.tokenize 同一实现做串行参照（同 vocab / max_len）。"""

    tokenize = WordSegmenter.tokenize

    def __init__(self, vocab: set[str], max_len: int):
        self.vocab = vocab
        self.max_len = max_len


def _synthetic_samples(n: int = 300) -> list[str]:
    """确定性合成语料：字符汤保证词涌现丰富；每 3 条以「的」开头，
    确保注入的跨样本 token（"\\n\\n的" 等）真实触发。"""
    import random
    rng = random.Random(7)
    pool = list("的了是我不在人他有这中大来说就和们到那要会着看好自己有事最么什 out 之由其")
    out = []
    for k in range(n):
        m = 30 + (k * 7) % 40
        body = "".join(rng.choice(pool) for _ in range(m))
        if k % 3 == 0:
            body = "的" + body
        out.append(body + SEP)
    return out


def main() -> None:
    t0 = time.perf_counter()

    # ── 公共材料 ──
    eval_text = EVAL.read_text(encoding="utf-8")
    samples = _synthetic_samples()
    full_text = "".join(samples)

    # ── A：词涌现并行 vs 串行（词集合一致）──
    print("[A] 词涌现并行（按 L）vs 串行")
    for tag, text in (("synthetic", full_text), ("eval", eval_text)):
        seg_s = WordSegmenter(text, **SEG_KWARGS)
        seg_p = build_segmenter_parallel(text, SEG_KWARGS, workers=4)
        same = seg_s.vocab == seg_p.vocab
        check(f"A.{tag} vocab 一致（|serial|={len(seg_s.vocab):,}）", same)
    seg_eval = WordSegmenter(eval_text, **SEG_KWARGS)

    # ── A2：head 模式 token 收集并行 vs 串行 ──
    print("[A2] head 模式 token 收集并行 vs 串行")
    toks_s = sorted(set(seg_eval.tokenize(eval_text)))
    toks_p = sorted(parallel_head_tokens(seg_eval, eval_text, workers=4))
    check(f"A2 eval token 集合一致（{len(toks_s):,}）", toks_s == toks_p)

    # ── B/B2：全量扫描锚点链并行 vs 串行贪心（token 流逐位一致）──
    print("[B] 全量扫描并行 vs 串行（真实涌现词表）")
    seg_syn = WordSegmenter(full_text, **SEG_KWARGS)
    stub = _StubSeg(seg_syn.vocab, SEG_KWARGS["max_len"])
    ref = stub.tokenize(full_text)
    par = list(iter_tokens_parallel(stub.vocab, stub.max_len, iter(samples),
                                    workers=2, group_chars=997))
    check(f"B synthetic 流一致（{len(ref):,} tokens）", par == ref)

    print("[B2] 注入跨样本 token 强制锚点链解析")
    vocab_x = set(seg_syn.vocab) | {"\n的", "\n\n的", "。\n的", "的\n\n", "\n\n"}
    stub_x = _StubSeg(vocab_x, SEG_KWARGS["max_len"])
    ref_x = stub_x.tokenize(full_text)
    n_cross = sum(1 for t in ref_x if "\n" in t and t.strip("\n"))
    check("B2 注入生效：跨样本 token 实际触发", n_cross > 0, f"n_cross={n_cross}")
    for workers in (2, 3):
        for gc in (37, 256, 100_000):
            par = list(iter_tokens_parallel(vocab_x, stub_x.max_len,
                                            iter(samples), workers, group_chars=gc))
            check(f"B2 w={workers} group={gc} 流一致"
                  f"（{len(ref_x):,} tokens，跨样本 token {n_cross} 个）", par == ref_x)
    seen_p, n_p = scan_vocab_parallel(vocab_x, stub_x.max_len, iter(samples),
                                      2, group_chars=256)
    check("B2 scan_vocab_parallel 集合/计数一致",
          seen_p == set(ref_x) and n_p == len(ref_x))

    # ── C：eval 语料端到端（并行流 ≡ 串行 StreamingTokenizer 流）──
    print("[C] eval 内置语料端到端对拍")
    ref_c = list(StreamingTokenizer(seg_eval, char_chunks(EVAL)))
    par_c = list(iter_tokens_parallel(seg_eval.vocab, seg_eval.max_len,
                                      char_chunks(EVAL), workers=3,
                                      group_chars=4096))
    check(f"C eval 流一致（{len(ref_c):,} tokens）", par_c == ref_c)

    # ── D：多进程数据加载 vs 串行（多文件语料，reorder 归并顺序验证）──
    print("[D] 多进程数据加载（PrefetchChars）vs 串行 char_chunks")
    import shutil as _shutil
    import tempfile
    tmpd = Path(tempfile.mkdtemp(prefix="vp_d_"))
    try:
        per = (len(samples) + 4) // 5
        for i in range(5):
            part = "".join(samples[i * per:(i + 1) * per])
            (tmpd / f"part_{i}.txt").write_text(part, encoding="utf-8")
        glob_path = tmpd / "part_*.txt"
        ref_d = list(char_chunks(glob_path))
        for dw in (1, 3):
            got = list(PrefetchChars(glob_path, workers=dw))
            check(f"D w={dw} 流一致（{len(ref_d):,} 样本 / 5 文件）", got == ref_d)
    finally:
        _shutil.rmtree(tmpd, ignore_errors=True)

    print("-" * 76)
    if _FAILURES:
        print(f"结果：{len(_FAILURES)} 例 FAIL → {_FAILURES}")
        sys.exit(1)
    print(f"结果：全部 PASS（{time.perf_counter() - t0:.1f}s）")


if __name__ == "__main__":
    main()
