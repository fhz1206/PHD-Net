"""train_1b 多核词表构建与全量扫描（逐位等价并行化）。

依据（fhz 2026-09-25 指令）
---------------------------
- 「词表构建的时候用多核cpu，用核心数量的0.8」→ `auto_workers()`
  = max(1, floor(cpu_count × 0.8))，`--vocab-workers 0`（默认）即取该值。
- 「数据加载也用多核」→ 见 `corpus_stream.PrefetchChars`（生产者进程预取）
  与本模块的并行扫描（parquet 解码由主进程流水供给 worker 池）。
- 「训练过程不要存在等待代码」→ 全模块零 sleep/轮询/忙等；
  所有同步均为 OS 级阻塞（队列、Future.result()）。

并行化结构与逐位等价声明（硬约束）
----------------------------------
1. 词涌现（WordSegmenter）按 L 层并行：
   各 L（2..max_len）的统计相互独立，`_induce_length` 与串行版是**同一份
   实现**、同一 numpy 运算序列 → 词集合逐位一致（集合无序，归并安全）。

2. 全量扫描按批（group）并行 —— 锚点链机制：
   贪心最长匹配是**无状态位置轨道**（下一决策只依赖当前位置：
   L(p) = 满足 text[p:p+L] ∈ vocab 的最大 L，否则 1）。
   把语料字符流切成正文块（每块带 2×max_len 字符前视）并行送 worker；
   组 g 的真实入口锚点 δ_g ∈ [0, max_len)（上一组轨道出口）。worker 对
   每个候选 δ 计算轨道，并以 δ=0 主链（spine）做因子化：各 δ 链 =
   前缀 + 主链切片（轨道相遇同一位置后必然重合，位置级 memo 共享计算）。
   编排器按 δ_0=0 起串行接链（每组 O(1) 查表），组间 token 严格无缝分区：
   并行输出 ≡ 串行 StreamingTokenizer 逐位一致（含跨组/跨样本 token）。
   workers=1 时调用方直接走串行原路径（本模块不参与）。

内存护栏：worker 峰值 ≈ 组文本 + 组 token 表（group_chars 缺省 1M 字符
≈ 数 MB/worker）；词涌现并行的瞬时峰值 ≈ 各层 W 矩阵之和（4M 字符
采样、max_len=6 时 ≈1.5 GB，12 GB 级机器可承受）。

P13（2026-09-28，fhz「线程 + GIL 解锁吃满核心」指令）：全量扫描热路径
numba 化（tokenizer_core.py，njit nogil）→ **ThreadPoolExecutor** 真多核：
共享内存零 pickle，主进程只做 µs 级 ndarray 接链与 bincount 位图合并。
服务器实测教训（进程版）：153 worker 全睡眠、主进程单核 100%——
组 32M 字符过大（并行度塌缩）+ 每组数百万 token 字符串 pickle 单点。
group_chars 缺省 32M → 1M（组数 ≫ worker 数，IPC/内存双降）。
numba 不可用时自动回退进程池路径（Python 版 _scan_group，逐位一致）。
"""

from __future__ import annotations

import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

try:
    from tokenizer_core import (EDGE_HASH_THRESHOLD, NUMBA_TOK_OK, OOV_FLAG,
                                build_edge_hash, build_trie,
                                scan_group_numba, text_to_codes)
except Exception:                                            # pragma: no cover
    NUMBA_TOK_OK = False

_TRIE_CACHE: dict = {}                                       # (max_len, frozenset) -> trie


def _trie_for(vocab, max_len: int):
    """trie 一次性构建（同词表跨组复用；frozenset 哈希做缓存键）。"""
    key = (max_len, frozenset(vocab))
    ent = _TRIE_CACHE.get(key)
    if ent is None:
        # trie +（大词表时）边哈希一次性构建并缓存（P15：大词表下二分退化
        # 32×，边哈希 O(1) 且带乘法混合；建表 265s -> 0.28s）
        tr = build_trie(vocab)
        n_edges = int(tr[1][-1]) if len(tr[1]) else 0
        eh = build_edge_hash(tr) if n_edges > EDGE_HASH_THRESHOLD else None
        ent = (tr, eh)
        _TRIE_CACHE.clear()                  # 只保留最新词表（生产场景词表唯一）
        _TRIE_CACHE[key] = ent
    return ent


def auto_workers() -> int:
    """fhz 指定口径：核心数量的 0.8（向下取整，至少 1）。"""
    return max(1, (os.cpu_count() or 1) * 8 // 10)


# ────────────────────────── 1. 词涌现按 L 并行 ──────────────────────────

def build_segmenter_parallel(text: str, seg_kwargs: dict | None = None,
                             workers: int = 0):
    """词涌现多核构建（P18：numba nogil + prange，层间**线程池**零 pickle）。

    引擎二选一（词集合逐位一致，verify_vocab_parallel A 例把关）：
      - numba 可用 → `phdnet.word_induce_numba`：层内 prange（默认核数/10 线程）
        + 层间 ThreadPoolExecutor（nogil 释放 GIL → 真并行）；
      - 否则 → 原 numpy `_induce_length` + ProcessPoolExecutor（回退路径）。
    workers≤1 或退化输入时走串行原路径。
    """
    from phdnet.word_encoder import WordSegmenter, _induce_length
    try:
        from phdnet.word_induce_numba import (NUMBA_WORD_OK,
                                              default_nthreads,
                                              induce_length_numba)
    except Exception:                            # pragma: no cover
        NUMBA_WORD_OK = False

    kwargs = dict(seg_kwargs or {})
    max_len = int(kwargs.get("max_len", 6))
    min_count = int(kwargs.get("min_count", 5))
    min_entropy = float(kwargs.get("min_entropy", 1.0))
    workers = auto_workers() if workers in (0, None) else int(workers)
    if workers <= 1 or max_len < 2 or len(text) < 2:
        return WordSegmenter(text, **kwargs)

    seg = WordSegmenter.__new__(WordSegmenter)
    seg.max_len = max_len
    seg.vocab = set()
    n = len(text)
    codes, inv_codes = np.unique(list(text), return_inverse=True)
    inv_codes = inv_codes.astype(np.int64)
    nw = min(workers, max_len - 1)               # 任务数只有 max_len-1 个

    if NUMBA_WORD_OK:
        nth = default_nthreads()                # 核内 prange 线程（核数/10）
        with ThreadPoolExecutor(max_workers=nw) as ex:
            futs = [ex.submit(induce_length_numba, codes, inv_codes, n, L,
                              min_count, min_entropy, nth)
                    for L in range(2, max_len + 1)]
            for f in futs:                       # 按提交序归并（集合无序，确定）
                seg.vocab |= f.result()
    else:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = [ex.submit(_induce_length, codes, inv_codes, n, L,
                              min_count, min_entropy)
                    for L in range(2, max_len + 1)]
            for f in futs:
                seg.vocab |= f.result()
    return seg


# ──────────────────── 2. 全量扫描按批并行（锚点链） ────────────────────

_VOCAB: frozenset | None = None                # worker 进程内的词表（initializer 注入）


def _scan_init(vocab) -> None:
    global _VOCAB
    _VOCAB = frozenset(vocab)


def _scan_group(group_text: str, body_len: int, max_len: int):
    """单批任务：δ=0 主链（spine）+ 各候选入口 δ 的前缀链（位置级 memo 共享）。

    返回 (spine_offsets, spine_tokens, exit0, per_delta)：
      spine_offsets/tokens —— 从 δ=0 起的贪心轨道（起点 < body_len 的全部 token）
      exit0                —— 主链出口 = 第一个 ≥ body_len 的轨道位置 − body_len
      per_delta[i]         —— (δ=i+1, prefix_tokens, hit, exit_δ)
        hit ≥ 0：该 δ 链在位置 hit 与主链重合 → 其 token = prefix + spine[hit:]
                 （exit_δ = exit0，重合后未来必然相同）
        hit = -1：未与主链相遇 → 其 token = prefix 全部（exit_δ = 自己的出口）
    """
    vocab = _VOCAB
    n = len(group_text)
    memo: dict[int, str] = {}                    # pos → 贪心 token（跨 δ 共享）

    def tok_at(p: int) -> str:
        t = memo.get(p)
        if t is None:
            for L in range(min(max_len, n - p), 1, -1):
                if group_text[p:p + L] in vocab:
                    t = group_text[p:p + L]
                    break
            else:
                t = group_text[p]                # 单字符回退（与串行同一判定）
            memo[p] = t
        return t

    spine_off: list[int] = []
    spine_tok: list[str] = []
    pos = 0
    while pos < body_len:
        t = tok_at(pos)
        spine_off.append(pos)
        spine_tok.append(t)
        pos += len(t)
    exit0 = pos - body_len

    spine_set = set(spine_off)
    per_delta = []
    for d in range(1, max_len):
        prefix: list[str] = []
        p = d
        hit = -1
        while p < body_len:
            if p in spine_set:
                hit = p
                break
            t = tok_at(p)
            prefix.append(t)
            p += len(t)
        if hit >= 0:
            per_delta.append((d, prefix, hit, exit0))
        else:
            per_delta.append((d, prefix, -1, p - body_len))
    return spine_off, spine_tok, exit0, per_delta


def _group_iter(samples, group_chars: int, lookahead: int):
    """字符级切批：yield (组全文 = 正文 + 前视, 正文长度)。

    组间正文严格无缝分区（carry 机制：上一组借出的前视残余在下一轮
    先归还进正文；前视借用也先消费 carry 再向源取）。组边界可为任意
    位置（锚点链对边界无假设）。
    P13：carry 改 list 收集 + 一次性 join（原 str += 是 O(n²) 复制，
    组 1M 字符 × 样本千级时主进程内存带宽成瓶颈）。
    """
    it = iter(samples)
    carry = ""                                   # 已取出、尚未划入正文的字符
    src_done = False
    while True:
        parts: list[str] = []
        plen = 0
        if carry:
            parts.append(carry)
            plen = len(carry)
            carry = ""
        while plen < group_chars and not src_done:
            try:
                s = next(it)
                parts.append(s)
                plen += len(s)
            except StopIteration:
                src_done = True
        if plen == 0:
            return
        full = parts[0] if len(parts) == 1 else "".join(parts)
        if len(full) < group_chars:
            # 攒不满一组 ⟹ 源已耗尽（plen>0 已保证）→ 最后一组，无前视可借
            yield full, len(full)
            return
        body = full[:group_chars]
        rest = full[group_chars:]
        while len(rest) < lookahead and not src_done:    # 借前视（不消费）
            try:
                rest += next(it)
            except StopIteration:
                src_done = True
        la = rest[:lookahead]                    # 纯前视（peek，不消费）
        carry = rest                             # rest 整体归还：下一组正文紧接 body
        yield body + la, len(body)


def _scan_task_numba(text: str, body_len: int, max_len: int, ent):
    """线程池任务体：文本 → 码点 → nogil 轨道（encode/nogil 均释放 GIL）。

    ent = (trie, edge_hash|None) 由 _trie_for 一次性构建并缓存。
    """
    tr, eh = ent
    return scan_group_numba(text_to_codes(text), body_len, max_len, tr, eh)


def _iter_group_tokens(vocab, max_len: int, samples, workers: int,
                       group_chars: int):
    """编排器：按序 yield 每组的真实 token 索引数组（锚点链接链）。

    返回元素 = np.ndarray uint32：词表内 token = vocab_list 下标（升序与
    build_trie 的 vocab_list 一致）；单字符 OOV = 0x8000_0000 | ord(ch)。
    P13：numba 可用时走 ThreadPoolExecutor（nogil 真多核、零 pickle）；
    否则回退进程池 Python 版（逐位一致）。
    预提交窗口 = 2×workers+2 组（有界 → 内存有界、背压自然形成）；
    消费按提交序（Future.result() 为 OS 级阻塞，无忙等）。
    """
    groups = _group_iter(samples, group_chars, 2 * max_len)
    if NUMBA_TOK_OK:
        _t_trie = time.perf_counter()
        ent = _trie_for(vocab, max_len)
        print(f"  [词表扫描] 分词结构构建（{len(vocab):,} 词）："
              f"{time.perf_counter() - _t_trie:.1f}s（一次性；此后各组复用）",
              flush=True)
        ex = ThreadPoolExecutor(max_workers=workers)
        submit = _scan_task_numba
        targs = (max_len, ent)
    else:
        ex = ProcessPoolExecutor(max_workers=workers,
                                 initializer=_scan_init,
                                 initargs=(tuple(vocab),))
        submit = _scan_group
        targs = (max_len,)
    tok2idx = None if NUMBA_TOK_OK else {w: i for i, w in enumerate(sorted(vocab))}
    try:
        window: deque = deque()                  # 待消费 Future 队列
        src_done = False
        delta = 0                                # 当前组真实入口锚点
        while True:
            while not src_done and len(window) < 2 * workers + 2:
                try:
                    text, body_len = next(groups)
                    window.append(ex.submit(submit, text, body_len, *targs))
                except StopIteration:
                    src_done = True
            if not window:
                return
            res = window.popleft().result()
            if NUMBA_TOK_OK:
                spine_off, spine_idx, exit0, per_delta = res
            else:                                # 进程版：str list → idx ndarray
                spine_off, spine_tok, exit0, per_delta = res
                spine_idx = np.fromiter(
                    (tok2idx.get(t, 0) if len(t) > 1 else
                     int(OOV_FLAG) | ord(t) for t in spine_tok),
                    dtype=np.uint32, count=len(spine_tok))
                spine_off = np.array(spine_off, dtype=np.int64)
                per_delta = [(d, np.fromiter(
                    (tok2idx.get(t, 0) if len(t) > 1 else
                     int(OOV_FLAG) | ord(t) for t in pref),
                    dtype=np.uint32, count=len(pref)) if pref else
                    np.empty(0, dtype=np.uint32), hit, ex_d)
                    for (d, pref, hit, ex_d) in per_delta]
            if delta == 0:
                toks = spine_idx
                new_delta = exit0
            else:
                d, prefix, hit, exit_d = per_delta[delta - 1]
                if hit >= 0:
                    i0 = int(np.searchsorted(spine_off, hit))
                    toks = np.concatenate([prefix, spine_idx[i0:]])
                    new_delta = exit0
                else:
                    toks = prefix
                    new_delta = exit_d
            yield toks
            delta = new_delta
    finally:
        for f in window:
            f.cancel()
        ex.shutdown(wait=False, cancel_futures=True)


def _idx_to_str(vocab_list: list[str], toks: np.ndarray):
    """idx ndarray → str 列表（OOV 高位码还原单字符；验证路径专用）。"""
    n_v = len(vocab_list)
    flag = int(OOV_FLAG)
    return [vocab_list[i] if i < n_v else chr(i & (flag - 1)) for i in toks.tolist()]


def iter_tokens_parallel(vocab, max_len: int, samples, workers: int,
                         group_chars: int = 1_000_000):
    """并行全量扫描：逐个 yield token（顺序 ≡ 串行 StreamingTokenizer）。"""
    vocab_list = sorted(vocab)
    for toks in _iter_group_tokens(vocab, max_len, samples, workers,
                                   group_chars):
        yield from _idx_to_str(vocab_list, toks)


def scan_vocab_parallel(vocab, max_len: int, samples, workers: int,
                        group_chars: int = 1_000_000, progress=None):
    """并行全量扫描（词表收集形态）：返回 (seen 集合, 扫过 token 总数)。

    P13：合并走 C 级 —— 词表内 token 用 bincount 位图 OR（µs 级/组），
    OOV 单字符 unique 后 chr 还原；不再逐 token Python set.update。
    progress(累计 tokens, 当前词表数) 在每组合并后回调。
    """
    vocab_list = sorted(vocab)
    n_v = len(vocab_list)
    mask = np.zeros(n_v + 1, dtype=bool)
    seen_oov: set[str] = set()
    n_total = 0
    flag = int(OOV_FLAG)
    for toks in _iter_group_tokens(vocab, max_len, samples, workers,
                                   group_chars):
        n_total += int(toks.size)
        tk = toks.astype(np.int64, copy=False)
        in_v = tk[tk < flag]
        if in_v.size:
            mask |= np.bincount(in_v, minlength=n_v + 1).astype(bool)
        oov = tk[tk >= flag]
        if oov.size:
            seen_oov.update(map(chr, np.unique(oov & (flag - 1)).tolist()))
        if progress is not None:
            progress(n_total, int(mask.sum()) + len(seen_oov))
    seen = set(np.array(vocab_list, dtype=object)[mask[:n_v]].tolist())
    seen |= seen_oov
    return seen, n_total


def parallel_head_tokens(seg, vocab_text: str, workers: int) -> set[str]:
    """head 模式 token 收集的多核版：把采样文本切块送锚点链扫描。

    与串行 `set(seg.tokenize(vocab_text))` 逐位一致（同一词表、同一贪心轨道）。
    P13：组大小取 max(200K, len/(2×workers))——太小则每组固定开销占比过高。
    """
    group_chars = max(200_000, len(vocab_text) // (workers * 2))
    seen, _ = scan_vocab_parallel(seg.vocab, seg.max_len,
                                  (vocab_text[i:i + group_chars]
                                   for i in range(0, len(vocab_text), group_chars)),
                                  workers, group_chars)
    return seen
