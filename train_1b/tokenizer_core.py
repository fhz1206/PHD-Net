# -*- coding: utf-8 -*-
"""numba nogil 贪心分词核心（P13，2026-09-28）——全量扫描多核线程化的热路径。

为什么是线程而不是进程（fhz 2026-09-28「线程 + GIL 解锁吃满核心」指令的落地）：
  原锚点链扫描是纯 Python 对象操作（dict/set 查找、str 切片），全程持有 GIL——
  线程化只能单核；进程化则被主进程 pickle 收发卡死（服务器实测 153 worker 全睡眠、
  主进程单核 100%）。本模块把热循环编译为释放 GIL 的机器码（numba njit nogil）：
  线程池并行 = 真多核，且共享内存零 pickle，主进程只做 µs 级接链。

逐位等价承诺（对拍验收 = tests/verifiers/verify_vocab_parallel.py 全部用例）：
  与 vocab_parallel._scan_group（Python 进程版）及串行 StreamingTokenizer 的
  贪心轨道**逐位一致**：最长匹配 L=max_len..2（单字符永不查词表、走 OOV 编码），
  memo 缓存不改变语义，锚点链 δ 前缀链语义原样保留。

词表编码：trie 只收录 L>=2 的词（与 Python 版「L=1 不查表」语义对齐）；
  单字符 OOV 用高位标记 0x8000_0000 | ord(ch)（chr() 无损还原，跨组零翻译）。
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    NUMBA_TOK_OK = True
except Exception:                                            # pragma: no cover
    NUMBA_TOK_OK = False

OOV_FLAG = np.uint32(0x8000_0000)                            # 单字符 OOV 高位标记
OOV_MASK = np.uint32(0x7FFF_FFFF)


def text_to_codes(text: str) -> np.ndarray:
    """str → uint32 码点数组（utf-32 零转换语义；C 级 memcpy）。"""
    return np.frombuffer(text.encode("utf-32-le"), dtype="<u4")


def build_trie(vocab) -> tuple:
    """词表 → CSR trie。

    返回 (child_start, child_end, child_char, child_node, term_idx, vocab_list)：
      节点 0 = 根；节点 n 的子节点占据 child_char/child_node 的
      [child_start[n], child_end[n]) 区间（按 char 升序 → 核内二分）；
      term_idx[n] >= 0 表示节点 n 是某词的结尾，值 = vocab_list 下标。
      长度 < 2 的词不进 trie（走 OOV 编码），但保留在 vocab_list（idx 稳定）。
    """
    vocab_list = sorted(set(vocab))
    tok2idx = {w: i for i, w in enumerate(vocab_list)}
    nxt = {}                               # (node, char) -> node
    term = {}                              # node -> vocab idx
    for w, i in tok2idx.items():
        if len(w) < 2:
            continue
        n = 0
        for ch in w:
            n = nxt.setdefault((n, ord(ch)), len(nxt) + 1)
        term.setdefault(n, i)
    # BFS 展平（保证子区间连续；char 排序留给展平阶段）
    order = [0]                            # 节点 → 展平 id
    flat = {}                              # 原节点 → 展平 id
    children = {}                          # 展平 id -> [(char, 原子节点)]
    frontier = [0]
    flat[0] = 0
    while frontier:
        n = frontier.pop(0)
        kids = sorted((c, m) for (p, c), m in nxt.items() if p == n)
        children[flat[n]] = kids
        for c, m in kids:
            if m not in flat:
                flat[m] = len(order)
                order.append(m)
                frontier.append(m)
    N = len(order)
    child_start = np.zeros(N, dtype=np.int64)
    child_end = np.zeros(N, dtype=np.int64)
    cc_list: list[int] = []
    cn_list: list[int] = []
    term_idx = np.full(N, -1, dtype=np.int64)
    for fid in range(N):
        kids = children[fid]
        child_start[fid] = len(cc_list)
        for c, m in kids:
            cc_list.append(c)
            cn_list.append(flat[m])
        child_end[fid] = len(cc_list)
        orig = order[fid]
        if orig in term:
            term_idx[fid] = term[orig]
    return (child_start, child_end,
            np.array(cc_list, dtype=np.int64), np.array(cn_list, dtype=np.int64),
            term_idx, vocab_list)


def scan_group_numba(codes, body_len, max_len, trie):
    """单组任务（线程内执行，nogil 并行）：δ=0 主链 + 各 δ 前缀链。

    返回 (spine_off, spine_idx, exit0, per_delta)：
      spine_off/spine_idx —— uint32/int64 ndarray（截断到实际长度）
      exit0               —— 主链出口 = end_pos - body_len
      per_delta[i]        —— (d, prefix_idx, hit, exit_d)，与进程版同语义
    """
    cs, ce, cc, cn, tidx, _ = trie
    n_cap = body_len + 2
    spine_off = np.empty(n_cap, dtype=np.int64)
    spine_idx = np.empty(n_cap, dtype=np.int64)
    empty = np.empty(0, dtype=np.uint8)
    k, end_pos, _ = track_until(codes, cs, ce, cc, cn, tidx, max_len,
                                0, body_len, empty, spine_idx, spine_off)
    spine_off = spine_off[:k].copy()
    spine_idx = spine_idx[:k].copy()
    exit0 = end_pos - body_len

    on_spine = np.zeros(codes.shape[0], dtype=np.uint8)
    on_spine[spine_off] = 1
    per_delta = []
    buf_idx = np.empty(n_cap, dtype=np.int64)
    buf_pos = np.empty(n_cap, dtype=np.int64)
    for d in range(1, max_len):
        k, end_d, hit = track_until(codes, cs, ce, cc, cn, tidx, max_len,
                                    d, body_len, on_spine, buf_idx, buf_pos)
        per_delta.append((d, buf_idx[:k].copy(),
                          hit, (end_d - body_len) if hit < 0 else exit0))
    return spine_off, spine_idx, exit0, per_delta


if NUMBA_TOK_OK:
    @njit(cache=True, nogil=True, fastmath=False)
    def _match_at(codes, cs, ce, cc, cn, tidx, p, n, max_len):
        """位置 p 起最长 L>=2 匹配；返回 (vocab idx 或 -1, 匹配长度)。"""
        node = 0
        best = -1
        best_len = 0
        lim = n - p
        if lim > max_len:
            lim = max_len
        for L in range(1, lim + 1):
            c = codes[p + L - 1]
            lo = cs[node]
            hi = ce[node]
            while lo < hi:                         # 子节点二分
                mid = (lo + hi) >> 1
                if cc[mid] < c:
                    lo = mid + 1
                elif cc[mid] > c:
                    hi = mid
                else:
                    lo = mid
                    hi = -2
                    break
            if hi != -2:
                break                              # 无此转移 → 停
            node = cn[lo]
            if tidx[node] >= 0:
                best = tidx[node]
                best_len = L
        return best, best_len

    @njit(cache=True, nogil=True, fastmath=False)
    def track_until(codes, cs, ce, cc, cn, tidx, max_len,
                    start, stop, on_spine, out_idx, out_pos):
        """贪心轨道：从 start 扫到 stop（匹配可延伸到 codes 末尾——前视区）。

        on_spine 非空时：p 命中 spine 位置 → 提前返回（δ 前缀链语义）。
        返回 (n_out, end_pos, hit)：end_pos = 轨道结束位置（可 > stop，溢出）；
        hit = 首个 spine 命中位置或 -1。OOV 单字符 = 0x80000000 | 码点。
        与 Python 版 tok_at 逐位一致（最长匹配 L=max_len..2，单字符回退）。
        """
        k = 0
        p = start
        hit = -1
        n = codes.shape[0]
        while p < stop:
            if on_spine.shape[0] > 0:
                if on_spine[p] != 0:
                    hit = p
                    break
            idx, L = _match_at(codes, cs, ce, cc, cn, tidx, p, n, max_len)
            if idx >= 0:
                out_idx[k] = idx
                out_pos[k] = p
                p += L
                k += 1
            else:
                out_idx[k] = 0x80000000 | codes[p]
                out_pos[k] = p
                p += 1
                k += 1
        return k, p, hit
