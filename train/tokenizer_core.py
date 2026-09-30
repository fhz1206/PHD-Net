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

# 边数阈值：超过则用边哈希（大词表二分退化），否则用 CSR 二分（小词表更快）
EDGE_HASH_THRESHOLD = 20_000

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
    # 一次遍历建邻接表（2026-09-28 修复：原实现每个节点全量扫描 nxt，
    # O(节点数 × 词表规模) → 5 万词词表建 trie 需数十亿次迭代、直接卡死；
    # 服务器 full 扫描实测命中此坑。改为 O(词表规模) 一次分组 + BFS 展平）。
    kids_of: dict = {}                    # 原节点 -> [(char, 子节点), ...]
    for (p, c), m in nxt.items():
        kids_of.setdefault(p, []).append((c, m))
    # BFS 展平（保证子区间连续；char 排序留给展平阶段）
    order = [0]                            # 节点 → 展平 id
    flat = {}                              # 原节点 → 展平 id
    children = {}                          # 展平 id -> [(char, 原子节点)]
    flat[0] = 0
    i = 0
    while i < len(order):
        orig = order[i]                      # order 存**原节点**
        i += 1
        fid = flat[orig]                     # 展平 id（children 的键）
        kids = sorted(kids_of.get(orig, ()))
        children[fid] = kids
        for c, m in kids:
            if m not in flat:
                flat[m] = len(order)
                order.append(m)
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


def scan_group_numba(codes, body_len, max_len, trie, edge_hash=None):
    """单组任务（线程内执行，nogil 并行）：δ=0 主链 + 各 δ 前缀链。

    返回 (spine_off, spine_idx, exit0, per_delta)：
      spine_off/spine_idx —— uint32/int64 ndarray（截断到实际长度）
      exit0               —— 主链出口 = end_pos - body_len
      per_delta[i]        —— (d, prefix_idx, hit, exit_d)，与进程版同语义
    """
    cs, ce, cc, cn, tidx, _vl = trie
    n_edges = int(ce[-1]) if len(ce) else 0
    use_hash = n_edges > EDGE_HASH_THRESHOLD
    if use_hash:
        eh = edge_hash if edge_hash is not None else build_edge_hash(trie)
        ekeys, evals, emask = eh.keys, eh.vals, eh.mask
    else:
        ekeys = np.zeros(1, dtype=np.int64)
        evals = np.zeros(1, dtype=np.int32)
        emask = np.int64(0)
    n_cap = body_len + 2
    spine_off = np.empty(n_cap, dtype=np.int64)
    spine_idx = np.empty(n_cap, dtype=np.int64)
    empty = np.empty(0, dtype=np.uint8)
    if use_hash:
        k, end_pos, _ = track_until_hash(codes, ekeys, evals, emask, tidx,
                                        max_len, 0, body_len, empty,
                                        spine_idx, spine_off)
    else:
        k, end_pos, _ = track_until_bin(codes, cs, ce, cc, cn, tidx, max_len,
                                       0, body_len, empty,
                                       spine_idx, spine_off)
    spine_off = spine_off[:k].copy()
    spine_idx = spine_idx[:k].copy()
    exit0 = end_pos - body_len

    on_spine = np.zeros(codes.shape[0], dtype=np.uint8)
    on_spine[spine_off] = 1
    per_delta = []
    buf_idx = np.empty(n_cap, dtype=np.int64)
    buf_pos = np.empty(n_cap, dtype=np.int64)
    for d in range(1, max_len):
        if use_hash:
            k, end_d, hit = track_until_hash(codes, ekeys, evals, emask, tidx,
                                            max_len, d, body_len, on_spine,
                                            buf_idx, buf_pos)
        else:
            k, end_d, hit = track_until_bin(codes, cs, ce, cc, cn, tidx,
                                            max_len, d, body_len, on_spine,
                                            buf_idx, buf_pos)
        per_delta.append((d, buf_idx[:k].copy(),
                          hit, (end_d - body_len) if hit < 0 else exit0))
    return spine_off, spine_idx, exit0, per_delta


class _EdgeHash:
    """trie 边的开放寻址哈希：键 = node·2¹⁶ + char（int64，< 2⁴⁰ 无溢出）。

    2026-09-28（P15）：大词表下 trie **二分查找**退化——根节点扇出可达数千，
    每层 log2(fanout) 次 cache-missy 比较 × 最多 6 层，实测 5 万词词表扫描
    从 21.4 掉到 0.66 Mchar/s（32×）。改为边哈希后每层 O(1)：键用小整数
    运算（node、char 均 < 2²⁴/2¹⁶），**不涉及大整数字面量**（前一轮 FNV-1a
    方案因 numba 把大常量提升到 float64 而全表失配，已废弃）。

    语义与二分版逐位一致（同一 trie、同一最长匹配规则），仅加速查找路径。
    """

    # 注：keys 存 key+1（key 可为 0；|1 会在 char 为偶数时与 (node,char+1)
    # 撞键——汉字码点多为偶数，不可避免）。空槽 = 0。
    __slots__ = ("keys", "vals", "mask", "tidx")

    def __init__(self, keys, vals, mask, tidx):
        self.keys = keys        # int64 键（0 = 空槽；键 |1 保证非零）
        self.vals = vals        # int32 子节点
        self.mask = mask
        self.tidx = tidx        # int64 终结标记（沿用 trie 的）


def build_edge_hash(trie, node_cap: int = 1 << 19) -> _EdgeHash:
    """从 build_trie 的产物生成边哈希表（容量按边数自动翻倍）。"""
    cs, ce, cc, cn, tidx, _vlist = trie
    n_edges = int(ce[-1]) if len(ce) else 0
    cap = 8
    while cap < max(16, n_edges * 2):
        cap <<= 1
    mask = cap - 1
    keys = np.zeros(cap, dtype=np.int64)
    vals = np.zeros(cap, dtype=np.int32)
    for node in range(len(cs)):
        for j in range(cs[node], ce[node]):
            key = node * 65536 + int(cc[j])
            # ⚠ 必须混合后再取槽：key 低 16 位是 char（汉字码点高位相同），
            # 直接 key & mask 会让「同字不同节点」全部撞槽 → 线性探测退化
            # （实测 5 万词建表 265s、扫描 0.01 Mchar/s）。乘法混合后均匀。
            slot = ((key * 2654435761) & 0xFFFFFFFFFFFFFFFF) & mask
            while keys[slot] != 0:
                slot = (slot + 1) & mask
            keys[slot] = key + 1              # +1 表示占用（key 本身可为 0）
            vals[slot] = int(cn[j])
    return _EdgeHash(keys, vals, mask, tidx)


if NUMBA_TOK_OK:
    def _match_at(codes, cs, ce, cc, cn, tidx, p, n, max_len):
        """位置 p 起最长 L>=2 匹配（CSR trie 二分）；返回 (idx 或 -1, 长度)。

        ⚠ 二分用「lower_bound + 命中检查」，不用 hi 哨兵：后者在 lo==hi 收敛
        时无法区分「未找到」，对**低扇出节点**（如单孩子）会漏匹配。
        仅用于**小词表**（边少时二分比哈希快，实测 1,270 词：21.4 vs
        11.2 Mchar/s）；大词表走 _match_at_hash（见 build_edge_hash）。
        """
        node = 0
        best = -1
        best_len = 0
        lim = n - p
        if lim > max_len:
            lim = max_len
        for L in range(1, lim + 1):
            c = codes[p + L - 1]
            lo = cs[node]
            end = ce[node]
            hi = end
            while lo < hi:
                mid = (lo + hi) >> 1
                if cc[mid] < c:
                    lo = mid + 1
                else:
                    hi = mid
            if lo >= end or cc[lo] != c:
                break                              # 无此转移 → 停
            node = cn[lo]
            if tidx[node] >= 0:
                best = tidx[node]
                best_len = L
        return best, best_len

    @njit(cache=True, nogil=True, fastmath=False)
    def _match_at(codes, cs, ce, cc, cn, tidx, p, n, max_len):
        """位置 p 起最长 L>=2 匹配（CSR trie 二分）；返回 (idx 或 -1, 长度)。

        ⚠ 二分用「lower_bound + 命中检查」，不用 hi 哨兵：后者在 lo==hi 收敛
        时无法区分「未找到」，对**低扇出节点**（如单孩子）会漏匹配。
        仅用于**小词表**（边少时二分更快：1,270 词实测 21.4 vs 11.2 Mchar/s）；
        大词表走 _match_at_hash（边数 > EDGE_HASH_THRESHOLD）。
        """
        node = 0
        best = -1
        best_len = 0
        lim = n - p
        if lim > max_len:
            lim = max_len
        for L in range(1, lim + 1):
            c = codes[p + L - 1]
            lo = cs[node]
            end = ce[node]
            hi = end
            while lo < hi:
                mid = (lo + hi) >> 1
                if cc[mid] < c:
                    lo = mid + 1
                else:
                    hi = mid
            if lo >= end or cc[lo] != c:
                break                              # 无此转移 → 停
            node = cn[lo]
            if tidx[node] >= 0:
                best = tidx[node]
                best_len = L
        return best, best_len

    @njit(cache=True, nogil=True, fastmath=False)
    def _match_at_hash(codes, ekeys, evals, emask, tidx, p, n, max_len):
        """位置 p 起最长 L>=2 匹配（**边哈希**）；返回 (idx 或 -1, 长度)。

        键 = node·2¹⁶ + codes[p+L-1]，存 key+1（0 表空槽）；槽位经 Knuth
        乘法混合（直接取低位会让同字不同节点全部撞槽 → 探测退化）。
        语义与二分版逐位一致（同一 trie，最长 L∈[2,max_len] 命中者）。
        """
        node = 0
        best = -1
        best_len = 0
        lim = n - p
        if lim > max_len:
            lim = max_len
        for L in range(1, lim + 1):
            key = node * 65536 + int(codes[p + L - 1])
            slot = (key * 2654435761) & emask        # 乘法混合（int64 自然溢出）
            found = False
            while ekeys[slot] != 0:
                if ekeys[slot] == key + 1:
                    node = evals[slot]
                    found = True
                    break
                slot = (slot + 1) & emask
            if not found:
                break                              # 无此转移 → 停
            if tidx[node] >= 0:
                best = tidx[node]
                best_len = L
        return best, best_len

    @njit(cache=True, nogil=True, fastmath=False)
    def track_until_bin(codes, cs, ce, cc, cn, tidx, max_len,
                        start, stop, on_spine, out_idx, out_pos):
        """贪心轨道（CSR 二分路径，小词表）。GIL 释放；语义与哈希路径逐位一致。"""
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

    @njit(cache=True, nogil=True, fastmath=False)
    def track_until_hash(codes, ekeys, evals, emask, tidx, max_len,
                         start, stop, on_spine, out_idx, out_pos):
        """贪心轨道（边哈希路径，大词表）。GIL 释放；语义与二分路径逐位一致。"""
        k = 0
        p = start
        hit = -1
        n = codes.shape[0]
        while p < stop:
            if on_spine.shape[0] > 0:
                if on_spine[p] != 0:
                    hit = p
                    break
            idx, L = _match_at_hash(codes, ekeys, evals, emask, tidx,
                                    p, n, max_len)
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


def scan_group_numba(codes, body_len, max_len, trie, edge_hash=None):
    """单组任务（线程内执行，nogil 并行）：δ=0 主链 + 各 δ 前缀链。

    返回 (spine_off, spine_idx, exit0, per_delta)：
      spine_off/spine_idx —— uint32/int64 ndarray（截断到实际长度）
      exit0               —— 主链出口 = end_pos - body_len
      per_delta[i]        —— (d, prefix_idx, hit, exit_d)，与进程版同语义
    """
    cs, ce, cc, cn, tidx, _vl = trie
    n_edges = int(ce[-1]) if len(ce) else 0
    use_hash = n_edges > EDGE_HASH_THRESHOLD
    if use_hash:
        eh = edge_hash if edge_hash is not None else build_edge_hash(trie)
        ekeys, evals, emask = eh.keys, eh.vals, eh.mask
    else:
        ekeys = np.zeros(1, dtype=np.int64)
        evals = np.zeros(1, dtype=np.int32)
        emask = np.int64(0)
    n_cap = body_len + 2
    spine_off = np.empty(n_cap, dtype=np.int64)
    spine_idx = np.empty(n_cap, dtype=np.int64)
    empty = np.empty(0, dtype=np.uint8)
    if use_hash:
        k, end_pos, _ = track_until_hash(codes, ekeys, evals, emask, tidx,
                                        max_len, 0, body_len, empty,
                                        spine_idx, spine_off)
    else:
        k, end_pos, _ = track_until_bin(codes, cs, ce, cc, cn, tidx, max_len,
                                       0, body_len, empty,
                                       spine_idx, spine_off)
    spine_off = spine_off[:k].copy()
    spine_idx = spine_idx[:k].copy()
    exit0 = end_pos - body_len

    on_spine = np.zeros(codes.shape[0], dtype=np.uint8)
    on_spine[spine_off] = 1
    per_delta = []
    buf_idx = np.empty(n_cap, dtype=np.int64)
    buf_pos = np.empty(n_cap, dtype=np.int64)
    for d in range(1, max_len):
        if use_hash:
            k, end_d, hit = track_until_hash(codes, ekeys, evals, emask, tidx,
                                            max_len, d, body_len, on_spine,
                                            buf_idx, buf_pos)
        else:
            k, end_d, hit = track_until_bin(codes, cs, ce, cc, cn, tidx,
                                            max_len, d, body_len, on_spine,
                                            buf_idx, buf_pos)
        per_delta.append((d, buf_idx[:k].copy(),
                          hit, (end_d - body_len) if hit < 0 else exit0))
    return spine_off, spine_idx, exit0, per_delta


class _EdgeHash:
    """trie 边的开放寻址哈希：键 = node·2¹⁶ + char（int64，< 2⁴⁰ 无溢出）。

    2026-09-28（P15）：大词表下 trie **二分查找**退化——根节点扇出可达数千，
    每层 log2(fanout) 次 cache-missy 比较 × 最多 6 层，实测 5 万词词表扫描
    从 21.4 掉到 0.66 Mchar/s（32×）。改为边哈希后每层 O(1)：键用小整数
    运算（node、char 均 < 2²⁴/2¹⁶），**不涉及大整数字面量**（前一轮 FNV-1a
    方案因 numba 把大常量提升到 float64 而全表失配，已废弃）。

    语义与二分版逐位一致（同一 trie、同一最长匹配规则），仅加速查找路径。
    """

    # 注：keys 存 key+1（key 可为 0；|1 会在 char 为偶数时与 (node,char+1)
    # 撞键——汉字码点多为偶数，不可避免）。空槽 = 0。
    __slots__ = ("keys", "vals", "mask", "tidx")

    def __init__(self, keys, vals, mask, tidx):
        self.keys = keys        # int64 键（0 = 空槽；键 |1 保证非零）
        self.vals = vals        # int32 子节点
        self.mask = mask
        self.tidx = tidx        # int64 终结标记（沿用 trie 的）


def build_edge_hash(trie, node_cap: int = 1 << 19) -> _EdgeHash:
    """从 build_trie 的产物生成边哈希表（容量按边数自动翻倍）。"""
    cs, ce, cc, cn, tidx, _vlist = trie
    n_edges = int(ce[-1]) if len(ce) else 0
    cap = 8
    while cap < max(16, n_edges * 2):
        cap <<= 1
    mask = cap - 1
    keys = np.zeros(cap, dtype=np.int64)
    vals = np.zeros(cap, dtype=np.int32)
    for node in range(len(cs)):
        for j in range(cs[node], ce[node]):
            key = node * 65536 + int(cc[j])
            # ⚠ 必须混合后再取槽：key 低 16 位是 char（汉字码点高位相同），
            # 直接 key & mask 会让「同字不同节点」全部撞槽 → 线性探测退化
            # （实测 5 万词建表 265s、扫描 0.01 Mchar/s）。乘法混合后均匀。
            slot = ((key * 2654435761) & 0xFFFFFFFFFFFFFFFF) & mask
            while keys[slot] != 0:
                slot = (slot + 1) & mask
            keys[slot] = key + 1              # +1 表示占用（key 本身可为 0）
            vals[slot] = int(cn[j])
    return _EdgeHash(keys, vals, mask, tidx)


if NUMBA_TOK_OK:
    def _match_at(codes, cs, ce, cc, cn, tidx, p, n, max_len):
        """位置 p 起最长 L>=2 匹配（CSR trie 二分）；返回 (idx 或 -1, 长度)。

        ⚠ 二分用「lower_bound + 命中检查」，不用 hi 哨兵：后者在 lo==hi 收敛
        时无法区分「未找到」，对**低扇出节点**（如单孩子）会漏匹配。
        仅用于**小词表**（边少时二分比哈希快，实测 1,270 词：21.4 vs
        11.2 Mchar/s）；大词表走 _match_at_hash（见 build_edge_hash）。
        """
        node = 0
        best = -1
        best_len = 0
        lim = n - p
        if lim > max_len:
            lim = max_len
        for L in range(1, lim + 1):
            c = codes[p + L - 1]
            lo = cs[node]
            end = ce[node]
            hi = end
            while lo < hi:
                mid = (lo + hi) >> 1
                if cc[mid] < c:
                    lo = mid + 1
                else:
                    hi = mid
            if lo >= end or cc[lo] != c:
                break                              # 无此转移 → 停
            node = cn[lo]
            if tidx[node] >= 0:
                best = tidx[node]
                best_len = L
        return best, best_len

    @njit(cache=True, nogil=True, fastmath=False)
    def _match_at(codes, cs, ce, cc, cn, tidx, p, n, max_len):
        """位置 p 起最长 L>=2 匹配（CSR trie 二分）；返回 (idx 或 -1, 长度)。

        ⚠ 二分用「lower_bound + 命中检查」，不用 hi 哨兵：后者在 lo==hi 收敛
        时无法区分「未找到」，对**低扇出节点**（如单孩子）会漏匹配。
        仅用于**小词表**（边少时二分更快：1,270 词实测 21.4 vs 11.2 Mchar/s）；
        大词表走 _match_at_hash（边数 > EDGE_HASH_THRESHOLD）。
        """
        node = 0
        best = -1
        best_len = 0
        lim = n - p
        if lim > max_len:
            lim = max_len
        for L in range(1, lim + 1):
            c = codes[p + L - 1]
            lo = cs[node]
            end = ce[node]
            hi = end
            while lo < hi:
                mid = (lo + hi) >> 1
                if cc[mid] < c:
                    lo = mid + 1
                else:
                    hi = mid
            if lo >= end or cc[lo] != c:
                break                              # 无此转移 → 停
            node = cn[lo]
            if tidx[node] >= 0:
                best = tidx[node]
                best_len = L
        return best, best_len

    @njit(cache=True, nogil=True, fastmath=False)
    def _match_at_hash(codes, ekeys, evals, emask, tidx, p, n, max_len):
        """位置 p 起最长 L>=2 匹配（**边哈希**）；返回 (idx 或 -1, 长度)。

        键 = node·2¹⁶ + codes[p+L-1]，存 key+1（0 表空槽）；槽位经 Knuth
        乘法混合（直接取低位会让同字不同节点全部撞槽 → 探测退化）。
        语义与二分版逐位一致（同一 trie，最长 L∈[2,max_len] 命中者）。
        """
        node = 0
        best = -1
        best_len = 0
        lim = n - p
        if lim > max_len:
            lim = max_len
        for L in range(1, lim + 1):
            key = node * 65536 + int(codes[p + L - 1])
            slot = (key * 2654435761) & emask        # 乘法混合（int64 自然溢出）
            found = False
            while ekeys[slot] != 0:
                if ekeys[slot] == key + 1:
                    node = evals[slot]
                    found = True
                    break
                slot = (slot + 1) & emask
            if not found:
                break                              # 无此转移 → 停
            if tidx[node] >= 0:
                best = tidx[node]
                best_len = L
        return best, best_len

    @njit(cache=True, nogil=True, fastmath=False)
    def track_until(codes, cs, ce, cc, cn, ekeys, evals, emask, tidx, max_len,
                    use_hash, start, stop, on_spine, out_idx, out_pos):
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
            if use_hash:
                idx, L = _match_at_hash(codes, ekeys, evals, emask, tidx,
                                        p, n, max_len)
            else:
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
