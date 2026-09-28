# -*- coding: utf-8 -*-
"""词涌现的 numba nogil + prange 多核核（P18，2026-09-28）。

fhz 问「为啥哪怕 nogil 也不会吃满所有核心」——因为 nogil 只解决「不占 GIL」，
不解决「把一个核的活拆到多核」。本模块用三层并行把词涌现真正铺满：

  层 1（核内 prange）：遍 1/遍 2 按**位置分块**，每线程持有**局部计数表**
        （无原子竞争），块结束复制进全局表并置零（经典 lock-free 合并）；
  层 2（遍 3 串行）：汇总只扫表槽（O(cap)），相对遍 1/2 是小头，串行以避免
        per-slot 并行缓冲（nthreads×4×cap）爆内存；
  层 3（层间线程池）：5 个 L 各自独立 → ThreadPoolExecutor（nogil 释放 GIL，
        真并行、零 pickle；旧实现是 ProcessPoolExecutor + pickle）。

替代原实现（`phdnet/word_encoder.py::_induce_length`）的两处热点：
  ① `np.unique(W, axis=0)` 对 (m, L) 滑窗矩阵整行排序去重（m ≈ 400 万，
     纯 numpy 不释放 GIL）；
  ② 熵的 Python 逐候选循环（全程持 GIL）。

⚠ 浮点口径：原实现是「压缩非零 + 1D pairwise Σ p log p」，本核用解析式
   H = log(Σc) − (Σ c log c)/Σc（数学等价、无灾难性抵消）。差异 1 ulp 量级，
   `tests/verifiers/verify_word_induce.py` 给出实测词集合差异数（预期 0）。
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    NUMBA_WORD_OK = True
except Exception:                                            # pragma: no cover
    NUMBA_WORD_OK = False

_P1 = 1000003               # 滚动 hash 乘子（< 2^31，避开 numba 大常量陷阱）


if NUMBA_WORD_OK:
    @njit(cache=True, nogil=True, fastmath=False)
    def _win_hash(inv, i, L):
        h = 0
        for j in range(L):
            h = h * _P1 + inv[i + j]
            h ^= h >> 7
        return h

    @njit(cache=True, nogil=True, fastmath=False)
    def induce_words_kernel(inv, n, L, B, min_count, min_entropy, nthreads):
        """单层 L-gram 涌现核（nogil + prange 多核）。返回 (K, L) 字符 id 矩阵。

        无原子竞争的并行策略：prange 块内用**局部表**，块末逐槽合并进全局表
        并清零局部表（每槽最多被 nthreads 个块写，故冲突面 = 槽数 × 块数）。
        """
        m = n - L + 1
        if m <= 0:
            return np.empty((0, L), dtype=np.int64)
        if nthreads < 1:
            nthreads = 1

        cap = 8
        while cap < 2 * m:
            cap <<= 1
        msk = cap - 1
        g_keys = np.full(cap, -1, dtype=np.int64)
        g_counts = np.zeros(cap, dtype=np.int64)
        g_first = np.full(cap, -1, dtype=np.int64)

        # ── 遍 1（prange）：窗口频次，块内局部表 ──
        chunk = (m + nthreads - 1) // nthreads
        # 局部表按**块大小**分配（条目上界 = 块内窗口数），而不是全表 cap：
        # 否则每线程要 memset 3×cap（4M 文本 × 6 线程 ≈ 1.2 GB 清零，纯浪费）。
        cap_l = 8
        while cap_l < 2 * chunk:
            cap_l <<= 1
        msk_l = cap_l - 1
        for bi in prange(nthreads):
            lo = bi * chunk
            hi = min(lo + chunk, m)
            if lo >= hi:
                continue
            l_keys = np.full(cap_l, -1, dtype=np.int64)
            l_counts = np.zeros(cap_l, dtype=np.int64)
            l_first = np.full(cap_l, dtype=np.int64, fill_value=-1)
            for i in range(lo, hi):
                h = _win_hash(inv, i, L)
                slot = h & msk_l
                while l_keys[slot] != -1:
                    if l_keys[slot] == h:
                        p = l_first[slot]
                        same = True
                        for j in range(L):
                            if inv[p + j] != inv[i + j]:
                                same = False
                                break
                        if same:
                            l_counts[slot] += 1
                            break
                    slot = (slot + 1) & msk_l
                else:
                    l_keys[slot] = h
                    l_counts[slot] = 1
                    l_first[slot] = i
            for slot in range(cap_l):
                # 块末合并：**必须走全局表的开放寻址插入**（按槽直写会让
                # 不同块的**不同键互相覆盖** → 丢条目 → 词频低估 → 少接受词）。
                if l_keys[slot] == -1:
                    continue
                h = l_keys[slot]
                cnt = l_counts[slot]
                fp = l_first[slot]
                gs = h & msk
                while g_keys[gs] != -1:
                    if g_keys[gs] == h:
                        g_counts[gs] += cnt
                        if fp < g_first[gs]:
                            g_first[gs] = fp
                        break
                    gs = (gs + 1) & msk
                else:
                    g_keys[gs] = h
                    g_counts[gs] = cnt
                    g_first[gs] = fp
                l_keys[slot] = -1
                l_counts[slot] = 0
                l_first[slot] = -1

        # ── 候选筛选（prange 友好：先数再标）──
        is_cand = np.zeros(cap, dtype=np.uint8)
        ncand = 0
        for slot in range(cap):
            if g_keys[slot] != -1 and g_counts[slot] >= min_count:
                is_cand[slot] = 1
                ncand += 1

        # ── 遍 2（prange）：候选 slot 的邻居身份计数（键含方向）──
        cap2 = 8
        while cap2 < 2 * m:
            cap2 <<= 1
        msk2 = cap2 - 1
        g2_keys = np.full(cap2, -1, dtype=np.int64)
        g2_counts = np.zeros(cap2, dtype=np.int64)
        # 遍 2 局部表：条目上界 = 块内窗口数 × 2（左右邻各一）
        cap2l = 8
        while cap2l < 4 * chunk:
            cap2l <<= 1
        msk2l = cap2l - 1
        for bi in prange(nthreads):
            lo = bi * chunk
            hi = min(lo + chunk, m)
            if lo >= hi:
                continue
            l2_keys = np.full(cap2l, -1, dtype=np.int64)
            l2_counts = np.zeros(cap2l, dtype=np.int64)
            for i in range(lo, hi):
                h = _win_hash(inv, i, L)
                slot = h & msk
                hit = -1
                while g_keys[slot] != -1:
                    if g_keys[slot] == h:
                        p = g_first[slot]
                        same = True
                        for j in range(L):
                            if inv[p + j] != inv[i + j]:
                                same = False
                                break
                        if same:
                            hit = slot
                            break
                    slot = (slot + 1) & msk
                if hit < 0 or is_cand[hit] == 0:
                    continue
                if i > 0:
                    key2 = hit * 2 * B + inv[i - 1]
                    s2 = key2 & msk2l
                    while l2_keys[s2] != -1:
                        if l2_keys[s2] == key2:
                            l2_counts[s2] += 1
                            break
                        s2 = (s2 + 1) & msk2l
                    else:
                        l2_keys[s2] = key2
                        l2_counts[s2] = 1
                if i + L < n:
                    key2 = (hit * 2 + 1) * B + inv[i + L]
                    s2 = key2 & msk2l
                    while l2_keys[s2] != -1:
                        if l2_keys[s2] == key2:
                            l2_counts[s2] += 1
                            break
                        s2 = (s2 + 1) & msk2l
                    else:
                        l2_keys[s2] = key2
                        l2_counts[s2] = 1
            for s2 in range(cap2l):
                if l2_keys[s2] == -1:
                    continue
                h = l2_keys[s2]
                cnt = l2_counts[s2]
                gs = h & msk2
                while g2_keys[gs] != -1:
                    if g2_keys[gs] == h:
                        g2_counts[gs] += cnt
                        break
                    gs = (gs + 1) & msk2
                else:
                    g2_keys[gs] = h
                    g2_counts[gs] = cnt
                l2_keys[s2] = -1
                l2_counts[s2] = 0

        # ── 遍 3（串行）：per-slot 汇总 Σc / Σ(c log c) + 判据 ──
        # 汇总只扫表槽（O(cap2)），相对遍 1/2（O(m) 次 hash + 探测）是小头，
        # 串行实现以避免 prange 的 per-slot 缓冲分配（nthreads×4×cap 会爆内存）。
        sum_l = np.zeros(cap, dtype=np.float64)
        hl_l = np.zeros(cap, dtype=np.float64)
        sum_r = np.zeros(cap, dtype=np.float64)
        hl_r = np.zeros(cap, dtype=np.float64)
        for s2 in range(cap2):
            key2 = g2_keys[s2]
            if key2 == -1:
                continue
            side = (key2 // B) & 1
            hit = (key2 // B) >> 1
            cnt = g2_counts[s2]
            if cnt <= 0:
                continue
            fc = float(cnt)
            lg = fc * np.log(fc)
            if side == 0:
                sum_l[hit] += fc
                hl_l[hit] += lg
            else:
                sum_r[hit] += fc
                hl_r[hit] += lg

        out = np.empty((ncand, L), dtype=np.int64)
        n_out = 0
        for slot in range(cap):
            if is_cand[slot] == 0:
                continue
            s = sum_l[slot]
            h_l = np.log(s) - hl_l[slot] / s if s > 0.0 else 0.0
            s = sum_r[slot]
            h_r = np.log(s) - hl_r[slot] / s if s > 0.0 else 0.0
            if h_l >= min_entropy and h_r >= min_entropy:
                p0 = g_first[slot]
                for j in range(L):
                    out[n_out, j] = inv[p0 + j]
                n_out += 1
        return out[:n_out]


def default_nthreads() -> int:
    """默认 prange 线程数 = **总核数 / 10**（fhz 2026-09-28 定调）。

    理由：① prange 的局部计数表按**块**分配（内存 O(chunk)），线程数过多会
    放大分配与合并开销；② 外层还有**层间线程池**（5 个 L 各自 nogil 并行），
    总并行度 = nthreads × 层数，故每层取核数的 1/10 量级即可铺满
    （191 核 → 19 × 5 ≈ 95 线程，留出主线程与 I/O 余量）；
    ③ 上限 64（再大收益递减且局部表缓存压力上升）。
    """
    import multiprocessing
    cpu = multiprocessing.cpu_count() or 1
    return max(1, min(64, cpu // 10))


def induce_length_numba(codes, inv_codes, n: int, L: int, min_count: int,
                        min_entropy: float, nthreads: int = 0) -> set[str]:
    """单层词涌现（nogil + prange 多核核 + Python 侧字符串还原）。"""
    B = int(codes.shape[0])
    if nthreads <= 0:
        nthreads = default_nthreads()
    words = induce_words_kernel(inv_codes, n, L, B, min_count, min_entropy,
                                nthreads)
    codes_list = [str(c) for c in codes]
    return {''.join(codes_list[i] for i in row) for row in words}
