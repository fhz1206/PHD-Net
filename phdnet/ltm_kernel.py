"""M4b 大空间表可 numba 化部分的核（P68）。

fhz 2026-09-29：「给可以用 numba 来加速的都用上」。本模块覆盖**不需要重构存储**
的两段每步都调的热点：

1. ~~`SparseLTM.encode`~~ —— **实测 0.84× 负收益，调用方已回退 Python**：
   核内去重是 O(n²) 线性扫描，输给 Python `set` 的 O(1) 哈希（200 维 →
   799 索引：304 → 362 µs）。核函数保留在此仅作记录，要重启需改成排序+相邻
   去重或小型开放寻址哈希表（参照 P18 词表合并）。真正上线的是：
1. `SparseLTM.recall` 的反投影累加（**已上线，404 → 50 µs = 8.07×**）：按被激活大空间索引经反向索引累加到
   n_dim 维（原为 Python 双层循环）。反向索引 `rev` 是**静态**的（`__init__`
   里一次性构建），因此可以零维护成本地预 CSR 化
   （`rev_indptr` / `rev_indices`，维度序保持升序 = 原 dict list 序 → 逐位）。

**为什么是 nogil 串行而不是 prange**：
- `recall` 里不同大空间索引会**共享同一个输出维度**（一个维度 j 被多个 big_i
  绑定）→ prange 并行写 `out[j] += s` 是数据竞态，numba 无原子保证 → **串行**；
- `encode` 的去重天然串行。
即便串行，收益也来自「消除 Python 解释开销 + 释放 GIL」（原实现每个绑定维度
约 50–100 ns 解释器开销，C 内约 1–2 ns）。

**逐位一致**（`tests/verifiers/verify_ltm_kernels.py` 把关）：遍历次序、累加
次序、浮点加法顺序全部与 Python 版一致。
`learn` / `predict` 的多核化需要把在线 CSR 存储改成连续数组（避免
gather/scatter 拷贝），属独立立项，本模块不含。
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    NUMBA_LTM = True
except ImportError:                                     # pragma: no cover
    from numba import njit, prange                      # numba 缺失时这里会抛
    NUMBA_LTM = True


@njit(cache=True, nogil=True, fastmath=False)
def _encode_hash_uniq(dims, idx, k_hash, n_slots):
    """激活维度 → k_hash 哈希索引去重（保持首次出现顺序，与 Python 版一致）。

    dims : (D,) 激活维度下标（升序，来自 np.nonzero）
    idx  : (n_dim, k_hash) 预计算的大空间索引表
    返回 (out, n)：去重后的索引与个数（写入 out[:n]）。
    """
    out = np.empty(n_slots, dtype=np.int64)
    n = 0
    D = dims.shape[0]
    for a in range(D):
        j = dims[a]
        for b in range(k_hash):
            i = idx[j, b]
            dup = False
            for t in range(n):
                if out[t] == i:
                    dup = True
                    break
            if not dup:
                out[n] = i
                n += 1
    return out, n


@njit(cache=True, nogil=True, fastmath=False)
def _recall_project(big_ids, weights, out, rev_indptr, rev_indices):
    """反投影累加：out[j] += Σ_t weights[t]（对 big_ids[t] 绑定的每个维度 j）。

    串行（共享输出维度 → prange 会竞态）。累加次序 = `big_ids` 的给定次序；
    重复键按出现次序逐次累加——与 Python 版
    `for i in active: for s: p[k] += v` 的浮点累加顺序完全一致 → 逐位相同。
    越界的 big_i 跳过，等价 `dict.get(big_i, ())` 返回空元组。
    """
    n_rev = rev_indptr.shape[0] - 1
    for t in range(big_ids.shape[0]):
        b = big_ids[t]
        if b < 0 or b >= n_rev:
            continue
        w = weights[t]
        for p in range(rev_indptr[b], rev_indptr[b + 1]):
            out[rev_indices[p]] += w


@njit(cache=True, nogil=True, fastmath=False)
def _recall_project_sparse(big_ids, weights, out, rev_keys, rev_off,
                            rev_indices):
    """**P166**：反投影累加（**稀疏键 + 二分**版，替代稠密 `rev_indptr`）。

    语义与 `_recall_project`（稠密版）**完全相同**，包括：
      · 累加次序 = `big_ids` 的给定次序（重复键逐次累加）；
      · 键不存在 → 跳过（等价 `dict.get(big_i, ())` 返回空元组）；
      · 维度序 = 原 dict list 序（j 升序）→ 逐位与Python 版相同。

    改动只在**怎么找 b 落在哪个区间**：
      稠密版：`rev_indptr[b]` 直接下标 → 数组长 n_keys（30b 档 4.0 GB，
              随机访问必然 cache miss）；
      稀疏版：在 rev_keys（长 = 实际键数 12288，**98 KB**）里二分，
              区间从 rev_off 取 → **全在 L1/L2**。

    ⚠ 二分的比较次数是 log2(12288) ≈ 14，但每次都在**几 KB 的热区**，
      而稠密版一次访问就要穿透 4 GB 的地址空间。
    """
    n_keys = rev_keys.shape[0]
    for t in range(big_ids.shape[0]):
        b = big_ids[t]
        if b < 0:
            continue
        # 二分：找 rev_keys 中等于 b 的位置（升序）。
        lo = 0
        hi = n_keys - 1
        pos = -1
        while lo <= hi:
            mid = (lo + hi) >> 1
            v = rev_keys[mid]
            if v == b:
                pos = mid
                break
            if v < b:
                lo = mid + 1
            else:
                hi = mid - 1
        if pos < 0:
            continue                      # 该 big_i 没绑定任何维度
        w = weights[t]
        for p in range(rev_off[pos], rev_off[pos + 1]):
            out[rev_indices[p]] += w


@njit(cache=True, nogil=True, fastmath=False)
def _recall_project_hashed(big_ids, weights, out, slot_keys, slot_pos,
                            rev_off, rev_indices):
    """**P166c**：开放寻址哈希，**命中后零次二分**（真O(1)）。

    ### 为什么改（P166b 的设计失误）
    P166b 哈希表只存**键**，命中后还要**再二分一次** `rev_keys` 才能拿到区间
    起点 → 实测比纯二分版**还慢 0.9×**（30b 档活跃 1024：二分 341.7 µs、
    哈希 389.9 µs）。二分白做了。

    P166c：**两个平行数组**（`slot_keys` / `slot_pos`），槽里同时放
    「键+1」与「该键在 rev_off 里的下标」→ 命中即得区间，**不再二分**。
    代价：两张表各128 KB（30b 档），共 256 KB —— 仍远小于稠密 4094 MB。

    语义与 `_recall_project`（稠密）/ `_recall_project_sparse`（二分）
      **完全相同**：键不存在 → 跳过；累加次序 = `big_ids` 给定次序；
      维度序 = 原 dict list 序（j 升序）。

    ⚠ 「键 + 1」：0 表示空槽。键 = 大空间索引< 2^32，int64 下无溢出。
    ⚠ 槽位哈希用**乘法混合**（MEMORY 的 numba 铁律：`key & mask` 在低位
      有结构时会退化）。
    """
    mask = slot_keys.shape[0] - 1
    for t in range(big_ids.shape[0]):
        b = big_ids[t] + 1                # +1 使「空槽 0」可区分
        if b == 0:                # 原 big_i == -1 → 跳过
            continue
        slot = (b * 2654435761) & mask
        while True:
            v = slot_keys[slot]
            if v == 0 or v == b:
                break
            slot = (slot + 1) & mask
        if slot_keys[slot] == 0:
            continue                      # 该键不存在（空槽）
        pos = slot_pos[slot]              # 命中即得，**无二次查找**
        w = weights[t]
        for q in range(rev_off[pos], rev_off[pos + 1]):
            out[rev_indices[q]] += w


@njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _ltm_learn_rows(K, V, S, tpi, tpi_post, ks, g_tpost, g_tpre,
                    eta, w_max, q, int8, m_out):
    """P78：imprint 的 `learn` 批量多核化（行间 prange，行内与原版同序）。

    K, V   : (R, cap) 活跃行缓冲（cap >= m_out；原路径写满会 `_grow_row` 扩容
             到 m_out，故按 m_out 分配即可，扩容在 scatter 侧补）
    S      : (R,) 有效槽数（就地更新）
    tpi    : (R,) 行前端迹（<=0 的行整行跳过，与原版一致）
    tpi_post: (R,) 行后端迹（LTD 用）
    ks / g_tpost / g_tpre : (C,) grow_order 与两条迹

    逐位要点：`ltp = eta*tpi*tpk`；`ltp <= 0` 跳过；`found` 取**首个**匹配槽；
    有槽 → `w + ltp - ltd`（ltd = eta*tpi_post*g_tpre[c]）；无槽且 `sz < m_out`
    → `min(ltp, w_max)` 生长（**与 growth_guidance 无关**，它只管 in_deg）；
    int8 写回 `int(round(clip(w)*q))`（banker's rounding 与 Python `round` 一致）。
    """
    R = K.shape[0]
    C = ks.shape[0]
    for r in prange(R):
        tp = tpi[r]
        if tp <= 0.0:
            continue
        tp_post = tpi_post[r]
        sz = S[r]
        for c in range(C):
            ltp = eta * tp * g_tpost[c]
            if ltp <= 0.0:
                continue
            k = ks[c]
            found = -1
            for s in range(sz):
                if K[r, s] == k:
                    found = s
                    break
            if found >= 0:
                if int8:
                    w = np.float64(np.int32(V[r, found])) / np.float64(q)
                else:
                    w = np.float64(V[r, found])
                ltd = eta * tp_post * g_tpre[c]
                w2 = w + ltp - ltd
                if w2 < 0.0:
                    w2 = 0.0
                elif w2 > w_max:
                    w2 = w_max
                if int8:
                    V[r, found] = np.int16(int(np.round(w2 * q)))
                else:
                    V[r, found] = np.float64(w2)
            elif sz < m_out:
                w2 = ltp if ltp < w_max else w_max
                K[r, sz] = k
                if int8:
                    V[r, sz] = np.int16(int(np.round(w2 * q)))
                else:
                    V[r, sz] = np.float64(w2)
                sz += 1
                S[r] = sz
