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
    from numba import njit
    NUMBA_LTM = True
except ImportError:                                     # pragma: no cover
    NUMBA_LTM = False

    def njit(*a, **kw):
        def wrap(fn):
            return fn
        return wrap if not a or not callable(a[0]) else a[0]


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
    """反投影累加：out[j] += Σ_{t: big_ids[t] 绑定了 j} weights[t]。

    串行（共享输出维度 → prange 会竞态）。累加次序 = big_ids 的给定次序，
    与 Python 版 `for big_i, s in scores.items(): for j in rev[big_i]` 完全一致。
    越界的 big_i 直接跳过（等价原 dict `.get(big_i, ())` 返回空元组）。
    """
    n_rev = rev_indptr.shape[0] - 1
    for t in range(big_ids.shape[0]):
        b = big_ids[t]
        if b < 0 or b >= n_rev:
            continue
        w = weights[t]
        for p in range(rev_indptr[b], rev_indptr[b + 1]):
            out[rev_indices[p]] += w
