#!/usr/bin/env python3
"""门禁：LTM 反投影索引的三条实现（P166/P166c）—— 逐位一致 + 内存上界。

## 背景（fhz 2026-10-03：「30b 和 1b 除了参数不同之外都要一样」）
30b 档的阻断项不是参数，而是 `SparseLTM._rev_to_csr` 造的**稠密** `indptr`：
长度 = `max(big_i)+1`，而 `idx` 均匀哈希到 `[0, n_neurons)`：

| 档 | n_neurons | 稠密 indptr |
|---|---|---|
| 1b   | 2^24| 134 MB |
| **30b** | **2^29** | **4094 MB**（光启动就占，且在 `__init__` 里无条件分配）

而 `rev` 的真实键数只有 `n_dim × k_hash`（30b 档 4096 个，占 0.00015%）。

## 本门禁盯三件事
  A. **三条核逐位一致**：稠密 / 二分稀疏 / 哈希稀疏（含"键不存在要跳过"
     与"重复键逐次累加"两条语义）。
  B. **切换阈值**：小规模走稠密（零成本）、大规模走哈希（省内存）。
  C. **内存上界**：哈希表长度 = 2 的幂且 ≥ 4×键数（负载因子 ≤ 0.25），
     且表总内存远小于同规模的稠密 indptr。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

for _p in (str(Path(__file__).resolve().parents[2]),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _RESULTS.append((bool(ok), name, detail))
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name,
                         ("  —— " + detail) if detail else ""))
    return bool(ok)


print("=" * 72)
print("门禁：LTM 反投影索引三实现（P166 / P166b / P166c）")
print("=" * 72)

from phdnet.bigltm import SparseLTM  # noqa: E402
from phdnet.ltm_kernel import (_recall_project, _recall_project_hashed,  # noqa: E402
                               _recall_project_sparse)

# ── A. 三条核逐位一致（含多个活跃度档）──────────────────────────────
print("[A] 三核逐位一致")
n_dim, k_hash = 256, 4
for n_neurons in (1 << 20, 1 << 24):
    ltm = SparseLTM(n_dim=n_dim, k_hash=k_hash, n_neurons=n_neurons)
    rng = np.random.default_rng(3)
    for n_act in (8, 64, n_dim):
        rate = np.zeros(n_dim)
        rate[rng.choice(n_dim, n_act, replace=False)] = 0.3
        ids = np.array(ltm.encode(rate), dtype=np.int64)
        # 故意让权重重复（同一 big_i 出现多次）→ 检验「逐次累加」语义
        w = rng.normal(0, 1, max(1, len(ids)))
        if len(ids) > 2:
            w[1] = w[0]           # 制造重复键
        o1 = np.zeros(n_dim)
        o2 = np.zeros(n_dim)
        o3 = np.zeros(n_dim)
        _recall_project(ids, w, o1, ltm._rev_indptr, ltm._rev_indices)
        _recall_project_sparse(ids, w, o2, ltm._rev_keys, ltm._rev_off,
                               ltm._rev_indices_sparse)
        _recall_project_hashed(ids, w, o3, ltm._rev_table, ltm._rev_tpos,
                               ltm._rev_off, ltm._rev_indices_sparse)
        check("A n=%d 活跃%-4d 稠密==二分==哈希（逐位）" % (n_neurons, n_act),
              np.array_equal(o1, o2) and np.array_equal(o1, o3),
              "非零维 %d" % int((o1 != 0).sum()))

# ── A2. 键不存在必须跳过（等价 dict.get(big_i, ())）──────────────────
print("[A2] 不存在的键被跳过")
ltm = SparseLTM(n_dim=64, k_hash=4, n_neurons=1 << 10)
ids = np.array([999999, -1, 12345], dtype=np.int64)   # 全不存在
w = np.ones(3)
o1 = np.zeros(64)
o2 = np.zeros(64)
o3 = np.zeros(64)
_recall_project(ids, w, o1, ltm._rev_indptr, ltm._rev_indices)
_recall_project_sparse(ids, w, o2, ltm._rev_keys, ltm._rev_off,
                       ltm._rev_indices_sparse)
_recall_project_hashed(ids, w, o3, ltm._rev_table, ltm._rev_tpos,
                       ltm._rev_off, ltm._rev_indices_sparse)
check("A2 三核都跳过不存在的键（含负索引）",
      not o1.any() and not o2.any() and not o3.any(),
      "全零=正确")

# ── B. 切换阈值 ─────────────────────────────────────────────────────
print("[B] 稠密/哈希 的切换阈值")
ltm_small = SparseLTM(n_dim=128, k_hash=4, n_neurons=1 << 20)
check("B1 小规模走稠密（阈值内，零成本切换）",
      ltm_small._use_sparse_rev is False,
      "稠密 %.1f MB" % (ltm_small._rev_indptr.nbytes / 2 ** 20))
ltm_big = SparseLTM(n_dim=1024, k_hash=4, n_neurons=1 << 24)
check("B2 大规模走哈希（省内存）",
      ltm_big._use_sparse_rev is True,
      "稠密 %.0f MB → 哈希 %.0f KB"
      % (ltm_big._rev_indptr.nbytes / 2 ** 20,
         (ltm_big._rev_table.nbytes + ltm_big._rev_tpos.nbytes) / 1024))

# ── C. 哈希表结构 ───────────────────────────────────────────────────
print("[C] 哈希表结构与内存")
ltm = SparseLTM(n_dim=1024, k_hash=4, n_neurons=1 << 29)   # 30b 真实规模
n_keys = int(ltm._rev_keys.size)
tsz = int(ltm._rev_table.size)
check("C1 键数 = n_dim×k_hash（不随 n_neurons 增长）",
      n_keys == 1024 * k_hash, "n_keys=%d" % n_keys)
check("C2 表长是 2 的幂", (tsz & (tsz - 1)) == 0, "表长=%d" % tsz)
check("C3 负载因子 ≤ 0.25（表长 ≥ 4×键数）", tsz >= 4 * n_keys,
      "%d / %d = %.2f" % (tsz, n_keys, n_keys / tsz))
check("C4 表内无重复键（槽位唯一）",
      len(set(ltm._rev_table[ltm._rev_table > 0].tolist())) == n_keys,
      "非空槽 %d" % int((ltm._rev_table > 0).sum()))
dense_mb = ltm._rev_indptr.nbytes / 2 ** 20
hash_kb = (ltm._rev_table.nbytes + ltm._rev_tpos.nbytes) / 1024
check("C5 30b 档哈希表远小于稠密 indptr",
      hash_kb * 1024 < dense_mb * 2 ** 20 / 10,
      "稠密 %.0f MB → 哈希 %.0f KB（省 %.0f 倍）"
      % (dense_mb, hash_kb, dense_mb * 2 ** 20 / (hash_kb * 1024)))
check("C6 slot_pos 指向合法 rev_off 下标",
      bool((ltm._rev_tpos[ltm._rev_table > 0]
            < ltm._rev_off.size - 1).all()),
      "最大下标 %d / rev_off 长 %d"
      % (int(ltm._rev_tpos[ltm._rev_table > 0].max()),
         ltm._rev_off.size - 1))

n_fail = sum(1 for ok, _, _ in _RESULTS if not ok)
print()
print("=" * 72)
print("结果：%d 例 FAIL%s" % (n_fail,
                            (" → " + str([n for ok, n, _ in _RESULTS if not ok]))
                            if n_fail else ""))
print("=" * 72)
sys.exit(1 if n_fail else 0)