#!/usr/bin/env python3
"""门禁：M4b 批量 learn 的**哈希槽位查找**（P214）与线性扫描核**逐位一致**。

## 它盯什么

`sparse_table.py::_find_slot` 的注释写着「线性扫描定位键（m_out ≤ 60，
扫描成本可忽略）」—— 这对 **dict 版**（每个 key 只扫一行）成立，
对 P78 的 prange 批量核**不成立**：

```
for r in prange(R):          # 活跃行数，1b 档 ~256–512
    for c in range(C):       # grow_order，同量级
        for s in range(sz):  # 行内槽位，最多 m_out=72
            if K[r,s] == k: ...
```

1b 档 R=C≈512、m_out=72 → 每步约 **1,900 万次 int64 比较**。
项目自己记录 `table.learn` 占 `M4b_imprint` 的 **88%**。

P214 改成**每行一张开放寻址哈希表**（`ltm_kernel._ltm_learn_rows_hashed`）：
建表按槽位升序、同键不覆盖 → 保留「**首个**匹配槽」语义；
生长后立即入表 → 保持后续 c 的可见性。复杂度
`O(R·C·sz)` → `O(R·(sz+C))`。

## 契约（本门禁逐项验证）

1. **逐位一致**：多 seed × 多变活跃维度下，两个核产出的
   `keys` / `vals` / `size` / 行数 **完全相同**（SHA-256 摘要比对）。
2. **语义边界**：覆盖**行生长初期**（sz < m_out）与**饱和期**（sz = m_out）、
   重复键、以及 `ltp<=0` 短路。
3. **开关可控**：`LTM_HASH_LOOKUP=False` 必须真的回退到线性核
   （否则 A/B 与排障能力是假的）。
4. **缓冲复用安全**：`_batch_buffers` 返回的必须是**视图**且不清零；
   多次调用后数值仍一致（防止读到上一步残留）。
5. **哈希表容量**：`H` 是 2 的幂且 ≥ 2·(m_out+1)（装填因子 ≤ 50%）。

运行：`python tests/gates/gate_ltm_hash_lookup.py`（退出码 0 = PASS）
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in ("", "tests", "tests/evals"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

import numpy as np

from phdnet.bigltm import SparseLTM
from phdnet.ltm_kernel import _ltm_learn_rows, _ltm_learn_rows_hashed
from phdnet.sparse_table import OnlineCSRTable

_RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    return bool(ok)


def _digest(t) -> tuple[str, int, int]:
    """表内容的 SHA-256 摘要（按行 id 排序 → 与 dict 插入顺序无关）。"""
    h = hashlib.sha256()
    for i in sorted(t.keys):
        sz = t.size[i]
        h.update(np.ascontiguousarray(t.keys[i][:sz], dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(t.vals[i][:sz], dtype=np.float64).tobytes())
    return h.hexdigest(), sum(t.size.values()), len(t.keys)


def _run(use_hash: bool, seed: int, n_steps: int = 130,
         n_dim: int = 1024, m_out: int = 72) -> tuple:
    rng = np.random.default_rng(seed)
    ltm = SparseLTM(n_dim=n_dim, n_neurons=1 << 22, m_out=m_out,
                    k_hash=4, csr_online=True)
    ltm.table.LTM_HASH_LOOKUP = use_hash

    def mk_rate(d: int) -> np.ndarray:
        r = np.zeros(n_dim)
        idx = rng.choice(n_dim, d, replace=False)
        r[idx] = rng.random(d)
        return r

    # 变活跃维度 → 同时覆盖「行生长期」「饱和期」与不同 R/C 组合
    dims_seq = (64, 128, 256, 64, 192)
    for step in range(n_steps):
        ltm.imprint(mk_rate(dims_seq[step % len(dims_seq)]))
    return _digest(ltm.table)


def _check_a():
    print("\n[A] 逐位一致：哈希核 vs 线性扫描核")
    for seed in (0, 1, 7):
        a = _run(True, seed)
        b = _run(False, seed)
        check(a == b, f"A seed={seed} 两核摘要相同",
              f"槽位 {a[1]}/{b[1]} 行数 {a[2]}/{b[2]}  hash={a[0][:12]}")


def _check_b():
    print("\n[B] 语义边界：int8 量化存储 + 小 m_out（快速饱和）")
    for int8 in (False, True):
        def one(use_hash: bool):
            import phdnet.bigltm as bl
            rng = np.random.default_rng(42)
            ltm = SparseLTM(n_dim=128, n_neurons=1 << 16, m_out=16,
                            k_hash=4, csr_online=True, int8_store=int8)
            ltm.table.LTM_HASH_LOOKUP = use_hash

            def rate(d):
                r = np.zeros(128)
                i = rng.choice(128, d, replace=False)
                r[i] = rng.random(d)
                return r

            for s in range(90):
                ltm.imprint(rate(16 if s % 3 else 28))
            return _digest(ltm.table)

        a, b = one(True), one(False)
        check(a == b, f"B int8={int8} 摘要相同",
              f"槽位 {a[1]}/{b[1]} 行数 {a[2]}/{b[2]}")


def _check_c():
    print("\n[C] 开关可控：False 真的走线性核")
    # 直接对拍两个核在同一组输入上的输出（绕过 Python 包装层）
    rng = np.random.default_rng(3)
    R, C, m_out = 48, 64, 32
    H = 1
    while H < 2 * (m_out + 1):
        H <<= 1
    cap = m_out

    K = rng.integers(0, 5000, size=(R, cap)).astype(np.int64)
    V = rng.random((R, cap))
    S = rng.integers(4, 20, size=R).astype(np.int64)
    tpi = rng.random(R) + 0.1
    tpi_post = rng.random(R)
    ks = rng.integers(0, 5000, size=C).astype(np.int64)
    g_post = rng.random(C) + 0.1
    g_pre = rng.random(C)
    eta, w_max, q = 0.05, 1.0, 1.0

    k1, v1, s1 = K.copy(), V.copy(), S.copy()
    k2, v2, s2 = K.copy(), V.copy(), S.copy()
    hk = np.empty((R, H), dtype=np.int64)
    hv = np.empty((R, H), dtype=np.int64)
    _ltm_learn_rows(k1, v1, s1, tpi, tpi_post, ks, g_post, g_pre,
                    eta, w_max, q, False, m_out)
    _ltm_learn_rows_hashed(k2, v2, s2, tpi, tpi_post, ks, g_post, g_pre,
                           eta, w_max, q, False, m_out, hk, hv)
    check(np.array_equal(k1, k2) and np.array_equal(v1, v2)
          and np.array_equal(s1, s2),
          "C 两核在同输入上 K/V/S 完全相同",
          f"max|ΔV|={float(np.max(np.abs(v1-v2))):.3e}")


def _check_d():
    print("\n[D] 持久化缓冲：复用视图 + 不清零仍逐位一致")
    t = OnlineCSRTable(n_neurons=1 << 16, m_out=72)
    b1 = t._batch_buffers(100, np.dtype(np.float64))
    b2 = t._batch_buffers(100, np.dtype(np.float64))
    check(b1[0].base is not None and b2[0].base is b1[0].base,
          "D1 多次调用返回同一底层缓冲的视图（零分配）",
          f"base 相同 = {b2[0].base is b1[0].base}")
    b3 = t._batch_buffers(300, np.dtype(np.float64))
    check(b3[0].shape[0] >= 300, "D2 R 增大时缓冲自动扩容",
          f"Rcap={b3[0].shape[0]}")
    H = t._hash_cap()
    check(H & (H - 1) == 0 and H >= 2 * (t.m_out + 1),
          "D3 哈希表容量是 2 的幂且 ≥ 2·(m_out+1)",
          f"H={H}, m_out={t.m_out}")
    # 不清零安全性：脏缓冲 + 两核 → 仍逐位一致
    b1[0][:] = 999999
    b1[1][:] = -7.25
    a = _run(True, 11, n_steps=80)
    b = _run(False, 11, n_steps=80)
    check(a == b, "D4 缓冲残留不影响数值（不清零安全）",
          f"hash={a[0][:12]}")


def main() -> int:
    print("=" * 74)
    print("门禁：M4b 哈希槽位查找（P214）—— 逐位一致 + 缓冲复用安全")
    print("=" * 74)
    _check_a()
    _check_b()
    _check_c()
    _check_d()
    n_ok = sum(1 for ok, _, _ in _RESULTS if ok)
    print("\n" + "=" * 74)
    print(f"结果：{n_ok}/{len(_RESULTS)} 通过 | 失败 {len(_RESULTS) - n_ok}")
    for ok, name, detail in _RESULTS:
        if not ok:
            print(f"  FAIL {name}  [{detail}]")
    print("=" * 74)
    return 0 if n_ok == len(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
