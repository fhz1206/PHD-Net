"""稀疏 CSR 向量化优化门禁：**逐位不变 + 无性能回归**（exit 0 = PASS）。

================================================================================
为什么需要这个文件
================================================================================
`phdnet/sparse_pc.py` 两处热点从逐行 Python 循环改成向量化实现：
  · `_from_dense_csr`：逐行 argsort+slice → 整行**全量** argsort
    （np.argsort(-np.abs(W), axis=1)[:, :k] + 行内 sort + take_along_axis）；
  · `SparsePCStack._dense`：逐行 for 赋值 → np.repeat + 高级索引一次写入。
收益要量、正确性要**逐位**证明——本门禁把两件事都固化成可复跑断言。
性能优化一旦破坏逐位（会被 verify_m2_kernels / verify_pc_learn_fused /
verify_readout_sparse_gate / run_tests --fast 等对拍门禁抓到），必须回退
优化而不是改门禁口径。

【A】向量化 vs 朴素参考实现**逐位对拍**（3 个规模 × tie-heavy 家族）
  为什么 tie 是这类改写的头号风险：argsort 对并列值（|w| 相同）的次序决定
  k 窗口边界处取哪几列 → 直接决定 CSR 的 idx 结构。numpy 默认
  introsort 的 tie-break 依赖「同一逻辑序列 + 同一算法」，所以实现必须用
  **全量 argsort**（⚠ 禁止 argpartition：它不给 tie-break 保证），而本节
  专门用 全 0 矩阵 / 大量重复值 / 两值矩阵 / k=1 / k=n_cols / NaN 行
  把边界情形打满。值路径 fp64→fp32 按位比（view(uint32)）。

【B】固定 seed 的 infer/learn 输出对拍 golden（golden 存 %TEMP%）
  · golden 在优化落地**前**由基准脚本生成：%TEMP%/perf_golden_.npz
    （115 个数组）+ %TEMP%/perf_golden.sha256（内容哈希 = 跨进程确定性证据）；
  · 本门禁重建同样构造 → 逐数组 np.array_equal + sha256 双重比对；
  · 同时跑**朴素实现自比对**：把模块里两个函数临时换回朴素参考实现再重建
    一遍 → 必须与向量化版逐位一致（证「向量化无副作用」+「结果确定性」），
    并在进程内重建第二遍证同进程确定性。

【C】性能断言：预热（吃掉 numba JIT / 分配器热身）+ best-of-7；
  阈值 = 优化后实测（两轮实测中较慢者）× 2 防抖动，输出打印实测 ms。

用法：py -3.14 tests/verifiers/verify_perf_no_regression.py
"""
from __future__ import annotations

import hashlib
import sys
import tempfile as _tmp
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import phdnet.sparse_pc as spc                              # noqa: E402
from phdnet.sparse_pc import SparsePCStack                  # noqa: E402

_RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    return bool(ok)


# ─────────────────────────────────────────────────────────── 朴素参考实现
# （= 优化前 sparse_pc.py 的原始代码，逐行照抄；仅作对拍基准，永不进生产）
def naive_from_dense_csr(W: np.ndarray, k: int):
    n_rows, n_cols = W.shape
    k = max(1, min(k, n_cols))
    indptr = np.arange(0, (n_rows + 1) * k, k, dtype=np.int64)
    idx = np.empty(n_rows * k, dtype=np.int64)
    val = np.empty(n_rows * k, dtype=np.float32)
    for r in range(n_rows):
        cols = np.argsort(-np.abs(W[r]))[:k]
        cols.sort()
        s = slice(r * k, (r + 1) * k)
        idx[s] = cols
        val[s] = W[r, cols]
    return indptr, idx, val


def naive_dense(indptr, idx, val, n_rows: int, n_cols: int) -> np.ndarray:
    W = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        p0, p1 = indptr[i], indptr[i + 1]
        W[i, idx[p0:p1]] = val[p0:p1]
    return W


def csr_equal(a, b) -> bool:
    return (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
            and np.array_equal(a[2].view(np.uint32), b[2].view(np.uint32)))


def best_of(fn, n: int = 7, warm: int = 2) -> float:
    """预热 warm 次（吃掉 numba JIT/首分配）后 best-of-n，返回 ms。"""
    for _ in range(warm):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return min(times)


# ───────────────────────────────────────────── (B) golden 构造（固定 seed）
# ⚠ 与 %TEMP%/perf_bench.py 的 golden 构造**逐行一致**（golden 在优化落地前
#   由它生成）；改动此构造 = golden 失效，须重新生成并说明。
def build_golden() -> dict:
    G: dict = {}
    rng = np.random.default_rng(1234)
    Wbig = rng.normal(size=(1024, 1024))
    G["Wbig"] = Wbig
    for k in (1, 64, 1024):
        ip, ix, vl = spc._from_dense_csr(Wbig, k)
        G[f"fdcsr_{k}_ip"] = ip
        G[f"fdcsr_{k}_ix"] = ix
        G[f"fdcsr_{k}_val"] = vl

    def stack_seq(fused, homeo=False):
        pc = SparsePCStack(128, 96, 64, eta_pc=0.05, eta_oja=0.01,
                           rng=np.random.default_rng(99), conn_k=16, fused=fused)
        s0 = np.random.default_rng(5).normal(size=128)
        out = {}
        for step in range(5):
            cache = pc.infer(s0, n_steps=2)
            pc.learn(cache, homeostasis=homeo)
            for kk, v in cache.items():
                out[f"step{step}_{kk}"] = np.asarray(v)
        out["up0_val"] = pc.up0[2].copy()
        out["up1_val"] = pc.up1[2].copy()
        out["dn0_val"] = pc.dn0[2].copy()
        out["dn1_val"] = pc.dn1[2].copy()
        out["hn_up0"] = pc._hn_up0.copy()
        out["dense_up0"] = SparsePCStack._dense(*pc.up0, pc.n1, pc.n0)
        out["dense_dn1"] = SparsePCStack._dense(*pc.dn1, pc.n1, pc.n2)
        return out

    for tag, kwargs in (("fused", dict(fused=True)), ("nonfused", dict(fused=False)),
                        ("fused_homeo", dict(fused=True, homeo=True))):
        for kk, v in stack_seq(**kwargs).items():
            G[f"{tag}_{kk}"] = v

    rng2 = np.random.default_rng(31)
    Ws = [rng2.normal(size=(48, 40)) for _ in range(4)]
    pcs = SparsePCStack.from_dense(*Ws, eta_pc=0.05, eta_oja=0.01, w_max=2.0, k=12)
    for nm, csr in zip(("up0", "up1", "dn0", "dn1"),
                       (pcs.up0, pcs.dn1, pcs.dn0, pcs.dn1)):
        G[f"fd_{nm}_ix"] = csr[1]
        G[f"fd_{nm}_val"] = csr[2]
    G["fd_val_w"] = pcs.up0[2].copy()
    return G


def golden_hash(G: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(G):
        a = np.ascontiguousarray(G[key])
        h.update(key.encode())
        h.update(a.dtype.str.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _eq(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)


# ───────────────────────────────────────────────────────────── (A) 对拍
def section_a() -> None:
    print("【A】向量化 vs 朴素参考 逐位对拍（3 规模 × tie-heavy）")
    rng = np.random.default_rng(20261007)
    scales = [(128, 96), (512, 384), (1024, 1000)]
    n_pass = 0
    total = 0
    for rows, cols in scales:
        makers = [
            ("random", lambda r, c: rng.normal(size=(r, c)), min(64, cols)),
            ("all-zeros", lambda r, c: np.zeros((r, c)), 7),
            ("all-zeros k=1", lambda r, c: np.zeros((r, c)), 1),
            ("heavy-repeats", lambda r, c: np.round(rng.normal(size=(r, c)) * 3) / 3.0,
             max(1, cols // 2)),
            ("two-valued k=1", lambda r, c: (rng.integers(0, 2, size=(r, c)) * 2.0 - 1.0), 1),
            ("int-repeats k=n_cols",
             lambda r, c: rng.integers(-3, 4, size=(r, c)).astype(np.float64), cols),
            ("nan-row", lambda r, c: np.where(np.arange(c)[None, :] == 3, np.nan,
                                               rng.normal(size=(r, c))), 8),
        ]
        for name, make, k in makers:
            total += 1
            W = make(rows, cols)
            a = naive_from_dense_csr(W, k)
            b = spc._from_dense_csr(W, k)
            ok = csr_equal(a, b)
            d_ok = np.array_equal(naive_dense(*b, rows, cols),
                                  SparsePCStack._dense(*b, rows, cols))
            if ok and d_ok:
                n_pass += 1
            check(ok and d_ok, f"A {rows}x{cols} {name} k={k}",
                  "idx/indptr/val+dense 逐位" if ok and d_ok else
                  f"fdcsr={'OK' if ok else 'DIFF'} dense={'OK' if d_ok else 'DIFF'}")
    # 显式空行 CSR（indptr 含 0 长度行）
    indptr = np.array([0, 3, 3, 6], dtype=np.int64)
    idx = np.array([0, 2, 5, 1, 4, 3], dtype=np.int64)
    val = np.arange(6, dtype=np.float32)
    total += 1
    ok = np.array_equal(naive_dense(indptr, idx, val, 3, 6),
                         SparsePCStack._dense(indptr, idx, val, 3, 6))
    if ok:
        n_pass += 1
    check(ok, "A _dense 空行 CSR (3 行含 0 度行)", "逐位")
    print(f"  -- A 小结: {n_pass}/{total} tie-heavy 对拍逐位一致")


# ──────────────────────────────────────────────────────── (B) golden 对拍
def section_b() -> None:
    print("【B】固定 seed infer/learn 输出对拍 golden（%TEMP%）")
    tmp = Path(_tmp.gettempdir())
    npz_path = tmp / "perf_golden_.npz"
    sha_path = tmp / "perf_golden.sha256"
    if not (npz_path.exists() and sha_path.exists()):
        check(False, "B golden 文件存在",
              f"缺 {npz_path if not npz_path.exists() else sha_path}（优化前基准产物）")
        return
    z = np.load(npz_path)
    golden = {k: z[k] for k in z.files}
    stored = sha_path.read_text().split()[0]

    out1 = build_golden()                       # B1 向量化重建
    check(_eq(golden, out1), "B1 重建输出 vs %TEMP% golden 逐数组 np.array_equal",
          f"{len(golden)} 数组")
    h1 = golden_hash(out1)
    check(h1 == stored == golden_hash(golden),
          "B2 重建 sha256 == 存档 golden sha256", h1[:16] + "...")

    out2 = build_golden()                       # B2b 同进程第二遍
    check(_eq(out1, out2), "B2b 同进程重建两遍逐位一致（确定性）")

    # B3 朴素实现自比对：临时把两个函数换回朴素版，重建必须逐位一致
    orig_fd, orig_dense = spc._from_dense_csr, SparsePCStack._dense
    try:
        spc._from_dense_csr = naive_from_dense_csr
        SparsePCStack._dense = naive_dense
        out_naive = build_golden()
    finally:
        spc._from_dense_csr = orig_fd
        SparsePCStack._dense = orig_dense
    check(_eq(out1, out_naive),
          "B3 朴素版自比对：朴素重建 == 向量化重建（逐位）")
    check(spc._from_dense_csr is orig_fd and SparsePCStack._dense is orig_dense,
          "B3b 对拍后现场恢复（未污染模块状态）")
    h_naive = golden_hash(out_naive)
    check(h_naive == stored, "B4 朴素版 sha256 == 存档 golden sha256", h_naive[:16] + "...")


# ────────────────────────────────────────────────────────── (C) 性能断言
# 阈值 = 优化后实测（两轮实测较慢值）× 2（防抖动）。换机器请重新基准
# （%TEMP%/perf_bench.py）后更新本表——但只能放松，逐位断言（A/B）永不放松。
PERF_CASES = [
    # (name, rows, cols, k, 阈值 ms, 优化后实测 ms[两轮较慢值])
    ("_from_dense_csr", 1024, 1024, 64, 141.0, 70.5),
    ("_from_dense_csr", 2048, 2048, 128, 582.0, 290.9),
    ("_dense", 2048, 2048, 128, 40.0, 19.8),
    ("_dense", 4096, 4096, 256, 160.0, 79.6),
]


def section_c() -> None:
    print("【C】性能断言（预热 + best-of-7；阈值 = 优化后实测 × 2）")
    rng = np.random.default_rng(4242)
    for name, rows, cols, k, limit, measured in PERF_CASES:
        W = rng.normal(size=(rows, cols))
        csr = spc._from_dense_csr(W, k)
        if name == "_from_dense_csr":
            ms = best_of(lambda: spc._from_dense_csr(W, k))
        else:
            ms = best_of(lambda: SparsePCStack._dense(*csr, rows, cols))
        check(ms <= limit, f"C {name} {rows}x{cols} k={k} <= {limit:.0f} ms",
              f"实测 {ms:.1f} ms（优化后基准 {measured} ms，阈值 2×）")


def main() -> int:
    section_a()
    section_b()
    section_c()
    n_ok = sum(1 for ok, _n, _d in _RESULTS if ok)
    print(f"\n== verify_perf_no_regression: {n_ok}/{len(_RESULTS)} PASS ==")
    if n_ok != len(_RESULTS):
        for ok, name, detail in _RESULTS:
            if not ok:
                print(f"  FAILED: {name}  {detail}")
        return 1
    print("全部通过：逐位不变 + 无性能回归。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
