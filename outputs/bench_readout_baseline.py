#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稀疏读出门 SpMV（生产真实形状）Rust(i32 原生) vs numba 对拍。

生产读出门 = vocab(73958) × n_h(3072) 的稀疏 CSR，连接率 k=128。
这是训练 net.step 的 73% 瓶颈（记忆：226.6 MB/步）。

P178 结论：idx 用 **i32 原生**（列下标 ≤ 65535，int32 无损）→ Rust gather 核
达 ≈38 GB/s（DDR4 峰值 ~88%），读出门 13.4ms(i64) → ~3ms(i32)。本脚本验证
原生 i32 落地后的端到端速度，并与 numba 对照。
"""
from __future__ import annotations
import os, sys, time
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numba  # noqa: E402

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_ROOT, "..", "phdnet_rs")))
from phdnet_rs import load  # noqa: E402

K, why = load()
if K is None:
    print("[FAIL] Rust 库加载失败：%s" % why); raise SystemExit(1)

N, COLS, KK = 73958, 3072, 128
rng = np.random.default_rng(3)
indptr = np.arange(0, N * KK + 1, KK, dtype=np.int64)
# 原生 int32 idx（生产路径 P178：列下标 ≤ 65535，int32 无损）
idx32 = rng.integers(0, COLS, N * KK).astype(np.int32)
val = rng.normal(0, 0.05, N * KK).astype(np.float32)
x = rng.normal(0, 1, COLS).astype(np.float32)
out = np.zeros(N, np.float32)
out_nb = np.zeros(N, np.float32)
out_nb32 = np.zeros(N, np.float32)


@numba.njit(parallel=True, fastmath=True, cache=True)
def spmv_nb(indptr, idx, val, x, out):
    # ⚠ fp32 累加，与 Rust（真 fp32）公平对比
    for i in numba.prange(indptr.shape[0] - 1):
        s = np.float32(0.0)
        for p in range(indptr[i], indptr[i + 1]):
            s = np.float32(s + val[p] * x[idx[p]])
        out[i] = s


def rust_spmv():
    K.m2_matvec(indptr, idx32, val, x, out, 8)


def nb_spmv():
    spmv_nb(indptr, idx32, val, x, out_nb)


def nb_i64_spmv():
    spmv_nb(indptr, idx32.astype(np.int64), val, x, out_nb32)


def best_of(fn, iters=7, warm=3, inner=3):
    for _ in range(warm):
        fn()
    best = 1e18
    for _ in range(iters):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        best = min(best, (time.perf_counter() - t0) / inner)
    return best * 1000.0


t_r = best_of(rust_spmv)
t_n = best_of(nb_spmv)
t_n64 = best_of(nb_i64_spmv)
print("=== 稀疏读出门 SpMV %d×%d k=%d (nnz=%.2fM) ===" % (N, COLS, KK, N * KK / 1e6))
print("  Rust(i32 原生, 池8):  %.3f ms" % t_r)
print("  numba(i32, 8):        %.3f ms" % t_n)
print("  numba(i64, 8):        %.3f ms" % t_n64)
print("  → Rust(i32) 相对 numba(i32) = %.2fx （%s）"
      % ((t_n / t_r), "Rust 更快" if t_r < t_n else "numba 更快"))
print("  → i64 拖慢 numba %.2fx（i32 才是原生格式）" % (t_n64 / t_n))

# 正确性（容差 1e-4：SIMD 改求和顺序）
rust_spmv()
nb_spmv()
rel = float(np.abs(out.astype(np.float64) - out_nb.astype(np.float64)).max()
           / max(1e-30, np.abs(out_nb.astype(np.float64)).max()))
print("  [正确] Rust(i32) vs numba(i32) 最坏 relerr=%.3e (容差1e-4: %s)"
      % (rel, "PASS" if rel < 1e-4 else "FAIL"))

# 带宽估算（i32 原生：val(4B) + idx(4B) + x(4B)）
bytes_i32 = (N * KK * 4) + (N * KK * 4) + (COLS * 4)
print("  近似字节流量(i32): %.1f MB → Rust %.1f GB/s / numba %.1f GB/s (DDR4 上限~43)"
      % (bytes_i32 / 1e6, bytes_i32 / 1e9 / (t_r / 1000), bytes_i32 / 1e9 / (t_n / 1000)))
