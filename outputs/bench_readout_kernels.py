#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P178 收口：gather vs scalar4 vs scalar1 vs numba（同进程交错）。

由 `PHDNET_CSR_KERNEL` 选择 Rust 内层；本脚本对每种模式跑一次，
外部用环境变量切换。**默认 gather 路径未改**。
"""
from __future__ import annotations
import sys, os, time
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numba  # noqa: E402

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "..", "phdnet_rs"))
from phdnet_rs import load  # noqa: E402

K, why = load()
if K is None:
    print("[FAIL] %s" % why); raise SystemExit(1)

MODE = os.environ.get("PHDNET_CSR_KERNEL", "gather")

N, COLS, KK = 73958, 3072, 128
rng = np.random.default_rng(3)
indptr = np.arange(0, N * KK + 1, KK, dtype=np.int64)
idx32 = rng.integers(0, COLS, N * KK).astype(np.int32)
val = rng.normal(0, 0.05, N * KK).astype(np.float32)
x = rng.normal(0, 1, COLS).astype(np.float32)
out_r = np.zeros(N, np.float32)
out_n = np.zeros(N, np.float32)


@numba.njit(parallel=True, fastmath=True, cache=True)
def spmv_nb(indptr, idx, val, x, out):
    for i in numba.prange(indptr.shape[0] - 1):
        s = np.float32(0.0)
        for p in range(indptr[i], indptr[i + 1]):
            s = np.float32(s + val[p] * x[idx[p]])
        out[i] = s


def t_rust(nt):
    K.m2_matvec(indptr, idx32, val, x, out_r, nt)

def t_numba(nt):
    numba.set_num_threads(nt)
    spmv_nb(indptr, idx32, val, x, out_n)


def once(fn, iters=9):
    for _ in range(3):
        fn()
    best = 1e18
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best * 1000.0


t_rust(8); t_numba(8); t_rust(1); t_numba(1)

best = {(k, nt): 1e18 for k in ("rust", "numba") for nt in (1, 8)}
for r in range(4):
    for nt in (1, 8):
        best[("rust", nt)] = min(best[("rust", nt)], once(lambda: t_rust(nt)))
        best[("numba", nt)] = min(best[("numba", nt)], once(lambda: t_numba(nt)))

r1, r8 = best[("rust", 1)], best[("rust", 8)]
n1, n8 = best[("numba", 1)], best[("numba", 8)]
byt = N * KK * 4 * 2 + COLS * 4
print("MODE=%-8s | Rust %6.2f/%6.2f ms (1T/8T, %.2fx) | numba %6.2f/%6.2f ms (%.2fx) | 8T vs numba %s"
      % (MODE, r1, r8, r1 / r8, n1, n8, n1 / n8,
         ("Rust %.2fx 更快" % (n8 / r8)) if r8 < n8 else ("numba %.2fx 更快" % (r8 / n8))))

# 正确性（对该模式自己的 numba 参考）
t_rust(8); t_numba(8)
rel = float(np.abs(out_r.astype(np.float64) - out_n.astype(np.float64)).max()
            / max(1e-30, np.abs(out_n.astype(np.float64)).max()))
print("           relerr=%.3e (%s) | Rust8T=%.1f GB/s" % (rel, "PASS" if rel < 1e-4 else "FAIL", byt / 1e9 / (r8 / 1000)))
