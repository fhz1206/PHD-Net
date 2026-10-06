#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M1/M2 基线：thread::scope(每次建/销线程) vs 常驻池 的真实差距。

M1 生产形状：读出门 vocab(73958) × n_h(3072) 稀疏 CSR k=128（M2 路径已用池）
           + 稠密 4096x4096 GEMV（M1 路径，mechanisms.rs 仍用 thread::scope）
"""
from __future__ import annotations
import sys, os, time
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "8")
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "..", "phdnet_rs"))
from phdnet_rs import load

K, why = load()
if K is None:
    print("[FAIL] %s" % why); raise SystemExit(1)

def once(fn, it=15, warm=5):
    for _ in range(warm): fn()
    b = 1e18
    for _ in range(it):
        t0 = time.perf_counter(); fn(); b = min(b, time.perf_counter()-t0)
    return b*1000

print("=== M1 稠密 GEMV (m1_gemv, mechanisms.rs 用 thread::scope) ===")
for (R, C) in [(4096, 4096), (73958, 512)]:
    W = np.ascontiguousarray(np.random.default_rng(0).normal(0, 0.01, (R, C)).astype(np.float32))
    x = np.random.default_rng(1).normal(0, 1, C).astype(np.float32)
    b = np.zeros(R, np.float32); o = np.zeros(R, np.float32)
    t1 = once(lambda: K.m1_gemv(W, x, b, o, 1))
    t8 = once(lambda: K.m1_gemv(W, x, b, o, 8))
    print("  %5dx%-5d (%.1f MB): 1T=%7.2fms  8T=%7.2fms  缩放=%.2fx" % (
        R, C, R*C*4/1e6, t1, t8, t1/t8))

print("=== M2 稀疏 CSR SpMV (m2_matvec, 已用常驻池) ===")
N, COLS, KK = 73958, 3072, 128
rng = np.random.default_rng(3)
indptr = np.arange(0, N*KK+1, KK, dtype=np.int64)
idx32 = rng.integers(0, COLS, N*KK).astype(np.int32)
val = rng.normal(0, 0.05, N*KK).astype(np.float32)
xs = rng.normal(0, 1, COLS).astype(np.float32)
out = np.zeros(N, np.float32)
t1 = once(lambda: K.m2_matvec(indptr, idx32, val, xs, out, 1))
t8 = once(lambda: K.m2_matvec(indptr, idx32, val, xs, out, 8))
print("  73958x3072 k=128: 1T=%7.2fms  8T=%7.2fms  缩放=%.2fx" % (t1, t8, t1/t8))
