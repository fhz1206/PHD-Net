"""O1-2 验证：主干**结构性稀疏连接**（SparsePCStack）的正确性与规模收益。

对拍设计（保证可比性）：用稠密栈的 4 个权重构造稀疏栈（`from_dense`，k = 全列），
两栈权重**完全相同** → 相同输入序列下 infer/learn 结果应在浮点容差内一致
（CSR 按列索引升序求和 vs BLAS 内部求和顺序不可控，故用容差而非逐位）。

同时报告结构性稀疏的规模收益（存在突触数 vs 稠密等价元素数、连接率）。

用法：python tools/verify_sparse_conn.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.pc import PredictiveCodingStack      # noqa: E402
from phdnet.sparse_pc import SparsePCStack        # noqa: E402

N0, N1, N2 = 256, 128, 64
N_STEPS = 150
ETA_PC, ETA_OJA = 0.02, 0.02
TOL = 1e-10

_ok = True


def check(name: str, ok: bool, extra: str = "") -> bool:
    global _ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))
    _ok &= ok
    return ok


def main() -> None:
    rng = np.random.default_rng(7)
    dense = PredictiveCodingStack(N0, N1, N2, ETA_PC, ETA_OJA, rng)
    # k = 全列 → 两栈权重完全相同
    full = SparsePCStack.from_dense(dense.W_up0, dense.W_up1, dense.W_dn0,
                                    dense.W_dn1, ETA_PC, ETA_OJA)
    print(f"稠密栈参数量: {dense.W_up0.size + dense.W_up1.size + dense.W_dn0.size + dense.W_dn1.size:,}")
    print(f"稀疏栈(全连接图) 存在突触: {full.n_synapses():,}"
          f"  连接率 {full.stats()['connectivity'] * 100:.1f}%")

    # 权重完全一致检查
    for name, (Wd, sp) in {
        "up0": (dense.W_up0, full.up0), "up1": (dense.W_up1, full.up1),
        "dn0": (dense.W_dn0, full.dn0), "dn1": (dense.W_dn1, full.dn1)}.items():
        Ws = SparsePCStack._dense(*sp, *Wd.shape)
        check(f"初始权重一致 ({name})", np.allclose(Wd, Ws, atol=0, rtol=0),
              f"max|Δ|={np.abs(Wd - Ws).max():.3e}")

    # 相同输入序列：infer + learn
    rng2 = np.random.default_rng(11)
    inps = [np.abs(rng2.normal(0, 1, N0)) for _ in range(N_STEPS)]
    max_r1 = max_r2 = 0.0
    for s0 in inps:
        cd = dense.infer(s0, 1)
        cs = full.infer(s0, 1)
        max_r1 = max(max_r1, float(np.abs(cd["r1"] - cs["r1"]).max()))
        max_r2 = max(max_r2, float(np.abs(cd["r2"] - cs["r2"]).max()))
        dense.learn(cd, eta_scale=1.0)
        full.learn(cs, eta_scale=1.0)

    check(f"{N_STEPS} 步前向 r1/r2 一致（容差 {TOL:g}）",
          max_r1 < TOL and max_r2 < TOL, f"max|Δr1|={max_r1:.3e} max|Δr2|={max_r2:.3e}")
    for name, (Wd, sp) in {
        "up0": (dense.W_up0, full.up0), "up1": (dense.W_up1, full.up1),
        "dn0": (dense.W_dn0, full.dn0), "dn1": (dense.W_dn1, full.dn1)}.items():
        Ws = SparsePCStack._dense(*sp, *Wd.shape)
        d = float(np.abs(Wd - Ws).max())
        check(f"{N_STEPS} 步学习后权重一致 ({name})", d < TOL, f"max|Δ|={d:.3e}")

    # 结构性稀疏的规模收益
    print("\n结构性稀疏规模收益（每神经元入边数 k）：")
    dense_total = (N1 * N0 + N2 * N1 + N0 * N1 + N1 * N2)
    print(f"  稠密等价元素数: {dense_total:,}")
    for k in (8, 16, 32):
        sp = SparsePCStack(N0, N1, N2, ETA_PC, ETA_OJA, np.random.default_rng(7), conn_k=k)
        st = sp.stats()
        print(f"  conn_k={k:<3} 存在突触 {st['synapses']:>7,}  "
              f"连接率 {st['connectivity'] * 100:>5.1f}%  存储压缩 {dense_total / st['synapses']:.1f}×")

    # 速度对比（同规模）
    print("\n速度对比（150 步 infer+learn）：")
    t0 = time.perf_counter()
    rng3 = np.random.default_rng(7)
    d2 = PredictiveCodingStack(N0, N1, N2, ETA_PC, ETA_OJA, rng3)
    for s0 in inps:
        d2.learn(d2.infer(s0, 1), eta_scale=1.0)
    t_dense = time.perf_counter() - t0
    print(f"  稠密栈           {t_dense * 1000:>7.0f} ms")
    for k in (8, 16, 32, 0):
        t0 = time.perf_counter()
        s2 = SparsePCStack(N0, N1, N2, ETA_PC, ETA_OJA, np.random.default_rng(7), conn_k=k)
        for s0 in inps:
            s2.learn(s2.infer(s0, 1), eta_scale=1.0)
        dt = time.perf_counter() - t0
        tag = "自动(1/8)" if k == 0 else str(k)
        print(f"  稀疏 conn_k={tag:<7} {dt * 1000:>7.0f} ms   相对稠密 {t_dense / dt:.2f}×")

    print("\n=== 通过：结构性稀疏栈与稠密栈数值一致；稀疏图压缩见上 ==="
          if _ok else "\n=== 存在超容差差异 ===")
    sys.exit(0 if _ok else 1)


if __name__ == "__main__":
    main()
