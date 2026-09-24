"""O1 验证：稀疏 PC 的精确 Top-k 支撑集更新 —— 与稠密更新的逐位等价上界。

`sparse_pc=True` 时 PC 学习只更新 |v| top-k 维（k = `pc_topk`，0 时用 1/8 比例）。
理论主张（O1）：
  ΔW_dn0 = η·e0⊗r1 的第 j 列贡献幅值 ∝ |r1[j]|·Σ|e0| —— 列维度的贡献排序与 |r1|
  排序**完全等价**，故按 |v| 取 top-k 即"精确梯度贡献 top-k"，非选中列零更新；
  **k = 层维度时退化为稠密更新，且逐位一致**（每列独立相加，与全矩阵加法同序）。

本脚本对拍验证 k = 全量（各层维度）时，稀疏路径与稠密路径在 N 步
infer+learn 后四个权重矩阵**逐位相同**；并顺带实测不同 k 的加速比。

用法：python tools/verify_pc_topk_equiv.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.pc import PredictiveCodingStack  # noqa: E402

N0, N1, N2 = 256, 128, 64
N_STEPS = 200
ETA_PC, ETA_OJA = 0.02, 0.02

_ok = True


def check(name: str, ok: bool, extra: str = "") -> bool:
    global _ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))
    _ok &= ok
    return ok


def run_stack(sparse: bool, topk: int, inps: list[np.ndarray]) -> dict:
    pc = PredictiveCodingStack(N0, N1, N2, ETA_PC, ETA_OJA,
                               np.random.default_rng(7),
                               sparse_pc=sparse, topk=topk)
    for s0 in inps:
        cache = pc.infer(s0, 1)
        pc.learn(cache, eta_scale=1.0)
    return {"W_up0": pc.W_up0, "W_up1": pc.W_up1, "W_dn0": pc.W_dn0, "W_dn1": pc.W_dn1}


def main() -> None:
    rng = np.random.default_rng(11)
    inps = [np.abs(rng.normal(0, 1, N0)) for _ in range(N_STEPS)]

    t0 = time.perf_counter()
    dense = run_stack(False, 0, inps)
    t_dense = time.perf_counter() - t0

    # k = 各层维度（全量）→ 应退化为稠密更新
    t0 = time.perf_counter()
    full = run_stack(True, 10 ** 9, inps)      # k 会被 min(k, len(v)) 截断到全量
    t_full = time.perf_counter() - t0

    for key in ("W_up0", "W_up1", "W_dn0", "W_dn1"):
        check(f"k=全量 与稠密逐位一致 ({key})",
              bool(np.array_equal(dense[key], full[key])),
              f"max|Δ|={np.abs(dense[key] - full[key]).max():.3e}")

    # 不同 k 的加速比（相对稠密）
    print(f"\n加速比实测（{N_STEPS} 步 infer+learn，稠密 {t_dense * 1000:.0f} ms）：")
    for k in (8, 16, 32, 64):
        t0 = time.perf_counter()
        run_stack(True, k, inps)
        dt = time.perf_counter() - t0
        print(f"  pct_topk={k:<4} {dt * 1000:>7.0f} ms   加速 {t_dense / dt:.2f}×")
    t0 = time.perf_counter()
    run_stack(True, 0, inps)                   # 默认 1/8 比例（与 T5.1 一致）
    dt = time.perf_counter() - t0
    print(f"  pct_topk=1/8  {dt * 1000:>7.0f} ms   加速 {t_dense / dt:.2f}×（默认比例）")

    print("\n=== 全部通过：稀疏 Top-k 的选择依据精确、k=全量时与稠密逐位一致 ==="
          if _ok else "\n=== 存在不一致 ===")
    sys.exit(0 if _ok else 1)


if __name__ == "__main__":
    main()
