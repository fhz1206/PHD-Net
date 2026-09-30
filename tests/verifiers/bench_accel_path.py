"""P28：加速读出热路径的等价性 + 带宽/耗时对照（1B 规模）。

在 CPU 上跑等价的两条实现（torch 设备 = cpu），量化：
  · 数值等价：AXPY（addmm_）vs 物化 outer 的结果差异（容差判据）；
  · 设备流量：按张量读写字节数统计（这是 NPU 上的主要瓶颈，故用字节而非时间）；
  · 墙钟耗时（仅参考：CPU 内存带宽与 NPU HBM 不等价，比例不可直接外推）。

用法：python tests/verifiers/bench_accel_path.py [--V 73958] [--H 3072] [--steps 20]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "train"))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

from phdnet.backends.accel_readout import AccelReadout      # noqa: E402


def legacy_update(W, p, t, ht, eta):
    """P28 之前的实现：物化 outer 再 add_（保留作对照）。"""
    dp = (p - t).to(W.dtype)
    W.add_(torch.outer(dp, ht), alpha=-float(eta))


def axpy_update(W, p, t, ht, eta):
    """P28 之后：rank-1 AXPY（不物化临时张量）。"""
    dp = (p - t).to(W.dtype)
    W.addmm_(dp.reshape(-1, 1), ht.reshape(1, -1), alpha=-float(eta))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=73_958)
    ap.add_argument("--H", type=int, default=3_072)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--w-mib", type=int, default=0,
                    help="单块 W 实际 MiB（默认由 V×H×4 推得）")
    args = ap.parse_args()

    V, H = args.V, args.H
    w_mib = args.w_mib or (V * H * 4 / 2 ** 20)
    rng = np.random.default_rng(0)
    W0 = (rng.normal(0, 0.05, (V, H)) * np.sqrt(H)).astype(np.float32)
    h = rng.normal(0, 1, H).astype(np.float32)
    tgt = np.zeros(V, dtype=np.float32)
    tgt[rng.integers(0, V)] = 1.0

    print(f"规模 V={V:,} H={H:,} → W = {w_mib:.0f} MiB (fp32)\n")

    # ── 1. 数值等价 ──
    print("[1] 数值等价：outer 物化 vs addmm_ AXPY")
    ro1 = AccelReadout(H, V, None, device="cpu", w0=W0)
    ro2 = AccelReadout(H, V, None, device="cpu", w0=W0)
    with torch.no_grad():
        p1 = torch.softmax(ro1.W @ torch.from_numpy(h), dim=0)
        p2 = torch.softmax(ro2.W @ torch.from_numpy(h), dim=0)
        legacy_update(ro1.W, p1, torch.from_numpy(tgt), torch.from_numpy(h), 0.05)
        axpy_update(ro2.W, p2, torch.from_numpy(tgt), torch.from_numpy(h), 0.05)
    d = float((ro1.W - ro2.W).abs().max())
    scale = float(ro1.W.abs().max())
    print(f"    max|ΔW| = {d:.3e}（W 尺度 {scale:.3e}，相对 {d / max(scale, 1e-12):.2e}）")
    print(f"    → {'容差一致' if d <= 1e-5 * scale else '**超出容差，需排查**'}\n")

    # ── 2. 设备流量 ──
    print("[2] 设备侧字节流量（每步，NPU 上真正的瓶颈）")
    h_bytes = H * 4
    y_bytes = V * 4
    t_bytes = V * 4
    legacy = {
        "W@h 读 W": V * H * 4,
        "outer 写临时张量": V * H * 4,
        "add_ 读 W + 读临时 + 写 W": 3 * V * H * 4,
        "H2D h×2 + target + y 回传": 2 * h_bytes + t_bytes + y_bytes,
    }
    p28 = {
        "W@h 读 W": V * H * 4,
        "addmm_ 读 W + 写 W": 2 * V * H * 4,
        "H2D h + target": h_bytes + t_bytes,
    }
    ls, ps = sum(legacy.values()), sum(p28.values())
    for k in legacy:
        print(f"    [旧] {k:<34s} {legacy[k] / 2**20:9.1f} MiB")
    print(f"    旧合计: {ls / 2**20:9.1f} MiB/步")
    print()
    for k in p28:
        print(f"    [新] {k:<34s} {p28[k] / 2**20:9.1f} MiB")
    print(f"    新合计: {ps / 2**20:9.1f} MiB/步")
    print(f"    → 流量 {(1 - ps / ls) * 100:.1f}% 降低；理论下限（读 W + 读写 W）"
          f" = {3 * V * H * 4 / 2**20:.0f} MiB/步\n")

    # ── 3. 墙钟（仅参考）──
    print("[3] 墙钟耗时（CPU 参考，比例不直接外推到 NPU）")
    for name, fn in (("outer 物化", legacy_update), ("addmm_ AXPY", axpy_update)):
        ro = AccelReadout(H, V, None, device="cpu", w0=W0)
        ht = torch.from_numpy(h)
        tt = torch.from_numpy(tgt)
        with torch.no_grad():
            for _ in range(3):
                p = torch.softmax(ro.W @ ht, dim=0)
                fn(ro.W, p, tt, ht, 0.01)
            if os.name == "nt":
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            for _ in range(args.steps):
                p = torch.softmax(ro.W @ ht, dim=0)
                fn(ro.W, p, tt, ht, 0.01)
            dt = (time.perf_counter() - t0) / args.steps
        print(f"    {name:<12s} {dt * 1000:8.2f} ms/步 "
              f"（含 softmax，不含 W@h）")
    print()
    print("结论：流量口径才是 NPU 上的真约束（GEMV 受带宽限制）；"
          "addmm_ 把 4 次全量 W 触碰降到 3 次。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
