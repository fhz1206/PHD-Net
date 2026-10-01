"""P114诊断：读出（NPU）热路径的同步点与字节流量拆解。

为什么需要它
================================================================================
P113 后读出占端到端 **64.7%**（7.83 / 12.09 ms/tok，1b 档昇腾 + NPU）。
它是下一个优化目标，但**盲目改算子**的效率很低——先要回答两个问题：

  Q1读出里**还有没有设备同步点**？CPU 被设备等住的时间不计入 NPU 利用率，
     但计入端到端。P112 已修掉一个每步 `torch.equal` 硬同步（等于把 P34 的
     nll_sync_every 收益又还回去），**不能假定那是唯一的**。
  Q2 读出的时间花在哪？是**带宽饱和**（无望，只能换布局）还是
     **算子效率 / 同步**（有希望）？

⚠ **带宽下界不是实测**。它是按「每步必须触及的字节数÷ 假设带宽」算出来的
**估算值**。真实带宽要用 npu-smi 或等效带宽实测拿（`tools/bench_accel.py`）。
把下界当实测是本项目反复踩过的坑（见 BUGS A8 / B12）。

口径
================================================================================
·每步触达字节 = 前向（读 W + idx + 写 y）× 1 + 更新（读 W + idx + 读/写 g）
  其中 g = 稀疏 gather 出的 (n_out, k) fp32 中间张量。
· **临时张量是真实流量**，不是零成本：`W * g` 物化一个 (n_out,k) fp32。
  1b 档 n_out=51,962, k=128 → 26.6 MiB/处；前向+更新各一处。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MIB = 1024 * 1024


def analyse(n_out: int, n_h: int, k: int, dtype_bytes: int = 4,
            idx_bytes: int = 8) -> dict:
    """按字节口径拆解读出热路径。返回各项 MiB 与总触达。"""
    W = n_out * k * dtype_bytes              #稀疏 val (n_out,k)
    I = n_out * k * idx_bytes               # 列索引 (n_out,k)
    g = n_out * k * dtype_bytes             # gather 出的 h 分量（同尺寸，临时）
    y = n_out * dtype_bytes                 # 输出 (n_out,)
    dp = n_out * dtype_bytes                # softmax 输出

    items = {
        "前向 读 W（val）": W,
        "前向 读 idx": I,
        "前向 读 h 本身": n_h * dtype_bytes,
        "前向 写 g（临时）": g,
        "前向 写 y": y,
        "更新 读 W（val）": W,
        "更新 读 idx（若缓存未命中）": I,
        "更新 读 g（临时）": g,
        "更新 读/写 dp（softmax 往返）": 3 * dp,
        "更新 写 W（原地）": W,
    }
    total = sum(items.values())
    # 稠密对照
    dense = n_out * n_h * dtype_bytes
    return {"items": items, "total": total, "dense_total": 3 * dense,
            "W": W, "I": I, "g": g}


def probe_sync_points(verbose: bool = True) -> int:
    """扫读出热路径里**所有可能触发设备同步**的调用。

    方法：静态扫 `accel_readout.py` 的方法体，标出返回 Python 标量
    （`.item()` / `bool(...)` / `float(tensor)` / `torch.equal` / `.cpu()`）
    或 numpy 转换的表达式。这些是「CPU 等设备」的位置。
    """
    import ast
    import io

    src_path = _ROOT / "phdnet" / "backends" / "accel_readout.py"
    src = io.open(src_path, encoding="utf-8").read()
    tree = ast.parse(src)

    SYNC = (".item", ".cpu", "numpy", "tolist", "equal")
    hits: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            seg = ""
            if isinstance(sub, ast.Call):
                f = sub.func
                if isinstance(f, ast.Attribute):
                    seg = f.attr
                elif isinstance(f, ast.Name):
                    seg = f.id
            if seg in SYNC:
                # `numpy` 只在显式 `.cpu().numpy()` 或 `np.` 前缀时算
                if seg == "numpy":
                    txt = ast.get_source_segment(src, sub) or ""
                    if ".cpu()" not in txt and "device=" not in txt:
                        continue
                hits.append((node.name, seg, sub.lineno))

    if verbose:
        print("=" * 78)
        print("[Q1] 读出后端里的同步候选点（静态扫描）")
        print("=" * 78)
        if not hits:
            print("  未发现同步候选点")
        else:
            by_fn: dict[str, list[tuple[str, int]]] = {}
            for fn, seg, ln in hits:
                by_fn.setdefault(fn, []).append((seg, ln))
            for fn, lst in sorted(by_fn.items()):
                segs = ", ".join(f"{s}@{ln}" for s, ln in lst)
                print(f"  {fn:<24} {segs}")
        print("\n  ⚠ 静态扫描**不等于运行时同步**：")
        print("    · 构造期/探测期的 .item() 不在热路径；")
        print("    · 检查点路径（_csr / W_cpu）的 .cpu() 只在存盘时调用。")
        print("    真正的判据是服务器日志的「读出 X ms/tok」与 CS/s。")
    return len(hits)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="P114 读出（NPU）热路径诊断：同步点 + 字节流量拆解")
    ap.add_argument("--n-out", type=int, default=51962,
                    help="词表大小（读出行数）。1b 档实测= 51,962")
    ap.add_argument("--n-h", type=int, default=3072,
                    help="读出列数= n_h。1b 档 = 3072（pred_in_readout 三拼）")
    ap.add_argument("--k", type=int, default=128,
                    help="稀疏入边数（生产默认 128）")
    ap.add_argument("--bw", type=float, default=800.0,
                    help="假设 HBM 带宽 GB/s（**估算用**，非实测；"
                         "910B 保守值；实测请用 tools/bench_accel.py）")
    ap.add_argument("--ms", type=float, default=7.83,
                    help="服务器实测读出 ms/tok（用于对比下界）")
    args = ap.parse_args()

    n_sync = probe_sync_points()

    print("\n" + "=" * 78)
    print("[Q2] 字节流量拆解（每步，1b 档）")
    print("=" * 78)
    a = analyse(args.n_out, args.n_h, args.k)
    print(f"配置：n_out={args.n_out:,}  n_h={args.n_h:,}  k={args.k}（连接率 "
          f"{args.k/args.n_h*100:.2f}%）")
    print(f"\n{'项':<34}{'MiB/步':>10}")
    print("-" * 44)
    for name, by in a["items"].items():
        print(f"{name:<34}{by/MIB:>10.2f}")
    print("-" * 44)
    print(f"{'合计触达':<34}{a['total']/MIB:>10.2f}")
    print(f"{'稠密对照（fp32, 3× 权重）':<34}{a['dense_total']/MIB:>10.2f}")
    print(f"\n稀疏/稠密 = {a['total']/a['dense_total']:.2f}×")

    tlb = a["total"] / (args.bw * 1e9) * 1e3
    print(f"\n带宽下界（**估算**，假设 {args.bw:.0f} GB/s）= {tlb:.3f} ms")
    if args.ms > 0:
        print(f"服务器实测 = {args.ms:.2f} ms→ 实测/ 下界 = "
              f"{args.ms/tlb:.1f}×")
        if args.ms / tlb > 3:
            print("\n  → **不是带宽饱和**（实测远高于下界）→ 时间花在"
                  "算子效率或同步上，有优化空间。")
            print("     优先查：① (W*g) 临时张量的物化开销；"
                  "② 能否改 CSR SpMM；③ 还有没有同步点。")
        else:
            print("\n  → 接近带宽饱和 → 只能换布局/降流量，"
                  "改算子收益有限。")
    return 0


if __name__ == "__main__":
    sys.exit(main())