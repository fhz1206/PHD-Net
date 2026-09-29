"""硬件后端适配基准（P10，2026-09-26 · CUDA / CANN(NPU) / ROCm / CPU）。

探测全部可用加速器（torch.cuda / torch_npu / HIP），并在每个可用设备 × 每个支持
精度（fp32/fp16/bf16）上跑读出热路径基准（前向 matvec + softmax 梯度更新，
1B 预设真实读出规模 V=9219 × H=3072）；无加速器时自动回退 CPU（参考路径）。

用法：python tools/bench_accel.py [--steps 50] [--V 9219] [--H 3072]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from phdnet.backends.torch_backend import (bench_readout,          # noqa: E402
                                           probe_devices)


def main() -> None:
    ap = argparse.ArgumentParser(description="读出热路径加速器基准（P10 适配）")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--V", type=int, default=73958,
                    help="读出行数（缺省 73958 = 1B 预设真实词表规模）")
    ap.add_argument("--H", type=int, default=3072)
    args = ap.parse_args()

    probes = probe_devices()
    print("=" * 78)
    print("设备探针（CUDA / ROCm / CANN-NPU / CPU）")
    print("=" * 78)
    for plat, info in probes.items():
        status = "✓ 可用" if info.get("ok") else "✗ 不可用"
        extra = ""
        if info.get("ok") and plat != "cpu":
            extra = f" | {info.get('name')} × {info.get('count')} | " \
                    f"后端版本 {info.get('version')}"
        elif plat == "npu" and not info.get("ok"):
            extra = f" | {info.get('note', '')}"
        print(f"  {plat:6s} {status}{extra}")

    print("=" * 78)
    print(f"读出热路径基准（V={args.V} × H={args.H}，{args.steps} 步均值）")
    print("=" * 78)
    results = []
    # 设备列表：按平台映射到 torch device 字符串
    dev_list = ["cpu"]
    if probes.get("rocm", {}).get("ok") or probes.get("cuda", {}).get("ok"):
        dev_list.insert(0, "cuda")
    if probes.get("npu", {}).get("ok"):
        dev_list.insert(0, "npu")
    for dev in dev_list:
        for dt in ("fp32", "fp16", "bf16"):
            # P29：dtype 此前**从未传入** bench_readout，且成功后立即 break
            # → fp16/bf16 档从来没被真测过。三档都跑，逐档诚实降级。
            try:
                r = bench_readout(device=dev, V=args.V, H=args.H,
                                  steps=args.steps, dtype=dt)
                results.append(r)
                print(f"  {dev:5s} {r['dtype']:5s}: 前向 {r['fwd_ms']:8.3f} ms | "
                      f"更新 {r['update_ms']:8.3f} ms | 合计 {r['total_ms']:8.3f} "
                      f"ms/token | W {r['W_MiB']:6.0f} MiB | "
                      f"等效 {r['eff_GBps']:6.1f} GB/s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  {dev:5s} {dt:5s}: 不可用（{type(e).__name__}: "
                      f"{str(e)[:60]}）", flush=True)
    print("-" * 78)
    print("说明：")
    print("  · 等效带宽 = 3×|W| / 耗时（读 W 一次 + 读写 W 各一次）。读出是 GEMV，")
    print("    **受带宽限制而非算力**，所以带宽是唯一可跨平台比较的指标。")
    print("  · dtype 即设备张量的存储+计算精度（softmax/NLL 主回路恒 fp32，P9 协议）。")
    print("  · fp8/fp4 在加速后端未实现（会回落 numba CPU），故本基准不测。")
    print("  · ⚠ 半精度的机制性代价：非目标行的更新量 ≈1e-6，比 fp16 半 ULP 还小")
    print("    15 倍 → 被舍入丢弃，学习规则退化为「只提升目标行」。见硬件后端报告。")
    with open(os.path.join(_ROOT, "outputs", "bench_accel.json"), "w",
              encoding="utf-8") as f:
        json.dump({"probes": probes, "results": results}, f,
                  ensure_ascii=False, indent=2, default=str)
    print(f"结果已写入 outputs/experiments/bench_accel.json")


if __name__ == "__main__":
    main()
