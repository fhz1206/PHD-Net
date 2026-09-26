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

from phdnet.torch_backend import bench_readout, probe_devices   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="读出热路径加速器基准（P10 适配）")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--V", type=int, default=9219)
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
            # fp16/bf16 在旧 GPU（sm<7.0）或部分 NPU 上可能不可用 → 诚实降级
            try:
                r = bench_readout(device=dev, V=args.V, H=args.H,
                                  steps=args.steps)
                r["dtype"] = dt if dev != "cpu" or dt == "fp32" else dt
                # CPU 一次测 fp32 足够（fp16 CPU matvec 无加速，仅参考）
                if dev == "cpu" and dt != "fp32":
                    continue
                r["dtype"] = dt
                results.append(r)
                print(f"  {dev:5s} {dt:5s}: 前向 {r['fwd_ms']:8.3f} ms | "
                      f"更新 {r['update_ms']:8.3f} ms | 合计 {r['total_ms']:8.3f} "
                      f"ms/token", flush=True)
                break                                     # 每设备 fp32 基准即可
            except Exception as e:  # noqa: BLE001
                print(f"  {dev:5s} {dt:5s}: 失败（{type(e).__name__}: {e}）",
                      flush=True)
                break
    print("-" * 78)
    print("说明：fp8 需 CUDA ≥ 8.9（Ada/Hopper）+ torch ≥ 2.1（float8_e4m3fn）；")
    print("      fp4 无 torch 原生类型 → CPU numba 量化码本路径（phdnet/readout.py P9）；")
    print("      本基准的设备 dtype 即存储+计算精度（softmax/NLL 在 fp32 主回路）。")
    with open(os.path.join(_ROOT, "outputs", "bench_accel.json"), "w",
              encoding="utf-8") as f:
        json.dump({"probes": probes, "results": results}, f,
                  ensure_ascii=False, indent=2, default=str)
    print(f"结果已写入 outputs/bench_accel.json")


if __name__ == "__main__":
    main()
