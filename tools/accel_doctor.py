# -*- coding: utf-8 -*-
"""加速器诊断（P20）：一次性回答「NPU/CUDA 到底能不能用、读出快多少」。

fhz 反馈服务器日志 `[能力] 检测到加速器 ['npu']` 但生产训练仍走 numba/CPU。
本脚本把「探测 → 解析 → 试分配 → 跑真实读出负载 → 计时」串起来，输出：

  1. 环境矩阵：torch / torch_npu / CANN 版本与配对情况；
  2. 设备解析：`resolve_device('auto')` 的结果（昇腾 → ROCm → CUDA → DML → CPU）；
  3. 张量试分配：在目标设备上分配一块读出规模（默认 73,958×3,072 fp32 ≈ 908 MB）
     的权重并做一次前向 matvec，**验证设备真的能算**（而非只是被探测到）；
  4. 性能对照：同一负载在 numba CPU 与设备上的 ms/token；
  5. 结论：能不能用、瓶颈在哪（显存/带宽/算子）、生产入口为何没走它。

用法：
    python tools/accel_doctor.py                      # 默认 1B 读出规模
    python tools/accel_doctor.py --V 20000 --H 3072   # 更小规模（显存不足时）
    python tools/accel_doctor.py --device npu         # 强制指定设备（不 auto）
    python tools/accel_doctor.py --no-bench           # 只诊断不跑基准
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))


def _sec(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto",
                    help="auto / npu / cuda / rocm / cpu")
    ap.add_argument("--V", type=int, default=73_958, help="词表大小（读出行数）")
    ap.add_argument("--H", type=int, default=3_072, help="输入维度（读出列数）")
    ap.add_argument("--steps", type=int, default=20, help="基准步数")
    ap.add_argument("--no-bench", action="store_true", help="只诊断不跑基准")
    args = ap.parse_args()

    # ── 1. 环境矩阵 ──
    _sec("1. 环境矩阵")
    try:
        import torch
        print(f"  torch        : {torch.__version__}")
        print(f"  torch.version.cuda : {torch.version.cuda}")
        print(f"  torch.version.hip  : {torch.version.hip}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  torch 不可用：{type(e).__name__}: {e}")
        return 2
    try:
        import torch_npu                                  # noqa: F401
        print(f"  torch_npu    : {getattr(torch_npu, '__version__', '?')}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  torch_npu    : 未安装 / 导入失败（{type(e).__name__}: "
              f"{str(e)[:80]}）")
    try:                                                     # CANN 版本
        import torch_npu.utils.collect_env as ce           # noqa: F401
        info = ce.get_ascend_version() if hasattr(ce, "get_ascend_version") else "?"
        print(f"  CANN         : {info}")
    except Exception:                                        # noqa: BLE001
        pass
    print("  ⚠ 版本配对：torch_npu 必须与 torch 版本严格配对（如 2.5.1 ↔ 2.5.1）；"
          "不匹配会出现「探测到设备但算子不可用」。")

    # ── 2. 设备解析 ──
    _sec("2. 设备解析（auto 择优：昇腾 → ROCm → CUDA → DirectML → CPU）")
    from phdnet.backends.multi_device import (capability_report, probe_multi,
                                              resolve_devices)
    probes = probe_multi()
    for k, v in probes.items():
        print(f"  {k:5s} ok={v.get('ok')} count={v.get('count')} "
              f"name={str(v.get('name'))[:40]}")
    try:
        devs = resolve_devices(args.device, allow_fallback=True)
        print(f"  → resolve_devices({args.device!r}) = {devs}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  → 解析失败：{type(e).__name__}: {e}")
        devs = ["cpu"]
    dev = devs[0]

    # ── 3. 张量试分配 + 真实读出负载 ──
    _sec(f"3. 试分配 + 真实读出负载（V={args.V:,} × H={args.H:,} "
         f"fp32 ≈ {args.V * args.H * 4 / 2**30:.2f} GiB）")
    import numpy as np
    dev_str = "cpu" if dev == "cpu" else ("cuda" if dev.startswith("cuda") else dev)
    W = None
    t0 = time.perf_counter()
    try:
        if dev == "cpu":
            W = torch.zeros((args.V, args.H), dtype=torch.float32)
        else:
            W = torch.zeros((args.V, args.H), dtype=torch.float32,
                            device=torch.device(dev_str))
        _ = (W @ torch.zeros(args.H, dtype=torch.float32,
                             device=W.device)).float().cpu().numpy()
        print(f"  ✅ 分配 + 前向 matvec 成功（{time.perf_counter() - t0:.2f}s）")
        print(f"     设备张量：{W.device} {W.dtype} {tuple(W.shape)}")
        if dev != "cpu":
            # 显存信息（各后端 API 名不同，取不到就跳过——诊断不该因它失败）
            try:
                mod = getattr(torch, dev_str, None)
                if mod is not None and hasattr(mod, "mem_get_info"):
                    free, total = mod.mem_get_info()
                    print(f"     显存：可用 {free / 2**30:.2f} / 共 "
                          f"{total / 2**30:.2f} GiB")
                elif hasattr(torch, "cuda") and dev_str == "cuda":
                    free, total = torch.cuda.mem_get_info()
                    print(f"     显存：可用 {free / 2**30:.2f} / 共 "
                          f"{total / 2**30:.2f} GiB")
            except Exception as e:                           # noqa: BLE001
                print(f"     （显存信息不可取：{type(e).__name__}）")
    except Exception as e:                                   # noqa: BLE001
        print(f"  ❌ 试分配失败：{type(e).__name__}: {str(e)[:200]}")
        print("     → 该设备「被探测到」但实际不可用（常见：算子缺失 / 显存不足 / "
              "版本不配对）")
        return 1

    if args.no_bench:
        capability_report(verbose=False)
        return 0

    # ── 4. 性能对照：设备 torch vs CPU torch vs numba 读出 ──
    _sec("4. 性能对照（同一负载，ms/token）")
    rng = np.random.default_rng(0)
    h = rng.normal(0, 1, args.H).astype(np.float32)
    W[torch.arange(min(64, args.V), device=W.device)] = 0.01

    def bench_torch(device_str: str) -> float | None:
        try:
            if device_str == "cpu":
                Wm = torch.zeros((args.V, args.H), dtype=torch.float32)
                ht = torch.from_numpy(h)
            else:
                Wm = torch.zeros((args.V, args.H), dtype=torch.float32,
                                 device=torch.device(device_str))
                ht = torch.from_numpy(h).to(Wm.device)
            t = torch.zeros(args.V, dtype=torch.float32, device=Wm.device)
            t[7] = 1.0
            for _ in range(3):                              # 预热
                y = Wm @ ht
                p = torch.softmax(y, dim=0)
                Wm.add_(torch.outer(p - t, ht), alpha=-0.01)
            t0 = time.perf_counter()
            for _ in range(args.steps):
                y = Wm @ ht
                p = torch.softmax(y, dim=0)
                Wm.add_(torch.outer(p - t, ht), alpha=-0.01)
            dt = (time.perf_counter() - t0) / args.steps
            del Wm
            return dt * 1000.0
        except Exception as e:                               # noqa: BLE001
            print(f"     （{device_str} 基准失败：{type(e).__name__}: {str(e)[:60]}）")
            return None

    res_dev = bench_torch(dev_str)
    res_cpu = bench_torch("cpu")
    print(f"  torch @ {dev_str:8s}: "
          + (f"{res_dev:8.2f} ms/token" if res_dev else "不可用"))
    print(f"  torch @ cpu      : "
          + (f"{res_cpu:8.2f} ms/token" if res_cpu else "不可用"))
    try:
        from phdnet.readout import Readout
        ro = Readout(args.H, args.V, np.random.default_rng(0), w_clip=0.0,
                     dtype="fp32")
        ro.W[:] = 0.0
        tt = np.zeros(args.V, dtype=np.float32)
        tt[7] = 1.0
        for _ in range(2):
            ro.learn_softmax(h, tt, 0.01)
        t0 = time.perf_counter()
        for _ in range(args.steps):
            ro.learn_softmax(h, tt, 0.01)
        dt_numba = (time.perf_counter() - t0) / args.steps * 1000.0
        print(f"  numba Readout   : {dt_numba:8.2f} ms/token（生产 CPU 基线）")
        if res_dev:
            print(f"  → 设备 vs numba 生产基线：{dt_numba / res_dev:.2f}×")
    except Exception as e:                                   # noqa: BLE001
        print(f"  numba 基线失败：{type(e).__name__}: {e}")

    # ── 5. 结论 ──
    _sec("5. 结论")
    rep = capability_report(verbose=False)
    print(f"  探测到的加速器：{rep['accelerators_present'] or '无'}")
    if devs == ["cpu"] or not rep["accelerators_present"]:
        print("  → 本机无可用加速器；生产走 numba/CPU 属正常（numba 只能编译 CPU）。")
    else:
        print(f"  → 设备 {devs} 可用。生产训练读出是否走它，取决于：")
        print("     ① --accel / accel_readout 开关（默认 auto，应自动上设备）；")
        print("     ② 训练日志 `[读出] 后端=...` 行（若显示 numba-cpu(回落)，")
        print("        同行会给出回落原因）；")
        print("     ③ PC 栈 / STDP / LTM 仍是 numba CPU（torch 栈缺 7 项机制），")
        print("        故只有读出这一块在设备上。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
