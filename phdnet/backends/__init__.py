"""硬件后端子包 —— CUDA / ROCm / CANN·昇腾 NPU / DirectML / CPU。

P30（2026-09-29）：删除了 `TorchSTDPCore` / `TorchReadout` / `selftest_torch`
与整套 torch 版 PHD-Net（`torch_lm.TorchPHDNet` 等）——它们权重不通用、缺 7 项
机制，从未进入生产路径。当前公共 API：

  - `AccelReadout`      **生产加速读出**（P19/P28：auto 选设备、AXPY 更新、
                        设备侧缓存；`accel_readout.py`）
  - `resolve_device()`  设备解析（auto：昇腾 → ROCm → CUDA → DirectML → CPU）
  - `probe_devices()`   设备探针（诚实降级 + 告警）
  - `bench_readout()`   读出基准（**含设备同步与等效带宽**，P29 修正）
  - 多卡：multi_device（`resolve_devices` / `plan_parallel` /
    `MultiDeviceReadout` / `capability_report`）
"""

from .accel_readout import (AccelReadout, pick_readout_backend,
                            resolve_accel_device)
from .torch_backend import bench_readout, probe_devices
from .torch_lm import resolve_device

__all__ = ["AccelReadout", "pick_readout_backend", "resolve_accel_device",
           "bench_readout", "probe_devices", "resolve_device"]
