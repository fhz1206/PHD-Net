"""硬件后端子包 —— CUDA / ROCm / CANN·NPU / CPU（P10，2026-09-27 目录化）。

公共 API（与旧 phdnet.torch_backend 完全一致，旧路径保留兼容 shim）：
  - TorchSTDPCore    STDP 时序关联核（torch 版）
  - TorchReadout     读出热路径（fp32/fp16/bf16；fp8 需 CUDA>=8.9）
  - probe_devices()  设备探针（诚实降级 + 告警）
  - selftest_torch() 等价性自检（容差判据；跨设备不宣称逐位）
  - bench_readout()  读出基准
"""

from .torch_backend import (TorchReadout, TorchSTDPCore, bench_readout,
                            probe_devices, selftest_torch)

__all__ = ["TorchReadout", "TorchSTDPCore", "bench_readout",
           "probe_devices", "selftest_torch"]
