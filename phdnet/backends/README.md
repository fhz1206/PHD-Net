# phdnet.backends —— 硬件后端子包

CUDA / ROCm / CANN·昇腾 NPU / DirectML / CPU（P10 目录化，P30 定稿 2026-09-29）。

## 当前文件

| 文件 | 内容 |
|---|---|
| `accel_readout.py` | **生产加速读出** `AccelReadout`（P19/P28：auto 选设备、`addmm_` rank-1 AXPY 更新、设备侧 (h,y) 缓存、`W_cpu`/`load_W` 检查点接口）+ `pick_readout_backend`（auto 择优/不兼容回落并记原因） |
| `multi_device.py` | 多卡（P14）：`resolve_devices` / `plan_parallel` / `MultiDeviceReadout`（读出列并行）/ `capability_report`（后端×设备能力矩阵） |
| `torch_lm.py` | 仅存 `resolve_device`（设备解析；auto：昇腾→ROCm→CUDA→DirectML→CPU）。其余（完整 torch 版 PHD-Net）已随 P30 删除 |
| `torch_backend.py` | 仅存 `probe_devices`（设备探针）+ `bench_readout`（读出基准，P29 修正：设备同步 + dtype 参数 + 等效带宽 GB/s + 默认测生产对象，默认 V=73,958） |

## 关键口径

- **读出加速只有一个实现**（`AccelReadout`）；旧 `TorchReadout`/`TorchSTDPCore`/
  `selftest_torch` 与整套 torch 版 PHD-Net 已随 P30 删除（权重不通用、缺 7 项机制、
  从未进生产）。
- 等价性判据：跨实现/跨设备用**容差**（归约顺序不同，max|Δ| ≈ 4e-06 量级），
  不宣称逐位；对拍脚本 `tests/verifiers/verify_accel_readout.py`（含 A4 接口
  完整性扫描）。
- `bench_accel.py` 报**等效带宽 GB/s**（读出是 GEMV，带宽是唯一可跨平台比较的指标）。
- 半精度（fp16/bf16）有机制性代价：非目标行更新量 ≈1e-6 < fp16 半 ULP →
  被舍入丢弃，学习规则退化为纯 Hebbian；切换前须在现行 eta=0.15 下重测 PPL。
