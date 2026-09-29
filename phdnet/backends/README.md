# phdnet.backends —— 硬件后端子包

CUDA / ROCm / CANN·昇腾 NPU / DirectML / CPU（P10 目录化，P30 定稿，P40–P52 增补，2026-09-29）。

## 当前文件

| 文件 | 内容 |
|---|---|
| `accel_readout.py` | **生产加速读出** `AccelReadout`（P19/P28：auto 选设备、`addmm_` rank-1 AXPY 更新、设备侧 (h,y) 缓存、`W_cpu`/`load_W` 检查点接口）+ `pick_readout_backend`（auto 择优/不兼容回落并记原因）。P40 起训练步走设备直通（见下"AccelReadout 训练直通"）。P44 起 `torch_compile_mode` 默认 `"default"` |
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

## AccelReadout 训练直通（P36/P40，P44/P45/P52 演进）

- **训练步设备直通**：`model.step` 对加速后端走 `forward_dev(h)`（设备张量直通，
  零 D2H）+ `target_idx`（int，onehot 在**设备上**构造，省 289 KiB H2D/步）。
  推理/评估路径保持 numpy。
- **识别标志（接口纪律）**：用类属性 `AccelReadout._is_accel = True` 显式标识，
  **不要用 `hasattr(readout, 'forward_dev')` 猜**——numba Readout 也有同名旧接口，
  曾导致误走设备路径崩溃（P40 教训）。
- **融合步核**：P38 `_train_step_core(y32, ht, t32, correct, eta)` 是融合候选
  （softmax/nll/addmm_ 一体），**接收前向缓存的 y32，不重复 matvec**。
- **P6 感知器 dp 复用**（P45）：直接复用 softmax 输出 p（`p[c] -= 1` 就地），
  不再 `zeros_like` 分配。
- **torch.compile（P44/P45 定稿）**：`torch_compile_mode` 默认 **"default"**
  （只融合 kernel，不启用 cudagraphs）。`reduce-overhead` 与本实现**本质冲突**：
  W 每步原地更新（`W.addmm_`）→ cudagraph 拒绝 mutate 输入 → 每步打印
  "skipping cudagraphs due to mutated inputs" 并静默回退；W 原地更新是硬约束
  （改非原地会让流量翻倍），故不修冲突、只锁默认值。
- **P45 实测**：修掉 P40 引入的 `zeros_like` 分配后，compile **开快 15%**
  （26.34 vs 30.98 ms/tok，本机 400-token smoke）。修之前测是反向的——
  教训：**A/B 必须在最终代码上重测**。

## M2 融合核（P52，`phdnet/sparse_pc.py::_pc_infer_fused`）

- `@njit(cache=True, nogil=True, parallel=True, fastmath=True)`：把 5 次
  `_csr_matvec` + n_steps 循环融进 1 次调用；中间数组核内一次分配 + 复用；
  返回独立数组（cache 跨步语义不变）。
- 实测（smoke 档 PC 栈）：n_steps=1 **2.10×**，n_steps=3 **2.16×**。
- ⚠ **容差一致（1 ulp），非逐位**：融合后编译器 fastmath 重结合决策与逐算子版
  不同，已如实记录待裁决——符合"容差不宣称逐位"的总体口径。

## fastmath 实验（P39，负结果已记录）

- 生产并行融合核（parallel=True）加 fastmath **无收益**（50.62 vs 40.85 ms——
  带宽饱和的广播乘加，浮点严格性不是瓶颈）。
- CPU numba 读出核已带宽饱和（43 GB/s ≈ DDR4 上限），不必再追浮点微优化。

## numba 缓存（P39）

- `NUMBA_CACHE_DIR=outputs/numba_cache` 持久化（train/infer/train_rl 顶部设置）；
  `readout.py` 7 个 inline-always 核补 `cache=True`（冷启动 3.96→2.60s）。

## 周边配套（子包外，但属同一批变更）

- **telemetry（P41）**：`phdnet/telemetry.py::Telemetry`——CPU/RAM/NPU 利用率/
  HBM/CS 率（非阻塞）；IPC 需 perf（Python 拿不到，诚实返回 None）。
- **三源采样器（P49）**：`tools/fetch_ms.py`（web/code/math，
  `--plan web=3,code=2,math=1`，零原始落盘，断点续跑）。
- **fast 门禁**：9/9 通过。
