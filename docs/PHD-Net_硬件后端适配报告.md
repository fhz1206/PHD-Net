# PHD-Net 硬件后端适配报告

> 适配对象：CUDA（NVIDIA）/ CANN·昇腾 NPU（华为）/ ROCm（AMD）/ CPU 参考
> 适配日期：2026-09-26（P10）
> 指令依据：fhz「我需要你适配 cuda, CANN, ROCm，完成后给出一份适配报告」
> 配套：《性能评估与迭代方案》（精度体系 P9 / 性能剖析 §6.13）

---

## 一、结论速览

| 平台 | 适配状态 | 验证状态 | 说明 |
|---|---|---|---|
| **CUDA** | ✅ 代码就绪 | ⚠ 结构验证（无硬件） | torch 原生路径；fp8 需 CUDA ≥ 8.9（Ada/Hopper）+ torch ≥ 2.1 |
| **ROCm** | ✅ 代码就绪 | ⚠ 结构验证（无硬件） | 走 HIP 化的 cuda 接口，**算子代码与 CUDA 完全相同**（`torch.version.hip` 判别） |
| **CANN·昇腾 NPU** | ✅ 代码就绪 | ⚠ 结构验证（无硬件） | 需安装 `torch_npu` 插件（版本须与 CANN/torch 配对）；import 即注册 `npu` 设备 |
| **CPU 参考** | ✅ 完整可用 | ✅ 实测通过 | torch 2.14.0+cpu 实测；numba 路径为主力生产路径 |

**适配原则**：同一份算子代码，仅 device/dtype 不同（后端只改 device/dtype，不改算法语义）；
每条设备路径启用前必须过**等价性自检**（`selftest_torch`，容差判据——跨设备归约顺序不同，
逐位等价在跨设备场景**不成立**，这是与 CPU 内部优化 P6/P7 的本质区别）。

## 二、架构与分层

| 层 | 现状 | 设备覆盖 |
|---|---|---|
| M3 STDP 时序关联核 | `TorchSTDPCore`（scatter-add 语义与 numpy 版一致，含等价自检） | 全平台 |
| **M6 读出热路径（P10 新增）** | `TorchReadout`：前向 matvec + softmax 梯度更新；`probe_devices()` 统一探针；`bench_readout()` 基准 | 全平台（fp32/fp16/bf16） |
| M1 编码 / M2 预测编码 / M4 记忆 / M4b 大空间表 | numpy/numba（CPU） | ⚠ 未迁移（见 §五 路线） |
| 词级 LM 训练循环 | `train_1b/train.py`（CPU 流水线） | ⚠ 未接入 torch 路径 |

**P9 精度体系与设备的关系**（`readout_dtype`：fp32 默认 / fp16 / bf16 / fp8 / fp4）：

| 精度 | CPU（numba 码本） | torch 设备（CUDA/ROCm/NPU） |
|---|---|---|
| fp32 | ✅ 原生（BLAS + 融合核） | ✅ 原生 |
| fp16 / bf16 | ✅ 码本 + LUT 位算法核 | ✅ 原生张量核（GPU 上才是真加速） |
| fp8 (e4m3fn) | ✅ 码本 + 位算法核 | ⚠ 需 CUDA ≥ 8.9 + torch ≥ 2.1（float8_e4m3fn）；CPU torch 无原生 fp8 |
| fp4 (e2m1) | ✅ 半字节打包 + 逐张量缩放 | ❌ torch 无原生类型 → 需自定义 kernel（独立立项） |

**质量实测**（冻结语料 4K 口径，`tools/audit_precision.py`）：fp16 −0.17% / bf16 −1.16% /
fp8 **−1.70%**（量化噪声的正则化效应，PPL 反而略优）/ fp4 +5.88%（e2m1 分辨率粗，需块缩放立项）；
**fp32 与历史 fp64 锚点逐位一致**（4K 96.7241 / 全语料 77.5261）——默认精度切换零质量损失。

## 三、启用方法（按平台）

```bash
# 通用依赖
pip install numpy numba pyarrow psutil

# CUDA（NVIDIA）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# ROCm（AMD）
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# CANN·昇腾 NPU（华为）—— torch_npu 版本须与 CANN/torch 配对（如 torch 2.1 ↔ torch_npu 2.1）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu                       # 或从 CANN 配套源安装

# 设备探针 + 读出基准（任一平台）
python tools/bench_accel.py                 # 打印各平台可用性 + fp32/fp16/bf16 读出基准
python -c "from phdnet.torch_backend import selftest_torch; print(selftest_torch('cuda'))"
```

## 四、本机实测（CPU 参考路径，torch 2.14.0+cpu）

| 项 | 结果 |
|---|---|
| 设备探针 | cpu ✓；cuda/rocm/npu 均不可用（无硬件，探针诚实降级并告警） |
| STDP 等价自检 | `selftest_torch('cpu')` PASS |
| 读出基准 | 见 `outputs/bench_accel.{log,json}`（V=9219 × H=3072） |

## 五、诚实边界与迁移路线

1. **LM 全栈尚未迁移**：M1/M2/M4/M4b 与词级训练循环仍为 numpy/numba。读出（M6）占单步
   ~85–90%（§6.13 剖析）故优先 torch 化；全栈迁移是一项独立工程（CSR 稀疏核需映射为
   dense mask 或 torch.sparse，大空间表需设备端稀疏张量）。
2. **fp8/fp4 的 GPU 加速**：fp8 走 torch 原生 `float8_e4m3fn`（仅 Ada/Hopper+）；fp4 需
   自定义 kernel（如双位打包 + LUT 反量化），CPU 码本路径已验证语义。
3. **无硬件验证边界**：CUDA/ROCm/NPU 的代码路径在本机只能做结构验证（探针降级、自检
   框架就位）；三平台的真机回归须在具备对应硬件的机器或 CI GPU runner 上执行
   （`tools/bench_accel.py` 即为 runner 就绪的基准入口）。
4. **精度判据**：跨设备用容差一致（`atol/rtol` + 低精度相关性 ≥0.99，沿用 selftest_torch
   双档判据）；不宣称跨设备逐位等价。
