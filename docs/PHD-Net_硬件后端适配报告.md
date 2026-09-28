# PHD-Net 硬件后端适配报告

> 适配对象：CUDA（NVIDIA）/ CANN·昇腾 NPU（华为）/ ROCm（AMD）/ CPU 参考
> 适配日期：2026-09-26（P10 基础层）；2026-09-28（P11 词级 LM 全栈迁移）
> 指令依据：fhz「我需要你适配 cuda, CANN, ROCm，完成后给出一份适配报告」
> 及「LM全栈请迁移（让子代理帮忙），要支持不同硬件」
> 配套：《性能评估与迭代方案》（精度体系 P9 / 性能剖析 §6.13）；`phdnet/backends/README.md`

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
| M6 读出热路径 | `TorchReadout`：前向 matvec + softmax 梯度更新；`probe_devices()` 统一探针；`bench_readout()` 基准 | 全平台（fp32/fp16/bf16） |
| **M1–M6 词级 LM 全栈（P11，2026-09-28）** | `backends/torch_lm.py`：`TorchWordLM` / `TorchPHDNet` —— M1 k-WTA 编码、M2 CSR 预测编码（边表示 + index_add 前向 + 向量化学习）、M3 STDP（`_ProdSTDPCore` 复刻 numba 热路径语义）、M4 门控 WM + 快慢 LTM、M5 Welford 调制器（标量态设备无关直接复用）、M6 softmax 读出；`resolve_device()` 诚实报错不静默回退 | 全平台（fp32/fp16/bf16） |
| 1B 生产路径（train_1b/） | numpy/numba CPU 多进程流水线（主力生产路径，readout fp32 融合核） | CPU |
| 大空间表（big_ltm）/ 其余扩展机制 | numpy/numba（CPU） | ⚠ torch 栈未迁移（见 §五） |

**P9 精度体系与设备的关系**（`readout_dtype`：fp32 默认 / fp16 / bf16 / fp8 / fp4）：

| 精度 | CPU（numba 码本） | torch 设备（CUDA/ROCm/NPU） |
|---|---|---|
| fp32 | ✅ 原生（BLAS + 融合核） | ✅ 原生 |
| fp16 / bf16 | ✅ 码本 + LUT 位算法核 | ✅ 原生张量核（GPU 上才是真加速） |
| fp8 (e4m3fn) | ✅ 码本 + 位算法核 | ⚠ 需 CUDA ≥ 8.9 + torch ≥ 2.1（float8_e4m3fn）；CPU torch 无原生 fp8 |
| fp4 (e2m1) | ✅ 半字节打包 + 逐张量缩放（代码保留） | ❌ **已禁用**（fhz 2026-09-27：e2m1 分辨率不足，4K 口径 ppl +5.88%；MX 块缩放立项后重新启用） |

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
| **LM 全栈等价性（P11）** | `tools/verify_torch_lm.py`：torch vs numpy 主实现（同 seed、冻结语料、base 配置）PPL 相对差 **0.00003%**（容差 1%），读出/STDP/PC 三条权重轨迹 Pearson 相关全部 **1.000000**（容差 0.99）→ PASS |
| **LM 冒烟训练（P11）** | `tools/train_torch_lm.py --preset smoke` 2,000 token 跑通（15 ms/token，OOV 0）；PPL 曲线中段上冲经 numpy 同口径对拍确认为 128 维小栈固有动态（两版曲线四位小数一致），非 torch 实现差异 |

## 五、诚实边界与迁移路线

1. **torch LM 栈未迁移机制**（`TorchPHDNet` 构造时逐项检查并显式拒绝，不做静默近似）：
   adaptive_lr / stdp_homeostasis / metaplasticity / ei_synapses / big_ltm /
   retrieval_topk / readout_hidden / readout_conn_k / lognormal_init /
   wm_summary_every / readout_recurrence / segment_check / plateau_sleep /
   multi_modulation；`readout_dtype` 仅 fp32/fp16/bf16（fp8 需 CUDA≥8.9 真机，
   fp4 已禁用）。1B 生产路径（稀疏 CSR/STDP 主干 + 大空间表）仍为 numba CPU。
2. **M3 的 numpy↔torch 每步互转**：n_top 维小向量，CPU 上开销可忽略；
   CUDA 真机上是已知优化点。
3. **fp8/fp4 的 GPU 加速**：fp8 走 torch 原生 `float8_e4m3fn`（仅 Ada/Hopper+）；
   fp4 已禁用（MX 块缩放立项后重新评估）。
4. **无硬件验证边界**：CUDA/ROCm/NPU 的代码路径在本机只能做结构验证（探针降级、自检
   框架就位）；三平台的真机回归须在具备对应硬件的机器或 CI GPU runner 上执行
   （`tools/bench_accel.py` + `tools/verify_torch_lm.py --device all` 即为 runner
   就绪的基准/验证入口）。
5. **精度判据**：跨设备用容差一致（`atol/rtol` + 低精度相关性 ≥0.99，沿用 selftest_torch
   双档判据）；不宣称跨设备逐位等价。
