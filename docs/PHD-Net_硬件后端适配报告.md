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
| **LM 全栈等价性（P11）** | `tests/verifiers/verify_torch_lm.py`：torch vs numpy 主实现（同 seed、冻结语料、base 配置）PPL 相对差 **0.00003%**（容差 1%），读出/STDP/PC 三条权重轨迹 Pearson 相关全部 **1.000000**（容差 0.99）→ PASS |
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
   （`tools/bench_accel.py` + `tests/verifiers/verify_torch_lm.py --device all` 即为 runner
   就绪的基准/验证入口）。
5. **精度判据**：跨设备用容差一致（`atol/rtol` + 低精度相关性 ≥0.99，沿用 selftest_torch
   双档判据）；不宣称跨设备逐位等价。

## 六、多卡自动适配（P14，2026-09-28）

### 结论先行：为什么不是 DDP / DataParallel

PHD-Net 是**逐 token 事件驱动的稀疏网络**：无 batch 维、无注意力，全局状态
（WM 情景缓冲 / STDP 印迹 / LTM 长时记忆）跨样本连续携带。标准 DDP 需要可分的
梯度与 batch 归约——本网络**没有**；DataParallel 需要 batch 轴可切——本网络**没有**。
因此多卡走**模型并行**，且只切真正可分的部分。

### 切分策略

| 部件 | 可分性 | 方案 | 状态 |
|---|---|---|---|
| 读出 W ∈ R^{V×H} | **完全可分**（逐行 dot + 逐行外积更新） | 按词表行（V 维）列并行到各卡 | ✅ 已实现 + 逐位验证 |
| PC 栈（稀疏 CSR 主循环） | 事件驱动、隐式耦合 | 不切 | 上游单卡（主设备） |
| STDP 印迹 / LTM 2^24 | CSR 天然可按神经元区间切 | `shard_ranges` 计划 + 跨卡共激活对通信 | 计划层（1B 真机待验） |
| 分词 / SDR 编码 | CPU numpy | 不切 | CPU |

读出列并行的通信量：每步两次小向量（全局 logits y 下发 + dp 分片上行），
V=9,219 时约 37 KB×2 ≈ 数 μs，不构成瓶颈。softmax 主回路在主卡 fp32
（与 P9 协议一致）。1B 预设读出占端到端 ~89% → N 卡上限 ≈ 1/(1-0.89+0.89/N)
（N=4 → ≈3.3×，**仅为读出部分的线性加速，非全模型加速**）。

### 公共 API（`phdnet/backends/multi_device.py`）

- `resolve_devices("auto" | "cuda:0,1" | "npu:0 1" | "cpu", max_devices=N)`
  → 设备列表；auto = 昇腾 → ROCm → CUDA → DirectML → CPU 择优并取**全部同型号设备**
  （与 `resolve_device` 同一优先级）；不可用设备诚实报错（`allow_fallback=True` 才回退）；
- `probe_multi()`：多卡视角探针（含各平台设备串列表）；
- `shard_ranges(n_items, n_devices)`：均衡切分（读出 V / LTM 神经元共用）；
- `plan_parallel(devices, V, H)`：自动选型 + 诚实预期（策略、通信量、未并行部分）；
- `configure_host_threads()`：多卡时 OMP/MKL 线程收敛为 1；
- `MultiDeviceReadout(W, devices, dtype)`：列并行读出（forward / learn_softmax /
  `to_numpy` / `load_rows` 检查点往返）。

### 接入点

- 训练：`tools/train_torch_lm.py --devices auto`（单设备 = 原生路径，零行为变更；
  多卡自动启用读出列并行 + host 线程收敛）；`TorchWordLM(devices=...)` /
  `TorchPHDNet(readout_devices=...)` 透传；
- 推理：`train_1b/infer.py --devices auto` 打印设备清单与计划，并**明确声明**
  生产推理为 numba CPU 单路（不假装多卡）；多卡推理走 torch 栈，**权重不通用**。

### 验证（`tests/verifiers/verify_multi_device.py`，24 例全 PASS）

- A 设备解析：auto / 显式列表 / 不可用设备报错 / fallback / max_devices；
- B 分片：均衡、余数分配（10/4 → 3,3,2,2）、空段剔除、覆盖完整；
- C 单设备 ≡ `TorchReadoutDense`（前向 / NLL / 更新后 W，**逐位** max|Δ|=0）；
- D 4 路分片 ≡ 单设备（**逐位** max|Δ|=0）；真实训练端到端 4 分片 PPL 与单卡
  完全一致（smoke preset 567.7016）；
- E 计划报告字段；F host 线程收敛。
- **诚实边界**：本机无加速器，跨**不同**卡的路径仅做结构验证（判据为容差一致，
  非逐位）；同设备分片为逐位一致。多卡真机回归待硬件到位后按 D 例扩到真机执行。

## 七、为什么昇腾机器上没有用 NPU（P18，2026-09-28）

**根因不是探测失败，而是后端选择**：

| 后端 | CPU | 昇腾 NPU | CUDA | ROCm | DirectML |
|---|---|---|---|---|---|
| numba（**生产**：`train_1b/train.py`、`infer.py`） | ✓ | ✗ | ✗ | ✗ | ✗ |
| torch 栈（`tools/train_torch_lm.py`、`phdnet.backends`） | ✓ | ✓ | ✓ | ✓ | ✓ |

numba 的 `@njit` **只能编译到 CPU 机器码**（物理限制）。生产路径的分词核、读出
融合核、稀疏主循环全部是 numba，因此 NPU 机器上它们依然跑在 CPU 上——日志里
的「numba nogil 线程 ×N」指的是 CPU 线程，不是 NPU。

**为什么此前不提示**：静默走 CPU 且不说明，容易被误判为「NPU 没被识别」。现已在
`phdnet.backends.multi_device.capability_report()` 显式给出矩阵 + 行动建议，
并在 train / infer 启动日志中提示（仅当探测到加速器时）。

**要用上 NPU 的实际路径**（诚实边界）：

```bash
python tools/train_torch_lm.py --device auto      # auto 择优：昇腾 → ROCm → CUDA → DirectML → CPU
python tools/train_torch_lm.py --devices auto     # 多卡：读出列并行（见 §六）
python tests/verifiers/verify_torch_lm.py --device all   # 三平台等价性自检（容差判据）
```

限制：torch 栈机制覆盖不全（M1–M6 + 部分 M7；`big_ltm`、自适应 LR 等 7 项显式
`NotImplementedError`），且**权重与生产 npz 检查点不通用**。生产 1B 模型的
numba 路径暂未提供 NPU 移植方案（需要重写全部算子，工作量以周计）。

## 八、计算型部件自动上设备（P19，fhz「有 cuda/cann/rocm 就跑对应设备」）

**已迁移：读出（M6）** —— 1B 预设里占端到端 ~89%（V×H = 73,958×3,072 fp32 ≈ 908 MB），
且逐 token 只有一个 h 向量参与计算 → 每步通信仅 12 KB 上行 + 少量下行，可忽略。

| 项 | 说明 |
|---|---|
| 开关 | `PHDNetConfig.accel_readout`，**默认 `auto`**；CLI `--accel auto\|cpu\|npu\|cuda\|rocm\|dml`（train 与 infer 均支持） |
| auto 行为 | 探测到加速器（昇腾/CUDA/ROCm/DirectML）即用（择优顺序同 P10）；**无加速器回落 numba CPU 原路径，逐位不变**；构造失败亦回落并记录 `_accel_fallback_reason`（如实报告，不静默） |
| 实现 | `phdnet/backends/accel_readout.py::AccelReadout`（W 常驻设备，接口对齐 `Readout`：forward / learn_softmax / W / n_synapses / W_cpu / load_W） |
| 精度 | softmax 主回路恒 fp32（与 P9 协议一致）；低精度档 fp16/bf16 可用 |
| 等价性 | **容差一致**（跨库 numpy BLAS ↔ torch 内核归约顺序不同，实测 max\|Δ\| ≈ 4e-06）；`tests/verifiers/verify_accel_readout.py` 12 例全 PASS |
| 日志 | 训练启动打印 `[读出] 后端=accel:<设备>` 或 `numba-cpu`（含回落原因） |

**未迁移（诚实说明）**：

- **分词/词表扫描**：变长字符串 + 状态化锚点链，设备侧不划算（需整段上传 + 变长
  输出），留 CPU（numba nogil 多核）。实测训练循环内分词占比 ~0.0%。
- **词涌现统计**：已多核化（numba prange，4.5×），但形态是可精确并行的整数统计 +
  浮点熵，迁设备需重写并引入容差判据；当前 CPU 已非瓶颈。
- **PC 栈 / STDP / LTM**：事件驱动稀疏 + 在线 CSR 生长；torch 栈缺 `big_ltm` 等
  7 项机制，整体迁移会**丢机制**。生产 1B 的全 NPU 移植需重写全部算子。
