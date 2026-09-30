# `phdnet.backends` —— 硬件后端子包

> **本目录只放硬件后端。** 本文只写：文件索引、**每层在哪个设备上跑**、怎么用、
> 关键口径、昇腾踩坑速查指针。
> **不写**：昇腾坑的完整根因与处置（→ `../../docs/PHD-Net_硬件后端适配报告.md`）、
> 任何性能数字（→ `../../docs/PHD-Net_性能评估与迭代方案.md`，唯一出处）。
> **数据截止：2026-09-30。**

支持的平台：**numba CPU（生产主力）/ CANN·昇腾 NPU / CUDA / ROCm / DirectML**。

## 一、文件索引

| 文件 | 公共 API | 职责 |
|---|---|---|
| `accel_readout.py` | `AccelReadout`、`pick_readout_backend(cfg, n_h, n_out, rng)`、`resolve_accel_device(spec)` | **生产加速读出**（P19/P28：auto 选设备、AXPY 更新、设备侧缓存）。`pick_readout_backend` 是模型侧唯一入口 |
| `torch_backend.py` | `probe_devices()`、`bench_readout(device, V, H, dtype, steps, use_accel)` | 设备探针（诚实降级 + 告警）；读出基准（**含设备同步与等效带宽 GB/s**，P29 修正） |
| `torch_lm.py` | `resolve_device(device, allow_fallback)` | **仅**设备解析。auto 择优顺序：**昇腾 → ROCm → CUDA → DirectML → CPU** |
| `multi_device.py` | `resolve_devices`、`probe_multi`、`shard_ranges`、`configure_host_threads`、`plan_parallel`、`capability_report`、`MultiDeviceReadout` | 多卡（读出**列并行**）、分片计划、host 线程收敛、能力矩阵 |
| `__init__.py` | 导出 `AccelReadout` / `pick_readout_backend` / `resolve_accel_device` / `bench_readout` / `probe_devices` / `resolve_device` | 子包门面 |

**已删除（P30）**：`TorchSTDPCore` / `TorchReadout` / `selftest_torch` 与整套 torch 版
PHD-Net（`torch_lm.TorchPHDNet` 等，约 800 行）。删除理由：① 权重与生产 npz 检查点
**不通用**；② **缺 7 项机制**（`big_ltm`、自适应 LR 等显式 `NotImplementedError`）；
③ 生产加速路径已由 `accel_readout` 单独实现（只迁读出，不动 numba 主循环）。
**不要试图复活它们。**

## 二、每层在哪个设备上跑

| 层 / 部件 | 设备 | 说明 |
|---|---|---|
| 分词、词表构建、onehot 缓冲 | **CPU** | 变长字符串处理，NPU 不擅长（要整段上传 + 变长输出） |
| M1 稀疏编码器 | **CPU**（numba 自写 GEMV / BLAS） | 昇腾 aarch64 的 numpy GEMV 病态 → 走自写 numba 核；x86 走 BLAS |
| M2 预测编码主干（CSR） | **CPU**（numba） | 结构稀疏 + 事件驱动 CSR 生长，numba 只能编译到 CPU |
| M3 STDP 时侧向 | **CPU**（numba） | 逐突触时序更新，**只走 numba CPU** |
| M4a 工作记忆 / M4b 大空间表 | **CPU** | M4b 的在线 CSR 生长是 CPU 结构 |
| M5 神经调制 | **CPU** | 标量/小向量运算 |
| **M6 读出** | **可上加速器**（`AccelReadout`） | 唯一值得迁移的部件：V×H 矩阵大、每步只传一个 h 向量 |
| 多卡（可选） | **读出按词表行分片到各卡** | `MultiDeviceReadout`；上游 PC 栈 / STDP / LTM 仍在主设备单卡 |

**结论**：这是**混合执行**架构 —— 数据侧 + 学习侧在 CPU，只有 M6 读出在设备上。
`numba 只能编译到 CPU 机器码` 是物理限制，不是探测失败；日志里的
「numba nogil 线程 ×N」指的是 **CPU 线程**，不是加速器算力。

## 三、怎么用

训练 / 推理入口：

```bash
python train/train.py --accel auto     # auto = 有加速器就用，否则回落 numba CPU
python train/infer.py  --accel auto
python train/train.py --accel npu       # 显式指定；不可用则回落并记原因
python train/train.py --devices auto    # 多卡：读出列并行（与 --accel 正交）
```

**确认加速是否真生效**：看启动日志的 `[读出] 后端=accel:<设备>` / `numba-cpu`；
回落时同行打印 `fallback reason: <原因>`。**不要用「探测到设备」判断加速可用** ——
探测 ≠ 能算，跑 `accel_doctor.py`。诊断与基准：

```bash
python tools/backend_probe.py     # 探测层：各平台 ok / count / name
python tools/accel_doctor.py      # 试分配 + 一次前向 matvec，验证设备真的能算
python tools/bench_accel.py       # 读出基准，报等效带宽 GB/s（跨平台唯一可比指标）
```

## 四、关键口径

### 回落纪律（不静默）

`pick_readout_backend` 在下列任一情况下**回落原 numba `Readout`**
（**默认路径逐位不变**），并把原因写在 `readout._accel_fallback_reason`：

| 触发条件 | 回落原因文案 |
|---|---|
| `--accel cpu/off/numba` | 不算回落，直接走 numba 主路径 |
| `readout_hidden > 0`（两级读出） | 加速后端未实现 |
| `readout_conn_k > 0`（稀疏读出） | 加速后端未实现 |
| `readout_dtype ∈ {fp8, fp4}` | 加速后端**禁用量化码本**（见下） |
| 无加速器 / torch 未安装 / 构造异常 | 物理不可用 |

设计原则：**宁可回落，也不要「能跑但语义不同」** —— 后者比回落到 numba 更糟。

### 精度：读出默认 fp16

| 项 | 口径 |
|---|---|
| 默认 | `--readout-dtype fp16`（**CLI 默认**；库 `config` 默认 bf16 —— 两者不一致是事实） |
| 动机 | **学习精度**，不是访存收益：bf16 半 ULP ≈ 2e-4 ≫ 非目标行更新 \|dp\| ≈ 1e-6 → 更新被舍 → 退化为纯 Hebbian |
| 加速器可用档位 | **fp32 / fp16 / bf16** |
| **fp8 / fp4** | **在所有加速器禁用**（P86）。昇腾实测 `Float8_e4m3fn has not been supported`（ERR01007），fp4 的 MX 块缩放同样无算子 |
| fp8 的唯一可用路径 | `--accel cpu --readout-dtype fp8` —— 走 numba CPU 的 P9/P12 位算法量化核，**不是加速能力** |
| 检查点 | bf16/fp8 存**位模式**（uint16 / uint8），加载侧按 `ckpt_dtype` 解码闭环（numpy 无原生 bf16，不这样存会崩） |
| 其它 | M1 编码器默认 fp64；分词 onehot 缓冲 fp16（值域 0/1，无损） |

各精度的 PPL / 带宽数字见《性能评估与迭代方案》，本文不复制。

### 等价性判据

跨实现 / 跨设备：只宣称**容差一致**（`atol/rtol`，归约顺序不同，**不宣称逐位**）。
**同设备**：torch CPU vs torch CPU 可逐位；多卡分片 vs 单设备可逐位。验证入口见文末 §六。

### 接口纪律（改代码前先读）

1. **用类属性 `AccelReadout._is_accel = True` 标识加速后端** ——
   **不要用 `hasattr(readout, 'forward_dev')` 猜**：numba `Readout` 也有同名旧接口，
   曾导致误走设备路径崩溃。
2. **必须对齐 `Readout` 的全部访问面**：`W` / `forward` / `__call__` / `learn` /
   `learn_softmax` / `n_synapses` / `conn_k` / `hidden` / `stats()` / `W_cpu` /
   `load_W`。缺一项会在**生产保存检查点**时才崩。
3. **`torch.compile` 是惰性编译**：构造期不编译、**首次调用**才编译 →
   首次失败**永久回落 eager + 告警**；`--torch-compile` **默认关**。
   `reduce-overhead` / cudagraphs 与本实现**本质冲突**（W 每步原地 mutate 输入），
   只锁 `torch_compile_mode="default"`。
4. **融合步核必须逐行镜像 eager 路径**：`target_idx` 路径的「就地改 p」曾只写在
   eager 分支，融合核漏改 → NPU 首跑即崩。
5. **可选依赖惰性 import + `getattr` 防御**：未定义的补丁函数曾让修改静默 no-op。
6. **`learn_softmax` 的 `accumulate` 参数当前未被使用**（加速后端静默忽略 minibatch
   梯度累积）。当前默认 1 → 无行为差异，**启用前必须实现或显式拒绝**。
7. **x86 的性能结论不构成证据**：M1 fp32、M2 融合核、`OMP_PLACES=cores` 三项在昇腾上
   分别慢 70× / 8–13× / 8×。跨平台改动必须在目标机器复测。

## 五、昇腾踩坑速查（**只列现象，根因见硬件后端适配报告**）

完整根因与处置见 **`docs/PHD-Net_硬件后端适配报告.md`**。此处仅作索引：

| 现象关键词 | 报告章节 |
|---|---|
| NPU 首跑崩 `FakeTensor - None`（融合核漏改分支） | §七 昇腾真机坑位 |
| `npu-smi` / Telemetry 恒 `--`（PATH 与列名解析） | §七 |
| 每步 `.item()` 把设备延迟全额暴露给 CPU、pageable H2D 阻塞 | §七 |
| OpenBLAS 拉满 191 线程、线程无核心亲和性 | §七 + 《并行与加速架构分析》 |
| `OMP_PLACES=cores` 反噬（已回滚，只留 `PROC_BIND`） | §七 |
| M1 aarch64 numpy GEMV 病态（两次误判） | §七 |
| bf16 检查点保存崩（numpy 无原生 bf16） | §七 |
| fp8 / fp4 ERR01007、量化码本禁用 | §四 精度与后端 |
| 多卡读出列并行为什么不是 DDP | §六 多卡 |

## 六、验证入口

| 脚本 | 覆盖 |
|---|---|
| `tests/verifiers/verify_accel_readout.py` | `AccelReadout` 对拍 + **接口完整性扫描**（扫全仓库 `readout.X` 访问面，断言加速后端具备） |
| `tests/verifiers/verify_accel_readout_p55.py` | 融合核/eager × 两种 target 路径的矩阵 |
| `tests/verifiers/verify_multi_device.py` | 设备解析 / 分片均衡与余数 / 单设备 ≡ `AccelReadout` / 分片 ≡ 单设备（逐位）/ 计划报告 / host 线程收敛 |
| `tests/verifiers/bench_accel_path.py` | 复现读出路径的设备流量与墙钟对照 |
| `python tests/run_tests.py fast` | 零回归门槛，**9/9** |

**周边配套**（子包外，同一批变更）：`phdnet/telemetry.py::Telemetry`（CPU/RAM/NPU
利用率、HBM、CS 率，非阻塞；IPC 需 perf，Python 拿不到就诚实返回 `None`）；
`NUMBA_CACHE_DIR=outputs/numba_cache` 持久缓存（train / infer / train_rl 顶部设置）。
