# PHD-Net 硬件后端适配报告

> 适配对象：CUDA（NVIDIA）/ CANN·昇腾 NPU（华为）/ ROCm（AMD）/ DirectML / CPU 参考
> 更新日期：2026-09-29（P28 NPU 性能根因修复 / P29 基准工具修正 / P30 旧 torch 栈删除）
> 历史节点：P10 基础层（09-26）→ P14 多卡（09-28）→ P19 只迁移读出（09-28）
> → P28/P29/P30（09-29，本版）
> 配套：《性能评估与迭代方案》（精度体系 P9）；`phdnet/backends/README.md`

---

## 一、结论速览

**架构决策（P30 定稿）**：加速的正确姿势是**只迁移读出**（M6），不维护平行实现。
生产路径 =

- **numba CPU 主循环**（PC 栈 / STDP / LTM / 分词，机制完整）；
- **`phdnet/backends/accel_readout.py::AccelReadout`**（torch 设备上的读出，W 常驻设备）。

| 平台 | 读出加速（AccelReadout） | 验证状态 | 说明 |
|---|---|---|---|
| **CUDA** | ✅ 代码就绪 | ⚠ 结构验证（无硬件） | fp8 需 CUDA ≥ 8.9（Ada/Hopper）+ torch ≥ 2.1 |
| **ROCm** | ✅ 代码就绪 | ⚠ 结构验证（无硬件） | 走 HIP 化接口，算子代码与 CUDA 完全相同（`torch.version.hip` 判别） |
| **CANN·昇腾 NPU** | ✅ 代码就绪 | ⚠ 结构验证（无硬件） | 需 `torch_npu` 插件（版本须与 torch 严格配对）；import 即注册 `npu` 设备 |
| **DirectML** | ✅ 代码就绪 | ⚠ 结构验证 | torch-directml |
| **CPU 参考** | ✅ 完整可用 | ✅ 实测通过 | numba 路径为主力生产路径 |

**适配原则**：同一份算子代码，仅 device/dtype 不同（后端只改 device/dtype，不改算法语义）。
跨设备判据为**容差一致**（`atol/rtol`，跨库归约顺序不同，逐位等价在跨设备场景不成立）；
`AccelReadout` 与 numba 读出的实测 max|Δ| ≈ 4e-06。多卡分片路径同设备逐位一致（§五）。

**测试现状（2026-09-29）**：fast 测试 **9/9** PASS；`verify_accel_readout` 全 PASS
（21+ 用例，含 A4 接口扫描：正则扫全仓库 `readout.X` 访问面，断言 AccelReadout 具备）；
`verify_multi_device` 全 PASS；`verify_rl` 全 PASS。

## 二、当前架构（主线）

| 层 | 实现 | 设备 |
|---|---|---|
| 分词 / 词表扫描 | numba nogil 多核（CPU） | CPU |
| M1–M5：PC 栈 / STDP 印迹 / LTM（含 big_ltm 2^24）/ 调制器 | numpy/numba **生产主循环**，机制完整 | CPU |
| M6 读出热路径 | `AccelReadout`（W 常驻设备，接口对齐 `Readout`：forward / learn_softmax / W / n_synapses / W_cpu / load_W） | 昇腾 / CUDA / ROCm / DirectML / CPU（torch） |
| M6 读出回落路径 | numba fp32 融合核 | CPU |
| 多卡 | `phdnet/backends/multi_device.py`：读出**列并行**（§五） | 多设备 |

读出占 1B 端到端 ~89%（V×H = 73,958×3,072 fp32 ≈ 867 MiB），是唯一值得上设备的部件；
其余部件迁移会丢机制或无收益（§九诚实边界）。

**接入开关**：`PHDNetConfig.accel_readout`，默认 `auto`；CLI `--accel auto|cpu|npu|cuda|rocm|dml`
（train 与 infer 均支持）。auto = 探测到加速器即择优使用（昇腾 → ROCm → CUDA → DirectML）；
**无加速器回落 numba CPU 原路径，逐位不变**；构造失败亦回落并记录 `_accel_fallback_reason`
（如实报告，不静默）。训练启动日志打印 `[读出] 后端=accel:<设备>` 或 `numba-cpu`。

精度：softmax 主回路恒 fp32（与 P9 协议一致）；fp16/bf16 可用但有机制性代价（§八）。

## 三、启用方法（按平台）

```bash
# 通用依赖
pip install numpy numba pyarrow psutil

# CUDA（NVIDIA）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# ROCm（AMD）
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# CANN·昇腾 NPU（华为）—— torch_npu 版本须与 torch 严格同版本（如 2.5.1 ↔ 2.5.1）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu                       # 或从 CANN 配套源安装

# 生产训练/推理（读出自动上设备）
python train_1b/train.py --accel auto
python train_1b/infer.py --accel auto

# 工具链
python tools/accel_doctor.py                # 诊断：设备真的能算吗 + 性能对照（§四）
python tools/bench_accel.py                 # 读出基准，默认 V=73,958（1B 真实词表），报等效带宽（§七）
python tools/backend_probe.py               # 设备探针
```

## 四、设备探针与能力报告

「探测到」不等于「能用」。三层递进回答这个问题：

| 工具 / 机制 | 回答的问题 |
|---|---|
| `torch_backend.probe_devices()` | 各平台 ok/count/name（探测层） |
| `multi_device.capability_report()` | 后端能力矩阵：numba 生产路径只能跑 CPU；torch 读出可跑全平台 + 行动建议 |
| `tools/accel_doctor.py` | 试分配 + 一次前向 matvec，验证设备**真的能算**；性能对照（设备 / torch-CPU / numba 基线 ms/token）；环境矩阵（torch_npu 必须与 torch 严格同版本，不匹配会出现「探测到但算子不可用」） |

```bash
python tools/accel_doctor.py                          # 默认 1B 读出规模（867 MiB）
python tools/accel_doctor.py --V 20000 --H 3072       # 显存不足时缩小规模
python tools/accel_doctor.py --device npu             # 强制指定设备
python tools/accel_doctor.py --no-bench               # 只诊断不跑基准
```

训练/推理启动日志会打印**读出设备的实际解析结果**（`[能力] 读出设备解析（auto）= npu`），
失败时给出一行可复制的诊断命令。真正生效的后端仍以 `[读出] 后端=...` 为准。

numba 的 `@njit` 只能编译到 CPU 机器码（物理限制）：NPU 机器上生产主循环依然跑 CPU，
日志里的「numba nogil 线程 ×N」指 CPU 线程。上 NPU 的唯一路径 = `--accel auto` 让
读出走 AccelReadout。

## 五、多卡：读出列并行（P14）

**为什么不是 DDP / DataParallel**：PHD-Net 是逐 token 事件驱动的稀疏网络——无 batch 维、
无注意力，全局状态（WM 情景缓冲 / STDP 印迹 / LTM）跨样本连续携带。DDP 需要可分梯度与
batch 归约、DataParallel 需要 batch 轴可切，本网络都没有。因此多卡走**模型并行**，且只切
真正可分的部分：

| 部件 | 可分性 | 方案 | 状态 |
|---|---|---|---|
| 读出 W ∈ R^{V×H} | **完全可分**（逐行 dot + 逐行外积更新） | 按词表行（V 维）列并行到各卡 | ✅ 已实现 + 同设备逐位验证 |
| PC 栈（稀疏 CSR 主循环） | 事件驱动、隐式耦合 | 不切 | 上游单卡（主设备） |
| STDP 印迹 / LTM 2^24 | CSR 天然可按神经元区间切 | `shard_ranges` 计划 + 跨卡共激活对通信 | 计划层（1B 真机待验） |
| 分词 / SDR 编码 | CPU numpy | 不切 | CPU |

**公共 API**（`phdnet/backends/multi_device.py`）：

- `resolve_devices("auto" | "cuda:0,1" | "npu:0 1" | "cpu", max_devices=N)`：auto 择优
  （昇腾 → ROCm → CUDA → DirectML → CPU）并取全部同型号设备；不可用设备诚实报错
  （`allow_fallback=True` 才回退）；
- `probe_multi()`：多卡视角探针；
- `shard_ranges(n_items, n_devices)`：均衡切分（读出 V / LTM 神经元共用）；
- `plan_parallel(devices, V, H)`：自动选型 + 诚实预期（策略、通信量、未并行部分）；
- `configure_host_threads()`：多卡时 OMP/MKL 线程收敛为 1；
- `MultiDeviceReadout(W, devices, dtype)`：列并行读出（forward / learn_softmax /
  `to_numpy` / `load_rows` 检查点往返）。

**接入点**：训练/推理 `--devices auto`（单设备 = 原生路径，零行为变更；多卡自动启用
读出列并行 + host 线程收敛）。生产推理为 numba CPU 单路 + 可选 accel 读出；
多卡读出并行与读出设备选择（`--accel`）正交。

**通信量**：每步两次小向量（全局 logits y 下发 + dp 分片上行），1B 词表约
289 KiB×2/步，不构成瓶颈。softmax 主回路在主卡 fp32。1B 预设读出占 ~89%
→ N 卡上限 ≈ 1/(1-0.89+0.89/N)（N=4 → ≈3.3×，**仅为读出部分的线性加速，非全模型加速**）。

**验证**（`tests/verifiers/verify_multi_device.py`，全 PASS）：设备解析 / 分片均衡与余数
（10/4 → 3,3,2,2）/ 单设备 ≡ AccelReadout（前向 / NLL / 更新后 W 逐位 max|Δ|=0）/
4 路分片 ≡ 单设备（逐位）+ 端到端 PPL 一致 / 计划报告字段 / host 线程收敛。
诚实边界：本机无加速器，跨**不同**卡的路径仅做结构验证（容差判据，非逐位）；
同设备分片为逐位一致。多卡真机回归待硬件到位后执行。

## 六、NPU 性能根因修复（P28，2026-09-29）

用户在昇腾上报告性能异常。逐行审计 `AccelReadout` 每 token 的设备交互后，定位到
**带宽超支 40% + 3 次硬同步**。

### 每步设备流量账（V=73,958，H=3,072，W = 867 MiB fp32）

| 项 | 修复前 | 修复后 |
|---|---|---|
| `W @ h` 读 W | 867 MiB | 867 MiB |
| `torch.outer(dp, h)` **物化临时张量** | **867 MiB** | 0（消除） |
| `W.add_(outer)` 读 W + 读临时 + 写 W | 2,600 MiB | 1,733 MiB（`addmm_` 读 W + 写 W） |
| H2D h（×2）+ target + **y D2H→H2D 回传** | 0.6 MiB | 0.3 MiB |
| **合计** | **4,334 MiB/步** | **2,600 MiB/步（−40.0%）** |

理论下限（读 W 一次 + 读写 W 各一次）= 2,600 MiB → **修复后正好触底**。
按昇腾 ~1.6 TB/s HBM 估，2.9 ms → 1.7 ms/步。

### 三处修正

1. **`torch.outer` → `W.addmm_(dp.reshape(-1,1), h.reshape(1,-1), alpha=-eta)`**
   rank-1 AXPY，不物化与 W 同尺寸的临时张量（1B 档 867 MiB），同时消除每步一次大块
   设备分配/释放对 caching allocator 的压力。**数值逐位相同**（实测 max|ΔW| = 0.000e+00）。
2. **设备侧 (h, y) 缓存**：`forward` 缓存设备张量，紧随其后的 `learn_softmax` 直接复用
   → 省掉「y D2H 289 KiB → H2D 传回」与 h 的重复上传。
3. **硬同步 3 → 1**：`correct` 改在主机侧算（target 本来就在主机），nll 合并成唯一一次 `.item()`。

### 实测

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 设备流量/步 | 4,334 MiB | 2,600 MiB（**−40%**） |
| 墙钟（CPU 参考，不外推 NPU） | 456.75 ms | **220.39 ms（2.07×）** |
| 硬同步/步 | 3 | 1 |

复现：`python tests/verifiers/bench_accel_path.py`

### 尚未处理（已知下一步）

- `PHDNet.step` 里 `y = self.readout(h)` 仍会把 y 拉回主机（`forward` 里的 `.cpu()`），
  而训练热路径只消费 `d["nll"]` → 一次可省的同步 + 289 KiB D2H。彻底消除需要
  `model.step` 走 `forward_dev` 并把返回 dict 里的 `y` 惰性化，涉及核心热路径与
  infer 的接口约定，**本轮未动**（收益约 3%，风险高于收益）。
- `AccelReadout.learn_softmax` 收了 `accumulate` 参数但未使用（minibatch 在加速后端被
  静默忽略）。当前 `minibatch_size` 默认 1 → 无行为差异，但启用前必须实现或显式拒绝。

## 七、基准工具的三个 bug：为什么现在的数字可信（P29，2026-09-29）

P29 之前，`tools/bench_accel.py` + `torch_backend.bench_readout` 本身有缺陷，
**它给出的所有加速器数字都不可信**。修复前先看旧账，理解为什么今天能信新数字：

1. **异步设备上从不 synchronize**：计时区间前后只调 `time.perf_counter()`，
   NPU/CUDA 上测到的是「kernel 提交耗时」而非真实耗时（队列还没执行完就返回），
   数字偏小且随负载剧烈波动。→ 已修：`_sync_device()`。
2. **`dtype` 从未传入 + 成功即 break**：`bench_readout()` 写死 fp32，调用方循环
   fp32/fp16/bf16 后**第一次成功就 break** → fp16/bf16 档**从来没有被真测过**。
   项目里「fp16 慢 17×」的负面数据只来自 CPU numba 码本路径（位算法量化，非原生
   半精度），不能外推到 NPU。→ 已修：dtype 参数真正传入。
3. **测的不是生产对象**：用的是旧 `TorchReadout`，而生产走 `AccelReadout`
   （P28 的 AXPY + 设备侧缓存）。→ 已修：默认 `AccelReadout`（P30 删除旧对象后
   此类「测错对象」bug 不可能再发生）。

修复后额外报告**等效带宽 GB/s**：读出是 GEMV，受带宽限制，带宽是唯一可跨平台
比较的指标。默认 V=73,958（1B 真实词表）。

```bash
python tools/bench_accel.py                 # 默认 V=73,958
python tools/bench_accel.py --steps 50
```

输出形如 `npu fp32: … 合计 X ms/token | W 867 MiB | 等效 Y GB/s`，
三档（fp32/fp16/bf16）横向对比即可判断 fp32 是不是主要损失来源。

**同时看训练日志的读出占比**（`train_1b/train.py` 已打印）：

```
token 100  … | 读出 1.234 ms/tok（accel:npu@npu, 占 12.3%）
```

占比 < 50% 即证明**瓶颈已转移到 CPU 侧**（PC 栈 / STDP / big_ltm 2^24 的
537 MB 随机访问），此时再优化读出精度收益有限，应转攻 CPU 侧。

## 八、NPU 利用率的诚实口径与半精度的机制性代价

**~50% 利用率是结构性上限，不是 bug**。读出是 GEMV，每步读满 W（fp32 867 MiB），
受**带宽**限制而非算力 → 「理论 100% 利用率」不成立，带宽饱和即到顶。
提高利用率的唯一途径是批处理——但 PHD-Net 是逐 token 状态连续语义（WM/STDP/LTM
跨样本携带），批处理会破坏该语义，**不可行**。因此对加速器读出的合理预期就是
带宽打满，而不是算力打满。

**降精度（fp16/bf16）可把流量减半（2,600 → 1,300 MiB/步），但有机制性代价**：

- 非目标行的更新量 ≈ 1e-6（|dp| ≈ 1/V），而 fp16 半 ULP ≈ 1.5e-5、bf16 ≈ 2e-4
  → **round-to-nearest 下被完全吞掉**；
- 即半精度读出退化为「只提升目标行、从不衰减其他行」的**纯 Hebbian 规则**，
  与 fp32 的 `p − t` 不是同一个学习规则；
- 现有 PPL 表（fp16 −0.17%）是 CPU 码本路径、在 `eta=0.05` 下测的；现行默认
  `eta=0.15` 更激进，**需在真机重测 PPL 后才能决策**（用 §七的 bench_accel 三档对比）。

## 九、诚实边界：哪些没迁移及原因

| 部件 | 状态 | 原因 |
|---|---|---|
| 分词/词表扫描 | 留 CPU（numba nogil 多核） | 变长字符串 + 状态化锚点链，设备侧不划算（整段上传 + 变长输出）；实测训练循环内占比 ~0.0% |
| 词涌现统计 | 留 CPU（已 nogil 化，7.00×，P22） | 迁设备需重写 + 容差判据；核内多线程实测只快 1.16×（内存带宽饱和），CPU 已非瓶颈 |
| PC 栈 / STDP / LTM | 留 CPU（生产主循环） | 事件驱动稀疏 + 在线 CSR 生长 + big_ltm 2^24；整体迁移需重写全部算子且 historically 平行实现赶不上机制数（P30 教训）。若读出占比 < 50%，优化此处收益更大 |
| fp8 | 代码路径就绪 | 需 CUDA ≥ 8.9 真机（Ada/Hopper）+ torch ≥ 2.1；CPU torch 无原生 fp8 |
| fp4 | **已禁用**（fhz 2026-09-27） | e2m1 分辨率不足，4K 口径 ppl +5.88%；MX 块缩放立项后重新启用 |
| 跨设备逐位等价 | 不宣称 | 跨库归约顺序不同，判据为容差一致（`atol/rtol`）；同设备分片路径可逐位 |
| 无硬件验证 | 结构验证 | CUDA/ROCm/NPU 真机回归须在对应硬件或 CI runner 上执行（`tools/accel_doctor.py` + `tools/bench_accel.py` 即 runner 就绪入口） |

## 十、SFT / RL / 多卡（与读出设备选择正交）

| 能力 | 入口 | 说明 |
|---|---|---|
| SFT | `--assistant-marker`（回复掩码）+ `--init-from`（两阶段） | 与 `--accel` / `--devices` 正交 |
| RL | `tools/train_rl.py` + `phdnet/rl.py`（REINFORCE，零新增算子） | `verify_rl` 全 PASS |
| 多卡 | `--devices auto`（读出列并行） | 见 §五 |

## 十一、旧 torch 栈删除记录（P30，2026-09-29，fhz「旧的可以删除」）

以下对象已全部删除（约 1,400 行），此处仅作历史记录，**不在任何当前路径中**：

- 已删除：完整 torch 版 PHD-Net 系列（torch_lm 下的全部分词器外模型类）、
  `torch_backend` 下的旧读出/STDP 类与其自检入口、对应训练与验证脚本、
  `phdnet/torch_backend.py` 旧位置 shim。
- 删除理由：① 权重与生产 npz 检查点不通用；② 缺 7 项机制（big_ltm 等显式
  NotImplementedError）；③ 从未进入生产路径。
- 一句话教训：**不要维护一套永远赶不上 numba 主循环机制数的平行实现**。
- 保留（生产依赖）：`torch_lm.resolve_device`（auto 择优：昇腾 → ROCm → CUDA →
  DirectML → CPU）、`torch_backend.probe_devices`、`torch_backend.bench_readout`
  （P29 修正版）、`multi_device` 全套、`AccelReadout`。

此后读出加速只有一个实现（AccelReadout），基准只测生产对象。

---

## 2026-09-29 增补（昇腾实测后的修复）

| 项 | 问题 | 修复 |
|---|---|---|
| **P55 崩溃** | NPU 首次真跑生产训练即崩：`FakeTensor - None` @ `accel_readout.py:159`。P45 的「`target_idx` 路径就地改 p」只实现在 **eager 分支**，`torch.compile` 融合核漏改（`t32=None`）；既有验证只走 target 数组路径 | 融合核逐行镜像 eager（`p−onehot` ≡ `p[c]−= 1`，数值等价）；新增 `verify_accel_readout_p55.py` 2×2 矩阵 9/9 |
| **inductor 惰性编译** | `torch.compile()` 构造期不编译，**首次调用**才编译 → 编译失败会崩生产 | 首次失败**永久回落 eager + 告警**（P19「回落 + 记原因」）；`--torch-compile` 现**默认关** |
| **P58 同步/流水** | 每步 `.item()`（`nll_sync_every` 默认为 1）把 NPU 延迟全额暴露给 CPU；h 的 pageable H2D 阻塞 | `--nll-sync-every` 默认 **8**；`_staged_to_dev` 用 pinned 暂存 + `non_blocking`（4 槽 + Event 覆写保护） |
| **P57/P63 遥测** | `Telemetry()` 无参构造 → 加速器分支永不触发；`npu-smi` 不在 PATH；解析依赖该机没有的 Bus-Id 列 | device 自动探测 + 三级定位 `npu-smi` + **按表头定位列**解析；AI Core% 改走 npu-smi（`torch.npu.utilization()` 会同步设备流） |
| **P71 绑核** | numba/OpenMP 线程无亲和性 → 核间漂移（CS/s 250 万+） | `--omp-proc-bind`（默认开）设 `OMP_PROC_BIND=close` / `OMP_PLACES=cores`；`--numba-threads` 默认 8（P22：核内 1→6 线程仅 1.16×） |
| **P73 BLAS 线程** | `*_NUM_THREADS` 从未限 → OpenBLAS 拉 191 线程跑小 sgemv | 在 **`import numpy` 之前**设四个环境变量为 8（仓库早有 `configure_host_threads()` 但无生产调用点） |
| **bf16 读出的机制代价** | 非目标行更新（\|dp\|≈1e-6）被 bf16 半 ULP（≈2e-4）舍掉 → 学习退化为纯 Hebbian，PPL 震荡不降 | 建议 `--readout-dtype fp32` + `--ckpt-dtype bf16`（**存储仍半精度**，学习规则精确） |
