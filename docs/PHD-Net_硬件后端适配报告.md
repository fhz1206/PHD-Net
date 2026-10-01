# PHD-Net 硬件后端适配报告

> **适用范围**：加速后端的**能力矩阵、分层归属、迁移路径、昇腾真机踩坑、配置口径**。
> **不写**：训练怎么跑（见 `README.md` 与《扩展指南》）、机制细节（见《架构设计》）、
> 任何性能数字（**唯一出处是 `PHD-Net_性能评估与迭代方案.md`**，本文只引用不复制）。
> **数据截止**：2026-10-01（对应 P92/P100：加速器全面禁用 FP8 张量；**int8 量化码本解禁**（910B 有 INT8 Cube）；读出默认 **int8**（fhz 2026-10-01），fp16 可选）。
> ⚠ 数据点：Ascend **910B4**（Atlas A2）——该芯片**没有 FP8 单元**（910C 部分支持、950 原生 MXFP8/MXFP4），所以 fp8/fp4 的 ERR01007 是硬件限制而非软件 bug。
> **子包索引**：`phdnet/backends/README.md`。**缺陷详情**：`BUGS.md`。

---

## 一、结论速览

**架构决策（P30 定稿，至今有效）：加速的正确姿势是只迁移读出（M6），不维护平行实现。**

生产路径 = numba CPU 主循环（机制完整）+ `phdnet/backends/accel_readout.py::AccelReadout`
（torch 设备上的读出，W 常驻设备）。理由：读出是唯一的大矩阵部件，其余部件迁移会**丢机制**
或**无收益**（逐条见 §七 诚实边界）。

**适配原则**：同一份算子代码，只改 device/dtype，**不改算法语义**。

**等价性判据**：跨实现 / 跨设备只宣称**容差一致**（`atol/rtol`）—— 归约顺序不同，逐位等价在跨设备
场景不成立；**同设备**的分片路径可逐位。⚠ 容差判据本身需要归因：P80 把 nll 换成 `F.cross_entropy`
（数学等价、浮点顺序不同，相对差 3.46e-5 vs 判据 1e-6）后 `verify_accel_readout` 长期报 1 FAIL，
`git stash` 双向确认不是当次回归 —— **门禁红了必须归因，不能等它变绿**。

**核心纪律（P19 起确立，四条）**：

| 纪律 | 含义 | 违反后果 |
|---|---|---|
| **回落 + 记原因** | 加速器不可用/构造失败 → 回落 numba CPU 原路径（逐位不变），并把原因写在 `_accel_fallback_reason` 上，启动日志打印 | 静默回落 = 无法判断「加速是否真的生效」（P88 的 507033 正是靠这条变成非事故） |
| **完整调用面对齐**（P23） | `AccelReadout` 必须对齐 `Readout` 的**全部**访问面（`W`/`learn_softmax`/`learn`/`__call__`/`n_synapses`/`conn_k`/`hidden`/`stats()`/`W_cpu`/`load_W`） | 缺一项就在**生产保存检查点**时才崩 |
| **未实现配置要 fail-fast** | 未实现的配置组合**显式回落并记原因**，绝不「能跑但语义不同」 | 静默换规则比回落更糟 |
| **`torch.compile` 惰性编译**（P58） | 构造期不编译、**首次调用**才编译 → 首次失败**永久回落 eager + 告警**；`--torch-compile` 默认关 | 编译失败崩生产训练 |

---

## 二、后端矩阵

| 平台 | 读出加速（`AccelReadout`） | 验证状态 | 说明 |
|---|---|---|---|
| **numba CPU**（生产主力） | 主循环全机制（PC 栈 / STDP / LTM / 分词） | ✅ 实测通过 | `@njit` 只能编译到 CPU 机器码 —— **物理限制**，加速器机器上主循环仍跑 CPU。**唯一保留 fp8/fp4 量化码本的路径**（P92） |
| **CANN·昇腾 NPU** | ✅ 代码就绪 | ✅ **真机实测**（P55–P92 系列） | 需 `torch_npu`，版本须与 torch **严格同版本**；import 即注册 `npu` 设备。⚠ **fp8/fp4 全禁**（§四） |
| **CUDA**（NVIDIA） | ✅ 代码就绪 | ⚠ 结构验证（无硬件） | 算子代码与 CANN 路径相同 |
| **ROCm**（AMD） | ✅ 代码就绪 | ⚠ 结构验证（无硬件） | 走 HIP 化接口（`torch.version.hip` 判别） |
| **DirectML** | ✅ 代码就绪 | ⚠ 结构验证 | torch-directml |
| **CPU（torch）** | ✅ 可用 | ✅ 实测通过 | ⚠ **fp8 不可用**：能分配 fp8 张量但**无 fp8 addmv**（P85 探针实测） |

**为什么只有昇腾是真机结论**：CUDA / ROCm / DirectML 三行都只有结构验证。引用这三行的任何
性能陈述都必须标注「未在真机验证」。

---

## 三、分层归属：哪一层在哪个设备

| 层 | 实现 | 设备 |
|---|---|---|
| 分词 / 词表扫描 | numba nogil 多核 | CPU |
| M1–M5：PC 栈 / STDP 印迹 / LTM（含 `big_ltm`）/ 调制器 | numpy/numba **生产主循环**，机制完整 | CPU |
| M6 读出热路径 | `AccelReadout`（W 常驻设备） | 昇腾 / CUDA / ROCm / DirectML / CPU(torch) |
| M6 读出回落路径 | numba 融合核 | CPU |
| 多卡 | `phdnet/backends/multi_device.py`：读出**列并行**（§六） | 多设备 |

**选路开关**：`PHDNetConfig.accel_readout`，默认 `auto`；CLI `--accel auto|cpu|npu|cuda|rocm|dml`
（train 与 infer 均支持）。auto 择优顺序 = **昇腾 → ROCm → CUDA → DirectML**；无加速器回落
numba CPU 原路径（逐位不变）。

**运行期自证**：启动日志打印 `[读出] 后端=accel:<设备>` 或 `numba-cpu(回落)`；回落时同行打印
`fallback reason: <原因>`。**以这一行为准，不以「探测到设备」为准。**
⚠ 这条纪律本可更早发现 BUGS B3（fp8 能力表遗漏导致默认 fp8 被静默回落，整轮改造在生产等于没生效）。

---

## 四、精度与后端（P92 定稿）

现行口径（`--readout-dtype`，**默认 `fp16`**）：

| 项 | 口径 |
|---|---|
| **加速后端可用精度** | **只有 fp32 / fp16 / bf16**（`_DT` 就这三项；fp8/fp4 连构造都不允许，立即报 unsupported dtype） |
| fp8 / fp4 | **仅 numba CPU 路径**保留 P9/P12 的位算法量化码本 → `--accel cpu --readout-dtype fp8` 仍可用。这是「除 CPU 外禁用」的那一半 |
| 默认为何 fp16 而非 bf16 | bf16 半 ULP ≈ 2e-4 ≫ 非目标行更新 \|dp\| ≈ 1e-6 → 更新被舍入，**训练退化为纯 Hebbian**（PPL 震荡的根因）。fp16 原生支持、成熟 GEMV、同样带宽、**保住微小更新** |
| 检查点 | `to_numpy` 对 bf16/fp8 存**位模式**（uint16 / uint8），加载侧按 `ckpt_dtype` 解码闭环（无损、不膨胀） |
| 其它精度 | M1 编码器默认 **fp64**（昇腾 aarch64 numpy GEMV 病态 → 平台自适应走自写 numba 核；x86 走 BLAS），`--encoder-dtype` 可切；分词 / onehot 缓冲 fp16（值域 0/1，无损） |
| 未实现的结构化配置 | `readout_hidden>0`（两级读出）、`readout_conn_k>0`（稀疏读出）在加速后端**未实现 → fail-fast 回落** |

**⚠ 昇腾 fp8 实测结论（P92 的证据，写在代码里）**：Ascend910B4 / CANN 8.5 / torch_npu 2.9 上
`float8_e4m3fn` 与 `float8_e5m2` 的 **create / cast / matmul 全部 ERR01007**；`torch.ops` 暴露
**零个 fp8 算子**；CPU torch 可分配 fp8 但无 fp8 addmv。→ 这是「**芯片有 fp8，框架没暴露**」，
不是芯片缺陷。`tools/probe_fp8.py` 保留了这个判定，CANN 跟进后一处改动即可翻回（探针三档结论：
标准 dtype 可用 / 有专用算子走 `torch.ops` / 保持拒绝）。**⚠ 能力表纪律**：
`accel_readout.py::_unsupported_reason()` 是「未实现项」黑名单 —— P84 实现了 fp8 并设为默认，
**忘了从这张表删掉** → 生产日志一直 `numba-cpu(回落)`，**整轮改造没生效且不报任何错**
（BUGS B3，本项目最严重的一条）。加 dtype 必须同步 `_DT` + 这张表。

各精度的 PPL / 带宽数字见《性能评估与迭代方案》，本文不复制。

---

## 五、配置口径（现行默认值）

| 开关 | 默认 | 来源 / 理由 |
|---|---|---|
| `--readout-dtype` | **fp16** | P85：昇腾与 CPU 都不支持 fp8（ERR01007 / addmv 无实现）；bf16 会让学习退化 |
| `--fp8-refresh` | 8 | ⚠ 仅 numba CPU 路径有意义（加速后端已禁 fp8） |
| `--nll-sync-every` | **8** | P62：默认 1 时每步 `.item()` 把设备延迟全额暴露给 CPU，「CPU 忙、NPU 空闲」就是这么来的 |
| `--torch-compile` | **默认关** | P58：inductor 在服务器不稳（调试噪声、编译时长不可控）；P45 的 +15% 仍可用 flag 复现 |
| `--numba-threads` | **8** | P62：191 核上 prange 拉满线程 → 进程只用 1.3 核、CS/s 250 万+；P22 实测 1→6 线程仅 1.16×（带宽饱和） |
| `--omp-proc-bind` | **默认开** | P71：设 `OMP_PROC_BIND=close`。⚠ **不设 `OMP_PLACES`** —— P74 实测 `OMP_PLACES=cores` 在 191 核上让 M2 再慢 8–13×，**已回滚，只留 PROC_BIND** |
| BLAS 线程 | 随 `--numba-threads` | P73：四个 `*_NUM_THREADS` 必须在 **`import numpy` 之前**设（OpenBLAS 初始化后再设无效） |
| `--m2-kernel` | **plain** | P76：昇腾上 fused 退化 3–4×。x86 仍快 2.1× → 跨平台运行应按机器挑 |
| `--encoder-dtype` | **fp64** | P77：aarch64 numpy GEMV 病态（fp32/fp64 都慢，是平台不是 dtype）；x86 走 BLAS |
| `--accel` | auto | 昇腾 → ROCm → CUDA → DirectML → CPU(numba) |
| `--ckpt-every` | 50000 | P83：异步保存（主线程 `.copy()` 快照 + 后台单写线程） |

---

## 六、多卡：读出列并行（P14）

**为什么不是 DDP / DataParallel**：PHD-Net 是逐 token 事件驱动的稀疏网络 —— **无 batch 维、无注意力、
无梯度**，全局状态（WM 情景缓冲 / STDP 印迹 / LTM）跨样本连续携带。DDP 需要可分梯度与 batch 归约，
DataParallel 需要 batch 轴可切，本网络两者都没有。因此走**模型并行**，且只切真正可分的部分：
读出 W ∈ R^(V×H) **完全可分**（逐行 dot + 逐行外积更新）→ 按词表行（V 维）列并行，✅ 已实现 +
同设备逐位验证；PC 栈（事件驱动、隐式耦合）与分词 / SDR 编码（CPU numpy）不切，上游单卡；
STDP 印迹 / LTM 2²⁴ 的 CSR 天然可按神经元区间切 → `shard_ranges` 计划 + 跨卡共激活对通信，
属**计划层**（1B 真机待验）。

**公共 API**（`phdnet/backends/multi_device.py`）：`resolve_devices()`（不可用设备**诚实报错**，
`allow_fallback=True` 才回退）、`probe_multi()`、`shard_ranges()`、`plan_parallel()`、
`capability_report()`、`configure_host_threads()`、`MultiDeviceReadout`（列并行 + 检查点往返）。
**接入点** `--devices auto`：单设备 = 原生路径、零行为变更；多卡读出并行与读出设备选择（`--accel`）**正交**。
**通信量**：每步两次小向量（全局 logits 下发 + dp 分片上行），1B 词表（V=73,958）约 289 KiB × 2 / 步，
不构成瓶颈。softmax 主回路在主卡。

**⚠ 诚实边界**：跨**不同**卡的路径只做结构验证（容差判据，非逐位）；同设备分片为逐位一致。
**多卡真机回归待硬件到位**。

---

## 七、昇腾真机七条坑（P55–P92）

全部是「第一次真机跑才暴露」，按发生顺序：

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| **P55** | NPU 首次真跑生产训练即崩：`FakeTensor - None` @ `accel_readout.py` | 「`target_idx` 路径就地改 p」只实现在 **eager 分支**，`torch.compile` 融合核漏改（`t32=None`）；既有验证只走 target 数组路径 | 融合核逐行镜像 eager（`p − onehot` ≡ `p[c] −= 1`，数值等价）；新增 `verify_accel_readout_p55.py` 2×2 矩阵（9/9） |
| **P58** | inductor 惰性编译 → 编译失败崩生产 | `torch.compile()` **构造期只是包装、不会失败**，**首次调用**才编译；回落写在了错误的层 | 首次执行失败 → **永久回落 eager + 告警**；`--torch-compile` 默认关。⚠ 附带：遥测自己同步设备流（`torch.npu.utilization()`），AI Core% 改走 `npu-smi` |
| **P57** | NPU / HBM 遥测恒 `--` | `Telemetry()` **无参构造** → `device=""` → 加速器分支永不触发（P41 遗留，P19 搬上 NPU 后没人回补） | 自动探测 device（`torch.npu` → `torch.cuda`，显式传入仍优先）；首次失败打印原因（原 `except: pass` 全吞）；`TeeLogger` 接管 `sys.stderr` |
| **P63 / P71** | AI Core% 恒 `--` | ① 训练**直接 `python train.py` 启动**、没 source Ascend `set_env.sh` → `shutil.which("npu-smi")` 落空；② 解析器找「含 `0x` 总线号的数据行」，**该机没有这一列** → 所有行被跳过 | 三级定位 `$NPU_SMI_PATH` → PATH → 常见安装路径；改为**按表头定位列**（找 `AICore` 表头 → 推出列序 → 按列号取值），**与型号无关**；失败时打印原始输出前 14 行一次 |
| **P88** | NPU 设备启动失败 `507033 / Failed to start the device` | **设备层**问题，最常见是**上一次训练进程未退出、仍占着设备**。回落机制正常工作（`numba-cpu(回落) | fallback reason: …`，没崩） | 训练启动期做一次**真实的设备张量探测**（建 net 之前），把 CANN 天书翻译成三条可操作检查（残留进程 / 容器设备映射 / 设备占满）+ `--accel cpu` 逃生口。**未修** —— 取决于服务器环境 |
| **P71 / P73 / P74** | CS/s 250 万–600 万；只用 1.1–3.2 核（191 核）；加 `OMP_PLACES` 后 M1 慢 70×、M2 慢 8–13× | ① numba/OpenMP 线程**无核心亲和性** → 核间漂移；② `*_NUM_THREADS` **从未限上限** → OpenBLAS 拉 **191 线程**跑 1024×2048 的小 sgemv（仓库早有 `configure_host_threads()` 但**无生产调用点** = 修了一半）；③ `OMP_PLACES=cores` 在 191 核上建 191 项 place 表，每次线程池同步都要遍历 | `--omp-proc-bind`（默认开）只设 `OMP_PROC_BIND=close`、**不设 PLACES**；`--numba-threads` 默认 8 且 **BLAS 变量必须在 `import numpy` 之前**设。⚠ P74 是**假设被实测否掉**的记录，保留以防有人再把 `OMP_PLACES` 加回来 |
| **P81 / P92** | bf16 检查点保存崩 `unsupported ScalarType BFloat16`；fp8 在昇腾全链路 ERR01007 | numpy 无原生 bf16；**芯片有 fp8 但 torch_npu 没暴露标准 dtype 路径** | `to_numpy` 存 uint16 位模式，与加载侧 `ckpt_dtype` 解码闭环（无损）；fp8/fp4 **从加速后端整体删除**，`_DT` 收缩为 `{fp32, fp16, bf16}`，**仅 numba CPU 路径保留量化码本**。⚠ P85 第一版探针只测**张量创建**（CPU 能分配）→ 假通过，炸在 addmm；**能力探测必须用真实算子** |

**平台差异铁律**（三次反噬）：

| 改动 | x86 | 昇腾 aarch64 |
|---|---|---|
| M1 编码器 fp32 | 快 **1.80×** | 慢 **约 70×** |
| M2 融合核 | 快 1.15–2.16× | 慢 **3–8×** |
| `OMP_PLACES=cores` | 无感 | 慢 **8×**、CS/s 升到 650 万 |

→ **x86 的性能结论不构成昇腾的证据**；跨平台项目里任何优化必须在目标机器复测。
`--m2-kernel` / `--encoder-dtype` 就是这条铁律的产物；`tools/bench_local.py` 明确**不产出文档数字**。

---

## 八、迁移路径与启用条件

### 8.1 已完成的迁移

| 节点 | 内容 |
|---|---|
| P10 / P14 | 基础层 `backends/` 目录化与统一探针 / 多卡读出列并行 |
| P19 / P23 | **只迁移读出**（`AccelReadout`），确立「回落 + 记原因」 / **完整调用面对齐**（根治三次崩溃） |
| P28 | NPU 带宽根因：`torch.outer` 物化同尺寸临时张量 → `W.addmm_` rank-1 AXPY；设备侧 (h, y) 缓存；硬同步 3 → 1。**数值逐位相同** |
| P29 / P30 | 基准工具三 bug 修正（**修复前的加速器数字一律不可信**）/ **旧 torch 栈整体删除**（约 1,400 行）—— 教训：**不要维护一套永远赶不上 numba 主循环机制数的平行实现** |
| P36–P52 | 训练步设备直通（`forward_dev` + `target_idx`）、融合步核、`_is_accel` 显式标识（**不要用 `hasattr` 猜** —— numba `Readout` 也有同名旧接口，曾导致误走设备路径崩溃） |
| P84 / P92 | fp8 forward 副本 + fp16 更新主副本 → 因昇腾不支持，**fp8/fp4 从加速后端整体禁用** |

### 8.2 启用条件（各后端跑起来的前提）

| 后端 | 启用条件 | 验证入口 |
|---|---|---|
| numba CPU | 默认即主路径 | `run_tests.py fast` + 18 个 verifier |
| 昇腾 NPU | `torch_npu` 与 torch **严格同版本**；启动前 source Ascend `set_env.sh`（否则 `npu-smi` 不在 PATH）；设备未被残留进程占用 | `tools/accel_doctor.py`（试分配 + 一次前向 matvec）+ `tools/bench_accel.py`（读出基准，报等效带宽 GB/s）+ 启动日志 `[读出]` 行 |
| CUDA / ROCm | 对应 torch 构建 + 硬件 | 同上（**真机回归须在对应硬件或 CI runner 上执行**） |
| DirectML | `torch-directml` | 同上（仅结构验证） |
| 多卡 | `--devices auto`；同型号 | `verify_multi_device.py`（⚠ 跨不同卡只做结构验证） |

### 8.3 下一步（尚未实施）

| 项 | 说明 |
|---|---|
| **M6 幂律稀疏化** | 架构级欠账（脑同构维度）。既省访存又补同构，方案已设计未实施 |
| 设备端稀疏张量 | 100B 容量栈上设备的前置条件，当前缺 |
| 段级流水 | M1/M2 与 M3/M4/M5 重叠，收益上限 = max(CPU 侧, 设备侧) |
| `accumulate` 参数 | `AccelReadout.learn_softmax` 收了 minibatch 梯度累积参数但**未使用**（加速后端会静默忽略）。当前默认 1 → 无行为差异，**启用前必须实现或显式拒绝** |
| 在线 CSR 镜像 | M4b 多核化的正确路径（零 gather），至今未做 |


---

## 九、诚实边界

| 部件 / 结论 | 状态 | 原因 |
|---|---|---|
| 分词 / 词表扫描 | 留 CPU（numba nogil 多核） | 变长字符串 + 状态化锚点链，设备侧不划算；实测训练循环内占比极小 |
| 词涌现统计 | 留 CPU（已 nogil 化） | 迁设备需重写 + 容差判据；核内多线程实测只快 1.16×（带宽饱和） |
| PC 栈 / STDP / LTM | 留 CPU（生产主循环） | 事件驱动稀疏 + 在线 CSR 生长 + `big_ltm` 2²⁴；整体迁移需重写全部算子，且平行实现赶不上机制数（P30 教训）。若读出占比 < 50%，优化此处收益更大 |
| 两级读出 / 稀疏读出 | **加速后端未实现 → fail-fast 回落** | 「能跑但语义不同」比回落更糟（`readout_hidden>0`、`readout_conn_k>0`） |
| 量化码本 fp8 / fp4 | **加速后端全禁，仅 numba CPU 保留** | 昇腾 ERR01007（create/cast/matmul 全失败、`torch.ops` 零算子）；CPU torch 无 addmv。是「**框架没暴露**」而非芯片缺陷 |
| 跨设备逐位等价 | **不宣称** | 跨库归约顺序不同，判据为容差一致；同设备分片可逐位 |
| CUDA / ROCm / DirectML 真机 | **结构验证** | 无硬件。真机回归须在对应硬件或 CI runner 上执行 |
| M4b imprint 多核收益 | **本机 1.0–1.1×，收益押在服务器侧** | x86 Python 在该规模已够快；预期 20 s/call → 毫秒级，**尚未经服务器复测** |
| 设备启动 507033 / fp8 检查点体积与加载耗时 | **未修**（环境）/ **未测** | 设备 retain 失败，需在服务器确认残留进程；后者是已知开放项 |

---

## 十、诊断与门禁

### 10.1 三层递进：「探测到」不等于「能用」

| 工具 / 机制 | 回答的问题 |
|---|---|
| `torch_backend.probe_devices()` | 各平台 ok / count / name（**探测层**） |
| `multi_device.capability_report()` | 后端能力矩阵：numba 生产路径只能跑 CPU；torch 读出可跑全平台 + 行动建议 |
| `tools/accel_doctor.py` | 试分配 + 一次前向 matvec，验证设备**真的能算**；性能对照；环境矩阵（`torch_npu` 必须与 torch 严格同版本，不匹配会出现「探测到但算子不可用」） |
| `tools/probe_fp8.py` | fp8 到底卡在芯片还是框架（CANN 跟进后据此翻回） |
| 启动日志 `[读出] … fallback reason: …` | **实际选中了哪个后端** —— 唯一可信的判据 |

`tools/bench_accel.py` 是**读出基准**（默认 V=73,958 / H=3,072，1B 真实词表规模），报**等效带宽
GB/s** —— 读出是 GEMV，带宽是唯一可跨平台比较的指标。P29 修正了它的三个 bug，**修复前的加速器数字
一律不可信**；此后只测生产对象 `AccelReadout`。**同时看训练日志的读出占比**
`[计时] 读出 X ms/tok（后端@设备，占 Y%）`：占比 < 50% 即证明**瓶颈已转移到 CPU 侧**，
此时再优化读出精度收益有限。

### 10.2 验证入口（18 个 verifier）

| 类别 | 脚本 |
|---|---|
| 逐位对拍 | `verify_ltm_kernels` / `verify_ltm_learn_batch` / `verify_pc_learn_fused` / `verify_readout_fused` / `verify_readout_sparse` / `verify_csr_equiv` / `verify_seg_equiv` / `verify_vocab_parallel` / `verify_parallel_consistency` / `verify_stream_tokenize` |
| 容差对拍 | `verify_accel_readout`（⚠ NLL 判据 1e-4，见 §一）/ `verify_accel_readout_p55`（2×2 矩阵）/ `verify_multi_device` / `verify_rl` |
| 检查点往返 | `verify_ckpt_roundtrip` / `verify_ms_stream`（41 用例：bf16 round-trip、语法门禁、`train.py --help` 子进程） |
| 其他 | `verify_repro_process` / `bench_accel_path` |

**零回归门槛**：`python tests/run_tests.py fast` = **9/9**。
⚠ **fast 门禁只跑 `tests/` 下的检查，从不 import `train/train.py`** —— 改训练入口后必须单独
`py_compile` + 跑一次 `--help`。

### 10.3 CI 两套的真实状态

| 项 | 状态 |
|---|---|
| **GitCode CI**（`.gitcode/workflows/ci.yml`） | ✅ 正常 |
| **GitHub Actions**（`.github/workflows/ci.yml`） | 🔴 **仍红**。已修三轮（`outputs/` 目录未建 → 合成语料 fallback → 依赖 import 损坏钉 `numpy<2.3`），但**无 admin token 拿不到日志**，只能看到匿名可读的 check-run annotations。**不要声称 CI 全绿。** |
| **Jenkins** | ⚠ **仓库没有 Jenkinsfile**（P9 时代已删）。旧文档记「CI 三套」是错的 —— **CI 只有两套** |
| CI 调用的 verifier | 两个 CI 配置**各只调 3 个** verifier（`verify_seg_equiv` / `verify_vocab_parallel` / `verify_parallel_consistency`）+ `tools/bench_accel.py`；⚠ 两边都引用了 **`verify_torch_lm.py`，而该文件已不存在** —— CI 的这条引用是悬空的 |
