# PHD-Net 硬件后端适配报告

> **适用范围**：加速后端的**能力矩阵、选路纪律、迁移路径、昇腾真机踩坑**。
> **不写**：训练怎么跑（见 `README.md` 与《扩展指南》）、机制细节（见《架构设计》）、
> 任何性能数字（**唯一出处是 `PHD-Net_性能评估与迭代方案.md`**，本文只引用不复制）。
> **数据截止**：2026-09-30（对应 P84：读出默认 fp16 forward + fp16 更新）。
> **子包索引**：`phdnet/backends/README.md`。

---

## 一、结论速览

**架构决策（P30 定稿，至今有效）：加速的正确姿势是只迁移读出（M6），不维护平行实现。**

生产路径 = numba CPU 主循环（机制完整）+ `phdnet/backends/accel_readout.py::AccelReadout`
（torch 设备上的读出，W 常驻设备）。理由：读出是唯一的大矩阵部件，其余部件迁移会**丢机制**
或**无收益**（逐条见 §八 诚实边界）。

**适配原则**：同一份算子代码，只改 device/dtype，**不改算法语义**。

**等价性判据**：跨实现 / 跨设备只宣称**容差一致**（`atol/rtol`）——归约顺序不同，逐位等价在
跨设备场景不成立；**同设备**的分片路径可逐位。

**核心纪律（P19 起确立，四条）**：

| 纪律 | 含义 | 违反后果 |
|---|---|---|
| **回落 + 记原因** | 加速器不可用/构造失败 → 回落 numba CPU 原路径（逐位不变），并把原因写在 `_accel_fallback_reason` 上，启动日志打印 | 静默回落 = 无法判断"加速是否真的生效" |
| **完整调用面对齐**（P23 教训） | `AccelReadout` 必须对齐 `Readout` 的**全部**访问面（`W` / `learn_softmax` / `learn` / `__call__` / `n_synapses` / `conn_k` / `hidden` / `stats()` / `W_cpu` / `load_W`） | 缺一项就在**生产保存检查点**时才崩 |
| **未实现配置要 fail-fast** | 未实现的配置组合**显式回落并记原因**，绝不"能跑但语义不同" | 静默换规则比回落更糟 |
| **`torch.compile` 惰性编译**（P58） | `torch.compile()` 构造期不编译、**首次调用**才编译 → 首次失败**永久回落 eager + 告警**；`--torch-compile` 默认关 | 编译失败崩生产训练 |

---

## 二、后端能力矩阵

| 平台 | 读出加速（`AccelReadout`） | 验证状态 | 说明 |
|---|---|---|---|
| **numba CPU**（生产主力） | 主循环全机制（PC 栈 / STDP / LTM / 分词） | ✅ 实测通过 | `@njit` 只能编译到 CPU 机器码——**物理限制**，加速器机器上主循环仍跑 CPU |
| **CANN·昇腾 NPU** | ✅ 代码就绪 | ✅ **真机实测**（P55–P83 系列） | 需 `torch_npu`，版本须与 torch **严格同版本**；import 即注册 `npu` 设备 |
| **CUDA**（NVIDIA） | ✅ 代码就绪 | ⚠ 结构验证（无硬件） | fp8 matmul 需 CUDA ≥ 8.9（Ada/Hopper）+ torch ≥ 2.1 |
| **ROCm**（AMD） | ✅ 代码就绪 | ⚠ 结构验证（无硬件） | 走 HIP 化接口，算子代码与 CUDA 完全相同（`torch.version.hip` 判别） |
| **DirectML** | ✅ 代码就绪 | ⚠ 结构验证 | torch-directml |
| **CPU（torch）** | ✅ 可用 | ✅ 实测通过 | **fp8 matmul 不存在 → 自动回落 fp16 主副本**（`tdtype` 同步回落） |

**为什么只有昇腾是真机结论**：CUDA / ROCm / DirectML 三行都只有结构验证。引用这三行的任何
性能陈述都必须标注"未在真机验证"。

---

## 三、分层归属：哪一层在哪个设备

| 层 | 实现 | 设备 |
|---|---|---|
| 分词 / 词表扫描 | numba nogil 多核 | CPU |
| M1–M5：PC 栈 / STDP 印迹 / LTM（含 `big_ltm`）/ 调制器 | numpy/numba **生产主循环**，机制完整 | CPU |
| M6 读出热路径 | `AccelReadout`（W 常驻设备） | 昇腾 / CUDA / ROCm / DirectML / CPU(torch) |
| M6 读出回落路径 | numba 融合核 | CPU |
| 多卡 | `phdnet/backends/multi_device.py`：读出**列并行**（§五） | 多设备 |

**选路开关**：`PHDNetConfig.accel_readout`，默认 `auto`；CLI `--accel auto|cpu|npu|cuda|rocm|dml`
（train 与 infer 均支持）。auto 择优顺序 = **昇腾 → ROCm → CUDA → DirectML**；
无加速器回落 numba CPU 原路径（逐位不变）。

**运行期自证**：训练/推理启动日志打印 `[读出] 后端=accel:<设备>` 或 `numba-cpu`；
回落时同行打印 `fallback reason: <原因>`。**以这一行为准**，不以"探测到设备"为准。

---

## 四、精度与后端（P84）

现行口径（**默认 `fp8`**，`--readout-dtype`）：

| 项 | 口径 |
|---|---|
| forward | fp8_e4m3fn **副本**（1B 档读 320 → 80 MB），每 `--fp8-refresh`（默认 8）步重建 |
| 更新 | **fp16 主副本**（保住微小更新） |
| **动机不是省访存** | 更新侧用 fp16 使总访存 **+42%**；买的是**学习精度**：bf16 半 ULP ≈ 2e-4 ≫ 非目标行更新 \|dp\| ≈ 1e-6 → 更新被舍 → 退化为纯 Hebbian（PPL 震荡的根因）。本机验证：fp16 主副本下 5722 个非目标行格点获得更新（bf16 下为 0） |
| fp8 matmul | **仅昇腾 / CUDA**；CPU torch 自动回落 fp16（`tdtype` 同步回落，否则 `addmv` dtype 不匹配） |
| 检查点 | `to_numpy` 对 bf16/fp8 存**位模式**（uint16 / uint8），加载侧按 `ckpt_dtype` 解码闭环 |
| 其它精度 | M1 编码器默认 **fp64**（昇腾 aarch64 numpy GEMV 病态 → 平台自适应走自写 numba 核；x86 走 BLAS），`--encoder-dtype` 可切；分词 / onehot 缓冲 fp16（值域 0/1，无损） |

⚠ **fp8 是默认精度，但它不是"到处都能跑"的精度**。`readout_hidden>0`（两级读出）、
`readout_conn_k>0`（稀疏读出）等结构化配置在加速后端**未实现**，会 fail-fast 回落（见 §六）。

各精度的 PPL / 带宽数字见《性能评估与迭代方案》，本文不复制。

---

## 五、多卡：读出列并行（P14）

**为什么不是 DDP / DataParallel**：PHD-Net 是逐 token 事件驱动的稀疏网络——**无 batch 维、
无注意力、无梯度**，全局状态（WM 情景缓冲 / STDP 印迹 / LTM）跨样本连续携带。DDP 需要可分
梯度与 batch 归约，DataParallel 需要 batch 轴可切，本网络两者都没有。因此走**模型并行**，
且只切真正可分的部分：

| 部件 | 可分性 | 方案 | 状态 |
|---|---|---|---|
| 读出 W ∈ R^(V×H) | **完全可分**（逐行 dot + 逐行外积更新） | 按词表行（V 维）列并行 | ✅ 已实现 + 同设备逐位验证 |
| PC 栈（稀疏 CSR 主循环） | 事件驱动、隐式耦合 | 不切 | 上游单卡（主设备） |
| STDP 印迹 / LTM 2²⁴ | CSR 天然可按神经元区间切 | `shard_ranges` 计划 + 跨卡共激活对通信 | 计划层（1B 真机待验） |
| 分词 / SDR 编码 | CPU numpy | 不切 | CPU |

**公共 API**（`phdnet/backends/multi_device.py`）：

- `resolve_devices("auto" | "cuda:0,1" | "npu:0 1" | "cpu", max_devices=N)`：auto 择优并取全部
  同型号设备；不可用设备**诚实报错**（`allow_fallback=True` 才回退）；
- `probe_multi()`：多卡视角探针；
- `shard_ranges(n_items, n_devices)`：均衡切分（读出 V / LTM 神经元共用）；
- `plan_parallel(devices, V, H)`：自动选型 + 诚实预期（策略、通信量、未并行部分）；
- `capability_report()`：后端 × 设备能力矩阵 + 行动建议；
- `configure_host_threads()`：多卡时 OMP/MKL 线程收敛；
- `MultiDeviceReadout(W, devices, dtype)`：列并行读出（`forward` / `learn_softmax` /
  `to_numpy` / `load_rows` 检查点往返）。

**接入点**：`--devices auto`。单设备 = 原生路径、零行为变更；多卡自动启用读出列并行 +
host 线程收敛。多卡读出并行与读出设备选择（`--accel`）**正交**。

**通信量**：每步两次小向量（全局 logits 下发 + dp 分片上行），1B 词表（V=73,958）约
289 KiB × 2 / 步，不构成瓶颈。softmax 主回路在主卡。

**验证**（`tests/verifiers/verify_multi_device.py`）：设备解析 / 分片均衡与余数 / 单设备 ≡
`AccelReadout`（前向 / NLL / 更新后 W）/ 分片 ≡ 单设备（逐位）+ 端到端一致性 / 计划报告字段 /
host 线程收敛。

**诚实边界**：跨**不同**卡的路径只做结构验证（容差判据，非逐位）；同设备分片为逐位一致。
多卡真机回归待硬件到位。

---

## 六、诊断：三层递进

「探测到」不等于「能用」。

| 工具 / 机制 | 回答的问题 |
|---|---|
| `torch_backend.probe_devices()` | 各平台 ok / count / name（**探测层**） |
| `multi_device.capability_report()` | 后端能力矩阵：numba 生产路径只能跑 CPU；torch 读出可跑全平台 + 行动建议 |
| `tools/accel_doctor.py` | 试分配 + 一次前向 matvec，验证设备**真的能算**；性能对照（设备 / torch-CPU / numba 基线）；环境矩阵（`torch_npu` 必须与 torch 严格同版本，不匹配会出现"探测到但算子不可用"） |

`tools/bench_accel.py` 是**读出基准**（默认 V=73,958 / H=3,072，1B 真实词表规模），报
**等效带宽 GB/s**——读出是 GEMV，带宽是唯一可跨平台比较的指标；三档精度横向对比即可判断
"是不是精度是主要损失来源"。

P29 修正了基准工具的三个 bug（异步设备无 synchronize / dtype 从未传入 + 成功即 break /
测的不是生产对象），**修复前的加速器数字一律不可信**；此后基准只测生产对象 `AccelReadout`。

**同时看训练日志的读出占比**：`[计时] 读出 X ms/tok（后端@设备，占 Y%）`。占比 < 50% 即证明
**瓶颈已转移到 CPU 侧**（PC 栈 / STDP / `big_ltm` 的随机访问），此时再优化读出精度收益有限。

---

## 七、迁移路径

### 7.1 已完成的迁移

| 节点 | 内容 |
|---|---|
| P10 | 基础层：`backends/` 目录化，统一探针 |
| P14 | 多卡：读出列并行 |
| P19 | **只迁移读出**（`AccelReadout`），确立"回落 + 记原因" |
| P23 | **完整调用面对齐**（根治三次崩溃） |
| P28 | NPU 带宽根因：`torch.outer` 物化同尺寸临时张量 → `W.addmm_` rank-1 AXPY；设备侧 (h, y) 缓存；硬同步 3 → 1。**数值逐位相同** |
| P29 | 基准工具三 bug 修正（见 §六） |
| P30 | **旧 torch 栈整体删除**（约 1,400 行）——教训：**不要维护一套永远赶不上 numba 主循环机制数的平行实现** |
| P36–P52 | 训练步设备直通（`forward_dev` + `target_idx`）、融合步核、`_is_accel` 显式标识（**不要用 `hasattr` 猜**——numba `Readout` 也有同名旧接口，曾导致误走设备路径崩溃） |
| P84 | fp8 forward 副本 + fp16 更新主副本 |

### 7.2 下一步（尚未实施）

| 项 | 说明 |
|---|---|
| **M6 幂律稀疏化** | 架构级欠账（脑同构维度，见《竞争力与脑同构性评估》）。既省访存又补同构，方案已设计未实施 |
| 设备端稀疏张量 | 100B 容量栈上设备的前置条件，当前缺 |
| 段级流水 | M1/M2 与 M3/M4/M5 重叠，收益上限 = max(CPU 侧, 设备侧)。论证见《并行与加速架构分析》 |
| `accumulate` 参数 | `AccelReadout.learn_softmax` 收了 minibatch 梯度累积参数但**未使用**（加速后端会静默忽略）。当前默认 1 → 无行为差异，**启用前必须实现或显式拒绝** |

---

## 八、昇腾真机六条坑（P55–P83）

全部是「第一次真机跑才暴露」，按发生顺序：

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| **P55** | NPU 首次真跑生产训练即崩：`FakeTensor - None` @ `accel_readout.py` | 「`target_idx` 路径就地改 p」只实现在 **eager 分支**，`torch.compile` 融合核漏改（`t32=None`）；既有验证只走 target 数组路径 | 融合核逐行镜像 eager（`p − onehot` ≡ `p[c] −= 1`，数值等价）；新增 `verify_accel_readout_p55.py` 2×2 矩阵 |
| **P57/P63** | NPU / HBM 遥测恒 `--`；AI Core% 仍 `--` | ① `Telemetry()` 无参构造 → `device=""` → 加速器分支永不触发；② `npu-smi` **不在非交互 shell 的 PATH**；③ 解析又依赖该机没有的 Bus-Id 列 | device 自动探测 + 三级定位 `npu-smi` + **按表头定位列**解析；AI Core% 改走 npu-smi（**不能用 `torch.npu.utilization()`——它同步设备流**） |
| **P58** | CPU 忙、NPU 空闲 | 每步 `.item()`（`nll_sync_every` 默认为 1）把设备延迟全额暴露给 CPU；h 的 **pageable H2D** 阻塞 | `--nll-sync-every` 默认 **8**（设备侧累积 → CPU/设备重叠）；`_staged_to_dev` 用 **pinned 暂存 + `non_blocking`**（4 槽 + Event 覆写保护） |
| **P71/P73** | CS/s 250 万–600 万；只用 1.1–3.2 核 | ① numba/OpenMP 线程**无核心亲和性** → 核间漂移；② `*_NUM_THREADS` 从未限 → OpenBLAS 拉 **191 线程**跑小 sgemv | `--omp-proc-bind`（默认开）设 `OMP_PROC_BIND=close`；`--numba-threads` 默认 8，**BLAS 线程同为 8**，且必须在 **`import numpy` 之前**设（P73：仓库早有 `configure_host_threads()` 但无生产调用点）。⚠ `OMP_PLACES=cores` 在 191 核上反而让 M2 再慢 8× → **已回滚，只留 `PROC_BIND`** |
| **P81 / P84** | bf16 检查点保存崩 `unsupported ScalarType BFloat16` | numpy 无原生 bf16 | `to_numpy` 存 uint16 **位模式**，加载侧按 `ckpt_dtype` 解码闭环（无损）；fp8 同理存 uint8 |
| **P76/P77** | M1 编码器在昇腾慢 70×（两次误判） | ① fp32 在 aarch64 病态；② 混合 dtype（fp32 权重 @ fp64 输入）脱离 BLAS | 输入 dtype 跟随权重 + **平台自适应 GEMV**（aarch64 → 自写 numba 核 `_gemv_rows`；x86 → BLAS），**不做全局翻转** |

**平台差异铁律**（三次反噬，见《并行与加速架构分析》与 BUGS.md「结构性教训汇总」）：

| 改动 | x86 | 昇腾 |
|---|---|---|
| M1 fp32 | 快 1.80× | **慢 70×** |
| M2 融合核 | 快 1.15–2.16× | **慢 8–13×** |
| `OMP_PLACES=cores` | 无感 | **慢 8×**、CS/s 650 万 |

→ **x86 的性能结论不构成证据**；跨平台项目里任何优化必须在目标机器复测。
`--m2-kernel {fused,plain}` 就是这条铁律的产物（默认 `plain`）。

---

## 九、诚实边界：哪些没迁移及原因

| 部件 | 状态 | 原因 |
|---|---|---|
| 分词 / 词表扫描 | 留 CPU（numba nogil 多核） | 变长字符串 + 状态化锚点链，设备侧不划算（整段上传 + 变长输出）；实测训练循环内占比极小 |
| 词涌现统计 | 留 CPU（已 nogil 化） | 迁设备需重写 + 容差判据；核内多线程实测只快 1.16×（内存带宽饱和），CPU 已非瓶颈 |
| PC 栈 / STDP / LTM | 留 CPU（生产主循环） | 事件驱动稀疏 + 在线 CSR 生长 + `big_ltm` 2²⁴；整体迁移需重写全部算子，且平行实现赶不上机制数（P30 教训）。若读出占比 < 50%，优化此处收益更大 |
| 两级读出 / 稀疏读出 | **加速后端未实现 → fail-fast 回落** | 「能跑但语义不同」比回落更糟（`readout_hidden>0`、`readout_conn_k>0`） |
| 量化码本（fp4） | **已禁用**（fhz 2026-09-27） | e2m1 分辨率不足；MX 块缩放立项后重新启用 |
| 跨设备逐位等价 | **不宣称** | 跨库归约顺序不同，判据为容差一致；同设备分片可逐位 |
| CUDA / ROCm / DirectML 真机 | **结构验证** | 真机回归须在对应硬件或 CI runner 上执行（`tools/accel_doctor.py` + `tools/bench_accel.py` 即 runner 就绪入口） |
| fp8 检查点体积与加载耗时 | **未测** | 已知开放项 |

---

## 十、CI 与门禁

| 项 | 状态 |
|---|---|
| `python tests/run_tests.py fast` | **9/9** |
| 专项 verifier | 18 个（逐位 / 容差各自有约定） |
| 语法门禁 | `verify_ms_stream` 会对改动文件 `py_compile` + 跑 `train.py --help` |
| CI 两套 | GitCode（`.gitcode/workflows/ci.yml`）、GitHub Actions（镜像，**当前仍红**，无日志权限，待 fhz 提供 traceback） |
