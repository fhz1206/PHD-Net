# PHD-Net 性能评估与迭代方案

> **适用范围**：`phdnet/` 包（v0.0.0）+ `train_1b/`（1B 档流式训练管线）的**性能数字**。
> 本文是**所有性能数字的唯一权威出处**——其他文档一律链接引用，不复制数字。
> **数据截止**：2026-09-30。
> 配套：为什么吃不满多核与 NPU 的**论证**见 `PHD-Net_并行与加速架构分析.md`；
> P53–P84 的**过程与教训**（含被否决方案）见 `PHD-Net_迭代优化与修复日志.md`；
> 缺陷台账见 `BUGS.md`；后端矩阵与昇腾踩坑见 `PHD-Net_硬件后端适配报告.md`。

---

## 一、测量方法

**没有测量方法就没有性能数字。** 本项目的性能结论全部来自下面这一套固定方法；
换方法即换口径，数字不可跨表比较。

### 1.1 开关

| 开关 | 默认 | 为什么影响测量 |
|---|---|---|
| `--step-profiling` | 关 | **唯一的分段计时来源**（P35 九段 + P72 主循环三段）。关闭时零开销；不开就只剩端到端一个数，无法归因 |
| `--readout-dtype` | **fp8**（P84） | 决定读出的访存量与计算精度。fp8 = forward 走 fp8 副本（读 320→80 MB）+ 更新走 fp16 主副本 |
| `--fp8-refresh` | 8 | fp8 forward 副本的重建间隔（步）。重建是一次设备算子，摊到 N 步 |
| `--m2-kernel` | **plain** | `fused` 在昇腾退化 8–13×（P76 实测），x86 快 1.15–2.16×。默认随平台定，测速必须显式带上 |
| `--encoder-dtype` | **fp64** | M1 编码器精度。numpy 混合 dtype（fp64 W @ fp32 x）会脱离 BLAS → 慢 5.9×（P61 实测） |
| `--numba-threads` | 8 | numba `prange` 线程上限（0 = 不限）。191 核上不限必然空转 |
| `--omp-proc-bind` | 开 | 设 `OMP_PROC_BIND=close`。**不再设 `OMP_PLACES=cores`**（P74 已撤，见 §五） |
| `--nll-sync-every` | 8 | nll 设备侧累积间隔。默认 1 时每步 `.item()` 把 NPU 延迟全额暴露给 CPU，流水永不重叠（P62） |
| `--torch-compile` | 关 | inductor 在昇腾首次调用才编译，失败会崩生产（P58）。测速时保持默认关闭，否则测的是编译回落路径 |
| `--prefetch-workers` | 0（单进程即吃满核） | 数据加载。pyarrow 解码在库内多线程且释放 GIL，多进程只增内存 |
| `--ckpt-every` | 50000 | 检查点硬停顿的触发间隔（当前 0.8–37 s，见 §四） |

### 1.2 日志字段与读法

| 字段 | 位置 | 读法 |
|---|---|---|
| `sliding PPL …  X ms/tok` | 训练主行 | 端到端步时。**跨 run 比较必须同数据、同 `--remote-fraction`、同机器** |
| `readout X ms/tok (后端@设备, Y% of total)` | 训练主行 | 读出分项（P20）。`% of total` 直接判断加速是否生效、耗时是否转移到 PC 栈 |
| `loop: tokenize A  encode_onehot B  step C` | 主行下一行（需 `--step-profiling`） | **主循环三段**（P72）。此前主循环开销从未被测量，导致「分段之和 < 总耗时」的缺口无法归因 |
| `segments: <名> <ms/tok> …` | 再下一行（需 `--step-profiling`） | **`net.step` 内部九段**，按耗时降序取前 6（`_prof` 累加值 ÷ 已完成 token 数） |
| `[ltm-diag] imprints=… prev=… cur=… combos=… rate_dims=…/… rows=… k_hash=…` | 每 10 次 imprint | LTM 规模演化。⚠ 门槛必须考虑**触发频率**——imprint 是条件触发（`mode=="encode"` 且 gate 达标），门槛设 1000 时 3500 token 一行都不出 |
| `[ltm-diag] recalls=… active=… scores=… bindings=…` | 每 10 次 recall | 召回侧遍历量。`scores`/`bindings` 随表增长，是 M4b 超线性的先行指标 |
| `CPU … proc X核 … CS/s … GC …M/gen2 N` | 遥测行 | `proc X核` = 进程实际用核数；`CS/s` = 上下文切换频率（线程空转的指纹）；`GC` 用于验证/证伪 GC 假设 |
| `NPU/GPU …%  HBM …GB` | 遥测行 | 设备利用率与显存。**AI Core% 走 `npu-smi` 外部子进程**（`torch.npu.utilization()` 内部同步设备流，会破坏被观测的流水线） |
| `[numba] cache dir = … | size = X MB` | 启动行 | 持久缓存是否命中：命中启动 ~2.6 s，未命中会全量编译 |
| `[parallel] numba prange threads = N … OMP_PROC_BIND=… BLAS/OpenMP threads=N` | 启动行 | 线程治理是否真的生效（见 §五） |
| `[readout] compute precision = …` / `[capability] readout device resolved (auto) = …` | 启动行 | 读出实际精度与设备。精度降级（如 CPU 上 fp8 回落 fp16）必须在这一行可见 |
| `[sample lang=zh] CJK=…% \| '…'` | 语种自检 | `--lang` 是否生效一眼可见，不再靠 PPL 猜 |
| `[benchmark]` / `tools/bench_accel.py` | 独立工具 | 加速器基准。⚠ 必须带 `dtype` 且先 `_sync_device()`，否则测到的是 kernel 提交耗时（P29 修掉的三个 bug） |

### 1.3 profiling 的正确读法

**九段**（`phdnet/model.py` 中的段名，与 `--step-profiling` 的 `segments` 输出一一对应）：
`M1_encode` / `M2_infer` / `M3_pred` / `M5_mod` / `M4a_wm` / `M4b_ltm`（内嵌 `M4b_imprint`）/ `PC_learn` / `STDP_learn`。

⚠ **`readout` 不在 `segments` 里**——它走独立的累计器（`_ro_ms_accum` / `_ro_calls`），
打印在训练主行的 `readout X ms/tok` 字段。所以 `segments` 之和天然**不含读出**，
求和时必须把主行的读出值加进来，否则会误判为「有未归因的缺口」。

⚠ **`M4b_ltm` 只覆盖 imprint 分支**。recall 分支（`elif retrieve_now`）**没有独立计时**，
它的时间落在「九段之和 ≠ `loop.step`」的差额里。因此：
**M4b 剩余成本归因（开放项 1）需要靠 `M4b_imprint` 与差额的联合判读**，
不能只看 `M4b_ltm` 一个数就断言 recall 侧便宜。

三步读法，缺一步就会得出错误结论：

1. **先对账**：`loop.step` 是否 ≈ 训练主行的 `ms/tok`。不等则差额在主循环（分词/onehot 之外）或在计时之外。
2. **再求和**：`segments` 九段之和 **+ 主行读出值** 是否 ≈ `loop.step`。不等则还有未计时的工作
   （已知两处：recall 分支、突触修剪 `critical_period` 分支在 `PC_learn` 计时之外）。
3. **最后看趋势**：同一 run 内把 `segments` 按 token 数排成序列。**绝对值小但超线性增长的段才是真元凶**——2026-09-29 的 M4b 就是这样定位的（token 1500 时 `M4b_ltm` 仅 0.14 ms/tok，token 3500 涨到 19.54，而同期 `M2_infer` 反而从 5.05 降到 2.70）。

> 只看「当前最大段」会追错目标：那一段可能正在下降，而下一段正在指数增长。

---

## 二、环境与口径

| 项 | 本机（x86） | 1B 服务器（昇腾） |
|---|---|---|
| CPU / 加速器 | Windows 11 / **8 线程** / 12.6 GB RAM / **无 GPU·NPU** | **191 核** + NPU（~1.6 TB/s HBM） |
| 解释器 | Python 3.13（托管 venv：numpy 2.5.3 / numba 0.67 / pyarrow 25 / torch 2.14+cpu） | 同 |
| 语料 | `eval_corpus/internal_corpus.txt`（27,034 字符冻结基准）+ `eval_corpus/ood_wiki.txt` | ModelScope `fhzfhz/Mixture-General-Mini` 前 30% 分片（HTTP Range 流式） |
| 词表口径 | 小栈 256 维栈 | **51,962 词表 × 3,072 隐层 = 1.6 亿读出参数**（bf16 = 320 MB） |
| 用途 | 正确性对拍、机制消融、本机微基准 | **性能数字只以本列为准** |

**PPL 口径**：字符归一 PPL = exp(Σ token NLL / 评估段字符数)。跨粒度比较统一用该口径。

⚠ **治理约定**：`docs/*.md` **不是数据集**。冻结基准在仓库根 `eval_corpus/`，是评测基准（非训练数据）。

---

## 三、当前性能基线（1B 档，昇腾 191 核 + NPU）

### 3.1 端到端

| 口径 | 值 | 来源 |
|---|---|---|
| **实测区间** | **20–110 ms/tok** | spec 现行口径（跨多轮优化，含最差段） |
| 最优段 | **26 ms/tok**（区间 20–83） | 2026-09-29 服务器快照 |
| 最低单点 | 20.0 ms/tok | P62 修复后（同步 + 线程限流生效） |
| 修完待复测的预期 | 20–110 区间下沿 | P70/P76/P77/P78/P80/P83 全部落地后**未复测** |
| 历史最差段 | 104–107 ms/tok | 带 `OMP_PLACES=cores` 时（已撤） |
| 超线性段 | 100.56 ms/tok（token 10000） | 根因是 M4b 表增长（P64 → P70） |

### 3.2 各段耗时

| 段 | 优化前（昇腾实测） | 现状 | 处置与出处 |
|---|---|---|---|
| `M4b_ltm` | **63.5 ms/tok（占 76%）** | 2.7–22（修后未复测到底） | 元凶是 `predict` 的 1.8 万次 dict 更新 → P70 `predict_arr` **44.6×**（本机 8.78 → **0.197 ms**）+ P78 learn 批量多核 |
| ↑ 其中 `M4b_imprint` | aarch64 单次 ≈ **20 s** | 待复测（内层段已加） | Python `learn`：65k 组合 × `_find_slot` 扫描，且只占 1 核 |
| `M1_encode` | 0.85 → **58–60** | 平台自适应 | P77：aarch64 走自写 numba GEMV，x86 走 BLAS。**aarch64 numpy GEMV 病态，fp32/fp64 都慢** |
| `M2_infer` | 2.4 → **19–32** | `--m2-kernel plain`（6.7–11.9，快 3–4×） | P76：A/B 实测融合核在昇腾退化 |
| `readout`（NPU） | 9.5–13.5 | **5.5–7** | P80 nll 单 kernel + pinned correct + 关图优化。**访存受限**，见 §四 |
| `PC_learn` / `STDP_learn` | <1 | <1 | 未优化也未成为瓶颈 |
| `M3_pred` / `M5_mod` / `M4a_wm` | <1 | <1 | 同上 |
| 主循环 `tokenize` | 从未测量 | **0.06 ms** | P72 新增。结论：主循环不是黑洞 |
| 主循环 `encode_onehot` | 从未测量 | **0.13 ms** | 同上 |
| 检查点 | 每 5 万步硬停 **0.8–37 s** | 向量化后仍 3.9 s/10 万行 | P83 `compact_csr` 向量化 + 异步写盘（7.2 → 3.9 s / 10 万行） |

### 3.3 资源利用率

| 指标 | 值 | 判读 |
|---|---|---|
| 进程实际用核 | **1.1–3.2 / 191** | **架构性**，不是配置问题（论证见《并行与加速架构分析》） |
| 上下文切换 | 250 万–600 万 /秒 | 线程空转 + 核间漂移；P71/P73 处理后仍在此量级 |
| CS/s（带 `OMP_PLACES`） | 650 万 | place 表让同步更频繁，是回滚的直接证据 |
| NPU AI Core% | 修复前恒 `--` → **98%** | P57 遥测修复 + P63 `npu-smi` 三级定位 + P71 绑核后可见 |

---

## 四、访存账（读出为什么是天花板）

读出 W **常驻设备**，每步只往返 h（12 KB 上行）与 y。流量全在 W 上。

| 口径 | 每步访存 | 备注 |
|---|---|---|
| bf16 全程 | **960 MB/步** | forward 读 W + update 读/写 W |
| P28 修复前 | 4,334 MiB/步 | `torch.outer` 物化了与 W 同尺寸的临时张量（1B 档 867 MiB） |
| P28 修复后 | **2,600 MiB/步（−40%）** | 改 `addmm_` rank-1 AXPY 原地融合，**正好触底**（读 W 一次 + 读写 W 各一次），数值逐位相同 |
| P84 fp8 forward + fp16 更新 | **1.36 GB/步（+42%）** | ⚠ **不是访存收益**，是学习精度收益（见下） |

**为什么是访存受限而不是算力受限**：

- 实测 5.5–7 ms/tok 对应 2×320 MB/步 ≈ **64 GB/s** 量级，远低于昇腾 ~1.6 TB/s HBM。
- 反过来算：320 MB 走满 HBM 只需 **0.2–0.3 ms**。实测比理论搬运下限高一个量级 →
  剩余时间是**同步、launch 开销与小 kernel 串行提交**（P55/P58/P62/P80 已分别处理）。
- 结论：**访存账是地板，launch/同步账是当前的天花板**。再往下需要 CANN 自定义算子
  （把 GEMV + softmax + addmm 融成一个核）或 msprof 内核级 profiling。

**P84 的 fp8 方案必须连着精度一起读**：

- 动机**不是省访存**（更新侧 fp16 使总访存 +42%），而是**学习精度**：
  bf16 半 ULP ≈ 2e-4 ≫ 非目标行更新 |dp| ≈ 1e-6 → 更新被舍入 → 学习退化为纯 Hebbian
  → PPL 震荡不降。本机验证：fp16 主副本下 **5722 个非目标行格点**获得更新（bf16 下为 0）。
- fp8 matmul 仅昇腾/CUDA；CPU 自动回落 fp16（`tdtype` 同步回落）并告警一次。
- 因此「bf16 读出 PPL 略优」这类**旧语料口径的低精度正则化结论已作废**——它测的是
  「更新被舍掉后」的行为，不是低精度计算的行为。

**要真正降访存，只有稀疏化一条路**（M6 读出目前 100% 稠密）：量级、方案与论证见
《并行与加速架构分析》§五。

---

## 五、线程治理（BLAS / numba / OpenMP）

在 191 核机器上，**线程设置本身就是头号性能杠杆，也是最容易反噬的地方**。

### 5.1 三个必须知道的时序约束

| 约束 | 原因 | 违反后果 |
|---|---|---|
| BLAS 线程上限必须在 **`import numpy` 之前**设 | OpenBLAS 初始化后再设环境变量**无效** | 191 线程去跑 1024×2048 的 sgemv → 1.1–3.2 核 + 250 万 CS/s |
| `numba.set_num_threads` 必须在**任何核首次执行之前** | 线程池在首次调用时定型 | 191 线程跑千行级 `prange` → 纯空转 |
| `OMP_PROC_BIND` 必须在**numba 初始化之前**进环境 | 绑核在 OpenMP 初始化时生效 | 绑核不生效，且无法事后补 |

实现见 `train_1b/train.py:59`（BLAS 四变量）与 `train_1b/train.py:409`（PROC_BIND + numba 线程）。
生效情况看启动行 `[parallel] …`。

### 5.2 为什么上限是 8 而不是更多

P22 实测：**核内 1 → 6 线程仅 1.16×**，瓶颈是内存流量（两张 ~64 MB 哈希表远超 L3）。
本机融合核带宽墙同样是 1→8 线程 4.1 → 9.3 GB/s 的次线性曲线。
→ **超过 8 线程不再有收益，只增加调度与空转成本。**

### 5.3 已回滚的配置

| 配置 | 状态 | 依据 |
|---|---|---|
| `OMP_PROC_BIND=close` | **保留** | 绑物理核而非到处迁移（P71） |
| `OMP_PLACES=cores` | **已撤** | 191 核 place 表 → 每次线程池同步都要遍历 → M2 再慢 **8×**（19–32 ms/tok）、CS/s 升到 650 万 |
| 核内 `prange`（词涌现） | **已回退** | 1→6 线程仅 1.16× |
| 局部词表复用 | **已放弃** | 破坏正确性。**性能优化先过对拍** |
| `torch.npu.utilization()` 遥测 | **禁用** | 它同步设备流，在热路径上砍一刀 |

> **教训**：绑核类环境变量在**超多核机器上可能反噬**（place 表规模 = 核数）。

---

## 六、平台差异（x86 ≠ aarch64，三次反噬）

| 改动 | x86 | 昇腾 |
|---|---|---|
| M1 编码器 fp32（P61） | 快 **1.80×**（606.7 → 337.7 µs） | **慢 70×**（0.85 → 58 ms） |
| M2 融合核（P52） | 快 **1.15–2.16×** | **慢 8–13×**（2.4 → 19–32 ms） |
| `OMP_PLACES=cores`（P74） | 无感 | **慢 8×**、CS/s 650 万 |

补充：fp32 权重 @ fp64 输入在 **x86 上就已经慢 5.9×**（脱离 BLAS 的逐元素慢路径）——
这是唯一一条在两个平台都成立的规则：**GEMV/GEMM 前必须 `W.dtype == x.dtype`**。

> **规则：x86 的性能结论不构成证据。** 任何优化必须在目标机器复测。
> 本文档中凡标「本机」的数字**只在本机成立**。

---

## 七、本机（x86 8 核）性能明细

以下数字**只在本机成立**，不可外推到昇腾。

### 7.1 1B 训练管线

| 项 | 实测 | 说明 |
|---|---|---|
| 词表扫描 | 33.4 亿 tokens / **500 s**（191 核服务器） | head/full 扫描；`--vocab-scan full` 数小时级 |
| 词表结构构建 | 5 万词建 trie **265 s → 0.48 s** | 修复 O(节点×词表) 卡死：一次遍历邻接表 + BFS 展平 |
| 锚点链扫描引擎 | 8.30 → **21.41 Mchar/s（2.58×）** | `tokenizer_core.py`：`@njit(nogil=True)` 的 CSR-trie 贪心轨道 + δ 前缀链 + `ThreadPoolExecutor`（共享内存、零 pickle）。**原锚点链热路径是纯 Python dict/set 查找，全程持 GIL → 单纯线程化只能单核**，故先编译为释放 GIL 的机器码再线程化 |
| 查找结构自适应 | 5 万词表 0.66 → **11.59 Mchar/s（17×）** | 小词表 CSR 二分 / 大词表边哈希（阈值 2 万边），best-of-3 |
| 词涌现 | 4.7M 字符 63.1 → **9.0 s（7.00×）** | numba nogil（hash 去重 + 边回比 + 解析式熵）+ 层间线程池，逐位一致 |
| 组大小 | 32M → **1M 字符** | 32M 组时并行度塌缩（组数 ≪ worker 数），且每组数百万 token 的字符串 pickle 成单点 |
| 数据加载 | **预取默认 1 进程** | 解码在 pyarrow 内多线程且释放 GIL，单进程即吃满核；多进程只增内存（每进程 ~100–200 MB）与调度 |
| 在途数据量 | 深度 64 → **8192** + `--prefetch-batch` | ⚠ 深度受**按文件序归并**约束：单生产者时囤积≈0，depth 不是主杠杆。真正有效的是 `--prefetch-workers`（多生产者）与 `--prefetch-batch` |
| numba 编译缓存 | 冷启动 3.96 → **2.60 s** | 全部 `@njit` 已带 `cache=True`（唯一例外是 `inline="always"` 小辅助，与 cache 互斥） |
| `torch.compile` | 开 26.34 vs 关 30.98 ms/tok（本机 CPU torch，快 15%） | **现默认关**（P58：inductor 首次调用才编译，昇腾失败会崩生产） |
| CPU 侧预计算 | onehot 14.12 → **0.33 µs（43×）** | ⚠ 占端到端仅 **0.069%**——方向对但量级很小。大头在 M2 推理（状态依赖无法预计算） |
| 读出参考路径 | fp32 **40.9 ms/token**（读出规模 9,219×3,072） | ⚠ 这是**小栈口径**，与 1B 档（51,962×3,072）不可混用 |

### 7.2 读出精度的性能含义

| 精度 | 存储（相对 fp32） | 计算受限程度 |
|---|---|---|
| fp32 | 1.0×（基准） | 更新核 **13.3 ms**（本机最快） |
| fp16 / bf16 | 0.5× | 更新核 **226 ms**——位运算重量化 > 带宽节省 |
| fp8 | 0.25× | 仅昇腾/CUDA 有原生 matmul；CPU 回落 fp16 |
| fp4 | 0.125× | e2m1 分辨率粗，当前逐张量缩放不足（块缩放 MX 立项候选，非当前 7 条开放项之一） |

> **诚实边界**：低精度在本机 CPU 上是**计算受限**，收益只在**存储压缩**与
> **GPU/加速器原生张量核**。且 P84 已证明 bf16 的真实代价在**学习精度**而非访存（§四）。

---

## 八、评测基线

复现入口 `tools/rebaseline.py`。配置：**256 维栈 / `eta_readout=0.15` / 读出 fp32**。

| 口径 | 现行锚点 |
|---|---|
| 4,000 字符段 | **394.4687** |
| 全语料 | **359.2603** |

⚠ 评测语料 2026-09-28 更换为中文维基条目合集。**旧口径 78.16 / 90.2480 / 73.1166 只存于
git 历史，不可与现行锚点混用。**

**泛化**（`tools/audit_gen_eval.py`）：跨域泛化不足是当前主短板，**根因是数据 ≫ 架构**。
总预算不变下，混合域训练使近域 inflation −19% / 远域 −23%，代价是域内 +27%（预算分摊）。
⚠ 该结论出自已删除的 mix 训练流（P42）；现行训练为两阶段全量
（`--data pretrain_zh` → `--data sft`）。

---

## 九、门禁现状

| 项 | 状态 |
|---|---|
| `python tests/run_tests.py fast` | **9/9** |
| 专项 verifier | **18 个**（逐位/容差各自有约定；容差一致的融合核明确标注） |
| 语法门禁 | `verify_ms_stream` 对改动文件 `py_compile` + 跑 `train.py --help`（P54 起） |
| 其他门禁 | `verify_ms_stream` 41/41、`verify_accel_readout_p55` 9/9、`verify_pc_learn_fused` 12/12、`verify_ltm_kernels` 10/10、`verify_ltm_learn_batch` 5/5、`verify_ckpt_roundtrip` PASS |
| CI | **两套**：GitCode（`.gitcode/workflows/ci.yml`）、GitHub Actions（镜像，**当前仍红**，无日志权限，待 fhz 提供 traceback） |

**门禁缺口**（本轮暴露，见迭代日志）：bf16/fp8 的检查点 round-trip 不在 fast 集合内 →
本机全绿、服务器崩（P83b）。建议并入 fast。

---

## 十、已知开放项

| # | 项 | 说明 |
|---|---|---|
| 1 | M4b 复测归因 | `M4b_imprint` 内层段已加，但 **recall 分支尚无独立计时**（§1.3）→ 需联合判读 imprint 段与「九段+读出 ≠ step」的差额 |
| 2 | fp8 上 PPL 是否开始下降 | 精度问题，决定后续所有性能实验的可信度 |
| 3 | 数据集语种矛盾 | 远程分片实测中文仅 2%、`unk` 47% 是代码，与文档记录（中文 95%）矛盾 → 需核对数据制备/上传环节 |
| 4 | 检查点 `compact_csr` | 向量化后仍 3.9 s/10 万行 |
| 5 | **M6 幂律稀疏化** | 架构欠账，方案已设计未实施；是访存量级的唯一杠杆 |
| 6 | GitHub Actions 仍红 | 无日志权限，待 traceback |
| 7 | bf16/fp8 检查点体积与加载耗时 | 未测 |

---

## 十一、复现入口

```bash
# 零回归门槛（9 项）
python tests/run_tests.py fast

# 性能：1B 生产配置模块级剖析（务必带 --step-profiling）
python train_1b/train.py --data pretrain --lang zh --remote-data \
    --remote-fraction 0.3 --step-profiling

# 精度体系（L1 逐位 / L2 带宽 / L3 PPL）
python tools/audit_precision.py
# 当前评测基线复测（394.4687 / 359.2603）
python tools/rebaseline.py
# LM 热路径 cProfile 剖析（写 outputs/experiments/profile_lm.log）
python tools/profile_lm.py
# 加速器探针 + 读出基准（CUDA/ROCm/NPU/CPU）
python tools/bench_accel.py
# 设备侧一次性诊断
python tools/accel_doctor.py
# 多核词表/加载逐位对拍
python tests/verifiers/verify_vocab_parallel.py
# 读出设备侧流量基准
python tests/verifiers/bench_accel_path.py
```
