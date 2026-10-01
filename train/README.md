# train —— 1B 档生产训练目录使用说明

> **适用范围**：本目录是 PHD-Net 的**唯一生产训练入口**（`train/train.py`）。
> **数据截止**：2026-09-30。开关默认值以 `train.py --help` 与源码为准。
> **相关文档**：六机制与容量账 → `../docs/PHD-Net_架构设计.md`；
> 性能数字 → `../docs/PHD-Net_性能评估与迭代方案.md`（唯一出处）；
> 语料制备 → `../tools/README.md`；根入口 → `../README.md`。

---

## 1. 快速开始

### 最小可跑（冒烟，分钟级，只验管线）

```bash
pip install -r requirements.txt

python train/train.py --preset smoke --data sft --tokens 2000
```

`smoke` 档 width=256 / conn_k=32 / big_n=2²⁰，只验证管线通不通，**不验证容量**。
日志会打印「总突触参数容量 ⚠ < 1B（smoke 档仅验证管线）」——这是预期输出，不是错误。

### 标准 1B 档

```bash
# 本地 datasets/
python train/train.py --preset 1b --data pretrain --tokens 1000000

# ModelScope 远程流式（HTTP Range，零落盘）
python train/train.py --preset 1b --data pretrain --remote-data

# 零 OOV 词表 + 1M context 里程碑 + 断点续训
python train/train.py --preset 1b --data pretrain \
    --vocab-scan full --context-milestone 1000000 --resume
```

### 推理 / 对话

```bash
python train/infer.py --model outputs/models/phdnet1b_1b_pretrain_final.npz --prompt "…"
python train/infer.py --model outputs/models/phdnet1b_1b_pretrain_final.npz --chat
```

### 只看检查点的大空间表统计

```bash
python train/train.py --preset 1b --data pretrain --report
```

### 门禁（改动后必跑）

```bash
python tests/run_tests.py fast
```

---

## 2. 完整命令行开关表

`--preset` 选档，其余开关默认值与「为什么是这个值」如下。
**默认值全部来自源码**（`train/train.py` 的 `main()`、`config_1b.py`、`phdnet/config.py`）。

### 2.1 档位与数据

| 开关 | 默认 | 为什么是这个值 |
|---|---|---|
| `--preset` | `1b` | 容量档位。`smoke`=管线验证（分钟级，< 1B）；`1b`=标准档（width=1024/conn_k=128/big_n=2²⁴/big_m=72）；`1b_max`=大主干档（width=4096/conn_k=512）；`30b`=2²⁹×56≈30.3B。`smoke` 只验管线不验容量 |
| `--data` | `sft` | 训练语料 glob。`sft` / `pretrain`（全量）/ `pretrain_zh`（**已废弃，等价 `pretrain --data-lang zh`**，保留兼容）/ `eval`（冻结语料）。默认给 `sft` 是因为它最小、最适合当冒烟默认 |
| `--remote-data` | 关 | 数据直读 ModelScope `fhzfhz/Mixture-General-Mini`，HTTP Range 流式**零落盘**。**默认关闭**是为了保证本地 `datasets/` 路径逐位不变；服务器 `glob 无匹配` 时才需打开。**仅支持 ModelScope** |
| `--remote-fraction` | `0.3` | `--remote-data` 时取**排序后前 30%** 分片。取前缀子集而非随机采样，是为了保持数据顺序语义——词表扫描与训练流开头逐字符一致。改这个值再 `--resume` 会静默改变数据分布，故数据口径随检查点落盘 |
| `--lang` | `en` | **终端输出语言**（P119）：`en`=英文（**默认**）/ `zh`=中文。**对训练结果零影响** —— 语料、词表、模型状态、数值全不变，只改打印文案。实现见 `phdnet/i18n.py`（`set_lang` + 输出层兜底翻译）。 |
| `--data-lang` | `all` | **训练语料**语言过滤（P119 从 `--lang` 拆出）：`all`=全量（默认，与旧逐位一致）/ `zh` / `en`，按 parquet 的 `lang` 列筛训练流。**⚠ 这个参数会改变训练结果**。词表扫描不受影响（词表是训练流的超集 → OOV 恒 0） |

### 2.2 精度与后端

| 开关 | 默认 | 为什么是这个值 |
|---|---|---|
| `--readout-dtype` | **`fp16`** | 选 fp16 是为**保住学习**，不是省访存。bf16 半 ULP≈2e-4 ≫ 非目标行更新 \|dp\|≈1e-6 → 更新被舍 → **退化为纯 Hebbian**（PPL 震荡不降的根因）。fp16 在昇腾有原生 GEMV 且能保住微小更新。`fp8`/`fp4` 选项保留但**实测昇腾与 CPU 都不可用**（昇腾 ERR01007、CPU 无 fp8 addmv）→ 只在 `--accel cpu` 下走量化码本路径。`fp32` = 精确规则对照 |
| `--encoder-dtype` | **`fp64`** | M1 编码器权重存储精度（迭代恒 fp32）。默认 fp64 是因为**昇腾 aarch64 上 fp32 sgemv 实测慢约 70 倍**（58 vs 0.85 ms/tok，该平台 sgemv 内核未针对此尺寸调优）→ 平台自适应走自写 numba 核；**x86 上 fp32 快 1.80×**，在 x86 训练应显式 `--encoder-dtype fp32` |
| `--ckpt-dtype` | `bf16` | **存储**精度（不是训练精度）。bf16/fp16/fp8 存**位模式**（uint16/uint8）后按 `meta["ckpt_dtype"]` 无损解码，确能减小 npz。`""` = fp32 不压缩。torch 做不到 fp8 matmul，且 fp8 会把感知器非目标行更新冲成 0，所以 fp8 只作为**存储**选项 |
| `--accel` | `auto` | 读出计算设备。`auto` = 有加速器就用（**昇腾 → ROCm → CUDA → DirectML**），否则回落 numba CPU 原路径（逐位不变）。也可显式 `cpu`/`npu`/`cuda`/`rocm`/`dml` |
| `--nll-sync-every` | `8` | 读出 nll 同步周期。`1`=每步 `.item()`（旧行为，会在每步暴露同步点）；`N>1` 时 nll 累积到设备、每 N 步同步一次 → CPU/NPU 重叠，NPU 场景端到端约 −30~40%。默认 8 是因为服务器实测（读出 12.9 ms/tok、CPU 仅占 1.3–3.2/191 核、CS/s 250 万+ = 线程空转等同步）确认每步 `.item()` 是主要暴露点。代价：PPL 统计滞后 N 步，滑动均值下可忽略 |
| `--fp8-refresh` | `8` | 读出 fp8 forward 副本的重建间隔（步）。量化 1.6 亿元素是一次设备算子，摊到 N 步。**仅在 `--readout-dtype fp8`（CPU 码本路径）下生效**，fp16 路径不用 forward 副本 |
| `--numba-threads` | `8` | numba prange 线程上限（`0`=用 numba 默认=全部核）。服务器 191 核上主循环只用 1.3 核、CS/s 250 万+（百万级上下文切换 = 线程空转等锁）；核内 1→6 线程仅 1.16×（访存带宽饱和）→ 8 线程足够。默认从「全部核」降到 8 就是为了这个 |
| `--omp-proc-bind` | **开** | 设 `OMP_PROC_BIND=close` 把 OpenMP 线程绑到物理核。⚠ 只设 PROC_BIND、**不设 `OMP_PLACES=cores`**——191 核 place 表会让线程池每次同步遍历，实测 M2 慢 8–13×。`--no-omp-proc-bind` 关闭 |
| `--torch-compile` | **关** | 默认 OFF：inductor 编译在服务器上不稳定（debug trace 干扰 + 编译耗时不可控）；eager 在 NPU 上是 4 个小 kernel 异步流提交。本机实测曾 +15%（26.34 vs 30.98 ms/tok），但跨平台不保证 |
| `--torch-compile-mode` | `default` | 显式开编译时的模式。**必须用 `default`**：读出每步原地更新 W，cudagraphs 拒绝 mutated inputs，`reduce-overhead` / `max-autotune` 会打印 `skipping cudagraphs due to mutated inputs` 并丢掉该收益 |
| `--readout-conn-k` | `0`（稠密） | 稀疏读出每输出单元入边数。`0`=稠密，是**主路径也是当前架构欠账**（M6 100% 稠密）。大词表时可设 512–2048 换成 CSR 稀疏读出 |

### 2.3 规模与容量

| 开关 | 默认 | 为什么是这个值 |
|---|---|---|
| `--width` | `0`（用预设） | 覆盖主干宽度。**0 = 用预设值**。显式给值时 `conn_k` 保持 `max(8, w//8)` ≈ 12.5% 连接率，不因覆盖而破坏稀疏性 |
| `--big-n` | `0`（用预设） | 覆盖大空间神经元数。0 = 用预设（1b 档 2²⁴）。M4b 是 1B 档容量主体（2²⁴×72 = 1.208e9，占约 88%） |
| `--csr-online` | **开** | 大空间表用在线可写 CSR。每条突触 ~16 B（int8 权重 + CSR 索引）vs dict 的 ~100+ B → 1B 档实测省 5.74×（91.8→16.0 B/条目）。内存只随**已生长**突触数增长、与容量无关 → 32 GB 机器上长跑/30B 档的安全前提。`--no-csr-online` 回退 dict（**仅短跑/调试**） |

> `csr_online` 与 `k_sparse` 是铁律四的两个例外（default-on）——它们是**性能/容量前提**，
> 不是行为语义变更。`--no-csr-online` 在 dict 后端已生长 > 5000 万时日志会给出告警。

### 2.4 训练预算

| 开关 | 默认 | 为什么是这个值 |
|---|---|---|
| `--seed` | `11` | 全链路种子（分段器、SDR 哈希、初始权重）。沿用既有基线值，改动会使 PPL 锚点不可比 |
| `--epochs` | `1` | 语料流过遍数。**状态跨 epoch 连续不重置**（只有首步的 prev 链接断开），所以 >1 不会重置记忆 |
| `--tokens` | `0`（不限） | 训练步预算。0 = 不限，由 `--minutes` / `SIGINT` 收尾。生产用显式预算便于对齐评测口径 |
| `--minutes` | `0.0`（不限） | 时间预算（分钟）。长跑的兜底闸门 |
| `--log-every` | `500` | 每 N token 打一行进度。太小会刷爆 `train_logs/`，太大看不出 PPL 趋势 |
| `--context-milestone` | `1000000` | 每跨过 N token 打一行里程碑，**显式证明状态连续未重置**（1M context 达标）。`0`=关闭 |
| `--step-profiling` | 关 | 打开后追加 `loop:`（主循环三段）与 `segments:`（`net.step` 内部九段取前 6）。**诊断用**，有计时开销 |

### 2.5 词表

| 开关 | 默认 | 为什么是这个值 |
|---|---|---|
| `--vocab-scan` | `head` | `head`=采样文本建词表（快，OOV 由回退兜底且不崩）；`full`=全量流式扫一遍（零 OOV，但大语料需数小时）。默认 `head` 是因为训练流里 OOV token **跳过该步并计数**，流永不断 |
| `--vocab-sample-chars` | `4000000` | `head` 模式的采样字符数。**只影响词表，不截断训练数据本身**——这是两个独立的量 |
| `--vocab-file` | 无 | 外部词表（每行一词，`#` 注释与空行忽略）。给了它就**跳过** head/full 扫描。`--resume` 时词表取自检查点（自包含），同样跳过扫描 |
| `--vocab-workers` | `0`（自动） | 词表构建/全量扫描并行进程数。`0`=自动=核心数×0.8；`1`=串行。词涌现按 L 层并行，全量扫描按批走锚点链并行，逐位等价 |

**词表来源优先级**（`train.py` 内 `_resume_vocab` / `_file_vocab` 分支）：
`--resume` 自包含 > `--vocab-file` > `--vocab-scan head/full`。
四个来源都会在启动时立刻落盘一份**词表快照**到 `outputs/models/vocab_<preset>_<data>.txt`
（含分词器候选集 + `max_len` + sha1）——崩溃后可直接 `--vocab-file <该文件>` 复用，免重扫
（`full` 扫描约 520 s）。

### 2.6 两阶段训练 / 续训

| 开关 | 默认 | 为什么是这个值 |
|---|---|---|
| `--resume` | 关 | 从 `models/` 检查点续训：快进至断点、**状态由检查点恢复**、**步数保留**。检查点不存在时打印提示并从头开始（不报错） |
| `--init-from` | 无 | 两阶段微调（P26）：从该检查点**初始化权重**但**步数归零**（= 在预训练权重上做 SFT）。文件不存在时按从头开始处理并原样报告 |
| `--assistant-marker` | `""` | SFT 回复掩码标记（如 `'助手：'` / `'Assistant:'`）。设置后只对该标记之后的**助手回复**计算损失；其前的系统/用户 prompt 用 `learn=False` 推进状态、**不更新权重**。留空 = 全 token 计损失（旧行为） |

> **`--resume` vs `--init-from` 的区别就是「步数是否归零」**：续训用 `--resume`，
> 换阶段（SFT）用 `--init-from`。详见 §5。

### 2.7 数据加载并行

| 开关 | 默认 | 为什么是这个值 |
|---|---|---|
| `--prefetch-depth` | `0`（类缺省 8192 批） | 预取队列深度。⚠ 深度受**按文件序归并**约束——单生产者时囤积≈0，**depth 不是数据供给的主杠杆** |
| `--prefetch-batch` | `0`（类缺省 64 样本） | 每批样本数。增大它提高单次 IPC 传输量、减少唤醒次数——**这是提升数据提供量的有效杠杆之一** |
| `--prefetch-workers` | `0`（自动） | 预取进程数。`0`=自动=min(8, 核数×0.8, 文件数)。解码在 pyarrow 内多线程，进程过多只增内存/调度。远程模式另受 `REMOTE_MAX_PROCS` 约束（并发 Range 请求可能触发服务端限流） |

### 2.8 产物与日志

| 开关 | 默认 | 为什么是这个值 |
|---|---|---|
| `--save-dir` | `outputs/models/` | 生产训练产物统一存这里（**不入 git**，见 `.gitignore`）。与 `test/` `experiments/` 的入库产物分离 |
| `--ckpt-every` | `50000` | 每 N token 存一次检查点。默认 5 万是因为 1B 档每次保存含约 867 MiB 读出权重 D2H + 写盘，**约数秒**，太密会拖慢训练 |
| `--log-file` | 无（自动） | 不给则写 `outputs/models/train_logs/train_1b_<preset>_<data>_<时间戳>.log`。TeeLogger 同时接管 **stdout + stderr**（行缓冲），所以 `warnings.warn` 与 torch inductor 日志也会落盘 |
| `--report` | 关 | 只打印检查点的大空间表统计后退出（不训练）。用于查已生长突触数 / 容量利用率 |

---

## 3. 数据格式与流式语义

### 3.1 两入口

| 入口 | 路径形态 | 说明 |
|---|---|---|
| 本地 | `datasets/sft/sft_000.*.parquet`、`datasets/pretrain/pretrain_*.parquet` | 独立 git 仓库（`fhzfhz/Mixture-General-Mini`），gitignore |
| 远程 | `--remote-data` → `ms://fhzfhz/Mixture-General-Mini/<split>/<glob>` | ModelScope 官方 SDK 列目录 + fsspec HTTP Range 可 seek 流 → pyarrow 直读，**零原始落盘** |

统一 schema：**`text` / `lang` / `src`**（`text` 是训练文本，`lang` 供 `--data-lang` 过滤，`src` 溯源）。
导入侧统一走 `phdnet/corpus.py`（同认 txt 与 parquet）。
样本边界：字符流按 `SEP = "\n\n"` 切分。

### 3.2 流式语义（铁律三的落地）

- **训练数据永不截断**：语料按字符流逐样本流过，token 边产边训。内存占用与语料总长无关
  ——4.6 GB 分片与 23 KB 内置语料走同一条路。**没有 `--max-chars` 参数**。
- **状态连续**：WM / STDP / LTM 状态跨 token、跨样本、跨分片、跨 epoch 连续携带，
  **全程不重置**。长程依赖由 big_ltm 事件驱动印迹承载。
  `--context-milestone`（默认 1M）打行显式证明达标；每 epoch 只有**首步的 prev 链接断开**。
- **每步一个 token**：不允许批处理。读出 W **每步原地更新**。
- **OOV 不崩**：训练流中 OOV token 跳过该步并计数，OOV 率进 `[METRIC]` 尾行。
- **分词逐位等价**：流式分词与全量贪心最长匹配逐位一致（`verify_stream_tokenize.py` 对拍）。
- **主循环零等待**：生产者进程预取（`PrefetchChars`），主循环无 sleep/轮询/忙等。
- `SIGINT`（Ctrl-C）**不丢检查点**：置停止标志 → 退出循环 → 等后台写盘队列 → 存滚动检查点。

---

## 4. 检查点

### 4.1 异步保存语义

`--ckpt-every` 到点时调 `save_model_async()`：

1. **主线程做一致快照**（`save_model_snapshot`）——`to_numpy` 对 torch 张量返回
   **共享内存视图**，必须 `.copy()`；否则后台写盘时主线程 `addmm_`/learn 正在原地改写 →
   **数据撕裂**。`compact_csr`/`sparse_csr` 返回新数组，天然安全。
2. **写盘在后台线程**（`ckpt-writer`）排队执行，主循环不阻塞。
3. 收尾先 `wait_pending_saves()` 再存 final，确保**退出前落盘完成**。
4. 后台失败会打印 `[ckpt] 后台保存失败`（fail-fast，不静默吞）。

产物：`outputs/models/phdnet1b_<preset>_<data>.npz`（滚动）+ 同名 `.json`（meta）
+ `phdnet1b_<preset>_<data>_final.npz`（本次运行最终模型，meta 里带 `final: true`）。
meta 落盘**数据口径**（数据路径 / `--remote-fraction` / 语言过滤），防止续训时静默换数据分布。

### 4.2 低精度存储：位模式

`--ckpt-dtype bf16|fp16|fp8` 时，**大矩阵（读出 W 等）存原始位模式**而不是降精度数值：

| `--ckpt-dtype` | 存储 dtype | 说明 |
|---|---|---|
| `bf16`（默认） | `uint16` | numpy 无 bfloat16 dtype → 必须存位模式才能真正减小 npz |
| `fp16` | `uint16` | 同上 |
| `fp8` | `uint8`（float8_e4m3fn） | 仅存储；torch 仍无法做 fp8 matmul |
| `""` | fp32 数组 | 不压缩 |

**加载侧按 `meta["ckpt_dtype"]` 无损解码回原精度**（闭环）。
⚠ 这是**存储**选项，与 `--readout-dtype`（计算精度）无关：计算用 fp16、存储用 bf16 是正常组合。

### 4.3 `--resume` vs `--init-from`

| | `--resume` | `--init-from` |
|---|---|---|
| 用途 | **续训**（同一训练被中断） | **换阶段**（预训练 → SFT） |
| 权重 | 恢复 | 初始化 |
| 步数 | **保留** | **归零** |
| 词表 | 取自检查点（自包含，跳过扫描） | 按 `--vocab-scan` / `--vocab-file` 走 |
| 大空间表 | 恢复已生长突触 | 重新生长 |

两者可只给其一；`--init-from` 的文件不存在时按从头开始处理并**原样报告**（不静默失败）。

---

## 5. SFT 用法（assistant marker 双标记状态机）

给 `--assistant-marker '助手：'` 后，`StreamingTokenizer` 的 `__next__` 从「yield 纯 token」
变成「yield `(token, trainable)`」。**双标记**：

| 标记 | 值 | 切换到 |
|---|---|---|
| 助手标记 | `--assistant-marker` 的值（如 `助手：` / `Assistant:`） | `mode=True`（**计损失**） |
| 用户标记 | 固定 `"用户："` | `mode=False`（**不计损失**） |

**状态机是零滞后的前缀匹配**（`corpus_stream.py::_consume`）：

1. 若某 token 落在某个 marker 的字符序列内（marker 跨 token 边界）→ 该 token 不可训练，
   并按前缀长度扣减 `_pending`；marker 刚被消费完的那一步**立即**切模式。
2. 否则看缓冲区前缀是否命中某个 marker（完整或前缀）：完整命中 → 立即切模式；
   部分命中 → 记 `_pending` 待续。
3. 都不命中 → 按当前 `mode` 返回可训练性。

> 早期实现用 `rfind` + 游标，marker 跨 token 边界时会**滞后 1~2 个 token**，
> 导致掩码错位（实测首个可训练 token 落在回复中间的「诗。」而不是开头）。
> 现实现消除了该滞后。

**为什么要有状态机**：SFT（instruction tuning）只应对**回复**计损失。
对 prompt 计损失会把模型往「复读用户问题」的方向拉。prompt 段仍以 `learn=False`
推进状态（记忆照常演化），只是不更新权重。

**损失归属**：损失由 `(p1, p2) → p0` 这一步产生，因此用 **`p0`（目标侧）** 的可训练标记
决定是否记 `seg_nll`，而不是 `p1`。收尾会打印 prompt 掩码步数：

```
[SFT] reply masking active (marker '助手：'): N prompt steps advance state only without weight updates; loss counted on assistant replies only
```

未设置 `--assistant-marker` 时行为与旧版**逐位一致**。

---

## 6. 日志字段速查表

### 6.1 进度行（每 `--log-every` 步）

```
  token  1,000,000  sliding PPL  394.4687    62.31 ms/tok  elapsed 103.8 min  | readout   6.412 ms/tok (accel-ascend@npu,  10.3% of total)
```

| 字段 | 含义 |
|---|---|
| `token N` | 已训练 token 数（**累计**，含 `--resume` 恢复的步数） |
| `sliding PPL` | 最近 `--log-every` 步的 `exp(mean(nll))` 滑动均值，**不是**全量 PPL |
| `ms/tok` | 端到端单 token 毫秒 = 总耗时 / 本次步数 |
| `elapsed` | 本次运行分钟数 |
| `readout X ms/tok` | M6 读出分项耗时。**访存受限的证据就看这一项** |
| `(accel-…@npu, N% of total)` | 读出后端 / 设备 / 占端到端比例 |

### 6.2 `loop:` 与 `segments:`（`--step-profiling`）

`loop:` 是**主循环三段**——主循环开销不在 `segments:` 里，差额就落在这里（此前从未被测量）：
`tokenize` / `encode_onehot` / `step`。

`segments:` 是 `net.step` 内部**九段**，按耗时降序取前 6。段名固定集合：
`M1_encode`、`M2_infer`、`M3_pred`、`M4a_wm`、`M4b_ltm`、`M4b_imprint`（与 recall 分开计时）、
`M5_mod`、`PC_learn`、`STDP_learn`。`M4b_ltm` 占比高是事件驱动表随训练变长的先行指标。

### 6.3 其它诊断行

| 行 | 触发 | 用途 |
|---|---|---|
| `[ltm-diag] imprints=… prev=… cur=… combos=… rate_dims=… rows=… k_hash=…` | 每 10 次 imprint | LTM 规模与稀疏度演化。⚠ 门槛须考虑触发频率（imprint 是条件触发，门槛设太高会一行不出） |
| `[ltm-diag] recalls=… active=… scores=… bindings=…` | 每 10 次 recall | 召回侧遍历量。`scores`/`bindings` 随表增长，是 M4b 超线性的先行指标 |
| `[big-ltm] grown … / capacity … (utilization …) \| memory ≈… MB` | 每检查点 + 每里程碑 | 已生长突触 / 容量上限 / 利用率 / 内存预估。**容量账的实时口径，不虚标** |
| `[sample lang=…] CJK=…% \| '…'` | 每 `--log-every` 步 | 用最近 8 个真实 target token 反查词表拼文本 + CJK 占比，**一眼看出 `--lang` 是否真生效** |
| `[numba] cache dir = … \| size = … MB` | 启动一次 | numba 持久缓存是否命中（首次全量编译约 2.6 s 启动） |
| `[device] ⚠️ NPU 初始化失败 → 读出将回落到 numba CPU。` | 启动一次 | **加速器启动期健康检查**。常见原因：① 上次训练的 python 进程还在（`npu-smi info` / `ps aux \| grep train_1b`）② 容器未映射设备 ③ 设备被占满（507033）。强制 CPU 用 `--accel cpu` |
| `[parallel] numba prange threads = … \| OMP_PROC_BIND=… BLAS/OpenMP threads=…` | 启动一次 | 确认线程上限真的生效（默认 cap 8） |
| `[readout] compute precision = …` / `[readout] backend=… \| fallback reason: …` | 启动一次 | 读出计算/存储精度 + 半精度舍入更新告警；回落原因（探测到 ≠ 能用） |
| `[context milestone] N tokens processed continuously (state never reset; ≥1M context achieved ×K)` | 每 `--context-milestone` | **逐 token 连续性的显式证明** |
| `[checkpoint] saved N tokens → path` | 每 `--ckpt-every` | 异步保存已触发（`[ckpt] 后台保存失败` 则是失败） |
| `[epoch K/N] continuing stream across epochs: state continuous (net not reset)` | 每 epoch | 跨 epoch 状态连续，仅首步 prev 断开 |
| `[SFT] reply masking active (marker …)` | 启动 + 收尾 | SFT 掩码生效与 prompt 掩码步数 |
| `[gc] freeze() + threshold(50000, 200, 200)` | 启动一次 | GC 已冻结（长跑防停顿） |
| `[METRIC] preset=… data=… tokens=… ms_per_token=… final_ppl=… oov_rate=…` | 收尾 | **机器可读的单行指标**，便于入库对比 |

**收尾还会打印**：`this run N tokens in M min (X ms/token) | OOV skipped … (rate)`、
`tail sliding PPL ≈ …`、`model: …`、`checkpoint: …`、`[log] log saved: …`。

### 6.5 终端语言

`--lang zh`（默认）终端全中文、`--lang en` 全英文。实现在 `phdnet/i18n.py`：
① `T("中文", "English")` 精确路径；② 输出层兜底（包装 stdout/stderr 按替换表转换遗留中文串）。
**只影响终端文案，模型数据（词表、语料、日志里的数值）一律不翻译。**
日志数值（`token` / `PPL` / `ms` / 字段名）保持原样以便 grep。

---

## 7. 本目录文件索引

| 文件 | 职责 |
|---|---|
| `train.py` | **生产训练主入口**。全部命令行开关、主循环、流式接入、异步保存编排、SFT 掩码、终端 i18n 安装 |
| `config_1b.py` | 档位预设（`smoke`/`1b`/`1b_max`/`30b`）、`build_cfg()`、`capacity_report()` 容量验算 |
| `corpus_stream.py` | `SEP` 样本边界、`char_chunks()` 字符流、`PrefetchChars` 多进程预取、`StreamingTokenizer`（含 SFT 双标记状态机） |
| `vocab_parallel.py` | `auto_workers()`、`build_segmenter_parallel()`、`parallel_head_tokens()`、`scan_vocab_parallel()` |
| `tokenizer_core.py` | 分词核心（字符 → 词候选的向量化最长匹配） |
| `ckpt_1b.py` | `save_model(_async/_snapshot)`（一致快照 + 位模式编码）、`wait_pending_saves`、`load_model`、`save_vocab_snapshot` |
| `infer.py` | 推理 / `--chat` 对话 / `--milestone` / 多设备读出（`--devices`） |

**生产训练入口单轨在本目录**，新增训练脚本不进 `tools/`。工具脚本索引见 `../tools/README.md`。

### 排障速查

| 症状 | 先查 |
|---|---|
| 训练慢、CPU 占用率极低 | 逐 token 串行是**架构性**的。确认 `--numba-threads`（默认 8）与 `[parallel]` 行 |
| `[device] ⚠️ NPU 初始化失败` | 杀掉残留进程（`npu-smi info` / `ps aux \| grep train_1b`）；确认容器映射设备；或 `--accel cpu` |
| `--lang zh` 看着没生效 | 看 `[sample lang=…]` 行的 CJK 占比（比猜 PPL 可靠）；注意 `--lang` 默认是 `all` |
| PPL 震荡不降 | 确认 `--readout-dtype int8`（bf16 把非目标行更新舍成纯 Hebbian，这是历史根因） |
| 首次启动慢 | numba 首次全量编译约 2.6 s；`[numba] cache` 看缓存目录与大小 |
| 检查点写盘慢 | 1B 档每次约 867 MiB D2H + 数秒写盘，属预期；调大 `--ckpt-every` |
| 想确认容量 | 启动时的容量验算表，或 `--report` 查已生长突触 / 利用率 |
| 门禁失败 | `python tests/run_tests.py fast`（9 项）+ `tests/verifiers/` 下 18 个专项 verifier |
