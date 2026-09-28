# train_1b —— PHD-Net 1B 参数模型训练与推理

PHD-Net 的 1B 档训练**与推理**子项目：在**事件驱动稀疏类脑语义**下，训练总突触参数
容量 ≥ 1×10^9 的词级语言模型。**训练数据无论多长不截断**（字符级流式，内存与语料
总长无关），**context 无硬窗口**（状态全程不重置，目标 ≥1M 连续 tokens）。不依赖
GPU 也能在小预算内真实训练与续训（每步成本只正比于活跃神经元数，与 1B 总容量无关）。

## 1M context 与「不截断」（2026-09-25）

- **流式训练**：语料按 `concat(样本_i + "\n\n")` 字符流逐样本流过，token 边产边训
  （`corpus_stream.py::StreamingTokenizer`，与全量贪心分词**逐位等价**，
  `verify_stream_tokenize.py` 四用例对拍 PASS）。4.6 GB pretrain 分片与 23 KB
  内置语料走同一条路，`--max-chars` 截断参数已废除。
- **context 语义**：PHD-Net 无位置编码/注意力窗口，context = 记忆机制的有效范围。
  训练循环保证 WM/STDP/LTM 状态**全程不重置**（跨样本、跨分片、跨 epoch 连续）；
  长程依赖由 big_ltm 事件驱动印迹（1B 突触容量）承载。每跨过
  `--context-milestone`（默认 1,000,000）token 打一行里程碑。
- **词表**：涌现词表来自采样文本（默认前 `--vocab-sample-chars 4M` 字符——只影响
  分词能力，训练数据本身全量流过）；`--vocab-scan full` 全量扫一遍构建零 OOV 词表
  （大语料需数小时）。训练流中 OOV token 跳过该步并计数（进 [METRIC] 的 oov_rate）。
- **中断续训**：检查点完整保存运行时状态（STDP 迹/调制器/step_count 等），
  resume 快进至断点——1M+ 连续训练可中断恢复。

## 多核词表构建与数据加载（2026-09-25，fhz 指令）

- **`--vocab-workers`**：词表构建并行进程数，**默认自动 = 核心数 × 0.8**（向下取整）；
  `1` = 串行原路径。两条并行化，均逐位等价（`verify_vocab_parallel.py` 对拍 PASS）：
  - **词涌现按 L 层并行**：`WordSegmenter` 各 L（2..max_len）统计相互独立，
    per-L 体抽为 `_induce_length`（串行/多核同一份实现、同一 numpy 运算序列），
    词集合完全一致；
  - **全量扫描 / head token 收集按批并行（锚点链）**：贪心最长匹配是无状态位置轨道，
    语料切批（缺省 32M 字符/批，各带 2×max_len 前视）并行送 worker，worker 以
    δ=0 主链因子化各候选入口的轨道，编排器按锚点链接链——并行 token 流与
    串行 `StreamingTokenizer` **逐位一致**（含跨批/跨样本 token；对拍含
    强制注入跨样本 token 的压力用例）。适用大语料 `--vocab-scan full`
    （数小时级 → ≈多核加速）；小语料进程启动开销占优时可用 `--vocab-workers 1`。
- **数据加载多核（多进程）**：训练流由 **W 个生产者进程**（默认 = 核心数×0.8，按
  文件数封顶）按连续文件片段并行解码 parquet/txt，各批带全局文件序号推入有界队列，
  主进程按文件顺序 reorder 归并——与 `char_chunks` 产出逐位一致（`verify_vocab_parallel.py`
  用例 D 对拍）；单文件语料自动退化为 1 进程。`--vocab-scan full` 的喂料同样走该通道。
- **零等待代码**：训练主循环无 sleep/轮询/忙等——仅有界队列与 Future 的
  OS 级阻塞；队列有界（64 批 × 64 样本，背压限内存 ≈ 数十 MB）。

## 容量口径（1B 在哪里）

| 组成 | 机制 | 规模（`--preset 1b` 默认） |
|---|---|---|
| **大空间事件驱动表（1B 主体）** | M4b 容量栈 `big_ltm`：结构可塑性，共激活神经元间生长新突触，每神经元 ≤60 条出边为硬容量约束 | 2^24 × 60 = **1,006,632,960** |
| 主干稀疏连接 | M2 预测编码栈，CSR 结构性稀疏（12.5% 连接率） | 4 × 1024×128 = 524,288 |
| STDP 侧向核 | M3 时序关联，每神经元 16 出边 | 1,024×16 = 16,384 |
| 编码器（稠密） | M1 稀疏编码，组合输入 2048→1024 | ≈ 2,098,176 |
| 读出头 | M6 softmax 感知器，h=3×1024 三拼 | 词表 × 3,072 |
| **总突触参数容量** | | **≈ 1.01–1.07×10^9 ≥ 1B** ✓ |

**诚实边界**：大空间表的 1B 是**容量**上限（`count_params` 计已生长突触），
训练初期利用率低、随经验增长；启动与日志均打印实时利用率，不做任何虚标。
"每步成本与总容量无关"成立的前提是 SDR 稀疏编码（6.25% 激活率）+ 事件驱动
局部计算——密集 1B 模型（权重+梯度+优化器 ≈16 GB、每步 ≥10^9 FLOPs）在
无独显机器上物理不可行，本档通过稀疏结构绕开，而非否认该物理约束。

## 文件

| 文件 | 职责 |
|---|---|
| `train.py` | 主入口：流式训练循环、容量验算、日志、检查点、1M 里程碑 |
| `corpus_stream.py` | 字符流 + StreamingTokenizer（逐位等价）+ PrefetchChars 多进程加载（按文件分片并行解码 + 顺序归并） |
| `vocab_parallel.py` | 多核词表构建：词涌现按 L 并行 + 全量扫描锚点链并行（默认核心数×0.8） |
| `config_1b.py` | 档位预设（smoke / 1b / 1b_max）、构建配置、容量验算 |
| `ckpt_1b.py` | 完整检查点：大空间表（CSR 快照 + 迹/时间戳）+ 词表 + 权重 |
| `infer.py` | 推理 / 对话：检查点自包含加载（不需要语料）、流式长 prompt、τ+top-k |
| `verify_stream_tokenize.py` | 对拍：流式分词 vs 全量分词逐位一致 |
| `verify_ckpt_roundtrip.py` | 对拍：保存→恢复→续训逐位等价 |
| `verify_vocab_parallel.py` | 对拍：多核词表/扫描 vs 串行逐位一致（含跨样本 token 压力用例） |

## 用法

```bash
# 冒烟：分钟级验证全管线（容量 <1B，仅功能验证）
python train_1b/train.py --preset smoke --data sft --tokens 2000

# 标准 1B 档，全量流式训练（任意长度不截断；1M context 里程碑自动打点）
python train_1b/train.py --preset 1b --data pretrain_zh --tokens 1000000

# 生产长跑（零 OOV 词表 + 断点续训）
python train_1b/train.py --preset 1b --data pretrain_zh --vocab-scan full --resume

# 推理 / 对话（检查点自包含加载，不需要语料）
python train_1b/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz \
    --prompt "用户：什么是机器学习？\n助手：" --n 200
python train_1b/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz --chat
```

产物位置（fhz 2026-09-28 指令：**生产训练产物统一存 `outputs/models/`**）：

- `outputs/models/phdnet1b_{preset}_{data}.npz` —— 滚动检查点（`--ckpt-every` 触发 + 收尾）
- `outputs/models/phdnet1b_{preset}_{data}_final.npz` —— 收尾另存的最终模型
- `outputs/train_logs/train_1b_*.log` —— 训练日志（含 `[METRIC]` 尾行）

## 检查点内容（相对生产版的增强）

`ckpt_1b.py` 保存/恢复**完整可续训状态**，生产版 `tools/train_production.py`
不包含前三项：

1. 大空间表邻接结构（`compact_csr` 快照，dict 版与在线 CSR 版均支持）
2. 突触迹与时间戳（t_pre/t_post/stamp_pre/stamp_post）+ 步数 + 入度
3. 词表与分词器（seg.vocab / max_len / tokens；SDR 哈希确定性重建）
4. 运行时状态：STDP 内部迹 / 调制器 Welford 统计 / net.step_count / 上一时刻
   发放率 / 任务门控（对拍 `verify_ckpt_roundtrip.py` 抓出的路径分叉缺口，已修复）
5. 主干 CSR 四权重、编码器、STDP、WM、读出（稠密 W 或稀疏 CSR 三元组）

续训要求 `--data` 与 `--vocab-sample-chars`（或 `--vocab-scan`）与原训练一致
（词表大小 fail-fast 校验）；**推理不需要语料**——检查点自包含（`infer.py`）。

## 硬件需求（`--preset 1b`，fp64）

| 项 | 需求 |
|---|---|
| 内存 | 静态 ≈0.5–0.6 GB（读出占大头）+ 大空间表已生长突触（dict 版 ≈100 B/条；`--csr-online` 切在线 CSR ≈10 B/条，逐位等价已由 `tests/verifiers/verify_csr_equiv.py` 验证） |
| 磁盘 | 检查点 ≈0.3–0.6 GB/份（npz 未压缩，速度优先） |
| 吞吐 | 本机纯 CPU 实测见日志 `ms/token`；10^9 token 生产训练需 GPU/集群（参考 `tools/estimate_scale.py` 外推：256M 档本机 157 ms/token ⇒ 10^9 token ≈ 5 年，1B 档生产训练必须换硬件） |

## 与既有 1B 工作的关系

- `tools/train_1b.py` —— 1B 突触容量的**演示核**（ABC 序列，无词表/语料）；
  本子项目是其**全管线生产化**（词级 LM + parquet 语料 + 检查点/续训/日志）。
- `tools/bench_1b_migrate.py` —— dict 版 vs 在线 CSR 版的性能对拍（容量口径同源）。
- `--preset 1b_max`：主干 4096（连接率不变）+ 读出稀疏化建议
  `--readout-conn-k 512`，供大内存机器放大固定突触部分。

## 词表与设备（P13–P22，2026-09-28）

### 词表来源（四选一，优先级从高到低）

| 来源 | 触发 | 说明 |
|---|---|---|
| 检查点自包含 | `--resume` 且 ckpt 存在 | **完全跳过词表阶段**（旧实现 resume 也会白扫一遍 full，几十分钟） |
| 已有快照（自动） | `outputs/models/vocab_<preset>_<data>.json` 存在 | 词表做出来时即落盘，下次启动**自动复用**，免重扫 |
| 外部词表 | `--vocab-file vocab.json` | JSON 权威格式（含 `words` + `seg_vocab` + `max_len`） |
| 扫描 | `--vocab-scan head\|full` | head = 采样文本（默认）；full = 全量锚点链扫描 |

- **JSON 是唯一权威格式**：词表含跨行 token（如 `\n的`），纯文本每行一词会切碎
  （实测 2,610 词读回只剩 2,571）。`.txt` 镜像可选（`txt_mirror=True`，不可回读）。
- 词表一确定就落盘 `outputs/models/`（模型构建之前）→ 训练崩溃不丢词表。
- 推理侧 `infer.py` 自动定位快照并与 `tok_tokens` **交叉校验**，不一致即报错退出。

### 词表构建性能（numba nogil）

| 阶段 | 实现 | 实测（4.7M 字符生产规模） |
|---|---|---|
| 词涌现 | nogil 核：开放寻址 hash 去重（取代 `np.unique(W, axis=0)` 整行排序）+ 边逐元素回比（精确非概率）+ 解析式熵；**层间线程池**并行 5 个 L | 63.1s → **9.0s（7.00×）**，逐位一致 |
| 全量扫描 | nogil 核 + `ThreadPoolExecutor`（共享内存零 pickle）；小词表 CSR 二分 / 大词表边哈希自适应 | 33.4 亿 tokens / 520s（191 核机器） |

核内 prange 曾实现并**已回退**（实测 1→6 线程仅 1.16×，瓶颈是内存带宽）。

### 设备与并行度

| 项 | 默认 | 说明 |
|---|---|---|
| `--accel` | `auto` | 读出计算设备：auto = 有 cuda/cann(npu)/rocm 就用，否则回落 numba CPU（**默认路径逐位不变**） |
| `--prefetch-workers` | `0`（自动 = **1 进程**） | parquet 解码在 pyarrow 内多线程且释放 GIL，单进程即吃满核；多进程只增内存 |
| `--prefetch-depth` | `0`（缺省 **256** = 在途数据量 4×） | 预取队列深度（批数）；缓冲 4× 防数据供给饿死，仍有界（背压成立） |
| `--vocab-workers` | `0`（核数×0.8） | 词表扫描线程数（nogil 真并行） |
| 读出计时 | 日志 | `token N … \| 读出 X ms/tok（后端@设备，占 Y%）` |

启动日志会打印并行度预算、能力矩阵（numba 只能上 CPU）与读出后端（回落时给原因）。
诊断加速器：`python tools/accel_doctor.py`。
