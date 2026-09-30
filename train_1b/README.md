# train_1b —— 1B 档生产训练目录使用说明

> **适用范围**：本目录是 PHD-Net 的**唯一生产训练入口**（`train_1b/train.py`）。
> **数据截止**：2026-09-30（P84：读出默认 fp8 forward + fp16 更新）。
> **相关文档**：机制原理与容量账 → `docs/PHD-Net_架构设计.md`；
> **所有性能数字** → `docs/PHD-Net_性能评估与迭代方案.md`（本文只引用不复制）；
> 平台差异与迁移 → `docs/PHD-Net_硬件后端适配报告.md`；
> 怎么加新机制/后端/dtype → `docs/PHD-Net_扩展指南.md`；
> 缺陷台账 → `BUGS.md`。写作纪律见 `docs/文档写作规范.md`。

本目录负责「训练 + 推理」两个动作。**训练入口已单轨化**：早期并存过
`tools/train_production.py`，该脚本**已删除**，其配方与 4M 基线只能从 git 历史复现。
不要在文档或脚本里再引用它。

---

## 1. 快速开始

### 1.1 冒烟（分钟级，只验管线通不通）

```bash
python train_1b/train.py --preset smoke --data sft --tokens 2000
```

`smoke` 档容量 <1B，**仅验证功能**。要跑几分钟而非几秒就 `--tokens 100000`。

### 1.2 生产长跑（1B 档）

```bash
python train_1b/train.py --preset 1b --data pretrain \
    --remote-data --remote-fraction 0.3 --resume
```

这条命令的每个部分都有理由，别随手删：

| 片段 | 为什么 |
|---|---|
| `--preset 1b` | 标准档，总突触容量 ≥1×10^9（口径见架构设计文档）。 |
| `--data pretrain` | 预训练分片（中英全量）。SFT 用 `--data sft`，两阶段分开。 |
| `--remote-data` | 数据直读 ModelScope，HTTP Range 流式、**零本地落盘**。服务器上没有本地数据集时必需。 |
| `--remote-fraction 0.3` | 取排序后前 30% 分片（前缀子集 → 词表与训练流开头一致）。默认即 0.3，写出来是为了让口径显式。 |
| `--resume` | 词表直接取自检查点（自包含，**跳过整个词表阶段**）+ 断点续训。 |

**本地数据入口**（`datasets/` 有分片时）：去掉 `--remote-data` /
`--remote-fraction` 即可，其余完全相同。本机数据集已按 fhz 指令删除，重训前先用
`tools/fetch_ms.py` 重新采样。

### 1.3 阶段二 SFT

```bash
python train_1b/train.py --preset 1b --data sft \
    --init-from outputs/models/phdnet1b_1b_pretrain.npz \
    --assistant-marker "助手：" \
    --vocab-file outputs/models/vocab_1b_pretrain.json
```

`--init-from` = **初始化权重、步数归零**；`--resume` = 继续同一状态并保留步数。
两者语义不同，别混用。详见 §5。

### 1.4 推理 / 对话

```bash
python train_1b/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz \
    --prompt "用户：什么是机器学习？\n助手：" --n 200
python train_1b/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz --chat
```

检查点**自包含**，推理不需要语料。`infer.py` 会自动定位词表快照并与
`tok_tokens` 交叉校验，不一致直接报错退出。

### 1.5 产物位置

| 路径 | 内容 |
|---|---|
| `outputs/models/phdnet1b_{preset}_{data}.npz` | 滚动检查点（`--ckpt-every` 触发 + 收尾各存一次） |
| `outputs/models/phdnet1b_{preset}_{data}_final.npz` | 收尾另存的最终模型 |
| `outputs/models/phdnet1b_{preset}_{data}.json` | 检查点元数据（`ckpt_dtype`、数据口径等） |
| `outputs/models/vocab_{preset}_{data}.json` | 词表快照（**权威格式**，见 §4） |
| `outputs/models/train_logs/train_1b_*.log` | 训练日志（含 `[METRIC]` 尾行） |
| `outputs/numba_cache/` | numba 持久编译缓存（`NUMBA_CACHE_DIR`，不被 `__pycache__` 清理波及） |

---

## 2. 命令行开关表

`python train_1b/train.py --help` 是权威来源（#2 教训：改 help 文本后必须实跑一次）。
下表是**默认值 + 为什么是这个值**。「为什么」列里带平台结论的，x86 与昇腾常常相反，
详见性能文档。

### 2.1 精度与后端（最需要按机器选的一组）

| 开关 | 默认 | 为什么是这个默认值 |
|---|---|---|
| `--readout-dtype` | `fp8` | **动机是学习精度，不是省访存。** forward 用 fp8_e4m3fn 副本（1B 档读流量 320→80 MB），**更新用 fp16 主副本**。bf16 半 ULP≈2e-4 ≫ 非目标行更新 \|dp\|≈1e-6 → 更新被舍 → 退化为纯 Hebbian；fp16 主副本下本机实测 5722 个非目标行格点获得更新，bf16 下为 0。代价：更新侧 fp16 使总访存 **+42%**。⚠ fp8 matmul **仅昇腾/CUDA**，CPU torch 自动回落 fp16 主副本（只告警一次）。`fp32` = 精确规则对照 |
| `--fp8-refresh` | `8` | fp8 forward 副本的重建间隔（步）。量化 1.6 亿元素是一次设备算子，摊到 N 步：N 越大越省，但 forward 用的副本越旧。8 是「量化成本可忽略」与「副本不过分陈旧」的折中。 |
| `--encoder-dtype` | `fp64` | 昇腾 aarch64 上 fp32 sgemv 实测**慢约 70×**（该平台 sgemv 内核未针对此尺寸调优），且 fp32/fp64 的 numpy GEMV 都病态 → aarch64 走自写 numba 核、迭代恒 fp32。**x86 上 fp32 快 1.80×**，在 x86 上训练应显式 `--encoder-dtype fp32`。混合 dtype 会让 numpy 脱离 BLAS 走逐元素慢路径（务必两个都改）。 |
| `--accel` | `auto` | 读出计算设备。auto = 有 cuda/cann(npu)/rocm 就用，否则回落 numba CPU 原路径（**默认路径逐位不变**）。也可显式 `cpu/npu/cuda/rocm/dml`。注意：**numba 只能编译到 CPU**，加速器上只有读出走设备，其余部件仍在 CPU。 |
| `--nll-sync-every` | `8` | nll 累积到设备、每 N 步同步一次 → 与 CPU 计算重叠。1 = 每步同步（旧行为）。PPL 统计滞后 N 步，滑动均值下可忽略。 |
| `--m2-kernel` | `plain` | M2 推理核。**昇腾上融合核退化 3–4×**（服务器 A/B：fused 20–27 ms/tok vs plain 6.7–11.9，prange+fastmath 在 aarch64 上退化）→ 跨平台训练默认取保守值。x86 上 fused 快 2.1×，在 x86 上可显式 `--m2-kernel fused`。数据与口径见性能文档的平台差异表。 |
| `--torch-compile` / `--no-torch-compile` | **默认关** | inductor 编译在服务器上不稳定（debug trace 干扰 + 编译耗时不可控）；eager 在 NPU 上是 4 个小 kernel 异步流提交，与编译版的实际差距需 `--step-profiling` 自行量化。想要融合核再显式打开。 |
| `--torch-compile-mode` | `default` | 只有开了 `--torch-compile` 才有意义。`default` 融合 kernel 但**不开 cudagraph** —— 读出每步原地更新 W，cudagraph 拒绝被改写的输入；选 `reduce-overhead`/`max-autotune` 会打印 `skipping cudagraphs` 并丢掉该收益。 |

### 2.2 线程与并行度

| 开关 | 默认 | 为什么是这个默认值 |
|---|---|---|
| `--numba-threads` | `8` | numba prange 线程上限（0 = 用 numba 默认 = 全部核）。服务器实测 191 核上主循环只用 1.1–3.2 核、CS/s 250 万+，大量上下文切换来自「用 191 线程跑千行级 prange」的线程空转等锁。核内实测 1→6 线程仅 1.16×（访存带宽已饱和）→ 8 线程足够。 |
| `--omp-proc-bind` | **默认开** | 设 `OMP_PROC_BIND=close` 把 OpenMP 线程绑到物理核。⚠ **只绑核、不设 `OMP_PLACES`** —— 191 核的 place 表会让线程池每次同步都遍历它，实测 M2 慢 8–13×。`--no-omp-proc-bind` 关闭。 |
| `--vocab-workers` | `0`（自动 = 核心数 × 0.8） | 词表构建/全量扫描的并行度。`1` = 串行原路径（小语料时进程启动开销占优）。 |
| `--prefetch-workers` | `0`（自动） | 语料预取进程数。parquet 解码在 pyarrow 内多线程且释放 GIL，单进程即能吃满核，进程过多只增内存。远程源另有 `REMOTE_MAX_PROCS = 4` 的上限（每进程持独立连接池 + 8 MB block 缓存，8 进程并发 Range 可能触发服务端限流）。 |
| `--prefetch-depth` / `--prefetch-batch` | `0`（类缺省 8192 / 64） | 预取队列深度与每批样本数。**注意深度受「按文件序归并」约束** —— 单生产者时囤积≈0，depth 不是数据供给的主杠杆，`--prefetch-batch` 才是。 |

BLAS 线程上限（`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS`）在 `train.py` 顶部
**`import numpy` 之前**设好（默认 `min(8, 核数)`），不需要也不能在运行时改 ——
OpenBLAS 初始化后再设环境变量无效。

### 2.3 数据

| 开关 | 默认 | 说明 |
|---|---|---|
| `--data {sft,pretrain,pretrain_zh,eval}` | `sft` | 训练数据一律走上传分片 parquet；`eval` 指冻结基准 `eval_corpus/internal_corpus.txt`（**27,034 字符**实测原文长度）。`--data mix` 已删除，训练是**两阶段**。`pretrain_zh` 保留兼容，等价 `--data pretrain --lang zh`。 |
| `--remote-data` | 关 | 直读 ModelScope `fhzfhz/Mixture-General-Mini`，HTTP Range 流式零落盘，**仅支持 ModelScope**。`--data eval` 时不生效。 |
| `--remote-fraction` | `0.3` | 取排序后前多少比例的分片（fhz 2026-09-29 指令）。前缀子集 → 词表扫描与训练流开头一致。 |
| `--lang {all,zh,en}` | `all` | 按 parquet 的 `lang` 列过滤**训练流**；`all` 逐位不变。**词表扫描不受影响**（词表是训练流的超集 → OOV 恒 0）。选择结果随检查点 meta 落盘。 |
| `--vocab-scan {head,full}` | `head` | `head` = 采样文本建词表（快，OOV 由回退兜底并计数）；`full` = 全量流式扫一遍（零 OOV，大语料需数小时）。⚠ 远程数据上 `full` 会把分片整读两遍 → 配 `--vocab-file` 复用快照。 |
| `--vocab-sample-chars` | `4_000_000` | head 模式的采样字符数。**只影响词表，训练数据本身全量流过不截断**。 |
| `--vocab-file` | 无 | 外部词表（JSON 权威格式，见 §4）。给了它就**跳过整个扫描阶段**。 |
| `--tokens` / `--minutes` / `--epochs` | `0` / `0.0` / `1` | 训练预算，0 = 不限。`--epochs > 1` 时状态跨 epoch 连续不重置（仅首步的 prev 链断开）。 |
| `--context-milestone` | `1_000_000` | 每跨过这么多 token 打一行里程碑（`0` = 关闭），显式证明连续 context 达标。 |

### 2.4 检查点与观测

| 开关 | 默认 | 说明 |
|---|---|---|
| `--ckpt-every` | `50000` | 每 N **token** 触发一次**异步**保存（语义见 §6）。 |
| `--ckpt-dtype` | `bf16` | 检查点**存储**精度（`""`/`fp8`/`bf16`/`fp16`），只作用于大矩阵，与训练计算精度无关。⚠ `*_final.npz` 不走这个开关（恒定原精度存储），详见 §6.4。 |
| `--log-every` | `500` | 主日志行间隔（token 数）。 |
| `--step-profiling` | 关 | 输出九段 `segments:` + 主循环三段 `loop:`。定位瓶颈用，一次加它会多花一点墙钟。 |
| `--report` | 关 | 只打印检查点的大空间表统计后退出（查利用率，不训练）。 |
| `--seed` | `11` | 默认即基线值，改它等于换一条实验线。 |
| `--csr-online` / `--no-csr-online` | **默认开** | 大空间表用在线可写 CSR（~16 B/条 vs dict 的 ~100+ B）→ 内存只随**已生长**突触增长、与容量无关，这是 32 GB 机器上长跑的前提。`--no-csr-online` 回 dict 版（逐位等价由 `tests/verifiers/verify_csr_equiv.py` 保证），仅供短跑/调试。 |
| `--readout-conn-k` | `0`（稠密） | 稀疏读出每输出单元入边数，大词表时建议 512–2048。⚠ 加速读出后端**未实现**稀疏读出 → 设了会回落 numba CPU 并在日志里给原因。 |
| `--width` / `--big-n` | `0`（用预设） | 覆盖主干宽度 / 大空间神经元数。 |
| `--preset` | `1b` | `smoke`（管线验证）/ `1b`（标准档）/ `1b_max`（大主干）/ `30b`（2^29 × 56 ≈ 30.3B，**依赖在线 CSR**，单机不现实）。 |
| `--save-dir` / `--log-file` | `outputs/models/` / 自动 | 产物与日志位置。 |

启动日志会打印：并行度预算、后端 × 设备能力矩阵、**读出后端的实际解析结果**
（探测到 ≠ 能用）、回落时的原因、以及数据口径（分片数 / 总量 / lang 过滤 / 远程比例）。

---

## 3. 数据格式与流式语义

### 3.1 不截断 / 1M context

- **训练数据永不截断**：语料按 `concat(样本_i + "\n\n")` 字符流逐样本流过，token 边产边训，
  内存占用与语料总长无关。`--max-chars` 截断参数已废除。4.6 GB 分片与 23 KB 内置语料
  走同一条路。
- **context = 记忆机制的有效范围**（无位置编码/注意力窗口）。训练循环保证 WM / STDP /
  LTM 状态**全程不重置**（跨样本、跨分片、跨 epoch 连续），长程依赖由 big_ltm 事件驱动
  印迹承载。每跨过 `--context-milestone` 打一行 `[context milestone]`。
- **流式分词与全量贪心匹配逐位等价**，对拍见 `tests/verifiers/verify_stream_tokenize.py`。
- **OOV**：训练流中遇到词表外 token 就**跳过该步**并计数（永不崩溃、永不截断），
  计入日志尾行的 `oov_rate`。

### 3.2 远程语料注意事项

1. `--remote-data` **仅支持 ModelScope**（`fhzfhz/Mixture-General-Mini`），走 HTTP Range。
2. `--vocab-scan full` 会把远程分片**整读两遍** → 配 `--vocab-file` 复用快照。
3. 分片级抽样是**前缀子集**（前 `remote_fraction` 比例），顺序语义与本地一致。
4. ⚠ **已知数据问题**（未修）：实测该仓库 52 个 pretrain 分片中，第 0–45 片
   `lang` 分布为 en 50% / unk 47% / **zh 仅 2%**（`unk` 实为代码），只有末尾第 51 片是
   中文（99.7%）。与设计记录「中文 3,697 万块 ≈ 95%」冲突。后果是 `--lang zh` 过滤后
   只剩约 2% 的行，`--remote-fraction 0.3` 取到的前 16 片又全是 m7core →
   「中文训练」实际只用到极小部分数据。**这是数据制备/上传环节的问题，不是代码问题**，
   详见 `BUGS.md` #18。
5. 数据口径（`data` / `remote` / `remote_fraction` / `lang` / 分片数）随检查点落盘，
   避免「改了 `--remote-fraction` 又 `--resume`，数据分布静默变化」。

---

## 4. 词表

四种来源，优先级从高到低：

| 来源 | 触发条件 | 说明 |
|---|---|---|
| 检查点自包含 | `--resume` 且检查点存在 | **完全跳过词表阶段**（旧实现 resume 也会白扫一遍 full，几十分钟） |
| 已有快照 | `--vocab-file vocab_*.json` | 词表做出来时即落盘，下次启动免重扫 |
| 外部词表 | `--vocab-file <路径>` | 同上；JSON 是权威格式 |
| 扫描 | `--vocab-scan head\|full` | head = 采样文本（默认）；full = 全量锚点链扫描 |

- **JSON 是唯一权威格式**：`words` + `seg_vocab`（分词器候选集）+ `max_len` + `sha1`。
  词表含跨行 token（如 `\n的`），纯文本每行一词会切碎。`.txt` 镜像可选但**不可回读**。
- 词表一确定就**先落盘再建模** → 训练崩溃不丢词表，下次 `--vocab-file` 直接复用。
- 快照缺 `seg_vocab` 时会打印告警并回退到 token 词表作候选集 —— 分词结果可能与训练
  不同（静默降级风险），优先用 `--resume`。
- 多核构建（`--vocab-workers`）两条并行化均**逐位等价**，对拍见
  `tests/verifiers/verify_vocab_parallel.py`。

---

## 5. SFT：回复掩码与两阶段

**为什么需要**：`--data sft` 早先只是「把 SFT 语料当普通语料训」（所有 token 都计损失）
—— 那是指令微调的数据、却用预训练的方式训练。

| 能力 | 用法 | 说明 |
|---|---|---|
| 回复掩码 | `--assistant-marker "助手："` | 只对**助手回复**计损失；系统/用户 prompt 段用 `learn=False` 推进状态、不更新权重。对 prompt 计损失会把模型往「复读用户问题」的方向拉 |
| 两阶段微调 | `--init-from <预训练ckpt>` | 从检查点**初始化权重但步数归零**；与 `--resume`（继续同一状态并保留步数）不同 |

- **双标记状态机**：遇到用户侧标记（默认 `用户：`）退出可训练段、遇到助手侧标记进入。
  多轮对话实测约 50% 的 token 参与损失。
- 掩码判定**零滞后**：marker 跨 token 边界也正确。早期版本用「最后一个 marker 之后」
  会漏掉前几轮（仅 0.2% 步计损失）。
- 未设 `--assistant-marker` 时行为与旧版**逐位一致**（全 token 计损失）。

`[SFT]` 收尾行会打印被掩掉的 prompt 步数，可用来确认掩码真的在生效。

---

## 6. 检查点

### 6.1 保存了什么

`ckpt_1b.py` 保存/恢复**完整可续训状态**：

1. 大空间表邻接结构（`compact_csr` 快照，dict 版与在线 CSR 版均支持）
2. 突触迹与时间戳（`t_pre/t_post/stamp_pre/stamp_post`）+ 步数 + 入度
3. 词表与分词器（`seg.vocab` / `max_len` / `tokens`，SDR 哈希确定性重建）
4. 运行时状态：STDP 内部迹 / 调制器 Welford 统计 / `net.step_count` / 上一时刻发放率 /
   任务门控（这一组是对拍 `verify_ckpt_roundtrip.py` 抓出的路径分叉缺口）
5. 主干 CSR 四权重、编码器、STDP、WM、读出（稠密 W 或稀疏 CSR 三元组）

### 6.2 异步保存语义

`--ckpt-every` 触发的是 `save_model_async`，分三步：

1. **主线程做一致快照**：临时把 `np.savez` 换成捕获函数，跑原 `save_model` 主体，
   再对每个数组 `.copy()`。**必须拷贝** —— `to_numpy` 对 torch 张量返回共享内存视图，
   不拷贝则后台写盘时主线程的原地更新会撕裂数据。`compact_csr`/`sparse_csr` 返回新
   数组，天然安全。
2. **后台单 worker 写盘**：daemon 线程 `ckpt-writer` 串行处理队列（不并发覆盖同一文件），
   `np.savez` + 写同名 `.json`。IO 释放 GIL，训练循环不停摆。
3. **收尾 `join`**：正常退出与 SIGINT 都先 `wait_pending_saves()` 再存 final，
   final 落盘后再 `wait_pending_saves()` 一次才打印完成。

意义：检查点原本的 0.8–37 s 硬停顿不再落在训练循环里。**代价**：进程被
`kill -9` 时队列里的快照会丢（最近一次已完成的滚动检查点仍在）。后台写盘失败会在
日志里打 `[ckpt] 后台保存失败`（fail-fast 精神，不静默吞掉）。

### 6.3 位模式存储与 `ckpt_dtype`

numpy 没有原生 `bfloat16` / `float8_e4m3fn`，所以这两种格式按**位模式**存：
bf16 → `uint16`、fp8 → `uint8`；加载侧按 `meta["ckpt_dtype"]` 解码回原精度闭环。
只对**大矩阵**（`ndim ≥ 2` 且 `size ≥ 4096`）压缩，小数组原样存。

`--ckpt-dtype` 是**存储**精度，与训练计算精度是两件事：
`--readout-dtype fp8` + `--ckpt-dtype bf16` 是合法且推荐的组合。

### 6.4 恢复方式与续训要求

```bash
# 从滚动检查点续训（词表也从检查点取）
python train_1b/train.py --preset 1b --data pretrain --remote-data --resume
```

- `--resume` 找不到检查点时**如实打印**「starting from scratch」，不静默。
- ⚠ **`*_final.npz` 不吃 `--ckpt-dtype`**：收尾那两次 `save_model` 没有传该参数，
  所以 final 恒以原精度存储、体积比滚动检查点大。要小体积就续训时重跑一段再拿滚动检查点。
- 续训要求 `--data` 与词表来源与原训练一致（读出层形状随词表大小对齐，不一致会报错）。
- **`--init-from` 不恢复运行时状态**（步数归零，等价「在预训练权重上重新开始」）。

---

## 7. 运行诊断

### 7.1 日志字段速查

| 日志片段 | 含义 | 怎么用 |
|---|---|---|
| `token 12,000  sliding PPL <P>  <X> ms/tok  elapsed 8.2 min \| readout <R> ms/tok (accel:npu@npu, <S>% of total)` | 主日志行（尖括号 = 实测填入）：滑动 PPL、墙钟 ms/tok、读出分项 | 读出占比高 → 先看 `--readout-dtype` / `--accel`；占比低 → 瓶颈在 CPU 侧，看 `segments:` |
| `loop:  tokenize <a>  encode_onehot <b>  step <c>  ms/tok` | **主循环三段**（`net.step` 之外的开销此前从未被测量） | 三段之和 ≠ 总 ms/tok 的差额在 `net.step` 内部；`tokenize` 大说明数据供给跟不上 |
| `segments:  M4b_ltm 1.70  M2_infer 1.20  M1_encode 0.85 …` | **九段分解**（按耗时降序取前 6）：`M1_encode`/`M2_infer`/`M3_pred`/`M5_mod`/`M4a_wm`/`M4b_ltm`/`M4b_imprint`/`PC_learn`/`STDP_learn` | 只在 `--step-profiling` 下输出。读出不在九段内（单独计时） |
| `[ltm-diag] imprints=… prev=…/cur=…/combos=…/rate_dims=…/rows=… k_hash=…` | LTM 规模与稀疏度演化（每 10 次 imprint） | `combos` 暴涨 = 结构可塑性成本上升 |
| `[ltm-diag] recalls=… active=…/scores=…/bindings=…` | 召回侧遍历量（每 10 次） | `active`/`scores` 决定 predict 的 Python 遍历量 |
| `[sample lang=zh] CJK=98% \| '…'` | 用**最近 8 个真实 target token** 反查词表拼文本 + CJK 占比 | 判断 `--lang` 是否真生效，一眼可见。⚠ 排查前先看日志 `[log] start: … argv:` 里到底有没有那个参数 |
| `GC <对象数>M/gen2 <次数>`（遥测行内） | 被跟踪对象数 / gen2 回收次数 | ms/tok 随步数超线性恶化时验证/排除 GC 假设 |
| `[numba] cache dir = … \| size = … MB \| 首次运行会全量编译，此后命中缓存（预期启动 ~2.6 s）` | numba 持久缓存状态 | 首次启动慢是正常的；size 异常小 = 核没带 `cache=True` |
| `[telemetry] npu-smi = <path>` / `[telemetry] ⚠ 未找到` / `[telemetry] npu-smi parse FAILED; raw head://…` | 设备遥测可用性 | `NPU/GPU --%` 恒为 `--` 时先看这三行 |
| `CPU 12% proc 1.3核 RAM 34%/12.1GB NPU/GPU 41% HBM 3.2/64.0GB CS/s 2500000` | 遥测单行 | 「核用不满」的判据在此（架构性，见并行分析文档） |
| `[gc] freeze() + threshold(50000, 200, 200); tracked objects = …` | GC 调优生效确认 | |
| `[big-ltm] grown … / capacity … (utilization …) \| memory ≈… MB` | 大空间表实时利用率 + 内存护栏 | 利用率长期极低是正常的（容量随经验生长，不虚标） |
| `[checkpoint] saved N tokens → <path>` | 滚动检查点已**入队**（不是已落盘） | 落盘完成看收尾 `wait_pending_saves()` |
| `[METRIC] preset=… tokens=… ms_per_token=… final_ppl=… oov_rate=…` | 机器可读尾行 | 批量实验采集用这一行 |
| `[capability] …` | 后端 × 设备能力矩阵 + 读出设备解析结果 | 「加速器没被识别」时先看这里 |

### 7.2 性能数字去哪看

**本文不给性能数字。** 所有 ms/tok、带宽、加速比、平台差异对照表的唯一出处是
`docs/PHD-Net_性能评估与迭代方案.md`；「为什么吃不满 191 核」的论证在
`docs/PHD-Net_并行与加速架构分析.md`；优化过程与被否决的方案在
`docs/BUGS.md#结构性教训汇总`。本文 §2 只解释**开关默认值为什么这么设**。

### 7.3 诊断工具

```bash
python tools/accel_doctor.py     # 加速器：环境矩阵 / 试分配 / 真实负载计时
python tools/backend_probe.py    # 硬件后端设备矩阵
python tools/bench_accel.py      # 读出基准（多档精度 + 等效带宽 GB/s）
python tests/verifiers/verify_accel_readout.py       # 读出加速逐位对拍
python tests/verifiers/verify_ltm_kernels.py         # LTM 核逐位（recall 全链路）
python tests/verifiers/verify_ckpt_roundtrip.py      # 保存 → 恢复 → 续训逐位等价
```

---

## 8. 目录内文件索引

| 文件 | 职责 |
|---|---|
| `train.py` | **主入口**：流式训练主循环、词表四源决策、容量验算、日志、检查点、context 里程碑。线程数治理必须在 `import numpy` 之前完成，改动这个文件时**先跑 `--help`**（#2/#3 教训） |
| `tokenizer_core.py` | numba `nogil` 贪心分词核心（全量扫描多核线程化的热路径）。释放 GIL → 线程池真多核、共享内存零 pickle。逐位等价承诺由 `tests/verifiers/verify_vocab_parallel.py` 验收 |
| `corpus_stream.py` | 字符流 + `StreamingTokenizer`（与全量贪心分词逐位等价，SFT 双标记状态机在此）+ `PrefetchChars` 多进程加载（按文件分片并行解码 + 按文件序归并，与 `char_chunks` 产出逐位一致） |
| `config_1b.py` | 档位预设（`smoke` / `1b` / `1b_max` / `30b`）、`build_cfg`、容量验算 `capacity_report`（按 `count_params` 同口径） |
| `vocab_parallel.py` | 多核词表构建：词涌现按 L 并行 + 全量扫描锚点链并行 |
| `ckpt_1b.py` | 完整检查点：保存/恢复、异步写盘队列、位模式压缩与解码、词表快照落盘与读取 |
| `infer.py` | 推理 / 对话：检查点自包含加载（不需要语料）、流式长 prompt、温度 + top-k 采样 |
| `README.md` | 本文件 |

对拍脚本统一在 `tests/verifiers/`（不在本目录）：`verify_stream_tokenize.py` /
`verify_vocab_parallel.py` / `verify_ckpt_roundtrip.py` / `verify_csr_equiv.py` /
`verify_ltm_kernels.py` / `verify_accel_readout.py` / `verify_accel_readout_p55.py` /
`verify_ms_stream.py` 等。

---

## 9. 与既有 1B 工具的关系

- `tools/train_1b_capacity.py` —— 容量口径验算工具（本目录 `config_1b.py` 的命令行版）。
- `tools/train_rl.py` —— REINFORCE 强化学习回路，读出即 policy logits，零新增算子
  （用法见该脚本 `--help` 与 `docs/PHD-Net_扩展指南.md`）。
- `tools/estimate_scale.py` —— 生产规模外推。
- 旧 `tools/train_production.py` —— **已删除**，配方仅存于 git 历史，不要引用。

## 10. 硬件需求（`--preset 1b`）

| 项 | 需求 |
|---|---|
| 内存 | 静态 ≈0.5–0.6 GB（读出占大头）+ 大空间表**已生长突触 × ~16 B**（在线 CSR 默认开，内存与表容量无关）。`--no-csr-online` 回 dict 版约 100 B/条 |
| 磁盘 | 检查点数十分之一 GB 量级/份（npz 未压缩，速度优先）；`--ckpt-dtype fp8` 可压到 1/4。`*_final.npz` 见 §6.4 |
| 吞吐 | 10^9 token 级生产训练必须换硬件（本机纯 CPU 实测见日志 `ms/token`；外推见 `tools/estimate_scale.py` 与性能文档） |
| 加速器 | 可选。有昇腾/CUDA 时读出走设备（`--accel auto`），其余部件仍在 CPU——numba 只能编译到 CPU |
