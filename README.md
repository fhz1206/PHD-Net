# PHD-Net

**预测编码 × Hebbian/STDP × 双记忆**：一个完全不依赖自注意力的类脑模型架构，逐 token 流式训练，
学习规则全部局部化（无反向传播），计算事件驱动稀疏（只触碰活跃通路）。
Python + numba + Cython（可选）+ PyTorch/昇腾 NPU（读出加速）。

**唯一生产训练入口是 `train/train.py`。**

适用范围：仓库根入口文档。数据截止 **2026-10-31（P211）**。
文档分工见 `docs/文档写作规范.md`；详细性能数字与迭代历史见
`docs/PHD-Net_性能评估与迭代方案.md`（性能唯一出处）。

---

## 1. 当前状态（P211，2026-10-31）

| 项 | 状态 |
|---|---|
| fast 回归 | `python tests/run_tests.py fast` = **9/9**（零回归门槛） |
| 专项门禁 | `tests/gates/gate_*.py` = **51 个，当前全绿**（数量用 `ls tests/gates/gate_*.py` 现查，别写死） |
| 生产默认 | 读出 fp32（P194）/ `eta_oja=0`（P198）/ SDR 新颖性印迹开（P197）/ `--step-profiling` 默认开（P198）/ `--lang en`（P199 消息目录）/ **前向only 路径开**（P211）/ 稀疏读出 `conn_k=128`（P108）/ 在线 CSR 大空间表（P50） |
| 已知红线 | `eta_oja>0` 生产规模正反馈发散（已默认 0 + RuntimeWarning）；1B 档大词表 × 少 token 曝光不足（见 §9） |

近期迭代主线（均已在本仓库实测验证）：

- **P194**：学习能力修复四件套——`micro_batch` 微批量、表征发育窗口（`eta_pc=0.02` 前 2 万步
  余弦退火后冻结，`pc_dev_steps=20000`）、读出默认改回 **fp32**（精确 p−t 语义）、稳态缩放。
- **P195**：DeepSeek-V4-Flash 主权重模式（`--readout-master-weight`）：fp8 影子前向 + fp32 master
  迭代。本机 A/B（固定探针集）：fp8 无 master PPL **2.84** → 有 master **2.15**
  （fp32 精确基线 **2.14**，差 ≈1%）。默认 off（多付 866.7 MiB fp32）。
- **P196**：`--data cc_traces`（ModelScope `choucisan/mimo-claude-code-traces-1k`，
  1,017 traces / 10 类 / 40.5 MB / MIT，英文代码+推理+工具调用）。
- **P197**：会话中学习——SDR 支撑集新颖性检测使 LTM 印迹触发率 **0.67% → 100%**；
  cc_traces 3,000 token 实测 PPL **2474 → 1308**、LTM 生长 **+81%**。
- **P198**：`eta_oja>0` 会正反馈发散（生产规模 3,000 步 probe_PPL **9.1e11**；`eta_oja=0` → 916），
  已默认 0 并对 >0 发 RuntimeWarning；`generate()` 支持 **top-k / nucleus(top-p)** 截断；
  词表预算诊断工具 `tools/diag_vocab_budget.py`；`--step-profiling` 默认开
  （日志自动带 `loop:` / `segments:` 分段用时）。
- **P199**：i18n 重写为**显式消息目录**（`phdnet/i18n.py::_CATALOG` 整句精确匹配，
  废弃正则词表替换）——`--lang en` 现在真正输出英文核心消息；TeeLogger 建立后重装
  翻译层的顺序 bug 已修。新代码用 `T("中文", "English")`。
- **P210**：**前向only 训练路径**（`--forward-only`，默认关）——调研市面上的
  forward-only 家族（FF / FP / FTP / NoProp / Sigprop），取 **FP**（标签 → 固定随机
  投影目标）+ **FTP**（沿模型自己的权重递推层目标）两条内核，给 **M2 主干**补上
  「任务信号到达中间层」的通道（此前任务信号只到 M6 读出，主干只学无监督重构）。
  **默认关闭**，关闭时训练轨迹逐位不变。**收益未达显著**（smoke 档 probe_PPL
  −0.32% / −2.52%，远低于项目 <20% 判据）→ 定位为「机制已实现、收益待服务器
  1b 档 A/B」的候选。详见 `docs/PHD-Net_前向only训练路径.md`。
  同批修复了一个**既有缺陷**：`--probe-every` 因 `tools/eval/` 不在 `sys.path`
  而从未真正跑通（README §4 规定的 A/B 判据失效）。
- **P211**：**前向only 改为默认路径**（fhz 指示「默认采用 forward-only，旧的非
  forward-only 可以删去」）+ **四类 CPU 优化**（全部逐位不变，final_PPL 恒为
  13,017.891）：
  - **语义反转**：`cfg.forward_only` 默认 True；CLI `--forward-only` 删去，只剩
    `--no-forward-only`。旧路径不再是并列模式，而是 **forward-only 的特例**
    （`--forward-only-eta 0`）—— 门禁 C3b 用训练轨迹摘要**逐位**验证二者等价。
    smoke 档 PPL 13017.891 vs 13355.328（**改善 2.53%**，方向明确为正，但仍
    **低于项目 <20% 判据** → 生产档判据仍需服务器 1b 档 probe_PPL A/B）。
  - **读出访存**：稀疏读出 CSR 列索引 int64 → **int32**（索引流量减半）→
    1B 档形态 CSR 前向 1.29×、更新 1.30×、`learn_softmax` 1.18×。
  - **forward-only 融合核**：6 次独立核调用 → **1 次**（0.110 → 0.045 ms，2.4×），
    表达式逐字照抄原子核 → 逐位等价。
  - **LTM**：dict 版 `table.learn` 3.74 → **1.06 ms（3.5×）**（循环不变的
    `_decay_read` 提循环外 + `_enc` 内联）。
  - **计时口径修复**：`M4b_imprint` 的计时窗嵌套在 `M4b_ltm` 内却被累加两次 →
    `other` 被低估 0.70 ms/tok；已修，各段之和与 step 总时长自洽。
  - ⚠ **诚实边界**：读出已是**访存下界**（对照实验：索引排序后更新核耗时完全
    相同 → 瓶颈是写带宽而非 gather），算法层无提速空间；LTM 的 dict 版收益
    **在生产默认的在线 CSR 路径上不适用**。详见
    `docs/PHD-Net_CPU性能优化实录_P211.md`。

---

## 2. 四条铁律

1. **禁自注意力 / 位置编码 / 堆叠层**——长程依赖由记忆机制承担，不由架构 shortcuts 承担。
2. **每个机制必须有神经认知对应**，不接受「工程上好用但无认知对应」的部件。
3. **逐 token 语义**：状态跨 token / 跨分片 / 跨 epoch 连续演化，读出 W 每步原地更新
   → **不允许批处理**（读出微批量只是 B 步梯度平均后一次提交，主干/记忆仍严格保序）。
4. **行为变更以 config 开关承载，默认关闭**。例外：`csr_online` / `k_sparse` /
   P197 的缺失能力补齐项（`sdr_novelty` / `task_modulation` / `error_gated_memory` /
   `plateau_sleep` / `readout_replay`）为 default-on——它们是**缺陷修复**而非行为偏好。

---

## 3. 六机制速览

| 机制 | 神经认知对应 | 本项目实现 | 稀疏性 |
|---|---|---|---|
| **M1** 稀疏分布式编码 | 皮层稀疏发放 + 侧抑制 WTA | `phdnet/sparse_encoder.py` k-WTA + SDR；1b 档 k_sparse = width/8 = 128（12.5% 活跃） | 12.5% 活跃 |
| **M2** 预测编码主干 | dense representation + sparse connectivity | `phdnet/sparse_pc.py::SparsePCStack`：上行识别 ×2 + 下行生成 ×2，全部 CSR 承载，`conn_k=128` | 12.5% 连接率 |
| **M3** STDP 时序预测 | 稀疏拓扑突触迹 | `phdnet/plasticity.py` + `stdp_kernels.py`，`m_lateral=16` | 1.56% 连接率 |
| **M4a** 工作记忆 | WM 锚定 | `phdnet/wm.py`（`n_wm_slots=4`，内容寻址可选） | — |
| **M4b** 大容量事件驱动稀疏表 | 长程印迹 | `phdnet/bigltm.py` + `sparse_table.py`，2²⁴ × 72 | 事件驱动，随经验生长 |
| **M5** 神经调制 | surprise z 分数门控（多通道可选） | `phdnet/modulator.py` | — |
| **M6** 读出 | 感知器输出层 | `phdnet/readout.py` + `backends/accel_readout.py`，`--readout-conn-k` 默认 **128** | **4.2%** 连接率（128/3072） |

**稀疏性是架构原则，不只是优化**：人脑长程连接率 ~2e-8、平均每神经元约 1744 条突触。
本项目 M2 = 12.5%、M3 = 1.56%、M4b = 事件驱动、M1 k-WTA 12.5%、M6 = 4.2%。
M6 幂律异质连接（`--readout-powlaw-alpha`，k_i ∝ freq^alpha）保留但**默认 0**：
加速器对非均匀行宽 fail-fast（开 alpha>0 回落 numba 读出），且词频目前是代理值 `1/rank`。

### 容量账（1b 档，`--preset 1b`，width=1024，词表 51,962，conn_k=128）

「1B」= **总突触参数容量 ≥ 1.0×10⁹**（容量上限口径，非已用量）。启动时由
`train/config_1b.py::capacity_report()` 打印验算表；词表随数据/采样变化，以实跑为准。

| 组成 | 规模 |
|---|---|
| M4b 大空间事件驱动表 | 2²⁴ 神经元 × 72 出边上限 = **1,207,959,552**（≈99.2%）|
| M6 读出（稀疏 k=128） | **6,651,136**（稠密等价 159,627,264 = 51,962×3,072）|
| M1 编码器 | 2,098,176 |
| M2 主干 CSR / M3 STDP 核 | 524,288 / 16,384 |
| **固定突触小计** | **9,289,984** |
| **总容量** | **≈ 1.217e9** |

⚠ 容量数字必须带 `readout_conn_k` 与词表口径，不可混用。M4b 突触**不是构建即存在**，
而是随经验生长（"fire together, wire together"）；**每步计算量只正比于活跃神经元数**
（事件驱动），与 1B 总容量无关——这是容量与算力解耦的关键。

---

## 4. 快速开始

### 安装

```bash
pip install -r requirements.txt
# 必装：numpy / numba / psutil / pyarrow / Cython；torch 可选（读出加速器路径用）
```

### 冒烟训练（分钟级，只验管线）

```bash
python train/train.py --preset smoke --data sft --tokens 2000
```

### P196：cc_traces 会话中学习（远程直读，零落盘）

```bash
python train/train.py --preset smoke --data cc_traces --remote-data --tokens 3000
```

内容是英文（代码 + 推理 + 工具调用）；`--data-lang zh` 会被自动拦下改为不过滤。
本地副本模式先转换：
`py -3.14 tools/data/cc_traces_to_text.py --out datasets/cc_traces/cc_000.parquet`。

### 英文终端（P199：`--lang en` 真正输出英文）

```bash
python train/train.py --lang en --preset smoke --data sft --tokens 2000
```

`--lang` 只影响终端文案，对训练结果零影响；语料语言过滤是另一个参数 `--data-lang`（默认 all）。

### 生成与评估

```bash
# 生成（P198：top-k 截断，CLI 暴露 --topk 默认 8；库级 generate() 另支持 nucleus top-p）
python train/infer.py --model outputs/models/phdnet1b_1b_pretrain_final.npz --prompt "…" --topk 8

# 交互对话（多轮共享网络状态）
python train/infer.py --model outputs/models/phdnet1b_1b_pretrain_final.npz --chat

# A/B 判据用固定探针集 probe_PPL（冻结 eval_corpus/internal_corpus.txt + 只读前向，零污染）
python train/train.py --preset 1b --data cc_traces --probe-every 1000 --probe-tokens 2048 --tokens 100000
```

⚠ 训练流 sliding PPL 变异系数大（实测信噪比 0.40），**任何 <20% 的差异都测不出来**；
A/B 判定一律用 `--probe-every` 的 `probe_PPL`，且只比差值（探针与训练流 PPL 绝对值不可比）。

### 标准 1B 档训练

```bash
# 本地 datasets/ 入口
python train/train.py --preset 1b --data pretrain --tokens 1000000

# ModelScope 远程流式入口（HTTP Range，零落盘）
python train/train.py --preset 1b --data pretrain --remote-data

# 零 OOV 词表 + 1M context 里程碑 + 断点续训
python train/train.py --preset 1b --data pretrain --vocab-scan full --context-milestone 1000000 --resume

# P195：fp8 影子前向 + fp32 主权重（默认关，显式开启）
python train/train.py --preset 1b --data pretrain --readout-dtype fp8 --readout-master-weight

# P198：词表预算诊断（训练前估算 exposure = tokens / vocab）
python tools/diag_vocab_budget.py --vocab-sample-chars 300000 --tokens 400000
```

### 门禁（改动后必跑）

```bash
python tests/run_tests.py fast        # 9/9
ls tests/gates/gate_*.py              # 47 个专项 verifier，按需跑
```

### 训练时间预估（P204 公式）

**主公式**（串行逐 token 架构，无 batch 维）：

```
T(h) = tokens × ms_per_token / 3,600,000

ms_per_token = B / (f_cpu × e_cpu)  +  S
               └── 可并行部分 ──┘     └ 串行/固定开销
```

| 符号 | 含义 | 1B 档实测锚点（服务器 aarch64 152 核 + NPU） |
|---|---|---|
| `tokens` | 训练 token 预算 | `--tokens N`（默认 ∞，用 `--minutes` 限时） |
| `B` | 单 token 的**串行**计算量（ms，单核口径） | 由 M1-M5 各段的单核耗时决定 |
| `f_cpu` | numba/OpenMP 有效并行度 | `--numba-threads`（**P204 默认 16**） |
| `e_cpu` | 并行效率（屏障/带宽退化） | 1 线程=1.0；**P201 本机 x86 实测**：2-8 线程 0.9–1.0，16 线程 ≈0.6，32 线程 ≈0.45（fork/join 4.7→24.7→64.3 µs/区） |
| `S` | 每 token 的固定开销（读出头往返 + replay 摊销 + 同步） | 见下 |

**`S` 的分解**（P202/P204 实测，服务器 1b 档）：

```
S = readout_npu/npu_overlap  +  replay_ms/replay_every  +  other
    ─────── 8.5% 端到端 ────    ── 可调 ──                ── 未打点小项 ──

实测（P204 默认 replay_every=16）：
  readout  2.04 ms/tok（NPU 异步，与 CPU 重叠 → 实际贡献 ≈ 串行残差）
  M6_replay 5.6/16 ≈ 0.35 ms/tok   （replay_every=4 时是 5.6，占 47%）
  other     ~2.2 ms/tok            （segments 未打点的余额，见 --step-profiling）
```

**1B 档实测量级**（`--preset 1b --data pretrain --remote-data`）：

| 配置 | ms/tok | 1M token 预计 | 10M token 预计 |
|---|---|---|---|
| `--replay-every 4`（P204 前默认） | ~12.3 | 3.4 h | 34 h |
| **`--replay-every 16`（P204 默认）** | **~9.3** | **2.6 h** | **26 h** |
| `--no-readout-replay` | ~8.3 | 2.3 h | 23 h |

**用法**：
1. 跑 2,000 token 热身，从日志 `METRIC` 行读 `ms_per_token`（首行含 JIT 冷编译，**必须用第二行之后**）；
2. `T(h) = tokens × ms_per_token / 3.6e6`；
3. 改配置后按 `S` 的分解线性外推——**只有 `S` 项随开关变化，`B/(f·e)` 不变**。

⚠ **三条纪律**：
- **不要用首行** `segments(cum)`（含 numba 冷编译 + LTM 生长期，实测把稳态 1.28 抬到 19.59 ms/tok）；看 `segments(win)` 与 `METRIC`。
- **ms/tok 与本机无关**：本项目已有**四次 x86 ≠ 昇腾方向相反**的实测，公式里的 `f_cpu/e_cpu` 必须用目标机器实测标定。
- **词表 × 曝光决定能否学会**，不决定时间：`--vocab-sample-chars` 只影响词表大小与 OOV，`ms/tok` 主要由 `n_out`（读出规模）与 `conn_k` 决定（见 `tools/diag_vocab_budget.py`）。


### 只看检查点的大空间表统计

```bash
python train/train.py --preset 1b --data pretrain --report
```

全部开关、默认值与「为什么是这个值」见 `train/README.md`。

---

## 5. 训练档位表（`train/config_1b.py::PRESETS`）

| 档位 | width | conn_k | big_n | big_m | 容量 | 用途 |
|---|---|---|---|---|---|---|
| `smoke` | 256 | 32 | 2²⁰ | 60 | < 1B（⚠ 仅验证管线，属预期日志） | 分钟级冒烟 |
| `1b` | 1024 | 128 | 2²⁴ | 72 | ≈1.217e9（V=51,962 口径） | 标准档 |
| `1b_max` | 4096 | 512 | 2²⁴ | 60 | 大主干档 | 大宽度实验 |
| `30b` | 1024 | 128 | 2²⁹ | 56 | ≈30.3B | 需在线 CSR + 大内存 + 多机预算 |

关键生产默认（全部来自 `train/train.py` argparse 与 `phdnet/config.py`，已核实）：

| 项 | 默认 | 说明 |
|---|---|---|
| `--preset` | `1b` | 档位选择 |
| `--data` | `sft` | `sft` / `pretrain`（全量）/ `pretrain_zh`（已废弃，等价 `pretrain --data-lang zh`）/ `eval` / `cc_traces` |
| `--lang` | `en` | 终端语言（P199 消息目录后 en 真正英文）；数据过滤用 `--data-lang` |
| `--readout-dtype` | `fp32` | P194 Q4：精确 p−t 语义；低精度可显式指定，配 `--readout-master-weight` 解除舍入丢失 |
| `--readout-master-weight` | off | P195：fp32 master + 低精度影子（默认 off，多付 866.7 MiB） |
| `--readout-conn-k` | `128` | 稀疏读出（≈4.2% 连接率）；0 = 稠密 |
| `--micro-batch` | `1` | P194 Q1：读出梯度 B 步平均提交（主干/记忆仍逐 token 保序） |
| `--no-forward-only` | 关（即前向only **开**） | P211：前向only 训练路径（FP+FTP 内核，给 M2 主干补任务信号）。**P211 起默认开启**；`--forward-only-eta 0` = 退回旧行为（逐位等价）；PPL 改善 2.53% 但低于 <20% 判据，见 `docs/PHD-Net_前向only训练路径.md` |
| `--step-profiling` | **开** | P198 起默认开：日志带 `loop:` 与 `segments:` 分段用时；`--no-step-profiling` 关 |
| `eta_oja`（config） | `0.0` | P198：>0 生产规模发散（9.1e11），默认 0 + RuntimeWarning |
| `--cython-kernels` | `auto` | 加载已编译扩展否则就地构建，失败静默回落 numba |
| `--m2-kernel` | `plain` | 昇腾实测比 fused 快 5.41× |
| `--numba-threads` / `--nll-sync-every` | `8` / `8` | 线程上限 / nll 同步周期 |
| `--csr-online` | 开 | 在线可写 CSR（内存 ~5.74× 省） |
| `--ckpt-dtype` | `fp16` | 检查点**存储**精度（位模式存取，与计算精度无关） |
| `--ckpt-every` | `50000` | 每 N token 存检查点（异步，SIGINT 不丢） |
| `--seed` / `--epochs` | `11` / `1` | 状态跨 epoch 连续不重置 |
| `--vocab-scan` / `--vocab-sample-chars` | `head` / `4,000,000` | 词表采样；预算匹配见 §9 |
| `--remote-data` / `--remote-fraction` | 关 / `0.3` | ModelScope HTTP Range 流式，零落盘；前缀子集保序 |

---

## 6. 数据集

### Mixture-General-Mini（通用预训练 / SFT）

| 入口 | 说明 |
|---|---|
| 本地 `datasets/` | 独立 git 仓库（`fhzfhz/Mixture-General-Mini`），`sft/` 与 `pretrain/` parquet 分片，schema `text / lang / src` |
| `--remote-data` | ModelScope HTTP Range 流式直读，**零本地落盘**（仅支持 ModelScope；`--remote-fraction` 是分片级前缀抽样，不是随机采样） |

### cc_traces（P196，会话中学习测试集）

- ModelScope `choucisan/mimo-claude-code-traces-1k`：1,017 traces / 10 任务类别 / 40.5 MB / MIT。
- **event-stream JSONL**（非 parquet）→ 专用读取路径 `phdnet/corpus.py::iter_cc_traces`：
  只抽语言内容、按 event 顺序还原对话，保留 `thinking` 块。
- 远程模式两级目录 `session/<category>/*.jsonl` 由 train.py 按 10 类逐个展开。
- 内容是**英文**；对拍门禁 `tests/gates/gate_cc_traces.py`（23 项，含远程实读）。

### 冻结评测基准（仓库根 `eval_corpus/`）

`internal_corpus.txt` **27,034 字符**（21,627 训练 / 5,407 评估）+ `ood_wiki.txt` **20,187 字符**（远域探针）。
不入训练集；`--probe-every` 探针用的就是这份冻结文本。

---

## 7. 任务导向文档地图

| 我要做的事 | 看这份 |
|---|---|
| **跑一次训练** | `train/README.md`（快速开始 + 开关表） |
| 搞懂六机制为什么这么设计、脑同构论证 | `docs/PHD-Net_架构设计.md` |
| **看性能数字 / 为什么吃不满多核** | `docs/PHD-Net_性能评估与迭代方案.md`（性能唯一出处） |
| 快速查现状 / 已定案结论 | `docs/PHD-Net_现状速查.md` |
| 加新机制 / 扩硬件后端 | `docs/PHD-Net_扩展指南.md`、`docs/PHD-Net_硬件后端适配报告.md` |
| 了解 forward-only（无反向传播）训练路径 | `docs/PHD-Net_前向only训练路径.md`（FF/FP/FTP/NoProp/Sigprop 调研 + P210/P211 实现 + A/B） |
| 查 CPU 侧提速的实测与边界 | `docs/PHD-Net_CPU性能优化实录_P211.md`（P211：int32 索引 / 融合核 / LTM 内联 / 计时口径 + **访存下界的诚实边界**） |
| 与 Transformer / 大脑对比 | `docs/PHD-Net_竞争力与脑同构性评估.md` |
| 查一个 bug 的症状→根因→门禁缺口 | `BUGS.md`（53 条） |
| 制备语料 / 数据集入口 | `tools/README.md`、`datasets/README.md` |
| 对外一页式介绍 | `docs/index.html` |
| 写文档要遵守的规则 | `docs/文档写作规范.md` |

目录速览：

```
train/          生产训练子项目（train.py / config_1b.py / corpus_stream.py / infer.py / README.md）
phdnet/         核心包（config / sparse_encoder / sparse_pc / stdp_kernels / plasticity / wm /
                ltm / bigltm / sparse_table / modulator / readout / model / word_encoder /
                word_lm / corpus / ms_stream / i18n / device / forward_only / backends/）
eval_corpus/    冻结评测基准 + OOD 探针
tests/          run_tests.py（fast 9 项）+ gates/（48 个 gate_*.py）
tools/          数据制备 / 评测 / 诊断（diag_vocab_budget.py、data/cc_traces_to_text.py 等）
docs/           架构、性能、扩展、后端、竞争力、现状速查
chat/           对话前端（tui.py，--selftest 可无模型走通 UI）
datasets/       训练语料（独立仓库，gitignore）
BUGS.md         缺陷台账（症状→根因→修复→门禁缺口）
```

---

## 8. 精度与平台口径

- **读出**：默认 **fp32**（P194）。P110 反证在案：低精度把感知器 p−t 非目标行更新
  （|dp|≈1e-6）舍掉 → 学习退化为纯 Hebbian（fp16 保留率仅 26.67%）。
  要低精度访存收益请配 `--readout-master-weight`（P195）：fp8/bf16/fp16 影子只做前向，
  所有更新落在 fp32 master（非目标行更新 100% 保留），每 `--readout-shadow-refresh`（默认 8）步
  重量化；检查点对 master 存 fp32 精确值逐位往返。
- **910B4 无原生 fp8 算子**（ERR01007 实测）：fp8 走「位模式（uint8 承载）存储 +
  CPU 转换 int8 计算」，`--fp8-conv` 选 torch/nogil 核。
- **M1 编码器**：生产 CLI `--encoder-dtype` 默认 **fp32**；库 config 默认 fp64（两个入口需区分）。
  昇腾 aarch64 走 numba GEMV 核，x86 走 BLAS——**x86 与昇腾的加速方向常常相反，不可混用**。
- **M2/M3/M4a**：fp32（P173 定案）。
- **ms/tok 必带平台与档位**：现行历史锚点是 P113 的 **12.09 ms/tok**（1b 档，昇腾 191 核 + NPU）。
  本机（x86，无 NPU）只能冒烟验证管线；昇腾端到端数字均来自服务器实测。
- P192 的 STDP 学习率语义修正改变了学习行为：**历史 PPL 锚点不再可比**，
  数字一律以《性能评估与迭代方案》§10.1 的记录为准，重测前不得引用旧锚点。

---

## 9. 已知边界（诚实）

1. **1B 档 PPL 仍高**：词表 51,962 × 少量 token = 每词曝光不足，probe_PPL 长期偏高。
   P198 词表预算诊断给出经验阈值：`exposure = tokens / vocab`，<30 严重不足、≥100 才充足。
2. **「正常对话」未达到**：1B 档生成 bigram 合法率仅 **24–35%**（小词表 + 小预算可到 78%，
   但那是 3,392 词的目标空间）。要接近正常对话需要 **≥10⁸ token 量级**的曝光，
   这不是算法问题而是数据量问题。
3. **`eta_oja>0` 是已知危险开关**：生产规模正反馈发散（3,000 步 probe_PPL 9.1e11）。
   库默认 0 且 `__post_init__` 发 RuntimeWarning；小规模机制测试可显式开启并自担。
4. **低精度默认是 fp32 而不是 fp8/fp16**：P110 的舍入反证未被推翻；P195 主权重模式
   解除了它，但要多付一份 fp32 master（1b 档 866.7 MiB）且引入 ≤N 步影子 staleness。
5. **幂律读出未接入加速器**：`--readout-powlaw-alpha>0` 回落 numba（fail-fast 是正确行为），
   且词频是 `1/rank` 代理值。与人脑 ~2e-8 连接率仍差约 6 个数量级。
6. **历史评测锚点不可比**：P192/P193 修正了 STDP 学习率语义（LTD 迹缺 λ^Δt 补偿），
   学习规则层面的语义变更 → 旧锚点只存于 git 历史；新基线待重建。
7. **CI**：GitCode 与 GitHub 两套 workflow（`.gitcode/workflows/ci.yml` 镜像 `.github/`），
   仓库**没有 Jenkinsfile**；GitHub 侧历史状态红，待复核。
8. **平台诚实**：本机无 NPU，所有昇腾端到端收益均为服务器历史实测；Cython 核
   （`--cython-kernels auto`）无昇腾端到端收益实测，且开后与 numba 有 fp32 级微小漂移
   （fastmath vs 严格 IEEE），**不得称逐位等价**。

---

## 10. 术语口径

- **逐 token**：一次前向 = 一个 token，状态连续演化。批处理在本架构下是语义错误。
- **容量 vs 已用量**：容量是突触上限（`--report` / `[big-ltm]` 日志查利用率），已用量是已生长数。
- **稠密 / 稀疏连接率**：不存在的突触**不存储、不计算**。
- **ms/tok**：单 token 毫秒数，必带平台与档位。
- **probe_PPL**：固定冻结探针集上的只读 PPL，A/B 判定的唯一可信口径；与训练流 PPL 绝对值不可比。
- **终端语言**：`--lang` 默认 `en`（P199 消息目录），`--lang zh` 中文；只影响文案不影响数据。
  语料过滤是 `--data-lang`。

写作规则见 `docs/文档写作规范.md`；新增训练脚本不进 `tools/`，一律进 `train/`。
