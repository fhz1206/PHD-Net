# PHD-Net

**预测编码 × Hebbian/STDP × 双记忆**：一个完全不依赖自注意力的类脑模型架构，逐 token 流式训练，
Python + numba + PyTorch/昇腾 NPU。

学习规则全部局部化（无反向传播），计算事件驱动稀疏（只触碰活跃通路）。
**唯一生产训练入口是 `train/train.py`。**

适用范围：仓库根入口文档。数据截止 **2026-09-30**。
文档分工见 `docs/文档写作规范.md`；本文件不写详细性能数字与历史。

---

## 1. 四条铁律

1. **禁自注意力 / 位置编码 / 堆叠层**——长程依赖由记忆机制承担，不由架构 shortcuts 承担。
2. **每个机制必须有神经认知对应**，不接受「工程上好用但无认知对应」的部件。
3. **逐 token 语义**：状态跨 token / 跨分片 / 跨 epoch 连续演化，读出 W 每步原地更新
   → **不允许批处理**。这是语义约束，不是性能取舍。
4. **行为变更以 config 开关承载，且默认关闭**。例外仅两个：`csr_online` / `k_sparse` default-on。

---

## 2. 六机制速览

| 机制 | 神经认知对应 | 本项目实现 | 稀疏性 |
|---|---|---|---|
| **M1** 稀疏分布式编码 | 皮层稀疏发放 + 侧抑制 WTA | `phdnet/sparse_encoder.py` k-WTA + SDR | 输出 **12.5%** 活跃 |
| **M2** 预测编码主干 | dense representation + sparse connectivity | `phdnet/sparse_pc.py` 结构性稀疏 CSR，4 层 | **12.5%** 连接率 |
| **M3** STDP 时序预测 | 稀疏拓扑突触迹 | `phdnet/plasticity.py` + `stdp_kernels.py`，`m_lateral=16` | **1.56%** 连接率 |
| **M4a** 工作记忆 | WM 锚定 | `phdnet/wm.py` | — |
| **M4b** 大容量事件驱动稀疏表 | 长程印迹 | `phdnet/bigltm.py` + `sparse_table.py`，2²⁴ × 72 | 事件驱动，随经验生长 |
| **M5** 神经调制 | surprise z 分数门控 / 四通道 | `phdnet/modulator.py` | — |
| **M6** 读出 | 感知器输出层 | `phdnet/readout.py` | **100% 稠密 = 架构欠账** |

**稀疏性是架构原则，不只是优化**：人脑长程连接率 ~2e-8、平均每神经元约 1744 条突触。
本项目 M2 = 12.5%、M3 = 1.56%、M4b = 事件驱动、M1 输出 k-WTA 12.5%。

> **明确的架构欠账**：**M6 读出目前 100% 稠密**，是全项目唯一没有稀疏化的部件，
> 与人脑 2e-8 连接率差约 4 个数量级。幂律稀疏化方案（读出按幂律分布稀疏到 5–12% 活跃度）
> **已设计、未实施**。在它落地前，任何「本架构稀疏度已达标」的说法都不成立。
> 详见 `docs/PHD-Net_架构设计.md`。

### 容量账（1B 档，`--preset 1b`，width=1024）

「1B」= **总突触参数容量 ≥ 1.0×10⁹**（容量上限口径，非已用量）。启动时由
`train/config_1b.py::capacity_report()` 打印验算表。

| 组成 | 规模 | 占比 |
|---|---|---|
| M4b 大空间事件驱动表 | 2²⁴ 神经元 × 72 出边上限 = **1.208e9** | **≈ 88%** |
| M6 读出（稠密） | 词表 × `n_h`（pred_in_readout 时 3×1024） | **≈ 11.6%** |
| M1 编码器（稠密） | 组合输入 2048 → 1024 + 偏置 | **≈ 0.16%** |
| M2 主干 CSR / M3 STDP 核 | 4 层 × 1024 × 128 / 1024 × 16 | 计入稀疏侧 |

**总容量 ≈ 1.370e9**（1.317e9 是 `big_ltm_m=60` 时代的旧值，已作废）。M4b 突触**不是构建即存在**，而是随经验生长（"fire together, wire together"）；
`count_params` 计的是**已生长突触数**，训练初期利用率低。**每步计算量只正比于活跃神经元数**
（事件驱动），与 1B 总容量无关——这是容量与算力解耦的关键。

---

## 3. 任务导向文档地图

**我要做什么 → 看哪份。**

| 我要做的事 | 看这份 | 别看 |
|---|---|---|
| **跑一次训练** | `train/README.md`（快速开始 + 开关表） | — |
| 搞懂六机制为什么这么设计、脑同构论证 | `docs/PHD-Net_架构设计.md` | 本文档 |
| **看性能数字 / 为什么吃不满多核** | `docs/PHD-Net_性能评估与迭代方案.md`（性能唯一出处）<br>`docs/PHD-Net_并行与加速架构分析.md`（吃不满的论证） | 本文档 |
| 加新机制 / 扩硬件后端 | `docs/PHD-Net_扩展指南.md`<br>`docs/PHD-Net_硬件后端适配报告.md` | — |
| 与 Transformer / 大脑对比 | `docs/PHD-Net_竞争力与脑同构性评估.md` | — |
| 查一个 bug 的症状→根因→门禁缺口 | `BUGS.md` | — |
| 制备语料 / 数据集入口 | `tools/README.md`、`datasets/README.md` | `docs/*.md`（**不是数据集**） |
| 对外一页式介绍 | `docs/index.html` | — |
| 写文档要遵守的规则 | `docs/文档写作规范.md` | — |

**冻结评测基准在 `eval_corpus/`**（仓库根）：`internal_corpus.txt` **27,034 字符**（实测原文长度）、
`ood_wiki.txt` 远域探针。`docs/*.md` 是文档，**不是数据集**。

---

## 4. 当前状态摘要（2026-09-30）

### 精度口径

- 读出 `--readout-dtype` 默认 **fp16**。动机是**学习精度**不是访存收益：bf16 半 ULP≈2e-4
  ≫ 非目标行更新 |dp|≈1e-6 → 更新被舍 → **退化为纯 Hebbian**；fp16 保住更新。
- **fp8/fp4 在所有加速器上已禁用**：实测昇腾 Ascend910B4 + CANN 8.5 + torch_npu 2.9 对
  float8_e4m3fn/e5m2 的建张量/cast/matmul 全部 ERR01007，`torch.ops` 零个 fp8 算子；
  CPU torch 能建张量但无 fp8 addmv → **numba CPU 路径仍可用量化码本**（`--accel cpu --readout-dtype fp8`）。
- M1 编码器 `--encoder-dtype` 默认 **fp64**（昇腾 aarch64 的 numpy GEMV 病态慢 → 平台自适应走
  自写 numba 核；x86 走 BLAS）。
- 分词 onehot 缓冲 fp16（无损，1B 词表 208→104 KB/步）。
- 检查点对低精度**存位模式**（uint16/uint8），加载侧按 `meta["ckpt_dtype"]` 无损解码。

### 性能区间（1B 档，昇腾 191 核 + NPU）

- **20–110 ms/tok**；读出 5.5–7 ms/tok，**访存受限**（960 MB/步 = 读 320 + 读 320 + 写 320，
  反算有效带宽 137–175 GB/s ≈ HBM 的 9–11%）。
- **191 核用不满是架构性的**：逐 token 串行 + 每步可并行工作量太小（核内 1→6 线程仅 1.16×）。
- 完整数字与平台差异铁律见 `docs/PHD-Net_性能评估与迭代方案.md`。

### 门禁与 CI

- `python tests/run_tests.py fast` = **9/9**（零回归门槛）；专项 verifier **18** 个（`tests/verifiers/`）。
- 语法门禁：`verify_ms_stream` 会对改动文件 `py_compile` 并真跑 `train.py --help`。
- CI **两套**：`.gitcode/workflows/ci.yml`、`.github/workflows/ci.yml`（镜像）。
  仓库**没有 Jenkinsfile**。GitHub 侧当前仍红（无日志权限，待 traceback）。

### 评测基线

- 口径：256 维栈、`eta_readout=0.15`。现行锚点 **4K 段 394.4687 / 全语料 359.2603**
  （评测语料 2026-09-28 换为中文维基条目合集）。
- 复测入口 `python tools/rebaseline.py`（**无命令行参数**）。
- ⚠ 旧口径 78.16 / 90.2480 / 73.1166 **只存于 git 历史，不可混用**。

### 已知开放项

M4b 复测归因 / fp16 下 PPL 是否下降 / 数据集语种（远程分片实测中文仅 2%、unk 47% **是代码问题**，
与旧记录矛盾）/ 检查点 `compact_csr` 向量化后仍 3.9 s/10 万行 /
**M6 幂律稀疏化（架构欠账）** / GitHub Actions 仍红。

---

## 5. 快速开始

### 最小可跑（冒烟，分钟级，只验管线）

```bash
pip install -r requirements.txt

python train/train.py --preset smoke --data sft --tokens 2000
```

### 门禁

```bash
python tests/run_tests.py fast
```

### 标准 1B 档训练

```bash
# 本地 datasets/ 入口
python train/train.py --preset 1b --data pretrain --tokens 1000000

# ModelScope 远程流式入口（HTTP Range，零落盘）
python train/train.py --preset 1b --data pretrain --remote-data
```

### 推理 / 对话

```bash
python train/infer.py --model outputs/models/phdnet1b_1b_pretrain_final.npz --chat
```

**全部开关、默认值与「为什么是这个值」见 `train/README.md`。**

### 训练数据两入口

| 入口 | 说明 |
|---|---|
| 本地 `datasets/` | gitignore 的独立仓库（`fhzfhz/Mixture-General-Mini`），由 `tools/` 转换脚本生成 |
| `--remote-data` | ModelScope HTTP Range 流式直读，**零本地落盘**（仅支持 ModelScope） |

---

## 6. 目录结构

```
train/
├── train/                   1B 生产训练子项目（唯一训练入口）
│   ├── train.py                生产训练主入口（流式 / SFT / 续训 / 远程数据）
│   ├── config_1b.py            1B 档预设与容量验算
│   ├── corpus_stream.py        字符流 → token 流、SFT 掩码状态机
│   ├── vocab_parallel.py       词表并行构建 / 全量扫描
│   ├── tokenizer_core.py       分词核心
│   ├── ckpt_1b.py              检查点存取（异步 / 位模式 / 词表快照）
│   ├── infer.py                推理 / 对话
│   └── README.md               训练目录说明（开关表在此）
├── phdnet/                     核心包
│   ├── config.py               PHDNetConfig（全部超参与开关；新增默认关闭）
│   ├── sparse_encoder.py       M1 稀疏编码器（k-WTA / SDR）
│   ├── sparse_pc.py            M2 预测编码 CSR 栈
│   ├── stdp_kernels.py         M3 STDP 计算核（numba + numpy 回退）
│   ├── plasticity.py           M3 时序关联核容器
│   ├── wm.py / ltm.py          M4a 工作记忆 / M4b 长期记忆
│   ├── bigltm.py               M4b 容量层适配器
│   ├── sparse_table.py         1B 印迹表 + 在线可写 CSR
│   ├── modulator.py            M5 神经调制器
│   ├── readout.py              M6 读出头（局部感知器 + softmax + 精度码本）
│   ├── model.py                组装与调度（PHDNet.step）
│   ├── word_encoder.py         M7 词涌现（无词典分词）+ 词级 SDR
│   ├── word_lm.py / lm.py      词级 / 字符级语言模型
│   ├── ms_stream.py            ModelScope HTTP Range 流式读取
│   ├── device.py               后端探测（cuda / cann·昇腾 / rocm / dml / cpu）
│   ├── i18n.py                 终端文案语言切换（zh / en）
│   ├── telemetry.py            系统 / 设备遥测
│   └── backends/               硬件后端子包（accel_readout / multi_device / …）
├── eval_corpus/                冻结评测基准（27,034 字符）+ OOD 探针
├── tests/                      回归门禁（run_tests.py）+ demo + verifiers/
├── tools/                      语料制备 / 评测审计 / 硬件诊断（索引见 tools/README.md）
├── docs/                       架构、扩展、性能、并行、后端、竞争力评估
├── chat/                       对话前端（TUI / OpenAI 兼容 / websearch 工具）
├── outputs/                    产物（test/ experiments/ 入库；smoke/ models/ 不入库）
├── datasets/                   训练语料（独立仓库，gitignore）
├── BUGS.md                     缺陷台账
└── requirements.txt
```

---

## 7. 术语口径

- **逐 token**：一次前向 = 一个 token，状态连续演化。批处理在本架构下是语义错误。
- **容量 vs 已用量**：容量是突触上限（`--report` / `[big-ltm]` 日志查利用率），已用量是已生长数。
- **稠密 / 稀疏连接率**：不存在的突触**不存储、不计算**。
- **ms/tok**：单 token 毫秒数，必带平台与档位。x86 与昇腾的加速方向常常相反，不可混用。
- **终端语言**：`--lang zh`（默认）终端全中文、`--lang en` 全英文，只影响文案不影响数据。

写作规则见 `docs/文档写作规范.md`；新增训练脚本不进 `tools/`，一律进 `train/`。
