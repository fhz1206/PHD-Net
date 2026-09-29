# PHD-Net（v0.0.0）

**预测编码 × Hebbian/STDP × 双记忆**：一个完全不依赖自注意力（Self-Attention）的类脑模型架构。

两条铁律：① 无自注意力、无位置编码、无堆叠层；② 每个机制都必须有神经认知对应。
学习规则全部局部化（无反向传播），计算事件驱动稀疏（只触碰活跃通路）。

主干唯一实现为**结构性稀疏 CSR**（`SparsePCStack` + numba 核，连接率 12.5% 起步，
稠密栈已移除）；读出支持 fp32 / fp16 / bf16（fp64 已停用，fp8 已移除）；
容量层支持 1B 突触印迹表（`big_ltm`）。生产路径为 numba CPU 主循环
（PC 栈 / STDP / 双记忆 / 分词，机制齐全，支持 1M 上下文状态不重置）+
NPU/CUDA 加速读出（M6，`phdnet/backends/accel_readout.py::AccelReadout`）；
torch 旧栈（TorchPCStack/TorchWordLM/TorchReadout/TorchSTDPCore）已于 2026-09-29 全部删除。

## 目录结构

```
train/
├── phdnet/                     核心包
│   ├── config.py               PHDNetConfig（全部超参与开关；新增一律默认关闭）
│   ├── sparse_encoder.py       M1 稀疏编码器（k-WTA / SDR）
│   ├── stdp_kernels.py         M3 STDP 计算核与自检（numba + numpy 回退）
│   ├── plasticity.py           M3 时序关联核容器（STDPCore）
│   ├── wm.py / ltm.py          M4a 工作记忆 / M4b 长期记忆
│   ├── memory.py               M4 双记忆兼容再导出
│   ├── modulator.py            M5 神经调制器（surprise z 分数门控 / 四通道 MultiModulator）
│   ├── readout.py              M6 读出头（局部感知器 + softmax + 精度码本）
│   ├── model.py                组装与调度（PHDNet.step / sleep）
│   ├── cognition.py            认知层：语义图 / 情景缓冲 / 联想链
│   ├── generate.py             M11 生成解码器（WM 锚定 + 温度采样）
│   ├── tokenizer.py            字符 → SDR 确定性哈希（全项目哈希基础设施）
│   ├── word_encoder.py         M7 词涌现（无词典分词）+ 词级 SDR + 上下文绑定
│                               （向量化后 4.7M 字符 63.1s → 9.0s，7.00×，逐位一致）
│   ├── word_lm.py              词级语言模型（PHDWordLM，评测主路径）
│   ├── ngram.py                n-gram 基线族（字符级 / 词级）
│   ├── lm.py                   字符级语言模型（PHDNetLM）
│   ├── sparse_table.py         容量层本体（1B 突触印迹表 + 在线可写 CSR）
│   ├── bigltm.py               容量层适配器（SparseLTM）
│   ├── context_memory.py       M13 上下文漂移情景记忆（TCM/CMR 式长程复制）
│   ├── device.py               硬件后端探测（cuda / cann·昇腾 NPU / rocm / dml / cpu）
│   └── backends/               硬件后端子包（accel_readout 读出加速 AccelReadout /
│                               multi_device 多卡列并行 / torch_backend 设备探针+基准 /
│                               torch_lm 仅存 resolve_device）
├── train_1b/                   1B 生产训练子项目（唯一训练入口）
│   ├── train.py                生产训练（多数据源 / 断点续训 / 多进程加载 / --readout-dtype /
│   │                           --accel auto / --devices auto 多卡读出列并行；
│   │                           词表四来源：resume 自包含 / 快照自动复用 / --vocab-file / head-full 扫描，
│   │                           词表快照自动落盘 outputs/models/）
│   ├── config_1b.py            1B 预设（big_ltm 2^24×60 + 稀疏 CSR/STDP 主干，总参 ≥1B）
│   ├── ckpt_1b.py              完整可续训检查点（含 CSR 快照 / Welford / STDP 迹）
│   ├── infer.py                推理与对话（--chat；检查点自包含加载；词表快照自动交叉校验）
│   ├── corpus_stream.py        流式语料（多进程 PrefetchChars + zh 过滤 + mix 轮转）
│   └── vocab_parallel.py       词表 / 分词多核构建（与串行逐位一致；
│                               33.4 亿 tokens / 520s @ 191 核）
├── datasets/                   语料（**自身是独立仓库** → atomgit.com/fhz1206/Mixture-General-Mini）
│   ├── sft/                    SFT 分片 parquet（≤100 MiB；331.9 万条）
│   ├── pretrain/               预训练分片 parquet（≤100 MiB；3,906 万条）
│   └── raw/                    原始件归档（不入库；读取统一走 phdnet/corpus.py）
├── eval_corpus/                冻结评测基准（internal_corpus.txt 23,504 字符 + OOD 探针；不入训练集）
├── tests/                      回归与验收
│   ├── run_tests.py            分层回归入口（fast 9 项 / --full）
│   ├── checks_core.py          核心行为检查
│   ├── checks_backend.py       硬件后端检查（--accel auto 设备解析 / 回落）
│   ├── verifiers/              逐位对拍验证脚本（分词 / 词表 24 例 / CSR / 读出加速核 /
│   │                           多卡 / RL / 检查点 round-trip 等 13 项）
│   ├── eval_suite.py           M5 评测入口（编排 + 判定矩阵）
│   ├── eval_tasks_*.py         六任务评测（LM / 长程 / 记忆 / 持续学习 / 多跳 / 效率）
│   ├── nano_gpt.py             对照模型：nanoGPT 级 Transformer（PyTorch）
│   └── demo_*.py               各里程碑消融与验收（M1–M9 / 泛化探针 / 长程复制 / 认知层）
├── tools/                      工程工具（数据制备 / 基准 / 审计）
│   ├── train_rl.py             RL 训练（REINFORCE；零新增算子；JSONL 提示集 + 可插拔 reward_fn）
│   ├── accel_doctor.py         加速后端一次性诊断（环境矩阵 / 试分配 / 带宽）
│   ├── bench_accel.py          读出基准（fp32 / fp16 / bf16 三档 + 等效带宽 GB/s；P29 修正版）
│   ├── rebaseline.py           基线复测（锚点对照）
│   ├── audit_precision.py      精度体系审计（L1 逐位 / L2 带宽 / L3 端到端）
│   ├── audit_gen_eval.py       泛化评测（域内 / 近域 / 远域三域 + 2-gram 无泄漏基线）
│   ├── audit_brain_parity.py   脑同构审计（12 项结构性指标对照生物学事实）
│   ├── hunt_bugs.py            开关矩阵冒烟 + 边界扫描
│   └── prepare_* / convert_* / fetch_* / pack_* / split_*    语料制备供应链
├── chat/                       对话与推理入口
│   ├── chat_openai.py          OpenAI 风格对话循环 + web_search 工具调用
│   ├── chat_r1sft.py           R1 SFT 模型对话 / 演示续写
│   ├── websearch_tool.py       联网搜索工具（OpenAI function calling 封装）
│   └── run_demo.py / run_demo_external.py   「训练 + 对话」演示
├── docs/                       文档与介绍网页（见文末索引）
├── outputs/                    运行产物
│   ├── test/                   测试产物：CI 回归 / 对拍 / 审计 / demo 日志（入库）
│   ├── experiments/            实验与基准产物（入库）
│   ├── smoke/                  smoke 档产物（模型 + 断点，不入库）
│   └── models/                 生产训练产物：模型检查点 + train_logs/（不入库）
├── .gitcode/workflows/ci.yml   GitCode Action 流水线
└── .github/workflows/ci.yml    GitHub Actions 镜像
```

## 快速开始

```bash
# 环境：Python 3.13+；依赖安装 pip install -r requirements.txt
# （numpy/numba/psutil/pyarrow；torch 可选——读出加速 cuda/npu/rocm/dml 与
#   Transformer 对照模型所需，缺失自动回落 numba CPU，默认路径逐位不变）
python tests/run_tests.py fast    # 快速回归（9 项，约 1 分钟）—— 零回归门槛
python tests/run_tests.py --full  # 全量验收（约 35 分钟）

python tests/eval_suite.py        # 六任务评测 + Transformer 对照 + 判定矩阵
python tests/demo_gen.py          # 泛化探针（域内 / 近域 / 远域 + 统计基线对照）
python tools/rebaseline.py        # 基线锚点复测（4K / 全语料双口径）
python tools/backend_probe.py     # 硬件后端探测（设备矩阵）
python tools/accel_doctor.py      # 加速后端一次性诊断（环境矩阵 / 试分配 / 带宽）
python tools/bench_accel.py       # 读出基准（fp32 / fp16 / bf16 三档 + 等效带宽 GB/s）

# 预训练（唯一生产入口；多进程加载 / 断点续训 / 精度可选 / --accel auto）
python train_1b/train.py --preset smoke --data sft --tokens 100000
python train_1b/train.py --data mix --resume   # sft + pretrain 中文子集混合

# SFT（回复掩码：只对助手回复计损失，多轮约 50% 步；--init-from 两阶段微调）
python train_1b/train.py --data sft --assistant-marker "助手：" \
    --init-from outputs/models/<预训练检查点>.npz

# RL（REINFORCE，零新增算子；JSONL 提示集 + 可插拔 reward_fn）
python tools/train_rl.py --init-from outputs/models/<SFT 检查点>.npz

# 推理 / 对话（词表快照自动交叉校验）
python train_1b/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz --chat

# 硬件验证（P30 后保留项）
python tests/verifiers/verify_accel_readout.py    # 读出加速逐位对拍
python tests/verifiers/verify_multi_device.py     # 多卡分片与单设备逐位一致（24 例）
python tests/verifiers/verify_vocab_parallel.py   # 词表并行构建逐位一致（24/24）
python tests/verifiers/verify_rl.py               # RL 训练回路
python tests/verifiers/verify_ckpt_roundtrip.py   # 检查点落盘 / 恢复 round-trip
```

子目录脚本自带 `sys.path` 引导，从任意工作目录运行都可。
评测一律读冻结语料 `eval_corpus/internal_corpus.txt`（编辑 docs 不影响基线）；
语料通道走 ModelScope（境内源 ~5 MB/s，境外源受限）。

## 当前基线与关键指标

> ⚠ 两组口径不可混用：`rebaseline`（256 维 BASE 栈，词级 LM 锚点）与
> `eval_suite` / `demo_m9`（各自独立配置）。以下均为冻结语料（23,504 字符）实测。

**现行权威锚点**（BASE 配置：n_sdr=256 / k_sparse=32 / eta_pc=0 / eta_readout=0.15 / 读出 fp32；
复测入口 `tools/rebaseline.py`；语料 = 中文维基高质量条目合集 27,405 字符，2026-09-28 更换——
旧口径 90.2480/73.1166 为自指文档语料，天然偏乐观，仅存于 git 历史）：

| 口径 | ppl_char | bpc | 吞吐 | 主干连接率 |
|---|---|---|---|---|
| 4,000 字符训练段 | **394.4687** | 8.624 | ~2.9 ms/token | 12.5% |
| 全语料（21,924 字符） | **359.2603** | 8.489 | ~2.5 ms/token | 12.5% |

**结构性能力**（与口径无关）：

| 指标 | 数值 |
|---|---|
| 持续学习遗忘率 BWT | **0.000**（Transformer 对照 −1.000） |
| 单样本记忆 | 100%（Transformer 原生不具备） |
| 长程复制（启用 M13，n=4/8/16） | **45.6% / 25.9% / 17.8%**（随机 8.3%） |
| 10 亿突触端到端单步 | **1.28 ms**（CPU；numba 核加速 5.8–51.5×） |
| 1B 训练态内存 | **197 MB**（稠密对照 ≥16 GB，本机不可运行） |
| 稳定正增益机制 | T3.4 内容寻址 WM −1.6%；回放稳定读出 −1.4% |
| 已证伪方向（勿复用） | 表征可塑性五项全负（+18%~+63%）；T3.2 top-k 检索 +3.2% 有害；minibatch 单遍协议下有害 |
| 脑同构审计 | 12 项结构性指标：一致 10 / 近似 2 / 偏离 0（`tools/audit_brain_parity.py`） |
| 工程吞吐（实测） | 词涌现构建 4.7M 字符 63.1s → 9.0s（7.00×，逐位一致）；词表扫描 33.4 亿 tokens / 520s（191 核） |
| 回归状态 | fast **9/9**（零回归门槛）；CI 三平台同源 |

**诚实边界**：内置语料仅约 2 万字符，结论只在该规模与语料下成立，不构成对通用 LLM 能力的宣示；
远域（维基 OOD）受词表覆盖墙限制，属数据规模问题而非架构单一问题；
读出加速已上设备并实测：NPU 迁移后设备流量 **−40%**（4,334 → 2,600 MiB/步，正好理论下限）；
GEMV 受带宽限制，**~50% 利用率是结构性上限**；主循环机制（PC 栈 / STDP / 记忆 / 分词）仍在 numba CPU；
fp16 有机制性代价，待真机实测（`tools/bench_accel.py` 三档精度基准）；
规模外推（100B vs GLM-5.3-Flash）为估算，见《竞争力与脑同构性评估》。

## 文档索引

- 架构与设计 → `docs/PHD-Net_架构设计.md`
- 对标 Transformer 的五轨道结论与开放项 → `docs/PHD-Net_对标Transformer优化路线图.md`
- 与 LLM 的竞争力 / 100B 预估 / 与人脑的同构性 → `docs/PHD-Net_竞争力与脑同构性评估.md`
- CPU/RAM 量化与迭代结果账 → `docs/PHD-Net_性能评估与迭代方案.md`
- 硬件后端适配 → `docs/PHD-Net_硬件后端适配报告.md`
- 一页式介绍 → `docs/index.html`

## CI/CD

- 回归入口 `tests/run_tests.py`（fast 9 项）+ 逐位对拍 `tests/verifiers/`。
- CI 双平台：`.gitcode/workflows/ci.yml`（GitCode Action）+ `.github/workflows/ci.yml`（GitHub 镜像）。
- 治理约定：架构介绍文档（docs/*.md）不是数据集；冻结评测基准位于 `eval_corpus/`。
