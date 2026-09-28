# PHD-Net（v0.0.0）

**预测编码 × Hebbian/STDP × 双记忆**：一个完全不依赖自注意力（Self-Attention）的类脑模型架构。

两条铁律：① 无自注意力、无位置编码、无堆叠层；② 每个机制都必须有神经认知对应。
学习规则全部局部化（无反向传播），计算事件驱动稀疏（只触碰活跃通路）。

> 2026-09-20 终态：五轨道 M1–M9 全部收官（每项都有实测结论，含负结果）；四份文档与介绍网页已按最终状态重写。
> 2026-09-21：**内置语料冻结**为 `eval_corpus/internal_corpus.txt`（23,504 字符，逐字复制自架构设计文档）——编辑文档不再改变基线；
> `data/` 更名 `datasets/`（下分 `eval/ sft/ pretrain/`）；中文维基预训练语料按指令删除（外部锚点暂停）。**预训练语料已拍板（2026-09-22）：Infinity-Instruct `7M_core`**（M7_Core，atomgit `BAAI/Infinity-Instruct`；75 parquet 分片 6.1 GB，实测 750 万条对话 / 全量文本 ~10.2 GB；`tools/convert_infinity.py` 流式转换落 `datasets/pretrain/infinity_m7core.txt`）——已就位：7,449,106 条 / 10.07 GB 文本；语言分布 en 89.9% / zh-cn 10.1%（约 75 万条中文），中文子集经 `langdetect` 字段筛出（约 1 GB 文本，为原维基语料的 ~19 倍）。
> **SFT 语料已就位（2026-09-23）**：OpenBMB `UltraInteract_sft`（fhz 指定）——288,579 条 / **0.603 GB 文本**，
> 77,023 条唯一指令（平均 3.75 响应/指令，含偏好树 `parent_id`）；任务构成 Coding 39.8% / Math 56.2% / Logic 4.0%；
> `tools/convert_ultrainteract.py` 转换落 `datasets/sft/ultrainteract_sft.txt`（flat / grouped 两模式）。
>
> **2026-09-24 三条更新**：① **性能 P7**——稠密读出更新改为 numba 融合并行核，
> `tools/_prof_step.py` 实测 11.373 → 5.053 ms/token（**2.25×**），三层对拍**逐位等价**
> （`tests/verifiers/verify_readout_fused.py`），基线 96.7241 / 77.5261 **一字未变**，fast 11/11；
> ② **新增两个数据集**——匠数 `deepctrl-sft-data`（中文 SFT，流式过滤后 150 MB）
> 与 `Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B`（R1 长链 CoT 300 MB，
> 结构判定归到预训练）；叠加 `Ultra-FineWeb-L3` 中文采样（8 万条 / 102 MB parquet）；
> ③ **语料导入改为 parquet**（`phdnet/corpus.py` 统一 txt/parquet/jsonl 读取入口），
> 压缩 1.91–6.26×，且都 <100 MiB 可直接入库。

---

## 目录结构

```
train/
├── phdnet/                     核心包
│   ├── config.py               PHDNetConfig（全部超参与开关；新增一律默认关闭）
│   ├── sparse_encoder.py       M1 稀疏编码器（k-WTA / SDR）
│   ├── stdp_kernels.py         M3 STDP 计算核与自检（numba + numpy 回退）
│   ├── plasticity.py           M3 时序关联核容器（STDPCore）
│   ├── wm.py / ltm.py          M4a 工作记忆 / M4b 长期记忆
│   ├── memory.py               M4 双记忆兼容再导出（旧引用仍可用）
│   ├── modulator.py            M5 神经调制器（surprise z 分数门控 / 四通道 MultiModulator）
│   ├── readout.py              M6 读出头（局部感知器 + softmax）
│   ├── model.py                M6 组装与调度（PHDNet.step / sleep）
│   ├── cognition.py            认知层：语义图 / 情景缓冲 / 联想链
│   ├── generate.py             M11 生成解码器（WM 锚定 + 温度采样）
│   ├── tokenizer.py            字符 → SDR 确定性哈希
│   ├── word_encoder.py         M7 词涌现（无词典分词）+ 词级 SDR + 上下文绑定
│   ├── word_lm.py              T2 词级语言模型（PHDWordLM）
│   ├── ngram.py                n-gram 基线族（字符级 / 词级）
│   ├── lm.py                   字符级语言模型（PHDNetLM）
│   ├── sparse_table.py         M4 容量层本体（1B 突触容量印迹表）
│   ├── bigltm.py               容量层适配器（SparseLTM）
│   ├── context_memory.py       M13 上下文漂移情景记忆（TCM/CMR 式长程复制）
│   ├── device.py               硬件后端探测（昇腾 NPU / ROCm / CUDA / DirectML / CPU）
│   └── torch_backend.py        torch 版 STDP 算子（同一份代码覆盖各加速器）
├── datasets/                   语料（**自身是独立仓库** → atomgit.com/fhz1206/Mixture-General-Mini）
│   ├── eval/                   内置语料（仅本地评测用，按 fhz 要求不随数据集仓库外发）
│   │   ├── internal_corpus.txt 内置冻结语料（23,504 字符，逐字复制自架构设计文档）
│   │   └── ood_wiki.txt        远域泛化探针语料（中文维基 26 篇 / 20,188 字符）
│   ├── sft/merged/split/       SFT 合并语料 parquet 分片（≤90 MiB，UltraInteract 英文 + deepctrl 中文，331.9 万条）
│   ├── pretrain/merged/split/  预训练合并语料 parquet 分片（≤90 MiB；= M7_Core 10.84 GB + Magpie-R1 CoT + Ultra-FineWeb-L3 中文采样）
│   └── （散装原始件在本地保留：infinity_m7core.txt 10.84 GB 等；读取统一走 phdnet/corpus.py）
│   ⚠ 远端单文件 ≤100 MiB 且 LFS 配额满 ⇒ 交付形态 = 按大小切分的 **parquet 分片**（tools/split_parquet.py，每片可直接 pyarrow 读取）
├── tests/                      回归入口与全部验收脚本
│   ├── run_tests.py            分层回归入口（fast 11 项 / --full）
│   ├── checks_common.py        检查项共享件（常量/断言/结果收集）
│   ├── checks_core.py          核心行为检查
│   ├── checks_backend.py       硬件后端检查
│   ├── eval_suite.py           M5 评测入口（编排 + 判定矩阵）
│   ├── eval_common.py          评测共享件（语料路径/公共配置/Transformer 辅助）
│   ├── eval_tasks_lm.py        任务 1 语言建模 / 任务 2 长程依赖
│   ├── eval_tasks_memory.py    任务 3 单样本记忆 / 4 持续学习 / 5 多跳推理
│   ├── eval_tasks_perf.py      任务 6 效率
│   ├── nano_gpt.py             对照模型：nanoGPT 级 Transformer（PyTorch）
│   ├── demo_phdnet.py          三项基础实验（序列 / 补全 / 单样本关联）
│   ├── demo_lm.py              P1 字符级 LM 验收
│   ├── demo_m1.py … demo_m4.py M1–M4 里程碑消融
│   ├── demo_m9.py              M9 五轨道全量消融（15 项 + 容量赛道）
│   ├── demo_gen.py             泛化探针（域内/近域/远域 + OOV 惩罚口径；--strength 为加强组）
│   ├── demo_strength.py        架构强化验收（稳态缩放 / 读出退火 / 错误触发检索）
│   ├── demo_copy.py            M13 长程复制（延迟复制任务 + 双向校验对比）
│   ├── demo_corpus.py          外部语料验收（中文维基上的词级 LM）
│   ├── demo_v2.py              认知层四实验
│   └── demo_dev.py / demo_replay.py / demo_multihop.py
│                               P3 自动发育 / P4 生成式回放 / P5 多跳检索
├── tools/                      工程工具（数据制备 / 基准 / 审计；验证脚本在 tests/verifiers/）
│   ├── bench.py                CPU/内存基准（优化前后对照）
│   ├── train_1b.py             10 亿突触容量模型在线训练（玩具 demo，非生产入口）
│   ├── fetch_modelscope.py     ModelScope 数据集文件下载（断点续传，实测 ~5 MB/s）
│   ├── prepare_wikicn.py       中文维基 JSONL 清洗 + 确定性划分（→ datasets/pretrain/；本轮用于 ood_wiki.txt 抽取）
│   ├── backend_probe.py        硬件后端体检（昇腾 / ROCm / CUDA / DirectML）
│   └── prepare_mimo.py / prepare_toolcall.py / prepare_mix.py / prepare_r1sft.py / train_r1sft.py
│                               旧语料制备与训练（数据集已按指令删除，脚本仅留档）
├── chat/                       对话与推理入口（fhz 2026-09-28 目录规范）
│   ├── chat_openai.py          OpenAI 风格对话循环 + web_search 工具调用
│   ├── chat_r1sft.py           R1 SFT 模型对话 / 演示续写
│   ├── websearch_tool.py       联网搜索工具（OpenAI function calling 封装）
│   └── run_demo.py / run_demo_external.py   「训练 + 对话」演示
├── docs/                       文档与介绍网页
│   ├── PHD-Net_架构设计.md                     总设计（基础架构 + 认知层 + 终态状态）
│   ├── PHD-Net_对标Transformer优化路线图.md     瓶颈定位 / 五轨道结论 / 开放项
│   ├── PHD-Net_竞争力与脑同构性评估.md          与 LLM 对比 / 100B vs GLM-5.3-Flash / 脑同构
│   ├── PHD-Net_性能评估与迭代方案.md            CPU·RAM 量化与 P1–P6/五轨道结果账
│   └── index.html                              架构介绍网页（打开即用）
├── outputs/                    运行产物（fhz 2026-09-28 分类规范）
│   ├── test/                   测试产物：CI 回归 / verify 对拍 / audit 审计 / demo 日志（入库）
│   ├── experiments/            实验与基准产物：experiment / scaling / bench / eval_suite（入库）
│   ├── smoke/                  smoke 档产物（模型 + 断点，不入库）
│   └── models/                 生产训练产物：模型检查点 + train_logs/（不入库）
└── .workbuddy/                 工作区记忆与日志（勿删）
```

## 快速开始

```bash
# 环境：Python 3.14（numpy / numba / psutil / torch 均可选，缺失自动回退）
python tests/run_tests.py          # 快速回归（11 项，约 40 秒）—— 零回归门槛
python tests/run_tests.py --full   # 全量验收（约 35 分钟）

python tests/demo_m2.py            # 词级 LM 消融（M2）
python tests/demo_m9.py            # 五轨道全量消融（M9）
python tests/eval_suite.py         # 六任务评测 + Transformer 对照 + 判定矩阵（M5）
python tests/demo_copy.py          # 长程复制（M13）
python tools/backend_probe.py      # 硬件后端体检
python tools/train_1b.py           # 10 亿突触容量训练

# 泛化探针（域内 / 近域 / 远域 + 统计基线对照）
python tests/demo_gen.py                    # 基准组（→ outputs/test/demo_gen.log）
python tests/demo_gen.py --strength         # 加强组（T3.4 内容寻址 + ② 回放稳定读出）
```

子目录脚本自带 `sys.path` 引导，**从任意工作目录运行都可**；内置语料为冻结副本
`eval_corpus/internal_corpus.txt`（23,504 字符，逐字复制自架构设计文档 2026-09-21 版）。

> ✅ **语料已与文档解耦（2026-09-21）**：全部评测一律读 `eval_corpus/internal_corpus.txt`，
> 编辑 `docs/PHD-Net_架构设计.md` **不再**改变语料与基线（原「语料即文档，编辑即漂移」条款作废）。
> 若有意更换语料：更新冻结副本 → 重跑 `eval_suite.py` + `demo_m9.py` → 回填全部文档数字。
>
> ℹ️ **网络事实（2026-09-20 实测）**：本机对境外源严重受限（wikimedia ~10 KB/s、huggingface 不可达），
> 国内源可达且快（ModelScope ~5 MB/s、清华 PyPI ~5 MB/s），故语料通道走 ModelScope。

## 关键实测（v0.0.0，2026-09-21 重测）

> ⚠ **两组数字不可混用**：`eval_suite`（默认 **256 维栈**）与 `demo_m9`（**128 维小栈**）网络宽度与分词粒度都不同。
> 下面按口径分开列。所有数值在冻结语料 `eval_corpus/internal_corpus.txt`（**23,504 字符**）上测得——编辑文档不再影响基线。

**A 组 · 评测套件口径（默认 256 维栈，训练 18,803 / 评估 4,701 字符，冻结语料 23,504 字符）**

| 指标 | 数值 |
|---|---|
| 内置语料 LM 基线（字符归一 PPL） | **78.16**（bpc 6.288；学习曲线 0.25/0.50/0.75/1.00 → 108.4 / 84.3 / 114.7 / 78.2） |
| Transformer 对照 0.52M（96d×2L, 527,236 参数） | 85.27（bpc 6.414；token PPL 1157.19，训练 1.0 s / CPU 4.1 s） |
| Transformer 对照 2.38M（192d×4L, 2,385,028 参数） | 85.66（bpc 6.421；token PPL 1165.76，训练 4.7 s / CPU 18.2 s） |
| 套件判定 | **相当（达标）**（主任务低 8.3%，3 项结构化任务占优） |
| 长程复制（默认栈，n=4/8/16） | 10.1% / 11.0% / 10.0%（Transformer 5.7% / 9.7% / 8.6%） |
| LM 路径效率 | **6.96 ms/token**（P6 热路径优化 2026-09-21，修复前 12.84；逐位等价验证 + fast 11/11，详见《性能评估》§6.4），可塑参数 1,712,388 |
| 泛化探针（2026-09-21 新增） | 近域四文档 inflation 0.68–1.05×（优）；远域维基 OOD 9.38×（不足，OOV 58.5% 覆盖墙）；见 `tests/demo_gen.py` |

**B 组 · M9 消融口径（128 维小栈，参照基线 R = 102.97，bpc 6.686）**

| 指标 | 数值 |
|---|---|
| 稳定正增益 | T3.4 内容寻址 WM **101.29（−1.6%）**；② 回放稳定读出 **101.48（−1.4%）**（② 四轮均为增益：−0.8% → −1.34% → −1.5% → −1.4%） |
| 符号翻转（不可依赖） | T3.2 top-k 检索 **106.24（+3.2%，有害）**——历史 −2.4% → −0.65% → −1.1% → +3.2%，**不推荐启用** |
| 表征可塑性五项 | 全部为负（+18.4% ~ +62.9%），跨 M1/M8/M9 多轮一致 |
| 组合 | 不叠加；排序对语料敏感，勿据单次数字下结论 |

**C 组 · 与口径无关的结构性指标**

| 指标 | 数值 |
|---|---|
| 持续学习遗忘率 BWT | **0.000** vs Transformer **−1.000** |
| 单样本记忆 | 100%（Transformer 原生不具备） |
| 长程复制（启用 M13 后，n=4/8/16） | **45.6% / 25.9% / 17.8%**（随机 8.3%） |
| 10 亿突触端到端单步 | **1.28 ms**（CPU；numba 核加速 5.8–51.5×） |
| 1B 训练态内存 | 197 MB（稠密对照 ≥16 GB，本机不可运行） |
| 回归状态 | `run_tests.py` fast **11/11**、**full 19/19 通过** |
| 瓶颈/开放项收口（2026-09-23） | **O4** 分词向量化 **36.9×**｜**O1/O3** 在线可写 CSR（逐位等价 12/12 PASS、内存 5.74×↓、1B 档验证）｜**O6** 消融（词涌现 **+177.5%** / 稀疏 k-WTA **+84.0%** / **M5 调制器 −4.2% 净负担**）｜**O7** 瓶颈在数据预算（数据轴 4k→16k −11.2% vs 参数轴 1.71M→17M +22.4%）｜**O2** 表征可塑 10 配置实测**仍不可解锁**｜**O5** 再入连接 −1.88%（κ=0.2 时整段 exact +50%；vote/多步清理有害）｜**B7** minibatch 单遍协议下有害 |
| O1 双栈统一（2026-09-23 完成） | **容量栈合并**：LM 栈可挂 1B 事件驱动容量栈（`big_ltm`），**ppl 95.48（−1.83%）**、参数 1.58M、容量上限 **1.0B 突触**｜**主干结构性稀疏连接**：`sparse_conn`（CSR 稀疏图 + numba），**连接率 3.1% 时 PPL 持平（+0.03%）、突触存储压缩 32×**，k=16 时 −0.51%、PC 层加速 **1.68–4.02×**，与稠密栈数值等价（≤1.1e-15）｜读出学习率默认 0.05 经网格实测**次优**（0.2 时 −4.4%~−7.3%，待确认后统一切换） |
| 脑同构审计（2026-09-23） | `tools/audit_brain_parity.py` **12 项结构性指标**逐项对照生物学事实（不以"模块存在"判对齐）。**三轮演进：一致 5/近似 6/偏离 1 → 6/6/0 → 10/2/0**。本轮把"近似"逐项改造对齐：**A5**（STDP 拓扑原判为误判，实为结构性稀疏 `(n,m)`）｜**A6** 激活稀疏度 k=16（6.2%）→ **−1.43%** 更优｜**A9** 独立抑制类群 抑制比 0.1 → **−1.17%** 正增益｜**A7** 重新判读（全架构无全局反传，监督源自自监督任务定义）｜**A4 新增两级群体读出**（速度 **6.8×**、参数 −60%，PPL +10~13% → 默认关闭）｜**A3** 可学习编码器实测有害（+33%）→ 确定性哈希为有据选择。默认路径 ppl 仍 **97.2596**、fast **11/11** |
| Bug 猎手（2026-09-23） | `tools/hunt_bugs.py`：编译 85 文件 0 错误 / 31 组开关矩阵冒烟 / 边界输入 / 接口一致性 → **45 项通过、0 bug**。本轮修复：维度契约 fail-fast（原为模糊 numpy 广播错误）、新增 `net.n_out` 查询接口 |
| 生产级训练（2026-09-23） | 原 `tools/train_production.py`（规模预设 / 断点续训 / 检查点 / 双预算 / 滑动 PPL）——**2026-09-28 功能重复治理已删除**：其为 `train_1b/train.py` 的严格子集（无 mix / 无 vocab-scan / ckpt 不支持稀疏读出恢复），训练入口单轨化为 `train_1b/train.py`。本机实测 4M 档（git 历史可复现）：**7.43M 参数、17.38 ms/token、PPL 607.9→341.9 持续下降**。**256M 生产模型本机不可行**（算力差 ~3 个数量级：157 ms/token ⇒ 10⁹ token ≈ 5 年；详见性能评估 §6.10）——内存可行（2.05 GB）、质量与数据量均差 1–2 个数量级；投产需 GPU 集群 + 10⁹ token 数据 + torch 后端接入 |
| **稀疏默认开启（2026-09-24）** | fhz 指令「稀疏要默认开启」→ `sparse_conn=True`、`k_sparse=16`（激活 6.25%）已设为**库默认**。**新基线**（评测脚本显式传 k_sparse=32）：4,000 字符口径 **96.7241**（原 97.2596，**−0.55%**）｜全语料口径 **77.5261**（原 78.4180，**−1.14%**）｜主干连接率 **12.5%**（32,768/262,144 突触）、存储压缩 **8×**。⚠ 期间发现并修复**初始化量纲错误**：稀疏权重原用 `1/√n_in`，按 fan-in 应为 `1/√k`（否则前向幅值偏低 √(k/n)≈0.35×）——修正前全语料口径 +37.6% 劣化，修正后 **−1.14% 更优**（同一错误也存在于稀疏读出，已一并修复） |

**诚实边界**：内置语料仅约 2 万字符，结论只在该规模与语料下成立，**不构成对通用 LLM 能力的宣示**；
远域（维基 OOD）受词表覆盖墙限制（OOV 58.5%），属数据规模问题而非架构单一问题；
旧外部维基锚点（300.66 vs 687.55）已随维基预训练语料按指令删除而暂停，**新预训练语料已就位**
（Infinity-Instruct 7M_core / M7_Core：7,449,106 条 / 10.07 GB，中文子集 ~75 万条 / ~1 GB）；
昇腾 NPU 与 ROCm 路径已完成代码适配与接口探测，但**未在对应硬件上实测**。
规模外推（100B vs GLM-5.3-Flash）为**估算**，见《竞争力与脑同构性评估》§二。

## 文档索引

- 架构与设计 → `docs/PHD-Net_架构设计.md`
- 对标 Transformer 的五轨道结论与开放项 → `docs/PHD-Net_对标Transformer优化路线图.md`
- 与 LLM 的竞争力 / 100B 预估 / 与人脑的同构性 → `docs/PHD-Net_竞争力与脑同构性评估.md`
- CPU/RAM 量化与迭代结果账 → `docs/PHD-Net_性能评估与迭代方案.md`
- 一页式介绍 → `docs/index.html`

## CI/CD 与硬件后端

- **测试流水线**：回归入口 `tests/run_tests.py`（fast 19 项）+ 逐位对拍 `tests/verifiers/`；
  CI：`.gitcode/workflows/ci.yml`（GitCode Action）+ `Jenkinsfile`（GitCode Jenkins）+ `.github/workflows/ci.yml`（GitHub 镜像）。
- **硬件后端**：CUDA / ROCm / CANN·NPU / CPU 适配与基准见《docs/PHD-Net_硬件后端适配报告》；
  探针与基准入口 `tools/bench_accel.py`。
- **读出精度**：`--readout-dtype`（fp32 默认 / fp16 / bf16 / fp8 / fp4；fp64 已停止支持）；
  逐位与质量验证 `tools/audit_precision.py`。
- **泛化评测**：`tools/audit_gen_eval.py`（域内 held-out / 近域 / 远域三域 + 2-gram 无泄漏基线）；
  混合域训练 `--data mix`（sft + pretrain 中文子集样本级轮转）。
- **治理约定**：架构介绍文档（docs/*.md）不是数据集；冻结评测基准位于 `eval_corpus/`。
