# PHD-Net（v0.0.0，更新 2026-09-29）

**预测编码 × Hebbian/STDP × 双记忆**：一个完全不依赖自注意力（Self-Attention）的类脑模型架构。

两条铁律：① 无自注意力、无位置编码、无堆叠层；② 每个机制都必须有神经认知对应。
学习规则全部局部化（无反向传播），计算事件驱动稀疏（只触碰活跃通路）。

主干唯一实现为**结构性稀疏 CSR**（`SparsePCStack` + numba 核，连接率 12.5% 起步，
稠密栈已移除）；读出支持 fp32 / fp16 / bf16（fp64 已停用，fp8 已移除）；
容量层支持 1B 突触印迹表（`big_ltm`）。生产路径为 numba CPU 主循环
（PC 栈 / STDP / 双记忆 / 分词，机制齐全，支持 1M 上下文状态不重置）+
NPU/CUDA 加速读出（M6，`phdnet/backends/accel_readout.py::AccelReadout`）；
torch 旧栈（TorchPCStack/TorchWordLM/TorchReadout/TorchSTDPCore）已于 2026-09-29 全部删除。

**默认精度与开关（P46/P47/P50/P44）**：训练计算默认 **bf16**（`--readout-dtype bf16`，
fp32 可回退——半精度会舍掉感知器非目标行更新，启动日志有机制警告）；
检查点存储默认 **bf16**（`--ckpt-dtype bf16`，位模式压缩 fp8/bf16/fp16 可选，加载无损解码）。
大空间表在线 CSR **默认开**（`--csr-online` 默认 True，`--no-csr-online` 回退 dict）：
内存 = 已生长突触 × ~16 B，与容量无关。NPU 读出 kernel 融合**默认开**
（torch_compile=True，`--no-torch-compile` 关闭；模式默认 `default`——cudagraph 与
W 原地更新冲突，reduce-overhead 会打印 skipping cudagraphs 并回退）。
M2 PC 推理已切换 numba 融合核 `_pc_infer_fused`（nogil+parallel+fastmath，
5 次 matvec 融成 1 次、中间数组核内复用；n_steps=1 **2.10×** / n_steps=3 **2.16×**，
容差一致 1 ulp（非逐位），已记录待裁决）。

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
│   ├── train.py                生产训练（两阶段 pretrain→sft / 断点续训 / 多进程加载 /
│   │                           --readout-dtype bf16 默认 / --ckpt-dtype bf16 默认 /
│   │                           --nll-sync-every NPU 读出重叠 / torch.compile 融合默认开 /
│   │                           --accel auto / --devices auto 多卡读出列并行；
│   │                           词表四来源：resume 自包含 / 快照自动复用 / --vocab-file / head-full 扫描，
│   │                           词表快照自动落盘 outputs/models/；运行时 print 全英文，help 保留中文）
│   ├── config_1b.py            预设族（smoke / 1b / 1b_max / 30b；30b = big_n 2^29 × big_m 56
│   │                           → 30.29B 参数，依赖 --csr-online 稀疏——稠密 113 GiB 不可行，
│   │                           速度随 LTM 访存线性放大）
│   ├── ckpt_1b.py              完整可续训检查点（含 CSR 快照 / Welford / STDP 迹；bf16 位模式默认）
│   ├── infer.py                推理与对话（--chat；检查点自包含加载；词表快照自动交叉校验）
│   ├── corpus_stream.py        流式语料（多进程 PrefetchChars；P43 起两阶段三源：
│   │                           pretrain 中英全量 3906 万块无 lang 过滤 / pretrain_zh 仅中文对照 / sft；
│   │                           mix 轮转已删除；SFT 回复掩码）
│   └── vocab_parallel.py       词表 / 分词多核构建（与串行逐位一致；
│                               33.4 亿 tokens / 500s @ 191 核）
├── datasets/                   语料（**自身是独立仓库** → atomgit.com/fhz1206/Mixture-General-Mini，
│   │                           限额 1 GiB < 交付 4.64 GiB 推不全；权威副本在 ModelScope
│   │                           fhzfhz/Mixture-General-Mini，`--remote-data` 可直读；
│   │                           本机副本已于 2026-09-29 按指令删除：307 文件 / 32.28 GiB
│   │                           （其中 parquet 64 个 / 9.62 GiB），未被 git 追踪 →
│   │                           只能从 ModelScope 重新下载/重建。删除时的数据源构成：
│   │                           sft = UltraInteract(288k) + Magpie-R1 CoT(40,619)；
│   │                           pretrain = M7_Core 中文 3,697 万 + Magpie-R1 英文 201 万
│   │                           + Ultra-FineWeb-L3 中文 8 万。逐文件明细见 git 历史
│   │                           `170ce0c^:docs/DATASETS_DELETED_MANIFEST.json`）
│   ├── sft/                    SFT 分片 parquet（≤100 MiB；331.9 万条）
│   ├── pretrain/               预训练分片 parquet（≤100 MiB；3,906 万条 = 中文 3708 万 + 英文 Magpie-R1 201 万）
│   └── raw/                    原始件归档（不入库；读取统一走 phdnet/corpus.py）
├── eval_corpus/                冻结评测基准（internal_corpus.txt 27,034 字符 + OOD 探针；不入训练集）
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
│   ├── fetch_ms.py             三源统一采样器（web/code/math；`--plan web=3,code=2,math=1`；
│   │                           ModelScope 流式零原始落盘，断点续跑；fetch_l3.py 能力已并入、保留）
│   └── prepare_* / convert_* / pack_* / split_*    语料制备供应链其余件
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

# 预训练 / SFT（唯一生产入口；多进程加载 / 断点续训 / --accel auto）
python train_1b/train.py --preset smoke --data sft --tokens 100000

# 阶段① 预训练：中英全量 3906 万块（中文 3708 万 + 英文 Magpie-R1 201 万，无 lang 过滤；
#          `mix` 轮转已删除）。默认 `--nll-sync-every 8`（nll 设备侧累积，CPU/NPU
#          重叠）+ `--numba-threads 8`（191 核上空转的 prange 线程上限，0=不限）
python train_1b/train.py --data pretrain --resume

# 语言选择（fhz 2026-09-29）：--lang zh / en / all（默认 all 逐位不变）；
#          `--data pretrain --lang zh` 等价旧 `--data pretrain_zh`；sft 同样适用
python train_1b/train.py --data pretrain --lang zh --resume

# 服务器无本地数据集时：ModelScope 直读（HTTP Range 流式零落盘，仅支持 ModelScope；
#          --remote-fraction 0.3 = 取排序后前 30% 分片，前缀子集顺序语义不变）
python train_1b/train.py --data pretrain --remote-data --remote-fraction 0.3 --resume

# 阶段② SFT：回复掩码——只对助手回复计损失（多轮约 50% 步）；
#          默认 bf16 计算 + bf16 检查点存储
python train_1b/train.py --data sft --assistant-marker "助手：" \
    --init-from outputs/models/phdnet1b_1b_pretrain.npz

# ⚠ 读出精度与学习能力（2026-09-29 实测）：bf16 读出会把非目标行更新
#   （|dp|≈1e-6）舍入掉（bf16 半 ULP≈2e-4）→ 学习退化为纯 Hebbian，PPL 震荡
#   不降。要精确学习规则就用 fp32 计算 + bf16 存储（模型文件仍是半精度）：
python train_1b/train.py --data pretrain --remote-data --readout-dtype fp32 \
    --ckpt-dtype bf16 --resume

# 30B 档（big_n 2^29 × big_m 56 → 30.29B；依赖默认开启的 --csr-online 稀疏）
python train_1b/train.py --preset 30b --data pretrain --resume

# 云端语料制备：三源统一采样器（web/code/math；零原始落盘，断点续跑；
#          三源 L3 合计 994.3 GB，50 GB 云端预算下采样 3–5%）
python tools/fetch_ms.py --plan web=3,code=2,math=1

# RL（REINFORCE，零新增算子；JSONL 提示集 + 可插拔 reward_fn）
python tools/train_rl.py --init-from outputs/models/<SFT 检查点>.npz

# 推理 / 对话（词表快照自动交叉校验）
python train_1b/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz --chat

# 硬件验证（P30 后保留项）
python tests/verifiers/verify_accel_readout.py    # 读出加速逐位对拍
python tests/verifiers/verify_accel_readout_p55.py  # 读出 2×2 矩阵（target_idx × compiled）9/9
python tests/verifiers/verify_pc_learn_fused.py  # M2 学习融合核逐位对拍 12/12 + 多规模计时
python tests/verifiers/verify_ms_stream.py       # ms:// 远程源纯函数 41 例（零网络）
python tests/verifiers/verify_multi_device.py     # 多卡分片与单设备逐位一致（24 例）
python tests/verifiers/verify_vocab_parallel.py   # 词表并行构建逐位一致（24/24）
python tests/verifiers/verify_rl.py               # RL 训练回路
python tests/verifiers/verify_ckpt_roundtrip.py   # 检查点落盘 / 恢复 round-trip
```

子目录脚本自带 `sys.path` 引导，从任意工作目录运行都可。
评测一律读冻结语料 `eval_corpus/internal_corpus.txt`（编辑 docs 不影响基线）；
**训练数据两个入口（P54，2026-09-29）**：① 本地 `datasets/`（默认，逐位不变）；
② `--remote-data` 直读 ModelScope `fhzfhz/Mixture-General-Mini`——**HTTP Range
流式零落盘（仅支持 ModelScope）**，`--remote-fraction` 取排序后前 N% 分片
（默认 0.3，前缀子集保证词表与训练流一致）。atomgit 限额 1 GiB < 交付 4.64 GiB，
故数据集仓库推不全，服务器走 ②。语料采样通道同样走 ModelScope（境内源 ~5 MB/s）。
numba 编译缓存持久化于 `outputs/numba_cache`（`NUMBA_CACHE_DIR`，不被
`__pycache__` 清理波及；readout.py 7 个核带 cache=True，冷启动 3.96→2.60s）。

## 当前基线与关键指标

> ⚠ 两组口径不可混用：`rebaseline`（256 维 BASE 栈，词级 LM 锚点）与
> `eval_suite` / `demo_m9`（各自独立配置）。以下均为冻结语料（27,034 字符原文长度）实测。

**现行权威锚点**（BASE 配置：n_sdr=256 / k_sparse=32 / eta_pc=0 / eta_readout=0.15 / 读出 fp32；
复测入口 `tools/rebaseline.py`；语料 = 中文维基高质量条目合集 27,405 字符，2026-09-28 更换——
旧口径 90.2480/73.1166 为自指文档语料，天然偏乐观，仅存于 git 历史）：

| 口径 | ppl_char | bpc | 吞吐 | 主干连接率 |
|---|---|---|---|---|
| 4,000 字符训练段 | **394.4687** | 8.624 | ~2.9 ms/token | 12.5% |
| 全语料（27,034 字符原文长度） | **359.2603** | 8.489 | ~2.5 ms/token | 12.5% |

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
| 工程吞吐（实测） | 词涌现构建 4.7M 字符 9.0s（7.00×，逐位一致）；词表扫描 33.4 亿 tokens / 500s（191 核） |
| 1B 训练步时（**服务器实测** 2026-09-29，昇腾 191 核 + ModelScope 30% 分片） | **26 ms/tok**（区间 26–42）；其中 **readout 12.9–13.4 ms/tok 在 NPU 上占 30–50%**（320 MB bf16 ×2 访存 ≈ 49 GB/s，远低于 NPU 带宽 → **同步/launch 暴露，非算力瓶颈**）；进程仅用 **1.3–3.2 / 191 核**、CS/s 250 万+ |
| P62 同步/空转修复 | `--nll-sync-every` 默认 1→**8**（每步 `.item()` 曾把 NPU 延迟全额暴露给 CPU =「CPU 忙 NPU 空闲」真相）；`--numba-threads` 默认 **8**（191 核跑千行级 prange 纯空转；P22 实测 1→6 线程仅 1.16×） |
| M2 学习侧融合核（P59） | `SparsePCStack.learn` 8 次核调用 → 1 个 nogil/parallel 核，**逐位一致**；26 万边 **1.72×** / 105 万边 1.15× / 419 万边 1.25×。（predictive 版实测**负收益** 0.91× → 已删） |
| M1 编码器半精度（P61） | 迭代 fp32 + 模型 fp32（`cfg.encoder_dtype`）：encode **606.7→337.7 µs（1.80×）**，top-k 逐位一致。⚠ numpy 路径下 bf16 存储**更慢**（每次付上采样转换），bf16 语义由读出侧（NPU 原生）承担 |
| H2D/图优化（P58/P63） | 读出 h 走 pinned 暂存 + `non_blocking`（pageable 会阻塞 CPU）；`--torch-compile` **默认关**（inductor 服务器不稳定）；遥测 NPU% 走 `npu-smi`（`torch.npu.utilization()` 会同步设备流）——**三级定位** `NPU_SMI_PATH` → PATH → 常见安装路径（非交互 shell 没有 set_env.sh 时也能读到） |
| M2 numba 融合核（P52） | `phdnet/sparse_pc.py::_pc_infer_fused`：n_steps=1 **2.10×** / n_steps=3 **2.16×**，容差一致 1 ulp（非逐位） |
| torch.compile A/B（P45） | 开 **26.34** vs 关 **30.98** ms/tok（快 15%；本机 CPU torch）。**现默认关**，需要时显式 `--torch-compile` |
| numba 编译缓存（P39） | readout 7 核 cache=True + 持久化缓存目录，冷启动 **3.96→2.60s** |
| CPU 侧预计算（P51） | `encode_composite`/`onehot` 缓冲预分配复用（onehot 14.12→0.33 μs）；诚实口径：占端到端 **0.069%**，大头在 M2 推理（状态依赖无法预计算） |
| 回归状态 | fast **9/9**（零回归门槛）；`verify_ms_stream` 41/41、`verify_accel_readout_p55` 9/9、`verify_pc_learn_fused` 12/12、`verify_ckpt_roundtrip` PASS；CI 三平台同源 |

**诚实边界**：内置语料仅约 2 万字符，结论只在该规模与语料下成立，不构成对通用 LLM 能力的宣示；
远域（维基 OOD）受词表覆盖墙限制，属数据规模问题而非架构单一问题；
读出加速已上设备并实测：NPU 迁移后设备流量 **−40%**（4,334 → 2,600 MiB/步，正好理论下限）；
GEMV 受带宽限制，**~50% 利用率是结构性上限**（逐 token 串行语义，批处理才能突破）；
⚠ **bf16 读出有机制性代价**：非目标行更新（|dp|≈1e-6）被 bf16 半 ULP（≈2e-4）舍掉 →
学习退化为纯 Hebbian，实测 PPL 震荡不降 → 要精确规则用 `--readout-dtype fp32`
+ `--ckpt-dtype bf16`（**存储仍半精度**）；
⚠ **M3/M4 STDP、LTM 不是瓶颈**（`n_top×16` 边、µs 级）——优化前务必先看
`--step-profiling` 九段分解；本轮两次「看起来是热点」的优化实测为**负收益**
已回滚（LTM 表 Python 循环向量化 0.73×、`learn_predictive` 融合 0.91×）；
fp16 有机制性代价，待真机实测（`tools/bench_accel.py` 三档精度基准）；
规模外推（100B vs GLM-5.3-Flash）为估算，见《竞争力与脑同构性评估》。

**训练数据规模（三源实测）**：L3 430.6 GB / 2.02 亿篇；Code-L3 441.1 GB（11 语言）；
Math-L3 122.5 GB（4 子集）；合计 **994.3 GB** → 50 GB 云端预算下经
`tools/fetch_ms.py` 采样 **3–5%**。

## 文档地图

**先按你要做的事找文档**（每个事实只有一处权威出处，其余链接引用）：

| 你要做的 | 看这份 |
|---|---|
| **了解项目是什么、怎么跑** | 本文件 |
| **扩展它**（加机制 / 后端 / dtype / 数据 / 训练阶段）| `docs/PHD-Net_扩展指南.md` |
| **理解六个机制与脑同构、容量账、铁律** | `docs/PHD-Net_架构设计.md` |
| **查性能数字**（唯一出处）| `docs/PHD-Net_性能评估与迭代方案.md` |
| **搞懂为什么吃不满多核与 NPU** | `docs/PHD-Net_并行与加速架构分析.md` |
| **迁移/适配硬件后端、昇腾踩坑** | `docs/PHD-Net_硬件后端适配报告.md` |
| **与 Transformer / 大脑对比的结论** | `docs/PHD-Net_竞争力与脑同构性评估.md` |
| **查某个坑的根因** | `BUGS.md`（33 条，症状 → 根因 → 门禁缺口） |
| 子目录用法 | `train_1b/README.md`、`tools/README.md`、`chat/README.md`、`datasets/README.md`、`phdnet/backends/README.md` |

## 当前状态（2026-09-30）

- **精度**：读出默认 **fp8 forward + fp16 更新**（保住 |dp|≈1e-6 的非目标行更新——
  bf16 会把它舍掉，导致学习退化为纯 Hebbian）；M1 默认 fp64（昇腾走自写 numba
  GEMV）；分词 onehot 缓冲 fp16。
- **性能（1B 档，昇腾 191 核 + NPU）**：20–110 ms/tok；读出 5.5–7 ms（访存受限）；
  本轮修掉 M4b（63.5 ms/tok）、M1 平台差异、M2 融合核退化、检查点 0.8–37 s 硬停顿。
  数字与口径见性能文档。
- **门禁**：`python tests/run_tests.py fast` = 9/9（零回归门槛）。
- **CI**：两套（GitCode + GitHub 镜像）；GitHub 侧当前仍红，待 traceback。
- **已知欠账**：M6 读出仍 100% 稠密（人脑连接率 2e-8）→ 幂律稀疏化方案已设计、
  未实施；数据集语种构成与文档记录矛盾（实测中文仅 2%）。

## 快速开始

```bash
# 训练（生产单轨；服务器用远程语料直读）
python train_1b/train.py --preset 1b --data pretrain --remote-data \
    --remote-fraction 0.3 --lang zh --resume --step-profiling

# 评测基线（复测入口）
python tools/rebaseline.py

# 门禁
python tests/run_tests.py fast
```

## CI/CD

- 回归入口 `tests/run_tests.py`（fast 9 项）+ 逐位对拍 `tests/verifiers/`。
- CI 双平台：`.gitcode/workflows/ci.yml`（GitCode Action）+ `.github/workflows/ci.yml`（GitHub 镜像）。
- 治理约定：架构介绍文档（docs/*.md）不是数据集；冻结评测基准位于 `eval_corpus/`。
