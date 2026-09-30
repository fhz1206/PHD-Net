# tools/ —— 工具脚本索引

适用范围：本目录下**工具型脚本**的用途分组与典型命令。**参数一律以源码为准**
（`python tools/<名>.py --help`）；未列出的参数表示该脚本没有该开关。
**数据截止：2026-09-30。** 相关文档：`README.md`、`train/README.md`、
`datasets/README.md`、`datasets/UPLOAD_README.md`。

> **生产入口单轨**：训练只有 `train/train.py`，推理只有 `train/infer.py`。
> 本目录**不放**新的训练入口（见文末「非生产轨」）。

---

## 一、语料制备（txt → parquet → 分片）

分片统一 schema `text (string) / lang (string: zh|en|code) / src (string)`；
数据读取侧统一走 `phdnet/corpus.py`（同认 `.txt` 与 `.parquet`）。

| 脚本 | 一句话 | 典型命令 |
|---|---|---|
| `fetch_ms.py` | ModelScope 多源 HTTP Range 流式采样，**零原始落盘**，直接产 `datasets/pretrain/` 分片 | `python tools/fetch_ms.py --dry-run`<br>`python tools/fetch_ms.py --plan web=3,code=2,math=1 --resume` |
| `fetch_l3.py` | Ultra-FineWeb-L3 单源流式采样（`fetch_ms.py` 的前身，保留可复现） | `python tools/fetch_l3.py --docs-per-lang 1000,1000 --resume` |
| `fetch_ultrafineweb.py` | L3 子集枚举与计数（`--list` 列出子集，`--subset` 可重复） | `python tools/fetch_ultrafineweb.py --list` |
| `fetch_modelscope.py` | 从 ModelScope 下载**单个**数据文件（国内直连） | `python tools/fetch_modelscope.py --ns AI-ModelScope --name wikipedia-cn-20230720-filtered --file wikipedia-cn-20230720-filtered.jsonl --out data/_raw/wikipedia-cn-filtered.jsonl` |
| `convert_infinity.py` | Infinity-Instruct M7_Core → parquet（`--sample` 等距抽样，`--stats` 只统计单分片并外推） | `python tools/convert_infinity.py --stats` |
| `convert_ultrainteract.py` | UltraInteract SFT → parquet（`--mode flat\|grouped`） | `python tools/convert_ultrainteract.py --stats` |
| `convert_deepctrl.py` | 匠数 deepctrl → parquet（`--lang zh\|en` 流式过滤，`--max-mb` 截断） | `python tools/convert_deepctrl.py --lang zh --max-mb 300` |
| `convert_magpie.py` | Magpie-R1 CoT → parquet | `python tools/convert_magpie.py --stats` |
| `convert_ultrafineweb.py` | Ultra-FineWeb-L3 中文 → parquet（`--step` 等距抽样） | `python tools/convert_ultrafineweb.py --stats` |
| `build_parquet.py` | 把 txt 转 parquet（zstd）；`--replace` 删原 txt，`--stats` 只统计 | `python tools/build_parquet.py --stats` |
| `merge_corpus.py` | 按**输出实际大小滚动**合并同目录语料；同源 txt 与 parquet 只取 parquet。`--split pretrain\|sft` 必填 | `python tools/merge_corpus.py --split pretrain --stats` |
| `split_parquet.py` | 把大 parquet 按**输出大小**切成 `--target-mb`（默认 90）的**合法 parquet**，pyarrow 可直读、无需还原 | `python tools/split_parquet.py datasets/pretrain/merged/pretrain_000.parquet --target-mb 90` |
| `pack_corpus.py` | gzip + 按字节分卷（`--chunk-mb` 默认 90）+ `SHA256SUMS.txt`；用于绕开远端单文件限额。还原：`cat x.gz.part-* > x.gz && gunzip x.gz` | `python tools/pack_corpus.py datasets/sft/x.txt` |
| `prepare_wikicn.py` | 中文维基 → 训练段/评估段纯文本，**确定性划分**（尾部为评估段，中间留 `--gap` 隔离带，避免同主题泄漏） | `python tools/prepare_wikicn.py --train-chars 20000000 --eval-chars 2000000` |
| `analyze_corpus_role.py` | 判定语料更适合 SFT 还是 pretrain（**结构统计，不依赖样本边界**） | `python tools/analyze_corpus_role.py datasets/sft/*.txt` |

分片大小、`.gitattributes` 与 LFS 的关系、发布限额：见 `datasets/UPLOAD_README.md`。

---

## 二、评测与审计

| 脚本 | 一句话 | 典型命令 |
|---|---|---|
| `rebaseline.py` | 冻结语料 BASE 配置的基线复测：两个口径 + 吞吐 + 主干连接率。**无命令行参数** | `python tools/rebaseline.py` |
| `audit_gen_eval.py` | 泛化实测：域内 held-out / 近域 / 远域 PPL + 词级 2-gram 参照（全程 readonly） | `python tools/audit_gen_eval.py --ckpt outputs/smoke/phdnet1b_smoke_sft.npz --skip-tokens 150000 --indomain-tokens 30000` |
| `audit_precision.py` | 精度体系三层审计：L1 各 dtype 融合核 vs numpy 参考**逐位**、L2 带宽、L3 端到端 PPL。**无命令行参数** | `python tools/audit_precision.py` |
| `audit_brain_parity.py` | 脑同构指标逐项开关实测（`--sparse-conn` / `--big-ltm` / `--cortical-init` / `--sparse-readout` / `--ei-synapses` / `--k-sparse` / `--two-level`） | `python tools/audit_brain_parity.py --big-ltm` |
| `audit_imprint_gate.py` | big_ltm 印迹门控阈值实验（`ltm_imprint_gate` 扫描）。**无命令行参数** | `python tools/audit_imprint_gate.py` |
| `ablation_modules.py` | 逐模块关闭/打开，按 `tests/eval_common.py` 的 BASE 口径跑完整训练+评估，输出 PPL 敏感度排序 | `python tools/ablation_modules.py` |
| `scaling_curve.py` | 参数轴缩放：三点实测 (参数量, ppl_char, bpc, ms/token)，幂律拟合 α 与 R² | `python tools/scaling_curve.py` |
| `scaling_data_axis.py` | 数据轴缩放：固定架构放大训练字符数，验证「瓶颈是数据预算还是参数规模」 | `python tools/scaling_data_axis.py` |
| `calibrate_scaling.py` | 缩放三点配置的实测校准（参数量 + 单遍耗时），**临时脚本**，不改 `phdnet/` | `python tools/calibrate_scaling.py` |

M5 公平评测（PHD-Net vs nanoGPT 级 Transformer）入口**不在本目录**：
`python tests/eval_suite.py`。回归门禁：`python tests/run_tests.py fast`（**9/9**）。

---

## 三、性能与诊断

**性能数字的唯一出处是 `docs/PHD-Net_性能评估与迭代方案.md`**；本目录脚本只产出实测，
不在 README 里抄结论。x86 与昇腾的加速方向常常相反，跨平台数字不可混用。

| 脚本 | 一句话 | 典型命令 |
|---|---|---|
| `bench_local.py` | 本机（x86）CPU 侧热点微基准 + numba 线程缩放 + 4M 档端到端 smoke；`--compare` 与上一份结果对比给出「变快/变慢」 | `python tools/bench_local.py --threads 1,2,4,8`<br>`python tools/bench_local.py --compare --no-smoke` |
| `bench_accel.py` | 读出热路径在各可用设备 × 各精度档的基准（`--steps` / `--V` / `--H`，默认 V=73,958、H=3,072） | `python tools/bench_accel.py --steps 50` |
| `accel_doctor.py` | 一次性回答「探测到 ≠ 能用」：环境矩阵 → 设备解析 → 读出规模试分配 + matvec → 与 numba 基线 ms/token 对照 | `python tools/accel_doctor.py --V 20000 --H 3072 --no-bench` |
| `estimate_scale.py` | 逐档放大主干/词表，实测 (参数量, ms/token) 并外推，同时给内存需求 | `python tools/estimate_scale.py --tokens 800` |
| `profile_lm.py` | LM 热路径剖析：cProfile 端到端 + 逐相手工计时（编码/PC/STDP/WM/LTM/读出），结果写 `outputs/experiments/profile_lm.log`。**无命令行参数**，只读不改模型 | `python tools/profile_lm.py` |
| `backend_probe.py` | 硬件后端探针：昇腾 NPU / ROCm / CUDA / DirectML / CPU，并跑本机可跑的等价性自检 | `python tools/backend_probe.py` |

⚠ `bench_local.py` 测的是**本机 x86** 数字：已实测三次「x86 更快、昇腾更慢」
（M1 fp32、M2 融合核、OMP place 表）。它只回答「这次改动让 CPU 侧变快还是变慢」，
文档里的 ms/tok 必须来自服务器实测。

---

## 四、能力探测

| 脚本 | 一句话 | 典型命令 |
|---|---|---|
| `probe_fp8.py` | 探测当前设备/框架**实际暴露**的 fp8 能力：`float8_e4m3fn` 的建张量/cast/matmul、`torch.ops` 里的 fp8 算子、`torch_npu` 量化 API、设备型号与 CANN 版本。秒级出结果，不需要训练数据 | `python tools/probe_fp8.py --json` |
| `probe_npu_quant.py` | 定向探测昇腾**量化矩阵乘 API**（fp8 张量不通之后的下一步）：`npu_weight_quant_batchmatmul` / `npu_quant_matmul` 等支持什么 dtype、签名能否吃读出的形状 | `python tools/probe_npu_quant.py --json` |

**结论（已定案，写在这里避免重复实验）**：fp8 / fp4 在**所有加速器**已禁用。
实测昇腾 Ascend910B4 + CANN 8.5 + torch_npu 2.9 上 `float8_e4m3fn` /
`float8_e5m2` 的 create / cast / matmul **全部 ERR01007**，`torch.ops` 零个 fp8 算子；
CPU torch 能建 fp8 张量但**无 fp8 addmv**。因此 `--accel cpu --readout-dtype fp8`
仍可用——走的是 numba CPU 的量化核，**不是加速能力**。根因与处置见
`docs/PHD-Net_硬件后端适配报告.md` 与 `BUGS.md`。

---

## 五、发布

| 脚本 | 一句话 | 典型命令 |
|---|---|---|
| `upload_modelscope.py` | 把 `datasets/` 的交付物推到 ModelScope `fhzfhz/Mixture-General-Mini`（排除 `.git` / `eval/` / `*/raw/` / `*.txt`）；`--verify` 只核对远端清单与本地应有清单的差集，不写入 | `python tools/upload_modelscope.py --verify` |

**上传动作由 fhz 本人执行**，脚本只做清单校验与分片推送，不做发布决策
（推哪个仓、要不要拆仓、限额怎么绕）。约束与限额现实见 `datasets/UPLOAD_README.md`。

---

## 六、非生产轨脚本（勿误用）

历史上有过多条训练入口，现已消歧。下列两个仍在 `tools/` 下，**都不是生产训练入口**：

| 脚本 | 定位 | 与生产的关系 |
|---|---|---|
| `train_1b_capacity.py` | 1B **容量验证**（在线 CSR，无词表 / 无流式管线），用于复现「1B 容量本机可训」 | 独立实现，权重与生产 ckpt 不通用 |
| `train_rl.py` | REINFORCE 实验（策略梯度直接复用 M6 读出感知器更新） | 实验轨，**权重与生产 ckpt 不通用**；REINFORCE 回路本身有 `tests/verifiers/verify_rl.py` 覆盖 |

`train_rl.py` 主要参数：`--init-from`（必填）、`--prompts`（必填）、
`--reward-fn module:function`、`--iters 100`、`--rl-lr 0.01`、`--n-tokens 32`、
`--tau 1.0`、`--topk 0`、`--baseline-decay 0.9`、`--kl-coef 0.0`、`--accel auto`。

`tools/archive/`：`bench.py`、`bench_1b_migrate.py`、`audit_prof_1b.py`、
`hunt_bugs.py`、`rerun_task6.py` —— 保留可追溯，**不再维护**，不要在新工作里引用。

一次性实验脚本（`experiment_*.py`）与已产出数据的语料制备脚本
（`prepare_{mimo,mix,r1sft,toolcall}.py`）已于 2026-09-30 删除；
`convert_*.py` / `fetch_*.py` 保留以保证语料可重现。
