# tools/ —— 工具脚本索引

适用范围：本目录下的**工具型脚本**（语料制备、评测审计、硬件诊断、发布）。
**生产训练入口单轨在 `train_1b/`**（`train_1b/train.py`），新增训练脚本不进本目录。
数据截止 2026-09-30。相关文档：`README.md`、`train_1b/README.md`、`datasets/README.md`。

脚本参数一律以源码为准（`python tools/<名>.py --help`）；未列出的参数表示该脚本无该开关。

## 语料制备（txt → parquet → 分片）

统一 schema `text / lang / src`；数据导入侧统一走 `phdnet/corpus.py`（同认 txt 与 parquet）。

| 脚本 | 一句话 | 典型用法 |
|---|---|---|
| `fetch_ms.py` | ModelScope 三源（web/code/math）HTTP Range 流式采样，**零原始落盘**，直接产出 `datasets/pretrain/` 分片 | `python tools/fetch_ms.py --dry-run`（只探规模）<br>`python tools/fetch_ms.py --plan web=3,code=2,math=1 --resume` |
| `convert_infinity.py` | Infinity-Instruct M7_Core → parquet | `python tools/convert_infinity.py --stats` |
| `convert_ultrainteract.py` | UltraInteract SFT → parquet（`--mode flat/grouped`） | `python tools/convert_ultrainteract.py --stats` |
| `convert_deepctrl.py` | 匠数 deepctrl → parquet（`--lang zh/en` 流式过滤） | `python tools/convert_deepctrl.py --lang zh --max-mb 300` |
| `convert_magpie.py` | Magpie-R1 CoT → parquet | `python tools/convert_magpie.py --stats` |
| `convert_ultrafineweb.py` | Ultra-FineWeb-L3 中文 → parquet（`--step` 等距抽样） | `python tools/convert_ultrafineweb.py --stats` |
| `build_parquet.py` | 把 `datasets/` 下 txt 转 parquet（zstd），原 txt 默认保留 | `python tools/build_parquet.py --stats`<br>`python tools/build_parquet.py --replace` |
| `merge_corpus.py` | 按**输出实际大小滚动**合并同目录语料（去重源：txt 与 parquet 同源只取 parquet） | `python tools/merge_corpus.py --split pretrain --stats`<br>`python tools/merge_corpus.py --split sft` |
| `split_parquet.py` | 把大 parquet 按输出大小切成 ≤`--target-mb` 的**合法 parquet**（可直接 pyarrow 读，无需还原） | `python tools/split_parquet.py datasets/pretrain/merged/pretrain_000.parquet --target-mb 90` |
| `pack_corpus.py` | gzip + 按字节分卷（`--chunk-mb`，默认 90）+ `SHA256SUMS.txt`；用于绕开远端单文件限额 | `python tools/pack_corpus.py datasets/sft/x.txt`<br>还原：`cat x.gz.part-* > x.gz && gunzip x.gz` |
| `analyze_corpus_role.py` | 判定语料更适合 SFT 还是 pretrain（结构统计，不依赖样本边界） | `python tools/analyze_corpus_role.py datasets/sft/*.txt` |

## 评测与审计

| 脚本 | 一句话 | 典型用法 |
|---|---|---|
| `rebaseline.py` | 冻结语料 + BASE 配置的基线复测（**无命令行参数**，两个口径 + 吞吐 + 主干连接率） | `python tools/rebaseline.py` |
| `audit_gen_eval.py` | 泛化实测：域内 held-out / 近域 / 远域 PPL + 词级 2-gram 参照（全程 readonly） | `python tools/audit_gen_eval.py --ckpt outputs/smoke/phdnet1b_smoke_sft.npz --skip-tokens 150000 --indomain-tokens 30000` |
| `audit_precision.py` | 精度审计（半精度格点/更新舍入） | `python tools/audit_precision.py` |
| `audit_brain_parity.py` / `audit_imprint_gate.py` | 脑同构指标 / 印迹门禁 | `python tools/audit_brain_parity.py` |
| `profile_lm.py` | 词级 LM 热路径剖析 | `python tools/profile_lm.py` |

M5 公平评测（PHD-Net vs nanoGPT 级 Transformer）入口不在本目录：
`python tests/eval_suite.py`。回归门禁：`python tests/run_tests.py fast`（9 项）。

## 性能与诊断

| 脚本 | 一句话 | 典型用法 |
|---|---|---|
| `accel_doctor.py` | 一次性回答「探测到 ≠ 能用」：环境矩阵 → 设备解析 → 读出规模试分配 + matvec → 与 numba 基线 ms/token 对照 | `python tools/accel_doctor.py`<br>`python tools/accel_doctor.py --V 20000 --H 3072 --no-bench` |
| `bench_accel.py` | 各可用设备 × 各精度档的读出热路径基准（`--steps/--V/--H`） | `python tools/bench_accel.py --steps 50` |
| `estimate_scale.py` | 逐档放大主干/词表，实测 (参数量, ms/token) 并外推，同时给内存需求 | `python tools/estimate_scale.py --tokens 800` |
| `backend_probe.py` | 设备探针（CUDA / ROCm / CANN·昇腾 / DirectML / CPU） | `python tools/backend_probe.py` |
| `calibrate_scaling.py` / `scaling_curve.py` / `scaling_data_axis.py` | 缩放律标定与数据轴扫描 | `python tools/calibrate_scaling.py` |

性能数字的**唯一出处**是 `docs/PHD-Net_性能评估与迭代方案.md`，本目录脚本只产出实测，
不在 README 里抄结论。x86 与昇腾的加速方向常常相反，跨平台数字不可混用。

## 数据获取

| 脚本 | 一句话 | 典型用法 |
|---|---|---|
| `fetch_modelscope.py` | 从 ModelScope 下载单个数据文件（国内直连） | `python tools/fetch_modelscope.py --ns AI-ModelScope --name wikipedia-cn-20230720-filtered --file wikipedia-cn-20230720-filtered.jsonl --out data/_raw/wikipedia-cn-filtered.jsonl` |
| `prepare_wikicn.py` | 中文维基 → 训练段/评估段纯文本，**确定性划分**（尾部为评估段，中间留 `--gap` 隔离带） | `python tools/prepare_wikicn.py --train-chars 2000000 --eval-chars 200000` |
| `fetch_l3.py` / `fetch_ultrafineweb.py` | 早期单源采样器（`fetch_ms.py` 的前身，保留可复现） | `python tools/fetch_ultrafineweb.py --list` |
| `prepare_mimo.py` / `prepare_mix.py` / `prepare_r1sft.py` / `prepare_toolcall.py` | SFT 子集制备（外部语料 → 训练/评估切分） | `python tools/prepare_r1sft.py --max-samples 1000` |

## 发布

| 脚本 | 一句话 | 典型用法 |
|---|---|---|
| `upload_modelscope.py` | 把 `datasets/` 的交付物推到 ModelScope `fhzfhz/Mixture-General-Mini`；`--verify` 只核对远端清单 | `python tools/upload_modelscope.py --verify` |

上传范围与限额约束见 `datasets/UPLOAD_README.md`。**上传动作由 fhz 执行**，
本脚本只做清单校验与推送。

## 非生产轨脚本（勿误用）

历史上有过三条训练入口，现已消歧；下列两个仍在 `tools/` 下，**都不是生产训练入口**：

- `train_1b_capacity.py`：1B **容量验证**（在线 CSR，无词表/无流式管线），仅用于复现
  「1B 容量本机可训」结论。
- `train_rl.py`：REINFORCE 实验（torch 栈，权重与生产 ckpt **不通用**）。

生产训练一律走 `train_1b/train.py`，生产推理走 `train_1b/infer.py`。

## 一次性实验与归档

- `experiment_*.py`、`ablation_modules.py`：一次性机制实验，结论已固化进 `docs/`。
- `archive/`：`bench.py`、`bench_1b_migrate.py`、`audit_prof_1b.py`、`hunt_bugs.py`、
  `rerun_task6.py` —— 保留可追溯，不再维护。
