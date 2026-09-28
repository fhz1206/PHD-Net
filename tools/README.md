# tools/ 索引（P14，2026-09-28 整理）

## 为什么整理

`tools/train_1b.py`（1B 突触容量验证）与 `train_1b/`（生产训练管线）目录名高度相似，
且历史上并存三条训练入口，容易误用。整理原则：**不删功能，只消歧 + 归类**（全部 `git mv`，
历史可复现）。

## 三条训练/推理入口的分工（务必按此选用）

| 入口 | 定位 | 何时用 |
|---|---|---|
| `train_1b/train.py` | **生产单轨**（numba CPU，1B 词级 LM 全量流式训练） | 1B 正式训练、SFT、评测 |
| `tools/train_torch_lm.py` | torch 栈轨（多卡自动适配 P14，权重与生产 ckpt **不通用**） | 有 GPU/多卡时的实验与加速验证 |
| `tools/train_1b_capacity.py` | 1B **容量验证**（BillionSynapseNet 在线 CSR，无词表/无流式管线） | 复现「1B 容量本机可训」结论（docs 引用） |

推理：`train_1b/infer.py`（生产 ckpt，numba CPU 单路）/ torch 栈见上表说明。

## 目录分类

- **语料工具**（`convert_*` / `prepare_*` / `build_parquet` / `merge_corpus` /
  `split_parquet` / `pack_corpus` / `fetch_*` / `upload_modelscope` / `analyze_corpus_role`）：
  数据导入统一走 `phdnet/corpus.py`。
- **审计与基准**：`rebaseline.py`（基线锚点）、`audit_precision.py`（精度）、
  `audit_gen_eval.py`（泛化）、`audit_brain_parity.py` / `audit_imprint_gate.py`（脑同构）、
  `bench_accel.py`（加速器基准）、`profile_lm.py`（LM 热路径剖析）。
- **实验脚本**（`experiment_*` / `ablation_modules.py`）：一次性机制实验，结论已入文档。
- **缩放与规划**：`estimate_scale.py` / `calibrate_scaling.py` / `scaling_curve.py` /
  `scaling_data_axis.py`。
- **`archive/`**：一次性旧基准与调试脚本（`bench.py`、`bench_1b_migrate.py`、
  `audit_prof_1b.py`、`hunt_bugs.py`、`rerun_task6.py`）——保留可追溯，不再维护；
  其结论已固化进 docs 与记忆日志。

## 命名约定

- 训练入口只保留 `train_1b/` 一条生产轨（`tools/train_production.py` 已于 P11 删除）；
- 一次性脚本带 `experiment_` / `audit_` 前缀，基准带 `bench_`，语料带 `convert_` / `prepare_`。
