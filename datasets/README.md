---
license: apache-2.0
task_categories: [text-generation, question-answering]
language: [zh, en]
tags: [sft, pretraining, mixture, brain-inspired]
pretty_name: Mixture-General-Mini
size_categories: [10M<n<100M]
configs:
- config_name: pretrain
  data_files:
  - split: train
    path: 'pretrain/*.parquet'
- config_name: sft
  data_files:
  - split: train
    path: 'sft/*.parquet'
---

# datasets/ —— 语料目录

适用范围：本目录的结构约定与数据入口。**评测冻结基准不在这里**（见下）。数据截止
2026-09-30。相关文档：`UPLOAD_README.md`（上传流程）、`train_1b/README.md`（训练侧消费）。

## 目录结构

```
datasets/                    ← 本目录，自身是独立 git 仓库（见下）
├── sft/                     SFT 分片 parquet
│   └── raw/                 原始件归档（.gitignore 排除，不入库）
├── pretrain/                预训练分片 parquet
│   └── raw/                 原始件归档（.gitignore 排除，不入库）
├── .gitattributes           parquet 存储方式（见 UPLOAD_README）
└── .gitignore               排除 eval/ 与 */raw/
```

`sft/`、`pretrain/` 下的分片统一 schema `text (string) / lang (string: zh|en|code) /
src (string: 来源语料名)`，均 ≤100 MiB，pyarrow / `datasets` 可直接读。`*/raw/` 是
**原始归档**（下载的 jsonl、转换前的 txt、merge 源），只供制备脚本直读，**不入库**，
训练入口不碰 `raw/`。`.gitignore` 同时排除 `eval/` —— 但冻结基准本来就不在本目录。

## 评测冻结基准在仓库根 `eval_corpus/`，不在这里

- `eval_corpus/internal_corpus.txt` —— 冻结基准，**27,034 字符**（实测原文长度；
  按 80/20 划分后训练段 18,803 / 评估段 4,701 是划分后口径，两者不要混用）。
- `eval_corpus/ood_wiki.txt` —— 远域（OOD）探针，20,187 字符。
- 该目录**不入训练集**。`tools/prepare_wikicn.py` 产出的 `wiki_train.txt` /
  `wiki_eval.txt` 是历史训练切分，与冻结基准是两套东西，不要互相替代。

注：`tools/audit_gen_eval.py` / `tools/rebaseline.py` 的 docstring 里写的语料字符数
（23,504 / 27,405）是旧口径 —— **以上表与 `docs/文档写作规范.md` 为准**。

## datasets 是独立仓库

`datasets/` 自带 `.git`，是**独立 git 仓库**（远端 atomgit.com/fhz1206/Mixture-General-Mini），
不是主仓库的子目录。主仓库只追踪四份文档（`README.md`、`UPLOAD_README.md`、
`.gitattributes`、`.gitignore`），**不追踪分片内容** —— 原因是交付体积超出远端限额
（详见 `UPLOAD_README.md`）。**权威副本在 ModelScope**，训练侧用 `--remote-data` 直读。

本机分片副本已于 2026-09-29 按指令删除（307 文件 / 32.28 GiB，其中 parquet 64 个 /
9.62 GiB），逐文件明细见主仓库 git 历史
`170ce0c^:docs/DATASETS_DELETED_MANIFEST.json`。**要恢复本地数据只能从 ModelScope
重新下载或重新制备**，不能靠 git checkout（分片不在主仓库索引里）。

## 远程数据集：ModelScope `fhzfhz/Mixture-General-Mini`

训练侧两入口二选一：

```bash
# ① 本地 datasets/（逐片 parquet 直读）
python train_1b/train.py --data pretrain

# ② ModelScope 直读：HTTP Range 流式，零本地落盘（仅支持 ModelScope）
python train_1b/train.py --data pretrain --remote-data --remote-fraction 0.3
```

`--remote-data` 的机制：把数据路径换成 `ms://fhzfhz/Mixture-General-Mini/<split>/<glob>`，
经 fsspec HTTP Range（206 分块随机读）把分片当可 seek 文件流交给 `pyarrow.ParquetFile`，
**1 GB 原始片从不落盘**。需要 `fsspec` + `aiohttp`。

`--remote-fraction F`（默认 0.3）是**分片级前缀抽样**：把分片名排序后取**前 F 比例**，
不是随机采样、不是按行采样；它是 `PHDNET_REMOTE_FRACTION` 环境变量，经 worker 进程继承。
四条代价：

1. **顺序语义不变 = 语料构成高度偏斜**。分片不是随机打散的，取前缀等于只吃了排在前面的
   那几个源。实测：52 片里第 0–45 片全是 `infinity_m7core`，`--remote-fraction 0.3`
   取前 16 片 → 实际只用到 M7_Core 一个源。
2. **改 fraction 会静默改变数据分布**。与 `--resume` 组合尤其危险：续训时改了 fraction
   等于换了数据。数据口径（`data` / `remote` / `remote_fraction` / `lang` / 分片数）
   会随检查点落盘，续训时能看出这次用的是哪份数据。
3. **跨 run 比较必须同 fraction**：性能与 PPL 数字只有同 fraction、同机器、同数据才可比。
4. **词表扫描会重复读**。`--vocab-scan full` 在远程模式下把分片整读两遍（建词表 + 训练），
   启动日志会 WARN；建议 `--vocab-file` 复用词表快照。

### ⚠ 已知问题：语种构成与早期记录矛盾（待核）

远程分片实测 lang 分布 **en 50% / `unk` 47% / zh 仅 2%**，其中 **`unk` 其实是代码**
（CJK 占比 0%）；只有末片是 `ultrafineweb_l3_zh`（中文 99.7%）。后果：
`--lang zh` 过滤后只剩约 2% 的行，且 `--remote-fraction 0.3` 取的前缀分片里几乎没有中文。

这与早期文档记录的「中文 95%」**直接矛盾**，属**数据问题**（制备/上传环节），不是训练
代码问题。核对结论尚未产出，引用语种比例前请以本节为准。台账见主仓库 `BUGS.md` #18。

## 本地读取

统一走 `phdnet/corpus.py`（同认 `.txt` 与 `.parquet`；给 parquet 路径即走列式读取，
支持只读部分 row group，大语料不必整文件载入内存）；或用 HuggingFace `datasets`
直读远端：

```python
from phdnet.corpus import load_text, iter_texts
text = load_text("datasets/pretrain/pretrain_000.000.parquet", limit_chars=4096)
for row in iter_texts("datasets/sft/sft_000.000.parquet"):
    ...

from datasets import load_dataset
ds = load_dataset('fhzfhz/Mixture-General-Mini', 'sft')
ds = load_dataset('fhzfhz/Mixture-General-Mini', 'pretrain', streaming=True)
```

## 版权与许可

各来源数据集均为 Apache License 2.0 或其原始开源许可；本仓库整体以 Apache License 2.0
发布。逐来源清单与分片数见 `UPLOAD_README.md`。

