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

适用范围：本目录的结构约定、数据入口与语种实测。**评测冻结基准不在这里**（见下）。
**数据截止：2026-09-30。**
相关文档：`UPLOAD_README.md`（上传/发布流程）、`../train/README.md`（训练侧消费）、
`../tools/README.md`（制备脚本索引）。

> ⚠ 本目录同时是 **ModelScope 数据集卡片**（上方 YAML 头）与 gitcode 数据集仓库的说明。
> 改结构时两张卡片都要复核。

## 一、目录结构

```
datasets/                    ← 本目录，自身是独立 git 仓库（见 §四）
├── sft/                     SFT 分片 parquet
│   └── raw/                 原始件归档（.gitignore 排除，不入库）
├── pretrain/                预训练分片 parquet
│   └── raw/                 原始件归档（.gitignore 排除，不入库）
├── .gitattributes           parquet 存储方式（见 UPLOAD_README.md）
└── .gitignore               排除 eval/ 与 */raw/
```

分片统一 schema：`text (string)` / `lang (string: zh|en|code)` / `src (string: 来源语料名)`，
均 ≤ 100 MiB，`pyarrow` / HuggingFace `datasets` 可直读。`*/raw/` 是**原始归档**
（下载的 jsonl、转换前的 txt、merge 源），只供制备脚本直读，**不入库**；训练入口不碰 `raw/`。
`.gitignore` 同时排除 `eval/` —— 但冻结基准本来就不在本目录（见 §二）。
制备工具（`fetch_*` / `convert_*` / `build_parquet` / `merge_corpus` / `split_parquet` /
`pack_corpus`）的用法见 `../tools/README.md`。

## 二、评测冻结基准在**仓库根 `eval_corpus/`**，不在这里

| 文件 | 字符数（实测原文长度） | 用途 |
|---|---|---|
| `eval_corpus/internal_corpus.txt` | **27,034** | 冻结基准，按 `int(len*0.8)` 划分为 **21,627 训练 / 5,407 评估** |
| `eval_corpus/ood_wiki.txt` | **20,187** | 远域（OOD）探针 |

「原文长度」与「划分后段长」是**两个口径**，不要混用。该目录**不入训练集**，
且保留在 PHD-Net 主仓库（评测基线必需），`.gitignore` 明确保留它。
`tools/prepare_wikicn.py` 产出的 `wiki_train.txt` / `wiki_eval.txt` 是**历史训练切分**，
与冻结基准是两套东西，**不要互相替代**。部分脚本 docstring 里还写着旧字符数
（23,504 / 27,405），那是旧口径；以上表与 `../docs/文档写作规范.md` 为准。

## 三、训练侧两个数据入口（二选一）

```bash
# ① 本地 datasets/（逐片 parquet 直读，默认路径，逐位不变）
python train/train.py --data pretrain

# ② ModelScope 直读：HTTP Range 流式，零本地落盘（仅支持 ModelScope）
python train/train.py --data pretrain --remote-data --remote-fraction 0.3
```

本地读取统一走 `phdnet/corpus.py`（同认 `.txt` 与 `.parquet`；给 parquet 路径即走列式读取，
支持只读部分 row group，大语料不必整文件载入内存）；或用 HuggingFace `datasets` 直读远端：
```python
from phdnet.corpus import load_text, iter_texts
text = load_text("datasets/pretrain/pretrain_000.000.parquet", limit_chars=4096)
for row in iter_texts("datasets/sft/sft_000.000.parquet"):
    ...

from datasets import load_dataset
ds = load_dataset('fhzfhz/Mixture-General-Mini', 'sft')
ds = load_dataset('fhzfhz/Mixture-General-Mini', 'pretrain', streaming=True)
```


## 三之二、P196：Claude Code traces（`--data cc_traces`）

fhz 指定的**测试训练集**：ModelScope `choucisan/mimo-claude-code-traces-1k`
（1,017 个 trace / 15,046 events / 11,995 messages / 5,271 tool calls /
10 个任务类别 / 40.5 MB / MIT License）。

⚠ **它不是 parquet，而是 event-stream JSONL**（每行一个 event：
`user` / `assistant` / `tool_use` / `tool_result` / 元数据行）——
通用 `iter_texts` 找不到 `text` 字段会得到**空流**，故有专用读取路径
（`phdnet/corpus.py::iter_cc_traces`）：只抽语言内容、按 event 原顺序
还原成对话文本，**保留 `thinking` 块**（reasoning trace 是这份数据的独特价值）。

```bash
# 远程直读（零本地落盘；两级目录 session/<category>/*.jsonl 由 train.py
# 按 10 个类别逐个展开——ms_stream 只支持单层 glob）
python train/train.py --preset smoke --data cc_traces --remote-data --tokens 3000

# 本地副本模式：先转成 parquet（tools/data/cc_traces_to_text.py）
py -3.14 tools/data/cc_traces_to_text.py --list
py -3.14 tools/data/cc_traces_to_text.py --out datasets/cc_traces/cc_000.parquet
python train/train.py --preset smoke --data cc_traces --tokens 3000
```

转换格式（一条 trace → 一段文本，`\n\n` 分隔多轮）：

```
user: The following Python code has a logic error bug. …
assistant: [思考] The user has shared a Python function …
assistant: The code doesn't actually have a logic error …
assistant: [调用工具 Bash] {"command": "pytest -q"}
user: [工具返回] 3 passed
```

**本机 smoke 实测**（`--preset smoke --tokens 3000`）：PPL **2474 → 1457**、
LTM imprints=430（长出 30,720 突触）、OOV 0.0000%、3.04 ms/tok。
对拍门禁：`tests/gates/gate_cc_traces.py`（23 项，含远程实读）。

⚠ 内容是**英文**（代码 + 推理 + 工具调用）→ `--data-lang zh` 会得到空流，
train.py 已提前拦下并自动改为不过滤。

**P198 训练预算建议**：词表规模 × 曝光度必须匹配——

| 词表采样 | 词表 | tokens | 每词曝光 | probe_PPL | 生成 bigram 合法率 |
|---|---|---|---|---|---|
| 4M 字符 | 20,913 | 100K | 4.8 | 3322（≈随机） | 5–13% |
| **300K 字符** | **3,392** | 100K | 29.5 | **389**（**11×**） | **78%** |

**推荐**：`--vocab-sample-chars 300000 --tokens 400000`（曝光 > 100 才能学到
中频词；`tools/diag_vocab_budget.py` 可在训练前估算）。
⚠ 生产档必须 `eta_oja=0`（P198：`eta_oja>0` 在生产规模正反馈发散——
上行识别权重改动 PC 前向，r1/r2 喂 STDP+读出形成反馈环，
3000 步 probe_PPL 从 3403 恶化到 9.1e11；`build_cfg` 已默认 0）。

## 四、datasets 是独立仓库

`datasets/` 自带 `.git`，是**独立 git 仓库**，不是主仓库的子目录：

- 远端：`gitcode.com/fhz1206/Mixture-General-Mini`；
- 主仓库 `.gitignore` 显式排除 `datasets/pretrain/`、`datasets/sft/`、`datasets/.git/`，
  因此主仓库只追踪本目录的 `README.md`、`UPLOAD_README.md`、`.gitattributes`、`.gitignore`
  四份文件，**不追踪分片内容**；
- 原因是交付体积超出远端限额（详见 `UPLOAD_README.md`）；
- **权威副本在 ModelScope**（`fhzfhz/Mixture-General-Mini`），
  训练侧用 `--remote-data` 直读。

⚠ 本机分片副本已按指令删除，**不能靠 `git checkout` 恢复**（分片不在任何仓库索引里）。
要恢复本地数据只能从 ModelScope 重新下载或重新制备。

## 五、远程数据集：ModelScope `fhzfhz/Mixture-General-Mini`

`--remote-data` 把数据路径换成 `ms://fhzfhz/Mixture-General-Mini/<split>/<glob>`，
经 fsspec HTTP Range（206 分块随机读，块大小 8 MB）把分片当**可 seek 文件流**交给
`pyarrow.ParquetFile`，**原始分片从不落盘**（需 `fsspec` + `aiohttp`）。
只支持 ModelScope（数据集托管 modelscope.cn；atomgit 限额 1 GiB 本就推不全）。
超时 60 s、重试 3 次、指数退避 2 s；远程模式预取进程上限 4（避免打爆服务端）。
实现见 `phdnet/ms_stream.py`，对拍见 `tests/gates/gate_ms_stream.py`。

**`--remote-fraction F`（默认 0.3）= 分片级前缀抽样**：把分片名排序后取**前 F 比例**，
不是随机采样、不是按行采样；经环境变量 `PHDNET_REMOTE_FRACTION` 传给 worker 进程。
设计意图是**保持数据顺序语义**（词表扫描与训练流开头逐字符一致）。**四条代价**：

1. **顺序语义 = 语料构成高度偏斜**。分片不是随机打散的，取前缀等于只吃了排在前面的
   少数源。实测 52 片里前 46 片同属一个源，`--remote-fraction 0.3` 取前 16 片
   → 实际只用到该单一源。
2. **改 fraction 会静默改变数据分布**。与 `--resume` 组合尤其危险：续训时改 fraction
   等于换了数据。数据口径（`data` / `remote` / `remote_fraction` / `lang` / 分片数）
   随检查点落盘，续训时能看出这次用的是哪份数据。
3. **跨 run 比较必须同 fraction**。性能与 PPL 数字只有同 fraction、同机器、同数据才可比。
4. **词表扫描会重复读**。`--vocab-scan full` 在远程模式下把分片整读两遍
   （建词表 + 训练），启动日志会 WARN；建议用 `--vocab-file` 复用词表快照。

## 六、⚠ 语种实测：与早期记录矛盾（待核）

远程分片实测 `lang` 分布：

| lang | 占比 | 实际内容 |
|---|---|---|
| `en` | **50%** | 英文 |
| `unk` | **47%** | **实为代码**（CJK 占比 0%） |
| `zh` | **2%** | 中文 |

只有末片是中文 L3 采样（中文 99.7%）。后果：`--lang zh` 过滤后只剩约 2% 的行，
且 `--remote-fraction 0.3` 取的前缀分片里几乎没有中文。

**这与早期文档记录的「中文 95%」直接矛盾。** 属**数据问题**（制备 / 上传环节），
不是训练代码问题；核对结论尚未产出。**引用语种比例前以本节为准**，标为**待核**。

## 七、版权与许可

各来源数据集均为 Apache License 2.0 或其原始开源许可；本数据集整体以 Apache License 2.0
发布（见文件头 YAML）。逐来源清单与分片数见 `UPLOAD_README.md`。
