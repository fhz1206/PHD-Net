---
license: apache-2.0
task_categories:
- text-generation
- question-answering
language:
- zh
- en
tags:
- sft
- pretraining
- mixture
- brain-inspired
pretty_name: Mixture-General-Mini
size_categories:
- 10M<n<100M
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

# Mixture-General-Mini

PHD-Net 训练语料集（SFT + 预训练混合），统一 schema：`text (string) / lang (string: zh|en) / src (string: 来源语料名)`，
共 54 个 ≤100 MiB 的 parquet 分片（pyarrow / datasets 可直接读取）。

## sft 配置（config_name=sft）：2 片 / 178 MB / 3,319,142 条
| 来源 | 语言 | 内容 |
|---|---|---|
| OpenBMB UltraInteract_sft | 英文 | Coding 39.8% / Math 56.2% / Logic 4.0%（指令-响应对） |
| 匠数 deepctrl-sft-data（中文子集） | 中文 89.6% | 流式过滤：丢弃上下文依赖续问 13%、过短/过长 9% |

## pretrain 配置（config_name=pretrain）：52 片 / 4.6 GB / 39,059,702 条
| 来源 | 条数 | 说明 |
|---|---|---|
| Infinity-Instruct 7M_core（M7_Core） | 36,967,551 | 高质量指令对话（预训练用途） |
| Magpie-Reasoning-V1-150K-CoT-DeepSeek-R1-Llama-70B | 2,012,151 | R1 长链推理（平均 7,661 字符） |
| Ultra-FineWeb-L3 中文采样 | 80,000 | OpenBMB 高质量中文网页（确定性等间隔抽样） |

## 读取示例
```python
from datasets import load_dataset
ds = load_dataset('fhzfhz/Mixture-General-Mini', 'sft')
ds = load_dataset('fhzfhz/Mixture-General-Mini', 'pretrain', streaming=True)
```

## 版权与许可
各来源数据集均为 Apache License 2.0 或其原始开源许可；本仓库整体以 Apache License 2.0 发布。
