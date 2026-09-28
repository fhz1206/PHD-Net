# chat/ —— 对话与推理入口

PHD-Net 训练产物的对话/演示脚本（fhz 2026-09-28 目录规范：聊天代码统一放 `chat/`）。

| 脚本 | 用途 |
|---|---|
| `chat_openai.py` | OpenAI Chat Completions 风格对话循环（本地 PHD-Net 后端 + web_search 工具；SFT 工具调用训练后的模型驱动 `<tool_call>` 生成） |
| `chat_r1sft.py` | 与 R1 蒸馏 SFT 子集训练的模型对话 / 演示续写 |
| `websearch_tool.py` | 联网搜索工具（OpenAI function calling 协议封装，为对话循环配发） |
| `run_demo.py` | 本地 CPU「训练 + 对话」演示（字符级 PHDNetLM） |
| `run_demo_external.py` | 外部语料训练 + 对话演示（mimo-claude-code-traces-1k） |

1B 生产模型的交互式对话走 `train_1b/infer.py`（`--chat`）——检查点自包含加载，
无需语料；模型位于 `outputs/models/`。

示例：

```bash
python chat/chat_openai.py                       # 工具调用对话循环
python chat/chat_r1sft.py                        # R1 SFT 模型对话
python train_1b/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz --chat
```
