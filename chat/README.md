# chat/ —— 对话与推理入口

PHD-Net 训练产物的对话/演示脚本（fhz 2026-09-28 目录规范：聊天代码统一放 `chat/`）。

## 推荐入口：`tui.py`（终端 TUI）

纯标准库（无 urwid/textual/rich 依赖），ANSI 转义序列实现：

```
┌──────────────────────────────────────────────────────────┐
│ PHD-Net TUI │ r1sft_model.pkl │ 词表 12,345 │ numpy-cpu │ 轮次 3 │   ← 反显状态栏（固定首行）
├──────────────────────────────────────────────────────────┤
│ 你 > 你好                                                        │  ← 用户：青色
│ 助手 > 你好，很高兴见到你……（逐字符打字机输出）                      │  ← 助手：绿色
│         （历史区滚动，追加式）                                     │
├──────────────────────────────────────────────────────────┤
│ > _                                                      │  ← 底部输入框
└──────────────────────────────────────────────────────────┘
```

- 斜杠命令：`/help` `/quit` `/clear` `/save <path>`
- 多行输入：行尾 `\` 续行，单独一行 `.` 结束多行输入并提交
- 助手回复逐 token/逐字符打印（打字机效果；PHDWordLM 一次性返回，渲染层模拟流式）
- Windows 兼容：启动时 `os.system("")` 激活 ANSI（Windows Terminal / 新版 conhost）；
  终端不支持时自动降级为普通逐行打印；`--force-tui` 可强制 TUI 渲染路径（管道调试）

```bash
# word 后端（默认）：pickle 模型，chat_r1sft.py 同款加载
python chat/tui.py --model outputs/r1sft_model.pkl

# 1b 后端：train_1b 自包含 npz 检查点（词表内嵌，无需语料）
python chat/tui.py --ckpt outputs/models/phdnet1b_1b_sft_final.npz

# 无模型时走通 UI（随机小模型）；或单次模式验证渲染路径
python chat/tui.py --selftest
python chat/tui.py --selftest --prompt "你好"
```

参数：`--backend {word,1b}`（默认 word；`--model *.npz` 自动切 1b）、`--model <pkl>`、
`--ckpt <npz>`（隐含 1b）、`--tau 0.8`、`--max-tokens 200`、`--topk 8`、
`--seed 0`、`--prompt <text>`（单次模式）、`--selftest`、`--force-tui`。

诚实边界：模型是词级联想续写器（PHD-Net 无注意力/位置编码），生成的是格式上
连贯的文本，不保证事实正确或真正的语义问答。

## 其他脚本

| 脚本 | 用途 |
|---|---|
| `chat_openai.py` | OpenAI Chat Completions 风格对话循环（本地 PHD-Net 后端 + web_search 工具；SFT 工具调用训练后的模型驱动 `<tool_call>` 生成） |
| `chat_r1sft.py` | 与 R1 蒸馏 SFT 子集训练的模型对话 / 演示续写（逐行 input 循环，无 TUI） |
| `websearch_tool.py` | 联网搜索工具（OpenAI function calling 协议封装，为对话循环配发） |
| `run_demo.py` | 本地 CPU「训练 + 对话」演示（字符级 PHDNetLM） |
| `run_demo_external.py` | 外部语料训练 + 对话演示（mimo-claude-code-traces-1k） |

1B 生产模型的交互式对话也可走 `train_1b/infer.py`（`--chat`）——检查点自包含加载，
无需语料；模型位于 `outputs/models/`。

示例：

```bash
python chat/chat_openai.py                       # 工具调用对话循环
python chat/chat_r1sft.py                        # R1 SFT 模型对话
python train_1b/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz --chat
```
