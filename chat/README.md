# chat/ —— 对话与推理用法

适用范围：本目录的聊天/演示脚本。1B 生产模型的推理入口在 `train_1b/infer.py`。
数据截止 2026-09-30。相关文档：`README.md`、`train_1b/README.md`。

入口选择：终端交互对话（推荐）→ `tui.py`；OpenAI Chat Completions 风格 + web_search
工具调用 → `chat_openai.py`；与 R1 蒸馏 SFT 模型对话（无 TUI）→ `chat_r1sft.py`；
1B 生产检查点 → `train_1b/infer.py`（见下）；本地 CPU「训练+对话」演示 →
`run_demo.py` / `run_demo_external.py`。

## `tui.py`（终端 TUI）

纯标准库实现（无 urwid/textual/rich 依赖），ANSI 转义序列渲染：顶部反显状态栏固定在首行
（模型名 / 词表规模 / 设备 / 对话轮次），中部对话历史区追加式滚动（用户青色、助手绿色、
系统提示暗色），底部是 `>` 输入框。斜杠命令 `/help` `/quit` `/clear` `/save <path>`；
多行输入用行尾 `\` 续行、单独一行 `.` 提交。打字机效果是**渲染层模拟**
（`PHDWordLM.generate` 一次性返回，字符逐个打印）。Windows 上启动时 `os.system("")`
激活 ANSI，不支持时自动降级为逐行打印，`--force-tui` 强制走 TUI 路径（管道调试）。

```bash
python chat/tui.py --model outputs/r1sft_model.pkl       # word 后端（默认）：pickle 模型
python chat/tui.py --ckpt outputs/models/phdnet1b_1b_sft_final.npz   # 1b 后端：自包含 npz
python chat/tui.py --selftest                            # 无模型时走通 UI（随机小模型）
python chat/tui.py --selftest --prompt "你好"              # 单次模式验证渲染路径
```

参数（`--help` 核对）：`--backend {word,1b}`（默认 word；`--model *.npz` 自动切 1b）、
`--model <pkl>`、`--ckpt <npz>`（隐含 1b）、`--tau 0.8`、`--max-tokens 200`、`--topk 8`、
`--seed 0`、`--prompt <text>`、`--selftest`、`--force-tui`。

## 模型加载：检查点 + 词表

**1b 后端（npz，自包含）** —— `train_1b/infer.py::load_from_ckpt`：词表（`seg.vocab` /
`tok.tokens` / SDR 哈希）**完整序列化在检查点内**，推理不需要语料、也不需要外部词表文件；
`cfg` 由检查点 `meta` 重建，训练与推理词表逐位一致。加载大矩阵时按 `meta["ckpt_dtype"]`
把位模式（uint8/uint16）**无损解回原精度** —— `ckpt_dtype` 是**存储**精度，不是计算精度。

**词表快照只做交叉校验，不作为词表来源**（不给 `--vocab-file` 时自动找检查点同目录 /
上一级 `models/` / `outputs/models/` 下的 `vocab_<preset>_<data>.json`）；**不一致即报错
退出**（退出码 2）：用错词表会让 SDR 哈希与读出层错位，表现为「能跑但输出退化」的静默
故障。比对对象是 **token 词表**（= `tok_tokens`，读出层行数），不是分词器候选集
（`tok_vocab`）。要改词表只能重训。**word 后端（pickle）** 则由 `chat/tui.py` /
`chat_r1sft.py` 直接 `pickle.load`，模型与词表在同一个 pkl 里。

```bash
python train_1b/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz \
    --vocab-file outputs/models/vocab_1b_pretrain.json --prompt "用户：你好\n助手："
```

## 生成参数

`--tau`（温度）默认 `tui.py` 0.8 / `infer.py` 与 `chat_r1sft.py` 0.7，越高越多样；
`--topk` 默认 8（`0` = 全分布不截断）；`--max-tokens` / `--n` = 单轮生成 token 数；
`--seed` 同值 + 同参数可复现。长 prompt 走 `StreamingTokenizer` **流式预热，不截断**
（任意长，含 1M 级），按 `--milestone` 打点；OOV token 跳过并打印计数（与训练侧同一
回退语义）。

## fp8 / bf16 推理的注意事项

**核心事实：推理侧没有学习更新。** 本目录与 `train_1b/infer.py` 的所有
`lm.net.step(..., learn=False)` 只做前向，权重在推理过程中**不被改写**。因此：

- 训练侧那个「低精度把微小更新舍掉 → 退化为纯 Hebbian」的机制问题（半 ULP ≫ 非目标行
  更新 |dp|）**在推理侧不存在** —— 推理不更新权重，没有更新可被舍掉；
- 低精度在推理侧**只影响生成质量**（读出分布的数值误差 → 采样结果偏移），不影响
  「学习能不能发生」。推理侧看到 fp8/bf16 表现下降时该查量化误差与采样参数；
- fp8 主副本 + fp16 更新副本那套双副本机制是**训练侧**设计，推理时只体现为前向副本
  的量化误差。

其他约束：fp8 matmul **只在昇腾 / CUDA 可用**，CPU 上回落 fp16（不报错，静默回落）。
推理入口**不暴露** `--readout-dtype` / `--ckpt-dtype`：精度由检查点里的 `cfg` 决定，
想要不同精度就用不同训练产物，而不是在推理时切换。

## 生产推理（`train_1b/infer.py`）

```bash
# 单次续写
python train_1b/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz \
    --prompt "用户：什么是机器学习？\n助手：" --n 200
# 交互式对话（多轮共享网络状态 —— 对话历史就是 context）
python train_1b/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz --chat
```

参数：`--model`（必填）、`--prompt`、`--chat`、`--n 200`、`--tau 0.7`、`--topk 8`、
`--seed 0`、`--milestone 1000000`（预热打点间隔，0=关闭）、`--accel auto`、`--vocab-file`、
`--devices auto`（打印设备清单与多卡计划）。该入口是 **numba CPU 生产路径（单路）**：
numba 只能编译到 CPU 机器码，多卡加速不在此路径。

## 其他脚本

| 脚本 | 用途 | 典型用法 |
|---|---|---|
| `chat_openai.py` | OpenAI Chat Completions 风格对话循环 + `web_search` 工具；主路径由模型自己生成 `<tool_call>` 标记，备用路径 `--fallback-rules`（默认开）按关键词兜底并打 `[fallback]` 标记 | `python chat/chat_openai.py --demo`（默认模型 `chat/tool_model.pkl`，会话存 `chat/openai_convo.json`） |
| `chat_r1sft.py` | 与 R1 蒸馏 SFT 模型对话/演示续写（逐行 input，Ctrl-C 退出） | `python chat/chat_r1sft.py --demo`（默认 `outputs/r1sft_model.pkl`） |
| `websearch_tool.py` | 联网搜索工具（OpenAI function calling 封装，无需 API key；Bing CN → Bing 国际 → DuckDuckGo Lite 逐级回退） | `python chat/websearch_tool.py 搜索词` |
| `run_demo.py` / `run_demo_external.py` | 本地 CPU「训练 + 对话」演示（字符级 `PHDNetLM` / 词级 `PHDWordLM`） | `python chat/run_demo.py` |

`chat_openai.py` / `chat_r1sft.py` 的前置训练分别是 `tools/prepare_toolcall.py`、
`tools/prepare_r1sft.py`。

## 诚实边界

模型是**词级联想续写器**（无注意力 / 无位置编码，条件化 = WM 锚定 + LTM 印迹 + 情景缓冲），
生成格式上连贯的文本，**不保证事实正确，也不保证真正回答了问题**。
`phdnet/generate.py::Generator` 是字符级 M11 遗留接口，与词级 LM **不兼容**，勿混用。
