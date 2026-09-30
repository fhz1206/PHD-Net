# chat/ —— 对话与推理用法

适用范围：本目录的聊天 / 演示脚本。**1B 生产模型的推理入口在 `train/infer.py`**
（不在本目录）。数据截止：2026-09-30。
相关文档：`train/README.md`、`phdnet/backends/README.md`（设备归属与回落纪律）。

入口选择：终端交互对话 → `tui.py`；OpenAI Chat Completions 风格 + web_search
工具调用 → `chat_openai.py`；与 R1 蒸馏 SFT 模型对话（无 TUI）→ `chat_r1sft.py`；
1B 生产检查点 → `train/infer.py`；本地 CPU「训练 + 对话」演示 → `run_demo.py` /
`run_demo_external.py`。

## 一、入口：`chat/tui.py`

纯标准库实现（无 urwid / textual / rich 依赖），ANSI 转义序列渲染：顶部反显状态栏固定在首行
（模型名 / 词表规模 / 设备 / 对话轮次），中部对话历史区追加式滚动（用户青色、助手绿色、
系统提示暗色），底部是 `>` 输入框。斜杠命令 `/help` `/quit` `/clear` `/save <path>`；
多行输入用行尾 `\` 续行或单独一行 `.` 提交。打字机效果是**渲染层模拟**
（`PHDWordLM.generate` 一次性返回，字符逐个打印）。终端不支持 ANSI（无 tty /
`TERM=dumb`）时自动退化为逐行打印，`--force-tui` 强制走 TUI 路径（管道调试用）。

```bash
python chat/tui.py --model outputs/r1sft_model.pkl                    # word 后端（默认）
python chat/tui.py --ckpt outputs/models/phdnet1b_1b_sft_final.npz    # 1b 后端：自包含 npz
python chat/tui.py --selftest                                         # 无模型：随机小模型走通 UI
python chat/tui.py --selftest --prompt "你好"                          # 单次模式验证渲染路径
```

参数（源码核对，`chat/tui.py`）：`--backend {word,1b}`（默认 `word`；`--model *.npz`
自动切 1b）、`--model <pkl>`（默认 `outputs/r1sft_model.pkl`）、
`--ckpt <npz>`（隐含 `--backend 1b`）、`--tau 0.8`、`--max-tokens 200`、
`--topk 8`（`0` = 全分布）、`--seed 0`、`--prompt <text>`（单次模式）、
`--selftest`、`--force-tui`。

## 二、模型加载：检查点 + 词表快照 + `--init-from`

**1b 后端（npz，自包含）** —— `train/infer.py::load_from_ckpt`：
词表（`seg.vocab` / `tok.tokens` / SDR 哈希）**完整序列化在检查点内**，
推理**不需要语料、不需要外部词表文件**；`cfg` 由检查点 `meta` 重建，
训练与推理词表逐位一致。加载大矩阵时按 `meta["ckpt_dtype"]` 把位模式
（uint8 / uint16）**无损解回原精度** —— `ckpt_dtype` 是**存储**精度，
不是计算精度（这点常被误解）。

**词表快照只做交叉校验，不作为词表来源**：不给 `--vocab-file` 时，自动找检查点同目录 /
上一级 `models/` / `outputs/models/` 下的 `vocab_<preset>_<data>.json`（训练侧落盘）。
**不一致即报错退出（退出码 2）** —— 用错词表会让 SDR 哈希与读出层错位，表现为
「能跑但输出退化」的**静默故障**。比对对象是 **token 词表**（`tok_tokens`，= 读出层行数），
**不是**分词器候选集（`tok_vocab`），两者语义不同。要改词表只能重训。

```bash
python train/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz \
    --vocab-file outputs/models/vocab_1b_pretrain.json --prompt "用户：你好\n助手："
```

**word 后端（pickle）** —— 由 `chat/tui.py` / `chat_r1sft.py` 直接 `pickle.load`，
模型与词表在同一个 pkl 里，**没有独立的词表校验环节**：加载到哪个模型就认哪个词表。

**关于 `--init-from`** —— 它是**训练侧**参数（`train/train.py`），**推理入口不提供**。
它与 `--resume` 的区别就是「步数是否归零」：续训用 `--resume`（步数接着走），
两阶段微调（在预训练权重上做 SFT）用 `--init-from`（步数**归零**）。
推理侧要换模型只有一条路：给 `--ckpt` 指向另一个检查点。详见 `train/README.md`。

---

## 三、生成参数

| 参数 | `tui.py` | `train/infer.py` | 说明 |
|---|---|---|---|
| 温度 | `--tau` 0.8 | `--tau` 0.7 | 越高越多样 |
| top-k | `--topk` 8 | `--topk` 8 | `0` = 全分布不截断 |
| 生成 token 数 | `--max-tokens` 200 | `--n` 200 | 单轮上限 |
| 随机种子 | `--seed` 0 | `--seed` 0 | 同值 + 同参数可复现 |

长 prompt 走 `StreamingTokenizer` **流式预热，不截断**（任意长，含 1M 级），
按 `--milestone`（默认 1,000,000，0 = 关闭）打点；OOV token 跳过并打印计数，
与训练侧同一回退语义。`infer.py` 另有 `--devices auto` 打印设备清单与多卡计划。

---

## 四、推理侧低精度：只影响质量，不影响「学习能不能发生」

**核心事实：推理侧没有学习更新。** 本目录与 `train/infer.py` 的所有
`lm.net.step(..., learn=False)` 只做前向，权重在推理过程中**不被改写**。因此：

- 训练侧那个「低精度把微小更新舍掉 → 退化为纯 Hebbian」的机制问题
  （bf16 半 ULP ≈ 2e-4 ≫ 非目标行更新 |dp| ≈ 1e-6 → 更新被舍）
  **在推理侧不存在** —— 没有更新，就没有被舍掉的东西；
- 低精度在推理侧**只影响生成质量**（读出分布的数值误差 → 采样结果偏移）。
  推理侧看到质量下降时该查量化误差与采样参数，而不是查学习率；
- 「fp8 前向副本 + fp16 更新主副本」那套双副本机制是**训练侧**设计
  （副本每 `--fp8-refresh` 步重建）。推理时它只体现为前向副本的量化误差，
  不再有任何「更新写回」环节。

### 当前加速器侧只支持 fp32 / fp16 / bf16

`--readout-dtype` 在**加速器后端**可用档位为 **fp32 / fp16 / bf16**；
**fp8 / fp4 已在所有加速器禁用**（P86）—— 不是「能力不足」，而是**没有算子**：
昇腾 Ascend910B4 + CANN 8.5 + torch_npu 2.9 上 `float8_e4m3fn` 的
create / cast / matmul 全部 ERR01007，fp4 的 MX 块缩放同样没有算子。

想用 fp8/fp4 的量化码本，只能走 numba CPU 路径
（`python train/train.py --accel cpu --readout-dtype fp8`）—— 那是 P9/P12 的
**位算法量化核**（CPU 上可用），**不代表加速器具备 fp8 能力**。
根因与实测证据见 `docs/PHD-Net_硬件后端适配报告.md` 与 `BUGS.md`。

其它精度口径（训练侧，推理时由检查点 `cfg` 决定）：M1 编码器默认 **fp64**、
分词 onehot 缓冲 **fp16**、检查点默认 `--ckpt-dtype bf16`（存位模式）。
**推理入口不暴露 `--readout-dtype` / `--ckpt-dtype`** —— 想要不同精度就用不同的训练产物，
不要在推理时切换。

---

## 五、生产推理（`train/infer.py`）

```bash
# 单次续写
python train/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz \
    --prompt "用户：什么是机器学习？\n助手：" --n 200
# 交互式对话（多轮共享网络状态 —— 对话历史就是 context）
python train/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz --chat
```

参数：`--model`（必填）、`--prompt`、`--chat`、`--n 200`、`--tau 0.7`、`--topk 8`、
`--seed 0`、`--milestone 1000000`、`--accel auto`、`--vocab-file`、`--devices auto`。

设备归属：读出（M6）可经 `phdnet/backends/AccelReadout` 上加速器，
**其余部件（分词/编码、M1–M5、STDP）仍在 CPU**（numba 只能编译到 CPU 机器码）。
不可用的加速器会**回落并打印原因**，不静默。见 `phdnet/backends/README.md`。

---

## 六、其它脚本

| 脚本 | 用途 | 典型用法 |
|---|---|---|
| `chat_openai.py` | OpenAI Chat Completions 风格对话循环 + `web_search` 工具；主路径由模型自己生成 `<tool_call>` 标记，备用路径 `--fallback-rules`（默认开）按关键词兜底并打 `[fallback]` 标记 | `python chat/chat_openai.py --demo` |
| `chat_r1sft.py` | 与 R1 蒸馏 SFT 模型对话 / 演示续写（逐行 input，Ctrl-C 退出） | `python chat/chat_r1sft.py --demo` |
| `websearch_tool.py` | 联网搜索工具（OpenAI function calling 封装，**无需 API key**；Bing CN → Bing 国际 → DuckDuckGo Lite 逐级回退） | `python chat/websearch_tool.py 搜索词` |
| `run_demo.py` / `run_demo_external.py` | 本地 CPU「训练 + 对话」演示（字符级 `PHDNetLM` / 词级 `PHDWordLM`） | `python chat/run_demo.py` |

**前置现状（诚实标注）**：`chat_r1sft.py` 的前置是 `tools/train_r1sft.py`（仍在）；
但两者各自的语料制备脚本（`tools/prepare_r1sft.py`、`tools/prepare_toolcall.py`）
已随一次性实验清理删除（见 `tools/README.md`），**重新训练需先自行恢复制备步骤**。
两个对话脚本本身可追溯、可用。

## 七、诚实边界

模型是**词级联想续写器**（无注意力 / 无位置编码，条件化 = WM 锚定 + LTM 印迹 +
情景缓冲），生成格式上连贯的文本，**不保证事实正确，也不保证真正回答了问题**。
`phdnet/generate.py::Generator` 是字符级 M11 遗留接口，与词级 LM **不兼容**，勿混用。
