# -*- coding: utf-8 -*-
"""P205a 设计评估：架构级提前计算的三层方案（决策记录，先写后实现）。

## 现状瓶颈链（服务器 1b 档，step 12.3ms/tok）

```
CPU: M1(0.3) → M2(0.6) → M3 → M4ab(2.3) → [readout提交] → PC_learn → STDP
NPU:                          └── forward_dev ──┘   └── learn_softmax ──┘
     └──────────── 12.3ms 串行 ────────────┘  NPU 2.04ms 只占 8.5%
```

NPU 的 2.04ms 是「提交后算完」的时间；但 CPU 从「提交读出」到「下一步提交
读出」之间还有 ~10ms 在跑 M4ab/PC_learn——这期间 NPU 空闲。**NPU 利用率
20-50% 的本质 = NPU 每步只有 2ms 的活，却有 12ms 的周期**。

## 三层方案（按侵入度排序）

### 层 1（已存在）：pinned 异步 H2D（P58 `_staged_to_dev`）
h 的上传已经 non_blocking——但 kernel 提交仍在本流，NPU 只提前了**搬运**，
没提前**计算**。收益已吃满。

### 层 2（本轮实现）：`--beta p2` 双流 H2D + `nll` 独立流
- **p2**：`_staged_to_dev` 的 H2D 提交到独立 `_h2d_stream`，主流
  `wait_stream` 后消费——H2D 与「上一步的读出 kernel」真并行（此前同流
  FIFO 串行）。单步读出 2ms 里 H2D 占比小（~µs 级），预期收益小但零风险。
- **nll 独立流**：`nll_sync_every>1` 时第 N 步的 `.item()` 是 CPU 硬同步点
  （等 NPU 把 nll 算完）。改成 `_nll_stream.wait_stream(主流)` 后再 item——
  CPU 只等「nll 累积链」完成，不等「同流后续 kernel」。1b 档每 8 步省一次
  全流排空。
- **`--cann-dispatch` 接线**：`wire_dispatch()` 调
  `torch_npu.npu.set_compile_mode(jit_compile=False)` 解除「task_queue 仅
  二进制场景生效」的静默失效——这是 P120 以来的**未接线项**。

### 层 3（记录，本轮不做）：跨步 pipeline（t+1 的 target 提前）
`learn_softmax` 的 `correct`（target 索引）在第 t+1 步开始时已知（t+1 的
target 就绪才进 step）——但 softmax 用的是**本步 y**，提前提交无意义。
真正能提前的是「把 t 步的 learn_softmax 提交延迟到 t+1 步的 forward_dev
之后、在同一个 NPU 批里执行」——**这改变 PC_learn 读到的 nll 时序**
（task_modulation 用 nll 门控本步主干学习率），P117 已裁决不做。
```
（P117 原文：前移会改变 _prev_rate/_exp 等状态的推进时机，等于换一套
  时序规则——这是学习规则层面的语义变更，收益 ~1.09× 不足以承担）
```

## 服务器 A/B 判据（P205a 实现后）
```bash
# 基线
python3 train/train.py --preset 1b --data pretrain --remote-data --replay-every 16
# + nll 独立流 + p2 双流（--beta p2）
python3 train/train.py --preset 1b --data pretrain --remote-data --replay-every 16 --beta p2
# + CANN 接线（set_compile_mode 解除 JIT 场景）
python3 train/train.py --preset 1b --data pretrain --remote-data --replay-every 16 --beta p2 --cann-dispatch
```
判据：`segments(win)` 的 readout ms/tok、端到端 ms/tok、`npu-smi` 的利用率
采样均值（目标：20-50% → 更高=空窗缩短）、`nll_sync_every` 的 PPL 无回归。
"""
