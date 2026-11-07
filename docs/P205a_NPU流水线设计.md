# NPU 加速特性现状（`--beta` 实验性优化档）

> **适用范围**：`--beta` 提供的 NPU 实验性优化档（p0/p1/p2/p3）。
> **数据截止**：2026-11-01（P214）。
> **相关文档**：[性能评估与迭代方案](PHD-Net_性能评估与迭代方案.md)（性能数字唯一出处）、
> [并行与加速架构分析](PHD-Net_并行与加速架构分析.md)（为什么吃不满）、
> [CPU 侧加速方案 P213](PHD-Net_CPU侧加速方案_P213.md)（CPU 侧时间预算）、
> [硬件后端适配报告](PHD-Net_硬件后端适配报告.md)（昇腾踩坑）。
> 与代码冲突时**一律以代码为准**。

---

## 0. 一句话

`--beta` 提供 4 个 NPU 优化档：**p0/p1/p2 已实现**（Graph Capture / 前向+学习融合图 /
双流 H2D + nll 独立流），**p3（双缓冲流水线）未实现**（`--help` 里列了，代码里没有）。
这些是针对 **NPU「假占用」**（设备占用 90%+ 但算力没跑满）的**现有手段**；
新增手段见 [NPU 利用率诊断与优化 P214](PHD-Net_NPU利用率诊断与优化_P214.md)。

---

## 1. 现状：NPU 为什么「假占用」

**现象**：设备监控显示占用 ~90%+，但实际算力/吞吐远未达对应水平。

**机制**（本项目已定案，见性能文档 §3.5 与并行文档 §3.5）：

```
CPU: M1(0.3) → M2(0.6) → M3 → M4ab(1.96) → [提交读出] → PC_learn → STDP
NPU:                        └── forward_dev ──┘   └── learn_softmax ──┘
     └──────────── 10.8 ms 串行周期 ───────────┘  NPU 3.5 ms 只占 32%
```

**NPU 每步只有 ~3.5 ms 的活，却有 10.8 ms 的周期** → 设备监控看到的是
「任务队列里有东西在跑」或「设备上下文处于活跃态」，**不等于 AI Core 在做有效计算**。

⚠ **「占用 90%」的指标本身要分层看**（P214 诊断结论）：
- **设备级占用**（npu-smi 的 HBM/上下文占用、进程在设备上的时间占比）
  → 高，但**不代表 AI Core 满载**；
- **AI Core 算力占用**（`aicore_time`）→ 本项目 P144 msprof 实测 **`aicore_time=0`**，
  即**有效计算时间几乎为 0**，时间都花在 **`aclnnIndex`（算子下发/索引，89.3%）**
  与 **Task Wait（6.8×）** 上。

→ 所以本项目的「假占用」**主要是算子下发开销 + 同步等待**，不是带宽饱和
（读出是**算子效率问题**，见性能文档 §3.5 的 26.1×）。

---

## 2. 四个 beta 档的现状

| 档 | 名称 | 状态 | 机制 | 代码位置 |
|---|---|---|---|---|
| **p0** | Graph Capture | ✅ 已实现 | 把读出前向+学习捕获成 NPU Graph，replay 时消除 **kernel launch 开销** | `accel_readout.py:1350` `enable_npu_graph()` |
| **p1** | Forward+Learn Graph Fusion | ✅ 已实现 | 前向与学习合到**一张图**（消除两图之间的下发间隙） | 同上，`p1` 分支 |
| **p2** | Async H2D/D2H 双流 | ✅ 已实现 | H2D 提交到**独立流** `_h2d_stream`；`nll` 累积链走 `_nll_stream`（`nll_sync_every>1` 时 CPU 只等 nll 链，不等主流排空） | `accel_readout.py:595-606, 1323-1328, 2421` |
| **p3** | 双缓冲流水线 | ❌ **未实现** | `--help` 里列了「p3=双缓冲流水线」，但**代码里没有任何 p3 分支** → 传 `--beta p3` **静默无效** | — |

### 2.1 p2 的两条流（唯一「减少同步等待」的已实现手段）

```python
self._h2d_stream = None      # accel_readout.py:595  H2D 独立流
self._nll_stream = None      # :605                  nll 独立流
self._nll_stream_ok = None   # :606                  None=未探测 / bool=探测结果
```

- **H2D 独立流**：读出输入 `h` 的上传走 `_h2d_stream`（non_blocking），
  与「上一步的读出 kernel」真正并行（此前同流 FIFO 串行）；
- **nll 独立流**：`nll_sync_every>1` 时，第 N 步的 `.item()` 是**CPU 硬同步点**
  （等 NPU 把 nll 算完）。改成 `_nll_stream.wait_stream(主流)` 后再 item ——
  CPU 只等「nll 累积链」完成，**不等同流后续 kernel**。
  1b 档 `nll_sync_every=8` → 每 8 步省一次全流排空。

⚠ **探测失败即降级**：无 `torch.npu.Stream`（torch-CPU / 老版本）→ `_nll_stream_ok=False`
→ 走回原路径，**不报错**。

### 2.2 `--cann-dispatch`（与 beta 正交的独立开关）

```bash
python train/train.py --preset 1b --data pretrain --remote-data --cann-dispatch
```

- **不新增任何行为**，只核对 `TASK_QUEUE_ENABLE=2` + `COMBINED_ENABLE=1` 是否真的生效
  （这两项自 P120 起已默认开）；
- ⚠ 官方列了两个**静默失效**条件且**都不打日志**：
  ① 仅**二进制场景**生效（JIT 路径下等于没开）；
  ② `ASCEND_LAUNCH_BLOCKING=1` 会**强制关闭** task_queue。
- 本项目**无实测**，属官方文档推荐 + 默认无害。

---

## 3. 怎么用 / 怎么测

### 3.1 启用

```bash
# p0+p1+p2（实验性，需配合 --cython-kernels auto）
python train/train.py --preset 1b --data pretrain --remote-data \
    --beta p0,p1,p2 --cython-kernels auto

# 范围写法也支持
--beta p0-p1-p2
```

⚠ **仅在 `--accel npu/cuda/rocm` 时生效**；CPU 路径不受影响。
⚠ **实验性功能** —— 启用前必须跑门禁，且**判据只能用 `probe_PPL`**（见 §3.2）。

### 3.2 服务器 A/B 判据（照抄，不要自己发明）

```bash
# 基线
python3 train/train.py --preset 1b --data pretrain --remote-data --replay-every 16

# + p2 双流
python3 train/train.py --preset 1b --data pretrain --remote-data --replay-every 16 --beta p2

# + CANN 接线
python3 train/train.py --preset 1b --data pretrain --remote-data --replay-every 16 --beta p2 --cann-dispatch
```

**判据**：
1. `segments(win)` 的 `readout` ms/tok；
2. 端到端 ms/tok；
3. `npu-smi` 利用率采样均值（⚠ 目标**不是**占用更高，而是**空窗缩短** →
   占用可能**下降**但吞吐**上升**，见 §4）；
4. `nll_sync_every` 变化下 **PPL 无回归**（用 `probe_PPL`，不是 sliding PPL）。

### 3.3 线程数 A/B（与 beta 正交，但同样影响 NPU 侧）

```bash
python tools/bench/bench_server_threads.py --threads 8,16,32
```

---

## 4. ⚠ 关键认知：「占用下降」不等于「变慢」

针对 NPU 的「假占用」，判据**不能只看设备占用率**：

| 指标 | 含义 | 期望方向 |
|---|---|---|
| 设备占用（npu-smi） | 任务队列/上下文的活跃度 | **可能随空窗缩短而下降**（这是好事） |
| **端到端 ms/tok** | 真实吞吐 | **下降才是收益** |
| **`segments(win)` 的 readout** | 读出段的真实耗时 | 下降 |
| **AI Core 算力占用**（`aicore_time`） | 有效计算时间占比 | **上升**（真正在算） |
| `aclnnIndex` 占比 | 算子下发/索引开销 | **下降**（说明少下发、少碎算子） |

P144 msprof 的关键读数正是这个逻辑：
**`aicore_time=0` 而 `aclnnIndex=89.3%`** → 设备「占用」很高但**几乎没在算**。
→ 优化的第一目标是**把 `aclnnIndex` 的时间挪到 `aicore_time`**，
而不是把占用率推得更高。

---

## 5. 现状小结

| 已有手段 | 能解决 | 不能解决 |
|---|---|---|
| p0 Graph Capture | kernel launch 开销 | 算子本身的碎粒度、Host↔Device 拷贝 |
| p1 前向+学习融合图 | 两图之间的下发间隙 | 图内算子碎粒度 |
| p2 双流 H2D + nll 独立流 | H2D 与 kernel 重叠、同步点摊薄 | 跨 token 流水（改变时序规则 → 铁律③） |
| `--cann-dispatch` | task_queue 静默失效 | — |
| `--nll-sync-every`（默认 **4**，P216） | `.item()` 硬同步摊薄；N=4 兼顾 nll 时效性 | 同步本身是「等 NPU 干完」，N 变小不会减少总等待 |
| ❌ p3 双缓冲流水线 | **未实现** | — |

⚠ **跨 token 流水（p3 的本意）当前不可做**：
`learn_softmax` 用的是**本步的 y**，提前提交无意义；而把「t 步的 learn_softmax」
延迟到「t+1 步的 forward_dev 之后、在同一个 NPU 批里执行」**会改变
`PC_learn` 读到的 nll 时序**（`task_modulation` 用 nll 门控本步主干学习率）
→ 等于换一套时序规则 → **违反铁律③**（P117 已裁决不做，收益 ~1.09× 不足以承担）。

新增手段与更深的诊断见
[NPU 利用率诊断与优化 P214](PHD-Net_NPU利用率诊断与优化_P214.md)。
