# NPU「假占用」诊断与优化（P214）

> **适用范围**：昇腾 NPU 上「设备占用 90%+ 但算力/吞吐没达到对应水平」的
> 定位、判据、已落实的优化与残留瓶颈。
> **数据截止**：2026-11-04（P214）。
> **数据来源**：`train_1b_1b_pretrain_20261101-140007.log`（1b 档、昇腾 aarch64
> 191 核 + CANN 8.5 + NPU、24.5k token 稳态）、项目既有 msprof（P141–P145）
> 与代码核查（`phdnet/backends/accel_readout.py`、`phdnet/config.py`）。
> **相关文档**：[性能评估与迭代方案](PHD-Net_性能评估与迭代方案.md)（性能数字唯一出处）、
> [并行与加速架构分析](PHD-Net_并行与加速架构分析.md)（为什么吃不满）、
> [P201 并行瓶颈诊断](PHD-Net_P201_M2-PC并行瓶颈诊断.md)（M2/PC 侧）、
> [NPU 加速特性现状](P205a_NPU流水线设计.md)（`--beta` 四档）、
> [CPU 侧加速方案 P213](PHD-Net_CPU侧加速方案_P213.md)。
> 与代码冲突时**一律以代码为准**。

---

## 0. 一句话

**「占用 90%」是真的，但它测的不是算力。** 本项目实测
**`aicore_time = 0`、`aiv_time = 99.6%`、`Task Wait = 6.8× 自身耗时`**
→ 设备一直「活跃」，但**有效矩阵计算几乎为零**，时间花在 Vector 单元的
索引型算子下发/排队上。这是**指标虚高**，不是真实计算利用率。

本轮落实了两件可验证的事：
1. **CPU 侧 LTM 优化**（哈希槽位查找 + 持久化缓冲）→ `imprint` **1.83×**，
   直接缩短 NPU 等 CPU 的空窗；
2. **`tools/probe/diagnose_npu_utilization.py`** —— 把「假占用」判读自动化，
   输入 msprof CSV 直接给判定 + 归因 + 判据（含 P142 的拒绝出结论保护）。

⚠ **服务器侧 A/B 仍待 fhz 执行**（清单见 §6），本文不把未实测的收益写成结论。

---

## 1. 现状时间预算（服务器 1b 档稳态）

| 项 | ms/tok | 占比 |
|---|---|---|
| **端到端 step** | **10.8** | 100% |
| CPU 侧（M1–M5 + other 未归因） | **~6.5** | **60%** |
| NPU 读出（M6） | 3.5 | 32% |

CPU 主要项：`M6_replay` 1.5–1.8 / `M4b_ltm` 0.9–2.2（波动大）/
`M2_infer` 0.65 / `M1_encode` 0.30 / `other` 内未归因 ~1.5。

```
CPU: M1 → M2 → M3 → M4ab → [提交读出] → PC_learn → STDP
NPU:                  └── forward_dev ──┘   └── learn_softmax ──┘
     └──────────── 10.8 ms 串行周期 ───────────┘  NPU 只有 3.5 ms 的活
```

**NPU 每步只有 3.5 ms 的活，却有 10.8 ms 的周期** → 队列长时间空转。

---

## 2. 「指标虚高」 vs 真实计算利用率：判断依据

### 2.1 指标口径必须分层

| 指标 | 实际测什么 | 能否代表算力 | 本项目读数 |
|---|---|---|---|
| 设备占用（npu-smi / 上下文活跃） | 进程在设备上、任务队列非空 | ❌ | ~90%+ |
| `aicore_time`（Cube/MAC） | 矩阵单元**有效执行**时间 | ✅ 最接近 | **0** |
| `aiv_time`（Vector） | 向量/访存型算子执行 | ⚠ 在算但效率低 | **99.6%** |
| `Task Duration` | **含排队**的墙上时间 | ❌ | — |
| `Task Wait Time` | 纯等待（没被调度到） | ❌ 恰恰是空转证据 | **6.8× 耗时** |

### 2.2 三条判读纪律（P144 踩过两次坑）

1. **算子名里的硬件名 ≠ 它跑在那个硬件上。**
   `aclnnIndex_IndexAiCore_Index` 名字带 "AiCore"，但 `aicore_time=0`、
   `aiv_time=99.6%` → 跑在 **Vector 单元**。"AiCore" 只是 CANN 的**分类标签**。
2. **`Task Duration` 含排队，不等于执行时间。**
   `Index` 的 Task Wait 是自身耗时的 **6.8×** →「算子慢」是错误表述，
   真因是**没被调度到**。
3. **每个 0 都要问「是真 0 还是缺数据」**：第一版解析器只取 `aicore_time`
   一个指标，把 0 当缺数据 → 误判。本项目已用多指标交叉验证确认为**真 0**。

### 2.3 量化依据：为什么是「算子效率」而非带宽饱和

| 证据 | 数值 | 含义 |
|---|---|---|
| 读出耗时 vs 带宽下界 | 7.83 ms vs **0.300 ms** = **26.1×** | 不是带宽墙 |
| 同访存量 Vector vs Cube | `Index` 8.1 GB/s vs `BatchMatMul` 38 GB/s = **4.7×** | Vector 单元低效 |
| `Index` 占比 | **89.3%**（msprof 600 步） | 稀疏 gather 是单点热点 |
| 每步 kernel 数 | ~5–12 个 eager 小算子 | 下发/队列主导 |
| CANN launch 开销 | 50–200 µs/kernel → ~0.5 ms/step | 只占读出 6%，**量级不足以单独解释** |
| CPU/NPU 串行相加 | 4.26 + 7.83 = **12.09 ms**（与端到端逐位一致） | **无任何隐含重叠** |

**结论**：主因是 **① 等 CPU（队列背压）** + **② Vector 单元低效** 的叠加；
`launch` 开销量级不够单独造成假占用；同步点已被 `nll_sync_every=8` 摊薄过一轮。

---

## 3. 逐项瓶颈核查（代码 + 运行时数据）

| 类别 | 核查结果 | 判定 |
|---|---|---|
| **算子间空隙 / 同步等待** | `forward_dev` 返回**设备张量、零同步**；`nll_sync_every=8` 把 `.item()` 摊到 1/8 步；`correct` 在主机侧算省一次 `.item()`；P205a `--beta p2` 双流（H2D 独立流 + nll 独立流）已实现 → **每步显式同步点已压到 1/8 步一次** | 已大幅缓解；**残留待 A/B 证实**（`--nll-sync-every 1 vs 8`） |
| **串行依赖 / 流水线断点** | 逐 token 语义**禁止数据并行**（铁律③）：t+1 的状态依赖 t 的学习结果。P192 核对：`12.09 = 4.26 + 7.83` 完全串行，但**代码里没有阻塞同步** → 成因是**运行时下发/队列背压**，不是显式同步 | **结构性主因**，不可通过改训练主循环消除 |
| **Host/Device 传输与拷贝** | `_staged_to_dev` 用 pinned 暂存 + `non_blocking`；`learn_softmax` 复用 `forward_dev` 缓存的 `ht`/`y`，**不重复 H2D**；权重常驻设备 | 已优化，非瓶颈 |
| **Host 侧下发 / launch** | 稀疏路径 `torch_compile` 默认 **False**（P58 指令）→ softmax/log/sub/addmm_ 各自一个 eager kernel；每步 ~5–12 个 | **是**（`wait 6.8×` 的直接来源），但量级仅 ~0.5 ms |
| **算子切分与融合不足** | `Index`（`_sp_gather` 的 `ht[Wi]`）占 89.3%，跑在 Vector；`--beta p0/p1` Graph Capture 与前向+学习融合图**已实现但默认没开** | **是**（单点最大热点） |
| **精度配置** | 读出 fp32（精确 `p−t` 规则）；混合 dtype 会让 `fp32 W @ fp64 x` 脱离 BLAS 走慢路径且**无断言检查**；fp16/bf16 会把非目标行更新舍掉 → 退化为纯 Hebbian | 正确配置，**不要为提速改精度** |
| **调度配置** | `TASK_QUEUE_ENABLE=2` + `COMBINED_ENABLE=1` 自 P120 默认开，但官方有两个**静默失效**条件（仅二进制场景生效；`ASCEND_LAUNCH_BLOCKING=1` 强制关闭）且**不打日志** → `--cann-dispatch` 只核对、不新增行为 | 待服务器核对 |

### 3.1 为什么「跨 token 流水 / 双缓冲」当前不能做

`--beta p3=双缓冲流水线` 在 `--help` 里列了，**代码里没有 p3 分支**（静默无效）。
更根本的原因：

- `learn_softmax` 用的是**本步的 y**，提前提交无意义；
- 把 t 步的 `learn_softmax` 延迟到 t+1 步 `forward_dev` 之后、在同一个 NPU 批里执行
  → 会改变 `PC_learn` 读到的 **nll 时序**（`task_modulation` 用 nll 门控本步主干学习率）
  → 等于换一套时序规则 → **违反铁律③**（P117 裁决不做，收益 ~1.09× 不足以承担）；
- `readout_recurrence=True` 时，流水会让 `recur_cue` 拿到**上一步**的 y →
  **静默语义错误**。

---

## 4. 已落实的优化（本轮，可验证）

### 4.1 CPU 侧 LTM：**1.83×**（直接缩短 NPU 等 CPU 的空窗）

`M4b_ltm` 是 CPU 侧最大项（0.9–2.2 ms/tok），项目自记
`table.learn` 占 `M4b_imprint` 的 **88%**。两个改动：

**(a) 哈希槽位查找**（`phdnet/ltm_kernel.py::_ltm_learn_rows_hashed`）

原核三重循环 `for r in prange(R): for c in range(C): for s in range(sz)`
→ 1b 档 R=C≈512、m_out=72 时**每步约 1,900 万次 int64 比较**。
`_find_slot` 注释「线性扫描成本可忽略」对 dict 版成立，对 prange 批量核**不成立**。

改为**每行一张开放寻址哈希表**（容量 2 的幂 ≥ 2·(m_out+1)，装填 ≤ 50%）：
建表按槽位升序、同键不覆盖 → 保留「**首个**匹配槽」语义；生长后立即入表。
复杂度 `O(R·C·sz)` → **`O(R·(sz+C))`**。

**(b) 持久化 K/V 缓冲**（`OnlineCSRTable._batch_buffers`）

原每步 `np.zeros((R, cap))×2` → calloc ~1.2 MB + 清零 + page fault，
外加 512 行 Python `max` 求 cap。改为按上限预分配一次、每步切片复用。
⚠ **不清零是安全的**：所有被 scatter 读取的列都被 gather 或核内生长写过
（门禁 D4 用脏缓冲验证）。`cap ≡ m_out` 由 `_new_row`/`_grow_row` 的上限证明。

**实测分解**（本机 x86 8 核，R≈512 的 1b 形态，`imprint` 端到端中位）：

| 组合 | 中位 | vs 原版 |
|---|---|---|
| A 线性核 + 每步 zeros（原版） | 10.91 ms | 1.00× |
| B 仅持久化缓冲 | 9.57 ms | 1.14× |
| C 仅哈希核 | 6.99 ms | 1.56× |
| **D 两者（生产现状）** | **5.97 ms** | **1.83×** |

⚠ 绝对数字是**本机**的（README §4：x86 数字不构成昇腾证据）；
**算法复杂度改善与平台无关**，但服务器收益必须由日志 `M4b_ltm` 段确认。

**门禁**：新增 `tests/gates/gate_ltm_hash_lookup.py`（**10/10**）：
多 seed 逐位一致 / int8 量化 / 小 m_out 饱和期 / 两核同输入 K/V/S 相同 /
缓冲视图复用 + 自动扩容 + 脏缓冲不清零安全 / 哈希表容量契约。
全量门禁 **53/53 通过**。

### 4.2 profiling 埋点：假占用判读自动化

新增 `tools/probe/diagnose_npu_utilization.py`：

```bash
# 服务器采集
bash tools/scripts/prof_readout_msprof.sh
# 判读（本机也行，纯文本）
python tools/probe/diagnose_npu_utilization.py outputs/prof/msprof_readout \
    --log train_1b_1b_pretrain_20261101-140007.log
# CI 用 JSON
python tools/probe/diagnose_npu_utilization.py <csv> --json
```

输出：四个口径汇总、Top 算子表、**假占用判定 + 判断依据**、按证据强度的
归因与建议动作、期望方向。
⚠ 继承 P142 教训：**列名子串匹配**；识别失败**拒绝出结论**（返回码 3），
绝不按位置猜（那会把 `Model ID` 读成算子名却照样打印表格）。

### 4.3 已有的手段（不是本轮新做，但必须知道）

| 手段 | 状态 | 能解决 |
|---|---|---|
| `--beta p0` Graph Capture | 已实现，默认关 | kernel launch 开销 |
| `--beta p1` 前向+学习融合图 | 已实现，默认关 | 两图之间的下发间隙 |
| `--beta p2` 双流 H2D + nll 独立流 | 已实现，默认关 | H2D 与 kernel 重叠、同步点摊薄 |
| `--beta p3` 双缓冲流水线 | ❌ **未实现**（help 有、代码无） | — |
| `--nll-sync-every 8` | 默认开 | `.item()` 硬同步摊薄 |
| `--cann-dispatch` | 默认关 | 核对 task_queue 静默失效 |
| `torch_compile` | 默认 False（P58 指令） | 稀疏路径算子融合 |

详见 [NPU 加速特性现状](P205a_NPU流水线设计.md)。

---

## 5. 判据：不要用「占用率」当优化目标

| 指标 | 期望方向 |
|---|---|
| **端到端 ms/tok** | **下降才是收益** |
| **`segments(win)` 的 readout** | 下降 |
| **`aicore_time` 占比** | **上升**（真正在算矩阵） |
| `aclnnIndex` 占比 | **下降**（少碎算子 / 少下发） |
| `Task Wait / Duration` | **下降**（少排队） |
| 设备占用（npu-smi） | **可能下降** —— 空窗缩短后占用下降是**好事** |

⚠ **PPL 判据一律用 `probe_PPL`**（`--probe-every 1000 --probe-tokens 2048`）：
训练流 sliding PPL 信噪比仅 0.40，<20% 的差异测不出来。

---

## 6. 服务器待执行清单（需要 fhz）

| # | 命令 | 判据 | 目的 |
|---|---|---|---|
| 1 | `bash tools/scripts/prof_readout_msprof.sh` → 回传 `op_summary*.csv` | 喂给 §4.2 判读工具 | 拿到本批次真实的 `aicore/aiv/wait` 分解 |
| 2 | `python tools/bench/bench_server_threads.py --threads 8,16,32 --preset 1b --data pretrain --remote-data --tokens 20000` | 端到端 ms/tok + `segments(win)` 各段 | 补 P204 遗留的线程 A/B 空白 |
| 3 | 同参数 `--nll-sync-every 1` vs `8` | readout ms/tok + probe_PPL | 证/否证「还有隐藏同步点」 |
| 4 | 基线 / `--beta p0,p1,p2 --cython-kernels auto` / 再加 `--cann-dispatch` | readout ms/tok + ms/tok + probe_PPL | 量化融合与 Graph 的收益 |
| 5 | 正常跑一轮 1b，对比 `M4b_ltm` 段 | 1.96 ms（旧日志） vs 新值 | 确认 §4.1 在昇腾的收益 |

---

## 7. 残留瓶颈与后续可优化点

**结构性的，改不动**：
- 逐 token 语义 → 不能数据并行；跨 token 流水违反铁律③；
- 每步只提交 ~5–12 个小 kernel → 队列背压是**架构性**的。

**还能做（按性价比）**：
1. **`Index` 算子换实现**：把稀疏 gather 从 Vector `index` 改成 Cube 友好的
   SpMM / 合批 —— 单点占 89.3%，是**唯一能大幅挪动 `aicore_time` 的方向**。
   但需自定义算子或改稀疏表示，**风险高，须先有 msprof 基线**。
2. **稀疏路径重评 `--torch-compile`**（P201 建议 5）：默认 False 是 P58 指令，
   但本机已实测「融合与 eager 逐位相同」→ 服务器 A/B 有空间。
3. **`--beta p0/p1/p2` 转默认**（若 §6 第 4 项证实收益且 probe_PPL 无回归）。
4. **M6_replay 摊销再评估**：`replay_every=16 → 32` 可省 ~0.8 ms/tok，
   但 P204 实测「replay 少反而略差」（P197 锚定有效）→ **必须 probe_PPL 判**。
5. **M4b 的 Python 循环**（gather/scatter/3× `_decay_read_arr` 占 66%）：
   理论上限 ~1.0 ms/tok，但 `_decay_read_arr` 有**突变测试契约**
   （必须逐个调 `_decay_read`），改动风险高。
6. **`--enable-cpu-parallel` 是死配置**（代码里从未被读取）→ 先决定接线还是删。

---

## 8. 变更清单（P214）

| 文件 | 改动 |
|---|---|
| `phdnet/ltm_kernel.py` | 新增 `_ltm_learn_rows_hashed`（哈希槽位查找）；原线性核保留作对拍/回退 |
| `phdnet/sparse_table.py` | 新增 `_batch_buffers`（持久化 K/V/S/HK/HV 复用）+ `LTM_HASH_LOOKUP` 开关；`_learn_batch` 去掉每步 `np.zeros×2` 与 512 行 Python max，默认走哈希核 |
| `tests/gates/gate_ltm_hash_lookup.py` | **新增**（10 项：逐位一致 / int8 / 饱和期 / 缓冲复用安全 / 容量契约） |
| `tools/probe/diagnose_npu_utilization.py` | **新增**：msprof 假占用判读（判定 + 依据 + 归因 + JSON；列名识别失败拒绝出结论） |
| `docs/PHD-Net_NPU利用率诊断与优化_P214.md` | **新增**：本文 |
