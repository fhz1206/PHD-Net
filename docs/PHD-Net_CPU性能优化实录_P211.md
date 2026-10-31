# PHD-Net CPU 性能优化实录（P211）

> **适用范围**：CPU 侧（x86 / aarch64）的模块级提速与 forward-only 路径改造。
> **数据截止**：2026-10-31（P211）。
> **相关文档**：[性能评估与迭代方案](PHD-Net_性能评估与迭代方案.md)（生产档性能唯一出处）、
> [前向only 训练路径](PHD-Net_前向only训练路径.md)（机制与 A/B）、
> [现状速查](PHD-Net_现状速查.md)（当前事实基线）。
> 与代码冲突时**一律以代码为准**。

---

## 0. 一句话

P211 在本机（x86 8 核、无 NPU）上完成了**四类 CPU 优化**（读出访存、forward-only
融合核、LTM 解释器开销、计时口径修复），**全部逐位不变**（端到端 final_PPL 恒为
13,017.891），并把 forward-only 变成默认路径。

**最重要的结论不是"快了多少"，而是"哪里快不了"**——见 §4 的诚实边界。

---

## 1. 测量口径（本节是全文前提）

| 项 | 值 |
|---|---|
| 平台 | Windows 11 / AMD64 / **8 核** / 无 NPU / OpenBLAS 0.3.34 |
| Python | 3.14.5，numpy 2.5.2，numba |
| 训练档 | `--preset smoke`（width=256）、`--data cc_traces`、3000 token |
| 读出形态 | **稀疏**（CLI 默认 `--readout-conn-k 128`），n_h = 768，词表 20,913 |
| 大空间表 | **在线 CSR**（CLI 默认 `--csr-online` 开） |
| 判据 | `[METRIC] ms_per_token`（4 次取最优）+ `final_ppl`（必须恒定） |

⚠ **本机不是生产平台**：生产是昇腾 aarch64 + NPU + 191 核。本节数字**只描述
CPU 侧算法与访存特性的相对变化**，不预测生产端到端。

⚠ **PPL 恒定是硬判据**：本次所有优化都要求**训练轨迹逐位不变**
（`final_ppl` 必须完全相同）。任何数值变更都算回归，不接受"差异在噪声内"。

---

## 2. 优化清单与实测收益

### 2.1 稀疏读出：CSR 列索引 int32（访存减半）

**问题**：`_from_dense_csr` / `_random_csr` 产出的列索引是 **int64**。读出的
CSR 前向与更新都是**流式访存**核（每趟顺序读 `idx` + `val`），墙钟由**字节数**
决定，与浮点运算量无关。

1B 档真实形态（V=51962 / k=128 / n_h=3072）：索引 53.2 MB + 值 26.6 MB。
索引字节数是值的两倍 → **压到 int32 直接砍掉一半索引流量**。

**改动**：`phdnet/sparse_pc.py::_compress_csr_idx`（新增）+ `readout.py` 构造时调用。
- 只在 `n_cols ≤ 2³¹−1` **且实测 `idx.max() ≤ 2³¹−1`** 时压缩；
- 否则**原样返回**（不静默截断 —— 截断会让寻址指向错误列 → 权重静默损坏）。

**实测**（1B 档真实形态）：

| | int64 | int32 | 加速 |
|---|---|---|---|
| CSR 内存 | 79.8 MB | **53.2 MB** | 1.50× |
| CSR 前向 | 4.18 ms | **3.25 ms** | 1.29× |
| CSR 更新 | 7.90 ms | **6.06 ms** | 1.30× |
| `learn_softmax` 总 | 12.03 ms | **10.23 ms** | 1.18× |

**逐位性**：int32 与 int64 索引的 matvec / add_outer **逐位相同**
（门禁 A3a/A3b，max|Δ| = 0）。索引只用于寻址、不参与浮点运算。

### 2.2 forward-only：融合核（6 次核调用 → 1 次）

**问题**：`learn_forward_target` 原路径 = 6 次独立 numba 核调用：
2×`_mv` + 2×`_add_outer` + 2×`_clip` + 2×`_scale_rows`（稳态缩放）。
每次调用都要付跨界与调度开销。

**改动**：`phdnet/sparse_pc.py::_pc_fo_fused`（新增 numba 融合核）——
核内完成「算预测 → 目标匹配残差 → 稀疏外积 → clip → 稳态缩放」全部 5 步，
**表达式与累加顺序逐字照抄各原子核** → 逐位等价。

**实测**（smoke 档 n=256）：0.110 → 0.045 ms（**2.4×**）；端到端 FO_learn 段
0.41 → 0.32 ms/tok。

**⚠ 一个被否决的优化（记录以免重踩）**：曾尝试「从 cache 的 `e0`/`e1` 复原预测」
以省掉 2 次 SpMV（`dn1·r2 = r1 − e1`）。**但它不逐位等价**：`pc.infer` 融合核
算 `e1` 时用 **fp32** 累加，而 `_csr_matvec` 用 **fp64** 累加 →
`r1 − e1` 与 `dn1·r2` **不逐位相同**（实测 max|Δ| = 3.2e-8 / 8.9e-8），
残差 dtype 也从 fp64 降为 fp32 → 端到端 `final_ppl` 13017.891 → 13019.683。
**fp32 的 e1 已丢失低位信息，无法从中还原 fp64 的乘积** ⇒ 只能选择
「保持逐位」或「接受数值变更」，本项目选前者 → 改为在融合核内**重算** pred。

### 2.3 forward-only：稳态缩放只作用于下行（省 2 次核调用）

**问题**：`pc.learn()`（homeostasis=True）走 `_pc_learn_fused_homeo` 核，
**核内已做 4 次 `scale_rows`**（含 up0/up1）；随后 `learn_forward_target`
又调 `_homeostatic_scale()` → **再缩放一次全部 4 个**。
而 forward-only **只改下行**（上行 up0/up1 恒为只读，P198 约束），
两次之间上行权重**没有任何变化** → 第二次缩放上行是**纯粹的重复劳动**。

**改动**：`_homeostatic_scale(down_only=True)` —— forward-only 路径只缩 dn0/dn1。
**语义更正确**（缩放只施加于实际被本路径改动的突触），且省 2 次核调用。

### 2.4 LTM dict 版：循环不变项提循环外 + `_enc` 内联

**问题**（cProfile 实测，smoke 档 64×64 组合）：

| 热点 | 调用数/200 次 learn | tottime 占比 |
|---|---|---|
| `learn` 本身 | 200 | 46% |
| `_enc` | **768,000** | **32%** |
| `max` / `min` 内建 | 各 1,536,000 | 23% |

且内层循环里有两个**循环不变**的 `_decay_read`：
- `_decay_read(t_post, stamp_post, i)` —— 只依赖 `i`，被重复算 |cur| 次；
- `_decay_read(t_pre, stamp_pre, k)` —— 只依赖 `k`，被重复算 |prev| 次。

**改动**（`phdnet/sparse_table.py::SparseSynapseTable.learn`）：
1. 两个 `_decay_read` 提到循环外（`i_tpost` 字典 + `g_tpre` 字典）；
2. `_enc` 在 float 模式内联（`float(min(max(w,0),w_max))`），int8 模式仍走 `_enc`。

**⚠ 严格等价的三处细节**：
- `_decay_read` 是**纯函数** → 提到循环外取值完全相同；
- 钳位写成**两个顺序独立的 if**（非 elif）——原式 `min(max(v,0),w_max)`
  在 `w_max < 0` 时返回 `w_max`（负），换序会返回 0；
- 突变测试契约保持 —— 仍**经由 `_decay_read` 取值**，负向对照可被检出。

**实测**：`table.learn` **3.74 → 1.87 → 1.06 ms（3.5×）**。

**⚠ 适用范围**：这是 **dict 版**（`--no-csr-online`）。**生产默认走在线 CSR**
（`--csr-online` 开），那条路径有 numba 批量核，本节的收益**不适用**——
见 §4.2。

### 2.5 计时口径修复：嵌套段不再重复计入

**问题**：`M4b_imprint` 的计时窗**完全嵌套在** `M4b_ltm` 内，但两者都被累加进
`_prof_step_accounted` ⇒ 「本步已计」比真实值**多出内层段** ⇒ `other`（残差）
被**低估**，且读者把各段相加会**重复计算** ltm（实测 smoke 档差 0.70 ms/tok）。

**改动**：`phdnet/model.py::_prof_nested_begin()` —— 内层段照常打印（细分可见），
但**不重复计入**「本步已计」基准。

**效果**：`other` 从 4.78 → 5.29 ms/tok（正是找回的被低估部分），
各段之和与 step 总时长**自洽**。**这是"没有准确分段数据就无法定位瓶颈"的
前提修复** —— 本次优化正是靠它才看清了 readout 才是真正的大头。

### 2.6 投影目标按标签缓存

`ForwardTargetProjector.project` 每步跑一遍 splitmix64 + tanh。语料 token 服从
zipf 分布，头部标签反复出现 → 按 `(label, layer)` 缓存（LRU 容量 4096）。
**不改变数值**（固定投影对同一 (label, layer) 恒得同一码，缓存只是重算→重取）。

---

## 3. 端到端实测

**严格 A/B**（同参数、`git stash` 对拍、4 次取最优）：

| | 基线（P210，显式 `--forward-only`） | P211（默认开启 + 全部优化） |
|---|---|---|
| `final_ppl` | 13,017.891 | **13,017.891**（逐位不变 ✓） |
| ms/tok（最优） | 8.26 | **8.01** |
| ms/tok（中位） | 8.35 | **8.12** |
| **readout 段** | 6.07 ms/tok | **4.93 ms/tok（1.23×）** |
| FO_learn 段 | 0.41 ms/tok | **0.32 ms/tok（1.28×）** |
| M4b_ltm 段 | 0.70 | 0.70（持平，见 §2.4 适用范围） |
| M6_replay 段 | 0.35 | 0.35（持平） |
| PC_learn 段 | 0.30 | 0.35（+0.05，融合核的额外开销） |

**⚠ 为什么端到端只快 1.03× 而分段显示 readout 快 1.23×**：
① 本机 8 核有其它负载，墙钟抖动 ±0.4 ms/tok，掩盖了分段收益；
② `readout` 段的 1.23× 只覆盖 4.93 ms 中的约 1.1 ms；
③ forward-only 本身**新增**了 FO_learn 段的开销（P210 也有，但默认开启后
每个跑训练的人都承担它）。

**forward-only 默认开启的 PPL 影响**（smoke 档，3000 token）：

| 配置 | final_ppl | ms/tok |
|---|---|---|
| forward-only **开**（新默认） | **13,017.891** | 8.12 |
| forward-only 关 | 13,355.328 | 7.61 |

**PPL 改善 2.53%**，方向明确为正；但**仍低于项目 <20% 判据**，
且在本机 smoke 档它换来的 PPL 改善与它自身的开销相抵。

---

## 4. 诚实边界（这一节比前面的数字更重要）

### 4.1 读出是访存下界，算法层已无空间

实测（1B 档真实形态）：读全量 64 MB 的 W 需 10.6 ms → **6.1 GB/s** 有效带宽；
而 BLAS 的 `W32 @ h32` 只需 2.7 ms（**23 GB/s**，分块 + 缓存友好）。
**稀疏 CSR 前向 3.25 ms ≈ 访存下界**。

进一步的对照实验（`bench_upd.py`）：

| 变体 | 更新核耗时 |
|---|---|
| 标准（随机索引） | 2.13 ms |
| **索引排序后**（顺序访问） | **2.13 ms（完全相同）** |

→ 更新核**不是 gather 瓶颈，是写带宽瓶颈**（写 `val` 比只读贵一倍）。
⇒ **在不改架构（稀疏化 / 量化 / 分块）的前提下，读出没有算法级提速空间。**
唯一剩下的杠杆是**减少 W 的体积**或**换更省访存的布局**——那是架构级改动，
不在本轮范围。

### 4.2 LTM 的 dict 版优化在生产默认路径上不生效

生产 CLI 默认 `--csr-online`（在线 CSR 表，有 numba 批量核 `_learn_batch`）；
本节的 3.5× 收益属于 **dict 版**（`--no-csr-online`）。
smoke 档实测：dict 版 `table.learn` 1.06 ms vs 在线 CSR 版 0.57 ms
（**CSR 版本来就更快**）→ 生产路径无收益。

**若要继续优化生产路径的 LTM**，方向是 `_learn_batch` 内的 gather/scatter
Python 循环（实测 gather 逐行拷贝 0.136 ms → `np.concatenate` 0.093 ms，
可省 ~0.04 ms/call），收益有限。

### 4.3 本机数字不可外推到生产

本机是 x86 8 核、无 NPU；生产是昇腾 aarch64 + NPU + 191 核。
`numba` 的 `prange` 行为、`fastmath` 的编译器实现、BLAS 的分块策略
在两个平台上都不同。本文的收益数字**只说明"这些访存/调用次数问题存在且可修"**，
不预测生产端到端提速比。生产数字须在服务器上按
[性能评估与迭代方案](PHD-Net_性能评估与迭代方案.md) 的口径重测。

### 4.4 顺带发现的两个既有问题（未修，记录待办）

1. **`--replay-every` 疑似未接线**：本轮实测 `--replay-every 4` 与
   `--replay-every 16` 的 `M6_replay` 段耗时**相同**（0.32–0.35 ms/tok），
   与 config 注释宣称的「replay_every=16 → M6_replay 1.2–1.4 ms/tok」
   不符 → 该 CLI 疑似**没接到 cfg**。**本轮未修**（超出范围）。
2. **`--forward-only` 探针的滑动窗口**：`segments(win)` 是 500 步滑窗，
   而 `probe_PPL` 只在第一次探针时算出（日志里三次探针值相同）→
   **probe_PPL 不是每个窗口都刷新**。判据使用时需注意。

---

## 5. 门禁

`tests/gates/gate_cpu_opt_p211.py`（P211 新增，**15 项**）：

| 组 | 内容 |
|---|---|
| A | int32 索引：放得下则压、放不下则**原样返回**、索引值不变、matvec/add_outer 与 int64 **逐位相同**、真实读出走 int32 |
| B | forward-only 融合核：与多核参考实现**逐位相同**、**不触碰上行**、不新增连接、稳态缩放只作用于下行 |
| C | LTM 内联：与原版 learn **逐位相同**（float + int8）、突变测试契约仍可检出 |

**C2 的场景构造要点**（缺一个就会**假通过**）——这是本轮踩过的坑：

1. **必须自增 `step_count`** —— `_decay_read` 的补偿量是 `λ^(step_count − stamp)`，
   而 `step_count` 由调用方（`bigltm.SparseLTM.imprint`）自增；裸调
   `table.learn()` 不自增 ⇒ dt 恒为 0 ⇒ 补偿永不生效 ⇒ 突变注入变成空操作。
2. **必须用小的 id 空间**（48 而非 2²⁰）—— id 空间大则各步几乎不重复 →
   `stamp` 永不滞后 → 补偿无从体现。
3. 必须让 **`k ∈ cur` 但 `k ∉ prev`**（P193 修的正是 LTD 的第二条迹 `t_pre[k]`）。

---

## 6. 变更清单

| 文件 | 改动 |
|---|---|
| `phdnet/config.py` | `forward_only` 默认 True；注释改为 P211 语义 |
| `phdnet/forward_only.py` | 投影目标按 `(label, layer)` 缓存 |
| `phdnet/model.py` | forward-only 默认化；嵌套段计时修复；FO 段局部绑定 |
| `phdnet/sparse_pc.py` | `_compress_csr_idx`（新）；`_pc_fo_fused`（新）；`_homeostatic_scale(down_only)` |
| `phdnet/sparse_table.py` | `_decay_read` 提循环外；`_enc` 内联 |
| `phdnet/readout.py` | 稀疏读出 CSR 索引压 int32 |
| `train/train.py` | `--forward-only` → `--no-forward-only`（默认开） |
| `tests/gates/gate_forward_only.py` | 30 项（P211 新契约） |
| `tests/gates/gate_cpu_opt_p211.py` | 新增 15 项 |
