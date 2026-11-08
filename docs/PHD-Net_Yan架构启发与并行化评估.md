# Yan 架构启发：单 token 计算量与并行化可行性评估

> **数据截止**：2026-11-07（P217）。
> **范围**：调研 RockAI「Yan / MCSD」架构，评估其思路能否简化本项目
> 单 token 计算量、能否把逐 token 串行改成并行。
> **结论性质**：评估与方案，**不是已实现的功能**。任何语义变更必须先过
> `probe_PPL` 门禁；本机 x86 数字不构成昇腾证据（铁律）。
> 相关：[NPU 假占用诊断](PHD-Net_NPU利用率诊断与优化_P214.md)（当前瓶颈实证）、
> [架构设计](PHD-Net_架构设计.md)（铁律四条）、
> [并行与加速架构分析](PHD-Net_并行与加速架构分析.md)。

---

## 0. 结论先行

**Yan 的「可并行训练」不能直接照搬**，原因是架构性质不同，不是优化没做到：

| | Yan / MCSD | PHD-Net |
|---|---|---|
| 训练范式 | **离线**：整条序列用**同一份权重**一次前向，再统一反向传播 | **在线学习**：每个 token 都更新权重（STDP / LTM imprint / PC learn） |
| 序列并行 | ✅ 下三角矩阵一次 matmul 算完 N 个位置 | ❌ token t+1 的前向依赖 token t 更新后的权重 |
| 推理状态 | O(1)（`S_slope`, `S_decay` 两个向量） | **已经是 O(1)**（固定容量 CSR 表 + 衰减迹） |
| 复杂度 | O(N·d) | **已经是 O(d)**（逐 token 增量，无历史重算） |
| 稀疏激活 | 类脑分区激活（部分神经元参与） | conn_k=128 结构稀疏 + 活跃行 CSR |

**换句话说**：Yan 用「并行训练 + 循环推理」把 Transformer 的 O(N²) 降下来；
我们**根本没有 O(N²) 项**，也没有 KV cache，早就是 Yan 想要的目标形态。
我们的串行**不是 Attention 造成的**，而是**在线学习规则**造成的（铁律③）。

**但 Yan 有两个思路确实可以落地**，且能直接打中当前实测的最大瓶颈
（`Index` 占 NPU 时间 **80.7%**，见 P214 §0）：

1. **预定义衰减 / 线性递推** → 我们的 `λ^Δt` 迹已是同构，
   可以改成 **associative scan** 并行（语义保持）。
2. **分区激活 / 只算需要的神经元** → 我们每步对**整个词表 V=51,962**
   做 `ht[Wi]` gather（6,651,136 元素 = 25.37 MiB），这是 `Index` 80.7% 的根源。

---

## 1. Yan / MCSD 到底是什么（核实过的公式）

来源：RockAI「Yan 架构」（2024-01 发布 Yan 1.0，非 Attention、非 RNN）、
MCSD 论文 **arXiv:2406.12230**《MCSD: An Efficient Language Model with Diverse Fusion》。

### 1.1 训练时：全序列矩阵形式（可并行）

输入 `X ∈ R^(N×D)` reshape 成 `C` 个通道 `(C, N, Dc)`，`D = C·Dc`。

```
U = X·W_U,  V = X·W_V        (slope 分支)
F = X·W_F,  E = X·W_E        (decay 分支)

V' = W_s · V                 W_s = (s_0..s_{C-1})，s_i 是下三角 softmax 矩阵
E' = W_d · E                 W_d = (de_0..de_{C-1})，de_i 是下三角 RMSNorm 矩阵

Θ_s = SiLU(V') ⊙ U
Θ_d = E' ⊙ sigmoid(F)
Θ   = Concat(Θ_s, Θ_d)
```

预定义权重（**不学习**，这是能一次 matmul 的关键）：

```
β_i = (2^(-8/C))^(i+1)          # slope，通道 i 的衰减速度
α_i = 1 - 2^(-5-i)              # decay，通道 i 的衰减系数
```

`W_s`/`W_d` 上三角为 `-∞`（因果 mask），所以 `V'`/`E'` 的第 n 行只依赖前 n 个位置
→ **一个 (N×N) @ (N×Dc) 的 matmul 就能并行算出整条序列**。

### 1.2 推理时：递推形式（O(1) 空间）

```
S_{n+1}^{decay,i} = α_i · S_n^{decay,i} + α_i · E_n          ← 线性递推，核心
S_{n+1}^{slope,i} = (1 - 1/Z_n)·S_n^{slope,i} + (1/Z_n)·V_n   Z_n = Σ_{j=0}^{n-1} e^{-jβ_i}
θ_{n+1}^i = U ⊙ SiLU(S^{slope}) + sigmoid(F) ⊙ RMSNorm(S^{decay})
```

每个通道只存两个 `R^(1×Dc)` 向量 → 状态规模 `2·C·Dc`，**与 N 无关**。

### 1.3 与我们的同构点

| Yan 概念 | 我们代码里的对应 |
|---|---|
| `α_i` 预定义衰减 | LTM 迹 `t_pre/t_post` 的 `λ^Δt` 补偿（`_decay_read`） |
| `S^{decay}` 线性递推 | 迹递推 `trace_t = λ·trace_{t-1} + x_t` |
| O(1) 状态 | `OnlineCSRTable` 固定容量 + 稀疏连接 |
| 类脑分区激活 | `conn_k=128` 结构稀疏、CSR 活跃行 |
| 可并行训练 | ❌ **不适用**（在线学习） |

---

## 2. 为什么「直接并行」行不通（必须说清）

`step(t)` 的真实依赖链（P192 已核对）：

```
M1_encode → M2_infer → M3_pred → M5_mod → M4a_wm → M4b_ltm   [CPU]
→ M6 forward_dev + learn_softmax                              [NPU]
→ PC_learn (用本步 M2 cache) → STDP_learn (用 _prev_rate)
跨 token 状态：_prev_rate ← rate(t)、_prev_pc ← cache(t)、bigltm._prev ← cur(t)
```

- `M2_infer(t+1)` 依赖 PC 权重，而它在 `PC_learn(t)` 里更新；
- `PC_learn(t)` 在 `M6(t)` 之后 → **`M6(t) → PC_learn(t) → M2(t+1)` 不可打断**；
- 前移 `learn_softmax` 会改变 `PC_learn` 读到的 nll 时序
  （`task_modulation` 用 nll 门控本步主干学习率）→ 违反铁律③；
- `readout_recurrence` 打开时，流水会让 `recur_cue` 拿到**上一步**的 y
  → **静默语义错误**。

**Yan 能并行是因为它的「权重」在整条序列里是常量**（`W_s/W_d` 甚至是预定义的），
前向算完才一起更新。我们每 token 都改权重，序列维不存在可并行的独立位置。

⚠ 想真正并行训练，唯一出路是**改变学习规则**（把在线学习改成批内延迟更新），
那是**语义变更**，必须 `probe_PPL` 判据，不能靠代码技巧绕过。

---

## 3. 可落地的三条路径（按风险排序）

### 路径 A（**语义保持、零风险**）：把 Vector `Index` 换成 Cube 友好的等价算子

**当前实测**（P214 §0，msprof）：
`aclnnIndex_IndexAiCore_Index` 21,245 次、**80.7%**、均 6.5 ms/次 ≈ **4 GB/s**；
同机 `BatchMatMul` 在 Cube 上是它的 **4.7×**。

前向是 `y[i] = Σ_j W[i,j] · ht[Wi[i,j]]`，`Wi` 形状 `(V=51962, k=128)`。
它数学上等价于一个**稀疏矩阵乘**：

```
S ∈ R^(V×H)，第 i 行只有 k 个非零：S[i, Wi[i,j]] = W[i,j]
y = S @ ht            # SpMM / GEMV，走 Cube
```

`Wi` 是**静态的**（初始化后不变），所以 CSR 结构可预构建常驻设备，
每步只需更新 `data`（跟着 `W` 变）。**数学等价，不碰学习规则。**

⚠ 本机 x86 原型（概念验证）：`max|Δ| ≈ 2e-6`（fp32 累加顺序差异，**容差级、
非逐位**）；「按列乘加」的朴素实现在 CPU 上反而慢 3.4×（128 次 Python 循环，
不是真 SpMM kernel）→ **这个数字绝不能外推昇腾**，必须 NPU 实测。

**先测的两件小事（零风险，开关已存在）**：
1. `--sp-gather take`（P134 已有的另一条 gather kernel）；
2. `--sparse-fwd-kernel einsum`（不物化 `(V,k)` 中间张量，省 25.37 MiB 读 + 写）。

### 路径 B（**语义保持**）：线性状态递推用 associative scan 并行

`trace_t = λ·trace_{t-1} + x_t` 是**可结合**的一阶线性递推，
可用 Blelloch parallel scan 在 `O(log N)` 深度算完整个序列，
结果**数学等价**（浮点结合律需容差判据）。

**适用范围有限**：只有「输入 `x_t` 不依赖前面 token 的学习结果」的那些状态才行
（例如纯编码侧的衰减迹）。LTM 的 `t_pre/t_post` 里 `prev/cur` 编码依赖表状态，
需要逐个确认。**推理路径（`learn=False`）最干净** —— 那时没有权重更新，
整条序列的状态传递可以完全并行。

⚠ 收益预期要诚实：迹更新在单 token 里占比很小，这条路径**不能解决
`Index` 80.7%**，但它是唯一「不改语义就能把某段串行改并行」的地方。

### 路径 C（**语义变更，高风险，需 probe_PPL**）：分区激活 / 批内延迟更新

| 做法 | 对应 Yan 思路 | 单 token 计算量 | 代价 |
|---|---|---|---|
| 只算 top-K 候选输出行，跳过其余 V | 类脑分区激活 | `V·k` → `K·k`（K=512 时 **~100×**） | softmax 分母有偏 → 需 sampled / 层次 softmax，改学习规则 |
| 批内 B 个 token 用同一份权重并行前向，批末一次性更新 | 可并行训练 | 前向摊薄 B× | **每个 token 的学习延迟 B 步**，违反在线语义 |
| 降 `conn_k` 128→96/64 | 结构稀疏 | 线性下降 | 模型容量变小，PPL 待判 |

⚠ **这三条都不能直接改默认值**。判据必须是 `probe_PPL`
（`--probe-every 1000 --probe-tokens 2048`），**不是** sliding PPL（信噪比 0.40）。
历史教训：P110 实测低精度导致非目标行更新保留率暴跌（fp32 99.95% →
fp16 26.67% → bf16 5.79%）→ 退化为纯 Hebbian。**「少算一些行」很容易
踩同一个坑。**

---

## 4. NPU 实测结果（2026-11-08，AtomGit Notebook）

环境：**Ascend910B4** × 1 / CANN 8.5 / torch 2.9.0+cpu + torch_npu /
Python 3.11.14。`V=51962, k=128, H=3072`，warmup 8 / repeat 50。
脚本：`tools/bench/npu_notebook_benchmark.py`（自包含，结果自动导出
`npu_gather_bench.json` / `.txt`）。

### fp16（生产读出计算 dtype）

| 实现 | ms/step | vs A | max\|Δ\| |
|---|---:|---:|---:|
| **A `ht[Wi]`（生产，`aclnnIndex`）** | **6.489** | 1.00× | — |
| B `torch.take` | 6.289 | 1.03× | **0** |
| C `einsum` | 7.029 | 0.92× | **3.906e-03 ⚠ 非等价** |
| **D `torch.index_select(ht,0,Wi.ravel())`** | **0.123** | **52.80×** | **0** |
| E 分区激活（只算 512 行） | 0.085 | 76.08× | 0 |
| E 分区激活（只算 2048 行） | 0.281 | 23.07× | 0 |
| F "稠密 SpMM" `S@ht`（V=4096） | 0.094 | 69.11× | **1.026 ⚠ 不等价** |

### fp32

| 实现 | ms/step | vs A | max\|Δ\| |
|---|---:|---:|---:|
| **A `ht[Wi]`（生产）** | **6.743** | 1.00× | — |
| B `torch.take` | 6.333 | 1.06× | **0** |
| C `einsum` | 7.957 | 0.85× | 1.192e-06 |
| **D `index_select`** | **0.228** | **29.63×** | **0** |
| E 分区激活（512 / 2048 行） | 0.095 / 0.285 | 70.76× / 23.67× | 0 |
| F "稠密 SpMM"（V=4096） | 0.043 | 157.55× | **1.051 ⚠ 不等价** |

### 判读（三条，别读错）

1. **`index_select` 是本次最大收获**：与生产 `ht[Wi]` **逐位相同**
   （`max|Δ| = 0`），却在 NPU 上快 **52.80×（fp16）/ 29.63×（fp32）**。
   它绕开了 `aclnnIndex`（Vector 单元，msprof 占 80.7%）。
   → 已接成 `--readout-gather-impl index_select`（**默认仍是 index**），
   门禁 `gate_gather_impl.py` 17/17 锁住逐位等价契约。
2. ⚠ **F 项名字有误导，别当等价实现**：`S.scatter_(1, Wi, W)` 对
   **重复列索引是覆盖而非累加**，与 `Σⱼ W[i,j]·ht[Wi[i,j]]` 分歧
   （`max|Δ|≈1.05`）。门禁的负向对照也确认了这点（`max|Δ|=5.6`）。
   真要做 Cube 友好的 SpMM，必须用 CSR 结构（重复列分别累加），
   不能靠 `scatter_` 稠密矩阵。而且 F 的 V 只有 4096（全量的 1/12），
   **157× 这个数字不能与 A 直接比**。
3. ⚠ **`einsum` 在 fp16 下不等价且更慢**：`max|Δ|=3.906e-03`、0.92×。
   所以 `--sparse-fwd-kernel einsum` 在 fp16 读出臂**不能用**
   （fp32 下 1.192e-06 属容差级，但同样 0.85× 更慢）。
4. **E 是理论上限，不是可用优化**：只算 512 行确实 76×，但
   `learn_softmax` 的 softmax 需要全 V 的 logits；真的砍行就得改
   学习规则 → `probe_PPL` 判。P110 的坑还在（少算行/降精度 →
   非目标行更新保留率暴跌）。
5. ⚠ **`index_select` 的 52.80× 是算子层面**，不代表生产端到端也快 52×：
   readout 只占端到端 ~40%，且生产 `_sp_gather` 还有缓存/量化分支。

⚠ **以上是孤立算子 benchmark**。生产 `_sp_gather` 还有缓存
（P111/P143）、fp8/int8 量化臂、`_gather_dtype` 降精度分支 ——
真实收益必须服务器 A/B + `probe_PPL` 才能定，**所以默认值没动**。

另注：脚本里的 `MiB_per_step_fp16 = 12.69`（`V·k·2 bytes`）是对的；
P214 文档写的 25.37 MiB 是 **fp32** 中间张量口径，两者不矛盾。

## 5. 建议的验证顺序（在 NPU notebook 上）

环境：AtomGit Notebook，**1 × NPU 910B4 / CANN 8.5 / Python 3.11**，与生产服务器同构。

1. ✅ **算子级 benchmark 已跑完**（§4）：结论是 `index_select` 逐位相同且
   快 30–53×。脚本 `tools/bench/npu_notebook_benchmark.py`，结果
   `npu_gather_bench.json/.txt`。
2. **下一步：生产路径 A/B** —— 用同一份数据、同参数跑
   `--readout-gather-impl index` vs `index_select`，判据：
   `segments(win)` 的 readout ms/tok + 端到端 ms/tok + **`probe_PPL`**
   （不看 sliding PPL，信噪比 0.40）。
3. **若确认收益且 PPL 无回归** → 才把 `readout_gather_impl` 默认改成
   `index_select`，并同步 `phdnet/config.py` 与文档。
4. 路径 B（推理路径 parallel scan）/ 路径 C（分区激活）排在后面，
   且 C 必须先有 `probe_PPL` 基线。

### 5.1 P218：一次「无效 A/B」的复盘（务必记住）

2026-11-08 用户回传两段生产日志，看起来像 `index` vs `index_select`
的 A/B，但**不能用它判定收益**，三个原因：

1. **服务器代码版本落后**：`train.py` 的 choices 仍是
   `{index,take}`（旧于 `fe4b596`），用户加 `--readout-gather-impl
   index_select` 直接报 `invalid choice`。→ 那两轮**根本没有跑过**
   `index_select`。A/B 前必须 `git fetch` 并核对 `--help`。
2. **只有 cum 口径的 readout 指标**：旧日志 `readout ms/tok` 是
   `_ro_ms_accum / _ro_calls`（从 token 0 起的累计平均），含 JIT 冷编译。
   两段日志第一段 500 步 **58.54 ms/tok**（`PC_learn 15.70` /
   `FO_learn 13.52` / `M1_encode 6.55`），第二段 500 步 **21.68 ms/tok**
   —— 这是**冷/热 numba cache** 的差异，不是改动收益；而两段的
   `segments(win)` 几乎完全一致，`probe_PPL` 四个采样点**逐位相同**。
3. **启动日志不打印生效值** → 无法区分「改了没效果」与「CLI 存了值、
   生产路径没接线」（P134/P186 踩过两次）。

**P218 已修**（`866a2bd`）：
- 日志同时打印 `readout X ms/tok (cum) / Y (win)`，`reset_prof_window()`
  同步清零 win；
- 启动行 `[readout] gather_impl=<实际生效> (requested <CLI 值>) |
  gather_dtype=... | conn_k=...` 自证开关；
- `gate_readout_win_timing.py`（8/8）锁住契约；
- `tools/analyze_readout_ab.py` 校验两轮生效值确实不同、且各自等于 CLI 值，
  再用 `probe_PPL`（语义）+ `readout (win)`（性能）出对比表。

**正确的 A/B 顺序**：`git fetch` → 核对 `--help` → **先跑一轮预热**
（填 numba cache）→ A(`index`) → B(`index_select`) → 必要时再跑一次 A
交叉验证；判据用 `probe_PPL` 与 `readout (win)`，**不看 cum、不看
sliding PPL**。

### 5.2 P218 生产 A/B 结果（2026-11-08，浏览器里跑完）

环境：AtomGit Notebook / Ascend910B4 / `866a2bd`，**先跑 1000 token 预热**
填 numba cache，然后两轮各 5000 token、`--probe-every 1000
--probe-tokens 2048 --step-profiling`，同数据同 seed。
启动行自证：`A gather_impl=index (requested index)` /
`B gather_impl=index_select (requested index_select)`。

| token | probe A / B | 相同 | 端到端 A→B | readout(win) A→B |
|---:|---|---|---:|---:|
| 1,000 | 31110.8992 / 31110.8992 | ✅ | 36.56 → 20.77 | 7.897 → 2.454 (3.22×) |
| 2,000 | 9100.0132 / 9100.0132 | ✅ | 32.88 → 17.51 | 8.003 → 2.462 (3.25×) |
| 3,000 | 2949.1149 / 2949.1149 | ✅ | 31.84 → 16.65 | 7.956 → 2.360 (3.37×) |
| 4,000 | 1061.1045 / 1061.1045 | ✅ | 31.52 → 16.17 | 7.992 → 2.350 (3.40×) |
| 5,000 | 601.8898 / 601.8898 | ✅ | 31.40 → 15.88 | 8.123 → 2.404 (3.38×) |

**均值：端到端 1.81×、`readout (win)` 3.14×；probe_PPL 9 个采样点逐位相同。**

注意 `readout(win)` 在整千/半千采样点呈 ~8 / ~3.8 两组（滑窗长度
`[500 steps]` vs `[2291 steps]` 交替，probe 会改变窗口），但 **A/B 比值
稳定在 3.1–3.4×**，不是噪声。

**为什么算子级 52.80× → 生产只有 3.14×**：算子 benchmark 只测
`(ht[Wi]*W).sum(1)`；生产 `readout ms/tok` 还含 `W` 的 fp8 反量化、
`learn_softmax` 的 logsumexp/softmax/rank-1 更新、CPU 提交与等待。
**端到端 1.81× 才是最终收益**（readout 只占一部分）。

**落地决策（P218 最终）**：默认改为 **`auto` 按平台择优**
（与 `--m2-kernel auto` 同一治理方式）：
- NPU / 昇腾 → `index_select`（有上面这份生产 A/B 证据）；
- CPU / CUDA / ROCm → `index`（本机 x86 实测 `index_select` **慢 11%**，
  其它后端无实测证据 → 不外推）；
- 显式 `index`/`take`/`index_select` 仍可覆盖，保留 A/B 与排障能力。

`resolve_gather_impl()` 的 9 个 case 由 `gate_gather_impl.py`（27/27）锁住。

---

## 6. 一句话回答用户的问题

> **「能不能简化单个 token 的计算量、变成并行？」**

- **简化单 token 计算量：能，而且已经实测出来** —— 当前 80.7% 花在
  对整个词表做 Vector `Index`；把它换成 `torch.index_select` 在
  Ascend910B4 上快 **52.80×（fp16）/ 29.63×（fp32）且逐位相同**（§4）。
  这**不碰铁律③**，已做成可选开关，待生产 A/B 后即可改默认。
- **把训练变成并行：不能在不改学习规则的前提下做到**。我们的串行来自
  「每个 token 都更新权重」，不是来自 Attention；Yan 的并行训练建立在
  「整条序列共用同一份权重」上。想并行就得改成批内延迟更新（路径 C），
  那是语义变更，要用 `probe_PPL` 判，而不是偷偷改代码。
- **推理路径可以并行**（无权重更新 + 线性递推 + associative scan），
  风险最低，但收益也最小。

---

## 7. 参考来源

- RockAI / 岩芯数智「Yan 架构」公开资料（Yan 1.0 非 Attention 通用大模型，
  2024-01；Yan 1.3 3B 多模态，MCSD + 类脑激活机制）。
- **Hua Yang, Duohai Li, Shiman Li**, *MCSD: An Efficient Language Model with
  Diverse Fusion*, arXiv:2406.12230, 2024.
  <https://arxiv.org/html/2406.12230v1>
- 中国证券报 / 上海市科协 / 澎湃新闻关于 Yan 架构的报道（O(n·d) 复杂度、
  O(1) 推理空间、7× 训练效率 / 5× 推理吞吐等厂商口径，**未经本项目复现**）。

⚠ 厂商宣称的 7×/5×/3× 是**同参数量对 LLaMA 2 的对照**，不是本项目的
可达收益；本文只借用其**结构思路**（预定义衰减递推、分区激活、
训练矩阵形式 / 推理递推形式对偶），不引用其性能数字作承诺。
