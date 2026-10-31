# PHD-Net 技术债清理与性能默认优化（P212）

> **适用范围**：P211 之后的代码修复、死代码清理、以及「默认采用性能更优实现」的改造。
> **数据截止**：2026-10-31（P212）。
> **相关文档**：[CPU 性能优化实录 P211](PHD-Net_CPU性能优化实录_P211.md)（上一轮的提速实录）、
> [前向only 训练路径](PHD-Net_前向only训练路径.md)（机制与 A/B）、
> [性能评估与迭代方案](PHD-Net_性能评估与迭代方案.md)（生产档性能唯一出处）、
> [现状速查](PHD-Net_现状速查.md)（当前事实基线）。
> 与代码冲突时**一律以代码为准**。

---

## 0. 一句话

P212 做了四件事：① 修 **2 个真 bug**（一个是 P211 自己引入的语义回归）；② 清理
**7 处死代码 / 孤立实现**（含 1 个 244 行的孤立模块）；③ **兑现一条从未落地的
性能优化**（fp8 转换查表，**3.06×**）；④ 让 **M2 推理核按平台自动择优**
（x86 默认从慢路径切到快路径）。

**最重要的一条**：P212 的审计过程暴露了一个 P211 掩盖的**真实设计缺陷**——
forward-only 在 RL 场景下**与策略梯度不对齐**，导致 RL 学不动。修它比"让门禁变绿"
重要得多，详见 §2.3。

---

## 1. 修的两个真 bug

### 1.1 `--replay-every` 被静默丢弃（P210 起就有）

**位置**：`train/train.py`（P212 前的接线处）

```python
if args.replay_every > 0 and args.replay_every != 4:   # ← 那个 4 是过时的库默认值
```

`cfg.replay_every` 的库默认在 P204 已从 **4 改成 16**，但这里仍拿过时的 `4` 当哨兵。
后果：用户传 `--replay-every 4`（help 文案明确说「⚠ 回 `--replay-every 4` 可复原
旧巩固节奏」）时，`4 > 0` 为真而 `4 != 4` 为假 → **条件不成立 → 该值被静默丢弃**，
仍用库默认 16。**与 help 文案直接矛盾。**

**修法**：`argparse` 的 `default=0` 已经表达了「0 = 沿库默认」的语义，故只需判 `> 0`
—— **不必拿任何库默认值当哨兵**（那种写法在库默认变更时必然再次失效）。

**实测（修后）**：

| 参数 | 修前 M6_replay 段 | 修后 M6_replay 段 |
|---|---|---|
| `--replay-every 4` | 0.33 ms/tok（**参数无效**） | **1.46 ms/tok** |
| `--replay-every 16` | 0.33 ms/tok（**参数无效**） | **0.41 ms/tok** |

即：修前两个参数**完全等价**（参数根本没生效）；修后按预期呈现 **3.6×** 差异。

### 1.2 `homeostasis` 参数被融合核忽略（**P211 引入的语义回归**）

**位置**：`phdnet/sparse_pc.py::_pc_fo_fused` + `learn_forward_target`

P211 把 forward-only 的 6 次原子核调用融合成 1 个 numba 核时，核内的
「⑥ 行范数稳态缩放」写成了**无条件执行** —— `learn_forward_target(..., homeostasis=False)`
与 `homeostasis=True` 的结果**完全相同**（参数被静默忽略）。

而 **P210 的原实现是尊重该标志的**（`if homeostasis: self._homeostatic_scale()`），
即 P211 相对 P210 引入了一处语义回归。

- **生产无影响**：`train/config_1b.py` 设 `homeostasis=True`，恒为 True。
- **库直用有影响**：`PHDNetConfig()` 默认 `homeostasis=False` → 任何直用库配置的
  路径（tests / tools）会拿到**静默不同**的数值。

**修法**：给核加 `homeostasis` 形参并条件化 ⑥ 段；门禁补 `homeostasis=False`
的对拍（B1b/B1c）。

**验证**：

```
forward_only=True (homeostasis=False) 摘要: 742523569ee10b70
forward_only=True (homeostasis=True)  摘要: d362ecfdece65c8e
→ 两者不同 ✓（标志现在真正生效）
```

---

## 2. 审计暴露的真实设计缺陷（比 bug 更重要）

### 2.1 现象

修完 1.2 后，`tests/gates/gate_rl_loop.py` 的 **D 组**（学习有效性：玩具奖励应上升）
从 PASS 变成 **FAIL**：

```
改动前（P211）：前三 0.375 → 后三 0.542   PASS
改动后（P212）：前三 0.375 → 后三 0.308   FAIL
```

### 2.2 定位过程（每一步都有实测支撑）

| 步骤 | 实验 | 结论 |
|---|---|---|
| 1 | `git stash` 对拍 | 确认是本轮引入，非既有失败 |
| 2 | 强制融合核无条件缩放（P211 行为） | RL PASS → 差异确在「缩放」 |
| 3 | 查 `cfg.homeostasis` 默认值 | 库默认 **False**；RL 门禁用 `PHDNetConfig()` → 走 `homeostasis=False` |
| 4 | 测残差幅度演化（‖τ − dn1·r2‖） | 有/无缩放**几乎相同**（5.29 vs 5.29）→ **不是饱和** |
| 5 | 试「残差归一化」方案 C | 0.308 → 0.367，**仍 FAIL** → 归一化不是关键 |
| 6 | 关掉 forward-only 重测 RL | **PASS**（0.375 → 0.425）→ 问题出在 **forward-only 本身** |
| 7 | 查 `phdnet/rl.py` 的学习率接线 | **找到根因**（见 §2.3） |

### 2.3 根因：策略梯度只调制读出，forward-only 以固定强度注入

`phdnet/rl.py` 把 REINFORCE 的策略梯度灌进**读出感知器**，用

```python
net.step(x, target=tgt, learn=True, learn_scale=self.lr * float(a))
```

即 `learn_scale = lr · advantage` 调制**读出**学习率（advantage>0 强化、<0 弱化）。

而 forward-only 此前只用 `pc_scale`（**固定强度**）→ 在 RL 里它以
**与奖励无关**的强度持续往主干注入「标签→随机码」目标：
**advantage≈0 的样本也在注入噪声**，advantage<0 的样本也照样注入（方向不对）。

→ 主干被与奖励无关的信号持续扰动，真正的策略梯度（只作用于读出）不足以对抗
→ **RL 策略学不动**。

### 2.4 修法：forward-only 与读出**同口径**接受 `learn_scale`

```python
_eta_fo = forward_only_eta * pc_scale * float(learn_scale)
if _eta_fo != 0.0:      # ⚠ 用 != 0.0 而非 > 0.0（见下）
```

- **普通训练 `learn_scale=1.0` → 逐位不变**；
- **RL 的 advantage 直接调制层目标注入强度**（含负号 → 反向更新）；
- ⚠ 判据用 `!= 0.0` 而非 `> 0.0`：RL 的 `learn_scale` **可以是负的**
  （advantage<0 的样本应当**反向**更新）；用 `> 0.0` 会把负 advantage 的样本
  **整段跳过** → 丢掉一半的策略梯度信号。

**结果**：RL 门禁 D 组 **0.375 → 0.417（PASS）**。

**门禁新增断言**（`gate_cpu_opt_p211` B1d/B1e）：

- B1d：`learn_scale=1.0` 与 `0` 的训练轨迹**不同**（确实受它调制）；
- B1e：`learn_scale=0` 的 forward-only 轨迹 == **关闭 forward-only** 的轨迹
  （**逐位相同**）→ 证明调制是干净的、且 `learn_scale=0` 是「不注入」的精确表达。

### 2.5 附带发现：RL 判据本身脆弱

在 P212 状态下把 RL 门禁的采样 seed 偏移 1/2/3 → **PASS / FAIL / FAIL**
（前三 0.325→0.342 / 0.375→0.292 / 0.325→0.275）。

即该判据在**微小数值扰动**下就翻转 —— 玩具任务的奖励只有 0/0.5 两档、
轮次只有 12，噪声大。**本轮没有去改这个判据**（改 seed 让它过 = 掩盖问题），
而是修掉了它暴露的**真实缺陷**。若后续要加固，建议：多 seed 取中位数、
或把奖励改成连续值。

---

## 3. 清理的死代码 / 冗余实现

| # | 对象 | 位置 | 处置 | 依据 |
|---|---|---|---|---|
| 1 | `forward_target_delta` | `phdnet/forward_only.py` | **删除** | 稠密 `np.outer` 实现，生产零调用，已被稀疏融合核取代；且**与铁律②「结构性稀疏」冲突**。门禁 B4 的三条断言改写到**真实生产路径**上（语义不变，测的代码变了） |
| 2 | `learn_forward_target` 的 `return` 后 17 行 | `phdnet/sparse_pc.py` | **删除** | 不可达死代码（自称「保留作 A/B 对照」）。门禁**自带**参考实现，不依赖它 |
| 3 | `self._fwd_cache` 死字段 + 误导注释 | `phdnet/model.py` | **删除** | 从未被读写；真缓存在 `forward_only.py::_cache`。那段注释描述了一个**不存在于本类**的实现 |
| 4 | `fp8_capability.py` 整模块（244 行） | `phdnet/backends/` | **删除** | **零 import**；其能力探测逻辑已被 `accel_readout.py` 的 `_probe` + 降级链取代。但它承载的 P147「转化效率」需求**未兑现** → 见 §4 |
| 5 | `_encode_hash_uniq` | `phdnet/ltm_kernel.py` | **删除** | P68 实测 0.84× 负收益已回滚到 Python `dict.fromkeys` 的**残留死核**（还白占一次 JIT 编译） |
| 6 | `_resolve_device` | `phdnet/backends/torch_backend.py` | **删除** | 零引用；其 docstring 声称的「A1 修复：回退时至少告警」**从未接线**（真正的设备解析是 `torch_lm.resolve_device`） |
| 7 | `local_name_of` / `DEFAULT_ITER_DTYPE` | `phdnet/ms_stream.py` / `phdnet/precision_policy.py` | **删除** | 零引用的孤立符号 |
| 8 | `readout_master_weight` / `readout_shadow_refresh` 重复定义 | `phdnet/config.py` | **删除重复项** | dataclass 里后定义覆盖先定义（值相同故无数值影响），但会让「改哪一处生效」含糊 |

**保留不动**（有实测依据的正确决策，勿误伤）：

| 实现 | 位置 | 依据 |
|---|---|---|
| `learn_predictive` 多核调用 | `sparse_pc.py` | P59 实测融合版 0.91–1.36× **负收益** |
| `SparseLTM.encode` 用 `dict.fromkeys` | `bigltm.py` | P95b/P98 逐项实测分解；`np.unique` 94.3 µs 更慢 |
| `readout` 前向走 BLAS 而非 numba | `readout.py` | 实测慢 0.68× 且非逐位等价 |
| fp8 转换默认走 torch | `fp8_int8_convert.py` | P154 实测 torch 快 5.6× |

---

## 4. 兑现一条从未落地的性能优化：fp8 转换查表

**问题**：`phdnet/backends/fp8_int8_convert.py` 里定义了一张 16 项的
`_E4M3_POW2` 查表，注释宣称「**P154 性能关键**」并给出实测对比
（查表 105.4 ms vs powf 120.6 ms）—— 但**这张表从未被索引过**。
核内一直是 `2.0 ** (exp - _E4M3_BIAS)`，被 numba 编译成 `powf` 调用
（注释自己解释过原因：「逐元素 powf 无法被向量化」）。

**即：注释宣称的优化从未落地，实际跑的是慢的那一档。**

**修法**：核内改用 `_E4M3_POW2[exp]`。**数值逐位相同**（表里全是 2 的幂，
精确可表示；`np.array_equal(2.0**(exp-7), _E4M3_POW2[exp])` = True，偏差 0.0）。

**实测**（本机 x86 8 核，1b 档规模 6,651,136 个 fp8 位模式）：

| | nogil 路径 |
|---|---|
| 改前（powf） | **154.6 ms** |
| 改后（查表） | **50.5 ms** |
| **加速** | **3.06×** |

（torch 路径 24.8 ms 不变 —— 它本来就不用这个核。）

### 4.1 顺带补上「转化效率」报告（P147 指令兑现）

P147 原文：「记得提供转化效率，转化放在 cpu 上」。该需求此前**只存在于零 import
的孤立模块** `fp8_capability.py`（已删）→ 从未兑现。

现在改由**实际在用的** `fp8_to_int8_codes(report=...)` 兑现，报告
`ms` / `MiB_in` / `MiB_out` / `MiB_per_s` / `ratio_vs_fp32_copy` / `conv`，
并在 `accel_readout.py` 的调用点打印告警。

### 4.2 补上门禁（保护链原本是断的）

`fp8_int8_convert.py` 的 docstring 声称「**逐位对拍由
`tests/gates/verify_fp8_int8_convert.py` 把关**」—— 但**该文件从来就不存在**
（P154/P151/P123 都反复引用它）。即这条保护链是**断的**，而代码里恰好藏着 §4 的缺陷。

新增 `tests/gates/gate_fp8_int8_convert.py`（**17 项**）覆盖：
查表与 powf 的逐位等价 / 核内确实用查表（源码级）/ numpy 参考路径一致性 /
边缘位模式（subnormal、符号位）/ NaN 槽位处理 / 报告字段完整性 / 向后兼容。

---

## 5. 默认采用性能更优的实现：M2 推理核按平台择优

**问题**：`--m2-kernel` 默认 `plain`（非融合路径，5 次独立 `_csr_matvec`）。
但代码注释里明确记载：

- `phdnet/sparse_pc.py`：「**x86 上融合核快 1.15-2.16×**」
- `train/train.py`：「⚠ **x86 结论不构成昇腾证据**：本机上 fused 比 plain 快」

即**在 x86 开发机上，默认走的是慢 1.15–2.16× 的路径**。

**本机实测**（2026-10-31，`M2_infer` 单次调用，200 次取最优）：

| 规模 | plain | fused | serial | fused/plain |
|---|---|---|---|---|
| n=256, k=32 | 0.2236 ms | **0.1089 ms** | 0.1213 ms | **2.05×** |
| n=1024, k=128 | 0.6961 ms | **0.5193 ms** | 1.7315 ms | **1.34×** |

`serial` 在两档都**最差**（x86 上比 plain 慢 2.5×）→ 永不作为默认。

**改法**：新增 `phdnet/sparse_pc.py::default_m2_kernel()`，**按平台**择优：

| 平台 | 选择 | 依据 |
|---|---|---|
| 昇腾 aarch64（191 核 + NPU） | `plain` | P113 服务器实测 M2_infer 6.92 → 1.28 ms/tok（**5.41×**），端到端 1.46×；P76/P99 实测融合核在昇腾慢 3–4×（**prange 屏障成本主导**） |
| x86（本机 8 核） | `fused` | 上表实测 1.34–2.05× |

⚠ **为什么可以按平台分**（与 P165 的教训不冲突）：P165 说「BLAS 快慢不是架构的
必然属性」，那是**数值库**的性质；而这里的差异来自 **prange 屏障的线程数**
（昇腾 191 核时 fork/join 成本被放大到远超计算本身，x86 8 核则相反）——
核数越多屏障越贵，是**可预期的系统性规律**。

⚠ 仍然是**启发式**：真正的判据是「核数多 + 单次计算量小」。故：
① 用户可用 `--m2-kernel` 显式覆盖（优先级最高）；
② `auto` 只在**未显式指定**时生效；
③ 启动时打印实际选择（本机 `[p212] M2 推理核：auto → fused`）。

### 5.1 ⚠ 数值影响（必须如实记录）

三态**不是逐位相同**（这一点项目文档此前记为「逐位完全相同」，实际那是
**跨 run 的日志观测**，受控对拍用的是 `_RTOL = 1e-6` 容差）：

| 对比 | max\|Δ\| |
|---|---|
| fused vs plain | **9.7e-8 ~ 1.5e-7**（fp32 舍入级） |
| serial vs plain | **2.2e-16 ~ 4.4e-16**（fp64/fp32 累加顺序差） |

端到端影响（smoke 档 / cc_traces / 3000 token）：`final_ppl` 13017.891（plain）
→ **13017.892**（fused），差 **1e-6 相对量**。

→ 这是**已知且可接受的容差级差异**（门禁 `gate_m2_infer_kernels` 以 `_RTOL=1e-6`
对拍三态），但**不是逐位不变** —— 与 P211 那批「逐位不变」的优化性质不同，
故在此明确区分。

---

## 6. 门禁

| 门禁 | 项数 | 覆盖 |
|---|---|---|
| `gate_cpu_opt_p211` | **19** | int32 索引 / FO 融合核逐位等价（含 `homeostasis=False` 对拍 B1b）/ **`learn_scale` 调制契约（B1d/B1e）** / LTM 内联逐位等价 |
| `gate_forward_only` | **30** | 固定随机投影 / 层目标学习 / 「eta=0 ≡ 旧路径」逐位判据；**B4 改写到真实生产路径**（删掉了稠密 `forward_target_delta` 依赖） |
| `gate_fp8_int8_convert` | **17**（新增） | fp8→int8 查表等价性 / 转换正确性 / 边缘位模式 / NaN 处理 / 转化效率报告 |

**全量**：52 个门禁，**49 通过**，3 个失败
（`gate_cli_lang_semantics` / `gate_encoder_gemv` / `gate_ltm_reverse_index`
—— 经 `git stash` 对拍确认为**改动前既有**的环境路径问题，与本轮无关）。

`tests/run_tests.py fast` = **9/9**。

---

## 7. 变更清单

| 文件 | 改动 |
|---|---|
| `train/train.py` | `--replay-every` 接线修复；`--m2-kernel` 默认 `auto` + 按平台择优 |
| `phdnet/sparse_pc.py` | `_pc_fo_fused` 加 `homeostasis` 形参并条件化缩放；删 `return` 后死代码；新增 `default_m2_kernel()` |
| `phdnet/model.py` | forward-only 强度接受 `learn_scale` 调制（RL 同口径）；删 `_fwd_cache` 死字段 |
| `phdnet/forward_only.py` | 删稠密 `forward_target_delta` |
| `phdnet/config.py` | 删重复字段定义；修正 fp8 能力探测的**错误注释** |
| `phdnet/backends/fp8_int8_convert.py` | 核内接线 `_E4M3_POW2` 查表（3.06×）；新增 `report=` 转化效率 |
| `phdnet/backends/accel_readout.py` | 调用点接上转化效率报告；修正注释 |
| `phdnet/backends/torch_backend.py` | 删孤立 `_resolve_device` |
| `phdnet/backends/fp8_capability.py` | **整个模块删除**（244 行） |
| `phdnet/ltm_kernel.py` | 删死核 `_encode_hash_uniq` |
| `phdnet/ms_stream.py` / `phdnet/precision_policy.py` | 删孤立符号 |
| `tests/gates/gate_cpu_opt_p211.py` | 新增 learn_scale 契约断言（B1d/B1e） |
| `tests/gates/gate_forward_only.py` | B4 断言改写到真实生产路径 |
| `tests/gates/gate_fp8_int8_convert.py` | **新增**（17 项） |
