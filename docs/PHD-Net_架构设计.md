# PHD-Net 架构设计

> **适用范围**：本文是 PHD-Net 的**最权威设计文档**——四条铁律、六机制（M1–M6）与脑同构对应、
> 数据流与逐 token 语义、稀疏性设计、容量账（四档预设对比）、语料与数据入口、评测口径。
> 文中每个关键数字都标注了源码位置（`文件:行号` 或符号名）；核对不到写「未验证」，不做推测。
>
> **本文不写**：
> - **性能数字**（ms/tok、加速倍数、访存带宽）——唯一出处是《PHD-Net_性能评估与迭代方案.md》，
>   本文只引用不复制。
> - 优化历史与被否决方案——见 `BUGS.md`。
> - 并行/加速吃不满多核的论证——见《PHD-Net_并行与加速架构分析.md》。
> - 硬件后端矩阵与昇腾踩坑——见《PHD-Net_硬件后端适配报告.md》。
> - 与 Transformer / 大脑的对比结论——见《PHD-Net_竞争力与脑同构性评估.md》。
>
> **数据截止：2026-10-08（P192）。** 与代码冲突时**一律以代码为准**。全项目事实基线与禁写项见
> `docs/文档写作规范.md`（该规范的基线截止日为 2026-09-30，本文以源码为准并在其之上更新）。
>
> **相关文档**：
> - `README.md` —— 项目一句话、怎么跑、任务导向文档地图
> - `docs/文档写作规范.md` —— 写作规范与全项目事实基线（写其它文档前必读）
> - `docs/PHD-Net_性能评估与迭代方案.md` —— **所有性能数字**
> - `docs/PHD-Net_并行与加速架构分析.md` —— 为什么吃不满多核/NPU（论证）
> - `docs/PHD-Net_扩展指南.md` —— 怎么加东西、检查单
> - `docs/PHD-Net_硬件后端适配报告.md` —— 后端矩阵与迁移路径
> - `docs/PHD-Net_竞争力与脑同构性评估.md` —— 与 Transformer / 大脑对比的结论
> - `BUGS.md` —— 缺陷台账

---

## 0. 设计出发点

Transformer 把「关联」建模为**全局成对相似度**（QK^T，O(T²·d)），把「记忆」建模为**无学习的
KV 缓存**，把「学习」寄托于**非局部反向传播**。PHD-Net 换一套组织原则：

| 关注点 | Transformer | PHD-Net |
|---|---|---|
| 关联 | 全局成对相似度 | **局部稀疏连接**（每神经元 k 条存在入边，CSR 承载） |
| 记忆 | 无学习的 KV 缓存 | **双记忆**（工作记忆漏衰减 + 事件驱动大空间表） |
| 学习 | 反向传播 + 梯度 | **局部规则**（Oja / STDP / 感知器），误差驱动 |
| 时间 | 位置编码显式注入 | **状态连续演化**（无位置编码，见铁律①③） |
| 稀疏性 | 稠密矩阵 + 零权重 | **结构性稀疏**（不存在的突触不存储、不计算） |

「稀疏」在本项目里是**两层不同的东西**，全文严格区分：

1. **稀疏表示**（activations sparse）：M1 的 k-WTA 只激活 k 个单元、M3 的发放率
   `PHDNet._rate` 只保留 top-`n//16`。这对应皮层的**稀疏发放**。
2. **稀疏连接**（synapses sparse）：M1/M2/M3/M6 只存储并遍历**存在**的突触。
   这对应皮层的 **dense representation + sparse connectivity**
   （`phdnet/sparse_pc.py:16-17`）：表示稠密、连接稀疏，信息由大量神经元的稀疏活动模式承载。

---

## 1. 四条铁律

### 铁律① 禁止自注意力 / 位置编码 / 堆叠层

源码中**不存在**任何 Q/K/V 投影、位置编码模块或跨层堆叠结构。M2 是一棵**两层层级**
（`n_sdr → n_mid → n_top`，`phdnet/model.py:41-46`），M3 在 `n_top` 上做单层侧向关联
（`phdnet/model.py:58-69`）。时序信息**不来自位置编码，而来自状态本身**（见铁律③）。

唯一的"顺序敏感"设计是 M3 STDP 的**突触迹**（`t_pre`/`t_post`，`phdnet/plasticity.py:45-48`）
—— 它是有方向的因果结构（pre 在前、post 在后），属于局部可塑性机制，不是位置编码。

### 铁律② 每个机制必须有神经认知对应

本文 §2 的每个机制都给出脑区/文献对应。源码注释里的对应声明即权威出处
（如 `phdnet/inits.py:1-18` 引 Song et al. 2005 的皮层突触重尾分布、
`phdnet/sparse_alloc.py:3-13` 引突触巩固/修剪的幂律倾斜）。

### 铁律③ 逐 token 语义（**禁止批处理**）

`PHDNet.step()` 是**单时间步**接口（`phdnet/model.py:204`）。状态（WM 槽位强度、STDP 迹、
LTM 突触、调制器 Welford 统计量、`step_count`、读出 W）**全部跨 token 连续演化**，
且读出 W **每步原地更新**。因此：

- 训练侧状态**永不重置**：`train/train.py:719` 打印
  `"[context] state never reset (WM/STDP/LTM carried continuously across samples/shards/epochs)"`。
- 跨 epoch 也不重置：`train/train.py:805` 打印
  `"continuing stream across epochs: state continuous (net not reset), first step's prev link severed"`
  —— 只有**第一步的 prev 链接被切断**（`p2 = None`，`train/train.py:801`），其余状态全保留。
- 主循环逐 token 调 `net.step(x, target=tgt, learn=trainable, target_idx=_i0)`
  （`train/train.py:838`），语料流是 `StreamingTokenizer`（`train/corpus_stream.py:243`，
  注释直书"逐 token 训练，语料任意长"）。
- 评估路径用 `readonly=True` **冻结所有持久状态**（`phdnet/model.py:211-213`），
  保证重复评估结果一致（2026-09-28 修复：此前 readonly 未真正冻结 PC/STDP/编码器学习）。

**推论**：批处理（batch 内并行多 token、梯度累积式大批量更新）会破坏"状态连续演化"这一
前提，**不允许**。唯一的例外是读出更新的 `minibatch_size`（`phdnet/config.py:277`），
它只累积**梯度**、不改变 token 推进的逐步语义，且**默认 1 = 关闭**。

### 铁律④ 行为变更用 config 开关承载，默认关闭

`phdnet/config.py` 里所有可选行为都带默认值；改变既有行为的开关一律 `= False`
（`auto_development` / `task_modulation` / `multi_modulation` / `critical_period` /
`staged_sleep` / `homeostasis` / `lognormal_init` / `readout_recurrence` 等）。

**三个明确的例外（默认开启）**：

| 项 | 值 | 位置 | 理由 |
|---|---|---|---|
| `sparse_conn` | `True` | `phdnet/config.py:187` | 稠密 PC 栈已于 2026-09-28 **删除**（`phdnet/pc.py` 移除），主干唯一实现是 `SparsePCStack`；传 `False` 在配置期 **fail-fast**（`phdnet/config.py:342-345`），不做静默稀疏化 |
| `k_sparse` | `16` | `phdnet/config.py:11` | 2026-09-24 起默认（原 32 = 12.5%）。实测依据（4,000 字符口径）：k=16（6.25%）PPL **−1.43%** 优于 k=32；k=8（3.125%）**+12.5%**（容量不足，有害）。**评测脚本显式传 `k_sparse=32`**（`tests/eval_common.py:16`），故该默认变更不影响既有评测口径 |
| `csr_online` | CLI `--csr-online` `default=True` | `train/train.py:281-282` | 在线可写 CSR 已是生产默认（dict 版每条目 ~100+ B → CSR 版 ~16 B）。库字段 `csr_online` 本身默认 `False`（`phdnet/config.py:262`），由 CLI 显式打开 |

---

## 2. 六机制 M1–M6

### 2.0 编排顺序（`PHDNet.step`）

`phdnet/model.py:204-459` 的实际执行顺序（**注意与模块编号不完全一致**）：

| 序 | 动作 | 源码行 | 机制 |
|---|---|---|---|
| 1 | `s0, _ = self.encoder.encode(x)` | `model.py:228` | M1 |
| 2 | `cache = self.pc.infer(s0, n_infer_steps)` | `model.py:231` | M2 |
| — | `rate = self._rate(cache["r2"])` | `model.py:233` | 发放率（供 M3/M4b/M5） |
| — | `fused = self._fused_feature(rate)` | `model.py:234` | T3.2，默认 `None` 零参与 |
| 3 | `pred = self.stdp.predict(self._prev_rate)` | `model.py:237` | M3 时序预测（诊断 + 读出特征） |
| — | `cos` / `seq_err` | `model.py:248-249` | 时序一致性诊断 |
| — | `surprise = ‖e0‖/√len` | `model.py:251` | 预测误差范数 |
| 4 | `gate, mode = self.modulator.observe(surprise)` | `model.py:254` | M5 |
| 5 | `wm.decay()` → `wm.write(r2, mem_gate, gate_thresh)` | `model.py:263-266` | M4a |
| 6 | `ltm.imprint(...)` 或 `ltm.recall(...)` | `model.py:289-301` | M4b |
| 7 | `h = self._build_h(r2, pred_feat, fused)` → `readout(h)` / `learn_softmax` | `model.py:308-341` | M6 |
| 8 | PC/STDP/编码器局部学习（×调制门） | `model.py:378-449` | M2/M3/M1 学习 |

**读出特征拼装**（`model.py:173-181`）：`h = [r2 ; WM]`，`pred_in_readout=True` 时
三拼 `[r2 ; WM ; STDP 时序预测]`，T3.2 开启时再拼 `retrieval_topk` 条召回线索。
1B 档 `n_h = 3 × 1024 = 3072`（`capacity_report` 的 `n_h` 口径，`train/config_1b.py:101`）。

---

### 2.1 M1 稀疏分布式编码（k-WTA）

**脑同构**：初级感觉皮层的稀疏放电与抑制性侧抑制（`phdnet/sparse_encoder.py:1`）。
皮层用**大量神经元各自的少量活动**承载信息，而不是靠大激活值。

**源码实现**（`SparseEncoder`，`phdnet/sparse_encoder.py:76-153`）：

```
u = W_enc·x + b                # 全层电位（感受野投影），sparse_encoder.py:126
idx = argpartition(-u, k-1)[:k] # k-WTA 竞争 = 侧抑制的抽象，sparse_encoder.py:126
s[idx] = norm(u[idx]) + 0.1     # 归一到 [0.1, 1.1]，sparse_encoder.py:129
```

- **W 初始化**：固定随机投影 `normal(0, 1/√n_input)`（`sparse_encoder.py:85`），
  **默认冻结**，"生产中可由 Oja 规则慢调，这里保持冻结以聚焦高层学习"。
- **`k_sparse` 默认 16 / `n_sdr` 256 → 激活率 6.25%**（`config.py:9-11`）。
  1B 档按 `k_sparse = max(8, w//8)` 缩放到 128/1024（`train/config_1b.py:76`）。
- **⚠ 输入维度契约**：`step(x)` 的 `x` 维度是 **`cfg.n_input`**（字符/词 SDR 的拼接输入），
  **不是** `cfg.k_sparse`；`target` 的维度是 **`net.n_out`**（读出输出维度）。
  两者在 `model.py:220-225` 与 `phdnet/readout.py:1042-1058` 都有 fail-fast 校验。
  词级 LM 侧 `n_input = 2 × n_sdr`（`phdnet/word_lm.py:41`，当前词 SDR ‖ 前词 SDR）。
- **`__post_init__` 校验**：`k_sparse > n_sdr` 直接 `ValueError`
  （`config.py:336-339`，2026-09-28 修复——原先构造成功、首个 step 才崩）。
- **存储 dtype**：迭代量**恒 fp32**（P61 口径），权重按 `encoder_dtype` 存储。
  库默认 `fp64`（`config.py:298`）；**bf16/fp16 存储 + fp32 迭代在 numpy/BLAS 路径必然
  每次付上采样转换**（W 读 8.4 MB + 写 16.8 MB + 再读 16.8 MB），比 fp32 直接 GEMV 更慢
  （`sparse_encoder.py:10-14`）。
- **平台自适应 GEMV**（`sparse_encoder.py:110-123`）：aarch64（昇腾）上 numpy 对该形状
  GEMV 病态慢 → 走自写 numba 行核 `_gemv_rows`；x86 上 BLAS 更快 → 走 BLAS。
  ⚠ aarch64 numba **不支持 float16 数组**（P107b 曾踩中，已回滚）。
- **可选可学习词典**（`learnable_encoder`，默认 `False`，`config.py:158`）：
  Foldiak 局部规则 `ΔW_i = η·s_i·(x − s_i·W_i)` + 行范数拉回（`sparse_encoder.py:132-153`），
  成本 `O(k·n_input)`。**实测有害 +6.1%**（`tools/audit_brain_parity.py:143`）→ 保持关闭。

---

### 2.2 M2 预测编码主干（结构性稀疏 CSR）

**脑同构**：皮层的**预测编码层级** + `dense representation + sparse connectivity`
组织原则。对应文献写在 `phdnet/sparse_pc.py:9-17` 的对照表里：
单个锥体神经元 ~10³–10⁴ 突触 / 潜在靶点 ~10¹¹ → 连接率 ~10⁻⁵（长程）～10% 量级（局部）。

**结构**（`SparsePCStack`，`phdnet/sparse_pc.py:368-609`）：四个 CSR 权重

| 权重 | 方向 | 形状（n0→n1→n2 = n_sdr→n_mid→n_top） | 初值 |
|---|---|---|---|
| `up0` | 上行识别 | n1 × n0，每行 k0 条 | `normal(0, 1/√k0)` |
| `up1` | 上行识别 | n2 × n1，每行 k1 条 | `normal(0, 1/√k1)` |
| `dn0` | 下行生成 | n0 × n1 | `up0` 转置 × **0.5** |
| `dn1` | 下行生成 | n1 × n2 | `up1` 转置 × **0.5** |

- **连接率**：`conn_k=0` 时 `k = n_in//8` ≈ **12.5%**（`sparse_pc.py:386-388`）。
  1B 档 `conn_k=128` / width 1024 = **12.5%**（`train/config_1b.py:80`）。
- **⚠ fan-in 归一化是硬要求**（`sparse_pc.py:390-394`）：稀疏栈每神经元只有 k 条入边，
  权重尺度必须取 `1/√k` 而非 `1/√n_in`，否则前向幅值偏低 `√(k/n_in)` 倍
  （k=32/n=256 时 ~0.35×），经 tanh 后表征趋 0。**该量纲错误在 4,000 字符短训练口径下
  被掩盖（PPL −0.30%），在全语料口径下暴露为 +37.6% 劣化。**
- **推理**（`sparse_pc.py:464-497`）：`r1 = tanh(up0@s0)`；`r2 = tanh(up1@r1)`；
  迭代 `n_infer_steps`（默认 1）步自由能下降：
  `e1 = r1 − dn1@r2`；`d2 = clip(up1@e1, ±0.5)`；`r2 = tanh(r2 + 0.15·d2)`；
  `e0 = s0 − dn0@r1`；`d1 = clip(up0@e0, ±0.5)`；`r1 = tanh(r1 + 0.15·d1)`。
  **步长 0.5 / 0.15 / 迭代 1 是硬编码的字面量**，不是配置项。
- **三态核选择**（`config.py:198`，`pc_fused_kernel`）：`True`/"fused"（并行融合核）、
  `"serial"`（单核融合核，去掉 prange 屏障）、`False`（原始 5 次核调用）。
  ⚠ **不能 `bool()`**（`sparse_pc.py:381-384`）——那会把 `"serial"` 变成 `True`，
  让单核核永不可达。**生产 CLI 默认 `--m2-kernel plain`（P113，`train/train.py:250`）**：
  P76 只否掉了「prange **融合**」（昇腾上10 个屏障区主导开销），**没否掉 `_csr_matvec`
  自身的行级 prange**——而 plain 走的正是后者。服务器实测（1b 档，昇腾 191 核 + NPU）：
  M2_infer **6.92 → 1.28 ms/tok（5.41×）**，端到端 17.63 → 12.09 ms/tok（1.46×），
  四个采样点 sliding PPL 与 serial 运行**逐位相同**（纯调度变化、零语义变化）。
  对拍见 `tests/verifiers/verify_m2_kernels.py`（13 例三态等价 + n_steps 1/2/3）。
  ⚠ **x86 上测不出这个差异**（本机 fused/serial/plain 三态`max|Δ|=0`，256 行规模下
  结果恰好相同）——x86 结论不构成昇腾证据。
- **学习**（`sparse_pc.py:500-529`）：下行 `ΔW += η_pc·e⊗r`（误差驱动稀疏外积）；
  上行 Oja `ΔW += η_oja·post·(pre − post·W)`；四权重 clip 到 `±pc_w_max`（默认 **2.0**，
  `config.py:29-31` —— 2026-09-28 修复：此前 `cfg.w_max` 从未传入 PC，签名默认 2.0 静默生效）。
- **可选机制**（全部默认关闭）：稳态突触缩放 `homeostasis`（Turrigiano 2008，
  `sparse_pc.py:579-584`）、逐神经元目标发放率 `neuron_target_rate`（内在可塑性，
  `sparse_pc.py:566-577`）、发育期后冻结 `pc_dev_steps`、时序预测目标
  `pc_predictive_target`（`sparse_pc.py:531-564`）、皮层式重尾初始化 `lognormal_init`
  （`phdnet/inits.py:24-40`，幅度 lognormal × E/I 符号比 0.8）。
- **⚠ 生产档表征冻结**：`train/config_1b.py:77` 设 `eta_pc=0.0, eta_oja=0.0`
  （注释「表征冻结（铁律：多轮实测最优解）」），同时开 `eta_stdp=0.02`。
  即生产档下 M2 只做前向推理，**不学习**；可塑性由 M3（STDP）与 M6（读出）承担。
- **可验证等价性**：`k = n_in`（全连接图）时本模块与稠密栈**数值一致**（CSR 求和按列索引升序；
  与 BLAS 内部求和顺序不保证逐位相同）。对拍判据见
  `tests/verifiers/verify_parallel_consistency.py`（`sparse_pc.py:25-27`）。
- **统计接口**：`pc.stats()` 返回 `synapses` / `dense_equivalent` / `connectivity`
  / `k0` / `k1`（`sparse_pc.py:602-609`）。

---

### 2.3 M3 时序关联核（STDP）

**脑同构**：海马 CA3 递归网络与皮层局部突触的 STDP（`phdnet/plasticity.py:1`）。

**结构**（`STDPCore`，`phdnet/plasticity.py:19-125`）：

- **稀疏拓扑**：每神经元 **m_lateral = 16** 条随机出边，`post_idx` 形状 `(n_top, 16)`，
  `W` 只存拓扑内权重（`plasticity.py:41-44`）。默认 `n_top=64` 时连接率 16/64 = 25%；
  1B 档 `n_top=1024` 时 16/1024 = **1.5625%**。
- **突触迹衰减** `lambda_trace = 0.35`（`config.py:26`，注释：「时间窗宽度，
  过宽会引入隔步混淆」）。
- **学习率** `eta_stdp = 0.03`（库默认）/ **0.02**（1B 档，`config_1b.py:78`）。
- **权重上限** `w_max = 1.0`（`config.py:28`，只作用于 STDP 核，主干用 `pc_w_max`）。

**`step` 的顺序是正确性关键**（`plasticity.py:64-125`）：

```
t_pre ← λ·t_pre + pre_rate                      # 1. 先吸收 pre 迹
raw = 2·t_pre[i]·post_rate[k] − t_post[k]·pre_rate[i]
      ↑ 用**更新前**的历史 post 迹算 LTD
W[i] += η·raw（clip 到 [0, w_max]）              # 2. 更新活跃边
t_post ← λ·t_post + post_rate                     # 3. 最后才吸收 post 迹
```

原注释（`plasticity.py:68-70`）：「否则 LTP 与 LTD 中的当前共激活项相消，学习失效」。

**计算量**：`O(活跃 × m)` 而非 `O(N×m)` —— numpy 回退路径只遍历
`np.nonzero(pre_rate > 0)` 的行（`plasticity.py:92`）。**稀疏化是 M3 正确工作的前提**
（`model.py:132-136`）：`PHDNet._rate` 把 tanh 表示半波整流后做 k-WTA 竞争
（`k = max(1, len(r2)//16)`，即 ~6% 激活），低激活率保证不同输入的支撑集区分度。

**平台约束（硬约束）**：**M3 只走 numpy(+numba) CPU**（P30 定稿，`model.py:49-57`）。
显式指定 torch 后端会 **fail-fast**，因为 STDP 的逐突触状态（自适应 LR / 稳态 /
元可塑性 / E-I）在 torch 上无法保留，迁移会丢机制。`NUMBA_OK` 分派条件见
`plasticity.py:84-88`：`adaptive / homeostasis / metaplasticity / ei_synapses`
任一开启即强制 numpy 路径（这些机制需要逐突触状态）。

**可选机制**（全部默认关闭，`config.py:76-82` / `119-135`）：
逐突触自适应 LR（`adaptive_lr`，二阶矩归一 `adapt_rho=0.9`、`adapt_eps=0.5`）、
多尺度双迹（`dual_trace`，长窗 `lambda_trace_slow=0.85` / 权重 `beta_slow_trace=0.5`）、
STDP 稳态缩放（`stdp_homeostasis`，`homeo_target=2.0`，只 downscale 不注能）、
BCM 元可塑性（`metaplasticity`，滑动阈值 θ，`bcm_tau=0.01`）、
E/I 突触类型（`ei_synapses`，`ei_ratio=0.2` = 皮层 ~80/20；抑制性出边权重为负、clip 到
`[−w_max, 0]`，此时 `predict` 不裁剪，因为预测含负向抑制贡献，`plasticity.py:60-62`）。

**`predict` 的作用**：`pred = W·post_idx·prev_rate`（`_predict_edges` 核），是"上一时刻发放率
→ 当前发放率"的时序前瞻。它既是诊断指标（`seq_err = 1 − cos(pred, rate)`），也在
`pred_in_readout=True` 时作为 M6 的第三路特征（`model.py:237-247`）。
⚠ 审计修复（`model.py:238-246`）：训练路径下 `_prev_rate is _last_rate`（同一对象别名），
原实现因此对同一份数据算了两遍；现在训练路径复用 `pred`，但 **`readonly=True` 时两者
确实不同 → 保留原调用**。

---

### 2.4 M4a 工作记忆（PFC 持续放电 + 基底核门控）

**脑同构**：背外侧前额叶的持续放电与基底核门控写入（`phdnet/wm.py:1`）。

**实现**（`WorkingMemory`，`phdnet/wm.py:8-125`）：

- **`n_wm_slots = 4`**（`config.py:34`，注释「有限容量 ≈7±2 的抽象」）。
  槽位 `slots` 形状 `(4, n_top)`，强度 `strength` 形状 `(4,)`。
- **漏衰减**：`gamma_wm = 0.85`（`config.py:35`），每步 `slots *= γ; strength *= γ`
  （`wm.py:31-48`）。**摘要槽不衰减** —— 对应 PFC 对重要内容的主动维持（rehearsal）。
- **门控写入**：`write(r, gate, thresh)` 仅当 `gate ≥ gate_thresh`（0.35）才写
  （`wm.py:88-118`）。写入目标默认「**最弱槽位**」（`_pick_weakest`，`wm.py:21-29`），
  **排除摘要槽**（B3 修复：否则长跑后 argmin 命中摘要槽，把免衰减保护的重要内容覆盖）。
- **可选内容寻址**（`wm_content_address`，默认 `False`，`config.py:170`）：
  写入目标改为「与新内容余弦相似度 ≥ `wm_sim_thresh`（0.6）的最相似槽位」，
  无命中回退最弱槽位 —— 相似内容聚合、减少同义槽位碎片化。
- **可选压缩摘要槽**（`wm_summary_every`，默认 `0` = 关闭，`config.py:93`）：
  每 N 步把 WM 内容经 LTM 吸引子压缩写进免衰减保留槽。大容量表路径必须先稀疏化
  （top-1/8 幅值率）再 `recall`，因为 ±1 稠密模式会被 A2 契约显式拦截（`wm.py:56-68`）。
- **读出**：`read()` 返回强度加权平均 `(strength ⊙ slots).sum(0) / strength.sum()`
  （`wm.py:120-125`）；全零强度时返回零向量。**这是 M6 特征的第二路**。

---

### 2.5 M4b 长期记忆（海马快印迹 / 皮层慢巩固 + 大空间事件驱动表）

M4b 有**两种实现**，由 `big_ltm` 开关切换（`model.py:73-84`）：

#### (a) 稠密双速率 LTM（`big_ltm=False`，库默认）

`LongTermMemory`（`phdnet/ltm.py`）：`W_fast`（n×n 海马快权重）+ `W_slow`（n×n 皮层慢权重）。
`imprint` 一次性 Hebbian 印迹 `W_fast += η_hip·outer(p_out, src)`，去对角、clip 到 `[-1,1]`；
`consolidate` 做 `W_slow ← (1−η_cortex)·W_slow + η_cortex·W_fast` + 可选遗忘；
`recall` 是 Hopfield 式吸引子 `r ← sign(W·r)` 迭代 `n_recall_steps = 6` 步，
`W = beta_slow·W_slow + beta_fast·W_fast`。

#### (b) 大空间事件驱动稀疏表（`big_ltm=True`，**1B 档生产路径**）

**脑同构**：海马/皮层印迹 —— **仅被经验触碰过的突触存在**；单神经元突触数有硬上限
（`tools/audit_brain_parity.py:96-101` 的 A2 条目）。

三层结构：

| 层 | 文件 | 职责 |
|---|---|---|
| 适配器 | `phdnet/bigltm.py`（`SparseLTM`） | n_dim 维表示 → 确定性哈希 → 大空间索引；召回反投影回 n_dim |
| 存储本体 | `phdnet/sparse_table.py`（`SparseSynapseTable` / `OnlineCSRTable`） | 邻接存储 + STDP 式生长 |

**哈希分布式编码**（`bigltm.py:44-53`）：每个维度 `j` 经 `U64(j)*7919 + arange(k_hash)*2654435761`
再 `_mix64` 取模 `n_neurons`，得 `k_hash` 个大空间索引（默认 `big_ltm_k = 4`，`config.py:99`）。
**反投影索引** `rev` 一次性构建后**零维护成本**预 CSR 化供 numba 核用（`bigltm.py:50-57`）。

**A2 契约校验**（`bigltm.py:95-111`）：`SparseLTM` 只接受**稀疏率**（激活维度 ≤ n_dim/4）。
传 ±1 稠密模式会一次性激活全部维度、生成 `n_dim × k_hash` 个索引 → 性能崩塌 + 语义错误
且**无任何报错**。故 `model.py:294` 显式分派：大容量表传 `rate`（稀疏率），
稠密 LTM 传 `sign(rate)`。

**事件驱动**（`model.py:277-301`，条件触发，不是每步）：

- **印迹**：`learn and mode == "encode" and mem_gate > ltm_imprint_gate`。
  `ltm_imprint_gate = 0.8`（`config.py:67-69`）：2026-09-25 审计发现调制器 z 分布
  `std ≈ 0.32` → `gate > 0.8` 仅 ~0.4% 触发，big_ltm 容量栈在 LM 训练中**几乎从不印迹**，
  故调低以激活。每次 `imprint` 学的是 `table.learn(self._prev, cur)`
  ——**相邻两次稀疏率模式的关联**（"fire together, wire together"），`self._prev` 跨调用保留。
- **检索**：`retrieve_now = (mode == "retrieve" and step_count % retrieve_interval == 0)`，
  默认 `retrieve_interval = 8`；`multi_modulation` 时延长为
  `max(2, round(8·(0.5 + ht)))`（5-HT 耐心 → 检索节奏，`model.py:283-284`）。
  可选 `error_triggered_retrieval`（错误触发检索，每 4 步至多一次）。
  召回向量 `rec` 以 `gate=0.5, thresh=0.0` **无条件写入 WM**（`model.py:301`）。
- 两者互斥（`if / elif`，`model.py:289-301`）。

**两种存储后端**（`csr_online` 切换，`model.py:80`）：

| | `SparseSynapseTable`（dict 邻接） | `OnlineCSRTable`（在线可写 CSR） |
|---|---|---|
| 存储 | `out: dict[int, dict[int, float]]` | `keys/vals/size` 三个 dict，每行定长数组 + 预留槽 |
| 每条目内存 | ~100+ B（dict entry + boxed int/float） | ~16 B（int8 权重 + CSR 索引；训练侧口径 5.74× 省：91.8→16.0 B/条目，`train/train.py:283-286`） |
| 行扩容 | dict 自动 | 满 `row_cap` 后按 **2× 增长**，上限 `m_out`（`sparse_table.py:219-226`） |
| 惰性迹衰减 | `trace[i] = trace.get(i,0)·λ^Δt + 1.0`，pre/post **各自独立时间戳**（共用会导致 Δt=0、迹无衰减累加爆炸） | 同 |
| 新突触生长 | 只由 **LTP** 驱动（相邻对约定），LTD 仅作用于**已存在**突触 | 同（槽位顺序 = 首次生长顺序 = dict 插入顺序 → 逐位等价） |
| 逐位等价 | — | 对拍见 `tests/verifiers/verify_csr_equiv.py` |
| 分片落盘 | — | `export_shards(out_dir, n_shards)` / `import_shards`：按行哈希分片写 np.memmap（`sparse_table.py:471+`） |

**⚠ `consolidate` 在容量表上语义退化**（`bigltm.py:233-271`）：没有稠密快/慢权重分离，
巩固退化为对**已生长突触**的轻微衰减 `f = 1 − 0.01·forget`。
⚠ **int8 量化台阶**（诚实边界）：码值域 10~32（q=127）时，1% 乘性衰减落在量化台阶内、
round 后回原码 → **consolidate 零效果**；需 `forget ≳ 3` 才开始起作用。属量化精度固有属性。

**分期睡眠**（`staged_sleep`，默认 `False`，`config.py:129-131`）：`sleep()` 交替
SWS（巩固 + 突触 downscaling，`ltm.downscale(0.9)`，Tononi & Cirelli 突触稳态假说）
与 REM（生成式回放，`slow_only=False` 混合重组；回放强度 × DA 信号）。

**其它可选项**（默认关闭）：`sparse_table_prune`（惰性淘汰最旧 25% 迹，
`sparse_table.py:68-76`）、`growth_guidance`（新突触生长偏好**低入度**目标，
避免 hub 过载）、`sparse_int8`（权重 int8 量化存储）、`sparse_mmap_dir/shards`（分片 mmap）。

---

### 2.6 M5 神经调制

**脑同构**：蓝斑 NE / 多巴胺的意外信号 + 胆碱能系统的编码-检索模式切换
（Hasselmo）。`phdnet/modulator.py:1-7`。

**单通道版**（`Neuromodulator`，`modulator.py:13-30`，默认）：

```
surprise → Welford 在线 z 标准化 → z
gate = σ(mod_gain · z)          # mod_gain = 1.5
mode = "encode" if z > 0.3 else "retrieve"
```

`surprise` 的定义在 `model.py:251`：`surprise = ‖cache["e0"]‖ / √len(e0)`
—— 归一化的 M2 底层预测误差范数。

**多通道版**（`MultiModulator`，`modulator.py:33-92`，`multi_modulation=True` 启用）：

| 通道 | 公式 | 脑对应（Yu & Dayan / Hasselmo） | 增益配置 |
|---|---|---|---|
| ACh | `σ(gain_ach · z)` | 编码/检索模式门控 | `gain_ach = 1.5` |
| NE | `σ(gain_ne·(nov − 0.5))`，`nov = \|z − z_prev\|` | 新颖性 → 可塑性增益 | `gain_ne = 2.0` |
| DA | `σ(−gain_da · z)` | 巩固信号（低 surprise → 巩固/回放） | `gain_da = 1.5` |
| 5-HT | `σ(gain_ht·(nov − 0.3))` | 耐心 / 探索-利用权衡 | `gain_ht = 1.0` |

最终 `gate = ach · (0.5 + 0.5·ne)`（`modulator.py:90`），接口与单通道完全兼容。
**下游接线**（`multi_modulation=True` 时）：

- **NE → STDP 学习率**（`model.py:440`）：`ne_gain = modulator.ne` 乘进 `stdp_scale`。
- **DA → 睡眠回放强度**（`model.py:487-489`）：`eff = replay_ratio · da_scale`。
- **5-HT → 检索节奏**（`model.py:283-284`）：`retrieve_interval = max(2, round(8·(0.5+ht)))`。
- **ACh → 门控**（`model.py:408`）：`mod_scale = (0.3 + 0.7·gate) · task_scale · cp_scale`。

**另一条独立调制通路 T1.1**（默认关闭，`config.py:65`）：读出 NLL 经 Welford z 标准化
→ 门控本步**主干**学习率（`task_modulation`，`model.py:367-371, 400-405`）；
滞后一步的 `_task_gate` 供 T1.3 错误驱动写入（`error_gated_memory`，`config.py:66`）
改变 WM 写入与 LTM 印迹的判据。

---

### 2.7 M6 读出头（稀疏感知器 + softmax）

**脑同构**：IT → 前额叶/前运动皮层的决策读出（`phdnet/readout.py:1`）。
**监督信号仅存在于最末端**（局部感知器规则，误差不反传主干）。

**读出特征** `h`（`model.py:173-181`）：`[r2 ; WM]` 或三拼 `[r2 ; WM ; STDP 预测]`
（`pred_in_readout`，1B 档 `True`）或再拼 top-k 召回线索（`retrieval_topk > 0`）。

**三条连接结构路径**（`Readout.__init__`，`readout.py:787-822`）：

| 路径 | 触发 | 存储 | 前向 | 学习 |
|---|---|---|---|---|
| **稠密** | `conn_k == 0` 且 `hidden == 0` | `(n_out, n_in)` 矩阵（fp32 或量化码本） | BLAS sgemv / 量化码本 LUT matvec | 融合核 / 量化核 / numpy 回退 |
| **结构性稀疏** | `conn_k > 0` | CSR `(indptr, idx, val)`，每行 k 条**存在**入边 | `_csr_matvec`（只遍历存在的边） | `_csr_add_outer`（只更新存在的边） |
| **两级群体读出** | `hidden > 0` | 两个 CSR：`W1`（h→隐藏群体）、`W2`（群体→输出） | `W1` → **k-WTA 侧抑制竞争** → `W2` | `W2` 监督 + `W1` 局部无监督 Oja |

**⚠ M6 稀疏化已经是现行默认（不是架构欠账）**：

- **库默认** `readout_conn_k = 0`（稠密，`config.py:194`）；
  **生产 CLI 默认 `--readout-conn-k 128`**（`train/train.py:291-295`，P108 / fhz 2026-10-01）：
  1B 档 128/3072 = **4.2% 连接率**，取自 4M 档 A/B（k=128 时 PPL 477 vs 稠密 608，
  速度持平）。**注意 CLI 默认 128 与库默认 0 不一致，是事实**（`config_1b.py:81` 透传
  显式参数，`0` 时才是稠密主路径）。
- **稀疏模式仅支持均匀 k**（`_random_csr` 的输出）。加速后端对**非均匀行宽 fail-fast**
  （`phdnet/backends/accel_readout.py:148-153`），因为非均匀需要变长 CSR + segment sum，
  语义与实现都不同。
- **稀疏模式固定 fp32 CSR**（`readout.py:774`）：`self.dtype_name = dtype if conn_k == 0 else "fp32"`
  —— 量化码本与 CSR 不叠加（组合另行立项）。
- **P111 起加速器已实现稀疏读出**：设备侧持有 `(n_out, k)` 稠密 val + `(n_out, k)` 列索引，
  前向 = `Σⱼ Wv[i,j]·ht[Wi[i,j]]`（**逐元素乘 W 再求和**）、更新 = 逐行 rank-1
  `Wv[i] -= η·dp[i]·h[Wi[i,:]]`。此前 `readout_conn_k > 0` 命中 `_unsupported_reason`
  → 整个读出回落 numba CPU（等于关掉了全部设备加速）。对拍见
  `tests/verifiers/verify_accel_sparse.py`。

#### 2.7.1 M6 的第三层稀疏化：幂律异质连接（P124，本轮新接入）

均匀 k-conn 有个结构性缺陷：**高频词（如"的"）与长尾词获得同等容量** ——
长尾被过分配、高频被欠分配。`phdnet/sparse_alloc.py` 实现了按词频分配行宽
（k_i ∝ counts_i^alpha），但从诞生起docstring 就写着「未接入 readout」，**P124 完成接线**。

| 层 | 结构 | 行宽 | 连接率 | 加速器 | 开关 |
|---|---|---|---|---|---|
| 均匀 k（P108） | CSR，`indptr` 等差 | 恒 = k | 4.2% | ✅ gather-GEMV（P111） | `--readout-conn-k 128` |
| **幂律（P124）** | CSR，`indptr` 用 **cumsum** | **不等** | 预算受控 | ❌ **fail-fast** | `--readout-powlaw-alpha`（默认 **0=关**） |

- **生物学依据**：突触巩固 / 修剪（synaptic consolidation & pruning）——
  发育期先铺设**过量**突触，随后活动依赖地修剪：被反复激活的通路被**巩固**
  （保留更多突触），长期沉默的低效通路被**消除**。结果是连接密度随**神经元实际
  使用频率**呈幂律倾斜。这正是 Hebbian 学习（"use it or lose it"）在**连接数**
  这一结构层面（而非仅权重数值层面）的表达。
- **零回归**：`alpha=0` 时**逐位走原均匀路径**（实测 `idx`/`val`/`indptr` 全等）。
- **实测**（Zipf counts，n_out=200, k=8）：

  | alpha | 行宽范围 | k 与词频秩相关 | 总 nnz |
  |---|---|---|---|
  | 0（基线） | 8~8 均匀 | −1.00 | = 预算 |
  | 0.10 | 不等 | +0.56 | **= 预算** |
  | 0.25 | 6~19 | +0.77 | **= 预算** |
  | 0.50 | 不等 | +0.95 | **= 预算** |

  **总 nnz 在任何 alpha 下恒等于预算、不膨胀** —— 这正是分配器做预算控制的意义
  （`assign_conn_counts` 用 `total_budget` 硬约束，撞上 `k_max` 时会花不完，
  那是夹紧的**正常**结果而非 bug）。
- **不变式**：每行入边 ≥ 1（读出行不允许 0 条边）、列索引**行内升序且无重复**
  （`_csr_matvec` 的隐含契约）。对拍 `tests/verifiers/verify_powlaw_readout.py`（32 例）。
- **⚠ 加速器不支持非均匀行宽**：行宽不等需要变长 CSR + segment sum，与均匀 k 的
  「稠密张量 gather-GEMV」是**不同算法**（P19纪律：不能「能跑但语义不同」）→
  开 `alpha>0` 会**回落 numba 读出**，NPU 加速失效。**这是正确行为，不是 bug。**

**⚠⚠ 必须如实记录的局限：词频目前是「代理值」**

构造读出发生在 `PHDNet.__init__`，那时词表刚由 `vocab_text` 建好、**语料尚未流过**
→ **拿不到真实 Zipf 频次**。`phdnet/word_lm.py::_powlaw_proxy_counts` 返回
`1/rank` 造出单调递减形状（Zipf 的形状），但：

> 词表是 `sorted(set(...))` = **字典序**，**不是频次序** → 该代理**依据不足**，
> 幂律倾斜的方向**可能是反的**。

故**默认 `alpha=0`（关闭）**。开启前**必须先 A/B 验证代理与真实频次是否相关**。
正解是两阶段：先用均匀 k-conn 跑一段收集真实频次，再重建读出 —— **未实现**
（它会改变 checkpoint 结构）。

**剩余欠账**：即使幂律开启，与人脑 ~2e-8 的连接率仍差约**6 个数量级**；
且频次来源尚是代理而非真实统计。
  ⚠ **idx 刻意用 int64** —— int32 会让 NPU 侧 gather 走类型转换。

**两种学习规则**（`readout.py:1061-1157`）：

- **感知器**（`readout_softmax=False`，库默认）：`ΔW = η·(t − y)⊗h`。
- **softmax 感知器**（`readout_softmax=True`，**词级 LM 强制开启**，
  `phdnet/word_lm.py:42-43`）：`ΔW = −η·(p − t)⊗h`，返回 `nll = −log p(correct)`。
  梯度只有末端一层，**不反传 M2/M3**。

**稀疏化的两难（诚实记录）**：`tools/audit_brain_parity.py:139-146` 记录了
256 维栈下 `readout_conn_k=8`（连接率 1.0%）的实测：**PPL +14.8%**（质量劣化）。
即**过小的 k 会伤害质量** —— 单层线性 softmax 读出的连接需求远高于分布式编码主干。
1B 档取 k=128（4.2%）而非 8，是这个约束下的折中（该处的速度收益见性能文档）。

**可选子机制**（全部默认关闭）：

- `readout_softmax`（`config.py:58`）、`eta_readout = 0.15`（`config.py:59`，
  注释「实测最优区间 0.15–0.20，0.25+ 退化，0.5 发散」）
- `eta_readout_anneal` / `eta_readout_floor`（乘性退火与下限，防后期样本覆盖早期学习）
- `readout_w_clip`（权重范数上限）
- `minibatch_size`（梯度累积，脑对应「突触巩固的时间整合」）
- `readout_hidden` / `readout_hid_k` / `readout_eta_hid`（两级群体读出）
- `lognormal_init`（皮层式重尾 + E/I 符号比 `exc_ratio = 0.8`）
- `segment_check` / `readout_recurrence`（长程整段复制，见 §3.3）
- 量化码本 `readout_dtype`（见 §5.2）

---

## 3. 数据流与逐 token 语义

### 3.1 训练侧主循环

`train/train.py:771-851`：

```
for ep in range(args.epochs):                     # 状态**不重置**
    src = PrefetchChars(...)                      # 多进程预取
    stream = StreamingTokenizer(lm.tok.seg, src, assistant_marker=...)
    p2, p1, t0 = None, next(), next()             # epoch 首步 prev 断开
    while t0 is not None:
        x = lm.tok.encode_composite(p1, p2)      # [当前词 SDR ; 前词 SDR]，2×n_sdr
        tgt = lm.tok.onehot(stoi[t0])
        d = lm.net.step(x, target=tgt, learn=trainable, target_idx=stoi[t0])
        p2, p1, t0 = p1, t0, next()               # 逐 token 推进
```

- **上下文窗口 = 2 词**（当前词 + 前词），由 `encode_composite` 的双 SDR 拼接实现
  （`phdnet/word_encoder.py:173-200`）。**没有位置编码、没有 K/V 缓存**。
  长程依赖由 big_ltm 印迹承担（`train/train.py:719`：`long-range dependency carried by
  big_ltm imprints → supports arbitrarily long continuous sequences`）。
- **OOV 整步跳过**（`train/train.py:838-847`）：任一端 OOV 时 `oov_skipped += 1`、流不断。
- **SFT 回复掩码**（`train/train.py` 的 `assistant_marker`，P26）：最后一个
  `assistant_marker` 之前的 prompt 段用 `learn=False` 推进状态（不更新权重），
  只有助手回复段计损失。语义依据：「对 prompt 计损失会把模型往『复读用户问题』的方向拉」
  （`train/corpus_stream.py:259-262`）。

### 3.2 状态跨什么连续

| 状态 | 载体 | 跨 token | 跨 shard | 跨 epoch |
|---|---|---|---|---|
| WM 槽位内容/强度 | `wm.slots` / `wm.strength` | ✓ | ✓ | ✓ |
| STDP 突触迹 | `t_pre` / `t_post` | ✓ | ✓ | ✓ |
| STDP 权重 | `stdp.W`（每步原地更新） | ✓ | ✓ | ✓ |
| LTM 印迹 | `table` 突触 | ✓ | ✓ | ✓ |
| LTM 模式指针 | `ltm._prev` | ✓ | ✓ | ✓（分段训练用 `begin_episode()` 显式切断，`bigltm.py:225-231`） |
| 调制器统计量 | `mu` / `m2` / `count` | ✓ | ✓ | ✓ |
| 读出 W | `readout.W`（**每步原地更新**） | ✓ | ✓ | ✓ |
| 发育经验计数 | `net._exp` | ✓ | ✓ | ✓ |
| `step_count` | `net.step_count` | ✓ | ✓ | ✓（它参与 `retrieve_interval` 判定，检索节奏跨 epoch 连续） |

**唯一被切断的**：`p2`（前词链接）在 epoch 首步断开，`prev = None` →
`encode_composite` 的后半段清零（`word_encoder.py:192-199`）。

### 3.3 O5 长程整段复制（默认关闭）

`readout_recurrence`（`config.py:271-272`）：把上一步读出的**预测线索**回灌为下一步
WM 线索（`model.py:270-272`，`recur_gain = 0.5`）。脑对应 PFC→感觉皮层的**反馈再入**
（reentry）。`segment_check > 0`：每 N 步做段内一致性校验，低置信输出降权。

⚠ **明确不是位置编码**（`config.py:270`）：「不引入位置编码 / 顺序无关性（铁律），
仅做输出回灌与段内一致性约束」。

---

## 4. 容量账

**口径**：`train/config_1b.py::capacity_report`（`config_1b.py:88-126`），
纯算术、不构建网络，与 `phdnet/model.py:558-585::count_params` **同口径**：
计入编码器 + PC 四权重（按存在突触）+ STDP 权重 + 读出权重 + WM 槽位/强度 + LTM 存储
（大容量表计入「已生长突触数」而非容量上限）。**不含**推理期临时迹/时间戳
（视为状态而非参数）—— 报告时须注明此口径。

⚠ 两条口径的差异：`capacity_report` 报的是**容量上限**（`big_ltm_N × big_ltm_m`），
`count_params` 报的是**已生长突触数**。训练初期大空间表利用率低，`count_params`
远小于容量；随训练增长趋近上限（`--report` 可随时查看 `stats`）。

### 4.1 1B 档

```
big_ltm 容量 = big_ltm_N × big_ltm_m = 2^24 × 72 = 1,207,959,552   (config_1b.py:48)
```

这是 1B 的**主体**（占总量的 99% 上下）。突触**不是构建即存在**，而是随经验生长
（"fire together, wire together"），每神经元 ≤ `big_ltm_m` 条出边是硬容量约束。
每步计算量只正比于**活跃神经元数**（事件驱动），与 1B 总容量无关。

### 4.2 四档预设对比

`capacity_report` 实际输出（`V = 51,962` 词表，`readout_conn_k = 128`；
由 `build_cfg(preset, readout_conn_k=128)` + `capacity_report(cfg, 51962)` 算得）：

| 项 | `smoke` | `1b` | `1b_max` | `30b` |
|---|---|---|---|---|
| width（`n_sdr`=`n_mid`=`n_top`） | 256 | **1024** | 4096 | 1024 |
| `k_sparse` | 32 | 128 | 512 | 128 |
| `conn_k`（M2 每神经元入边） | 32 | **128** | 512 | 128 |
| M2 连接率 | 12.5% | **12.5%** | 12.5% | 12.5% |
| `big_ltm_N` | 2^20 | **2^24** | 2^24 | **2^29** |
| `big_ltm_m` | 60 | **72** | 60 | **56** |
| M1 编码器 | 131,328 | 2,098,176 | 33,558,528 | 2,098,176 |
| M2 主干 CSR | 32,768 | 524,288 | 8,388,608 | 524,288 |
| M3 STDP | 4,096 | 16,384 | 65,536 | 16,384 |
| M6 读出（稀疏 k=128） | 6,651,136 | 6,651,136 | 6,651,136 | 6,651,136 |
| **固定突触小计** | 6,819,328 | 9,289,984 | 48,663,808 | 9,289,984 |
| **M4b 容量** | 62,914,560 | **1,207,959,552** | 1,006,632,960 | 30,064,771,072 |
| **总容量** | 69,733,888 | **1,217,249,536** | 1,055,296,768 | 30,074,061,056 |
| **≈ ×10⁹** | 0.0697 | **1.2172** | 1.0553 | **30.0741** |
| M4b 占比 | 90.2% | 99.2% | 95.4% | 99.97% |
| 神经元索引映射内存 | 33.6 MB | 536.9 MB | 536.9 MB | 17,179.9 MB |
| 静态内存 | 55 MB | 76 MB | 423 MB | 76 MB |

预设定义见 `train/config_1b.py:45-58`。**用途**：`smoke` = 管线验证（分钟级，
`⚠ < 1B` 是预期的）；`1b` = 标准档；`1b_max` = 大主干档（width 4096 但 m 降到 60
以守住 ~1B）；`30b` = 2^29 神经元 × 56 突触（`m` 从 72 降到 56 是因为 2^29×72 = 38.6B 会超）。

⚠ **词表口径必须标明**：`capacity_report` 的 `n_ro = vocab_size × (readout_conn_k or n_h)`
（`config_1b.py:102`）。上表用 **51,962**（1B 档实建词表）。若按设计态满词表
**73,958**（`tools/bench_accel.py` 的缺省 `V`，用于访存上界估算），1B 档读出为
9,466,624、总容量 **1,220,065,024 ≈ 1.2201e9**；读出改回稠密（`readout_conn_k=0`）时
1B 档总容量为 1,437,797,376 ≈ **1.4378e9**（V=73,958） / 1,370,225,664 ≈ **1.3702e9**（V=51,962）。
**三个数都对，区别只在词表与稀疏档位。**

### 4.3 30B 档的诚实边界（`config_1b.py:50-56`）

① 稠密下 30.3B × fp32 = **121 GB** 权重（依赖大空间表的在线 CSR 稀疏存储才落得住；
服务器内存需 ≥ 200 GB）。
② 训练速度随 LTM 访存线性放大（**具体数字见性能文档**），**单机 1M tokens 不现实**，
需要多机/长跑预算。
③ 词表/读出规模不变。

### 4.4 内存口径

`capacity_report` 的 `mem`（`config_1b.py:107-114`）：

- 编码器 `n_enc × 8`（fp64 权重 + fp32 偏置）
- 主干 CSR `n_pc × 12`（**int32 索引 + fp64 权重**）
- STDP / 读出 `n × 8`（fp32）
- **大空间表按已生长突触计**：dict 版 ≈100 B/条，在线 CSR 版 ≈10 B/条
  （训练侧按 int8 权重 + CSR 索引口径报 ≈16 B/条目，`train/train.py:283-286`）。
  **内存只随已生长突触数增长，与容量无关** —— 这是 32 GB 机器上长跑 / 30B 档的安全前提。

---

## 5. 稀疏性设计

### 5.1 全局稀疏账

| 层 | 机制 | 稀疏维度 | 1B 档口径 |
|---|---|---|---|
| M1 | k-WTA top-k | 激活单元 | `k_sparse` 128 / `n_sdr` 1024 = **12.5%** |
| M2 | CSR 入边 | **连接** | `conn_k` 128 / 1024 = **12.5%** |
| M3 | 出边拓扑 + 发放率 k-WTA | 连接 + 激活 | 16/1024 = **1.5625%**（连接）；`n//16` = 6.25%（发放） |
| M4b | 事件驱动生长 | **突触存在性** | 每神经元 ≤ 72 出边；仅被触碰过的存在 |
| M6 | CSR 入边（均匀 k） | **连接** | 128/3072 = **4.2%**（CLI 默认）/ 0 = 稠密（库默认） |

**M2 稀疏的实测依据**（`config.py:182-184`）：连接率 12.5%（k=32）时 PPL 与稠密持平
（+0.03%~−0.30%），k=16（6.25%）时 −0.51%；**突触存储压缩 6.7–32×**。
（这些是**质量**数字；加速倍数属性能文档。）

### 5.2 精度作为稀疏的替代维度（读出码本）

P9 精度体系（`readout.py:354-368`）：低精度格式**按原生位型存储为码本**，计算时
LUT 反量化到 fp32、更新后重量化写回。**内存流量 ∝ 存储位宽**。

`RO_DTYPES`（**`readout.py:377`**）：`fp32` / `fp16` / `bf16` / `int8` / `int16` / `int32`。
⚠ **这张表里既没有 `fp8` 也没有 `fp4`**：
`int4`/`fp4` 已按 **P191e**（P152 定案：每步 unpack 抵消存储收益）整体删除，
显式请求构造期 fail-fast；`fp8` 在 **numba 侧**只是 `int8` 的**兼容别名**
（`_RO_DTYPE_ALIASES = {"fp8": "int8"}`，`readout.py:375`，P100 正名）。
**softmax/NLL 一律在 fp64 上计算** —— 这是低精度训练的标准「高精度主回路」结构。

⚠ **但默认不是 `RO_DTYPES` 里的任何一项 —— 默认是 `fp8`**（`config.py:318`
`readout_dtype: str = "fp8"`；`train/train.py:324` `default="fp8"`，两处一致）。
P189（fhz 2026-10-07「模型 fp8、迭代 fp16」）的 `fp8` 是**叠加在码本之上的存储/迭代方案**，
**不是** `RO_DTYPES` 的一个条目：
W 以 **fp8 e4m3fn 位模式由 uint8 承载**（1 B/元素，访存收益完整），
迭代在 **fp16 域**完成后 RNE 舍回 fp8 位模式；CLI 的 `choices` 也只有
`["fp32","fp16","bf16","fp8"]`（无 int 族，int 族显式请求仍禁 P163）。
⚠ 另：**稀疏读出（`conn_k>0`）内部强制 fp32**（`readout.py:542,660`
`dtype_name = dtype if conn_k == 0 else "fp32"`）—— 稀疏的意义正是省流量，
不叠加语义损失。

判据是 `tools/probe_readout_precision.py` 实测的「舍入后元素实际发生变化的比例」
（= 更新是否被保留），在 `|W|~1e-2` / `|dp|~1e-6` 档：

| 格式 | 非目标行更新保留率 | 目标行保留率 |
|---|---|---|
| fp8（**默认**，P189） | **最低**（步长比 fp16 粗约 4 倍） | 100% |
| fp32 | **99.95%**（精确 `p − t`，显式 opt-in） | 100% |
| fp16 | 26.67% | 100% |
| bf16 | 5.79% | 100% |

**低精度下「只保留目标行提升」→ 学习规则退化为纯 Hebbian。**
⚠ **默认 fp8 意味着这是默认就在承担的代价**，不是可选项：fp8 的粗步长会吞掉
≈1e-6 的非目标行更新（抑制项丢失），学习偏向「只抬目标行」的全行抬升。
这是 fhz 2026-10-07 的**知情取舍**（验收靠 `--probe-every` 固定探针集做 PPL A/B，
见《文档写作规范》§2.4）。要精确 `p − t` 语义，显式 `--readout-dtype fp32`。

⚠ **M1 编码器权重默认 `fp64`**（`config.py:330` `encoder_dtype: str = "fp64"`）
而**非 bf16** —— numpy/BLAS 路径下低精度存储每次都要付上采样转换，
实测比 fp32 直接 GEMV 更慢；生产 CLI `--encoder-dtype` 覆盖为 `fp32`
（`train/train.py:408`）。**读出侧精度与本字段无关**，由 `readout_dtype` 决定
（CLI `--readout-dtype`，默认 fp8）。

⚠ 量化码本在**加速器**上有能力表（`_unsupported_reason`，`accel_readout.py:1767-1790`）：
`int16`/`int32` 被**显式拒绝并说明原因**（P112，能力表与 `_DT` 对齐）；
`fp4`/`int4` 被拒（P191e）；`fp8`/`int8` **放行**（`:1776-1777` 返回 `None`）
—— fp8 曾被误禁（P92 教训：加新 dtype 时必须同步能力表），P189 起改为
运行时探测（`backends/fp8_capability.py`「真跑一次」）而非静态剔除。

### 5.3 幂律稀疏连接分配器（`phdnet/sparse_alloc.py`）

**脑同构**：突触巩固 / 修剪（synaptic consolidation & pruning）——
发育期先铺设**过量**突触，随后活动依赖地修剪：被反复激活的通路被**巩固**（保留更多入边），
长期沉默的低效通路被**消除**。结果是连接密度随**神经元实际使用频率**呈**幂律倾斜**
（`sparse_alloc.py:3-13`）。这是 Hebbian「use it or lose it」在**连接数**这一结构层面
（而非仅权重数值层面）的表达。

**两个公开 API**（纯 numpy、无副作用、确定性，不依赖随机数）：

| 函数 | 作用 |
|---|---|
| `assign_conn_counts(counts, density, alpha, k_min, k_max, total_budget, n_in)` | 按 `k_i ∝ counts_i^alpha` 分配每行入边数，返回 `(k,) int64` |
| `build_powlaw_csr(rng, n_out, n_in, counts, alpha, density, ...)` | 生成**行宽不等**的 CSR（`indptr` 用 cumsum 而非等差） |

**与均匀 k-conn 的区别**（`sparse_alloc.py:20-27`）：均匀 k-conn 每行**恰好** k 条入边
→ 高频词与长尾词获得同等容量；幂律分配行宽不等 → 总 nnz 由 `density`/`total_budget`
**统一控预算**，不因倾斜而膨胀。`alpha = 0` **精确退化**到均匀 k-conn
（`build_powlaw_csr` 的 RNG 调用顺序与 `_random_csr` 刻意保持一致 → 逐位一致）。

**分配算法**：**最大余数法**（largest remainder / Hare quota）+ 水位式迭代
（`_bounded_alloc`，`sparse_alloc.py:83-131`）：每轮按 `w` 比例给连续配额 → 向下取整 →
剩余单元按**小数部分降序**补给（先剔除已顶到 `k_max` 的行再补给 1，否则突破上限）→
撞 `k_max` 的行退出，下轮在剩余行重新分配。保证 `k_min ≤ k ≤ k_max` 且
`k.sum() ≤ total_budget`；同一 `(w, budget, lo, hi)` → 唯一解（无随机）。

**已知局限（务必阅读，`sparse_alloc.py:29-37`）**：

1. `alpha=1` 在强偏斜词频（Zipf(1.0)，max/min 可达 1e4~1e6）上**极易撞下限**：
   预算摊给长尾词的份额不足 1 整数 → 大量行被 `k_min` 夹紧，秩相关下降。
   **实践中 `alpha ∈ [0, 0.5]` 更合适**（`alpha` 实为**压缩指数**）。
2. 高频行同样可能撞 `k_max`（甚至 `k_max = n_in` 即该行退化为稠密）。两端同时饱和时
   秩相关必然被 ties 拉低 —— 这是**预算的真实约束**，不是 bug。
3. ⚠ **本模块是纯分配函数：不参与前向/反向，也未接入 `readout.py`。**
   接线需自行保证 `indptr/idx/val` 与下游 `_csr_matvec` 核的约定一致
   （列索引行内升序、无放回；本模块已保证）。
   当前状态：加速后端对非均匀行宽 **fail-fast**（`accel_readout.py:148-153`）；
   `tools/bench_readout_sparse.py:113-160` 通过候选名探测 + 本地 `csr_from_widths`
   回退做幂律分组扫描，**回退路径不崩整轮**。

---

## 6. 语料与数据入口

### 6.1 冻结评测基准（**在仓库根 `eval_corpus/`，不在 `datasets/`**）

| 文件 | 字符数 | 用途 |
|---|---|---|
| `eval_corpus/internal_corpus.txt` | **27,034** | 内部基准，按 `int(len*0.8)` 划分 |
| └ 训练段 | **21,627** | 25%→50%→75%→100% 四档学习曲线（`eval_tasks_lm.py:26-33`） |
| └ 评估段 | **5,407** | 字符归一 PPL / bpc |
| `eval_corpus/ood_wiki.txt` | **20,187** | 远域（OOD）探针 |

⚠ **字符数口径**：`27,034` = Python `open(..., encoding="utf-8").read()` 的返回值
（`tests/eval_suite.py:38-40` 的实际口径，也是本文一切评测数字的基准）。
该文件在磁盘上是 **CRLF** 行尾（363 个 `\r\n`、73,934 字节），按字节解码是 **27,397**
字符、按 `\n` 归一后是 **27,034**。`ood_wiki.txt` 同理（磁盘 CRLF 20,212 / 归一 20,187，
56,891 字节、25 行）。**`tools/rebaseline.py:29` 的注释写「27,405 字符」，与这三个口径
都不符** —— 见 §9。

**评测配置**（`tests/eval_common.py:15-18`）：`BASE` = 256 维栈（`n_sdr=n_mid=n_top=256`）、
`k_sparse=32`、`eta_pc=eta_oja=0`、`eta_stdp=0.02`、`seed=11`、`pred_in_readout=True`；
`SEG` = `dict(max_len=6, min_count=5, min_entropy=1.0)`。
⚠ **评测显式传 `k_sparse=32`**（12.5%），与库默认 16（6.25%）**不同口径**。

### 6.2 训练数据两入口

| 入口 | 路径 | 机制 |
|---|---|---|
| **本地** | `datasets/{sft,pretrain}/*.parquet` | `DATA_FILES`（`train/train.py:118-128`）。`pretrain` 与 `pretrain_zh` 指向同一目录，由 `PrefetchChars(lang=)` 区分 |
| **远程** | `--remote-data` | ModelScope `fhzfhz/Mixture-General-Mini`（`train/train.py:116`），**HTTP Range 流式零落盘**，仅支持 ModelScope |

远程路径（`phdnet/ms_stream.py`）：

- 列目录：官方 SDK `HubApi.get_dataset_files`（分页，`_PAGE_SIZE = 100`）。
- 读文件：fsspec HTTP Range 可 seek 流（`_BLOCK_MB = 8`）→ `pq.ParquetFile(filelike)`，
  **零原始落盘**（依赖 fsspec + aiohttp）。
- **分片级抽样**：`--remote-fraction 0.3`（默认）= 取排序后**前 30%** 分片 ——
  **前缀子集**，保持数据顺序语义，词表扫描与训练流开头逐字符一致。
- 鲁棒性：`_HTTP_TIMEOUT_S = 60`、`_READ_RETRIES = 3`、`_RETRY_BACKOFF_S = 2.0`（指数退避；
  重试只重建流、不改数值 → 逐位等价）；预取进程上限 `REMOTE_MAX_PROCS = 4`
  （每进程独立 aiohttp 连接池 + 8 MB block 缓存，8 进程可能打爆服务端限流）。
- 路径格式：`ms://<owner>/<dataset>/<路径或 glob>`。

**统一读取层**（`phdnet/corpus.py`）：`load_text` / `iter_texts` / `write_parquet` /
`corpus_stats` / `open_parquet_source`，支持 txt / parquet / jsonl 三形态。
**兼容性铁律**（`corpus.py:14-19`）：只**新增**读取能力，不改动既有默认路径 ——
`tests/eval_common.py` 仍读纯文本 `.txt`，行为逐位不变。

**其它 CLI 语料开关**：`--data-lang {all,zh,en}`（默认 `all`；按 parquet `lang` 列过滤，⚠ P119 从 `--lang` 拆出——`--lang` 现只管终端语言、默认 `en`；原 `--lang {all,zh,en}`（默认 `all`；按 parquet `lang` 列过滤，
**词表扫描不受影响** —— 词表是训练流的超集，OOV 恒 0）、`--vocab-scan {head,full}`
（默认 `head`，采样 `--vocab-sample-chars 4000000` 字符建词表）、
`--vocab-file`（外部词表，给了就跳过扫描）、`--remote-fraction`。
入口参数全表见 `train/README.md`。

### 6.3 词法（T2.1 / T2.3）

`SEG_KWARGS = dict(max_len=6, min_count=5, min_entropy=1.0)`（`train/config_1b.py:42`）。
`StreamingTokenizer` 是**在线贪心最长匹配**，与离线 `WordSegmenter.tokenize`
**逐位一致**，内存 `O(max_len)` → 支持任意长语料流（`train/corpus_stream.py:243-249`）。

---

## 7. 评测口径

### 7.1 三个层级

| 层级 | 命令 / 脚本 | 口径 |
|---|---|---|
| **零回归门禁** | `python tests/run_tests.py fast` | 分层回归（版本一致性、核心三实验、容量层小规模、词级 LM 冒烟…），**< 1 分钟**。`--full` 叠加耗时验收 |
| **专项验证** | `tests/verifiers/*.py` | **21 个**独立验证脚本（CSR 等价、融合核逐位、加速器对拍、稀疏读出门禁、检查点往返、分片一致性、精度探针…） |
| **基线锚点** | `tools/rebaseline.py` | 两个口径（4,000 字符 / 全语料）+ 吞吐 + 主干连接率 |

`tests/eval_suite.py` 是**跨任务对照评测**（PHD-Net vs nanoGPT 级 Transformer，
同语料 / 同字符预算）：任务 1 语言建模、任务 2 延迟复制、任务 3 单样本记忆、
任务 4 持续学习 BWT、任务 5 多跳链式推理、任务 6 效率。

### 7.2 PPL 口径

- **字符归一 PPL**（`ppl_char`）= `exp(平均 NLL)`，其中 NLL 逐 token 累加再除以
  **字符数**（不是 token 数）—— 这是跨分词粒度可比的关键。
- **bpc** = `log2(ppl_char)`。
- 学习曲线：训练比例 `f ∈ (0.25, 0.5, 0.75, 1.0)`，每档训完在**同一 eval 段**上评
  （`eval_tasks_lm.py:26-33`）。
- 1B 档语料训练循环产出 `seg_nll` 滑动均值 → `exp(mean)` 即 sliding PPL
  （`train/train.py:857-859`）。

### 7.3 现行锚点

⚠ **锚点是历史值，与代码无绑定**。`tools/rebaseline.py:29` 的 `ANCHOR` 字典：

| 口径 | 锚点 |
|---|---|
| 4,000 字符 | **394.4687** |
| 全语料 | **359.2603** |

**旧锚点 90.2480 / 73.1166 / 78.16 属自指语料口径，只存于 git 历史，不可混用**
（2026-09-28 更换为中文维基语料后重测）。

⚠ **归属与内部矛盾**：`docs/文档写作规范.md` §2.7 把锚点列在其事实基线里；
`rebaseline.py` 的 docstring（`rebaseline.py:9-10`）仍写「4,000 字符 96.7241 /
全语料 77.5261」与「冻结语料 23,504 字符」—— **与同文件 `ANCHOR` 字典和当前语料均不符**，
见 §9。

### 7.4 语料语种实测（待核）

远程分片实测语种：**en 50% / unk 47%（实为代码，CJK 0%）/ zh 仅 2%** ——
与早期记录「中文 95%」**直接矛盾**，标为**待核**（数据问题，非训练代码问题）。
出处：`docs/文档写作规范.md` §2.3。

---

## 8. 六机制一览表

| 机制 | 脑同构 | 核心文件 | 关键配置（默认） | 稀疏维度 |
|---|---|---|---|---|
| M1 稀疏分布式编码 | 初级感觉皮层稀疏放电 + 侧抑制（k-WTA） | `phdnet/sparse_encoder.py` | `n_sdr=256`、`k_sparse=16`、`encoder_dtype="fp64"`（**库默认**；生产 CLI 覆盖为 `fp32`，P107） | 激活 6.25% |
| M2 预测编码主干 | 皮层预测编码层级 + sparse connectivity | `phdnet/sparse_pc.py` | `n_mid=128`、`n_top=64`、`n_infer_steps=1`、`eta_pc=eta_oja=0.02`、`pc_w_max=2.0`、`conn_k=0` | 连接 12.5% |
| M3 时序关联核 | 海马 CA3 递归 + 皮层局部突触 STDP | `phdnet/plasticity.py`、`stdp_kernels.py` | `m_lateral=16`、`lambda_trace=0.35`、`eta_stdp=0.03`、`w_max=1.0` | 连接 1.56%（1B 档） |
| M4a 工作记忆 | PFC 持续放电 + 基底核门控 | `phdnet/wm.py` | `n_wm_slots=4`、`gamma_wm=0.85`、`gate_thresh=0.35` | 有限槽位 |
| M4b 长期记忆 | 海马快印迹 + 皮层慢巩固 + 吸引子补全 | `phdnet/bigltm.py`、`sparse_table.py`、`ltm.py` | `big_ltm_N=2^24`、`big_ltm_m=60`、`big_ltm_k=4`、`ltm_imprint_gate=0.8` | 突触存在性 |
| M5 神经调制 | 蓝斑 NE / DA + 胆碱能编码-检索切换（Hasselmo、Yu & Dayan） | `phdnet/modulator.py` | `mod_gain=1.5`、`multi_modulation=False` | — |
| M6 读出 | IT → 前额叶/前运动皮层 | `phdnet/readout.py`、`backends/accel_readout.py`、`sparse_alloc.py` | `readout_softmax=False`、`eta_readout=0.15`、`readout_conn_k=0`（库）/ **128（生产 CLI）**、`readout_powlaw_alpha=0.0`（默认关）、`readout_dtype="fp8"`（**P189**，库默认即 fp8） | 均匀 k 连接 4.2%（生产）；幂律**已接入但默认关闭**（词频是代理，见 §2.7.1） |

**跨机制的架构强化开关**（全部默认关闭）：稳态突触缩放 `homeostasis`、
错误触发检索 `error_triggered_retrieval`、发育期临界期 `critical_period` + 突触修剪
`prune_threshold`、验证驱动睡眠 `plateau_sleep`、情景记忆双向校验 `bidir_check`、
储备库回放 `readout_replay`、稀疏 int8 量化 `sparse_int8`、PC 发育期冻结 `pc_dev_steps`。

**P192 新增的计算后端开关**（全部默认关闭，铁律④）：
`cython_kernels`（`off`/`auto`/`force`，M2/M3 换 Cython nogil 核）、
`cann_dispatch`（只核对 P120 已默认开的 CANN 下发项是否真生效）。
⚠ `readout_pipeline` **已停用**（传了直接报错退出）—— 依据见
`docs/PHD-Net_性能评估与迭代方案.md` §8.3。

---

## 9. 源码与既有文档口径不一致清单（本次核对发现）

以下都是我**从源码核对**出来的既存不一致。本文采用「源码口径」，
但需回修源头或 `docs/文档写作规范.md`：

| # | 位置 | 不一致 | 本文采用 |
|---|---|---|---|
| 1 | `phdnet/config.py:197` vs `train/train.py:467` | `readout_conn_k` **库默认 0（稠密）**、**生产 CLI 默认 128** | 两者都写，并标明各自口径 |
| 2 | `docs/文档写作规范.md` §2.2 表 | 仍写「M6 读出**100% 稠密 = 架构欠账**」 | **已过时**。P108（2026-10-01）起 CLI 默认 `readout_conn_k=128`（4.2% 连接率）；P111 起加速器已实现均匀 k 的 gather-GEMV |
| 20 | `tools/rebaseline.py:9-10` / `phdnet/sparse_pc.py:29-32` | P124 之后 M6 已**三层稀疏化**（均匀 k / 加速器 gather-GEMV / 幂律异质），但 `phdnet/sparse_alloc.py` docstring 的「未接入 `readout.py`」**已过时** | **P124 已接入**（`alpha=0` 逐位等价）。建议把该 docstring 改成「P124 起已接入，默认 alpha=0」 |
| 21 | 幂律的**词频来源** | 文档若写「按词频分配」而不提来源，会误导读者以为用的是真实 Zipf 频次 | 实为**代理值** `1/rank`（`word_lm.py::_powlaw_proxy_counts`），因构造期语料未流过；且词表是**字典序非频次序** → 代理依据不足。**默认 alpha=0**，见 §2.7.1 |
| 22 | `phdnet/backends/accel_readout.py:148-153` | `AccelReadout.__init__` 只在**构造期**检查行宽均匀性 | 幂律路径下会走到构造期抛 `ValueError` → 被 `pick_readout_backend` 兜底 except 吞掉 → **回落原因丢失**。**P124 已在 `_unsupported_reason` 提前拒绝**（P19 纪律） |
| 3 | `tools/rebaseline.py:29` 注释 | 写「冻结语料 **27,405** 字符」 | 三个口径：磁盘 CRLF **27,397** / 归一 **27,034**（= `eval_suite` 实际）/ 73,934 字节。**27,405 与三者都不符** |
| 4 | `tools/rebaseline.py:9-10` docstring | 仍写「冻结语料 **23,504** 字符」+ 旧锚点 `96.7241 / 77.5261` | 与同文件 `ANCHOR` 字典（`394.4687 / 359.2603`）**自相矛盾** |
| 5 | `docs/文档写作规范.md` §2.2 / §2.7 | 1B 档容量写 `1.370e9`（稠密读出、V=51,962 口径） | **三个都对**：V=51,962/k=128 → **1.2172e9**；V=73,958/k=128 → 1.2201e9；V=51,962/k=0 → 1.3702e9。差异来自词表与稀疏档位，必须标明 |
| 6 | `docs/文档写作规范.md` §2.2 表 | M1 写「输出 **12.5%** 稀疏」 | 库默认 `k_sparse=16`/`n_sdr=256` = **6.25%**；12.5% 是评测口径（`k_sparse=32`）与 1B 档（128/1024） |
| 7 | ~~`docs/文档写作规范.md` §2.4 / §2.8 写「读出 **CLI 默认 fp16**（库 config 默认 bf16）」~~ | ✅ **两处均已过时，现行口径 = CLI 与库 config 都是 `fp8`**（`config.py:318`、`train/train.py:324`，P189 2026-10-07）。`文档写作规范.md` §1 禁止表与 §2.4 已同步为 fp8，并注明「fp32 默认是 P110 的历史口径」 | 🟢 已修。本表保留此行作为**「修正一次 ≠ 永久正确」**的实例：P110 改 fp32 → P163 改 fp16 → **P189 改 fp8** |
| 8 | `docs/文档写作规范.md` §2.8 | 写 `--readout-conn-k 0` | 实际 **128**（P108，`train/train.py:467`） |
| 9 | `phdnet/sparse_encoder.py:12-14` 模块 docstring vs `config.py:330` | docstring 说「M1 默认 `fp32`」（且第 14 行仍写「读出侧 bf16 已由 `--readout-dtype` **默认启用**」—— 读出侧默认早已不是 bf16，见本表第 7 行） | **config 是 `fp64`**（`sparse_encoder.py` 的形参默认是 `dtype="fp32"`，但 `model.py:49` 传的是 `cfg.encoder_dtype`） |
| 10 | ~~`train/train.py:250-255`~~ | ~~`--m2-kernel` 的 `default="serial"`，但 help 文本写「**默认 plain**」~~ | **P113 已修**：默认改`plain`（有服务器实测支撑），help 同步为实测数字（现行 `train/train.py:394`）。见§1 M2 三态核 |
| 11 | `phdnet/sparse_pc.py:29-32` 文档字符串 | 说「开关（默认关闭，默认路径逐位不变）：`cfg.sparse_conn` —— True 时主干改用本模块」 | **已过时**（写于稠密栈仍在时）。2026-09-28 稠密栈已删除，`sparse_conn` 恒 `True`，False 会 fail-fast |
| 12 | `phdnet/config.py:84-85` vs `phdnet/model.py:53-57` | config 仍保留 `backend` / `torch_dtype` 字段和「昇腾部分型号建议 float16」的说明 | `model.py` 对非 `auto/numpy/cpu` 的 backend **fail-fast**（旧 torch 栈已随 P30 删除）。字段为**兼容保留** |
| 13 | ~~`tools/bench_readout_sparse.py:113-115` 注释~~ | ~~「`sparse_alloc.py` 由另一位同事并行实现中」~~ | ✅ **已过时并已解决**：`phdnet/sparse_alloc.py` 已存在，**且 P124 已接入 `readout.py`**（`:574-579`，幂律路径 import `sparse_alloc.build_powlaw_csr`，`alpha=0` 逐位等价于均匀 k）。该注释仍在声称「预留接口 / 未就绪」→ 🔴 **活跃的误导性注释，建议回修**（与本表第 20 行是同一件事的两面） |
| 14 | `tools/audit_brain_parity.py:139-146` | A4 结论仍写「直接稀疏化读出不可接受」 | 该结论基于 256 维栈 `conn_k=8`（1.0%）。1B 档取 k=128（4.2%）后 PPL 反而**更优**（477 vs 608，4M 档）—— 两者不矛盾但**口径不同**，须标明维度 |
| 15 | `phdnet/config.py` 中 `encoder_dtype` 上方的注释块 | 🔴 **2026-10-07 已重写，但重写得只对了一半** —— **本条是活跃缺陷，不是历史问题**。旧注释（写「`bf16` 语义在读出侧已由 `readout_dtype`（**默认 bf16**）落地」）**确实已删除**；现注释（`:320-329`）开头正确声明「本字段自 P75 起**默认 fp64**」、并正确写「读出侧精度由 `readout_dtype`（**默认 fp8**，P189）决定，与本字段无关」。**残留错误在上一段**：`readout_dtype` 字段正上方 `:305` 仍写「默认 **fp32**（fhz 2026-10-01 授权改回）」，而下一行 `:318` 就是 `readout_dtype: str = "fp8"` —— **同一段注释内部自相矛盾，且紧贴它所描述的字段** | 文档采用 `fp8`（`config.py:318` + `train/train.py:324` 双重核对）。**建议回修 `config.py:305`**：删掉「默认 fp32」，改写为「默认 fp8（P189 2026-10-07）；P110 的 fp32 默认是历史口径」，并把 P110 的 fp8 保留率警告保留在案 |

---

## 10. 附：新增机制的接线检查单（摘自 `docs/PHD-Net_扩展指南.md`）

1. 新机制是否**有神经认知对应**（铁律②）？写在模块 docstring 首位。
2. 是否**改变了既有行为**？→ 必须 `config` 开关 + 默认关闭（铁律④），
   并在 `phdnet/config.py` 注明「0/False = 关闭，保持旧行为」。
3. 是否**尊重逐 token 语义**（铁律③）？不能引入批处理、不能重置跨 token 状态。
4. 是否引入**位置编码 / 自注意力 / 堆叠层**？→ 违反铁律①，否决。
5. 是否**只从源码取事实**、每个数字能指到 `文件:行号`？
6. 性能数字**不写在本仓库文档** → 只写进 `docs/PHD-Net_性能评估与迭代方案.md`
   并标明平台（x86 8 核 / 昇腾 191 核）与档位（smoke / 1b / 1b_max / 30b）。
   **x86 的性能结论不构成证据** —— x86 与昇腾常常方向相反。
7. 改动是否过了 `python tests/run_tests.py fast`（零回归门禁）？
8. 涉及稀疏 / 精度 / 后端的**能力表**是否同步更新？
   （P92 教训：加新 dtype 时漏改能力表，导致默认配置被静默回落。）
