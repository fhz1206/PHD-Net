# PHD-Net 架构设计

> **适用范围**：本文是 PHD-Net 的**最权威设计文档**——四条铁律、六机制（M1–M6）与脑同构对应、
> 逐 token 数据流、学习能力层（P194–P198）、容量账、精度体系、语料与评测口径。
> 每个关键数字标注来源：`[源码]`（文件/符号）、`[本会话实测]`（P194–P199 迭代窗口）、
> `[历史锚点]`（更早的服务器/工具实测）。核对不到写「未验证」，不做推测。
>
> **本文不写**：
> - 性能数字（ms/tok、倍数、带宽）——唯一出处是《PHD-Net_性能评估与迭代方案.md》，本文只引用不复制。
> - 优化历史与被否决方案——见 `BUGS.md`。
> - 硬件后端矩阵与昇腾踩坑——见《PHD-Net_硬件后端适配报告.md》。
> - 与 Transformer / 大脑的对比结论——见《PHD-Net_竞争力与脑同构性评估.md》。
>
> **数据截止：2026-10-17（P199）。** 与代码冲突时**一律以代码为准**。
>
> **相关文档**：`README.md`（项目入口）｜`docs/PHD-Net_现状速查.md`（一页式现状与开关默认值）｜
> `docs/PHD-Net_性能评估与迭代方案.md`（所有性能数字）｜`docs/PHD-Net_竞争力与脑同构性评估.md`（对比结论）｜
> `docs/PHD-Net_扩展指南.md`｜`docs/PHD-Net_硬件后端适配报告.md`｜`train/README.md`（训练侧开关全表）｜`BUGS.md`（缺陷台账）

---

## 0. 设计出发点

Transformer 把「关联」建模为**全局成对相似度**（QK^T），把「记忆」建模为**无学习的 KV 缓存**，
把「学习」寄托于**非局部反向传播**。PHD-Net 换一套组织原则：

| 关注点 | Transformer | PHD-Net |
|---|---|---|
| 关联 | 全局成对相似度 | **局部稀疏连接**（每神经元 k 条存在入边，CSR 承载） |
| 记忆 | 无学习的 KV 缓存 | **双记忆**（工作记忆漏衰减 + 事件驱动大空间表） |
| 学习 | 反向传播 + 梯度 | **局部规则**（Oja / STDP / 感知器），误差经 M5/T1.1 门控 |
| 时间 | 位置编码显式注入 | **状态连续演化**（无位置编码，见铁律①③） |
| 稀疏性 | 稠密矩阵 + 零权重 | **结构性稀疏**（不存在的突触不存储、不计算） |

「稀疏」在本项目里是**两层不同的东西**，全文严格区分：

1. **稀疏表示**（activations sparse）：M1 的 k-WTA 只激活 k 个单元、M3/M4b 输入的发放率
   `PHDNet._rate` 只保留 top-`n//16`。对应皮层的**稀疏发放**。
2. **稀疏连接**（synapses sparse）：M2/M3/M6 只存储并遍历**存在**的突触，M4b 突触随经验**生长**。
   对应皮层 **dense representation + sparse connectivity** 组织原则
   （`phdnet/sparse_pc.py` 模块头）。

---

## 1. 四条铁律

### 铁律① 禁止自注意力 / 位置编码 / 堆叠层

源码中不存在任何 Q/K/V 投影、位置编码模块或跨层堆叠。M2 是一棵**两层层级**
（`n_sdr → n_mid → n_top`，`SparsePCStack`），M3 在 `n_top` 上做单层侧向关联。
时序信息不来自位置编码，而来自状态本身（铁律③）。M3 STDP 的**突触迹**
（`t_pre`/`t_post`，`phdnet/plasticity.py`）是有方向的因果结构（pre 在前、post 在后），
属局部可塑性机制，不是位置编码。

### 铁律② 每个机制必须有神经认知对应

§2 每个机制都给出脑区/文献对应；源码 docstring 首位的对应声明即权威出处
（如 `phdnet/inits.py` 引 Song et al. 2005 皮层突触重尾分布、`phdnet/sparse_alloc.py`
引突触巩固/修剪的幂律倾斜）。

### 铁律③ 逐 token 语义（禁止批处理）

`PHDNet.step()` 是**单时间步**接口（`phdnet/model.py`）。WM 槽位强度、STDP 迹、LTM 突触、
调制器 Welford 统计量、`step_count`、读出 W 全部跨 token 连续演化，读出 W 每步原地更新。

- 训练侧状态**永不重置**：`train/train.py` 打印
  `"[context] state never reset (WM/STDP/LTM carried continuously across samples/shards/epochs)"`；
  跨 epoch 也不重置，只有 epoch 首步的 prev 链接被切断（`p2 = None`）。
- 主循环逐 token 调 `net.step(x, target=tgt, learn=trainable, target_idx=_i0)`；
  语料流是 `StreamingTokenizer`（逐 token 训练，语料任意长）。
- 评估路径用 `readonly=True` **冻结所有持久状态**（`model.py` step 的 readonly 分支），
  保证重复评估一致。
- **P194 例外（有边界）**：`micro_batch>1` 时读出梯度累积 B 步后一次提交
  （`commit_grad`），但**主干（PC/STDP/LTM）与状态推进仍按 token 严格保序**——
  脑同构对应「微小兴奋性神经元簇并行放电 + 突触时间整合」，不改变逐 token 时序语义
  （`model.py` P194 (Q1) 块）。另一例外 `minibatch_size`（B7）同理，默认 1 = 关闭。

### 铁律④ 行为变更用 config 开关承载

改变既有行为的开关进 `phdnet/config.py` 并注明「0/False = 旧行为」。**现行默认开启的例外**
（均有实测依据，[源码] `phdnet/config.py` + `train/config_1b.py`）：

| 项 | 值 | 依据 |
|---|---|---|
| `k_sparse` | 16（库）/ `max(8, w//8)`（1b 档 128） | k=16 PPL −1.43% 优于 32（4,000 字符口径） |
| `csr_online`（CLI） | **开** | 在线可写 CSR ~16 B/突触 vs dict ~100+ B，1b 档实测省 5.74×；库字段默认 False，由 CLI 打开 |
| `readout_conn_k`（CLI） | 128（4.2% 连接率） | 4M 档 A/B：k=128 PPL 477 vs 稠密 608 且速度持平；库默认 0（稠密） |
| `homeostasis`（1b 档） | True | P194：稳态缩放防表征发散（Turrigiano 2008） |
| **P197 四开关** | `sdr_novelty` / `task_modulation` / `error_gated_memory` / `plateau_sleep` / `readout_replay` = **True** | 缺失能力修复（会话中学习），见 §3.3 |
| `step_profiling`（CLI） | 开（P198 起） | 分段耗时是观测前提，开销 ~0.2 µs/段 |

---

## 2. 六机制 M1–M6

### 2.0 编排顺序与数据流（`PHDNet.step`，`phdnet/model.py`）

```
            ┌──────────────────── 逐 token（状态永不重置）────────────────────┐
            │                                                                │
 语料流 ──► 分词（贪心最长匹配，OOV 跳过不断流）                              │
            │                                                                │
            ▼                                                                │
      ┌──────────┐   x = [cur SDR ; prev SDR]（2×n_sdr 组合输入）            │
      │ M1 稀疏编码│  k-WTA → s0（k_sparse 个活跃单元）                        │
      └────┬─────┘                                                          │
           ▼                                                                │
      ┌──────────┐   CSR 前向 + n_infer_steps 步自由能迭代                    │
      │ M2 预测编码│  r1 = tanh(up0@s0); r2 = tanh(up1@r1)                    │
      │ (SparsePCStack)  e0/e1 = 重构误差（surprise 的来源）                  │
      └────┬─────┘                                                          │
           ▼                                                                │
      rate = _rate(r2)（半波整流 + k-WTA，~6% 激活）                         │
           │                                                                │
     ┌─────┼──────────────┬───────────────┐                                 │
     ▼     ▼              ▼               ▼                                 │
 ┌──────┐ ┌──────────┐ ┌────────────┐ ┌─────────┐                          │
 │M3 STDP│ │M5 调制    │ │M4a 工作记忆 │ │M4b LTM   │                         │
 │predict│ │surprise→z │ │decay→write  │ │imprint / │◄── 新颖性覆盖（P197）   │
 │(prev) │ │gate/mode  │ │(task gate)  │ │recall    │                         │
 └──┬───┘ └────┬─────┘ └─────┬──────┘ └────┬─────┘                          │
    │          │             │             │                                 │
    ▼          ▼             ▼             ▼                                 │
      h = [r2 ; WM] (+ STDP 预测, pred_in_readout) (+ 检索线索, 可选)          │
           │                                                                │
           ▼                                                                │
      ┌──────────┐   三臂：稠密码本 / CSR 稀疏 / 加速器（NPU 等）             │
      │ M6 读出   │  softmax 感知器：ΔW = −η·(p−t)⊗h（仅末端）                │
      └────┬─────┘  master 模式：更新落 fp32 master，影子只做前向（P195）      │
           ▼                                                                │
      下一 token 预测（top-k / nucleus 截断，P198）──────────────────────────┘
           │
           ▼
      局部学习：PC（误差驱动 + Oja，×调制门 ×发育窗口）/ STDP（×NE 增益）/
      读出（micro_batch 累积或逐步）
```

执行顺序（`model.py::step`）：M1 编码 → M2 推理 → rate/fused → M3 预测（诊断 + 读出特征）→
surprise/novelty → M5（新颖性可覆盖 mode/gate）→ M4a decay+write → M4b imprint/recall →
M6 前向/学习 → 主干局部学习（PC × `pc_dev_gate` × `mod_scale`；STDP × `ne_gain`）。

**读出特征拼装**：`h = [r2 ; WM]`；`pred_in_readout=True` 时三拼 `[r2 ; WM ; STDP 时序预测]`；
`retrieval_topk>0` 时再拼召回线索。1B 档 `n_h = 3 × 1024 = 3072`。

---

### 2.1 M1 稀疏分布式编码（k-WTA）

**脑同构**：初级感觉皮层的稀疏放电与抑制性侧抑制（`phdnet/sparse_encoder.py`）。
皮层用大量神经元各自的少量活动承载信息，而不是靠大激活值。

**实现**（`SparseEncoder.encode`）：`u = W_enc·x + b` → `argpartition(-u, k-1)[:k]`
（k-WTA 竞争 = 侧抑制的抽象）→ 胜者归一到 `[0.1, 1.1]`。

**关键参数**：

- `k_sparse` 库默认 16 / `n_sdr` 256 → 6.25%；1B 档 `max(8, w//8)` = 128/1024 → **12.5%**。
- `encoder_dtype`：库默认 `fp64`，**生产 CLI 默认 `fp32`**（x86 上 fp32 快 1.80×；
  迭代量恒 fp32——P61 口径）。bf16/fp16 存储 + fp32 迭代在 numpy/BLAS 路径必付上采样转换，更慢。
- W 初始化为固定随机投影（`normal(0, 1/√n_input)`），默认冻结；可选 `learnable_encoder`
  （Foldiak 局部规则，实测有害 +6.1% → 保持关闭）。
- **GEMV 平台自适应**：`_prefer_blas_gemv` 运行时探测（P165）——BLAS 快走 BLAS，
  BLAS 病态慢（曾见于 aarch64 某构建）走自写 numba 行核 `_gemv_rows_ilp`（行内 4 路 ILP，1.25×）。
  aarch64 numba 不支持 float16 数组（P107b 教训，核内保持 fp32/fp64）。
- 输入维度契约：`step(x)` 的 `x` 是 `cfg.n_input`（词级 = 2×n_sdr），`target` 是 `net.n_out`；
  `k_sparse > n_sdr` 在 `__post_init__` 直接 ValueError。

**验证状态**：`gate_encoder_gemv.py`（GEMV 路径）；编码器 fp32 迭代经全部门禁回归。

---

### 2.2 M2 预测编码主干（结构性稀疏 CSR，唯一实现）

**脑同构**：皮层预测编码层级（Rao & Ballard 1999）+ **dense representation + sparse
connectivity**——单个锥体神经元 ~10³–10⁴ 突触 / 潜在靶点 ~10¹¹ → 连接率 ~10⁻⁵（长程）～
10%（局部）。稠密栈已于 2026-09-28 删除；稀疏性开关本身也已删除（P193）——
**稀疏是唯一形态，不存在「关掉稀疏」的选项**。

**结构**（`SparsePCStack`，`phdnet/sparse_pc.py`）：四个 CSR 权重

| 权重 | 方向 | 形状（n0→n1→n2 = n_sdr→n_mid→n_top） | 初值 |
|---|---|---|---|
| `up0` | 上行识别 | n1 × n0，每行 k0 条 | `normal(0, 1/√k0)` |
| `up1` | 上行识别 | n2 × n1，每行 k1 条 | `normal(0, 1/√k1)` |
| `dn0` | 下行生成 | n0 × n1 | `up0` 转置 × 0.5 |
| `dn1` | 下行生成 | n1 × n2 | `up1` 转置 × 0.5 |

**关键参数与事实**：

- **连接率**：`conn_k=0` 时 `k = n_in//8` ≈ 12.5%；1B 档 `conn_k=128` / width 1024 = **12.5%**。
- **fan-in 归一化是硬要求**：权重尺度必须 `1/√k` 而非 `1/√n_in`，否则前向幅值偏低
  `√(k/n_in)` 倍，经 tanh 后表征趋 0（短训练口径掩盖、全语料口径 +37.6% 劣化）。
- **推理**（`infer`，步长/迭代数硬编码）：`r1 = tanh(up0@s0)`；`r2 = tanh(up1@r1)`；
  迭代 `n_infer_steps`（默认 1）步自由能下降：`e1 = r1 − dn1@r2`；`d2 = clip(up1@e1, ±0.5)`；
  `r2 = tanh(r2 + 0.15·d2)`；`e0 = s0 − dn0@r1`；`d1 = clip(up0@e0, ±0.5)`；
  `r1 = tanh(r1 + 0.15·d1)`。
- **三态核选择**（`pc_fused_kernel`）：`True/"fused"`（prange 融合核）/`"serial"`（单核融合）/`False`
  = plain（原始 5 核调用，**有行级 prange**）。不能 `bool()` 三态值。**生产 CLI 默认 `plain`**
  （P113 服务器实测：M2_infer 6.92→1.28 ms/tok，端到端 17.63→12.09，四个采样点 PPL 逐位相同
  ——纯调度变化）。对拍 `tests/gates/gate_m2_infer_kernels.py`（三态等价 + n_steps 1/2/3）。
- **学习**：下行 `ΔW += η_pc·e⊗r`（误差驱动稀疏外积）；上行 Oja
  `ΔW += η_oja·post·(pre − post·W)`；四权重 clip 到 `pc_w_max`（默认 2.0）。
  **η_oja > 0 已知危险**（P198 正反馈发散，见 §3.4）；`η_pc` 由**发育窗口**控制（§3.1）。
- **精度**：val fp32（P173）、融合核中间量 fp32（P176）——「fp32 存储 + fp32 累加」统一。
- **可选机制**（默认关闭）：稳态突触缩放 `homeostasis`（**1b 档开**）、逐神经元目标发放率
  `neuron_target_rate`、时序预测目标 `pc_predictive_target`、皮层式重尾初始化 `lognormal_init`。
- **可验证等价性**：`k = n_in`（全连接图）时与稠密栈数值一致（对拍判据
  `gate_parallel_consistency.py`）。`stats()` 返回 `synapses/dense_equivalent/connectivity/k0/k1`。

**验证状态**：`gate_m2_infer_kernels` / `gate_parallel_consistency` / `gate_m2_learn_fused` /
`gate_cython_kernels`（Cython 臂容差等价，41/41 历史例数）。

---

### 2.3 M3 时序关联核（STDP）

**脑同构**：海马 CA3 递归网络与皮层局部突触的 STDP（`phdnet/plasticity.py`；
Markram et al. 1997 / Bi & Poo 1998）。

**实现**（`STDPCore`）：每神经元 `m_lateral=16` 条随机出边，`W` 只存拓扑内权重（fp32，P173）。
`step` 顺序是正确性关键：

```
t_pre ← λ·t_pre + pre_rate            # 1. 先吸收 pre 迹
raw = 2·t_pre[i]·post_rate[k] − t_post[k]·pre_rate[i]   # 用更新前的 post 迹算 LTD
W[i] += η·raw（clip 到 [0, w_max]）    # 2. 更新活跃边
t_post ← λ·t_post + post_rate          # 3. 最后才吸收 post 迹
```

否则 LTP 与 LTD 中的当前共激活项相消，学习失效。学习率 `eta_stdp` 库默认 0.03 /
1B 档 0.02；`lambda_trace = 0.35`（时间窗宽度）；`w_max = 1.0`（只作用于 STDP 核）。
η 缩放语义已修正（P192 审计）：LTP/LTD 两项**同乘** `eta·eta_scale`，与 numpy 参考公式一致
——属学习规则语义变更，历史 PPL 锚点因此不可比（归因记录见性能文档 §10.1）。

**计算量**：`O(活跃 × m)` 而非 `O(N×m)`——只遍历 `pre_rate > 0` 的行。
**稀疏化是 M3 正确工作的前提**：`PHDNet._rate` 把 tanh 表示半波整流后 k-WTA
（`k = max(1, len(r2)//16)`，~6% 激活），低激活率保证支撑集区分度。

**平台约束（硬约束）**：M3 只走 numba/numpy CPU（P30 定稿）。显式指定 torch 后端 fail-fast——
STDP 的逐突触状态（自适应 LR / 稳态 / 元可塑性 / E-I）在 torch 上无法保留。numba 分派条件：
`adaptive/homeostasis/metaplasticity/ei_synapses` 任一开启即强制 numpy 路径。
可选机制（全部默认关闭）：`adaptive_lr`（逐突触二阶矩归一）、`dual_trace`（长窗
`lambda_trace_slow=0.85` / 权重 0.5）、`stdp_homeostasis`（只 downscale 不注能）、
`metaplasticity`（BCM 滑动阈值）、`ei_synapses`（E/I 比 0.2，抑制出边负权重、clip 到 [−w_max,0]）。

**`predict` 的作用**：`pred = W·post_idx·prev_rate` 是「上一时刻发放率 → 当前发放率」的
时序前瞻，既是诊断（`seq_err = 1 − cos(pred, rate)`），也在 `pred_in_readout=True` 时
作为 M6 第三路特征。训练路径下 `_prev_rate is _last_rate`（同一对象别名），复用 `pred`
避免重复计算；readonly 时两者确实不同，保留原调用。

**验证状态**：`gate_stdp_parity.py`（numba/numpy 对拍）、`gate_ltm_trace_semantics.py`
（32/32，P193 起全绿——LTD 迹 `_decay_read` 补偿修复）、`gate_wm_modulator.py`。

---

### 2.4 M4a 工作记忆（PFC 持续放电 + 基底核门控）

**脑同构**：背外侧前额叶持续放电与基底核门控写入（Miller & Cohen 2001）。

**实现**（`WorkingMemory`，`phdnet/wm.py`；slots fp32）：

- `n_wm_slots = 4`（有限容量 ≈7±2 的抽象）；漏衰减 `gamma_wm = 0.85`；摘要槽不衰减。
- 门控写入 `write(r, gate, thresh)`：`gate ≥ gate_thresh`（0.35）才写；
  写入目标默认「最弱槽位」（排除摘要槽）。**P197 起**：`error_gated_memory=True`（默认）时
  写入门改用**滞后一步的任务误差门** `self._task_gate`（T1.3）——监督信号从读出头广播到记忆写入。
- 可选内容寻址（`wm_content_address`，默认关）：写入目标改为最相似槽位（阈值 0.6）。
- 可选压缩摘要槽（`wm_summary_every`，默认 0）：每 N 步把 WM 内容经 LTM 吸引子压缩进免衰减槽。
- 读出 `read()`：强度加权平均——M6 特征第二路。

**验证状态**：`gate_wm_modulator.py`；P197 门禁覆盖写入路径（B 组印迹/熟悉化用例）。

---

### 2.5 M4b 长期记忆（海马快印迹 + 大空间事件驱动表，1B/30B 容量主体）

M4b 由 `big_ltm` 切换：`False` = 稠密双速率 LTM（`phdnet/ltm.py`，W_fast/W_slow + Hopfield
吸引子 recall）；`True` = **生产路径**的大空间事件驱动稀疏表。

**脑同构**：海马/皮层印迹——仅被经验触碰过的突触存在；单神经元突触数有硬上限
（`tools/eval/audit_brain_parity.py` A2）。

三层结构：

| 层 | 文件 | 职责 |
|---|---|---|
| 适配器 | `phdnet/bigltm.py`（`SparseLTM`） | n_dim 维稀疏率 → 确定性哈希 → 大空间索引；召回反投影回 n_dim |
| 存储本体 | `phdnet/sparse_table.py`（`SparseSynapseTable` dict 版 / `OnlineCSRTable` 在线可写 CSR） | 邻接存储 + STDP 式生长 |
| 稀疏内核 | `phdnet/ltm_kernel.py`（numba） | `predict_arr`（gather，44.6×）、`_recall_project*`（反投影，21.9×） |

**哈希分布式编码**：每维度 `j` 经 `U64(j)*7919 + arange(k_hash)*2654435761` 再 `mix64`
取模 `n_neurons`，得 `big_ltm_k`（默认 4）个大空间索引。反投影索引 `rev` 一次性构建、
零维护成本预 CSR 化（30b 档走稀疏键 + 开放寻址哈希表，稠密 indptr 在 2²⁹ 时要 4 GB——P166）。

**A2 契约**（`_check_sparse`）：只接受**稀疏率**（激活维度 ≤ n_dim/4）；±1 稠密模式会
一次性激活全部维度 → 显式拦截。故 `model.py` 分派：大容量表传 `rate`，稠密 LTM 传 `sign(rate)`。

**事件驱动**（`model.py` M4b 块，条件触发，不是每步）：

- **印迹**：`learn and mode == "encode" and mem_gate > ltm_imprint_gate`。
  `ltm_imprint_gate` 现默认 **0.3**（P197；旧 0.8 使 big_ltm 在 LM 训练中几乎从不印迹）。
  每次 `imprint` 学的是 `table.learn(self._prev, cur)`——**相邻两次稀疏率模式的关联**
  （"fire together, wire together"）。`amortize>1` 已移除（P122 审计：收益 0 + 权重污染，
  传 >1 直接 ValueError）。`begin_episode()` 显式切断跨情节 `_prev` 指针。
- **检索**：`mode == "retrieve" and step_count % 8 == 0`（multi_modulation 时
  `max(2, round(8·(0.5+ht)))`）；可选 `error_triggered_retrieval`（默认关）。
  召回向量以 `gate=0.5, thresh=0.0` **无条件写入 WM**。两者互斥（if/elif）。

**两种存储后端**（`csr_online` 切换；**CLI 默认开**）：

| | `SparseSynapseTable`（dict 邻接） | `OnlineCSRTable`（在线可写 CSR） |
|---|---|---|
| 存储 | `out: dict[int, dict[int, float]]` | 每行定长数组 + 预留槽，满 `row_cap` 后 2× 增长、上限 `m_out` |
| 每条目内存 | ~100+ B | ~16 B（int8 权重 + CSR 索引；1b 档实测 91.8→16.0 B/条目，5.74× 省） |
| 惰性迹衰减 | `trace[i] = trace[i]·λ^Δt + 1.0`，pre/post **各自独立时间戳** | 同；LTD 读侧走 `_decay_read` 补偿（P193 修复） |
| 新突触生长 | 只由 LTP 驱动（相邻对约定），LTD 仅作用于已存在突触 | 同 |
| 逐位等价 | — | `gate_ltm_csr_equivalence.py` |
| 分片落盘 | — | `export_shards/import_shards`（按行哈希分片 npz） |

**诚实边界**：`consolidate` 在容量表上无稠密快/慢权重分离，退化为对已生长突触的
乘性衰减 `f = 1 − 0.01·forget`；int8 量化码值域 10~32 时 1% 衰减落在量化台阶内
→ **consolidate 零效果**，需 `forget ≳ 3`（量化精度固有属性）。
可选：`sparse_table_prune`（淘汰最旧 25% 迹）、`growth_guidance`（生长偏好低入度目标）、
`sparse_int8`（int8 量化存储）、`staged_sleep`（SWS 巩固+downscale / REM 回放，默认关）。

**验证状态**：`gate_ltm_csr_equivalence` / `gate_ltm_kernels`（recall 全链路逐位）/
`gate_ltm_learn_batch` / `gate_ltm_trace_semantics`（32/32）/ `gate_ltm_reverse_index`。
**P197 会话中学习验证**见 §3.3。

---

### 2.6 M5 神经调制

**脑同构**：蓝斑 NE / 多巴胺的意外信号 + 胆碱能编码-检索模式切换（Hasselmo；Yu & Dayan）。

**单通道版**（默认，`Neuromodulator`）：

```
surprise = ‖e0‖/√len(e0)   （M2 底层预测误差范数）
z = Welford 在线 z 标准化（2026-10-07 修复：用更新前统计量算 z）
gate = σ(mod_gain · z)      （mod_gain = 1.5）
mode = "encode" if z > 0.3 else "retrieve"
```

**P197 新颖性覆盖**：`sdr_novelty=True`（默认）时，顶层 SDR 支撑集（`rate>0` 的下标集合）
经顺序无关哈希查 `_seen_sdr` 表（LRU，容量 65536）；**新模式 → 强制 `mode="encode"` 且
`gate ≥ novelty_gate_floor(0.85)`**（放行印迹门控）。动机（本会话实测）：surprise 对输入
幅度盲视（M1 k-WTA + min-max 归一化抹掉幅度，2σ→0.164 / 10σ→0.167），旧门控下印迹触发率
仅 0.67%。只读路径用 `_peek_sdr_novelty`（不登记，评估不污染训练期集合）。

**多通道版**（`multi_modulation=True` 启用，默认关）：ACh（编码门控）/ NE（新颖性→可塑性，
乘进 STDP `ne_gain`）/ DA（巩固→睡眠回放强度）/ 5-HT（耐心→检索节奏）；
`gate = ach·(0.5 + 0.5·ne)`，接口与单通道兼容。

**另一条独立通路 T1.1/T1.3**（P197 起默认开）：读出 NLL 经 Welford z 标准化
→ `task_modulation` 门控本步**主干**学习率（`task_scale`，下限 `task_scale_floor=0.3`）；
滞后一步的 `_task_gate` 供 `error_gated_memory` 改变 WM 写入与 LTM 印迹判据。
**监督信号不再只到读出头**——「学什么」的缺失已补。

**验证状态**：`gate_insession_learning.py`（A 组新颖性检测/只读不污染/LRU 淘汰；
D 组默认值契约）。

---

### 2.7 M6 读出头（稀疏感知器 + softmax，三臂）

**脑同构**：IT → 前额叶/前运动皮层的决策读出（`phdnet/readout.py`）。
**监督信号仅存在于最末端**（局部感知器规则，误差不反传主干）。

**三条连接结构路径**（`Readout.__init__` / `pick_readout_backend`）：

| 路径 | 触发 | 存储 | 前向 | 学习 |
|---|---|---|---|---|
| **稠密** | `conn_k == 0` 且 `hidden == 0` | `(n_out, n_in)` fp32 或量化码本（RO_DTYPES：fp32/fp16/bf16/int8/int16/int32） | BLAS sgemv / LUT matvec | 融合核 / 量化核 / numpy 回退 |
| **结构性稀疏** | `conn_k > 0` | CSR `(indptr, idx, val)`，每行 k 条存在入边，**固定 fp32** | `_csr_matvec`（只遍历存在的边） | `_csr_add_outer`（只更新存在的边） |
| **加速器** | `accel_readout != cpu` 且设备可用 | 设备常驻 W（fp32/fp16/bf16/int8/fp8 位模式） | gather-GEMV（P111，`einsum` 默认） | 设备 rank-1；P195 master |

- **库默认 `readout_conn_k = 0`（稠密）**；**生产 CLI 默认 128** → 1B 档 128/3072 = **4.2%**
  连接率（P108；4M 档 A/B：k=128 PPL 477 vs 稠密 608）。两者不一致是事实（`config_1b.py`
  透传显式参数）。
- 稀疏模式仅支持**均匀 k**；**幂律异质连接已接入**（`--readout-powlaw-alpha`，`alpha=0`
  逐位走均匀路径）：行宽 `k_i ∝ counts_i^alpha`（最大余数法 + 水位迭代，总 nnz 恒等于预算，
  每行 ≥1 条入边，列索引行内升序无重复；`tests/gates/gate_readout_powlaw.py` 32 例）。
  ⚠ **词频目前是代理值**：构造期语料尚未流过，`_powlaw_proxy_counts` 返回 `1/rank`，
  而词表是字典序非频次序 → 倾斜方向可能相反，故默认 `alpha=0`；开启前必须 A/B 验证代理
  与真实频次的相关性。**加速器对非均匀行宽 fail-fast**（不同算法，不「能跑但语义不同」）——
  开 `alpha>0` 回落 numba 读出，是正确行为不是 bug。
- **两级群体读出**（`readout_hidden>0`，默认关）：W1（h→隐藏群体，稀疏投射 + k-WTA 竞争 +
  局部 Oja）→ W2（群体→输出，监督）。fan-in 补偿 ×√(fan_in/k)。
- **学习规则**：感知器（`readout_softmax=False` 库默认）或 **softmax 感知器**
  （词级 LM 强制开启）：`ΔW = −η·(p − t)⊗h`，返回 `nll = −log p(correct)`；
  softmax/NLL 一律 fp64 主回路。`eta_readout = 0.15`（实测最优区间 0.15–0.20，0.5 发散）。
- 可选：`eta_readout_anneal/floor`（乘性退火）、`readout_w_clip`、`minibatch_size`（B7）、
  `lognormal_init`、`readout_recurrence/segment_check`（O5 再入，默认关，明确不是位置编码）。

**验证状态**：`gate_readout_sparse` / `gate_readout_sparse_structure` / `gate_readout_powlaw` /
`gate_accel_sparse_readout`（21 例）/ `gate_readout_fused` / `gate_readout_master_weight`（§3.2）。

---

## 3. 学习能力层（P194–P198，本会话主线）

> P192–P199 迭代的主命题：让「逐 token 局部学习」真正学得动。
> 诊断结论：主干不学（eta_pc=0 时代）+ 印迹门控空转（0.67%）+ 低精度吞更新（P110）
> + 上行 Oja 发散（P198）。P194–P198 逐条对症。

### 3.1 P194 学习能力四件套

| 件 | 开关（默认） | 机制 | 脑对应 |
|---|---|---|---|
| Q1 | `micro_batch`（CLI 默认 1） | 读出梯度 B 步累积，`commit_grad` 一次提交；**主干仍逐 token**。语义 = **梯度和**（等效 n 个串行在线步 × η），不是均值——首版 `eta/n` 实测 PPL 随 B 单调恶化，已修正 | 微小神经元簇并行放电 + 突触时间整合 |
| Q2 | `pc_dev_steps`（库默认 0；**1b 档 20000**） | 表征发育余弦窗口：前 N 步 `eta_pc` 按 `1−cos(π·t/N)/2` 从 0.02 退火到 0，之后冻结。**取代旧的「eta_pc 恒 0 表征冻结」**——历史冻结是没有发育窗口时的无奈之举 | 发育临界期（表征先塑、后固） |
| Q3 | `readout_conn_k` / `readout_powlaw_alpha` | 读出结构性稀疏（128 = 4.2%）+ 幂律异质连接（默认关，见 §2.7） | 皮层→输出投射稀疏；突触巩固/修剪 |
| Q4 | `readout_dtype`（**默认 fp32**） | 精确 `p−t` 语义。P110 反证在案：fp16 非目标行更新保留率仅 26.67% → 退化为纯 Hebbian | — |

1B 档 `build_cfg` 生产组合：`eta_pc=0.02, pc_dev_steps=20000, eta_stdp=0.02,
readout_dtype="fp32", homeostasis=True, big_ltm=True, conn_k=128, pred_in_readout=True`。

### 3.2 P195 主权重模式（DeepSeek-V4-Flash 式）

fhz 指令「模仿 DeepSeek-V4-Flash：保持 fp8 的精度和 fp16 的迭代精度」。
§6.1 的失败模式不是 fp8 本身的问题，而是**没有高精度累加器**：

| 组件 | 精度 | 作用 |
|---|---|---|
| **master weight** | **fp32**（常驻） | 所有更新作用于此 → 非目标行更新 **100% 保留**（fp8 直存口径为 26.67%→更低） |
| **影子副本** | fp8/bf16/fp16 位模式 | 只用于前向 GEMV → 保留低精度访存收益 |
| 重量化 | 每 `shadow_refresh`（默认 8）步 | 摊薄量化成本；前向有 ≤N 步 staleness |

- **两条臂行为一致**（GAP-4 修复）：`AccelReadout` 与 numba `Readout` 均实现 master
  （原参数只接线到加速臂，`--accel cpu` 下 silently 失效）。微批量 `commit_grad` 同样走 master。
- **联动约束**：master 模式下 **torch.compile 融合核被禁用**（图按 `self.W` 的 dtype 固化，
  掺 master 会静默「更新影子、master 不变」）；**NPU Graph 的重量化移出图外**（图内只更新
  master 内存，replay 后 `_graph_master_sync` 推进影子簿记）。
- **ckpt 存 master 逐位**：`ro_W` 跳过 `ckpt_dtype` 压缩，fp32 精确值往返（否则 §6.1 的
  收益归零）；`W_authoritative()` 给出无歧义权威视图（RL KL 参考/统计必须用它）。
- **本机 A/B**（`tests/checks_p195.py --ab`，固定探针集，[本会话实测]）：accel fp8 无 master
  PPL **2.84**（+33.5% vs fp32 基线）→ **fp8 + master(r=8) 2.15**（−24.2%，vs fp32 基线 +1.1%）；
  numba int8 + master 2.15 同结论。`r=4→2.15 / r=16→2.20`。
- **代价**：多一份 fp32 master（1b 档 866.7 MiB）→ **默认 off**（`--readout-master-weight`
  显式开启；`readout_dtype=fp32` 时开启无收益，启动打印提示）。
- **门禁**：`gate_readout_master_weight.py`（A 组 master/影子不变量、B 组全 dtype 分派
  BugC 回归、C 组 ckpt 往返逐位、D 组 compile/graph/micro_batch 联动、E 组默认 off 逐位）。

### 3.3 P197 会话中学习（印迹从 0.67% → ~100%）

诊断（本会话实测）：1B 预设下 LTM 印迹触发率 **0.67%**（600 步只 4 次），1B 记忆栈几乎
完全空转；根因是 surprise 幅度盲视 + 印迹门 0.8 过高。四项修复（全部默认开，铁律④例外）：

| 开关（默认 True） | 机制 |
|---|---|
| `sdr_novelty` | SDR 支撑集新颖性检测（海马**模式分离**的工程等价）：新顶层激活模式 → 强制 encode + gate≥0.85 放行印迹；表容量 `sdr_novelty_cap=65536`（LRU） |
| `task_modulation` / `error_gated_memory` | 误差广播：读出 NLL z 分数门控主干学习率；任务误差门（滞后一步）接管 WM 写入 / LTM 印迹判据 |
| `plateau_sleep` | 验证 PPL 平台期（连续 `plateau_patience=2` 段无改善）→ 触发 `sleep()` 巩固——睡眠时机科学化 |
| `readout_replay` | 储备旧 (输入,目标) 对（`replay_buffer=256`），每 `replay_every=4` 步在**当前表征**下只读前向重训读出（`replay_eta=0.5`）——缓解「表征漂移 × 恒定读出率」的灾难性遗忘 |

门控值：`ltm_imprint_gate` **0.8 → 0.3**（写入由「真新颖」驱动而非「意外度幅值」）。

**验证状态**（`gate_insession_learning.py` + [本会话实测]）：

- 印迹触发率 **0.67% → ~100%**（smoke 档 200 步随机输入；门禁判据 >30%；重复刺激
  熟悉化后回落——表容量限制，非门控问题）。
- cc_traces 3,000 token：PPL **2474 → 1308**、LTM 生长 **+81%**。
- 读出头单对记忆：nll 降 >50%；不泛化到未见输入（防过拟合假象）。
- `sdr_novelty=False` 逐位回旧行为（D2 关闭契约）。

### 3.4 P198 发散防线 + 生成截断 + 词表预算

- **`eta_oja > 0` 生产规模正反馈发散**（[本会话实测]，2026-10-17）：
  上行识别权重改动 `pc.infer` 前向 → r1/r2 喂 STDP + 读出 → 反馈环。
  `eta_oja=0.02` → 3000 步 probe_PPL **9.1e11**（logits 动态范围 55.9，softmax 后
  非最大项全下溢，nll=27.6）；`eta_oja=0` → **916**。与 `homeostasis` 无关（关掉也发散），
  与 `eta_pc` 无关（0.005 也差）→ **下行（eta_pc）安全、上行（eta_oja）危险，二者分开决策**。
  处置：库默认 0.0；`__post_init__` 对 >0 发 **RuntimeWarning**（不 raise——小规模/短跑的
  机制测试合法地用它）；`build_cfg` 显式 0.0。
- **表征发育窗口实测收益**（`eta_pc=0.02, eta_oja=0`，40K token / 3,392 词表）：
  probe_PPL 3403（随机）→ 1K 步 1512 → 5K 步 620 → 30K 步 **389**（≈11×）；
  bigram 合法率 **78%**（旧配置 5–13%）、distinct 76.5%。
- **`generate()` top-k / nucleus 截断**（`phdnet/word_lm.py::generate`，
  `topk`/`topp` 参数；`train/infer.py::sample_next` 同款 top-k）：缺它时低质量分布被温度
  放大成「长尾乱码」（τ=0.8 实测 bigram 合法率 6%）。截断是生成可读性的**必要**条件。
  评估工具 `tools/eval/eval_generation.py`（bigram 合法率 + distinct）。
  门禁 `gate_generate_truncation.py`。
- **词表预算工具**（`tools/diag_vocab_budget.py`）：训练前估算
  `exposure = tokens / vocab`（每词平均曝光）。经验阈值（本会话实测，非理论推导）：
  词表 20,913 + 100K token → PPL 3322；词表 3,392 + 100K token → 1310（目标空间缩小 6×
  → PPL 降 61%）。exposure < 30 严重不足 / 30–100 勉强 / ≥100 充足。
  **1B 档 PPL 高的根因即词表 × 曝光**（见 §10 诚实边界）。

### 3.5 P196 数据扩展：cc_traces 会话数据链路

`--data cc_traces`：ModelScope `choucisan/mimo-claude-code-traces-1k`
（1,017 traces / 10 类 / 40.5 MB / MIT；event-stream JSONL，session/<类别>/*.jsonl）。
本地走转换 parquet；远程零落盘直读 `.jsonl`（`phdnet/corpus.py` P196 新增分支）；
保留 thinking 块；`lang` 列恒 en。门禁 `gate_cc_traces.py`（A 本地 fixture / B 远程 /
C 流式 / D 幂等 / E train.py 接线）。

---

## 4. 状态跨什么连续

| 状态 | 载体 | 跨 token | 跨 shard | 跨 epoch |
|---|---|---|---|---|
| WM 槽位内容/强度 | `wm.slots` / `wm.strength` | ✓ | ✓ | ✓ |
| STDP 突触迹 / 权重 | `t_pre`/`t_post` / `stdp.W`（每步原地更新） | ✓ | ✓ | ✓ |
| LTM 印迹 | `table` 突触 | ✓ | ✓ | ✓ |
| LTM 模式指针 | `ltm._prev` | ✓ | ✓ | ✓（`begin_episode()` 显式切断） |
| 调制器统计量 / 任务门 | `mu/m2/count` / `_task_gate` | ✓ | ✓ | ✓ |
| 读出 W（含 master） | `readout.W` / `_W_master` | ✓ | ✓ | ✓ |
| 新颖性表 | `_seen_sdr`（LRU） | ✓ | ✓ | ✓ |
| 发育经验 / `step_count` | `net._exp` / `net.step_count` | ✓ | ✓ | ✓（参与检索节奏判定） |
| 回放储备库 | `_replay_buf` | ✓ | ✓ | ✓ |

**唯一被切断的**：`p2`（前词链接）在 epoch 首步断开。

---

## 5. 容量账

**口径**：`train/config_1b.py::capacity_report`（纯算术，不构建网络，与
`phdnet/model.py::count_params` 同口径：大容量表计入「容量上限」，`count_params`
计入「已生长突触数」——训练初期利用率低，随训练趋近上限）。

### 5.1 1B 档主体

```
big_ltm 容量 = big_ltm_N × big_ltm_m = 2^24 × 72 = 1,207,959,552
```

突触不是构建即存在，而是随经验生长（事件驱动）；每步计算量只正比于**活跃神经元数**，
与总容量无关——容量与算力解耦。

### 5.2 四档预设对比（`PRESETS`，V = 51,962 词表，readout_conn_k = 128）

| 项 | `smoke` | `1b` | `1b_max` | `30b` |
|---|---|---|---|---|
| width（n_sdr=n_mid=n_top） | 256 | **1024** | 4096 | 1024 |
| k_sparse / conn_k | 32/32 | **128/128** | 512/512 | 128/128 |
| big_ltm_N × m | 2²⁰×60 | **2²⁴×72** | 2²⁴×60 | **2²⁹×56** |
| M1 编码器 | 131,328 | 2,098,176 | 33,558,528 | 2,098,176 |
| M2 主干 CSR | 32,768 | 524,288 | 8,388,608 | 524,288 |
| M3 STDP | 4,096 | 16,384 | 65,536 | 16,384 |
| M6 读出（k=128） | 6,651,136 | 6,651,136 | 6,651,136 | 6,651,136 |
| 固定突触小计 | 6,819,328 | **9,289,984** | 48,663,808 | 9,289,984 |
| M4b 容量 | 62,914,560 | **1,207,959,552** | 1,006,632,960 | 30,064,771,072 |
| **总容量 ≈ ×10⁹** | 0.0697 | **1.2172** | 1.0553 | **30.0741** |
| M4b 占比 | 90.2% | 99.2% | 95.4% | 99.97% |
| 索引映射内存 | 33.6 MB | 536.9 MB | 536.9 MB | 17,179.9 MB |

**词表口径必须标明**：满词表 73,958 时 1B 档总容量 1.2201e9；稠密读出（conn_k=0）时
1.3702e9（V=51,962）——三个数都对，区别只在词表与稀疏档位。
⚠ `capacity_report` 的内存估算注释仍写「CSR = int32 + fp64」，实际 val 已是 fp32（P173）——
静态内存为**上界估计**，以启动时打印与 `[big-ltm]` 行为准。

### 5.3 30B 档诚实边界

① 稠密下 30.3B × fp32 = 121 GB（依赖在线 CSR 稀疏存储；服务器内存 ≥ 200 GB）；
② 训练速度随 LTM 访存线性放大，单机 1M tokens 不现实；③ 词表/读出规模不变。

---

## 6. 精度体系

| 层 | 精度 | 说明 |
|---|---|---|
| M6 读出**计算/存储默认** | **fp32**（P194 Q4） | P110 反证：fp32 非目标行更新保留率 **99.95%** / fp16 26.67% / bf16 5.79%（目标行均 100%）→ 低精度把 `p−t` 退化为纯 Hebbian |
| M6 低精度 + 主权重（P195） | fp8/bf16/fp16 影子 + fp32 master | 要低精度访存收益时的正确姿势（§3.2）；numba 侧 fp8 请求归一为 int8 码本，位模式存储只在加速器臂 |
| 检查点 | `--ckpt-dtype fp16`（默认，uint16 位模式） | 存储选项，与计算精度无关；bf16/fp8 存位模式无损解码；**master 模式下 ro_W 存 fp32 逐位**（跳过压缩） |
| M1 编码器 | 库默认 fp64 / **CLI 默认 fp32** | 迭代量恒 fp32；低精度存储在 numpy 路径必付上采样转换 |
| M2 CSR val / 融合核中间量 / M3 STDP W / M4a WM | **fp32**（P173/P176） | 「fp32 存储 + fp32 累加」统一；fp32 才能 SIMD（AVX2 是 f32 指令） |
| softmax / NLL | **fp64 主回路** | 低精度训练的标准结构 |
| RO_DTYPES | fp32/fp16/bf16/int8/int16/int32 | **fp4/int4 已整体删除**（P191e：每步 unpack 抵消存储收益，从未上生产；显式请求构造期 fail-fast）——此为已删除能力的历史注记 |

---

## 7. 硬件与后端

- **M1–M5：numba/CPU 为主**（Cython nogil 核可选加速：`--cython-kernels off/auto/force`，
  **CLI 默认 auto**——加载已编译扩展否则就地构建，失败静默回落 numba；`force` 失败 raise）。
  ⚠ Cython（`/O2` 严格 IEEE）与 numba（`fastmath=True`）有 **fp32 级漂移（~1 ulp/累加）**，
  **不得称逐位等价**；昇腾端到端收益无实测（本机无昇腾）。`readout_pipeline` **已停用**
  （传了直接 SystemExit——训练步设备路径已零同步，额外流水无保持语义的实现）。
- **M6 读出三臂**：稠密码本 / CSR 稀疏 / 加速器（`--accel auto` = 昇腾→ROCm→CUDA→
  DirectML→否则 numba 回落并记原因）。加速器实现均匀 k 的 gather-GEMV（P111）+ P195
  master + NPU Graph（beta 实验位）。
- **CANN 环境治理**（`phdnet/backends/cann_env.py`，须在 import numpy/torch 前加载）：
  `TASK_QUEUE_ENABLE=2` / `COMBINED_ENABLE=1` / expandable_segments /
  MULTI_STREAM_MEMORY_REUSE（P120 起默认，不覆盖用户显式值）；
  `--cann-dispatch` 只核对是否真生效（不新增行为；官方列了两个静默失效条件）。
- **线程**：`--numba-threads` 默认 8（191 核服务器实测核内 1→6 线程仅 1.16×，
  访存带宽饱和）；`OMP_PROC_BIND=close`、无 OMP_PLACES（P74 撤销，place 表让 M2 慢 8–13×）。
- M3 与 M4b 生长是 CPU 结构操作，不上加速器（迁移丢机制 / 定长行生长不可设备化）。

---

## 8. 语料与数据入口

| 入口 | 路径 | 说明 |
|---|---|---|
| 本地 | `datasets/{sft,pretrain}/*.parquet` | `DATA_FILES`；`pretrain` = 全量（zh+en），`pretrain_zh` = lang 过滤对照 |
| 远程 | `--remote-data` | ModelScope `fhzfhz/Mixture-General-Mini`，HTTP Range 流式零落盘；`--remote-fraction 0.3` = 排序后前 30% 分片（前缀子集，顺序语义） |
| 会话数据 | `--data cc_traces` | P196（§3.5） |
| 冻结评测 | `eval_corpus/`（仓库根） | internal_corpus 27,034 字符（80/20 划分）+ ood_wiki 20,187 |

**词法**：`SEG_KWARGS = dict(max_len=6, min_count=5, min_entropy=1.0)`；
`StreamingTokenizer` 在线贪心最长匹配，与离线分词逐位一致，内存 O(max_len)。
词表来源优先级：`--resume` 自包含 > `--vocab-file` > `--vocab-scan head/full`；
词表快照落盘 `outputs/models/vocab_*.txt`。OOV 整步跳过、流不断。
**SFT 回复掩码**：`--assistant-marker` 双标记状态机（零滞后前缀匹配）——只对助手回复计损失，
prompt 段 `learn=False` 推进状态。

---

## 9. 评测与门禁

- **零回归门禁**：`python tests/run_tests.py fast` = **9/9**（< 1 分钟）。
- **专项验证**：`tests/gates/gate_*.py` 共 **47 个**（`ls` 现查，别写死数量）；
  P199 时点 **47/47 全绿**（`outputs/test/reorg_gates/summary.json`；
  `gate_csr_vectorization_perf` 曾 26/27 属 golden 基准文件缺失，后续已绿）。
  P194–P199 新增：`gate_insession_learning` / `gate_readout_master_weight` /
  `gate_generate_truncation` / `gate_cc_traces` / `gate_cli_i18n`。
  ⚠ fast 门禁不 import `train/train.py` → 改 `train/*.py` 后必须单独 `py_compile` + `--help`。
- **PPL 口径**：`ppl_char = exp(Σ token NLL / 评估段字符数)`；bpc = log2(ppl_char)。
- **锚点现状**：历史 fp32 锚点 4K **394.4687** / 全语料 **359.2603**（干净 HEAD `da8f294`
  可复现）；P192 工作区 + fp32 = 465.4778 / 868.3641——η 缩放修正与 LTD 迹补偿均为
  **学习规则语义变更**，旧锚点只存 git 历史、**不可比**；P198 的 40K-token
  发育窗口验证（3403→389）是**另一组口径**（3,392 词表），亦不可混用。
  复测 `tools/eval/rebaseline.py`（只对比不更新 ANCHOR）。
- **i18n**（P199）：`--lang {en,zh}`（默认 en）只管终端文案。实现从正则词表替换重写为
  **显式消息目录**（`_CATALOG` 整句精确匹配 + `_WORDMAP` 多字词整词扫描）——结构上排除
  「大 empty 间表」类碎片乱码；未登记消息原样保留中文（可读、无损）。
  TeeLogger 顺序 bug 已修（先建 Tee 再装 filter，`train/train.py` P199 注）。
  门禁 `gate_cli_i18n.py`（30 项显式契约 + 旧坏例回归）。

---

## 10. 诚实边界

1. **1B 档 PPL 仍高，根因是词表 × 曝光**：词表 51,962 时 exposure=tokens/vocab，
   <30 严重不足（PPL 接近随机）、≥100 才充足（`tools/diag_vocab_budget.py` 训练前估算）。
   P113 服务器日志的 sliding PPL（19602–38101）主要由此贡献，非算法失效。
2. **「正常对话」未达到**：1B 档生成 bigram 合法率 24–35%；3,392 词表 + 40 万 token 可到
   78%（那是缩小 6× 的目标空间）。接近正常对话需 **≥10⁸ token 量级曝光**——数据量问题。
3. **`eta_oja>0` 是已知危险开关**（§3.4）；显式开启收 RuntimeWarning，仅限小规模机制测试。
4. **低精度默认是 fp32**：P195 主权重解除了 P110 的舍入丢失，但要多付一份 fp32 master
   且引入 ≤N 步影子 staleness。
5. **幂律读出默认关**（词频代理依据不足）；开启回落 numba（加速器 fail-fast，正确行为）。
6. **历史 PPL 锚点不可比**（P192/P193 学习语义修正）；昇腾端到端数字均为服务器历史实测，
   本机 x86 只能验管线，**x86 结论不构成昇腾证据**（已多次方向反转）。
7. **SM 会话学习已达**（印迹 ~100%、误差广播生效），但生成质量需 ≥10⁸ token 曝光——
   机制在、数据预算不在，两者不可混为一谈。

---

## 11. 六机制一览表（现行默认）

| 机制 | 脑同构 | 核心文件 | 关键配置（默认） | 稀疏维度 |
|---|---|---|---|---|
| M1 稀疏编码 | 皮层稀疏放电 + 侧抑制 | `sparse_encoder.py` | `k_sparse=16`（1b:128）、`encoder_dtype` 库 fp64/CLI fp32 | 激活 6.25%（1b 12.5%） |
| M2 预测编码 | 皮层预测编码层级 + sparse connectivity | `sparse_pc.py` | `conn_k=0`→n/8（1b:128）、`eta_pc=0.02`、`eta_oja=0`（P198）、`pc_w_max=2.0`、`pc_dev_steps`（1b: 20000） | 连接 12.5% |
| M3 STDP | 海马 CA3 + 皮层局部突触 | `plasticity.py`、`stdp_kernels.py` | `m_lateral=16`、`lambda_trace=0.35`、`eta_stdp=0.03`（1b: 0.02）、`w_max=1.0` | 连接 1.56%（1b） |
| M4a 工作记忆 | PFC 持续放电 + 基底核门控 | `wm.py` | `n_wm_slots=4`、`gamma_wm=0.85`、`gate_thresh=0.35`、`error_gated_memory=True`（P197） | 有限槽位 |
| M4b 长期记忆 | 海马快印迹 + 事件驱动表 | `bigltm.py`、`sparse_table.py` | `big_ltm_N=2^24`、`big_ltm_m=60`（1b: 72）、`big_ltm_k=4`、`ltm_imprint_gate=0.3`（P197）、`sdr_novelty=True` | 突触存在性 |
| M5 调制 | NE/DA + 胆碱能切换 | `modulator.py` | `mod_gain=1.5`、`multi_modulation=False`、`task_modulation=True`（P197） | — |
| M6 读出 | IT→PFC/前运动 | `readout.py`、`backends/accel_readout.py`、`sparse_alloc.py` | `readout_conn_k` 库 0/CLI 128、`readout_dtype=fp32`（P194）、`readout_master_weight=False`（P195）、`readout_powlaw_alpha=0` | 均匀 k 连接 4.2%（CLI）；幂律可选 |
