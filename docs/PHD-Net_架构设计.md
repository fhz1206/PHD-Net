# PHD-Net：预测-赫布-双记忆网络

> 📌 **现状快照（2026-09-27 整理；2026-09-29 硬件后端定稿复核）**：本文已按「删除过时与历史消息、只保留现状」的原则整理；引用已删除语料/旧基线的历史段落以现状结论为准，当前基线与口径见《性能评估与迭代方案（现状版)》。
### Predictive coding × Hebbian plasticity × Dual-memory Network

> 一个完全不依赖自注意力（Self-Attention）的类脑模型架构——总设计文档（基础架构 M1–M6 ＋ 认知层 M7–M12，外扩 M13 上下文漂移情景记忆）。
> 版本：v0.0.0 ｜ 日期：2026-09-16（基础架构）／ 2026-09-18（并入认知层）／ 2026-09-19（脑同构 5 项统一、工程状态定稿）／ 2026-09-20（M9 五轨道收官与语料更换后的终态校订）／ 2026-09-29（硬件后端定稿：生产加速读出 AccelReadout（P28）、读出基准修正（P29）、旧 torch 栈删除（P30））
> 配套实现：`phdnet/`（Python 3.14）｜ 配套文档：《性能评估与迭代方案》《竞争力与脑同构性评估》《对标 Transformer 优化路线图》
> 版本治理：统一使用 **v0.0.0**，分层表述用「基础架构 M1–M6 / 认知层 M7–M12」；是否升 v0.1.0 需 fhz 明确确认（当前保持 v0.0.0）。
>
> ✅ **语料解耦（2026-09-21 起）**：内置语料已冻结为逐字副本 `eval_corpus/internal_corpus.txt`（**23,504 字符**），
> `tests/eval_common.py` 的 `DOC` 指向该文件——**编辑本文不再改变语料与基线**（原「本文即语料，编辑即漂移」条款作废）。
> 若有意更换语料：更新冻结副本 → 重跑 `eval_suite.py` + `demo_m9.py` → 回填全部文档数字。
> 远域泛化探针语料：`eval_corpus/ood_wiki.txt`（中文维基 26 篇 / 20,188 字符）。

---

## 0. 设计动机与总原则

Transformer 的自注意力把"关联"建模为**全局成对相似度**（QK^T，复杂度 O(T²·d)），
把"记忆"建模为**无学习的 KV 缓存**，把"学习"寄托于**非局部的反向传播**。
这三点都与人脑的工作方式相去甚远：

| 人脑做法 | Transformer 做法 | PHD-Net 的选择 |
|---|---|---|
| 突触局部可塑性，无需全局梯度 | 反向传播（全局、非局部） | 全部学习规则**局部化**（每个突触只看自己两端） |
| 稀疏放电（~1–5% 神经元活跃） | 稠密激活 | **k-WTA 稀疏编码**（SDR，稀疏分布式表示） |
| 海马快印迹 + 皮层慢巩固（双系统） | 单一权重 + KV cache | **双速率记忆**：快权重（海马）+ 慢权重（皮层） |
| 前额叶有限容量工作记忆（≈7±2） | 上下文窗口随 T 线性/平方膨胀 | **有界槽位工作记忆** + 长期记忆检索 |
| 层级预测：高层级预测低层级输入 | 无生成式层级 | **预测编码层级**（自由能最小化） |
| 神经调制（DA/NE/ACh）门控学习 | 固定学习率 | **全局调制标量**门控学习率与编码/检索模式 |

**四条总原则：**
1. **无自注意力**：任何位置都不出现 QKV 投影与全局成对相似度计算；
   时序关联由**突触迹（STDP）**与**状态迹**承载，计算量与活跃神经元数成正比，而非与历史长度平方成正比。
2. **学习规则局部化**：每个突触的更新只依赖其前/后端神经元活性、局部迹、及一个全局标量调制信号——无反向传播。
3. **稀疏即效率**：全网络采用稀疏分布式码（SDR），激活率 ~10%；计算与存储只触碰活跃通路。
4. **双速率记忆**：一次性经验写入快权重（海马式印迹），统计规律缓慢巩固入慢权重（皮层式学习）。

---

## 1. 架构总览与信息流

```
            ┌──────────── 全局调制信号 M（surprise → 学习率门控 / 编码-检索切换）───────────┐
            │                                                                            │
 输入 x_t   ▼                                                                            │
  ─────▶ M1 稀疏编码器 ── s_t ──▶ M2 预测编码层级 ── r2_t ──▶ M3 时序关联核(STDP) ───────┤
            (初级感觉皮层)        (皮层层级结构)          (海马 CA3 / 皮层局部突触)          │
                                      │  上行误差 e ↑  下行预测 ŷ ↓                       │
                                      │                                                  │
                                      ▼                                                  ▼
                              M4a 工作记忆（PFC 槽位+门控+衰减）◀── 检索线索 ── M4b 长期记忆
                                      │                          (快权重印迹+慢权重巩固)
                                      ▼                                  ▲
                              M6 读出头 ◀── h = [r2 ; WM ; 检索结果] ──────┘
                                (IT→前运动皮层输出)
```

**五条通路：**
- **前馈流**：x → SDR 稀疏码 → 逐层自下而上识别 → 顶层表示；
- **反馈流**：高层级向低层级发送预测，低层级只上传**预测误差**（预测编码核心）；
- **横向流**：层内 Hebbian/Oja 关联（特征绑定），跨时间步 STDP 因果绑定（序列结构）；
- **记忆流**：工作记忆 ⟷ 长期记忆（写 = 印迹；读 = 吸引子模式补全）；
- **调制流**：底层预测误差范数 → 意外度（surprise）→ 门控所有模块的学习率与工作模式。

---

## 2. 模块详解（M1–M6）

### M1 稀疏编码器 Sparse Encoder
- **大脑对应**：初级感觉皮层的稀疏放电；抑制性中间神经元（SST/PV）介导的**侧抑制**与胜者全取（WTA）竞争。
- **机制**：输入 x ∈ R^d 经固定（或 Oja 慢调）随机投影得全层电位 u，经 **k-WTA** 竞争仅保留 top-k 个胜者，
  其余置零，得到稀疏分布式表示 s（SDR），典型激活率 k/N ≈ 12.5%。
- **信息流**：感受野投影 → 竞争 → 稀疏码 s 与胜者索引 idx（供后续稀疏快速通路）。
- **数学**：u = W_enc·x + b；s_i = normalize(u_i) 若 i ∈ top-k(u)，否则 0。

### M2 预测编码层级 Predictive Coding Stack
- **大脑对应**：皮层层级 V1→V2→V4→IT 的**预测编码**（Rao & Ballard 1999；Friston 自由能原理）；
  深层锥体细胞携带内容表示，浅层颗粒细胞携带误差信号。
- **结构**：L 层。每层有识别（上行）权重 W_up 与生成（下行）权重 W_dn：
  - 上行：r_{l+1} = φ(W_up,l · e_l)，φ = tanh；
  - 下行：ŷ_l = W_dn,l · r_{l+1}（生成式预测）；误差 e_l = s_l − ŷ_l；
  - 推理时做 T_infer 步误差驱动精炼（近似自由能最小化）。
- **信息流**：误差自下而上逐层上传、预测自上而下逐层下发，构成闭环；顶层表示 r_L 是当前内容的抽象。
- **学习**（局部，无需反传）：
  - 生成权重：ΔW_dn,l = η · e_l ⊗ r_{l+1}（预测误差 × 上层表示，最小化误差）；
  - 识别权重：**Oja 规则** ΔW_up,l = η · r_{l+1} ⊙ (e_l − r_{l+1} ⊙ W_up,l)
    （Hebb 项 + 归一化项，防止权重无界增长，隐式做类 PCA 特征提取）。

### M3 时序关联核 Sequential Association Core（STDP）
- **大脑对应**：海马 CA3 递归可塑性网络；皮层局部突触的**脉冲时序依赖可塑性**
  （STDP：先发放的突触前神经元增强后发放的突触后连接——因果学习，Bi & Poo 1998）。
- **结构**：稀疏固定拓扑。每个神经元只向 m 个随机后继发出连接（模拟皮层局部连接组），
  总潜在突触数 = N·m，但存储与计算只触碰活跃通路。
- **机制**（突触迹模型）：
  - 突触前迹 P_pre ← λ·P_pre + rate(r_{t-1})；突触后迹 P_post ← λ·P_post + rate(r_t)；
  - Δw = η · (P_pre ⊗ post − P_post ⊗ pre)₊，截断于 [0, w_max]；
  - 直觉：**"先响后响 → 增强；后响先响 → 抑制"**——网络自发学到时间因果结构。
- **信息流**：预测 p_t = f(W·rate(r_{t-1}))；p_t 与当前表示的偏差即**时序意外度**。
- **工程注记**：STDP 核由 numba JIT 实现（`_predict_edges`/`_stdp_delta`），带运行时自检；
  自检已升级为三重门槛（scatter 语义 + 一致缩放下权重增长 + numba/numpy 逐位等价），
  实测 **NUMBA_OK = True**，加速 **5.8–51.5×**（n=8192/m=32 → 51.5×）。

### M4a 工作记忆 Working Memory
- **大脑对应**：背外侧前额叶（dlPFC）的**持续放电**（delay activity）；基底核-丘脑回路的**门控**写入。
- **结构**：C 个槽位（默认 4，对应有限容量）；每槽保存一个顶层表示及其强度。
- **机制**：每步全部槽位衰减 ×γ_wm（模拟持续放电的漏积分）；
  写入由调制门控 gate 决定——只有"意外"的信息才占用槽位（对应基底核门控假说）；
  读出为强度加权的槽位总和。
- **信息流**：接收顶层表示写入；向读出头与长期记忆检索提供线索。

### M4b 长期记忆 Long-Term Memory（双速率）
- **大脑对应**：**海马**（快速一次性印迹，索引理论）+ **新皮层**（慢速统计学习）；
  互补学习系统理论 CLS（McClelland, McNaughton & O'Reilly 1995）。
- **结构**：两套联想矩阵——
  - 快权重 W_fast：单步 Hebbian 外积印迹 ΔW = η_hip·p ⊗ p（对称自联想，去对角防自激），
    或非对称 ΔW = η_hip·p_target ⊗ p_cue（配对关联）；
  - 慢权重 W_slow：**睡眠巩固**阶段以指数滑动平均吸收快权重
    W_slow ← (1−η_c)·W_slow + η_c·W_fast（模拟离线回放与系统巩固）。
- **检索**：**Hopfield 式吸引子动力学**——r ← sign((β_s·W_slow + β_f·W_fast)·r)，迭代至收敛；
  天然支持**模式补全**（给残缺线索恢复完整记忆）。
- **信息流**：编码模式下新经验快速印迹；检索模式下由线索（来自 WM 或感官）收敛到吸引子。

### M5 神经调制器 Neuromodulator
- **大脑对应**：蓝斑去甲肾上腺素/腹侧被盖多巴胺的**意外与奖赏信号**；
  基底前脑胆碱能系统在"编码模式/检索模式"间的切换（Hasselmo）。
- **机制**：意外度 = 底层预测误差范数（||e_0||），经在线 Welford 标准化为 z 分数：
  - 学习门控 gate = σ(a·z)（意外大 → 可塑性开大：先见者学，重复者忘）；
  - 模式切换：z 高 → 编码模式（印迹 + WM 写入）；z 低 → 检索模式（吸引子提取）。
- **信息流**：一个全局标量广播至 M2/M3/M4，缩放各自的学习率。

> **脑同构升级（2026-09-19，默认关闭）**：单标量 z 门控已升级为四通道 `MultiModulator`（`multi_modulation` 开关）。四个神经调质通道各司其职——**ACh** 编码/检索模式门控（Hasselmo）、**NE** 新颖性驱动的可塑性增益（|Δsurprise| 骤变→可塑性开大，Yu & Dayan）、**DA** 巩固信号（surprise 低→高 DA→海马→皮层巩固/回放）、**5-HT** 耐心（低唤醒→延长检索等待、降低冲动写入）。返回接口与 `Neuromodulator` 兼容，NE/DA/5-HT 分别细化 STDP 学习率、睡眠回放强度、检索节奏。默认路径仍用 `Neuromodulator`，逐位不变。

### M6 读出头 Readout
- **大脑对应**：IT → 前额叶/前运动皮层的决策读出。
- **机制**：h = [r_L ; h_WM ; h_recall]，y = softmax(W_ro·h)；
  读出权重用**感知器式局部规则** ΔW_ro = η·(t − y) ⊗ h（监督信号仅存在于最末端，主干完全无监督）。

---

## 3. 连接关系汇总

| 连接 | 类型 | 载体 | 对应脑通路 |
|---|---|---|---|
| x → s | 前馈投影 + WTA | W_enc（稀疏竞争） | 感觉通路 → V1 |
| s ↔ r1 ↔ r2 | 上行识别 / 下行生成 | W_up / W_dn | 皮层层级前馈/反馈纤维 |
| r2(t-1) → r2(t) | 稀疏拓扑可塑性 | W_seq（STDP） | CA3 递归 / 皮层局部连接 |
| r2 → WM | 门控写入 | 槽位阵列 | 皮层 → PFC |
| WM/r2 → LTM | 线索检索 / 印迹写入 | W_fast / W_slow | PFC/皮层 → 海马 → 皮层 |
| e0 → 全局 | 标量广播 | 调制信号 | NE/DA/ACh 弥散投射 |
| [r2,WM,recall] → y | 线性读出 | W_ro | IT → 前运动 |

---

## 4. 学习规则总表（全部局部，无反向传播）

| # | 规则 | 公式 | 作用 |
|---|---|---|---|
| 1 | k-WTA 竞争 | s = top-k(normalize(u)) | 稀疏编码 |
| 2 | Oja 规则 | Δw_ij = η·y_i·(x_j − y_i·w_ij) | 稳定的 Hebbian 特征学习（防发散） |
| 3 | PC 误差规则 | ΔW_dn = η·e ⊗ r_hi | 生成模型拟合数据分布 |
| 4 | STDP 迹 | Δw = η·(P_pre·post − P_post·pre)₊ | 时间因果结构学习 |
| 5 | 海马印迹 | W_fast += η_h·p_out ⊗ p_in | 一次性经验快速存储 |
| 6 | 皮层巩固 | W_slow ← (1−η_c)W_slow + η_c·W_fast | 慢速统计学习（睡眠回放抽象） |
| 7 | 感知器读出 | ΔW_ro = η·(t − y) ⊗ h | 仅末端使用监督 |
| 8 | 调制门控 | η_t = η₀·σ(a·z(surprise)) | 全局可塑性调度 |

**稳定性设计**：Oja 归一化项防权重爆炸；STDP 截断于 [0, w_max]；
Hopfield 检索用 sign 吸引子保证收敛；双速率分离避免快学习破坏旧知识（灾难性遗忘的结构性对策）。

---

## 5. 推理流程（单个时间步）

```
输入 x_t
 1. M1:  s_t = kWTA(W_enc·x_t)                      # 稀疏编码
 2. M2:  自下而上生成 r1、r2；自上而下算误差 e0、e1
         并做 T_infer 步误差驱动精炼（自由能下降）
 3. M3:  p_t = f(W_seq·rate(r2_{t-1}))              # 时序预测
         seq_err = 1 − cos(p_t, rate(r2_t))
 4. M5:  surprise = ‖e0‖ → z 标准化 → gate、模式(编码/检索)
 5. M4a: WM 衰减；若 gate > τ 则写入新槽位
 6. M4b: 若编码模式且 z 极高 → LTM 快速印迹
         若检索模式 → 由 WM/r2 线索做吸引子检索 → 回填 WM
 7. M6:  y_t = readout([r2 ; WM ; recall])
 8. 学习（若开启）：所有突触按 §4 局部规则 × gate 同步更新
```

**训练即推理**：在线、单样本、无 epoch 概念；每 K 步执行一次
`sleep()`——巩固快权重入慢权重、衰减过时印迹（离线回放的抽象）。

---

## 6. 规模化设计（1B 参数如何在 CPU 上训练）

人脑 860 亿神经元、~10¹⁴–¹⁵ 突触，却只消耗 20W——秘诀是
**稀疏连接 + 事件驱动的局部计算**：任一时刻只有极少数神经元活跃，
计算量正比于**活跃突触数**而非**总突触数**。PHD-Net 继承这一性质：

- 总潜在突触 = N·m（如 N = 2²⁴ ≈ 1678 万神经元 × m = 60 出边 ≈ **10 亿突触**）；
- 连接拓扑由**确定性伪随机生成**（每神经元的出边索引由 seed+i 派生），
  冻结的结构连接**不占内存**——这正是"基因预连接"的抽象；
- 可训练突触只存储**被活跃触碰过的部分**（SDR 每步仅 ~k 个神经元活跃，
  每步只读写 k·m ≈ 数千条边），用有界印迹表存储，内存占用数百 MB 以内；
- 因此 1B 参数模型的**单步训练在 CPU 上是毫秒级**的——密集 1B 模型
  （权重+梯度+优化器状态 ≈ 16GB 且每步 10⁹× 数次 FLOPs）在无独显机器上物理不可行，
  而事件驱动的稀疏类脑架构使其可行。
- **1B 规模实测（2026-09-19 复跑 `tools/train_1b_capacity.py` 定稿）**：容量 N=2²⁴ × m=60 ≈ 1.007B，
  **单步训练耗时 1.28 ms/步**（1500 步合计 1.92 s，已生长 3,072 条突触、利用率 0.0003%）；
  突触迹与时间戳改为 dict 按需存储，RSS 增量约 0 MB（相对 N 长度数组方案 ~197 MB 的实质改进）。
  注：早前容量层独立微基准（`tests/demo_m4.py` 的 `SparseSynapseTable` 查表核）曾报 0.47 ms/步，
  1.28 ms/步 为含完整 `BillionSynapseNet` 前/后向 + 印迹/检索的端到端单步耗时，作为口径统一后的头条数字。

---

## 7. 与 Transformer 的机制对照

| 维度 | Transformer | PHD-Net |
|---|---|---|
| 关联机制 | 全局成对点积 QK^T，O(T²·d) | 局部突触可塑性 + 迹，O(活跃×m) |
| 记忆 | KV cache（无学习、随长度膨胀） | 双速率可塑权重 + 吸引子（可学习、可补全、有界） |
| 上下文 | 有限窗口 | WM 有界槽位 + LTM 无限检索，代价恒定 |
| 学习 | 反向传播（非局部、需成批） | 局部规则（在线、单样本） |
| 归纳偏置 | 无（纯数据驱动） | 稀疏性、时序因果、层级预测、竞争 |
| 在线适应 | 需微调 + 重放 | 天然单样本印迹 |
| 硬件亲和 | GPU 密集矩阵乘 | 事件驱动稀疏读写（CPU 友好） |

**坦白的取舍**：PHD-Net 放弃了注意力的"任意两 token 一步直达"能力，
长程关联依赖 WM 槽位与 LTM 检索的间接通路；在纯语言建模这类极端长程任务上
预计弱于同规模 Transformer。其优势场景：流式在线学习、少样本关联、
模式补全、低功耗/边缘设备、持续学习而不灾难性遗忘。

---

## 8. 超参数表（默认配置）

| 超参 | 默认值 | 含义 |
|---|---|---|
| n_sdr / k_sparse | 256 / 32 | SDR 维度与稀疏度（M1） |
| n_mid / n_top | 128 / 64 | PC 中间层/顶层维度（M2） |
| n_infer_steps | 2 | 误差驱动精炼步数 |
| n_wm_slots / gamma_wm | 4 / 0.85 | 工作记忆容量/衰减（M4a） |
| m_lateral / lambda_trace | 16 / 0.7 | STDP 出边数/迹衰减（M3） |
| eta_pc / eta_stdp / eta_oja | 0.01–0.05 | 各模块基础学习率 |
| eta_hip / eta_cortex | 0.08 / 0.02 | 海马印迹率/皮层巩固率（M4b） |
| beta_fast / beta_slow | 0.6 / 1.0 | 检索时快/慢权重配比 |
| task_modulation / task_mod_gain | False / 2.0 | T1.1 误差广播调制开关与增益（默认关，路线图 M1） |
| error_gated_memory | False | T1.3 错误驱动记忆写入（默认关） |
| pred_in_readout | False | T3.1 预测通路三拼 h=[r2;WM;pred]（默认关） |
| multi_modulation / gain_ach,ne,da,ht | False / 1.5,2.0,1.5,1.0 | M5 四通道神经调制：ACh/NE/DA/5-HT（默认关，脑同构升级） |
| stdp_homeostasis / homeo_target | False / 2.0 | M3 STDP 稳态突触缩放（行范数超阈拉回，防发散；默认关） |
| metaplasticity / bcm_tau | False / 0.01 | M3 BCM 元可塑性滑动阈值 θ（默认关） |
| staged_sleep / sleep_downscale | False / 0.9 | M4b 分期睡眠 SWS/REM + 突触 downscaling（默认关） |
| ei_synapses / ei_ratio | False / 0.2 | E/I 突触类型：抑制性出边权重为负（默认关） |
| critical_period / crit_rate / prune_threshold | False / 0.001 / 0.01 | M6 发育期临界期 + 突触修剪（默认关） |

**M9 期新增开关（全部默认关，括号内为 M9 实测结论）**

| 超参 | 默认值 | 含义与实测 |
|---|---|---|
| `retrieval_topk` / `retrieval_gain` | 0 / 1.0 | T3.2 常态 top-k 检索融合（**符号不稳：−2.4% → −0.65% → −1.1% → 冻结语料 +3.2%（有害）；不推荐启用**） |
| `wm_content_address` / `wm_sim_thresh` | False / 0.6 | T3.4 WM 余弦内容寻址写入（冻结语料 **−1.6%**，稳定正增益） |
| `readout_replay` / `replay_every` / `replay_eta` | False / 4 / 0.5 | ② 回放稳定读出（冻结语料 **−1.4%**；四轮均为增益 −0.8% → −1.34% → −1.5% → −1.4%） |
| `episodic_len` | 0 | T3.3 情景缓冲接入生成（PPL 中性；bigram 合法率 100%→99.2%） |
| `plateau_sleep` / `plateau_patience` | False / 2 | T4.4 验证驱动睡眠（±0.0%） |
| `learnable_encoder` / `eta_enc` | False / 0.01 | T2.2 可学习稀疏词典（+6.1%，不建议） |
| `pc_predictive_target` / `pc_pred_mix` | False / 1.0 | T1.2 PC 预测目标化（+15.9%，不建议） |
| `pc_dev_steps` | 0 | ① 发育期后强制巩固表征（+15.6%，不建议） |
| `neuron_target_rate` / `eta_homeo` | False / 0.01 | ③ 逐神经元目标发放率（+37.9%，不建议） |
| `bidir_check` / `bidir_w` | False / 1.0 | 复制任务双向校验（−9.9pp，不建议） |
| `sparse_conn` / `conn_k` | True / 0 | **O1-2 主干结构性稀疏连接（唯一实现）**（CSR 稀疏图 + numba 核；脑对应：皮层 dense representation + sparse connectivity）。`conn_k` = 每神经元入边数（0 = 各层 n_in//8）。实测：连接率 3.1%（k=8）时 PPL 与稠密持平（+0.03%）、突触存储压缩 32×；k=16 时 −0.51%。**2026-09-28 起（fhz 指令）稠密栈已删除**，`sparse_conn=False` 在 config 层 fail-fast（原 `sparse_pc`/`pc_topk` 字段随之移除） |
| `readout_conn_k` | 0 | **O1-3 读出结构性稀疏**（CSR）。实测 k=8（连接率 1.0%）：**速度 13.3×（7.95→0.60 ms/token）但 PPL +14.8%** ⇒ 吞吐换精度，**默认不启用**；副产物：读出是全项目最大计算瓶颈（占 7.9 ms/token 主体） |
| `lognormal_init` / `exc_ratio` | False / 0.8 | **O1-4 皮层式权重初始化**（脑对应：突触强度对数正态/重尾 + E/I ≈ 80/20，Song et al. 2005）。启用后权重分布判定为"重尾"，初始 E/I 比即为设定值；默认关闭以保历史基线口径 |
| `readout_hidden` / `readout_hid_k` / `readout_eta_hid` | 0 / 0 / 0.002 | **A4 两级群体读出**（脑对应：皮层→输出为**多级中继 + 群体编码**，非单层线性分类器）：h → 隐藏群体（稀疏投射 + k-WTA 侧抑制竞争 + **局部无监督 Oja**）→ 输出（稀疏投射 + 任务监督）。实测：速度 **6.828→0.999 ms/token（6.8×）**、参数 −60%，但 PPL **+10~13%**（h=1024 时 +10.3% 最优）→ 与单级稀疏化同量级代价，**默认关闭** |
| `ei_synapses` / `ei_ratio` | False / 0.2 | **A9 独立抑制类群**（脑对应：抑制性中间神经元只发抑制性投射）。实测：**抑制比 0.1 → PPL −1.17%（正增益）**、0.2/0.3 无损 → **推荐 `ei_synapses=True, ei_ratio=0.1`**（默认关闭以保基线口径） |
| `k_sparse`（推荐值 16） | 32 | **A6 激活稀疏度**（脑对应：皮层 1–5% 稀疏发放）。实测：k=16（6.2%）→ **PPL −1.43%**；k=8（3.1%）→ +12.5%（容量不足有害）⇒ **推荐 16**（更类脑且更优） |
| `growth_guidance` | False | T5.2 拓扑生长引导（入度上限 32 生效；因果重合 0.50/0.50 中性） |
| `readout_w_clip` | 0.0 | C8 读出权重范数上限（0 = 关闭，保持旧行为） |
| `sparse_table_prune` | 0 | C1 迹/时间戳字典惰性淘汰阈值（0 = 关闭） |
| `consolidate_every` | 1 | C2 每 N 次 sleep 执行一次巩固（1 = 旧行为） |

> 铁律口径：上表（含 M9 新增）**全部默认关闭/保持旧值**，因此默认路径逐位不变；
> `pred_in_readout` 在库配置中同样为 `False`，仅在评测基线配置 `tests/eval_common.py` 的 `BASE` 中置 `True`。

---

## 11. 硬件后端适配（昇腾 NPU / ROCm / CUDA / DirectML / CPU）

同一份算子代码覆盖所有后端：torch 路径只切换 device/dtype，语义与 numpy 版一致
（生产加速读出 `AccelReadout` 的前向为 GEMV、更新为 rank-1 AXPY；对拍判据见下方等价性门槛）。

| 层次 | 默认实现 | 加速实现（P30 后现状，2026-09-29） |
|---|---|---|
| M6 读出（1B 预设占比 ~89%） | numpy / numba（CPU） | **`phdnet/backends/accel_readout.py::AccelReadout`（生产加速读出，torch）** |
| M3 关联核（热点） | numpy + numba（CPU，加速 5.8–51.5×） | —（torch 化 STDP 核已随 P30 删除，见下方注记） |
| M1/M2/M4a/M4b | numpy | — |
| 1B 大容量印迹表 | dict 邻接表（CPU） | —（待迁移设备端稀疏张量） |

**生产加速读出（AccelReadout，P28）**：torch 实现，W 常驻设备；
`addmm_` 做 rank-1 AXPY 更新（不物化与 W 同尺寸的临时张量）、(h, y) 设备侧缓存、
硬同步 3→1——每步流量 4,334 → 2,600 MiB（**−40%** 触底），数值与 numpy 路径**逐位相同**。

- 设备解析（torch 路径）：`phdnet/backends/torch_lm.py::resolve_device`
  （**auto 择优：昇腾 NPU → ROCm → CUDA → DirectML → CPU**；显式设备不可用时诚实报错，
  仅 `allow_fallback=True` 才在 RuntimeWarning 后回退 CPU）；
- 探测与选择（numpy 路径）：`phdnet/device.py`（`probe()` / `select_backend()` / `use_torch_backend()` + 各生态安装指引）；
- 设备探针与读出基准：`phdnet/backends/torch_backend.py` 仅存
  `probe_devices`（统一加速器探针：昇腾 NPU / ROCm / CUDA / DirectML / CPU）与
  `bench_readout`（P29 修正：每个计时区间前后设备同步 + `dtype` 参数 + 报告等效带宽 GB/s；
  默认测生产对象 `AccelReadout`，默认 V=73,958）；
- 多卡并行：`phdnet/backends/multi_device.py`——`resolve_devices`（auto 解析全部同型号设备）/
  `plan_parallel`（单卡零变更，多卡读出列并行）/ `MultiDeviceReadout`（按词表行切分，每步仅两次小向量通信）/
  `capability_report`（能力矩阵）。诚实定位：这是**读出列并行的模型并行**，不是 DDP/DataParallel
  （逐 token 事件驱动、无 batch 与反向图，标准数据并行没有梯度可分）；
- 体检脚本：`tools/backend_probe.py`；后端检查：`tests/checks_backend.py`
  （P30 后精简为设备探测，torch 栈用例随栈删除）；
- 等价性门槛：非 numpy 后端启用前须通过对拍验证
  （`tests/verifiers/verify_accel_readout.py`，容差判据，跨设备不宣称逐位；
  旧 `selftest_torch()` 已随 P30 删除——沿用 numba 自检 bug 的教训：不能只判「权重有没有增长」）；
- 开关：`config.backend`（`auto` / `numpy` / `torch` / `npu` / `rocm` / `cuda` / `cpu`）+
  `PHDNetConfig.accel_readout`（读出加速，默认 `auto`）；
- **静默降级已修复（A1，2026-09-19）**：显式请求本机不可用的加速器（如 `backend="npu"`/`"rocm"` 而无对应硬件）
  不再静默在 CPU 上以 torch 训练——`use_torch_backend` 返回 `False` 并发出 `RuntimeWarning`、
  `select_backend` 返回 `kind="unavailable"`；
  默认 `backend="auto"` 无加速器时仍走 numpy 路径，行为逐位不变。

> **P30（2026-09-29）：旧 torch 栈已全部删除（约 1,400 行）**——
> `torch_lm.TorchPHDNet` / `TorchWordLM` / `TorchSparsePC` / `TorchLTM` /
> `TorchReadoutDense` / `TorchSparseEncoder` / `_ProdSTDPCore`、
> `torch_backend.TorchReadout` / `TorchSTDPCore` / `selftest_torch`、
> `tools/train_torch_lm.py`、`tests/verifiers/verify_torch_lm.py`、
> 以及旧 shim `phdnet/torch_backend.py`。理由：① 权重与生产 npz 检查点不通用
> （不同实现、不同初始化顺序）；② 缺 `big_ltm` 等 7 项机制，整体迁移会丢机制；
> ③ 从未进入生产——加速需求已由 `AccelReadout`（只迁移读出、不动 numba 主循环）单独覆盖。

诚实边界：昇腾 NPU 与 ROCm 路径目前**仅完成代码适配与接口探测**——本机为
Windows + 无 NPU/AMD GPU（torch 2.13.0+cpu），需在具备硬件的环境运行
`tools/backend_probe.py` 才能完成实测验证；ROCm 官方仅支持 Linux，Windows 上的
AMD GPU 建议 DirectML 路径。

> **工程状态（2026-09-19 终态）**：原《代码审计报告》已按指令删除，其修复记录并入本说明。
> 全部 A/B/C 类代码审计风险已修复——A1 静默降级、A2 `SparseLTM` 契约校验、
> A3 非 softmax 读出配置生效、B1–B7（含 `max_len` 配置化、dtype 纳入等价门禁、`readonly` 可复现评估等）、
> C1/C2/C6/C7/C8（迹剪枝、`consolidate` 节流、`count_params`、重置结果、读出权重上限）。
> 铁律：新增行为一律走 config 开关且默认关闭、默认路径逐位不变；`tests/run_tests.py` fast **11/11 通过**。
> **语料布局（2026-09-29 更新）**：`data/` → `datasets/`（分子目录 `sft/`、`pretrain/`，`raw/` 存原始件不入库）；
> **冻结评测基准已迁到仓库根 `eval_corpus/`**（`internal_corpus.txt` 23,504 字符 + `ood_wiki.txt` 探针）——
> 2026-09-27 指令「docs/*.md 不是数据集」。训练数据有两个入口：本地 `datasets/`（默认，逐位不变）
> 与 `--remote-data` 直读 ModelScope（HTTP Range 流式零落盘，`--remote-fraction` 取前缀分片；
> atomgit 限额 1 GiB < 交付 4.64 GiB，故服务器走远程）。
> 中文维基预训练语料（原 54.0 + 5.4 MB）已按指令删除。**预训练语料已拍板（2026-09-22）：Infinity-Instruct `7M_core`（M7_Core）**——atomgit `BAAI/Infinity-Instruct`，75 parquet 分片 6.1 GB，实测 750 万条对话 / 全量文本 ~10.2 GB，经 `tools/convert_infinity.py` 流式转换落 `datasets/pretrain/infinity_m7core.txt`（**已就位：7,449,106 条 / 10.07 GB 文本**；语言分布 en 89.9% / zh-cn 10.1%，中文子集约 75 万条 / ~1 GB）。
> **SFT 语料已就位（2026-09-23）**：OpenBMB `UltraInteract_sft`——288,579 条 / **0.603 GB 文本**，77,023 条唯一指令（偏好树 `parent_id`，平均 3.75 响应/指令），任务构成 Coding 39.8% / Math 56.2% / Logic 4.0%；`tools/convert_ultrainteract.py` 转换落 `datasets/sft/ultrainteract_sft.txt`。
> 旧的多份 SFT / 会话轨迹语料（`tool_train.txt` 等）此前已全部删除，详见《性能评估与迭代方案》§5.2。

---

## 9. 代码映射表

| 架构模块 | 文件 | 核心类/函数 |
|---|---|---|
| M1 稀疏编码器 | `phdnet/sparse_encoder.py` | `SparseEncoder.encode` |
| M2 预测编码层级 | `phdnet/sparse_pc.py` | `SparsePCStack.infer / learn`（CSR 结构性稀疏；稠密栈 2026-09-28 删除） |
| M3 STDP 计算核与自检 | `phdnet/stdp_kernels.py` | `_predict_edges` / `_stdp_delta` / `NUMBA_OK` |
| M3 时序关联核容器 | `phdnet/plasticity.py` | `STDPCore`（稀疏拓扑 + 突触迹 + 自适应/双迹） |
| M4a 工作记忆 | `phdnet/wm.py` | `WorkingMemory`（兼容再导出：`phdnet/memory.py`） |
| M4b 长期记忆 | `phdnet/ltm.py` | `LongTermMemory.imprint / recall / consolidate` |
| M5 神经调制器 | `phdnet/modulator.py` | `Neuromodulator.observe` |
| M6 读出头 | `phdnet/readout.py` | `Readout.learn / learn_softmax` |
| M6 组装与调度 | `phdnet/model.py` | `PHDNet.step / sleep`（再导出 `Readout`） |
| M7 词涌现与词编码 | `phdnet/word_encoder.py` | `WordSegmenter` / `WordTokenizer`（T2.3 上下文绑定） |
| M7–M12 认知层 | `phdnet/cognition.py` ｜ `phdnet/generate.py` | `SemanticGraph` / `EpisodicBuffer` / `chain_query` ｜ `Generator` |
| 认知层验收 | `tests/demo_v2.py`（开发期文件名沿用） | 四实验：知识调用 / 链式推理 / 文本生成 / 时序回放 |
| M1 升级验收 | `tests/demo_m1.py` | 增量消融：预测通路三拼 / 误差广播调制 / 错误驱动写入 |
| M2 词级语言接口 | `phdnet/word_lm.py` | `PHDWordLM`（词法见 word_encoder.py，基线见 ngram.py） |
| M2 验收 | `tests/demo_m2.py` | 词级 LM 五变体消融 + n-gram 参照 + 生成合法率 |
| M3 优化调度（T4.1/T4.2） | `phdnet/plasticity.py` | `STDPCore`（逐突触自适应 + 多尺度双迹） |
| M3 课程学习（T4.3） | `phdnet/word_lm.py` | `PHDWordLM.train_stream(curriculum_top=)` |
| M4 容量层本体 | `phdnet/sparse_table.py` | `SparseSynapseTable`（2^24×60 ≈ 1.01B 突触容量） |
| M4 容量层适配器 | `phdnet/bigltm.py` | `SparseLTM`（表示维度 ↔ 大空间哈希对接） |
| M4 验收 | `tests/demo_m4.py` | 容量层因果学习 + 接入对照 |
| M5 对照模型 | `tests/nano_gpt.py` | nanoGPT 级 Transformer（PyTorch，0.44M / 2.22M） |
| M5 评测套件 | `tests/eval_suite.py` | 编排与判定矩阵（任务实现见 `tests/eval_tasks_*.py`，共享件 `tests/eval_common.py`） |
| M8 架构强化验收 | `tests/demo_strength.py` | 稳态缩放 / 读出退火 / 错误触发检索 |
| M13 上下文漂移情景记忆 | `phdnet/context_memory.py` | `ContextMemory`（漂移/印迹/保持槽/模式分离） |
| M13 验收 | `tests/demo_copy.py` | 延迟复制 n=4/8/16（含模式分离与遗忘扫描） |
| 硬件后端适配（numpy 路径） | `phdnet/device.py` | `probe()` / `select_backend()` / `use_torch_backend()` |
| 生产加速读出 | `phdnet/backends/accel_readout.py` | `AccelReadout`（torch；`addmm_` rank-1 AXPY + 设备侧缓存） |
| 设备解析（torch 路径） | `phdnet/backends/torch_lm.py` | `resolve_device`（auto：昇腾→ROCm→CUDA→DirectML→CPU） |
| 设备探针与读出基准 | `phdnet/backends/torch_backend.py` | `probe_devices` / `bench_readout`（P29 修正） |
| 多卡并行 | `phdnet/backends/multi_device.py` | `resolve_devices` / `plan_parallel` / `MultiDeviceReadout` / `capability_report` |
| 后端检查 | `tests/checks_backend.py` | 设备探测（torch 栈用例已随 P30 删除） |
| 后端体检 | `tools/backend_probe.py` | 探测昇腾/ROCm/CUDA/DirectML + 等价性自检 |
| 回归总入口 | `tests/run_tests.py` | fast（默认）/ `--full` 分层回归 |
| 演示实验 | `tests/demo_phdnet.py` | 序列预测 / 模式补全 / 少样本关联 |
| 1B 训练 | `tools/train_1b_capacity.py` | `BillionSynapseNet`（§6 规模化设计） |
| M9 五轨道验收 | `tests/demo_m9.py` | 15 项配置对比 + 容量赛道（int8 / CSR / 稀疏 PC / 生长引导） |
| 外部语料获取 | `tools/fetch_modelscope.py` | ModelScope 数据集文件下载（含断点续传，实测 ~5 MB/s；本期间用于 ood_wiki.txt Range 抽取） |
| 外部语料制备 | `tools/prepare_wikicn.py` | 中文维基 JSONL → `datasets/pretrain/wiki_{train,eval}.txt`（清洗 + 确定性划分；维基语料已删，脚本留档待新锚点） |
| 外部语料验收 | `tests/demo_corpus.py` | 中文维基上的词级 LM（OOV 安全评估） |

**工程实现注记**：
1. numba 核（`_predict_edges`/`_stdp_delta`）带**运行时行为自检**（`_selftest_numba`）：
   三重门槛（scatter 语义 + 一致缩放下权重增长 + numba/numpy 逐位等价），通过则启用 JIT，
   否则回退纯 numpy 向量化实现，行为完全一致（实测 `NUMBA_OK=True`，加速 5.8–51.5×）。
2. STDP 的生物学要点在实现中体现为三条工程约束：
   突触前/后迹的惰性衰减时间戳必须独立；迹更新与学习的时间顺序决定 LTP/LTD
   是否相消；发放率输入必须稀疏化（稠密输入会使历史迹残留污染因果学习）。
3. 规模化版（1B）中，相邻成对约定下新突触生长只由 LTP 驱动，
   LTD 仅作用于已存在突触以提供稳态平衡。

---

## 10. 认知层模块（M7–M12）：面向 LLM 功能目标的架构升级

> 开发期代号"v2"（验收脚本 `tests/demo_v2.py` 沿用该文件名）；官方版本号统一为 **v0.0.0**。
> 在基础架构（M1–M6）之上新增认知层，使架构机制上覆盖 LLM 的四项核心功能：
> 自然语言理解、上下文处理、知识存储与调用、文本生成。
> **铁律合规**：全程无自注意力、无位置编码、无堆叠层；全部机制对应神经认知特性。

### 10.1 LLM 功能的类脑机制映射

| LLM 功能 | LLM 的实现 | PHD-Net 认知层的实现 | 认知对应 |
|---|---|---|---|
| 自然语言理解 | 自注意力 + 堆叠层学分布式语义 | M7 组合编码 + M2 层级压缩：字符 SDR 经 STDP 绑定为词/短语表示，预测编码层级逐级抽象 | 视觉/语言皮层的层级特征组织 |
| 上下文处理 | KV cache（无学习）+ 位置编码 | M8 情景缓冲（逐模式寻址 + 突触链时序）+ M10 分层摘要 | 海马情景记忆 + 前额叶分层工作记忆 |
| 知识存储与调用 | 参数隐式存储 + 注意力检索 | M9 语义关联图（三元组印迹）+ M12 联想链（多步激活） | 海马-皮层语义记忆网络 + 联想扩散激活 |
| 文本生成 | 自回归 + 注意力条件化 | M11 生成解码器：下行预测通路反向展开 + WM 主题锚定 + 调制噪声采样 | 内言语与运动皮层生成通路 |
| 逐步推理 | 思维链提示（仍是注意力） | M12 联想链：检索结果回填 WM 作为下一线索，迭代展开 | 人类联想式推理（扩散激活理论） |
| 指令遵循 | 监督微调 + RLHF | M5 调制门控 + 读出强化（设计层） | 强化学习与多巴胺门控 |

### 10.2 两项铁律的机制级替代（不是"不用 attention 的近似"，而是不同信息原理）

1. **自注意力 → 局部可塑性关联**：LLM 用全局成对相似度实现"任意位置关联"；
   认知层用**联想链**（检索→回填→再检索）与**情景缓冲索引**实现多跳关联——
   代价是逐步串行（对应人脑串行注意），收益是每步 O(活跃×m) 而非 O(T²)。
2. **位置编码 → 突触链时序**：LLM 把位置注入表示；这里的时间信息由
   **STDP 因果链**（先发-后发绑定）与情景缓冲的**链式回放**承载——顺序即连接，
   "第 k 个位置"不存在于任何向量中，而是链上第 k 次激活（实验 D 验证）。
3. **堆叠层 → 层级预测编码 + 摘要层级**：LLM 靠 N 层堆叠逐层抽象；
   这里抽象由 M2 预测编码层级（生成-识别闭环，非堆叠的残差流）与
   M10 分层摘要（句/段/篇递归压缩）实现——每层是独立脑区式模块，
   有自己的记忆与调制，而非同一残差流的重复块。

### 10.3 模块详解

**M7 组合编码器（词汇涌现）**
- 机制：字符 SDR 呈现时，M3 的 STDP 链把共现字符绑定为词级表示；
  词表示 = 该词字符序列在关联核中的激活轨迹终点（P5 已验证多跳链）。
- 认知对应：视觉词形区（VWFA）的组合调谐。状态：**已实现**
  （`phdnet/word_lm.py`：无词典统计分词实现词涌现——n-gram 频次 + 边界熵判据，
  为 STDP 共现绑定的工程等价物；词级 LM 见路线图 §八 M2 实测）。

**M8 情景缓冲（Episodic Buffer）——上下文的对应物**
- 机制：逐模式寻址的事件序列存储。每个事件（SDR）印迹到 LTM 并在缓冲中形成 STDP 链；
  任意历史事件可作为检索线索，链式回放恢复**顺序**。
- 对照 KV cache：KV cache 是无学习的原文堆叠；情景缓冲是**可学习、可补全、可链式回放**的
  记忆——回忆是重构而非复制（符合记忆的认知本质）。状态：已实现，实验 D 验证。

**M9 语义关联图（Semantic Graph）——知识的对应物**
- 机制：三元组 (主语 s, 关系 r, 宾语 o) 以组合线索印迹：W += η·outer(o, [s ; r])，
  其中 [s ; r] 为拼接绑定（组合编码，类脑中对应相位绑定/神经振荡同步的工程替代）。
- 检索：cue = [s ; r] → 吸引子收敛 → o；**无需注意力**，一次矩阵-向量乘。
- 容量：N=512 吸引子容量 ~70 三元组，可扩展到 10⁶⁺（事件驱动存储）。状态：已实现，实验 A 验证。

**M10 分层摘要（Hierarchical Summaries）——分层知识组织**
- 机制：P5 摘要槽的推广——句级摘要（短窗口压缩）→ 段级摘要 → 篇级主题锚（长期保留槽）。
  各层独立衰减率，形成"细节易逝、要旨长存"的梯度。状态：摘要槽机制已实现（P5），三级层级为配置扩展。

**M11 生成解码器（文本生成）**
- 机制：字符级 LM（P1）的读出头反向使用——每步生成：① 当前 WM 主题锚定；
  ② 读出分布 p(next|state)；③ 调制噪声采样（温度 τ 对应神经噪声/随机变异性）；
  ④ 生成字符回填输入，自回归但条件化机制是 WM 锚定而非注意力。状态：已实现，实验 C 验证。

**M13 上下文漂移情景记忆（长程依赖 / 串行回忆）** `phdnet/context_memory.py`
- **机制**：上下文随经历缓慢漂移（c ← ρ·c + (1−ρ)·item），并把「上下文 → 项目」印迹；
  回忆时用当前上下文做**内容寻址检索**，回忆出的项目又驱动上下文继续漂移 →
  **串行顺序回忆自然涌现**（TCM, Howard & Kahana 2002；CMR, Polyn et al. 2009）。
- **三个配套**：① **事件边界触发回放**（'#' 等边界触发链式回放）；
  ② **PFC 免衰减保持槽**维持情节起始上下文（延迟期持续放电），回放从本段开头开始；
  ③ **DG 模式分离**为每段情景混入去相关编码——**决定性组件**，关闭时准确率≈随机。
- **认知对应**：海马时间上下文与情景序列回忆 + 齿状回模式分离 + 前额叶延迟活动。
- **实测**：延迟复制位置准确率 n=4/8/16 = 45.6% / 25.9% / 17.8%
  （随机 8.3%，字符级 LM 基线 ~10%）——3.6× 随机；整段完全复现率仍低（n=4 约 7.5%）。

**M12 联想链推理引擎（逐步推理）**
- 机制：`state ← WM；for r in 链: state ← query(state, r)；state 回填 WM`。
  每步是局部联想检索，结果作为下一步线索——与"思维链"功能同构但机制为扩散激活。
  状态：已实现，实验 B 验证 2 跳演绎推理。

### 10.4 LLM 功能对照与差距评估

| 功能维度 | 当前已具备 | 与 LLM 的差距 | 差距性质 |
|---|---|---|---|
| 字符级语言建模 | ✓ PPL 99（优于 3-gram 5×） | GPT 级 PPL ~1.1/字符（英文） | **数据 + 规模**：需 MB 级语料与词级编码 |
| 词/语义理解 | 机制具备（M7），未训练 | 完整语义能力 | 数据 + 训练管线 |
| 上下文长度 | 情景缓冲无硬限（链式检索） | LLM 128K–1M token | 容量可扩展；长程连贯未验证 |
| 知识规模 | 机制具备（M9），实测 10⁰–10² 三元组 | LLM 万亿 token 隐式知识 | **数据 + 存储**，架构无障碍 |
| 多步推理 | ✓ 2–3 跳联想链（实测） | GPT-4 级复杂推理（>10 步 + 反事实） | 链深受吸引子噪声限制；需 M10 摘要辅助 |
| 文本生成 | ✓ 字符级自回归 + 主题锚定（实测） | 段落级连贯、指令跟随 | 语义级生成需词级 LM（M7 训练） |
| 对话/指令 | 设计层（M5 门控） | 完整对话能力 | 训练管线 + 意图表示 |

**诚实的总差距声明**："与 LLM 相当"的能力差距主要是**经验规模**而非**架构原理**：
LLM 的能力 = 架构 × 10²⁵ FLOPs 训练 × 万亿 token。认知层在架构层面闭合了四项功能的
机制对应物，并在小规模上逐项验证；要达到 LLM 级表现，需要（按序）：
MB→GB 级语料、词级编码（M7 训练）、10⁹⁺ 突触知识印迹、以及云端算力——
这些是工程与资源问题，不受两条铁律约束。

### 10.5 实验验证结果（demo_v2.py 实测，2026-09-18）

| 实验 | 验证机制 | 实测结果 | 判定 |
|---|---|---|---|
| A 知识存储与调用 | M9 三元组印迹 + 检索（12 条干扰竞争） | **4/6 命中（67%）** | △ 接近目标；小样本波动 ±16%，扩容与实体正交化可继续提升 |
| B 逐步推理 | M12 联想链 2 跳演绎 | **链式 3/4（75%）vs 直接查询 0/4（0%）**——未存储的事实（"苏格拉底会死"）由链式检索推断得出 | ✓ 通过（核心对照显著） |
| C 文本生成 | M11 温度采样 + 主题锚定 | **2-gram 合法率 98.1% vs 随机对照 1.0%**；生成样本为字符级伪文本（语料为 markdown 技术文档，含表格符号，如实反映字符级水平） | ✓ 机制通过（语义级生成需词级 LM） |
| D 顺序上下文 | M8 非对称外积联想链（无位置编码） | **链式回放 5 步位置重合度 100%** | ✓ 通过 |

**关键对照**：实验 B 中，未印迹的组合（苏格拉底 × 具有）直接查询命中率 0%，
而联想链（苏格拉底→是→人→具有→会死）达 75%——**"推理"作为独立于"存储"的
能力被机制性证明**：新事实由既有关联的串行激活合成，这正是逐步推理的类脑本质。

工程说明：实体表示必须使用确定性哈希（`hash()` 进程间随机化会导致不可复现）；
M8 采用与 M9 同构的非对称外积联想链（STDP 拓扑核的稀疏出边会使回放分布散布，
重合度仅 41%，已替换）。

---

## 12. 当前状态（2026-09-21 终态重测）

### 12.1 M9 收官结论（五轨道）

重测条件：冻结语料 `eval_corpus/internal_corpus.txt` **23,504 字符**（训练 18,803 / 评估 4,701）、seed 11、
128 维小栈词表口径 OOV 0%，参照基线 R = **102.97**（bpc 6.686）。结果写入 `outputs/test/demo_m9_result.json`。

| 配置 | 字符归一 PPL | 相对 R | 结论 |
|---|---|---|---|
| **R 参照（W2，开关全关）** | 102.97 | — | 对拍基准，bpc 6.686 |
| T3.4 内容寻址 + WM×2 | **101.29** | **−1.6%** | 本轮最优，稳定正增益 |
| ② 回放稳定读出 | 101.48 | −1.4% | 正增益（四轮均为增益：−0.8% → −1.34% → −1.5% → −1.4%） |
| T4.4 验证驱动睡眠 | 102.97 | +0.0% | 中性 |
| T3.2 top-k 检索融合（k=8） | 106.24 | **+3.2%** | **符号翻转转负（有害）**：历史 −2.4% → −0.65% → −1.1% → +3.2%，不推荐启用 |
| T2.2 可学习稀疏词典 | 121.88 | +18.4% | 负 |
| ① 发育期巩固（1500 步后冻结） | 125.60 | +22.0% | 负 |
| ①+T1.2 预测主目标化 | 125.61 | +22.0% | 负 |
| V4′ 全程可塑（η_pc 0.02 + 稳态 + 退火） | 126.73 | +23.1% | 负 |
| ③ 逐神经元目标发放率 | 167.72 | +62.9% | 负（最差） |

其他：T3.3 情景缓冲接入生成，bigram 合法率 96.7% / 92.5% / **100.0%**（len=0/2/4）。
容量赛道：int8 量化 cos = **0.9999**（max|Δ|=0.0142）、值内存 190.1 → 63.4 KB（**3.0×↓**）、CSR 快照 8,112 条；
拓扑生长引导入度上限 **32**（均值 17.95，结构生效但因果中性 0.50/0.50）；
稀疏 PC 25.80 → 13.49 ms/步（**1.91×**，单步相对偏差 4.0%）。

> ⚠ **排序对语料敏感，请勿据单次数字下结论**：检索/记忆三项（T3.2 / T3.4 / ②）在四轮语料上排序反复反转，
> 且 T3.2 在冻结语料上**符号翻转为有害（+3.2%）**——检索融合增益不可依赖；
> T3.4 与 ② 四轮均为增益（幅度 −0.2% ~ −1.6% 噪声带内）。
> 唯一超稳定结论仍是：**表征可塑性五项跨 M1/M8/M9 多轮全负**，`eta_pc=0` 仍是硬约束；
> 在 256 维默认栈上，PHD-Net 已优于同预算 Transformer（见 §12.3）。

> 完整矩阵与组成 ablation 见《对标 Transformer 优化路线图》§三。

### 12.2 开放项（下一阶段）

| # | 开放项 | 卡点 |
|---|---|---|
| O1 | 双栈完整统一（T5.1 剩余） | 稠密 PC 千维以上 O(n²)；1B 印迹表仍为 dict 邻接 |
| O2 | 表征可塑性解锁 | 不是幅度而是稳定性问题（灾难性遗忘 / 行范数震荡） |
| O3 | 100B 在线 CSR 迁移 | dict 邻接约 100+ B/条目 → 100B 规模不可行 |
| O4 | 分词器向量化 | `WordSegmenter` 为 O(候选×n) 纯 Python，全量语料评估需数小时 |
| O5 | 长程**整段**完全复制 | 位置准确率尚可（45.6/25.9/17.8%），整段复现率仍低（n=4 约 7.5%） |
| O6 | 脑同源机制的功效消融 | 12/12 模块对齐 ≠ 有贡献，需逐模块敏感度排序 |
| O7 | PHD-Net 自身 scaling curve | 目前只有 ~0.2M 一个稠密点，尚无 α 值（O1 前置） |

### 12.3 语料与评测口径

| 语料 | 路径 | 规模 | 用途 |
|---|---|---|---|
| 内置（冻结） | `eval_corpus/internal_corpus.txt` | **23,504 字符**（逐字复制自本文 2026-09-21 版） | 默认回归与 ours 对拍（**编辑本文不再影响语料**） |
| 探针（远域） | `eval_corpus/ood_wiki.txt` | 26 篇 / 20,188 字符（ModelScope Range 抽取） | 泛化探针 `tests/demo_gen.py` |
| 外部预训练 | `datasets/pretrain/` 或 `--remote-data`（ModelScope `fhzfhz/Mixture-General-Mini`）：**Infinity-Instruct M7_Core**（7.45M dialogues）+ **Magpie-R1**（201 万条英文 CoT）+ **Ultra-FineWeb-L3 中文**（8 万） | — | 预训练 / 跨轮稳定锚点（维基语料已按指令删除） |

统一口径：**字符归一 PPL** = exp(总 token NLL / 评估段字符数)，并报 bpc。
Transformer 对照为自建 nanoGPT 级模型（PyTorch，**0.52M（96d×2L）/ 2.38M（192d×4L）**），须在**同语料**下重测才可比。

> ⚠ **同文档也存在两种口径，勿混用**：
> - **默认 256 维栈**（`tests/eval_common.py` BASE：n_sdr=256 / k_sparse=32 / eta_pc=0 / **eta_readout=0.15**、读出 fp32；复测入口 `tools/rebaseline.py`）→ **现行锚点：4,000 字符段 394.4687（bpc 8.624）／全语料 21,924 字符 359.2603（bpc 8.489）**，评测语料 2026-09-28 更换为中文维基高质量条目合集（旧口径 78.16 / 90.2480 / 73.1166 仅存于 git 历史，不可混用），`tests/eval_suite.py`
> - **M9 消融 128 维小栈**（`tests/demo_m9.py` BASE：n_sdr=128、分词 min_count=8）→ 冻结语料当前 R = **102.97**（bpc 6.686）
>
> 两者网络宽度与分词粒度都不同，绝对数值不可比；跨组做除法得到的"提升倍数"没有意义。
> 泛化能力结论（域内良好 / 近域优 / 远域受 OOV 覆盖墙限制）见《性能评估与迭代方案》§5.4。

## 附：词表查找结构（P15，2026-09-28）

分词热路径的查找结构按**词表规模自适应**（`train_1b/tokenizer_core.py`）：

- **CSR trie + 二分**（边数 ≤ 2 万）：小词表下二分常数更小；
- **边哈希**（边数 > 2 万）：键 = `node·2¹⁶ + char`，槽位经 Knuth 乘法混合
  （直接取低位会让「同字不同节点」全部撞槽，探测退化——实测建表 265s）；
  大词表下根扇出数千，二分退化为每层 log2(fanout) 次 cache-missy 比较。

两条路径**逐位等价**（同一 trie、同一最长匹配规则），由
`tests/verifiers/verify_vocab_parallel.py` 的 24 例对拍保证（含 6 例
极小词表/低扇出用例——低扇出正是旧二分实现漏匹配的暴露面）。

## 附：词表来源与设备张量兼容（P17，2026-09-28）

**词表快照（P17，fhz「词表做出来第一时间存入 outputs/models」）**：
词表一确定就落盘 `outputs/models/vocab_<preset>_<data>.json`（权威）+ `.txt`（人读镜像），
训练崩溃/中断也不丢；下次 `--vocab-file <同名>.json` 直接复用，免重扫（full 扫描 520s）。
快照**必须同时存两个集合**：`words`（token 词表 = 读出层行数 = n_readout）与
`seg_vocab`（分词器候选集，训练时是涌现词表，通常是真子集）——用 token 词表
代替候选集会让分词结果与训练不一致（静默降质）。另：词表可能含**跨行 token**
（`"\n的"` 等），故 JSON 为权威格式（纯文本每行一词会把它切碎：实测 2,610 → 2,571）。
推理侧 `infer.py` 自动定位同目录 / `outputs/models/` 的快照并**交叉校验**
（比对 `tok_tokens`），不一致即报错退出——防「用错词表静默降质」。

**词表来源三选一**（优先级从高到低，`train_1b/train.py`）：

1. `--resume` + 检查点存在 → **跳过构建**，用检查点内 `tok_vocab` /
   `tok_tokens` / `tok_max_len`（`ckpt_1b.peek_tokenizer` 只读取词表以对齐
   `cfg.n_readout`，随后 `load_model` 幂等恢复分词器与 SDR 哈希）；
2. `--vocab-file vocab.txt` → 外部词表（每行一词，`#` 注释与空行忽略）；
3. `--vocab-scan head|full` → 原有扫描（head = 采样文本，多核 token 收集；
   full = 全量锚点链扫描，多核 nogil 线程）。

**设备张量兼容**：`phdnet/model.py::to_numpy()` 与 `_nelem()` 统一把
numpy / torch（含 CUDA·NPU·ROCm 设备张量）转 numpy 再统计或存档。
动机：昇腾机器（CANN 8.5 aarch64 + torch_npu）实测 `np.asarray(npu_tensor)`
直接抛错，会让参数统计与检查点保存双双失败。

## 附：部件 × 执行后端总览（P22，2026-09-28；P30 复核 2026-09-29）

| 部件 | 执行后端 | 并行方式 | 备注 |
|---|---|---|---|
| 词涌现（`_induce_length`） | numba nogil 核 | 层间线程池（5 个 L 各占一核） | 核内串行（prange 已回退：实测 1→6 线程仅 1.16×，内存带宽饱和） |
| 词表全量扫描（锚点链） | numba nogil 核 | `ThreadPoolExecutor`（组间，零 pickle） | 小词表 CSR 二分 / 大词表边哈希自适应 |
| 语料解码（parquet） | pyarrow（释放 GIL） | 单进程内多线程（默认 1 进程） | 多进程只增内存，不增吞吐 |
| 分词（训练循环内） | 纯 Python | — | 占端到端 ~0.0%，不优化 |
| 读出（M6，占 1B 预设 ~89%） | **auto**：`AccelReadout`（torch 设备：昇腾 NPU/ROCm/CUDA/DirectML）→ 否则 numba CPU | 单设备（W 常驻设备）；多卡为读出列并行（`multi_device.MultiDeviceReadout`） | `PHDNetConfig.accel_readout`，W 巨大且逐 token 只传 h；P28：`addmm_` rank-1 AXPY + 设备侧 (h,y) 缓存 + 硬同步 3→1，流量 4,334→2,600 MiB/步（−40%），数值逐位相同 |
| PC 栈 / STDP / LTM | numba（事件驱动稀疏 + 在线 CSR） | 单核（内含 prange 融合核） | 旧 torch 栈缺 `big_ltm` 等 7 项机制，已于 P30（2026-09-29）整体删除，不再有迁移分叉 |

**为什么是这个组合**：numba 只能编译到 CPU 机器码（物理限制），所以「能上设备的部件」
必须用 torch 写。当前只有读出满足（纯矩阵、W 大、每步通信 ~12 KB）。其余部件要么是
变长字符串（分词/词表扫描，设备不划算且占比极低），要么是事件驱动稀疏结构
（旧 torch 整栈迁移会丢机制，该栈已于 P30 删除）——这是当前的诚实边界，不是未做清单。
