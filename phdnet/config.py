"""全局超参配置 —— 对应架构文档 §8 超参数表。"""

from dataclasses import dataclass


@dataclass
class PHDNetConfig:
    # M1 稀疏编码器
    n_input: int = 32          # 原始输入维度
    n_sdr: int = 256           # SDR 稀疏码维度 N0
    k_sparse: int = 16         # 稀疏度（激活率 **6.25%**，更接近皮层 1–5% 稀疏发放）
    # 2026-09-24 起默认 16（原 32 = 12.5%）：fhz 指令「稀疏要默认开启」。
    # 实测依据（4,000 字符口径）：k=16（6.2%）→ PPL **−1.43%**（优于 k=32）；
    # k=8（3.1%）→ +12.5%（容量不足，有害）。评测脚本显式传 k_sparse=32，
    # 故该默认变更**不影响既有评测口径**，仅作用于未显式指定的调用。

    # M2 预测编码层级
    n_mid: int = 128           # 中间层维度 N1
    n_top: int = 64            # 顶层维度 N2
    n_infer_steps: int = 1     # 误差驱动精炼步数（自由能下降迭代）
    eta_pc: float = 0.02       # 生成权重学习率（发育期快速形成表征）
    eta_oja: float = 0.02      # 识别权重 Oja 学习率（Hebb+归一化，防发散）

    # M3 时序关联核（STDP）
    m_lateral: int = 16        # 每神经元出边数（稀疏拓扑）
    lambda_trace: float = 0.35  # 突触迹衰减（时间窗宽度，过宽会引入隔步混淆）
    eta_stdp: float = 0.03     # STDP 学习率
    w_max: float = 1.0         # 权重截断上限（STDP 核；PC 主干用 pc_w_max）
    pc_w_max: float = 2.0      # PC 主干（稠密/稀疏）权重截断上限——2026-09-28 修复：
                               # 此前 cfg.w_max 从未传入 PC（签名默认 2.0 静默生效），
                               # 现显式化为独立字段，默认值 = 历史实际行为（逐位不变）

    # M4a 工作记忆
    n_wm_slots: int = 4        # 槽位数（有限容量 ≈7±2 的抽象）
    gamma_wm: float = 0.85     # 每步衰减（模拟持续放电的漏积分）
    gate_thresh: float = 0.35  # 写入门限（基底核门控）

    # M4b 长期记忆（双速率）
    eta_hip: float = 0.08      # 海马快速印迹率
    eta_cortex: float = 0.05   # 皮层慢巩固率
    beta_fast: float = 0.6     # 检索时快权重配比
    beta_slow: float = 1.0     # 检索时慢权重配比
    n_recall_steps: int = 6    # 吸引子迭代步数

    # M5 神经调制器
    mod_gain: float = 1.5      # z 分数 → 门控的 sigmoid 增益
    seed: int = 7

    # P3 自动发育调度（默认关闭 = 手工 learn_pc/learn_stdp 开关，非破坏）
    auto_development: bool = False  # True = 依表征稳定度自动门控 PC/STDP 学习
    dev_gain: float = 3.0      # 不稳定度 → 门控的缩放增益
    dev_alpha: float = 0.05    # 不稳定度 EMA 平滑系数
    dev_floor: float = 0.1     # STDP 门控下限（表征未稳定时保留微弱学习）
    eta_dev_pc: float = 0.02   # 自动模式下 PC 学习率（表征期快速发育）

    # M6 读出头（P1 语言接口扩展；默认值保持旧行为，非破坏）
    n_readout: int = 0          # 读出输出维度；0 = 与 n_input 相同（旧行为）
    readout_softmax: bool = False  # True = 下一 token 预测用 softmax 感知器（交叉熵局部梯度）
    eta_readout: float = 0.15   # softmax 感知器学习率（fhz 2026-09-27 拍板：0.05 → 0.15，实测最优区间 0.15–0.20，0.25+ 退化，0.5 发散）
    eta_readout_anneal: float = 0.0  # 读出学习率每步乘性退火（0=关闭；如 0.9997）
    eta_readout_floor: float = 0.004 # 退火下限（发育后巩固，防后期样本覆盖早期学习）
    readout_w_clip: float = 0.0 # C8：读出权重范数上限（0=关闭，保持旧行为）

    # 路线图 M1 升级（2026-09-18，三项开关默认关闭 = 旧行为，非破坏）
    task_modulation: bool = False    # T1.3 误差广播调制：读出 NLL z 分数门控 PC/STDP 学习率
    error_gated_memory: bool = False # T1.3 错误驱动写入：WM 写入/LTM 印迹门控改用任务误差（滞后一步）
    ltm_imprint_gate: float = 0.8    # LTM 印迹的 mem_gate 阈值（2026-09-25 审计：默认 0.8 下
                                     #   调制器 z 分布 std≈0.32 → gate>0.8 仅 ~0.4% 触发，
                                     #   big_ltm 容量栈在 LM 训练中几乎从不印迹；调低以激活）
    pred_in_readout: bool = False    # T3.1 预测通路三拼：h = [r2 ; WM ; STDP 时序预测]
                                     #   （首版用 LTM 原始召回向量，实测噪声压垮读出头，已修订）
    task_mod_gain: float = 2.0       # 任务误差 z → 门控的 sigmoid 增益
    task_scale_floor: float = 0.3    # 任务门控下限（预测良好时保留的基础学习率比例）

    # 路线图 M3 优化调度（2026-09-18，默认关闭 = 旧行为，非破坏）
    adaptive_lr: bool = False        # T4.1 逐突触自适应学习率（局部二阶矩归一，突触自稳态）
    adapt_rho: float = 0.9           # T4.1 二阶矩 EMA 系数
    adapt_eps: float = 0.5           # T4.1 分母稳定项（取 0.5 使初始尺度≈1，便于消融可比）
    dual_trace: bool = False         # T4.2 多尺度双迹（短窗 LTP/长窗巩固）
    lambda_trace_slow: float = 0.85  # T4.2 长窗迹衰减
    beta_slow_trace: float = 0.5     # T4.2 长窗迹在组合迹中的权重

    # 硬件后端（2026-09-18；默认不启用 torch 后端 = 旧行为，非破坏）
    backend: str = "auto"        # auto（无加速器则 numpy）/ numpy / torch / npu / rocm / cuda / cpu
    torch_dtype: str = "float32"  # torch 后端数据类型（昇腾部分型号建议 float16，需另行验证）

    # 架构强化（2026-09-18；默认关闭 = 旧行为，非破坏）
    homeostasis: bool = False        # 稳态突触缩放：把 PC 各行权重范数拉回初始值，
                                     # 稳定 Hebb/误差驱动的在线学习（Turrigiano 2008）
    error_triggered_retrieval: bool = False  # 错误触发检索：任务误差高时额外检索（T3.2-lite）

    # P5 摘要槽接线（0 = 关闭，保持旧行为；>0 = 每 N 步压缩 WM 内容进保留槽）
    wm_summary_every: int = 0

    # 路线图 M4 容量栈（默认关闭 = 稠密 LTM 旧行为，非破坏）
    big_ltm: bool = False            # True = 长期记忆改用大空间事件驱动印迹表
    big_ltm_N: int = 1 << 24         # 大空间神经元数（× big_ltm_m = 突触容量）
    big_ltm_m: int = 60              # 每神经元可塑出边上限（60 × 2^24 ≈ 1.01B）
    big_ltm_k: int = 4               # 每个表示维度哈希到大空间的神经元数
    sparse_table_prune: int = 0      # C1：迹/时间戳字典惰性淘汰阈值（0=关闭，保持旧行为）

    # 维护性与资源开关（默认保守，保持旧行为）
    consolidate_every: int = 1       # C2：每 N 次 sleep 才执行一次 consolidate（1=每次，旧行为）

    # ───────────────────────────────────────────────────────────────────────
    # 脑同构统一（2026-09-19，用户确认"全部可行项"）：5 项机制，全部默认关闭 =
    # 旧行为（非破坏）。每项均经 tests/run_tests.py fast 零回归验证。
    # ───────────────────────────────────────────────────────────────────────

    # (1) M5 多通道神经调制：单标量 surprise→gate 升级为 ACh/NE/DA/5-HT 四通道
    #     脑对应 Hasselmo（ACh 编码/检索切换）、Yu & Dayan（NE 新颖性→可塑性、
    #     DA 巩固、5-HT 耐心）。
    multi_modulation: bool = False   # True = Neuromodulator 替换为 MultiModulator
    gain_ach: float = 1.5           # ACh 编码门控增益（高 surprise → 编码模式）
    gain_ne: float = 2.0            # NE 新颖性增益（|Δsurprise| → 可塑性）
    gain_da: float = 1.5            # DA 巩固增益（低 surprise → 巩固/回放）
    gain_ht: float = 1.0            # 5-HT 耐心增益（低唤醒 → 延长检索等待）

    # (2) M3 稳态突触缩放扩展至 STDP + 元可塑性（BCM 滑动阈值）
    #     针对 PC/表征发散与灾难性遗忘根因（此前被迫冻结 eta_pc=0）。
    stdp_homeostasis: bool = False  # True = STDP 出行权范数超阈则拉回（防发散）
    homeo_target: float = 2.0       # STDP 行范数上限（超则按比例 downscale）
    metaplasticity: bool = False    # True = BCM 元可塑性：LTP 受滑动阈值 θ 门控
    bcm_tau: float = 0.01           # θ 二阶矩 EMA 系数

    # (3) M4b 分期睡眠 SWS/REM + 突触 downscaling
    #     脑对应 Tononi & Cirelli（突触稳态假说：SWS 整体 downscaling）
    #     + 标准两阶段睡眠（SWS 巩固 / REM 重组回放）。
    staged_sleep: bool = False      # True = sleep() 交替 SWS(巩固+downscale)/REM(回放)
    sleep_downscale: float = 0.9    # SWS 突触 downscaling 因子（<1）

    # (4) 抑制/兴奋（E/I）突触类型
    #     脑对应皮层 ~80% 兴奋 / ~20% 抑制；抑制性出边权重为负（侧向抑制）。
    ei_synapses: bool = False       # True = 分配 E/I 类型，I 出边符号翻转+负权重
    ei_ratio: float = 0.2           # 抑制性神经元占比

    # (5) 发育期临界期 + 突触修剪
    #     脑对应发育临界期（经验累积后关闭可塑性窗口）+ 突触消除（低效连接剪除）。
    critical_period: bool = False   # True = 学习率随经验累积衰减（临界期关闭）
    crit_rate: float = 0.001        # 经验→学习率衰减系数
    prune_threshold: float = 0.01   # 突触修剪阈值（|w| 低于则归零）

    # ───────────────────────────────────────────────────────────────────────
    # 路线图第二轮全量优化（2026-09-19，M9）：补齐剩余轨道
    # T1.2 / T2.2 / T3.2 / T3.3 / T3.4 / T4.4 / T5.1–T5.3
    # + PC 突破三方向 + 情景记忆双向校验。全部默认关闭 = 旧行为（非破坏）。
    # ───────────────────────────────────────────────────────────────────────

    # T1.2 预测编码主目标化：生成权重学习「下一时间步」的表示（时序预测目标），
    #      与逐步重构目标凸组合（pc_pred_mix：0=纯重构旧行为，1=纯预测）。
    #      仅在 eta_pc>0（表征可塑）时有作用；脑对应皮层生成模型的时序预测。
    pc_predictive_target: bool = False
    pc_pred_mix: float = 1.0        # 预测项权重（1−mix 为重构项权重）

    # T2.2 可学习稀疏词典：M1 W_enc 由 Foldiak 局部规则慢调
    #      （活跃单元 Oja 更新 + 行范数稳态双约束，防 Oja 塌缩）。
    #      脑对应初级皮层 simple cell 感受野学习。
    learnable_encoder: bool = False
    eta_enc: float = 0.01           # 编码器学习率（慢调）

    # T3.2 检索常态化 + top-k 融合：>0 = 每步检索 LTM，把 |score| 最大的
    #      top-k 召回线索（归一化）拼入读出特征（PFC 读出聚合情景线索）。
    #      扩散激活的持续背景检索；0 = 关闭（每 8 步定频旧行为）。
    retrieval_topk: int = 0
    retrieval_gain: float = 1.0     # 融合特征缩放

    # T3.4 WM 内容寻址写入：写入目标从「最弱槽位」改为
    #      「与新内容最相似的槽位（相似度 ≥ 阈值）否则最弱槽位」——
    #      相似内容聚合、减少同义槽位碎片化。扩容直接调 n_wm_slots。
    wm_content_address: bool = False
    wm_sim_thresh: float = 0.6      # 内容寻址命中阈值（余弦相似度）

    # T4.4 验证驱动睡眠：train_stream 内按验证 PPL 平台期触发 sleep()
    #      （连续 plateau_patience 段无改善 → 巩固）。睡眠稳态调节的科学化时机。
    plateau_sleep: bool = False
    plateau_patience: int = 2       # 连续 N 个验证段无改善才触发

    # O1-2：主干**结构性稀疏连接**（sparse connectivity）。
    # 脑对应：皮层组织原则是 dense representation + sparse connectivity ——
    # 单个锥体神经元仅 ~10³–10⁴ 突触（潜在靶点 ~10¹¹，连接率 ~10⁻⁵～10%），
    # 权重的存在性是结构性的（不存在的突触不存储、不计算），而非"零值稠密矩阵"。
    # **2026-09-24 起默认 True**（fhz 指令「稀疏要默认开启」）：
    #   实测依据——连接率 12.5%（k=32）时 PPL 与稠密持平（+0.03%~−0.30%），
    #   k=16（6.25%）时 −0.51%；突触存储压缩 6.7–32×、PC 层加速 1.68–4.02×。
    # **2026-09-28 起（fhz 指令「删稠密功能」）稠密栈已删除**（phdnet/pc.py 移除），
    #   主干唯一实现为 SparsePCStack；本字段仅为配置兼容保留，False 会 fail-fast。
    sparse_conn: bool = True
    conn_k: int = 0             # 每神经元入边数 k（0 = 自动取各层 n_in//8 ≈ 12.5%）

    # O1-3：**读出头的结构性稀疏连接**（默认关闭）。
    # 脑对应：皮层→输出投射同样稀疏（每个下游细胞只接收部分上游输入），
    # 而非稠密全连接矩阵。启用后读出权重以 CSR 承载（每输出单元 k 条存在的入边），
    # 学习规则不变（softmax 交叉熵的末端梯度），仅连接存在性结构改变。
    readout_conn_k: int = 0     # 0 = 稠密读出（旧行为）；>0 = 每输出单元 k 条入边

    # O1-4：**皮层式权重初始化**（默认关闭）。脑对应：皮层突触强度呈对数正态/重尾
    # （少数强连接 + 大量弱连接，Song et al. 2005），且兴奋/抑制比 ~80/20（E/I 平衡）。
    # 高斯对称初始化会低估强连接、且 E/I 比固定 50/50。
    lognormal_init: bool = False  # True = 主干/读出权重按 lognormal(重尾) × E/I 符号初始化
    exc_ratio: float = 0.8        # 兴奋性（正权重）比例（皮层 ~0.8）

    # A4：**两级群体读出**（默认 0 = 单层线性，旧行为）。脑对应：皮层→输出的投射
    # 是多级（多突触中继）且靠群体编码，而非单层线性分类器。
    # 第一级 h → 隐藏群体（稀疏投射 + k-WTA 侧抑制竞争，局部无监督 Oja 学习）；
    # 第二级 群体 → 输出（稀疏投射 + 任务监督）。同时改善 A4（多级/稀疏）
    # 与 A7（误差只到第二级，不反传第一级/主干）。
    readout_hidden: int = 0       # >0 = 隐藏群体规模（如 512）
    readout_hid_k: int = 0        # 第一级每隐藏单元入边数（0 = n_in//8）
    readout_eta_hid: float = 0.002  # 中间层局部 Hebbian/Oja 学习率

    # T5.2 拓扑生长引导：大空间表新突触生长偏好低入度目标神经元
    #      （发育期突触生长的结构引导，避免 hub 过载）。
    growth_guidance: bool = False

    # T5.3 稀疏量化存储：印迹表权重 int8 量化存储（值对象内存 ~×8↓），
    #      另提供 compact_csr() 扁平快照（CSR）供读密集阶段使用。
    sparse_int8: bool = False

    # PC 突破①：发育期后强制巩固表征——前 pc_dev_steps 步 PC/编码器可塑，
    #      之后冻结（eta_pc/eta_oja/eta_enc → 0），针对「表征漂移 × 恒定
    #      读出率 → 灾难性遗忘」的根因（M8.A 诊断链的对症机制）。
    pc_dev_steps: int = 0

    # PC 突破②：回放稳定读出——储备库保存旧 (输入, 目标) 对，每 replay_every
    #      步把一条旧输入在**当前表征**下前向（只读）并重训读出——
    #      以旧目标重新锚定漂移后的表征，缓解读出灾难性遗忘。
    readout_replay: bool = False
    replay_buffer: int = 256        # 储备库容量（旧输入条数）
    replay_every: int = 4           # 每隔多少步重演一条
    replay_eta: float = 0.5         # 回放读出学习率倍率（防回放主导）

    # PC 突破③：逐神经元目标发放率（内在可塑性）——每神经元维护激活 EMA，
    #      按目标率温和缩放其上行输入行（细粒度稳态，補 k-WTA 的全局稀疏）。
    neuron_target_rate: bool = False
    target_rate: float = 0.06       # 目标激活率（≈ k-WTA 的 1/16）
    eta_homeo: float = 0.01         # 激活 EMA 与增益步长

    # T6 后续：情景记忆双向校验（item→context→item 一致性检验，
    #      低置信回忆按一致性降权——串行回忆级联误差的对症机制）。
    bidir_check: bool = False
    bidir_w: float = 1.0            # 一致性权重指数（0=不降权）

    # T3.3 情景缓冲接入生成：词级生成时每步把最近 episodic_len 个已生成
    #      token 作为「情节流」重放观察（学习关闭），再采样下一 token——
    #      内言语受情景流约束。0 = 关闭（仅 WM 锚定旧行为）。
    episodic_len: int = 0

    # ───────────────────────────────────────────────────────────────────────
    # 工程收口（2026-09-22）：开放项 O1/O3/O5 与瓶颈 B7 的机制开关。
    # 全部默认关闭 / 默认值 = 旧行为（非破坏）；默认路径逐位不变。
    # ───────────────────────────────────────────────────────────────────────

    # O1/O3：稀疏印迹表——在线可写 CSR + 分片 mmap + 稀疏 Top-k 误差反传。
    # 脑对应：突触结构的在线生长（结构可塑性）+ 长时程存储的物理分片。
    csr_online: bool = False        # True = 印迹表切换为在线可写 CSR（定长行 + 预留槽）
    csr_grow_chunk: int = 64        # CSR 行初始容量（在线追加预留槽数）
    sparse_mmap_dir: str = ""       # 非空 = 按分片 mmap 落盘（大容量表的物理分片存储）
    sparse_mmap_shards: int = 8     # 分片数（每片独立文件，按哈希归属）
    sparse_topk_backprop: int = 0   # >0 = 稀疏 Top-k 误差反传（k；0 = 关闭，旧行为）

    # O5：长程整段复制——级联误差治理（读出再入 + 分段校验）。
    # 脑对应：PFC→感觉皮层的反馈再入（reentry）与序列分段边界校验。
    # 注意：不引入位置编码 / 顺序无关性（铁律），仅做输出回灌与段内一致性约束。
    readout_recurrence: bool = False  # True = 读出预测回灌为下一步 WM 线索（再入连接）
    recur_gain: float = 0.5           # 回灌强度（0=不回灌；1=全量替换线索）
    segment_check: int = 0            # >0 = 每 N 步做段内一致性校验，低置信输出降权

    # B7：统计效率——读出更新 minibatch 化（累积 N 步梯度后平均更新）。
    # 脑对应：突触巩固的时间整合（单次事件不足以触发长期可塑性）。
    minibatch_size: int = 1           # 1 = 逐步更新（旧行为）；N>1 = 累积 N 步平均后更新

    # P6b：读出权重 float32 化（性能开关，默认关闭）。
    # 动机：P6 剖析显示读出学习占训练热路径 79%（词级读出 9.46 MB float64
    # 矩阵的稠密更新），其耗时受内存带宽支配 —— float32 使带宽减半。
    # 数值影响：读出前向/更新的浮点精度由 fp64 降为 fp32（属数值变化行为，
    # 故默认关闭；默认路径逐位不变）。预计训练加速 ~1.3–1.5×。
    readout_dtype: str = "fp32"     # P9 精度体系（fhz 2026-09-26：停止 fp64；默认 fp32）
    # P19 读出加速器：auto=有加速器就用（昇腾→ROCm→CUDA→DirectML），否则回落
    # numba CPU 原路径（逐位不变）；cpu/off/numba = 强制原路径；
    # npu/cuda/rocm/dml = 显式设备（不可用则回落并如实报告）。
    accel_readout: str = "auto"
                                    # fp4 已禁用（fhz 2026-09-27）：+5.88% 劣化，待 MX 块缩放；代码保留
                                    #   可选 fp32 / fp16 / bf16 / fp8 / fp4
                                    #   低精度 = 原生位型码本存储 + 查表反量化计算
                                    #   （softmax/NLL 保持 fp64 主回路）


    def __post_init__(self):
        # 2026-09-28 修复：k_sparse > n_sdr 时 SparseEncoder 的 argpartition kth
        # 越界（构造成功、首个 step 才崩）。fail-fast 提前到配置期。
        if self.k_sparse > self.n_sdr:
            raise ValueError(
                f"k_sparse({self.k_sparse}) 不得超过 n_sdr({self.n_sdr})"
                "——k-WTA 每层只有 n_sdr 个神经元可选。")
        # 2026-09-28（fhz 指令「删稠密功能」）：稠密 PC 栈已删除，主干唯一实现
        # 为 SparsePCStack；sparse_conn=False 显式拒绝（不做静默稀疏化）。
        if self.sparse_conn is not True:
            raise ValueError(
                "sparse_conn=False 已不可用（fhz 2026-09-28 删除稠密 PC 功能）："
                "主干唯一实现为 SparsePCStack（CSR 稀疏图），请使用默认 True。")
