"""M2 预测编码层级 —— 对应皮层层级（V1→V2→V4→IT）的预测编码
（Rao & Ballard 1999；Friston 自由能原理）。

架构对应（文档 §2 M2）：
    上行识别：r_{l+1} = tanh(W_up · e_l)     自下而上误差流
    下行生成：ŷ_l = W_dn · r_{l+1}           自上而下预测流
    误差：    e_l = s_l − ŷ_l                只把"意外"上传
    学习：    ΔW_dn = η·e ⊗ r_hi（误差最小化）
              ΔW_up = Oja 规则（Hebb + 归一化，防发散）
"""

import numpy as np


class PredictiveCodingStack:
    def __init__(self, n0: int, n1: int, n2: int, eta_pc: float, eta_oja: float,
                 rng: np.random.Generator, w_max: float = 2.0,
                 sparse_pc: bool = False, topk: int = 0,
                 lognormal_init: bool = False, exc_ratio: float = 0.8):
        self.eta_pc, self.eta_oja, self.w_max = eta_pc, eta_oja, w_max
        self.sparse_pc = sparse_pc            # T5.1（lite）稀疏学习更新（默认关闭）
        self.topk = int(topk)                 # O1：稀疏支撑集精确 k（0 = 默认 1/8 比例）
        # 识别（上行）权重：感觉通路
        if lognormal_init:                    # O1-4：皮层式（重尾 + E/I 比）
            from .inits import cortical_init
            self.W_up0 = cortical_init(rng, (n1, n0), 1 / np.sqrt(n0), exc_ratio)
            self.W_up1 = cortical_init(rng, (n2, n1), 1 / np.sqrt(n1), exc_ratio)
        else:
            self.W_up0 = rng.normal(0, 1 / np.sqrt(n0), (n1, n0))   # e0/s0 -> r1
            self.W_up1 = rng.normal(0, 1 / np.sqrt(n1), (n2, n1))   # r1 -> r2
        # 生成（下行）权重：反馈通路，以识别权重的转置初始化（生成-识别共享先验）
        self.W_dn0 = (self.W_up0.T * 0.5).copy()                 # r1 -> ŝ0 (n0, n1)
        self.W_dn1 = (self.W_up1.T * 0.5).copy()                 # r2 -> r̂1 (n1, n2)
        # 稳态目标：各行权重的初始 L2 范数（突触缩放把它们拉回此值）
        self._hn_up0 = np.linalg.norm(self.W_up0, axis=1)
        self._hn_up1 = np.linalg.norm(self.W_up1, axis=1)
        self._hn_dn0 = np.linalg.norm(self.W_dn0, axis=1)
        self._hn_dn1 = np.linalg.norm(self.W_dn1, axis=1)
        # PC 突破③：逐神经元激活 EMA（内在可塑性；默认关闭不分配逻辑差异）
        self.act_ema = None

    # ---------- 推理：自下而上 + 误差驱动精炼（自由能下降） ----------
    def infer(self, s0: np.ndarray, n_steps: int = 1) -> dict:
        r1 = np.tanh(self.W_up0 @ s0)                 # 自下而上：中间层
        r2 = np.tanh(self.W_up1 @ r1)                 # 自下而上：顶层
        for _ in range(n_steps):
            e1 = r1 - self.W_dn1 @ r2                 # 中层预测误差
            # 精炼增量限幅（防正反馈发散：误差驱动更新必须是小步修正）
            d2 = np.clip(self.W_up1 @ e1, -0.5, 0.5)
            r2 = np.tanh(r2 + 0.15 * d2)
            e0 = s0 - self.W_dn0 @ r1                 # 底层预测误差（意外度来源）
            d1 = np.clip(self.W_up0 @ e0, -0.5, 0.5)
            r1 = np.tanh(r1 + 0.15 * d1)
        e0 = s0 - self.W_dn0 @ r1
        e1 = r1 - self.W_dn1 @ r2
        return {"s0": s0, "r1": r1, "r2": r2, "e0": e0, "e1": e1}

    # ---------- 学习：全局部规则，无反向传播 ----------
    def _support(self, v: np.ndarray) -> np.ndarray:
        """稀疏 PC 的活跃支撑集：|v| 最大的前 k 维（学习更新只触达它们）。

        选择依据（O1 · 精确 top-k）：ΔW_dn0 = η·e0⊗r1 的第 j 列贡献幅值
        ∝ |r1[j]|·Σ|e0| —— 列维度的贡献排序与 |r1| 排序**完全等价**，
        故按 |v| 取 top-k 即"精确梯度贡献 top-k"，非选中列零更新。
        k = `self.topk`（>0）或默认 `len(v)//8`；k ≥ len(v) 时退化为稠密更新
        （与稠密路径逐位一致，见 tools/verify_pc_topk_equiv.py）。
        """
        k = self.topk if self.topk > 0 else max(1, len(v) // 8)
        k = min(k, len(v))
        return np.argpartition(-np.abs(v), k - 1)[:k]

    def learn(self, cache: dict, eta_scale: float = 1.0,
              homeostasis: bool = False) -> None:
        # P6 性能修复（2026-09-21，数值恒等）：eta_pc=eta_oja=0（表征冻结，
        # 评测基线配置）时全矩阵更新是乘零无效功（~2.1 MB 权重 × 8 遍遍历
        # ≈ 每 token 1.1 ms，占训练热路径 11%）——直接短路。
        # 对有限值逐位等价：W += 0·X ≡ W（clip 对未变化的权重同样恒等）；
        # homeostasis=True 时稳态缩放仍有实际作用，不短路。
        if self.eta_pc == 0.0 and self.eta_oja == 0.0 and not homeostasis:
            return
        s0, r1, r2 = cache["s0"], cache["r1"], cache["r2"]
        e0, e1 = cache["e0"], cache["e1"]
        if homeostasis:
            # 稳态第一层：教学信号限界——把预测误差归一到单位范数再进权重更新，
            # 使每步更新幅度与误差**方向**成正比而与幅值无关（幅值无界正是
            # 此前三次发散的入口；仅行缩放不足以根除）。
            e0 = e0 / max(float(np.linalg.norm(e0)), 1e-9)
            e1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
        # T5.1（lite）稀疏 PC：把外积更新截断到活跃支撑集（近似等价，
        # 非支撑集元素贡献被舍弃），学习成本 O(n²)→O(k·n)
        if self.sparse_pc:
            c1, c2 = self._support(r1), self._support(r2)
            self.W_dn0[:, c1] += self.eta_pc * eta_scale * np.outer(e0, r1[c1])
            self.W_dn1[:, c2] += self.eta_pc * eta_scale * np.outer(e1, r2[c2])
            u1 = self._support(s0)
            self.W_up0[:, u1] += self.eta_oja * eta_scale * (
                r1[:, None] * (s0[u1][None, :] - r1[:, None] * self.W_up0[:, u1]))
            self.W_up1[:, c1] += self.eta_oja * eta_scale * (
                r2[:, None] * (r1[c1][None, :] - r2[:, None] * self.W_up1[:, c1]))
        else:
            # 生成权重：预测误差 × 上层表示（最小化自由能）
            self.W_dn0 += self.eta_pc * eta_scale * np.outer(e0, r1)
            self.W_dn1 += self.eta_pc * eta_scale * np.outer(e1, r2)
            # 识别权重：Oja 规则 ΔW = η·y⊙(x − y⊙W)，逐行 Hebb+归一化
            self.W_up0 += self.eta_oja * eta_scale * (r1[:, None] * (s0[None, :] - r1[:, None] * self.W_up0))
            self.W_up1 += self.eta_oja * eta_scale * (r2[:, None] * (r1[None, :] - r2[:, None] * self.W_up1))
        np.clip(self.W_up0, -self.w_max, self.w_max, out=self.W_up0)
        np.clip(self.W_up1, -self.w_max, self.w_max, out=self.W_up1)
        np.clip(self.W_dn0, -self.w_max, self.w_max, out=self.W_dn0)
        np.clip(self.W_dn1, -self.w_max, self.w_max, out=self.W_dn1)
        if homeostasis:
            self._homeostatic_scale()

    def learn_predictive(self, prev: dict, cache: dict, eta_scale: float = 1.0,
                         mix: float = 1.0, homeostasis: bool = False) -> None:
        """T1.2 预测编码主目标化（默认关闭）：生成权重学习「下一时间步」的表示。

        用**上一步**的上层表示预测**本步**的下层表示（时序预测目标）：
            e0p = s0_now − W_dn0 @ r1_prev    （底层时序预测误差）
            e1p = r1_now − W_dn1 @ r2_prev    （中层时序预测误差）
        与逐步重构目标凸组合（mix：0=纯重构，1=纯预测）——
        对应皮层生成模型的时序预测目标（把"生成式预训练"等价物引入主干）。
        识别权重 Oja 更新与稳态缩放沿用 learn()（本方法复用其逻辑：把
        组合后的误差塞回 cache 形状后调用 learn 的更新公式不合适——
        时序项的 Hebb 因子是 prev 表示而非当前表示，故独立实现外积项）。
        """
        # P6 性能修复：与 learn() 相同的零学习率短路（eta_pc=eta_oja=0 且
        # 无稳态缩放时，全矩阵更新为乘零无效功）
        if self.eta_pc == 0.0 and self.eta_oja == 0.0 and not homeostasis:
            return
        s0, r1, r2 = cache["s0"], cache["r1"], cache["r2"]
        p_r1, p_r2 = prev["r1"], prev["r2"]
        e0 = cache["e0"]; e1 = cache["e1"]
        if homeostasis:
            e0 = e0 / max(float(np.linalg.norm(e0)), 1e-9)
            e1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
        e0p = s0 - self.W_dn0 @ p_r1              # 底层时序预测误差
        e1p = r1 - self.W_dn1 @ p_r2              # 中层时序预测误差
        if homeostasis:
            e0p = e0p / max(float(np.linalg.norm(e0p)), 1e-9)
            e1p = e1p / max(float(np.linalg.norm(e1p)), 1e-9)
        eta = self.eta_pc * eta_scale
        # 凸组合：重构项（当前表示为因子）+ 时序预测项（上一表示为因子）
        self.W_dn0 += eta * ((1.0 - mix) * np.outer(e0, r1) + mix * np.outer(e0p, p_r1))
        self.W_dn1 += eta * ((1.0 - mix) * np.outer(e1, r2) + mix * np.outer(e1p, p_r2))
        # 识别权重：Oja 规则（与 learn() 相同，作用于当前步）
        self.W_up0 += self.eta_oja * eta_scale * (r1[:, None] * (s0[None, :] - r1[:, None] * self.W_up0))
        self.W_up1 += self.eta_oja * eta_scale * (r2[:, None] * (r1[None, :] - r2[:, None] * self.W_up1))
        np.clip(self.W_up0, -self.w_max, self.w_max, out=self.W_up0)
        np.clip(self.W_up1, -self.w_max, self.w_max, out=self.W_up1)
        np.clip(self.W_dn0, -self.w_max, self.w_max, out=self.W_dn0)
        np.clip(self.W_dn1, -self.w_max, self.w_max, out=self.W_dn1)
        if homeostasis:
            self._homeostatic_scale()

    def homeostatic_rate(self, r2: np.ndarray, target: float,
                         eta_h: float) -> None:
        """PC 突破③：逐神经元目标发放率（内在可塑性，默认关闭）。

        每神经元维护激活 EMA，按目标率温和缩放其上行输入行：
        激活高于目标 → 下调该神经元的输入增益（W_up1 行）；
        低于目标 → 上调。增益经 clip(±0.5) 限幅，防止单步大幅震荡。
        这是 k-WTA（全局稀疏）之外的**逐神经元**细粒度稳态，
        对应皮层神经元的内在可塑性（intrinsic plasticity）。
        """
        if self.act_ema is None:
            self.act_ema = np.full(len(r2), target)
        act = (r2 + 1.0) * 0.5                    # tanh → [0,1] 发放率代理
        self.act_ema += eta_h * (act - self.act_ema)
        gain = np.exp(np.clip((target - self.act_ema) * 2.0, -0.5, 0.5))
        self.W_up1 *= gain[:, None]

    def _homeostatic_scale(self) -> None:
        """稳态突触缩放（Turrigiano 2008）：把每行权重范数拉回初始值。

        生物依据：突触缩放（synaptic scaling）是神经系统稳定 Hebbian 可塑性的
        经典机制——在保留权重**相对结构**的同时按比例整体缩放，防止
        误差驱动 / Hebb 更新无界增长把预测误差推入发散区。
        本项目此前 PC 在线学习三次发散（n≥256 配置下 PPL 达 4.7×10⁴），
        均因缺少这一稳态约束而被迫冻结表征（eta_pc = eta_oja = 0）。
        """
        for W, hn in ((self.W_dn0, self._hn_dn0), (self.W_dn1, self._hn_dn1),
                      (self.W_up0, self._hn_up0), (self.W_up1, self._hn_up1)):
            nrm = np.maximum(np.linalg.norm(W, axis=1), 1e-12)
            W *= (hn / nrm)[:, None]
