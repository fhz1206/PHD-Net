"""M5 神经调制器 —— 对应蓝斑 NE / 多巴胺的意外信号，与胆碱能系统的
编码-检索模式切换（Hasselmo）。

架构对应（文档 §2 M5）：
    surprise（底层预测误差范数）→ 在线 z 标准化
    gate = σ(a·z)                意外大 → 可塑性开大（先见者学，重复者忘）
    z 高 → 编码模式；z 低 → 检索模式
"""

import numpy as np


class Neuromodulator:
    def __init__(self, gain: float = 1.5):
        self.gain = gain
        self.mu = 0.0            # 意外度在线均值（Welford）
        self.m2 = 1.0            # 在线方差累积
        self.count = 1.0

    def observe(self, surprise: float):
        """输入当前意外度，返回 (gate ∈ [0,1], mode ∈ {encode, retrieve})。"""
        self.count += 1.0
        delta = surprise - self.mu
        self.mu += delta / self.count
        self.m2 += delta * (surprise - self.mu)
        std = (self.m2 / self.count) ** 0.5 + 1e-6
        z = (surprise - self.mu) / std
        gate = float(1.0 / (1.0 + np.exp(-self.gain * z)))   # σ(a·z)
        mode = "encode" if z > 0.3 else "retrieve"
        return gate, mode


class MultiModulator:
    """M5 多通道神经调制（脑同构升级，2026-09-19）：

    将单标量 surprise→gate 扩展为四个神经调质通道，对应 Hasselmo（ACh 编码/检索
    切换）与 Yu & Dayan（NE 新颖性→可塑性、DA 巩固、5-HT 耐心）的分工：

        ACh 乙酰胆碱 : 编码/检索模式门控（surprise 高 → 编码模式，高 ACh）
        NE  去甲肾上腺素: 新颖性驱动的可塑性增益（|Δsurprise| 骤变 → 可塑性开大）
        DA  多巴胺   : 巩固信号（surprise 低、环境可预测 → 高 DA → 海马→皮层巩固/回放）
        5-HT血清素  : 耐心 / 探索-利用权衡（低唤醒 → 高耐心 → 延长检索等待、降低冲动写入）

    返回接口与 Neuromodulator 完全兼容：(gate, mode)；四通道同时作为属性
    (self.ach / self.ne / self.da / self.ht) 暴露，供 model.step 在
    multi_modulation=True 时细化各模块学习率分配（NE→STDP、DA→sleep 回放、
    5-HT→检索节奏）。默认路径（multi_modulation=False）仍用 Neuromodulator，逐位不变。
    """

    def __init__(self, gain: float = 1.5, gain_ach: float | None = None,
                 gain_ne: float | None = None, gain_da: float | None = None,
                 gain_ht: float | None = None):
        self.gain = gain
        self.gain_ach = gain_ach if gain_ach is not None else gain
        self.gain_ne = gain_ne if gain_ne is not None else 2.0
        self.gain_da = gain_da if gain_da is not None else 1.5
        self.gain_ht = gain_ht if gain_ht is not None else 1.0
        self.mu = 0.0            # 意外度在线均值（Welford）
        self.m2 = 1.0
        self.count = 1.0
        self._z_prev = 0.0
        # 暴露的四通道（observe 前给中性初值，避免未初始化访问）
        self.ach = 0.5
        self.ne = 0.5
        self.da = 0.5
        self.ht = 0.5

    def observe(self, surprise: float):
        """输入当前意外度，返回 (gate ∈ [0,1], mode ∈ {encode, retrieve})。

        gate 以 ACh 为主、NE 增强（与旧单标量 gate 语义尽量靠拢，便于对比）；
        mode 仍由 surprise 的 z 分数决定（z>0.3 视为新颖 → encode）。
        """
        self.count += 1.0
        delta = surprise - self.mu
        self.mu += delta / self.count
        self.m2 += delta * (surprise - self.mu)
        std = (self.m2 / self.count) ** 0.5 + 1e-6
        z = (surprise - self.mu) / std
        # ACh：surprise 高 → 编码模式
        self.ach = float(1.0 / (1.0 + np.exp(-self.gain_ach * z)))
        # NE：意外度的"变化率" → 新颖性 → 可塑性（一阶差分经 z 标准化）
        nov = abs(z - self._z_prev)
        self.ne = float(1.0 / (1.0 + np.exp(-self.gain_ne * (nov - 0.5))))
        # DA：surprise 低（环境可预测）→ 巩固
        self.da = float(1.0 / (1.0 + np.exp(self.gain_da * z)))   # = σ(-gain_da·z)
        # 5-HT：低新颖性/低唤醒 → 高耐心（延长检索、降低冲动写入）
        self.ht = float(1.0 / (1.0 + np.exp(self.gain_ht * (nov - 0.3))))
        self._z_prev = z
        gate = float(self.ach * (0.5 + 0.5 * self.ne))
        mode = "encode" if z > 0.3 else "retrieve"
        return gate, mode
