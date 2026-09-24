"""M3 时序关联核 —— 对应海马 CA3 递归网络与皮层局部突触的 STDP。

架构对应（文档 §2 M3）：
    稀疏拓扑：每神经元仅 m 条出边，复杂度 O(活跃×m)
    突触迹：  P_pre ← λ·P_pre + pre ；  P_post ← λ·P_post + post
    更新：    Δw = η·(P_pre·post − P_post·pre)₊  截断于 [0, w_max]

计算核与自检已拆分至 phdnet/stdp_kernels.py（NUMBA_OK 在此再导出，兼容旧引用）。"""

import numpy as np

from .stdp_kernels import (NUMBA_OK, NUMBA_AVAILABLE, _predict_edges,
                           _predict_edges_np, _selftest_numba, _stdp_delta)

__all__ = ["STDPCore", "NUMBA_OK"]



class STDPCore:
    """稀疏拓扑 + 突触迹的时序可塑性关联核。"""

    def __init__(self, n: int, m_edges: int, lam: float, eta: float, w_max: float,
                 rng: np.random.Generator, adaptive: bool = False,
                 adapt_rho: float = 0.9, adapt_eps: float = 0.5,
                 dual: bool = False, lam_slow: float = 0.85, beta_slow: float = 0.5,
                 homeostasis: bool = False, homeo_target: float = 2.0,
                 metaplasticity: bool = False, bcm_tau: float = 0.01,
                 ei_synapses: bool = False, ei_ratio: float = 0.2):
        self.n, self.m, self.lam, self.eta, self.w_max = n, m_edges, lam, eta, w_max
        # M3：T4.1 逐突触自适应；T4.2 多尺度双迹（默认关闭 = 旧行为）
        self.adaptive, self.adapt_rho, self.adapt_eps = adaptive, adapt_rho, adapt_eps
        self.dual, self.lam_slow, self.beta_slow = dual, lam_slow, beta_slow
        # 稳态突触缩放扩展至 STDP（默认关闭 = 旧行为）
        self.homeostasis, self.homeo_target = homeostasis, homeo_target
        # 元可塑性 BCM 滑动阈值（默认关闭 = 旧行为）
        self.metaplasticity, self.bcm_tau = metaplasticity, bcm_tau
        # E/I 突触类型（默认关闭 = 全为兴奋性，旧行为）
        self.ei_synapses, self.ei_ratio = ei_synapses, ei_ratio
        self.is_inh = (rng.random(n) < ei_ratio) if ei_synapses else np.zeros(n, dtype=bool)
        # 确定性稀疏拓扑：每神经元 m 条随机出边（模拟皮层局部连接）
        self.post_idx = np.empty((n, m_edges), dtype=np.int64)
        for i in range(n):
            self.post_idx[i] = rng.choice(n, size=m_edges, replace=False)
        self.W = np.zeros((n, m_edges), dtype=np.float64)   # 只存拓扑内权重
        self.t_pre = np.zeros(n)                             # 突触前迹（短窗）
        self.t_post = np.zeros(n)                            # 突触后迹（短窗）
        self.t_pre_slow = np.zeros(n)                        # T4.2 长窗前迹
        self.t_post_slow = np.zeros(n)                       # T4.2 长窗后迹
        self.v = np.full((n, m_edges), 0.25)                 # T4.1 每突触梯度二阶矩
        self.theta = np.zeros(n)                             # BCM 滑动阈值（按突触后神经元）
        self._predict = _predict_edges if NUMBA_OK else _predict_edges_np

    def predict(self, pre_rate: np.ndarray) -> np.ndarray:
        """由上一时刻发放率预测当前发放率（时序前瞻）。

        默认（ei_synapses=False）裁剪到 [0, w_max]（旧行为，逐位不变）；
        E/I 开启时抑制性出边权重为负 → 预测含负向抑制贡献，故不裁剪。
        """
        p = self._predict(self.W, self.post_idx, pre_rate, self.n)
        if self.ei_synapses:
            return p
        return np.clip(p, 0.0, self.w_max)

    def step(self, pre_rate: np.ndarray, post_rate: np.ndarray, eta_scale: float = 1.0) -> None:
        """一步时序学习：更新迹并按 STDP 规则修改活跃边权重。

        顺序要点：先吸收 pre 迹、用**更新前**的历史 post 迹计算 LTD、
        最后才吸收 post 迹 —— 否则 LTP 与 LTD 中的当前共激活项相消，学习失效。
        """
        self.t_pre = self.lam * self.t_pre + pre_rate
        # T4.2 多尺度双迹：短窗（因果细节）+ 长窗（统计巩固）在核外组合，
        # 组合后的迹直接送原核——numba 路径无需改动即可支持
        if self.dual:
            self.t_pre_slow = self.lam_slow * self.t_pre_slow + pre_rate
            self.t_post_slow = self.lam_slow * self.t_post_slow + post_rate
            tp = self.t_pre + self.beta_slow * self.t_pre_slow
            tp_hist = self.t_post + self.beta_slow * self.t_post_slow
        else:
            tp, tp_hist = self.t_pre, self.t_post
        # T4.1 需要逐突触状态 → 只能在 numpy 路径实现（numba 核无该参数，
        # 且本环境 numba 自检未通过，实际始终走 numpy）
        # 元可塑性 / STDP 稳态 / E-I 同样需要逐突触、按通道状态 → 一并强制 numpy。
        # 默认（adaptive/homeostasis/metaplasticity/ei 全 False）仍走 numba 核，逐位不变。
        if NUMBA_OK and not (self.adaptive or self.homeostasis
                             or self.metaplasticity or self.ei_synapses):
            _stdp_delta(self.W, self.post_idx,
                        self.eta * eta_scale * tp, tp_hist,
                        pre_rate, post_rate, 1.0, self.w_max)
        else:
            # numpy 回退（真事件驱动）：仅遍历活跃突触前神经元的出边行，
            # 计算量 O(活跃×m) 而非 O(N×m) —— 与 numba 核语义一致
            for i in np.nonzero(pre_rate > 0.0)[0]:
                k_t = self.post_idx[i]                     # (m,) 出边目标
                raw = 2.0 * tp[i] * post_rate[k_t] - tp_hist[k_t] * pre_rate[i]
                if self.adaptive:
                    self.v[i] = (self.adapt_rho * self.v[i]
                                 + (1.0 - self.adapt_rho) * raw * raw)
                    scale = 1.0 / (np.sqrt(self.v[i]) + self.adapt_eps)
                else:
                    scale = 1.0
                if self.metaplasticity:
                    # BCM 元可塑性：LTP 受突触后滑动阈值 θ 门控
                    # （post 超阈才增强，否则仅 LTD）→ 抑制弱连接被随机噪声推高，
                    # 是"灾难性遗忘入口之一：恒定读出学习率 + 表征漂移"的对症约束。
                    ltp = 2.0 * tp[i] * post_rate[k_t]
                    ltd = tp_hist[k_t] * pre_rate[i]
                    bcm = np.maximum(post_rate[k_t] - self.theta[k_t], 0.0)
                    dw = self.eta * eta_scale * (ltp * bcm - ltd) * scale
                else:
                    dw = self.eta * eta_scale * raw * scale
                # E/I 突触类型：抑制性神经元出边权重为负（侧向抑制动力学）
                sign = -1.0 if self.is_inh[i] else 1.0
                lo = -self.w_max if self.is_inh[i] else 0.0
                hi = 0.0 if self.is_inh[i] else self.w_max
                np.clip(self.W[i] + sign * dw, lo, hi, out=self.W[i])
            # BCM 滑动阈值：跟踪突触后活动二阶矩（"近期平均发放强度"）
            if self.metaplasticity:
                self.theta += self.bcm_tau * (np.square(post_rate) - self.theta)
            # STDP 稳态突触缩放：行范数超阈则按比例拉回（仅 downscale，
            # 不向零行注能），约束 Hebb/STDP 驱动的权重能量发散。
            if self.homeostasis:
                nrm = np.maximum(np.linalg.norm(self.W, axis=1), 1e-12)
                over = nrm > self.homeo_target
                self.W[over] *= (self.homeo_target / nrm[over])[:, None]
        self.t_post = self.lam * self.t_post + post_rate
