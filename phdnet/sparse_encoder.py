"""M1 稀疏编码器 —— 对应初级感觉皮层的稀疏放电与抑制性侧抑制（k-WTA 竞争）。

架构对应（文档 §2 M1）：
    u = W_enc·x + b          全层电位（感受野投影）
    s_i = norm(u_i) 若 i ∈ top-k，否则 0   —— 胜者全取得到 SDR 稀疏分布式表示
"""

import numpy as np


class SparseEncoder:
    def __init__(self, n_input: int, n_sdr: int, k: int, rng: np.random.Generator):
        self.n_input, self.n_sdr, self.k = n_input, n_sdr, k
        # 固定随机投影（生产中可由 Oja 规则慢调，这里保持冻结以聚焦高层学习）
        self.W = rng.normal(0.0, 1.0 / np.sqrt(n_input), size=(n_sdr, n_input))
        self.b = rng.uniform(0.0, 0.1, size=n_sdr)
        # T2.2 稳态目标：各行权重的初始 L2 范数（Foldiak 学习后拉回，防塌缩）
        self._hn = np.linalg.norm(self.W, axis=1)

    def encode(self, x: np.ndarray):
        """输入 x -> (s, idx)。s: SDR 稀疏向量；idx: 胜者索引（供稀疏快速通路）。"""
        u = self.W @ x + self.b
        idx = np.argpartition(-u, self.k - 1)[: self.k]      # k-WTA 竞争（侧抑制的抽象）
        s = np.zeros_like(u)
        win = u[idx]
        s[idx] = (win - win.min()) / (win.max() - win.min() + 1e-9) + 0.1  # 归一到 [0.1, 1.1]
        return s, np.sort(idx)

    def learn(self, x: np.ndarray, s: np.ndarray, eta: float) -> None:
        """T2.2 可学习稀疏词典（Foldiak 局部规则，默认关闭）。

        仅活跃（胜者）行更新：ΔW_i = η·s_i·(x − s_i·W_i) —— Hebb 项学习
        输入相关感受野，逐行 Oja 归一化项防塌缩；随后把每行范数拉回初始值
        （稳态双约束之二）。k-WTA 竞争（encode 内）即 Foldiak 的侧抑制分权，
        与稳态缩放共同构成路线图要求的「k-WTA + 稳态双约束」。

        局部性：只有 s_i>0 的行被触碰，成本 O(k·n_input)；无全局非局部梯度。
        """
        act = np.nonzero(s > 0.0)[0]
        if act.size == 0:
            return
        u = self.W @ x                                        # 当前全层电位（感受野投影）
        sa = s[act]
        self.W[act] += eta * (sa[:, None] * (x[None, :] - sa[:, None] * self.W[act]))
        # 稳态突触缩放：行范数拉回初始值（保留相对结构，防 Oja/Hebb 漂移塌缩）
        nrm = np.maximum(np.linalg.norm(self.W[act], axis=1), 1e-12)
        self.W[act] *= (self._hn[act] / nrm)[:, None]
