"""M4b 长期记忆 —— 海马快印迹 + 皮层慢巩固（CLS）与吸引子模式补全。

（自 memory.py 拆分；原路径 `from phdnet.memory import LongTermMemory` 仍可用）"""

import numpy as np


class LongTermMemory:
    """M4b 长期记忆：快权重（海马）+ 慢权重（皮层）+ 吸引子检索。"""

    def __init__(self, n: int, eta_hip: float, eta_cortex: float,
                 beta_fast: float, beta_slow: float, n_steps: int):
        self.n = n
        self.eta_hip, self.eta_cortex = eta_hip, eta_cortex
        self.beta_fast, self.beta_slow = beta_fast, beta_slow
        self.n_steps = n_steps
        self.W_fast = np.zeros((n, n))          # 海马快权重（一次性印迹）
        self.W_slow = np.zeros((n, n))          # 皮层慢权重（统计巩固）
        self.diag_idx = np.diag_indices(n)

    def imprint(self, p_out: np.ndarray, p_in: np.ndarray | None = None) -> None:
        """一次性 Hebbian 印迹：对称自联想（p_in=None）或非对称配对关联。"""
        src = p_out if p_in is None else p_in
        self.W_fast += self.eta_hip * np.outer(p_out, src)
        self.W_fast[self.diag_idx] = 0.0        # 去对角防自激
        np.clip(self.W_fast, -1.0, 1.0, out=self.W_fast)

    def consolidate(self, forget: float = 1.0) -> None:
        """睡眠巩固：慢权重 EMA 吸收快权重；可选衰减快权重（遗忘旧痕）。"""
        self.W_slow = (1.0 - self.eta_cortex) * self.W_slow + self.eta_cortex * self.W_fast
        self.W_fast *= forget

    def downscale(self, alpha: float) -> None:
        """SWS 突触 downscaling（Tononi & Cirelli 突触稳态假说）：

        整体按因子 α(<1) 收缩快/慢权重，释放冗余突触容量、压低整体能耗，
        同时保留相对连接结构（按比例缩放而非随机剪除）。仅 staged_sleep=True
        的 SWS 阶段调用——旧行为（staged_sleep=False）从不触发，逐位不变。
        """
        self.W_fast *= alpha
        self.W_slow *= alpha


    def recall(self, cue: np.ndarray) -> np.ndarray:
        """Hopfield 式吸引子检索：r ← sign(W·r) 迭代至收敛（模式补全）。"""
        W = self.beta_slow * self.W_slow + self.beta_fast * self.W_fast
        r = np.sign(cue)
        r[r == 0] = 1.0
        for _ in range(self.n_steps):
            r = np.sign(W @ r)
            r[r == 0] = 1.0
        return r

    def recall_scores(self, cue: np.ndarray) -> np.ndarray:
        """T3.2 top-k 融合用的分级召回分数：收敛态的前符号激活 W·r_final。

        与 recall() 相同的吸引子动力学（只读、不改状态），但返回收敛前
        最后一次线性激活的**分级**分数（±1 二值向量的置信度），供
        读出端做 top-k 线索加权融合。
        """
        W = self.beta_slow * self.W_slow + self.beta_fast * self.W_fast
        r = np.sign(cue)
        r[r == 0] = 1.0
        for _ in range(self.n_steps):
            r = np.sign(W @ r)
            r[r == 0] = 1.0
        return W @ r

    def replay(self, n_patterns: int, rng: np.random.Generator,
               n_mix: int = 3, slow_only: bool = True) -> list[np.ndarray]:
        """P4 生成式回放（睡眠梦境的抽象）：

        从随机噪声线索出发，由吸引子网络收敛采样"梦境模式"，供重放机制
        重新印迹，实现不接触原始数据的知识保持。

        slow_only=True（默认）：仅由慢权重（已巩固的旧知识）驱动收敛 ——
        若混入快权重，梦境会以新模式为主，重放等效于加强新学习、进一步
        挤压旧记忆（已被实验证伪）。
        """
        W = self.beta_slow * self.W_slow if slow_only else \
            (self.beta_slow * self.W_slow + self.beta_fast * self.W_fast)
        dreams: list[np.ndarray] = []
        for _ in range(n_patterns):
            r = np.sign(rng.normal(0, 1, self.n))
            for _ in range(max(1, n_mix)):
                r = np.sign(W @ r)
                r[r == 0] = 1.0
            dreams.append(r)
        return dreams
