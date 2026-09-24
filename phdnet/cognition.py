"""PHD-Net v2 认知层 —— M8 情景缓冲 / M9 语义关联图 / M12 联想链推理。

架构对应（v2 文档 §2）：
    SemanticGraph  ：M9 三元组 (s, r, o) 以组合线索 [s; r] 印迹（Hopfield 扩展），
                     检索 = 一次矩阵-向量乘 + sign —— 无注意力的知识调用
    EpisodicBuffer ：M8 事件序列的 STDP 突触链存储，链式回放恢复顺序 ——
                     时间由连接承载（无位置编码的时序）
    chain_query    ：M12 联想链推理：检索结果回填为下一步线索，逐步展开 ——
                     扩散激活式推理（思维链的功能同构、机制不同）
"""

import numpy as np


class SemanticGraph:
    """M9 语义关联图：知识以 (主语, 关系, 宾语) 三元组印迹存储与调用。"""

    def __init__(self, n: int, eta: float = 0.06, rng: np.random.Generator | None = None):
        self.n = n
        self.rng = rng or np.random.default_rng(0)
        self.W = np.zeros((n, 2 * n))          # 宾语 ← [主语 ; 关系]
        self.eta = eta

    @staticmethod
    def _cue(s: np.ndarray, r: np.ndarray) -> np.ndarray:
        return np.concatenate([s, r])

    def bind(self, s: np.ndarray, r: np.ndarray, o: np.ndarray) -> None:
        """印迹一条知识：W += η·o ⊗ [s ; r]（组合绑定）。"""
        self.W += self.eta * np.outer(o, self._cue(s, r))
        np.clip(self.W, -1.0, 1.0, out=self.W)

    def query(self, s: np.ndarray, r: np.ndarray) -> np.ndarray:
        """调用知识：o = sign(W·[s ; r])（单步吸引子，无注意力打分）。"""
        return np.sign(self.W @ self._cue(s, r))

    def chain_query(self, s: np.ndarray, relations: list[np.ndarray]) -> np.ndarray:
        """M12 联想链推理：每步检索结果回填为下一步的主语，逐步展开。

        例：chain_query(苏格拉底, [是, 具有])
            → query(苏格拉底, 是) = 人 → query(人, 具有) = 会死
        """
        o = s
        for r in relations:
            o = self.query(o, r)
        return o


class EpisodicBuffer:
    """M8 情景缓冲：事件序列以非对称外积关联链存储（与 M9 同构的联想形式），
    时间 = 连接拓扑（无位置编码）；任意起点可链式回放恢复顺序 ——
    记忆是重构而非复制。"""

    def __init__(self, n: int, eta: float = 0.5, k: int = 64,
                 rng: np.random.Generator | None = None):
        self.n = n
        self.k = k
        self.eta = eta
        self.rng = rng or np.random.default_rng(0)
        self.W = np.zeros((n, n))              # 非对称：列 = 前事件，行 = 后事件
        self.prev = np.zeros(n)

    def store(self, event: np.ndarray) -> None:
        """存储一个事件（稀疏 0/1 SDR），与前事件形成因果关联。"""
        if self.prev.any():
            self.W += self.eta * np.outer(event, self.prev)
        self.prev = event

    def replay(self, start: np.ndarray, steps: int,
               k: int | None = None) -> list[np.ndarray]:
        """从 start 线索链式回放：每步取关联分布的 top-k 为下一事件。"""
        k = k or self.k
        seq = []
        cur = start
        for _ in range(steps):
            p = self.W @ cur
            idx = np.argpartition(-p, k - 1)[:k]
            nxt = np.zeros(self.n)
            nxt[idx] = 1.0
            seq.append(nxt)
            cur = nxt
        return seq
