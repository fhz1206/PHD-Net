"""M4a 工作记忆 —— 对应背外侧前额叶的持续放电与基底核门控写入。

（自 memory.py 拆分；原路径 `from phdnet.memory import WorkingMemory` 仍可用）"""

import numpy as np


class WorkingMemory:
    """M4a 工作记忆：有限槽位 + 门控写入 + 漏衰减 + P5 压缩摘要槽。"""

    def __init__(self, n: int, n_slots: int, gamma: float,
                 content_address: bool = False, sim_thresh: float = 0.6):
        self.slots = np.zeros((n_slots, n))     # 槽位内容
        self.strength = np.zeros(n_slots)       # 槽位强度（持续放电幅值）
        self.gamma = gamma
        self.summary_slot: int | None = None    # P5：摘要槽（不参与漏衰减）
        # T3.4 内容寻址写入（默认关闭 = 最弱槽位旧行为）
        self.content_address = content_address
        self.sim_thresh = sim_thresh

    def _pick_weakest(self) -> int:
        """最弱槽位选择（旧行为），跳过摘要槽（B3 修复逻辑）。"""
        order = np.argsort(self.strength)              # 最弱在前
        if self.summary_slot is not None and len(order) > 1:
            for s in order:                            # 跳过摘要槽，取下一个最弱
                if int(s) != self.summary_slot:
                    return int(s)
            return int(order[0])
        return int(order[0])

    def decay(self) -> None:
        """每步漏积分衰减（模拟 PFC 持续放电的自然消退）。

        摘要槽位不衰减 —— 对应前额叶对重要内容的主动维持（rehearsal）。
        """
        mask = np.ones(len(self.strength), dtype=bool)
        if self.summary_slot is not None:
            mask[self.summary_slot] = False
        self.slots[mask] *= self.gamma
        self.strength[mask] *= self.gamma

    def summarize(self, ltm, slot: int | None = None) -> bool:
        """P5 压缩摘要：把当前槽位加权内容经吸引子收敛为稳定表示，
        写入保留槽位（不衰减）—— 使重要信息跨越长时干扰仍可读出。

        ltm：LongTermMemory 引用（提供吸引子动力学）。

        O1（2026-09-23）：大容量表（`SparseLTM`，有 `n_dim` 属性）的 A2 契约要求
        **稀疏率**线索——稠密 ±1 会被显式拦截（一次性激活全部维度，语义与
        性能双错）。故大容量表路径把摘要内容稀疏化为 top-1/8 幅值率，
        并把返回的分级分数二值化为槽内容；稠密 LTM 路径仍用 `np.sign`（逐位不变）。
        """
        content = self.read()
        if np.abs(content).sum() < 1e-9:
            return False
        if hasattr(ltm, "n_dim"):                 # 大容量事件驱动表
            k = max(1, len(content) // 8)
            thr = np.partition(np.abs(content), -k)[-k]
            cue = np.where(np.abs(content) >= thr, np.abs(content), 0.0)
            r = np.sign(ltm.recall(cue))
        else:                                     # 稠密双速率 LTM（旧行为）
            r = ltm.recall(np.sign(content))
        if slot is None:
            slot = int(np.argmin(self.strength))
        self.slots[slot] = r.astype(float)
        self.strength[slot] = 1.0
        self.summary_slot = slot
        return True

    def write(self, r: np.ndarray, gate: float, thresh: float) -> bool:
        """门控写入：仅当调制门 > 门限时写入（基底核门控假说）。

        B3 修复：排除免衰减的摘要槽（P5 压缩槽）。否则长跑衰减后所有槽强度 ≥1.0，
        argmin 可能命中摘要槽，把"免衰减保护的重要内容"覆盖成无关输入，长期污染读出。

        T3.4 内容寻址（content_address=True，默认关闭）：写入目标改为
        「与新内容余弦相似度 ≥ sim_thresh 的最相似槽位」（相似内容聚合更新，
        减少同义内容碎片化）；无命中时回退最弱槽位。
        """
        if gate < thresh:
            return False
        if self.content_address:
            norms = np.linalg.norm(self.slots, axis=1)
            nz = norms > 1e-9
            if nz.any():
                sims = np.full(len(self.slots), -1.0)
                sims[nz] = (self.slots[nz] @ r) / (norms[nz] * (np.linalg.norm(r) + 1e-9))
                if self.summary_slot is not None:
                    sims[self.summary_slot] = -1.0     # 摘要槽不参与内容寻址覆盖
                best = int(np.argmax(sims))
                if sims[best] >= self.sim_thresh:
                    self.slots[best] = r
                    self.strength[best] = gate
                    return True
            slot = self._pick_weakest()
        else:
            slot = self._pick_weakest()
        self.slots[slot] = r
        self.strength[slot] = gate
        return True

    def read(self) -> np.ndarray:
        """强度加权读出。"""
        total = self.strength.sum()
        if total < 1e-9:
            return np.zeros_like(self.slots[0])
        return (self.strength[:, None] * self.slots).sum(axis=0) / total
