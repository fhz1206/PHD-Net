"""M4a 工作记忆 —— 对应背外侧前额叶的持续放电与基底核门控写入。

（自 memory.py 拆分；原路径 `from phdnet.memory import WorkingMemory` 仍可用）"""

import numpy as np


class WorkingMemory:
    """M4a 工作记忆：有限槽位 + 门控写入 + 漏衰减 + P5 压缩摘要槽。"""

    def __init__(self, n: int, n_slots: int, gamma: float,
                 content_address: bool = False, sim_thresh: float = 0.6):
        # ⚠⚠ **P173（fhz 指令）：fp64 → fp32**（理由同 sparse_pc.py）。
        #   ⚠ M4a 是**逐元素**运算（衰减/门控），本来就访存受限；
        #     fp32 让每步的 slots 流量**减半**。
        self.slots = np.zeros((n_slots, n), dtype=np.float32)     # 槽位内容
        self.strength = np.zeros(n_slots, dtype=np.float32)       # 槽位强度（持续放电幅值）
        self.gamma = gamma
        self.summary_slot: int | None = None    # P5：摘要槽（不参与漏衰减）
        # T3.4 内容寻址写入（默认关闭 = 最弱槽位旧行为）
        self.content_address = content_address
        self.sim_thresh = sim_thresh

    def _pick_weakest(self) -> int | None:
        """最弱槽位选择（旧行为），跳过摘要槽（B3 修复逻辑）。
        2026-10-07 修复（审计 4）：n_slots==1 时旧守卫 len(order)>1 为假 → 摘要槽保护
        静默失效、write 覆盖免衰减摘要；改为始终跳过摘要槽，找不到非摘要槽返回 None。"""
        order = np.argsort(self.strength)              # 最弱在前
        if self.summary_slot is not None:
            for s in order:                            # 跳过摘要槽，取下一个最弱
                if int(s) != self.summary_slot:
                    return int(s)
            return None                                # 2026-10-07 修复（审计 4）：只剩摘要槽 → 无可写槽
        return int(order[0])

    def decay(self) -> None:
        """每步漏积分衰减（模拟 PFC 持续放电的自然消退）。

        摘要槽位不衰减 —— 对应前额叶对重要内容的主动维持（rehearsal）。

        审计 M1：`summary_slot is None` 时（默认，`wm_summary_every=0` 从不触发）
        原实现仍构造全 True 的布尔掩码并做**花式索引**——把 (4, 1024) 数组拷贝
        出来乘完再散射写回。改成纯原地乘法（本机 11.6 → 2.2 μs/步），数值逐位
        相同（同一 gamma、同一顺序）。
        """
        if self.summary_slot is None:
            self.slots *= self.gamma
            self.strength *= self.gamma
            return
        mask = np.ones(len(self.strength), dtype=bool)
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
        # 2026-10-07 修复（审计 2）：NaN 比极为 False 会放行 NaN cue（NaN<1e-9 为 False）；
        # isfinite 哨兵先挡，避免把 NaN 摘要写进免衰减槽永久污染读出。
        if not np.all(np.isfinite(content)) or np.abs(content).sum() < 1e-9:
            return False
        if hasattr(ltm, "n_dim"):                 # 大容量事件驱动表
            k = max(1, len(content) // 8)
            # 2026-10-07 修复（审计 3）：旧判定 |content|>=thr 在幅值并列时激活数暴涨
            # （4 槽同型实测 64/64）→ 超 SparseLTM._check_sparse 的 n_dim/4 上限而崩；
            # argpartition 精确取 k 位，无论并列多少。
            idx = np.argpartition(np.abs(content), -k)[-k:]
            cue = np.zeros_like(content)
            cue[idx] = np.abs(content)[idx]
            r = np.sign(ltm.recall(cue))
        else:                                     # 稠密双速率 LTM（旧行为）
            r = ltm.recall(np.sign(content))
        if slot is None:
            # 2026-09-28 修复：与 _pick_weakest 同口径排除当前摘要槽——否则每次
            # summarize 都把免衰减保护「搬家」，上一个受保护槽立即失去保护并可被
            # 普通写入覆盖（B3 修复的同类漏洞）。
            order = np.argsort(self.strength)
            slot = int(order[0])
            if (self.summary_slot is not None and len(order) > 1
                    and slot == self.summary_slot):
                for s in order[1:]:
                    if int(s) != self.summary_slot:
                        slot = int(s)
                        break
        # 2026-10-07 修复（交叉验证漏报 3）：吸引子收敛结果同样可能非有限
        # （LTM 权重被污染时 np.sign/recall 会吐 NaN）→ 写进免衰减槽 = 永久毒化。
        if not np.all(np.isfinite(r)):
            return False
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
        # 2026-10-07 修复（审计 2）：NaN gate 旧判定（NaN<thresh 为 False）会放行写入
        # → strength=NaN → read() 永久输出 NaN；isfinite 哨兵拒绝非有限 gate。
        if not np.isfinite(gate) or gate < thresh:
            return False
        # 2026-10-07 修复（交叉验证漏报 3）：**内容侧**同样要设防 ——
        # 只挡 gate 是不够的：write(r=NaN, gate=1.0) 照样返回 True，
        # 槽位被写成 NaN 后 decay 衰不掉（NaN*α 仍是 NaN）→ read() 永久输出 NaN。
        # 实测（第二意见复核）：write(r=NaN, gate=1.0)=True → decay×2 → read 全 NaN。
        r = np.asarray(r, dtype=float)
        if not np.all(np.isfinite(r)):
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
        if slot is None:
            return False   # 2026-10-07 修复（审计 4）：无非摘要槽可写 → 拒绝写入，保护免衰减摘要
        self.slots[slot] = r
        self.strength[slot] = gate
        return True

    def read(self) -> np.ndarray:
        """强度加权读出。

        2026-10-07 修复（交叉验证漏报 3）：**自愈**。写入侧已挡 NaN 的 gate/r，
        但槽里可能残留**历史版本**写入的 NaN（或外部恢复的快照带毒）——
        实测一旦写入，decay 衰不掉（NaN×α 仍是 NaN）→ read() 永久输出 NaN。
        这里把「非有限强度」与「含非有限内容的槽」一律视作 0 权，
        保证 read() 的输出恒有限（代价：毒化槽被静默忽略，写入侧的告警负责可见性）。
        ⚠ 注意 0×NaN 仍是 NaN，所以**槽本身也要换掉**，不能只改权重。
        """
        s = self.strength
        slots = self.slots
        bad_s = ~np.isfinite(s)
        bad_c = ~np.isfinite(slots).all(axis=1)
        if bad_s.any() or bad_c.any():
            s = np.where(bad_s, 0.0, s) * (~bad_c)
            slots = np.where(bad_c[:, None], 0.0, slots)
        total = float(s.sum())
        if not np.isfinite(total) or total < 1e-9:
            return np.zeros_like(slots[0])
        return (s[:, None] * slots).sum(axis=0) / total

    # ══════════════════════════════════════════════════════════════════════
    # P192 的「**不做**」记录（这是结论，不是遗漏）
    # ══════════════════════════════════════════════════════════════════════
    # `phdnet/_cykernels.pyx` 里有一个 `wm_decay_read` 核（把 `decay()` 与
    # `read()` 合并成一趟，避免一次中间数组），**但本模块刻意不接它**。理由：
    #
    # ① **收益不可知且必然极小**：M4a 的规模是 `n_slots × d_model`，
    #    1b 档 n_slots 是几十量级、d_model 量级 1e3 → 总元素数 ~1e4，
    #    一次 np 乘法在 BLAS/内存带宽面前已是微秒级。Cython 核能省的
    #    只是**一次 Python 层函数调用 + 一次临时数组分配**，
    #    相对 M2（CSR，n×k=1e5~1e6 元素）差 2~3 个数量级。
    #    性能文档 §2.2 已把 CPU 侧 4.26 ms 拆开，**M4a 不在热点里**。
    #
    # ② **代价明确且是真实回归风险**：`read()` 上方的自愈逻辑
    #    （2026-10-07「交叉验证漏报 3」修的）要求「非有限强度」与「含非有限
    #    内容的槽」都按 0 权处理。Cython 核为了快，会走**无检查的 fp32 路径**
    #    → NaN 一旦进槽，`NaN×γ` 仍是 NaN，`read()` 永久输出 NaN ——
    #    正是那个已被修掉的 bug。把它接回去等于**用一个微秒级优化换一个
    #    已修复的正确性缺陷**，不划算。
    #
    # ③ 因此 `wm_decay_read` 核**保留在 .pyx 里**（自检 C3/C4 会跑它，
    #    保证它一旦将来被启用是对的），但**默认无任何调用点**。
    #    门禁 `verify_cykernels.py` 的 C3/C3b/C4 组就是它的保护网。
