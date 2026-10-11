"""T6 长程复制增强：上下文漂移的情景记忆（TCM / CMR 式串行回忆）。

神经认知依据（无注意力、无位置向量）：
  1. **时间上下文模型 TCM**（Howard & Kahana 2002）/ **CMR**（Polyn et al. 2009）：
     当前上下文由过去项目缓慢累积而成（c ← ρ·c + (1−ρ)·item）；回忆 = 用当前上下文
     做内容寻址检索；回忆出的项目又反过来驱动上下文继续漂移 —— 于是**串行顺序回忆
     自然涌现**。这是人类自由回忆/序列回忆的标准模型（解释近因与邻接效应）。
  2. **事件边界触发的回放**：分隔符（'#'）作为事件边界（事件分割理论 /
     海马在边界处的重放），触发整段序列的链式回放。
  3. **前额叶延迟期持续放电**：序列起始上下文由工作记忆的**免衰减保持槽**维持
     （PFC delay activity），因此回放可以「从这一段的开头」开始，而不需要检索。
  4. 与 Transformer 的区别：顺序**不存在于任何向量中**（无位置编码），也不靠
     成对相似度打分（无注意力）；顺序 = 上下文状态的连续漂移轨迹，检索 = 模式补全。

用法（延迟复制）：
  编码段：begin_episode() → observe(item) × n
  边界  ：ctx ← slot（保持槽）
  复制段：recall(steps) → 逐步输出 item_1 … item_n
"""

from __future__ import annotations

import numpy as np

from .tokenizer import U64, _mix64


class ContextMemory:
    """上下文漂移的双向联想记忆（context→item 单向显式存储 + 漂移律隐式回放）。"""

    def __init__(self, n_dim: int, rho: float = 0.8, eta: float = 0.2,
                 k_clean: int = 32, forget: float = 0.99, sep: float = 0.0,
                 kappa: float = 0.0, seed: int = 7, dtype=np.float32,
                 bidir: bool = False, bidir_w: float = 1.0):
        self.n = n_dim
        self.rho = rho                      # 上下文漂移率（越大=漂移越慢=保持越久）
        self.eta = eta                      # 印迹学习率
        # 2026-10-07 修复（审计 6）：k_clean>n_dim 会在首次 _clean 的 argpartition 抛越界、k<=0 返回全零使漂移链静默退化；构造期 clamp 到 [1, n_dim]。
        self.k = int(max(1, min(k_clean, n_dim)))  # 清理时保留的活跃位数（k-WTA 侧抑制）
        self.forget = forget                # 每段情节结束后的轻微遗忘（近因加权）
        self.sep = sep                      # 模式分离强度（0=关闭）
        self.kappa = kappa                  # 再入强度（0=关闭；CMR 的 item→context 反馈）
        self.seed = seed
        # T6 后续：双向校验（item→context→item 一致性，默认关闭）
        self.bidir = bidir
        self.bidir_w = bidir_w
        self.W = np.zeros((n_dim, n_dim), dtype=dtype)   # context → item 联想矩阵
        self.W_ic = np.zeros((n_dim, n_dim), dtype=dtype)  # item → 后随上下文（再入）
        self.ctx = np.zeros(n_dim, dtype=dtype)
        self.slot = np.zeros(n_dim, dtype=dtype)         # PFC 免衰减保持槽

    # ---------- 模式分离：为每段情节派生去相关的起始上下文（DG/CA3） ----------
    def _episode_code(self, idx: int) -> np.ndarray:
        x = U64(idx) * U64(7919) + np.arange(self.k, dtype=np.uint64) * U64(2654435761)
        i = (_mix64(x + U64(self.seed)) % U64(self.n)).astype(np.int64)
        c = np.zeros(self.n, dtype=self.ctx.dtype)
        c[i] = 1.0
        return c

    # ---------- k-WTA 清理（稀疏化，对应侧抑制竞争） ----------
    def _clean(self, v: np.ndarray) -> np.ndarray:
        # 2026-10-07 修复（审计 7）：全零激活时旧实现凭 top-k 造出任意 k 位伪「回忆」；
        # 显式返回全零——无信号就是无信号（下游按空读出处理）。
        if not np.any(v):
            return np.zeros_like(v)
        idx = np.argpartition(-v, self.k - 1)[: self.k]
        s = np.zeros_like(v)
        s[idx] = 1.0
        return s

    # ---------- 编码 ----------
    def begin_episode(self, ep_idx: int | None = None) -> None:
        """情节开始：可选做模式分离（DG 为正交化相似情景），并把上下文存入保持槽。"""
        if self.sep > 0.0 and ep_idx is not None:
            code = self._episode_code(ep_idx)
            self.ctx = (1.0 - self.sep) * self.ctx + self.sep * code
        self.slot = self.ctx.copy()

    def observe(self, item: np.ndarray) -> None:
        """观察一个项目：把「当前上下文 → 该项目」印迹，然后上下文按漂移律更新。"""
        self.W += self.eta * np.outer(item, self.ctx)
        self.ctx = self.rho * self.ctx + (1.0 - self.rho) * item
        # 再入连接（CMR）：项目 → 它之后的上下文 —— 回忆时用它把漂移轨迹
        # 拉回「学习时的轨迹」，抑制串行回忆的误差级联
        if self.kappa > 0.0:
            self.W_ic += self.eta * np.outer(self.ctx, item)
            np.clip(self.W_ic, -1.0, 1.0, out=self.W_ic)
        np.clip(self.W, -1.0, 1.0, out=self.W)

    def end_episode(self) -> None:
        """情节结束：轻微遗忘（避免长期累积导致的干扰饱和）。"""
        if self.forget < 1.0:
            self.W *= self.forget
            if self.kappa > 0.0:
                self.W_ic *= self.forget

    # ---------- 检索（串行回忆） ----------
    def _retrieval(self, c: np.ndarray) -> np.ndarray:
        """context→item 检索算子，recall 与 recall_scored 共用。
        2026-10-07 修复（审计 1）：scored 原来对 clean_steps>=2 用另一套多步递推
        （对固定 c 反复加漂移、clean 在 W 之前），评分轨迹≠状态轨迹、同参两 API 分叉；
        抽成公共算子后两者的评分与状态推进走同一 operator。"""
        return self._clean(self.W @ c)

    def _advance(self, c: np.ndarray, item: np.ndarray,
                 clean_steps: int) -> np.ndarray:
        """上下文推进：漂移律 +（可选）再入校正 +（可选）多步清理。"""
        c = self.rho * c + (1.0 - self.rho) * item
        if self.kappa > 0.0:
            reentered = self._clean(self.W_ic @ item)   # 项目找回它的后随上下文
            c = (1.0 - self.kappa) * c + self.kappa * reentered
        for _ in range(max(0, clean_steps - 1)):        # 多步清理：再检索一遍自身轨迹
            c = self._clean(self.W @ c)
        return c

    def recall(self, steps: int, clean_steps: int = 1) -> list[np.ndarray]:
        """从保持槽的起始上下文出发，逐步回忆：上下文→项目→新上下文→下个项目…"""
        out: list[np.ndarray] = []
        c = self.slot.copy()
        for _ in range(steps):
            item = self._retrieval(c)       # 2026-10-07 修复（审计 1）：与 recall_scored 共用同一算子
            out.append(item)
            c = self._advance(c, item, clean_steps)
        return out

    def _bidir_consistency(self, mat: np.ndarray) -> np.ndarray:
        """T6 后续：item→context→item 双向一致性检验（bidir=True 时启用）。

        Hebb 存储 W ≈ Σ item⊗ctx 的转置 W.T 是 item→context 的近似逆映射：
        对每个候选 item_v，恢复「编码它的上下文」再正向检索一次，
        一致性 = 重检索 item 与原候选的重合度。真（学习时被编码过的）
        候选往返一致（高分）；漂移误差产生的假候选往返不一致（低分）。
        返回 (V,) 一致性向量 ∈ [0, k]（重合位数）。
        """
        V = len(mat)
        cons = np.empty(V, dtype=np.float64)
        for vi in range(V):
            c_back = self.W.T @ mat[vi]          # item → context（转置近似逆）
            i_back = self._clean(self.W @ c_back)  # context → item（正向检索）
            cons[vi] = float(i_back @ mat[vi])   # 往返重合位数
        return cons

    def recall_scored(self, steps: int, candidates: dict[str, np.ndarray],
                      clean_steps: int = 1, segment: int = 0,
                      seg_thresh: float = 0.5, vote: bool = False) -> list[str]:
        """回忆并与候选 token 匹配（读出时与词表比对，等价于 LM 的 softmax 输出）。

        bidir=True：候选得分按双向一致性降权 —— sims·consistency^bidir_w，
        低置信（往返不一致）的回忆被抑制，串行回忆的级联误差被截断。

        O5 分段校验（segment=N>0，默认 0=关闭）：每 N 步为一个"段"，
        段末用段起点上下文按同一推进律（漂移 + 再入 + 清理）**重放**段内
        已回忆项目序列，得到重放终点 c_rep；若 c_rep 与当前终点 c 的稀疏重合度
        < seg_thresh，判定该段偏离"轨迹自洽路径"（级联误差），
        以 c_rep 重锚定上下文继续回忆 —— 将长程误差限制在一个段内
        （段内自洽是弱约束，不需要位置编码/注意力/顺序无关化）。

        O5-opt 双通路交叉验证（vote=True，默认关闭，2026-09-23）：对同一上下文
        用**两条检索通路**各自打分——单步前向 v1 = W·c 与二次迭代
        v2 = W·clean(W·c)，候选得分取**乘积** sims = s1·s2（AND 逻辑：
        两条通路都支持才胜出）。单通路下漂移噪声可推高假候选，而假候选难以
        同时通过两条通路的检验 → 抑制"自洽漂移"型误差。
        """
        if not candidates:
            # 2026-10-07 修复（审计 7）：空候选显式报错，不再在 np.stack 处以晦涩 ValueError 崩溃
            raise ValueError("recall_scored: candidates 为空，无可评分候选")
        names = list(candidates.keys())
        mat = np.stack([candidates[k] for k in names])     # (V, n)
        out = []
        c = self.slot.copy()
        cons = self._bidir_consistency(mat) if self.bidir else None
        c_seg_start = c.copy()
        for t in range(steps):
            if segment > 0 and t > 0 and t % segment == 0:
                c_seg_start = c.copy()
            v = self._retrieval(c)    # 2026-10-07 修复（审计 1）：与 recall 共用同一算子（原多步递推是另一套，cs>=2 时评分轨迹≠状态轨迹）
            sims = mat @ v                                  # 与候选 token 的重合度
            if vote:                                       # O5-opt：双通路 AND 融合
                v2 = self.W @ self._clean(self.W @ c)
                sims = sims * (mat @ v2)
            if cons is not None and self.bidir_w > 0.0:
                sims = sims * (cons ** self.bidir_w)        # 双向校验降权
            if not np.any(sims):
                # 2026-10-07 修复（审计 7）：sims 全 0 = 无读出信号，显式停止，
                # 不再让 argmax 恒选 names[0]（静默假读出）
                break
            best = names[int(np.argmax(sims))]
            out.append(best)
            c = self._advance(c, candidates[best], clean_steps)
            if segment > 0 and (t + 1) % segment == 0:      # 段末一致性校验
                c_rep = c_seg_start.copy()
                for it in out[-segment:]:
                    c_rep = self._advance(c_rep, candidates[it], clean_steps)
                agree = float(self._clean(c_rep) @ self._clean(c)) / float(self.k)
                if agree < seg_thresh:
                    c = c_rep          # 低置信段 → 以重放轨迹重锚定（截断级联）
        return out
