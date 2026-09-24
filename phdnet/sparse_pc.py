"""M2 稀疏连接版预测编码层级 —— 结构性稀疏（sparse connectivity）。

================================================================================
为什么需要这个模块（神经认知依据，铁律②）
================================================================================
`pc.py` 的 `PredictiveCodingStack` 用**稠密矩阵**承载皮层权重（n×n 全连接，
连接率 100%）。这与皮层的真实结构不符：

| 项 | 皮层（实测文献） | 稠密栈（旧） |
|---|---|---|
| 单个锥体神经元突触数 | ~10³–10⁴ | n（满连接） |
| 潜在靶点 | ~10¹¹ | n |
| **连接率** | **~10⁻⁵（长程）～10% 量级（局部）** | **100%** |
| 计算/存储单位 | **只对存在的突触** | 全矩阵 |

皮层的组织原则是 **dense representation + sparse connectivity**（表示稠密、
连接稀疏）：信息由大量神经元的稀疏活动模式承载，而不是靠两两全连接。
本模块把主干改造为**结构性稀疏**：

  - 每个神经元只有 k 条**存在**的入边（默认 k = n_in/8 ≈ 12.5%，接近局部
    皮层连接率的量级），以 CSR 存储，**不存在的连接不存在**（不是"零权重"）；
  - 前向/学习只遍历存在的边（O(k·n) 而非 O(n²)）；
  - 学习规则与稠密版逐条对应（误差驱动 ΔW_dn、Oja 规则 ΔW_up、突触缩放稳态）。

数值关系（可验证）：**k = n_in（全连接图）时本模块与稠密栈数值一致**
（CSR 求和顺序按列索引升序；与 BLAS 内部求和顺序不保证逐位相同，
对拍判据见 tools/verify_sparse_conn.py）。

开关（默认关闭，默认路径逐位不变）：
  cfg.sparse_conn  —— True 时主干改用本模块
  cfg.conn_k       —— 每神经元入边数 k（0 = 自动取 n_in//8）
"""

from __future__ import annotations

import numpy as np

from .plasticity import NUMBA_OK

if NUMBA_OK:                                        # pragma: no cover
    from numba import njit

    @njit(cache=True, fastmath=True)
    def _csr_matvec(indptr, idx, val, x):
        """y[i] = Σ_{p∈入边(i)} val[p]·x[idx[p]]（只遍历存在的突触）。"""
        n = indptr.shape[0] - 1
        y = np.zeros(n)
        for i in range(n):
            s = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                s += val[p] * x[idx[p]]
            y[i] = s
        return y

    @njit(cache=True, fastmath=True)
    def _csr_add_outer(indptr, idx, val, a, b, eta):
        """稀疏外积累加：val[p] += eta·a[i]·b[idx[p]]（边 (i, idx[p])）。"""
        n = indptr.shape[0] - 1
        for i in range(n):
            ai = a[i]
            if ai == 0.0:
                continue
            for p in range(indptr[i], indptr[i + 1]):
                val[p] += eta * ai * b[idx[p]]

    @njit(cache=True, fastmath=True)
    def _csr_oja_up(indptr, idx, val, post, pre, eta):
        """稀疏 Oja：Δw = η·post[i]·(pre[j] − post[i]·w)（边 (i, j)）。"""
        n = indptr.shape[0] - 1
        for i in range(n):
            pi = post[i]
            if pi == 0.0:
                continue
            for p in range(indptr[i], indptr[i + 1]):
                val[p] += eta * pi * (pre[idx[p]] - pi * val[p])

    @njit(cache=True, fastmath=True)
    def _csr_clip(val, w_max):
        for p in range(val.shape[0]):
            v = val[p]
            if v > w_max:
                val[p] = w_max
            elif v < -w_max:
                val[p] = -w_max

    @njit(cache=True, fastmath=True)
    def _csr_row_norms(indptr, val):
        n = indptr.shape[0] - 1
        out = np.zeros(n)
        for i in range(n):
            s = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                s += val[p] * val[p]
            out[i] = s ** 0.5
        return out

    @njit(cache=True, fastmath=True)
    def _csr_scale_rows(indptr, val, target):
        n = indptr.shape[0] - 1
        for i in range(n):
            s = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                s += val[p] * val[p]
            nrm = s ** 0.5
            if nrm < 1e-12:
                nrm = 1e-12
            f = target[i] / nrm
            for p in range(indptr[i], indptr[i + 1]):
                val[p] *= f
else:                                               # pandas 回退（纯 numpy）
    def _csr_matvec(indptr, idx, val, x):
        y = np.zeros(indptr.shape[0] - 1)
        for i in range(len(y)):
            p0, p1 = indptr[i], indptr[i + 1]
            if p1 > p0:
                y[i] = float(val[p0:p1] @ x[idx[p0:p1]])
        return y

    def _csr_add_outer(indptr, idx, val, a, b, eta):
        for i in range(len(a)):
            if a[i] != 0.0:
                p0, p1 = indptr[i], indptr[i + 1]
                val[p0:p1] += eta * a[i] * b[idx[p0:p1]]

    def _csr_oja_up(indptr, idx, val, post, pre, eta):
        for i in range(len(post)):
            if post[i] != 0.0:
                p0, p1 = indptr[i], indptr[i + 1]
                val[p0:p1] += eta * post[i] * (pre[idx[p0:p1]] - post[i] * val[p0:p1])

    def _csr_clip(val, w_max):
        np.clip(val, -w_max, w_max, out=val)

    def _csr_row_norms(indptr, val):
        out = np.zeros(indptr.shape[0] - 1)
        for i in range(len(out)):
            p0, p1 = indptr[i], indptr[i + 1]
            out[i] = float(np.linalg.norm(val[p0:p1])) if p1 > p0 else 0.0
        return out

    def _csr_scale_rows(indptr, val, target):
        for i in range(len(target)):
            p0, p1 = indptr[i], indptr[i + 1]
            nrm = max(float(np.linalg.norm(val[p0:p1])) if p1 > p0 else 0.0, 1e-12)
            val[p0:p1] *= target[i] / nrm


def _random_csr(rng: np.random.Generator, n_rows: int, n_cols: int, k: int,
                scale: float, lognormal: bool = False,
                exc_ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每行随机 k 条入边（无放回，列索引升序）——发育期随机连接的抽象。

    `lognormal=True` 时权重按皮层式分布（重尾对数正态 × E/I 符号比，O1-4）。
    """
    k = max(1, min(k, n_cols))
    indptr = np.arange(0, (n_rows + 1) * k, k, dtype=np.int64)
    idx = np.empty(n_rows * k, dtype=np.int64)
    val = np.empty(n_rows * k, dtype=np.float64)
    if lognormal:
        from .inits import cortical_init
        for r in range(n_rows):
            cols = np.sort(rng.choice(n_cols, size=k, replace=False))
            s = slice(r * k, (r + 1) * k)
            idx[s] = cols
            val[s] = cortical_init(rng, (k,), scale, exc_ratio).ravel()
        return indptr, idx, val
    for r in range(n_rows):
        cols = np.sort(rng.choice(n_cols, size=k, replace=False))
        s = slice(r * k, (r + 1) * k)
        idx[s] = cols
        val[s] = rng.normal(0.0, scale, k)
    return indptr, idx, val


def _from_dense_csr(W: np.ndarray, k: int):
    """把稠密权重按每行 |w| 最大的 k 列转为 CSR（列索引升序，值原样保留）。

    生物学对应：突触修剪保留强连接（"use it or lose it"）——用于
    稠密→稀疏蒸馏，以及**可比性对拍**（k = 全列时两栈权重完全相同）。
    """
    n_rows, n_cols = W.shape
    k = max(1, min(k, n_cols))
    indptr = np.arange(0, (n_rows + 1) * k, k, dtype=np.int64)
    idx = np.empty(n_rows * k, dtype=np.int64)
    val = np.empty(n_rows * k, dtype=np.float64)
    for r in range(n_rows):
        cols = np.argsort(-np.abs(W[r]))[:k]
        cols.sort()
        s = slice(r * k, (r + 1) * k)
        idx[s] = cols
        val[s] = W[r, cols]
    return indptr, idx, val


def _transpose_csr(indptr, idx, val, n_new_rows: int, scale: float = 1.0):
    """把 (n_rows × k) 稀疏图转置为 (n_cols × k') 的 CSR（值 × scale）。

    生成权重以识别权重的转置初始化（生成-识别共享先验，与稠密版一致）。
    """
    src_row = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    new_rows = idx.copy()
    order = np.lexsort((src_row, new_rows))
    flat_src, flat_val = src_row[order], val[order] * scale
    counts = np.bincount(new_rows, minlength=n_new_rows)
    new_indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return new_indptr, flat_src.astype(np.int64), flat_val


class SparsePCStack:
    """结构性稀疏预测编码栈（接口与 `PredictiveCodingStack` 对齐，可 drop-in）。

    四个权重（上行识别 W_up0/W_up1、下行生成 W_dn0/W_dn1）均以 CSR 承载：
      indptr[i]:indptr[i+1] 给出第 i 个**目标**神经元存在的入边（源索引 + 权重）。
    学习只更新存在的边；`k = n_in` 时退化为全连接图（与稠密栈数值一致）。
    """

    def __init__(self, n0: int, n1: int, n2: int, eta_pc: float, eta_oja: float,
                 rng: np.random.Generator, w_max: float = 2.0,
                 conn_k: int = 0, lognormal_init: bool = False,
                 exc_ratio: float = 0.8):
        self.eta_pc, self.eta_oja, self.w_max = eta_pc, eta_oja, w_max
        self.n0, self.n1, self.n2 = n0, n1, n2
        k0 = conn_k if conn_k > 0 else max(1, n0 // 8)
        k1 = conn_k if conn_k > 0 else max(1, n1 // 8)
        self.k0, self.k1 = min(k0, n0), min(k1, n1)
        # 上行（识别）稀疏图（O1-4：可选皮层式权重分布）
        # ⚠ 初始化尺度按 **fan-in 归一化**（He/Glorot）：稀疏栈每神经元只有 k 条入边，
        #   故权重标准差取 1/√k 而非 1/√n_in —— 否则前向幅值偏低 √(k/n_in) 倍
        #   （k=32/n=256 时 ~0.35×），経 tanh 后表征趋 0、区分度下降。
        #   该量纲错误在 4,000 字符短训练口径下被掩盖（PPL −0.30%），
        #   在全语料口径下暴露为 +37.6% 劣化；按 fan-in 修正后两者应一致。
        self.up0 = _random_csr(rng, n1, n0, self.k0, 1.0 / np.sqrt(self.k0),
                               lognormal_init, exc_ratio)
        self.up1 = _random_csr(rng, n2, n1, self.k1, 1.0 / np.sqrt(self.k1),
                               lognormal_init, exc_ratio)
        # 下行（生成）稀疏图 = 上行图转置 × 0.5（与稠密版同一初始化语义）
        self.dn0 = _transpose_csr(*self.up0, n_new_rows=n0, scale=0.5)
        self.dn1 = _transpose_csr(*self.up1, n_new_rows=n1, scale=0.5)
        # 稳态目标：各行权重初始 L2 范数（突触缩放拉回此值）
        self._hn_up0 = _csr_row_norms(self.up0[0], self.up0[2])
        self._hn_up1 = _csr_row_norms(self.up1[0], self.up1[2])
        self._hn_dn0 = _csr_row_norms(self.dn0[0], self.dn0[2])
        self._hn_dn1 = _csr_row_norms(self.dn1[0], self.dn1[2])
        self.act_ema = None

    # ---------- 兼容接口：暴露稀疏矩阵形式的"权重"（只读快照） ----------
    @classmethod
    def from_dense(cls, W_up0: np.ndarray, W_up1: np.ndarray, W_dn0: np.ndarray,
                   W_dn1: np.ndarray, eta_pc: float, eta_oja: float,
                   w_max: float = 2.0, k: int = 0) -> "SparsePCStack":
        """由稠密权重构造稀疏栈（每行取 |w| 最大的 k 列；k≤0 或 k≥列数 → 全列）。

        用途：(a) **可比性对拍**——k = 全列时两栈权重完全相同，前向/学习应一致；
              (b) 稠密→稀疏蒸馏（保留幅值最大的连接，对应"强突触优先保留"）。
        """
        self = cls.__new__(cls)
        self.eta_pc, self.eta_oja, self.w_max = eta_pc, eta_oja, w_max
        self.n0, self.n1, self.n2 = W_up0.shape[1], W_up0.shape[0], W_up1.shape[0]
        self.k0 = min(k, W_up0.shape[1]) if k > 0 else W_up0.shape[1]
        self.k1 = min(k, W_up1.shape[1]) if k > 0 else W_up1.shape[1]
        self.up0 = _from_dense_csr(W_up0, self.k0)
        self.up1 = _from_dense_csr(W_up1, self.k1)
        self.dn0 = _from_dense_csr(W_dn0, W_dn0.shape[1])
        self.dn1 = _from_dense_csr(W_dn1, W_dn1.shape[1])
        self._hn_up0 = _csr_row_norms(self.up0[0], self.up0[2])
        self._hn_up1 = _csr_row_norms(self.up1[0], self.up1[2])
        self._hn_dn0 = _csr_row_norms(self.dn0[0], self.dn0[2])
        self._hn_dn1 = _csr_row_norms(self.dn1[0], self.dn1[2])
        self.act_ema = None
        return self

    @staticmethod
    def _dense(indptr, idx, val, n_rows: int, n_cols: int) -> np.ndarray:
        W = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            p0, p1 = indptr[i], indptr[i + 1]
            W[i, idx[p0:p1]] = val[p0:p1]
        return W

    @property
    def W_up0(self):                        # 只读稠密视图（测试/分析用，勿写入）
        return self._dense(*self.up0, self.n1, self.n0)

    @property
    def W_up1(self):
        return self._dense(*self.up1, self.n2, self.n1)

    @property
    def W_dn0(self):
        return self._dense(*self.dn0, self.n0, self.n1)

    @property
    def W_dn1(self):
        return self._dense(*self.dn1, self.n1, self.n2)

    def n_synapses(self) -> int:
        """实际存在的突触数（结构性稀疏的存储/计算单位）。"""
        return int(sum(len(v[2]) for v in (self.up0, self.up1, self.dn0, self.dn1)))

    # ---------- 推理（稀疏前向，逻辑与稠密版逐条对应） ----------
    def infer(self, s0: np.ndarray, n_steps: int = 1) -> dict:
        r1 = np.tanh(_csr_matvec(*self.up0, s0))
        r2 = np.tanh(_csr_matvec(*self.up1, r1))
        for _ in range(n_steps):
            e1 = r1 - _csr_matvec(*self.dn1, r2)
            d2 = np.clip(_csr_matvec(*self.up1, e1), -0.5, 0.5)
            r2 = np.tanh(r2 + 0.15 * d2)
            e0 = s0 - _csr_matvec(*self.dn0, r1)
            d1 = np.clip(_csr_matvec(*self.up0, e0), -0.5, 0.5)
            r1 = np.tanh(r1 + 0.15 * d1)
        e0 = s0 - _csr_matvec(*self.dn0, r1)
        e1 = r1 - _csr_matvec(*self.dn1, r2)
        return {"s0": s0, "r1": r1, "r2": r2, "e0": e0, "e1": e1}

    # ---------- 学习（只更新存在的突触） ----------
    def learn(self, cache: dict, eta_scale: float = 1.0,
              homeostasis: bool = False) -> None:
        if self.eta_pc == 0.0 and self.eta_oja == 0.0 and not homeostasis:
            return
        s0, r1, r2 = cache["s0"], cache["r1"], cache["r2"]
        e0, e1 = cache["e0"], cache["e1"]
        if homeostasis:
            e0 = e0 / max(float(np.linalg.norm(e0)), 1e-9)
            e1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
        eta = self.eta_pc * eta_scale
        # 下行生成权重：误差 × 上层表示（稀疏外积）
        _csr_add_outer(*self.dn0, e0, r1, eta)
        _csr_add_outer(*self.dn1, e1, r2, eta)
        # 上行识别权重：Oja 规则
        eo = self.eta_oja * eta_scale
        _csr_oja_up(*self.up0, r1, s0, eo)
        _csr_oja_up(*self.up1, r2, r1, eo)
        _csr_clip(self.up0[2], self.w_max)
        _csr_clip(self.up1[2], self.w_max)
        _csr_clip(self.dn0[2], self.w_max)
        _csr_clip(self.dn1[2], self.w_max)
        if homeostasis:
            self._homeostatic_scale()

    def learn_predictive(self, prev: dict, cache: dict, eta_scale: float = 1.0,
                         mix: float = 0.5, homeostasis: bool = False) -> None:
        """时序预测项 + 重构项的凸组合（与稠密版逐条对应，稀疏更新）。"""
        if self.eta_pc == 0.0 and self.eta_oja == 0.0 and not homeostasis:
            return
        s0, r1, r2 = cache["s0"], cache["r1"], cache["r2"]
        p_r1, p_r2 = prev["r1"], prev["r2"]
        e0, e1 = cache["e0"], cache["e1"]
        if homeostasis:
            e0 = e0 / max(float(np.linalg.norm(e0)), 1e-9)
            e1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
        e0p = s0 - _csr_matvec(*self.dn0, p_r1)
        e1p = r1 - _csr_matvec(*self.dn1, p_r2)
        if homeostasis:
            e0p = e0p / max(float(np.linalg.norm(e0p)), 1e-9)
            e1p = e1p / max(float(np.linalg.norm(e1p)), 1e-9)
        eta = self.eta_pc * eta_scale
        one_minus = 1.0 - mix
        _csr_add_outer(*self.dn0, e0, r1, eta * one_minus)
        _csr_add_outer(*self.dn0, e0p, p_r1, eta * mix)
        _csr_add_outer(*self.dn1, e1, r2, eta * one_minus)
        _csr_add_outer(*self.dn1, e1p, p_r2, eta * mix)
        eo = self.eta_oja * eta_scale
        _csr_oja_up(*self.up0, r1, s0, eo)
        _csr_oja_up(*self.up1, r2, r1, eo)
        _csr_clip(self.up0[2], self.w_max)
        _csr_clip(self.up1[2], self.w_max)
        _csr_clip(self.dn0[2], self.w_max)
        _csr_clip(self.dn1[2], self.w_max)
        if homeostasis:
            self._homeostatic_scale()

    def homeostatic_rate(self, r2: np.ndarray, target: float,
                         eta_h: float) -> None:
        """逐神经元目标发放率（内在可塑性）：按增益缩放该神经元存在的入边。"""
        if self.act_ema is None:
            self.act_ema = np.full(len(r2), target)
        act = (r2 + 1.0) * 0.5
        self.act_ema += eta_h * (act - self.act_ema)
        gain = np.exp(np.clip((target - self.act_ema) * 2.0, -0.5, 0.5))
        indptr, _idx, val = self.up1
        for i in range(len(gain)):
            p0, p1 = indptr[i], indptr[i + 1]
            val[p0:p1] *= gain[i]

    def _homeostatic_scale(self) -> None:
        """突触缩放（Turrigiano 2008）：每行存在的权重范数拉回初始值。"""
        _csr_scale_rows(self.up0[0], self.up0[2], self._hn_up0)
        _csr_scale_rows(self.up1[0], self.up1[2], self._hn_up1)
        _csr_scale_rows(self.dn0[0], self.dn0[2], self._hn_dn0)
        _csr_scale_rows(self.dn1[0], self.dn1[2], self._hn_dn1)

    def prune_silence(self, threshold: float) -> int:
        """发育期突触修剪（`critical_period`）：把 |w| < threshold 的**存在突触**
        权重置零并返回被沉默的边数。

        生物学对应「突触沉默」（silent synapse）——结构保留、传递效率归零，
        是可逆的早期修剪形态。真正的**结构性删除**（从 CSR 中移除该边，
        使行度不齐）需要变长 CSR / 重新压缩，列为后续项。
        与稠密版语义对齐：稠密版同样是把低于阈值的元素置零。
        """
        n = 0
        for (_ip, _idx, val) in (self.up0, self.up1, self.dn0, self.dn1):
            mask = np.abs(val) < threshold
            n += int(mask.sum())
            val[mask] = 0.0
        return n

    def stats(self) -> dict:
        """结构性稀疏统计：存在突触数 vs 等维度稠密矩阵的元素数。"""
        dense_total = (self.n1 * self.n0 + self.n2 * self.n1
                       + self.n0 * self.n1 + self.n1 * self.n2)
        syn = self.n_synapses()
        return {"synapses": syn, "dense_equivalent": dense_total,
                "connectivity": syn / dense_total if dense_total else 0.0,
                "k0": self.k0, "k1": self.k1}
