"""M6 读出的**异质（幂律）稀疏连接分配** —— 频次决定入边数。

================================================================================
生物学依据：突触巩固 / 修剪（synaptic consolidation & pruning）
================================================================================
人脑皮层的连接不是"每个神经元接收同样多条突触"。发育期先由分子信号铺设**过量**
的突触，随后**活动依赖**地修剪：被反复激活的通路被**巩固**（stabilize，保留
更多突触 Strength），长期沉默的低效通路被**消除**（eliminate）。结果是发育完成
的皮层中，连接密度随**神经元的实际使用频率**呈**幂律倾斜**：高频单元拥有显著更多
的入边，低频单元被修剪到接近 k_min。这正是 Hebbian 学习（"use it or lose it"）
在*连接数*这一结构层面（而非仅权重数值层面）的表达，也是稀疏编码文献中的标准
描述（Sparse distributed memory, SDM / Kanerva； hubs-and-authorities 式的
"rich get richer"）。

本模块只决定**连接数 k_i 的分配**，不决定连哪几列、也不决定权重值——后者沿用
`phdnet/sparse_pc.py::_random_csr` 的发育期随机连接 + `inits.cortical_init` 的
重尾/E-I 权重逻辑，保证与既有稀疏读出路径**同源**。

================================================================================
与均匀 k-conn（现有 `Readout(conn_k=k)`）的区别
================================================================================
* 均匀 k-conn：每行**恰好** k 条入边 → `indptr` 是等差数列，总 nnz = n_out * k。
  高频词（如"的"）与长尾词获得**同等容量**，长尾词被过分配、高频词被欠分配。
* 幂律分配（本模块）：`k_i ∝ counts_i^alpha`，行宽不等 → `indptr` 用 cumsum，
  总 nnz 由 `density` / `total_budget` **统一控预算**，不因倾斜而膨胀。
* `alpha = 0` 精确退化到均匀 k-conn（与 `_random_csr` 逐位一致，见
  `build_powlaw_csr`）。`alpha` 越大越倾斜，但受 `k_min` / `k_max` 夹紧。

**已知局限（务必阅读）**
1. `alpha=1` 在强偏斜词频（如 Zipf(1.0)，max/min 可达 1e4~1e6）上**极易撞上下限**：
   预算 `B` 摊给长尾词的份额不足 1 整数 → 大量行被 `k_min` 夹紧，秩相关下降。
   实践中 `alpha ∈ [0, 0.5]` 更合适（`alpha` 实为**压缩指数**）。
2. 高频行同样可能撞上 `k_max`（甚至 `k_max = n_in` 即退化为该行稠密）。
   两端同时饱和时秩相关必然被 ties 拉低——这是**预算的真实约束**，不是 bug。
3. 本模块是**纯分配函数**：不参与前向/反向，也未接入 `readout.py`。接线需自行
   保证 `indptr/idx/val` 与下游 `_csr_matvec` 核的约定一致（列索引行内升序、
   无放回），本模块已保证。

纯 numpy、无副作用、确定性（同一 counts + 超参 → 唯一输出，不依赖随机数）。
"""

from __future__ import annotations

import numpy as np

__all__ = ["assign_conn_counts", "build_powlaw_csr"]


# ── 内部工具 ────────────────────────────────────────────────────────────────
def _as_counts(counts) -> np.ndarray:
    """校验并归一 counts → 1-D float64（非有限值/负值按 0 处理）。"""
    if counts is None:
        return np.zeros(0, dtype=np.float64)
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim != 1:
        raise ValueError(f"counts 必须是一维 (n_out,)，收到 shape={c.shape}")
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(c, 0.0, out=c)                      # counts >= 0 是契约前提
    return c


def _power_weights(counts: np.ndarray, alpha: float) -> np.ndarray:
    """连续权重 `w ∝ counts^alpha`。

    用 `counts / max(counts)` 做归一后再取幂：等价于 `counts^alpha` 的**正比
    缩放**（`(c/cmax)^a = c^a / cmax^a`），但数值恒在 [0, 1]，**不会 overflow**。
    alpha=0 → 全 1（均匀，退化到旧 k-conn 行为）。
    全零 counts → 全 1（无信息可分，退化为均匀）。
    """
    n = counts.size
    if n == 0 or alpha == 0.0:
        return np.ones(n, dtype=np.float64)
    cmax = float(counts.max())
    if cmax <= 0.0:                                  # 全零向量：不除零
        return np.ones(n, dtype=np.float64)
    s = counts / cmax                                # ∈ [0, 1]，安全
    w = s ** alpha
    # 下溢保护：alpha 很大时低频行权重会变成 0，给一个极小正底座，保证它们
    # 仍按比例参与最大余数分配（而不是被当作"无权重"跳过）。
    return np.maximum(w, np.finfo(np.float64).tiny)


def _bounded_alloc(w: np.ndarray, budget: int, k_min: np.ndarray,
                   k_max: np.ndarray) -> np.ndarray:
    """在 `[k_min, k_max]` 框内把 `budget` 个整数单元按权重 `w` 分配。

    **最大余数法**（largest remainder / Hare quota）+ 水位式迭代：
    每轮按 `w` 比例给出连续配额，向下取整后把剩余单元按**小数部分降序**补给
    行；撞到 `k_max` 的行退出，下一轮在剩余行上重新按比例分配。
    保证 `k.sum() == budget`（预算够时）且 `k_min <= k <= k_max` —— 即
    `k.sum()` **不超过** budget。同一 `(w, budget, lo, hi)` → 唯一解（无随机）。
    """
    k = k_min.astype(np.int64).copy()
    n_rows = int(k.size)
    remaining = int(budget) - int(k.sum())
    if remaining < 0:
        raise ValueError(
            f"预算不足：sum(k_min)={int(k.sum())} > total_budget={int(budget)}；"
            "请调小 k_min 或提高 total_budget/density")
    w = np.asarray(w, dtype=np.float64).copy()

    while remaining > 0:
        room = k_max - k
        active = np.flatnonzero(room > 0)
        if active.size == 0:                         # 全部顶到 k_max，预算花不完
            break
        wa = w[active]
        if not np.isfinite(wa).all() or wa.sum() <= 0.0:
            wa = np.ones(active.size, dtype=np.float64)
            w[active] = wa
        exact = wa / wa.sum() * remaining
        floor = np.floor(exact)
        base = np.minimum(floor.astype(np.int64), room[active])
        k[active] += base
        remaining -= int(base.sum())
        if remaining <= 0:
            break
        # 最大余数：按小数部分降序、同分数按行号升序（确定性 tie-break）。
        # 必须**先剔除已顶到 k_max 的行**再补给 1，否则会突破上限。
        frac_full = np.zeros(n_rows, dtype=np.float64)
        frac_full[active] = exact - floor
        cand = active[room[active] > base]           # 仍有余量的候选行
        if cand.size == 0:
            break
        cand = cand[np.lexsort((cand, -frac_full[cand]))]
        n_take = int(min(remaining, cand.size))
        if n_take <= 0:                              # 无进展 → 退出，防死循环
            break
        k[cand[:n_take]] += 1
        remaining -= n_take
    return k


# ── 公开 API ────────────────────────────────────────────────────────────────
def assign_conn_counts(counts, density=None, alpha: float = 1.0, k_min: int = 1,
                       k_max=None, total_budget=None, n_in: int | None = None,
                       ) -> np.ndarray:
    """按幂律分配每个输出单元的入边数（纯函数，确定性）。

    参数
    ----
    counts : (n_out,) 每个输出单元的频次（>= 0；全零向量合法）
    density : 目标平均连接率 = mean(k) / n_in；与 `total_budget` 二选一。
        需要配合 `n_in`（或 `k_max`）使用。
    alpha : 幂律指数。0 = 均匀（退化为旧 conn_k 行为）；1 = k ∝ freq；越大越倾斜。
    k_min : 每行最少入边（默认 1）。
    k_max : 每行最多入边（默认 `n_in`；`n_in` 未知时默认 = 预算，即不设上限）。
    total_budget : 总入边数上限（与 `density` 二选一，**优先**）。
    n_in : 输入侧维度，仅用于 `density` 的量纲换算与 `k_max` 默认值。

    返回
    ----
    (k,) int64，满足 `k_min <= k <= k_max`、`k.sum() <= total_budget`（若给了），
    且 `alpha > 0` 时 `k` 与 `counts` 的秩相关为正。

    约束冲突（`sum(k_min) > total_budget`）时抛 `ValueError`——静默截断会让
    "预算上限"变成谎言。预算花不完（全部撞上 `k_max`）时 `k.sum() < budget`，
    这是夹紧的**正常**结果。
    """
    c = _as_counts(counts)
    n_out = int(c.size)
    if n_out == 0:
        return np.zeros(0, dtype=np.int64)

    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError(f"alpha 必须是非负有限数，收到 {alpha}")

    lo_v = int(k_min)
    if lo_v < 0:
        raise ValueError(f"k_min 必须 >= 0，收到 {k_min}")

    # ── 上限框 ──
    if k_max is None:
        hi_v = int(n_in) if n_in is not None else None
    else:
        hi_v = int(k_max)
    if hi_v is not None and hi_v < lo_v:
        raise ValueError(f"k_max({hi_v}) < k_min({lo_v})")
    lo = np.full(n_out, lo_v, dtype=np.int64)

    # ── 预算 ──
    uniform = (alpha == 0.0) or (float(c.max()) <= 0.0) or (float(c.max()) == float(c.min()))
    if total_budget is not None:
        budget = int(total_budget)
        if budget < 0:
            raise ValueError(f"total_budget 必须 >= 0，收到 {total_budget}")
    else:
        if density is None:
            raise ValueError("density 与 total_budget 必须给一个")
        if n_in is None:
            raise ValueError("density 需要 n_in 才能换算 mean(k) = density * n_in")
        density = float(density)
        if not np.isfinite(density) or density < 0.0:
            raise ValueError(f"density 必须是 [0, +inf) 的有限数，收到 {density}")
        mean_k = density * float(n_in)                 # mean(k)
        if mean_k < lo_v:                             # 连 k_min 都铺不满
            raise ValueError(f"density * n_in = {mean_k:.6g} < k_min = {lo_v}")
        if uniform:
            # 均匀分配下 k 必须全整数 → 最接近的均匀配置就是 round(mean_k)。
            # 这一步同时保证 alpha=0 路径得到**逐行相同**的 k（便于与
            # _random_csr 逐位对拍，也避免"前若干行多 1 条"的系统性偏置）。
            budget = int(round(mean_k)) * n_out
        else:
            budget = int(round(mean_k * n_out))

    if hi_v is None:
        hi = np.full(n_out, max(budget, lo_v), dtype=np.int64)
    else:
        hi = np.full(n_out, hi_v, dtype=np.int64)

    w = _power_weights(c, alpha)
    if uniform:                                      # 精确均匀（避免 ties 偏置）
        w = np.ones(n_out, dtype=np.float64)
    return _bounded_alloc(w, budget, lo, hi)


def build_powlaw_csr(rng: np.random.Generator, n_out: int, n_in: int, counts,
                     alpha: float = 1.0, density: float = 0.125, k_min: int = 1,
                     k_max=None, scale: float = 1.0, exc_ratio: float = 0.8,
                     lognormal: bool = False,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按幂律分配生成 CSR `(indptr, idx, val)`，**每行入边数不同**。

    与 `phdnet/sparse_pc.py::_random_csr`（`readout.py` 中经 `conn_k` 路径调用）
    的唯一差别：**行宽按幂律变化**（`indptr` 用 cumsum 而非等差）。权重初始化
    完全复用同源逻辑：

    * `lognormal=True` → `inits.cortical_init(rng, (k,), scale, exc_ratio)`
      （重尾对数正态 × E/I = exc_ratio），与原实现逐行同序调用；
    * `lognormal=False` → `rng.normal(0.0, scale, k)`。

    每行的 RNG 调用顺序也刻意与原实现保持一致（先 `rng.choice` 取列、后生成权重），
    因此 `alpha=0` 且行宽等于旧 `conn_k` 时，返回结果与 `_random_csr` **逐位一致**。

    参数 `alpha` / `k_min` / `k_max` / `density` 的语义见 `assign_conn_counts`。
    本函数额外把 `k_min` 夹到 `>= 1`：读出行不允许 0 条入边（`_random_csr` 内部
    同样用 `max(1, min(k, n_cols))`），且能保证 RNG 消耗序列可控。

    `counts=None` 表示"无频次信息" → 退化为均匀 k-conn（等价于 `_random_csr`）。
    """
    n_out = int(n_out)
    n_in = int(n_in)
    if n_out <= 0:
        raise ValueError(f"n_out 必须 > 0，收到 {n_out}")
    if n_in <= 0:
        raise ValueError(f"n_in 必须 > 0，收到 {n_in}")

    c = _as_counts(counts)
    if c.size == 0:                                  # None → 均匀退化路径
        c = np.zeros(n_out, dtype=np.float64)
    elif c.size != n_out:
        raise ValueError(f"counts 长度 {c.size} != n_out {n_out}")

    k = assign_conn_counts(c, density=density, alpha=alpha,
                           k_min=max(1, int(k_min)),
                           k_max=n_in if k_max is None else k_max,
                           n_in=n_in)

    # 行宽不等 → indptr 必须 cumsum（这是与旧路径的第二个差别）
    indptr = np.concatenate([[0], np.cumsum(k)]).astype(np.int64)
    nnz = int(indptr[-1])
    idx = np.empty(nnz, dtype=np.int64)
    val = np.empty(nnz, dtype=np.float64)

    if lognormal:
        from .inits import cortical_init
        for r in range(n_out):
            cols = np.sort(rng.choice(n_in, size=int(k[r]), replace=False))
            s = slice(int(indptr[r]), int(indptr[r + 1]))
            idx[s] = cols
            val[s] = cortical_init(rng, (int(k[r]),), scale, exc_ratio).ravel()
        return indptr, idx, val

    for r in range(n_out):                           # 与 _random_csr 同序：先列后权重
        kr = int(k[r])
        cols = np.sort(rng.choice(n_in, size=kr, replace=False))
        s = slice(int(indptr[r]), int(indptr[r + 1]))
        idx[s] = cols
        val[s] = rng.normal(0.0, scale, kr)
    return indptr, idx, val