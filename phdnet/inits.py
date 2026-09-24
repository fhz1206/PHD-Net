"""皮层式权重初始化 —— 突触权重的**重尾分布 + 兴奋/抑制比**。

================================================================================
神经认知依据（铁律②）
================================================================================
1. **重尾 / 对数正态**：皮层突触权重不是高斯对称分布，而是**对数正态/重尾**——
   少数强连接 + 大量弱连接（Song, Sjöström, Reigl, Nelson & Chklovskii 2005,
   PLoS Biol：皮层锥体细胞间连接强度呈对数正态；Montgomery & Madison 2004）。
   高斯初始化会**低估强连接的存在**，弱化"稀疏强连接承载主要信息"的组织原则。
2. **兴奋/抑制比**：皮层突触约 **80% 兴奋性 / 20% 抑制性**（E/I balance，
   是稳定动力学与增益控制的前提）。单矩阵符号平衡是它的最简抽象。

本模块提供 `cortical_init`：幅度服从对数正态（重尾）+ 符号按 E/I 比例分配，
并归一到与既有高斯初始化相同的尺度（保持前向激活幅值可比）。

默认关闭（`cfg.lognormal_init=False` → 各模块仍用 `rng.normal`，默认路径逐位不变）。
"""

from __future__ import annotations

import numpy as np


def cortical_init(rng: np.random.Generator, shape, scale: float,
                  exc_ratio: float = 0.8, sigma: float = 1.0) -> np.ndarray:
    """皮层式初始化：幅度 lognormal(重尾) × 符号(E/I = exc_ratio)。

    参数
    ----
    shape      : 权重形状
    scale      : 整体尺度（与高斯初始化同用 1/sqrt(n_in) 等，保持激活幅值可比）
    exc_ratio  : 兴奋性（正）比例，皮层 ~0.8
    sigma      : 对数正态的 σ（越大尾部越重）；1.0 给出中位数/均值比 ~1.65

    归一化：乘 exp(−σ²/2) 使 E[幅度] = 1，因此期望幅值与 `rng.normal(0, scale)`
    的尺度一致（重尾体现在**分布形状**而非整体量级）。
    """
    mag = rng.lognormal(mean=0.0, sigma=sigma, size=shape)
    mag *= np.exp(-0.5 * sigma * sigma)                 # E[mag] = 1
    sign = np.where(rng.random(shape) < exc_ratio, 1.0, -1.0)
    return (mag * sign * scale).astype(np.float64)
