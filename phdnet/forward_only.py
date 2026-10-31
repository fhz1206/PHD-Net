"""P210 前向only训练路径（forward-only training path）—— 层目标投影 + 局部目标匹配。

调研与动机
----------
市面上的「前向only」（forward-only / 无反向传播）训练路径主要有五支：

| 方法 | 核心机制 | 信用分配方式 |
|---|---|---|
| **Forward-Forward** (Hinton 2022) | 数据与标签假设拼接后各自前向，比对各假设的「goodness」 | 每层只看到自己的激活与一个标量 goodness 差异 |
| **Forward Projection** (Nat Commun 2026, s41467-026-69161-1) | 用**随机非线性投影**把标签投影成各层的膜电位目标；局部损失用**闭式回归**求解 | 目标从标签经固定随机投影生成，逐层独立 |
| **Forward Target Propagation** (arXiv 2506.11030) | 第二次前向扫描，用固定随机矩阵 G 从输出预测与目标标签估计逐层目标 | τ1 = σ(Gy) − σ(Gh_L) + h1，之后 τi = σ(W_i·τ_{i−1}) 用**同一套前向权重**递归 |
| **NoProp** (arXiv 2503.24322) | 每层独立学习去噪一个带噪标签；完全不做前向/反向传播 | 每层独立去噪，无层间信用分配 |
| **Sigprop** | 前向信号传播：把全局学习信号投影进网络成为层目标 | 前向投影 + 逐层局部学习 |

共同内核（本模块逐条对齐）
------------------------
1. **不使用反向传播** —— 梯度不沿层间回传。
2. **层目标由「标签 → 固定随机投影」生成** —— 投影矩阵**固定不训练**（FP/FTP 都用固定
   随机矩阵 G），故不存在「权重对称性 / 权重传输（weight transport）」问题。
3. **每层用自己的局部目标做局部学习** —— 损失只看本层激活与本层目标。

PHD-Net 的接入点
----------------
PHD-Net 本身**已经是无反向传播架构**（预测编码 + 赫布 + STDP，全部局部规则）。
但存在一个真实缺口：**任务（下一 token）信号只到达 M6 读出，主干 M2 完全拿不到
任务信号**——主干只学无监督的重构预测（P194 起上行识别权重更被 P198 冻结，
`eta_oja=0`）。这正是 forward-only 家族要补的信用分配缺口。

本模块提供的是**主干侧的「层目标」通道**，取 **Forward Projection (FP)** 的
「标签 → 固定随机投影目标」与 **Forward Target Propagation (FTP)** 的
「用同一套权重递推层目标」两条内核，落地为一个**纯前向、逐层、局部**的学习规则：

    目标生成（一次前向，无回传）
        τ_r1 = σ(Proj(label))                 ← FP 内核：固定随机投影
        τ_s0 = σ(dn0 · τ_r1)                  ← FTP 内核：用模型**自己的**下行
                                                权重把目标递推到更浅的一层

    局部学习（逐层独立，无跨层梯度）
        dn1 += η·(τ_r1 − dn1·r2) ⊗ r2         ← 每层用自己的目标做「目标匹配」
        dn0 += η·(τ_s0 − dn0·r1) ⊗ r1

关键语义（务必读）
------------------
**τ 是「下一 token 的随机码」，不是「正确答案」。** 固定随机投影不承载语义，
它提供的是一个**非退化、逐层可区分**的目标方向：同一个标签永远给同一个方向，
不同标签给不同方向。于是下行生成权重会学到「从当前状态预测下一 token 的随机码」
——即**下行生成权重成为「未来预测器」**，这与预测编码框架里生成权重的既有语义
（T1.2 `pc_predictive_target` 让生成权重预测下一时间步表示）同源，区别只在于
目标从「上一时刻表征」换成了「前向only的标签随机码」。

设计取舍（为什么不用闭式回归）
------------------------------
FP 的闭式回归需要**整数据集统计**（对偶/核方法），与本项目的**逐 token 流式**
铁律（README §2 铁律③）冲突。故本模块保留 forward-only 的**信用分配内核**
（固定随机投影 + 逐层目标 + 局部匹配），把「闭式求解」换成**流式增量**版本：
每步只用当前 token 的 (h_i, τ_i) 做一次局部更新。这是有意的取舍，已在
`docs/PHD-Net_前向only训练路径.md` 记录。

约束与诚实边界
--------------
- 本模块**不触碰上行识别权重**（up0/up1）——P198 实测 `eta_oja>0` 会正反馈发散
  （3000 步 probe_PPL 9.1e11），故上行保持只读。只更新**下行生成权重** dn0/dn1。
- 层目标 τ 由**固定随机投影**产生，与标签的具体语义无关 → 它**不保证**比
  无监督重构更好的 PPL；**收益必须由 `probe_PPL` A/B 判定**，不得先验断言。
- ⚠ **P211（fhz 2026-10-31 指示「默认采用 forward-only，旧的非 forward-only 路径
  删去」）**：本路径**默认开启**（`cfg.forward_only=True`），且旧的「非 forward-only
  路径」**不再是并列模式**，而是本路径的一个**特例**——把 `forward_only_eta` 设为 0
  即完全等价于旧行为（`learn_forward_target` 首行 `eta == 0` 直接返回）。
  门禁 `gate_forward_only` 的 C3b 用**训练轨迹摘要逐位**验证了这条等价关系。
"""

from __future__ import annotations

import numpy as np

__all__ = ["ForwardTargetProjector", "hash_codes"]


# ---------------------------------------------------------------------------
# 确定性随机码（固定投影的按需实现）
# ---------------------------------------------------------------------------
def _splitmix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 混淆（uint64 环运算，numpy 的**数组** uint64 溢出静默回绕）。

    用于把 (标签 id, 维度, 种子) 确定性地映射成一个 64 位随机数。
    选它而不是 `np.random.default_rng` 的原因：**跨进程/跨 numpy 版本可复现**
    （不依赖 RNG 流的内部状态），且**按需计算、无需存表**（词表 51,962 ×
    层维度 的稠密投影表会是数百 MB）。

    ⚠ 全部用**数组**运算：numpy 的 uint64 **标量**溢出会发 RuntimeWarning
      （值仍正确回绕，但警告噪声大）。数组的 uint64 溢出是静默回绕。
    """
    z = x + np.full_like(x, np.uint64(0x9E3779B97F4A7C15))
    z = (z ^ (z >> np.uint64(30))) * np.full_like(z, np.uint64(0xBF58476D1CE4E5B9))
    z = (z ^ (z >> np.uint64(27))) * np.full_like(z, np.uint64(0x94D049BB133111EB))
    return z ^ (z >> np.uint64(31))


def hash_codes(label_idx: int, dim: int, seed: int,
               gain: float = 1.0) -> np.ndarray:
    """标签 id → 固定随机码（长度 dim），确定性、可复现、跨进程一致。

    这是 **固定随机投影 G 的按需实现**：数学上等价于「取一个固定随机矩阵 G 的
    第 label_idx 列」，但只算需要的那一列，不存矩阵。

    分布：各分量近似独立 U(-1, 1)（经 splitmix64 → 取高 52 位映射到 [0,1)）。
    `gain` 线性缩放（调用方再过 tanh 压到激活值域内）。
    """
    dim = int(dim)
    if dim <= 0:
        raise ValueError(f"hash_codes：dim 必须 > 0，实际 {dim}")
    j = np.arange(dim, dtype=np.uint64)
    # ⚠ 基址混合用 **Python 整数**先算完（模 2^64），再转成 uint64 数组：
    #   numpy 的 uint64 **标量/0-d 数组**乘加溢出会发 RuntimeWarning
    #   （值仍正确回绕，但警告噪声大，且可能被 -Werror 升级成错误）；
    #   Python 大整数天然任意精度，取模即等价且无警告。
    #   数组级的 uint64 溢出是静默回绕，正是 splitmix64 要的。
    _M64 = (1 << 64) - 1
    _c = 0x9E3779B97F4A7C15
    _s = 0xC2B2AE3D27D4EB4F
    base = ((int(label_idx) & _M64) * _c
            + (int(seed) & _M64) * _s) & _M64
    base = np.full(dim, np.uint64(base), dtype=np.uint64)
    h = _splitmix64(j + base)
    u = (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
    return (2.0 * u - 1.0) * float(gain)


# ---------------------------------------------------------------------------
# 固定随机投影器
# ---------------------------------------------------------------------------
class ForwardTargetProjector:
    """标签信号 → 层目标的**固定随机投影**（不训练、不回传、无大矩阵）。

    对齐 **Forward Projection**（Nat Commun 2026）与 **Forward Target
    Propagation**（arXiv 2506.11030）的共同内核：用一个**固定的随机映射**把
    「任务信号」投影成各层的目标激活。固定（而非学习得到的）投影是这两条路径的
    关键——它**不需要对称反馈连接**，因此没有 weight transport 问题。

    脑对应
    ------
    固定随机投影对应**弥散式（diffuse）皮层反馈**：真实的皮层下行投射并非前馈
    权重的转置副本（那是 BP 的隐含假设），而是**随机连接但统计上匹配**的弥散通路。
    本实现用固定随机码表达这一点——它给出一个「与前馈权重无关」的先验方向。

    参数
    ----
    n_out : int
        标签空间维度（读出输出维度）——仅用于**校验标签 id 越界**。
    n_layers : tuple[int, ...]
        需要生成目标的各层维度（从深到浅）。PHD-Net 的 M2 用两层：
        r1 空间（n_mid）与 s0 空间（n_sdr）。
    seed : int
        随机种子（固定 → 同一模型内目标方向恒定）。
    gain : float
        目标码的线性增益（再经 tanh 压到激活值域）。
    """

    def __init__(self, n_out: int, n_layers: tuple[int, ...], seed: int,
                 gain: float = 1.0, cache_size: int = 4096):
        self.n_out = int(n_out)
        self.n_layers = tuple(int(d) for d in n_layers)
        self.seed = int(seed)
        self.gain = float(gain)
        if any(d <= 0 for d in self.n_layers):
            raise ValueError(f"ForwardTargetProjector：层维度必须 > 0，"
                             f"实际 {self.n_layers}")
        # P211 性能：按标签缓存已算出的目标码。
        # 语料里 token 服从 zipf 分布，头部标签在相邻 token 上反复出现 →
        # 缓存命中率可观。而 `hash_codes` 每步都要跑 splitmix64（多次 uint64
        # 乘加 + 位运算）+ tanh，属纯重复计算。
        # ⚠ **不改变数值**：固定投影对同一 (label, layer) 恒得同一码，缓存
        #   只是把「重算」换成「重取」。返回的数组是**共享副本**——调用方
        #   不得原地修改它（`recurse` 只读，外部也不改）。
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        self._cache_size = int(cache_size)

    def project(self, label_idx: int, layer: int) -> np.ndarray:
        """把标签 ``label_idx`` 投影成第 ``layer`` 层的**目标激活**。

        返回经过 ``tanh`` 压到 (-1, 1) 的稠密向量（长度 = 该层维度），
        与 M2 各层激活（`tanh` 输出）同值域。**不做任何学习、不依赖模型权重。**

        P211：命中缓存时直接返回**同一对象**（共享副本，调用方只读）。
        """
        li = int(label_idx)
        if not (0 <= li < self.n_out):
            raise ValueError(
                f"ForwardTargetProjector：标签 id {li} 越界 [0, {self.n_out})")
        li_layer = int(layer)
        key = (li, li_layer)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        d = self.n_layers[li_layer]
        code = hash_codes(li, d, self.seed, self.gain)
        out = np.tanh(code)
        # 简单容量上限：超限就清空（LRU 太重，且命中率对 FIFO 几乎无差）。
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[key] = out
        return out

    def recurse(self, tau: np.ndarray, weight_mv) -> np.ndarray:
        """用模型**自己的**权重把层目标递推到更浅一层（FTP 内核）。

        `weight_mv` 是一个「向量 → 向量」的映射（CSR 稀疏 matvec）。
        递推关系与 FTP 的 τ_i = σ(W_i·τ_{i+1}) 同构：目标**沿前向权重**移动，
        不需要任何额外的反馈矩阵，也不做任何梯度回传。
        """
        out = np.asarray(weight_mv(np.asarray(tau, dtype=np.float64)),
                         dtype=np.float64)
        return np.tanh(out)
