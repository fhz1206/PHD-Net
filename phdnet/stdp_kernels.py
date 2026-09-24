"""M3 STDP 计算核与运行时自检（numba 编译 + numpy 回退 + 等价性门槛）。

自 plasticity.py 拆分：本文件只负责"算子与自检"，STDPCore 容器见 plasticity.py。
NUMBA_OK 由自检结果决定（scatter 语义 + 一致缩放下权重增长 + 与 numpy 逐位等价）。
"""

import numpy as np

try:                                    # numba 可选加速（缺失时功能不受影响）
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:                     # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):          # 恒等装饰器回退
        def wrap(fn):
            return fn
        return wrap if not args or not callable(args[0]) else args[0]


@njit(cache=True)
def _predict_edges(W, post_idx, pre, n):
    """稀疏拓扑前向（scatter 语义）：p[k] += W[i,j]·pre[i]，即预测信号
    从活跃突触前神经元沿出边流向目标 —— 与 STDP 学习的边方向一致。"""
    p = np.zeros(n)
    m = W.shape[1]
    for i in range(n):
        pi = pre[i]
        if pi != 0.0:
            for j in range(m):
                p[post_idx[i, j]] += W[i, j] * pi
    return p


@njit(cache=True)
def _stdp_delta(W, post_idx, t_pre, t_post, pre, post, eta, w_max):
    """事件驱动 STDP：只更新**活跃突触前神经元**的出边（皮层局部可塑性）。

    LTP = A₊·前端迹 × 后端当前发放；LTD = A₋·后端历史 post 迹 × 前端当前发放。
    幅度不对称 A₊ = 2·A₋（标准 STDP 设定）：高频重复配对下 LTD 迹稳态高于
    LTP 迹，若 A₊ = A₋ 则净梯度为负、已学权重会被洗回 0。
    """
    n, m = W.shape
    for i in range(n):
        if pre[i] > 0.0:
            for j in range(m):
                k = post_idx[i, j]
                dw = eta * (2.0 * t_pre[i] * post[k] - t_post[k] * pre[i])
                w = W[i, j] + dw
                if w < 0.0:
                    w = 0.0
                elif w > w_max:
                    w = w_max
                W[i, j] = w


def _selftest_numba() -> bool:
    """numba 核运行时正确性自检（行为级 + 数值等价级）。

    历史教训（2026-09-18 修正）：此前自检把 t_pre 乘 0.1 传给核、而 LTD 项
    用的是完整 t_post —— LTP 被压小 10 倍使净增量恒为负、权重被 clip 到 0，
    于是 numba 与 numpy 路径**同时**返回 0，被误判为「numba 静默错编译」，
    项目因此长期空跑慢速 numpy 回退。实测证明两者结果逐位一致，numba 从未误编译。
    现自检：① scatter 语义；② 一致缩放下权重必须增长；③ numba 与 numpy 逐位一致。
    任一失败即回退 numpy 向量化实现。
    """
    if not NUMBA_AVAILABLE:
        return False
    try:
        # 1) predict 核 scatter 语义
        pidx = np.array([[1, 2], [2, 0], [0, 1]])
        W = np.zeros((3, 2))
        W[1, 0] = 0.5
        p = _predict_edges(W, pidx, np.array([0.0, 1.0, 0.0]), 3)
        if abs(p[2] - 0.5) > 1e-9 or abs(p[0]) > 1e-9:
            return False

        # 2) STDP 核：numba vs numpy 逐位一致 + 一致缩放下权重必须增长
        n, m = 128, 8
        rng = np.random.default_rng(0)
        pidx2 = np.array([rng.choice(n, m, replace=False) for _ in range(n)])

        def learn(W_ref, kernel):
            pre = np.zeros(n)
            post = np.zeros(n)
            pre[:8] = 0.8
            post[8:16] = 0.8
            t_pre = np.zeros(n)
            t_post = np.zeros(n)
            for _ in range(40):
                t_pre = 0.35 * t_pre + pre
                if kernel is not None:
                    kernel(W_ref, pidx2, t_pre, t_post, pre, post, 1.0, 1.0)
                else:
                    for i in np.nonzero(pre > 0.0)[0]:
                        k = pidx2[i]
                        np.clip(W_ref[i] + (2.0 * t_pre[i] * post[k]
                                            - t_post[k] * pre[i]), 0.0, 1.0,
                                out=W_ref[i])
                t_post = 0.35 * t_post + post
            return W_ref

        W_nb = learn(np.zeros((n, m)), _stdp_delta)
        W_np = learn(np.zeros((n, m)), None)
        if not np.allclose(W_nb, W_np, atol=1e-12):     # 数值等价门槛
            return False
        return float(W_nb.sum()) > 0.5                  # 行为门槛
    except Exception:
        return False


NUMBA_OK = _selftest_numba()


def _predict_edges_np(W, post_idx, pre, n):
    """numpy 回退：scatter 到出边目标（语义与 numba 核一致）。"""
    p = np.zeros(n)
    np.add.at(p, post_idx.ravel(), (W * pre[:, None]).ravel())
    return p
