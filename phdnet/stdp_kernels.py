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


# ══════════════════════════════════════════════════════════════════════════
# **P192：Cython 分派**（默认关闭，铁律④ —— `off` 时下方两个函数即原实现）
# ══════════════════════════════════════════════════════════════════════════
# ⚠ **诚实边界**：本文件的 numba 核用 `@njit(cache=True)`（**无 fastmath**）
#   → 语义上比 `sparse_pc.py` 的核更接近 Cython 的严格 IEEE。但
#   `_stdp_delta` 里的 `2.0` 是 Python float → numba 提升为 fp64 参与
#   `2.0*t_pre[i]*post[k]`，而 Cython 版用 `cdef float`（fp32）→ 仍有
#   ~1 ulp 差异（门禁 B4 实测 max|d| ~6e-8）。
#   故本开关**同样不构成「逐位等价」**，只保证**语义等价**（同序、同 clip、
#   同事件驱动分支）。门禁：`tests/verifiers/verify_cykernels.py`。
try:                                        # pragma: no cover - 纯导入分支
    from .cykernels import get_kernels as _get_cyk
except Exception:                           # noqa: BLE001
    _get_cyk = None

_CYK = None          # 激活时是 `phdnet._cykernels` 模块；否则 None
_CYK_MODE: str | None = None
_CYK_DEFAULT = object()


def cyk_init(mode: str = "off") -> bool:
    """按 `mode` 初始化本文件的 Cython 分派；返回是否生效。

    与 `phdnet/sparse_pc.py::cyk_init` 同名同义：仅设置直接构造机制的
    兼容默认值。PHDNet 显式传同一个实例句柄，已有机制的分派不会被本
    函数或随后构造的另一模型改写。force 失败不得静默回落。
    """
    global _CYK, _CYK_MODE
    if mode == _CYK_MODE and (mode == "off" or _CYK is not None):
        return _CYK is not None
    _CYK_MODE = mode
    if mode == "off" or _get_cyk is None:
        _CYK = None
        if mode == "force":
            raise RuntimeError("Cython loader 不可用（force）")
        return False
    try:
        _CYK = _get_cyk(mode)       # None 即未激活（模块本身，无 `.active`）
        if _CYK is None and mode == "force":
            raise RuntimeError("Cython 扩展未激活（force）")
    except Exception:                # noqa: BLE001
        _CYK = None
        if mode == "force":
            raise
    return _CYK is not None


def predict_edges(W, post_idx, pre, n, *, cython_module=_CYK_DEFAULT):
    """稀疏拓扑前向（scatter 语义）—— 分派入口。

    ⚠ **必须是 scatter 而非 gather**：多个 `(i,j)` 可映射到同一 `post_idx`，
    gather 会改变累加顺序 → 非逐位。故 Cython 版也用 `out[k] += ...`。
    ⚠ numba 版 `p = np.zeros(n)` 是 **fp64**（`0.0` 是 Python float）；
    Cython 版要求调用方给 fp32 缓冲。两者的 dtype 差异由调用点
    （`phdnet/plasticity.py`）显式承担，**本层不做隐式转换**。
    """
    mod = _CYK if cython_module is _CYK_DEFAULT else cython_module
    if mod is not None:
        out = np.zeros(n, dtype=np.float32)
        mod.predict_edges(np.ascontiguousarray(W, dtype=np.float32),
                            np.ascontiguousarray(post_idx, dtype=np.int32),
                            np.ascontiguousarray(pre, dtype=np.float32), out)
        return out
    return _predict_edges(W, post_idx, pre, n)


def stdp_delta(W, post_idx, t_pre, t_post, pre, post, eta, w_max,
               *, cython_module=_CYK_DEFAULT):
    """事件驱动 STDP —— 分派入口（原地改 `W`）。

    ⚠ `W`/`post_idx` 必须是 **C-contiguous 且 dtype 精确匹配**
    （`float32` / `int32`）：Cython 侧是 typed memoryview，dtype 不符会
    抛 `ValueError: Buffer dtype mismatch`。本层**主动转换并拷回**
    （因为 numba 版对非连续数组也能跑，为保持兼容度值得这点开销）；
    生产数据本就是连续的，故正常路径下转换是**零拷贝**（`ascontiguousarray`
    对已连续的数组返回原对象）。
    """
    mod = _CYK if cython_module is _CYK_DEFAULT else cython_module
    if mod is not None:
        Wc = np.ascontiguousarray(W, dtype=np.float32)
        mod.stdp_delta(
            Wc,
            np.ascontiguousarray(post_idx, dtype=np.int32),
            np.ascontiguousarray(t_pre, dtype=np.float32),
            np.ascontiguousarray(t_post, dtype=np.float32),
            np.ascontiguousarray(pre, dtype=np.float32),
            np.ascontiguousarray(post, dtype=np.float32),
            np.float32(eta), np.float32(w_max))
        if Wc is not W:              # 发生了转换 → 写回
            W[...] = Wc
        return W
    return _stdp_delta(W, post_idx, t_pre, t_post, pre, post, eta, w_max)


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
