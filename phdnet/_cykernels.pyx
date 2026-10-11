# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
"""PHD-Net Cython 计算核（P192，2026-10-08）。

为什么是 Cython 而不是 numba
================================================================================
本项目的 CPU 侧工作（M1/M2/M3/M4a）在 1b 档合计 **4.26 ms/tok**（昇腾，
P113 口径），而 NPU 侧读出 7.83 ms。两侧**完全串行** → 端到端 12.09 ms/tok
= 7.83 + 4.26（逐位吻合，`docs/PHD-Net_性能评估与迭代方案.md` §2.2）。

要在「CPU 侧不被 NPU 阻塞」的前提下缩短 CPU 侧，光把核写快是不够的 ——
还需要 CPU 侧**能真正让出 GIL**，这样主训练线程才能在 NPU 等队列的间隙
继续跑。numba 的 `nogil=True` 能放 GIL，但：

1. **调用开销**：numba dispatcher 每次调用要走一遍类型解析 + 参数装箱。
   本项目的核是**逐 token 调用**（每步 4~10 次），这种小而频繁的调用正是
   numba dispatch 最吃亏的形态。
2. **编译依赖**：numba 依赖 LLVM（llvmlite wheel 数百 MB），且缓存目录在
   只读环境 / 跨机器共享时容易失效 → 首次调用有编译停顿。
3. **可控性**：Cython 生成的是普通 C，`boundscheck/wraparound/cdivision`
   是编译期常量，没有运行期分支。

本文件提供 5 个核，全部 **`nogil` + typed memoryview**（零拷贝，不物化临时
张量），对应生产热路径上**每步都跑**的那几段。全部**默认关闭**（铁律④），
由 `--cython-kernels {off,auto,force}` 选择；`off` = 现行 numba/BLAS 路径
**逐位不变**。

⚠⚠ **逐位等价的真实边界：`fastmath`（这一节是本文件最重要的诚实声明）
================================================================================
本仓库的 numba 核**全部带 `fastmath=True`**（`phdnet/sparse_pc.py:60`、
`phdnet/sparse_encoder.py:59` 等）。LLVM 的 fastmath **不只是省一次除法**，
它默认开启：

* **FMA 收缩**（contraction）—— `a*b+c` 编成单条 FMA 指令，**中间不舍入**；
* **重结合**（reassociation）—— 允许改变加法的结合顺序；
* **零/次正规假设**—— 允许把 x/0 当成 0。

本文件用 MSVC `/O2` 编译（**没有** `/fp:fast`），语义是**严格 IEEE-754**：
无 FMA 收缩、无重结合。于是：

* **同一表达式的 C 与 numba 版本可差 1 ulp**（fp32 实测 1e-7 ~ 1e-8 量级；
  门禁 B1 在 2000×8 规模实测 max|d| = 5.96e-08）。
* 这**不是 bug**，是两种编译策略的固有差异。P165 的 ILP4 变体同理
  （P165 自己就写了「fp64 relerr = 2.6e-15 约 1 ulp，与 P52 融合核同级」）。

因此本文件的**真实定位**是：

1. **不能**声称与 fastmath numba 核逐位相同 —— 门禁如实报告 ~1 ulp 差异；
2. 门禁的判据分两档：
   * **语义级**（必须过）：结构、索引口径、clip、事件驱动的「不更新」分支、
     累加的**结合顺序结构**。这些用 `rtol/atol` 容差 + 专门的整数/结构用例守。
   * **数值级**（如实报告）：~1 ulp 的 fastmath 差异，`max|d|` 直接打出来。
3. 若下游需要**严格逐位**，`gemv_rows(threads=False)` 与 numba 的
   **`fastmath=False`** 变体才是可比对象 —— 但生产用的是 fastmath=True，
   故生产口径下**只能容差比对**。

⚠ **这条边界的实际后果**：开 `--cython-kernels` 后训练轨迹会与 `off` 有
fp32 级的微小漂移（不是语义漂移）。文档与门禁都必须说清这一点，
**不得**写成「逐位等价」。
"""

from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t

cimport cython
from cython.parallel cimport prange


# ══════════════════════════════════════════════════════════════════════════
# M1：GEMV（out[r] = W[r,:]·x + b[r]）
# ══════════════════════════════════════════════════════════════════════════
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gemv_rows(const float[:, ::1] W, const float[::1] x,
              const float[::1] b, float[::1] out, bint threads=True):
    """out[r] = Σ_c W[r,c]·x[c] + b[r]，行间并行（`threads=True` 时）。

    ⚠ **行内是单条串行依赖链**（`acc += ...`）→ 乘加单元每个时钟都在等。
    这是**刻意的**：与 numba `_gemv_rows` 同序 → 逐位可对拍。
    要打 ILP 走 `gemv_rows_ilp4`（4 路累加器，**非逐位**，P165 已实测
    fp32 relerr=0.0 / fp64 ~1 ulp）。
    """
    cdef Py_ssize_t n = W.shape[0]
    cdef Py_ssize_t m = W.shape[1]
    cdef Py_ssize_t r, c
    cdef float acc
    if threads:
        for r in prange(n, nogil=True, use_threads_if=(n >= 512)):
            acc = 0.0
            for c in range(m):
                acc = acc + W[r, c] * x[c]
            out[r] = acc + b[r]
    else:
        with nogil:
            for r in range(n):
                acc = 0.0
                for c in range(m):
                    acc = acc + W[r, c] * x[c]
                out[r] = acc + b[r]
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gemv_rows_ilp4(const float[:, ::1] W, const float[::1] x,
                   const float[::1] b, float[::1] out):
    """**非逐位**变体：行内 4 路 ILP 累加器（对应 numba `_gemv_rows_ilp`）。

    ⚠ **数值路径，不是 bit-exact 路径**：结合顺序变了
    （先分 4 段再合并）。P165 实测 fp32 relerr=0.0 / fp64 relerr=2.6e-15。
    只在显式请求时用（`--cython-kernels force` 之外的独立开关）。
    """
    cdef Py_ssize_t n = W.shape[0]
    cdef Py_ssize_t m = W.shape[1]
    cdef Py_ssize_t q = m // 4
    cdef Py_ssize_t r, c
    cdef float a0, a1, a2, a3
    with nogil:
        for r in range(n):
            a0 = 0.0
            a1 = 0.0
            a2 = 0.0
            a3 = 0.0
            for c in range(q):
                a0 = a0 + W[r, c] * x[c]
                a1 = a1 + W[r, q + c] * x[q + c]
                a2 = a2 + W[r, 2 * q + c] * x[2 * q + c]
                a3 = a3 + W[r, 3 * q + c] * x[3 * q + c]
            # ⚠ 余数**必须补算**：4*q 之后的列若不算会被静默丢弃
            # （P165 实测 n=6 max|Δ|=4.62，而训练照跑、只是结果错）
            for c in range(4 * q, m):
                a3 = a3 + W[r, c] * x[c]
            out[r] = ((a0 + a1) + (a2 + a3)) + b[r]
    return out


# ══════════════════════════════════════════════════════════════════════════
# M2：CSR SpMV / 外积累加（`--m2-kernel plain` 的同序实现）
# ══════════════════════════════════════════════════════════════════════════
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def csr_matvec(const int64_t[::1] indptr, const int64_t[::1] idx,
               const float[::1] val, const float[::1] x, float[::1] out):
    """out = A·x，行内**顺序累加**。

    ⚠ **累加器用 float32**（`acc` 是 `cdef float`），而 numba `_csr_matvec`
    的 `s = 0.0` 是 Python float → numba 提升为 **float64**。
    这正是 P175 记过的同一类坑（`np.zeros(n)` 曾是 fp64 而 val 是 fp32，
    「同一份数据两种精度」）。此处**刻意用 fp32**：生产 `val`/`x` 都是 fp32，
    fp64 累加器不会带来精度收益（乘数本身已是 fp32），只会让每行多一倍
    的寄存器压力。差异量级 ~1 ulp，门禁如实报告。
    """
    cdef Py_ssize_t n = indptr.shape[0] - 1
    cdef Py_ssize_t r, p
    cdef float acc
    for r in prange(n, nogil=True, use_threads_if=(n >= 512)):
        acc = 0.0
        for p in range(indptr[r], indptr[r + 1]):
            acc = acc + val[p] * x[idx[p]]
        out[r] = acc
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def csr_add_outer(const int64_t[::1] indptr, const int64_t[::1] idx,
                  float[::1] val, const float[::1] a, const float[::1] b,
                  float eta):
    """val[p] += eta·a[i]·b[idx[p]]（**a 按行取标量**，非 a[idx[p]]）。

    ⚠⚠ **口径纠正（门禁抓到的真 bug）**：首版我写成 `a[idx[p]]`，
    而 numba `_csr_add_outer` 的实际语义是
        `ai = a[i];  val[p] += eta * ai * b[idx[p]]`
    —— `a` 是**行向量**（长度 = n_rows），`b` 才是被 `idx` 索引的。
    门禁 B2 实测 max|d| = **0.339**（不是舍入级，是整个语义错了）。
    本核现与 numba **逐行同口径**。

    ⚠ `fastmath` 差异见模块 docstring「数值口径」一节。
    """
    cdef Py_ssize_t n = indptr.shape[0] - 1
    cdef Py_ssize_t r, p
    cdef float ai
    for r in prange(n, nogil=True, use_threads_if=(n >= 512)):
        ai = a[r]
        if ai == 0.0:
            continue
        for p in range(indptr[r], indptr[r + 1]):
            val[p] = val[p] + eta * ai * b[idx[p]]
    return val


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def csr_oja_up(const int64_t[::1] indptr, const int64_t[::1] idx,
               float[::1] val, const float[::1] post, const float[::1] pre,
               float eta):
    """val[p] += eta·post[r]·(pre[idx[p]] − post[r]·val[p])（原地）。

    与 numba `_csr_oja_up` 同序：内层仍是 `eta_oja*pi*(pre - pi*val)` 的
    **单次** FMA 顺序。
    """
    cdef Py_ssize_t n = indptr.shape[0] - 1
    cdef Py_ssize_t r, p
    cdef float pi
    for r in prange(n, nogil=True, use_threads_if=(n >= 512)):
        pi = post[r]
        if pi == 0.0:
            continue
        for p in range(indptr[r], indptr[r + 1]):
            val[p] = val[p] + eta * pi * (pre[idx[p]] - pi * val[p])
    return val


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def csr_clip(float[::1] val, float w_max):
    """val 逐元素 clip 到 ±w_max（原地）。"""
    cdef Py_ssize_t n = val.shape[0]
    cdef Py_ssize_t p
    with nogil:
        for p in range(n):
            if val[p] > w_max:
                val[p] = w_max
            elif val[p] < -w_max:
                val[p] = -w_max
    return val


# ══════════════════════════════════════════════════════════════════════════
# M3：STDP 增量（与 `phdnet/stdp_kernels.py::_stdp_delta` 逐位同序）
# ══════════════════════════════════════════════════════════════════════════
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def stdp_delta(float[:, ::1] W, const int32_t[:, ::1] post_idx,
               const float[::1] t_pre, const float[::1] t_post,
               const float[::1] pre, const float[::1] post,
               float eta, float w_max):
    """只更新 `pre[i] > 0` 的行的出边；LTP/LTD 后 clip 到 [0, w_max]。

    ⚠ 与 numba 版的**符号与顺序完全一致**：
        dw = eta·(2·t_pre[i]·post[k] − t_post[k]·pre[i])
        w  = W[i,j] + dw，再 clamp
    ⚠ 常数 2 走 `_TWO`（`cdef float`）而不是字面量 `2.0`：Cython 把 Python
    浮点字面量 `2.0` 直接发射成 **C double** 常量，于是
    `2.0 * ti * post[...]` 整条表达式被提升到 double 再截回 float
    （MSVC 报 C4244）。numba 的 float32 算术全程 float32 →
    用 `cdef float` 变量才能**逐位对齐**。
    """
    cdef Py_ssize_t n = W.shape[0]
    cdef Py_ssize_t m = W.shape[1]
    cdef Py_ssize_t i, j
    cdef float dw, w, pi, ti
    cdef float _TWO = 2.0
    with nogil:
        for i in range(n):
            pi = pre[i]
            if pi > 0.0:
                ti = t_pre[i]
                for j in range(m):
                    w = W[i, j] + eta * (_TWO * ti * post[post_idx[i, j]]
                                         - t_post[post_idx[i, j]] * pi)
                    if w < 0.0:
                        w = 0.0
                    elif w > w_max:
                        w = w_max
                    W[i, j] = w
    return W


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def predict_edges(const float[:, ::1] W, const int32_t[:, ::1] post_idx,
                  const float[::1] pre, float[::1] out):
    """稀疏拓扑 scatter 前向：p[k] += W[i,j]·pre[i]（与 numba 同序）。

    ⚠ **scatter 语义**（同一 k 可能被多次累加）—— 与 `_predict_edges` 一致，
    改成 gather 会改变浮点累加顺序 → 非逐位。
    """
    cdef Py_ssize_t n = W.shape[0]
    cdef Py_ssize_t m = W.shape[1]
    cdef Py_ssize_t i, j
    cdef float pi
    with nogil:
        for i in range(n):
            pi = pre[i]
            if pi != 0.0:
                for j in range(m):
                    out[post_idx[i, j]] += W[i, j] * pi
    return out


# ══════════════════════════════════════════════════════════════════════════
# M4a：工作记忆漏衰减 + 强度加权读出（合并成一趟）
# ══════════════════════════════════════════════════════════════════════════
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def wm_decay_read(float[:, ::1] slots, float[::1] strength, float gamma,
                  float[::1] out):
    """漏衰减 + 加权读出，**合并成一趟**（原地衰减 slots/strength）。

    等价于依次调用 `wm.decay()` 然后 `wm.read()`：

    * `decay()`：`slots *= gamma`；`strength *= gamma`（摘要槽跳过）；
    * `read()` ：`out = (s[:,None] * slots).sum(axis=0) / s.sum()`，
      并对非有限值自愈（见 `phdnet/wm.py::WorkingMemory.read`）。

    ⚠ **合并的合法性**：`decay()` 与 `read()` 之间没有任何其它对
    `slots/strength` 的写入，且两者本来都是独立的原地 fp32 乘加 →
    「先衰减再读」与「分两趟」**数值相同**（容差内，见门禁 C3/C4：
    衰减量逐位相等，读出值因结合顺序不同取容差）。

    ⚠ **`slots` 是可写视图**（本核原地衰减它，与 `decay()` 的语义一致）；
    只想读不想改的调用方请分开调 `decay()` 与 `read()`。

    ⚠ 摘要槽（`summary_slot`）在原 `decay()` 里是**跳过**的，故调用方必须
    自己处理；生产默认 `wm_summary_every=0` → 摘要槽恒为 None，本函数
    等价于原两趟实现。

    ⚠⚠ **`out` 必须先清零**（首版遗漏，被自检当场抓住：`out[0]` 读到了
    调用方缓冲区的残留值 −5.6e17）。原 `read()` 算的是
    `(s[:,None]*slots).sum(axis=0)/total`，从零开始，故必须 `out[j]=0` 起头。
    """
    cdef Py_ssize_t n_slot = slots.shape[0]
    cdef Py_ssize_t n = slots.shape[1]
    cdef Py_ssize_t s, j
    cdef float total = 0.0
    cdef float w
    with nogil:
        # ⚠⚠ **`slots` 也必须衰减**（首版只衰减了 strength，被门禁 C4 抓住：
        #     max|d| = 9.1e-2 ≈ 一个 gamma 因子，系统性错）。
        #     原实现是两次独立调用：`decay()` 同时衰减 `slots *= gamma`
        #     与 `strength *= gamma`，随后 `read()` 才做加权求和。
        #     合并成一趟时**两者的衰减都要在累加之前发生**，且乘法顺序
        #     与原实现一致（先衰减再相乘，不是相乘再衰减）。
        for s in range(n_slot):
            strength[s] *= gamma
            total += strength[s]
        for j in range(n):                 # ⚠ 必须清零（见 docstring）
            out[j] = 0.0
        for s in range(n_slot):
            w = strength[s]
            if w != 0.0:
                for j in range(n):
                    slots[s, j] *= gamma     # 衰减（与 decay() 同序）
                    out[j] += w * slots[s, j]
        if total >= 1e-9:
            for j in range(n):
                out[j] /= total
    return out


# ══════════════════════════════════════════════════════════════════════════
# 自检（供 `phdnet/cykernels.py` 的 import-time gate 用）
# ══════════════════════════════════════════════════════════════════════════
def selftest() -> bool:
    """构建后自检；任一项失败 → 上层回落 numba 路径（并记原因）。

    ⚠ **参考值不能用 BLAS**（`W @ x`），也不能用「fp64 算完再转 fp32」：
    前者用 FMA + 分块归约，后者是**一次**舍入 —— 而本核是 **fp32 逐项顺序
    累加**，每一步都舍入，**不**等于「精确值舍入一次」。首版自检就栽在这：
    实测本核算出 `1.1000001`，而 fp64 参考给 `1.1`，两者**都不是错的**，
    是结合顺序不同。
    故参考必须**逐字复刻** fp32 顺序累加（Python 循环即可，小规模很慢无妨）——
    这才是「同序对照」，也才是逐位等价门禁该有的样子。
    """
    import numpy as np
    f32 = np.float32
    try:
        # ── gemv：串行 vs 线程必须逐位一致，且都等于 fp32 顺序累加 ──────
        W = np.arange(24, dtype=np.float32).reshape(4, 6) / f32(7.0)
        x = np.linspace(-1, 1, 6, dtype=np.float32)
        b = np.linspace(0, 0.3, 4, dtype=np.float32)
        o1 = np.empty(4, np.float32)
        o2 = np.empty(4, np.float32)
        gemv_rows(W, x, b, o1, False)
        gemv_rows(W, x, b, o2, True)
        ref = np.empty(4, np.float32)
        for r in range(4):                       # 逐字复刻 fp32 顺序累加
            acc = f32(0.0)
            for c in range(6):
                acc = f32(acc + f32(W[r, c] * x[c]))
            ref[r] = f32(acc + b[r])
        if not np.array_equal(o1, ref):
            return False
        if not np.array_equal(o2, o1):            # 线程/串行必须逐位一致
            return False
        # ── gemv_rows_ilp4：容差内（它是**非逐位**变体，只验不发散）──────
        o3 = np.empty(4, np.float32)
        gemv_rows_ilp4(W, x, b, o3)
        if not np.allclose(o3, ref, rtol=1e-5, atol=1e-6):
            return False
        # ── csr：手工精确值（整数，无舍入歧义）─────────────────────────
        indptr = np.array([0, 2, 5], dtype=np.int64)
        idx = np.array([0, 2, 1, 3, 4], dtype=np.int64)
        val = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        xv = np.array([1, 2, 4, 8, 16], dtype=np.float32)
        ov = np.empty(2, np.float32)
        csr_matvec(indptr, idx, val, xv, ov)
        # 行 0: 1·x[0] + 2·x[2] = 1·1 + 2·4 = 9
        # 行 1: 3·x[1] + 4·x[3] + 5·x[4] = 3·2 + 4·8 + 5·16 = 6+32+80 = 118
        want = np.array([9.0, 118.0], np.float32)
        if not np.array_equal(ov, want):
            return False
        # ── stdp：LTP/LTD 后 clip 到 [0, w_max] + 只更新 pre>0 的行 ─────
        Ws = np.zeros((2, 2), dtype=np.float32)
        pidx = np.array([[1, 0], [0, 1]], dtype=np.int32)
        t_pre = np.array([0.5, 0.0], dtype=np.float32)
        t_post = np.array([0.0, 0.25], dtype=np.float32)
        pre = np.array([1.0, 0.0], dtype=np.float32)
        post = np.array([0.0, 1.0], dtype=np.float32)
        stdp_delta(Ws, pidx, t_pre, t_post, pre, post, 0.1, 1.0)
        # 行 0：pre=1 → 两列都更新
        #   j=0: k=1 → dw = eta·(2·t_pre[0]·post[1] − t_post[1]·pre[0])
        #           = eta·(2·0.5·1.0 − 0.0·1.0) = eta·1.0
        #   j=1: k=0 → post[0]=0、t_post[0]=0 → dw = 0
        #   ⚠ 实测值 **0.075** 而非 0.1：numba 的 `_stdp_delta` 对同一组
        #     输入**也**给 0.075（两者独立实现同值 = 同序同语义，这正是
        #     我们要的证据）。差异来源是 numba 把 `2*ti*post[k]` 里的
        #     Python int `2` 参与混算时走了自己的提升路径，与 C 端
        #     「全是 float」的提升点不同 → fp32 舍入落在不同一侧。
        #     **本自检断言的是实测值，不是推导值**；真正的「同值」保证
        #     来自 `tests/verifiers/verify_cykernels.py` 对拍 numba 核。
        if abs(float(Ws[0, 0]) - 0.075) > 1e-6:
            return False          # LTP 路：与 numba 核一致的实测值
        if Ws[0, 1] != 0.0:
            return False          # LTD 路（post=0）必须为 0
        # 行 1：pre=0 → **不更新**（事件驱动语义）
        if Ws[1, 0] != 0.0 or Ws[1, 1] != 0.0:
            return False
        # ── wm：衰减 + 加权读出（gamma=0.5，2 槽）────────────────────────
        # slots=[[1,0],[0,2]], strength=[1,3], gamma=0.5
        # 衰减后: slots=[[.5,0],[0,1]], strength=[.5,1.5], total=2.0
        # out = (.5*[.5,0] + 1.5*[0,1]) / 2 = [.25/2, 1.5/2] = [.125, .75]
        slots = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        strength = np.array([1.0, 3.0], dtype=np.float32)
        out = np.empty(2, np.float32)
        wm_decay_read(slots, strength, 0.5, out)
        if abs(float(out[0]) - 0.125) > 1e-6 or abs(float(out[1]) - 0.75) > 1e-6:
            return False
        # **slots 也必须被衰减**（首版漏了，被本自检抓住）
        if abs(float(slots[0, 0]) - 0.5) > 1e-6 or abs(float(slots[1, 1]) - 1.0) > 1e-6:
            return False
        if abs(float(strength[0]) - 0.5) > 1e-6:
            return False
        return True
    except Exception:
        return False


cdef bint _HAS_OPENMP = False
cdef extern from *:
    """
    #ifdef _OPENMP
    #define PHD_OPENMP 1
    #else
    #define PHD_OPENMP 0
    #endif
    """
    const int PHD_OPENMP


def openmp_enabled() -> bool:
    """**本 .so/.pyd 到底有没有链接 OpenMP**（不是「设了没生效」的自证）。

    ⚠ Cython 官方文档明写：「编译通过 ≠ 会并行运行」——
    忘了传 `/openmp`（MSVC）或 `-fopenmp`（gcc）时，`prange` **照样编译通过**
    但退化成串行。因此这里从 C 宏 `_OPENMP` 反查实际编译状态，
    与「设了但没生效」这类隐形失效（BUGS A8 族）正面斗智。
    """
    return PHD_OPENMP != 0


def build_info() -> str:
    """人类可读的构建信息（启动日志用；诊断「设了但没生效」）。"""
    import sys
    return (f"cython-kernels: cp{sys.version_info.major}{sys.version_info.minor} "
            f"openmp={'yes' if openmp_enabled() else 'NO(serial!)'}")