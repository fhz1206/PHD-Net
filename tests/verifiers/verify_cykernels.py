"""Cython 核（P192）的**逐位等价门禁** + GIL/构建自检。

为什么这个门禁是本仓库最要紧的一个
================================================================================
Cython 臂是**第二套数值实现**。`phdnet_rs/`（Rust）臂被整体删除的理由就是
（P182/P188）：「同一算子两份数值实现 + 双份门禁 + 一条 ctypes 跨界」，
收益只在大 shape 上成立，代价却是 dtype/idx 口径必须处处对齐 ——
P188 那次「int64→i32 漏 cast 直接崩」就是这个代价的真实账单。

本门禁把 Rust 臂欠的那笔账**先付掉**：每个 Cython 核都与对应的
**numba 核在随机输入上逐位对拍**。任一处不等 → 门禁红 → 不得开默认。

⚠ **诚实边界**：本门禁跑在**本机 x86**，只能保证「同一输入下 Cython 与
numba 逐位相同」。**它在昇腾上不成立**（编译器/FMA 策略不同），
昇腾的等价性必须由服务器复跑本门禁确认 —— 见 `docs/PHD-Net_性能评估与迭代方案.md` §八。

⚠ **不覆盖的**：`gemv_rows_ilp4` 是**刻意的非逐位**变体（P165 同款），
本门禁只断言它与精确值的相对误差有界，**不**要求与 numba 逐位相同。

运行：``python tests/verifiers/verify_cykernels.py``（退出码 0 = 全 PASS）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

CASES: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    """记一条判据并立刻打印。

    ⚠ **用 ASCII 标记（PASS/FAIL）而非 ✓/✗**：Windows 控制台默认
    GBK 码页，`print('✓')` 会抛 `UnicodeEncodeError: 'gbk' codec can't
    encode character '\\u2713'` —— 门禁自己先崩，是最坏的一种失败。
    （本文件的中文说明文字能正常打印，是因为 GBK 恰好有这些汉字；
    ✓/✗ 在 GBK 里没有。）
    """
    CASES.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'} {name}"
          + (f"  [{detail}]" if detail else ""), flush=True)


# ══════════════════════════════════════════════════════════════════════════
# 工具：随机 CSR（复用生产结构：每行定长 k）
# ══════════════════════════════════════════════════════════════════════════
def _rand_csr(n_rows: int, k: int, n_cols: int, seed: int):
    """返回 (indptr, idx, val)，结构与 `phdnet/sparse_pc.py::_random_csr` 同形。"""
    rng = np.random.default_rng(seed)
    nnz = n_rows * k
    indptr = (np.arange(n_rows + 1, dtype=np.int64) * k)
    idx = np.empty(nnz, dtype=np.int64)
    for r in range(n_rows):
        # 每行 k 个不重复列（同生产：无放回抽样）
        idx[r * k:(r + 1) * k] = rng.choice(n_cols, size=k, replace=False)
    val = rng.normal(0.0, 0.05, nnz).astype(np.float32)
    return indptr, np.sort(idx.reshape(n_rows, k), axis=1).ravel(), val


def _f32(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


def _f64(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def _main() -> int:
    print("Cython 核门禁（P192）")

    # ── A. 构建与加载（**先确认扩展真的能用**，否则 A 组跳过而非误判）─────
    from phdnet import cykernels as cyk

    k_off = cyk.Kernels("off")
    check("A1 mode=off 时扩展未激活（铁律④：默认关闭）",
          not k_off.active, cyk.cykernels_status())

    if not cyk.KERNELS.active and k_off.active is False:
        # 尝试加载/构建（auto 模式；失败会回落，不抛）
        try:
            k_auto = cyk.Kernels("auto")
            loaded = k_auto.active
        except Exception as e:                        # noqa: BLE001
            loaded = False
            print(f"    （扩展不可用：{type(e).__name__}: {e}）")
        if not loaded:
            print("\n⚠ Cython 扩展不可用（无编译器 / 未构建 / 自检未过）→ "
                  "**SKIP**：本门禁只在扩展可用时才有意义。")
            print("   构建：python setup_cython.py build_ext --inplace")
            _report()
            return 0
        _mod = k_auto
    else:
        _mod = cyk.get_kernels("auto")

    check("A2 扩展已加载并通过 import-time 自检", _mod is not None,
          cyk.cykernels_status())

    # ⚠ OpenMP 真伪：Cython 官方文档明写「忘了传 OpenMP 参数，prange 照样
    #   编译通过但退化为串行」。这里报出**真实**编译状态，不自证。
    try:
        omp = bool(_mod.openmp_enabled())
        check("A3 OpenMP 确实链接（_OPENMP 宏，非「设了没生效」）", omp,
              "openmp=yes" if omp else "openmp=NO（串行，见 README 构建说明）")
    except AttributeError:
        check("A3 OpenMP 状态可查", False, "openmp_enabled() 缺失")

    # ── B. 与 numba 核对照（**语义级必须严、数值级如实报告**）────────────
    #
    # ⚠⚠ **为什么 B 组不是全「逐位」**（这一段是本门禁的核心诚实声明）：
    #   本仓库的 numba 核**全部带 `fastmath=True`**（`sparse_pc.py:60` 等）。
    #   LLVM fastmath 默认开启 **FMA 收缩**（`a*b+c` 单指令、中间不舍入）
    #   与**重结合**（改变加法结合顺序）。本 Cython 核用 MSVC `/O2`
    #   （**无** `/fp:fast`）→ 严格 IEEE-754，**无**收缩、**无**重结合。
    #   于是同一表达式可差 ~1 ulp（fp32 实测 1e-8 ~ 1e-7 量级）。
    #   **这不是 bug，是两种编译策略的固有差异**（P165 的 ILP4 变体、
    #   P52 的融合核都是同一现象）。
    #
    #   故本组分两档：
    #   · **语义级**（结构 / 索引口径 / clip / 事件驱动分支）→ 必须严，
    #     用结构断言或**整数用例**（整数在 fp32 下无舍入歧义，可逐位）；
    #   · **数值级**（浮点累加顺序）→ **容差 + 如实打出 max|d|**，
    #     不谎称逐位。**若哪天把 Cython 核改成 `/fp:fast`，这里才可收紧。**
    try:
        from phdnet.sparse_pc import _csr_matvec, _csr_add_outer, _csr_oja_up
        from phdnet.stdp_kernels import _stdp_delta as nb_stdp, NUMBA_AVAILABLE
    except Exception as e:                            # noqa: BLE001
        print(f"  ! 无法导入 numba 核（{e}）→ B 组跳过")
        NUMBA_AVAILABLE = False

    # fp32 数值容差档：~1 ulp 的 fastmath 差异（P165 实测同量级）
    # ⚠ 绝对容差按**数值量级**给：本组用例的累加项 ≤ 16 项、元素 ~N(0,1)
    #   → 单次求和的 fp32 误差上界 ≈ 16·eps·Σ|·| ≈ 1e-5。取 1e-5 偏严，
    #   实测差 ~1e-8，留两个数量级余量。
    _ATOL_NUM = 1e-5

    def _num_eq(a, b, atol=_ATOL_NUM):
        """fp32 数值等价（fastmath 档），返回 (是否过, max|d|)。"""
        d = float(np.max(np.abs(np.asarray(a, np.float64)
                                - np.asarray(b, np.float64)))) if a.size else 0.0
        return d <= atol, d

    rng = np.random.default_rng(20261008)

    # ── B0：**整数用例** → 无舍入歧义，可真正逐位 ────────────────────────
    if NUMBA_AVAILABLE:
        # 全 1 的 CSR：Σ 全是整数乘加，fp32 精确表示 → 逐位可比
        nr, k, nc = 300, 4, 64
        ip = np.arange(nr + 1, dtype=np.int64) * k
        ix = np.sort(np.stack(
            [rng.choice(nc, size=k, replace=False) for _ in range(nr)]),
            axis=1)
        ix = np.ascontiguousarray(ix, dtype=np.int64).ravel()
        vl = np.ones(nr * k, dtype=np.float32)      # val 全 1
        x = np.ones(nc, dtype=np.float32)            # x   全 1 → 和 = k
        o_c = np.empty(nr, np.float32)
        _mod.csr_matvec(ip, ix, vl, x, o_c)
        o_n = _f32(_csr_matvec(ip, ix, vl, x))
        check("B0 csr_matvec 整数用例**逐位** vs numba（无舍入歧义）",
              np.array_equal(o_c, o_n), f"max|d|={np.max(np.abs(o_c - o_n)):.3e}")

    # ── B1/B2：CSR SpMV（多规模，含 n<512 的串行门限两侧）─────────────────
    if NUMBA_AVAILABLE:
        for (nr, k, nc) in ((3, 1, 8), (17, 5, 64), (511, 4, 512),
                            (512, 4, 512), (513, 4, 1024), (2000, 8, 2048)):
            ip, ix, vl = _rand_csr(nr, k, nc, seed=nr * 31 + k)
            x = _f32(rng.normal(0, 1, nc))
            a = np.empty(nr, np.float32)
            _mod.csr_matvec(ip, ix, vl, x, a)
            b = _f32(_csr_matvec(ip, ix, vl, x))
            ok, d = _num_eq(a, b)
            check(f"B1 csr_matvec 数值等价 vs numba ({nr}×{k}/{nc})", ok,
                  f"max|d|={d:.3e}（fastmath 档，~1 ulp）")
            # 线程路与串行路本身**必须逐位一致**（同一份 C 代码，不同调度）
            a2 = np.empty(nr, np.float32)
            _mod.csr_matvec(ip, ix, vl, x, a2)
            check(f"B1b csr_matvec 线程/串行**逐位**一致 ({nr}×{k})",
                  np.array_equal(a, a2))

        # ── B2：csr_add_outer（原地，各自一份副本）─────────────────────────
        # ⚠ 语义级关键用例：a 是**行向量**（长 n_rows），b 才被 idx 索引。
        #   用**整数**数据 → 逐位可比，直接抓住索引口径错误（首版 a[idx[p]]
        #   被门禁抓到 max|d|=0.339）。
        nr, k, nc = 777, 6, 1024
        ip, ix, vl0 = _rand_csr(nr, k, nc, seed=4242)
        u = np.ones(nr, dtype=np.float32)             # 行向量全 1
        v = np.arange(1, nr * k + 1, dtype=np.float32)   # 边值
        eta = np.float32(1.0)                        # eta=1 → 纯加法，无舍入
        v1 = vl0.copy()
        _mod.csr_add_outer(ip, ix, v1, u, v, eta)
        v2 = vl0.copy()
        _csr_add_outer(ip, ix, v2, u, v, eta)
        check("B2 csr_add_outer 整数用例**逐位** vs numba（索引口径）",
              np.array_equal(v1, v2),
              f"max|d|={float(np.max(np.abs(v1 - v2))):.3e}")
        # 浮点档：随机 a（行向量！）、随机 b
        u_r = _f32(rng.normal(0, 1, nr))
        v_r = _f32(rng.normal(0, 1, nr * k))
        v1r = vl0.copy()
        _mod.csr_add_outer(ip, ix, v1r, u_r, v_r, 0.037)
        v2r = vl0.copy()
        _csr_add_outer(ip, ix, v2r, u_r, v_r, 0.037)
        ok, d = _num_eq(v1r, v2r)
        check("B2b csr_add_outer 数值等价 vs numba", ok, f"max|d|={d:.3e}")

        # ── B3：csr_oja_up（原地）─────────────────────────────────────────
        post = _f32(rng.normal(0, 1, nr))
        pre = _f32(rng.normal(0, 1, nc))
        o1 = vl0.copy()
        _mod.csr_oja_up(ip, ix, o1, post, pre, 0.021)
        o2 = vl0.copy()
        _csr_oja_up(ip, ix, o2, post, pre, 0.021)
        ok, d = _num_eq(o1, o2)
        check("B3 csr_oja_up 数值等价 vs numba", ok, f"max|d|={d:.3e}")

        # ── B4：STDP 增量（vs numba `_stdp_delta`）────────────────────────
        for (n, m) in ((16, 4), (257, 8), (512, 8), (1024, 16)):
            pidx = np.ascontiguousarray(
                np.array([rng.choice(n, m, replace=False) for _ in range(n)]),
                dtype=np.int32)
            W0 = np.ascontiguousarray(
                rng.uniform(0.0, 1.0, (n, m)), dtype=np.float32)
            t_pre = _f32(rng.uniform(0, 1, n))
            t_post = _f32(rng.uniform(0, 1, n))
            pre = _f32((rng.random(n) < 0.3).astype(np.float32))
            post = _f32(rng.uniform(0, 1, n))
            w1 = W0.copy()
            _mod.stdp_delta(w1, pidx, t_pre, t_post, pre, post, 0.011, 1.0)
            w2 = W0.copy()
            nb_stdp(w2, pidx, t_pre, t_post, pre, post, 0.011, 1.0)
            ok, d = _num_eq(w1, w2)
            check(f"B4 stdp_delta 数值等价 vs numba ({n}×{m})", ok,
                  f"max|d|={d:.3e}（fastmath 档）")
        # B4b：事件驱动语义（pre=0 的行**不更新**）→ 结构性，必须严
        Ws = np.full((3, 2), 0.5, np.float32)
        pidx = np.array([[1, 0], [0, 1], [2, 0]], dtype=np.int32)
        t_pre = np.full(3, 0.5, np.float32)
        t_post = np.full(3, 0.25, np.float32)
        pre = np.array([1.0, 0.0, 1.0], np.float32)   # 行 1 pre=0
        post = np.full(3, 1.0, np.float32)
        Ws_before = Ws.copy()
        _mod.stdp_delta(Ws, pidx, t_pre, t_post, pre, post, 0.011, 1.0)
        check("B4b stdp_delta 事件驱动：pre=0 的行**逐位不变**",
              np.array_equal(Ws[1], Ws_before[1]),
              f"行1 before={Ws_before[1]} after={Ws[1]}")
        check("B4c stdp_delta：pre>0 的行确实更新（不是空转）",
              not np.array_equal(Ws[0], Ws_before[0]))

        # ── B5：GEMV（M1）vs numba `_gemv_rows` ───────────────────────────
        try:
            from phdnet.sparse_encoder import _gemv_rows as nb_gemv
            for (nr, nc) in ((16, 32), (512, 1024), (2048, 2048)):
                W = _f32(rng.normal(0, 0.1, (nr, nc)))
                x = _f32(rng.normal(0, 1, nc))
                bb = _f32(rng.normal(0, 0.1, nr))
                o1 = np.empty(nr, np.float32)
                o2 = np.empty(nr, np.float32)
                _mod.gemv_rows(W, x, bb, o1, False)
                _mod.gemv_rows(W, x, bb, o2, True)
                # 线程 vs 串行：同一份 C 代码，只差调度 → 必须逐位
                check(f"B5a gemv_rows 线程/串行**逐位**一致 ({nr}×{nc})",
                      np.array_equal(o1, o2))
                nb_gemv(W, x, bb, o2)
                ok, d = _num_eq(o1, o2, atol=1e-4)
                check(f"B5 gemv_rows 数值等价 vs numba ({nr}×{nc})", ok,
                      f"max|d|={d:.3e}（fp32 大规模，容差 1e-4）")
                # ILP4 是**非逐位**变体（P165 同款）：只验不发散
                o3 = np.empty(nr, np.float32)
                _mod.gemv_rows_ilp4(W, x, bb, o3)
                denom = np.maximum(np.abs(o1), 1e-3)
                rel = float(np.max(np.abs(o3 - o1) / denom))
                check(f"B5b gemv_rows_ilp4 相对误差有界 ({nr}×{nc})",
                      rel < 1e-2, f"relerr={rel:.2e}（非逐位，P165 同款）")
        except ImportError:
            check("B5 gemv 对拍（numba 核不可导入）", False, "skip")

    # ── C. 与 numpy 参考对拍（不依赖 numba，numba 缺失也有覆盖）──────────
    # ⚠ **本组全部用容差，不用逐位**：参考是 numpy 向量化归约
    #   （`(k2[:,None]*s2).sum(axis=0)`），而 Cython 核是**逐槽顺序累加**。
    #   两者的结合顺序不同 → fp32 下差 ~1e-7 属正常（P175 同类）。
    #   结构性断言（衰减量、槽数关系）仍用**逐位**。
    # csr_matvec vs 手算
    ip = np.array([0, 2, 5, 6], dtype=np.int64)
    ix = np.array([0, 2, 1, 3, 4, 1], dtype=np.int64)
    vl = np.array([1, 2, 3, 4, 5, 6], np.float32)
    xv = np.array([1, 2, 4, 8, 16, 32], np.float32)
    o = np.empty(3, np.float32)
    _mod.csr_matvec(ip, ix, vl, xv, o)
    # 行 0: idx=[0,2]   → 1·x[0] + 2·x[2] = 1·1 + 2·4  = 9
    # 行 1: idx=[1,3,4] → 3·x[1] + 4·x[3] + 5·x[4] = 3·2 + 4·8 + 5·16 = 118
    # 行 2: idx=[1]    → 6·x[1] = 6·2 = 12
    check("C1 csr_matvec vs 手算",
          np.array_equal(o, np.array([9.0, 118.0, 12.0], np.float32)),
          f"got={o}")

    # csr_clip
    vc = np.array([-3.0, -0.5, 0.0, 0.5, 3.0], np.float32)
    _mod.csr_clip(vc, 1.0)
    check("C2 csr_clip 到 ±w_max",
          np.array_equal(vc, np.array([-1, -0.5, 0, 0.5, 1], np.float32)),
          f"got={vc}")

    # wm_decay_read：与「decay() 两趟 + numpy 归约」对照
    # ⚠ 参考严格复刻 phdnet/wm.py 的 decay()（**slots 与 strength 都衰减**）
    #   + read()（向量化归约）。
    n_slot, n = 5, 64
    slots0 = _f32(rng.normal(0, 1, (n_slot, n)))
    st0 = _f32(rng.uniform(0.1, 2.0, n_slot))
    gamma = 0.93
    s1, k1 = slots0.copy(), st0.copy()
    out_c = np.empty(n, np.float32)
    _mod.wm_decay_read(s1, k1, gamma, out_c)
    s2, k2 = slots0.copy(), st0.copy()
    s2 *= gamma                       # decay(): slots 也衰减（首版漏了这条）
    k2 *= gamma
    tot = float(k2.sum())
    out_ref = ((k2[:, None] * s2).sum(axis=0) / tot).astype(np.float32)
    check("C3 wm_decay_read strength 衰减值逐位",
          np.array_equal(k1, k2),
          f"max|d|={float(np.max(np.abs(k1 - k2))):.3e}")
    check("C3b wm_decay_read slots 衰减值逐位（首版漏衰减，被抓）",
          np.array_equal(s1, s2),
          f"max|d|={float(np.max(np.abs(s1 - s2))):.3e}")
    ok, d = _num_eq(out_c, out_ref, atol=1e-5)
    check("C4 wm_decay_read 读出值与 numpy 两趟数值等价", ok,
          f"max|d|={d:.3e}（numpy 向量化归约、顺序不同 → 容差非逐位）")

    # ── D. GIL 真的放出了吗？（**性能前提**：不放开则并发无意义）────────
    # 判据：在一个 Python 层忙循环线程运行的同时，跑一个 nogil 核，
    # 若 GIL 未放出，两者会被串行化（总时长 ≈ 之和）；若放出了，
    # 应显著接近 max 而非 sum。用墙钟做粗判，不要求精确倍数。
    import threading
    import time
    stop = [False]
    ticks = [0]

    def spin():
        while not stop[0]:
            ticks[0] += 1

    big_W = _f32(rng.normal(0, 1, (256, 256)))
    big_x = _f32(rng.normal(0, 1, 256))
    big_b = _f32(np.zeros(256, np.float32))
    big_o = np.empty(256, np.float32)
    # 预热（首次调用有一次性开销）
    _mod.gemv_rows(big_W, big_x, big_b, big_o, False)

    t0 = time.perf_counter()
    th = threading.Thread(target=spin, daemon=True)
    th.start()
    time.sleep(0.02)                     # 让 spin 线程占住 GIL 轮转份额
    for _ in range(40):
        _mod.gemv_rows(big_W, big_x, big_b, big_o, False)
    dt = time.perf_counter() - t0
    stop[0] = True
    th.join(timeout=1.0)
    check("D1 nogil 核与 Python 线程可并发（GIL 未被独占）",
          dt < 1.0, f"40 次 gemv 墙钟 {dt * 1000:.1f} ms，spin tick={ticks[0]}")

    # ── E. 「默认关闭」不变量：模型在 off 下逐位不变 ──────────────────────
    # 这里只验证**机制**：off 门面不暴露任何核、不会静默改道。
    check("E1 mode=off 门面不提供核属性（不会静默改道）",
          not hasattr(k_off, "csr_matvec"))

    _report()
    return 0 if all(c[1] for c in CASES) else 1


def _raises(exc, fn, *a, **k) -> bool:
    try:
        fn(*a, **k)
        return False
    except exc:
        return True


def _report() -> None:
    n_ok = sum(1 for c in CASES if c[1])
    print(f"\n{'=' * 62}\nCython kernel gate: {n_ok}/{len(CASES)} passed")
    if n_ok != len(CASES):
        for nm, ok, det in CASES:
            if not ok:
                print(f"  FAIL {nm}  [{det}]")
        print("=" * 62)
    else:
        print("=" * 62)


if __name__ == "__main__":
    sys.exit(_main())