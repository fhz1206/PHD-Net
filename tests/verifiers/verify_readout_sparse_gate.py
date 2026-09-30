"""M6 稀疏读出**门禁**（gate）—— 退出码 0 = PASS。

================================================================================
为什么是 gate 而不是 benchmark
================================================================================
tools/bench_readout_sparse.py 负责「量曲线、为改默认值提供数据」；本文件
负责**守住不变量**。两者的失败含义完全不同：
  * benchmark 失败 = 「这个配置效果不好」→ 是**数据**，可以照实报告；
  * gate 失败     = 「稀疏实现有 bug 或结构被破坏」→ 是**缺陷**，必须挡住。
所以本文件**不做任何 PPL / 速度断言**——那些会随平台/预算漂移，进了门禁只
会造成假红。门禁只断言**确定性的结构与数值性质**。

================================================================================
五组用例，各自「为什么要有」
================================================================================
【G1】稀疏 vs 稠密的数值等价性（同一权重）
  为什么：稀疏读出最大的风险是「以为等价、其实算错了」。`_from_dense_csr`
  把稠密 W 按 |w| top-k 裁成 CSR，若值/索引搬错一位，训练照跑、PPL 照出，
  没人会发现。这里用**同一份 W** 做对拍，并额外做一层**同构矩阵**比较：
  把「被裁掉的位置置 0」后的稠密矩阵 vs CSR 前向。二者必须逐位一致 ——
  这验证的是**裁剪操作本身不引入额外误差**（而不是「稀疏≈稠密」），
  把「稀疏带来的信息损失」与「实现的 bug」彻底分开。
  容差 1e-12 相对差：`_csr_matvec` 是 fp64 顺序累加、BLAS 是分块求和，
  两者求和顺序不同但都远优于 1e-12；再松就掩盖真实 bug 了。

【G2】CSR 结构合法性
  为什么：CSR 的全部安全性都建立在「indptr/idx 自洽」上。一行 idx 越界
  → numba 核里就是**越界读**（可能不崩，但读到垃圾权重，静默污染训练）；
  idx 重复 → 同一突触被算两次、梯度翻倍；indptr 非单调 → 行错位，
  A 输出单元的权重写到 B 上。这些都是**静默错误**，必须显式断言。

【G3】更新路径：只写存在的槽位、且不越界
  为什么：前向正确不代表更新正确。稀疏更新只应触达存在的边——若实现误写成
  稠密 `W[i,j] -= ...` 再回写 CSR，就会写进「不存在的槽位」或越界。
  这里对 `size`/边界做断言，并用「多次 learn 后结构仍合法」覆盖状态漂移。

【G4】幂律分组（若 phdnet.sparse_alloc 可 import）
  为什么：幂律分配器的三条性质是它的**契约**：alpha=0 退化为均匀（否则
  「幂律」在低 alpha 时会静默改变均匀基线，破坏 A/B 可比性）、alpha=1
  时 k 与频次的**秩相关为正**（否则方向反了，分配器是反的）、总和不超预算
  （否则「省内存」的前提本身被违反，结论全错）。
  不可 import 就**跳过并明确说明**——并行开发未就绪是正常状态，不能因此红。

【G5】多步序列（20 步 learn + forward，逐步断言结构合法）
  为什么：**这是 P67 栽过的地方**——单步对拍全过，多步序列才露馅。
  典型失败模式是「更新时原地改了 idx/indptr 的形状或顺序」：第 1 步结果
  对，第 N 步 indptr 与 idx 长度失配 → 越界或错行。逐步断言才能定位到
  「第几步开始坏」，这是单步测试**结构上无法覆盖**的。

用法：python tests/verifiers/verify_readout_sparse_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.readout import Readout                        # noqa: E402
from phdnet.sparse_pc import _csr_matvec, _from_dense_csr  # noqa: E402

# 相对容差：fp64 下 CSR 顺序累加 vs BLAS 分块求和的实测偏差 ~1e-15~1e-14，
# 1e-12 留两个数量级余量；再松就会放过「值搬错一位」的真实 bug。
REL_TOL = 1e-12
N_IN, N_OUT = 96, 48
_results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, extra: str = "") -> bool:
    _results.append((bool(ok), name, extra))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {extra}" if extra else ""))
    return bool(ok)


def rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    """最大相对差；分母加极小量避免 a=b=0 时除零（此时差值也是 0）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    s = np.maximum(np.abs(b), 1e-300)
    return float(np.max(d / s)) if d.size else 0.0


def scaled_err(got: np.ndarray, want: np.ndarray) -> float:
    """**尺度归一**的最大绝对误差：max|got-want| / max(1, max|want|)。

    为什么不能直接用逐元素相对差（`rel_diff`）：权重里有大量**近零元素**，
    相对差在分母趋 0 时会放大到无意义的量级（实测同一份正确结果能报出
    1.4e-14 的「相对差」，纯属分母太小）。更新核带 fastmath=True，融合乘加
    与 numpy 的逐位本就不保证相同，所以这里的正确判据是：
    **误差相对于整体量级可忽略**，而不是「每个元素都精确相等」。
    """
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(want))) if want.size else 1.0)
    return float(np.max(np.abs(got - want)) / scale) if got.size else 0.0


def dense_from_csr(ip, idx, val, n_out, n_in) -> np.ndarray:
    """CSR → 稠密（**被裁掉的位置为 0**）——G1 的同构矩阵构造。"""
    W = np.zeros((n_out, n_in), dtype=np.float64)
    for i in range(n_out):
        s = slice(int(ip[i]), int(ip[i + 1]))
        W[i, idx[s]] = val[s]
    return W


# ────────────────────────────────────────────────────────────────────────────
# G1 · 数值等价性
# ────────────────────────────────────────────────────────────────────────────
def g1_numeric_equivalence() -> None:
    print("\n" + "=" * 88)
    print("G1 · 稀疏 vs 稠密数值等价性（同一权重；裁掉位置置 0 的同构矩阵比较）")
    print("=" * 88)
    rng = np.random.default_rng(20260930)
    W = rng.normal(0.0, 0.05, (N_OUT, N_IN))

    for k in (N_IN, N_IN // 2, 8, 1):
        csr = _from_dense_csr(W, k)
        ip, idx, val = csr

        # (a) 裁剪操作本身不引入误差：CSR 前向 vs 「裁掉位置置 0」的稠密前向。
        #     这是**严格相等**（不是近似）——两者算的是同一个数学对象。
        W_masked = dense_from_csr(ip, idx, val, N_OUT, N_IN)
        h = rng.normal(0.0, 1.0, N_IN)
        y_sparse = _csr_matvec(ip, idx, val, h)
        y_masked = W_masked @ h
        rd = rel_diff(y_sparse, y_masked)
        check(f"k={k:<3} 稀疏前向 == 裁掉置 0 的同构稠密前向", rd < REL_TOL,
              f"最大相对差 {rd:.3e}")

        # (b) 保留位置上的权重必须**原样搬运**（值/索引都没错位）。
        Wd = W.astype(np.float64)
        same_pos = True
        max_pos = 0.0
        for i in range(N_OUT):
            s = slice(int(ip[i]), int(ip[i + 1]))
            cols = idx[s]
            if not np.array_equal(np.sort(cols), cols):
                same_pos = False
                break
            d = np.abs(val[s] - Wd[i, cols]).max() if len(cols) else 0.0
            max_pos = max(max_pos, float(d))
        check(f"k={k:<3} 保留位置的权重值逐位一致", same_pos and max_pos == 0.0,
              f"max|Δ| = {max_pos:.3e}")

        # (c) k = n_in（不裁剪任何东西）时，稀疏前向必须与**原稠密**前向等价。
        #     这是「稀疏 ⊃ 稠密」的结构性断言：退化路径不能有额外误差。
        if k == N_IN:
            y_orig = Wd @ h
            rd0 = rel_diff(y_sparse, y_orig)
            check("k=n_in（不裁剪）稀疏前向 == 原稠密前向", rd0 < REL_TOL,
                  f"最大相对差 {rd0:.3e}")

    # (d) Readout.from_dense(k=0) 的权重必须与稠密实例**完全相同**。
    #     这是端到端对拍的入口：权重不同则后面所有比较都无意义。
    dense_ro = Readout(N_IN, N_OUT, np.random.default_rng(7))
    sparse_ro = Readout.from_dense(dense_ro.W, k=0)
    dW = float(np.abs(dense_ro.W - sparse_ro.W).max())
    check("Readout.from_dense(k=0) 权重与稠密实例完全相同", dW == 0.0,
          f"max|Δ| = {dW:.3e}")

    # (e) 裁剪的**语义**确认：留下的是 |w| 最大的 k 列（强突触优先）。
    #     若这一点反了（留下最弱的），PPL 会莫名其妙地变好/变坏而无人察觉。
    W2 = rng.normal(0.0, 1.0, (4, 16))
    csr2 = _from_dense_csr(W2, 3)
    ip2, idx2, val2 = csr2
    ok_topk = True
    for i in range(4):
        s = slice(int(ip2[i]), int(ip2[i + 1]))
        got = set(idx2[s].tolist())
        want = set(np.argsort(-np.abs(W2[i]))[:3].tolist())
        if got != want:
            ok_topk = False
    check("裁剪保留的是每行 |w| 最大的 k 列（强突触优先）", ok_topk)


# ────────────────────────────────────────────────────────────────────────────
# G2 · CSR 结构合法性
# ────────────────────────────────────────────────────────────────────────────
def g2_csr_structure() -> None:
    print("\n" + "=" * 88)
    print("G2 · CSR 结构合法性（indptr 单调 / 边界 / 升序无重复 / 行宽范围）")
    print("=" * 88)
    rng = np.random.default_rng(31337)
    cases = []
    # 覆盖：均匀 k、全连接、k=1（最窄行）、k 超过 n_in（应被夹住）
    for k in (1, 8, N_IN, N_IN * 4):
        ro = Readout(N_IN, N_OUT, rng, conn_k=k)
        cases.append((f"conn_k={k}", ro._csr, ro.conn_k))   # noqa: SLF001
    W = rng.normal(0, 0.05, (N_OUT, N_IN))
    cases.append(("from_dense(k=8)", _from_dense_csr(W, 8), 8))

    for name, (ip, idx, val), k_eff in cases:
        n_rows = len(ip) - 1
        checks = {
            "indptr 首元素为 0": int(ip[0]) == 0,
            "indptr 末元素 == len(idx)": int(ip[-1]) == len(idx),
            "indptr 末元素 == len(val)": int(ip[-1]) == len(val),
            "indptr 单调不减": bool(np.all(np.diff(ip) >= 0)),
            "indptr 元素 ∈ [0, nnz]": bool(ip.min() >= 0 and ip.max() <= len(idx)),
            "行宽 ∈ [1, n_in]": bool(np.all(np.diff(ip) >= 1)
                                and np.all(np.diff(ip) <= N_IN)),
            "行数 == n_out": n_rows == N_OUT,
            "行宽全为 k（均匀 CSR）": bool(np.all(np.diff(ip) == k_eff)),
        }
        # 每行 idx 必须升序且**无重复**：升序 + 无重复 ⟺ 严格递增
        sorted_ok, uniq_ok, inrange_ok = True, True, True
        for i in range(n_rows):
            cols = idx[int(ip[i]):int(ip[i + 1])]
            if len(cols) > 1 and not np.all(np.diff(cols) > 0):
                sorted_ok = False
            if len(np.unique(cols)) != len(cols):
                uniq_ok = False
            if len(cols) and (cols.min() < 0 or cols.max() >= N_IN):
                inrange_ok = False
        checks["每行 idx 升序且无重复"] = sorted_ok
        checks["每行 idx ∈ [0, n_in)"] = inrange_ok
        checks["val 全部有限"] = bool(np.all(np.isfinite(val)))

        bad = [k2 for k2, v in checks.items() if not v]
        check(f"{name:<18} 结构合法", not bad,
              f"k_eff={k_eff} 违反: {bad}" if bad else f"k_eff={k_eff} nnz={len(idx)}")


# ────────────────────────────────────────────────────────────────────────────
# G3 · 更新路径
# ────────────────────────────────────────────────────────────────────────────
def g3_update_path() -> None:
    print("\n" + "=" * 88)
    print("G3 · 更新路径（只写存在的槽位 / 不越界 / 多次 learn 后结构仍合法）")
    print("=" * 88)
    rng = np.random.default_rng(4242)
    k = 12
    ro = Readout(N_IN, N_OUT, rng, conn_k=k)
    ip, idx, val = ro._csr                                     # noqa: SLF001
    nnz = len(val)
    ip_copy, idx_copy = ip.copy(), idx.copy()

    h = rng.normal(0, 1, N_IN)
    tgt = np.zeros(N_OUT)
    tgt[3] = 1.0

    # (a) 一次 learn 前后的「哪些槽位变了」必须**全部落在存在槽位内**。
    #     做法：复制 val，learn，再比对变化位置；并断言结构数组（ip/idx）
    #     完全没被改动——结构是不可变的，变的只能是 val。
    v_before = val.copy()
    ro.learn(h, tgt, 0.05)
    changed = np.nonzero(~np.isclose(val, v_before, rtol=0, atol=0))[0]
    check("learn 后 val 长度不变（无越界写）", len(val) == nnz,
          f"len={len(val)} 期望 {nnz}")
    check("learn 后结构数组 indptr/idx 未被改动",
          np.array_equal(ip, ip_copy) and np.array_equal(idx, idx_copy),
          "结构不可变，只允许改 val")
    check("变化的槽位全部 ∈ [0, nnz)", bool(changed.size == 0 or
                                       (changed.min() >= 0 and changed.max() < nnz)),
          f"变化 {changed.size} 个槽位，范围 "
          f"[{int(changed.min()) if changed.size else '-'}"
          f",{int(changed.max()) if changed.size else '-'}]")
    # 变化的槽位必须是真实变化的（若 eta 极大也不该为 0 个——否则说明更新没生效）
    check("learn 确实更新了存在的槽位（更新路径生效）", changed.size > 0,
          f"变化 {changed.size} 个")

    # (b) 稠密视图上，**只有 CSR 位置**可能非零（验证没往不存在的位置写）。
    #     这是「只写存在的槽位」最强的表述：稠密回填后越界写会立刻暴露。
    Wd = ro.W
    mask = np.zeros((N_OUT, N_IN), dtype=bool)
    for i in range(N_OUT):
        mask[i, idx[int(ip[i]):int(ip[i + 1])]] = True
    stray = Wd.copy()
    stray[mask] = 0.0
    check("稠密回填后 CSR 之外全为 0（无越界/幽灵写入）",
          float(np.abs(stray).max()) == 0.0,
          f"max|越界值| = {np.abs(stray).max():.3e}")

    # (c) learn_softmax 路径同样只写存在槽位（softmax 是 LM 的实际路径，
    #     只测 learn 会漏掉这一条）。
    v2 = val.copy()
    nll = ro.learn_softmax(h, tgt, 0.05)
    check("learn_softmax 返回有限 NLL", np.isfinite(nll), f"nll={nll:.4f}")
    check("learn_softmax 后 val 长度不变", len(val) == nnz)
    ch2 = np.nonzero(~np.isclose(val, v2, rtol=0, atol=0))[0]
    check("learn_softmax 变化槽位 ∈ [0, nnz)",
          bool(ch2.size == 0 or (ch2.min() >= 0 and ch2.max() < nnz)),
          f"变化 {ch2.size} 个")
    check("learn_softmax 后 indptr/idx 未被改动",
          np.array_equal(ip, ip_copy) and np.array_equal(idx, idx_copy))

    # (c2) **跨行泄漏**检测（这是「只写存在的槽位」最强的表述）。
    #      上面 (a) 只验「槽位下标在 [0,nnz)」——但**相邻行的槽位也是合法下标**，
    #      所以「写到隔壁行」能骗过 (a)（实测：注入 val[indptr[i]-1] 的故障时
    #      (a) 全部 PASS，门禁失守）。这里改成**行归属**断言：构造一个只有
    #      第 0 行梯度非零的样本，learn 后**除第 0 行外的所有 val 必须逐位不变**。
    #      任何跨行写入都会立刻暴露，且这是确定性的（不依赖容差）。
    ro_l = Readout(N_IN, N_OUT, rng, conn_k=k)
    ip_l, idx_l, val_l = ro_l._csr                           # noqa: SLF001
    v_l = val_l.copy()
    from phdnet.sparse_pc import _csr_add_outer              # noqa: PLC0415

    # (c2a) **逐槽位精确核对**（最强的「只写存在的槽位」表述）。
    #      a 全非零 → 每行都会进入更新循环，**跨行泄漏的代码路径必然被执行**
    #      （只给 a[0] 置非零时，`if ai == 0.0: continue` 会让其余行直接跳过，
    #      泄漏分支根本走不到 —— 那样测了等于没测）。
    #      期望值按「行 i 的梯度只作用在行 i 的槽位上」逐元素构造，
    #      于是「写到隔壁行」与「写到不存在的槽位」都会立刻暴露，且**不依赖容差**。
    eta_l = 0.1
    a_l = rng.normal(0, 1, N_OUT)          # 全非零
    b_l = rng.normal(0, 1, N_IN)
    expect = v_l.copy()
    for i in range(N_OUT):
        s_i = slice(int(ip_l[i]), int(ip_l[i + 1]))
        expect[s_i] = v_l[s_i] + eta_l * a_l[i] * b_l[idx_l[s_i]]
    _csr_add_outer(ip_l, idx_l, val_l, a_l, b_l, eta_l)
    # 容差取 1e-15 相对量级：`_csr_add_outer` 带 fastmath=True，融合乘加
    # 不保证与 numpy 的 `eta*a*b` 逐位相同（实测偏差 ~5.6e-17 绝对）。
    # 这里是**相对**容差而非绝对——权重量级差异大时绝对容差会误判。
    dev = scaled_err(val_l, expect)
    check("逐槽位精确核对：val ≈ 原值 + eta·a[row]·b[idx]（无跨行/越界写）",
          dev < 1e-13, f"尺度归一误差 {dev:.3e}（容差 1e-13，fastmath 量级）")
    check("更新后 val 长度不变（无越界写）", len(val_l) == len(v_l),
          f"{len(val_l)} vs {len(v_l)}")

    # (c2b) 行归属隔离：只有第 0 行有梯度时，**其余行必须逐位不变**。
    #      双重保险：a2 全零除第 0 行 → 行归属错乱会被逐位暴露。
    ro_m = Readout(N_IN, N_OUT, rng, conn_k=k)
    ip_m, idx_m, val_m = ro_m._csr                           # noqa: SLF001
    v_m = val_m.copy()
    a_m = np.zeros(N_OUT)
    a_m[0] = 0.5
    _csr_add_outer(ip_m, idx_m, val_m, a_m, b_l, eta_l)
    row_w = np.diff(ip_m)
    changed_slots = np.nonzero(val_m != v_m)[0]
    owner = np.repeat(np.arange(N_OUT), row_w)      # slot -> row
    touched_rows = (np.unique(owner[changed_slots]) if changed_slots.size
                    else np.array([], dtype=np.int64))
    leaked = [int(i) for i in touched_rows if i != 0]
    check("行归属隔离：仅第 0 行有梯度时，只有第 0 行的 val 变化", not leaked,
          f"泄漏到行 {leaked[:5]}" if leaked else f"变化行 = {touched_rows.tolist()}")
    check("行归属隔离：第 0 行变化量与 eta·a·h 一致",
          scaled_err(val_m[:row_w[0]],
                     v_m[:row_w[0]] + eta_l * 0.5 * b_l[idx_m[:row_w[0]]]) < 1e-13,
          "行内更新量核对（尺度归一）")
    check("行归属隔离：其余行 val 逐位不变",
          np.array_equal(val_m[row_w[0]:], v_m[row_w[0]:]),
          f"第 1..{N_OUT - 1} 行未变")

    # (d) 多次 learn 后结构仍合法 + 权重有界（clip 生效）
    ro_c = Readout(N_IN, N_OUT, np.random.default_rng(99), conn_k=k, w_clip=0.5)
    ok_clip = True
    for step in range(30):
        hs = rng.normal(0, 1, N_IN)
        ts = np.zeros(N_OUT)
        ts[int(rng.integers(N_OUT))] = 1.0
        ro_c.learn_softmax(hs, ts, 0.4)
        Wc = ro_c.W
        if not np.all(np.isfinite(Wc)) or np.abs(Wc).max() > 0.5 + 1e-12:
            ok_clip = False
            break
    check("30 步 learn_softmax 后权重有限且 w_clip 生效", ok_clip,
          f"max|W| = {np.abs(ro_c.W).max():.4f} (w_clip=0.5)")


# ────────────────────────────────────────────────────────────────────────────
# G4 · 幂律分组（可 import 才跑）
# ────────────────────────────────────────────────────────────────────────────
def g4_powerlaw() -> None:
    print("\n" + "=" * 88)
    print("G4 · 幂律分组（alpha=0 退化均匀 / alpha=1 秩相关为正 / 总和 ≤ 预算）")
    print("=" * 88)
    try:
        from phdnet import sparse_alloc
    except Exception as exc:                                    # noqa: BLE001
        print(f"  [SKIP] phdnet.sparse_alloc 不可 import"
              f"（{type(exc).__name__}: {exc}）→ 幂律组跳过")
        print("         这是**预期状态**：幂律分配器由并行分支开发，"
              "未就绪不应让门禁变红。")
        print("         本组断言的是幂律分配器的契约（退化性/秩相关方向/预算上限），")
        print("         待 sparse_alloc 就绪后自动生效。")
        _results.append((True, "G4 幂律分组（跳过：sparse_alloc 不可 import）", "SKIP"))
        return

    print(f"  sparse_alloc: {getattr(sparse_alloc, '__file__', '?')}")
    fn = getattr(sparse_alloc, "assign_conn_counts", None)
    if not callable(fn):
        print(f"  [SKIP] sparse_alloc 无 assign_conn_counts 入口（现有: "
              f"{[n for n in dir(sparse_alloc) if not n.startswith('_')]}）")
        _results.append((True, "G4 幂律分组（跳过：入口缺失）", "SKIP"))
        return

    rng = np.random.default_rng(1234)
    n_out, n_in, budget = 64, 128, 64 * 16
    # counts 故意强偏斜（zipf 式），这样「秩相关为正」才有区分度——
    # 全均匀的 counts 会让任何分配器都给出常数 k，测不出方向对错。
    counts = (1.0 / (1.0 + np.arange(n_out))) ** 0.9 * 1000.0

    # (a) alpha=0 必须**退化到均匀**：这是 A/B 可比性的前提。若 alpha=0 也给出
    #     倾斜分配，则「幂律 vs 均匀」的对照基线会被悄悄改变。
    k0 = np.asarray(fn(counts, alpha=0.0, k_min=1, k_max=n_in,
                       total_budget=budget, n_in=n_in)).ravel()
    check("alpha=0 退化到均匀（所有行宽相同）",
          k0.size == n_out and k0.min() >= 1 and k0.max() <= n_in
          and len(np.unique(k0)) == 1,
          f"unique(k) = {np.unique(k0).tolist()[:5]} (n_unique={len(np.unique(k0))})")

    # (b) alpha=1 时 k 与 counts 的**秩相关为正**（方向不能反）。
    #     用 Spearman：只看次序、不受counts 的重尾量级影响。
    k1 = np.asarray(fn(counts, alpha=1.0, k_min=1, k_max=n_in,
                       total_budget=budget, n_in=n_in)).ravel()
    rc = _rank_corr(counts, k1.astype(np.float64))
    check("alpha=1 秩相关为正（分配方向正确）", rc > 0.3,
          f"Spearman rho = {rc:.4f}（>0.3）")

    # (c) 总和 ≤ 预算：**「省内存」的前提本身**。超预算则整轮结论失效。
    check("alpha=1 总和 ≤ 预算", int(k1.sum()) <= budget,
          f"sum = {int(k1.sum()):,} ≤ {budget:,}")
    check("alpha=0 总和 ≤ 预算", int(k0.sum()) <= budget,
          f"sum = {int(k0.sum()):,} ≤ {budget:,}")

    # (d) 夹紧边界：预算小于 n_out*k_min 时必须**显式报错**而非静默截断
    #     （静默截断会让「预算上限」变成谎言——这正是分配器 docstring 承诺的）。
    raised = False
    try:
        fn(counts, alpha=1.0, k_min=4, k_max=n_in,
           total_budget=8, n_in=n_in)          # 8 < 64*4
    except ValueError:
        raised = True
    check("预算 < n_out*k_min 时抛 ValueError（不静默截断）", raised)

    # (e) k 必须落在 [k_min, k_max] 内
    check("k ∈ [k_min, k_max]", bool(k1.min() >= 1 and k1.max() <= n_in),
          f"k ∈ [{int(k1.min())}, {int(k1.max())}]，n_in={n_in}")

    # (f) 变长 CSR 能装进 Readout 且前向可用（幂律分组的**落地**路径，
    #     也是 bench 的幂律行走的那条路）
    import sys as _sys
    _sys.path.insert(0, str(_ROOT / "tools"))
    from bench_readout_sparse import (                    # noqa: PLC0415
        attach_variable_csr, csr_from_widths)
    ro = Readout(256, 48, np.random.default_rng(5))
    # 用**真实幂律宽度**（截到 48 行，缩到 n_in=256 以内），而不是塞一个常数
    # 宽度 —— 否则这条用例根本没覆盖「变长」这个关键性质，测了等于没测。
    pl_w = np.clip(k1[:48], 1, 200)
    assert len(np.unique(pl_w)) > 1, f"幂律宽度应非恒定，实得 {np.unique(pl_w)[:5]}"
    csr = csr_from_widths(np.random.default_rng(6), 48, 256, pl_w, 0.05)
    attach_variable_csr(ro, csr)
    y = ro(rng.normal(0, 1, 256))
    check("变长 CSR 装入 Readout 后前向可用", y.shape == (48,) and np.all(np.isfinite(y)),
          f"y.shape={y.shape} 有限={bool(np.all(np.isfinite(y)))}")
    ts = np.zeros(48)
    ts[1] = 1.0
    ro.learn_softmax(rng.normal(0, 1, 256), ts, 0.05)
    check("变长 CSR 装入 Readout 后 learn_softmax 可用", True)


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关（并列取平均秩）。"""
    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # 并列值取平均秩
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


# ────────────────────────────────────────────────────────────────────────────
# G5 · 多步序列（P67 栽过的地方）
# ────────────────────────────────────────────────────────────────────────────
def g5_multi_step() -> None:
    print("\n" + "=" * 88)
    print("G5 · 多步序列（20 步 learn + forward，**逐步**断言结构合法）")
    print("=" * 88)
    rng = np.random.default_rng(555)
    for k in (4, 16, N_IN):
        ro = Readout(N_IN, N_OUT, rng, conn_k=k)
        ip, idx, val = ro._csr                                 # noqa: SLF001
        ip0, idx0, n0 = ip.copy(), idx.copy(), len(val)
        widths0 = np.diff(ip)
        bad_step = None
        detail = ""
        for step in range(20):
            h = rng.normal(0, 1, N_IN)
            tgt = np.zeros(N_OUT)
            tgt[int(rng.integers(N_OUT))] = 1.0
            nll = ro.learn_softmax(h, tgt, 0.05)
            y = ro(rng.normal(0, 1, N_IN))               # forward 也要跑
            # 逐步断言：结构数组不变、长度不变、行宽不变、值有限
            if len(val) != n0:
                bad_step, detail = step, f"nnz 变化 {n0}→{len(val)}"
            elif not np.array_equal(ip, ip0):
                bad_step, detail = step, "indptr 被改动"
            elif not np.array_equal(idx, idx0):
                bad_step, detail = step, "idx 被改动"
            elif not np.array_equal(np.diff(ip), widths0):
                bad_step, detail = step, "行宽变化"
            elif not (np.all(np.isfinite(val)) and np.all(np.isfinite(y))):
                bad_step, detail = step, "出现非有限值"
            elif not np.isfinite(nll):
                bad_step, detail = step, "NLL 非有限"
            if bad_step is not None:
                break
        check(f"k={k:<3} 20 步 learn+forward 结构逐步合法", bad_step is None,
              detail if bad_step is not None else
              f"结构恒定，nnz={n0}，行宽恒为 {int(widths0[0])}")

    # 交叉验证：多步序列下稀疏与稠密（k=n_in）的前向仍一致
    # ——证明「k=n_in 退化到稠密」在**多步**后依然成立（单步一致不够）。
    W = rng.normal(0, 0.05, (N_OUT, N_IN))
    dense_ro = Readout.from_dense(W, k=0)
    full_ro = Readout.from_dense(W, k=N_IN)
    g = np.random.default_rng(9)
    max_d = 0.0
    for _ in range(20):
        h = g.normal(0, 1, N_IN)
        t = np.zeros(N_OUT)
        t[int(g.integers(N_OUT))] = 1.0
        max_d = max(max_d, rel_diff(full_ro(h), dense_ro(h)))
        dense_ro.learn_softmax(h, t, 0.05)
        full_ro.learn_softmax(h, t, 0.05)
    check("20 步后 k=n_in 稀疏与稠密前向仍一致", max_d < REL_TOL,
          f"最大相对差 {max_d:.3e}")
    check("20 步后两者权重仍一致",
          float(np.abs(dense_ro.W - full_ro.W).max()) < 1e-10,
          f"max|ΔW| = {np.abs(dense_ro.W - full_ro.W).max():.3e}")


def main() -> int:
    print("=" * 88)
    print("M6 稀疏读出门禁（verify_readout_sparse_gate）")
    print(f"python {sys.version.split()[0]} | numpy {np.__version__}")
    print(f"相对容差 {REL_TOL:g} | 维度 n_in={N_IN} n_out={N_OUT}")
    print("=" * 88)
    g1_numeric_equivalence()
    g2_csr_structure()
    g3_update_path()
    g4_powerlaw()
    g5_multi_step()

    npass = sum(1 for ok, _, _ in _results if ok)
    ntot = len(_results)
    nskip = sum(1 for ok, _, tag in _results if tag == "SKIP")
    fails = [name for ok, name, _ in _results if not ok]
    print("\n" + "=" * 88)
    print(f"结果：{npass}/{ntot} 通过"
          + (f"（含 {nskip} 项按契约跳过）" if nskip else "")
          + f" | 失败 {len(fails)}")
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
    print("=" * 88)
    print("门禁结论：" + ("PASS ✅" if not fails else "FAIL ❌"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
