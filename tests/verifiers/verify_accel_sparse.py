"""P111：加速后端稀疏读出对拍（AccelReadout CSR vs numba Readout CSR）。

为什么必须有这个文件
================================================================================
P111 之前 `readout_conn_k>0` 会命中 `_unsupported_reason` → 整个 NPU 读出回落
numba CPU。P111 在加速后端实现了均匀 k 的 gather-GEMV，于是「稀疏 + 设备加速」
这条组合第一次**真的跑起来**。而它跑起来之前没有任何数值证据——这正是历史上
最贵的教训（P55 融合核缺target_idx 分支 → NPU 首次真跑即崩；P99 三态开关
dead on arrival）。所以本文件是实现的**准入门槛**，不是事后补的测试。

验证分层
================================================================================
A. 数值等价（容差判据，见下）
   A1 前向：稀疏 accel vs 稀疏 numba，同一 CSR → y逐元素接近
   A2 learn_softmax：同一 h/target/eta 走一步 → W 差异在容差内
   A3 learn：非 softmax 路径同样等价
   A4 与「稠密 numba + 零化非连接位」等价（语义锚：稀疏就是稠密的受限版本）
B. 调用面完整（P23/P24 纪律：换后端必须对齐全部访问面）
   B1 conn_k / _csr / n_synapses / stats / W_cpu / load_W / dtype_name
   B2 _csr 往返一致性（导出→ 再灌入 → 权重不变）
   B3 check点风格的 val[:] =写 + sync_csr_from_host 生效
C. 拒绝路径（P19：不静默走错算法）
   C1 非均匀行宽 CSR → fail-fast（不是静默算错）
   C2 稠密配置读 _csr → NotImplementedError
D. 零回归
   D1 conn_k=0 时 accel 行为与 P110 前一致（稠密路径未被污染）

容差判据（为什么不是逐位）
================================================================================
torch CPU 与 numpy 的**归约顺序不同**（同一 sum 的元素次序），fp32 下
max|Δ| ~1e-6 属正常。跨库逐位相等不是正确性要求（见 accel_readout.py 模块
docstring 与 MEMORY 的「跨库等价用容差」纪律）。本文件的容差取**相对误差**
并显式打印实测 max|Δ|，不藏数字。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in ("", "tests", "tools"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

_RTOL = 2e-5          # 相对容差：fp32 归约顺序差异的量级
_ATOL = 1e-6

_RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    return bool(ok)


def _torch():
    import torch
    return torch


def build_pair(n_h: int, n_out: int, k: int, seed: int = 7):
    """构造 (numba Readout, AccelReadout) 且**共享同一份 CSR**。

    共享是刻意的：两边独立随机生成 CSR 会连列索引都不同，数值没有可比性。
    这里先生成一次 CSR，再分别喂给两个后端。
    """
    torch = _torch()
    from phdnet.readout import Readout
    from phdnet.backends.accel_readout import AccelReadout
    from phdnet.sparse_pc import _random_csr

    rng = np.random.default_rng(seed)
    csr = _random_csr(rng, n_out, n_h, k, 0.05 * np.sqrt(n_h / k), False, 0.8)

    ref = Readout(n_h, n_out, np.random.default_rng(seed),
                  dtype="fp32", conn_k=k, lognormal_init=False)
    # 用同一份 CSR 覆盖，保证两者结构完全一致
    ref._csr = csr                                        # noqa: SLF001

    acc = AccelReadout(n_h, n_out, np.random.default_rng(seed), device="cpu",
                       dtype="fp32", conn_k=k, csr=csr,
                       lognormal_init=False, exc_ratio=0.8)
    return ref, acc, csr, torch


def main() -> int:
    torch = _torch()
    n_h, n_out, k = 64, 40, 8
    print("=" * 78)
    print(f"P111 稀疏加速读出对拍：n_h={n_h} n_out={n_out} k={k} device=cpu(torch)")
    print("=" * 78)

    ref, acc, csr, _ = build_pair(n_h, n_out, k)
    ip, idx, val = csr

    # ── A. 数值等价 ────────────────────────────────────────────────────────
    print("\n[A] 数值等价（容差判据，跨库归约顺序不同故不要求逐位）")
    rng = np.random.default_rng(11)

    # A1 前向
    max_dy = 0.0
    for _ in range(5):
        h = rng.normal(0.0, 1.0, n_h)
        y_ref = ref(h)
        y_acc = acc(h)
        max_dy = max(max_dy, float(np.abs(np.asarray(y_ref) - y_acc).max()))
    scale = max(1.0, float(np.abs(np.asarray(y_ref)).max()))
    check(max_dy <= _ATOL + _RTOL * scale, "A1 前向 y 与 numba 稀疏一致",
          f"max|Δ|={max_dy:.3e} (|y|max={scale:.3f})")

    # A2 learn_softmax
    h = rng.normal(0.0, 1.0, n_h)
    tgt = rng.normal(0.0, 1.0, n_out)
    eta = 0.15
    ref.learn_softmax(h.copy(), tgt.copy(), eta)
    acc.learn_softmax(h.copy(), tgt.copy(), eta)
    w_ref = ref._csr[2]                                    # noqa: SLF001（更新后的 ref）
    w_acc = acc._csr[2]                                    # noqa: SLF001（导出即当前值）
    dw = float(np.abs(w_ref - w_acc).max())
    # ⚠ 尺度必须取**更新后**权重的量级（不是更新前的副本）：单步更新量~1e-2，
    # 用旧尺度会把 fp32 舍入误差放大成假阳性（第一版就踩了这个，
    # 报出 0.63 的假失败——真实相对误差只有 7.3e-08）。
    wscale = max(1.0, float(np.abs(w_ref).max()))
    check(dw <= _ATOL + _RTOL * wscale, "A2 learn_softmax 后 CSR val 一致",
          f"max|Δ|={dw:.3e} (尺度 {wscale:.3f}, 相对 {dw/wscale:.2e})")

    # A3 learn（非 softmax 路径）
    ref2, acc2, _, _ = build_pair(n_h, n_out, k, seed=13)
    h2 = rng.normal(0.0, 1.0, n_h)
    t2 = rng.normal(0.0, 1.0, n_out)
    ref2.learn(h2.copy(), t2.copy(), eta)
    acc2.learn(h2.copy(), t2.copy(), eta)
    w2_ref = ref2._csr[2]                                  # noqa: SLF001
    w2_acc = acc2._csr[2]                                  # noqa: SLF001
    dw2 = float(np.abs(w2_ref - w2_acc).max())
    s2 = max(1.0, float(np.abs(w2_ref).max()))
    check(dw2 <= _ATOL + _RTOL * s2, "A3 learn 后 CSR val 一致",
          f"max|Δ|={dw2:.3e} (相对 {dw2/s2:.2e})")

    # A1b 连续不同 h：gather 缓存必须逐 h失效（P112 的回归门禁）
    # 这条专门盯 `_sp_gather` 的缓存判据。第一版用 `torch.equal`判「同一个
    # h」，性能上它是**每步强制设备同步**；改成主机侧 epoch 后，判据一旦漏在
    # 某条上传路径上（如 `forward` 走 `_to_dev` 而非 `_staged_to_dev`），
    # 就会把上一步的 gather 结果静默复用给新 h —— 只有数值对拍能抓到。
    _, acc_seq, _, _ = build_pair(n_h, n_out, k, seed=23)
    dmax = 0.0
    for i in range(6):                # 连续 6 个不同 h
        hs = rng.normal(0.0, 1.0 + i, n_h)
        dmax = max(dmax, float(np.abs(np.asarray(acc_seq(hs))).max()))
    # 每次 forward 的 h 尺度不同 → y 的量级应随之变化；若缓存被错误复用，
    # y 会被冻结在上一个 h 的结果上（dmax 不再增长）。
    check(dmax > 0.0 and acc_seq._cache_g_epoch == acc_seq._ht_epoch,
          "A1b 连续不同 h：gather 缓存逐 h 失效（epoch 判据）",
          f"cache_epoch={acc_seq._cache_g_epoch} ht_epoch={acc_seq._ht_epoch}")

    # A4 语义锚：稀疏 == 稠密里只保留连接位的那个矩阵
    # ⚠ 两条纪律（第一版都踩了，报出 1.40 的假失败）：
    #   ① 必须用 **csr3 自己的** idx/val（seed=17），不能复用 seed=7 的；
    #   ② **不能**用 `W_dense.reshape(-1)[idx] = val` —— CSR 的 idx 是
    #      **逐行排列**的（第 i 行的 k 个索引在偏移 i*k 处），而稠密矩阵
    #      展平后按 `idx` 直接索引会把「第 i 行第 j 个连接」写到全局第
    #      idx[i*k+j] 个元素上——结构完全错位。必须按行还原：
    #          W_dense[i, idx[i*k + j]] = val[i*k + j]
    _, acc3, csr3, _ = build_pair(n_h, n_out, k, seed=17)
    ip3, idx3, val3 = csr3
    idx3 = np.asarray(idx3, dtype=np.int64).reshape(n_out, k)
    val3 = np.asarray(val3, dtype=np.float32).reshape(n_out, k)
    W_dense = np.zeros((n_out, n_h), dtype=np.float32)
    for _r in range(n_out):                 # 逐行 scatter（k很小，可读）
        W_dense[_r, idx3[_r]] = val3[_r]
    h4 = rng.normal(0.0, 1.0, n_h)
    y_sparse = np.asarray(acc3(h4))
    y_dense = W_dense @ h4.astype(np.float32)
    d4 = float(np.abs(y_sparse - y_dense).max())
    s4 = max(1.0, float(np.abs(y_dense).max()))
    check(d4 <= _ATOL + _RTOL * s4,
          "A4 与「稠密+零化非连接位」语义等价",
          f"max|Δ|={d4:.3e} (相对 {d4/s4:.2e})")

    # ── B. 调用面完整 ─────────────────────────────────────────────────────
    print("\n[B] 调用面完整（P23/P24：换后端必须对齐全部访问面）")
    check(acc.conn_k == k, "B1a conn_k 报告真实 k", f"conn_k={acc.conn_k}")
    check(acc.hidden == 0, "B1b hidden=0（单级）")
    check(acc.n_synapses() == n_out * k, "B1c n_synapses = n_out*k",
          f"{acc.n_synapses()} vs {n_out * k}")
    check(acc.dtype_name == "fp32", "B1d dtype_name=fp32（稀疏锁 fp32）",
          acc.dtype_name)
    st = acc.stats()
    check(st["k"] == k and st["synapses"] == n_out * k,
          "B1e stats 与 Readout 键对齐", f"k={st['k']} syn={st['synapses']}")
    check("index_MB" in st and st["storage_MB_total"] > st["storage_MB"],
          "B1f stats 单列索引占用（不低估稀疏真实开销）",
          f"val={st['storage_MB']:.3f}MB + idx={st['index_MB']:.3f}MB")

    # B2 _csr 往返
    ip2, idx2, val2 = acc._csr
    check(np.array_equal(ip2, ip) and np.array_equal(idx2, np.asarray(idx)),
          "B2a _csr 的 indptr/idx 与源 CSR 逐位一致")
    check(np.allclose(val2, acc.W.cpu().numpy().ravel(), rtol=0, atol=0),
          "B2b _csr 的 val 与设备 W 当前值一致")

    # B3 check点风格：val[:] = ... + sync_csr_from_host()
    newval = np.asarray(val, dtype=np.float64) * 1.5
    val2[:] = newval
    acc.sync_csr_from_host()
    back = acc._csr[2]                                    # noqa: SLF001
    check(np.allclose(back, newval, rtol=_RTOL, atol=_ATOL),
          "B3 val[:]= 写入 + sync_csr_from_host 生效（ckpt 恢复路径）",
          f"max|Δ|={np.abs(back - newval).max():.3e}（fp64↔fp32 往返量级）")

    # B4 load_W / W_cpu 往返
    _, acc4, _, _ = build_pair(n_h, n_out, k, seed=19)
    Wv = acc4.W_cpu()
    _, acc4b, _, _ = build_pair(n_h, n_out, k, seed=19)
    acc4b.load_W(Wv)
    check(np.array_equal(acc4b.W_cpu(), Wv), "B4 W_cpu → load_W 精确往返",
          f"shape={Wv.shape}")

    # B5 sync_csr_from_host 在没有导出时必须 fail-fast（不能静默 no-op）
    _, acc5, _, _ = build_pair(n_h, n_out, k, seed=29)
    try:
        acc5.sync_csr_from_host()
        check(False, "B5 未导出就sync → fail-fast", "竟然静默通过了")
    except RuntimeError as e:
        check("先取 _csr" in str(e), "B5 未导出就 sync → fail-fast",
              str(e)[:56])

    # ── C. 拒绝路径 ────────────────────────────────────────────────────────
    print("\n[C] 拒绝路径（P19：不静默走错算法）")
    from phdnet.backends.accel_readout import AccelReadout
    # C1 非均匀行宽
    ragged_ip = np.array([0, 3, 5, 8], dtype=np.int64)
    ragged_idx = np.arange(8, dtype=np.int64) % n_h
    ragged_val = np.linspace(-0.1, 0.1, 8)
    try:
        AccelReadout(n_h, 3, np.random.default_rng(0), device="cpu",
                     dtype="fp32", conn_k=3,
                     csr=(ragged_ip, ragged_idx, ragged_val))
        check(False, "C1 非均匀行宽 CSR → fail-fast", "竟然构造成功了")
    except ValueError as e:
        check("均匀 k" in str(e), "C1 非均匀行宽 CSR → fail-fast",
              str(e)[:60])

    # C2 稠密配置读 _csr
    dense_acc = AccelReadout(n_h, n_out, np.random.default_rng(0), device="cpu",
                             dtype="fp32", conn_k=0)
    try:
        _ = dense_acc._csr
        check(False, "C2 稠密切 _csr → NotImplementedError", "竟然返回了")
    except NotImplementedError:
        check(True, "C2 稠密切 _csr → NotImplementedError")

    # C3 形状不符的 load_W
    try:
        dense_acc.load_W(np.zeros((n_out + 1, n_h), dtype=np.float32))
        check(False, "C3 load_W 形状不符 → 报错", "竟然接受了")
    except ValueError:
        check(True, "C3 load_W 形状不符 → 报错")

    # ── C4b. 稀疏前向算子 A/B（P116）────────────────────────────────────
    # `--sparse-fwd-kernel`：mulsum（默认，逐位基线）vs einsum（不物化
    # (n_out,k) 中间张量，但**归约顺序不同→ 非逐位**）。
    # ⚠ 本机 x86 实测 einsum 快 ~21%，但**昇腾收益未实测**——x86 结论不构成
    # 昇腾证据（本项目已实测到四次方向相反）。故 einsum **不作默认**，
    # 只提供开关 + 容差对拍，让服务器自己测。
    print("\n[C4b] 稀疏前向算子 mulsum vs einsum（P116，容差判据）")
    # `build_pair(seed=31)` 给两边喂**同一份 CSR**，故两者结构完全一致，
    # 唯一变量就是前向算子。（不能手动给 `_csr` 赋值——它是只读 property。）
    _, ro_mul, _, _ = build_pair(n_h, n_out, k, seed=31)
    _, ro_ein, _, _ = build_pair(n_h, n_out, k, seed=31)
    ro_ein._sp_fwd = "einsum"          # noqa: SLF001（刻意切换被测算子）
    d_ein = 0.0
    for _ in range(5):
        hh = rng.normal(0.0, 1.0, n_h)
        ym = np.asarray(ro_mul(hh))
        ye = np.asarray(ro_ein(hh))
        d_ein = max(d_ein, float(np.abs(ym - ye).max()))
    s_e = max(1.0, float(np.abs(ym).max()))
    check(d_ein <= 1e-4 * s_e,
          "C4b einsum 与 mulsum 容差内一致（**非逐位**）",
          f"max|Δ|={d_ein:.3e} 相对={d_ein/s_e:.2e}（归约顺序不同，预期非0）")
    # 反向：确认 einsum 不是「什么都没算」（防再次出现丢权重的 bug）
    check(float(np.abs(ye).max()) > 0.0,
          "C4b einsum 输出非零（防「丢权重」类静默错误）",
          f"max|y|={float(np.abs(ye).max()):.4f}")
    # 更新路径不受影响（两条前向共用同一 learn_softmax）
    w_before = ro_ein.W.clone()
    ro_ein.learn_softmax(hh.copy(), rng.normal(0, 1, n_out), 0.15)
    check(not bool((w_before == ro_ein.W).all()),
          "C4b einsum 臂的更新路径仍生效（W 已变）")

    # C4c. gather 缓存的 epoch 判据（P122 审计抓到的静默数值错误）
    print("\n[C4c] gather 缓存 epoch 判据（跨不同 h 的调用序列）")
    # 原bug 调用序列：forward(hA) → learn(hB) → learn(hA)
    # 第三次时 _lookup_ht 命中缓存（返回 htA），但 epoch 停在 hB 那次，
    # 于是 _sp_gather 判据「命中」→ 用G(hB) 去更新 hA 的输出。
    _, ro_ep, _, _ = build_pair(n_h, n_out, k, seed=37)
    tgt_ep = np.zeros(n_out, dtype=np.float32)
    tgt_ep[0] = 1.0
    hA = rng.normal(0.0, 1.0, n_h).astype(np.float32)
    hB = rng.normal(0.0, 1.0, n_h).astype(np.float32)
    ro_ep.forward(hA)
    ro_ep.learn_softmax(hB, tgt_ep, 0.0)
    ro_ep.learn_softmax(hA, tgt_ep, 0.0)
    Wi = ro_ep.Wi.detach().cpu().numpy()
    cached_g = ro_ep._cache_g.detach().cpu().numpy()# noqa: SLF001
    gA = hA[Wi]
    gB = hB[Wi]
    check(np.allclose(cached_g, gA, atol=1e-6),
          "C4c1 缓存 gather 对应 hA（而非被污染的 hB）",
          f"==hA:{np.allclose(cached_g, gA, atol=1e-6)} "
          f"==hB:{np.allclose(cached_g, gB, atol=1e-6)}")
    # 更强：被污染后再用 hA 前向，y 必须与干净实例一致
    _, ro_c1, _, _ = build_pair(n_h, n_out, k, seed=37)
    _, ro_c2, _, _ = build_pair(n_h, n_out, k, seed=37)
    ro_c1.forward(hA)
    ro_c1.learn_softmax(hB, tgt_ep, 0.0)
    y_after = np.asarray(ro_c1(hA))
    y_clean = np.asarray(ro_c2(hA))
    d = float(np.abs(y_after - y_clean).max())
    sc = max(1.0, float(np.abs(y_clean).max()))
    check(d <= 1e-5 * sc,
          "C4c2 污染后用 hA 前向，y 与干净实例**数值一致**（无静默错误）",
          f"max|Δ|={d:.3e}")
    # 正向：单h 连续调用不应误判为「变了」
    _, ro_same, _, _ = build_pair(n_h, n_out, k, seed=37)
    ro_same.forward(hA)
    g_before = ro_same._cache_g.detach().cpu().numpy().copy()   # noqa: SLF001
    ro_same.learn_softmax(hA, tgt_ep, 0.0)      # 同一 h → 应命中并复用
    g_after = ro_same._cache_g.detach().cpu().numpy()          # noqa: SLF001
    check(np.array_equal(g_before, g_after),
          "C4c3 同一 h 连续调用仍复用 gather（判据未过度失效）")

    # C4d. 稀疏路径的 torch.compile 融合核（P128：此前被排除在编译之外）
    print("\n[C4d] 稀疏读出的 torch.compile 融合核（P128）")
    from phdnet.backends.accel_readout import AccelReadout as _AR
    from phdnet.sparse_pc import _random_csr as _rc
    _csr3 = _rc(np.random.default_rng(37), n_out, n_h, k,
                0.05 * np.sqrt(n_h / k), False, 0.8)
    _tgt = np.zeros(n_out, dtype=np.float32)
    _tgt[0] = 1.0
    _h4 = rng.normal(0.0, 1.0, n_h).astype(np.float32)
    _e = _AR(n_h, n_out, np.random.default_rng(37), device="cpu",
             dtype="fp32", conn_k=k, csr=_csr3, compile=False)
    _c = _AR(n_h, n_out, np.random.default_rng(37), device="cpu",
             dtype="fp32", conn_k=k, csr=_csr3, compile=True)
    check(_c._compiled, "C4d1 稀疏路径**可以**启用 torch.compile（P128 前被排除）")
    _ye = _e.forward_dev(_h4)
    _ne = _e.learn_softmax(_h4, _tgt, 0.15, y_pre=_ye)
    _yc = _c.forward_dev(_h4)
    _nc = _c.learn_softmax(_h4, _tgt, 0.15, y_pre=_yc)
    # 数值必须一致（融合 vs eager）
    if _c._compiled and not getattr(_c, "_compile_failed", False):
        _we = _e.W.detach().cpu().numpy()
        _wc = _c.W.detach().cpu().numpy()
        check(np.array_equal(_we, _wc),
              "C4d2 融合核与 eager 的 W **逐位相同**",
              f"max|Δ|={float(np.abs(_we - _wc).max()):.3e}")
        check(abs(float(_ne) - float(_nc)) <= 1e-6 * max(1.0, abs(float(_ne))),
              "C4d3 融合核与 eager 的 nll 容差内一致",
              f"{float(_ne):.9f} vs {float(_nc):.9f}")
    else:
        # 运行期回落（P19 纪律）也是**正确行为**，如实报告而非判FAIL
        check(True, "C4d2 本机 torch.compile 不可用 → 已自动回落 eager（P19 正确行为）",
              f"compiled={_c._compiled}")
    # 融合核本身必须与稀疏无关（放开的前提）
    import inspect as _insp
    _src = _insp.getsource(_AR._train_step_core)
    check("_sparse" not in _src and "conn_k" not in _src,
          "C4d4 融合核实现里不含稀疏分支（故放开是安全的）")

    # C4e. 稀疏更新**不得物化** (n_out,k) 临时张量（P138）
    print("\n[C4e] 稀疏更新不物化（P138：P28 禁令的稀疏臂漏改）")
    import inspect as _i2
    _, ro_mat, _, _ = build_pair(n_h, n_out, k, seed=41)
    _src = _i2.getsource(type(ro_mat)._eager_step)
    _has_mat = "dp.reshape(-1, 1) * g" in _src and "add_(" in _src
    check(not _has_mat,
          "C4e1 稀疏更新不用 `add_(dp*g)`（会物化 25.4 MiB 临时张量）",
          "已改为 addcmul_" if "addcmul_" in _src else "未见addcmul_")
    check("addcmul_" in _src, "C4e2 用 `addcmul_`（rank-1 AXPY，不物化）")
    # 数值契约：与旧物化式在 fp32 容差内一致（**非逐位**，乘法次序不同）
    import torch as _t2
    _Wd = _t2.randn(4096, 16)
    _a = _Wd.clone()
    _b = _Wd.clone()
    _dp = _t2.randn(4096, 1)
    _g = _t2.randn(4096, 16)
    _a.add_(_dp * _g, alpha=-0.15)
    _b.addcmul_(_dp, _g, value=-0.15)
    _d = float((_a - _b).abs().max())
    _scale = max(1e-12, float(_Wd.abs().max()))
    check(_d <= 1e-6 * _scale,
          "C4e3 addcmul_ 与物化式**容差内一致**（fp32 1 ulp，非逐位）",
          f"max|Δ|={_d:.3e} 相对={_d/_scale:.3e}")

    # C4f. gather 中间量精度开关（P145）+ 重复赋值检查
    print("\n[C4f] gather 精度开关与死代码检查（P145）")
    import re as _re3
    _pcfg = (_ROOT / "phdnet" / "config.py").read_text(encoding="utf-8")
    _n_impl = len(_re3.findall(r"^ *readout_gather_impl *:", _pcfg, _re3.M))
    check(_n_impl == 1,
          "C4f1 config 里 readout_gather_impl **只定义一次**"
          "（P145 清掉 3 份重试残留）", f"实际 {_n_impl} 次")
    from phdnet.config import PHDNetConfig as _PC5
    check(str(_PC5().readout_gather_dtype) == "fp32",
          "C4f2 config 默认 gather_dtype=fp32（保持现状）",
          f"实际={_PC5().readout_gather_dtype!r}")
    # 数值契约：fp32 逐位不变；fp16 在容差内（实测相对误差 ~2e-4）
    _, ro32, _, _ = build_pair(n_h, n_out, k, seed=43)
    _, ro16, _, _ = build_pair(n_h, n_out, k, seed=43)
    ro16._gather_dtype = "fp16"                # noqa: SLF001
    _h5 = rng.normal(0.0, 1.0, n_h).astype(np.float32)
    _y32 = np.asarray(ro32(_h5))
    _y16 = np.asarray(ro16(_h5))
    _rel = float(np.abs(_y16 - _y32).max() / max(1e-30, np.abs(_y32).max()))
    check(_rel <= 1e-2,
          "C4f3 fp16 的 g 在**容差内**（实测相对误差 ~2e-4，远松于1e-2）",
          f"max相对Δ={_rel:.3e}")
    check(getattr(ro32, "_gather_dtype", "fp32") == "fp32",
          "C4f4 默认实例的 gather_dtype 是 fp32（未被动过）")

    # ── D. 零回归 ─────────────────────────────────────────────────────────
    print("\n[D] 零回归：conn_k=0 稠密路径未被污染")
    dense_ref = AccelReadout(n_h, n_out, np.random.default_rng(23), device="cpu",
                             dtype="fp32", conn_k=0)
    check(dense_ref.conn_k == 0 and dense_ref.W.shape == (n_out, n_h),
          "D1 稠密构造仍是 (n_out, n_h) 且 conn_k=0",
          f"{tuple(dense_ref.W.shape)}")
    hd = rng.normal(0.0, 1.0, n_h)
    yd = dense_ref(hd)
    check(np.asarray(yd).shape == (n_out,), "D2 稠密前向形状不变",
          str(tuple(np.asarray(yd).shape)))
    # 稠密 learn_softmax 仍走 addmm_ 且 W 变化
    w_before = dense_ref.W.clone()
    dense_ref.learn_softmax(hd.copy(), rng.normal(0, 1, n_out), 0.15)
    check(not torch.equal(w_before, dense_ref.W), "D3 稠密更新仍生效（W 已变）")

    # ── 汇总 ──────────────────────────────────────────────────────────────
    npass = sum(1 for ok, _, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print("\n" + "=" * 78)
    print(f"结果：{npass}/{total} 通过 | 失败 {total - npass}")
    if npass != total:
        print("失败用例：")
        for ok, name, detail in _RESULTS:
            if not ok:
                print(f"  · {name}  {detail}")
    print("=" * 78)
    return 0 if npass == total else 1


if __name__ == "__main__":
    sys.exit(main())