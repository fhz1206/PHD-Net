"""P211 门禁：CPU 优化（int32 CSR 索引 / forward-only 融合核 / LTM 内联）的
不变量与逐位等价性。

跑法：py tests/gates/gate_cpu_opt_p211.py

覆盖三组：

[A] **稀疏读出 int32 索引**（`_compress_csr_idx`）
    A1  列索引放得下时压到 int32
    A2  放不下时**原样返回**（不静默截断 —— 截断会让寻址指向错误列）
    A3  int32 与 int64 索引的 matvec / add_outer **逐位相同**
    A4  索引值本身未被修改（只改 dtype）

[B] **forward-only 融合核**（`_pc_fo_fused`）
    B1  融合核与多核参考实现**逐位等价**（dn0/dn1 权重）
    B2  融合核**不触碰上行**（P198：eta_oja>0 会正反馈发散）
    B3  融合核不新增连接（CSR 行宽不变）
    B4  稳态缩放只作用于下行（上行本路径不改）

[C] **LTM dict 版内联优化**
    C1  `_enc` 内联后与原实现逐位相同（含 int8 模式）
    C2  `_decay_read` 提到循环外后逐位相同
    C3  突变测试契约仍成立：`_decay_read` 注入负向对照能被检出
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "phdnet")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phdnet.readout import Readout                              # noqa: E402
from phdnet.sparse_pc import (SparsePCStack, _compress_csr_idx,  # noqa: E402
                              _csr_add_outer, _csr_matvec)
from phdnet.sparse_table import SparseSynapseTable               # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))


# ── A 组：稀疏读出 int32 索引 ────────────────────────────────────────────────
def group_a() -> None:
    print("\n[A] 稀疏读出 int32 索引（访存优化）")

    # A1：列数放得下 → 压到 int32
    ip = np.arange(0, 4 * 3 + 1, 3, dtype=np.int64)
    idx = np.array([0, 2, 5, 1, 3, 7, 2, 4, 6, 0, 1, 3], dtype=np.int64)
    val = np.arange(12, dtype=np.float32)
    ip2, idx2, val2 = _compress_csr_idx((ip, idx, val), n_cols=8)
    check(idx2.dtype == np.int32, "A1a 列数放得下 → 索引压到 int32",
          str(idx2.dtype))
    check(np.array_equal(idx2.astype(np.int64), idx),
          "A1b 压缩后索引**值**逐位不变")
    check(val2 is val, "A1c 值数组不被复制/修改")

    # A2：列数放不下 → 原样返回（不截断）
    ip3, idx3, val3 = _compress_csr_idx((ip, idx, val), n_cols=0x80000000)
    check(idx3.dtype == np.int64 and idx3 is idx,
          "A2a 列数超 int32 → 原样返回 int64（不静默截断）", str(idx3.dtype))

    # A3：int32 与 int64 的数值等价（matvec / add_outer 逐位）
    rng = np.random.default_rng(0)
    V, H, K = 4000, 256, 32
    ip = np.arange(0, (V + 1) * K, K, dtype=np.int64)
    idx64 = rng.integers(0, H, size=V * K).astype(np.int64)
    idx32 = idx64.astype(np.int32)
    val = rng.standard_normal(V * K).astype(np.float32)
    h = rng.standard_normal(H).astype(np.float32)

    y64 = _csr_matvec(ip, idx64, val, h)
    y32 = _csr_matvec(ip, idx32, val, h)
    check(np.array_equal(y64, y32), "A3a matvec：int32 索引与 int64 逐位相同",
          f"max|Δ|={float(np.abs(y64.astype(np.float64) - y32.astype(np.float64)).max()):.3e}")

    v64, v32 = val.copy(), val.copy()
    dp = rng.standard_normal(V)
    _csr_add_outer(ip, idx64, v64, dp, h, -0.15)
    _csr_add_outer(ip, idx32, v32, dp, h, -0.15)
    check(np.array_equal(v64, v32), "A3b add_outer：int32 索引与 int64 逐位相同",
          f"max|Δ|={float(np.abs(v64.astype(np.float64) - v32.astype(np.float64)).max()):.3e}")

    # A4：真实 Readout 的稀疏读出走 int32
    ro = Readout(n_in=128, n_out=500, rng=np.random.default_rng(0), conn_k=16)
    check(ro._csr[1].dtype == np.int32,
          "A4 真实稀疏读出的 CSR 索引为 int32", str(ro._csr[1].dtype))


def _ref_learn(pc, cache, tau_r1, tau_s0, eta, homeostasis=True) -> None:
    """forward-only 学习的多核参考实现（原子核逐条调用）。

    用途：对拍融合核 `_pc_fo_fused` 的**逐位等价性**。
    ⚠ 这是**参考基准**，不是生产路径 —— 生产走融合核（1 次调用，2.4×）。
    """
    r1, r2 = cache["r1"], cache["r2"]
    pred_r1 = pc._mv(pc.dn1, r2)
    pred_s0 = pc._mv(pc.dn0, r1)
    d_r1 = np.asarray(tau_r1, dtype=np.float64) - pred_r1
    d_s0 = np.asarray(tau_s0, dtype=np.float64) - pred_s0
    pc._add_outer(pc.dn1, d_r1, r2, eta)
    pc._add_outer(pc.dn0, d_s0, r1, eta)
    pc._clip(pc.dn0[2], pc.w_max)
    pc._clip(pc.dn1[2], pc.w_max)
    if homeostasis:
        pc._scale_rows(pc.dn0, pc._hn_dn0)
        pc._scale_rows(pc.dn1, pc._hn_dn1)


# ── B 组：forward-only 融合核 ───────────────────────────────────────────────
def group_b() -> None:
    print("\n[B] forward-only 融合核（_pc_fo_fused）")

    n0 = n1 = n2 = 256
    pc_a = SparsePCStack(n0, n1, n2, eta_pc=0.02, eta_oja=0.0,
                         rng=np.random.default_rng(11), conn_k=32)
    pc_b = SparsePCStack(n0, n1, n2, eta_pc=0.02, eta_oja=0.0,
                         rng=np.random.default_rng(11), conn_k=32)

    rng = np.random.default_rng(0)
    s0 = rng.standard_normal(n0).astype(np.float32)
    cache = pc_a.infer(s0, 2)
    tau_r1 = np.tanh(rng.standard_normal(n1))
    tau_s0 = np.tanh(rng.standard_normal(n0))

    # B1：融合核 vs 多核参考实现（homeostasis=True）
    pc_a.learn_forward_target(cache, tau_r1, tau_s0, 0.01, homeostasis=True)
    _ref_learn(pc_b, cache, tau_r1, tau_s0, 0.01, homeostasis=True)

    for name in ("dn0", "dn1"):
        a = getattr(pc_a, name)[2]
        b = getattr(pc_b, name)[2]
        check(np.array_equal(a, b),
              f"B1 {name}：融合核与多核参考实现逐位相同（homeostasis=True）",
              f"max|Δ|={float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()):.3e}")

    # B1b（P212 新增）：**homeostasis=False 时也必须逐位相同**。
    # ⚠ 这条锁住 P211 首版的真实回归：融合核的 ⑥ 行范数缩放曾写成
    #   **无条件执行** → 忽略 `homeostasis` 参数语义（生产恒为 True 故
    #   未暴露，但库默认 `PHDNetConfig().homeostasis=False` → 直用库
    #   配置者会拿到静默不同的数值）。
    pc_c = SparsePCStack(n0, n1, n2, eta_pc=0.02, eta_oja=0.0,
                         rng=np.random.default_rng(11), conn_k=32)
    pc_d = SparsePCStack(n0, n1, n2, eta_pc=0.02, eta_oja=0.0,
                         rng=np.random.default_rng(11), conn_k=32)
    cache_c = pc_c.infer(s0, 2)
    pc_c.learn_forward_target(cache_c, tau_r1, tau_s0, 0.01, homeostasis=False)
    _ref_learn(pc_d, cache_c, tau_r1, tau_s0, 0.01, homeostasis=False)
    ok_b1b = True
    det_b1b = []
    for name in ("dn0", "dn1"):
        a = getattr(pc_c, name)[2]
        b = getattr(pc_d, name)[2]
        same = np.array_equal(a, b)
        ok_b1b = ok_b1b and same
        det_b1b.append(f"{name}:{same}")
    check(ok_b1b,
          "B1b homeostasis=False：融合核与参考实现同样逐位相同"
          "（P212：修复了融合核无条件缩放的语义回归）",
          " ".join(det_b1b))

    # B1c（P212 新增）：homeostasis=False 与 True 的结果**必须不同**
    # （否则说明标志根本没生效 —— 即 B1 之前的 bug 形态）。
    nrm_false = float(np.sqrt((pc_c.dn1[2] ** 2).sum()))
    nrm_true = float(np.sqrt((pc_a.dn1[2] ** 2).sum()))
    check(nrm_false != nrm_true,
          "B1c homeostasis=False 与 True 结果不同（标志确实生效）",
          f"Σ‖dn1‖：False={nrm_false:.6f} True={nrm_true:.6f}")

    # B1d（P212 新增）：**`learn_scale` 调制**（与读出同口径）。
    # 起因：RL 把策略梯度灌进读出，用 `learn_scale = lr·advantage` 缩放
    # 读出学习率；而 forward-only 此前只用固定强度 `pc_scale` → 在 RL 里
    # 它以**与奖励无关**的强度持续注入噪声 → RL 学不动
    # （实测 `gate_rl_loop` D 组：0.375 → 0.308 不升）。
    # → forward-only 现在同口径接受 `learn_scale`。
    # 本组断言：① learn_scale=1.0 与不调制**逐位相同**（普通训练不受影响）；
    #          ② learn_scale=0 → 该步**不产生任何** forward-only 更新。
    from phdnet.config import PHDNetConfig
    from phdnet.model import PHDNet

    def _net_and_digest(ls: float):
        cfg = PHDNetConfig(n_input=32, n_sdr=64, n_mid=64, n_top=32,
                           k_sparse=16, n_readout=50, readout_softmax=True,
                           eta_readout=0.15, seed=11, step_profiling=False,
                           accel_readout="cpu:0", forward_only=True)
        net = PHDNet(cfg)
        rng2 = np.random.default_rng(0)
        for i in range(20):
            x = rng2.standard_normal(32).astype(np.float32)
            t = np.zeros(50)
            t[i % 50] = 1.0
            net.step(x, target=t, learn_scale=ls)
        h = hashlib.sha256()
        for arr in (net.pc.dn0[2], net.pc.dn1[2], net.pc.up0[2],
                    net.pc.up1[2], net.readout.W):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()

    d_ls1 = _net_and_digest(1.0)
    d_ls0 = _net_and_digest(0.0)
    check(d_ls1 != d_ls0,
          "B1d learn_scale=1.0 与 0 的训练轨迹不同（forward-only 确实受它调制）",
          f"ls=1:{d_ls1[:12]} ls=0:{d_ls0[:12]}")

    # learn_scale=0 时 forward-only 不注入 → 与「关掉 forward-only」等价
    def _net_no_fo(ls: float):
        cfg = PHDNetConfig(n_input=32, n_sdr=64, n_mid=64, n_top=32,
                           k_sparse=16, n_readout=50, readout_softmax=True,
                           eta_readout=0.15, seed=11, step_profiling=False,
                           accel_readout="cpu:0", forward_only=False)
        net = PHDNet(cfg)
        rng2 = np.random.default_rng(0)
        for i in range(20):
            x = rng2.standard_normal(32).astype(np.float32)
            t = np.zeros(50)
            t[i % 50] = 1.0
            net.step(x, target=t, learn_scale=ls)
        h = hashlib.sha256()
        for arr in (net.pc.dn0[2], net.pc.dn1[2], net.pc.up0[2],
                    net.pc.up1[2], net.readout.W):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()

    check(d_ls0 == _net_no_fo(0.0),
          "B1e learn_scale=0 的 forward-only 轨迹 == 关闭 forward-only 的轨迹",
          f"ls0={d_ls0[:12]}")

    # B2：不触碰上行（P198）
    up0_before = pc_a.up0[2].copy()
    up1_before = pc_a.up1[2].copy()
    pc_a.learn_forward_target(cache, tau_r1, tau_s0, 0.01, homeostasis=True)
    check(np.array_equal(pc_a.up0[2], up0_before)
          and np.array_equal(pc_a.up1[2], up1_before),
          "B2 融合核不触碰上行识别权重（P198 上行只读约束）")

    # B3：不新增连接（CSR 行宽不变）
    shapes_before = (len(pc_a.dn0[2]), len(pc_a.dn1[2]))
    pc_a.learn_forward_target(cache, tau_r1, tau_s0, 0.01, homeostasis=True)
    shapes_after = (len(pc_a.dn0[2]), len(pc_a.dn1[2]))
    check(shapes_before == shapes_after,
          "B3 融合核只更新已存在的边（不新增连接）",
          f"{shapes_before} → {shapes_after}")

    # B4：稳态缩放只作用于下行 —— 上行行范数在 forward-only 前后不变
    up0_norm_before = float(np.sqrt((pc_a.up0[2] ** 2).sum()))
    up1_norm_before = float(np.sqrt((pc_a.up1[2] ** 2).sum()))
    pc_a.learn_forward_target(cache, tau_r1, tau_s0, 0.01, homeostasis=True)
    up0_norm_after = float(np.sqrt((pc_a.up0[2] ** 2).sum()))
    up1_norm_after = float(np.sqrt((pc_a.up1[2] ** 2).sum()))
    check(up0_norm_before == up0_norm_after
          and up1_norm_before == up1_norm_after,
          "B4 稳态缩放只作用于下行（上行行范数不变）",
          f"up0 {up0_norm_before:.6f}→{up0_norm_after:.6f}")


# ── C 组：LTM dict 版内联优化 ───────────────────────────────────────────────
def _orig_learn(t, prev, cur):
    """P210 原版 learn（逐字复刻，作逐位对照基准）。"""
    if not cur:
        return
    t._touch(cur, t.t_post, t.stamp_post)
    if not prev:
        return
    t._touch(prev, t.t_pre, t.stamp_pre)
    out, eta, w_max = t.out, t.eta, t.w_max
    int8 = t.int8_store
    q = t._q
    grow_order = cur
    if t.growth_guidance and len(cur) > 1:
        grow_order = sorted(cur, key=lambda k: t.in_deg.get(k, 0))
    for i in prev:
        tpi = t._decay_read(t.t_pre, t.stamp_pre, i)
        if tpi <= 0.0:
            continue
        bucket = out.get(i)
        for k in grow_order:
            tpk = t.t_post.get(k, 0.0)
            ltp = eta * tpi * tpk
            if ltp <= 0.0:
                continue
            if bucket is not None and k in bucket:
                w_now = bucket[k] / q if int8 else bucket[k]
                ltd = (eta * t._decay_read(t.t_post, t.stamp_post, i)
                       * t._decay_read(t.t_pre, t.stamp_pre, k))
                bucket[k] = t._enc(min(max(w_now + ltp - ltd, 0.0), w_max))
            else:
                if bucket is None:
                    bucket = out[i] = {}
                if len(bucket) < t.m_out:
                    bucket[k] = t._enc(min(ltp, w_max))
                    if t.growth_guidance:
                        t.in_deg[k] = t.in_deg.get(k, 0) + 1


def group_c() -> None:
    print("\n[C] LTM dict 版内联优化（_enc / _decay_read 提循环外）")

    for int8 in (False, True):
        ta = SparseSynapseTable(1 << 20, 60, 0.7, 0.08, 1.0, seed=1,
                                int8_store=int8)
        tb = SparseSynapseTable(1 << 20, 60, 0.7, 0.08, 1.0, seed=1,
                                int8_store=int8)
        rng = np.random.default_rng(7)
        for _ in range(40):
            cur = list(rng.choice(1 << 20, 48, replace=False))
            prev = list(rng.choice(1 << 20, 48, replace=False))
            seed_w = {i: {k: float(rng.random()) * 0.9
                          for k in rng.choice(cur[:12], 8, replace=False)}
                      for i in prev[:18]}
            for t in (ta, tb):
                t._touch(cur, t.t_post, t.stamp_post)
                t._touch(prev, t.t_pre, t.stamp_pre)
                for i, bmap in seed_w.items():
                    b = t.out.setdefault(i, {})
                    for k, v in bmap.items():
                        b[k] = t._enc(v)
            _orig_learn(ta, prev, cur)
            tb.learn(prev, cur)

        same_keys = set(ta.out) == set(tb.out)
        diffs = [(i, k) for i in ta.out if i in tb.out
                 for k in ta.out[i]
                 if k in tb.out[i] and ta.out[i][k] != tb.out[i][k]]
        n_edges = sum(len(b) for b in ta.out.values())
        check(same_keys and not diffs,
              f"C1 int8={int8}：内联后与原版 learn 逐位相同",
              f"边数={n_edges} 不同={len(diffs)}")

    # C2：突变测试契约 —— `_decay_read` 注入负向对照必须能被检出。
    # ⚠ 判据用「**惰性 vs 急切参考**」对比（与既有的
    #   `gate_ltm_trace_semantics` 同款）：关掉 λ^Δt 补偿后，惰性表的
    #   ΣW 会偏离急切参考 → 必须能被检出。
    #
    # ⚠⚠ **三个场景构造要点（缺一个就会「假通过」）**：
    #   ① **必须自增 `step_count`** —— 补偿量是 `λ^(step_count − stamp)`，
    #      而 `step_count` 由**调用方**（`bigltm.SparseLTM.imprint`）
    #      自增；裸调 `table.learn()` 不自增 ⇒ dt 恒为 0 ⇒ 补偿永不生效
    #      ⇒ 突变注入变成空操作（假通过）。
    #   ② **必须用小的 id 空间**（如 48 而非 2²⁰）—— id 空间大则
    #      各步几乎不重复 → `stamp` 永不滞后 → 补偿无从体现。
    #   ③ 必须让 **`k ∈ cur` 但 `k ∉ prev`**（P193 修的正是 LTD 的
    #      第二条迹 `t_pre[k]`）。
    UNIV = 48
    rng = np.random.default_rng(3)
    seq = []
    for _ in range(60):
        np_prev = rng.choice(UNIV, size=int(rng.integers(3, 9)), replace=False)
        np_cur = rng.choice(UNIV, size=int(rng.integers(3, 9)), replace=False)
        seq.append(([int(x) for x in np_prev], [int(x) for x in np_cur]))

    def _drive(t, seq_) -> None:
        for prev, cur in seq_:
            t.learn(prev, cur)
            t.step_count += 1

    ta = SparseSynapseTable(1 << 12, 8, 0.7, 0.08, 1.0, seed=1)
    tb = SparseSynapseTable(1 << 12, 8, 0.7, 0.08, 1.0, seed=1)
    tb._decay_read = lambda trace, stamp, i: trace.get(i, 0.0)
    _drive(ta, seq)
    _drive(tb, seq)
    wa = sum(sum(b.values()) for b in ta.out.values())
    wb = sum(sum(b.values()) for b in tb.out.values())
    ratio = (wb / wa) if wa else 1.0
    check(abs(ratio - 1.0) > 0.05,
          "C2 突变测试契约仍成立：关掉 λ^Δt 补偿可被检出",
          f"ratio={ratio:.4f}（偏离 1.0 达 {abs(ratio - 1.0) * 100:.1f}%）")


def main() -> None:
    print("=" * 68)
    print("P211 门禁：CPU 优化（int32 CSR 索引 / FO 融合核 / LTM 内联）")
    print("=" * 68)
    group_a()
    group_b()
    group_c()
    n_pass = sum(1 for ok, _, _ in _results if ok)
    n_all = len(_results)
    print("-" * 68)
    print(f"结果：{n_pass}/{n_all} 通过 | 失败 {n_all - n_pass}")
    print("=" * 68)
    if n_pass != n_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
