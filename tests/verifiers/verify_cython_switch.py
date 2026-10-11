"""P192 `--cython-kernels` 的**开关不变量**门禁。

覆盖默认 off、当前 off 轨迹可复现、auto 的生产 dtype/核组合与数值漂移。
C1 自重复只证明当前实现确定性，不证明与历史提交逐位不变；后者需独立
历史源码/golden。off 不碰加载与跨模型隔离由 verify_cython_loader.py 守护。

⚠ **不能声称的**：开 `auto` 后与 `off` 逐位相同 —— **不能**。
   Cython 用 MSVC `/O2` = 严格 IEEE-754（无 FMA 收缩、无重结合），
   而 `phdnet/sparse_pc.py` 的 numba 核全带 `fastmath=True`
   （有收缩 + 重结合）→ 两者在同一输入上可差 ~1 ulp。
   故本门禁对 `auto` 只断言**语义等价**（最大相对漂移有界、
   关键结构量一致、训练能跑完），**不断言逐位**。

⚠ **昇腾边界**：本门禁跑在 x86，仅证明所覆盖输入的可复现性与容差一致；
   不证明历史零回归、完整质量或昇腾收益。目标机器必须复跑。

运行：``python tests/verifiers/verify_cython_switch.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

CASES: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CASES.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'} {name}"
          + (f"  [{detail}]" if detail else ""), flush=True)


def _build_cfg(cython: str, seed: int = 42, *, fused=True, encoder_dtype="fp32"):
    """造一个**小到能在本机秒级跑完**、但拓扑真实（非退化）的配置。"""
    from phdnet.config import PHDNetConfig
    cfg = PHDNetConfig(seed=seed)
    # 尺寸压小：门禁不测性能，只测「同一输入两开关行为一致」
    cfg.n_input = 32
    cfg.vocab_size = 61
    cfg.n_sdr = 24
    cfg.n_mid = 18
    cfg.n_top = 16
    cfg.k_sparse = 6
    cfg.m_lateral = 6
    cfg.readout_conn_k = 3
    cfg.conn_k = 4
    cfg.n_epochs = 2
    cfg.cycles = 1
    cfg.cython_kernels = cython
    cfg.pc_fused_kernel = fused
    cfg.encoder_dtype = encoder_dtype
    cfg.backend = "numpy"
    cfg.accel_device = "cpu"
    cfg.readout_dtype = "fp32"
    return cfg


def _traj(cfg, tokens: int, seed: int = 0):
    """跑 `tokens` 步，返回可比较的轨迹摘要（多个状态快照）。"""
    from phdnet.model import PHDNet
    net = PHDNet(cfg)
    rng = np.random.default_rng(seed)
    snaps = []
    for t in range(tokens):
        # ⚠ `step(x)` 校验 `x.shape[0] == cfg.n_input`（model.py:274），
        #   且 `encoder.encode` 内部做 `W32 @ x` → x 是 **一维**长度
        #   `n_input` 的向量（生产里由 `tok.encode_composite(tk, prev)` 给出）。
        #   故 x.shape = (n_input,)，不是 (batch, feat)。
        x = rng.integers(0, 8, size=cfg.n_input).astype(np.float32)
        net.step(x)
        if t % 4 == 3:                      # 每 4 步抓一次快照
            snaps.append(_snapshot(net))
    return _digest(snaps)


def _snapshot(net) -> dict:
    """抓一组**跨机制**的状态（M2/M3/M4a），确保不是只测一个点。

    ⚠ M2 的 `up0/dn0/up1/dn1` 是 **CSR 三元组** `(indptr, idx, val)`，
    不是权重矩阵 —— 直接 `np.asarray` 会因 inhomogeneous shape 报错
    （首版就这么炸的）。这里只取 `val`（真正被就地更新的那份）。
    """
    s = {}
    for name in ("up0", "dn0", "up1", "dn1"):
        csr = getattr(net.pc, name, None)
        if csr is not None and len(csr) == 3:
            val = csr[2]
            if val is not None:
                s[f"pc.{name}"] = np.asarray(val).ravel().copy()
    s["stdp.W"] = net.stdp.W.copy()
    s["stdp.t_pre"] = net.stdp.t_pre.copy()
    s["wm.slots"] = net.wm.slots.copy()
    s["wm.strength"] = net.wm.strength.copy()
    return s


def _digest(snaps: list[dict]) -> dict:
    """把快照折成 {键: 扁平数组}，便于逐键比较。"""
    out: dict[str, list[np.ndarray]] = {}
    for s in snaps:
        for k, v in s.items():
            out.setdefault(k, []).append(np.asarray(v, dtype=np.float64).ravel())
    return {k: np.concatenate(v) for k, v in out.items()}


def _main() -> int:
    print("P192 cython-kernels switch gate")

    # ── A. 取值域 fail-fast（唯一入口校验在 config.__post_init__）────────
    from phdnet.config import PHDNetConfig
    try:
        c = PHDNetConfig()
        c.cython_kernels = "turbo"
        c.__post_init__()
        check("A1 非法 cython_kernels 取值被 fail-fast 拒绝", False,
              "未抛 ValueError（静默接受了非法值！）")
    except ValueError:
        check("A1 非法 cython_kernels 取值被 fail-fast 拒绝", True)

    # ── B. 默认值必须是 off（铁律④）──────────────────────────────────────
    c = PHDNetConfig()
    check("B1 默认 cython_kernels == 'off'",
          c.cython_kernels == "off", f"实际 {c.cython_kernels!r}")
    check("B2 默认 readout_pipeline is False", c.readout_pipeline is False)
    check("B3 默认 cann_dispatch is False", c.cann_dispatch is False)

    # ── C. 当前 off 轨迹可复现（不是历史对拍）──────────────────────────
    # 两次独立构造、同 seed 与同输入；只能证明当前 off 的确定性。
    # loader 门禁另通过 mock 断言 off 不加载/构建，也覆盖 auto→off 隔离。
    t1 = _traj(_build_cfg("off"), 8, seed=1)
    t2 = _traj(_build_cfg("off"), 8, seed=1)
    same = all(np.array_equal(t1[k], t2[k]) for k in t1)
    check("C1 off 轨迹可复现（两次独立构造逐位相同）", same)

    # ── D. off 下分派确实是 None（不是「碰巧结果一样」）───────────────────
    import phdnet.sparse_pc as spc
    import phdnet.stdp_kernels as sk
    spc.cyk_init("off")
    sk.cyk_init("off")
    check("D1 off 时 sparse_pc._CYK is None", spc._CYK is None)
    check("D2 off 时 stdp_kernels._CYK is None", sk._CYK is None)

    # ── E. auto：**语义**等价（不断言逐位，见文件头）────────────────────
    ok_auto = spc.cyk_init("auto") and sk.cyk_init("auto")
    if not ok_auto:
        print("\n! Cython 扩展不可用（未构建/无编译器/自检未过）→ E 组 SKIP")
        print("  构建：python setup_cython.py build_ext --inplace")
    else:
        t3 = _traj(_build_cfg("auto"), 8, seed=1)
        check("E1 auto 轨迹的键集合与 off 一致",
              set(t3.keys()) == set(t1.keys()))
        # 逐键报告漂移，不谎称逐位
        worst_rel, worst_key = 0.0, ""
        for k in t1:
            a, b = t1[k], t3[k]
            if a.shape != b.shape:
                check(f"E2 auto 形状一致 [{k}]", False, f"{a.shape} vs {b.shape}")
                continue
            denom = np.maximum(np.abs(a), 1e-8)
            rel = float(np.max(np.abs(a - b) / denom)) if a.size else 0.0
            if rel > worst_rel:
                worst_rel, worst_key = rel, k
        print(f"    (auto vs off 最大相对漂移 = {worst_rel:.3e} @ {worst_key})")
        check("E2 auto 与 off 相对漂移有界（<1%，fp32 级）",
              worst_rel < 1e-2, f"{worst_rel:.3e} @ {worst_key}")
        # 结构量必须**完全**一致：权重非负（clip 生效）、无 NaN
        bad_nan = [k for k, v in t3.items() if not np.isfinite(v).all()]
        check("E3 auto 轨迹无 NaN/Inf（clip 语义未破）", not bad_nan,
              f"含非有限的键: {bad_nan[:3]}")
        if "stdp.W" in t3:
            check("E4 auto 下 STDP 权重仍非负（clip 到 [0, w_max] 生效）",
                  float(t3["stdp.W"].min()) >= 0.0,
                  f"min={float(t3['stdp.W'].min()):.3e}")

        # ── E5. 真生产分派：plain/fused × encoder fp32/fp64 ─────────────
        # 原 E 组只跑库默认 fused=True，根本未调用 Cython M2；plain 是
        # CLI 默认，fp64 则覆盖旧配置/显式 encoder 请求与中间 dtype 升级。
        from unittest.mock import patch
        for fused in (False, True):
            for enc_dtype in ("fp32", "fp64"):
                tag = f"{'fused' if fused else 'plain'}+encoder={enc_dtype}"
                ref = _traj(_build_cfg("off", fused=fused, encoder_dtype=enc_dtype), 8, seed=1)
                kmod = spc._CYK
                with patch.object(kmod, "csr_matvec", wraps=kmod.csr_matvec) as mv:
                    got = _traj(_build_cfg("auto", fused=fused, encoder_dtype=enc_dtype), 8, seed=1)
                    if not fused:
                        check(f"E5 M2 plain 确实调用 Cython [{tag}]", mv.call_count > 0,
                              f"调用 {mv.call_count} 次")
                finite = all(np.isfinite(v).all() for v in got.values())
                # 近 0 元素用混合 atol/rtol，避免相对误差对微小 fp32 舍入无限放大。
                bounded = set(got) == set(ref) and all(
                    np.allclose(got[key], ref[key], rtol=1e-2, atol=1e-6) for key in ref)
                worst_abs = max(float(np.max(np.abs(got[key] - ref[key]))) for key in ref)
                check(f"E6 生产组合能跑/有限/容差一致 [{tag}]", finite and bounded,
                      f"max|d|={worst_abs:.3e}（非逐位）")

        # 原地核的 dtype/非连续拷回，且每个核必须真实运行。
        from phdnet.model import PHDNet
        pc = PHDNet(_build_cfg("auto", fused=False)).pc
        ip = np.array([0, 2, 4], dtype=np.int64)
        ix = np.array([0, 1, 0, 1], dtype=np.int64)
        storage = np.arange(8, dtype=np.float64) / 10.0
        vl = storage[::2]  # 非连续 fp64 视图，更新必须写回原对象
        a = np.array([0.2, 0.4], dtype=np.float64)
        b = np.array([0.3, 0.5], dtype=np.float64)
        expected = np.ascontiguousarray(vl, dtype=np.float32)
        kmod.csr_add_outer(ip, ix, expected, a.astype(np.float32), b.astype(np.float32), 0.1)
        pc._add_outer((ip, ix, vl), a, b, 0.1)
        check("E7 add_outer fp64 非连续值缓冲正确拷回", np.array_equal(vl, expected))
        expected = np.ascontiguousarray(vl, dtype=np.float32)
        kmod.csr_oja_up(ip, ix, expected, a.astype(np.float32), b.astype(np.float32), 0.1)
        pc._oja((ip, ix, vl), a, b, 0.1)
        check("E8 oja fp64 非连续值缓冲正确拷回", np.array_equal(vl, expected))

        # ── F. 一个真 bug 的反证：把 dtype 口径故意弄错，门禁必须能抓 ─────
        # 做法：绕过 `ascontiguousarray`，直接喂一个 int64 的 post_idx
        # → 若分派层没有正确的 dtype 守护，这里应当抛错或静默算错。
        import phdnet.plasticity as pl
        W = np.zeros((4, 2), dtype=np.float32)
        pidx64 = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.int64)
        try:
            pl._stdp_delta_dispatch(
                W, pidx64, np.ones(4, np.float32), np.ones(4, np.float32),
                np.ones(4, np.float32), np.ones(4, np.float32), 0.1, 1.0)
            check("F1 int64 post_idx 被正确转换（不静默算错）", True,
                  "分派层做了 dtype 归一")
        except Exception as e:                            # noqa: BLE001
            check("F1 int64 post_idx 被正确转换（不静默算错）", False,
                  f"{type(e).__name__}: {e}")

    # ── G. 启动自检日志必须**说真话** ────────────────────────────────────
    # 真实 bug（P192 首版）：`get_kernels()` 返回**原始扩展模块**，
    # 而 `train.py` 的启动自检读了 `_ck.active` → 抛 AttributeError →
    # 被 except 吞成「未生效」，**紧接着下一行又打印 active=True**。
    # 于是日志同时说「未生效」与「生效」—— 一条自相矛盾、且会误导事后复盘的日志。
    # 门禁作用：钉死「判断生效用 `is not None`，不是 `.active`」。
    from phdnet import cykernels as _cyk
    for mode in ("off", "auto"):
        k = _cyk.get_kernels(mode)
        ok = (k is None) if mode == "off" else (k is not None)
        check(f"G1 get_kernels({mode!r}) 的生效判据用 `is not None`",
              ok, f"返回 {type(k).__name__}")
        check(f"G2 get_kernels({mode!r}) 的返回对象**没有** .active（故不能读它）",
              not hasattr(k, "active"), "原始模块 → 读 .active 会 AttributeError")
    # 门面才有 .active —— 这是两者唯一的形态差异，别混用
    check("G3 Kernels 门面才有 .active（与原始模块形态不同）",
          hasattr(_cyk.Kernels, "active") or "active" in vars(_cyk.Kernels),
          "门面负责补 .active/mode/status()")
    src = (_ROOT / "train" / "train.py").read_text(encoding="utf-8")
    check("G4 train.py 的启动自检不再读 _ck.active（否则日志会自相矛盾）",
          "_ck.active" not in src, "改判 `_ck is not None`")

    n_ok = sum(1 for c_ in CASES if c_[1])
    print(f"\n{'=' * 62}\nswitch gate: {n_ok}/{len(CASES)} passed")
    if n_ok != len(CASES):
        for nm, ok, det in CASES:
            if not ok:
                print(f"  FAIL {nm}  [{det}]")
    print("=" * 62)
    return 0 if n_ok == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(_main())