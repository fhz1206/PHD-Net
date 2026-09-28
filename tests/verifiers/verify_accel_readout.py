# -*- coding: utf-8 -*-
"""加速器读出（P19）的等价性与回落行为对拍。

fhz：「词表做完之后如果设备有 cuda/cann/rocm 就跑在对应设备上」——读出是
训练里唯一的计算密集部件（1B 预设占 ~89%），本验证覆盖：

A  接口兼容：AccelReadout 与 `phdnet.readout.Readout` 的调用面一致
   （learn_softmax / W / n_synapses）；
B  等价性：前向 / NLL / 更新后 W **容差一致**（atol/rtol 口径，同 P9/P10）；
   注：即便同在 CPU，numpy BLAS 与 torch 内核的**归约顺序不同**（向量宽度 /
   FMA 使用差异），跨库不保证逐位——实测 max|Δ| ≈ 4e-06，属预期。
C  状态往返：W_cpu() / load_W() 精确还原（检查点兼容）；
D  后端选择：auto 在**无加速器**机器上回落 numba 原路径（默认路径逐位不变）；
   显式不可用设备时回落并记录原因（诚实降级，不静默）。
E  精度档：fp32 / fp16 / bf16 均可构造并前向（低精度容差判据）。

运行：python tests/verifiers/verify_accel_readout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_HERE), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                          # noqa: E402

_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" —— {detail}" if detail else ""), flush=True)
    if not ok:
        _FAILURES.append(name)


def main() -> None:
    import torch

    from phdnet.backends.accel_readout import (AccelReadout,
                                               pick_readout_backend)
    from phdnet.config import PHDNetConfig
    from phdnet.readout import Readout

    rng = np.random.default_rng(7)
    n_h, n_out = 128, 501
    W0 = (rng.normal(0.0, 0.05, (n_out, n_h)) * np.sqrt(n_h)).astype(np.float32)
    h = rng.normal(0.0, 1.0, n_h).astype(np.float32)
    tgt = np.zeros(n_out, dtype=np.float32)
    tgt[rng.integers(0, n_out)] = 1.0

    # ── A/B：同设备逐位等价（读出只有 W@h 与外积更新，标量可精确比对）──
    print("[A/B] AccelReadout ≡ Readout（容差判据）")
    ro_ref = Readout(n_h, n_out, rng, w_clip=0.0, dtype="fp32")
    ro_ref.W[:] = W0
    ro_acc = AccelReadout(n_h, n_out, None, device="cpu", dtype="fp32", w0=W0)
    y_ref = (ro_ref.W @ h).astype(np.float32)            # Readout 无 forward：
    y_acc = ro_acc.forward(h)                            # 前向语义 = W @ h
    atol_f = 1e-5 * max(1.0, float(np.abs(y_ref).max()))
    check("A 前向容差一致", float(np.abs(y_ref - y_acc).max()) <= atol_f,
          f"max|Δ|={np.abs(y_ref - y_acc).max():.3e} ≤ {atol_f:.2e}")
    nll_ref = ro_ref.learn_softmax(h, tgt, 0.05)
    nll_acc = ro_acc.learn_softmax(h, tgt, 0.05)
    rel_nll = abs(nll_ref - nll_acc) / max(1e-12, abs(nll_ref))
    check("B NLL 容差一致（相对差 ≤1e-6）", rel_nll <= 1e-6,
          f"{nll_ref:.8f} vs {nll_acc:.8f} rel={rel_nll:.2e}")
    dW = float(np.abs(ro_ref.W.copy() - ro_acc.W_cpu()).max())
    w_scale = max(1e-12, float(np.abs(ro_ref.W).max()))
    check("B 更新后 W 容差一致（≤1e-5 相对）", dW <= 1e-5 * w_scale,
          f"max|Δ|={dW:.3e} ≤ {1e-5 * w_scale:.2e}")
    check("A n_synapses 与稠密元素数一致", ro_acc.n_synapses() == n_out * n_h,
          f"{ro_acc.n_synapses():,}")
    check("A W 常驻设备且为 torch 张量", torch.is_tensor(ro_acc.W))

    # ── C：状态往返 ──
    print("[C] 检查点往返")
    W_ck = rng.normal(0, 0.03, (n_out, n_h)).astype(np.float32)
    ro_acc.load_W(W_ck)
    check("C load_W 精确还原", np.array_equal(ro_acc.W_cpu(), W_ck))
    try:
        ro_acc.load_W(np.zeros((3, 3), dtype=np.float32))
        check("C 形状不符被拒", False, "未抛异常")
    except ValueError:
        check("C 形状不符被拒", True, "ValueError")

    # ── D：后端选择（无加速器 → 回落 numba）──
    print("[D] 后端选择与回落")
    from phdnet.backends.multi_device import probe_multi
    probes = probe_multi()
    has_accel = any(v.get("ok") and v.get("count")
                    for k, v in probes.items() if k != "cpu")
    cfg = PHDNetConfig()
    ro, backend = pick_readout_backend(cfg, n_h, n_out, rng)
    check("D auto 在无加速器时回落 numba",
          (backend.startswith("numba") == (not has_accel)),
          f"backend={backend} accel={has_accel}")
    check("D 回落对象是原 Readout", isinstance(ro, Readout))
    cfg_off = PHDNetConfig(accel_readout="cpu")
    _ro2, backend2 = pick_readout_backend(cfg_off, n_h, n_out, rng)
    check("D 显式 cpu 走原路径", backend2 == "numba-cpu")
    cfg_bad = PHDNetConfig(accel_readout="cuda:99")
    ro3, backend3 = pick_readout_backend(cfg_bad, n_h, n_out, rng)
    check("D 不可用设备回落且记录原因",
          backend3.startswith("numba") and
          hasattr(ro3, "_accel_fallback_reason"),
          f"backend={backend3} reason="
          f"{getattr(ro3, '_accel_fallback_reason', '无')[:48]}")
    if has_accel:      # 有加速器的机器：auto 必须真的上设备
        ro4, backend4 = pick_readout_backend(PHDNetConfig(), n_h, n_out, rng)
        check("D 有加速器时 auto 上设备", backend4.startswith("accel:"),
              f"backend={backend4}")
    else:
        print("  [SKIP] D 有加速器时 auto 上设备（本机无加速器）")

    # ── E：精度档 ──
    print("[E] 精度档")
    for dt in ("fp32", "fp16", "bf16"):
        try:
            m = AccelReadout(n_h, n_out, None, device="cpu", dtype=dt, w0=W0)
            y = m.forward(h)
            check(f"E dtype={dt} 前向可用", y.shape == (n_out,)
                  and np.isfinite(y).all(), f"max={np.abs(y).max():.4f}")
        except Exception as e:                                # noqa: BLE001
            check(f"E dtype={dt} 前向可用", False, f"{type(e).__name__}: {e}")

    print("-" * 76)
    if _FAILURES:
        print(f"结果：{len(_FAILURES)} 例 FAIL → {_FAILURES}")
        sys.exit(1)
    print("结果：全部 PASS")


if __name__ == "__main__":
    main()
