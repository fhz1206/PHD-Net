# -*- coding: utf-8 -*-
"""P204 门禁：多卡读出列并行的等价性 / 行为对拍。

用例
----
A  单卡路径零回归：--accel-devices 1（默认）走原 AccelReadout（工厂类型断言）
B  多卡构造：分片计划均衡、行覆盖完整、w0 行对应、单卡 fail-fast
C  列并行 ≡ 单卡（**同设备多分片**）：
   C1 前向 y（cat 序 = 全局行序）—— 判据 **逐位**
   C2 learn_softmax 后权重 —— 判据 **容差 1e-6**（见下）
   C3 nll（主卡全局 softmax，与单卡同协议）—— 判据 1e-9
D  稀疏读出多卡 fail-fast（诚实报错，不静默）
E  工厂分发：accel_readout_devices>1 → 工厂分派；无加速器环境诚实回落

⚠ **为什么 C2 是容差而非逐位**（P204 实测，必须说清）：
  单卡 `W.add_(torch.outer(dp, ht), alpha=-eta)` 一次写整张 W；
  多卡把 dp 按行**切片**后各分卡算 `torch.outer(dp_i, ht)` 再写自己那片。
  逐行看两者语义相同，但 `torch.outer` 的**分片实现走不同的 kernel/分块**
  → 实测 max|d| = 1.49e-08（fp32 相对 ~1e-7 级）。这与 checkpoint
  `gate_multi_device.py` D 例的「同设备分片逐位」口径**不同**——
  那个 gate 是先在主卡 cat 出完整 y 再做更新，本类是**更新也分发**。
  故 P204 的等价判据 = **fp32 容差**（1e-6），与 Cython 开关同口径；
  已如实写进 `sharded_readout.py` docstring，不得称逐位等价。

运行：python tests/gates/gate_multi_device_sharded.py
本机无 NPU：C 例用 device="cpu" 的两个分片（跨设备为容差一致判据）。
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

from phdnet.backends.accel_readout import (AccelReadout,     # noqa: E402
                                          pick_readout_backend)
from phdnet.backends.sharded_readout import ShardedAccelReadout  # noqa: E402

FAILURES: list[str] = []
TOTAL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    """⚠ 参数顺序 = (name, ok, detail)——与其它 gate 一致。"""
    global TOTAL
    TOTAL += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" —— {detail}" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(name)


def main() -> int:
    rng = np.random.default_rng(11)
    V, H = 517, 129                          # 质数维度：切分不整齐
    W0 = (rng.standard_normal((V, H)) * 0.05).astype(np.float32)
    h = rng.standard_normal(H).astype(np.float32)
    target = np.zeros(V, dtype=np.float32)
    target[int(rng.integers(0, V))] = 1.0

    print("[A] 单卡路径零回归")
    ro1 = AccelReadout(H, V, np.random.default_rng(7), device="cpu",
                       dtype="fp32", w0=W0)
    check("A1 默认路径仍是原 AccelReadout（无 parts 属性）",
          isinstance(ro1, AccelReadout) and not hasattr(ro1, "parts"))

    print("\n[B] 多卡构造")
    ro_m = ShardedAccelReadout(H, V, np.random.default_rng(7),
                               devices=["cpu", "cpu"], dtype="fp32", w0=W0)
    check("B1 两分片", len(ro_m.parts) == 2, f"{ro_m.shards}")
    cover = sum(e - s for s, e in ro_m.shards)
    check("B2 行覆盖完整且连续",
          cover == V and ro_m.shards[0][0] == 0 and ro_m.shards[-1][1] == V)
    sizes = [e - s for s, e in ro_m.shards]
    check("B3 分片均衡（±1）", max(sizes) - min(sizes) <= 1, f"{sizes}")
    w_p0 = ro_m.parts[0].W.detach().cpu().numpy()
    check("B4 w0 行对应（分片 0 首行 = W0[0]）",
          np.allclose(w_p0[0], W0[0], rtol=0, atol=1e-6))
    try:
        ShardedAccelReadout(H, V, devices=["cpu"], dtype="fp32")
        check("B5 单设备 fail-fast", False, "未报错")
    except RuntimeError as e:
        check("B5 单设备 fail-fast（诚实报错）",
              "≥2" in str(e) or ">=2" in str(e))

    print("\n[C] 列并行 ≡ 单卡（同设备两分片）")
    y1 = ro1.forward(h)
    ym = ro_m.forward(h)
    check("C1 前向 y 逐位一致（行切分无归约差异）",
          np.array_equal(y1, ym), f"max|d|={np.abs(y1 - ym).max():.2e}")

    eta = 0.05
    ro1b = AccelReadout(H, V, np.random.default_rng(7), device="cpu",
                        dtype="fp32", w0=W0)
    ro_mb = ShardedAccelReadout(H, V, np.random.default_rng(7),
                                devices=["cpu", "cpu"], dtype="fp32", w0=W0)
    n1 = ro1b.learn_softmax(h, target, eta, y_pre=None,
                            target_idx=int(np.argmax(target)))
    nm = ro_mb.learn_softmax(h, target, eta, y_pre=None,
                             target_idx=int(np.argmax(target)))
    check("C3 nll 一致（主卡全局 softmax，1e-9）", abs(n1 - nm) < 1e-9,
          f"单卡 {n1:.6f} / 多卡 {nm:.6f}")
    w_single = ro1b.W.detach().cpu().numpy()
    w_multi = np.concatenate([p.W.detach().cpu().numpy() for p in ro_mb.parts],
                            axis=0)
    d1 = float(np.abs(w_single - w_multi).max())
    check("C2 首步更新后权重一致（fp32 容差 1e-6，见文件头说明）",
          d1 < 1e-6, f"max|d|={d1:.2e}")

    h2 = rng.standard_normal(H).astype(np.float32)
    t2 = np.zeros(V, dtype=np.float32)
    t2[int(rng.integers(0, V))] = 1.0
    y1d = ro1b.forward_dev(h2)
    n1b = ro1b.learn_softmax(h2, t2, eta, y_pre=y1d,
                             target_idx=int(np.argmax(t2)))
    ymd = ro_mb.forward_dev(h2)
    nmb = ro_mb.learn_softmax(h2, t2, eta, y_pre=ymd,
                              target_idx=int(np.argmax(t2)))
    w_single = ro1b.W.detach().cpu().numpy()
    w_multi = np.concatenate([p.W.detach().cpu().numpy() for p in ro_mb.parts],
                            axis=0)
    d2 = float(np.abs(w_single - w_multi).max())
    check("C2b 第二步（设备直通 + 缓存复用）一致",
          d2 < 1e-6 and abs(n1b - nmb) < 1e-9, f"max|d|={d2:.2e}")

    print("\n[D] 稀疏多卡 fail-fast")
    try:
        ShardedAccelReadout(H, V, devices=["cpu", "cpu"], conn_k=8)
        check("D1 稀疏多卡诚实报错", False, "未报错")
    except NotImplementedError as e:
        check("D1 稀疏多卡诚实报错", "P204" in str(e))

    print("\n[E] 工厂分发")
    class _Cfg:                                   # noqa: N801
        accel_readout = "cpu"
        readout_dtype = "fp32"
        readout_w_clip = 0.0
        readout_master_weight = False
        readout_shadow_refresh = 8
        readout_gather_impl = "index"
        readout_gather_dtype = "fp32"
        readout_int8_compute = "auto"
        fp8_conv = "torch"
        nll_sync_every = 1
        torch_compile = False
        torch_compile_mode = "default"
        readout_conn_k = 0
        lognormal_init = False
        exc_ratio = 0.8
        sparse_fwd_kernel = "mulsum"
        accel_readout_devices = 1

    ro_a, name_a = pick_readout_backend(_Cfg(), H, V,
                                        np.random.default_rng(3))
    # ⚠ `accel_readout="cpu"` 时工厂**正确地**直接走 numba Readout
    #   （spec in ("cpu","off","numba") 短路）——所以本机（无加速器）下
    #   合法结果是 numba 路径。E1 判据 = 「不抛异常 + 不是多卡对象」。
    check("E1 工厂：devices=1 → 单卡路径（numba 或 AccelReadout，非多卡）",
          (not isinstance(ro_a, ShardedAccelReadout)), name_a)
    _Cfg.accel_readout = "auto"          # auto：有加速器则走设备路径
    _Cfg.accel_readout_devices = 1
    ro_a2, name_a2 = pick_readout_backend(_Cfg(), H, V,
                                          np.random.default_rng(3))
    check("E1b 工厂：auto + devices=1 → 非多卡",
          not isinstance(ro_a2, ShardedAccelReadout), name_a2)
    _Cfg.accel_readout_devices = 2
    try:
        ro_b, name_b = pick_readout_backend(_Cfg(), H, V,
                                            np.random.default_rng(3))
        # cpu 上解析不出 2 设备 → 回落 numba CPU（诚实回落也是正确行为）
        check("E2 工厂：devices=2 → 多卡或诚实回落",
              (isinstance(ro_b, ShardedAccelReadout)
               or "回落" in name_b or "numba" in name_b), name_b)
    except Exception as e:                        # noqa: BLE001
        check("E2 工厂分发不抛异常", False, f"{type(e).__name__}: {e}")

    print(f"\n{'PASS' if not FAILURES else 'FAIL'}: "
          f"{TOTAL - len(FAILURES)}/{TOTAL} 项通过")
    for n in FAILURES:
        print(f"  x {n}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
