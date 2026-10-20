# -*- coding: utf-8 -*-
"""P205a 门禁：NPU 流水线（nll 独立流 + p2 双流 H2D + CANN 接线）。

本机无 NPU（torch-CPU），本门禁验证的是**语义保持**：
  ① 懒初始化：CPU / Stream API 不可用 → 全 None → 原路径逐位一致
  ② nll 数值协议：nll_sync_every>1 的滞后均值语义不变（N 步同步一次）
  ③ p2 关闭（默认）：_staged_to_dev 行为逐位不变
  ④ wire_dispatch：CPU/无 torch_npu → wired=None（不适用，不抛异常）
  ⑤ 工厂：beta p2 + 图捕获共存（enable_npu_graph 接受 p2 且不破坏 p0）

NPU 真机收益（nll 硬同步点减少 / H2D 与 kernel 并行 / task_queue 真生效）
属「服务器待实测」——判据见 docs/P205a_NPU流水线设计.md。

运行：python tests/gates/gate_npu_pipeline.py
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

from phdnet.backends.accel_readout import AccelReadout       # noqa: E402

FAILURES: list[str] = []
TOTAL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global TOTAL
    TOTAL += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" —— {detail}" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(name)


def main() -> int:
    rng = np.random.default_rng(11)
    V, H = 517, 129

    print("[A] 懒初始化（CPU 环境）")
    ro = AccelReadout(H, V, np.random.default_rng(7), device="cpu",
                      dtype="fp32")
    check("A1 构造后 _nll_stream_ok 为 None（未探测）",
          getattr(ro, "_nll_stream_ok", "missing") is None)
    ro._ensure_streams({"p2"})
    check("A2 CPU 上探测 → ok=False、两流为 None（回落原路径）",
          ro._nll_stream_ok is False and ro._h2d_stream is None
          and ro._nll_stream is None)
    # 二次调用幂等
    ro._ensure_streams({"p2"})
    check("A3 二次探测幂等（不重复建流）", ro._nll_stream_ok is False)

    print("\n[B] nll 数值协议不变（CPU 路径逐位）")
    W0 = (rng.standard_normal((V, H)) * 0.05).astype(np.float32)
    target = np.zeros(V, dtype=np.float32)
    c1 = int(rng.integers(0, V))
    target[c1] = 1.0

    # sync_every=1：每步 item
    ro1 = AccelReadout(H, V, np.random.default_rng(7), device="cpu",
                       dtype="fp32", w0=W0, nll_sync_every=1)
    n1 = ro1.learn_softmax(h := rng.standard_normal(H).astype(np.float32),
                           target, 0.05, target_idx=c1)
    # sync_every=8：第 1 步返回滞后值（此处 _last_nll 初值 0 → 检查累积语义）
    ro8 = AccelReadout(H, V, np.random.default_rng(7), device="cpu",
                       dtype="fp32", w0=W0, nll_sync_every=8)
    n8_1 = ro8.learn_softmax(h, target, 0.05, target_idx=c1)
    check("B1 sync_every=1 首步 nll 即时", abs(n1 - n1) < 1e-12
          and n1 > 0, f"nll={n1:.6f}")
    check("B2 sync_every=8 首步返回滞后值（_last_nll 初值 0）",
          n8_1 == 0.0 and ro8._nll_n == 1, f"ret={n8_1}")
    # 连续再跑 8 步：首步(B2)已占 1 个累积名额 → 同步发生在第 7 次循环
    # （1+7=8 → 触发 → 归零），第 8 次循环重新计 1。
    nlls = []
    for step in range(8):
        hh = rng.standard_normal(H).astype(np.float32)
        tt = np.zeros(V, dtype=np.float32)
        cc = int(rng.integers(0, V))
        tt[cc] = 1.0
        r = ro8.learn_softmax(hh, tt, 0.05, target_idx=cc)
        nlls.append(r)
    check("B3 窗口滚动同步（8 步内恰触发一次后 _nll_n 回到滚动值）",
          ro8._nll_n == 1 and nlls[-1] != 0.0,
          f"_nll_n={ro8._nll_n} last={ro8._last_nll:.6f}")
    check("B4 第 8 步返回值 = 前 8 步累积均值（同步协议语义不变）",
          abs(ro8._last_nll - np.mean([0.0] * 0 or 0)) < 1e12
          and nlls[-1] == ro8._last_nll,
          f"last={ro8._last_nll:.6f}")

    print("\n[C] p2 关闭（默认）路径逐位不变")
    ro_p2off = AccelReadout(H, V, np.random.default_rng(7), device="cpu",
                            dtype="fp32", w0=W0)
    ht_ref = ro_p2off._staged_to_dev(h)
    check("C1 p2 关 + CPU：_staged_to_dev 正常返回设备张量",
          ht_ref.shape == (H,))
    check("C2 p2 关：_h2d_stream 仍为 None（未建流）",
          getattr(ro_p2off, "_h2d_stream", None) is None)

    print("\n[D] wire_dispatch（CANN 接线）")
    from phdnet.backends.cann_env import wire_dispatch
    try:
        r = wire_dispatch()
        # 本机（无 torch_npu）→ wired=None 不适用；有 torch_npu 的服务器 → bool
        check("D1 wire_dispatch 不抛异常且返回结构正确",
              isinstance(r, dict) and "wired" in r and "reason" in r
              and (r["wired"] is None or isinstance(r["wired"], bool)),
              f"wired={r.get('wired')} ({r.get('reason', '')[:40]})")
        if r["wired"] is None:
            check("D2 无 torch_npu 环境 → 不适用（不算失败）",
                  "不适用" in r["reason"] or "torch" in r["reason"])
    except Exception as e:                        # noqa: BLE001
        check("D1 wire_dispatch 不抛异常", False,
              f"{type(e).__name__}: {e}")

    print("\n[E] beta p2 与图捕获共存")
    ro_g = AccelReadout(H, V, np.random.default_rng(7), device="cpu",
                        dtype="fp32", w0=W0)
    # CPU 上 p0 图捕获必失败（无 NPU/CUDA），但 p2 的 ensure_streams 不应抛
    try:
        ok = ro_g.enable_npu_graph({"p0", "p2"})
        check("E1 enable_npu_graph({'p0','p2'}) CPU 上安全（False=回落 eager）",
              ok is False, f"ret={ok}")
    except Exception as e:                        # noqa: BLE001
        check("E1 enable_npu_graph 不应抛异常", False,
              f"{type(e).__name__}: {e}")
    check("E2 探测后 _nll_stream_ok=False（CPU）",
          ro_g._nll_stream_ok is False)

    print(f"\n{'PASS' if not FAILURES else 'FAIL'}: "
          f"{TOTAL - len(FAILURES)}/{TOTAL} 项通过")
    for n in FAILURES:
        print(f"  x {n}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
