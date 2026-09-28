# -*- coding: utf-8 -*-
"""多卡自动适配的等价性 / 行为对拍（P14，fhz 2026-09-28）。

用例
----
A  设备解析：auto / 显式列表 / 不可用设备诚实报错 / max_devices 截断
B  分片计划：均衡性、覆盖完整、空段剔除、参数校验
C  读出等价性：单设备 ≡ `TorchReadoutDense`（前向与更新后权重，**逐位**）
D  列并行分片 ≡ 单设备：4 路分片（同一设备上按行切分）前向 / NLL / 更新后权重**逐位**
E  并行计划报告：策略、预期加速、通信量、未并行部分齐备
F  host 线程收敛：多卡建议值 1

运行：python tests/verifiers/verify_multi_device.py
GPU 真机路径（多张不同卡）待硬件到位后另测——本机无加速器，
D 例以「同设备多路分片」验证分片逻辑的逐位等价（跨设备为容差一致判据，
已在 multi_device.py docstring 显式声明）。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_HERE), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                          # noqa: E402

from phdnet.backends.multi_device import (                   # noqa: E402
    MultiDeviceReadout, configure_host_threads, plan_parallel,
    probe_multi, resolve_devices, shard_ranges)
from phdnet.backends.torch_lm import TorchReadoutDense        # noqa: E402

_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" —— {detail}" if detail else ""), flush=True)
    if not ok:
        _FAILURES.append(name)


def _bitwise(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def main() -> None:
    rng = np.random.default_rng(11)
    V, H = 517, 129                          # 质数维度：切分不整齐，覆盖空段边界
    W0 = rng.standard_normal((V, H)).astype(np.float32) * 0.05
    h = rng.standard_normal(H).astype(np.float32)
    tgt = np.zeros(V, dtype=np.float32)
    tgt[rng.integers(0, V)] = 1.0

    # ── A 设备解析 ──
    print("[A] 设备解析")
    probes = probe_multi()
    devs = resolve_devices("auto")
    check("A.auto 返回列表且非空", isinstance(devs, list) and len(devs) >= 1,
          f"devices={devs} | probe=" + ", ".join(
              f"{k}:{v.get('count')}" for k, v in probes.items() if v.get("ok")))
    check("A.cpu 显式", resolve_devices("cpu") == ["cpu"])
    explicit = ",".join(devs[:1])
    check("A 显式列表回显", resolve_devices(explicit) == devs[:1], explicit)
    try:
        resolve_devices("cuda:99")
        check("A 不可用设备诚实报错", False, "未抛异常")
    except RuntimeError:
        check("A 不可用设备诚实报错", True, "RuntimeError（如实）")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fb = resolve_devices("cuda:99", allow_fallback=True)
        check("A allow_fallback 回退 cpu", fb == ["cpu"] and len(w) == 1)
    if len(devs) > 1:
        check("A max_devices 截断", len(resolve_devices("auto", max_devices=1)) == 1)
    else:
        print("  [SKIP] A.max_devices（本机单设备，截断等价恒等）")

    # ── B 分片计划 ──
    print("[B] 分片计划")
    r4 = shard_ranges(100, 4)
    check("B 均衡切分", r4 == [(0, 25), (25, 50), (50, 75), (75, 100)], str(r4))
    r3 = shard_ranges(10, 4)
    check("B 余数分配（10/4 → 3,3,2,2）", r3 == [(0, 3), (3, 6), (6, 8), (8, 10)],
          str(r3))
    r_empty = shard_ranges(2, 4)
    check("B 空段剔除（2/4）", r_empty == [(0, 1), (1, 2)], str(r_empty))
    r_big = shard_ranges(517, 4)
    cover = sum(e - s for s, e in r_big)
    check("B 覆盖完整且不重叠", cover == 517
          and r_big[0][0] == 0 and r_big[-1][1] == 517, str(r_big))
    try:
        shard_ranges(10, 0)
        check("B 非法设备数被拒", False)
    except ValueError:
        check("B 非法设备数被拒", True, "ValueError")

    # ── C 单设备 ≡ TorchReadoutDense ──
    print("[C] 单设备 ≡ TorchReadoutDense（逐位）")
    ref = TorchReadoutDense(W0.copy(), "cpu", __import__("torch").float32)
    mr1 = MultiDeviceReadout(W0.copy(), ["cpu"], dtype="fp32")
    y_ref = ref.forward(__import__("torch").as_tensor(h))
    y_1 = mr1.forward(h)
    check("C 前向逐位一致", _bitwise(y_ref.numpy(), y_1.cpu().numpy()),
          f"max|Δ|={np.abs(y_ref.numpy() - y_1.cpu().numpy()).max():.3e}")
    tt = __import__("torch").as_tensor(tgt)
    nll_ref = ref.learn_softmax(__import__("torch").as_tensor(h), tt, 0.15)
    nll_1 = mr1.learn_softmax_np(h, tgt, 0.15)
    check("C NLL 逐位一致", nll_ref == nll_1, f"{nll_ref!r} vs {nll_1!r}")
    check("C 更新后 W 逐位一致", _bitwise(ref.W.numpy(), mr1.to_numpy()),
          f"max|Δ|={np.abs(ref.W.numpy() - mr1.to_numpy()).max():.3e}")

    # ── D 列并行分片 ≡ 单设备（同设备 4 路分片，逐位）──
    print("[D] 读出列并行分片 ≡ 单设备（4 路分片，逐位）")
    mr4 = MultiDeviceReadout(W0.copy(), ["cpu"] * 4, dtype="fp32")
    check("D 分片区间覆盖完整", sum(e - s for s, e in mr4.ranges) == V
          and mr4.ranges[-1][1] == V, str(mr4.ranges))
    y4 = mr4.forward_np(h)
    check("D 前向逐位一致（4 分片 ≡ 单设备）", _bitwise(y_1.cpu().numpy(), y4),
          f"max|Δ|={np.abs(y_1.cpu().numpy() - y4).max():.3e}")
    nll4 = mr4.learn_softmax_np(h, tgt, 0.15)
    check("D NLL 逐位一致", nll_1 == nll4, f"{nll_1!r} vs {nll4!r}")
    check("D 更新后 W 逐位一致（4 分片 ≡ 单设备）",
          _bitwise(mr1.to_numpy(), mr4.to_numpy()),
          f"max|Δ|={np.abs(mr1.to_numpy() - mr4.to_numpy()).max():.3e}")
    # 权重取回不破坏形状/内容
    check("D to_numpy 形状还原", mr4.to_numpy().shape == (V, H))
    # dtype 档
    for dt in ("fp16", "bf16"):
        m = MultiDeviceReadout(W0.copy(), ["cpu", "cpu"], dtype=dt)
        y = m.forward_np(h)
        check(f"D dtype={dt} 可前向", y.shape == (V,), f"dtype={y.dtype}")

    # ── E 并行计划报告 ──
    print("[E] 自动选型计划")
    p1 = plan_parallel(["cpu"], V, H)
    p4 = plan_parallel(["cpu"] * 4, V, H)
    check("E 单卡策略为原生路径", "单设备" in p1["strategy"]
          and p1["expected_speedup_readout_share"] == 1.0)
    check("E 多卡策略含列并行", "列并行" in p4["strategy"]
          and p4["n_devices"] == 4, p4["strategy"])
    check("E 通信量字段", p4["comm_per_step_bytes"] == V * 4 * 2
          and p1["comm_per_step_bytes"] == 0)
    check("E 未并行部分如实列出", len(p4["not_parallel"]) >= 3
          and "容差" in p4["limits"])

    # ── F host 线程收敛 ──
    print("[F] host 线程收敛")
    import os
    n = configure_host_threads(1)
    check("F 多卡建议 1 线程", n == 1
          and os.environ.get("OMP_NUM_THREADS") == "1")

    print("-" * 76)
    if _FAILURES:
        print(f"结果：{len(_FAILURES)} 例 FAIL → {_FAILURES}")
        sys.exit(1)
    print("结果：全部 PASS")


if __name__ == "__main__":
    main()
