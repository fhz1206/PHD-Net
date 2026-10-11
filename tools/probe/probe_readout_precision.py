"""P110：读出精度机制性代价的数值验证（不改任何生产代码，纯测量）。

问题：低精度读出权重会不会把感知器 `p - t` 的**非目标行**更新舍入丢弃，
      使学习规则退化为「只提升目标行」的纯 Hebbian？

方法：按 P106 实测的量级构造更新量，逐步比较 fp32 / fp16 / bf16 下
      非目标行更新被保留的比例（保留 = 该元素实际发生变化）。
      同时验证 checkpoint 存盘往返（fp16 存盘）是否会二次截断。

判据：保留率 <100% 即表示该精度改变了学习规则（不是数值噪声，是语义丢失）。
"""
from __future__ import annotations

import numpy as np


_NP_DTYPE = {"fp32": np.float32, "fp16": np.float16, "bf16": None}  # bf16 走 ml_dtypes


def _to_bf16(a: np.ndarray) -> np.ndarray:
    """bf16 = 截断 float32 高 16 位后还原（与 torch/numpy 位模式一致）。"""
    return (a.astype(np.float32).view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def _round(a: np.ndarray, dtype: str) -> np.ndarray:
    """把数组舍入到目标精度并回到 fp64 域（模拟生产：码本舍入 → LUT 反量化）。"""
    if dtype == "bf16":
        return _to_bf16(a.astype(np.float32)).astype(np.float64)
    return a.astype(_NP_DTYPE[dtype]).astype(np.float64)


def probe(W0: np.ndarray, dp: np.ndarray, dtype: str) -> tuple[float, float]:
    """返回 (非目标行保留率, 目标行保留率)。

    保留率 = 「舍入后的 W+dp」与「同精度舍入后的 W」**实际不同**的元素比例。
    ⚠ 基线必须经过**同一精度**的舍入，否则跨精度比较是假阳性（fp32 值 vs
    fp16 值几乎总不相等，会得到虚高的"保留率"）。
    """
    baseline = _round(W0, dtype)
    updated = _round(W0 + dp, dtype)
    changed = updated != baseline
    # 第 0 行当作目标行（模拟 p − t 里目标行是唯一被提升的）
    target = changed[0]
    non_target = changed[1:]
    return float(non_target.mean()), float(target)


def main() -> None:
    rng = np.random.default_rng(20261001)
    print("=" * 78)
    print("P110 读出精度验证：低精度是否丢弃非目标行更新（|dp| 量级扫描）")
    print("=" * 78)

    # 真实读出权重量级：P9/P106 实测 W 元素多在 1e-2 ~ 1e-1
    for scale in (1e-1, 3e-2, 1e-2, 3e-3):
        W0 = rng.normal(0.0, scale, size=4096)
        print(f"\n--- |W| ~ {scale:.0e}（4096 行）---")
        print(f"{'|dp|':>10} | {'fp32 非目标':>12} | {'fp16 非目标':>12} | {'bf16 非目标':>12}"
              f" | {'fp32 目标':>9} | {'fp16 目标':>9} | {'bf16 目标':>9}")
        for dp_mag in (1e-4, 1e-5, 1e-6, 1e-7):
            dp = rng.normal(0.0, dp_mag, size=4096)
            r = {d: probe(W0, dp, d) for d in ("fp32", "fp16", "bf16")}
            print(f"{dp_mag:10.0e} | {r['fp32'][0]*100:11.2f}% | {r['fp16'][0]*100:11.2f}%"
                  f" | {r['bf16'][0]*100:11.2f}% | {r['fp32'][1]*100:8.2f}%"
                  f" | {r['fp16'][1]*100:8.2f}% | {r['bf16'][1]*100:8.2f}%")

    print("\n" + "=" * 78)
    print("结论判读")
    print("=" * 78)
    print("· fp32 非目标行保留率 100% = 精确的 p − t 规则（默认应为此）")
    print("· fp16/bf16 保留率 <100% = 非目标行更新被舍入丢弃 → 退化为纯 Hebbian")
    print("· fp16 半 ULP 约为 bf16 的 1/8，但仍远大于 1e-6 量级的 |dp|")

    # checkpoint 存盘往返：fp16 存盘本身也是一次截断
    print("\n--- checkpoint 存盘往返（训练中 fp32 累加，每 5k 步存一次 fp16）---")
    W = rng.normal(0.0, 3e-2, size=4096)
    for n_steps in (1, 10, 100):
        Wacc = W.copy()
        for _ in range(n_steps):
            Wacc = Wacc + rng.normal(0.0, 1e-6, size=4096)   # 每步非目标行更新
        kept_in_training = Wacc != W
        # 存盘一次 fp16
        survived = Wacc.astype(np.float16).astype(np.float64) != W
        print(f"  {n_steps:4d} 步累积：训练期保留 {kept_in_training.mean()*100:6.2f}%"
              f" → 经 1 次 fp16 存盘后剩 {survived.mean()*100:6.2f}%"
              f"  (存盘净损失 {(kept_in_training.mean()-survived.mean())*100:6.2f} pp)")


if __name__ == "__main__":
    main()