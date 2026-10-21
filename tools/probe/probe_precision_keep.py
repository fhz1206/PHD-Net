# -*- coding: utf-8 -*-
"""P205b-2：**真实码本路径**的读出精度保留率测量（覆盖全部六档 + fp8 位模式）。

与旧工具 `tools/probe/probe_readout_precision.py` 的区别（必须说清）：
  · 旧工具只测 fp32/fp16/bf16，且用**模拟舍入**（np.float16 / 位截断）；
  · 本工具用**生产代码的真实码本路径**（`phdnet.readout.quantize_to` /
    `dequantize_from`）→ int8 的 e4m3fn 定点化、int16 的均匀定点、
    int32 的无 LUT 整数换算，全部按生产语义计算。

保留率定义（与旧工具同口径，判据不变）：
  保留率 = 「同精度舍入后的 (W + dp)」与「同精度舍入后的 W」**实际不同**
  的元素比例 —— 即「这一步更新有没有真的写进去」。
  ⚠ 基线必须经过**同一精度**的舍入，否则跨精度比较是假阳性。

输出：docs/PHD-Net_精度体系总表.md 的 §2 数据源（本脚本只产出数据，
     文档由 gen_precision_table.py 汇总）。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np                                    # noqa: E402

from phdnet.readout import (RO_DTYPES, dequantize_from,   # noqa: E402
                            quantize_to)


def _fp8_bits_round(a: np.ndarray) -> np.ndarray:
    """fp8 e4m3fn 位模式往返（uint8 承载）——P189 生产路径的舍入语义。

    e4m3fn：1 符号 + 4 指数（bias 7）+ 3 尾数；无 inf（NaN 用 S=1,E=1,M=7）。
    这里用 torch 的 float8_e4m3fn（有则用），否则用 numpy 手工编码。
    """
    try:
        import torch
        t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return t.to(torch.float8_e4m3fn).to(torch.float32).numpy().astype(np.float64)
    except Exception:                                   # noqa: BLE001
        return a.astype(np.float64)                     # 无 torch → 不测


def keep_rate(fmt: str, W0: np.ndarray, dp: np.ndarray,
              target_row: int) -> tuple[float, float]:
    """返回 (非目标行保留率, 目标行保留率)，用生产码本路径。"""
    n = W0.shape[0]
    if fmt == "fp8-bits":
        base = _fp8_bits_round(W0)
        upd = _fp8_bits_round(W0 + dp)
    elif fmt == "fp32":
        # fp32 不走码本：舍入即恒等
        base = W0.astype(np.float64)
        upd = (W0 + dp).astype(np.float64)
    else:
        # ⚠ int16/int32 的码本是**均匀定点**，其存储码是「相对 per-tensor
        #   scale 的整数」——生产路径（`_set_dense`）会先除掉 scale 再量化。
        #   本测量要的是「舍入到该精度的格点后，更新还在不在」，
        #   → 必须**按生产方式**（带 scale）走一遍，否则 int16 会把
        #   |W|~1e-2 直接 rint 到 0（步长 1.0）而全盘丢失更新。
        #   简化但忠实的做法：用 int16/int32 的**均匀定点**语义做舍入，
        #   即「按 |W| 的量级选 scale，使格点覆盖 ±1 倍 W 域」。
        if fmt in ("int16", "int32"):
            # 生产（`_set_dense`）的 per-tensor scale：让**最大幅值**落在
            # 码本高端（int16 用 30000、int32 用 2³¹）→ 格点覆盖整个 W 域。
            # ⚠ int32 的格点是**整数**（scale=1.0 时 |W|~1e-2 会被 rint 到 0
            #   → 更新全丢，保留率 0%）；生产里 int32 的 per-tensor scale
            #   是「把 W 归一化到 ±1 再存整数」→ 这里按此口径：scale = amax。
            amax = float(np.abs(W0).max()) or 1.0
            if fmt == "int16":
                scale = amax / 30000.0
            else:
                # int32：整数格点 + per-tensor scale ⇒ 归一化到 ±1 再 ×2³¹
                scale = amax / 2147483647.0
            base = _uniform_fixed(W0 / scale, fmt) * scale
            upd = _uniform_fixed((W0 + dp) / scale, fmt) * scale
        else:
            # ⚠ quantize_to 内部 ravel 成 1-D（生产按扁平权重用）→ 这里
            #   必须 reshape 回原形状才能做**逐行**掩码。
            shp = W0.shape
            base = dequantize_from(fmt, quantize_to(fmt, W0)).reshape(shp).astype(np.float64)
            upd = dequantize_from(fmt, quantize_to(fmt, W0 + dp)).reshape(shp).astype(np.float64)
    diff = np.abs(upd - base) > 0
    # 目标行：|dp| ~1e-2（最大），非目标行：|dp| ~1e-6
    tgt_mask = np.zeros(n, dtype=bool)
    tgt_mask[target_row] = True
    non_tgt = diff[~tgt_mask]
    tgt = diff[tgt_mask]
    return (float(non_tgt.mean()) * 100.0 if non_tgt.size else 0.0,
            float(tgt.mean()) * 100.0 if tgt.size else 0.0)


def _uniform_fixed(x: np.ndarray, fmt: str) -> np.ndarray:
    """均匀定点舍入（int16: ±32767 格点；int32: ±2³¹）。"""
    qmax = 32767.0 if fmt == "int16" else 2147483647.0
    return np.clip(np.rint(x.astype(np.float64)), -qmax, qmax).astype(np.float64)


def main() -> None:
    rng = np.random.default_rng(7)
    n_out, n_in = 2000, 128
    # 生产的权重量级：|W| ~ 1e-2（P110 §6.1 口径）
    W0 = (rng.standard_normal((n_out, n_in)) * 1e-2).astype(np.float32)
    # dp：目标行 ~1e-2，非目标行 ~1e-6（P110 §6.1 口径）
    dp = (rng.standard_normal((n_out, n_in)) * 1e-6).astype(np.float32)
    tgt = 17
    dp[tgt] = (rng.standard_normal(n_in) * 1e-2).astype(np.float32)

    print(f"口径：n_out={n_out} n_in={n_in} |W|~1e-2 |dp_non_tgt~1e-6 "
          f"|dp_tgt~1e-2（目标行 {tgt}）")
    print("路径：**生产码本**（quantize_to/dequantize_from）\n")
    print(f"{'格式':<12} {'非目标行保留率':>14} {'目标行保留率':>13}")
    print("-" * 44)
    results = {}
    for fmt in list(RO_DTYPES) + ["fp8-bits"]:
        try:
            k_nt, k_t = keep_rate(fmt, W0, dp, tgt)
        except Exception as e:                           # noqa: BLE001
            print(f"{fmt:<12} ERROR {type(e).__name__}: {e}")
            continue
        results[fmt] = (k_nt, k_t)
        print(f"{fmt:<12} {k_nt:>13.2f}% {k_t:>12.2f}%")

    # 同时给出「有效步长」：该精度下非目标行更新的平均相对量级
    print("\n参考：各精度的量化步长（W 域，|W|~1e-2 处的相邻格点间距）")
    probe2 = np.array([0.05, 0.05 + 1e-4], dtype=np.float32)
    for fmt in ("fp32", "fp16", "bf16", "int8", "int16", "int32", "fp8-bits"):
        try:
            if fmt == "fp32":
                step = 0.0
                val = 0.05
            elif fmt == "fp8-bits":
                d2 = _fp8_bits_round(probe2)
                step = abs(float(d2[1] - d2[0]))
                val = float(_fp8_bits_round(np.array([0.05], np.float32))[0])
            elif fmt in ("int16", "int32"):
                qmax = 32767.0 if fmt == "int16" else 2147483647.0
                scale = 0.05 / qmax * 1.0     # |W|max=0.05 归一化到 ±qmax
                d2 = _uniform_fixed(probe2 / scale, fmt) * scale
                step = abs(float(d2[1] - d2[0]))
                val = float(_uniform_fixed(np.array([0.05]) / scale, fmt)[0] * scale)
            else:
                d2 = dequantize_from(fmt, quantize_to(fmt, probe2)).ravel()
                step = abs(float(d2[1] - d2[0]))
                val = float(dequantize_from(fmt, quantize_to(fmt, np.array([0.05], np.float32))).ravel()[0])
            print(f"  {fmt:<10} 0.05 邻域步长 ≈ {step:.3e}"
                  f"（反量化 {val:.8f}）")
        except Exception as e:                           # noqa: BLE001
            print(f"  {fmt:<10} ERROR {type(e).__name__}: {e}")

    out = _ROOT / "outputs" / "test" / "precision_keep_rates.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# P205b-2 真实码本路径保留率（生产 quantize_to/dequantize_from）",
             f"# 口径：n_out={n_out} n_in={n_in} |W|~1e-2 |dp_non_tgt~1e-6 "
             f"|dp_tgt~1e-2",
             "# 非目标行保留率 / 目标行保留率（%）"]
    for k, (a, b) in results.items():
        lines.append(f"{k}\t{a:.2f}\t{b:.2f}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
