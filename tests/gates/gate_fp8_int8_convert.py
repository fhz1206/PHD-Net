"""P212 门禁：fp8 → int8 转换的正确性与查表等价性。

跑法：py tests/gates/gate_fp8_int8_convert.py

⚠ **本门禁的存在理由**：P212 审计发现 `phdnet/backends/fp8_int8_convert.py`
的 docstring 声称「**逐位对拍由 `tests/gates/verify_fp8_int8_convert.py`
把关**」—— 但**该文件从来就不存在**（P154/P151/P123 都反复引用它）。
即：这条保护链是**断的**，而代码里恰好藏着两个真实缺陷（本门禁现已覆盖）：

  ① `_E4M3_POW2` 查表**定义了却从未被索引** —— 核内一直是
     `2.0 ** (exp - _E4M3_BIAS)`（编译成 `powf` 调用）。
     注释宣称「查表 105.4 ms vs powf 120.6 ms」，实际跑的是 powf 那档。
     → P212 接线查表（**数值逐位相同**，见 A2）。
  ② 「转化效率」需求（P147 指令「记得提供转化效率，转化放在 cpu 上」）
     只存在于零 import 的孤立模块 `fp8_capability.py` → 从未兑现。
     → P212 改由 `fp8_to_int8_codes(report=...)` 兑现。

覆盖组：
  A  查表等价性与转换正确性
  B  转化效率报告（P147 指令兑现）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.backends.fp8_int8_convert import (            # noqa: E402
    _E4M3_BIAS, _E4M3_POW2, _e4m3_bits_to_f64,
    fp8_bits_to_int8_torch, fp8_to_int8_codes,
    fp8_to_int8_codes_nogil)

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))


def _make_bits(n: int, seed: int = 0) -> np.ndarray:
    """随机 fp8 e4m3fn 位模式（避开 NaN 槽位 0x7F / 0xFF）。"""
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 256, size=n, dtype=np.uint8)
    return np.where(np.isin(b, [0x7F, 0xFF]), 0x00, b).astype(np.uint8)


def group_a() -> None:
    print("\n[A] 查表等价性与转换正确性")

    # A1：查表与 powf 逐位相同（所有值都是 2 的幂 → 精确可表示）
    exp = np.arange(16)
    powf = 2.0 ** (exp - _E4M3_BIAS)
    table = _E4M3_POW2[exp]
    check(np.array_equal(powf, table),
          "A1a 查表值与 `2.0**(exp-7)` 逐位相同（偏差 0.0）",
          f"max|Δ|={float(np.abs(powf - table).max()):.3e}")

    # A2：核内确实走查表（源码层面确认不再有**可执行**的变量指数 `**`）
    src = (_ROOT / "phdnet" / "backends" / "fp8_int8_convert.py").read_text(
        encoding="utf-8")
    # ⚠ 只查**代码行**（排除注释行）：注释里提到旧写法是正常的说明性文字；
    #   numpy 参考路径 `_e4m3_bits_to_f64` 用的是 `np.ldexp`（向量化，无 powf）。
    _code_lines = [ln for ln in src.splitlines()
                   if "2.0 ** (exp - _E4M3_BIAS)" in ln
                   and not ln.lstrip().startswith("#")]
    check(not _code_lines,
          "A2a 核内不再有变量指数的 `**`（powf）",
          f"残留 {len(_code_lines)} 处")
    check("_E4M3_POW2[exp]" in src,
          "A2b 核内确实索引了 `_E4M3_POW2` 查表")
    # A2c：numpy 参考路径用 ldexp（同样避开 powf）
    check("np.ldexp" in src,
          "A2c numpy 参考路径用 np.ldexp（向量化，无 powf）")

    # A3：numpy 参考实现与两条转换路径逐位一致
    bits = _make_bits(200_000, seed=3)
    ref = _e4m3_bits_to_f64(bits)
    amax = float(np.abs(ref).max())
    scale = (2.0 * amax / 127.0) if amax > 0 else 1.0
    want = np.clip(np.rint(ref / scale), -127, 127).astype(np.int8)

    c_nogil, s_nogil = fp8_to_int8_codes_nogil(bits)
    c_torch, s_torch = fp8_bits_to_int8_torch(bits)
    check(np.array_equal(c_nogil, want),
          "A3a nogil 路径与 numpy 参考逐位一致",
          f"scale={s_nogil:.6g} vs {scale:.6g}")
    check(np.array_equal(c_torch, want),
          "A3b torch 路径与 numpy 参考逐位一致")
    check(s_nogil == s_torch,
          "A3c 两条路径的 per-tensor scale 相同")

    # A4：边缘位模式（0 / ±0 / subnormal / 最小规格化 / 最大值）
    edge = np.array([0x00, 0x80, 0x01, 0x81, 0x08, 0x88, 0x7E, 0xFE],
                     dtype=np.uint8)
    c_e, s_e = fp8_to_int8_codes(edge, conv="nogil")
    c_et, s_et = fp8_bits_to_int8_torch(edge)
    check(np.array_equal(c_e, c_et),
          "A4 边缘位模式（含 subnormal / 符号位）两条路径一致",
          f"codes={c_e.tolist()}")

    # A5：NaN 槽位被跳过（不污染 amax，输出有限）
    with_nan = np.array([0x7F, 0xFF, 0x00, 0x08], dtype=np.uint8)
    c_n, s_n = fp8_to_int8_codes(with_nan, conv="nogil")
    check(np.isfinite(c_n).all() and c_n[0] == 0 and c_n[1] == 0
          and c_n[2] == 0 and c_n[3] != 0,
          "A5 NaN 槽位被跳过（置 0），正常位模式仍正确量化",
          f"codes={c_n.tolist()} scale={s_n:.6g}")


def group_b() -> None:
    print("\n[B] 转化效率报告（P147 指令兑现）")

    bits = _make_bits(500_000, seed=5)

    rep: dict = {}
    fp8_to_int8_codes(bits, conv="nogil", report=rep)
    need = ("conv", "ms", "MiB_in", "MiB_out", "MiB_per_s",
            "ratio_vs_fp32_copy")
    missing = [k for k in need if k not in rep]
    check(not missing, "B1 report dict 含全部约定字段", f"缺={missing}")
    check(rep.get("conv") == "nogil", "B2a report 记录实际路径",
          str(rep.get("conv")))
    check(rep.get("ms", 0) > 0, "B2b report 记录耗时 > 0",
          f"{rep.get('ms', 0):.3f} ms")
    check(rep.get("MiB_in", 0) > 0 and rep.get("MiB_out", 0) > 0,
          "B2c report 记录输入/输出字节数",
          f"{rep.get('MiB_in', 0):.2f} → {rep.get('MiB_out', 0):.2f} MiB")
    check(rep.get("MiB_per_s", 0) > 0, "B2d report 记录吞吐 > 0",
          f"{rep.get('MiB_per_s', 0):.1f} MiB/s")
    check(np.isfinite(rep.get("ratio_vs_fp32_copy", np.nan)),
          "B2e report 的「相对 fp32 纯拷贝」倍数有限")

    # B3：不传 report 时不应报错（向后兼容）
    try:
        c, _ = fp8_to_int8_codes(bits, conv="nogil")
        ok = c.shape == bits.shape
    except Exception as e:                                   # noqa: BLE001
        ok = False
        print(f"    异常: {e}")
    check(ok, "B3 不传 report 时调用正常（向后兼容）")

    # B4：torch 路径同样能出报告
    rep2: dict = {}
    fp8_to_int8_codes(bits, conv="torch", report=rep2)
    check(rep2.get("conv") == "torch" and rep2.get("ms", 0) > 0,
          "B4 torch 路径也能出报告",
          f"{rep2.get('ms', 0):.3f} ms")


def main() -> None:
    print("=" * 68)
    print("P212 门禁：fp8 → int8 转换（查表等价性 + 转化效率报告）")
    print("=" * 68)
    group_a()
    group_b()
    n_pass = sum(1 for ok, _, _ in _results if ok)
    n_all = len(_results)
    print("-" * 68)
    print(f"结果：{n_pass}/{n_all} 通过 | 失败 {n_all - n_pass}")
    print("=" * 68)
    if n_pass != n_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
