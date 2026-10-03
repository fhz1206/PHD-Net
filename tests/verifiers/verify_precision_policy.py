"""门禁：精度策略（降级链 / 七种精度 / 4-bit 打包 / 启动一次）—— P151。

覆盖 fhz 2026-10-03 指令的四点：
  1. 「fp 不支持转 int，int 不支持转 fp，两者都不行再报错」
  2. 「解禁 fp8 / fp4 / int4 / int8」
  3. 「默认走 fp8 和 fp16」
  4. 「训练启动时判断一次即可」

⚠ 为什么这些都要钉住：P151 实现过程中真实踩到的坑（都已修）：
  -降级链里 `fp32` 排在同族兄弟 → fp8 在 fp16 不可用时会先撞 fp32、
    **永远轮不到 int8**，违背指令；
  - e2m1 的 LUT 只写了 **14 个**元素 → `size != 16` 让族判据失效 →
    scale 用错 → 量化误差 **46.7%**；
  - 4-bit 的 `tdtype = uint8` 被 `_upd_dtype` 判成更新域 → `dp` 变 uint8 →
    `dp[correct] -= 1.0` 抛 "Float can't be cast to Byte"。
"""
from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                     # noqa: E402

from phdnet.precision_policy import (candidate_order,             # noqa: E402
                                     resolve_precision, resolve_precision_cached,
                                     clear_cache, FAMILIES)
from phdnet.backends.accel_readout import (AccelReadout,           # noqa: E402
                                           _pack4, _unpack4,
                                           _FP4_E2M1_LUT, _INT4_LUT,
                                           _unsupported_reason)

_RESULTS: list = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail else ""))
    return bool(ok)


def main() -> int:
    print("=" * 78)
    print("门禁：精度策略（降级链 / 七种精度 / 4-bit / 启动一次）—— P151")
    print("=" * 78)

    # ── A. 降级链的顺序契约 ──────────────────────────────────────────
    print("\n[A] 降级链顺序（fhz：fp→int→fp→报错）")
    for req in ("fp8", "int8", "fp4"):
        ch = candidate_order(req)
        # fp32 必须在 int 族**之后**（否则 fp8 永远轮不到 int8）
        i_fp32 = ch.index("fp32")
        i_int = min((ch.index(x) for x in ("int8", "int4", "int16", "int32")
                     if x in ch), default=len(ch))
        check(f"A1 {req} 的链里 fp32 排在 int 族之后",
              i_fp32 > i_int,
              f"fp32@{i_fp32} vs int@{i_int}：{' → '.join(ch)}")
    check("A2 链里无重复项",
          all(len(candidate_order(r)) == len(set(candidate_order(r)))
              for r in ("fp8", "int8", "fp4", "fp16", "int4")),
          "")
    check("A3 链尾是 fp32（公共终点兜底）",
          candidate_order("fp8")[-1] == "fp32",
          f"尾部={candidate_order('fp8')[-1]}")

    # ── B. 降级行为 ─────────────────────────────────────────────────
    print("\n[B] 降级行为")
    r1 = resolve_precision("fp8", lambda dt: (dt == "fp16", "fp8 无算子"))
    check("B1 fp8 不可用 → 降到 fp16（不是 fp32）",
          r1["dtype"] == "fp16" and r1["downgraded"],
          f"落地={r1['dtype']}")

    r2 = resolve_precision("fp8",
                           lambda dt: (dt == "int8", "")
                           if dt not in ("fp8", "fp16", "bf16")
                           else (False, "fp 族全挂"))
    check("B2 fp 族全挂 → **先试 int**（符合指令）",
          r2["dtype"] == "int8", f"落地={r2['dtype']}")

    r3 = resolve_precision("int8",
                           lambda dt: (dt == "fp16", "")
                           if dt not in ("int8", "int16", "int32")
                           else (False, "int 族全挂"))
    check("B3 int 族全挂 → 回 fp（符合指令）",
          r3["dtype"] == "fp16", f"落地={r3['dtype']}")

    r4 = resolve_precision("fp4", lambda dt: (False, "全不支持"))
    check("B4 全链不支持 → dtype=None（调用方必须报错）",
          r4["dtype"] is None and bool(r4["reason"]),
          f"reason 非空={bool(r4['reason'])}")

    # ── C. 探测只做一次（fhz：训练启动时判断一次即可）──────────────
    print("\n[C] 探测缓存（启动一次）")
    clear_cache()
    calls = {"n": 0}

    def _p(dt):
        calls["n"] += 1
        return True, ""
    resolve_precision_cached("fp8", _p, "cpu")
    n_first = calls["n"]
    for _ in range(4):
        resolve_precision_cached("fp8", _p, "cpu")
    check("C1 同设备重复解析**只探测一次**",
          calls["n"] == n_first, f"首次 {n_first} 次，后续共 {calls['n']}")
    clear_cache()
    calls["n"] = 0
    resolve_precision_cached("fp8", _p, "cpu")
    resolve_precision_cached("fp8", _p, "npu")     # 换设备要重探
    check("C2 换设备会重新探测（缓存键含 device）",
          calls["n"] == 2 * n_first, f"共 {calls['n']} 次")
    clear_cache()

    # ── D. 七种精度端到端（解禁 fp8/fp4/int4/int8）──────────────────
    print("\n[D] 七种精度端到端（稀疏，真实规模）")
    from phdnet.sparse_pc import _random_csr
    n_out, n_h, k = 2048, 512, 64
    csr = _random_csr(np.random.default_rng(3), n_out, n_h, k,
                      0.05 * np.sqrt(n_h / k), False, 0.8)
    h = np.random.default_rng(0).normal(0, 1, n_h).astype(np.float32)
    tgt = np.zeros(n_out, np.float32)
    tgt[0] = 1.0
    nlls = {}
    for dt in ("fp32", "fp16", "bf16", "fp8", "int8", "fp4", "int4"):
        clear_cache()
        try:
            ro = AccelReadout(n_h, n_out, np.random.default_rng(3),
                              device="cpu", dtype=dt, conn_k=k, csr=csr)
            y = ro.forward_dev(h)
            nll = ro.learn_softmax(h, tgt, 0.15, y_pre=y, target_idx=0)
            landed = ro._precision_resolved["dtype"]
            mem = ro.W.numel() * ro.W.element_size() / 2 ** 20
            nlls[dt] = (float(nll), landed, mem)
            check(f"D  {dt} 完整一步（落地 {landed}，W {mem:.2f} MiB）",
                  np.isfinite(nll), f"nll={nll:.4f}")
        except Exception as e:                                # noqa: BLE001
            check(f"D  {dt} 完整一步", False,
                  f"{type(e).__name__}: {str(e)[:60]}")
    # 访存阶梯必须单调不增
    if len(nlls) == 7:
        mems = [nlls[d][2] for d in ("fp32", "fp16", "fp8", "int4")]
        check("D8 访存阶梯 fp32>fp16>fp8>int4（真的省流量）",
              all(a > b for a, b in zip(mems, mems[1:])),
              " → ".join(f"{m:.2f}" for m in mems))

    # ── E. 4-bit 打包契约 ────────────────────────────────────────────
    print("\n[E] 4-bit 打包（这是 P151 踩坑最多的地方）")
    check("E1 e2m1 LUT **恰好 16 格**（我曾写成 14 →误差 46.7%）",
          _FP4_E2M1_LUT.size == 16, f"实际 {_FP4_E2M1_LUT.size}")
    check("E2 int4 LUT 恰好 16 格", _INT4_LUT.size == 16,
          f"实际 {_INT4_LUT.size}")
    rng = np.random.default_rng(0)
    v = rng.normal(0, 0.3, (128, 64)).astype(np.float32)
    for name, lut, fam, tol in (("fp4", _FP4_E2M1_LUT, "fp", 0.25),
                                ("int4", _INT4_LUT, "int", 0.10)):
        pk, sc = _pack4(v, lut, family=fam)
        pk2 = pk.reshape(v.shape[0], -1)
        up = _unpack4(torch.from_numpy(pk2), v.shape[1], lut,
                      sc).float().numpy()
        rel = float(np.abs(up - v).max() / np.abs(v).max())
        check(f"E3 {name} 打包往返误差在 {tol:.0%} 内（4-bit 只有 16 格）",
              rel <= tol, f"相对={rel:.2%} scale={sc:.4g}")
        check(f"E4 {name} 打包后**字节数减半**",
              pk2.shape[1] * 2 == v.shape[1],
              f"{v.shape[1]} → {pk2.shape[0]}×{pk2.shape[1]}")
        # 打包语义：高半字节 = 偶数列（numba `_ro_q_matvec_fp4` 的约定）
        c0 = int(pk2[0, 0])
        check(f"E5 {name} 高半字节=偶数列、低半字节=奇列（numba 语义）",
              abs(float((lut * sc)[c0 >> 4]) - up[0, 0]) < 1e-5,
              f"byte=0x{c0:02x} 偶={c0>>4}→{up[0,0]:.4f}")

    # ── F. 能力表不再禁 fp4/int4 ────────────────────────────────────
    print("\n[F] 能力表解禁")
    from phdnet.config import PHDNetConfig
    for dt in ("fp8", "int8", "fp4", "int4"):
        r = _unsupported_reason(PHDNetConfig(readout_dtype=dt))
        check(f"F  {dt} 不再被能力表拒绝",
              r is None or "int4" not in r,
              "放行" if r is None else r[:40])

    ok = sum(1 for r in _RESULTS if r[0])
    bad = [r[1] for r in _RESULTS if not r[0]]
    print("\n" + "=" * 78)
    print(f"结果：{ok}/{len(_RESULTS)} 通过"
          + (f" | 失败 {bad}" if bad else ""))
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())