#!/usr/bin/env python3
"""门禁：M1 GEMV 路径（P165）—— 探测存在、ILP 核数值正确、默认不退化。

P165 改了 `sparse_encoder.encode` 的分派：
- 判据从「平台名硬编码」换成**运行时探测** `_prefer_blas_gemv`；
- 新增 ILP 核 `_gemv_rows_ilp`（行内 4 路累加器）。

本门禁盯三件事：
  A. 探测函数存在且**只探测一次**（有缓存），返回 bool；
  B. ILP 核与原核在 fp32 下**逐位一致**、fp64 下容差在 1-2 ulp（与 P52 同级）；
  C. `encode` 端到端可跑，且**不再依赖平台名判断**（源码里没有
     `.startswith("aarch64")` 这类硬编码）。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

for _p in (str(Path(__file__).resolve().parents[2]),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _RESULTS.append((bool(ok), name, detail))
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name,
                         ("  —— " + detail) if detail else ""))
    return bool(ok)


print("=" * 70)
print("门禁：M1 GEMV 路径（P165）")
print("=" * 70)

from phdnet import sparse_encoder as SE  # noqa: E402

# ── A. 探测函数 ────────────────────────────────────────────────────────
print("[A] 运行时探测")
check("A1 _prefer_blas_gemv 存在",
      callable(getattr(SE, "_prefer_blas_gemv", None)))
_cache = getattr(SE, "_BLAS_GEMV_OK", None)
check("A2 有进程级缓存（避免每步探测）",
      isinstance(_cache, dict), type(_cache).__name__)

for dt in (np.float32, np.float64):
    try:
        r1 = SE._prefer_blas_gemv(np.dtype(dt))
        n_before = len(_cache)
        r2 = SE._prefer_blas_gemv(np.dtype(dt))
        n_after = len(_cache)
        check("A3 %s 探测返回 bool 且**幂等**（第二次不再新增缓存项）"
              % np.dtype(dt).name,
              isinstance(r1, bool) and r1 == r2 and n_after == n_before,
              "第1次=%s第2次=%s 缓存 %d→%d" % (r1, r2, n_before, n_after))
    except Exception as exc:                                # noqa: BLE001
        check("A3 %s 探测" % np.dtype(dt).name, False,
              "%s: %s" % (type(exc).__name__, str(exc)[:50]))

# ── B. ILP 核的数值正确性 ─────────────────────────────────────────────
print("[B] ILP 核数值契约")
check("B1 _gemv_rows_ilp 存在", hasattr(SE, "_gemv_rows_ilp"))
check("B2 ILP 开关存在且默认开", getattr(SE, "_GEMV_ILP", None) is True,
      "PHD_GEMV_ILP 可关（回到逐位路径）")

rng = np.random.default_rng(7)
n_sdr, n_in = 256, 512
for dt, tag, tol in ((np.float32, "fp32", 0.0),
                     (np.float64, "fp64", 1e-14)):
    W = rng.normal(0, 0.05, (n_sdr, n_in)).astype(dt)
    x = rng.normal(0, 1, n_in).astype(dt)
    b = rng.normal(0, 0.01, n_sdr).astype(dt)
    o1 = np.empty(n_sdr, dtype=dt)
    o2 = np.empty(n_sdr, dtype=dt)
    SE._gemv_rows(W, x, b, o1)
    SE._gemv_rows_ilp(W, x, b, o2)
    d = float(np.abs(o1.astype(np.float64) - o2.astype(np.float64)).max())
    scale = max(1e-30, float(np.abs(o1).max()))
    rel = d / scale
    if tol == 0.0:
        check("B3 %s ILP 与原核**逐位一致**（P165 实测 max|Δ|=0）" % tag,
              d == 0.0, "max|Δ|=%.3e" % d)
    else:
        check("B4 %s ILP 相对误差 ≤ %.0e（1-2 ulp，与 P52 同级）" % (tag, tol),
              rel <= tol, "relerr=%.3e" % rel)
    # 与 numpy 参考对照（两条路径都该正确）
    ref = W @ x + b
    e1 = float(np.abs(o1.astype(np.float64) - ref.astype(np.float64)).max()
               / max(1e-30, float(np.abs(ref).max())))
    # ⚠ 容差必须**按 dtype**：fp32 eps ≈ 1.19e-7（用 1e-12 会永远失败）；
    #   fp64 eps ≈ 2.2e-16，但**归约顺序不同**（核按列序、BLAS 分块），
    #   k=512 的累积误差实测约 4 eps → 容差取 16 eps。
    eps = float(np.finfo(dt).eps)
    tol = 16 * eps
    check("B5 %s 原核与 numpy 参考一致（容差 16x eps = %.1e）" % (tag, tol),
          e1 <= tol, "relerr=%.3e（eps=%.2e）" % (e1, eps))

# ── C. 分派不再用平台名硬编码 ──────────────────────────────────────────
print("[C] 分派逻辑")
src = Path(SE.__file__).read_text(encoding="utf-8")
enc_body = src.split("def encode")[1].split("\n    def learn")[0] \
    if "def encode" in src else ""
# ⚠ 必须**剥掉注释**再看：P77 的历史说明里会出现 aarch64 字样，那不是代码。
_code = "\n".join(l.split("#", 1)[0] for l in enc_body.split("\n"))
check("C1 encode 代码里不再有 aarch64 硬编码（注释不算）",
      "aarch64" not in _code,
      "P165：改用运行时探测（BLAS 快慢不是架构的必然属性）")
check("C2 encode 里调用 _prefer_blas_gemv",
      "_prefer_blas_gemv" in enc_body)
check("C3 encode 里调用 ILP 核",
      "_gemv_rows_ilp" in enc_body)

# ── D. 端到端 ────────────────────────────────────────────────────────
print("[D] 端到端 encode")
for dt in ("fp32", "fp64"):
    try:
        enc = SE.SparseEncoder(1024, 512, 64, np.random.default_rng(0),
                               dtype=dt)
        x = np.random.default_rng(1).normal(0, 1, 1024).astype(
            np.float64 if dt == "fp64" else np.float32)
        s, idx = enc.encode(x)
        check("D1 %s encode 可跑（s 与 idx 形状/有限性）" % dt,
              s.shape == (512,) and idx.shape == (64,)
              and bool(np.isfinite(s).all())
              and len(set(idx.tolist())) == 64,
              "s%s idx%s 唯一胜者=%d" % (s.shape, idx.shape, len(set(idx.tolist()))))
    except Exception as exc:                                # noqa: BLE001
        check("D1 %s encode 可跑" % dt, False,
              "%s: %s" % (type(exc).__name__, str(exc)[:60]))

n_fail = sum(1 for ok, _, _ in _RESULTS if not ok)
print()
print("=" * 70)
print("结果：%d 例 FAIL%s" % (n_fail,
                            (" → " + str([n for ok, n, _ in _RESULTS if not ok]))
                            if n_fail else ""))
print("=" * 70)
sys.exit(1 if n_fail else 0)