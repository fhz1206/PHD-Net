"""P124：M6 幂律异质连接分配的接入门禁。

背景
================================================================================
P108 起M6 读出已稀疏化（`readout_conn_k`，均匀 k-conn），P111 起在加速器上
实现。但**均匀 k 有个结构问题**：高频词（如"的"）与长尾词获得**同等容量**——
长尾被过分配、高频被欠分配。`phdnet/sparse_alloc.py` 实现了幂律分配
（k_i ∝ counts_i^alpha），但**长期未接入 readout**（其 docstring 明写
「未接入」）。P124 完成接线。

生物学依据：突触巩固/修剪（synaptic consolidation & pruning）——
"use it or lose it" 在**连接数**这一结构层面的表达。

四层验证
================================================================================
A. **零回归**：`alpha=0`（默认）必须与 P108 的均匀 k-conn **逐位一致**
   （idx / val / indptr 全等）—— 这是「默认关闭」铁律的兑现。
B. 分配正确性
   B1 alpha 越大 → 行宽倾斜越强（秩相关单调增）
   B2 **总 nnz 受预算控制、不膨胀**（这是幂律分配存在的核心原因）
   B3 行宽在 [k_min, k_max] 内，且 **>= 1**（读出行不允许 0 条入边）
   B4 列索引行内**升序且无重复**（`_csr_matvec` 的隐含契约）
C. 前向/学习语义
   C1 前向输出形状不变、无 NaN
   C2 学习后权重确实变了（幂律臂与均匀臂都试）
   C3 与 numba 参考实现（同CSR）在容差内一致
D. **配置与 CLI 默认值一致**（默认 alpha=0 = 关闭），且**加速器路径也放行**
   —— ⚠ 加速器的 `AccelReadout` 对**非均匀行宽 fail-fast**（P111 设计），
   所以 alpha>0 时 accel 会回落 numba。这是**已知且正确的**行为
   （P19 纪律：不能「能跑但语义不同」），但必须被记录与验证。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in ("", "tests", "tools"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

_RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    return bool(ok)


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """秩相关（不依赖 scipy——本环境没有）。"""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _zipf(n: int, exponent: float = 0.9) -> np.ndarray:
    return 1.0 / (np.arange(1, n + 1, dtype=np.float64) ** exponent)


# ── A. alpha=0 逐位零回归 ──────────────────────────────────────────────────
def section_a() -> None:
    print("\n[A] 零回归：alpha=0 与 P108 均匀 k-conn **逐位一致**")
    from phdnet.readout import Readout
    n_h, n_out, k = 96, 160, 8
    a = Readout(n_h, n_out, np.random.default_rng(7), conn_k=k,
                lognormal_init=False)
    b = Readout(n_h, n_out, np.random.default_rng(7), conn_k=k,
                lognormal_init=False, powlaw_alpha=0.0)
    ipa, idxa, vala = a._csr                                   # noqa: SLF001
    ipb, idxb, valb = b._csr                                   # noqa: SLF001
    check(np.array_equal(ipa, ipb), "A1 indptr 逐位相同",
          f"nnz={idxa.size}")
    check(np.array_equal(idxa, idxb), "A2 idx 逐位相同")
    check(np.array_equal(vala, valb), "A3 val 逐位相同")
    check(np.array_equal(np.diff(ipa), np.diff(ipb))
          and (np.diff(ipa) == k).all(),
          "A4 行宽全为 k（均匀）", f"min={np.diff(ipa).min()} max={np.diff(ipa).max()}")


# ── B. 分配正确性 ───────────────────────────────────────────────────────────
def section_b() -> None:
    print("\n[B] 幂律分配正确性")
    from phdnet.readout import Readout
    n_h, n_out, k = 96, 200, 8
    counts = _zipf(n_out)
    budget = k * n_out
    rhos = []
    for alpha in (0.0, 0.1, 0.25, 0.5):
        ro = Readout(n_h, n_out, np.random.default_rng(7), conn_k=k,
                     lognormal_init=False, powlaw_alpha=alpha,
                     powlaw_counts=counts)
        ip = ro._csr[0]                                       # noqa: SLF001
        kw = np.diff(ip)
        rhos.append(_rank_corr(counts, kw))
        check(kw.min() >= 1, f"B3.{alpha} 每行 >= 1 条入边（不允许 0）",
              f"min={kw.min()}")
        check(kw.sum() <= budget,
              f"B2.{alpha} 总 nnz 受预算控制（不膨胀）",
              f"nnz={kw.sum()} ≤ 预算 {budget}")
        # B4 列索引行内升序 + 无重复（_csr_matvec 的隐含契约）
        idx = ro._csr[1]                                      # noqa: SLF001
        ok_sorted = True
        for r in range(n_out):
            cols = idx[ip[r]:ip[r + 1]]
            if cols.size > 1 and np.any(np.diff(cols) <= 0):
                ok_sorted = False
                break
        check(ok_sorted, f"B4.{alpha} 列索引行内升序且无重复")
    # B1 单调性：alpha 越大秩相关越强（0 后应递增）
    tail = [r for r in rhos[1:] if not np.isnan(r)]
    check(len(tail) >= 2 and all(tail[i] <= tail[i + 1] + 1e-9
                                 for i in range(len(tail) - 1)),
          "B1 秩相关随 alpha 单调不减（倾斜增强）",
          " → ".join(f"{r:+.4f}" for r in rhos))
    # alpha>0 时确实不等宽
    ro = Readout(n_h, n_out, np.random.default_rng(7), conn_k=k,
                 lognormal_init=False, powlaw_alpha=0.25,
                 powlaw_counts=counts)
    kw = np.diff(ro._csr[0])                                 # noqa: SLF001
    check(kw.min() != kw.max(), "B5 alpha>0 → 行宽不等（异质连接）",
          f"min={kw.min()} max={kw.max()} mean={kw.mean():.2f}")
    check(_rank_corr(counts, kw) > 0.3,
          "B6 k 与词频正相关（高频词入边更多 = rich-get-richer）",
          f"rho={_rank_corr(counts, kw):+.4f}")


# ── C. 前向 / 学习语义 ──────────────────────────────────────────────────────
def section_c() -> None:
    print("\n[C] 前向与学习语义")
    from phdnet.readout import Readout
    n_h, n_out, k = 96, 120, 8
    counts = _zipf(n_out)
    rng = np.random.default_rng(11)
    h = rng.normal(0, 1, n_h).astype(np.float32)
    tgt = rng.normal(0, 1, n_out).astype(np.float32)

    for alpha in (0.0, 0.25):
        ro = Readout(n_h, n_out, np.random.default_rng(7), conn_k=k,
                     lognormal_init=False, powlaw_alpha=alpha,
                     powlaw_counts=counts)
        y = np.asarray(ro(h))                                # ← learn 之前取 y
        ok_shape = y.shape == (n_out,)
        check(ok_shape and np.isfinite(y).all(),
              f"C1.{alpha} 前向形状正确且无 NaN",
              f"shape={y.shape} |y|max={np.abs(y).max():.4f}")
        w0 = np.array(ro._csr[2], copy=True)                 # noqa: SLF001
        # C3 与「直接按 CSR 手算」一致 —— 证明前向用了真权重。
        # ⚠ 必须用 **learn 之前**的权重来手算：第一版把这段放在 learn 之后，
        # 而 learn 会改 `_csr[2]` → 手算用的是新权重、y 是旧权重的结果，
        # 报出 max|Δ|=6.7 的**假失败**。（第一版就踩了，与 P122 的 verifier
        # 假通过是同一个教训：比对的两端必须取自同一时刻。）
        ip, idx, val = ro._csr                                # noqa: SLF001
        ref = np.array([val[ip[i]:ip[i + 1]] @ h[idx[ip[i]:ip[i + 1]]]
                        for i in range(n_out)])
        d = float(np.abs(y - ref).max())
        check(d <= 1e-5 * max(1.0, float(np.abs(ref).max())),
              f"C3.{alpha} 前向与 CSR 手算一致（权重真被用上）",
              f"max|Δ|={d:.3e}")
        # C2 学习后权重确实变化（放在 C3 之后，此时已验证前向语义正确）
        ro.learn(h, tgt, 0.15)
        w1 = ro._csr[2]                                       # noqa: SLF001
        changed = not np.array_equal(np.sort(w0), np.sort(w1))
        check(changed, f"C2.{alpha} 学习后权重确实变化")


# ── D. 配置与 CLI ───────────────────────────────────────────────────────────
def section_d() -> None:
    print("\n[D] 配置/CLI 默认值与加速器路径")
    from phdnet.config import PHDNetConfig
    c = PHDNetConfig()
    check(float(c.readout_powlaw_alpha) == 0.0,
          "D1 config.readout_powlaw_alpha 默认 0.0（关闭）", f"实际={c.readout_powlaw_alpha}")
    check(int(c.readout_powlaw_kmin) == 1, "D2 config kmin 默认 1")
    check(int(c.readout_powlaw_kmax) == 0, "D3 config kmax 默认 0（=不设上限）")

    src = (_ROOT / "train" / "train.py").read_text(encoding="utf-8").replace("\n", " ")
    check('"--readout-powlaw-alpha", type=float, default=0.0' in src,
          "D4 CLI --readout-powlaw-alpha 默认 0.0")
    check('"--readout-powlaw-kmin"' in src and '"--readout-powlaw-kmax"' in src,
          "D5 CLI kmin/kmax 均存在")

    # D6 加速器对非均匀行宽 fail-fast（P111 设计）—— 记录为**预期行为**
    from phdnet.backends.accel_readout import _unsupported_reason
    reason_u = _unsupported_reason(PHDNetConfig(readout_conn_k=128,
                                                 readout_powlaw_alpha=0.25))
    reason_d = _unsupported_reason(PHDNetConfig(readout_conn_k=128,
                                                 readout_powlaw_alpha=0.0))
    check(reason_d is None or "conn_k" not in str(reason_d),
          "D6 均匀 k 时加速器照常放行（P111 路径未受影响）", f"reason={reason_d}")
    check(reason_u is not None and "行宽不等" in str(reason_u),
          "D7 幂律（行宽不等）时加速器**提前 fail-fast** 并说明原因",
          f"reason={str(reason_u)[:76]}")


def main() -> int:
    print("=" * 78)
    print("P124 门禁：M6 幂律异质连接接入 readout")
    print("=" * 78)
    section_a()
    section_b()
    section_c()
    section_d()
    npass = sum(1 for ok, _, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print("\n" + "=" * 78)
    print(f"结果：{npass}/{total} 通过 | 失败 {total - npass}")
    if npass != total:
        print("失败用例：")
        for ok, name, detail in _RESULTS:
            if not ok:
                print(f"  · {name}  {detail}")
    print("=" * 78)
    return 0 if npass == total else 1


if __name__ == "__main__":
    sys.exit(main())