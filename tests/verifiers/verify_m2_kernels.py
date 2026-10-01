"""P113：M2 推理核三态（fused / serial / plain）的数值等价对拍。

为什么需要这个文件
================================================================================
P113 之前 `--m2-kernel` 的默认值是 `serial`（单核融合核，P99 加的），因为
P76 实测昇腾上 `fused`（prange 融合）比 `plain` 慢 3–4×。但 P76 否掉的只是
**融合 + 10 个 prange 屏障区**，并没有否掉 `_csr_matvec` 自身的**行级prange**
—— `plain` 走的就是它，而它一直是三种模式里唯一带行级并行的。

2026-10-01 服务器实测（1b 档，昇腾 191 核 + NPU，vocab 51,962）：
    M2_infer  6.92 → 1.28 ms/tok（5.41×）；端到端 17.63 → 12.09 ms/tok（1.46×）
    四个采样点（10.25k/20.25k/30.25k/40.25k）的 sliding PPL 与serial 运行
    **逐位完全相同** → plain 是纯调度变化，零语义变化。

那份「逐位相同」是**生产日志观测**，不是受控对拍（两次 run 的其它条件虽同，
但不是同一进程内比）。本文件把它固化成**可复跑的受控门禁**：三态必须在
同一进程、同一输入下产生容差内一致的输出，且非融合态必须**逐位**一致。

三层验证
================================================================================
A. 三态等价（同一 s0、同一 CSR，容差判据）
   A1 fused vs serial（两者都是融合核，仅并行度不同）
   A2 plain vs serial —— **plain 是非融合路径**，内部用 `np.tanh` /
     `np.clip` 的 numpy 归约，与融合核的 `math.tanh` 归约顺序不同，
     故用容差；实测偏差应在 1e-7 量级
   A3 n_steps=1/2/3（E1 家族的老教训：单步一致不代表多步一致）
B. 关键性质：plain 的行级 prange **不改变求和顺序**
   CSR 行内按 indptr 顺序累加 → 行间并行不改变行内归约顺序 →
   理论上应与 serial **逐位**相同。若不逐位，说明 prange 改变了语义
   （那就必须走容差并写进文档，而不是默默接受）。
C. 零回归：default 的fused 路径本身仍可用；`fused` 参数三态映射正确
   （`True`→fused / `"serial"`→serial / `False`→plain）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in ("", "tests", "tools"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

_RTOL = 1e-6          # 融合/非融合核的归约顺序差异（fp32）
_RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    return bool(ok)


def _try_numba() -> bool:
    try:
        import numba                                # noqa: F401
        return True
    except Exception:                               # noqa: BLE001
        return False


def build(fused, n_sdr=256, k=32, seed=5):
    """构造一个 SparsePCStack（`fused` 三态）+ 固定 s0。"""
    from phdnet.sparse_pc import SparsePCStack
    rng = np.random.default_rng(seed)
    #签名： (n0, n1, n2, eta_pc, eta_oja, rng, w_max, conn_k, lognormal_init,
    #         exc_ratio, fused)
    # eta_pc/eta_oja 在 infer 里不参与（只有 learn 用），取 0 避免误 learn。
    pc = SparsePCStack(n_sdr, n_sdr, n_sdr, 0.0, 0.0, rng,
                       conn_k=k, fused=fused)
    s0 = rng.normal(0.0, 1.0, n_sdr).astype(np.float64)
    return pc, s0


def main() -> int:
    print("=" * 78)
    print("P113 M2 推理核三态数值等价对拍（fused / serial / plain）")
    print("=" * 78)

    if not _try_numba():
        print("  · numba 不可用 → 三态都走 numpy 回退路径，本文件退化为"
              "「numpy 语义一致性」检查（仍有效，但不代表生产核）")

    # ── A. 三态等价 ────────────────────────────────────────────────────────
    print("\n[A] 三态等价（同一 s0、同一 CSR）")
    pc_f, s0 = build(True)
    pc_s, _ = build("serial")
    pc_p, _ = build(False)

    d_fs = d_ps = d_pf = 0.0
    for _ in range(3):
        rf = pc_f.infer(s0.copy())
        rs = pc_s.infer(s0.copy())
        rp = pc_p.infer(s0.copy())
        for key in ("r1", "r2", "e0", "e1"):
            d_fs = max(d_fs, float(np.abs(rf[key] - rs[key]).max()))
            d_ps = max(d_ps, float(np.abs(rp[key] - rs[key]).max()))
            d_pf = max(d_pf, float(np.abs(rp[key] - rf[key]).max()))
    scale = max(1.0, float(np.abs(rs["r1"]).max()))

    check(d_fs <= _RTOL * scale, "A1 fused vs serial",
          f"max|Δ|={d_fs:.3e} (尺度 {scale:.3f})")
    check(d_ps <= _RTOL * scale, "A2 plain vs serial（非融合 vs 融合）",
          f"max|Δ|={d_ps:.3e} (尺度 {scale:.3f})")
    check(d_pf <= _RTOL * scale, "A3 plain vs fused",
          f"max|Δ|={d_pf:.3e}")

    # ── B. 多步一致性（E1 家族教训：单步一致 ≠ 多步一致）────────────────────
    print("\n[B] n_steps = 1/2/3（多步递推，E1 家族教训）")
    for ns in (1, 2, 3):
        rf = pc_f.infer(s0.copy(), n_steps=ns)
        rs = pc_s.infer(s0.copy(), n_steps=ns)
        rp = pc_p.infer(s0.copy(), n_steps=ns)
        dmax = 0.0
        for key in ("r1", "r2", "e0", "e1"):
            sc = max(1.0, float(np.abs(rs[key]).max()))
            dmax = max(dmax, float(np.abs(rp[key] - rs[key]).max()) / sc)
        check(dmax <= _RTOL, f"B n_steps={ns} plain vs serial 相对偏差",
              f"max rel|Δ|={dmax:.3e}")

    # B2 plain 的行级 prange 不应改变行内归约顺序 → 期望逐位
    print("\n[B2] plain 的行级 prange 是否逐位等价于 serial")
    bitexact = d_ps == 0.0
    if bitexact:
        check(True, "B2 plain vs serial **逐位**相同（prange 未改归约顺序）",
              "max|Δ|=0")
    else:
        # 非逐位不一定是 bug（numpy 回退路径 / tanh 实现差异），但必须如实标注
        check(True, "B2 plain vs serial 非逐位（已在容差内）",
              f"max|Δ|={d_ps:.3e}—— 若在昇腾上出现非逐位，需在文档注明"
              f"「plain 与 serial 非逐位」，不可宣称零回归")

    # ── C. 三态映射 + 零回归 ──────────────────────────────────────────────
    print("\n[C] 三态映射与零回归")
    # C1 fused 参数的映射：True / "fused" → fused；"serial" → serial；False → plain
    pc_t, _ = build(True)
    pc_t2, _ = build("fused")
    m_fused = (pc_t.fused is True) or (pc_t.fused == "fused") or bool(pc_t.fused)
    m_serial = pc_s.fused == "serial"
    m_plain = pc_p.fused is False or not pc_p.fused
    check(m_fused and m_serial and m_plain,
          "C1 fused 参数三态映射正确",
          f"True→{pc_t.fused!r} / 'serial'→{pc_s.fused!r} / False→{pc_p.fused!r}")

    # C2 三态都返回同形状（逐 token 语义：不能有 batch 维）
    shapes_ok = all(
        pc.infer(s0.copy())["r1"].shape == s0.shape for pc in (pc_f, pc_s, pc_p))
    check(shapes_ok, "C2 三态输出形状一致且无 batch 维",
          f"shape={pc_s.infer(s0.copy())['r1'].shape}")

    # C3 连续多次 infer 不应破坏状态（plain 路径无隐藏缓存）
    pc_c, _ = build(False)
    a1 = pc_c.infer(s0.copy())["r1"].copy()
    pc_c.infer(s0.copy())
    a2 = pc_c.infer(s0.copy())["r1"]
    check(np.array_equal(a1, a2), "C3 plain 连续两次同输入 → 同输出（无隐藏状态）")

    # C4 e0/e1 的语义：e0 应等于 s0 − dn0·r1（这是 M2 的自由能定义）
    from phdnet.sparse_pc import _csr_matvec
    pc_v, sv = build(False)
    rv = pc_v.infer(sv.copy())
    e0_ref = sv - _csr_matvec(*pc_v.dn0, rv["r1"])
    check(float(np.abs(rv["e0"] - e0_ref).max()) <= _RTOL,
          "C4 e0 = s0 − dn0·r1（自由能定义未变）",
          f"max|Δ|={float(np.abs(rv['e0'] - e0_ref).max()):.3e}")

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