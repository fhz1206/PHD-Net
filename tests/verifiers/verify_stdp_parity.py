"""门禁：M3 STDP **双路径对拍**（numba 核 vs numpy 参考）—— exit 0 = PASS。

为什么要有这条门禁（2026-10-07 审计 P0，已实测）：
    `STDPCore.step` 默认走 numba 核，numpy 分支只在 adaptive/homeostasis/
    metaplasticity/ei 开启时走到 —— 于是「两条路径是否等价」长期**没有**任何
    自动化断言。原实现把 `eta * eta_scale` 折进 t_pre 却给核传 `eta=1.0`、
    `tp_hist` 未缩放 → **LTP 带 η·ηs、LTD 一个 η 都没有**，而 numpy 分支
    两项都乘。实测（同种子 30 步）：
        η=0.03, ηs=1.0 → numba W sum=1.31 vs numpy 2.31（91% vs 85% 零权）
        η=0.03, ηs=0.5 → 0.59 vs 1.15（M3 近死）
        η=ηs=1.0      → 逐位相同
    而 `stdp_kernels._selftest_numba` **只在 η=ηs=1.0 那一档对拍** → 自检恒绿、
    fast 门禁 9/9 全绿，生产（η=0.03 默认）两条路径早已分叉 —— 这就是本文件
    要堵的「门禁缺口」：**必须覆盖真实超参，而不是只覆盖自检用的那一档**。

断言三件事：
  A. 多组 (η, ηs)（含生产默认 0.03 / 0.05）下 numba 与 numpy **逐位一致**；
  B. numba 路径下权重**确实会动**（堵「静默学不动」：全零 = 学习死了没人知道）；
  C. `predict`（numba vs numpy scatter 核）逐位一致。

用法：py -3.14 tests/verifiers/verify_stdp_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import phdnet.plasticity as P                                   # noqa: E402
from phdnet.plasticity import STDPCore                          # noqa: E402

_RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def _run(use_numba: bool, eta: float, eta_scale: float, steps: int = 30,
         dual: bool = False) -> np.ndarray:
    """固定种子跑 steps 步，返回最终权重（切 numba/numpy 走同一份输入序列）。"""
    core = STDPCore(40, 6, 0.35, eta, 1.0, np.random.default_rng(7), dual=dual)
    rnd = np.random.default_rng(11)
    old = P.NUMBA_OK
    P.NUMBA_OK = bool(use_numba)
    try:
        for _ in range(steps):
            pre = np.zeros(40)
            pre[rnd.integers(0, 40, 5)] = 1.0
            post = np.zeros(40)
            post[rnd.integers(0, 40, 2)] = 1.0
            core.step(pre, post, eta_scale=eta_scale)
    finally:
        P.NUMBA_OK = old
    return core.W.copy()


def main() -> int:
    print("=" * 78)
    print("门禁：M3 STDP numba ↔ numpy 双路径对拍（覆盖真实超参）")
    print("=" * 78)
    if not P.NUMBA_OK:
        print("  [SKIP] 本机 NUMBA_OK=False：无法跑 numba 路径（仍跑 B 组活性断言）")

    # ── A. 双路径逐位一致（生产默认 η=0.03 必须在列）────────────────────
    print("\n[A] 双路径对拍")
    grid = [(0.03, 1.0), (0.03, 0.5), (0.05, 1.0), (0.15, 1.0), (1.0, 1.0)]
    for eta, es in grid:
        if not P.NUMBA_OK:
            continue
        nb, np_ = _run(True, eta, es), _run(False, eta, es)
        d = float(np.abs(nb - np_).max())
        check(f"A  η={eta} ηs={es} numba 与 numpy 逐位一致", d == 0.0,
              f"max|ΔW|={d:.3e}（numba sum={nb.sum():.5f} / numpy sum={np_.sum():.5f}）")

    # ── B. 权重必须真的在动（堵「静默学不动」）────────────────────────────
    print("\n[B] 学习活性")
    for eta, es in ((0.03, 1.0), (0.03, 0.5)):
        w = _run(True, eta, es)
        moved = float((np.abs(w) > 0).mean())
        check(f"B  η={eta} ηs={es} 权重确实在更新（非全零）", moved > 0.05,
              f"非零占比={moved:.1%} sum={w.sum():.4f}")

    # ── C. dual_trace（T4.2）同样要双路径一致 ────────────────────────────
    print("\n[C] dual_trace 路径")
    if P.NUMBA_OK:
        nb, np_ = _run(True, 0.03, 1.0, dual=True), _run(False, 0.03, 1.0, dual=True)
        d = float(np.abs(nb - np_).max())
        check("C  dual=True 时两路径仍逐位一致", d == 0.0, f"max|ΔW|={d:.3e}")

    # ── D. predict（scatter 核）对拍 ──────────────────────────────────────
    print("\n[D] predict scatter 核")
    core = STDPCore(32, 5, 0.35, 0.03, 1.0, np.random.default_rng(3))
    rate = np.zeros(32)
    rate[[1, 7, 20]] = 1.0
    p_nb = core.predict(rate)
    from phdnet.stdp_kernels import _predict_edges_np
    p_np = _predict_edges_np(core.W.astype(np.float64), core.post_idx, rate, core.n)
    p_np = np.clip(p_np, 0.0, core.w_max)
    d = float(np.abs(p_nb - p_np).max())
    check("D  predict numba 与 numpy 一致", d <= 1e-9, f"max|Δ|={d:.3e}")

    ok = sum(1 for r in _RESULTS if r[0])
    bad = [r[1] for r in _RESULTS if not r[0]]
    print("\n" + "=" * 78)
    print(f"结果：{ok}/{len(_RESULTS)} 通过" + (f" | 失败 {bad}" if bad else ""))
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
