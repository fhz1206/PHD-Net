"""P0-1 / P2-7 / P2-8 门禁：**评估必须是只读的（零状态副作用）** —— exit 0 = PASS。

背景（复核者实测，修复前）
================================================================================
`phdnet/lm.py:63` 的 evaluate 只传了 `learn=False`、漏了 `readonly=True`
（对照 `phdnet/word_lm.py:227` 早已传）。`learn=False` 只关「权重学习」；
`readonly=False` 时 model.step 仍执行 modulator.observe、wm.decay/write、
step_count+=1、_last_rate/_prev_rate/STDP 迹推进。后果：
  · 同文本连评两次 PPL **9.741009 vs 9.660860**（相对差 8.228e-3），
    net.step_count 4399→4917（+518）——评估不可复现；
  · A 模型在两段训练之间只多跑 1 次 evaluate，与 B（不评估）在**完全相同数据**
    上继续训练后权重分叉：net.stdp.W maxΔ=0.0225、net.wm.slots maxΔ=1.19，
    heldout PPL **9.5564 vs 9.8475（−3.0%）** ——「评估」污染了训练，A/B 对照失真。
`tests/eval_tasks_lm.py` 任务 2 同病：三个测试长度共用同一个**非只读**模型 →
结果随执行顺序变（n=8 首测 0.0940，先跑 n=4 再测 0.0909，Δ=−0.0031）。

断言（exit 0 = PASS）
================================================================================
(a) 连评两次必须逐位相等：同模型同文本 evaluate×2 → PPL 完全相等，
    且 net.step_count 不变。
(b) 评估零副作用：evaluate 前后快照 net.stdp.W / net.wm.slots / net._last_rate /
    modulator 内部状态 → 全部 np.array_equal（标量逐位相等）。
(b*) `_nll` 兼容路径同步检查（同样零副作用）。
(c) 评估后再训练与「不评估」等权：两份同 seed 模型，一份中间插一次 evaluate，
    之后喂相同数据 → 权重逐位一致；并带**负向对照**（模拟旧的非只读评估 →
    必须分叉），证明本快照具备检出力、不是恒真断言。
(d) eval_tasks_lm 的顺序无关：同一模型先跑 n=8 再跑 n=4 vs 反序 → 结果相同。
(e) 源码守卫：三处调用点显式 readonly=True + task1 计时两口径分离。

(a)(b)(b*)(c) 用小规模配置（n_sdr/n_mid/n_top=64），整段 < 30 s。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _d in (str(_ROOT), str(_ROOT / "tests")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np

_RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""),
          flush=True)
    return bool(ok)


# ── 小规模配置与语料（自包含，不依赖语料文件） ──────────────────────────────
SMALL = dict(n_sdr=64, k_sparse=16, n_mid=64, n_top=64,
             eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
             pred_in_readout=True)
_WORDS = ("alpha ", "beta ", "gamma ", "delta ")
TRAIN = "".join(_WORDS[i % 4] for i in range(60))              # 360 字符
EVAL = "".join(_WORDS[(i * 3 + 1) % 4] for i in range(30))     # 180 字符
VOCAB = "".join(sorted(set(TRAIN + EVAL)))


def _make_lm(pred_in_readout: bool = True):
    from phdnet.config import PHDNetConfig
    from phdnet.lm import PHDNetLM
    return PHDNetLM(VOCAB, PHDNetConfig(**{**SMALL,
                                           "pred_in_readout": pred_in_readout}))


def _arr(v):
    """任意权重/状态对象 → np.ndarray（无法转换则打标记，两端同标记视为已跳过）。"""
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return np.array(v, copy=True)
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().to(torch.float32).numpy()
    except Exception:
        pass
    try:
        return np.array(v, copy=True)
    except Exception:
        return ("<unconverted>", type(v).__name__)


def _snapshot(lm) -> dict:
    """全量持久状态快照：权重 + 运行状态 + 调制器内部。

    ⚠ 刻意**排除**诊断计数器（_prof/_prof_win/_prof_win_n/_ro_ms_accum/_ro_calls）：
    它们由 model.py 的计时探针无条件累积（_prof_step 在 step 入口、_ro_calls 在
    读出后），属于观测开销而非模型状态，readonly 语义不覆盖它们。
    """
    net = lm.net
    out: dict = {}

    def put(k, v):
        if v is None:
            out[k] = None
        else:
            out[k] = _arr(v)

    # 权重 / 记忆
    put("stdp.W", net.stdp.W)
    put("wm.slots", net.wm.slots)
    put("wm.strength", net.wm.strength)
    put("readout.W", getattr(net.readout, "W", None))
    for name in ("up0", "up1", "dn0", "dn1"):                 # PC 主干 CSR
        v = getattr(net.pc, name, None)
        if isinstance(v, (tuple, list)):
            for i, e in enumerate(v):
                put(f"pc.{name}[{i}]", e)
        else:
            put(f"pc.{name}", v)
    # 运行状态
    put("_last_rate", net._last_rate)
    put("_prev_rate", net._prev_rate)
    put("_prev_sup", net._prev_sup)
    for k in ("step_count", "_exp", "_task_gate", "_last_da", "_ro_eta",
              "_sleep_count", "_instab_ema", "learn_pc", "learn_stdp"):
        out[k] = getattr(net, k)
    # 调制器内部状态（M5 + T1.1 任务调制器）
    for tag, obj in (("mod", net.modulator), ("task_mod", net.task_mod)):
        for k in ("mu", "m2", "count", "ach", "ne", "da", "ht", "_z_prev"):
            if hasattr(obj, k):
                out[f"{tag}.{k}"] = getattr(obj, k)
    return out


def _cmp(a: dict, b: dict) -> tuple[bool, str]:
    if set(a) != set(b):
        return False, f"快照键不一致：{sorted(set(a) ^ set(b))[:5]}"
    bad, skipped = [], 0
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, tuple) or isinstance(y, tuple):       # 无法转换 → 跳过
            skipped += 1 if x == y else 0
            if x != y:
                bad.append(f"{k}: 不可比对象不一致")
            continue
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            ok = isinstance(x, np.ndarray) and isinstance(y, np.ndarray) \
                and x.shape == y.shape and np.array_equal(x, y)
            d = (float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                 if ok is False and isinstance(x, np.ndarray)
                 and isinstance(y, np.ndarray) and x.shape == y.shape and x.size
                 else "n/a")
        elif x is None or y is None:
            ok, d = x is None and y is None, "None"
        elif isinstance(x, (float, np.floating)) or isinstance(y, (float, np.floating)):
            ok, d = float(x) == float(y), f"Δ={float(y) - float(x):.3e}"
        else:
            ok, d = bool(x == y), f"{x!r} vs {y!r}"
        if not ok:
            bad.append(f"{k}: {d}")
    n = len(a) - skipped
    if bad:
        return False, f"{len(bad)}/{n} 项不一致 → " + "; ".join(bad[:5])
    return True, f"{n} 项全部逐位一致" + (f"（另 {skipped} 项不可比已跳过）" if skipped else "")


# ── (a)(b) 连评逐位相等 + 评估零副作用 ───────────────────────────────────────
def section_ab() -> None:
    print("\n[(a)(b)] 同文本连评两次：PPL 逐位相等、step_count 不变、全状态零副作用")
    lm = _make_lm()
    lm.train_stream(TRAIN)
    sc0 = lm.net.step_count
    pre = _snapshot(lm)
    p1 = lm.evaluate(EVAL)
    mid = _snapshot(lm)
    p2 = lm.evaluate(EVAL)
    post = _snapshot(lm)
    sc2 = lm.net.step_count

    check(np.isfinite(p1) and np.isfinite(p2) and p1 > 0,
          "(a0) PPL 是有效正数（评估真的跑了）", f"ppl={p1!r}")
    check(p1 == p2, "(a1) 两次 evaluate 的 PPL **逐位相等**（==，非容差）",
          f"{p1!r} vs {p2!r}  Δ={p2 - p1:.3e}")
    check(sc0 == sc2, "(a2) evaluate 不推进 net.step_count",
          f"{sc0} → {sc2}（修复前 +{len(EVAL) - 1}）")
    ok1, d1 = _cmp(pre, mid)
    ok2, d2 = _cmp(mid, post)
    check(ok1 and ok2, "(b) evaluate 前后快照全一致（stdp.W/wm.slots/_last_rate/modulator）",
          f"第1次: {d1} | 第2次: {d2}")

    # (b*) 兼容路径 _nll：不调用 step()，同样必须零副作用
    x = lm.tok.encode(EVAL[0])
    tgt = lm.tok.onehot(lm.tok.stoi[EVAL[1]])
    pre_n = _snapshot(lm)
    nll = lm._nll(x, tgt)
    post_n = _snapshot(lm)
    okn, dn = _cmp(pre_n, post_n)
    check(okn and np.isfinite(nll), "(b*) _nll 兼容路径零状态副作用", f"nll={nll:.6f} | {dn}")


# ── (c) 评估后再训练 vs 不评估 ────────────────────────────────────────────────
def section_c() -> None:
    print("\n[(c)] 一份中间插 evaluate、一份不评估 → 继续同数据训练后等权")
    a, b = _make_lm(), _make_lm()
    a.train_stream(TRAIN)
    b.train_stream(TRAIN)
    ok_same, d_same = _cmp(_snapshot(a), _snapshot(b))
    check(ok_same, "(c0) 两份同 seed 模型训练后逐位一致（后续对比才有意义）", d_same)

    a.evaluate(EVAL)                       # 只有 A 多跑一次评估
    tail = TRAIN[::-1]                     # 第二段：两份完全相同的数据
    a.train_stream(tail)
    b.train_stream(tail)
    ok, d = _cmp(_snapshot(a), _snapshot(b))
    check(ok, "(c1) 中间插一次 evaluate 后继续训练 → 权重与状态逐位一致", d)

    # 负向对照：用**非只读** step 复刻旧实现的评估，快照必须检出分叉（证明非恒真）
    a2, b2 = _make_lm(), _make_lm()
    a2.train_stream(TRAIN)
    b2.train_stream(TRAIN)
    for t in range(len(EVAL) - 1):         # 旧 lm.py:63 —— 只有 learn=False
        a2.net.step(a2.tok.encode(EVAL[t]), learn=False)
    a2.train_stream(tail)
    b2.train_stream(tail)
    ok2, d2 = _cmp(_snapshot(a2), _snapshot(b2))
    check(not ok2, "(c2) 负向对照：非只读评估确实分叉（快照具备检出力）",
          ("检出分叉 " + d2[:110]) if not ok2 else "未检出分叉 → 快照太弱/断言恒真！")


# ── (d) eval_tasks_lm 顺序无关 ────────────────────────────────────────────────
def section_d() -> None:
    print("\n[(d)] eval_tasks_lm 任务2：先 n=8 后 n=4 vs 反序 → 结果相同")
    from eval_tasks_lm import _gen_copy, copy_eval_acc
    from phdnet.config import PHDNetConfig
    from phdnet.lm import PHDNetLM

    vocab = "abcdefghijkl#."
    train_txt, _ = _gen_copy(n=8, n_seq=40, seed=1)

    def mk():
        lm = PHDNetLM(vocab, PHDNetConfig(**{**SMALL, "pred_in_readout": False}))
        lm.train_stream(train_txt)
        return lm

    m1, m2 = mk(), mk()
    ok0, d0 = _cmp(_snapshot(m1), _snapshot(m2))
    check(ok0, "(d0) 两份同 seed 模型评估前逐位一致", d0)
    sc0 = m1.net.step_count
    r1 = {8: copy_eval_acc(m1, 8), 4: copy_eval_acc(m1, 4)}   # 先 8 后 4
    sc1 = m1.net.step_count
    r2 = {4: copy_eval_acc(m2, 4), 8: copy_eval_acc(m2, 8)}   # 先 4 后 8
    sc2 = m2.net.step_count
    check(r1[8] == r2[8] and r1[4] == r2[4],
          "(d1) 顺序无关：两种执行顺序的准确率**逐位相等**",
          f"n=8: {r1[8]:.6f} vs {r2[8]:.6f} | n=4: {r1[4]:.6f} vs {r2[4]:.6f}")
    check(sc0 == sc1 and sc0 == sc2, "(d2) 评估循环不推进 step_count",
          f"{sc0} → {sc1} / {sc2}")


# ── (e) 源码守卫 ──────────────────────────────────────────────────────────────
def section_e() -> None:
    print("\n[(e)] 源码守卫：调用点显式 readonly=True + task1 计时两口径分离")
    lm_src = (_ROOT / "phdnet" / "lm.py").read_text(encoding="utf-8")
    ev_src = (_ROOT / "tests" / "eval_tasks_lm.py").read_text(encoding="utf-8")

    check('learn=False, readonly=True)["y"]' in lm_src,
          "(e1) phdnet/lm.py evaluate 的 step 显式 readonly=True")
    check(lm_src.count("learn=False, readonly=True") >= 1 and "net.step" in lm_src,
          "(e2) phdnet/lm.py 评估用 step 调用已逐个对齐（无裸 learn=False 评估）",
          f"裸评估调用 {ev_src.count('learn=False)[\"y\"]')} 处（eval_tasks_lm）")
    check('readonly=True)["y"]' in ev_src,
          "(e3) eval_tasks_lm 复制评估循环 readonly=True")
    for key in ("ms_per_token_train", "ms_per_token_eval", "ms_per_token_incl_eval"):
        check(key in ev_src, f"(e4) task1 输出含 {key}（两口径分别计时）")
    check("t_train" in ev_src and "t_eval" in ev_src
          and "time.perf_counter() - t0" in ev_src,
          "(e5) task1 训练/评估分别计时（不再共用一个 dt）")


def main() -> int:
    print("=" * 78)
    print("门禁：评估必须只读（readonly=True）—— 连评逐位相等 / 零副作用 / 评估不污染训练")
    print("=" * 78)
    section_ab()
    section_c()
    section_d()
    section_e()
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
