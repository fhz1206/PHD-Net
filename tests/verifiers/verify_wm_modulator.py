"""2026-10-07 审计门禁：wm.py / modulator.py / context_memory.py 七项缺陷修复验收。

覆盖审计 1-7（见各 check 的标注）；任一 FAIL → exit 1。
运行：py -3.14 tests/verifiers/verify_wm_modulator.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phdnet.context_memory import ContextMemory   # noqa: E402
from phdnet.modulator import MultiModulator, Neuromodulator  # noqa: E402
from phdnet.wm import WorkingMemory               # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(("PASS" if cond else "FAIL"), "|", name, "|", detail)
    if not cond:
        FAILURES.append(name)


def _build(seed=1):
    cm = ContextMemory(64, rho=0.8, eta=0.5, k_clean=8, seed=seed)
    rng = np.random.default_rng(0)
    items = []
    for _ in range(5):
        v = np.zeros(64)
        v[rng.choice(64, 8, replace=False)] = 1.0
        items.append(v)
    # 情节 0 先把 ctx 跑成非零，再开情节 1 检索（slot 在 begin_episode 捕获，
    # 与 demo_copy 的跨情节用法一致；全新零上下文按审计 7 返回空读出而非伪回忆）
    cm.begin_episode(0)
    for v in items:
        cm.observe(v)
    cm.end_episode()
    cm.begin_episode(1)
    for v in items:
        cm.observe(v)
    return cm, {f"t{i}": items[i] for i in range(5)}


def audit1():
    """scored 评分向量与 recall 状态轨迹共用同一算子（clean_steps>=2 不再分叉）。"""
    for cs in (1, 2, 3):
        cm, cands = _build()
        names = list(cands)
        mat = np.stack([cands[k] for k in names])
        scored = cm.recall_scored(6, cands, clean_steps=cs)
        rec = cm.recall(6, clean_steps=cs)
        rec_names = [names[int(np.argmax(mat @ it))] for it in rec]
        check(f"审计1 recall==recall_scored (clean_steps={cs})",
              scored == rec_names, f"scored={scored} recall={rec_names}")
    # 状态轨迹级：两条路径的逐步状态/检索向量逐位一致（修复前 cs=2 overlap=[4,0,0,4,0]）
    cm, cands = _build()
    names = list(cands)
    mat = np.stack([cands[k] for k in names])
    cs = 2
    scored = cm.recall_scored(6, cands, clean_steps=cs)
    c_s, c_r, states_ok = cm.slot.copy(), cm.slot.copy(), True
    for t in range(6):
        item_r = cm._retrieval(c_r)
        v_s = cm._retrieval(c_s)
        states_ok &= np.array_equal(item_r, v_s)
        states_ok &= np.array_equal(cm._clean(c_r), cm._clean(c_s))
        pick = names[int(np.argmax(mat @ v_s))]
        states_ok &= pick == scored[t]
        c_r = cm._advance(c_r, item_r, cs)
        c_s = cm._advance(c_s, cands[scored[t]], cs)
    check("审计1 状态轨迹逐位一致 (cs=2, 6 步)", states_ok)
    # 共用算子门禁：recall_scored 的评分向量必须来自公共算子 _retrieval
    # （修复前它自带另一套多步递推，不会调用 _retrieval → 0 次）
    cm2, cands2 = _build()
    calls = []
    orig = cm2._retrieval
    def spy(c):
        calls.append(np.array(c, copy=True))
        return orig(c)
    cm2._retrieval = spy
    cm2.recall_scored(6, cands2, clean_steps=2)
    check("审计1 recall_scored 经公共算子 _retrieval 评分 (6 步=6 次)",
          len(calls) == 6, f"calls={len(calls)}（修复前 0 次）")


def audit2():
    wm = WorkingMemory(8, 4, 0.9)
    wrote = wm.write(np.ones(8), float("nan"), 0.5)
    check("审计2 write(NaN gate) 被拒", wrote is False,
          f"write={wrote} strength={wm.strength.tolist()}")
    check("审计2 write 后 strength/read 无 NaN",
          not np.isnan(wm.strength).any() and not np.isnan(wm.read()).any())

    class L:
        n_dim = 8
        def recall(self, c):
            return np.ones(8)
    wm2 = WorkingMemory(8, 4, 0.9)
    wm2.slots[:] = np.nan
    wm2.strength[:] = 1.0
    ok = wm2.summarize(L()) is False
    check("审计2 summarize(NaN cue) 被拒", ok)

    # ── 交叉验证漏报 3（2026-10-07）：**内容侧** NaN —— 只挡 gate 不够 ──
    # 实测（第二意见复核）：write(r=NaN, gate=1.0)=True → decay×2 → read() **永久全 NaN**
    wm3 = WorkingMemory(8, 4, 0.9)
    wrote3 = wm3.write(np.full(8, np.nan), 0.9, 0.5)
    check("漏报3 write(NaN 内容, 正常 gate) 被拒", wrote3 is False,
          f"write={wrote3} read_is_finite={bool(np.all(np.isfinite(wm3.read())))}")
    check("漏报3 该次写入后 read() 仍有限",
          bool(np.all(np.isfinite(wm3.read()))), "")

    # 自愈：**历史数据**里已带毒的槽不得永久污染读出（0×NaN 仍是 NaN，槽也要换掉）
    wm4 = WorkingMemory(8, 4, 0.9)
    wm4.slots[0] = np.nan
    wm4.slots[1] = np.ones(8)
    wm4.strength[:] = 1.0
    out = wm4.read()
    check("漏报3 read() 自愈：毒化槽不污染输出", bool(np.all(np.isfinite(out))),
          f"out={np.round(np.asarray(out), 3).tolist()[:4]}")

    # summarize 的吸引子结果非有限 → 拒绝写入免衰减槽
    class BadLTM:
        n_dim = 8
        def recall(self, c):
            return np.full(8, np.nan)

    wm5 = WorkingMemory(8, 4, 0.9)
    wm5.slots[0] = np.ones(8)
    before = wm5.slots.copy()
    ok5 = wm5.summarize(BadLTM()) is False
    check("漏报3 summarize(吸引子吐 NaN) 被拒且槽未被写", ok5
          and bool(np.array_equal(wm5.slots, before)),
          f"rejected={ok5}")


def audit3():
    class BigLTM:
        n_dim = 64
        def __init__(self):
            self.cues = []
        def recall(self, c):
            self.cues.append(np.array(c, copy=True))
            return c
    wm = WorkingMemory(64, 4, 0.9)
    wm.slots[0, :] = 1.0        # 4 槽同型 → 全幅值并列
    wm.strength[0] = 1.0
    ltm = BigLTM()
    ret = wm.summarize(ltm)
    cue = ltm.cues[-1]
    nz = int((cue != 0).sum())
    check("审计3 并列幅值 cue 恰好 top-k=8 位",
          ret and nz == 8, f"激活 {nz}/64（修复前 64/64，超 n_dim/4=16 上限）")


def audit4():
    class L2:
        n_dim = 4
        def recall(self, c):
            return np.array([1.0, 0.0, 0.0, 0.0])
    wm = WorkingMemory(4, 1, 0.9)          # n_slots == 1
    wm.slots[0] = np.ones(4)
    wm.strength[:] = 1.0
    ret = wm.summarize(L2())
    check("审计4 n_slots==1 summarize 成功建立摘要槽", ret is True and wm.summary_slot == 0,
          f"ret={ret} summary_slot={wm.summary_slot}")
    before = wm.slots[0].copy()
    wrote = wm.write(np.array([0.0, 1.0, 0.0, 0.0]), 0.9, 0.5)
    check("审计4 n_slots==1 write 被拒且摘要不被覆盖",
          wrote is False and np.allclose(wm.slots[0], before),
          f"write={wrote} overwritten={not np.allclose(wm.slots[0], before)}")


def audit5():
    def z_of(gate, gain=1.5):
        g = float(np.clip(gate, 1e-12, 1.0 - 1e-12))
        return float(np.log(g / (1.0 - g)) / gain)
    m1 = Neuromodulator()
    g1, _ = m1.observe(1.0)
    z1 = z_of(g1)
    m2 = Neuromodulator()
    g2, _ = m2.observe(100.0)
    g1_exp = float(1.0 / (1.0 + np.exp(-1.5 * 1.0)))    # z = surprise/(m2/count)^0.5 = 1.0
    # 修复前：更新后统计 → s=1→gate .704、s=100→gate .818（z 都≈1，与幅值脱钩）
    check("审计5 单通道首样本 z 与幅值成正比",
          abs(g1 - g1_exp) < 1e-5 and g1 >= 0.8175 and g2 > 0.9999,
          f"s=1→z={z1:.4f}(gate {g1:.4f}, 修复前 .704)；s=100→gate {g2:.6f}(z≈100, 修复前 .818)")

    mm = MultiModulator()
    g_first, _ = mm.observe(1.0)
    z_first = z_of(mm.ach, mm.gain_ach)
    check("审计5 MultiModulator 首步 novelty 中性（跳过虚构基线）",
          mm.ne == 0.5 and mm.ht == 0.5 and abs(z_first - 1.0) < 1e-3,
          f"ne={mm.ne}(修复前 .566) ht={mm.ht} z1={z_first:.4f} gate1={g_first:.4f}(修复前 .542)")
    g_second, _ = mm.observe(1.0)
    # 第二步 z2=(1-0.5)/sqrt(1.5/2)=0.5774 → nov=|0.5774-1|=0.4226 → ne=σ(2*(nov-0.5))
    ne2 = float(1.0 / (1.0 + np.exp(-2.0 * (abs(0.5773502691 - 1.0) - 0.5))))
    check("审计5 第二步 novelty 用真实上一步 z 计算",
          abs(mm.ne - ne2) < 1e-3 and mm.ne != 0.5,
          f"ne2={mm.ne:.4f} 期望 {ne2:.4f}")


def audit6():
    cm = ContextMemory(16, k_clean=100)
    check("审计6 k_clean>n_dim 被 clamp", cm.k == 16, f"k={cm.k}")
    cm.begin_episode(0)
    cm.observe(np.ones(16))
    try:
        cm.recall(2)
        ok = True
        err = ""
    except Exception as e:                       # noqa: BLE001
        ok, err = False, f"{type(e).__name__}: {e}"
    check("审计6 k=100 recall 不再 argpartition 越界", ok, err)
    cm2 = ContextMemory(16, k_clean=0)
    cm2.begin_episode(0)
    cm2.observe(np.ones(16))   # 第 1 次观察：ctx 仍为 0，W 不变
    cm2.observe(np.ones(16))   # 第 2 次：W 非零（ctx 已漂移非零）
    cm2.begin_episode(1)       # 重捕获非零 slot，保证检索有信号
    item = cm2.recall(1)[0]
    check("审计6 k<=0 被 clamp 到 1（不再全零静默退化）",
          cm2.k == 1 and int(item.sum()) == 1, f"k={cm2.k} active={int(item.sum())}")


def audit7():
    cm = ContextMemory(16, k_clean=4)
    try:
        cm.recall_scored(2, {})
        ok, err = False, "no exception"
    except ValueError:
        ok, err = True, ""
    except Exception as e:                       # noqa: BLE001
        ok, err = False, f"{type(e).__name__}: {e}"
    check("审计7 空候选显式 ValueError", ok, err)

    check("审计7 _clean(全零) 不造伪回忆",
          not np.any(cm._clean(np.zeros(16))), f"sum={float(cm._clean(np.zeros(16)).sum())}")

    cm0 = ContextMemory(16, k_clean=4)           # W 全零的新记忆
    cm0.begin_episode(0)
    cands = {"a": np.eye(16)[0], "b": np.eye(16)[1]}
    out = cm0.recall_scored(3, cands)
    check("审计7 全零激活 → 空读出而非恒选 names[0]",
          out == [], f"out={out}（修复前 ['a','a']）")


if __name__ == "__main__":
    audit1()
    audit2()
    audit3()
    audit4()
    audit5()
    audit6()
    audit7()
    print("=" * 72)
    if FAILURES:
        print(f"FAILURES({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("ALL PASS (审计 1-7 全绿)")
    sys.exit(0)
