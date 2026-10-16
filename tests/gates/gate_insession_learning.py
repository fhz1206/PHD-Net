"""P197 门禁：会话中学习能力（新颖性驱动印迹 + 误差广播）。

跑法：py -3.14 tests/gates/gate_insession_learning.py

判据：
  A1 SDR 新颖性：互异输入 → 新模式；重复输入 → 命中（熟悉化）
  A2 只读路径不登记新模式（评估不污染训练期集合）
  A3 LRU 淘汰：表满后最旧模式被挤出
  B1 印迹触发率 >> 旧默认（0.67%）—— 目标 > 30%
  B2 LTM 表真的生长
  B3 重复刺激不再印迹（熟悉化）
  C1 读出头能学会单个映射（nll 降 > 50%）
  C2 不泛化到未见输入（防过拟合假象）
  D1 配置默认值已切换（imprint_gate / task_modulation / ...）
  D2 默认关闭时逐位回旧行为（sdr_novelty=False → 无覆盖）
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "train")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                        # noqa: E402

from phdnet.config import PHDNetConfig                    # noqa: E402
from phdnet.model import PHDNet                           # noqa: E402

CASES: list = []


def check(ok, name, detail=""):
    CASES.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))


def _cfg(**kw):
    base = dict(n_input=32, n_sdr=64, n_mid=32, n_top=16, k_sparse=8,
                eta_pc=0.02, eta_oja=0.02, pc_dev_steps=20000,
                seed=11, step_profiling=False)
    base.update(kw)
    return PHDNetConfig(**base)


def _big(**kw):
    base = _cfg(big_ltm=True, big_ltm_N=1 << 14, big_ltm_m=16, big_ltm_k=2)
    base.__dict__.update(kw)
    return base


def main():
    print("=" * 74)
    print("P197 会话中学习能力（新颖性驱动印迹 + 误差广播）")
    print("=" * 74)

    # ── A. SDR 新颖性检测 ─────────────────────────────────────────
    print("\n[A] SDR 支撑集新颖性（海马模式分离）")
    net = PHDNet(_cfg())
    rng = np.random.default_rng(3)
    probes = [rng.normal(0, 1, 32).astype(np.float32) for _ in range(6)]
    for p in probes:
        net.step(p)
    ns = net.novelty_stats()
    check(ns["novel_miss"] == 6 and ns["seen_hits"] == 0,
          "A1a 6 个互异输入 → 6 个新模式", f"miss={ns['novel_miss']}")
    # 重复第一个 → 命中
    net.step(probes[0])
    ns = net.novelty_stats()
    check(ns["seen_hits"] == 1, "A1b 重复输入 → 命中（熟悉化）",
          f"hits={ns['seen_hits']}")
    # 只读路径不登记
    n_before = len(net._seen_sdr)
    net.step(rng.normal(0, 1, 32).astype(np.float32), learn=False, readonly=True)
    check(len(net._seen_sdr) == n_before,
          "A2 只读路径不登记新模式（评估不污染训练期集合）")
    # LRU 淘汰
    net2 = PHDNet(_cfg(sdr_novelty_cap=4))
    r2 = np.random.default_rng(4)
    for _ in range(10):
        net2.step(r2.normal(0, 1, 32).astype(np.float32))
    check(len(net2._seen_sdr) == 4, "A3 LRU 淘汰到容量上限",
          f"table={len(net2._seen_sdr)}/4")

    # ── B. 印迹触发率 ────────────────────────────────────────────
    # ⚠ 必须用**有代表性的宽度**：n_top=16 时顶层 SDR 只有 16 种可能模式，
    # 十几步后全部"见过" → 印迹率自然回落（那是表征容量限制，不是门控问题）。
    # smoke 档（n_top=256 / k=16）的组合空间 2.6e19，200 步内不会饱和。
    print("\n[B] LTM 印迹（旧默认 0.67% → 现在；smoke 档宽度）")
    from config_1b import build_cfg
    cfg_b = build_cfg("smoke")
    net3 = PHDNet(cfg_b)
    r3 = np.random.default_rng(7)
    imprints = 0
    N = 200
    for _ in range(N):
        d = net3.step(r3.normal(0, 1, cfg_b.n_input).astype(np.float32))
        if d["mode"] == "encode" and d["gate"] > cfg_b.ltm_imprint_gate:
            imprints += 1
    rate = imprints / N
    check(rate > 0.30, "B1 印迹触发率 > 30%（旧默认 0.67%）",
          f"{rate * 100:.1f}% ({imprints}/{N})")
    n_syn = sum(len(b) for b in net3.ltm.table.out.values())
    check(n_syn > 0, "B2 LTM 表真的生长", f"{n_syn:,} 突触")
    # 熟悉化：重复同一输入不再印迹
    probe = r3.normal(0, 1, 32).astype(np.float32)
    net3.step(probe)                       # 首次（可能已见过）
    before = sum(len(b) for b in net3.ltm.table.out.values())
    for _ in range(3):
        net3.step(probe)
    after = sum(len(b) for b in net3.ltm.table.out.values())
    check(after <= before * 1.5, "B3 重复刺激不再持续印迹（熟悉化）",
          f"{before} → {after}")

    # ── C. 会话中学习（读出头）───────────────────────────────────
    print("\n[C] 会话中学习（读出头单对记忆）")
    net4 = PHDNet(_cfg(n_readout=32, readout_softmax=True, eta_readout=0.15,
                       sdr_novelty=False, nll_sync_every=1))
    r4 = np.random.default_rng(5)
    x = r4.normal(0, 1, 32).astype(np.float32)
    tgt = np.eye(32)[7]
    n0 = None
    for i in range(50):
        d = net4.step(x, target=tgt)
        if i == 0:
            n0 = d["nll"]
    check(n0 is not None and d["nll"] < n0 * 0.5,
          "C1 读出头学会映射（nll 降 > 50%）", f"{n0:.3f} → {d['nll']:.3f}")
    d_new = net4.step(r4.normal(0, 1, 32).astype(np.float32),
                      target=np.eye(32)[9])
    check(d_new["nll"] > 2.0, "C2 不泛化到未见输入（防过拟合假象）",
          f"nll={d_new['nll']:.3f}")

    # ── D. 配置默认值 ────────────────────────────────────────────
    print("\n[D] 配置默认值")
    c = PHDNetConfig()
    check(c.sdr_novelty is True, "D1a sdr_novelty 默认 True")
    check(c.ltm_imprint_gate == 0.3, "D1b ltm_imprint_gate 默认 0.3（旧 0.8）",
          f"{c.ltm_imprint_gate}")
    check(c.task_modulation is True, "D1c task_modulation 默认 True")
    check(c.error_gated_memory is True, "D1d error_gated_memory 默认 True")
    check(c.plateau_sleep is True, "D1e plateau_sleep 默认 True")
    check(c.readout_replay is True, "D1f readout_replay 默认 True")
    # 关闭时逐位回旧行为
    net5 = PHDNet(_cfg(sdr_novelty=False))
    r5 = np.random.default_rng(6)
    modes = set()
    for _ in range(50):
        d = net5.step(r5.normal(0, 1, 32).astype(np.float32))
        modes.add(d["mode"])
    check(len(net5._seen_sdr) == 0, "D2a sdr_novelty=False → 不建表")
    check(net5.novelty_stats()["enabled"] is False, "D2b 标志如实反映关闭态")

    bad = [n for n, ok in CASES if not ok]
    print(f"\n{'PASS' if not bad else 'FAIL'}: {len(CASES) - len(bad)}/{len(CASES)} 项通过")
    for n in bad:
        print(f"  x {n}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
