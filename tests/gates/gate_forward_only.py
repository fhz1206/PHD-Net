"""P210 门禁：前向only 训练路径（forward-only training path）的不变量。

跑法：py -3.14 tests/gates/gate_forward_only.py

判据（全部**可执行**，不是静态断言）：
  A 组｜固定随机投影（FP 内核）
    A1 确定性：同 (标签, 维度, 种子) → 逐位相同的随机码
    A2 区分性：不同标签 → 不同码；不同种子 → 不同码
    A3 值域：码经 tanh 后落在 (-1, 1)（与 M2 各层激活同值域）
    A4 无溢出告警：uint64 混合用 Python 整数取模（不得发 RuntimeWarning）
    A5 标签越界 → 明确报错（不静默纠正）
  B 组｜层目标学习（FTP 内核 + 局部匹配）
    B1 目标递推：沿模型自己的权重把目标递推到更浅一层（形状/值域正确）
    B2 局部性：层目标学习**只动下行生成权重**，上行 up0/up1 **逐位不变**
        （P198：eta_oja>0 会正反馈发散，上行必须只读）
    B3 稀疏性：只更新**已存在**的边（CSR 行宽不变），不新增连接
    B4 目标匹配方向：τ > h 时权重沿 pre 增大（下一层激活朝目标靠）
    B5 eta=0 → 权重逐位不变（无信号即无更新）
  C 组｜接线与零回归（铁律④）
    C1 默认关闭 → 不构造投影器（关闭时零开销、零行为变化）
    C2 关闭时训练轨迹与旧版**逐位一致**（同种子同摘要）
    C3 开启时训练轨迹**确实改变**（机制真的生效，不是空接线）
    C4 配置 fail-fast：非法 eta / gain 立即报错
    C5 CLI 开关存在且默认关闭（--forward-only）
"""

import hashlib
import subprocess
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "train")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                        # noqa: E402

from phdnet.config import PHDNetConfig                    # noqa: E402
from phdnet.forward_only import (ForwardTargetProjector,  # noqa: E402
                                 forward_target_delta, hash_codes)
from phdnet.model import PHDNet                           # noqa: E402

CASES: list = []


def check(ok: bool, name: str, detail: str = "") -> None:
    CASES.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))


def _digest(net: PHDNet) -> str:
    """主干 + 读出权重的摘要（用于轨迹对比）。"""
    h = hashlib.sha256()
    for a in (net.pc.dn0[2], net.pc.dn1[2], net.pc.up0[2], net.pc.up1[2],
              net.readout.W):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


def _mk_net(forward_only: bool = False, eta: float = 0.01, **kw):
    cfg = PHDNetConfig(
        n_input=32, n_sdr=64, n_mid=64, n_top=32, k_sparse=16,
        n_readout=50, readout_softmax=True, eta_readout=0.15,
        seed=11, step_profiling=False, accel_readout="cpu:0",
        forward_only=forward_only, forward_only_eta=eta, **kw)
    return PHDNet(cfg)


def _train(net: PHDNet, steps: int = 20) -> None:
    rng = np.random.default_rng(0)
    for i in range(steps):
        x = rng.standard_normal(32).astype(np.float32)
        tgt = np.zeros(50)
        tgt[i % 50] = 1.0
        net.step(x, target=tgt)


# ── A 组：固定随机投影 ─────────────────────────────────────────────────────
def group_a() -> None:
    print("\n[A] 固定随机投影（FP 内核）")

    a1 = hash_codes(7, 64, 11)
    a2 = hash_codes(7, 64, 11)
    check(np.array_equal(a1, a2), "A1a 同标签同种子 → 逐位相同")
    check(np.array_equal(hash_codes(7, 32, 11), hash_codes(7, 32, 11)),
          "A1b 不同维度各自确定")
    check(not np.array_equal(hash_codes(7, 64, 11), hash_codes(8, 64, 11)),
          "A2a 不同标签 → 不同码")
    check(not np.array_equal(hash_codes(7, 64, 11), hash_codes(7, 64, 12)),
          "A2b 不同种子 → 不同码")

    proj = ForwardTargetProjector(n_out=100, n_layers=(32, 16), seed=11)
    t = proj.project(5, 0)
    check(t.shape == (32,), "A3a 目标维度 = 该层维度", str(t.shape))
    check(float(t.min()) > -1.0 and float(t.max()) < 1.0,
          "A3b 目标值域 ⊂ (-1,1)（与 tanh 激活同值域）",
          f"[{t.min():.3f},{t.max():.3f}]")

    # A4：uint64 混合不得触发溢出告警（-W error 升级为失败）
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            hash_codes(51961, 64, 11)
            ok4 = True
            d4 = ""
        except RuntimeWarning as e:
            ok4 = False
            d4 = str(e)[:60]
    check(ok4, "A4 大标签 id 不触发 uint64 溢出告警", d4)

    try:
        proj.project(200, 0)
        check(False, "A5 标签越界 → 明确报错", "未报错")
    except ValueError:
        check(True, "A5 标签越界 → 明确报错")


# ── B 组：层目标学习 ───────────────────────────────────────────────────────
def group_b() -> None:
    print("\n[B] 层目标学习（FTP 内核 + 局部目标匹配）")

    net = _mk_net(forward_only=True)
    proj = net._fwd_projector
    tau_r1 = proj.project(5, 0)
    tau_s0 = proj.recurse(tau_r1, lambda v: net.pc._mv(net.pc.dn0, v))
    check(tau_r1.shape == (net.cfg.n_mid,), "B1a r1 目标维度 = n_mid",
          str(tau_r1.shape))
    check(tau_s0.shape == (net.cfg.n_sdr,), "B1b s0 目标维度 = n_sdr",
          str(tau_s0.shape))
    check(float(tau_s0.min()) > -1.0 and float(tau_s0.max()) < 1.0,
          "B1c 递推目标仍在 (-1,1)")

    # B2：只动下行，上行逐位不变
    net = _mk_net(forward_only=True)
    up0_before = net.pc.up0[2].copy()
    up1_before = net.pc.up1[2].copy()
    dn1_before = net.pc.dn1[2].copy()
    # 直接调层目标学习（eta=0）→ 应逐位不变
    net.pc.learn_forward_target({"s0": np.zeros(net.cfg.n_sdr),
                                  "r1": np.zeros(net.cfg.n_mid),
                                  "r2": np.zeros(net.cfg.n_top)},
                                 np.zeros(net.cfg.n_mid),
                                 np.zeros(net.cfg.n_sdr), 0.0)
    check(np.array_equal(net.pc.up0[2], up0_before)
          and np.array_equal(net.pc.up1[2], up1_before),
          "B2a eta=0 时上行权重逐位不变（P198 上行只读约束）")

    # 真实调用路径：跑一步带监督的 step，比较上行/下行
    net2 = _mk_net(forward_only=True)
    u0 = net2.pc.up0[2].copy()
    u1 = net2.pc.up1[2].copy()
    d1 = net2.pc.dn1[2].copy()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32).astype(np.float32)
    tgt = np.zeros(50)
    tgt[3] = 1.0
    net2.step(x, target=tgt)
    check(np.array_equal(net2.pc.up0[2], u0)
          and np.array_equal(net2.pc.up1[2], u1),
          "B2b 前向only 路径**不触碰上行识别权重**（P198）")
    check(not np.array_equal(net2.pc.dn1[2], d1),
          "B2c 前向only 路径确实更新了下行生成权重")

    # B3：稀疏性 —— CSR 行宽不变（不新增连接）
    net3 = _mk_net(forward_only=True)
    widths_before = [len(net3.pc.dn1[1]), len(net3.pc.dn0[1])]
    _train(net3, steps=5)
    widths_after = [len(net3.pc.dn1[1]), len(net3.pc.dn0[1])]
    check(widths_before == widths_after,
          "B3 只更新已存在的边（CSR 行宽不变，不新增连接）",
          f"{widths_before} → {widths_after}")

    # B4：目标匹配方向 —— τ > h 时该行权重沿 pre 增大
    h = np.zeros(3)
    tau = np.array([1.0, -1.0, 0.0])
    pre = np.array([1.0, 2.0])
    dw = forward_target_delta(h, tau, pre, eta=1.0)
    check(dw[0, 0] > 0, "B4a τ>h → 权重沿 pre 增大（激活朝目标靠）")
    check(dw[1, 0] < 0, "B4b τ<h → 权重沿 pre 减小")
    check(np.allclose(dw[:, 1], dw[:, 0] * 2.0),
          "B4c 增量 ∝ pre（外积语义）")

    # B5：eta=0 → 逐位不变
    net5 = _mk_net(forward_only=True, eta=0.0)
    before = net5.pc.dn1[2].copy()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32).astype(np.float32)
    tgt = np.zeros(50)
    tgt[3] = 1.0
    net5.step(x, target=tgt)
    # 注意：eta=0 只关掉前向only 这一项，PC 的无监督重构学习仍会动 dn1。
    # 故此处只断言「不因前向only 而额外改变」→ 与关闭开关的轨迹对比。
    net5b = _mk_net(forward_only=False)
    before_off = net5b.pc.dn1[2].copy()
    net5b.step(x, target=tgt)
    check(np.array_equal(net5.pc.dn1[2], net5b.pc.dn1[2]),
          "B5 eta=0 时前向only 零贡献（与关闭开关轨迹逐位相同）")


# ── C 组：接线与零回归 ────────────────────────────────────────────────────
def group_c() -> None:
    print("\n[C] 接线与零回归（P211：默认开启 + 旧路径已删）")

    # C1（P211 语义反转）：**默认开启** → 默认就构造投影器。
    net_def = PHDNetConfig(n_input=32, n_sdr=64, n_mid=64, n_top=32,
                           k_sparse=16, n_readout=50, readout_softmax=True,
                           eta_readout=0.15, seed=11, step_profiling=False,
                           accel_readout="cpu:0")
    check(net_def.forward_only is True,
          "C1a 默认配置 forward_only=True（P211 起默认开启）")
    net_def_net = PHDNet(net_def)
    check(net_def_net._fwd_projector is not None,
          "C1b 默认配置即构造投影器（forward-only 已是默认路径）")

    net_off = _mk_net(forward_only=False)
    check(net_off._fwd_projector is None,
          "C1c 显式关闭 → 不构造投影器（零开销，门禁/实验用）")

    net_a = _mk_net(forward_only=False)
    _train(net_a)
    net_b = _mk_net(forward_only=False)
    _train(net_b)
    d_a, d_b = _digest(net_a), _digest(net_b)
    check(d_a == d_b, "C2a 关闭时同种子两次训练逐位一致（可复现）",
          d_a[:12])

    net_on = _mk_net(forward_only=True)
    _train(net_on)
    d_on = _digest(net_on)
    check(d_on != d_a, "C3 开启时训练轨迹确实改变（机制真的生效）",
          f"off={d_a[:12]} on={d_on[:12]}")

    # C3b（P211 新契约）：**eta=0 等价于旧行为** —— 旧路径已不再是并列模式，
    # 而是 forward-only 的特例。这条断言锁住「删旧路径」的实现方式。
    net_eta0 = _mk_net(forward_only=True, eta=0.0)
    _train(net_eta0)
    d_eta0 = _digest(net_eta0)
    check(d_eta0 == d_a,
          "C3b eta=0 的 forward-only 轨迹 == 旧路径轨迹（逐位相同）",
          f"eta0={d_eta0[:12]} old={d_a[:12]}")

    # C4：配置 fail-fast
    try:
        PHDNetConfig(forward_only=True, forward_only_eta=-0.1)
        check(False, "C4a 非法 eta → 立即报错", "未报错")
    except ValueError:
        check(True, "C4a 非法 eta → 立即报错")
    try:
        PHDNetConfig(forward_only=True, forward_only_gain=0.0)
        check(False, "C4b 非法 gain → 立即报错", "未报错")
    except ValueError:
        check(True, "C4b 非法 gain → 立即报错")

    # C5：CLI 开关（P211：默认开，旧开关删除 → 只剩 --no-forward-only）
    r = subprocess.run([sys.executable, str(_ROOT / "train" / "train.py"),
                        "--help"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    check("--no-forward-only" in r.stdout,
          "C5a CLI 存在 --no-forward-only（默认开启，故只保留关闭开关）")
    check("--forward-only-eta" in r.stdout, "C5b CLI 存在 --forward-only-eta")
    check("--forward-only " not in r.stdout,
          "C5c 旧的 --forward-only 开关已删除（旧路径不再是并列模式）")


def main() -> None:
    print("=" * 68)
    print("P210 门禁：前向only 训练路径（forward-only training path）")
    print("=" * 68)
    group_a()
    group_b()
    group_c()
    n_ok = sum(1 for _, ok in CASES if ok)
    print("-" * 68)
    print(f"RESULT {n_ok}/{len(CASES)} passed")
    if n_ok != len(CASES):
        print("FAILED: " + ", ".join(n for n, ok in CASES if not ok))
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
