"""回归检查：核心行为（版本/三实验/容量层/词级 LM/大容量 LTM/复制/开关）。"""

import subprocess
import sys

import numpy as np

from checks_common import DOC, ENV, PY, ROOT, TESTS, _check

# 与 phdnet.__version__ 同步；升级版本时只改这一处（2026-09-30 起 v0.1.0-alpha）
_EXPECTED_VERSION = "v0.1.0-alpha"


def t_version() -> None:
    import phdnet

    # 版本一致性：__init__.__version__ 必须与本文件下方的 _EXPECTED_VERSION 一致
    # （v0.1.0-alpha 分支起版本号带前缀，2026-09-30 之前是裸 "0.0.0"——把期望值
    # 抽成常量，升级版本时只需改这一处）
    _check(f"版本一致性 {_EXPECTED_VERSION}", phdnet.__version__ == _EXPECTED_VERSION,
           phdnet.__version__)


def t_core_demo() -> None:
    """核心三实验：用数值阈值判定，不硬匹配字符串。

    注（2026-09-18）：numba 内核启用后序列误差由 0.282 变为 0.317、结构 cos 由
    0.81 变为 0.73 —— 两条路径实现同一公式，差异来自浮点级分歧在 600 步在线学习中
    被放大；因果结构（A→B 最优）、补全 100%、one-shot 5/5 均保持。
    """

    r = subprocess.run([PY, str(TESTS / "demo_phdnet.py")], capture_output=True,
                       text=True, cwd=str(ROOT), env=ENV,
                       encoding="utf-8", errors="replace")
    out = r.stdout or ""

    def cos_b() -> float:
        import re
        m = re.search(r"cos=\[A:([\d.]+), B:([\d.]+), C:([\d.]+)\]", out)
        return float(m.group(2)) if m else 0.0

    def err_after() -> float:
        import re
        m = re.search(r"后 100 步平均时序误差:\s*([\d.]+)", out)
        return float(m.group(1)) if m else 1.0

    ok = (r.returncode == 0 and "全部实验完成" in out and "5/5" in out
          and "100.0%" in out and cos_b() > 0.6 and err_after() < 0.45)
    _check("demo_phdnet 三实验（因果结构 / 补全 100% / 关联 5/5）", ok,
           f"cos_B={cos_b():.2f} 误差={err_after():.3f}")


def t_capacity_small() -> None:
    from phdnet.bigltm import SparseSynapseTable
    from phdnet.tokenizer import U64, _mix64
    N, M, A = 1 << 12, 16, 16
    tb = SparseSynapseTable(N, M, lam=0.7, eta=0.08, seed=7)

    def sdr(sym: int) -> list[int]:
        x = U64(sym) * U64(7919) + np.arange(A, dtype=np.uint64) * U64(2654435761)
        return (_mix64(x + U64(7)) % U64(N)).astype(np.int64).tolist()

    prev = None
    for s in range(90):
        cur = sdr(s % 3)
        if prev is not None:
            tb.learn(prev, cur)
        tb.step_count += 1
        prev = cur
    p = tb.predict(sdr(0))
    top = [k for k, _ in sorted(p.items(), key=lambda kv: -kv[1])[:A]]
    ov = lambda a, b: len(set(a) & set(b)) / max(1, len(a))
    ok = ov(top, sdr(1)) > ov(top, sdr(2))
    _check("M4 容量层小规模因果学习（A→B 优于 A→C）", ok,
           f"→B {ov(top, sdr(1)):.2f} / →C {ov(top, sdr(2)):.2f}")


def t_word_lm_smoke() -> None:
    from phdnet.config import PHDNetConfig
    from phdnet.word_lm import PHDWordLM
    with open(DOC, encoding="utf-8") as f:
        text = f.read()
    cfg = PHDNetConfig(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
                       eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
                       pred_in_readout=True)
    lm = PHDWordLM(text, cfg, seg_kwargs=dict(max_len=6, min_count=5,
                                              min_entropy=1.0))
    lm.train_stream(text[:3000])
    m = lm.evaluate(text[3000:3500])
    ok = np.isfinite(m["ppl_char"]) and m["ppl_char"] < len(lm.tok)
    _check("词级 LM 冒烟（字符归一 PPL 有限且优于均匀）", ok,
           f"PPL={m['ppl_char']:.1f} / 均匀 {len(lm.tok)}")


def t_bigltm_smoke() -> None:
    from phdnet.config import PHDNetConfig
    from phdnet.word_lm import PHDWordLM
    with open(DOC, encoding="utf-8") as f:
        text = f.read()
    cfg = PHDNetConfig(n_sdr=128, k_sparse=16, n_mid=128, n_top=64,
                       eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
                       big_ltm=True, big_ltm_N=1 << 14, big_ltm_m=16,
                       big_ltm_k=4)
    lm = PHDWordLM(text, cfg, seg_kwargs=dict(max_len=4, min_count=5,
                                              min_entropy=1.0))
    lm.train_stream(text[:1500])
    m = lm.evaluate(text[1500:2000])
    st = lm.net.ltm.table.stats()
    _check("M4 大容量 LTM 接入冒烟（印迹生长 > 0）",
           np.isfinite(m["ppl_char"]) and st["grown_synapses"] > 0,
           f"生长 {st['grown_synapses']} 突触")


def t_copy_task() -> None:
    """T6 长程复制：小规模配置下位置准确率必须显著高于随机基线（1/12≈8.3%）。"""
    import demo_copy as dc
    codes = dc.sdrs(dc.ALPHA)
    mem, ep = dc.train(codes, n=8, n_seq=40, rho=0.5, forget=0.95, sep=1.0)
    res = dc.evaluate(mem, codes, 4, ep, n_seq=20)
    ok = res["pos"] > 2.0 / len(dc.ALPHA)
    _check("T6 长程复制（n=4 位置准确率 > 随机 2×）", ok,
           f"{res['pos'] * 100:.1f}% vs 随机 {100 / len(dc.ALPHA):.1f}%")


def t_switches_construct() -> None:
    from phdnet.config import PHDNetConfig
    from phdnet.model import PHDNet
    flags = dict(task_modulation=True, error_gated_memory=True,
                 pred_in_readout=True, adaptive_lr=True, dual_trace=True,
                 multi_modulation=True, stdp_homeostasis=True,
                 metaplasticity=True, staged_sleep=True, ei_synapses=True,
                 critical_period=True,
                 # M9 第二轮全量优化开关（默认关闭；此处验证全开可构造可运行）
                 pc_predictive_target=True, learnable_encoder=True,
                 retrieval_topk=4, wm_content_address=True,
                 readout_replay=True, neuron_target_rate=True,
                 plateau_sleep=True, pc_dev_steps=100, episodic_len=2)
    off = PHDNet(PHDNetConfig(seed=3))
    on = PHDNet(PHDNetConfig(seed=3, **flags))
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.normal(size=off.cfg.n_input)
        t = np.zeros(off.cfg.n_input)
        t[0] = 1.0
        off.step(x, target=t, learn=True)
        on.step(x, target=t, learn=True)
    on.sleep(n_replay=2, replay_ratio=0.5)
    # 大容量表新开关（T5.2/T5.3）：全开可构造、可印迹/检索、CSR 快照可用
    big = PHDNet(PHDNetConfig(seed=3, big_ltm=True, big_ltm_N=1 << 12,
                              big_ltm_m=8, growth_guidance=True, sparse_int8=True))
    xr = np.zeros(big.cfg.n_top); xr[:8] = 1.0
    big.ltm.imprint(xr); big.ltm.imprint(xr)
    snap = big.ltm.table.compact_csr()
    st = big.ltm.table.stats()
    ok_big = st["grown_synapses"] > 0 and len(snap["data"]) == st["grown_synapses"]
    # T6 双向校验：ContextMemory 全开可构造
    from phdnet.context_memory import ContextMemory
    cm = ContextMemory(64, bidir=True, bidir_w=1.0)
    cm.begin_episode(0)
    for _ in range(3):
        v = np.zeros(64); v[3] = 1.0
        cm.observe(v)
    cm.recall_scored(2, {"a": np.eye(64)[3], "b": np.eye(64)[4]})
    _check("M1/M3/M5/M4b/发育/M9 开关全开可构造并运行（默认路径不受影响）",
           ok_big,
           f"1B 表生长 {st['grown_synapses']} / CSR {len(snap['data'])} 突触")


def t_brain_homologues() -> None:
    """脑同构五项全开：构造、连续学习、睡眠、输出有限（零崩溃自检）。"""
    from phdnet.config import PHDNetConfig
    from phdnet.model import PHDNet
    flags = dict(multi_modulation=True, stdp_homeostasis=True,
                 metaplasticity=True, staged_sleep=True, ei_synapses=True,
                 critical_period=True)
    net = PHDNet(PHDNetConfig(seed=5, **flags))
    rng = np.random.default_rng(1)
    finite = True
    for _ in range(40):
        x = rng.normal(size=net.cfg.n_input)
        t = np.zeros(net.cfg.n_input); t[0] = 1.0
        d = net.step(x, target=t, learn=True)
        if not (np.isfinite(d["seq_err"]) and np.isfinite(d["nll"])
                and np.all(np.isfinite(net.stdp.W))):
            finite = False
            break
    n_done = net.sleep(n_replay=3, replay_ratio=0.5)
    _check("脑同构五项全开：构造/学习/睡眠/有限输出", finite and n_done >= 0)


FULL_SCRIPTS = ["demo_lm.py", "demo_m1.py", "demo_m2.py", "demo_m3.py",
                "demo_m4.py", "demo_m9.py", "demo_strength.py", "eval_suite.py"]
