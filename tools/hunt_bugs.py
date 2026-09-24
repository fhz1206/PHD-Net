"""Bug 猎手：系统化扫描（编译 / 开关矩阵 / 边界输入 / 数值健全 / 接口一致性）。

设计：用小配置跑「短路径」以覆盖尽可能多的代码分支，任何崩溃、NaN/Inf、
接口不一致都视为 bugs 并逐条打印（便于修复与复测）。

用法：python tools/hunt_bugs.py
"""
from __future__ import annotations

import itertools
import py_compile
import sys
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.config import PHDNetConfig          # noqa: E402
from phdnet.model import PHDNet, count_params   # noqa: E402

BUGS: list[str] = []
OK: list[str] = []


def bug(msg: str) -> None:
    BUGS.append(msg)
    print(f"  [BUG] {msg}")


def ok(msg: str) -> None:
    OK.append(msg)


SMALL = dict(n_sdr=64, k_sparse=8, n_mid=64, n_top=64,
             eta_pc=0.02, eta_oja=0.02, eta_stdp=0.02, seed=11,
             pred_in_readout=True)

# 单个开关（每项独立开启）+ 关键组合
SINGLE = [
    ("sparse_pc", dict(sparse_pc=True)),
    ("sparse_conn", dict(sparse_conn=True)),
    ("big_ltm", dict(big_ltm=True, big_ltm_N=1 << 16)),
    ("big_ltm+csr", dict(big_ltm=True, big_ltm_N=1 << 16, csr_online=True, csr_grow_chunk=4)),
    ("big_ltm+int8", dict(big_ltm=True, big_ltm_N=1 << 16, sparse_int8=True)),
    ("readout_fp32", dict(readout_fp32=True)),
    ("readout_conn_k", dict(readout_conn_k=4)),
    ("lognormal_init", dict(lognormal_init=True, exc_ratio=0.8)),
    ("minibatch", dict(minibatch_size=4)),
    ("recur", dict(readout_recurrence=True, recur_gain=0.5)),
    ("segcheck", dict(segment_check=10)),
    ("homeostasis", dict(homeostasis=True)),
    ("stdp_homeo", dict(stdp_homeostasis=True)),
    ("multi_mod", dict(multi_modulation=True)),
    ("target_rate", dict(neuron_target_rate=True)),
    ("critical", dict(critical_period=True)),
    ("wm_summary", dict(wm_summary_every=5)),
    ("learn_enc", dict(learnable_encoder=True)),
    ("pc_pred", dict(pc_predictive_target=True)),
    ("err_retrieval", dict(error_triggered_retrieval=True)),
    ("retrieval_topk", dict(retrieval_topk=4)),
    ("adaptive_lr", dict(adaptive_lr=True)),
    ("episodic", dict(episodic_len=4)),
    ("plateau_sleep", dict(plateau_sleep=True, plateau_patience=1)),
    ("growth", dict(growth_guidance=True)),
    ("wm_content", dict(wm_content_address=True)),
    ("readout_replay", dict(readout_replay=True, replay_every=2)),
    ("sparse_pc+topk", dict(sparse_pc=True, pc_topk=8)),
]

COMBO = [
    ("all_brain_sparse", dict(sparse_conn=True, conn_k=8, big_ltm=True, big_ltm_N=1 << 16,
                              csr_online=True, readout_conn_k=4, lognormal_init=True)),
    ("sparse+plastic", dict(sparse_conn=True, homeostasis=True, pc_dev_steps=2,
                            readout_replay=True, auto_development=True)),
    ("mlp_all", dict(minibatch_size=2, readout_fp32=True, readout_recurrence=True,
                     segment_check=5, multi_modulation=True)),
]


def smoke(name: str, ov: dict, steps: int = 6) -> None:
    try:
        cfg = PHDNetConfig(**{**SMALL, **ov})
        net = PHDNet(cfg)
        rng = np.random.default_rng(3)
        for t in range(steps):
            # 注意：x 维度 = cfg.n_input；target 维度 = net.n_out（读出输出维度）
            x = np.zeros(cfg.n_input)
            x[: max(1, cfg.n_input // 3)] = 1.0
            tgt = np.zeros(net.n_out)
            tgt[(t * 7) % net.n_out] = 1.0
            d = net.step(x, target=tgt, learn=True)
            for key in ("nll", "rate", "r2"):
                v = np.asarray(d[key])
                if not np.all(np.isfinite(v)):
                    bug(f"{name}: 第 {t} 步 `{key}` 出现 NaN/Inf")
                    return
        ok(f"{name}")
    except Exception as e:
        bug(f"{name}: 异常 {e!r}\n{''.join(traceback.format_exc().splitlines(True)[-4:])}")


def main() -> None:
    print("=" * 84)
    print("1) 编译检查（全部 .py）")
    print("=" * 84)
    n_py = 0
    for d in ("phdnet", "tests", "tools"):
        for py in sorted((_ROOT / d).glob("*.py")):
            n_py += 1
            try:
                py_compile.compile(str(py), doraise=True)
            except Exception as e:
                bug(f"编译失败 {py.name}: {e}")
    print(f"  检查 {n_py} 个文件；编译错误 {len([b for b in BUGS if '编译失败' in b])} 个")

    print("\n" + "=" * 84)
    print("2) 开关矩阵冒烟（单开 28 项 + 组合 3 项，各 6 步）")
    print("=" * 84)
    for name, ov in SINGLE:
        smoke(name, ov)
    for name, ov in COMBO:
        smoke(name, ov)

    print("\n" + "=" * 84)
    print("3) 边界输入")
    print("=" * 84)
    # 边界输入（注意：x 维度 = cfg.n_input）
    try:
        cfg = PHDNetConfig(**SMALL)
        net = PHDNet(cfg)
        d = net.step(np.zeros(cfg.n_input), target=np.zeros(net.n_out), learn=True)
        if not np.all(np.isfinite(d["nll"])):
            bug("全零输入：nll 非有限")
        else:
            ok("全零输入（含全零 target）")
    except Exception as e:
        bug(f"全零输入异常: {e!r}")
    # 极端幅值
    try:
        cfg = PHDNetConfig(**SMALL)
        net = PHDNet(cfg)
        x = np.full(cfg.n_input, 1e6)
        d = net.step(x, target=np.ones(net.n_out) / net.n_out, learn=True)
        if not np.all(np.isfinite(d["r2"])):
            bug("极端幅值输入：r2 非有限")
        else:
            ok("极端幅值输入 1e6")
    except Exception as e:
        bug(f"极端幅值异常: {e!r}")
    # 契约校验本身应给出明确错误（而非 numpy 广播错误）
    try:
        cfg = PHDNetConfig(**SMALL)
        net = PHDNet(cfg)
        try:
            net.step(np.zeros(cfg.n_input), target=np.zeros(cfg.n_top), learn=True)
            if cfg.n_top != net.n_out:
                bug("维度契约校验未触发（应抛明确 ValueError）")
            else:
                ok("target 维度契约（n_top == n_out，无需报错）")
        except ValueError as ve:
            if "读出输出维度" in str(ve):
                ok("错误 target 维度 → 明确契约错误")
            else:
                bug(f"契约校验信息不明确: {ve}")
    except Exception as e:
        bug(f"契约校验检查异常: {e!r}")
    # 空 spec 的评估路径
    try:
        from phdnet.word_lm import PHDWordLM
        seg = dict(max_len=4, min_count=2, min_entropy=0.5)
        lm = PHDWordLM("", PHDNetConfig(**SMALL), seg_kwargs=seg)
        ok("空语料构造（不崩溃）")
    except Exception as e:
        bug(f"空语料异常: {e!r}")

    print("\n" + "=" * 84)
    print("4) 接口一致性")
    print("=" * 84)
    # 4a. count_params 与各子模块口径一致（稀疏/稠密两种主干）
    for tag, ov in (("dense", {}), ("sparse_conn", dict(sparse_conn=True, conn_k=8)),
                    ("big_ltm", dict(big_ltm=True, big_ltm_N=1 << 16)),
                    ("sparse_conn+big_ltm", dict(sparse_conn=True, conn_k=8,
                                                 big_ltm=True, big_ltm_N=1 << 16)),
                    ("readout_k4", dict(readout_conn_k=4))):
        try:
            net = PHDNet(PHDNetConfig(**{**SMALL, **ov}))
            n = count_params(net)
            if not isinstance(n, int) or n <= 0:
                bug(f"count_params({tag}) = {n}（应为正整数）")
            else:
                ok(f"count_params({tag}) = {n:,}")
        except Exception as e:
            bug(f"count_params({tag}) 异常: {e!r}")
    # 4b. readout.n_synapses 与 stats 一致
    try:
        from phdnet.readout import Readout
        r = Readout(32, 16, np.random.default_rng(0), conn_k=4)
        st = r.stats()
        if st["synapses"] != r.n_synapses() or st["synapses"] != 16 * 4:
            bug(f"readout.stats 不一致: {st} vs n_synapses={r.n_synapses()}")
        else:
            ok("readout.stats 与 n_synapses 一致")
    except Exception as e:
        bug(f"readout.stats 异常: {e!r}")
    # 4c. SparsePCStack 的 W property 与 CSR 一致
    try:
        from phdnet.sparse_pc import SparsePCStack
        sp = SparsePCStack(32, 16, 8, 0.02, 0.02, np.random.default_rng(0), conn_k=4)
        ip, idx, val = sp.up0
        W = sp.W_up0
        mismatch = 0
        for i in range(16):
            for p in range(ip[i], ip[i + 1]):
                if W[i, idx[p]] != val[p]:
                    mismatch += 1
        if mismatch:
            bug(f"SparsePCStack.W_up0 视图与 CSR 不一致（{mismatch} 处）")
        else:
            ok("SparsePCStack.W_* 只读视图与 CSR 一致")
    except Exception as e:
        bug(f"SparsePCStack 视图异常: {e!r}")
    # 4d. wm.summarize 在两种 LTM 下均可用（契约分派正确）
    for tag, ov in (("dense_ltm", {}), ("sparse_ltm", dict(big_ltm=True, big_ltm_N=1 << 16))):
        try:
            cfg = PHDNetConfig(**{**SMALL, **ov, "wm_summary_every": 2, "n_wm_slots": 4})
            net = PHDNet(cfg)
            for t in range(6):
                x = np.zeros(cfg.n_input)
                x[:2] = 1.0
                net.step(x, target=None, learn=True)
            ok(f"wm.summarize（{tag}）")
        except Exception as e:
            bug(f"wm.summarize（{tag}）异常: {e!r}")
    # 4e. 可复现性：两次同 seed 构造的初始权重应一致
    try:
        a = PHDNet(PHDNetConfig(**SMALL))
        b = PHDNet(PHDNetConfig(**SMALL))
        pairs = [("pc.W_up0", a.pc.W_up0, b.pc.W_up0),
                 ("pc.W_dn0", a.pc.W_dn0, b.pc.W_dn0),
                 ("readout.W", a.readout.W, b.readout.W),
                 ("encoder.W", a.encoder.W, b.encoder.W)]
        bad = [nm for nm, x, y in pairs
               if not np.array_equal(np.asarray(x), np.asarray(y))]
        if bad:
            bug(f"同 seed 两次构造权重不一致：{bad}")
        else:
            ok("同 seed 构造可复现（pc/readout/encoder 权重一致）")
    except Exception as e:
        bug(f"可复现性检查异常: {e!r}")

    print("\n" + "=" * 84)
    print(f"结果：通过 {len(OK)} 项 ｜ 发现 bugs {len(BUGS)} 个")
    print("=" * 84)
    for b in BUGS:
        print("  •", b.splitlines()[0])
    sys.exit(1 if BUGS else 0)


if __name__ == "__main__":
    main()
