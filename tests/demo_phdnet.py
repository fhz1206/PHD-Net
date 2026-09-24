"""PHD-Net 演示 —— 三个实验验证架构核心能力（架构文档 §5 推理/学习流程）。

实验一 序列预测   ：M1+M2+M3+M5 全链路在线学习，验证 STDP 时序因果学习
实验二 模式补全   ：M4b 吸引子检索，验证一次性印迹 + 快/慢双速率记忆
实验三 少样本关联 ：M4b 非对称印迹 + M5 调制门控，验证"先见者学，重复者忘"

运行：python demo_phdnet.py  （Python 3.14，依赖 numpy；numba 可选加速）
"""

# --- 目录结构调整（2026-09-18）：脚本位于 tests/ 或 tools/ 子目录 ---
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]     # 项目根目录
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))              # 保证 `import phdnet` 可用
_DOC = _ROOT / "datasets" / "eval" / "internal_corpus.txt"    # 默认语料
# --- 引导结束 ---

import time

import numpy as np

from phdnet.config import PHDNetConfig
from phdnet.model import PHDNet
from phdnet.memory import LongTermMemory
from phdnet.modulator import Neuromodulator
from phdnet.plasticity import NUMBA_OK as NUMBA_AVAILABLE

BLOCKS = "▁▂▃▄▅▆▇█"


def spark(vals, width=40):
    """误差序列 → ASCII 迷你趋势图。"""
    step = max(1, len(vals) // width)
    m = [sum(vals[i:i + step]) / step for i in range(0, len(vals), step)]
    lo, hi = min(m), max(m)
    return "".join(BLOCKS[int((v - lo) / (hi - lo + 1e-9) * 7)] for v in m)


def sparse_symbol(n, k, rng):
    """生成随机稀疏符号码（SDR 风格，非负）。"""
    s = np.zeros(n)
    idx = rng.choice(n, k, replace=False)
    s[idx] = rng.uniform(0.3, 1.0, k)
    return s


# ─────────────────────────── 实验一：序列预测（全链路） ───────────────────────────
def exp1_sequence(n_steps=600):
    print("─" * 62)
    print("实验一  序列预测（M1 稀疏编码 → M2 预测编码 → M3 STDP → M5 调制）")
    rng = np.random.default_rng(0)
    sym = [sparse_symbol(64, 16, rng) for _ in range(3)]          # A B C 三个符号码
    # 顶层 256 维、发放率稀疏度 ~6%：不同符号的支撑集几乎不重叠（SDR 区分性）。
    # PC 表征冻结（固定随机投影，类基因预连接），学习聚焦 M3 关联核。
    cfg = PHDNetConfig(n_input=64, n_sdr=256, k_sparse=32,
                       n_mid=256, n_top=256, eta_pc=0.0, eta_oja=0.0,
                       eta_stdp=0.12, seed=11)
    net = PHDNet(cfg)
    errs = []
    t0 = time.perf_counter()
    for t in range(n_steps):                                      # 在线学习：ABCABC…
        x = sym[t % 3]
        d = net.step(x, learn=True)
        errs.append(d["seq_err"])
    dt = time.perf_counter() - t0
    early = sum(errs[:30]) / 30
    late = sum(errs[-100:]) / 100
    print(f"  前 30 步平均时序误差 : {early:.3f}")
    print(f"  后 100 步平均时序误差: {late:.3f}   （下降 {100 * (1 - late / early):.1f}%）")
    print(f"  误差趋势: {spark(errs)}")
    print(f"  {n_steps} 步在线学习耗时: {dt:.2f}s"
          f"（numba 加速: {'开启' if NUMBA_AVAILABLE else '自检未过 → numpy 回退'}）")

    # 结构检验：让"上一时刻"= A，预测下一步 —— 应该最像 B（学到 A→B→C→A 转移）
    def top_rate(x):
        s0, _ = net.encoder.encode(x)
        return net._rate(net.pc.infer(s0, net.cfg.n_infer_steps)["r2"])

    rates = [top_rate(s) for s in sym]
    pred = net.stdp.predict(rates[0])          # W·rate(A) → 预测 A 之后应出现的符号
    cos = [float(pred @ rt / (np.linalg.norm(pred) * np.linalg.norm(rt) + 1e-9))
           for rt in rates]
    names = "ABC"
    best = int(np.argmax(cos))
    print(f"  结构检验: 给 A 预测下一步 → 最相似符号 = {names[best]}"
          f"  cos=[A:{cos[0]:.2f}, B:{cos[1]:.2f}, C:{cos[2]:.2f}]  "
          f"{'✓ 学到 A→B 因果' if best == 1 else '✗'}")
    return errs


# ─────────────────────────── 实验二：模式补全（双速率记忆） ───────────────────────────
def exp2_completion(n_pat=10, n_dim=256, flip=0.25):
    print("─" * 62)
    print("实验二  模式补全（M4b：一次性印迹 → 吸引子检索 → 睡眠巩固）")
    rng = np.random.default_rng(1)
    pats = rng.choice([-1.0, 1.0], size=(n_pat, n_dim))           # 平衡 ±1 记忆模式
    ltm = LongTermMemory(n_dim, eta_hip=0.05, eta_cortex=0.05,
                         beta_fast=0.6, beta_slow=1.0, n_steps=6)

    accs_immediate = []
    for p in pats:                                                # 每个模式只呈现一次
        ltm.imprint(p)                                            # 海马式 one-shot 印迹
        cue = p.copy()
        k = int(flip * n_dim)
        idx = rng.choice(n_dim, k, replace=False)
        cue[idx] *= -1                                            # 25% 位翻转的残缺线索
        rec = ltm.recall(cue)
        accs_immediate.append(float((rec == p).mean()))
    print(f"  印迹后立即检索（快权重主导）准确率: {np.mean(accs_immediate):.1%}")

    for _ in range(30):                                           # “睡眠”：巩固 30 次
        ltm.consolidate()
    ltm.W_fast *= 0.2                                             # 模拟海马旧痕消退
    accs_consolidated = []
    for p in pats:
        cue = p.copy()
        idx = rng.choice(n_dim, int(flip * n_dim), replace=False)
        cue[idx] *= -1
        rec = ltm.recall(cue)
        accs_consolidated.append(float((rec == p).mean()))
    print(f"  巩固+快权重消退后检索（慢权重接管）准确率: {np.mean(accs_consolidated):.1%}")
    print(f"  → 验证双速率记忆：快速印迹可被皮层慢权重巩固接管（CLS 理论）")


# ─────────────────────────── 实验三：少样本关联 + 调制门控 ───────────────────────────
def exp3_association(n_pairs=5, n_dim=128):
    print("─" * 62)
    print("实验三  少样本配对关联（M4b 非对称印迹 + M5 调制门控）")
    rng = np.random.default_rng(2)
    X = rng.choice([-1.0, 1.0], size=(n_pairs, n_dim))            # 线索模式
    Y = rng.choice([-1.0, 1.0], size=(n_pairs, n_dim))            # 目标模式
    ltm = LongTermMemory(n_dim, eta_hip=0.06, eta_cortex=0.02,
                         beta_fast=1.0, beta_slow=0.0, n_steps=1)

    hits = 0
    for j in range(n_pairs):
        ltm.imprint(Y[j], X[j])                                   # 单次呈现即建立 X→Y 关联
        y = np.sign(ltm.W_fast @ X[j])                            # 非对称单步检索
        cos = [float(y @ Y[i] / n_dim) for i in range(n_pairs)]
        if int(np.argmax(cos)) == j:
            hits += 1
    print(f"  每对仅呈现 1 次，检索命中率: {hits}/{n_pairs}"
          f"  （自注意力模型需完整训练流程；此处为在线 one-shot）")

    mod = Neuromodulator(gain=2.0)                                # 调制门控演示
    novelty = [1.0, 0.9, 0.5, 0.2, 0.1, 0.1]                      # 重复呈现同一刺激的新颖度
    gates = []
    for nv in novelty:
        gate, mode = mod.observe(nv)
        gates.append(gate)
    print(f"  同一刺激重复呈现的调制门: {' '.join(f'{g:.2f}' for g in gates)}")
    print(f"  → 意外大时门开（多学），重复后门关（少学）——“先见者学，重复者忘”")


# ─────────────────────────── numba 加速对比（可选） ───────────────────────────
def benchmark():
    if not NUMBA_AVAILABLE:
        print("─" * 62)
        print("numba 自检未通过（或未安装），已回退纯 numpy，跳过加速对比")
        return
    from phdnet.plasticity import _predict_edges, _predict_edges_np
    rng = np.random.default_rng(3)
    n, m = 4096, 64
    W = rng.random((n, m))
    post_idx = np.array([rng.choice(n, m, replace=False) for _ in range(n)])
    pre = rng.random(n)
    _predict_edges(W, post_idx, pre, n)                           # 触发编译
    reps = 300
    t0 = time.perf_counter()
    for _ in range(reps):
        _predict_edges(W, post_idx, pre, n)
    t_nb = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(reps):
        _predict_edges_np(W, post_idx, pre, n)
    t_np = time.perf_counter() - t0
    print("─" * 62)
    print(f"numba 加速对比（n={n} 神经元 × m={m} 出边 × {reps} 次）: "
          f"numba {t_nb * 1000:.1f}ms vs numpy {t_np * 1000:.1f}ms "
          f"→ {t_np / t_nb:.2f}x")


if __name__ == "__main__":
    print("PHD-Net 演示（预测编码 × Hebbian/STDP × 双记忆，无自注意力）")
    exp1_sequence()
    exp2_completion()
    exp3_association()
    benchmark()
    print("─" * 62)
    print("全部实验完成。")
