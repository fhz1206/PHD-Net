"""P5 验收：WM 压缩摘要槽 + 多跳关联（长间隔因果对）。

协议：5 对 (X_i → Y_i)。呈现 X_i 后插入 12 个干扰模式再呈现 Y_i；
Y 呈现时刻用 WM 内容与 Y 建立非对称关联。测试：给 X 检索 Y。
对照 = 无摘要槽（X 痕迹被 12 步干扰挤出/稀释）；
实验组 = X 呈现后立即 summarize()（保留槽不衰减）。
命中判定：检索结果与 Y_i 的相似度高于其他 Y。
"""

# --- 目录结构调整（2026-09-18）：脚本位于 tests/ 或 tools/ 子目录 ---
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]     # 项目根目录
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))              # 保证 `import phdnet` 可用
_DOC = _ROOT / "datasets" / "eval" / "internal_corpus.txt"    # 默认语料
# --- 引导结束 ---

import numpy as np

from phdnet.memory import LongTermMemory, WorkingMemory

N, N_PAIRS, N_DISTRACT = 256, 5, 12


def cue_of(p, rng, flip=0.2):
    c = p.copy()
    idx = rng.choice(len(p), int(flip * len(p)), replace=False)
    c[idx] *= -1
    return c


def run(use_summary: bool, seed: int = 9) -> float:
    rng = np.random.default_rng(seed)
    X = rng.choice([-1.0, 1.0], size=(N_PAIRS, N))
    Y = rng.choice([-1.0, 1.0], size=(N_PAIRS, N))
    D = rng.choice([-1.0, 1.0], size=(N_DISTRACT, N))
    ltm = LongTermMemory(n=N, eta_hip=0.05, eta_cortex=0.0,
                         beta_fast=1.0, beta_slow=0.0, n_steps=6)
    for p in list(X) + list(Y) + list(D):        # 自联想：保证检索可收敛到模式
        ltm.imprint(p)
    wm = WorkingMemory(n=N, n_slots=4, gamma=0.85)
    W_pair = np.zeros((N, N))                    # 非对称关联 X→Y

    for i in range(N_PAIRS):
        wm.decay()
        wm.write(X[i].astype(float), 1.0, 0.0)   # 线索呈现
        if use_summary:
            wm.summarize(ltm)                    # P5：摘要槽固定线索
        for d in D:                              # 12 步干扰
            wm.decay()
            wm.write(d.astype(float), 1.0, 0.0)
        wm.decay()
        wm.write(Y[i].astype(float), 1.0, 0.0)   # 目标呈现：以 WM 内容建立关联
        W_pair += 0.08 * np.outer(Y[i], wm.read())

    hits = 0
    for i in range(N_PAIRS):
        x_clean = ltm.recall(cue_of(X[i], rng))  # 线索补全
        y = np.sign(W_pair @ x_clean)
        cos = [float(y @ Y[j] / N) for j in range(N_PAIRS)]
        if int(np.argmax(cos)) == i:
            hits += 1
    return hits / N_PAIRS


if __name__ == "__main__":
    print("=" * 62)
    print("P5 多跳关联验收（线索与目标间隔 12 步干扰）")
    print("=" * 62)
    acc_ctrl = np.mean([run(False, s) for s in range(3)])
    acc_summ = np.mean([run(True, s) for s in range(3)])
    print(f"对照（无摘要槽）命中率: {acc_ctrl:.0%}")
    print(f"实验组（摘要槽）命中率: {acc_summ:.0%}")
    ok = acc_summ > 0.6 and acc_summ > acc_ctrl
    print("-" * 62)
    print(f"P5 结论: {'✓ 通过' if ok else '✗ 未达标'}"
          f"（目标 >60% 且高于对照；随机水平 = 20%）")
