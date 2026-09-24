"""P4 验收：生成式回放（睡眠梦境）对旧知识保持的作用。

协议：
  1. 学习 10 个旧模式（一次性印迹）→ 睡眠巩固（快权重遗忘 0.2 模拟海马消退）
  2. 继续在线学习 10 个新模式（会与旧记忆竞争、干扰）
  3. 对照：普通睡眠 vs 睡眠+生成式回放
  4. 检验旧模式检索保持率（目标 >90%）
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

from phdnet.memory import LongTermMemory


def make_ltm():
    return LongTermMemory(n=256, eta_hip=0.05, eta_cortex=0.05,
                          beta_fast=0.6, beta_slow=1.0, n_steps=6)


def cue_of(p, rng, flip=0.2):
    c = p.copy()
    idx = rng.choice(len(p), int(flip * len(p)), replace=False)
    c[idx] *= -1
    return c


def old_recall_acc(ltm, old_pats, rng):
    accs = [float((ltm.recall(cue_of(p, rng)) == p).mean()) for p in old_pats]
    return float(np.mean(accs))


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    old_pats = rng.choice([-1.0, 1.0], size=(15, 256))
    new_pats = rng.choice([-1.0, 1.0], size=(15, 256))   # 高干扰：总模式数逼近容量

    def stable_dreams(ltm, k, rng):
        """生成 + 稳定性过滤：加噪重检索不变的吸引子才保留（剔除伪态）。"""
        out = []
        for d in ltm.replay(k, rng):
            noise = np.sign(rng.normal(0, 1, ltm.n)) * 0.2
            d2 = ltm.recall(np.sign(d + noise))
            if float((d2 == d).mean()) > 0.97:
                out.append(d)
        return out

    for tag, use_replay in [("普通睡眠（仅巩固）", False),
                            ("睡眠+生成式回放", True)]:
        ltm = make_ltm()
        for p in old_pats:                       # 阶段 1：旧知识 one-shot 印迹
            ltm.imprint(p)
        for _ in range(30):
            ltm.consolidate()                    # 巩固 + 海马旧痕消退
        ltm.W_fast *= 0.2
        acc_before = old_recall_acc(ltm, old_pats, rng)

        for p in new_pats:                       # 阶段 2：高干扰新学习（每模式 2 次）
            ltm.imprint(p)
            ltm.imprint(p)

        if use_replay:                           # 睡眠：巩固 + 过滤后的梦境回放
            for d in stable_dreams(ltm, 20, rng):
                ltm.W_fast += 1.0 * 0.05 * np.outer(d, d)   # 等效一次完整印迹
            ltm.W_fast[ltm.diag_idx] = 0.0
            np.clip(ltm.W_fast, -1.0, 1.0, out=ltm.W_fast)

        acc_after = old_recall_acc(ltm, old_pats, rng)
        new_acc = float(np.mean([float((ltm.recall(cue_of(p, rng)) == p).mean())
                                 for p in new_pats]))
        print(f"[{tag}] 旧模式保持率: 新学习前 {acc_before:.1%} → "
              f"新学习+睡眠后 {acc_after:.1%}（Δ {acc_after - acc_before:+.1%}）  "
              f"新模式检索 {new_acc:.1%}")

    print("-" * 62)
    print("P4 结论: 高干扰协议下对比回放与普通睡眠的旧知识保持率（目标：回放 >90% 且 > 对照）。")
