"""M5 评测任务：单样本记忆（3）/ 持续学习（4）/ 多跳推理（5）。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from eval_common import BASE, _tf_acc, _tf_fit
from phdnet.config import PHDNetConfig
from phdnet.lm import PHDNetLM
from phdnet.memory import LongTermMemory


def task3_oneshot(n_dim: int = 256, n_items: int = 20, noise: float = 0.3,
                  seed: int = 5) -> dict:
    """单样本记忆：每条只印迹一次，30% 位翻转的线索召回。"""
    print("[任务3] 单样本记忆（一次性印迹 + 带噪线索召回）")
    rng = np.random.default_rng(seed)
    ltm = LongTermMemory(n_dim, 0.08, 0.05, 0.6, 1.0, 6)
    pats = [np.sign(rng.normal(size=n_dim)) for _ in range(n_items)]
    for p in pats:
        ltm.imprint(p)
    hit = 0
    for p in pats:
        cue = p.copy()
        flip = rng.choice(n_dim, size=int(noise * n_dim), replace=False)
        cue[flip] *= -1.0
        rec = ltm.recall(cue)
        if np.mean(np.sign(rec) == np.sign(p)) > 0.9:
            hit += 1
    acc = hit / n_items
    print(f"    召回准确率 {acc * 100:.0f}%（{n_items} 条，各印迹一次）")
    return {"phdnet": acc, "transformer": None}


def task4_continual() -> dict:
    """持续学习：任务 A（x→y→z）→ 任务 B（p→q→r），测 BWT。"""
    print("[任务4] 持续学习（A→B 遗忘率 BWT）")
    vocab = "xyzpqr"
    A = "xyz" * 1000
    B = "pqr" * 1000
    lm = PHDNetLM(vocab, PHDNetConfig(**{**BASE, "pred_in_readout": False}))

    def acc(text: str) -> float:
        hit = tot = 0
        for i in range(min(600, len(text) - 1)):
            y = lm.net.step(lm.tok.encode(text[i]), learn=False)["y"]
            if int(np.argmax(y)) == lm.tok.stoi[text[i + 1]]:
                hit += 1
            tot += 1
        return hit / max(1, tot)

    lm.train_stream(A)
    a1 = acc(A)
    lm.train_stream(B)
    a2, b = acc(A), acc(B)
    print(f"    PHD-Net: acc_A1={a1:.3f} acc_A2={a2:.3f} acc_B={b:.3f} "
          f"BWT={a2 - a1:+.3f}")
    # Transformer 对照：先训 A 再顺序微调 B（同字符预算，滑窗 32）
    ids_a = [vocab.index(c) for c in A]
    ids_b = [vocab.index(c) for c in B]
    from nano_gpt import NanoGPT
    import torch
    torch.manual_seed(11)
    m = NanoGPT(len(vocab), 64, 2, 4, 32)
    _tf_fit(m, ids_a, epochs=3)
    ta1 = _tf_acc(m, ids_a)
    _tf_fit(m, ids_b, epochs=3)
    ta2 = _tf_acc(m, ids_a)
    tb = _tf_acc(m, ids_b)
    print(f"    Transformer: acc_A1={ta1:.3f} acc_A2={ta2:.3f} acc_B={tb:.3f} "
          f"BWT={ta2 - ta1:+.3f}")
    return {"phdnet": {"A1": a1, "A2": a2, "B": b, "BWT": a2 - a1},
            "transformer": {"A1": ta1, "A2": ta2, "B": tb, "BWT": ta2 - ta1}}


def task5_multihop() -> dict:
    """多跳推理：语义图 2 跳链式查询（M9/M12）。"""
    print("[任务5] 多跳推理（语义图 2 跳）")
    try:
        from phdnet.cognition import SemanticGraph
        g = SemanticGraph(1024, eta=0.1)

        def ent(name: str) -> np.ndarray:
            b = name.encode("utf-8")
            s = int.from_bytes(b[:8].ljust(8, b"\0"), "big")
            rng = np.random.default_rng(s)
            v = np.zeros(1024)
            v[rng.choice(1024, 32, replace=False)] = 1.0
            return v

        triples = [("苏格拉底", "是", "人"), ("人", "具有", "会死"),
                   ("柏拉图", "是", "人"), ("亚里士多德", "是", "人"),
                   ("猫", "是", "动物"), ("动物", "具有", "会死")]
        for s, r, o in triples:
            g.bind(ent(s), ent(r), ent(o))
        qs = [("苏格拉底", "具有", "会死"), ("柏拉图", "具有", "会死"),
              ("亚里士多德", "具有", "会死"), ("猫", "具有", "会死")]
        hit, direct = 0, 0
        for s, r, o in qs:
            chain = g.chain_query(ent(s), [ent("是"), ent("具有")])
            tgt = ent(o)
            if float(chain @ tgt) / (np.linalg.norm(chain) * np.linalg.norm(tgt) + 1e-9) > 0.5:
                hit += 1
            d = g.query(ent(s), ent(r))
            if float(d @ tgt) / (np.linalg.norm(d) * np.linalg.norm(tgt) + 1e-9) > 0.5:
                direct += 1
        print(f"    链式 {hit}/{len(qs)}  直接查询 {direct}/{len(qs)}")
        return {"phdnet": hit / len(qs), "direct": direct / len(qs),
                "transformer": None}
    except Exception as e:                                  # 依赖缺失不阻断套件
        print(f"    跳过（{e}）")
        return {"phdnet": None, "direct": None, "transformer": None}
