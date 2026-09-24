"""M5 公平评测（入口）—— 编排六任务 + Transformer 对照 + 判定矩阵。

任务实现已按类目拆分（结构整理 2026-09-18）：
    tests/eval_common.py       共享常量与跨任务辅助
    tests/eval_tasks_lm.py     任务 1 语言建模 / 任务 2 长程依赖
    tests/eval_tasks_memory.py 任务 3 单样本记忆 / 4 持续学习 / 5 多跳推理
    tests/eval_tasks_perf.py   任务 6 效率

口径（路线图 §5）：同语料、同划分、同字符预算；跨粒度统一用字符归一 PPL 与 bpc。
判定矩阵按路线图 §5.4：主任务差距 ≤10% 且 ≥2 项结构化任务显著占优 = 「相当」。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import math

import numpy as np

from eval_common import BASE, DOC, SEG
from eval_tasks_lm import task1_lm, task2_longrange
from eval_tasks_memory import task3_oneshot, task4_continual, task5_multihop
from eval_tasks_perf import task6_efficiency
from nano_gpt import train_eval as tf_train_eval
from phdnet.config import PHDNetConfig
from phdnet.word_lm import PHDWordLM


def main() -> None:
    print("=" * 70)
    print("M5 公平评测：PHD-Net vs nanoGPT 级 Transformer（同语料/同预算）")
    print("=" * 70)
    with open(DOC, encoding="utf-8") as f:
        text = f.read()
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split], text[split:]
    print(f"语料 {len(text):,} 字符（训练 {len(train_txt):,} / 评估 {len(eval_txt):,}）\n")

    r1 = task1_lm(train_txt, eval_txt, text)
    print(f"    学习曲线（训练比例→字符归一 PPL）: "
          f"{[(f'{f:.2f}', round(p, 1)) for f, p in r1['curve']]}")
    print(f"    PHD-Net 字符归一 PPL = {r1['ppl_char']:.2f}  bpc = {r1['bpc']:.3f}")

    print("\n[任务1 对照] Transformer（同语料、同字符预算）")
    probe = PHDWordLM(text, PHDNetConfig(**BASE), seg_kwargs=SEG)
    toks_tr = probe.tokenize(train_txt)
    toks_ev = probe.tokenize(eval_txt)
    cpt = len(train_txt) / max(1, len(toks_tr))
    ids_tr = [probe.tok.stoi[t] for t in toks_tr]
    ids_ev = [probe.tok.stoi[t] for t in toks_ev]
    tf_small = tf_train_eval(ids_tr, ids_ev, len(probe.tok), d_model=96,
                             n_layer=2, block_size=64, batch_size=32,
                             epochs=3, seed=11, chars_per_token=cpt)
    tf_big = tf_train_eval(ids_tr, ids_ev, len(probe.tok), d_model=192,
                           n_layer=4, block_size=64, batch_size=32,
                           epochs=3, seed=11, chars_per_token=cpt)

    print()
    r2 = task2_longrange()
    print()
    r3 = task3_oneshot()
    print()
    r4 = task4_continual()
    print()
    r5 = task5_multihop()
    print()
    r6 = task6_efficiency(train_txt, text)

    print("\n" + "=" * 70)
    print("判定矩阵（路线图 §5.4）")
    print("=" * 70)
    best_tf = min(tf_small["ppl_char"], tf_big["ppl_char"])
    gap = (r1["ppl_char"] - best_tf) / best_tf * 100
    print(f"主任务 字符归一 PPL: PHD-Net {r1['ppl_char']:.2f} vs "
          f"Transformer {best_tf:.2f}（差距 {gap:+.1f}%）")
    print(f"  bpc: PHD-Net {r1['bpc']:.3f} vs Transformer "
          f"{math.log(best_tf) / math.log(2):.3f}")
    wins = []
    if r3["phdnet"] and r3["phdnet"] > 0.8:
        wins.append(f"单样本记忆 {r3['phdnet'] * 100:.0f}%（Transformer 原生不具备）")
    if r5.get("phdnet"):
        wins.append(f"多跳链式推理 {r5['phdnet'] * 100:.0f}% vs 直接查询 "
                    f"{r5['direct'] * 100:.0f}%")
    if r4["phdnet"]["BWT"] > r4["transformer"]["BWT"]:
        wins.append(f"持续学习 BWT {r4['phdnet']['BWT']:+.3f} vs "
                    f"{r4['transformer']['BWT']:+.3f}")
    print(f"结构化任务占优项（{len(wins)}）: " + ("；".join(wins) if wins else "无"))
    if abs(gap) <= 10 and len(wins) >= 2:
        verdict = "相当（达标）"
    elif gap <= 0 and len(wins) >= 3:
        verdict = "更强"
    else:
        verdict = "未达成"
    print(f"\n判定: {verdict}")


if __name__ == "__main__":
    main()
