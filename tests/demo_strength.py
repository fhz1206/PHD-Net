"""架构强化验收：稳态突触缩放（解锁 PC 表征学习）+ 错误触发检索。

背景（路线图 B1，最大瓶颈）：PC 在线学习在 n≥256 配置下三次发散
（PPL 达 4.7×10⁴），表征被迫冻结（eta_pc = eta_oja = 0）——任务信号无处可去。
本次以**稳态突触缩放**（Turrigiano 2008，突触缩放是神经系统稳定 Hebbian
可塑性的经典机制）为稳定器，尝试解锁表征学习。

变体（同语料、同划分、同预算；基线 = M2 最优 W2 配置）：
  V0 基线        = 表征冻结（PC 关）
  V1 PC 开放     = + eta_pc/eta_oja>0 + P3 自动调度（预期发散，复现已知缺陷）
  V2 PC 开放+稳态= V1 + homeostasis=True（预期：不发散且优于 V0）
  V3 V2+错误触发 = V2 + error_triggered_retrieval（预测失败时额外检索）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根目录

import time

import numpy as np

from phdnet.config import PHDNetConfig
from phdnet.word_lm import PHDWordLM

if __name__ == "__main__":
    doc = r"D:\AiModel\train\docs\PHD-Net_架构设计.md"
    with open(doc, encoding="utf-8") as f:
        text = f.read()
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split], text[split:]
    print("=" * 72)
    print("架构强化验收：稳态突触缩放（解锁 PC 表征学习）+ 错误触发检索")
    print("=" * 72)

    base = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
                eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
                pred_in_readout=True)
    pc_on = {**base, "eta_pc": 0.02, "eta_oja": 0.01, "auto_development": True}
    seg = dict(max_len=6, min_count=5, min_entropy=1.0)

    variants = [
        ("V0 基线（表征冻结）", PHDNetConfig(**base)),
        ("V1 PC 开放（无稳态，预期发散）", PHDNetConfig(**pc_on)),
        ("V2 PC 开放 + 稳态缩放",
         PHDNetConfig(**pc_on, homeostasis=True)),
        ("V3 = V2 + 错误触发检索",
         PHDNetConfig(**pc_on, homeostasis=True,
                      error_triggered_retrieval=True)),
        ("V4 = V2 + 读出退火（0.9997/步）",
         PHDNetConfig(**pc_on, homeostasis=True, eta_readout_anneal=0.9997)),
    ]

    results = []
    for name, cfg in variants:
        lm = PHDWordLM(text, cfg, seg_kwargs=seg)
        t0 = time.perf_counter()
        lm.train_stream(train_txt)
        dt = time.perf_counter() - t0
        m = lm.evaluate(eval_txt)
        diverged = m["ppl_char"] > 500
        results.append((name, m, dt, diverged))
        print(f"{name}\n    训练 {dt:5.1f}s  token PPL = {m['ppl_token']:.2f}"
              f"  字符归一 PPL = {m['ppl_char']:.2f}  bpc = {m['bpc']:.3f}"
              f"{'   ⚠ 发散' if diverged else ''}\n")

    print("-" * 72)
    print(f"{'配置':<26}{'字符归一PPL':>12}{'bpc':>8}{'vs 基线':>10}{'耗时':>9}")
    ref = results[0][1]["ppl_char"]
    for name, m, dt, div in results:
        d = f"{(m['ppl_char'] - ref) / ref * 100:+.1f}%"
        mark = " ⚠发散" if div else ""
        print(f"{name:<28}{m['ppl_char']:>10.2f}{m['bpc']:>8.3f}{d:>10}{dt:>7.1f}s{mark}")

    v4 = results[4][1]["ppl_char"]
    v4_div = results[4][3]
    if not v4_div and v4 < ref:
        print(f"\n结论: ✓ 稳态缩放 + 读出退火解锁表征学习（V4={v4:.2f} < 冻结基线 {ref:.2f}）"
              f"——可解除 eta_pc=0 的冻结约束")
    elif not v4_div:
        print(f"\n结论: ○ 稳态缩放 + 读出退火消除了发散（V4={v4:.2f}，较无稳态发散 "
              f"{results[1][1]['ppl_char']:.0f} 改善 "
              f"{results[1][1]['ppl_char'] / v4:.0f}×），但表征漂移仍损失 "
              f"{(v4 - ref) / ref * 100:.0f}%，冻结约束暂保持")
    else:
        print("\n结论: ✗ 稳态缩放未能抑制发散——需更强约束（如逐神经元目标发放率）")
