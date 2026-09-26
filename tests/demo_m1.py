"""路线图 M1 验收（v2 修订）：T1.1 误差广播调制 + T1.3 错误驱动写入 + T3.1 预测通路三拼。

首版教训（2026-09-18）：T3.1 原设计把 LTM 原始召回向量（±1 值）直接拼进读出，
印迹饱和后近似噪声，PPL 从 155.8 崩到 3.7 万——已修订为"STDP 时序预测接入读出"
（LM 最需要的下一状态预测信号此前只用于诊断、从未进入读出特征）。

增量消融设计（逐开关隔离归因；同语料、同字符数、同预算）：
  A 基线          = P1 复现（开关全关）
  B 仅 T3.1'      = + STDP 预测通路接入读出
  C B + T1.1      = + 任务误差广播调制
  D 完整 M1       = C + T1.3 错误驱动写入

注：语料为《PHD-Net_架构设计.md》当前版（并入认知层后 13,322 字符），
基线参考点较 P1 时的 99.36（旧 9K 字符语料）移动属预期，以本轮 A 为参照。
验收目标（路线图 §六 M1）：显著优于 A；默认路径零回归（flags 全关）。
"""

# --- 目录结构调整（2026-09-18）：脚本位于 tests/ 或 tools/ 子目录 ---
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]     # 项目根目录
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))              # 保证 `import phdnet` 可用
_DOC = _ROOT / "eval_corpus" / "internal_corpus.txt"    # 默认语料
# --- 引导结束 ---

import time

import numpy as np

from phdnet.config import PHDNetConfig
from phdnet.lm import PHDNetLM

if __name__ == "__main__":
    with open(str(_DOC), encoding="utf-8") as f:
        text = f.read()
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split], text[split:]
    print("=" * 66)
    print("M1 验收 v2：预测通路三拼 / 误差广播调制 / 错误驱动写入（增量消融）")
    print("=" * 66)
    print(f"语料: internal_corpus.txt（{len(text):,} 字符）"
          f"  训练 {len(train_txt):,} / 评估 {len(eval_txt):,}\n")

    base = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
                eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11)

    variants = [
        ("A 基线（开关全关）", PHDNetConfig(**base)),
        ("B 仅 T3.1'（预测通路三拼）",
         PHDNetConfig(**base, pred_in_readout=True)),
        ("C B + T1.1（误差广播调制）",
         PHDNetConfig(**base, pred_in_readout=True, task_modulation=True)),
        ("D 完整 M1（C + T1.3 错误驱动写入）",
         PHDNetConfig(**base, pred_in_readout=True, task_modulation=True,
                      error_gated_memory=True)),
    ]

    results = []
    for name, cfg in variants:
        lm = PHDNetLM(text, cfg)
        t0 = time.perf_counter()
        lm.train_stream(train_txt)
        dt = time.perf_counter() - t0
        ppl = lm.evaluate(eval_txt)
        results.append((name, ppl, dt))
        print(f"{name}\n    训练 {dt:.1f}s（{dt / (len(train_txt) - 1) * 1000:.2f} ms/字符）"
              f"  评估 PPL = {ppl:.2f}\n")

    ppl_a = results[0][1]
    print("-" * 66)
    print(f"{'配置':<34}{'PPL':>10}{'vs 基线':>12}{'训练耗时':>10}")
    for name, ppl, dt in results:
        delta = f"{(ppl - ppl_a) / ppl_a * 100:+.1f}%" if ppl_a else "-"
        print(f"{name:<36}{ppl:>8.2f}{delta:>12}{dt:>8.1f}s")
    best, best_name = min((r[1], r[0]) for r in results)
    verdict = "✓ 达成（≤70）" if best <= 70 else "△ 未达 ≤70（以实测为准，进入调参迭代）"
    print(f"\n结论: 最优 {best_name} PPL = {best:.2f}  {verdict}")
