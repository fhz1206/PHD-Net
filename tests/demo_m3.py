"""M3 验收：优化调度（T4.1 逐突触自适应学习率 / T4.2 多尺度双迹 / T4.3 课程学习）。

基线 = M2 最优配置 W2（T2.1+T2.3 + T3.1' 预测三拼，字符归一 PPL 89.15）。
增量消融（同语料、同划分、同预算）：
  R 参照  = W2（应复现 89.15）
  A       = R + T4.1 逐突触自适应学习率
  B       = R + T4.2 多尺度双迹
  C       = R + T4.3 高频词课程（先高频骨架，后全量）
  D       = R + T4.1 + T4.2 + T4.3（组合）
指标：字符归一 PPL（跨粒度公平口径）为主，bpc 与训练耗时并报。
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
from phdnet.word_lm import PHDWordLM

if __name__ == "__main__":
    with open(str(_DOC), encoding="utf-8") as f:
        text = f.read()
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split], text[split:]
    print("=" * 70)
    print("M3 验收：优化调度 T4.1 自适应学习率 / T4.2 双迹 / T4.3 课程")
    print("=" * 70)

    # W2 基线配置（M2 最优）
    base = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
                eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
                pred_in_readout=True)
    seg_kwargs = dict(max_len=6, min_count=5, min_entropy=1.0)

    variants = [
        ("R 参照（M2 最优 W2）", PHDNetConfig(**base), 0.0),
        ("A = R + T4.1 自适应学习率",
         PHDNetConfig(**base, adaptive_lr=True), 0.0),
        ("B = R + T4.2 多尺度双迹",
         PHDNetConfig(**base, dual_trace=True), 0.0),
        ("C = R + T4.3 高频词课程", PHDNetConfig(**base), 0.2),
        ("D = R + T4.1+T4.2+T4.3",
         PHDNetConfig(**base, adaptive_lr=True, dual_trace=True), 0.2),
    ]

    results = []
    for name, cfg, curr in variants:
        lm = PHDWordLM(text, cfg, seg_kwargs=seg_kwargs)
        t0 = time.perf_counter()
        lm.train_stream(train_txt, curriculum_top=curr)
        dt = time.perf_counter() - t0
        m = lm.evaluate(eval_txt)
        results.append((name, m, dt))
        print(f"{name}\n    训练 {dt:.1f}s  token PPL = {m['ppl_token']:.2f}"
              f"  字符归一 PPL = {m['ppl_char']:.2f}  bpc = {m['bpc']:.3f}")

    print("-" * 70)
    print(f"{'配置':<28}{'字符归一PPL':>12}{'bpc':>8}{'vs 参照':>10}{'耗时':>9}")
    ref = results[0][1]["ppl_char"]
    for name, m, dt in results:
        d = f"{(m['ppl_char'] - ref) / ref * 100:+.1f}%"
        print(f"{name:<28}{m['ppl_char']:>10.2f}{m['bpc']:>8.3f}{d:>10}{dt:>7.1f}s")
    best = min((r[1]["ppl_char"], r[0]) for r in results)
    print(f"\n结论: 最优 {best[1]} = {best[0]:.2f}"
          f"（M2 最优 89.15；字符级基线 155.80）")
