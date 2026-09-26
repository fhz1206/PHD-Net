"""P1 验收：PHD-Net 字符级语言模型 vs 3-gram 基线。

语料：本项目《PHD-Net 架构设计》文档（真实中文技术文本，自包含）。
前 80% 训练 / 后 20% 评估。指标：字符级困惑度（PPL，越低越好）。
参考下限：均匀随机 PPL = 字符表大小。
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
from phdnet.lm import NGramBaseline, PHDNetLM

if __name__ == "__main__":
    with open(str(_DOC), encoding="utf-8") as f:
        text = f.read()
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split], text[split:]
    print("=" * 62)
    print("P1 语言接口层验收：PHD-Net 字符级 LM vs 3-gram 基线")
    print("=" * 62)
    print(f"语料: internal_corpus.txt（{len(text):,} 字符）"
          f"  训练 {len(train_txt):,} / 评估 {len(eval_txt):,}")

    # 基线：均匀随机（下界参照 = 无任何信息）
    cfg = PHDNetConfig(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
                       eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11)
    # 字符表按全量文本构建（评估段可能含训练段未见字符）
    probe = PHDNetLM(text, cfg)
    V = len(probe.tok)
    print(f"字符表大小 V = {V}  → 均匀基线 PPL = {V}")

    ngram = NGramBaseline(train_txt, vocab_text=text, n=3)
    ppl_ngram = ngram.evaluate(eval_txt)
    print(f"\n3-gram 基线（加 1 平滑）评估 PPL: {ppl_ngram:.2f}")

    print(f"\n开始 PHD-Net 在线训练（单样本、无反向传播、无 epoch）…")
    t0 = time.perf_counter()
    nlls = probe.train_stream(train_txt, log_every=2000)
    dt = time.perf_counter() - t0
    print(f"训练耗时: {dt:.1f}s（{dt / (len(train_txt) - 1) * 1000:.2f} ms/字符）")
    print("训练 NLL 下降轨迹:")
    for i, v in enumerate(nlls):
        print(f"  段 {i + 1}: NLL = {v:.3f}  PPL = {np.exp(v):.2f}")

    ppl = probe.evaluate(eval_txt)
    print("-" * 62)
    print(f"PHD-Net 评估 PPL : {ppl:.2f}")
    print(f"3-gram 基线 PPL  : {ppl_ngram:.2f}")
    print(f"均匀随机 PPL     : {V}")
    verdict = "✓ 优于 3-gram 基线" if ppl < ppl_ngram else "✗ 未优于 3-gram 基线"
    print(f"结论: {verdict}")
