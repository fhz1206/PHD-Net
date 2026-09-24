"""M2 验收：词涌现 + 上下文绑定编码（T2.1 + T2.3），M1 开关在词级上的再评估。

增量消融（同语料、同划分、同预算；字符级参照 = demo_lm 的 3-gram 与 PHD-Net 字符级基线）：
  参照  字符级 3-gram / 词级 bigram / trigram（token n-gram，加 1 平滑）
  W1    = T2.1 + T2.3（M1 开关全关）
  W2    = W1 + T3.1' 预测通路三拼
  W3    = W2 + T1.1 误差广播 + T1.3 错误驱动写入
  W4    = W3 + PC 开放发育（eta_pc/eta_oja=0.01，P3 自动调度）
  W5    = W3 + 周期睡眠巩固（每 400 token 一次 sleep）

指标：字符归一 PPL（跨粒度公平口径）为主，词级 PPL 与 bpc 并报；
另测生成 bigram 合法率（M11 词级版）。
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
from phdnet.lm import NGramBaseline
from phdnet.word_lm import PHDWordLM, WordNGram

if __name__ == "__main__":
    with open(str(_DOC), encoding="utf-8") as f:
        text = f.read()
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split], text[split:]
    print("=" * 70)
    print("M2 验收：词涌现 + 上下文绑定编码（T2.1 + T2.3）")
    print("=" * 70)

    base = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
                eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11)
    base4 = {**base, "eta_pc": 0.01, "eta_oja": 0.01}

    seg_kwargs = dict(max_len=6, min_count=5, min_entropy=1.0)

    # 词表统计（一次构建，各变体共享——确定性哈希，可复现）
    probe = PHDWordLM(text, PHDNetConfig(**base), seg_kwargs=seg_kwargs)
    toks_all = probe.tokenize(text)
    toks_train = probe.tokenize(train_txt)
    toks_eval = probe.tokenize(eval_txt)
    n_multi = sum(1 for t in probe.tok.tokens if len(t) > 1)
    avg_len = float(np.mean([len(t) for t in toks_train]))
    print(f"语料 {len(text):,} 字符 → {len(toks_all):,} token"
          f"（压缩比 {len(text) / len(toks_all):.2f}×，平均 token 长 {avg_len:.2f} 字）")
    print(f"词表 {len(probe.tok)}（多字词 {n_multi} + 单字符 {len(probe.tok) - n_multi}）")

    print("\n--- 参照基线 ---")
    ngram3c = NGramBaseline(train_txt, vocab_text=text, n=3)
    print(f"字符级 3-gram：字符归一 PPL = {ngram3c.evaluate(eval_txt):.2f}")
    for n in (2, 3):
        ng = WordNGram(toks_train, n=n)
        nll = ng.nll(toks_eval)
        n_char = sum(len(t) for t in toks_eval)
        print(f"词级 {n}-gram：token NLL 均值 {nll:.3f}"
              f"（字符归一 PPL = {np.exp(nll * len(toks_eval) / n_char):.2f}）")

    variants = [
        ("W1 T2.1+T2.3（开关全关）", PHDNetConfig(**base), 0),
        ("W2 = W1 + T3.1' 预测三拼",
         PHDNetConfig(**base, pred_in_readout=True), 0),
        ("W3 = W2 + T1.1 + T1.3",
         PHDNetConfig(**base, pred_in_readout=True, task_modulation=True,
                      error_gated_memory=True), 0),
        ("W4 = W3 + PC 开放发育",
         PHDNetConfig(**base4, pred_in_readout=True, task_modulation=True,
                      error_gated_memory=True, auto_development=True), 0),
        ("W5 = W3 + 周期睡眠巩固",
         PHDNetConfig(**base, pred_in_readout=True, task_modulation=True,
                      error_gated_memory=True), 400),
    ]

    results = []
    print("\n--- PHD-Net 词级 LM ---")
    for name, cfg, sleep_every in variants:
        lm = PHDWordLM(text, cfg, seg_kwargs=seg_kwargs)
        t0 = time.perf_counter()
        lm.train_stream(train_txt, sleep_every=sleep_every)
        dt = time.perf_counter() - t0
        m = lm.evaluate(eval_txt)
        results.append((name, m, dt))
        print(f"{name}\n    训练 {dt:.1f}s  token PPL = {m['ppl_token']:.2f}"
              f"  字符归一 PPL = {m['ppl_char']:.2f}  bpc = {m['bpc']:.3f}")

    # 生成合法性（用 W1 与最优变体各测一次；词级 M11）
    print("\n--- 生成 bigram 合法率（M11 词级） ---")
    for name, cfg, sleep_every in (variants[0], variants[-1]):
        lm = PHDWordLM(text, cfg, seg_kwargs=seg_kwargs)
        lm.train_stream(train_txt)
        gen = lm.generate(seed_token=toks_eval[0], n_tokens=200, tau=0.8)
        val = lm.bigram_validity(gen, toks_train)
        print(f"{name}: {val * 100:.1f}%（随机基线 ≈ 1/词表 = "
              f"{100 / len(lm.tok):.2f}%）  样本: {''.join(gen[:20])!r}")

    print("-" * 70)
    print(f"{'配置':<30}{'字符归一PPL':>12}{'bpc':>8}{'训练耗时':>10}")
    ref = ngram3c.evaluate(eval_txt)
    print(f"{'字符级 3-gram 参照':<32}{ref:>10.2f}"
          f"{np.log(ref) / np.log(2):>8.3f}{'—':>10}")
    for name, m, dt in results:
        print(f"{name:<32}{m['ppl_char']:>10.2f}{m['bpc']:>8.3f}{dt:>8.1f}s")
    best = min(r[1]['ppl_char'] for r in results)
    print(f"\n结论: 最优字符归一 PPL = {best:.2f}"
          f"（字符级 3-gram 参照 {ref:.2f}，PHD-Net 字符级基线 155.80）")
