"""脑同构"近似项"的实测：把审计中判定为 △ 的项逐一对齐大脑（可启用版）。

覆盖：
  A9 E/I 平衡 —— `ei_synapses`（独立抑制类群，已内置于 STDP 层）启用效果 + 抑制比扫参
  A6 激活稀疏度 —— `k_sparse` 从 12.5%（32/256）降到皮层量级 3.1%（8/256）/ 6.2%（16）
  A3 编码器学习 —— `learnable_encoder` 在更小 `eta_enc` 下是否可收敛（旧实测 +6.1% 有害）
  A4 读出稀疏 —— 已知有损（−14.8%），此处仅作对照
  A7 学习规则局部性 —— 无代码改动（误差不反传主干；监督来自自监督任务定义），仅记录

口径：冻结语料，训练段前 4,000 字符（基线 ppl_char = **97.2596**），评估段后 20%。

用法：python tools/experiment_brain_parity.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.config import PHDNetConfig          # noqa: E402
from phdnet.word_lm import PHDWordLM            # noqa: E402

CORPUS = _ROOT / "datasets" / "eval" / "internal_corpus.txt"
BASE = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
            eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11, pred_in_readout=True)
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)
TRAIN_CHARS = 4000
BASE_PPL = 97.2596

CONFIGS = [
    ("baseline", {}, "基准（k_sparse=32 = 12.5% 激活）"),
    # A9 E/I 独立抑制类群
    ("A9_ei_0.2", dict(ei_synapses=True, ei_ratio=0.2), "E/I 独立抑制类群（抑制比 20%）"),
    ("A9_ei_0.1", dict(ei_synapses=True, ei_ratio=0.1), "抑制比 10%"),
    ("A9_ei_0.3", dict(ei_synapses=True, ei_ratio=0.3), "抑制比 30%"),
    # A6 激活稀疏度（皮层 ~1–5%）
    ("A6_k8_3.1%", dict(k_sparse=8), "激活 8/256 = 3.1%（皮层量级）"),
    ("A6_k16_6.2%", dict(k_sparse=16), "激活 16/256 = 6.2%"),
    # A3 编码器学习（旧实测 eta_enc=0.01 有害 +6.1%）
    ("A3_enc_e0.001", dict(learnable_encoder=True, eta_enc=0.001), "可学习编码器 η=0.001"),
    ("A3_enc_e0.003", dict(learnable_encoder=True, eta_enc=0.003), "可学习编码器 η=0.003"),
]


def main() -> None:
    text = CORPUS.read_text(encoding="utf-8")
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split][:TRAIN_CHARS], text[split:]
    print(f"语料 {len(text)} | 训练段 {len(train_txt)} | 评估段 {len(eval_txt)} | "
          f"基线 ppl_char = {BASE_PPL}")
    print("=" * 96)
    rows = []
    for name, ov, note in CONFIGS:
        try:
            cfg = PHDNetConfig(**{**BASE, **ov})
            lm = PHDWordLM(text, cfg, seg_kwargs=SEG)
            t0 = time.perf_counter()
            lm.train_stream(train_txt)
            m = lm.evaluate(eval_txt)
            dt = time.perf_counter() - t0
            act = f"{cfg.k_sparse}/{cfg.n_sdr} = {cfg.k_sparse / cfg.n_sdr * 100:.1f}%"
            rows.append((name, float(m["ppl_char"]), dt, note))
            print(f"{name:<16} ppl_char={m['ppl_char']:>9.4f}  "
                  f"Δ={(m['ppl_char'] - BASE_PPL) / BASE_PPL * 100:>+7.2f}%  "
                  f"激活 {act:<14} ({dt:.0f}s)  {note}", flush=True)
        except Exception as e:
            print(f"{name:<16} 失败: {e!r}")
            rows.append((name, None, 0.0, note))

    print("=" * 96)
    print(f"{'配置':<16}{'ppl_char':>10}{'ΔPPL%':>9}  解读")
    for name, ppl, dt, note in rows:
        if ppl is None:
            print(f"{name:<16}{'FAIL':>10}{'--':>9}  {note}")
            continue
        d = (ppl - BASE_PPL) / BASE_PPL * 100
        verdict = ("✓ 可启用（无损）" if abs(d) <= 1.0 else
                   "△ 小损" if d <= 3.0 else "✗ 有害")
        print(f"{name:<16}{ppl:>10.4f}{d:>+9.2f}  {verdict}｜{note}")


if __name__ == "__main__":
    main()
