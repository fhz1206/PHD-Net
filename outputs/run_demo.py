"""临时演示驱动：在本地 CPU 上跑「训练 + 对话」。

- 训练：字符级 PHD-Net 语言模型（PHDNetLM）在《PHD-Net_架构设计.md》上在线训练
  （单样本、无反向传播、无 epoch），并报告 NLL 轨迹与评估 PPL。
- 对话：用 Generator（M11 生成解码器）对若干中文提示做自回归续写，
  模拟「给提示 → 模型生成」的对话式交互。

非侵入式：仅读取 docs 语料，不修改任何项目代码。
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # outputs/.. = 项目根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from phdnet.config import PHDNetConfig
from phdnet.lm import PHDNetLM
from phdnet.generate import Generator


def main():
    with open(str(ROOT / "datasets" / "eval" / "internal_corpus.txt"), encoding="utf-8") as f:
        text = f.read()

    TRAIN_N = 9000
    train_txt, eval_txt = text[:TRAIN_N], text[TRAIN_N:]

    cfg = PHDNetConfig(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
                       eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11)

    print("#" * 64, flush=True)
    print("# 训练：字符级 PHD-Net 语言模型（在线、单样本、无反向传播）", flush=True)
    print("#" * 64, flush=True)
    print(f"语料: PHD-Net_架构设计.md  总 {len(text):,} 字符", flush=True)
    print(f"训练段 {len(train_txt):,} / 评估段 {len(eval_txt):,} 字符", flush=True)

    t0 = time.perf_counter()
    lm = PHDNetLM(text, cfg)
    print(f"模型构建完成（字符表 V = {len(lm.tok):,}）", flush=True)

    nlls = lm.train_stream(train_txt, log_every=2000)
    dt = time.perf_counter() - t0
    print(f"在线训练耗时: {dt:.1f}s（{dt / (len(train_txt) - 1) * 1000:.2f} ms/字符）", flush=True)
    print("训练 NLL 下降轨迹:", flush=True)
    for i, v in enumerate(nlls):
        print(f"  段 {i + 1}: NLL = {v:.3f}  PPL = {np.exp(v):.2f}", flush=True)

    ppl = lm.evaluate(eval_txt)
    print(f"评估段字符级 PPL: {ppl:.2f}（对照：均匀随机基线 = V = {len(lm.tok):,}）", flush=True)

    print("", flush=True)
    print("#" * 64, flush=True)
    print("# 对话：Generator 生成解码（M11，温度采样 + top-k 截断）", flush=True)
    print("#" * 64, flush=True)
    gen = Generator(lm, tau=0.7, topk=8)
    prompts = ["预测", "学习", "记忆", "网络"]
    for raw in prompts:
        # 只保留词表中存在的字符，避免 KeyError
        prompt = "".join(c for c in raw if c in lm.tok.stoi)
        if not prompt:
            prompt = lm.tok.chars[0]
        out = gen.generate(prompt, n_chars=80)
        print(f"\n[用户提示] {prompt!r}", flush=True)
        print(f"[模型续写] {prompt + out}", flush=True)

    print("", flush=True)
    print("=" * 64, flush=True)
    print("说明：该模型是字符级下一字符预测器，并非聊天模型；", flush=True)
    print("训练语料仅 9K 字符，故续写以局部模式/重复为主，", flush=True)
    print("不具语义连贯——这是容量与语料规模决定的诚实边界。", flush=True)


if __name__ == "__main__":
    main()
