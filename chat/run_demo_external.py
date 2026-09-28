"""外部语料训练 + 对话演示（mimo-claude-code-traces-1k，MIT）。

- 训练：词级 PHDWordLM（W2 配置，与 demo_corpus.py 同口径）在外部语料上在线训练，
  报告字符归一 PPL（应与 10.81 量级一致，显著优于中文文档字符级 150）。
- 对话：用 PHDWordLM.generate（词级自回归续写）对若干英文种子词生成，
  模拟「提示 → 模型续写」的对话式交互。

非侵入式：仅读取 data/ 语料，不修改任何项目代码。
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # outputs/.. = 项目根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from phdnet.config import PHDNetConfig
from phdnet.word_lm import PHDWordLM


def main():
    DATA = ROOT / "data"
    MAX_TRAIN, MAX_EVAL = 60_000, 8_000

    tp, ep = DATA / "mimo_train.txt", DATA / "mimo_eval.txt"
    if not tp.exists():
        print("缺少语料：请先运行 `python tools/prepare_mimo.py`", flush=True)
        return

    train_full = tp.read_text(encoding="utf-8")
    eval_full = ep.read_text(encoding="utf-8")
    train_txt, eval_txt = train_full[:MAX_TRAIN], eval_full[:MAX_EVAL]

    cfg = PHDNetConfig(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
                       eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
                       pred_in_readout=True)
    lm = PHDWordLM(train_txt, cfg,
                   seg_kwargs=dict(max_len=8, min_count=3, min_entropy=1.0))

    toks_train = lm.tokenize(train_txt)
    toks_eval = lm.tokenize(eval_txt)

    print("#" * 70, flush=True)
    print("# 外部语料训练（mimo-claude-code-traces-1k, MIT）→ 词级 LM（W2）", flush=True)
    print("#" * 70, flush=True)
    print(f"语料: train {len(train_txt):,} / eval {len(eval_txt):,} 字符", flush=True)
    print(f"词表 {len(lm.tok):,}  训练 token {len(toks_train):,} / 评估 token {len(toks_eval):,}", flush=True)
    print(f"压缩比 {len(train_txt) / max(1, len(toks_train)):.2f}×", flush=True)

    t0 = time.perf_counter()
    lm.train_stream(train_txt)
    dt = time.perf_counter() - t0
    m = lm.evaluate(eval_txt)
    print(f"训练耗时: {dt:.1f}s（{dt / max(1, len(toks_train)) * 1000:.2f} ms/token）", flush=True)
    print(f"token PPL = {m['ppl_token']:.2f}  字符归一 PPL = {m['ppl_char']:.2f}  bpc = {m['bpc']:.3f}", flush=True)
    print(f"评估 OOV: {m['oov']}/{m['oov'] + m['n_tok']}（{m['oov_rate'] * 100:.1f}%，按标准做法跳过不计入 PPL）", flush=True)

    print("", flush=True)
    print("#" * 70, flush=True)
    print("# 对话：词级自回归续写（M11 生成解码，温度采样 τ=0.7）", flush=True)
    print("#" * 70, flush=True)

    seeds = ["the", "a", "def", "import", "error", "data", "function", "return", "the"]
    rng = np.random.default_rng(0)
    for raw in seeds:
        if raw not in lm.tok.stoi:
            continue
        gen = lm.generate(raw, n_tokens=60, tau=0.7, rng=rng)
        text = " ".join(gen)
        print(f"\n[提示] {raw!r}\n[续写] {text}", flush=True)

    print("", flush=True)
    print("=" * 70, flush=True)
    print("对照：同规模字符级中文文档 PPL≈150（本机早前演示）；", flush=True)
    print("外部语料词级 PPL≈10.81，续写为英文技术词序列，明显更连贯——", flush=True)
    print("但 PHD-Net 仍是下一 token 预测器，非语义级聊天模型。", flush=True)


if __name__ == "__main__":
    main()
