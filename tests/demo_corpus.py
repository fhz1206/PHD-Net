"""外部语料验收：在中文维基百科语料上训练词级 LM。

⚠️ 2026-09-21：维基预训练语料已按指令删除（预训练语料已拍板：Infinity-Instruct 7M_core / M7_Core，2026-09-22 已就位 7,449,106 条 / 10.07 GB 文本，见 tools/convert_infinity.py），
本脚本暂停使用——运行会以明确报错退出；新语料就位后按下列路径放置即可复用：
语料：`datasets/pretrain/wiki_train.txt` / `datasets/pretrain/wiki_eval.txt`（由 `tools/prepare_wikicn.py` 从
中文维基百科 dump 抽取清洗生成，**页面级**确定性划分，CC BY-SA 3.0）。
与文档语料（`eval_corpus/internal_corpus.txt`）不同，这是**固定外部语料**——
不随项目文档编辑而漂移，适合作为跨轮比较的稳定锚点。

默认截取 train 60,000 / eval 8,000 字符以控制 CPU 预算；80/20 比例是按字符数
硬截断得到的（可能切开条目），若要严格页面级划分请用 prepare_wiki.py 的输出原样。

用法：
    python tests/demo_corpus.py                     # 默认 60K / 8K 字符
    python tests/demo_corpus.py --train 200000 --eval 50000
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根目录

import numpy as np

from phdnet.config import PHDNetConfig
from phdnet.ngram import WordNGram
from phdnet.word_lm import PHDWordLM

DATA = Path(__file__).resolve().parents[1] / "datasets"
TRAIN_PATH = DATA / "wiki_train.txt"
EVAL_PATH = DATA / "wiki_eval.txt"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="中文维基语料上的外部语料验收")
    ap.add_argument("--train", type=int, default=60_000, help="训练字符数预算")
    ap.add_argument("--eval", type=int, default=8_000, help="评估字符数预算")
    args = ap.parse_args(argv)

    if not TRAIN_PATH.exists():
        sys.exit("缺少语料（已按指令删除）：请先运行 `python tools/fetch_modelscope.py` 与 "
                 "`python tools/prepare_wikicn.py` 重建 datasets/pretrain/wiki_{train,eval}.txt")
    train_full = TRAIN_PATH.read_text(encoding="utf-8")
    eval_full = EVAL_PATH.read_text(encoding="utf-8")
    train_txt, eval_txt = train_full[: args.train], eval_full[: args.eval]

    print("=" * 70)
    print("外部语料验收：中文维基百科（CC BY-SA）→ 词级 LM")
    print("=" * 70)
    print(f"语料文件: train {len(train_full):,} / eval {len(eval_full):,} 字符"
          f"（本运行截取 {len(train_txt):,} / {len(eval_txt):,}）\n")

    # 词表只从**训练段**构建（无评估泄漏）；评估走 OOV 安全路径
    lm = PHDWordLM(train_txt, PHDNetConfig(
        n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
        eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11, pred_in_readout=True),
        seg_kwargs=dict(max_len=8, min_count=3, min_entropy=1.0))
    toks_train = lm.tokenize(train_txt)
    toks_eval = lm.tokenize(eval_txt)
    print(f"词表 {len(lm.tok)}  训练 token {len(toks_train):,} / 评估 token {len(toks_eval):,}"
          f"  压缩比 {len(train_txt) / max(1, len(toks_train)):.2f}×\n")

    print("--- 参照基线 ---")
    n_char = sum(len(t) for t in toks_eval)
    ng = WordNGram(toks_train, n=2)
    nll2 = ng.nll(toks_eval)
    ppl2 = float(np.exp(nll2 * len(toks_eval) / max(1, n_char)))
    print(f"词级 2-gram: token NLL 均值 {nll2:.3f}（字符归一 PPL = {ppl2:.2f}）")

    print("\n--- PHD-Net 词级 LM（W2 配置）---")
    t0 = time.perf_counter()
    lm.train_stream(train_txt)
    dt = time.perf_counter() - t0
    m = lm.evaluate(eval_txt)
    print(f"训练 {dt:.1f}s（{dt / max(1, len(toks_train)) * 1000:.2f} ms/token）")
    print(f"token PPL = {m['ppl_token']:.2f}  字符归一 PPL = {m['ppl_char']:.2f}"
          f"  bpc = {m['bpc']:.3f}")
    print(f"评估 OOV: {m['oov']}/{m['oov'] + m['n_tok']}"
          f"（{m['oov_rate'] * 100:.1f}%，按标准做法跳过、不计入 PPL）")

    print("-" * 70)
    print(f"{'配置':<26}{'字符归一PPL':>12}{'bpc':>8}")
    print(f"{'词级 2-gram 参照':<24}{ppl2:>10.2f}{np.log(ppl2) / np.log(2):>8.3f}")
    print(f"{'PHD-Net 词级（W2）':<24}{m['ppl_char']:>10.2f}{m['bpc']:>8.3f}")
    verdict = "✓ 优于词级 2-gram 基线" if m["ppl_char"] < ppl2 else "✗ 未优于基线"
    print(f"\n结论: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
