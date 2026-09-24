"""在 R1 蒸馏 SFT 子集上训练 PHDWordLM（词级），并保存模型。

前置：先运行 tools/prepare_r1sft.py 生成 data/r1sft_train.txt / r1sft_eval.txt。

说明（诚实边界）：PHD-Net 是词级下一 token 预测器 + 联想记忆，不是语义聊天模型。
本脚本只让它学到中文问答的**表面结构与行文风格**，生成"看起来像问答"的续写，
并不真正理解指令或推理。训练文本建议截取子集（本机 CPU 预算，见 prepare 的 --max-samples）。

用法：
    python outputs/train_r1sft.py --small           # 小网络加速
    python outputs/train_r1sft.py --train data/r1sft_train.txt
"""

import argparse
import os
import pickle
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from phdnet.config import PHDNetConfig
from phdnet.word_lm import PHDWordLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/r1sft_train.txt")
    ap.add_argument("--eval", default="data/r1sft_eval.txt")
    ap.add_argument("--model-out", default="outputs/r1sft_model.pkl")
    ap.add_argument("--small", action="store_true",
                    help="使用更小的网络（n_sdr/n_mid/n_top=128, k=16）以加速 CPU 训练")
    ap.add_argument("--eta-stdp", type=float, default=0.03)
    ap.add_argument("--max-len", type=int, default=4,
                    help="分词最大 n-gram 长度（小值显著加速词表构建）")
    ap.add_argument("--min-count", type=int, default=8,
                    help="词表最低出现次数（大值减少熵剪枝候选数）")
    args = ap.parse_args()

    with open(args.train, encoding="utf-8") as f:
        train_text = f.read()
    ns, ks = (128, 16) if args.small else (256, 32)
    nm = 128 if args.small else 256
    nt = 128 if args.small else 256
    cfg = PHDNetConfig(n_sdr=ns, k_sparse=ks, n_mid=nm, n_top=nt,
                        eta_pc=0.0, eta_oja=0.0, eta_stdp=args.eta_stdp,
                        seed=11, pred_in_readout=True)
    seg_kwargs = dict(max_len=args.max_len, min_count=args.min_count,
                      min_entropy=1.0)

    t0 = time.time()
    print(f"[1/3] 构建词表（训练文本 {len(train_text)} 字符）...", flush=True)
    lm = PHDWordLM(train_text, cfg, seg_kwargs=seg_kwargs)
    print(f"      词表大小={len(lm.tok)}  用时 {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    print("[2/3] 训练（在线词级 LM）...", flush=True)
    hist = lm.train_stream(train_text, log_every=50000, sleep_every=0)
    n_seg = len(hist)
    print(f"      训练完成 用时 {time.time()-t1:.1f}s  日志段数={n_seg}  "
          f"末段NLL={hist[-1] if hist else float('nan'):.3f}", flush=True)

    with open(args.eval, encoding="utf-8") as f:
        eval_text = f.read()
    m = lm.evaluate(eval_text)
    print(f"[3/3] 评估：ppl_token={m['ppl_token']:.2f}  ppl_char={m['ppl_char']:.2f}  "
          f"bpc={m['bpc']:.3f}  oov_rate={m['oov_rate']*100:.1f}%", flush=True)

    with open(args.model_out, "wb") as f:
        pickle.dump(lm, f)
    print(f"      模型已保存：{args.model_out}", flush=True)


if __name__ == "__main__":
    main()
