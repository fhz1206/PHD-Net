"""把 ModelScope liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT 转写为
PHD-Net 词级 LM 训练文本（中文 Q&A 连续流）。

数据集：单文件 distill_r1_110k_sft.jsonl（678 MB / 110k 样本），每条含
prompt（问题）+ output（R1 蒸馏的思考链+答案）+ repo_name（溯源）。
本脚本流式读取、只取前 N 条，转写为「用户：<prompt>\n助手：<output>」格式，
供 PHDWordLM 词级训练——让模型学到中文问答的表面结构与行文风格。

用法：
    python tools/prepare_r1sft.py --max-samples 1000 --out-train data/r1sft_train.txt \
        --out-eval data/r1sft_eval.txt --eval-split 0.05

字段容错：动态探测 prompt/question/instruction/input 作问题，
output/answer/response 作答案；缺失则退化为仅 output。
"""

import argparse
import json
import os


SRC = "data/ds_r1/distill_r1_110k_sft.jsonl"

Q_KEYS = ("prompt", "question", "instruction", "input", "query")
A_KEYS = ("output", "answer", "response", "completion")


def _pick(d: dict, keys):
    for k in keys:
        if k in d and isinstance(d[k], str) and d[k].strip():
            return d[k].strip()
    # 退化：取第一个看起来像长文本的字段
    for k, v in d.items():
        if isinstance(v, str) and len(v) > 20 and k not in ("repo_name", "source"):
            return v
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--max-chars", type=int, default=0,
                    help="训练文本字符上限（0=不限，按样本数截取）")
    ap.add_argument("--out-train", default="data/r1sft_train.txt")
    ap.add_argument("--out-eval", default="data/r1sft_eval.txt")
    ap.add_argument("--eval-split", type=float, default=0.05)
    args = ap.parse_args()

    if not os.path.exists(args.src):
        raise SystemExit(f"未找到数据集：{args.src}\n请先 `git lfs pull` 拉取 LFS 文件。")

    buf = []
    neval = max(1, int(args.max_samples * args.eval_split))
    n_train = args.max_samples - neval
    n = 0
    with open(args.src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(d, dict):
                continue
            q = _pick(d, Q_KEYS)
            a = _pick(d, A_KEYS)
            if not a:
                continue
            rec = f"用户：{q}\n助手：{a}\n\n" if q else f"助手：{a}\n\n"
            buf.append(rec)
            n += 1
            if n >= args.max_samples:
                break

    train_recs = buf[:n_train]
    eval_recs = buf[n_train:]
    train_text = "".join(train_recs)
    eval_text = "".join(eval_recs)

    if args.max_chars and len(train_text) > args.max_chars:
        train_text = train_text[:args.max_chars]

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    with open(args.out_train, "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(args.out_eval, "w", encoding="utf-8") as f:
        f.write(eval_text)

    print(f"样本数={n} 训练={len(train_recs)} 评估={len(eval_recs)}")
    print(f"训练文本字符数={len(train_text)} 评估字符数={len(eval_text)}")
    print(f"写出: {args.out_train} / {args.out_eval}")


if __name__ == "__main__":
    main()
