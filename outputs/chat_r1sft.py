"""与训练好的 PHD-Net（R1 蒸馏 SFT 子集）跑对话 / 演示续写。

前置：先运行 outputs/train_r1sft.py 生成 outputs/r1sft_model.pkl。

用法：
    python outputs/chat_r1sft.py                 # 交互式：逐行输入问题，Ctrl-C 退出
    python outputs/chat_r1sft.py --demo          # 用内置几个问题跑演示续写

诚实边界：模型是词级联想续写器，不是语义聊天机器人。它会顺着"用户：..助手："
的格式生成**看起来像问答的中文文本**，但不保证事实正确或真正回答你的问题。
"""

import argparse
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from phdnet.word_lm import PHDWordLM


def load(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def reply(lm, prompt, n_tokens=160, tau=0.7, seed=0):
    """把 prompt 作为上下文"预热"网络状态，再自回归续写 n_tokens。"""
    rng = np.random.default_rng(seed)
    toks = lm.tokenize(prompt)
    prev, cur = None, None
    for t in toks:                       # 预热：更新内部状态，不学习
        lm.net.step(lm.tok.encode_composite(t, prev), learn=False)
        prev, cur = t, t
    if cur is None:
        cur = lm.tok.tokens[0]
    out = []
    for _ in range(n_tokens):
        y = lm.net.step(lm.tok.encode_composite(cur, prev), learn=False)["y"] / max(tau, 1e-6)
        y -= y.max()
        p = np.exp(y)
        p /= p.sum()
        nxt = lm.tok.tokens[int(rng.choice(len(p), p=p))]
        out.append(nxt)
        prev, cur = cur, nxt
    return "".join(out)


DEMO_QUERIES = [
    "用户：什么是机器学习？\n助手：",
    "用户：请用一句话解释量子纠缠。\n助手：",
    "用户：推荐一本适合初学者读的书。\n助手：",
    "用户：1+1 等于几？\n助手：",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/r1sft_model.pkl")
    ap.add_argument("--demo", action="store_true", help="用内置问题跑演示")
    ap.add_argument("--n-tokens", type=int, default=160)
    ap.add_argument("--tau", type=float, default=0.7)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"未找到模型 {args.model}\n请先运行 outputs/train_r1sft.py 训练。")
    print("加载模型中...", flush=True)
    lm = load(args.model)
    print(f"模型已加载（词表 {len(lm.tok)}）。\n", flush=True)

    if args.demo:
        for q in DEMO_QUERIES:
            print("─" * 60)
            print(q.strip())
            ans = reply(lm, q, n_tokens=args.n_tokens, tau=args.tau)
            print("助手（续写）：" + ans)
        return

    print("交互模式：输入问题后回车。Ctrl-C 退出。\n")
    try:
        while True:
            q = input("你 > ").strip()
            if not q:
                continue
            prompt = f"用户：{q}\n助手："
            print("助手 > " + reply(lm, prompt, n_tokens=args.n_tokens, tau=args.tau))
    except (EOFError, KeyboardInterrupt):
        print("\n再见。")


if __name__ == "__main__":
    main()
