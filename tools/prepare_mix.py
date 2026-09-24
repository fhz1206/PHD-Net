"""混合制备 CodeX-2M-Thinking + Qwen3.8-27B-Distillation-40K 训练文本。

两库均 Apache 2.0。CPU 预算策略：
- CodeX（英文编程+推理）：input 截 600 字符、output 截 1600 字符，均匀抽 --codex-n 条；
- Qwen38（code/math/science/logic 四域蒸馏）：messages[0]=user、messages[1]=assistant
  （含 <think>…</think>），user 截 500、assistant 截 2000，按域轮转抽 --qwen-n 条；
- 输出「用户：…\n助手：…\n\n」连续流，末尾 5% 作评估段。

用法：python tools/prepare_mix.py   # 默认 codex 120 条 + qwen 100 条 ≈ 50 万字符
"""

import argparse
import os

import pyarrow.parquet as pq


def cap(s: str, n: int) -> str:
    s = (s or "").strip()
    return s[:n] if len(s) <= n else s[:n].rstrip() + "…"


def load_codex(path, n):
    t = pq.read_table(path, columns=["input", "output"]).to_pydict()
    total = len(t["input"])
    idx = [int(i * total / n) for i in range(n)]
    out = []
    for i in idx:
        out.append(f"用户：{cap(t['input'][i], 600)}\n助手：{cap(t['output'][i], 1600)}\n\n")
    return out


def load_qwen(path, n):
    t = pq.read_table(path).to_pydict()
    msgs, domains = t["messages"], t["domain"]
    by = {}
    for m, d in zip(msgs, domains):
        by.setdefault(d, []).append(m)
    order, k = [], 0
    while len(order) < n:                       # 按域轮转，保证四域均衡
        d = list(by)[k % len(by)]
        if by[d]:
            order.append(by[d].pop(0))
        k += 1
    out = []
    for m in order:
        u = next((x["content"] for x in m if x["role"] == "user"), "")
        a = next((x["content"] for x in m if x["role"] == "assistant"), "")
        out.append(f"用户：{cap(u, 500)}\n助手：{cap(a, 2000)}\n\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codex", default="data/codex2m_shard0.parquet")
    ap.add_argument("--qwen", default="data/qwen38_train0.parquet")
    ap.add_argument("--codex-n", type=int, default=120)
    ap.add_argument("--qwen-n", type=int, default=100)
    ap.add_argument("--out-train", default="data/mix_train.txt")
    ap.add_argument("--out-eval", default="data/mix_eval.txt")
    ap.add_argument("--eval-split", type=float, default=0.05)
    args = ap.parse_args()

    recs = []
    if os.path.exists(args.codex):
        recs += load_codex(args.codex, args.codex_n)
        print(f"CodeX 取 {args.codex_n} 条")
    else:
        print(f"[warn] 缺 {args.codex}，跳过")
    if os.path.exists(args.qwen):
        recs += load_qwen(args.qwen, args.qwen_n)
        print(f"Qwen38 取 {args.qwen_n} 条")
    else:
        print(f"[warn] 缺 {args.qwen}，跳过")
    if not recs:
        raise SystemExit("没有任何数据可用")

    neval = max(1, int(len(recs) * args.eval_split))
    train_text = "".join(recs[:-neval])
    eval_text = "".join(recs[-neval:])
    for p, s in [(args.out_train, train_text), (args.out_eval, eval_text)]:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(s)
    print(f"总 {len(recs)} 条 | 训练 {len(recs)-neval} 条 {len(train_text)} 字符 | "
          f"评估 {neval} 条 {len(eval_text)} 字符")
    print(f"写出 {args.out_train} / {args.out_eval}")


if __name__ == "__main__":
    main()
