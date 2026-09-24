"""工具调用 SFT 制备：hermes-function-calling-v1 → PHD-Net 训练文本。

数据源（均 Apache 2.0）：
- data/hermes_func_calling.json：1893 组多轮工具对话（OpenAI 风格 <tool_call> 标记）
- data/hermes_glaive_5k.json：5209 组（含无工具调用的普通问答）
- data/qwen38_train0.parquet：取少量中文问答，保住中文应答风格

序列化（与 outputs/chat_openai.py 运行时渲染逐字一致）：
  系统：可用工具：{OpenAI tools JSON}
  用户：{question}
  助手：<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
  工具结果：{json}
  助手：{final answer}

合成增强：从语料的真实用户问题生成 120 组 web_search 调用对话
（运行时注册的就是这个工具，让调用签名与运行时绑定）。
"""

import argparse
import json
import os
import random

import pyarrow.parquet as pq

WS_SCHEMA = [{"type": "function", "function": {
    "name": "web_search",
    "description": "联网搜索：查询实时信息（新闻、天气、事件、事实核查）。",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "搜索关键词"},
        "n": {"type": "integer", "description": "返回条数，默认 5"}}}}}]


def cap(s, n):
    s = (s or "").strip()
    return s[:n] if len(s) <= n else s[:n].rstrip() + "…"


def serialize_conv(sys_json, turns):
    """完整序列化一轮对话：turns = [(kind, value)]，kind ∈ 用户/助手/工具结果。"""
    parts = [f"系统：可用工具：{sys_json}\n"]
    for kind, val in turns:
        parts.append(f"{kind}：{val}\n")
    return "".join(parts)


def load_hermes(path, n_tool, n_notool, cap_sys=400, cap_u=250, cap_a=500, cap_t=250):
    """从 ShareGPT 数组抽 n_tool 组带 <tool_call> 的对话 + n_notool 组无调用对话。

    按原顺序完整保留多轮（调用→工具结果→最终回答 的模式必须完整）。
    """
    data = json.load(open(path, encoding="utf-8"))
    rng = random.Random(7)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    out_tool, out_plain = [], []
    for i in idx:
        rec = data[i]
        convs = rec["conversations"]
        tools = rec.get("tools") or []
        if not tools:
            continue
        sys_json = cap(json.dumps(tools, separators=(",", ":"), ensure_ascii=False), cap_sys)
        has_call = any("<tool_call>" in c["value"] for c in convs if c["from"] == "gpt")

        def build_turns(conv_list):
            turns = []
            for c in conv_list:
                if c["from"] == "human":
                    turns.append(("用户", cap(c["value"], cap_u)))
                elif c["from"] == "gpt":
                    turns.append(("助手", cap(c["value"], cap_a)))
                elif c["from"] == "tool":
                    v = c["value"]
                    a, b = v.find("<tool_response>"), v.find("</tool_response>")
                    inner = v[a + len("<tool_response>"):b] if -1 not in (a, b) else v
                    turns.append(("工具结果", cap(inner, cap_t)))
            return turns

        if has_call:
            if len(out_tool) < n_tool:
                out_tool.append(serialize_conv(sys_json, build_turns(convs)))
        elif len(out_plain) < n_notool and len(convs) >= 3:
            last_h = next((c["value"] for c in reversed(convs) if c["from"] == "human"), "")
            last_g = next((c["value"] for c in reversed(convs) if c["from"] == "gpt"), "")
            if last_h and last_g:
                out_plain.append(serialize_conv(
                    sys_json, [("用户", cap(last_h, 200)), ("助手", cap(last_g, 400))]))
        if len(out_tool) >= n_tool and len(out_plain) >= n_notool:
            break
    return out_tool, out_plain


def synth_websearch(n, questions, seed=11):
    """合成 web_search 调用对话（真实用户问题 → 运行时工具签名）。"""
    rng = random.Random(seed)
    sys_json = cap(json.dumps(WS_SCHEMA, separators=(",", ":"), ensure_ascii=False), 300)
    out = []
    for q in questions[:n]:
        q = cap(q, 80)
        call = json.dumps({"name": "web_search",
                           "arguments": {"query": q, "n": 5}},
                          ensure_ascii=False, separators=(",", ":"))
        res = json.dumps({"results": [
            {"title": f"{q}·要点{i}", "url": f"https://www.example.com/s?k={i}",
             "snippet": f"与「{q}」相关的公开资料摘要 {i}。"} for i in (1, 2)]},
            ensure_ascii=False, separators=(",", ":"))
        final = f"根据搜索结果，以下是关于{q}的公开信息要点，供你参考。"
        out.append("".join([
            f"系统：可用工具：{sys_json}\n",
            f"用户：{q}\n",
            f"助手：<tool_call>\n{call}\n</tool_call>\n",
            f"工具结果：{res}\n",
            f"助手：{final}\n",
        ]))
    return out


def chinese_qa(path, n, cap_u=200, cap_a=600):
    t = pq.read_table(path).to_pydict()
    out, k = [], 0
    step = max(1, len(t["messages"]) // n)
    for m in t["messages"][::step]:
        u = next((x["content"] for x in m if x["role"] == "user"), "")
        a = next((x["content"] for x in m if x["role"] == "assistant"), "")
        a = a.split("</think>")[-1]                 # 去思考链，只留正文
        if 6 <= len(u) <= 200 and len(a) >= 15:
            out.append(f"用户：{cap(u, 200)}\n助手：{cap(a, 600)}\n")
            k += 1
        if k >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-train", default="data/tool_train.txt")
    ap.add_argument("--out-eval", default="data/tool_eval.txt")
    ap.add_argument("--eval-split", type=float, default=0.05)
    ap.add_argument("--hermes-n", type=int, default=70)
    ap.add_argument("--glaive-n", type=int, default=90)
    ap.add_argument("--plain-n", type=int, default=40)
    ap.add_argument("--synth-n", type=int, default=130)
    ap.add_argument("--zh-n", type=int, default=25)
    args = ap.parse_args()

    recs = []
    questions = []

    # 1) hermes 多轮 + glaive 单轮
    t1, p1 = load_hermes("data/hermes_func_calling.json", args.hermes_n, 20)
    print(f"hermes_func_calling: 工具 {len(t1)} / 无调用 {len(p1)}")
    recs += t1 + p1
    t2, p2 = load_hermes("data/hermes_glaive_5k.json", args.glaive_n, args.plain_n)
    print(f"hermes_glaive_5k: 工具 {len(t2)} / 无调用 {len(p2)}")
    recs += t2 + p2

    # 2) 合成 web_search（问题取自语料的真实用户提问）
    data = json.load(open("data/hermes_glaive_5k.json", encoding="utf-8"))
    for r in data:
        for c in r["conversations"]:
            if c["from"] == "human" and 6 <= len(c["value"]) <= 60:
                questions.append(c["value"])
    random.Random(3).shuffle(questions)
    syn = synth_websearch(args.synth_n, questions)
    print(f"合成 web_search: {len(syn)}")
    recs += syn

    # 3) 中文问答保底
    if os.path.exists("data/qwen38_train0.parquet"):
        zh = chinese_qa("data/qwen38_train0.parquet", args.zh_n)
        print(f"中文问答: {len(zh)}")
        recs += zh

    random.Random(5).shuffle(recs)
    neval = max(1, int(len(recs) * args.eval_split))
    train_text = "".join(r if r.endswith("\n\n") else r + "\n" for r in recs[:-neval])
    eval_text = "".join(r if r.endswith("\n\n") else r + "\n" for r in recs[-neval:])
    for p, s in [(args.out_train, train_text), (args.out_eval, eval_text)]:
        with open(p, "w", encoding="utf-8") as f:
            f.write(s)
    print(f"总 {len(recs)} 组 | 训练 {len(recs)-neval} 组 {len(train_text)} 字符 | "
          f"评估 {neval} 组 {len(eval_text)} 字符")
    print(f"写出 {args.out_train} / {args.out_eval}")


if __name__ == "__main__":
    main()
