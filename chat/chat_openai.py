"""OpenAI Chat Completions 风格对话循环（本地 PHD-Net 后端 + web_search 工具）。

工具调用（v2，SFT 工具调用训练后的模型驱动版）：
- 主路径：模型自己生成 <tool_call>\n{"name":...,"arguments":{...}}\n</tool_call>
  （训练自 hermes-function-calling-v1，SFT 制备见 tools/prepare_toolcall.py）。
  harness 解析该标记 → 转成 OpenAI assistant.tool_calls 消息 → 执行 → 追加
  role="tool" 消息 → 继续生成最终回答（最多 2 轮工具调用）。
- 备用路径（--fallback-rules，默认开）：模型没发出调用时，按关键词规则兜底合成调用，
  打印 [fallback] 标记以示区分。
- 协议面全程 OpenAI：system/user/assistant(tool_calls)/tool 消息、tool_call_id、
  每轮打印新增消息 JSON，全程会话存 openai_convo.json。

用法：
    python outputs/chat_openai.py --demo
    python outputs/chat_openai.py --model outputs/tool_model.pkl
"""

import argparse
import json
import os
import pickle
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from phdnet.word_lm import PHDWordLM                    # noqa: E402
from websearch_tool import TOOLS_SCHEMA, execute_tool_call  # noqa: E402

SYSTEM_PROMPT = ("可用工具：" +
                 json.dumps(TOOLS_SCHEMA, ensure_ascii=False, separators=(",", ":")))

SEARCH_HINTS = (
    "搜索", "搜一下", "查一下", "查查", "查一查", "联网", "最新", "最近", "今天",
    "今日", "现在", "目前", "新闻", "天气", "股价", "汇率", "比分", "发布", "上线",
    "多少钱", "哪一年", "什么时候", "是谁", "网址", "官网",
)

STOP_FINAL = ("\n用户：", "\n系统：")          # 防止续写出假想的下一轮


def load(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def needs_search(text: str) -> bool:
    return any(h in text for h in SEARCH_HINTS)


def make_query(text: str) -> str:
    q = re.sub(r"^(请|帮我|麻烦|帮我一下|给我)?(搜索|搜一下|查一下|查查|查一查|联网查|联网搜索)?(一下)?",
               "", text.strip())
    q = re.sub(r"[?？!！。，,]+$", "", q).strip()
    return q[:48] if q else text.strip()[:48]


# ---------------------------------------------------------------- 生成
def _warmup(lm, prompt, max_ctx=6000):
    prompt = prompt[-max_ctx:]
    prev = None
    for t in lm.tokenize(prompt):
        lm.net.step(lm.tok.encode_composite(t, prev), learn=False)
        prev = t
    return prev


def generate(lm, prompt, n_tokens=200, tau=0.7, seed=0, stops=()):
    """预热 + 自回归采样；遇到 stops 中的子串立即截断（工具调用停止序列）。"""
    rng = np.random.default_rng(seed)
    prev = _warmup(lm, prompt)
    cur = prev if prev is not None else lm.tok.tokens[0]
    text = ""
    for _ in range(n_tokens):
        y = lm.net.step(lm.tok.encode_composite(cur, prev), learn=False)["y"] / max(tau, 1e-6)
        y -= y.max()
        p = np.exp(y)
        p /= p.sum()
        nxt = lm.tok.tokens[int(rng.choice(len(p), p=p))]
        text += nxt
        cut = -1
        for s in stops:
            i = text.find(s)
            if i >= 0 and (cut < 0 or i < cut):
                cut = i
        if cut >= 0:
            return text[:cut]
        prev, cur = cur, nxt
    return text


def parse_tool_calls(text):
    """解析模型生成的 <tool_call>{json}</tool_call> → [ {"name","arguments"} ]。"""
    calls = []
    for body in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        try:
            d = json.loads(body)
            if isinstance(d, dict) and d.get("name"):
                calls.append({"name": d["name"],
                              "arguments": d.get("arguments") or {}})
        except json.JSONDecodeError:
            continue
    return calls


# ---------------------------------------------------------------- OpenAI 渲染
def render_messages(messages) -> str:
    """OpenAI messages → 训练分布内的文本（与 prepare_toolcall.py 逐字一致）。"""
    parts = []
    for m in messages:
        role, content = m.get("role"), m.get("content") or ""
        if role == "system":
            if content:
                parts.append(f"系统：{content}\n")
        elif role == "user":
            parts.append(f"用户：{content}\n")
        elif role == "assistant":
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    call = {"name": tc["function"]["name"], "arguments": args}
                    parts.append("助手：<tool_call>\n"
                                 + json.dumps(call, ensure_ascii=False,
                                              separators=(",", ":"))
                                 + "\n</tool_call>\n")
            elif content:
                parts.append(f"助手：{content}\n")
        elif role == "tool":
            c = content if content.startswith("{") else json.dumps({"content": content},
                                                                   ensure_ascii=False)
            parts.append(f"工具结果：{c}\n")
    return "".join(parts)


def run_turn(lm, messages, n_tokens=200, tau=0.7, force_search=None,
             fallback_rules=True, verbose=True):
    """处理一条用户消息 → 更新 messages（OpenAI 协议）→ 返回最终回复文本。"""
    last_user = messages[-1]["content"]
    new_msgs, via = [], []

    def emit(m, tag=None):
        messages.append(m)
        new_msgs.append(m)
        if verbose:
            print("  [OpenAI] " + (f"[{tag}] " if tag else "") +
                  json.dumps(m, ensure_ascii=False)[:300])

    rounds = 0
    while True:
        prompt = render_messages(messages) + "助手："
        stops = STOP_FINAL if rounds else STOP_FINAL + ("</tool_call>",)
        out = generate(lm, prompt, n_tokens=n_tokens, tau=tau, stops=stops)
        calls = parse_tool_calls(out) if rounds < 2 else []

        if calls:
            tcs = [{"id": f"call_{int(time.time()*1000)%10**8:08d}_{i}",
                    "type": "function",
                    "function": {"name": c["name"],
                                 "arguments": json.dumps(c["arguments"],
                                                         ensure_ascii=False)}}
                   for i, c in enumerate(calls)]
            emit({"role": "assistant", "content": None, "tool_calls": tcs},
                 tag="model")
            for tc in tcs:
                result = execute_tool_call(tc)
                try:
                    result["_query"] = json.loads(tc["function"]["arguments"]).get("query", "")
                except json.JSONDecodeError:
                    pass
                emit(result)
            rounds += 1
            continue

        # 备用：模型没发调用 → 关键词规则兜底（仅第一轮）
        if (rounds == 0 and fallback_rules and force_search is not False
                and needs_search(last_user) and last_user.strip()):
            emit({"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{int(time.time()*1000)%10**8:08d}",
                "type": "function",
                "function": {"name": "web_search",
                             "arguments": json.dumps(
                                 {"query": make_query(last_user), "n": 5},
                                 ensure_ascii=False)}}]}, tag="fallback")
            result = execute_tool_call(messages[-1]["tool_calls"][0])
            result["_query"] = make_query(last_user)
            emit(result)
            rounds += 1
            continue

        final_text = re.sub(r"<tool_call>.*?</tool_call>", "", out, flags=re.S).strip()
        emit({"role": "assistant", "content": final_text})
        return final_text


DEMO_TURNS = [
    ("今天有什么值得关注的科技新闻？", None),
    ("用一句话介绍什么是机器学习。", False),
    ("PHD 是什么学位？现在博士就业形势如何？", None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "tool_model.pkl"))
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--n-tokens", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--no-fallback", action="store_true",
                    help="关闭关键词兜底，只用模型自己发出的 tool_call")
    ap.add_argument("--save", default=os.path.join(HERE, "openai_convo.json"))
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"未找到模型 {args.model}\n请先训练（见 tools/prepare_toolcall.py）。")
    print("加载模型...", flush=True)
    lm = load(args.model)
    print(f"模型已加载（词表 {len(lm.tok)}）。已注册工具："
          f"{[t['function']['name'] for t in TOOLS_SCHEMA]}"
          f"（fallback={'关' if args.no_fallback else '开'}）\n", flush=True)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if args.demo:
        for q, force in DEMO_TURNS:
            print("─" * 60)
            print("你 > " + q)
            messages.append({"role": "user", "content": q})
            run_turn(lm, messages, n_tokens=args.n_tokens, tau=args.tau,
                     force_search=force, fallback_rules=not args.no_fallback)
    else:
        print("交互模式（OpenAI 协议 + web_search 工具）。Ctrl-C 退出。\n")
        try:
            while True:
                q = input("你 > ").strip()
                if not q:
                    continue
                messages.append({"role": "user", "content": q})
                run_turn(lm, messages, n_tokens=args.n_tokens, tau=args.tau,
                         fallback_rules=not args.no_fallback)
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")

    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"\n完整 OpenAI 会话已存：{args.save}")


if __name__ == "__main__":
    main()
