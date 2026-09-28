"""联网搜索工具（OpenAI function calling 风格封装）。

为 PHD-Net 对话循环配发的真实网络搜索工具，协议面完全按 OpenAI 规范：
- TOOLS_SCHEMA：OpenAI Chat Completions 的 tools 数组（type="function" + JSON Schema 参数）。
- web_search(query, n)：工具执行体，返回 OpenAI tool 消息 content 所需的文本。
- execute_tool_call(tool_call)：按 assistant.tool_calls 的 JSON 结构解析并执行，
  返回 {"role": "tool", "tool_call_id": ..., "content": ...} 标准消息。

搜索后端（无需 API key）：Bing CN → Bing 国际版 → DuckDuckGo Lite，逐级回退；
HTML 用正则轻量解析（标题/链接/摘要），不引入 bs4 依赖。

自测：python outputs/websearch_tool.py [查询词]
"""

import html as _html
import json
import re

import requests

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_HEADERS = {"User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}


# ---------------------------------------------------------------- OpenAI 协议面
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "联网搜索工具：查询实时信息（新闻、天气、近期事件、"
                           "事实核查、任何训练语料中可能没有的内容）。"
                           "返回若干条搜索结果（标题、摘要、链接）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "n": {"type": "integer", "description": "返回结果条数，默认 5",
                          "default": 5},
                },
                "required": ["query"],
            },
        },
    }
]


def execute_tool_call(tool_call: dict) -> dict:
    """按 OpenAI assistant.tool_calls 结构执行工具，返回标准 role="tool" 消息。

    tool_call = {"id": "call_xxx", "type": "function",
                 "function": {"name": "web_search", "arguments": "{\"query\": ...}"}}
    """
    fn = tool_call.get("function", {})
    name = fn.get("name", "")
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}
    if name != "web_search":
        content = f"[tool error] 未知工具：{name}（本会话只注册了 web_search）"
    else:
        content = web_search_text(args.get("query", ""), int(args.get("n", 5)))
    return {"role": "tool", "tool_call_id": tool_call.get("id", ""),
            "content": content}


# ---------------------------------------------------------------- 工具执行体
def _strip(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = _html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def _get(url: str, timeout: float = 10.0) -> str:
    r = requests.get(url, headers=_HEADERS, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    return r.text


def _parse_bing(page: str, n: int):
    out = []
    for block in re.findall(r'<li class="b_algo".*?</li>', page, re.S)[:n * 2]:
        m = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                      block, re.S)
        if not m:
            continue
        url, title = m.group(1), _strip(m.group(2))
        sm = re.search(r'<p[^>]*>(.*?)</p>', block, re.S)
        snippet = _strip(sm.group(1)) if sm else ""
        if title:
            out.append({"title": title, "url": url, "snippet": snippet[:200]})
        if len(out) >= n:
            break
    return out


def _parse_ddg(page: str, n: int):
    out = []
    links = re.findall(r"<a[^>]*href=\"(http[^\"]+)\"[^>]*>(.*?)</a>", page, re.S)
    snippets = re.findall(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
                          page, re.S)
    seen = set()
    for i, (url, title) in enumerate(links):
        title = _strip(title)
        if not title or "duckduckgo.com" in urlparse_host(url):
            continue
        snippet = _strip(snippets[i]) if i < len(snippets) else ""
        if url in seen:
            continue
        seen.add(url)
        out.append({"title": title, "url": url, "snippet": snippet[:200]})
        if len(out) >= n:
            break
    return out


def urlparse_host(url: str) -> str:
    m = re.match(r"https?://([^/]+)/", url + "/")
    return (m.group(1) if m else "").lower()


def web_search(query: str, n: int = 5, backend: str = "auto") -> dict:
    """执行搜索。返回 {"backend": 名称, "results": [{title,url,snippet}]}。"""
    query = (query or "").strip()
    if not query:
        return {"backend": "none", "results": []}
    q = requests.utils.quote(query)
    backends = [
        ("bing_cn", f"https://cn.bing.com/search?q={q}&setlang=zh-CN", _parse_bing),
        ("bing", f"https://www.bing.com/search?q={q}", _parse_bing),
        ("ddg", f"https://lite.duckduckgo.com/lite/?q={q}", _parse_ddg),
    ]
    if backend != "auto":
        backends = [b for b in backends if b[0] == backend]
    last_err = None
    for name, url, parser in backends:
        try:
            results = parser(_get(url), n)
            if results:
                return {"backend": name, "results": results}
            last_err = RuntimeError(f"{name} 返回 0 条结果")
        except Exception as e:                      # noqa: BLE001 - 工具层兜底
            last_err = e
    return {"backend": "failed", "results": [],
            "error": f"{type(last_err).__name__}: {last_err}"}


def web_search_text(query: str, n: int = 5) -> str:
    """工具执行体的文本形态（即 OpenAI tool 消息的 content）。"""
    r = web_search(query, n)
    if r["backend"] == "failed":
        return f"[web_search 失败] {r.get('error', '未知错误')}"
    if not r["results"]:
        return f"[web_search] 「{query}」没有搜到结果（后端 {r['backend']}）"
    lines = [f"[web_search 后端={r['backend']}] 「{query}」共 {len(r['results'])} 条："]
    for i, it in enumerate(r["results"], 1):
        lines.append(f"{i}. {it['title']}")
        if it["snippet"]:
            lines.append(f"   摘要：{it['snippet']}")
        lines.append(f"   链接：{it['url']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "人工智能 最新新闻"
    print(web_search_text(q, 5))
