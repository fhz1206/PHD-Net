"""Claude Code traces → 训练文本（session/*.jsonl → 对话式纯文本）。

数据源：ModelScope ``choucisan/mimo-claude-code-traces-1k``
（1,017 个 trace / 15,046 events / 11,995 messages / 5,271 tool calls，
10 个任务类别，JSONL event-stream 格式，MIT License）。

为什么要转换而不是直接喂 JSONL
--------------------------------
训练流消费的是**纯文本**（`WordTokenizer` / `PHDWordLM` 按字符/词切分）。
原始 event-stream 里有 ``parentUuid`` / ``uuid`` / ``timestamp`` / ``gitBranch``
等大量**与语言建模无关的元数据**，直接拼进去会让模型把容量浪费在记 UUID
上。本脚本只抽取**语言内容**并按对话顺序还原：

    user: <用户消息>
    assistant: <思考 + 回复>
    tool: <工具调用名 + 参数摘要>
    tool_result: <工具返回>

保留 ``thinking`` 块（reasoning trace 是这份数据的独特价值——它展示
「面对问题时如何一步步想」，对下一 token 预测是有信息的监督信号）。

用法
----
    # 列出全部类别与文件数
    py -3.14 tools/data/cc_traces_to_text.py --list

    # 转成一个 parquet（本地路径，供 --data 使用）
    py -3.14 tools/data/cc_traces_to_text.py --out datasets/cc_traces/cc_000.parquet

    # 只取调试类 + 生成式，按 trace 分片（每片 ~100 traces）
    py -3.14 tools/data/cc_traces_to_text.py --categories debugging code_generation \\
        --out datasets/cc_traces/cc_000.parquet --per-shard 100

    # 直接从本地目录读（已下载的副本）
    py -3.14 tools/data/cc_traces_to_text.py --local-dir /path/to/session \\
        --out datasets/cc_traces/cc_000.parquet

    # 只打印第一条 trace 的转换结果（调试用）
    py -3.14 tools/data/cc_traces_to_text.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MS_REPO = "choucisan/mimo-claude-code-traces-1k"

# 与数据集 README 一致的 10 个类别
CATEGORIES = [
    "algorithms", "api_integration", "code_generation", "data_processing",
    "debugging", "hf_trace", "math_problems", "refactoring", "shell_devops",
    "supplement",
]

SEP = "\n\n"          # trace 之间的分隔（与 corpus 层约定一致）
MAX_TOOL_INPUT = 400  # 工具参数摘要的最大字符数（防超长命令行刷屏）


# ── 远程列目录 ────────────────────────────────────────────────────────
def list_trace_files(categories: list[str] | None = None) -> list[tuple[str, int]]:
    """列 ModelScope 上的 trace 文件 → [(相对路径, 字节数)]，确定性排序。"""
    from phdnet.ms_stream import _api
    api = _api()
    cats = categories or CATEGORIES
    out: list[tuple[str, int]] = []
    for cat in cats:
        page = 1
        while True:
            files = api.get_dataset_files(
                repo_id=MS_REPO, root_path=f"/session/{cat}",
                recursive=False, page_number=page, page_size=100)
            if not files:
                break
            for f in files:
                if f.get("Type") == "blob" and str(f.get("Path", "")).endswith(".jsonl"):
                    out.append((f["Path"], int(f.get("Size") or 0)))
            page += 1
    return sorted(out)


def fetch_trace(rel_path: str) -> str:
    """取单个 trace 原文（ms:// HTTP Range，零落盘）。"""
    from phdnet.ms_stream import open_ms_file
    with open_ms_file(f"ms://{MS_REPO}/{rel_path}") as fh:
        raw = fh.read()
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw


# ── event-stream → 对话文本 ─────────────────────────────────────────
def _clip(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + "\n…[截断]"


def _content_to_text(content, thinking: bool = True) -> str:
    """把 message.content（str 或 block 列表）转成可读文本。"""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for b in content:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text" and b.get("text"):
            parts.append(b["text"].strip())
        elif bt == "thinking" and thinking and b.get("thinking"):
            # reasoning trace：这份数据的独特价值，保留（明确标注来源）
            parts.append("[思考] " + b["thinking"].strip())
        elif bt == "tool_use":
            name = str(b.get("name", "?"))
            inp = json.dumps(b.get("input", {}), ensure_ascii=False)
            parts.append(f"[调用工具 {name}] {_clip(inp, MAX_TOOL_INPUT)}")
        elif bt == "tool_result":
            c = b.get("content")
            if isinstance(c, list):
                c = "\n".join(str(x.get("text", "")) for x in c
                              if isinstance(x, dict))
            parts.append("[工具返回] " + _clip(str(c or ""), MAX_TOOL_INPUT))
    return "\n".join(p for p in parts if p)


def trace_to_text(raw: str, thinking: bool = True) -> str:
    """单个 JSONL event-stream → 一段对话文本。

    只保留 ``type ∈ {user, assistant}`` 的行（跳过 mode / permission-mode /
    last-prompt 等会话元数据行）；按原文件顺序输出（event-stream 本身即时序）。
    """
    lines: list[str] = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
        except Exception:
            continue
        if not isinstance(o, dict):
            continue
        t = o.get("type")
        if t not in ("user", "assistant"):
            continue
        msg = o.get("message") or {}
        role = msg.get("role") or ("user" if t == "user" else "assistant")
        body = _content_to_text(msg.get("content"), thinking=thinking)
        if not body:
            continue
        lines.append(f"{role}: {body}")
    return SEP.join(lines)


# ── 写出 ────────────────────────────────────────────────────────────
def write_parquet(texts: list[str], dst: Path, lang: str = "en") -> int:
    """写成 parquet（列 text / lang / src），与 datasets/ 既有 schema 对齐。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    dst.parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.table({
        "text": pa.array(texts, type=pa.string()),
        "lang": pa.array([lang] * len(texts), type=pa.string()),
        "src": pa.array(["cc-traces"] * len(texts), type=pa.string()),
    })
    pq.write_table(tbl, dst, compression="zstd")
    return len(texts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Claude Code traces → 训练文本（parquet）")
    ap.add_argument("--list", action="store_true",
                    help="只列出远程 trace 文件（类别/数量/总字节）后退出")
    ap.add_argument("--dry-run", action="store_true",
                    help="只转换第一条 trace 并打印（不写文件）")
    ap.add_argument("--local-dir", type=Path, default=None,
                    help="从本地目录读 trace（已下载的副本；递归找 *.jsonl）")
    ap.add_argument("--categories", nargs="*", default=None,
                    help="只取指定类别（默认全部 10 类）")
    ap.add_argument("--per-shard", type=int, default=100,
                    help="每个 parquet 分片包含多少条 trace（默认 100）")
    ap.add_argument("--max-traces", type=int, default=0,
                    help="最多转换多少条（0=不限；调试用）")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出 parquet 路径（分片时自动加 _NNN 序号）")
    ap.add_argument("--no-thinking", action="store_true",
                    help="丢弃 assistant 的 thinking 块（只留最终回复）")
    args = ap.parse_args()

    # ── 本地目录模式 ──
    if args.local_dir is not None:
        files = sorted(str(p) for p in args.local_dir.rglob("*.jsonl"))
        if not files:
            print(f"本地目录无 *.jsonl: {args.local_dir}")
            sys.exit(1)
        print(f"[local] {len(files)} 个 trace 文件（{args.local_dir}）")
        items = [(f, 0) for f in files]
        fetcher = lambda rel: Path(rel).read_text(encoding="utf-8")  # noqa: E731
    else:
        items = list_trace_files(args.categories)
        if not items:
            print("[remote] 未找到任何 trace 文件（检查类别名 / 网络 / MS_TOKEN）")
            sys.exit(1)
        by_cat: dict[str, int] = {}
        for rel, _ in items:
            cat = rel.split("/")[1] if "/" in rel else "?"
            by_cat[cat] = by_cat.get(cat, 0) + 1
        total_mb = sum(sz for _, sz in items) / 1e6
        print(f"[remote] {len(items)} 个 trace，合计 {total_mb:.1f} MB")
        for c in sorted(by_cat):
            print(f"    {c:20} {by_cat[c]:>5}")
        if args.list:
            return
        fetcher = fetch_trace

    if args.max_traces:
        items = items[:args.max_traces]

    if args.dry_run:
        rel = items[0][0]
        print(f"\n=== dry-run: {rel} ===")
        print(trace_to_text(fetcher(rel), thinking=not args.no_thinking))
        return

    if args.out is None:
        print("--out 未给（或用 --dry-run / --list）")
        sys.exit(1)

    # ── 转换 + 分片写出 ──
    buf: list[str] = []
    shard = 0
    n_ok = n_empty = 0
    for i, (rel, _sz) in enumerate(items):
        try:
            txt = trace_to_text(fetcher(rel), thinking=not args.no_thinking)
        except Exception as e:                      # noqa: BLE001
            print(f"  ! {rel}: {type(e).__name__}: {e}")
            continue
        if txt.strip():
            buf.append(txt)
            n_ok += 1
        else:
            n_empty += 1
        if len(buf) >= args.per_shard:
            dst = (args.out if shard == 0 and len(items) <= args.per_shard
                   else args.out.with_name(f"{args.out.stem}_{shard:03d}{args.out.suffix}"))
            n = write_parquet(buf, dst)
            print(f"  → {dst.name}: {n} traces")
            buf, shard = [], shard + 1
        if (i + 1) % 100 == 0:
            print(f"  … {i + 1}/{len(items)}")
    if buf:
        dst = (args.out if shard == 0
               else args.out.with_name(f"{args.out.stem}_{shard:03d}{args.out.suffix}"))
        n = write_parquet(buf, dst)
        print(f"  → {dst.name}: {n} traces")
    print(f"\n完成：{n_ok} 条转换成功，{n_empty} 条为空，共 {shard + (1 if buf else 0)} 个分片")


if __name__ == "__main__":
    main()
