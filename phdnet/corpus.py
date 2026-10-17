"""统一语料读取层 —— txt / parquet / jsonl 三种形态，一套接口。

================================================================================
为什么（fhz 2026-09-24：「修改模型导入，改为 parquet」）
================================================================================
纯文本语料的三个痛点：① 体积大（M7 Core 10.8 GB，压缩前）；② 无结构（SFT 的
instruction/output 边界只能靠空行猜，导致 `analyze_corpus_role.py` 初版统计失真）；
③ 无法按列抽样/过滤（要筛中文子集必须全量扫一遍）。

parquet 的好处：列式 + 内建压缩（实测 UltraInteract 616 MB → 87 MB，7.05×）、
保留字段结构（`text` / `lang` / `src`）、支持只读部分 row group（大语料可分片
流式训练，不必整文件载入内存）。

**兼容性铁律**：本模块只**新增**读取能力，不改动任何既有默认路径——
`tests/evals/eval_common.py` 等仍读 `eval_corpus/internal_corpus.txt`（纯文本），
行为逐位不变；parquet 仅在显式给出 `.parquet` 路径时启用。

用法：
    from phdnet.corpus import load_text, iter_texts, write_parquet
    text = load_text("datasets/sft/x.parquet", limit_chars=4_000)
    for t in iter_texts("datasets/pretrain/y.parquet"):   # 流式，不全量载入
        ...
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

__all__ = ["load_text", "iter_texts", "write_parquet", "corpus_stats",
           "open_parquet_source", "cc_trace_to_text", "iter_cc_traces",
           "load_cc_traces"]

_SEP = "\n\n"


def is_remote_path(path) -> bool:
    """是否 ModelScope 远程 spec（ms://…；phdnet.ms_stream.is_remote_path 转发）。"""
    from phdnet.ms_stream import is_remote
    return is_remote(path)


def open_parquet_source(p):
    """Path 或 MsFile → pyarrow ParquetFile。

    本地：``ParquetFile(str(p))``（与旧行为**完全一致**，逐位不变）；
    远程：``ms://`` spec → HTTP Range 可 seek 流（phdnet/ms_stream.py），
    网络抖动按指数退避重试（审计 C1：长跑训练不该被单次 5xx 打断；
    重试只重建流，不改变任何数值 → 逐位等价）。
    """
    pq = _require_pyarrow()
    if not is_remote_path(p):
        return pq.ParquetFile(str(p))
    import time as _time
    from phdnet.ms_stream import open_ms_file, _READ_RETRIES, _RETRY_BACKOFF_S
    last: Exception | None = None
    for k in range(_READ_RETRIES):
        try:
            return pq.ParquetFile(open_ms_file(str(p)))
        except Exception as e:                      # noqa: BLE001
            last = e
            if k + 1 < _READ_RETRIES:
                _time.sleep(_RETRY_BACKOFF_S ** (k + 1))
    raise RuntimeError(
        f"ms:// 打开失败（重试 {_READ_RETRIES} 次）: {p}") from last


def _require_pyarrow():
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet as pq
        return pq
    except ImportError as e:      # pragma: no cover
        raise SystemExit(
            "读取 parquet 需要 pyarrow：`pip install pyarrow`"
            "（国内源：pip install pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple）"
        ) from e


def expand_paths(path: str | Path | list) -> list[Path]:
    """路径展开：支持 glob 通配符（如 `pretrain_*.parquet`），返回确定性排序的文件列表。

    远程（fhz 2026-09-29）：`ms://<ns>/<path>` 走 ModelScope 直连
    （phdnet/ms_stream.py，HTTP Range 流式零落盘）；本地逻辑一字未动。

    P196：**接受已展开的列表**（原样返回）——调用方若已自行展开
    （如 cc_traces 的两级目录按类别逐个 glob），不能再走一遍 expand_ms
    （会重复列目录 + 把跨目录 pattern 拒掉）。这让
    `expand_paths` 对下游（PrefetchChars / iter_texts）幂等。
    """
    if isinstance(path, (list, tuple)):
        return list(path)
    if is_remote_path(path):
        from phdnet.ms_stream import expand_ms
        return expand_ms(str(path))
    p = Path(path)
    if any(ch in p.name for ch in "*?["):
        hits = sorted(p.parent.glob(p.name))
        if not hits:
            raise FileNotFoundError(f"glob 无匹配文件: {path}")
        return hits
    return [p]


def _iter_one(p, column: str, batch_size: int):
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        pf = open_parquet_source(p)
        cols = [c for c in (column,) if c in pf.schema_arrow.names] or None
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            col = batch.column(column if cols else 0)
            for i in range(batch.num_rows):
                v = col[i].as_py()
                if v:
                    yield v
    elif suffix in (".jsonl", ".json"):
        # P196：远程 .jsonl 直读（ms:// 经 HTTP Range 流式，零落盘）。
        # 用途：Claude Code traces（choucisan/mimo-claude-code-traces-1k，
        # 1,017 个 session/*.jsonl）这类**非 parquet** 的 ModelScope 数据集。
        # ⚠ 远程流是**二进制**且可能跨块断行 → 按字节读入后自行 splitlines，
        #   不依赖 fsspec 的行缓冲（其 block_size 语义对行级迭代不可靠）。
        if is_remote_path(p):
            from .ms_stream import open_ms_file
            with open_ms_file(str(p)) as fh:
                raw = fh.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if isinstance(d, dict):
                    v = d.get(column) or d.get("text") or d.get("content")
                else:
                    v = d
                if v:
                    yield v
            return
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if isinstance(d, dict):
                    v = d.get(column) or d.get("text") or d.get("content")
                else:
                    v = d
                if v:
                    yield v
    else:                                   # 纯文本：按空行分块（既有约定）
        with open(p, encoding="utf-8") as f:
            buf = []
            for line in f:
                if line.strip() == "":
                    if buf:
                        yield "".join(buf).strip()
                        buf = []
                else:
                    buf.append(line)
            if buf:
                yield "".join(buf).strip()


def iter_texts(path: str | Path, column: str = "text",
               batch_size: int = 2000):
    """流式迭代语料中的每条文本；path 支持单个文件或 glob 模式（多分片顺序拼接）。"""
    for p in expand_paths(path):
        yield from _iter_one(p, column, batch_size)


def load_text(path: str | Path, limit_chars: int | None = None,
              sep: str = _SEP) -> str:
    """把语料拼成一个字符串（供 WordTokenizer / PHDWordLM 直接消费）。

    path 支持单文件或 glob（如 `datasets/pretrain/pretrain_*.parquet`，分片按文件名
    排序拼接）；parquet/jsonl 每条样本按 `sep` 连接，纯文本原样读入（与旧行为一致）。
    """
    parts: list[str] = []
    n = 0
    any_txt = False
    paths = expand_paths(path)          # 展开一次复用（远程 glob 原本会二次
                                        # 列目录发网络请求，审计 B4）
    for p in paths:
        if p.suffix.lower() in (".parquet", ".jsonl", ".json"):
            # P196：远程 .jsonl 走 open_ms_file 整文件流式读入（零落盘），
            # 与本地 jsonl 同口径（逐行 JSON → column 字段）。
            if is_remote_path(p) and p.suffix.lower() in (".jsonl", ".json"):
                from .ms_stream import open_ms_file
                with open_ms_file(str(p)) as fh:
                    raw = fh.read()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    v = None
                    if isinstance(d, dict):
                        v = d.get("text") or d.get("content") or d.get(column)
                    elif isinstance(d, str):
                        v = d
                    if v:
                        parts.append(v)
                        n += len(v) + len(sep)
                        if limit_chars and n >= limit_chars:
                            return (sep.join(parts)[:limit_chars])
                continue
            for t in _iter_one(p, "text", 2000):
                parts.append(t)
                n += len(t) + len(sep)
                if limit_chars and n >= limit_chars:
                    return (sep.join(parts)[:limit_chars])
        else:
            if is_remote_path(p):
                raise NotImplementedError(
                    f"ms:// 远程源仅支持 .parquet（收到 {p.name}）")
            any_txt = True
            with open(p, encoding="utf-8") as f:
                s = f.read(limit_chars - n) if limit_chars else f.read()
            parts.append(s)
            n += len(s)
            if limit_chars and n >= limit_chars:
                break
    if any_txt:
        return "".join(parts) if not any(
            p.suffix.lower() in (".parquet", ".jsonl", ".json")
            for p in paths) else sep.join(parts)[:limit_chars] if limit_chars else sep.join(parts)
    return sep.join(parts)[:limit_chars] if limit_chars else sep.join(parts)


# ── P196：Claude Code traces（event-stream JSONL → 对话文本）────────────
# 数据源：ModelScope choucisan/mimo-claude-code-traces-1k（1,017 traces）。
# 每行是一个 event（user / assistant / tool_use / tool_result / 元数据），
# **没有** `text` 字段 → 通用 `_iter_one` 会跳过全部行。故单独一条路径：
# 只抽语言内容、按 event 原顺序还原成对话文本。
_CC_SEP = "\n\n"
_CC_MAX_TOOL_INPUT = 400


def _cc_content_to_text(content, thinking: bool = True) -> str:
    """message.content（str 或 block 列表）→ 可读文本。"""
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
            # reasoning trace 是这份数据的独特价值（展示「如何一步步想」）
            parts.append("[思考] " + b["thinking"].strip())
        elif bt == "tool_use":
            inp = json.dumps(b.get("input", {}), ensure_ascii=False)
            parts.append(f"[调用工具 {b.get('name', '?')}] "
                         + (inp if len(inp) <= _CC_MAX_TOOL_INPUT
                            else inp[:_CC_MAX_TOOL_INPUT] + "\n…[截断]"))
        elif bt == "tool_result":
            c = b.get("content")
            if isinstance(c, list):
                c = "\n".join(str(x.get("text", "")) for x in c
                              if isinstance(x, dict))
            s = str(c or "").strip()
            parts.append("[工具返回] "
                         + (s if len(s) <= _CC_MAX_TOOL_INPUT
                            else s[:_CC_MAX_TOOL_INPUT] + "\n…[截断]"))
    return "\n".join(p for p in parts if p)


def cc_trace_to_text(raw: str, thinking: bool = True) -> str:
    """单个 JSONL event-stream → 一段对话文本。

    只保留 ``type ∈ {user, assistant}`` 的行（跳过 mode / permission-mode /
    last-prompt 等会话元数据）；按原文件顺序输出（event-stream 本身即时序）。
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
        body = _cc_content_to_text(msg.get("content"), thinking=thinking)
        if body:
            lines.append(f"{role}: {body}")
    return _CC_SEP.join(lines)


def iter_cc_traces(path, thinking: bool = True):
    """流式迭代 Claude Code traces → 每条 trace 一段对话文本。

    ``path`` 支持单个文件 / glob / **已展开的文件列表**（与 ``iter_texts``
    同口径）。三种来源均可：
      · 远程 ``ms://.../*.jsonl``（HTTP Range 零落盘）；
      · 本地 ``*.jsonl``（原始 event-stream）；
      · 本地 ``*.parquet``（P196 转换产物：列 ``text`` 已是转换后的文本，
        直接 yield，**不再走 JSONL 解析**）。

    ⚠ 本地 parquet 分支是必需项：训练一旦用了 `--data cc_traces` 的本地
    副本模式（`datasets/cc_traces/cc_000.parquet`），`DATA_FILES` 的 glob
     就是 parquet，而 parquet 是**二进制**——按 utf-8 文本读会
     `UnicodeDecodeError`（P198 实测）。
    """
    for p in expand_paths(path):
        suf = p.suffix.lower()
        if suf == ".parquet":
            # 已转换的文本列（`tools/data/cc_traces_to_text.py` 的产物）
            pf = open_parquet_source(p)
            names = pf.schema_arrow.names
            col = "text" if "text" in names else 0
            for batch in pf.iter_batches(batch_size=256, columns=[col] if col == "text" else None):
                for v in batch.column(0 if col == 0 else col).to_pylist():
                    if v and v.strip():
                        yield v
            continue
        if is_remote_path(p):
            from .ms_stream import open_ms_file
            with open_ms_file(str(p)) as fh:
                raw = fh.read()
            raw = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        else:
            with open(p, encoding="utf-8") as f:
                raw = f.read()
        txt = cc_trace_to_text(raw, thinking=thinking)
        if txt.strip():
            yield txt


def load_cc_traces(path, limit_chars: int | None = None,
                   thinking: bool = True) -> str:
    """把 traces 拼成一个字符串（供 WordTokenizer / PHDWordLM 消费）。"""
    parts: list[str] = []
    n = 0
    for t in iter_cc_traces(path, thinking=thinking):
        parts.append(t)
        n += len(t) + len(_CC_SEP)
        if limit_chars and n >= limit_chars:
            return _CC_SEP.join(parts)[:limit_chars]
    return _CC_SEP.join(parts)[:limit_chars] if limit_chars else _CC_SEP.join(parts)


def write_parquet(texts, dst: str | Path, lang: str | None = None,
                  src: str | None = None, batch_size: int = 50_000,
                  compression: str = "zstd") -> int:
    import pyarrow as pa

    pq = _require_pyarrow()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    buf: list[str] = []
    writer = None
    schema = pa.schema([("text", pa.string())]
                       + ([("lang", pa.string())] if lang else [])
                       + ([("src", pa.string())] if src else []))

    def flush(buf: list[str]):
        cols = {"text": pa.array(buf, pa.string())}
        if lang:
            cols["lang"] = pa.array([lang] * len(buf), pa.string())
        if src:
            cols["src"] = pa.array([src] * len(buf), pa.string())
        return pa.Table.from_pydict(cols, schema=schema)

    for t in texts:
        buf.append(t)
        if len(buf) >= batch_size:
            tbl = flush(buf)
            if writer is None:
                writer = pq.ParquetWriter(str(dst), tbl.schema,
                                          compression=compression)
            writer.write_table(tbl)
            n += len(buf)
            buf = []
    if buf:
        tbl = flush(buf)
        if writer is None:
            writer = pq.ParquetWriter(str(dst), tbl.schema, compression=compression)
        writer.write_table(tbl)
        n += len(buf)
    if writer is not None:
        writer.close()
    return n


def corpus_stats(path: str | Path, max_items: int = 50_000) -> dict:
    """轻量统计：条数、字符数、平均长度、中文占比。"""
    if is_remote_path(path):                       # 审计 B4：Path(ms://) 会
        from phdnet.ms_stream import ms_total_size  # 摧毁远程语义并误报 glob 无匹配
        size_mb: float | None = ms_total_size(expand_paths(path)) / 1e6
        p = str(path)
    else:
        p = Path(path)
        size_mb = None
    n = chars = cjk = 0
    lens = []
    for t in iter_texts(p):
        n += 1
        chars += len(t)
        lens.append(len(t))
        cjk += sum(1 for c in t[:2000] if "\u4e00" <= c <= "\u9fff")
        if n >= max_items:
            break
    return {"path": str(p), "items": n, "chars": chars,
            "mean_len": (sum(lens) / n) if n else 0.0,
            "cjk_ratio": (cjk / max(1, min(chars, 2000 * n))) if n else 0.0,
            "size_mb": (os.path.getsize(p) / 1e6) if size_mb is None else size_mb}
