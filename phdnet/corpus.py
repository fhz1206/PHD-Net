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
`tests/eval_common.py` 等仍读 `eval_corpus/internal_corpus.txt`（纯文本），
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
           "open_parquet_source"]

_SEP = "\n\n"


def is_remote_path(path) -> bool:
    """是否 ModelScope 远程 spec（ms://…；phdnet.ms_stream.is_remote_path 转发）。"""
    from phdnet.ms_stream import is_remote
    return is_remote(path)


def open_parquet_source(p):
    """Path 或 MsFile → pyarrow ParquetFile。

    本地：``ParquetFile(str(p))``（与旧行为**完全一致**，逐位不变）；
    远程：``ms://`` spec → HTTP Range 可 seek 流（phdnet/ms_stream.py）。
    """
    pq = _require_pyarrow()
    if is_remote_path(p):
        from phdnet.ms_stream import open_ms_file
        return pq.ParquetFile(open_ms_file(str(p)))
    return pq.ParquetFile(str(p))


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


def expand_paths(path: str | Path) -> list[Path]:
    """路径展开：支持 glob 通配符（如 `pretrain_*.parquet`），返回确定性排序的文件列表。

    远程（fhz 2026-09-29）：`ms://<ns>/<path>` 走 ModelScope 直连
    （phdnet/ms_stream.py，HTTP Range 流式零落盘）；本地逻辑一字未动。
    """
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
    for p in expand_paths(path):
        if p.suffix.lower() in (".parquet", ".jsonl", ".json"):
            for t in _iter_one(p, "text", 2000):
                parts.append(t)
                n += len(t) + len(sep)
                if limit_chars and n >= limit_chars:
                    return (sep.join(parts)[:limit_chars])
        else:
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
            for p in expand_paths(path)) else sep.join(parts)[:limit_chars] if limit_chars else sep.join(parts)
    return sep.join(parts)[:limit_chars] if limit_chars else sep.join(parts)


def write_parquet(texts, dst: str | Path, lang: str | None = None,
                  src: str | None = None, batch_size: int = 50_000,
                  compression: str = "zstd") -> int:
    """把文本序列写成 parquet（列 `text`，可选 `lang` / `src`）。"""
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
    p = Path(path)
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
            "size_mb": os.path.getsize(p) / 1e6}
