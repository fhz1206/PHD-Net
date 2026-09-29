#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ModelScope 三大预训练源统一流式采样器（P49）。

fhz 指令：把三个数据源统一接进预训练——
  ① OpenBMB/Ultra-FineWeb-L3     （通用网页，中英 × qa/multi_style）
  ② OpenBMB/UltraData-Code-L3    （代码，11 个编程语言子集）
  ③ OpenBMB/UltraData-Math-L3    （数学，4 个合成子集）

与 `fetch_l3.py` 同源，但**泛化**为多源：每个源用一张配置表描述
（仓库 / 子集枚举 / 文本字段 / 语言标签 / 质量闸），输出统一 schema
`text/lang/src` 的 parquet 分片，直接落进 `datasets/pretrain/`，
训练侧 `--data pretrain` 无需任何新参数即可消费。

规模（实测，P48/P49 探查）
    L3        430.6 GB / 2.02 亿篇
    Code-L3   ~11 语言 × ~30 GB（cpp 28 片 29.7 GB，cs 28 片 29.7 GB …）
    Math-L3   4 子集 × ~30 GB（Conversation 50 片 30.8 GB …）
    合计 ≈ 880 GB —— 50 GB 云端预算下只能采样（建议 3~5%）

零原始落盘：fsspec HTTP Range 随机读直喂 pyarrow，1 GB 原始片从不落盘。

用法
----
    python tools/fetch_ms.py --dry-run                     # 探查三源规模
    python tools/fetch_ms.py --plan web=3,code=2,math=1   # 按权重采样（百万篇）
    python tools/fetch_ms.py --plan web=1 --resume         # 增量续跑
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))

API_TREE = ("https://www.modelscope.cn/api/v1/datasets/{ns}/repo/tree"
            "?Revision=master&Root={path}")
API_FILE = ("https://www.modelscope.cn/api/v1/datasets/{ns}/repo"
            "?Revision=master&FilePath={path}")

SCHEMA = pa.schema([("text", pa.string()),
                    ("lang", pa.string()),
                    ("src", pa.string())])

# ── 源配置表 ────────────────────────────────────────────────────────────
# text_field : 取哪一列作为训练文本（三个源都已实测）
# subsets    : None = 自动递归枚举两层子目录（深度到 parquet）
# lang       : 写进 parquet 的 lang 标签
SOURCES = {
    "web": {
        "ns": "OpenBMB/Ultra-FineWeb-L3",
        "root": "data",
        "match": "ultrafineweb_",
        "text_field": "content",
        "lang": None,                # 从子目录名解析（ultrafineweb_zh_l3 → zh）
        "lang_from_subdir": ("ultrafineweb_", "_l3"),
    },
    "code": {
        "ns": "OpenBMB/UltraData-Code",
        "root": "data/UltraData-Code-L3",
        "match": None,                # 11 个语言目录直接用
        "text_field": "content",      # 实测：uuid/content/.../full_content
        "lang": "code",
    },
    "math": {
        "ns": "OpenBMB/UltraData-Math",
        "root": "data/UltraData-Math-L3",
        "match": None,
        "text_field": "content",      # 实测：uid/content
        "lang": "math",
    },
}


# ───────────────────────── ModelScope 直连 ─────────────────────────

def list_files(ns: str, path: str) -> list[dict]:
    url = API_TREE.format(ns=ns, path=path)
    with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as r:
        return json.loads(r.read())["Data"]["Files"]


def walk_parquets(ns: str, root: str, match: str | None, depth: int = 2):
    """递归枚举 parquet 分片（限深，避免把 400 个目录全展开）。"""
    out = []
    stack = [(root, 0)]
    while stack:
        p, d = stack.pop()
        try:
            items = list_files(ns, p)
        except Exception as e:                            # noqa: BLE001
            print(f"    !! 列目录失败 {p}: {e}", flush=True)
            continue
        for f in items:
            if f.get("Type") == "blob" and f["Name"].endswith(".parquet"):
                if match is None or match in f["Path"]:
                    out.append(f)
            elif f.get("Type") == "tree" and d < depth:
                if match is None or match in f["Name"] or match in f["Path"]:
                    stack.append((f["Path"], d + 1))
    return out


def open_stream(ns: str, path: str, block_mb: int = 4):
    import fsspec
    url = API_FILE.format(ns=ns, path=path)
    fs = fsspec.filesystem("http", client_kwargs={"trust_env": True})
    return fs.open(url, "rb", block_size=block_mb << 20)


# ───────────────────────── 质量闸 ─────────────────────────

def cjk_ratio(s: str) -> float:
    return sum(1 for ch in s if "一" <= ch <= "鿿") / max(1, len(s))


def latin_ratio(s: str) -> float:
    lat = sum(1 for ch in s if ("a" <= ch <= "z") or ("A" <= ch <= "Z"))
    return lat / max(1, len(s))


def acceptable(text: str, lang: str, min_chars: int, max_chars: int) -> bool:
    t = (text or "").strip()
    if not (min_chars <= len(t) <= max_chars):
        return False
    if lang == "zh":
        return cjk_ratio(t) >= 0.15
    if lang == "en":
        return latin_ratio(t) >= 0.15
    return True                       # code / math：不按语言卡（代码里中文注释常见）


# ───────────────────────── 采样 + 写分片 ─────────────────────────

def sample_source(name: str, cfg: dict, quota: int, min_chars: int,
                  max_chars: int, batch_rows: int, max_shards: int):
    ns = cfg["ns"]
    files = walk_parquets(ns, cfg["root"], cfg.get("match"))
    if not files:
        print(f"  [{name}] 未找到 parquet 分片", flush=True)
        return
    step = max(1, len(files) // max(1, min(len(files), max_shards)))
    picked = files[::step][:max_shards]
    total_bytes = sum((f.get("Size") or 0) for f in files)
    print(f"  [{name}] {len(files)} 片 / {total_bytes / 1e9:.1f} GB；"
          f"本次取 {len(picked)} 片配额 {quota:,} 篇", flush=True)
    field = cfg["text_field"]
    got = 0
    t0 = time.perf_counter()
    for f in picked:
        if got >= quota:
            break
        try:
            fh = open_stream(ns, f["Path"])
            pf = pq.ParquetFile(fh)
            if field not in pf.schema_arrow.names:
                print(f"    !! {f['Path']} 无字段 {field}"
                      f"（实际 {pf.schema_arrow.names}）", flush=True)
                fh.close()
                continue
            for batch in pf.iter_batches(batch_size=batch_rows, columns=[field]):
                if got >= quota:
                    break
                for v in batch.column(field):
                    txt = v.as_py()
                    if acceptable(txt, cfg["lang"] or "en", min_chars, max_chars):
                        yield (txt, cfg["lang"] or "en",
                               f"{ns.split('/')[-1]}/{f['Path'].split('/')[-2]}")
                        got += 1
                        if got >= quota:
                            break
            fh.close()
        except Exception as e:                          # noqa: BLE001
            print(f"    !! {f['Path']} 失败: {type(e).__name__}: {e}", flush=True)
    print(f"  [{name}] 完成 {got:,} 篇（{time.perf_counter() - t0:.0f}s）", flush=True)


def write_shards(out_dir: Path, prefix: str, rows_iter, shard_mib: int,
                 resume: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = int(shard_mib * 2 ** 20)
    buf: list = []
    nbytes = 0
    idx = 0
    shards = 0
    rows = 0
    skipped = 0

    def _flush(buf):
        nonlocal idx, shards, rows, skipped
        if not buf:
            return
        if resume:
            while (out_dir / f"{prefix}_{idx:05d}.parquet").exists():
                idx += 1
        path = out_dir / f"{prefix}_{idx:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(
            [{"text": t, "lang": lg, "src": s} for t, lg, s in buf],
            schema=SCHEMA), path, compression="snappy")
        idx += 1
        shards += 1
        rows += len(buf)

    for row in rows_iter:
        buf.append(row)
        nbytes += len(row[0]) * 3
        if nbytes >= limit:
            _flush(buf)
            buf = []
            nbytes = 0
    _flush(buf)
    return {"shards": shards, "rows": rows, "skipped": skipped}


# ───────────────────────── main ─────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="ModelScope 三源统一流式采样")
    ap.add_argument("--out", type=Path, default=_ROOT / "datasets" / "pretrain")
    ap.add_argument("--plan", default="",
                    help="各源配额（百万篇），如 web=3,code=2,math=1；"
                         "留空则只做 --dry-run")
    ap.add_argument("--shard-mib", type=int, default=100)
    ap.add_argument("--batch-rows", type=int, default=2048)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=20000)
    ap.add_argument("--max-shards", type=int, default=24,
                    help="每源最多取多少片（片级等距铺开）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("三源规模探查（ModelScope 直连，未下载任何分片）")
    print("=" * 74)
    grand = 0
    for name, cfg in SOURCES.items():
        try:
            files = walk_parquets(cfg["ns"], cfg["root"], cfg.get("match"))
            b = sum((f.get("Size") or 0) for f in files)
            grand += b
            print(f"  {name:5s} {cfg['ns']:28s} {len(files):4d} 片 "
                  f"{b / 1e9:8.1f} GB")
        except Exception as e:                          # noqa: BLE001
            print(f"  {name:5s} 探查失败: {type(e).__name__}: {e}")
    print(f"  {'合计':5s} {'':28s}       {grand / 1e9:8.1f} GB")
    if args.dry_run or not args.plan:
        print("\n(--dry-run 或未给 --plan：到此为止)")
        return 0

    plan: dict = {}
    for item in args.plan.split(","):
        k, _, v = item.partition("=")
        if k.strip() and v.strip():
            plan[k.strip()] = int(float(v.strip()) * 1_000_000)

    print("=" * 74)
    print(f"采样计划：{plan} → {args.out}")
    print("=" * 74)
    t0 = time.perf_counter()
    tot = {"shards": 0, "rows": 0, "skipped": 0}
    for name, quota in plan.items():
        cfg = SOURCES.get(name)
        if cfg is None:
            print(f"  未知源 {name}（可用：{list(SOURCES)}）")
            continue
        rows = sample_source(name, cfg, quota, args.min_chars,
                            args.max_chars, args.batch_rows, args.max_shards)
        st = write_shards(args.out, f"ms_{name}", rows, args.shard_mib, args.resume)
        for k in tot:
            tot[k] += st[k]
        print(f"  → {name}: +{st['rows']:,} 篇 / {st['shards']} 片", flush=True)
    print("=" * 74)
    print(f"完成：{tot['rows']:,} 篇 / {tot['shards']} 片，"
          f"{time.perf_counter() - t0:.0f}s")
    print("训练接入：分片已在 datasets/pretrain/ → --data pretrain 直接消费")
    print("词表：--vocab-file 复用旧快照，或 --vocab-scan head/full 在新语料上重建")
    return 0


if __name__ == "__main__":
    sys.exit(main())
