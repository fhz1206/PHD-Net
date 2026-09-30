#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ultra-FineWeb-L3 → 本项目 parquet 分片（流式，零原始落盘）。

fhz 指令：L3 作为训练数据源，攻克「数据集导向」。云端只有 50 GB 存储，而 L3 全量
**430.6 GB / 2.02 亿篇**，因此必须**按配额采样**而非全量搬运。

设计要点
--------
1. **零原始落盘**：用 fsspec 的 HTTP Range 随机读直接喂给
   `pyarrow.parquet.ParquetFile`——只拉需要的字节区间，**不下载原始分片到磁盘**
   （原始 1,075 MB/片 × N 会瞬间撑爆磁盘）。
2. **分层采样**：语言（zh/en）× 风格（qa/multi_style）四个子集各自按配额均匀取，
   避免只吃到 qa 或只吃到某一语言。
3. **恒定内存**：按 batch 迭代（默认 2048 行），不整片载入。
4. **断点续跑**：已存在的分片跳过（`--resume`），中断后重跑不重复下载。
5. **质量闸**：长度区间（去极短/超长）、非空、CJK/拉丁字符占比合理性。

分片输出遵循本项目规范（`datasets/README.md`）：snappy parquet、列 `text/lang/src`、
单片 ≤ `--shard-mib`（默认 100 MiB，符合 gitcode 单文件限制）。

用法
----
    # 探查（不落盘，只打印各子集规模）
    python tools/fetch_l3.py --dry-run

    # 采样：中文 300 万篇 + 英文 200 万篇 → datasets/pretrain/l3_*.parquet
    python tools/fetch_l3.py --docs-per-lang 3,000,000,2,000,000 --out datasets/pretrain

    # 增量：再加 100 万篇（跳过已有分片）
    python tools/fetch_l3.py --docs-per-lang 1,000,000,1,000,000 --out datasets/pretrain --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))

NS_NAME = "OpenBMB/Ultra-FineWeb-L3"
API_TREE = ("https://www.modelscope.cn/api/v1/datasets/{ns}/repo/tree"
            "?Revision=master&Root={path}")
API_FILE = ("https://www.modelscope.cn/api/v1/datasets/{ns}/repo"
            "?Revision=master&FilePath={path}")
SUBSETS = [("zh", "qa"), ("zh", "multi_style"),
           ("en", "qa"), ("en", "multi_style")]


# ───────────────────────── ModelScope 直连 ─────────────────────────

def list_files(subset_path: str) -> list[dict]:
    url = API_TREE.format(ns=NS_NAME, path=subset_path)
    with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as r:
        files = json.loads(r.read())["Data"]["Files"]
    return [f for f in files if f.get("Type") == "blob"]


def open_stream(path: str, block_mb: int = 4):
    """可 seek 的 HTTP 文件流（Range 随机读）→ 供 pyarrow 直接消费。"""
    import fsspec
    url = API_FILE.format(ns=NS_NAME, path=path)
    fs = fsspec.filesystem("http", client_kwargs={"trust_env": True})
    return fs.open(url, "rb", block_size=block_mb << 20)


# ───────────────────────── 质量闸 ─────────────────────────

def cjk_ratio(s: str) -> float:
    if not s:
        return 0.0
    cjk = sum(1 for ch in s if "一" <= ch <= "鿿")
    return cjk / max(1, len(s))


def latin_ratio(s: str) -> float:
    if not s:
        return 0.0
    lat = sum(1 for ch in s if ("a" <= ch <= "z") or ("A" <= ch <= "Z"))
    return lat / max(1, len(s))


def acceptable(text: str, lang: str, min_chars: int, max_chars: int) -> bool:
    t = (text or "").strip()
    if not (min_chars <= len(t) <= max_chars):
        return False
    r = cjk_ratio(t) if lang == "zh" else latin_ratio(t)
    return r >= 0.15                     # 该语言的主导字符占比 ≥ 15%


# ───────────────────────── 采样 + 写分片 ─────────────────────────

SCHEMA = pa.schema([("text", pa.string()),
                    ("lang", pa.string()),
                    ("src", pa.string())])


def write_shards(out_dir: Path, prefix: str, rows_iter, shard_mib: int,
                 resume: bool) -> dict:
    """把行迭代器写成 ≤ shard_mib 的 parquet 分片。返回统计。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = int(shard_mib * 2 ** 20)
    buf: list = []
    nbytes = 0
    shards = 0
    written_rows = 0
    skipped = 0
    for text, lang, src in rows_iter:
        buf.append((text, lang, src))
        nbytes += len(text) * 3          # 约 3 B/字符（UTF-8 上界估计）
        if nbytes >= limit:
            idx = shards + (1 if not resume else 0)
            # 续跑时按现有文件数往后编号
            path = out_dir / f"{prefix}_{idx:05d}.parquet"
            if path.exists():
                skipped += 1
            else:
                pq.write_table(pa.Table.from_pylist(
                    [{"text": t, "lang": lg, "src": s} for t, lg, s in buf],
                    schema=SCHEMA), path, compression="snappy")
                shards += 1
                written_rows += len(buf)
            buf = []
            nbytes = 0
    if buf:
        idx = shards + (1 if not resume else 0)
        path = out_dir / f"{prefix}_{idx:05d}.parquet"
        if path.exists():
            skipped += 1
        else:
            pq.write_table(pa.Table.from_pylist(
                [{"text": t, "lang": lg, "src": s} for t, lg, s in buf],
                schema=SCHEMA), path, compression="snappy")
            shards += 1
            written_rows += len(buf)
    return {"shards": shards, "rows": written_rows, "skipped": skipped}


def sample_subset(lang: str, style: str, quota: int, min_chars: int,
                  max_chars: int, batch_rows: int, log_every: int):
    """流式采样一个子集，产出 (text, lang, src) 行。"""
    subset = f"data/ultrafineweb_{lang}_l3/{style}"
    files = list_files(subset)
    if not files:
        return
    # 均匀铺开：按索引等距取片（避免只吃前 N 片的主题偏差）
    step = max(1, len(files) // max(1, min(len(files), 24)))
    picked = files[::step][:24]
    src_tag = f"Ultra-FineWeb-L3/{lang}/{style}"
    got = 0
    t0 = time.perf_counter()
    for fi, f in enumerate(picked):
        if got >= quota:
            break
        try:
            fh = open_stream(f["Path"])
            pf = pq.ParquetFile(fh)
            for batch in pf.iter_batches(batch_size=batch_rows,
                                        columns=["content"]):
                if got >= quota:
                    break
                for v in batch.column("content"):
                    txt = v.as_py()
                    if acceptable(txt, lang, min_chars, max_chars):
                        yield (txt, lang, src_tag)
                        got += 1
                        if got % log_every < batch_rows:
                            print(f"    [{lang}/{style}] {got:,}/{quota:,} "
                                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
                        if got >= quota:
                            break
            fh.close()
        except Exception as e:                          # noqa: BLE001
            print(f"    !! {f['Path']} 失败: {type(e).__name__}: {e}",
                  flush=True)
    print(f"  [{lang}/{style}] 完成 {got:,} 篇（{time.perf_counter() - t0:.0f}s）",
          flush=True)


# ───────────────────────── main ─────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="流式采样 Ultra-FineWeb-L3")
    ap.add_argument("--out", type=Path, default=_ROOT / "datasets" / "pretrain",
                    help="输出目录（项目规范：datasets/pretrain/）")
    ap.add_argument("--prefix", default="l3",
                    help="分片文件名前缀（默认 l3 → l3_00000.parquet）")
    ap.add_argument("--docs-per-lang", default="1000,1000",
                    help="中文,英文 各采样多少篇（如 3,000,000,2,000,000）")
    ap.add_argument("--styles", default="qa,multi_style",
                    help="采样的风格子集（逗号分隔）")
    ap.add_argument("--shard-mib", type=int, default=100,
                    help="单分片上限 MiB（默认 100，符合 gitcode 限制）")
    ap.add_argument("--batch-rows", type=int, default=2048)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=20000,
                    help="超长文档截断上限（字符）")
    ap.add_argument("--resume", action="store_true", help="跳过已存在的分片")
    ap.add_argument("--dry-run", action="store_true", help="只探查规模，不落盘")
    args = ap.parse_args()

    quotas = [int(x.replace(",", "")) for x in args.docs_per_lang.split(",")]
    if len(quotas) != 2:
        print("--docs-per-lang 需要 'zh,en' 两个数字", file=sys.stderr)
        return 2
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    # ── 探查规模 ──
    print("=" * 72)
    print(f"Ultra-FineWeb-L3 规模探查（{NS_NAME}）")
    print("=" * 72)
    total_bytes = 0
    for lang, style in SUBSETS:
        if style not in styles:
            continue
        try:
            fs = list_files(f"data/ultrafineweb_{lang}_l3/{style}")
        except Exception as e:                          # noqa: BLE001
            print(f"  {lang}/{style}: 列举失败 {e}")
            continue
        b = sum((f.get("Size") or 0) for f in fs)
        total_bytes += b
        print(f"  {lang:2s}/{style:11s} {len(fs):4d} 片  {b / 1e9:7.1f} GB  "
              f"≈ {len(fs) * 505269 / 1e6:.1f} 百万篇")
    print(f"  {'合计':14s}        {total_bytes / 1e9:7.1f} GB")
    if args.dry_run:
        print("\n(--dry-run：到此为止，未写任何文件)")
        return 0

    # ── 采样 ──
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f"采样：zh={quotas[0]:,} 篇 / en={quotas[1]:,} 篇 → {out_dir}")
    print("=" * 72)
    t0 = time.perf_counter()
    grand = {"shards": 0, "rows": 0, "skipped": 0}
    for (lang, style), quota in zip(SUBSETS, quotas):
        if style not in styles or quota <= 0:
            continue
        per_style = quota // max(1, len(styles))
        rows = sample_subset(lang, style, per_style, args.min_chars,
                            args.max_chars, args.batch_rows, 100000)
        st = write_shards(out_dir, f"{args.prefix}_{lang}_{style[:4]}", rows,
                          args.shard_mib, args.resume)
        for k in grand:
            grand[k] += st[k]
        print(f"  → {lang}/{style}: +{st['rows']:,} 篇, {st['shards']} 片"
              f"{f'（跳过 {st[chr(39)+chr(39)] if False else st['skipped']} 片已存在）' if st['skipped'] else ''}",
              flush=True)
    print("=" * 72)
    print(f"完成：{grand['rows']:,} 篇 / {grand['shards']} 片，"
          f"用时 {time.perf_counter() - t0:.0f}s")
    print(f"输出目录：{out_dir}")
    print("下一步：")
    print("  1) 重建词表（流式，~500s 量级按数据量）：")
    print(f"     python train/train.py --data pretrain --vocab-scan full ...")
    print("  2) 或先复用旧快照：--vocab-file outputs/models/vocab_*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
