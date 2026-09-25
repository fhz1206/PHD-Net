"""把 datasets/{pretrain,sft} 下的语料**合并成 ~1 GB 一个的 parquet**。

fhz 2026-09-24 指令：「pretrain 合并成几个 parquet（每个 1 GB），sft 也是」。

要点
----
1. **去重复源**：同一语料同时有 `.txt` 与 `.parquet` 时只取 parquet（内容相同，
   parquet 是压缩后的列式副本），避免训练时同一批数据被学两遍。
2. **滚动切分**：按**输出文件实际大小**滚动（非按条数），每达到 `--target-mb`
   （默认 1024）就关掉当前 writer 开新文件：`pretrain_000.parquet / _001 / …`。
3. **统一 schema**：`text / lang / src`，`src` 记录来自哪个语料文件，便于回溯与
   后续按比例采样（例如只取中文子集）。
4. **流式**：源侧逐块/逐行读，不把 10 GB 语料整体载入内存。

用法：
  python tools/merge_corpus.py --split sft
  python tools/merge_corpus.py --split pretrain --target-mb 1024
  python tools/merge_corpus.py --split pretrain --stats      # 只看会并入哪些源
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import pyarrow as pa                                   # noqa: E402
import pyarrow.parquet as pq                           # noqa: E402

from phdnet.corpus import iter_texts                   # noqa: E402

SCHEMA = pa.schema([("text", pa.string()),
                    ("lang", pa.string()),
                    ("src", pa.string())])


def _lang_of(t: str) -> str:
    cjk = sum(1 for c in t[:1000] if "\u4e00" <= c <= "\u9fff")
    lat = sum(1 for c in t[:1000] if c.isascii() and c.isalpha())
    if cjk > 100 and cjk >= lat:
        return "zh"
    if lat > 100:
        return "en"
    return "unk"


def pick_sources(split_dir: Path) -> list[Path]:
    """每个语料名只取一份：parquet 优先于 txt。"""
    out = []
    seen: set[str] = set()
    for p in sorted(split_dir.glob("*")):
        if p.is_dir():
            continue
        stem = p.stem
        if stem in seen:
            continue
        pq_v = split_dir / f"{stem}.parquet"
        chosen = pq_v if pq_v.exists() else p
        if chosen.suffix.lower() in (".parquet", ".txt", ".jsonl"):
            seen.add(stem)
            out.append(chosen)
    return out


def merge(split: str, target_mb: int = 1024, out_dir: Path | None = None,
          stats_only: bool = False) -> list[dict]:
    src_dir = _ROOT / "datasets" / split
    out_dir = out_dir or (src_dir / "merged")
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = pick_sources(src_dir)
    if stats_only:
        for s in sources:
            print(f"  {s.stat().st_size/1e6:>9.1f} MB  {s.name}")
        return []

    target = target_mb * 1_000_000
    files = []
    idx = 0
    writer = None
    cur = out_dir / f"{split}_{idx:03d}.parquet"
    buf: list[tuple[str, str, str]] = []
    n_total = 0

    def flush() -> pa.Table | None:
        if not buf:
            return None
        return pa.Table.from_pydict(
            {"text": pa.array([b[0] for b in buf], pa.string()),
             "lang": pa.array([b[1] for b in buf], pa.string()),
             "src": pa.array([b[2] for b in buf], pa.string())},
            schema=SCHEMA)

    for s in sources:
        print(f"[源] {s.name}（{s.stat().st_size/1e6:.1f} MB）", flush=True)
        n_src = 0
        for t in iter_texts(s):
            if not t or not t.strip():
                continue
            buf.append((t.strip(), _lang_of(t), s.stem))
            n_src += 1
            n_total += 1
            if len(buf) >= 20_000:
                tbl = flush()
                if writer is None:
                    writer = pq.ParquetWriter(str(cur), SCHEMA, compression="zstd")
                writer.write_table(tbl)
                buf = []
                if cur.exists() and cur.stat().st_size >= target:
                    writer.close()
                    files.append({"path": cur, "mb": cur.stat().st_size / 1e6})
                    print(f"  → {cur.name} {cur.stat().st_size/1e6:.1f} MB "
                          f"（累计 {n_total:,} 条）", flush=True)
                    idx += 1
                    cur = out_dir / f"{split}_{idx:03d}.parquet"
                    writer = None
        print(f"      {s.stem}: {n_src:,} 条", flush=True)

    if buf:
        tbl = flush()
        if writer is None:
            writer = pq.ParquetWriter(str(cur), SCHEMA, compression="zstd")
        writer.write_table(tbl)
    if writer is not None:
        writer.close()
    if cur.exists():
        files.append({"path": cur, "mb": cur.stat().st_size / 1e6})
        print(f"  → {cur.name} {cur.stat().st_size/1e6:.1f} MB", flush=True)
    print(f"[完成] {split}: {len(files)} 个文件 / {n_total:,} 条")
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["pretrain", "sft"], required=True)
    ap.add_argument("--target-mb", type=int, default=1024)
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    merge(a.split, a.target_mb, stats_only=a.stats)


if __name__ == "__main__":
    main()
