"""把大 parquet 按**输出文件大小**切成多个 ≤target 的小 parquet。

用途：atomgit/gitcode 的 pre-receive 钩子拒绝 >100 MiB 单文件，而 LFS 配额已满。
与 gzip 分卷（tools/pack_corpus.py）不同，本工具的每个分片**本身就是合法的
parquet**（保留 schema，可直接 pyarrow 读取、iter_batches、load_text），无还原步骤。

用法：
  python tools/split_parquet.py datasets/pretrain/merged/pretrain_000.parquet
  python tools/split_parquet.py x.parquet --target-mb 90 --out-dir out/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def split(src: Path, target_mb: float = 90.0, out_dir: Path | None = None,
          batch_rows: int = 20_000) -> list[Path]:
    pf = pq.ParquetFile(str(src))
    schema = pf.schema_arrow
    out_dir = out_dir or src.parent / "split"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    target = target_mb * 1_048_576

    parts: list[Path] = []
    idx = 0
    writer = None
    cur: Path | None = None
    rows_this = 0
    rows_total = 0

    def close():
        nonlocal writer, cur, rows_this
        if writer is not None:
            writer.close()
            parts.append(cur)
            print(f"  {cur.name}  {cur.stat().st_size/1e6:.1f} MB / {rows_this:,} 行", flush=True)
        writer, cur, rows_this = None, None, 0

    for batch in pf.iter_batches(batch_size=batch_rows):
        if writer is None:
            cur = out_dir / f"{stem}.{idx:03d}.parquet"
            idx += 1
            writer = pq.ParquetWriter(str(cur), schema, compression="zstd")
        writer.write_batch(batch)
        rows_this += batch.num_rows
        rows_total += batch.num_rows
        if cur.exists() and cur.stat().st_size >= target:
            close()
    close()
    print(f"[完成] {src.name}（{rows_total:,} 行）→ {len(parts)} 片", flush=True)
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--target-mb", type=float, default=90.0)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    split(a.src, a.target_mb, a.out_dir)


if __name__ == "__main__":
    main()
