"""Ultra-FineWeb-L3（OpenBMB）中文子集 → 采样 → parquet（datasets/pretrain/）。

================================================================================
为什么必须采样
================================================================================
全量 1,764 个 parquet × ~1.08 GB ≈ **1.9 TB**（en 400B+ / zh 200B+ tokens）；
仅中文两个子集也有 596 个分片 ≈ 640 GB。本机磁盘与带宽都装不下，故：
  · 每个中文子集只**下载 1 个分片**（tools/fetch_ultrafineweb.py，下完即删）；
  · 分片内按**确定性等间隔**抽样（不是只取前 N 行——那样会拿到同一站点/主题
    的连续内容，分布偏斜），每 `--step` 行取 1 行，上限 `--max-rows` 条。

输出 parquet（列 `text` / `lang=zh` / `src`），供 `phdnet.corpus.load_text` 直接读。

用法：
  python tools/convert_ultrafineweb.py --stats              # 只看分片规模
  python tools/convert_ultrafineweb.py --max-rows 40000      # 每子集采样 4 万行
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from phdnet.corpus import write_parquet        # noqa: E402

SRC = _ROOT / ".tmp_ufw_data"
DST = _ROOT / "datasets" / "pretrain" / "raw" / "ultrafineweb_l3_zh.parquet"
COL = "content"


def iter_sampled(files: list[Path], step: int, max_rows: int):
    """确定性等间隔抽样：跨文件连续计数，每 step 行取 1 行。"""
    n = kept = 0
    for f in files:
        pf = pq.ParquetFile(str(f))
        for batch in pf.iter_batches(batch_size=4096, columns=[COL]):
            col = batch.column(COL)
            for i in range(batch.num_rows):
                n += 1
                if n % step:
                    continue
                v = col[i].as_py()
                if v and len(v.strip()) >= 200:
                    yield v.strip()
                    kept += 1
                    if kept >= max_rows:
                        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--max-rows", type=int, default=40_000)
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    files = sorted(a.src.glob("*/*.parquet")) or sorted(a.src.glob("*.parquet"))
    if not files:
        raise SystemExit(f"未找到分片：{a.src}（先跑 tools/fetch_ultrafineweb.py）")
    total = 0
    for f in files:
        pf = pq.ParquetFile(str(f))
        n = pf.metadata.num_rows
        total += n
        print(f"  {f.parent.name:<16}{n:>10,} 行  {f.name[:44]}")
    print(f"合计 {total:,} 行")
    if a.stats:
        return
    print(f"采样：每 {a.step} 行取 1，每子集上限 {a.max_rows:,} 条")
    n = write_parquet(iter_sampled(files, a.step, a.max_rows * len(files)),
                      a.dst, lang="zh", src="ultrafineweb_l3_zh")
    mb = os.path.getsize(a.dst) / 1e6
    print("-" * 66)
    print(f"输出 {a.dst}  {n:,} 条 / {mb:.1f} MB")


if __name__ == "__main__":
    main()
