"""Infinity-Instruct 7M_core（M7_Core）parquet → PHD-Net 预训练纯文本转换。

数据源：atomgit BAAI/Infinity-Instruct，config=7M_core（1.4M 指令核心子集，
官方口径达到全量 7M 指令 95.7% 性能）。落地于 datasets/pretrain/（Request E 拍板）。

设计（RAM 12.6 GB 约束）：
  - 流式：逐分片 ParquetFile.iter_batches，batch=2000 行，绝不全量载入；
  - 拼接：每条样本各轮对话 value 按序拼接（"\n" 连接轮次），样本间 "\n\n"；
  - 抽样：--sample N 等步长抽样（用于小规模实验）；--stats 只统计不写文件。

用法：
  python tools/convert_infinity.py --stats            # 仅统计（单分片外推）
  python tools/convert_infinity.py --sample 50000     # 抽 5 万条
  python tools/convert_infinity.py                    # 全量 1.4M 条
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SRC = _ROOT / ".tmp_ii_clone" / "7M_core"
_DEFAULT_DST = _ROOT / "datasets" / "pretrain" / "infinity_m7core.txt"

CONV_FIELD = "conversations"      # Infinity-Instruct 对话字段（list of {from, value}）
ROLE_SEP = "\n"                   # 轮次间分隔
SAMPLE_SEP = "\n\n"               # 样本间分隔


def iter_samples(src: Path, stride: int = 1):
    """流式产出每条样本的拼接文本。stride=2 表示每 2 条取 1。"""
    n_seen = 0
    shards = sorted(src.glob("*.parquet"))
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=2000, columns=[CONV_FIELD]):
            col = batch.column(0)
            for item in col:
                if n_seen % stride != 0:
                    n_seen += 1
                    continue
                n_seen += 1
                turns = item.as_py() or []
                parts = [t.get("value", "") for t in turns
                         if isinstance(t, dict) and t.get("value")]
                if parts:
                    yield ROLE_SEP.join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=_DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=_DEFAULT_DST)
    ap.add_argument("--sample", type=int, default=0, help="等步长抽样条数（0=全量）")
    ap.add_argument("--stats", action="store_true", help="只统计单分片并外推，不写文件")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"源目录不存在: {args.src}")
        sys.exit(1)

    if args.stats:
        # 只用已落地的真实分片（跳过尚未 lfs pull 的 ~133B 指针文件）
        real = [s for s in sorted(args.src.glob("*.parquet"))
                if s.stat().st_size > 1000]
        if not real:
            print("无已落地分片（全部仍为 LFS 指针），请等待 git lfs pull 完成")
            sys.exit(1)
        shard = real[0]
        pf = pq.ParquetFile(shard)
        n_rows = pf.metadata.num_rows
        n_chars = n_samples = 0
        for batch in pf.iter_batches(batch_size=2000, columns=[CONV_FIELD]):
            for item in batch.column(0):
                turns = item.as_py() or []
                text = ROLE_SEP.join(t.get("value", "") for t in turns
                                     if isinstance(t, dict) and t.get("value"))
                if text:
                    n_samples += 1
                    n_chars += len(text)
        est = n_chars * 75  # 75 分片
        print(f"单分片 {shard.name}: {n_rows:,} 行, 有效样本 {n_samples:,}, "
              f"{n_chars:,} 字符")
        print(f"全量外推（×75 分片）: ~{n_rows * 75:,} 行, "
              f"~{est:,} 字符（{est / 1e9:.1f} GB 文本）")
        return

    stride = 0
    if args.sample > 0:
        # 先知道总行数才能定步长：用 75 分片的 metadata 求和（便宜）
        total = sum(pq.ParquetFile(s).metadata.num_rows
                    for s in sorted(args.src.glob("*.parquet")))
        stride = max(1, total // args.sample)
        print(f"总行数 {total:,}，抽样目标 {args.sample:,} → 步长 {stride}")
    else:
        stride = 1

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    n = n_chars = 0
    with open(args.dst, "w", encoding="utf-8") as f:
        for text in iter_samples(args.src, stride):
            f.write(text + SAMPLE_SEP)
            n += 1
            n_chars += len(text) + len(SAMPLE_SEP)
            if n % 50_000 == 0:
                print(f"  已写 {n:,} 条 / {n_chars / 1e9:.2f} GB", flush=True)
    print(f"完成: {args.dst}  样本 {n:,} 条, {n_chars / 1e9:.2f} GB 文本")


if __name__ == "__main__":
    main()
