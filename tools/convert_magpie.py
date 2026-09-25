"""Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B（ModelScope）→ SFT 文本。

数据源：`Magpie-Align/Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B`
150,000 条指令 + DeepSeek-R1（Llama-70B 蒸馏）的长链推理回答（parquet 6 分片
共 1.11 GB，解压 2.62 GB）。字段含 `instruction` / `response` / `task_category` /
`difficulty` / `input_quality` / `language`。

过滤（同 deepctrl 的「名带 sft 不等于都适合做 SFT」原则）：
  · response 过短（<200 字符）—— 长链推理是本数据集的价值所在，短答无意义；
  · response 过长（>4000 字符）—— 截断会破坏推理链，直接丢弃；
  · instruction 重复；
  · `input_quality` 标记为 poor 的样本（数据自带质量标签）。

输出：`instruction\n\nresponse`，样本间空行；上限 `--max-mb` 控磁盘。

用法：
  python tools/convert_magpie.py --stats
  python tools/convert_magpie.py --max-mb 200
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _ROOT / ".tmp_magpie" / "data"
_DEFAULT_DST = _ROOT / "datasets" / "pretrain" / "raw" / "magpie_reasoning_r1.txt"

SEP = "\n\n"
# 实测：150,000 条 **平均 response 7,661 字符**（R1 长链 CoT），全 EN，
# input_quality 仅 good/excellent。故上限放到 16,000（截断会破坏推理链，
# 超限直接丢弃；下限 200 过滤无推理内容的空壳回答）。
MIN_RESP = 200
MAX_RESP = 16000
COLS = ["instruction", "response", "input_quality", "task_category",
        "difficulty", "language"]


def iter_rows(src_dir: Path, limit: int = 0):
    files = sorted(src_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"未找到 parquet：{src_dir}（先跑 tools/_fetch_magpie.py）")
    n = 0
    for f in files:
        pf = pq.ParquetFile(str(f))
        cols = [c for c in COLS if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=2000, columns=cols):
            cd = {c: batch.column(c) for c in cols}
            for i in range(batch.num_rows):
                yield tuple(cd[c][i].as_py() for c in cols)
                n += 1
                if limit and n >= limit:
                    return


def stats(src_dir: Path) -> None:
    cat, qua, lang = Counter(), Counter(), Counter()
    n = ln = 0
    for row in iter_rows(src_dir):
        ins, resp = (row[0] or ""), (row[1] or "")
        n += 1
        ln += len(resp)
        cat[row[3] if len(row) > 3 else None] += 1
        qua[row[2] if len(row) > 2 else None] += 1
        lang[row[5] if len(row) > 5 else None] += 1
    print(f"{n:,} 条  平均 response {ln/max(1,n):.0f} 字符")
    print("task_category:", cat.most_common(8))
    print("input_quality:", qua.most_common(6))
    print("language:", lang.most_common(5))


def convert(src_dir: Path, dst: Path, max_mb: float = 200.0,
            max_samples: int = 0, dry_run: bool = False) -> dict:
    seen: set[int] = set()
    n_read = n_keep = n_short = n_long = n_dup = n_poor = 0
    bytes_out = 0
    limit = int(max_mb * 1_000_000)
    fh = None if dry_run else open(dst, "w", encoding="utf-8", newline="\n")
    try:
        for row in iter_rows(src_dir):
            n_read += 1
            if max_samples and n_keep >= max_samples:
                break
            ins, resp = (row[0] or "").strip(), (row[1] or "").strip()
            if not ins or not resp:
                n_short += 1
                continue
            if len(ins) < 8:
                n_short += 1
                continue
            if len(resp) < MIN_RESP:
                n_short += 1
                continue
            if len(resp) > MAX_RESP:
                n_long += 1
                continue
            q = row[2] if len(row) > 2 else None
            if isinstance(q, str) and q.lower() in ("poor", "bad", "low"):
                n_poor += 1
                continue
            key = hash(ins)
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            s = ins + SEP + resp
            b = len(s.encode("utf-8")) + len(SEP)
            if bytes_out + b > limit:
                break
            if fh is not None:
                fh.write(s if n_keep == 0 else SEP + s)
            bytes_out += b
            n_keep += 1
    finally:
        if fh is not None:
            fh.close()
    return dict(read=n_read, kept=n_keep, mb=bytes_out / 1e6, short=n_short,
                long=n_long, dup=n_dup, poor=n_poor,
                dst=str(dst) if not dry_run else "(dry-run)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=_SRC_DIR)
    ap.add_argument("--dst", type=Path, default=_DEFAULT_DST)
    ap.add_argument("--max-mb", type=float, default=200.0)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.stats:
        stats(a.src)
        return
    a.dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Magpie Reasoning R1 → {a.dst}（上限 {a.max_mb:.0f} MB）")
    r = convert(a.src, a.dst, a.max_mb, a.max_samples, a.dry_run)
    print("-" * 70)
    print(f"读取 {r['read']:,} 条 → 保留 {r['kept']:,} 条（{r['mb']:.1f} MB）")
    print(f"丢弃：过短 {r['short']:,} ／ 过长 {r['long']:,} ／ 重复 {r['dup']:,} "
          f"／ 低质 {r['poor']:,}")
    print(f"输出：{r['dst']}")


if __name__ == "__main__":
    main()
