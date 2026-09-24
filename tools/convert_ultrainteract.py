"""UltraInteract_sft（OpenBMB）parquet → PHD-Net SFT 纯文本（`datasets/sft/`）。

数据源：atomgit `OpenBMB/UltraInteract_sft`（fhz 2026-09-23 指定）——
288,579 条，170.7 MB parquet；字段 `task/dataset/instruction/response/id/parent_id`；
`parent_id` 构成偏好树（同一 instruction 的多个 response 共享 parent_id）。
语言以英文为主（Coding / Math / 等复杂推理任务）。

输出模式：
  --mode flat（默认）：每条样本 → `instruction\\n\\nresponse`，样本间 `\\n\\n`
    —— SFT 标准形态（同指令的多个响应各成一条）。
  --mode grouped：按 parent_id 分组 → 一条 instruction 后接其全部 response
    —— 保留偏好树结构（供后续偏好学习/对比使用）。

用法：
  python tools/convert_ultrainteract.py --stats          # 只统计（不写文件）
  python tools/convert_ultrainteract.py --limit 20000    # 前 2 万条
  python tools/convert_ultrainteract.py                  # 全量（flat）
  python tools/convert_ultrainteract.py --mode grouped
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SRC = _ROOT / ".tmp_sft_clone" / "0000_sft.parquet"
_DEFAULT_DST = _ROOT / "datasets" / "sft" / "ultrainteract_sft.txt"

SEP = "\n\n"


def iter_rows(src: Path, limit: int = 0):
    pf = pq.ParquetFile(str(src))
    n = 0
    for batch in pf.iter_batches(batch_size=2000,
                                 columns=["task", "dataset", "instruction",
                                          "response", "parent_id"]):
        cols = {c: batch.column(c) for c in
                ("task", "dataset", "instruction", "response", "parent_id")}
        for i in range(batch.num_rows):
            yield tuple(cols[c][i].as_py() for c in
                        ("task", "dataset", "instruction", "response", "parent_id"))
            n += 1
            if limit and n >= limit:
                return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=_DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=_DEFAULT_DST)
    ap.add_argument("--mode", choices=["flat", "grouped"], default="flat")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"源文件不存在: {args.src}")
        sys.exit(1)

    if args.stats:
        n = n_chars = 0
        tasks: Counter = Counter()
        datasets: Counter = Counter()
        parents: set = set()
        for task, ds, ins, resp, pid in iter_rows(args.src, args.limit):
            n += 1
            n_chars += len(ins or "") + len(resp or "") + len(SEP)
            tasks[task] += 1
            datasets[ds] += 1
            parents.add(pid)
        print(f"样本数: {n:,}")
        print(f"字符量: {n_chars:,}（{n_chars / 1e9:.3f} GB）")
        print(f"唯一 parent_id（唯一指令数）: {len(parents):,}"
              f" → 平均 {n / max(1, len(parents)):.2f} 响应/指令")
        print("task 分布（前 8）:")
        for k, v in tasks.most_common(8):
            print(f"  {k:<24}{v:>9,}  ({v / n * 100:.1f}%)")
        print("dataset 分布（前 8）:")
        for k, v in datasets.most_common(8):
            print(f"  {k:<24}{v:>9,}  ({v / n * 100:.1f}%)")
        return

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    n = n_chars = 0
    if args.mode == "flat":
        with open(args.dst, "w", encoding="utf-8") as f:
            for _task, _ds, ins, resp, _pid in iter_rows(args.src, args.limit):
                if not ins or not resp:
                    continue
                text = ins + SEP + resp
                f.write(text + SEP)
                n += 1
                n_chars += len(text) + len(SEP)
                if n % 50_000 == 0:
                    print(f"  已写 {n:,} 条 / {n_chars / 1e9:.3f} GB", flush=True)
    else:  # grouped：按 parent_id 保留偏好树
        groups: dict[str, list[tuple[str, str]]] = {}
        for _task, _ds, ins, resp, pid in iter_rows(args.src, args.limit):
            if not ins or not resp:
                continue
            groups.setdefault(pid, []).append((ins, resp))
        with open(args.dst, "w", encoding="utf-8") as f:
            for pid, items in groups.items():
                head = items[0][0]
                block = head + SEP + SEP.join(r for _, r in items)
                f.write(block + SEP)
                n += len(items)
                n_chars += len(block) + len(SEP)
        print(f"  分组数: {len(groups):,}")
    print(f"完成: {args.dst}  样本 {n:,} 条, {n_chars / 1e9:.3f} GB 文本")


if __name__ == "__main__":
    main()
