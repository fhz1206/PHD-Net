"""把 datasets/ 下的纯文本语料转成 parquet（列式 + zstd 压缩）。

fhz 2026-09-24 指令「修改模型导入，改为 parquet」：训练/评测的数据入口统一走
`phdnet.corpus.load_text`，它同时认 `.txt` 与 `.parquet`，给出 parquet 路径即
走列式读取（支持只读部分 row group，大语料不必整文件载入内存）。

转出来的 parquet 内建压缩，体积显著小于原文（见 `outputs/build_parquet.log`）：
  · 可直接替换原 txt 供训练使用；
  · 原 txt **默认保留**（`--replace` 才删除），避免破坏既有脚本与基线。

用法：
  python tools/build_parquet.py --stats                 # 只统计
  python tools/build_parquet.py                         # 转换 datasets/ 下全部 txt
  python tools/build_parquet.py datasets/sft/x.txt      # 只转指定文件
  python tools/build_parquet.py --replace               # 转换后删除原 txt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from phdnet.corpus import corpus_stats, iter_texts, write_parquet   # noqa: E402

DATASETS = _ROOT / "datasets"


def detect_lang(text: str) -> str:
    cjk = sum(1 for c in text[:2000] if "\u4e00" <= c <= "\u9fff")
    lat = sum(1 for c in text[:2000] if c.isascii() and c.isalpha())
    if cjk > 200 and cjk > lat:
        return "zh"
    if lat > 200:
        return "en"
    return "unk"


def convert(src: Path, dst: Path | None = None, replace: bool = False,
            stats_only: bool = False) -> dict:
    st = corpus_stats(src, max_items=20_000)
    lang = detect_lang(next(iter_texts(src), ""))
    st["lang"] = lang
    if stats_only:
        return st
    dst = dst or src.with_suffix(".parquet")
    n = write_parquet(iter_texts(src), dst, lang=lang, src=src.name)
    out_mb = os.path.getsize(dst) / 1e6
    st.update({"written": n, "dst": str(dst), "dst_mb": out_mb,
               "ratio": st["size_mb"] / max(1e-9, out_mb)})
    if replace and dst.exists():
        os.remove(src)
        st["removed_src"] = True
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    files = [Path(f) for f in a.files] or sorted(
        p for d in ("sft", "pretrain") for p in (DATASETS / d / "raw").glob("*.txt"))
    print(f"{'语料':<34}{'条数':>10}{'语言':>6}{'原 MB':>9}{'parquet MB':>11}{'压缩':>8}")
    print("-" * 80)
    for f in files:
        if not f.exists():
            print(f"{str(f):<34}  (不存在)")
            continue
        r = convert(f, replace=a.replace, stats_only=a.stats)
        if a.stats:
            print(f"{Path(r['path']).name:<34}{r['items']:>10,}{r['lang']:>6}"
                  f"{r['size_mb']:>9.1f}{'-':>11}{'-':>8}")
        else:
            print(f"{Path(r['path']).name:<34}{r['written']:>10,}{r['lang']:>6}"
                  f"{r['size_mb']:>9.1f}{r['dst_mb']:>11.1f}{r['ratio']:>7.2f}x",
                  flush=True)


if __name__ == "__main__":
    main()
