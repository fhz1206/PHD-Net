"""Ultra-FineWeb-L3（ModelScope）分片下载 —— 只取采样所需的少量分片。

数据集全量 1,764 个 parquet × ~1.08 GB ≈ **1.9 TB**（en 400B+ / zh 200B+ tokens），
本机磁盘与带宽都装不下，故**按子集各取 1 个分片**做采样（下载后立即删除原始）。

用法：
  python tools/fetch_ultrafineweb.py --list                 # 列出分片与大小
  python tools/fetch_ultrafineweb.py --subset zh_qa --n 1   # 下载该子集前 1 个分片
"""
from __future__ import annotations

import argparse
import os
import subprocess
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REPO = "OpenBMB/Ultra-FineWeb-L3"
_BASE = f"https://www.modelscope.cn/api/v1/datasets/{_REPO}/repo?Revision=master&FilePath="
_OUT = _ROOT / ".tmp_ufw_data"

SUBSETS = {
    "en_multi_style": "data/ultrafineweb_en_l3/multi_style",
    "en_qa": "data/ultrafineweb_en_l3/qa",
    "zh_multi_style": "data/ultrafineweb_zh_l3/multi_style",
    "zh_qa": "data/ultrafineweb_zh_l3/qa",
}


def all_files() -> list[str]:
    repo = _ROOT / ".tmp_ufw"
    if repo.exists():
        out = subprocess.run(["git", "ls-tree", "-r", "HEAD", "--name-only"],
                             capture_output=True, text=True, cwd=str(repo)).stdout
        return [f for f in out.splitlines() if f.endswith(".parquet")]
    return []


def download(path: str, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"[skip] {dst.name}")
        return dst.stat().st_size
    with urllib.request.urlopen(_BASE + path, timeout=600) as r, open(dst, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            total += len(chunk)
            f.write(chunk)
    print(f"[ok] {dst.name} {total/1e6:.1f} MB", flush=True)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=list(SUBSETS), action="append")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    files = all_files()
    if a.list or not files:
        for k, d in SUBSETS.items():
            fs = sorted(f for f in files if f.startswith(d + "/"))
            print(f"{k:<16}{len(fs):>5} 个分片  {d}")
        if not files:
            print("（先 git clone --filter=blob:none --no-checkout 到 .tmp_ufw 才能列文件）")
        return
    for s in a.subset or list(SUBSETS):
        fs = sorted(f for f in files if f.startswith(SUBSETS[s] + "/"))[:a.n]
        for f in fs:
            download(f, _OUT / s / Path(f).name)


if __name__ == "__main__":
    main()
