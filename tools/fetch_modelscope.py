"""从 ModelScope 下载数据集文件（国内直连，实测 ~4 MB/s）。

用法：
    python tools/fetch_modelscope.py --ns AI-ModelScope --name wikipedia-cn-20230720-filtered \
        --file wikipedia-cn-20230720-filtered.jsonl --out data/_raw/wikipedia-cn-filtered.jsonl

说明：本机对境外源（dumps.wikimedia.org / huggingface.co）实测仅 ~10 KB/s，
而 ModelScope 与清华 PyPI 可达 4–5 MB/s —— 语料获取统一走 ModelScope 通道。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request

UA = "Mozilla/5.0"
API = "https://www.modelscope.cn/api/v1/datasets/{ns}/{name}/repo"


def fetch(ns: str, name: str, file: str, out: str, timeout: int = 120) -> int:
    url = API.format(ns=ns, name=name) + f"?Revision=master&FilePath={file}"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".part"
    pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"User-Agent": UA}
    if pos:
        headers["Range"] = f"bytes={pos}-"
        print(f"[info] 断点续传，已存在 {pos/1e6:.1f} MB")
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "ab") as fh:
        if pos and r.status != 206:
            print("[warn] 服务端不支持续传，从头下载")
            fh.seek(0)
            fh.truncate(0)
            pos = 0
        total = 0
        last_t = t0
        last_n = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            pos += len(chunk)
            total += len(chunk)
            now = time.time()
            if now - last_t >= 10:
                print(f"[prog] {pos/1e6:8.1f} MB  {(pos-last_n)/1e6/(now-last_t):6.2f} MB/s"
                      f"  用时 {(now-t0)/60:5.1f} min", flush=True)
                last_t, last_n = now, pos
    os.replace(tmp, out)
    size = os.path.getsize(out)
    print(f"[done] {out}  {size/1e6:.1f} MB  用时 {(time.time()-t0)/60:.1f} min")
    return size


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="下载 ModelScope 数据集文件")
    ap.add_argument("--ns", default="AI-ModelScope")
    ap.add_argument("--name", default="wikipedia-cn-20230720-filtered")
    ap.add_argument("--file", default="wikipedia-cn-20230720-filtered.jsonl")
    ap.add_argument("--out", default="data/_raw/wikipedia-cn-filtered.jsonl")
    args = ap.parse_args(argv)
    try:
        fetch(args.ns, args.name, args.file, args.out)
    except Exception as exc:  # noqa: BLE001
        print(f"[fail] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
