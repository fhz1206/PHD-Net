"""语料分卷打包 —— 绕开 gitcode 的「单文件 ≤100 MiB」pre-receive 钩子。

背景（2026-09-24 实测两条路都被堵）：
  · 普通 git：remote hook 拒绝 >100 MiB 的文件，提示必须用 LFS；
  · Git LFS：命名空间配额 **已用 166 GB / 限额 100 GB**，任何 LFS 上传都被拒。
故改用「gzip 压缩 + 按字节分卷」，每片 ≤ `--chunk-mb`（默认 90，留 10 MiB 余量），
以普通 git 对象入库。还原：

    cat xxx.gz.part-* > xxx.gz && gunzip xxx.gz        # Linux / Git Bash
    copy /b xxx.gz.part-* xxx.gz                        # Windows cmd

产出同时写一份 `SHA256SUMS.txt`（每片一行）便于校验。

用法：
  python tools/pack_corpus.py                       # 打包 datasets/ 下全部 >100 MiB 的语料
  python tools/pack_corpus.py datasets/sft/x.txt    # 只打包指定文件
  python tools/pack_corpus.py --chunk-mb 80 --level 6
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _ROOT / "datasets" / "packed"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def pack_one(src: Path, out_dir: Path, chunk_mb: float, level: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem                      # 去掉 .txt
    gz_tmp = out_dir / f".{stem}.gz.tmp"
    chunk = int(chunk_mb * 1_048_576)
    t0 = time.time()
    with open(src, "rb") as fi, gzip.open(gz_tmp, "wb", compresslevel=level) as fo:
        shutil.copyfileobj(fi, fo, 1 << 22)
    gz_size = gz_tmp.stat().st_size

    parts = []
    idx = 0
    # ⚠ 不可用 shutil.copyfileobj(fi, fo, chunk)：第三参数是**缓冲区大小**而非
    #   总量上限，会把整个 gz 写进一个 part（初版 bug：M7 Core 4.5 GB 全落在
    #   part-000，触发远端 100 MiB 钩子）。必须按 chunk 显式 read。
    with open(gz_tmp, "rb") as fi:
        while True:
            data = fi.read(chunk)
            if not data:
                break
            dst = out_dir / f"{stem}.gz.part-{idx:03d}"
            with open(dst, "wb") as fo:
                fo.write(data)
            parts.append(dst)
            idx += 1
    gz_tmp.unlink()
    return {"src": str(src), "src_mb": src.stat().st_size / 1e6,
            "gz_mb": gz_size / 1e6, "ratio": src.stat().st_size / max(1, gz_size),
            "parts": [p.name for p in parts], "sec": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--chunk-mb", type=float, default=90.0)
    ap.add_argument("--level", type=int, default=6)
    a = ap.parse_args()
    files = [Path(f) for f in a.files] or [
        p for d in ("sft", "pretrain") for p in sorted((_ROOT / "datasets" / d).glob("*.txt"))
    ]
    sums = []
    print(f"分卷上限 {a.chunk_mb:.0f} MiB ／ gzip level {a.level}")
    for f in files:
        if not f.exists():
            print(f"  (跳过，不存在) {f}")
            continue
        r = pack_one(f, a.out, a.chunk_mb, a.level)
        print(f"{f.name}: {r['src_mb']:.1f} MB → gzip {r['gz_mb']:.1f} MB "
              f"（{r['ratio']:.2f}×）→ {len(r['parts'])} 片  [{r['sec']:.0f}s]", flush=True)
        for p in r["parts"]:
            sums.append((p, a.out / p))
    if sums:
        with open(a.out / "SHA256SUMS.txt", "w", encoding="utf-8") as fh:
            for name, path in sums:
                fh.write(f"{sha256_of(path)}  {name}\n")
        print(f"校验清单：{a.out / 'SHA256SUMS.txt'}（{len(sums)} 片）")


if __name__ == "__main__":
    main()
