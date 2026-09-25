"""把 datasets/ 的上传交付物推到 ModelScope（fhzfhz/Mixture-General-Mini）。

范围：54 个 ≤100 MiB 的 parquet 分片（sft/ 2 片 + pretrain/ 52 片）
     + .gitattributes / README.md / UPLOAD_README.md。
排除：.git / eval（内置语料不外发）/ */raw（原始归档）/ *.txt。

用法：python tools/upload_modelscope.py            # 全量
     python tools/upload_modelscope.py --verify    # 上传后核对远端文件清单
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)

TOKEN = "ms-6222a6fc-cc01-452e-8193-f4b669ec7c01"
REPO = "fhzfhz/Mixture-General-Mini"
IGNORE = [".git/**", ".git/*", "raw/**", "raw/*", "eval/**", "eval/*",
          "sft/raw/**", "pretrain/raw/**", "*.txt"]


def local_expect() -> list[str]:
    """应上传的文件（相对 datasets/ 的路径）。"""
    out = []
    for pat in ("sft/*.parquet", "pretrain/*.parquet"):
        out += [p.replace("\\", "/").split("datasets/", 1)[1]
                for p in glob.glob(f"datasets/{pat}")]
    for f in (".gitattributes", "README.md", "UPLOAD_README.md"):
        out.append(f)
    return sorted(set(out))


def verify() -> None:
    from modelscope.hub.api import HubApi
    api = HubApi()
    api.login(TOKEN)
    remote = set(api.get_repo_files(repo_id=REPO, repo_type='dataset')) \
        if hasattr(api, "get_repo_files") else None
    if remote is None:
        print("SDK 无 get_repo_files，改用 API 查询")
        import urllib.request, json
        u = (f"https://www.modelscope.cn/api/v1/datasets/{REPO}/repo/files"
             f"?Revision=master&Recursive=true")
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read().decode("utf-8"))
        files = d.get("Data", {}).get("Files", [])
        remote = {f.get("Path") or f.get("Name") for f in files}
    expect = set(local_expect())
    missing = sorted(expect - remote)
    print(f"远端文件 {len(remote)} 个；本地应有 {len(expect)} 个")
    print("缺失：", missing if missing else "无 —— 上传完整 ✓")


def upload() -> None:
    from modelscope.hub.api import HubApi
    api = HubApi()
    api.login(TOKEN)
    t0 = time.time()
    api.upload_folder(
        repo_id=REPO,
        folder_path="datasets",
        repo_type="dataset",
        commit_message="Mixture-General-Mini：SFT 331.9 万条 + 预训练 3,905.9 万条"
                       "（54 个 ≤100MiB parquet 分片，schema: text/lang/src）",
        ignore_patterns=IGNORE,
    )
    print(f"[完成] 用时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    verify() if a.verify else upload()
