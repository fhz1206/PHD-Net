"""把 ModelScope `wikipedia-cn-20230720-filtered` 转换为 PHD-Net 纯文本语料。

数据源：`AI-ModelScope/wikipedia-cn-20230720-filtered`（ModelScope，参见 README 的引用与许可说明）。
该数据集基于中文维基百科 2023-07-20 dump，**保留了 254,547 条高质量词条**：
已过滤 Template/Category/File/Draft/Help 等特殊页与低质量页，已完成简繁转换与大陆习惯用词转换。
每条为 JSONL，`completion` 字段即正文（jsonl 体积 520.7 MB）。

产出：
    datasets/pretrain/wiki_train.txt   训练段（默认 2,000 万字符）
    datasets/pretrain/wiki_eval.txt    评估段（默认 200 万字符）

划分是**确定性的**：按词条顺序取尾部作为评估段，中间留 `--gap` 条隔离带，
避免同一主题相邻词条跨段泄漏；不随机、不洗牌，因此可复现。

用法：
    python tools/prepare_wikicn.py                       # 默认 20M / 2M 字符
    python tools/prepare_wikicn.py --train-chars 2000000 --eval-chars 200000
    python tools/prepare_wikicn.py --no-join             # 保留条目换行（默认条目内连成一段）
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IN = os.path.join(_ROOT, "datasets", "_raw", "wikipedia-cn-filtered.jsonl")
DEFAULT_TRAIN = os.path.join(_ROOT, "datasets", "pretrain", "wiki_train.txt")
DEFAULT_EVAL = os.path.join(_ROOT, "datasets", "pretrain", "wiki_eval.txt")

# 数据集已做过结构化清理，这里只去掉残余的行内标记与空白噪音
_RE_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.S)
_RE_HTML = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]{0,10}(\s[^<>]{0,60})?/?>")
_RE_WIKI_LINK = re.compile(r"\[\[([^\]|]{1,80}\|)?([^\]]{1,80})\]\]")
_RE_EMPH = re.compile(r"'''?")
_RE_FILE = re.compile(r"\[\[(文件|File|檔案|图像|Image):[^\]]*\]\]", re.I)
_RE_END = re.compile(r"[。！？；…）)\"」』】]|[.?!;]$")
_RE_BULLET = re.compile(r"^[\*#]{1,3}\s*")


def _join(lines: list[str]) -> str:
    """条目内的换行恢复为自然语言连接：

    数据集清洗后丢弃了换行位置信息，直接拼接会把小标题粘连成正文。
    这里按"前一行是否以句末标点结束"决定：未结束则补句号，遇到以括号开头的
    新段落先断句。这样既保留连续语流，也恢复句子边界。
    """
    out = lines[0]
    for ln in lines[1:]:
        prev = out[-1] if out else ""
        out += ln if (not prev or _RE_END.search(prev)) else "。" + ln
    return out

_RE_WS = re.compile(r"[ \t　]+")
_RE_BLANKS = re.compile(r"\n{2,}")


def clean(text: str, join_lines: bool) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = _RE_REF.sub("", t)
    t = _RE_FILE.sub("", t)
    t = _RE_WIKI_LINK.sub(r"\2", t)
    t = _RE_EMPH.sub("", t)
    t = _RE_HTML.sub("", t)
    t = _RE_WS.sub(" ", t)
    lines = [ln.strip() for ln in t.split("\n")]
    # 丢掉纯装饰行（孤立的 "*"/"-"/"|"/"!"）与 wiki 列表/标题标记
    lines = [_RE_BULLET.sub("", ln).strip() for ln in lines]
    lines = [ln for ln in lines if ln and ln not in {"*", "-", "|", "!", "#"}]
    if join_lines:
        t = _join(lines) if lines else ""
    else:
        t = "\n".join(lines)
    t = _RE_BLANKS.sub("\n", t)
    return t.strip()


def build(in_path: str, out_train: str, out_eval: str, *,
          train_chars: int, eval_chars: int, gap: int,
          min_chars: int, join_lines: bool, progress: int = 100000) -> dict:
    if not os.path.exists(in_path):
        sys.exit(f"缺少语料：{in_path}\n请先运行 python tools/fetch_modelscope.py")

    # 第一遍：统计可用词条数，确定评估段起点
    t0 = time.time()
    n_items = 0
    with io.open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                n_items += 1
    n_eval_start = n_items - int(eval_chars / max(1, min_chars) * 1.6) - gap
    n_eval_start = max(0, min(n_eval_start, n_items - 1))
    print(f"[info] 词条总数 {n_items:,}；评估段从第 {n_eval_start:,} 条起")

    train_parts: list[str] = []
    eval_parts: list[str] = []
    n_tr, n_ev = 0, 0
    i = 0
    with io.open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            i += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = rec.get("completion") or ""
            if len(raw) < min_chars:
                continue
            txt = clean(raw, join_lines)
            if len(txt) < min_chars:
                continue

            if i >= n_eval_start:
                if n_ev < eval_chars and i >= n_eval_start + gap:
                    eval_parts.append(txt)
                    n_ev += len(txt) + 1
            elif n_tr < train_chars:
                train_parts.append(txt)
                n_tr += len(txt) + 1

            if progress and i % progress == 0:
                print(f"[prog] 扫描 {i:,}/{n_items:,}  train {n_tr/1e6:.1f}M  "
                      f"eval {n_ev/1e6:.2f}M  {time.time()-t0:.0f}s", flush=True)
            if n_tr >= train_chars and n_ev >= eval_chars:
                break

    os.makedirs(os.path.dirname(out_train) or ".", exist_ok=True)
    with io.open(out_train, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(train_parts))
    with io.open(out_eval, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(eval_parts))
    return {
        "train_chars": n_tr,
        "eval_chars": n_ev,
        "train_bytes": os.path.getsize(out_train),
        "eval_bytes": os.path.getsize(out_eval),
        "train_items": len(train_parts),
        "eval_items": len(eval_parts),
        "scanned": i,
        "seconds": time.time() - t0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="wikipedia-cn-filtered → PHD-Net 文本语料")
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    ap.add_argument("--out-train", default=DEFAULT_TRAIN)
    ap.add_argument("--out-eval", default=DEFAULT_EVAL)
    ap.add_argument("--train-chars", type=int, default=20_000_000)
    ap.add_argument("--eval-chars", type=int, default=2_000_000)
    ap.add_argument("--gap", type=int, default=500, help="训练/评估之间的隔离词条数")
    ap.add_argument("--min-chars", type=int, default=200, help="丢弃过短词条")
    ap.add_argument("--no-join", action="store_true", help="保留条目内换行（默认连成一段）")
    args = ap.parse_args(argv)

    r = build(args.in_path, args.out_train, args.out_eval,
              train_chars=args.train_chars, eval_chars=args.eval_chars,
              gap=args.gap, min_chars=args.min_chars, join_lines=not args.no_join)
    print("-" * 62)
    print(f"train: {r['train_items']:,} 条 / {r['train_chars']:,} 字符 "
          f"({r['train_bytes']/1e6:.1f} MB) → {args.out_train}")
    print(f"eval : {r['eval_items']:,} 条 / {r['eval_chars']:,} 字符 "
          f"({r['eval_bytes']/1e6:.2f} MB) → {args.out_eval}")
    print(f"扫描 {r['scanned']:,} 条，用时 {r['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
