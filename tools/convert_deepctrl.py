"""匠数 deepctrl-sft-data（ModelScope）→ PHD-Net SFT 纯文本（`datasets/sft/`）。

================================================================================
为什么要先过滤（fhz 2026-09-24：「有些我个人觉得不适合作为 sft 的语料」）
================================================================================
该数据集名为 SFT，但实测（探针 tools/_probe_deepctrl.py，扫 15.9 万条）含大量
**不具备指令跟随训练价值**的样本：

  1. **上下文依赖的续问**：`history` 非空而 `input` 是"好的，那接下来…"/
     "嗯，谢谢你介绍的做法很详细，但我不喜欢吃鸡蛋…"——脱离历史后 input 不可
     理解，单拿出来训练等于教模型"答非所问"。占比可观（探针中大量出现）。
  2. **极短无信息 output**：如"你好，有什么我可以帮助你的吗？"（<40 字符）。
  3. **过短/空 input**、**重复 input**、**超长 output**（尾部截断风险）。

过滤策略：① 续问 → 丢弃；② 自足的多轮 → **把 history 拼成上下文**保留（多轮
对话是有价值的 SFT 形态）；③ 短 output / 短 input / 重复 / 超长 → 丢弃。

================================================================================
磁盘约束（fhz：「磁盘占用不能太大」）
================================================================================
原始文件 zh 17.19 GB + en 12.01 GB = **29 GB**，不落盘：用 ModelScope 直连 API
**流式读取**，边读边过滤边写，达到 `--max-samples` / `--max-mb` 即停。
磁盘只增长输出文件（默认 zh ≤ 300 MB / en ≤ 120 MB）。

用法：
  python tools/convert_deepctrl.py --lang zh                 # 默认 300 MB 上限
  python tools/convert_deepctrl.py --lang en --max-mb 120
  python tools/convert_deepctrl.py --lang zh --max-samples 5000 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# 控制台编码容错（2026-10-07 修复）：与 train/train.py 同源 —— 中文 Windows 的
# GBK 控制台打不出 ⚠/emoji → `UnicodeEncodeError` 让 `--help` 直接崩。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass
URL = ("https://www.modelscope.cn/api/v1/datasets/deepctrl/deepctrl-sft-data/repo"
       "?Revision=master&FilePath=sft_data_{lang}.jsonl")

SEP = "\n\n"

# 续接词：这些开头的 input 在多轮语境里是"接着上一句说"，脱离 history 即不可解
CONTINUE_PREFIX = (
    "好的", "好，", "好吧", "嗯", "那", "那么", "谢谢", "多谢", "再", "还有",
    "另外", "是的", "对，", "对的", "明白", "可以", "接下来", "接着", "继续",
    "然后", "哦", "行", "不错", "很好", "太好了", "ok", "OK", "yes", "Yes",
    "thanks", "Thanks", "okay", "Okay", "Great", "great", "sure", "Sure",
    "and then", "Also", "also", "Next", "next",
)
MIN_INPUT = 4
MIN_OUTPUT = 40
MAX_OUTPUT = 3000
SHORT_INPUT_CTX = 15      # history 非空时，input 短于此视为依赖上下文


def _ctx_dependent(inp: str, has_history: bool) -> bool:
    """是否依赖上下文（脱离 history 后 input 不可理解）。"""
    if not has_history:
        return False
    s = inp.strip()
    if len(s) < SHORT_INPUT_CTX:
        return True
    return s.startswith(CONTINUE_PREFIX)


def _clean(t: str | None) -> str:
    if not t:
        return ""
    return t.replace("\r\n", "\n").strip()


def _fmt(instruction: str, inp: str, history: list, out: str) -> str:
    parts = []
    if history:                                  # 自足的多轮：把历史拼成上下文
        hist = []
        for turn in history[-3:]:                # 最多带 3 轮，控体积
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                u, a = _clean(turn[0]), _clean(turn[1])
                if u or a:
                    hist.append(f"用户：{u}\n助手：{a}")
        if hist:
            parts.append("历史对话：\n" + "\n".join(hist))
    head = _clean(instruction)
    body = _clean(inp)
    q = (head + "\n" + body).strip() if head else body
    if q:
        parts.append(q)
    parts.append(_clean(out))
    return "\n\n".join(parts)


def convert(lang: str, dst: Path, max_samples: int = 0, max_mb: float = 300.0,
            dry_run: bool = False) -> dict:
    seen: set[int] = set()
    n_read = n_keep = n_short_out = n_ctx = n_dup = n_long = n_bad = 0
    bytes_out = 0
    t0 = time.perf_counter()
    fh = None if dry_run else open(dst, "w", encoding="utf-8", newline="\n")
    try:
        limit = int(max_mb * 1_000_000)
        with urllib.request.urlopen(URL.format(lang=lang), timeout=300) as r:
            for raw in r:                        # 按行流式，原始文件不落盘
                n_read += 1
                if max_samples and n_keep >= max_samples:
                    break
                if bytes_out >= limit:
                    break
                try:
                    d = json.loads(raw.decode("utf-8"))
                except Exception:
                    n_bad += 1
                    continue
                inp = _clean(d.get("input"))
                out = _clean(d.get("output"))
                if len(inp) < MIN_INPUT:
                    n_short_out += 1
                    continue
                if len(out) < MIN_OUTPUT or len(out) > MAX_OUTPUT:
                    n_long += 1
                    continue
                hist = d.get("history") or []
                if _ctx_dependent(inp, bool(hist)):
                    n_ctx += 1
                    continue
                key = hash(inp)
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                s = _fmt(d.get("instruction"), inp, hist, out)
                b = len(s.encode("utf-8")) + len(SEP)
                if bytes_out + b > limit:
                    break
                if fh is not None:
                    fh.write(s if n_keep == 0 else SEP + s)
                bytes_out += b
                n_keep += 1
                if n_keep % 20000 == 0:
                    print(f"  ... {n_keep:,} 条 / {bytes_out/1e6:.1f} MB "
                          f"（已读 {n_read:,}）", flush=True)
    finally:
        if fh is not None:
            fh.close()
    dt = time.perf_counter() - t0
    return dict(lang=lang, read=n_read, kept=n_keep, mb=bytes_out / 1e6,
                drop_short=n_short_out, drop_len=n_long, drop_ctx=n_ctx,
                drop_dup=n_dup, bad=n_bad, sec=dt,
                dst=str(dst) if not dry_run else "(dry-run)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "zh"], default="en",
                    help="终端输出语言（P119：默认英文；--lang zh 输出中文。"
                         "不影响转换结果）")
    ap.add_argument("--dst", type=Path, default=None)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-mb", type=float, default=300.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dst = a.dst or (_ROOT / "datasets" / "sft" / "raw" / f"deepctrl_sft_{a.lang}.txt")
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"deepctrl sft_data_{a.lang}.jsonl → {dst}")
    print(f"上限：{a.max_mb:.0f} MB"
          + (f" / {a.max_samples:,} 条" if a.max_samples else ""))
    r = convert(a.lang, dst, a.max_samples, a.max_mb, a.dry_run)
    print("-" * 70)
    print(f"读取 {r['read']:,} 条  →  保留 {r['kept']:,} 条（{r['mb']:.1f} MB）"
          f"  用时 {r['sec']:.0f}s")
    print(f"丢弃：上下文依赖 {r['drop_ctx']:,} ／ 过短或过长 {r['drop_len']:,} "
          f"／ 空 input {r['drop_short']:,} ／ 重复 {r['drop_dup']:,} "
          f"／ 解析失败 {r['bad']:,}")
    print(f"输出：{r['dst']}")


if __name__ == "__main__":
    main()
