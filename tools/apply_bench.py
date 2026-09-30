#!/usr/bin/env python3
"""把云端基准结果（`tools/bench_server.py` 的 JSON）回填进性能文档。

**性能数字的唯一权威处是 `docs/PHD-Net_性能评估与迭代方案.md` 的基线表**，
其它文档只链接引用（见 `docs/文档写作规范.md` §3）。本脚本只改那一张表，
并把 JSON 归档到 `outputs/experiments/`，这样「数字 ↔ 测量配置 ↔ commit」
三者永远绑定在一起——避免再次出现「同一组数字抄在 4 份文档里然后漂移」。

用法：
```bash
python tools/apply_bench.py outputs/experiments/server_bench_20260930-190000.json
# 预览（不改文件）：
python tools/apply_bench.py <json> --dry-run
```
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PERF_DOC = _ROOT / "docs" / "PHD-Net_性能评估与迭代方案.md"
MARK_START = "<!-- BENCH:BEGIN -->"
MARK_END = "<!-- BENCH:END -->"


def _fmt_run(run: dict) -> str:
    """一组配置 → 一行表格（稳态 ms/tok + readout + PPL + 是否回落）。"""
    s = run.get("summary")
    if not s:
        return (f"| `{run['name']}` | — | — | — | "
                f"失败 rc={run.get('returncode')} |")
    fb = "⚠️ " + run["fallback"][:40] if run.get("fallback") else "—"
    extra = " ".join(a for a in run["argv"]
                     if a.startswith(("--m2", "--encoder", "--readout-dtype",
                                     "--numba", "--no-omp")))
    return (f"| `{run['name']}` | {s['ms_per_tok_steady']} | "
            f"{s['readout_ms_steady']} | {s['ppl_last']:.1f} | {fb} |"
            f" `{extra or 'baseline'}` |")


def render(bench: dict) -> str:
    env = bench.get("env", {})
    args = bench.get("args", {})
    lines = [
        MARK_START,
        f"### 云端实测基线（{env.get('ts', '?')} 实测，commit `{env.get('commit', '?')}`）",
        "",
        f"- 环境：{env.get('platform', '?')} / {env.get('machine', '?')} / "
        f"{env.get('cpu_count', '?')} 核 / torch {env.get('torch', '?')} / "
        f"numba {env.get('numba', '?')} / numpy {env.get('numpy', '?')}",
        f"- 配置：`--preset {args.get('preset')}`、`--tokens {args.get('steps')}`、"
        f"`--log-every {args.get('log_every')}`、`--remote-fraction "
        f"{args.get('fraction')}`、`--profile {args.get('profile')}`",
        "",
        "| 配置 | 稳态 ms/tok | readout ms/tok | 末段 PPL | 回落 | 差异开关 |",
        "|---|---|---|---|---|---|",
    ]
    for run in bench.get("runs", []):
        lines.append(_fmt_run(run))
    ok = [r for r in bench.get("runs", []) if r.get("summary")]
    if ok:
        base = next((r for r in ok if r["name"] == "baseline"), ok[0])
        bs = base["summary"]
        lines += ["", "**基线逐段分解**（`--step-profiling`，稳态后 20% 均值）：", ""]
        segs = base.get("segments") or {}
        loop = base.get("loop") or {}
        if segs:
            lines.append("| 段 | ms/tok |")
            lines.append("|---|---|")
            for k, v in sorted(segs.items(), key=lambda kv: -kv[1]):
                lines.append(f"| {k} | {v} |")
        if loop:
            lines += ["", "| 主循环 | ms/tok |", "|---|---|"]
            for k, v in loop.items():
                lines.append(f"| {k} | {v} |")
        tel = base.get("telemetry")
        if tel:
            lines += ["", f"- 遥测：CPU {tel['cpu_pct']}%（进程 {tel['cores']} 核）、"
                          f"加速器 {tel['acc_util']}、HBM {tel['hbm']}、"
                          f"CS/s {tel['ctx_switches']:,}、"
                          f"GC {tel['gc_objs_M']}M/gen2 {tel['gc_gen2']}"]
        trend = bs.get("trend_ms_per_tok") or []
        if trend:
            lines += ["", f"- 步时趋势（采样点）：{trend} → 若单调上升说明仍在超线性恶化",
                      ]
    lines.append(MARK_END)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()
    if not args.json_path.is_absolute():
        args.json_path = _ROOT / "outputs" / "experiments" / args.json_path
    if not args.json_path.exists():
        print(f"找不到 {args.json_path}", file=sys.stderr)
        return 1
    bench = json.loads(args.json_path.read_text(encoding="utf-8"))
    block = render(bench)

    if args.dry_run:
        print(block)
        return 0

    if not PERF_DOC.exists():
        print(f"性能文档不存在：{PERF_DOC}", file=sys.stderr)
        return 1
    text = PERF_DOC.read_text(encoding="utf-8")
    if MARK_START in text and MARK_END in text:
        new = re.sub(re.escape(MARK_START) + r".*?" + re.escape(MARK_END),
                     block, text, flags=re.S)
        action = "替换已有基线块"
    else:
        marker = "\n## "
        idx = text.find(marker, text.find(MARK_START) + 1 if MARK_START in text else 0)
        pos = text.find("\n## ", text.find("## ") + 1)
        anchor = text.find("\n## ", text.find("\n## ") + 1)
        pos = anchor if anchor > 0 else len(text)
        new = text[:pos] + "\n\n" + block + "\n" + text[pos:]
        action = "插入新基线块"
    PERF_DOC.write_text(new, encoding="utf-8")
    print(f"已{action}：{PERF_DOC.relative_to(_ROOT)}")
    print(f"基线日期：{date.today()}；来源 JSON：{args.json_path.name}")
    print("提示：README / index.html / 架构设计里如引用了具体步时，请一并核对"
          "（规范要求它们只链接本文，不复制数字）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
