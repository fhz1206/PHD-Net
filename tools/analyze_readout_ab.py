#!/usr/bin/env python3
"""解析 `--readout-gather-impl index` vs `index_select` 的生产 A/B 日志。

用法：
    python tools/analyze_readout_ab.py <A日志> <B日志>

判据（项目纪律，别用错）：
  1. **probe_PPL 必须相同**（固定探针，逐位可比）→ 语义零回归；
     训练流 sliding PPL 波动 ±20%，不可用作判据。
  2. **readout ... (win)** 是滑窗口径（P218 新增），用它判性能；
     `(cum)` 含 JIT 冷编译，不判。
  3. `segments(win)` 的 CPU 侧各段，用于确认变化的到底是哪一段。
  4. 启动行 `[readout] gather_impl=<实际> (requested <CLI>)` 必须两轮不同
     且各自等于 CLI 值 —— 否则「开关没接线」，本次 A/B 无效。
"""

from __future__ import annotations

import argparse
import os
import re
import sys


def parse(path: str) -> dict:
    rows = []
    cfg = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[readout] gather_impl=" in line:
                m = re.search(r"gather_impl=(\S+)\s+\(requested\s+(\S+)\)", line)
                if m:
                    cfg["actual"] = m.group(1)
                    cfg["requested"] = m.group(2)
            if "[readout] backend=" in line:
                cfg["backend"] = line.strip()
            # 采样行：token ... sliding PPL ... probe_PPL ... ms/tok ... | readout X (cum) / Y (win)
            m = re.search(
                r"token\s+([\d,]+)\s+sliding PPL\s+([\d.]+)"
                r"(?:\s+probe_PPL\s+([\d.]+))?"
                r".*?([\d.]+)\s+ms/tok"
                r".*?\|\s*readout\s+([\d.]+)\s+ms/tok\s*\(cum\)"
                r"\s*/\s*([\d.]+)\s*\(win\)", line)
            if m:
                rows.append({
                    "token": int(m.group(1).replace(",", "")),
                    "sliding_ppl": float(m.group(2)),
                    "probe_ppl": (float(m.group(3)) if m.group(3) else None),
                    "ms_per_tok": float(m.group(4)),
                    "readout_cum": float(m.group(5)),
                    "readout_win": float(m.group(6)),
                })
            # segments(win)
            mw = re.search(r"segments\(win\):\s*(.+?)\s*\[", line)
            if mw and rows:
                segs = {}
                for k, v in re.findall(r"([A-Za-z0-9_]+)\s+([\d.]+)", mw.group(1)):
                    segs[k] = float(v)
                rows[-1]["segments_win"] = segs
    return {"cfg": cfg, "rows": rows, "path": path}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="index 轮日志")
    ap.add_argument("b", help="index_select 轮日志")
    args = ap.parse_args()
    for p in (args.a, args.b):
        if not os.path.exists(p):
            print(f"文件不存在: {p}")
            return 2

    A, B = parse(args.a), parse(args.b)
    print("=" * 92)
    print("生产 A/B：readout gather 实现")
    print("=" * 92)
    for tag, d in (("A", A), ("B", B)):
        c = d["cfg"]
        print(f"  {tag}: {os.path.basename(d['path'])}")
        print(f"     生效 gather_impl = {c.get('actual')}  "
              f"(CLI requested = {c.get('requested')})")
        print(f"     {c.get('backend', '(无 backend 行)')}")

    if A["cfg"].get("actual") == B["cfg"].get("actual"):
        print("\n  ❌ 两轮生效值相同 → A/B 无效（开关没接线或用了同一代码）")
        return 1
    if (A["cfg"].get("actual") != A["cfg"].get("requested") or
            B["cfg"].get("actual") != B["cfg"].get("requested")):
        print("\n  ❌ 生效值 != CLI 值 → 配置传递链断了")
        return 1

    ra = {r["token"]: r for r in A["rows"]}
    rb = {r["token"]: r for r in B["rows"]}
    common = sorted(set(ra) & set(rb))
    print(f"\n  共同采样点 {len(common)} 个: {common}")

    print(f"\n{'token':>7} {'probe A':>12} {'probe B':>12} {'相同?':>6} "
          f"{'ms/tok A':>10} {'B':>8} {'Δ':>8} "
          f"{'ro(win) A':>10} {'B':>9} {'加速':>8}")
    print("-" * 92)
    ppl_ok = True
    speedups, ro_ratios = [], []
    for t in common:
        a, b = ra[t], rb[t]
        pa, pb = a["probe_ppl"], b["probe_ppl"]
        same = (pa is not None and pb is not None and abs(pa - pb) < 1e-6)
        if pa is not None and pb is not None and not same:
            ppl_ok = False
        da = a["ms_per_tok"] - b["ms_per_tok"]
        ratio = a["readout_win"] / max(1e-9, b["readout_win"])
        sp = a["ms_per_tok"] / max(1e-9, b["ms_per_tok"])
        speedups.append(sp); ro_ratios.append(ratio)
        print(f"{t:>7,} {(f'{pa:.4f}' if pa else '—'):>12} "
              f"{(f'{pb:.4f}' if pb else '—'):>12} "
              f"{('✅' if same else ('—' if pa is None else '❌')):>6} "
              f"{a['ms_per_tok']:>10.2f} {b['ms_per_tok']:>8.2f} {da:>+8.2f} "
              f"{a['readout_win']:>10.3f} {b['readout_win']:>9.3f} "
              f"{ratio:>7.2f}×")

    if speedups:
        n = len(speedups)
        print("-" * 92)
        print(f"  均值: 端到端 {sum(speedups)/n:.2f}× | "
              f"readout(win) {sum(ro_ratios)/n:.2f}×")
    print(f"\n  语义判据: probe_PPL {'✅ 完全相同（零回归）' if ppl_ok else '❌ 有差异，需排查'}")
    print("  ⚠ 若 readout(win) 无差异，检查：① 是否真走稀疏臂 conn_k>0；"
          "② gather 缓存是否每步命中；③ 启动行生效值。")
    print("=" * 92)
    return 0 if ppl_ok else 1


if __name__ == "__main__":
    sys.exit(main())
