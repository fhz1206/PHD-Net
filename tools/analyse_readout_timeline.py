"""P141：解析 msprof 导出的 timeline，**自动找出我们没数到的算子**。

为什么需要这个
================================================================================
P135~P140 的外部测量已到极限：裸算子合计 0.35 ms，而设备单步真实执行
**9.2~14.1 ms**（昇腾 1b 档），差 26 倍。逐行打点显示 Python 侧除 `.item()`
外所有行都 < 0.2 ms —— **说明设备上确实在执行我们没数到的 kernel**。

本脚本读msprof 导出的 timeline CSV，做一件人做不了的事：
**把设备侧算子按耗时排序，与「我们已知的 8 个」做差集**，直接列出
「多出来的」是什么 —— 那就是答案所在。

用法
================================================================================
    # 1) 先采集（昇腾上）
    bash tools/prof_readout_msprof.sh
    # 2) 解析（本机也行，纯文本处理）
    python tools/analyse_readout_timeline.py outputs/prof/msprof_readout
    # 或直接给 CSV
    python tools/analyse_readout_timeline.py path/to/timeline.csv

⚠ **列名是猜的**：msprof 不同版本的 CSV 列名不同。本脚本会先打印实际列名，
   找不到就按位置猜，并在输出里**明确标注「列名是猜的」**。
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from collections import defaultdict

# 我们已知的算子（P135 在昇腾上实测的裸算子耗时，单位 ms）
KNOWN = {
    "h[wi]": 0.020, "index_select": 0.025, "take": 0.016,
    "einsum": 0.142, "gather_sparse_index": 0.034,
    "addcmul_": 0.061, "addmm_": 0.061, "cross_entropy": 0.104,
    "memcpy_dtoh": 0.038, "memcpy_htod": 0.038,
}


def _find_csv(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    out: list[str] = []
    for pat in ("**/*.csv", "**/*.timeline", "**/*.asc"):
        out += glob.glob(os.path.join(path, pat), recursive=True)
    # 优先设备侧 timeline（通常最大/名字含 device）
    out.sort(key=lambda p: -(os.path.getsize(p)))
    return out


def _pick_cols(header: list[str]) -> tuple[int, int, int, str] | None:
    """返回 (name_col, dur_col, start_col, 判据说明)。找不到返回 None。"""
    low = [h.strip().lower() for h in header]
    dur = next((i for i, h in enumerate(low)
                if h in ("duration", "dur", "duration(us)", "time")), None)
    name = next((i for i, h in enumerate(low)
                 if h in ("name", "op", "op name", "kernel", "event")), None)
    start = next((i for i, h in enumerate(low)
                  if h in ("start", "begin", "ts", "start(us)")), None)
    if dur is not None and name is not None:
        return name, dur, (start if start is not None else -1), "按列名"
    return None


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "outputs/prof/msprof_readout"
    files = _find_csv(target)
    if not files:
        print(f"[analyse] 在 {target} 下没找到 CSV/timeline。")
        print("          先跑 `bash tools/prof_readout_msprof.sh` 采集，")
        print("          或用 `find <dir> -name '*.csv'` 确认产物位置。")
        return 2

    print("=" * 84)
    print("P141 读出 timeline 分析 —— 找出「我们没数到的算子」")
    print("=" * 84)
    for f in files[:3]:
        print(f"  候选: {f}（{os.path.getsize(f)/2**20:.1f} MiB）")
    path = files[0]
    print(f"\n使用: {path}")

    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rdr = csv.reader(fh)
        header = next(rdr, None)
        if not header:
            print("  空文件/无表头")
            return 2
        print(f"\n实际列名: {header}")
        picked = _pick_cols(header)
        guessed = False
        if picked is None:
            # 按位置猜：msprof 常见布局是 id,name,start,duration,...
            guessed = True
            name_c, dur_c = (1, 3) if len(header) > 3 else (0, len(header) - 1)
            start_c = 2 if len(header) > 2 else -1
            print("⚠ **列名识别失败，按位置猜测**："
                  f"name=col{name_c}, duration=col{dur_c}"
                  "（结果请谨慎解读）")
        else:
            name_c, dur_c, start_c, how = picked
            print(f"列名识别: {how} → name=col{name_c}, duration=col{dur_c}")

        # 累加：按算子名聚合
        agg: dict[str, list[float]] = defaultdict(list)
        n = 0
        for row in rdr:
            if len(row) <= max(name_c, dur_c):
                continue
            nm = row[name_c].strip()
            if not nm:
                continue
            try:
                d = float(row[dur_c])
            except ValueError:
                continue
            agg[nm].append(d)
            n += 1
    if not n:
        print("没有可解析的数据行")
        return 2

    unit = "（列名含 (us) 则单位是微秒）" if "us" in str(header[dur_c]).lower() else ""
    total = sum(sum(v) for v in agg.values())
    print(f"\n共 {n:,} 条记录，{len(agg)} 种算子，累计 {total:,.1f}{unit}")

    print("\n" + "=" * 84)
    print(f"{'算子':<52}{'次数':>8}{'累计':>14}{'占比':>8}")
    print("-" * 84)
    items = sorted(agg.items(), key=lambda kv: -sum(kv[1]))
    for nm, vals in items[:25]:
        s = sum(vals)
        print(f"{nm[:50]:<52}{len(vals):>8}{s:>14,.1f}"
              f"{s/total*100:>7.1f}%")

    # 差集：我们已知的那几个
    print("\n" + "=" * 84)
    print("与「我们已知的算子」对照（P135 在昇腾实测的裸算子耗时）")
    print("=" * 84)
    hit, miss = [], []
    for nm, vals in items:
        low = nm.lower()
        matched = next((k for k in KNOWN if k in low), None)
        if matched:
            hit.append((nm, sum(vals), matched))
        else:
            miss.append((nm, sum(vals), len(vals)))
    if hit:
        print("\n【已覆盖的算子】（我们能解释的）")
        for nm, s, k in hit[:10]:
            print(f"  {nm[:48]:<50}{s:>12,.1f}  (裸算子 {KNOWN[k]:.3f})")
    print(f"\n【**未覆盖的算子**】← 答案在这里（占累计 "
          f"{sum(s for _, s, _ in miss)/total*100:.1f}%）")
    for nm, s, c in miss[:15]:
        print(f"  {nm[:48]:<50}{s:>12,.1f}  ×{c}")
    if not miss:
        print("  （无—— 所有算子都能被已知的 8个解释）")
    print("\n→ 未覆盖的算子就是我们**没数到**的那26 倍。"
          "看它们的名称与累计耗时，即可判断是"
          " dtype 转换 / 内存分配 / 搬运 / 隐式同步 中的哪一种。")
    if guessed:
        print("\n⚠ 本次**列名是猜的**，结论请对照 `实际列名` 那一行复核。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())