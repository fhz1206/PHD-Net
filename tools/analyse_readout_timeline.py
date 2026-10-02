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


# ⚠ **P142 修正**：msprof 的 `op_summary` 真实列名（2026-10-02 实测）是：
#   'Op Name', 'OP Type', 'Task Start Time(us)', 'Task Duration(us)', ...
#   另有**极有价值**的分解列：
#   'aicore_time(us)'（AI Core 占用）、'aic_mte1/mte2_ratio'（内存搬运）、
#   'aic_scalar_ratio'、'aic_mac_ratio'、'cube_utilization(%)'、
#   'Task Wait Time(us)'（**排队等待**）、'aiv_time/aiv_vec_time'（Vector 单元）
# → **列名匹配必须用「子串包含」**，不是精确相等：第一版用精确相等
#   全部落空 → 按位置猜到 col1/col3（那是 'Model ID'/'Stream ID'）→
#   把算子名读成 `4294967295`（Model ID 的值），**结论完全无效且看不出来**。
#教训：解析失败时**必须显式报错并拒绝出结论**，不能「猜了就出表」。
_SUBSTR_DUR = ("task duration", "duration")
_SUBSTR_NAME = ("op name", "kernel name", "op_type", "optype")


def _pick_cols(header: list[str]) -> tuple[int, int, int, str] | None:
    """返回 (name_col, dur_col, start_col, 说明)。找不到返回 None。"""
    low = [h.strip().lower() for h in header]
    dur = next((i for i, h in enumerate(low)
                if any(k in h for k in _SUBSTR_DUR)), None)
    name = next((i for i, h in enumerate(low)
                 if any(k in h for k in _SUBSTR_NAME)), None)
    start = next((i for i, h in enumerate(low)
                  if "start" in h), None)
    if dur is not None and name is not None:
        return name, dur, (start if start is not None else -1), "按列名(子串)"
    return None


# 有价值的分解列（用于「时间去哪了」的判读）
_SUBSTR_AICORE = ("aicore_time",)
_SUBSTR_WAIT = ("task wait time",)
_SUBSTR_MTE1 = ("aic_mte1_ratio",)
_SUBSTR_MTE2 = ("aic_mte2_ratio",)
_SUBSTR_SCALAR = ("aic_scalar_ratio",)
_SUBSTR_MAC = ("aic_mac_ratio",)
_SUBSTR_CUBE = ("cube_utilization",)
# P144：**别只看 aicore_time**。fhz 质疑「Index 是不是测占用率的」——
# 数据显示 `aicore_time = 0` 而 `Task Wait Time` 巨大 →它**根本没占 AI Core**
#，而是在 **Vector（AIV）** 单元上跑。gather 是访存/向量型，
# 跑AIV 才是它的自然归属。第一版只取 aicore，把 0 当「没数据」→ 结论错。
_SUBSTR_AIV = ("aiv_time",)
_SUBSTR_AIVVEC = ("aiv_vec_time",)
_SUBSTR_MTE1T = ("aic_mte1_time",)
_SUBSTR_MTE2T = ("aic_mte2_time",)
_SUBSTR_FIXPIPE = ("aic_fixpipe_time",)


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
            # ⚠⚠ **P142：识别失败就**拒绝出结论**。第一版「按位置猜」，
            #   猜到了 col1/col3（实际是 'Model ID'/'Stream ID'）→ 把算子名读成
            #   `4294967295`，而**表格照样打印、看不出任何异常** ——
            #   这是比「报错」更糟的失败模式。宁可不给结果。
            print("✗ **列名识别失败，拒绝出结论**（不再按位置猜测）。")
            print(f"  实际列名: {header}")
            print("  期望包含 'Op Name' 与 'Task Duration'（可用子串匹配）。")
            print("  若列名确实不同，请把本工具的 _SUBSTR_NAME/_SUBSTR_DUR "
                  "按实际表头调整。")
            return 3
        else:
            name_c, dur_c, start_c, how = picked
            print(f"列名识别: {how} → name=col{name_c}, duration=col{dur_c}")

        # 分解列索引（有则用，无则跳过）
        def _ci(sub):
            return next((i for i, h in enumerate(header)
                         if any(k in h.strip().lower() for k in sub)), -1)
        c_wait = _ci(_SUBSTR_WAIT)
        c_aic = _ci(_SUBSTR_AICORE)
        c_mte1 = _ci(_SUBSTR_MTE1)
        c_mte2 = _ci(_SUBSTR_MTE2)
        c_mac = _ci(_SUBSTR_MAC)
        c_scalar = _ci(_SUBSTR_SCALAR)
        c_cube = _ci(_SUBSTR_CUBE)
        c_aiv = _ci(_SUBSTR_AIV)
        c_aivvec = _ci(_SUBSTR_AIVVEC)
        c_mte1t = _ci(_SUBSTR_MTE1T)
        c_mte2t = _ci(_SUBSTR_MTE2T)
        c_fix = _ci(_SUBSTR_FIXPIPE)

        # 累加：按算子名聚合
        agg: dict[str, list[float]] = defaultdict(list)
        dec: dict[str, list[float]] = defaultdict(
            lambda: [0.0] * 10)     # 0wait 1aicore 2mte1% 3mte2% 4mac 5scalar
                                        # 6aiv 7aiv_vec 8mte1_t 9mte2_t
        cube: list[float] = []

        def _num(row, i):
            if i < 0 or i >= len(row):
                return 0.0
            try:
                return float(row[i])
            except ValueError:
                return 0.0

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
            dec[nm][0] += _num(row, c_wait)
            dec[nm][1] += _num(row, c_aic)
            dec[nm][2] += _num(row, c_mte1)
            dec[nm][3] += _num(row, c_mte2)
            dec[nm][4] += _num(row, c_mac)
            dec[nm][5] += _num(row, c_scalar)
            dec[nm][6] += _num(row, c_aiv)
            dec[nm][7] += _num(row, c_aivvec)
            dec[nm][8] += _num(row, c_mte1t)
            dec[nm][9] += _num(row, c_mte2t)
            if c_cube >= 0:
                cube.append(_num(row, c_cube))
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
    # ── 时间去向：AI Core vs 搬运 vs 排队（msprof 的分解列）──────────────
    if dec and (c_aic >= 0 or c_wait >= 0 or c_mte1 >= 0):
        print("\n" + "=" * 92)
        print("时间去向（msprof 分解列，单位 μs；ratio 类列是 **%**，不是 μs）")
        print("=" * 92)
        hdr = (f"{'算子':<38}{'耗时':>11}{'AICore':>10}{'AIV/Vector':>12}"
               f"{'mte1_t':>9}{'mte2_t':>9}{'排队':>12}")
        print(hdr)
        print("-" * 92)
        for nm, vals in items[:12]:
            s_ = sum(vals)
            d = dec[nm]
            print(f"{nm[:36]:<38}{s_:>11,.0f}{d[1]:>10,.0f}{d[6]:>12,.0f}"
                  f"{d[8]:>9,.0f}{d[9]:>9,.0f}{d[0]:>12,.0f}")
        # 硬件单元归属判读（P144：这是 fhz 质疑引出的关键维度）
        print("\n  硬件单元归属判读：")
        for nm, vals in items[:12]:
            s_ = sum(vals)
            if s_ <= 0:
                continue
            d = dec[nm]
            wa, wv = d[1], d[6]
            mv = d[8] + d[9]
            wait = d[0]
            if wa / s_ > 0.5:
                where = "**AI Core（Cube）** ← 真的在算矩阵"
            elif wv / s_ > 0.3:
                where = "**Vector/AIV** ← 访存/向量型（gather 属此类）"
            elif mv / s_ > 0.3:
                where = "**搬运为主**（mte1/mte2）← memory-bound"
            elif wait > 3 * s_:
                where = ("**几乎全在排队**（wait ≫ 耗时）→ 不是它慢，"
                         "是**没被调度到**")
            else:
                where = "混合/待判读"
            print(f"{nm[:36]:<38}→ {where}")
        tot_d = sum(sum(v) for v in agg.values())
        tot_a = sum(d[1] for d in dec.values())
        tot_w = sum(d[0] for d in dec.values())
        if tot_d > 0:
            print("-" * 92)
            print(f"{'合计':<40}{tot_d:>11,.0f}{tot_a:>11,.0f}")
            print(f"  **AI Core 占比 = {tot_a/tot_d*100:.1f}%**"
                  f"　　**排队等待占比 = {tot_w/tot_d*100:.1f}%**")
            print("  → AI Core 占比高 = 设备**在算**（则问题在真实计算量）")
            print("  → 排队占比高 = 设备**在等资源**（则调度/并发度不足）")
        if cube:
            _cv = [c for c in cube if c > 0]
            if _cv:
                print(f"  cube_utilization(%)：中位 {sorted(_cv)[len(_cv)//2]:.1f}，"
                      f"最大 {max(_cv):.1f}")
    else:
        print("\n（本文件没有 msprof 的分解列，跳过「时间去向」分析）")

    print("\n→ 未覆盖的算子就是我们**没数到**的那26 倍。"
          "看它们的名称与累计耗时，即可判断是"
          " dtype 转换 / 内存分配 / 搬运 / 隐式同步 中的哪一种。")
    if guessed:
        print("\n⚠ 本次**列名是猜的**，结论请对照 `实际列名` 那一行复核。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())