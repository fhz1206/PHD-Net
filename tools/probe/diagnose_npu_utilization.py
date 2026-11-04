#!/usr/bin/env python3
"""NPU「假占用」诊断：区分**指标虚高**与**真实计算利用率**，并给出归因。

## 问题

`npu-smi` / 设备监控常显示占用 ~90%+，但**算力/吞吐并未达到相应水平**。
原因是「占用」这个指标本身有多个口径，混用必然误判：

| 指标 | 它实际测什么 | 能不能代表算力 |
|---|---|---|
| 设备占用（npu-smi / 上下文活跃） | 进程在设备上、任务队列非空 | ❌ 不能 |
| `aicore_time`（Cube/MAC） | 矩阵计算单元的**有效执行**时间 | ✅ 最接近 |
| `aiv_time`（Vector） | 向量/访存型算子的执行时间 | ⚠ 在算，但效率低 |
| `Task Duration` | 含**排队等待**的墙上时间 | ❌ 不能 |
| `Task Wait Time` | 纯等待（没被调度到） | ❌ 恰恰是空转证据 |

**本项目已定案的读数**（P144 msprof，昇腾 1b 档）：
- `aicore_time = 0`（**真的 0**，多指标交叉验证过，不是缺数据）；
- `aiv_time = 99.6%` → `Index` 算子跑在 **Vector 单元**
  （算子名 `aclnnIndex_IndexAiCore_Index` 里的 "AiCore" 只是 CANN **分类标签**）；
- `Task Wait` 是自身耗时的 **6.8×** →「算子慢」是错误表述，
  真因是**没被调度到**（每步只提交 ~12 个小 kernel，NPU 大部分时间空转等 CPU）；
- 同一份访存量：Vector `Index` 8.1 GB/s vs Cube `BatchMatMul` 38 GB/s（**差 4.7×**）；
- 读出 7.83 ms 是带宽下界 0.300 ms 的 **26.1×** → 算子效率问题，不是带宽饱和。

## 本脚本做什么

1. 读 msprof 导出的 `op_summary` / timeline CSV；
2. 汇总 `aicore_time` / `aiv_time` / `wait` / `duration` 四个口径；
3. **自动判定**「假占用」与否，并给出**判断依据**（哪一列支持/反驳）；
4. 按证据强度列出归因（等 CPU / Host 下发 / Vector 低效 / 同步等待 / H2D-D2H / 算子切分）；
5. 输出本项目**可直接执行的 A/B 命令**与期望方向。

⚠ **列名用子串匹配**（P142 教训：精确匹配会全部落空；按位置猜更糟，
   会把 `Model ID` 读成算子名却照样打印表格 → 比报错更危险）。
   识别失败时**拒绝出结论**（返回码 3）。

## 用法

```bash
# 服务器上采集
bash tools/scripts/prof_readout_msprof.sh

# 判读（本机也行，纯文本处理）
python tools/probe/diagnose_npu_utilization.py outputs/prof/msprof_readout

# 指定单个 CSV + 训练日志（日志用于给出端到端 ms/tok）
python tools/probe/diagnose_npu_utilization.py path/to/op_summary.csv \
    --log train_1b_1b_pretrain_20261101-140007.log

# 只输出机器可读的 JSON（供 CI）
python tools/probe/diagnose_npu_utilization.py <csv> --json
```

判据与红线见 `docs/PHD-Net_NPU利用率诊断与优化_P214.md`。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

# msprof 列名子串（不同版本列名有差异 → 子串匹配）
_SUBSTR = {
    "name": ["op name", "op_name", "kernel name", "算子名"],
    "dur": ["task duration", "duration", "耗时"],
    "wait": ["task wait", "wait time", "等待"],
    "aicore": ["aicore_time", "aic_time", "aicore time"],
    "aiv": ["aiv_time", "aiv_vec_time", "vector_time"],
    "cube": ["cube_utilization", "cube util"],
    "mte": ["mte1", "mte2", "mte_ratio"],
    "start": ["task start time", "start time"],
}


def _find_csv(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    out: list[str] = []
    for pat in ("**/*.csv", "**/*.timeline"):
        out += glob.glob(os.path.join(path, pat), recursive=True)
    # 优先 op_summary（含硬件单元分解列）
    out.sort(key=lambda p: (("op_summary" not in os.path.basename(p).lower()),
                            -os.path.getsize(p)))
    return out


def _col(header: list[str], keys: list[str]) -> int:
    for i, h in enumerate(header):
        hl = h.strip().lower().replace(" ", "")
        if any(k.replace(" ", "") in hl for k in keys):
            return i
    return -1


def _num(row, i: int) -> float:
    if i < 0 or i >= len(row):
        return 0.0
    try:
        return float(str(row[i]).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def analyse(path: str) -> dict:
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rdr = csv.reader(fh)
        header = next(rdr, None)
        if not header:
            return {"error": "空文件/无表头"}
        idx = {k: _col(header, v) for k, v in _SUBSTR.items()}

    if idx["name"] < 0 or idx["dur"] < 0:
        return {"error": "列名识别失败，拒绝出结论", "header": header,
                "note": "按位置猜测会把 Model ID 读成算子名却照样打印表格 "
                        "（P142 教训）—— 宁可不给结果"}

    agg: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "dur": 0.0, "wait": 0.0, "aicore": 0.0,
                 "aiv": 0.0, "mte": 0.0})
    total = {"n": 0, "dur": 0.0, "wait": 0.0, "aicore": 0.0, "aiv": 0.0}
    cube_vals: list[float] = []

    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rdr = csv.reader(fh)
        next(rdr, None)
        for row in rdr:
            if len(row) <= max(idx["name"], idx["dur"]):
                continue
            nm = row[idx["name"]].strip()
            if not nm:
                continue
            d = _num(row, idx["dur"])
            rec = agg[nm]
            rec["n"] += 1
            rec["dur"] += d
            rec["wait"] += _num(row, idx["wait"])
            rec["aicore"] += _num(row, idx["aicore"])
            rec["aiv"] += _num(row, idx["aiv"])
            rec["mte"] += (_num(row, idx["mte"]) if idx["mte"] >= 0 else 0.0)
            total["n"] += 1
            total["dur"] += d
            total["wait"] += _num(row, idx["wait"])
            total["aicore"] += _num(row, idx["aicore"])
            total["aiv"] += _num(row, idx["aiv"])
            if idx["cube"] >= 0:
                cv = _num(row, idx["cube"])
                if cv > 0:
                    cube_vals.append(cv)

    d = total["dur"] or 1.0
    metrics = {
        "kernel_count": total["n"],
        "total_duration_us": total["dur"],
        "total_wait_us": total["wait"],
        "aicore_time_us": total["aicore"],
        "aiv_time_us": total["aiv"],
        "aicore_ratio": total["aicore"] / d,
        "aiv_ratio": total["aiv"] / d,
        "wait_ratio": total["wait"] / d,
        "wait_over_duration": (total["wait"] / total["dur"])
                              if total["dur"] else 0.0,
        "cube_utilization_mean": (sum(cube_vals) / len(cube_vals))
                                 if cube_vals else None,
        "has_hardware_columns": idx["aicore"] >= 0 or idx["aiv"] >= 0,
    }
    top = sorted(agg.items(), key=lambda kv: -kv[1]["dur"])[:12]
    metrics["top_ops"] = [
        {"name": nm, "count": v["n"], "duration_us": round(v["dur"], 1),
         "wait_us": round(v["wait"], 1), "aicore_us": round(v["aicore"], 1),
         "aiv_us": round(v["aiv"], 1),
         "share": round(v["dur"] / d, 4)}
        for nm, v in top
    ]
    return metrics


def verdict(m: dict) -> dict:
    """把读数翻译成「假占用」判定 + 归因（带判断依据）。"""
    if "error" in m:
        return {"verdict": "无法判定", "reason": m["error"]}

    reasons: list[str] = []
    causes: list[tuple[str, str, str]] = []   # (原因, 证据, 期望方向)
    fake = False

    # ① 硬件单元分解列缺失 → 拒绝下结论（P142）
    if not m["has_hardware_columns"]:
        return {"verdict": "无法判定",
                "reason": "CSV 缺 aicore_time / aiv_time 列，"
                          "无法区分「设备占用」与「算力占用」",
                "note": "需 msprof op_summary（不是纯 timeline）"}

    # ② aicore 占比低 = 有效矩阵计算几乎没跑
    if m["aicore_ratio"] < 0.10:
        fake = True
        reasons.append(
            f"aicore_time 占 Task Duration 仅 {m['aicore_ratio']*100:.1f}% "
            f"（{m['aicore_time_us']:.0f} / {m['total_duration_us']:.0f} µs）"
            " → 有效矩阵计算时间近乎为 0")
    else:
        reasons.append(
            f"aicore_time 占比 {m['aicore_ratio']*100:.1f}% → 有实质矩阵计算")

    # ③ Vector 单元在跑但效率低
    if m["aiv_ratio"] > 0.30 and m["aicore_ratio"] < 0.10:
        fake = True
        reasons.append(
            f"aiv_time 占比 {m['aiv_ratio']*100:.1f}% 而 aicore 近乎 0 → "
            "算子跑在 **Vector 单元**（访存/索引型），不是 Cube/MAC。"
            "⚠ 算子名里的 'AiCore' 只是 CANN 分类标签，不代表硬件归属")
        causes.append((
            "算子切分与融合不足 / Vector 低效",
            "同访存量：Vector `Index` 8.1 GB/s vs Cube `BatchMatMul` 38 GB/s "
            "（差 4.7×）；`Index` 占 ~89%",
            "把稀疏 gather 从 Vector 索引改成 Cube 友好的 SpMM / 合批，"
            "或启用 `--beta p0,p1` Graph 融合"))

    # ④ wait ≫ duration = 没被调度到（不是算子慢）
    if m["wait_over_duration"] > 2.0:
        fake = True
        reasons.append(
            f"Task Wait / Task Duration = {m['wait_over_duration']:.1f}× → "
            "**不是算子慢，是没被调度到**：每步只提交少量小 kernel，"
            "设备大部分时间空转等 Host 下发")
        causes.append((
            "Host 侧下发 / kernel launch 开销 + 队列背压",
            f"wait {m['wait_over_duration']:.1f}× 于自身耗时；"
            "CANN 上 launch ~50–200 µs/kernel，每步 ~5–12 个 eager 算子",
            "启用 `--beta p0,p1`（Graph Capture 消除 launch）；"
            "或重评稀疏路径 `--torch-compile`（默认 False）"))
        causes.append((
            "CPU→NPU 串行依赖 / 等 CPU",
            "CPU 侧约 6.5 ms + NPU 读出 3.5 ms 相加 ≈ 端到端 10.8 ms，"
            "无隐含重叠（P192 已核对：代码里没有阻塞同步，是队列背压）",
            "先做 CPU 侧加速（P214 的 LTM 哈希查找已落地 1.83×）；"
            "线程 A/B 用 `tools/bench/bench_server_threads.py`"))

    # ⑤ 同步点 / 拷贝
    causes.append((
        "同步等待与 H2D/D2H",
        "代码核查：`forward_dev` 返回设备张量、零同步；"
        "`nll_sync_every=8` 把 `.item()` 摊到 1/8 步；"
        "`--beta p2` 双流（H2D 独立流 + nll 独立流）已实现",
        "跑 `--nll-sync-every 1 vs 8` A/B："
        "若 1 明显更慢 → 还有隐藏同步点；若相同 → 无隐藏同步"))

    return {
        "verdict": "假占用（指标虚高）" if fake else "真实计算为主",
        "evidence": reasons,
        "causes": [{"cause": c, "evidence": e, "action": a}
                   for c, e, a in causes],
        "caveat": "优化目标是**把 aicore_time 提上去 / 把 wait 降下来**，"
                  "不是把设备占用率推得更高；占用下降但 ms/tok 下降是好事",
    }


def _log_summary(path: str) -> dict:
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return {}
    ms = [float(x) for x in re.findall(r"([\d.]+) ms/tok\s+elapsed", txt)]
    ro = [float(x) for x in re.findall(r"readout\s+([\d.]+) ms/tok", txt)]
    out = {}
    if ms:
        out["end_to_end_ms_per_tok_last"] = ms[-1]
    if ro:
        out["readout_ms_per_tok_last"] = ro[-1]
    if ms and ro:
        out["cpu_side_approx"] = round(ms[-1] - ro[-1], 2)
        out["npu_share"] = round(ro[-1] / ms[-1], 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="msprof CSV 文件或目录")
    ap.add_argument("--log", default="", help="训练日志（给出端到端 ms/tok）")
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    args = ap.parse_args()

    files = _find_csv(args.target)
    if not files:
        print(f"没找到 CSV：{args.target}\n"
              "先跑 `bash tools/scripts/prof_readout_msprof.sh` 采集。")
        return 2
    path = files[0]
    m = analyse(path)
    v = verdict(m)
    log = _log_summary(args.log) if args.log else {}

    if args.json:
        print(json.dumps({"source": path, "metrics": m, "verdict": v,
                          "log": log}, ensure_ascii=False, indent=2))
        return 0 if "error" not in m else 3

    print("=" * 80)
    print("NPU「假占用」诊断 —— 指标虚高 vs 真实计算利用率")
    print("=" * 80)
    print(f"  数据源: {path}")
    if "error" in m:
        print(f"\n✗ {m['error']}")
        if "header" in m:
            print(f"  实际列名: {m['header']}")
        return 3

    print(f"\n  kernel 数 {m['kernel_count']}  |  "
          f"Task Duration {m['total_duration_us']:.0f} µs  |  "
          f"Task Wait {m['total_wait_us']:.0f} µs "
          f"({m['wait_over_duration']:.1f}×)")
    print(f"  aicore_time {m['aicore_time_us']:.0f} µs "
          f"({m['aicore_ratio']*100:.1f}%)  |  "
          f"aiv_time {m['aiv_time_us']:.0f} µs "
          f"({m['aiv_ratio']*100:.1f}%)")
    if m["cube_utilization_mean"] is not None:
        print(f"  cube_utilization 均值 {m['cube_utilization_mean']:.2f}%")

    print("\n  Top 算子（按 Task Duration）")
    print(f"  {'算子':<44} {'次数':>6} {'耗时µs':>10} {'占比':>7} {'等待µs':>10}")
    for o in m["top_ops"]:
        print(f"  {o['name'][:44]:<44} {o['count']:>6} "
              f"{o['duration_us']:>10.0f} {o['share']*100:>6.1f}% "
              f"{o['wait_us']:>10.0f}")

    print("\n" + "-" * 80)
    print(f"  判定：{v['verdict']}")
    print("  判断依据：")
    for r in v.get("evidence", []):
        print(f"    · {r}")
    print("\n  归因（按证据强度）与建议动作：")
    for i, c in enumerate(v.get("causes", []), 1):
        print(f"    {i}. {c['cause']}")
        print(f"       证据: {c['evidence']}")
        print(f"       动作: {c['action']}")
    print(f"\n  ⚠ {v.get('caveat', '')}")

    if log:
        print("\n  训练日志摘要：")
        for k, val in log.items():
            print(f"    {k}: {val}")

    print("\n  服务器 A/B（按项目判据，PPL 一律用 probe_PPL）：")
    print("    bash  : python tools/bench/bench_server_threads.py --threads 8,16,32")
    print("    beta  : --beta p0,p1,p2 --cython-kernels auto")
    print("    同步  : --nll-sync-every 1  vs  8")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
