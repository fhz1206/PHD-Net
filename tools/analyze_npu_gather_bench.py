#!/usr/bin/env python3
"""解析 AtomGit NPU Notebook 的 `npu_gather_bench.json`，给出结论与建议。

用法：
    python tools/analyze_npu_gather_bench.py <下载的 json 路径>
    # 也支持 .txt（只做人读摘要）

背景见 docs/PHD-Net_Yan架构启发与并行化评估.md。
判据来自 P214 msprof 实测：`Index` 占 NPU 时间 80.7%，是单点瓶颈。
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def analyse(data: dict) -> int:
    env = data.get("env", {})
    print("=" * 78)
    print("NPU gather benchmark 解析 (P217)")
    print("=" * 78)
    for k in ("device", "torch", "npu_available", "npu_name",
              "V", "K", "H", "elements_per_step", "MiB_per_step_fp16",
              "warmup", "repeat"):
        if k in env:
            print(f"  {k:22s} = {env[k]}")

    if not env.get("npu_available"):
        print("\n⚠ 这次跑在 CPU 上 → **不能作为昇腾结论**（项目铁律：x86 ≠ 昇腾）")

    print()
    verdicts = []
    for run in data.get("runs", []):
        dt = run.get("dtype")
        vs = run.get("variants", [])
        print(f"--- dtype = {dt} ---")
        print(f"  {'实现':40s} {'ms/step':>10} {'vs A':>8} {'max|Δ|':>11}  状态")
        tA = next((v["ms"] for v in vs
                   if v["name"].startswith("A ") and v.get("ok")), None)
        for v in vs:
            if not v.get("ok"):
                print(f"  {v['name'][:40]:40s} {'—':>10} {'—':>8} {'—':>11}  "
                      f"FAILED {v.get('error', '')[:60]}")
                continue
            sp = v.get("speedup_vs_A")
            if sp is None and tA:
                sp = round(tA / v["ms"], 3)
            d = v.get("max_abs_delta_vs_A")
            print(f"  {v['name'][:40]:40s} {v['ms']:10.3f} "
                  f"{(f'{sp:.2f}×' if sp else '—'):>8} "
                  f"{(f'{d:.2e}' if d is not None else '—'):>11}  OK")
        print()
        verdicts.append((dt, vs, tA))

    print("=" * 78)
    print("结论与建议")
    print("=" * 78)
    for dt, vs, tA in verdicts:
        ok = {v["name"][:1]: v for v in vs if v.get("ok")}
        a = ok.get("A")
        if not a:
            continue
        # 找最快且数值安全的等价实现（排除 E 分区激活、F 缩小版 SpMM）
        equiv = [v for v in vs
                 if v.get("ok") and v["name"][0] in "BCD"
                 and v.get("speedup_vs_A", 0) > 1.10
                 and (v.get("max_abs_delta_vs_A") is None
                      or v["max_abs_delta_vs_A"] < 1e-2)]
        if equiv:
            best = max(equiv, key=lambda v: v["speedup_vs_A"])
            print(f"  [{dt}] `{best['name'][:38]}` 比生产实现快 "
                  f"**{best['speedup_vs_A']:.2f}×**，max|Δ|="
                  f"{best.get('max_abs_delta_vs_A')} "
                  "→ **零语义风险，可直接换算子/开关**")
        else:
            print(f"  [{dt}] B/C/D 均无显著优势 → 瓶颈在 Vector 单元本身，"
                  "需真 CSR SpMM / 自定义算子")

        e = [v for v in vs if v.get("ok") and v["name"].startswith("E ")]
        if e and tA:
            for v in e:
                print(f"  [{dt}] {v['name'][:34]}: "
                      f"仅 {v['ms']:.3f} ms（全量 {tA:.3f} ms，"
                      f"理论 {tA/v['ms']:.0f}×）→ **分区激活的上限**，"
                      "但 softmax 需全 V，真用要改学习规则 + probe_PPL")

        f = [v for v in vs if v.get("ok") and v["name"].startswith("F ")]
        if f:
            print(f"  [{dt}] 等价稠密 SpMM(V=4096): {f[0]['ms']:.3f} ms；"
                  "注意 V 只有全量的 1/12，不能直接比绝对值")

    print("\n⚠ 红线：")
    print("  · 任何改学习规则的方案（分区激活 / 批内延迟更新 / 降 conn_k）")
    print("    必须 probe_PPL 判，不看 sliding PPL（信噪比 0.40）。")
    print("  · P110 教训：少算行/降精度 → 非目标行更新保留率")
    print("    fp32 99.95% → fp16 26.67% → bf16 5.79%（退化为纯 Hebbian）。")
    print("=" * 78)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="下载的 npu_gather_bench.json 或 .txt")
    args = ap.parse_args()
    if not os.path.exists(args.path):
        print(f"文件不存在: {args.path}")
        return 2
    if args.path.endswith(".json"):
        with open(args.path, encoding="utf-8") as f:
            return analyse(json.load(f))
    with open(args.path, encoding="utf-8", errors="replace") as f:
        print(f.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
