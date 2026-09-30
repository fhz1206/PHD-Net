#!/usr/bin/env python3
"""云端（昇腾）性能重测：跑一组配置，产出**机器可读**的 JSON 基线。

fhz 2026-09-30：「性能数字全部重新测试（云端）」。本脚本就是那把尺子：
一次跑完所有需要测的组合，输出 `outputs/experiments/server_bench_<ts>.json`，
再由 `tools/apply_bench.py` 回填进《性能评估与迭代方案》（性能数字的唯一出处）。

**为什么要有这个工具**：今天的性能数字散在多份文档、且大多是「本机 x86 推算 +
服务器零散日志」拼出来的。规范要求「一个事实只在一处权威处写」，那就必须有
**可复算的采集端**（本脚本）与**唯一的回填端**（apply_bench.py）。

## 用法（服务器上）

```bash
# 快速档（约 15 分钟）：验证精度/线程/核选择的影响
python tools/bench_server.py --preset 1b --steps 3000 --profile quick

# 完整档（约 2 小时）：含 30B 容量档与步时趋势
python tools/bench_server.py --preset 1b --steps 20000 --profile full
```

跑完把 JSON 传回（或直接说「跑完了」，我读 `outputs/experiments/` 下最新的），
再执行：

```bash
python tools/apply_bench.py outputs/experiments/server_bench_<ts>.json
```

## 采集内容
- 每组配置：总 ms/tok、九段 + 主循环三段分解、readout ms/tok 与占比；
- 步时趋势（每 500 token 一点）→ 判断是否仍在超线性恶化；
- 稳态遥测：CPU 核数、NPU%/HBM、CS/s、GC 对象数；
- 环境指纹：设备、torch/numpy/numba 版本、CPU 核数、commit。

**纪律**：只测**真实存在的配置**（不再推断）；每组都记 argv 与 commit，
避免「数字漂移但不知道测的是哪一版」。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# 采集哪些配置：name -> 额外 CLI（None = 基线）
PROFILES: dict[str, list[tuple[str, list[str]]]] = {
    "quick": [
        ("baseline", []),
        ("m2_fused", ["--m2-kernel", "fused"]),
        ("encoder_fp32", ["--encoder-dtype", "fp32"]),
        ("readout_fp16", ["--readout-dtype", "fp16"]),
        ("readout_fp32", ["--readout-dtype", "fp32"]),
        ("no_omp_bind", ["--no-omp-proc-bind"]),
    ],
    "full": [
        ("baseline", []),
        ("m2_fused", ["--m2-kernel", "fused"]),
        ("encoder_fp32", ["--encoder-dtype", "fp32"]),
        ("readout_fp16", ["--readout-dtype", "fp16"]),
        ("readout_fp32", ["--readout-dtype", "fp32"]),
        ("no_omp_bind", ["--no-omp-proc-bind"]),
        ("numba_threads_16", ["--numba-threads", "16"]),
        ("numba_threads_1", ["--numba-threads", "1"]),
    ],
}

LOG_RE = re.compile(
    r"^\s*token\s+([\d,]+)\s+sliding PPL\s+([\d.]+)\s+([\d.]+) ms/tok\s+elapsed\s+"
    r"([\d.]+) min\s+\|\s+readout\s+([\d.]+) ms/tok \(([^)]*)\)\s*"
    r"(?:.*?([\d.]+)% of total)?")
LOOP_RE = re.compile(r"loop: (.+)$")
SEG_RE = re.compile(r"segments: (.+)$")
TEL_RE = re.compile(r"CPU (\d+)%.*?proc ([\d.]+)核.*?NPU/GPU (\S+).*?HBM (\S+)"
                    r".*?CS/s (\d+).*?GC ([\d.]+)M/gen2 (\d+)")


def _fingerprint(commit: str) -> dict:
    info = {
        "commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:                                        # noqa: BLE001
        pass
    try:
        import numba
        info["numba"] = numba.__version__
    except Exception:                                        # noqa: BLE001
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_npu"] = getattr(torch, "__version__", None)
    except Exception:                                        # noqa: BLE001
        pass
    return info


def run_one(args, name: str, extra: list[str]) -> dict:
    """跑一组配置并解析日志。子进程独立，避免互相污染 numba/线程状态。"""
    cmd = [sys.executable, str(_ROOT / "train_1b" / "train.py"),
           "--preset", args.preset, "--data", "pretrain",
           "--remote-data", "--remote-fraction", str(args.fraction),
           "--resume", "--step-profiling",
           "--tokens", str(args.steps), "--log-every", str(args.log_every)] + extra
    print(f"\n=== [{name}] {' '.join(extra) or '(baseline)'} ===", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    out = {"name": name, "argv": cmd[1:], "wall_min": round(wall / 60, 2),
           "returncode": proc.returncode, "points": [], "segments": {},
           "loop": {}, "telemetry": None, "fallback": None}
    if proc.returncode != 0:
        out["error"] = (proc.stderr or proc.stdout)[-800:]
        print(f"  失败（rc={proc.returncode}）", flush=True)
        return out
    fb = re.findall(r"\[readout\].*(回落|fallback).*", proc.stdout)
    if fb:
        out["fallback"] = fb[0][:200]
    for line in proc.stdout.splitlines():
        m = LOG_RE.match(line)
        if m:
            tok, ppl, ms, mins, ro_ms, ro_tag = m.group(1), m.group(2), \
                float(m.group(3)), m.group(4), float(m.group(5)), m.group(6)
            out["points"].append({
                "token": int(tok.replace(",", "")), "ppl": float(ppl),
                "ms_per_tok": float(ms), "elapsed_min": float(mins),
                "readout_ms": ro_ms, "readout_tag": ro_tag,
            })
        m2 = SEG_RE.search(line)
        if m2:
            out["segments"] = {k: float(v) for k, v in
                               (p.split() for p in m2.group(1).split("  ") if " " in p)}
        m3 = LOOP_RE.search(line)
        if m3:
            out["loop"] = {k: float(v) for k, v in
                           (p.split() for p in m3.group(1).split("  ") if " " in p)}
        m4 = TEL_RE.search(line)
        if m4:
            out["telemetry"] = {
                "cpu_pct": int(m4.group(1)), "cores": float(m4.group(2)),
                "acc_util": m4.group(3), "hbm": m4.group(4),
                "ctx_switches": int(m4.group(5)),
                "gc_objs_M": float(m4.group(6)), "gc_gen2": int(m4.group(7)),
            }
    if out["points"]:
        tail = out["points"][-max(1, len(out["points"]) // 5):]     # 后 20% 均值
        out["summary"] = {
            "ms_per_tok_steady": round(sum(p["ms_per_tok"] for p in tail) / len(tail), 2),
            "readout_ms_steady": round(sum(p["readout_ms"] for p in tail) / len(tail), 3),
            "ppl_last": tail[-1]["ppl"],
            "trend_ms_per_tok": [p["ms_per_tok"] for p in out["points"]][::max(1, len(out["points"]) // 8)],
        }
        print(f"  稳态 {out['summary']['ms_per_tok_steady']} ms/tok | "
              f"readout {out['summary']['readout_ms_steady']} | "
              f"PPL {out['summary']['ppl_last']}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="1b", choices=["1b", "30b", "4m"])
    ap.add_argument("--steps", type=int, default=3000, help="每组配置的 token 预算")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--fraction", type=float, default=0.3)
    ap.add_argument("--profile", default="quick", choices=sorted(PROFILES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            cwd=str(_ROOT), capture_output=True, text=True
                            ).stdout.strip() or "unknown"
    runs = []
    for name, extra in PROFILES[args.profile]:
        try:
            runs.append(run_one(args, name, extra))
        except KeyboardInterrupt:
            print("\n中断，已完成的结果仍会写出", flush=True)
            break
    outdir = _ROOT / "outputs" / "experiments"
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = Path(args.out) if args.out else outdir / f"server_bench_{ts}.json"
    path.write_text(json.dumps(
        {"env": _fingerprint(commit), "args": vars(args), "runs": runs},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n结果已写入 {path}")
    print("把它交给 AI（或说「跑完了」）即可回填文档："
          f"\n  python tools/apply_bench.py {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
