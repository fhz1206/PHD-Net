#!/usr/bin/env python3
"""服务器（昇腾 aarch64）**端到端 numba 线程 A/B** —— 回答「32 线程到底快不快」。

## 为什么需要这个脚本

`--numba-threads` 的默认值是 **16**（P204 从 8 改成 16），但：

- `train/train.py` 的 help 明确写着「本机 x86 实测 16/32 因 fork/join 屏障
  （4.7→24.7→64.3 µs/区）负收益；**但本机 ≠ 昇腾**（项目已有四次方向相反）
  → **服务器 A/B 定论**：`--numba-threads 8/16/32`，判据是 `segments(win)` 的
  M2_infer + 端到端 ms/tok + CS/s」；
- `docs/PHD-Net_性能评估与迭代方案.md` §P204 行同样记着「**服务器须 A/B**」。

**也就是说：这条 A/B 从 P204 至今从未做过。** 服务器日志（191 核）显示
`[parallel] numba prange threads = 16 (cap 16)` —— 191 核只用了 16 个
numba 线程，且没人验证过 32 是否更好。

本脚本**只做测量、不改任何默认**：跑完给出一张判据表，由人决定要不要改默认。

## 用法（服务器上）

```bash
# 标准 A/B：8 / 16 / 32 线程，各跑一轮
python tools/bench/bench_server_threads.py --threads 8,16,32

# 更细：1,2,4,8,16,32,64（含 1 线程基线，看并行加速比曲线）
python tools/bench/bench_server_threads.py --threads 1,2,4,8,16,32,64

# 与生产同参（1b 档 + 远程数据），跑久一点让稳态段出现
python tools/bench/bench_server_threads.py --threads 8,16,32 \
    --preset 1b --data pretrain --remote-data --tokens 20000

# 只看 CPU 侧各段（排除 NPU 读出的抖动）
python tools/bench/bench_server_threads.py --threads 8,16,32 --segments-only
```

## 判据（与 README §4 / 性能文档一致）

1. **端到端 `ms/tok`** —— 主判据（取各轮末窗的中位，避免 JIT/预热噪声）；
2. **`segments(win)` 的 CPU 侧各段** —— 定位是「哪一段变快/变慢」；
3. **CS/s（context switch）** —— 线程空转的信号（`/proc/self/status`
   的 `voluntary_ctxt_switches`）；线程越多若 CS/s 暴涨 → 屏障空转在主导；
4. ⚠ **PPL 不作判据**（README §4：训练流 sliding PPL 信噪比仅 0.40，
   任何 <20% 的差异都测不出来）。本脚本**只报不判**。

## 输出

- 控制台：一张对齐的判据表；
- `outputs/experiments/server_threads_<ts>.json`：机器可读的原始数据。

⚠ **本脚本的数字只在「同一台机器、同一 commit、同一数据、同一 token 预算」
下可比**（README §4 铁律）。跨机器的绝对值无意义。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _parse_segments(line: str) -> dict[str, float]:
    """解析 `segments(win): other 5.35  M6_replay 1.73 ... ms/tok` → {段: 值}。

    ⚠ 段名与数值之间是**空格**分隔（不是 `=`）——与 `segments(cum)` 同格式。
    """
    out: dict[str, float] = {}
    toks = line.split()
    i = 0
    while i + 1 < len(toks):
        try:
            out[toks[i]] = float(toks[i + 1])
            i += 2
        except ValueError:
            i += 1
    return out


def _ctx_switches() -> tuple[int, int]:
    """(voluntary, involuntary) context switches —— 线程空转的信号。"""
    try:
        txt = Path("/proc/self/status").read_text()
    except OSError:
        return (0, 0)
    vol = inv = 0
    for ln in txt.splitlines():
        if ln.startswith("voluntary_ctxt_switches:"):
            vol = int(ln.split()[1])
        elif ln.startswith("nonvoluntary_ctxt_switches:"):
            inv = int(ln.split()[1])
    return (vol, inv)


def _python_exe() -> str:
    """返回**带 numpy 的**解释器路径 —— **复用 `phdnet/_python_exe.py`**。

    ⚠ P215 审计修复：本函数原是 `run_gates.py` 那套探测的**独立副本**，
      但漏了后来加的加固项：① 扫描 `%LOCALAPPDATA%/Programs/Python/...`
      等常见安装位置；② **把 `py.exe` 启动器降级到兜底**（它的解释器选择
      依赖被运行文件的路径/关联，实测 `py -c` 与 `py <脚本>` 会挑不同解释器）。
      用裸 `python`（WorkBuddy 内置 3.13 无 numpy）启动本脚本时，
      旧版会选中 `C:\\Windows\\py.EXE` → 子进程可能全灭。
      现统一走单一实现，避免两份逻辑漂移。

    ⚠ 按**文件路径**加载，不 `import phdnet._python_exe`：
      `phdnet/__init__.py` 会连带 import numpy/numba，在没装 numpy 的
      解释器里直接炸 —— 而本模块自己必须能在那种环境里跑。
    """
    import importlib.util
    src = _ROOT / "phdnet" / "_python_exe.py"
    spec = importlib.util.spec_from_file_location("_phdnet_python_exe", src)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.python_exe()


def run_once(threads: int, args: argparse.Namespace) -> dict:
    """跑一轮训练，返回该轮的判据数据。"""
    cmd = [
        _python_exe(), str(_ROOT / "train" / "train.py"),
        "--preset", args.preset,
        "--data", args.data,
        "--tokens", str(args.tokens),
        "--numba-threads", str(threads),
        "--step-profiling",
    ]
    if args.remote_data:
        cmd.append("--remote-data")
    if args.lang:
        cmd += ["--lang", args.lang]
    if args.extra:
        cmd += args.extra.split()

    print(f"\n=== numba-threads={threads} ===")
    print("  " + " ".join(cmd[1:]))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=args.timeout)
    wall = time.perf_counter() - t0
    out = proc.stdout or ""

    # 末窗分段（取最后 3 个滑窗的中位，避免单窗抖动）
    win_lines = re.findall(r"segments\(win\): (.+?) ms/tok", out)
    win_all = [_parse_segments(ln) for ln in win_lines]
    win_tail = win_all[-args.windows:] if win_all else []
    seg_med: dict[str, float] = {}
    if win_tail:
        keys = set().union(*[set(d) for d in win_tail])
        for k in keys:
            vals = [d[k] for d in win_tail if k in d]
            if vals:
                seg_med[k] = statistics.median(vals)

    # 端到端 ms/tok（取日志里的所有 [METRIC] 行的中位）
    metric = re.findall(r"ms_per_token=([0-9.]+)", out)
    ms_vals = [float(v) for v in metric]
    ms_med = statistics.median(ms_vals) if ms_vals else float("nan")

    # 读出段（NPU）
    ro = re.findall(r"readout\s+([0-9.]+) ms/tok", out)
    ro_med = statistics.median([float(v) for v in ro]) if ro else float("nan")

    # 实际生效的 numba 线程数（训练进程自己打印的）
    eff = re.findall(r"numba prange threads = (\d+)", out)
    eff_n = int(eff[-1]) if eff else -1

    # 最终 PPL（**只报不判** —— README §4：sliding PPL 信噪比仅 0.40）
    ppl = re.findall(r"final_ppl=([0-9.]+)", out)
    ppl_f = float(ppl[-1]) if ppl else float("nan")

    rec = {
        "threads_requested": threads,
        "threads_effective": eff_n,
        "ms_per_token": ms_med,
        "readout_ms": ro_med,
        "segments": seg_med,
        "final_ppl": ppl_f,
        "wall_s": wall,
        "returncode": proc.returncode,
        "stderr_tail": (proc.stderr or "")[-400:],
    }
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="8,16,32",
                    help="逗号分隔的 numba 线程数（默认 8,16,32）")
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--data", default="cc_traces")
    ap.add_argument("--tokens", type=int, default=6000)
    ap.add_argument("--lang", default="")
    ap.add_argument("--remote-data", action="store_true")
    ap.add_argument("--extra", default="",
                    help="额外透传给 train.py 的参数（空格分隔）")
    ap.add_argument("--windows", type=int, default=3,
                    help="取末尾几个滑窗做中位（默认 3）")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--segments-only", action="store_true",
                    help="只打印分段表")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ths = [int(t) for t in args.threads.split(",") if t.strip()]
    if not ths:
        print("--threads 为空", file=sys.stderr)
        return 2

    # ⚠ `segments(win)` 是 **500 步滑窗**（train.py 的打印间隔）→ token 数
    #   不足时**一次都不会打印** → 分段判据全空（ms/tok 仍在）。
    #   故要求 ≥ 3 个滑窗 = 1500 token 起。
    if args.tokens < 1500:
        print(f"⚠ --tokens={args.tokens} 太小：`segments(win)` 是 500 步滑窗，"
              f"需 ≥1500 token（3 个滑窗）才有分段判据。自动提到 1500。",
              file=sys.stderr)
        args.tokens = 1500

    print("=" * 78)
    print("服务器 numba 线程 A/B（端到端）")
    print(f"  机器: {os.uname().nodename if hasattr(os, 'uname') else '?'}"
          f" | CPU 核数: {os.cpu_count()}")
    print(f"  参数: preset={args.preset} data={args.data} "
          f"tokens={args.tokens} threads={ths}")
    print("  ⚠ PPL 只报不判（README §4：sliding PPL 信噪比仅 0.40）")
    print("=" * 78)

    records = []
    for t in ths:
        try:
            records.append(run_once(t, args))
        except subprocess.TimeoutExpired:
            print(f"  ⚠ threads={t} 超时（{args.timeout}s），跳过")
            records.append({"threads_requested": t, "timeout": True})

    # ⚠ P216 修复：原判据是 `returncode == 0`。但昇腾服务器上**训练成功完成
    #   后进程退出时会崩**（`Fatal Python error: PyGILState_Release ...
    #   Python runtime state: finalizing`，returncode = **-6 = SIGABRT**）——
    #   实测 2026-11-07 `--numba-threads 32 --tokens 20000`：20,000 token 跑完、
    #   `METRIC ... ms_per_token=9.97 final_ppl=9572.690`、model 与 checkpoint
    #   都已保存，只是 finalize 阶段 abort。
    #   → 判据改为「**有有效数据**」（解析到 ms/tok 或 segments），
    #     退出码非 0 但数据完整时**单独标注**，不丢弃这一轮。
    def _has_data(r: dict) -> bool:
        if r.get("timeout"):
            return False
        return (r.get("ms_per_token") is not None) or bool(r.get("segments"))

    ok = [r for r in records if _has_data(r)]
    aborted = [r for r in ok if r.get("returncode") != 0]
    if not ok:
        print("\n⚠ 没有任何一轮产出有效数据（下面的 stderr 是直接原因）")
        for r in records:
            if r.get("timeout"):
                print(f"  · threads={r.get('threads_requested')}: 超时")
            else:
                print(f"  · threads={r.get('threads_requested')}: "
                      f"returncode={r.get('returncode')}")
                print(f"      stderr: {r.get('stderr_tail', '')[-300:]}")
        print("\n⚠ 常见原因：① `--tokens` 太小 → 滑窗（500 步）没打印过一次 "
              "`segments(win)`；② 数据路径不存在；③ 训练进程被 OOM 杀掉。")
        print("   建议 `--tokens` ≥ 1500（smoke 档）以确保至少打印 3 次滑窗。")
        return 1
    if aborted:
        print(f"\n⚠ 有 {len(aborted)} 轮「训练完成但退出时崩溃」（数据有效，已纳入对比）：")
        for r in aborted:
            print(f"  · threads={r.get('threads_requested')}: "
                  f"returncode={r.get('returncode')} "
                  f"ms/tok={r.get('ms_per_token')}")
        print("   已知：昇腾 + numba prange 在解释器 finalize 阶段 abort"
              "（`PyGILState_Release ... finalizing`），"
              "\n   发生在 model/checkpoint 保存**之后**，不影响数值；"
              "判据仍可用，但别把它当「训练失败」。")

    # ── 判据表 ──
    base = ok[0]
    print("\n" + "=" * 78)
    print("判据表")
    print("=" * 78)
    print(f"{'threads':>8} {'eff':>4} {'ms/tok':>9} {'vs 1st':>8} "
          f"{'readout(NPU)':>13} {'CPU侧合计':>10} {'final_ppl':>11}")
    print("-" * 78)
    for r in ok:
        segs = r.get("segments", {})
        # CPU 侧合计 = 端到端 − NPU 读出
        cpu = r["ms_per_token"] - r["readout_ms"]
        ratio = (r["ms_per_token"] / base["ms_per_token"]
                 if base["ms_per_token"] else float("nan"))
        print(f"{r['threads_requested']:>8} {r['threads_effective']:>4} "
              f"{r['ms_per_token']:>9.2f} {ratio:>7.3f}× "
              f"{r['readout_ms']:>13.2f} {cpu:>10.2f} {r['final_ppl']:>11.1f}")

    if not args.segments_only:
        print("\n分段明细（末窗中位，ms/tok）")
        seg_keys: list[str] = []
        for r in ok:
            for k in r.get("segments", {}):
                if k not in seg_keys:
                    seg_keys.append(k)
        seg_keys.sort(key=lambda k: -base.get("segments", {}).get(k, 0))
        hdr = f"{'段':<16}" + "".join(f"{r['threads_requested']:>9}线程"
                                      for r in ok)
        print(hdr)
        print("-" * len(hdr))
        for k in seg_keys:
            row = f"{k:<16}"
            for r in ok:
                row += f"{r.get('segments', {}).get(k, float('nan')):>10.2f}"
            print(row)

    best = min(ok, key=lambda r: r["ms_per_token"])
    print("\n" + "-" * 78)
    print(f"最佳: numba-threads={best['threads_requested']} "
          f"（实际生效 {best['threads_effective']}）"
          f" → {best['ms_per_token']:.2f} ms/tok")
    if len(ok) > 1:
        worst = max(ok, key=lambda r: r["ms_per_token"])
        print(f"      最差 {worst['threads_requested']} 线程 "
              f"→ {worst['ms_per_token']:.2f} ms/tok"
              f"（差异 {(worst['ms_per_token'] / best['ms_per_token'] - 1) * 100:+.1f}%）")
    print("⚠ 结论只对「本机 + 本 commit + 同数据 + 同 token 预算」有效。")
    print("  改默认前请把本表贴进 docs/PHD-Net_性能评估与迭代方案.md。")
    print("-" * 78)

    out_dir = _ROOT / "outputs" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out) if args.out else out_dir / f"server_threads_{ts}.json"
    out_path.write_text(json.dumps({
        "preset": args.preset, "data": args.data, "tokens": args.tokens,
        "cpu_count": os.cpu_count(), "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"原始数据 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
