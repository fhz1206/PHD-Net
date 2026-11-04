#!/usr/bin/env python3
"""**统一门禁 runner** —— 一次跑完 `tests/gates/gate_*.py`，自动处理解释器问题。

## 为什么需要它

本机（Windows）有多个 Python，`python` 可能指向**没有 numpy** 的解释器
（WorkBuddy 内置 3.13）→ 直接 `python tests/gates/xxx.py` 会报
`ModuleNotFoundError: No module named 'numpy'`，看起来像门禁坏了，
实际是**启动器用错**。

实测（2026-11-01）：
- `gate_encoder_gemv` / `gate_ltm_reverse_index` —— 用裸 `py <file>` 跑**失败**，
  用 `sys.executable`（同一解释器）跑**通过** → 代码没问题，是启动方式问题；
- `gate_cli_lang_semantics` —— **真路径缺陷**（`eval_common` 在
  `tests/evals/`，门禁只加了 `tests/`）→ 已修（P213），现 20/20。

本 runner 用 `phdnet._python_exe.python_exe()` 选一个**实测能 import numpy**
的解释器来跑每个门禁 → 不再受调用方用什么启动器影响。

## 用法

```bash
# 全跑（按文件名排序）
python tools/run_gates.py

# 只跑指定的
python tools/run_gates.py gate_forward_only gate_cpu_opt_p211

# 按关键字筛选
python tools/run_gates.py --match ltm

# 只看失败详情
python tools/run_gates.py --show-failures

# 并发跑（省时；每个门禁独立进程，互不干扰）
python tools/run_gates.py --jobs 4
```

## 输出

- 每行：`PASS/FAIL  <门禁名>  (耗时)`；FAIL 时附最后一行错误；
- 末尾：汇总（通过数 / 失败数 / 失败清单）；
- JSON：`outputs/experiments/gates_<ts>.json`（供 CI 消费）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_python_exe_helper():
    """按**文件路径**加载 `phdnet/_python_exe.py`，**不触发 `phdnet/__init__.py`**。

    ⚠ 为什么要绕开包初始化：`phdnet/__init__.py` 会 `from .model import PHDNet`
    → 连带 import numpy/numba → 在**没装 numpy 的解释器**里直接炸
    （本模块自己就要能在那种环境里跑，才能去选一个好的解释器！）
    —— 典型的「先要梯子才能找梯子」死锁。
    """
    import importlib.util
    src = _ROOT / "phdnet" / "_python_exe.py"
    spec = importlib.util.spec_from_file_location("_phdnet_python_exe", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.python_exe


python_exe = _load_python_exe_helper()

_GATE_DIR = _ROOT / "tests" / "gates"

# ⚠ **性能敏感门禁**：内含**墙钟时限**断言（如「词级替换 < 200µs」），
#   与其它门禁并发时 CPU 被抢 → **假失败**（实测：gate_cli_i18n 在
#   --jobs 4 下 FAIL、单独跑 PASS）。→ 这些门禁**单独串行**跑。
#   判据：名字里带 timing/perf/bench，或门禁自己声明了时限断言。
_TIMING_SENSITIVE = ("gate_cli_i18n",)


def _summary(stdout: str) -> str:
    """抓门禁自己的汇总行（各门禁格式不一，取第一个含「通过/失败」的）。"""
    for ln in stdout.splitlines():
        if ("通过" in ln and ("失败" in ln or "PASS" in ln)) or \
           ln.strip().startswith("结果：") or "ALL PASS" in ln:
            return ln.strip()
    return ""


def run_one(path: Path, exe: str, timeout: int) -> dict:
    t0 = time.perf_counter()
    try:
        r = subprocess.run([exe, str(path)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout, cwd=str(_ROOT))
        rc, out, err = r.returncode, r.stdout or "", r.stderr or ""
        timed_out = False
    except subprocess.TimeoutExpired:
        rc, out, err = -1, "", f"超时（>{timeout}s）"
        timed_out = True
    dt = time.perf_counter() - t0
    ok = (rc == 0)
    tail = ""
    if not ok:
        lines = [ln for ln in (err or out).splitlines() if ln.strip()]
        tail = lines[-1][:160] if lines else f"returncode={rc}"
    return {
        "gate": path.stem,
        "ok": ok,
        "returncode": rc,
        "seconds": round(dt, 2),
        "summary": _summary(out),
        "tail": tail,
        "timed_out": timed_out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gates", nargs="*", help="门禁名（不含 gate_ 前缀也行）")
    ap.add_argument("--match", default="", help="按关键字筛选")
    ap.add_argument("--jobs", type=int, default=1, help="并发数（默认 1）")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--show-failures", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    exe = python_exe()
    print(f"[runner] 探测到的解释器 = {exe}"
          f" | 当前进程 sys.executable = {sys.executable}")

    paths = sorted(_GATE_DIR.glob("gate_*.py"))
    if args.gates:
        want = {g if g.startswith("gate_") else f"gate_{g}" for g in args.gates}
        paths = [p for p in paths if p.stem in want]
    if args.match:
        paths = [p for p in paths if args.match in p.stem]

    if not paths:
        print("没有匹配的门禁")
        return 2

    print(f"[runner] {len(paths)} 个门禁 | 解释器 {exe} | jobs={args.jobs}")
    t0 = time.perf_counter()
    # 性能敏感门禁 → 串行、且在并发跑完之后跑（避免被抢 CPU）
    timing_paths = [p for p in paths if p.stem in _TIMING_SENSITIVE]
    par_paths = [p for p in paths if p.stem not in _TIMING_SENSITIVE]

    results = []
    if args.jobs > 1 and par_paths:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            results.extend(ex.map(lambda p: run_one(p, exe, args.timeout),
                                  par_paths))
    else:
        for p in par_paths:
            rec = run_one(p, exe, args.timeout)
            results.append(rec)
            mark = "PASS" if rec["ok"] else "FAIL"
            print(f"  {mark}  {rec['gate']:44s} ({rec['seconds']:6.2f}s)"
                  + ("" if rec["ok"] else f"  {rec['tail']}"))
    # 性能敏感门禁：串行、独占
    for p in timing_paths:
        rec = run_one(p, exe, args.timeout)
        results.append(rec)
        mark = "PASS" if rec["ok"] else "FAIL"
        print(f"  {mark}  {rec['gate']:44s} ({rec['seconds']:6.2f}s)"
              f"  [串行：性能敏感门禁]"
              + ("" if rec["ok"] else f"  {rec['tail']}"))
    total = time.perf_counter() - t0

    n_ok = sum(1 for r in results if r["ok"])
    n_all = len(results)
    failed = [r for r in results if not r["ok"]]

    print("\n" + "=" * 70)
    print(f"结果：{n_ok}/{n_all} 通过 | 失败 {len(failed)} | "
          f"总耗时 {total:.1f}s")
    if failed:
        print("失败清单：")
        for r in failed:
            print(f"  · {r['gate']}  ({r['seconds']}s)  {r['tail']}")
        if args.show_failures:
            for r in failed:
                print(f"\n--- {r['gate']} ---")
                print(r["tail"])
    print("=" * 70)

    out_dir = _ROOT / "outputs" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = (Path(args.out) if args.out
                else out_dir / f"gates_{ts}.json")
    out_path.write_text(json.dumps(
        {"python": exe, "total": n_all, "passed": n_ok,
         "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细 → {out_path}")
    return 0 if n_ok == n_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
