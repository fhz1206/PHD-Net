#!/usr/bin/env python3
"""本地（x86）性能**回归基准**——回答「这次改动让 CPU 侧变快还是变慢」。

fhz 2026-09-30：「云端重测工具链不要了，还是本地测试吧」。

## 为什么本地数字不能写进文档

今天已经在本机踩了三次**平台方向相反**的坑：M1 fp32（x86 快 1.80× / 昇腾慢
70×）、M2 融合核（x86 快 1.15–2.16× / 昇腾慢 8–13×）、OMP place 表（x86 无感 /
昇腾慢 8×）。**规则：x86 的绝对数字不构成昇腾的证据。**

所以本脚本的定位很明确：
- ✅ 用途一：**回归检测**——同一台机器、同一 commit 前后对比「变快/变慢」；
- ✅ 用途二：测**与平台无关的部分**（算法复杂度、Python 开销、内存分配、线程缩放）；
- ❌ 不用途：不产出文档里的 ms/tok（那必须来自服务器实测）。

## 用法

```bash
# 基线（当前 commit），输出 JSON 到 outputs/experiments/local_bench_<ts>.json
python tools/bench_local.py

# 与上次结果对比（回归结论：哪些变快/变慢）
python tools/bench_local.py --compare

# 只测线程缩放（P22 结论：核内 1→6 线程仅 1.16×）
python tools/bench_local.py --threads 1,2,4,8
```

## 测什么（全部与平台无关或明确标注）
1. **CPU 侧热点**：`SparseEncoder.encode`（M1）、`SparsePCStack.infer`（M2）、
   `OnlineCSRTable.predict_arr` / `_ltm_learn_rows`（M4b）、主循环 glue。
2. **numba 线程缩放**：同一工作负载在 1/2/4/8 线程下的耗时（验证线程上限设置）。
3. **端到端 smoke**：`train_1b/train.py --preset 4m` 跑几百 token，取分段耗时
   （**仅作趋势**，不是 1B 档数字）。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
OUT = _ROOT / "outputs" / "experiments"


def _best(fn, reps: int = 5) -> float:
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b


def _env() -> dict:
    import numpy as np
    e = {"python": sys.version.split()[0], "platform": platform.platform(),
         "machine": platform.machine(), "cpu_count": os.cpu_count(),
         "numpy": np.__version__}
    for mod, key in (("numba", "numba"), ("torch", "torch")):
        try:
            e[key] = __import__(mod).__version__
        except Exception:                                     # noqa: BLE001
            e[key] = None
    try:
        e["commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_ROOT),
            capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:                                         # noqa: BLE001
        e["commit"] = "unknown"
    e["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return e


def bench_mechanisms(threads: int | None) -> dict:
    """CPU 侧热点微基准（与加速器无关）。"""
    import numpy as np
    from phdnet.sparse_encoder import SparseEncoder
    from phdnet.sparse_pc import SparsePCStack
    from phdnet.sparse_table import OnlineCSRTable
    if threads:
        import numba
        numba.set_num_threads(threads)
    rng = np.random.default_rng(0)
    out: dict = {}

    def _time_us(fn) -> float | None:
        try:
            return round(_best(fn) * 1e6, 1)
        except Exception as e:                                # noqa: BLE001
            return f"失败: {type(e).__name__}: {e}"[:60]

    # M1 编码器（2048→1024 SDR，fp64：与生产默认一致）
    try:
        enc = SparseEncoder(2048, 1024, 128, rng, dtype="fp64")
        x = rng.normal(0, 1, 2048)
        out["M1_encode_us"] = _time_us(lambda: enc.encode(x))
    except Exception as e:                                    # noqa: BLE001
        out["M1_encode_us"] = f"失败: {type(e).__name__}: {e}"[:60]

    # M2 推理（CSR，conn_k=128，1024 宽）——两个核都要测（平台差异敏感项）
    for fused in (True, False):
        try:
            pc = SparsePCStack(1024, 1024, 1024, 0.0, 0.0, rng, conn_k=128,
                               fused=fused)
            s0 = rng.random(1024)
            out[f"M2_infer_{'fused' if fused else 'plain'}_us"] = _time_us(
                lambda pc=pc, s0=s0: pc.infer(s0, 1))
        except Exception as e:                                # noqa: BLE001
            out[f"M2_infer_{'fused' if fused else 'plain'}_us"] = \
                f"失败: {type(e).__name__}: {e}"[:60]

    # M4b recall 的 predict_arr（活跃 256 行 × 72 槽，与服务器 [ltm-diag] 同量）
    try:
        tbl = OnlineCSRTable(n_neurons=1 << 22, m_out=72, lam=0.9, eta=0.01,
                             seed=0)
        act = [int(v) for v in rng.choice(1 << 22, size=256, replace=False)]
        r2 = np.random.default_rng(1)
        for i in act:
            tbl._new_row(i)
            for k in r2.choice(1 << 16, size=72, replace=False):
                tbl._append(int(i), int(k), float(r2.random()))
        out["M4b_predict_arr_us"] = _time_us(lambda: tbl.predict_arr(act))
        # LTM encode（每步经 imprint/recall 调用；P95 优化后 1.44-1.53x）
        from phdnet.bigltm import SparseLTM
        ltm = SparseLTM(n_dim=1024, n_neurons=1 << 20, m_out=72, k_hash=4,
                        seed=0)
        rt = np.zeros(1024)
        rt[:256] = 1.0
        out["LTM_encode_us"] = _time_us(lambda: ltm.encode(rt))
    except Exception as e:                                    # noqa: BLE001
        out["M4b_predict_arr_us"] = f"失败: {type(e).__name__}: {e}"[:60]
    return out


def bench_smoke(tokens: int) -> dict:
    """4M 档端到端 smoke——只作趋势，不是 1B 数字。"""
    cmd = [sys.executable, str(_ROOT / "train_1b" / "train.py"),
           "--preset", "4m", "--data", "pretrain", "--tokens", str(tokens),
           "--log-every", str(max(50, tokens // 4)), "--step-profiling"]
    print(f"  smoke: {' '.join(cmd[1:])}", flush=True)
    p = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        return {"error": (p.stderr or p.stdout)[-400:]}
    import re
    ms, segs, loop = [], {}, {}
    for line in p.stdout.splitlines():
        m = re.search(r"token\s+([\d,]+).*?([\d.]+) ms/tok", line)
        if m:
            ms.append(float(m.group(2)))
        m2 = re.search(r"segments: (.+)$", line)
        if m2:
            segs = {k: float(v) for k, v in
                    (x.split() for x in m2.group(1).split("  ") if " " in x)}
        m3 = re.search(r"loop: (.+)$", line)
        if m3:
            loop = {k: float(v) for k, v in
                    (x.split() for x in m3.group(1).split("  ") if " " in x)}
    return {"ms_per_tok": ms[-1] if ms else None, "segments": segs, "loop": loop}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="", help="逗号分隔，如 1,2,4,8")
    ap.add_argument("--smoke-tokens", type=int, default=600)
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--compare", action="store_true", help="与上一份结果对比")
    args = ap.parse_args()

    ths = [int(t) for t in args.threads.split(",") if t.strip()] or [None]
    result = {"env": _env(),
              "note": "本地 x86 回归基准；**不可作为昇腾/文档数字**"
                      "（今天已实测三次平台方向相反）",
              "mechanisms": {}, "smoke": None}
    for t in ths:
        label = f"threads={t}" if t else "threads=default"
        print(f"\n=== 微基准（{label}）===", flush=True)
        result["mechanisms"][label] = bench_mechanisms(t)
        for k, v in result["mechanisms"][label].items():
            if isinstance(v, (int, float)):
                print(f"  {k:<24} {v:>10.1f} us", flush=True)
            else:
                print(f"  {k:<24} {v}", flush=True)
    if not args.no_smoke:
        print("\n=== 4M 档 smoke（趋势）===", flush=True)
        result["smoke"] = bench_smoke(args.smoke_tokens)
        print(f"  {result['smoke']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = OUT / f"local_bench_{ts}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    print(f"\n结果：{path.relative_to(_ROOT)}")

    if args.compare:
        prevs = sorted(OUT.glob("local_bench_*.json"))
        if len(prevs) < 2:
            print("没有可对比的历史结果（需要至少两份）")
            return 0
        old = json.loads(prevs[-2].read_text(encoding="utf-8"))
        new = result
        print("\n=== 与上一份对比 ===")
        for label, rec in new["mechanisms"].items():
            orec = old.get("mechanisms", {}).get(label)
            if not orec:
                continue
            for k, v in rec.items():
                ov = orec.get(k)
                if ov:
                    mark = "↑慢" if v > ov * 1.05 else ("↓快" if v < ov * 0.95 else "  ≈")
                    print(f"  {k:<24} {ov:>9.1f} -> {v:>9.1f} us  {mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
