#!/usr/bin/env python3
"""探测当前设备/框架**实际暴露的 fp8 能力**（fhz 2026-09-30：「fp8 理论上来讲
昇腾是支持的」）。

**背景**：P85 实测 `torch_npu` 在 `.to(float8_e4m3fn)` 处抛
`Float8_e4m3fn has not been supported`（ERR01007），于是 P86 把加速后端的
fp4/fp8 一律禁用。但那只是**标准 dtype 张量**这条路不通；Ascend 910B/910C
硬件本身有 fp8 算子，可能走的是别的入口（`torch.ops.npu.*`、`torch_npu` 的
专用 API、或 910C/A5 支持而 910B 不支持）。本脚本把这些入口一次问清。

**用法**（服务器上，秒级出结果，不需要训练数据）：
```bash
python tools/probe_fp8.py            # 人类可读报告
python tools/probe_fp8.py --json     # 机器可读（供 AI 分析）
```

结果决定 P86 的去留：
- 若存在可用 fp8 入口 → 接入（走该入口做读出 GEMV / 权重存储）；
- 若全部不可用 → 保留 P86 的禁用，并把结论写成「芯片有 fp8、框架未接通」，
  等 CANN / torch_npu 补齐后一条命令切回。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys


def probe() -> dict:
    out: dict = {
        "env": {"python": sys.version.split()[0], "platform": platform.platform(),
                "machine": platform.machine()},
        "torch": {}, "torch_npu": {}, "ops_float8": [], "dtypes": {},
        "capabilities": [], "verdict": "",
    }
    try:
        import torch
    except Exception as e:                                   # noqa: BLE001
        out["verdict"] = f"torch 不可用：{e}"
        return out
    out["torch"] = {"version": torch.__version__}
    try:
        import torch_npu                                     # noqa: PLC0415
        out["torch_npu"] = {"version": getattr(torch_npu, "__version__", "?")}
    except Exception as e:                                   # noqa: BLE001
        out["torch_npu"] = {"error": f"{type(e).__name__}: {e}"}
        out["verdict"] = "torch_npu 不可用（非昇腾机器？）"
        return out

    # ── 1. 标准 fp8 dtype：能否建张量 / cast / matmul ──
    for name in ("float8_e4m3fn", "float8_e5m2"):
        dt = getattr(torch, name, None)
        rec: dict = {}
        if dt is None:
            rec["exists"] = False
        else:
            rec["exists"] = True
            for dev in ("cpu", "npu", "cuda"):
                if dev == "npu" and not hasattr(torch, "npu"):
                    continue
                if dev == "cuda" and not torch.cuda.is_available():
                    continue
                try:
                    a = torch.zeros(4, dtype=dt, device=dev)
                    rec[f"create@{dev}"] = "ok"
                except Exception as e:                       # noqa: BLE001
                    rec[f"create@{dev}"] = f"{type(e).__name__}: {e}"[:80]
                try:
                    b = torch.zeros(4, dtype=torch.float32, device=dev)
                    c = b.to(dt)
                    _ = c.to(torch.float32)
                    rec[f"roundtrip@{dev}"] = "ok"
                except Exception as e:                       # noqa: BLE001
                    rec[f"roundtrip@{dev}"] = f"{type(e).__name__}: {e}"[:80]
                try:
                    x = torch.zeros(4, 4, dtype=dt, device=dev)
                    v = torch.zeros(4, dtype=dt, device=dev)
                    _ = x @ v
                    rec[f"matmul@{dev}"] = "ok"
                except Exception as e:                       # noqa: BLE001
                    rec[f"matmul@{dev}"] = f"{type(e).__name__}: {e}"[:80]
        out["dtypes"][name] = rec

    # ── 2. torch.ops.npu 里与 float8 相关的算子（可能才是正路）──
    try:
        for ns in ("npu", "aten"):
            o = getattr(torch.ops, ns, None)
            if o is None:
                continue
            names = [n for n in dir(o) if "float8" in n.lower() or "fp8" in n.lower()]
            if names:
                out["ops_float8"] += [f"{ns}::{n}" for n in names]
    except Exception as e:                                   # noqa: BLE001
        out["ops_float8"] = [f"enum failed: {e}"]

    # ── 3. torch_npu 上的 fp8 相关属性/API ──
    for attr in dir(torch_npu):
        if any(k in attr.lower() for k in ("float8", "fp8", "quant", "cast")):
            out["capabilities"].append(f"torch_npu.{attr}")

    # ── 4. 设备与 CANN 版本（判断是 910B/910C/A5）──
    try:
        out["env"]["npu_name"] = torch.npu.get_device_name(0)
        out["env"]["cann"] = getattr(torch_npu, "cann_version", lambda: "?")()
    except Exception:                                        # noqa: BLE001
        pass

    # ── 判定 ──
    fp8 = out["dtypes"].get("float8_e4m3fn", {})
    dev_ok = any(v == "ok" for k, v in fp8.items() if "matmul" in k)
    cast_ok = any(v == "ok" for k, v in fp8.items() if "roundtrip" in k)
    if dev_ok and cast_ok:
        out["verdict"] = ("可用：标准 fp8 张量在设备上可 cast + matmul → "
                          "P86 的禁用应当撤销")
    elif out["ops_float8"]:
        out["verdict"] = ("标准 dtype 不通，但存在 fp8 专用算子 "
                          f"（{out['ops_float8'][:4]}）→ 需走 torch.ops 路径接入")
    else:
        out["verdict"] = ("标准 dtype 与专用算子均不可用 → 保持 P86 禁用；"
                          "结论应记为「芯片有 fp8、框架未接通」")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="输出机器可读")
    args = ap.parse_args()
    r = probe()
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
        return 0
    print("=== fp8 能力探测 ===")
    print(f"环境: {r['env'].get('platform')} / {r['env'].get('machine')}"
          f" | npu={r['env'].get('npu_name', '?')} | CANN={r['env'].get('cann', '?')}")
    print(f"torch {r['torch'].get('version')} | torch_npu {r['torch_npu'].get('version', r['torch_npu'].get('error'))}")
    for name, rec in r.get("dtypes", {}).items():
        print(f"\n[{name}] exists={rec.get('exists')}")
        for k, v in rec.items():
            if k != "exists":
                print(f"   {k:<18} {v}")
    if r["ops_float8"]:
        print(f"\nfp8 专用算子: {r['ops_float8'][:10]}")
    if r["capabilities"]:
        print(f"\ntorch_npu 相关 API: {r['capabilities'][:10]}")
    print(f"\n>>> 判定：{r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
