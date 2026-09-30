#!/usr/bin/env python3
"""定向探测：昇腾的**量化矩阵乘** API 到底能不能用（fp8 张量不通之后的下一步）。

fhz 提供的 `tools/probe_fp8.py` 结果（2026-09-30 19:15，Ascend910B4 / CANN 8.5 /
torch_npu 2.9.0 / torch **2.9.0+cpu**）：

- `float8_e4m3fn` 与 `float8_e5m2` 在 **npu 上 create / roundtrip / matmul 全部
  ERR01007「has not been supported」**；`torch.ops` 里也没有任何 float8 算子；
- 但 `torch_npu` 暴露了 **44 个量化相关 API**，其中与「读出」直接相关的有：
  `npu_weight_quant_batchmatmul`（权重量化 + 批量矩阵乘）、`npu_quant_matmul`、
  `npu_quant_matmul_dequant`、`npu_dynamic_quant`、`npu_group_quant`、
  `npu_quantize_per_tensor`、`npu_trans_quant_param` …

**因此下一步不是继续找 fp8，而是问：这些 API 支持什么 dtype（大概率 int8 / int4）、
签名是什么、能否吃我们的形状（读出是 (n_out, n_h) @ (n_h,)）。**

⚠ 同时要核对一个可疑点：`torch 2.9.0+cpu` 与 `torch_npu 2.9.0` 组合——torch 是
**CPU 版**，而 torch_npu 通常要求与之配套的 torch 构建；版本/构建不匹配会让一部分
算子静默不可用。本脚本会一并记录，并在报错时提示这一点。

## 用法（服务器）
```bash
python tools/probe_npu_quant.py            # 报告
python tools/probe_npu_quant.py --json     # 机器可读
```
"""
from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys

# 与「读出」最相关的候选：权重量化 matmul、量化 matmul、量化/反量化、参数变换
CANDIDATES = [
    "npu_weight_quant_batchmatmul",
    "npu_quant_matmul",
    "npu_quant_matmul_dequant",
    "npu_quant_matmul_reduce_sum",
    "npu_dynamic_quant",
    "npu_group_quant",
    "npu_quantize_per_tensor",
    "npu_trans_quant_param",
    "npu_anti_quant",
    "npu_dtype_cast",
    "npu_format_cast",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    out: dict = {"env": {}, "apis": {}, "verdict": ""}
    try:
        import torch
        import torch_npu                                    # noqa: PLC0415
        out["env"] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", "?"),
            "torch_build": torch.__config__.show().split("\n")[0][:120],
        }
        try:
            out["env"]["npu"] = torch.npu.get_device_name(0)
        except Exception:                                   # noqa: BLE001
            out["env"]["npu"] = "?"
    except Exception as e:                                  # noqa: BLE001
        out["verdict"] = f"torch/torch_npu 不可用：{type(e).__name__}: {e}"
        print(json.dumps(out, ensure_ascii=False, indent=2) if args.json else out["verdict"])
        return 0

    for name in CANDIDATES:
        fn = getattr(torch_npu, name, None)
        rec: dict = {"exists": fn is not None}
        if fn is not None:
            try:
                rec["signature"] = str(inspect.signature(fn))[:220]
            except Exception:                               # noqa: BLE001
                rec["signature"] = "(签名不可读)"
            rec["doc"] = (fn.__doc__ or "").strip().split("\n")[0][:220]
            # 真正调用一次：小形状，float32 输入，看它接受什么
            for dt_name, dt in (("float32", torch.float32),
                                ("float16", torch.float16),
                                ("bfloat16", torch.bfloat16)):
                try:
                    a = torch.zeros(8, dtype=dt, device="npu")
                    b = torch.zeros(8, 4, dtype=dt, device="npu")
                    _ = fn(a, b)
                    rec[f"call_{dt_name}"] = "ok"
                except Exception as e:                       # noqa: BLE001
                    rec[f"call_{dt_name}"] = f"{type(e).__name__}: {e}"[:90]
            out["apis"][name] = rec
    else:
        pass

    ok = [n for n, r in out["apis"].items()
          if any(v == "ok" for k, v in r.items() if k.startswith("call_"))]
    if ok:
        out["verdict"] = (f"可用 API：{ok} → 读出可尝试走量化权重路径"
                          f"（注意：这些多半是 int8/int4，权重低精度 + 更新高精度的"
                          f"组合需要另行设计）")
    else:
        out["verdict"] = ("所有候选量化 matmul API 均调用失败 → 若 torch/torch_npu "
                          "构建不匹配（当前 torch 是 **+cpu** 版），先修正安装；"
                          "否则量化读出在 910B4 上不可行，维持 fp16 读出")
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    else:
        print("=== 昇腾量化 API 探测 ===")
        print(f"环境：{out['env'].get('npu')} | torch {out['env']['torch']} | "
              f"torch_npu {out['env']['torch_npu']}")
        for n, r in out["apis"].items():
            print(f"\n[{n}] {r['signature']}")
            if r.get("doc"):
                print(f"    doc: {r['doc'][:100]}")
            for k, v in r.items():
                if k.startswith("call_"):
                    print(f"    {k:<18} {v}")
        print(f"\n>>> {out['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
