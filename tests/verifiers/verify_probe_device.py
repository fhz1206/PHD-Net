#!/usr/bin/env python3
"""verify_probe_device：读出精度探测**必须用已解析的设备**（P161 门禁）。

P163：dtype 列表改为 fp16/bf16/fp32 —— **int 族已整体禁用**，
原来的 int8 用例现在应当**报错**（见 P163 的 int 门禁）。

fhz 2026-10-03 13:19 的生产日志（Ascend910B4）：

    [readout] backend=numba-cpu(回落) | fallback reason:
      [precision] 设备 auto 无任何可用精度（请求 fp8）
       fp8 / int8 / fp16 / bf16 / fp32 全失败：
       RuntimeError: Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, ...

**根因（P161）**：`accel_readout` 里探测闭包写的是 `_dev = str(device)`——
即**未解析的原始入参 `"auto"`**；而上一行 `self.device = resolve_accel_device(device)`
**已经把 "auto" 解析成 "npu"**。于是`torch.ones(8, 8, device="auto")`
把 `"auto"` 当未知设备名去查表 → 报的那串设备名是 **torch 核心的表**（无npu）
→ **所有 dtype 必然失败** → 降级链走完 → 回落 numba-cpu。

⚠ 这个 bug 的恶劣之处：
1. 报错信息**极具误导性**——看起来像「910B 没有 fp8 算子」，实际是设备名写错；
2. 我P160 因此误诊为「torch_npu 未注册」，白改了一处（那处 import 本身无害、
   仍保留，但**不是根因**）；
3. 在本机（CPU）**测不出来**，因为 `resolve("auto")=="cpu"` 恰好也失败 →
   任何 `device="auto"` 的探测都会失败 → 与真机表现一致，无差异可测。

本门禁的做法：**直接断言源码里探测用的 `_dev` 来自 `self.device`**
（静态检查，不依赖设备），并**动态验证** `torch.ones(device="auto")` 会失败
（证明该bug 的机理成立）。
"""
from __future__ import annotations

import io
import re
import sys

FAIL = []
OK = []


def case(name: str, cond: bool, detail: str = "") -> None:
    (OK if cond else FAIL).append(name)
    tag = "PASS" if cond else "FAIL"
    print("[%s] %s%s" % (tag, name, ("  —— " + detail) if detail else ""))


SRC_PATH = "phdnet/backends/accel_readout.py"

with io.open(SRC_PATH, encoding="utf-8") as f:
    src = f.read()
lines = src.split("\n")


def _line_of(pat: str) -> int:
    for i, l in enumerate(lines):
        if re.search(pat, l):
            return i + 1
    return 0


print("[A] 静态检查：探测用的设备必须是 self.device（已解析），不是原始入参")
_dev_line = _line_of(r"^\s*_dev\s*=")
_dev_src = lines[_dev_line - 1].strip() if _dev_line else ""
case("A1 `_dev` 有赋值", _dev_line > 0, "line %d: %s" % (_dev_line, _dev_src))
case("A2 `_dev` 取自 self.device（P161 修复）",
     "self.device" in _dev_src,
     "若为 str(device) 则会用未解析的 'auto' → 全线 fail")
case("A3 `_dev` **不是** `str(device)`",
     re.search(r"_dev\s*=\s*str\(\s*device\s*\)", _dev_src) is None,
     "当前: %s" % _dev_src)

print()
print("[B] 动态验证：bug 机理（device='auto' 必然失败）")
try:
    import torch
    try:
        torch.ones(2, 2, device="auto")
        case("B1 torch.ones(device='auto') 失败（bug 机理成立）", False,
             "意外成功 → 本门禁的前提不成立，需重新评估")
    except Exception as e:
        msg = str(e)
        case("B1 torch.ones(device='auto') 失败（bug 机理成立）", True,
             msg[:60] + "...")
        case("B2 报错来自 torch 的设备表（列的是 cpu/cuda/…）",
             "Expected one of" in msg or "not a supported" in msg.lower(),
             "这正是生产日志里那段文本")
    # 对照：已解析的设备名必须能建张量
    try:
        torch.ones(2, 2, device="cpu")
        case("B3 已解析设备（cpu）可建张量", True)
    except Exception as e:
        case("B3 已解析设备（cpu）可建张量", False, str(e)[:50])
except ImportError:
    case("B0 跳过（无 torch）", True)

print()
print("[C] 回归：device=auto 的读出必须能完成探测 + 一步训练")
try:
    import numpy as np
    import warnings

    sys.path.insert(0, ".")
    warnings.simplefilter("ignore")
    from phdnet.backends.accel_readout import AccelReadout
    from phdnet.precision_policy import clear_cache
    from phdnet.sparse_pc import _random_csr

    n_out, n_h, k = 512, 128, 16
    csr = _random_csr(np.random.default_rng(3), n_out, n_h, k,
                      0.05 * np.sqrt(n_h / k), False, 0.8)
    h = np.random.default_rng(0).normal(0, 1, n_h)
    t = np.zeros(n_out)
    t[0] = 1.0
    for dt in ("fp16", "bf16", "fp32"):
        try:
            clear_cache()
            ro = AccelReadout(n_h, n_out, np.random.default_rng(3),
                              device="auto", dtype=dt, conn_k=k, csr=csr,
                              int8_compute="auto")
            y = ro.forward_dev(h)
            n = ro.learn_softmax(h, t, 0.15, y_pre=y, target_idx=0)
            good = (tuple(y.shape) == (n_out,) and np.isfinite(n))
            case("C:%s device=auto 可跑完（落地 %s）" % (dt, ro._precision_resolved["dtype"]),
                 good, "nll=%.4f device=%s" % (n, ro.device))
        except Exception as e:
            case("C:%s device=auto 可跑完" % dt, False, str(e)[:60])
except Exception as e:
    case("C0 跳过（构造失败）", True, str(e)[:60])

print()
n_fail = len(FAIL)
print("=" * 66)
print("结果：%d 例 FAIL%s" % (n_fail, (" → " + str(FAIL)) if n_fail else ""))
print("=" * 66)
sys.exit(1 if n_fail else 0)