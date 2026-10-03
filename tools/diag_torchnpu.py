#!/usr/bin/env python3
"""P159：诊断「读出回落numba-cpu」—— torch_npu 注册失败的真因。

fhz 的生产训练日志（2026-10-03 12:56）：

    [capability] 探测到设备 ['npu:0'] 可用
    [readout] backend=numba-cpu(回落) | fallback reason:
      RuntimeError: [precision] ❌ 设备auto **无任何可用精度**（请求 fp8）。
        ✗ fp8  RuntimeError: Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl,
                              opencl, ideep, hip, xpu, mps, meta, hpu, xla, mkldnn…

**这行报错是 torch_npu 未完成注册的典型症状**：设备字符串 `npu:0` 被
torch 当成普通字符串去查「已知设备表」，表里没有它 → 报错里列的是
**torch 核心认识的设备**（cpu/cuda/xpu/mps/…），**一个 npu 都没有**。

⚠ 与 `[capability] 探测到设备 ['npu:0'] 可用` **矛盾** → 因为那条走的是
   `resolve_device()`，它可能只做了 `torch.npu.is_available()` 之类而
   **没有真正 import torch_npu**。

本脚本按顺序定位，逐条给出结论。
"""
from __future__ import annotations

import os
import sys
import traceback


def step(n, title):
    print()
    print("=" * 70)
    print("STEP %d%s" % (n, title))
    print("=" * 70)


step(0, "环境事实")
print("  python      :", sys.version.split()[0])
print("  可执行文件  :", sys.executable)
print("  cwd         :", os.getcwd())
print("  环境变量:")
for k in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "LD_LIBRARY_PATH",
          "PYTHONPATH", "PYTORCH_NPU_ALLOC_CONF", "ASCEND_OPP_PATH"):
    v = os.environ.get(k)
    print("    %-22s = %s" % (k, (v[:70] + "...") if v and len(v) > 70 else v))

step(1, "torch 是否装对")
try:
    import torch
    print("  torch.__version__   =", torch.__version__)
    print("  torch.__file__      =", torch.__file__)
    print("  version.cuda        =", torch.version.cuda)
    print("  编译时的 torch 平台 =", torch.__config__.show().split("\n")[0]
          if hasattr(torch, "__config__") else "?")
except Exception as e:
    print("  FAIL", e)
    sys.exit(1)

step(2, "torch 认识的设备表（关键：里面有没有 npu）")
try:
    # torch 的 device 解析表就是这些名字
    print("  torch.device('npu:0') ->", end=" ")
    try:
        d = torch.device("npu:0")
        print("OK  type=%s index=%s" % (d.type, d.index))
    except Exception as e:
        print("FAIL  %s: %s" % (type(e).__name__, str(e)[:100]))
    print()
    print("  手动建一个 npu 张量（这才是真正的检验）:")
    try:
        a = torch.ones(4, 4, dtype=torch.float16, device="npu:0")
        print("    ✅ 成功:", a.device, a.dtype)
        print("    → torch_npu **已注册**，设备字符串 npu:0 可用")
    except Exception as e:
        print("    ❌ 失败: %s: %s" % (type(e).__name__, str(e)[:140]))
        print()
        print("    ⚠ 若报错里列出的设备名**不含 npu** → torch_npu 没注册成功")
        print("      典型原因：")
        print("       1. `import torch_npu` 失败（版本不匹配 / CANN 找不到）")
        print("       2. 环境变量 LD_LIBRARY_PATH 没指向 Ascend 驱动库")
        print("       3. 以root/其他用户跑，但 /usr/local/Ascend/cann-8.5.0")
        print("          owner 不同（日志里有 owner mismatch 警告）")
        print("       4. torch 与 torch_npu 版本不严格配对")
except Exception:
    traceback.print_exc()

step(3, "torch_npu 能否 import（最关键）")
try:
    import torch_npu
    print("  ✅ import torch_npu 成功")
    print("  torch_npu.__version__ =", getattr(torch_npu, "__version__", "?"))
    print("  torch_npu.__file__    =", getattr(torch_npu, "__file__", "?"))
    print()
    print("  npu 相关 API:")
    for name in ("npu", "npu_matmul", "npu_quantize_per_tensor",
                 "npu_quant_matmul", "npu_anti_quant", "npu_format_cast",
                 "npu_dtype_cast", "npu_weight_quant_batchmatmul"):
        print("    %-32s %s" % (name, hasattr(torch_npu, name)))
    print()
    print("  device_count:", torch.npu.device_count() if hasattr(torch, "npu") else "?")
    print("  is_available:", torch.npu.is_available()
          if hasattr(torch, "npu") else "?")
except Exception as e:
    print("  ❌ import torch_npu 失败: %s: %s" % (type(e).__name__, str(e)[:160]))
    print()
    print("  → **这才是根因**。日志里的『无任何可用精度』是因为连张量都建不出来，")
    print("    于是 fp8/fp16/int8 **全部探测失败**，降级链走到终点 → 回落 numba-cpu。")
    print()
    print("  请把下面的输出贴给 fhz：")
    print("    python -c \"import torch_npu\" 2>&1")
    print("    python -V; pip show torch torch-npu 2>&1 | grep -E 'Name|Version'")

step(4, "版本配对核对")
try:
    import torch_npu
    tv = getattr(torch_npu, "__version__", "?")
    t = torch.__version__.split("+")[0]
    print("  torch       =", t)
    print("  torch_npu   =", tv)
    print("  ✅ 若主版本.次版本一致（2.x.y↔ 2.x.y）→ 配对正常")
except Exception:
    print("  （torch_npu 不可用，跳过）")

step(5, "项目代码侧的判定")
try:
    sys.path.insert(0, os.getcwd())
    from phdnet.backends.accel_readout import resolve_accel_device
    dev = resolve_accel_device("auto")
    print("  resolve_accel_device('auto') =", dev)
    try:
        t = torch.ones(4, 4, device=dev)
        print("  ✅ 在 %s 上建张量成功" % dev)
    except Exception as e:
        print("  ❌ 在 %s 上建张量失败: %s" % (dev, str(e)[:120]))
        print()
        print("  → 这**完全解释了**生产日志的回落：")
        print("    [capability] 说'设备可用'（只查了 is_available）")
        print("    [precision] 说'无任何可用精度'（真去建张量，失败）")
        print("    → 两者不是矛盾，是**探测深度不同**。")
except Exception:
    traceback.print_exc()

step(6, "结论与下一步")
print("""
  判定顺序：
    STEP 3 失败 → **根因是 torch_npu 装不上/未注册**（环境问题，与本项目代码无关）
    STEP 3 成功但 STEP 2 失败 → torch_npu 装了但没 import 生效（检查 import 时机/
                是否在 import torch 之前 import torch_npu）
    STEP 2、3 都成功 → 环境正常，需要复现 `[precision]` 那一步的真实报错

  ⚠ 重要：`[capability] 探测到设备 ['npu:0'] 可用` **不能当作读出已上设备的证据**
    —— 日志已经证明它是假阳性。真正的证据是训练日志里的
    `读出 X ms/tok（后端@设备）` 行（见项目 MEMORY「性能定性结论」）。
""")