"""CANN / torch_npu 运行环境治理（P120，2026-10-01）。

为什么需要这个模块
================================================================================
P115 实测：读出段7.83 ms/tok，而按字节流量的带宽下界是 0.300 ms ——
**实测是下界的 26.1 倍**，所以瓶颈**不在算力、也不在 HBM 带宽**。
P120 用 `npu-smi info -t usages` 补上带宽利用率遥测后，要进一步定位就需要
**把设备侧的下发/调度开销降下来**。

昇腾官方文档给出的两类手段（本模块覆盖环境变量类）：

1. **TASK_QUEUE_ENABLE** —— task_queue 算子下发队列优化等级。
   -默认 **1**（Level 1）：把算子下发任务分两段，aclnn 调用放二级流水，
     用队列传递、并行掩盖一部分下发耗时。
   - **2**（Level 2）：在 Level 1 基础上**进一步平衡一二级流水的负载**
     （把 workspace 相关任务也迁到二级流水），**掩盖效果更好**。
   - ⚠ 仅在**二进制场景**生效（即 `set_compile_mode(jit_compile=False)`）。
   - ⚠ `ASCEND_LAUNCH_BLOCKING=1` 时 task_queue 被强制关闭，本设置失效。
   - ⚠ Level 2 因内存并发，**NPU 内存峰值会上升**（我们内存极宽裕：HBM
     0.1/29 GB，无风险）。

2. **COMBINED_ENABLE** —— 把**非连续的两个算子**组合合并下发，减少 kernel
   启动开销。默认 **0**（关闭）。存在相邻非连续算子组合时开启。

3. **PYTORCH_NPU_ALLOC_CONF=expandable_segments:True** —— 内存池扩展段，
   按需创建内存块，减少碎片。训练场景官方推荐开启。

4. **MULTI_STREAM_MEMORY_REUSE=1** —— 跨流内存复用，减少显存占用。

5. **CPU_AFFINITY_CONF=1** —— 任务绑到固定 CPU 核，减少上下文切换。
   ⚠ **本项目要慎用**：我们已经用 `OMP_PROC_BIND=close` + BLAS cap 8 在管
   CPU 线程，而 CPU 侧只占 1.1/191 核——CPU 不是瓶颈，绑核收益可能为0，
   反而与 numba/BLAS 的线程布局打架。故**默认不设**，留 `--cann-affinity`。

设计纪律
================================================================================
· **必须在 `import torch` / `import numpy` 之前设置** —— 环境变量在库初始化
  时被读取，事后设置无效（同 BLAS 线程的教训）。
· **只设「用户没显式设过」的**（`os.environ.get(k) is None`）—— 绝不覆盖
  用户在 shell 里的显式配置。
· **默认开启的是官方标注「训练场景建议开」的**；有风险的（affinity）默认关。
· 每条都打印实际生效值 —— 与「设了但没生效」这种隐形失效斗智（BUGS A8）。
"""
from __future__ import annotations

import os

# 键 → (默认是否开启, 官方建议, 说明)
_CANN_VARS: dict[str, tuple[bool, str, str]] = {
    "TASK_QUEUE_ENABLE": (
        True, "2",
        "task_queue 下发队列 Level 2（默认 1）。把 workspace 任务也迁到"
        "二级流水，掩盖 CPU 下发开销。⚠ 仅二进制场景生效；"
        "ASCEND_LAUNCH_BLOCKING=1 时失效。"),
    "COMBINED_ENABLE": (
        True, "1",
        "非连续算子组合下发（默认 0）。减少 kernel 启动次数。"),
    "PYTORCH_NPU_ALLOC_CONF": (
        True, "expandable_segments:True",
        "内存池扩展段，按需创建内存块，减少碎片（大模型训练官方推荐）。"),
    "MULTI_STREAM_MEMORY_REUSE": (
        True, "1",
        "跨流内存复用，减少显存占用。"),
    "CPU_AFFINITY_CONF": (
        False, "1",
        "任务绑固定 CPU 核。⚠ **本项目默认不设**：CPU 侧仅占 1.1/191 核，"
        "不是瓶颈；而我们已用 OMP_PROC_BIND=close + BLAS cap 8 管线程，"
        "再绑核可能与 numba/BLAS 线程布局打架。需实测才开。"),
}


def apply_cann_env(verbose: bool = True) -> dict:
    """设置 CANN/torch_npu 环境变量（幂等）。返回实际生效值。

    ⚠ 必须在 `import torch` / `import numpy` 之前调用。
    """
    applied: dict = {}
    lines = []
    for k, (on, val, why) in _CANN_VARS.items():
        cur = os.environ.get(k)
        if cur is not None:
            # 用户显式设过 → 尊重，不覆盖（P120 纪律）
            applied[k] = cur
            lines.append(f"  {k} = {cur}   (显式设置，保留)")
            continue
        if on:
            os.environ[k] = val
            applied[k] = val
            lines.append(f"  {k} = {val}   ← 默认开启")
        else:
            applied[k] = None
            lines.append(f"  {k} = (unset)  ← 默认关闭：{why.split('：')[0]}")
    if verbose:
        print("[cann-env] CANN/torch_npu 运行环境（须在 import torch 前设置）:",
              flush=True)
        for ln in lines:
            print(ln, flush=True)
    return applied


def describe() -> str:
    """返回当前进程里这些变量的实际状态（诊断用，不修改）。"""
    out = []
    for k in _CANN_VARS:
        out.append(f"{k}={os.environ.get(k, '(unset)')}")
    return " ".join(out)


if __name__ == "__main__":  # pragma: no cover
    apply_cann_env()
    print(describe())