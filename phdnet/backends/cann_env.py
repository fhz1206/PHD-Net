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


# ══════════════════════════════════════════════════════════════════════════
# **P168：ROCm（AMD）运行时环境** —— 与 CANN 同机制，同样须在 import torch 前
# ══════════════════════════════════════════════════════════════════════════
# 依据 **ROCm 官方文档**（2026-10-03 检索）：
#   · `TORCH_BLAS_PREFER_HIPBLASLT=1`
#     「Explicitly prefers hipBLASLt for GEMM operations in PyTorch, which can
#       improve linear layer performance」—— 官方 env 参考页。
#     → **直接命中本项目**：M1 编码器（P165 实测 BLAS 比自写核快 **5.4×**）
#       与 M6 读出都走 GEMV/GEMM → 显式让 torch 选 hipBLASLt 而非 hipBLAS。
#   · `NCCL_MIN_NCHANNELS=112`（MI300X 专用）
#     → **本项目单卡读出，不设**：那是**多卡 RCCL** 的互联调优，单卡无收益
#       且占内存。**明确写下来是为了防止以后照抄。**
#   · `HSA_OVERRIDE_GFX_VERSION` —— 是**兼容性覆盖**（在不支持的卡上强开），
#     本项目**不设**：它会绕过真实的架构判定，与 P167 的「按真跑探测」相悖。
#   · `torch.compile(backend='rocdynamo')`：社区数据称有 ~25% 融合收益，
#     但本机 P155 已实测 `torch.compile` **不可用**（无 C++ 编译器
#     `cl is not found`），且未在 ROCm/昇腾上实测 → **默认不设**。
#
# ⚠ **铁律**：任何一项都**没有本项目的实测数据**（本机无 AMD 卡）。
#   这里只做「官方文档推荐 + 默认无害」，**不宣称收益**。
#   真要开→ 一次改一个 + `--step-profiling` 对比 `segments(win)`。
_ROCM_VARS: dict[str, tuple[bool, str, str]] = {
    "TORCH_BLAS_PREFER_HIPBLASLT": (
        True, "1",
        "官方推荐：让 PyTorch 的 GEMM 优先走 hipBLASLt（ROCm 官方 env 参考页）。"
        "⚠ 本项目 M1 编码器与 M6 读出都是 GEMV/GEMM 形态 → 直接命中。"
        "⚠ **本机无 AMD 卡，本项无实测数据**；若变慢请 `PHD_ROCM_BLASLT=0` 关掉。"),
    "NCCL_MIN_NCHANNELS": (
        False, "112",
        "MI300X 多卡 RCCL 互联调优。⚠ **本项目单卡读出，不该设** —— "
        "单卡无 all-reduce，设了只占内存。（写下来是防以后照抄社区清单。）"),
    "HSA_OVERRIDE_GFX_VERSION": (
        False, "-",
        "⚠ **不要设**：它是「在不支持的卡上强行覆盖架构」的兼容性手段，"
        "会绕过真实判定 —— 与 P167「支持与否由真跑探测决定」相悖。"),
    "TORCHINDUCTOR_COMPILE_THREADS": (
        False, "-",
        "torch.compile 的编译线程数。本项目**默认不 compile**"
        "（P155 实测本机无 C++ 编译器）；用 rocdynamo 后端时才需调。"),
}


def apply_rocm_env(verbose: bool = True) -> dict:
    """设置 ROCm/hipBLASLt 环境变量（幂等）。返回实际生效值。

    ⚠ 必须在 `import torch` / `import numpy` 之前调用。
    ⚠ **本项目无 AMD 卡的实测数据** —— 这里只落官方推荐项且默认无害；
    真要判断收益必须在 ROCm 机器上跑 `--step-profiling` 对比。
    """
    applied: dict = {}
    lines = []
    # 全局逃生阀：PHD_ROCM_ENV=0 一次关掉全部（便于 A/B 与排障）
    if str(os.environ.get("PHD_ROCM_ENV", "1")).strip().lower() in (
            "0", "false", "no", "off"):
        if verbose:
            print("[rocm-env] 已由 PHD_ROCM_ENV=0 关闭（全部不设置）",
                  flush=True)
        return {k: None for k in _ROCM_VARS}
    for k, (on, val, why) in _ROCM_VARS.items():
        cur = os.environ.get(k)
        if cur is not None:
            applied[k] = cur
            lines.append(f"  {k} = {cur}   (显式设置，保留)")
            continue
        #单项逃生阀：PHD_ROCM_<KEY>=0 可单独关掉某项
        if str(os.environ.get("PHD_ROCM_" + k, "1")).strip().lower() in (
                "0", "false", "no", "off"):
            applied[k] = None
            lines.append(f"  {k} = (unset)  ← 已由 PHD_ROCM_{k}=0 关闭")
            continue
        if on:
            os.environ[k] = val
            applied[k] = val
            lines.append(f"  {k} = {val}   ← 默认开启")
        else:
            applied[k] = None
            lines.append(f"  {k} = (unset)  ← 默认关闭：{why.split('：')[0]}")
    if verbose:
        print("[rocm-env] ROCm/AMD 运行环境（须在 import torch 前设置; "
              "⚠ 无本机实测数据，见 docs）:", flush=True)
        for ln in lines:
            print(ln, flush=True)
    return applied


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


def describe_rocm() -> str:
    """返回 ROCm 变量的实际状态（P168 诊断用，不修改）。"""
    return " ".join(f"{k}={os.environ.get(k, '(unset)')}"
                    for k in _ROCM_VARS)


# ══════════════════════════════════════════════════════════════════════════
# **P192：`--cann-dispatch` 的下发流水**核对器
# ══════════════════════════════════════════════════════════════════════════
# ⚠⚠ **先说清楚这个开关「不是」什么**，否则它会被误当成一项新优化：
#
#   `TASK_QUEUE_ENABLE=2` 与 `COMBINED_ENABLE=1` **P120 起就已经默认开启**
#   （见上面的 `_CANN_VARS`）。所以 `--cann-dispatch` **不新增任何行为** ——
#   它做的是把「官方推荐的这两项到底有没有真的生效、有没有被别的设置
#   作废」**核对出来并打印**。
#
# 为什么这件事值得一个开关：
#   CANN 官方文档明确写了两个会让它**静默失效**的条件，而这两个条件
#   **都不在环境变量里、也不会打任何日志**：
#     ① `TASK_QUEUE_ENABLE` **仅在二进制场景生效**（需
#        `torch_npu.npu.set_compile_mode(jit_compile=False)`）→ 我们若走
#        JIT 编译路径，它**根本没在起作用**；
#     ② `ASCEND_LAUNCH_BLOCKING=1`（CANN runtime 层）会**强制关闭**
#        task_queue → 设了 `TASK_QUEUE_ENABLE=2` 也白设。
#   注意 ① 与另一个环境变量 `ASCEND_LAUNCH_BLOCKING`（**torch_npu 层**，
#   同名不同层）是两回事 —— 这正是「设了但没生效」（BUGS A8 族）的典型。
#   故本函数把它们**逐条核对**出来，而不是假定「设了 = 生效了」。

def verify_dispatch(report: bool = True) -> dict:
    """核对下发流水相关设置的真实有效性，返回核对结果。

    返回字典：
      `{"effective": bool|None, "checks": [(名, 通过?, 说明)], "notes": [..]}`
      · `effective` 为 `None` 表示非昇腾环境，或关键状态未知（不宣称已生效）。
      · `checks` 里每一项都带说明，供日志直接打印。

    ⚠ 本函数**不修改任何环境变量**（调用点必须早于 `apply_cann_env`，
      或仅用于事后诊断），因此可以在训练中途安全调用。
    """
    checks: list[tuple[str, bool | None, str]] = []
    notes: list[str] = []

    # 只有在真的能 import torch_npu 时才谈得上「昇腾场景」
    try:
        import torch  # noqa: F401
        _has_torch = True
    except Exception as e:                            # noqa: BLE001
        _has_torch = False
        notes.append(f"torch 不可用（{type(e).__name__}）→ 本项不适用")

    npu_present = any(
        os.environ.get(k) is not None for k in ("ASCEND_RT_VISIBLE_DEVICES",)
    ) or os.environ.get("PHDNET_NPU") is not None
    if not _has_torch:
        return {"effective": None, "checks": checks, "notes": notes}
    try:
        import torch_npu  # noqa: F401
        _is_npu = True
    except Exception:                                  # noqa: BLE001
        _is_npu = False

    if not _is_npu:
        notes.append("非 torch_npu 环境（无昇腾）→ 下发流水不适用，"
                     "本项不算失败")
        return {"effective": None, "checks": checks, "notes": notes}

    # ① task_queue 等级
    tq = os.environ.get("TASK_QUEUE_ENABLE")
    checks.append(("TASK_QUEUE_ENABLE", tq == "2",
                   f"={tq or '(unset)'}（Level 2 = workspace 任务也迁入二级流水；"
                   f"默认 CANN 值是 1）"))

    # ② **失效条件 1**：ASCEND_LAUNCH_BLOCKING（CANN runtime 层）
    blocking = {k: os.environ.get(k) for k in
                ("ASCEND_RT_LAUNCH_BLOCKING", "ASCEND_LAUNCH_BLOCKING")}
    # Explicit zero is nonblocking; either layer set to one disables queuing.
    blocked = any(v == "1" for v in blocking.values())
    known = all(v in (None, "0", "1") for v in blocking.values())
    nonblocking = False if blocked else (True if known else None)
    checks.append(("ASCEND_LAUNCH_BLOCKING 非阻塞", nonblocking,
                   f"{blocking}；置 1 会强制关闭 task_queue；0/未设置不阻塞"))

    # ③ **失效条件 2**：JIT 编译路径（task_queue 仅二进制场景生效）
    jit_mode = None
    try:
        from torch_npu.npu import npuConfig
        jit_mode = npuConfig.is_jit_compile_false()
    except Exception as e:                            # noqa: BLE001
        notes.append(f"无法读取 jit_compile 状态（{type(e).__name__}: {e}）")
    binary_mode = None if jit_mode is None else bool(jit_mode)
    checks.append(("二进制场景（jit_compile=False）", binary_mode,
                   f"is_jit_compile_false()={jit_mode}；官方文档："
                   f"TASK_QUEUE_ENABLE **仅在二进制场景生效**"))

    # ④ 算子合并下发
    comb = os.environ.get("COMBINED_ENABLE")
    checks.append(("COMBINED_ENABLE", comb == "1",
                   f"={comb or '(unset)'}（非连续算子组合下发，默认 CANN 值 0）"))

    if any(ok is False for _, ok, _ in checks):
        effective = False
    elif checks and all(ok is True for _, ok, _ in checks):
        effective = True
    else:
        effective = None
    if report:
        print("[cann-dispatch] 下发流水生效核对（不修改任何变量）:",
              flush=True)
        for nm, ok, why in checks:
            flag = 'UNKNOWN' if ok is None else ('OK' if ok else 'WARN')
            print(f"  {flag} {nm}: {why}", flush=True)
        for nt in notes:
            print(f"  -- {nt}", flush=True)
        print(f"  => effective={effective}"
              + ("（⚠ 有项未满足，task_queue 可能未真正生效）"
                 if effective is False else ""), flush=True)
    return {"effective": effective, "checks": checks, "notes": notes}


if __name__ == "__main__":  # pragma: no cover
    apply_cann_env()
    print(describe())