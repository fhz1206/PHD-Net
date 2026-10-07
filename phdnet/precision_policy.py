"""统一精度策略：**自动降级链**（P151，fhz 2026-10-03 指令）。

指令原文
================================================================================
「精度逻辑改写成 fp 该多少是多少，如果 fp 的不支持就自动转为 int，int 如果不支持
自动转为 fp，如果两者都不行再报错；解禁 fp8, fp4, int4, int8，默认走 fp8 和 fp16」

降级链
================================================================================
    fpX  ──不支持──►  intY（同位宽或更宽）
    intY ──不支持──►  fpX
    两者都不支持 ──►  **报错**（带可选方案）

「支持」的判据= **真跑一次**（不是查表）。这是本项目反复踩过的坑：
- P86 曾把昇腾 fp8 硬编码禁用，而驱动会升级 → 探测比禁令可靠；
- P148 我写的探测是 `a.to(fp16) @ b` → **全程没跑 fp8 kernel**，
  在任何设备上都成功 → `has_fp8` 恒真，等于没有探测。
所以这里的每个候选都要**实际构造并跑一步**才算「支持」。

同族替代（same-family sibling）
================================================================================
降级时优先在**同一族内**换（fp8 → fp16 → fp32），因为它们能复用同一条算子
路径；跨族（fp → int）要换 scale 语义，代价更大但仍在自动完成之列。
"""
from __future__ import annotations

import os                    # P167：PHD_FP8 环境变量（fp8 候选开关）

# ── 精度族定义（位宽从窄到宽）────────────────────────────────────────────────
#   每项:(名称, 存储字节/元素, 族, 同族替代顺序)
FAMILIES: dict[str, dict] = {
    # ── 浮点族（bit-exact 格点：e4m3fn / bf16 / fp16 / fp32）──────────────
    # ⚠⚠ **P152（fhz 2026-10-03）：fp8 的降级**优先落到 int8**，
    #   而不是 fp16。
    #   理由（fhz 指出 + 社区数据佐证）：
    #   ① **访存相同**：fp8 与 int8 都是 1 字节/元素 → 从 fp8 换到 int8
    #      **访存一点不增**，而换到 fp16 会**访存翻倍**（2 字节）——
    #      在 msprof 证明「访存是瓶颈」（Index 占 81%）的前提下这是关键差异。
    #   ② **社区实测二者吞吐相同**：H100 上 FP8 1979 TFLOPS、INT8 1979 TFLOPS
    #      （tutorialq 的实测表）→ 降级到 int8 **不会更慢**。
    #   ③ **社区的 fallback 其实是 bf16/fp16**（TE 的 fp8_autocast 在无 fp8
    #      算子时回 bf16）—— 但那是 H100 有 fp8 tensor core 的情形。
    #      **本项目的 910B 实测无 fp8 算子**（P86：ERR01007），所以「同字节、
    #      同吞吐」的 int8 才是等价替代。
    #   ⚠ int8 的代价：① **动态范围固定**（±127×scale），不像 fp8 的
    #     对数间距自适应 → 这正是社区说「INT8 在Transformer 里易溢出」的点；
    #     ② 需 calibration（本项目用 per-tensor scale = 2·max|W|/127，
    #     即「无校准集的即时 calibration」）。
    #   → 顺序改为 **int8 紧跟 fp8 之后**，fp16/bf16 退到 int 族之后。
    "fp8":  {"bytes": 1, "family": "fp",
             "siblings": ["int8", "fp16", "bf16", "fp32"]},
    "fp16": {"bytes": 2, "family": "fp", "siblings": ["bf16", "fp32", "fp8"]},
    "bf16": {"bytes": 2, "family": "fp", "siblings": ["fp16", "fp32", "fp8"]},
    "fp32": {"bytes": 4, "family": "fp", "siblings": []},
    # ⚠ P152（fhz 2026-10-03：「禁用 fp4 和 int4」）：4-bit **从降级链移除**。
    #   实测理由（P151 的门禁数据）：4-bit 每步要 unpack 出 fp16 副本
    #   （51962×128 → 12.7 MiB），**unpack 开销很可能抵消 4× 的存储收益**，
    #   即「省下的流量又吐回来」。在 gather-GEMV 这个形态下 4-bit 不划算。
    #   → 不再进 FAMILIES，故既不会被请求、也不会出现在降级链里。
    # ── 定点族（per-tensor scale + 最近格点）────────────────────────────
    "int8": {"bytes": 1, "family": "int", "siblings": ["int16", "int32", "fp8"]},
    # ⚠ P152：int4 同上禁用（4-bit 整体退出）。
    "int16": {"bytes": 2, "family": "int", "siblings": ["int32", "fp16"]},
    "int32": {"bytes": 4, "family": "int", "siblings": ["fp32"]},
}

# ⚠⚠⚠ **P163（fhz 2026-10-03 决定）：int 族整体禁用**──────────────────────
# 「就按照刚刚测试的版本吧，别的多余的脚本不要了，**所有 int 全部禁用**」
#
# 依据（Ascend910B4 生产实测，outputs/_t2_int8.log）：
#   · fp8 不可用（ERR01007，P86/P158 复核）
#   · **int8 也没顶上** —— 降级链 fp8 → int8 → fp16 里，int8 探测失败，
#     最终落到 **fp16**；即本轮实际跑的是 **fp16 存储 + fp16 计算**。
#   · int 族在本项目的稀疏 gather-GEMV 形态下**从未显示收益**：
#     P156 实测 int8 路径在 CPU 上比 fp16 慢 1.2×（连 int16 中间都不如 fp16），
#     而「真int8 累加」数学上不可能（127×127 > 127）。
#   → 结论：**int 族退出**。默认 fp16（= 刚刚测试通过的那一版）。
#
# ⚠ **这里只做「禁用」判定，不删 FAMILIES 条目**—— 保留它们是为了
#   历史日志/文档可追溯，且日后若换平台要恢复只需翻这一个开关。
INT_FAMILY_ENABLED = False

# 「默认走 fp16」——**P163**：模型本体与迭代统一 fp16（刚刚验证通过的那一版）
DEFAULT_MODEL_DTYPE = "fp16"
DEFAULT_ITER_DTYPE = "fp16"


def candidate_order(requested: str) -> list[str]:
    """给出 `requested` 的**降级尝试顺序**。

    顺序：同族替代（窄→宽）→ 跨族（先同位宽的，再逐级放宽）→ fp32 兜底。
    例：`fp8` → fp8, fp16, bf16, fp32, int8, int16, int32
        `int8` → int8, int16, int32, fp8, fp16, bf16, fp32
        `fp4` → fp4, fp8, fp16, bf16, fp32, int4, int8, int16, int32
    ⚠ `fp32` **始终在链上**（它是所有族的公共终点，且一定能跑），
      但**只有当前面全部不可用时才会被选中** —— 否则 fp8 就退化成 fp32 了。
    """
    req = str(requested or "").lower()
    info = FAMILIES.get(req)
    if info is None:
        # 未知名字：直接交给调用方按 fp32 处理
        return ["fp32"]
    fam = info["family"]
    sibs = [s for s in info.get("siblings", []) if s != req]
    other = "int" if fam == "fp" else "fp"
    # 跨族：先挑位宽相同的，再按位宽排
    other_fam = sorted((k for k, v in FAMILIES.items()
                        if v["family"] == other),
                       key=lambda k: FAMILIES[k]["bytes"])
    same_w = [k for k in other_fam if FAMILIES[k]["bytes"] >= info["bytes"]]
    # ⚠ **顺序很重要**（fhz 指令「fp 不支持→int，int 不支持→fp」）：
    #   1) 请求本身
    #   2) **同族**替代（fp8 → fp16 → bf16）—— 复用同一条算子路径，代价最小
    #   3) **跨族**（fp → int），位宽相同或更宽的优先
    #   4) 跨族再回 fp 族（int → fp）：**fp32 放这里**，而不是第 2 步
    # ⚠ 否则 fp8 在 fp16 不可用时会**先撞上 fp32**（它是 fp 族最宽的），
    #   永远轮不到 int8 —— 违背「先试 int」的指令。
    # ⚠ 同族兄弟里**跳过 fp32**（它是 fp 族最宽的）：若 fp8→fp16/bf16 都不行，
    #   直接跳到 fp32 会**绕过 int族** —— 违背「先试 int」的指令。
    sibs_mid = [x for x in sibs if x != "fp32"]
    seq = [req] + sibs_mid + same_w
    if fam == "int":
        # 从 int 出发：回 fp 族（同位宽→更宽），fp32 在最后
        back_fp = [k for k in other_fam
                   if FAMILIES[k]["bytes"] >= info["bytes"]]
        seq += back_fp + ["fp32"]
    else:
        # 从 fp 出发：跨族 int 全试完，再回到 fp32 兜底
        seq += ["fp32"]
    # 去重且保序
    seen: set[str] = set()
    out = [x for x in seq if not (x in seen or seen.add(x))]

    # ══════════════════════════════════════════════════════════════════════
    # **P163：int 族整体禁用**（fhz 2026-10-03「所有 int 全部禁用」）
    # ══════════════════════════════════════════════════════════════════════
    if not INT_FAMILY_ENABLED:
        # 请求 int 时**不静默换人**：报错，因为「禁 int」是硬约束，
        # 悄悄 fp16 会让调用方以为跑的是 int。
        if fam == "int":
            raise ValueError(
                f"int 族已禁用（P163，fhz 2026-10-03）：请求 {req!r} 无效。"
                f"可用：fp32 / fp16 / bf16。"
                f"依据：Ascend910B4 实测 fp8 与 int8 均不可用，实际落到 fp16；"
                f"且 P156 实测 int8 路径在 gather-GEMV 下比 fp16 慢 1.2×。")
        out = [x for x in out if FAMILIES.get(x, {}).get("family") != "int"]
    # ══════════════════════════════════════════════════════════════════════
    # **P167→P189：fp8 的可用性交给「真跑一次」的探测，不再静态剔除**──────
    # ══════════════════════════════════════════════════════════════════════
    # P189（fhz 2026-10-07 指令「精度改为 fp8，迭代用 fp16」）：显式请求 fp8
    # 时**必须把 fp8 放回链首**——910B 无 fp8 算子时由 `_probe` 探测失败后
    # 走「fp8 位模式存储 + int8 算子计算」的降级（见 AccelReadout 的
    # `_fp8_bits` 模式），而不是静默落 fp16（那是 P163 的旧口径，已废）。
    # fp4/int4 仍剔除（P152 定案：unpack 开销抵消存储收益）。
    out = [x for x in out if x not in ("fp4", "int4")]
    if req.lower() == "fp8" and "fp8" not in out:
        out = ["fp8"] + out
    # 去掉裸 fp32 中间项（若已被排除则只剩 fp16/bf16）
    if out and out[0] != "fp32" and "fp32" in out[1:]:
        out = [out[0]] + [x for x in out[1:] if x != "fp32"]
    return out


def _fp8_candidate_enabled() -> bool:
    """fp8 是否进降级候选链（P167）。

    默认 **False**（= P163 的口径，依据是 910B 实测）。设 `PHD_FP8=1` 打开，
    用于**别的平台真的有 fp8 时**（例如 ROCm gfx942 / NVIDIA sm89+）——
    打开后仍由 `resolve_precision` 的 `_probe` **真跑一次**决定，
    跑不通会正常降级，故打开**没有正确性风险**，只有一点启动开销。
    """
    return str(os.environ.get("PHD_FP8", "0")).strip().lower() in (
        "1", "true", "yes", "on")


def describe_chain(requested: str) -> str:
    """人读的一行摘要（用于告警/报错文案）。"""
    return " → ".join(candidate_order(requested))


def resolve_precision(requested: str, probe, device: str = "cpu",
                       allow_reorder: bool = True) -> dict:
    """按降级链探测，返回第一个「真跑得通」的精度。

    `probe(dt) -> (ok: bool, note: str)`：**必须真跑一次**并返回成败。
    本函数**不做**任何静态猜测。

    返回 `{"dtype": str|None, "requested": str, "attempts": [...],
           "downgraded": bool, "chain": str, "reason": str}`
    - `dtype is None` → 链上全部不可用 → **调用方必须报错**（指令要求）。
    """
    req = str(requested or "").lower()
    order = candidate_order(req)
    attempts: list[tuple[str, bool, str]] = []
    for dt in order:
        try:
            ok, note = probe(dt)
        except Exception as e:                              # noqa: BLE001
            ok, note = False, f"{type(e).__name__}: {str(e)[:70]}"
        attempts.append((dt, bool(ok), note))
        if ok:
            return {"dtype": dt, "requested": req, "attempts": attempts,
                    "downgraded": dt != req, "chain": describe_chain(req),
                    "reason": ""}
    return {"dtype": None, "requested": req, "attempts": attempts,
            "downgraded": False, "chain": describe_chain(req),
            "reason": "; ".join(f"{d}: {n}" for d, _, n in attempts[:4])}


def unsupported_message(res: dict, device: str = "") -> str:
    """链上全不支持时的**报错文案**（指令：两者都不行再报错）。"""
    lines = [f"[precision] ❌ 设备 {device or '?'} **无任何可用精度**"
             f"（请求 {res['requested']}）。"]
    lines.append(f"  降级链：{res['chain']}")
    lines.append("  试过的结果：")
    for dt, ok, note in res["attempts"][:8]:
        lines.append(f"    {'✓' if ok else '✗'} {dt:<6} {note}")
    lines.append("  可选方案：① 显式 `--readout-dtype fp32`（所有设备必然支持）"
                 "；② 检查算子库/CANN 版本是否支持低精度；"
                 "③ 若设备内存吃紧，可试 `int8`/`fp8`（1 字节）。")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# 进程级探测缓存（P151，fhz 2026-10-03：「训练启动时判断一次即可」）
# ══════════════════════════════════════════════════════════════════════════
#为什么需要：`AccelReadout` 每步都会被构造吗？不——但**每次 `pick_readout_backend`
# 都会构造一个**，而真实训练里可能因为resume / 多臂 A/B 而反复构造。
#   探测本身要真跑 matmul（8×8 小矩阵，单次 ~0.1~1 ms，fp8 首次还要 JIT），
#   重复做纯属浪费，且**同一进程内设备能力不会变**。
# ⚠ 缓存键必须含 `device`（CPU 与 NPU 能力不同）。
_CACHE: dict = {}


def resolve_precision_cached(requested: str, probe, device: str = "cpu",
                             use_cache: bool = True) -> dict:
    """`resolve_precision` + 进程级缓存（fhz：启动时判断一次即可）。"""
    key = (str(requested or "").lower(), str(device))
    if use_cache and key in _CACHE:
        hit = _CACHE[key]
        return dict(hit, cached=True)
    res = resolve_precision(requested, probe, device)
    res["cached"] = False
    if use_cache:
        _CACHE[key] = dict(res)
    return res


def clear_cache() -> None:
    """清缓存（测试用；换设备/换驱动后必须调用）。"""
    _CACHE.clear()
