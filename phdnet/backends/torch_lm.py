# -*- coding: utf-8 -*-
"""设备解析（torch 设备字符串 ↔ 可用性判定）。

P30（2026-09-29，fhz「旧的可以删除」）：本文件原先还承载**完整的 torch 版
PHD-Net**（`TorchPHDNet` / `TorchWordLM` / `TorchSparsePC` / `TorchLTM` 等，
约 800 行）。该栈已废弃并删除，理由：

  ① **权重与生产 npz 检查点不通用**（不同实现、不同初始化顺序）；
  ② **缺 7 项机制**（`big_ltm`、自适应 LR 等显式 `NotImplementedError`）——
     整体迁移会丢机制，与项目铁律冲突；
  ③ 生产加速路径已由 `phdnet/backends/accel_readout.py::AccelReadout`
     单独实现（只迁移读出，不动 numba 主循环），无需整套 torch 栈。

保留 `resolve_device()`：它是**生产依赖**（`accel_readout.resolve_accel_device`
经它把 `auto` 解析成昇腾/ROCm/CUDA/DirectML/CPU），也是 P10 起统一
「设备解析口径」的实现（auto 择优顺序：昇腾 → ROCm → CUDA → DirectML → CPU）。
"""

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

# ⚠ P160（fhz 2026-10-03 生产日志）：`import torch_npu` **必须在本模块之后立即
#   执行**才生效 —— torch_npu 靠 import 副作用把 `npu` 注册进 torch 的设备表。
#   生产日志里读出全线回落（fp8/fp16/int8 全部「Expected one of cpu, cuda, …」
#   且列表里**没有 npu**）→ 就是这里缺了 import。
#   ⚠ 那个局部探测函数里曾有 import，但写在探测内部 + `except: pass`
#   静默吞异常 → 失败时日志里看不到任何线索。统一收敛到下面这个函数。
_NPU_IMPORT_ERROR: str | None = None


def ensure_npu_registered() -> str | None:
    """import torch_npu 以把 `npu` 注册进 torch 设备表。返回错误原因或 None。"""
    global _NPU_IMPORT_ERROR
    if _NPU_IMPORT_ERROR is None:
        try:
            import torch_npu  # noqa: F401   # 副作用式注册
        except Exception as exc:                            # noqa: BLE001
            _NPU_IMPORT_ERROR = f"{type(exc).__name__}: {str(exc)[:120]}"
    return _NPU_IMPORT_ERROR


from .torch_backend import probe_devices


def resolve_device(device: str = "auto", allow_fallback: bool = False) -> str:
    """把 auto/cpu/cuda/rocm/npu/dml/任意 torch 设备串解析为可执行设备。

    - "auto"：probe_devices() 按 **昇腾 NPU → ROCm → CUDA → DirectML → CPU**
      择优（与 `phdnet/device.py::probe()` 声明的顺序一致；2026-09-28 补齐
      DirectML，此前本函数缺该项，会跳过 Windows 上的 AMD/Intel GPU 直通）；
    - 显式设备不可用时抛 RuntimeError（诚实报错）；allow_fallback=True 才
      在 RuntimeWarning 后回退 CPU。
    """
    if torch is None:
        raise RuntimeError("未安装 torch，无法使用 torch LM 栈")
    dev = str(device).strip().lower()
    # ⚠ P160：`probe_devices()` 报 npu ok **不代表 torch 真能用 npu** ——
    #   torch_npu 的 import 副作用才把 npu 写进 torch 的设备表。
    #   → auto 分支在相信 npu 之前**先确保已注册**（成本：一次 import 缓存）。
    _npu_err = ensure_npu_registered()
    if dev in ("", "auto"):
        probes = probe_devices()
        for key in ("npu", "rocm", "cuda"):
            if probes.get(key, {}).get("ok"):
                if key == "npu" and _npu_err is not None:
                    # 注册失败 → 不能声称可用（否则下游建张量必失败）
                    continue
                return "npu" if key == "npu" else "cuda"
        if probes.get("dml", {}).get("ok"):
            return probes["dml"]["device"] or "privateuseone:0"
        return "cpu"
    if dev == "cpu":
        return "cpu"

    def _ok(kind: str) -> bool:
        probes = probe_devices()
        if kind == "cuda":                      # ROCm 走 cuda 接口，二者互认
            return bool(probes.get("cuda", {}).get("ok")
                        or probes.get("rocm", {}).get("ok"))
        return bool(probes.get(kind, {}).get("ok"))

    if dev in ("cuda", "rocm", "npu", "gpu"):
        kind = {"gpu": "cuda", "rocm": "cuda"}.get(dev, dev)
        if _ok(kind):
            return "npu" if dev == "npu" else "cuda"
        probes = probe_devices()
        msg = (f"请求设备 '{device}' 在本机不可用（probe_devices: {probes}）；"
               "torch LM 栈诚实报错，不做静默回退。")
        if allow_fallback:
            import warnings
            warnings.warn(msg + " 已按 allow_fallback=True 回退 CPU。", RuntimeWarning)
            return "cpu"
        raise RuntimeError(msg)
    if dev == "dml":
        probes = probe_devices()
        if probes.get("dml", {}).get("ok"):
            return probes["dml"]["device"] or "privateuseone:0"
        msg = (f"请求设备 'dml' 在本机不可用（probe_devices: {probes}）；"
               "torch LM 栈诚实报错，不做静默回退。")
        if allow_fallback:
            import warnings
            warnings.warn(msg + " 已按 allow_fallback=True 回退 CPU。", RuntimeWarning)
            return "cpu"
        raise RuntimeError(msg)
    # 任意 torch 设备字符串（cuda:1 / mps / privateuseone …）：直接探测
    try:
        t = torch.zeros(1, device=dev)
        del t
        return dev
    except Exception as e:  # noqa: BLE001
        msg = f"请求设备 '{device}' 不可用（{type(e).__name__}: {e}）。"
        if allow_fallback:
            import warnings
            warnings.warn(msg + " 已按 allow_fallback=True 回退 CPU。", RuntimeWarning)
            return "cpu"
        raise RuntimeError(msg) from e


# ---------------------------------------------------------------------------
# M1 稀疏编码器（torch 化；初始化数组由 numpy 参考实现抽取，逐位一致）
# ---------------------------------------------------------------------------
