"""STDP 关联核的 torch 实现 —— 同一份算子覆盖 CPU / CUDA / ROCm(HIP) / 昇腾 NPU。

要点：
  1. **算法语义与 `phdnet/plasticity.STDPCore` 完全一致**：predict 为 scatter-add
     （`index_add_`），学习为「先更新 pre 迹 → 用历史 post 迹算 LTD → 再更新 post 迹」；
     仅活跃突触前行参与更新，权重截断于 [0, w_max]。
  2. **后端只改 device/dtype**：昇腾用 `npu`、ROCm 走 `cuda` 接口（torch.version.hip）、
     DirectML 用 `privateuseone`；算子代码完全相同。
  3. **启用前必须过等价性对拍**：`tests/verifiers/verify_accel_readout.py`（容差判据；跨设备不宣称逐位）
     （沿用 numba 自检 bug 的教训：不能只靠「权重有没有增长」判据）。
  4. 暂不支持逐突触自适应学习率（T4.1，需逐突触状态与循环）——该开关开启时
     请使用 numpy 后端，避免语义分歧。

已知限制（诚实）：目前仅 STDP 热点（M3）完成 torch 化；M1/M2/M4/M6 仍为 numpy/numba；
1B 大容量印迹表（dict 邻接表）仍为 CPU 结构，尚未迁移到设备端稀疏张量。
"""

from __future__ import annotations

# P30：TorchSTDPCore / TorchReadout / selftest_torch 已删除——
#   生产加速读出走 accel_readout.AccelReadout（AXPY + 设备侧缓存）；
#   完整 torch 版 PHD-Net（torch_lm.TorchPHDNet）因权重不通用 + 缺 7 项机制已废弃。
#   本文件只保留：设备探针 + 读出基准（含设备同步与等效带宽）。

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


_DTYPE_ALIASES = {
    "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float64": torch.float64, "fp64": torch.float64, "double": torch.float64,
}


def _resolve_dtype(dtype):
    """把配置里的字符串（cfg.torch_dtype）映射为 torch.dtype。

    2026-09-28：此前未知值**静默回退 float32**，把拼写错误（`fp8`、
    `float_16`）伪装成合法配置。现改为对无法识别的取值抛 ValueError。
    torch 官方名（float32/float16/bfloat16/float64）与常见简写
    （fp32/fp16/bf16/half/double/float）均继续支持；也接受带 `torch.`
    前缀的写法（`torch.float16`）。
    """
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).strip().lower()
    if name.startswith("torch."):
        name = name[len("torch."):]
    if name in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[name]
    raise ValueError(
        f"无法识别的 dtype {dtype!r}；可用取值："
        f"{sorted(_DTYPE_ALIASES)}（亦接受 torch.float16 这类带前缀写法）。"
        f"注意：fp8/fp4 无 torch 原生类型，不在支持范围。")


def _resolve_device(device: str) -> str:
    """校验设备可用性：不可用则回退 cpu 并发出告警（避免静默在 CPU 上训练）。

    A1 修复：此前异常被完全吞掉，请求 npu/cuda 却无硬件时静默跑 CPU 且无提示；
    现改为回退时至少告警一次，便于发现「误配后端」而非得到一个慢且无提示的结果。
    """
    if device == "cpu":
        return "cpu"
    try:
        probe_t = torch.zeros(1, device=device)
        del probe_t
        return device
    except Exception as e:  # noqa: BLE001
        import warnings
        warnings.warn(
            f"设备 '{device}' 不可用（{type(e).__name__}: {e}），已回退 CPU。",
            RuntimeWarning)
        return "cpu"


def probe_devices() -> dict:
    """统一加速器探针：返回 {平台: 可用信息}。

    - CUDA：torch.cuda.is_available()（NVIDIA，**compute capability ≥ 7.0**
      建议；用 `torch.cuda.get_device_capability()` 查询）；
    - ROCm：torch.cuda.is_available() 且 torch.version.hip 非空（AMD，走 HIP 化的
      cuda 接口，算子代码与 CUDA 完全相同）。
      ⚠ P167 修正文档：AMD 用 **gfx 架构号**而非 NVIDIA 的 sm/capability：
      fp16 matmul 需 **gfx90a+**（MI210/MI250/MI300），bf16 需 gfx90a+，
      **fp8 需 gfx942**（MI300X/MI325X）。用 `torch.cuda.get_device_properties(0).gcnArchName`
      取架构串。
    - CANN/昇腾 NPU：torch_npu 插件注册的 `npu` 设备（需按 CANN 版本安装
      torch_npu，如 torch 2.1 ↔ torch_npu 2.1）；
    - DirectML：torch_directml 插件（Windows AMD/Intel GPU，设备串
      `privateuseone:0`）；
    - CPU：恒可用（fp32/fp16/bf16 全支持；fp8 仅模拟）。

    平台键与择优顺序（昇腾 → ROCm → CUDA → DirectML → CPU）现与
    `phdnet/device.py::probe()` 声明的顺序一致。
    """
    out = {"cpu": {"ok": True, "note": "参考路径（fp32/fp16/bf16 全支持）"}}
    if torch is None:
        out["torch"] = {"ok": False, "note": "torch 未安装"}
        return out
    is_hip = bool(getattr(torch.version, "hip", None))
    cuda_ok = torch.cuda.is_available()
    _gpu = {
        "ok": cuda_ok,
        "count": torch.cuda.device_count() if cuda_ok else 0,
        "name": (torch.cuda.get_device_name(0) if cuda_ok else None),
        "version": (torch.version.hip if is_hip else torch.version.cuda),
    }
    # ⚠ P167：**ROCm 下同时写 "cuda" 与 "rocm" 两个键**（内容相同）。
    #   原来只写 `cuda if not is_hip else rocm`（**互斥**），而
    #   `torch_lm.resolve_device._ok("cuda")` 靠 `or` 去捞另一个方向的键 ——
    #   能跑但**脆弱**：任何只查 `probes["cuda"]` 的新代码在 ROCm 上会静默
    #   拿到 None。两个键都写 → 两种查法都对。
    out["rocm" if is_hip else "cuda"] = _gpu
    if is_hip:
        out["cuda"] = _gpu            # 同一份 dict（内容相同，键都可用）
        # 记录AMD 架构串，便于「gfx90a+/gfx942」级别的能力判断
        if cuda_ok:
            try:
                out["rocm"]["gcn_arch"] = (
                    torch.cuda.get_device_properties(0).gcnArchName)
            except Exception:                     # noqa: BLE001
                out["rocm"]["gcn_arch"] = None
    try:
        import torch_npu  # noqa: F401  （CANN 插件：import 即注册 npu 设备）
        npu_ok = torch.npu.is_available()
        out["npu"] = {"ok": npu_ok,
                      "count": torch.npu.device_count() if npu_ok else 0,
                      "name": (torch.npu.get_device_name(0) if npu_ok else None),
                      "version": getattr(torch_npu, "__version__", None)}
    except ImportError:
        out["npu"] = {"ok": False,
                      "note": "torch_npu 未安装（CANN 适配需单独安装该插件）"}
    # DirectML（与 device.py::probe() 的择优链对齐；此前 probe_devices 无此项，
    # 导致 resolve_device('auto') 会跳过 Windows 上的 AMD/Intel GPU 直通）
    try:
        import torch_directml                                   # noqa: WPS433
        dev = torch_directml.device()
        out["dml"] = {"ok": dev is not None, "device": str(dev) if dev else None,
                      "version": getattr(torch_directml, "__version__", None)}
    except ImportError:
        out["dml"] = {"ok": False,
                      "note": "torch_directml 未安装（Windows AMD/Intel GPU 路径）"}
    return out


def _sync_device(device: str) -> None:
    """同步设备队列（**计时的前置条件**）。

    P29 修复：旧实现直接 `time.perf_counter()` 前后包住 kernel 调用，在
    NPU/CUDA 这类**异步设备**上测到的是「提交耗时」而非真实耗时——队列还没
    执行完就返回了，数字会小一个数量级且随负载剧烈波动。
    """
    dev = str(device).lower()
    try:
        if dev.startswith("npu") and hasattr(torch, "npu"):
            torch.npu.synchronize()
        elif dev.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:                                        # noqa: BLE001
        pass


def bench_readout(device: str = "auto", V: int = 73958, H: int = 3072,
                  steps: int = 50, dtype: str = "fp32",
                  use_accel: bool = True) -> dict:
    """读出热路径基准（前向 + 更新），返回 ms/token **与等效带宽**。

    P29 修正三处（否则这个工具的结论不可信）：
      ① 每个计时区间前后 `_sync_device`——异步设备上不等待测的是提交耗时；
      ② 支持 `dtype`（fp32/fp16/bf16）——此前写死 fp32，**fp16/bf16 从未被测过**，
         却按「fp16 可能更慢」下过结论；
      ③ `use_accel=True` 时用**生产同款** `phdnet.backends.accel_readout.AccelReadout`
         （AXPY 更新 + 设备侧缓存），而不是旧 `TorchReadout`；
      ④ 报告等效带宽（GB/s）——读出是 GEMV，受带宽限制，**带宽才是可移植的指标**。
    """
    import time
    rng = np.random.default_rng(0)
    # P30：旧 `TorchReadout` 已删除（被 AccelReadout 取代），因此现在**只有**
    # 生产实现可测——这正是修复的意义：基准必须测生产对象。
    if not use_accel:
        raise ValueError(
            "旧 TorchReadout 已删除（P30）；基准只能测生产实现 AccelReadout。"
            "如需对比历史实现，请从 git 历史取回。")
    from .accel_readout import AccelReadout
    core = AccelReadout(H, V, None, device=device, dtype=dtype,
                        w0=(rng.standard_normal((V, H)) * 0.01).astype("float32"))
    h = (np.abs(rng.standard_normal(H)) + 0.1).astype(np.float32)
    t = np.zeros(V, dtype=np.float32)
    t[V // 2] = 1.0
    core.forward(h)                                          # 预热（JIT/上下文）
    core.learn_softmax(h, t, 0.05)
    _sync_device(getattr(core, "device", device))
    t0 = time.perf_counter()
    for _ in range(steps):
        core.forward(h)
    _sync_device(getattr(core, "device", device))
    fwd = (time.perf_counter() - t0) / steps * 1000
    t0 = time.perf_counter()
    for _ in range(steps):
        core.learn_softmax(h, t, 0.05)
    _sync_device(getattr(core, "device", device))
    upd = (time.perf_counter() - t0) / steps * 1000
    esize = {"fp32": 4, "fp16": 2, "bf16": 2}.get(str(getattr(core, "dtype_name",
                                                               "fp32")), 4)
    # 流量口径（每步）：读 W 一次 + 读写 W 各一次
    w_bytes = V * H * esize
    total_ms = fwd + upd
    gbs = (3 * w_bytes / 1e9) / (total_ms / 1000) if total_ms > 0 else 0.0
    return {"device": getattr(core, "device", str(device)),
            "dtype": getattr(core, "dtype_name", dtype),
            "fwd_ms": fwd, "update_ms": upd, "total_ms": total_ms,
            "W_MiB": w_bytes / 2 ** 20, "eff_GBps": gbs}
