"""硬件后端适配：昇腾 NPU / ROCm(HIP) / CUDA / DirectML / CPU 的统一探测与选择。

设计原则：
  1. **同一份算子代码覆盖所有后端**——torch 后端只改 device/dtype，不改动算法语义
     （加速读出 `AccelReadout` 的对拍见 tests/verifiers/verify_accel_readout.py）。
  2. **永远可回退**：无加速器或缺少依赖时自动回退 numpy(+numba) CPU 路径，
     默认行为与既有实现逐位一致（零回归）。
  3. **数值等价有门槛**：任何非 numpy 后端启用前必须通过对拍验证
     （`tests/verifiers/verify_accel_readout.py`；容差判据，跨设备不宣称逐位）
     与 numpy 参考实现的等价性检验（沿用 numba 自检 bug 的教训）。

生态说明（诚实）：
  - **昇腾 NPU**：需安装 CANN 工具包与 `torch_npu`（版本须与 torch 严格匹配），
    设备名为 `npu`；本文件通过 `import torch_npu` + `torch.npu.is_available()` 探测。
  - **ROCm**：AMD GPU + ROCm 版 PyTorch（`torch.version.hip` 非空），
    沿用 `cuda` 设备接口；**ROCm 官方仅支持 Linux**，Windows 上的 AMD GPU 建议
    走 DirectML（PyTorch-DirectML 插件，设备 `privateuseone`）。
  - 二者在无硬件环境下**无法实测**，本文件只负责探测、选择与给出安装指引。
"""

from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class BackendInfo:
    name: str            # numpy-cpu / torch-cpu / torch-cuda / torch-rocm / torch-npu / torch-dml
    kind: str            # cpu / cuda / rocm / npu / dml / unavailable
    device: str          # 传给 torch 的设备字符串
    torch_version: str | None
    hip: str | None
    npu_plugin: bool
    numba: bool
    verified: bool       # 是否在本机完成数值等价自检
    notes: str

    def __str__(self) -> str:  # pragma: no cover
        return (f"{self.name} (device={self.device}, torch={self.torch_version}, "
                f"verified={self.verified}) — {self.notes}")


def _torch():
    if importlib.util.find_spec("torch") is None:
        return None
    import torch  # noqa: WPS433
    return torch


def use_torch_backend(prefer: str = "auto") -> bool:
    """是否应启用 torch 后端。默认（auto 且无加速器）= False → 保持 numpy 路径零回归。

    A1 修复：显式请求某加速器种类但该种类在本机不可用时，**拒绝**并告警，
    回退 numpy(+numba) CPU 路径，而不再静默在 torch-CPU 上训练却自称目标后端。
    """
    info = probe()
    if prefer == "numpy" or info.torch_version is None:
        return False
    if prefer == "auto":
        return info.kind in ("npu", "rocm", "cuda", "dml")
    if prefer == "torch":
        # 显式 torch：有 torch 即用（设备由 probe 选出，含 CPU），与 numpy 默认形成对照
        return True
    # 显式指定加速器种类：必须本机确实可用，否则拒绝并告警
    if info.kind != prefer:
        warnings.warn(
            f"backend='{prefer}' 在本机不可用（探测到 {info.kind}），"
            f"PHD-Net 将回退 numpy(+numba) CPU 路径而非静默跑 torch-CPU。",
            RuntimeWarning)
        return False
    return True


@lru_cache(maxsize=8)
def probe(verify: bool = True) -> BackendInfo:
    """探测当前环境可用的计算后端（按 昇腾 → ROCm → CUDA → DirectML → CPU 顺序）。

    结果缓存：探测涉及 torch 导入与等价性自检，构造多个 PHDNet 时不得重复执行
    （2026-09-18 修复：此前每次构造模型都会触发一次完整探测）。
    """
    numba_ok = False
    try:                                    # numba 是否可用（CPU 加速路径）
        from .plasticity import NUMBA_OK    # noqa: WPS433
        numba_ok = bool(NUMBA_OK)
    except Exception:
        numba_ok = False

    torch = _torch()
    if torch is None:
        return BackendInfo("numpy-cpu", "cpu", "cpu", None, None, False, numba_ok,
                           True, "未安装 torch；使用 numpy(+numba) CPU 路径（默认，零回归）")

    hip = getattr(torch.version, "hip", None)
    npu_plugin = importlib.util.find_spec("torch_npu") is not None
    verified = False
    notes = ""

    # ── 昇腾 NPU（最高优先级；可用则直接返回）─────────────────────────────
    if npu_plugin:
        try:
            import torch_npu  # noqa: F401,WPS433  （注册 npu 设备）
            if hasattr(torch, "npu") and torch.npu.is_available():
                verified = _verify(torch, "npu", hip) if verify else False
                return BackendInfo("torch-npu", "npu", "npu", torch.__version__, hip,
                                   True, numba_ok, verified,
                                   "检测到昇腾 NPU（torch_npu 可用）")
            notes = "已装 torch_npu 但 NPU 不可用（检查 CANN 驱动与 npu-smi）"
        except Exception as e:
            notes = f"torch_npu 导入失败：{type(e).__name__}: {e}"

    # ⚠ 2026-10-07 修复（审计 P1-7）+ **同日返工**：这里原来是 `elif hip` —— 只要机器**装了
    #   torch_npu 但没有 NPU 硬件**，上面的 npu 块设完 notes 就把整条 elif 链
    #   跳过，**同机的 ROCm/CUDA 直接被忽略**（实测：fake torch_npu +
    #   `torch.cuda.is_available()` 桩 True → 返回 torch-cpu；拔掉 torch_npu
    #   同样条件 → torch-cuda）。NPU 不可用只该是**附注**，不能短路后续探测；
    #   且 torch_backend.probe_devices 是各平台独立探测、不受此影响 →
    #   两套「统一口径」在同机上会给出不同答案（torch_lm 注释却声称一致）。
    # ⚠ 2026-10-07：**把 notes 带进这两个 return** —— 否则 npu 块里记下的
    #   「已装 torch_npu 但 NPU 不可用」这类附注会在这里被**静默丢弃**，
    #   诊断信息只在最终的 torch-cpu 分支才带上（交叉验证发现的口径不一致）。
    _suffix = f"；{notes}" if notes else ""
    if hip:
        if torch.cuda.is_available():
            verified = _verify(torch, "cuda", hip) if verify else False
            return BackendInfo("torch-rocm", "rocm", "cuda", torch.__version__, hip,
                               False, numba_ok, verified,
                               f"检测到 ROCm/HIP 运行时 {hip}（ROCm 走 cuda 设备接口）"
                               + _suffix)
        notes = f"torch 为 ROCm 构建（hip={hip}）但未检测到可用设备"
        _suffix = ""
    elif torch.cuda.is_available():
        verified = _verify(torch, "cuda", hip) if verify else False
        return BackendInfo("torch-cuda", "cuda", "cuda", torch.__version__, hip,
                           False, numba_ok, verified,
                           "检测到 NVIDIA CUDA 设备" + _suffix)

    # DirectML（Windows 上的 AMD/Intel GPU 路径）
    if importlib.util.find_spec("torch_directml") is not None:
        try:
            import torch_directml  # noqa: WPS433
            dev = torch_directml.device()
            if dev is not None:
                return BackendInfo("torch-dml", "dml", str(dev), torch.__version__,
                                   hip, False, numba_ok, False,
                                   "检测到 torch_directml（Windows AMD/Intel GPU 路径，未做等价自检）")
        except Exception as e:
            notes += f"；DirectML 不可用（{type(e).__name__}）"

    verified = _verify(torch, "cpu", hip) if verify else False
    return BackendInfo("torch-cpu", "cpu", "cpu", torch.__version__, hip, False,
                       numba_ok, verified,
                       (notes + "；" if notes else "") +
                       "未检测到加速器，回退 torch CPU 路径（numpy/numba 仍为默认）")


def _verify(torch, device: str, hip) -> bool:
    """torch 后端等价性自检 —— P30 后**恒为 False**（自检已随栈删除）。

    2026-10-07 清理：原实现是「先 `raise NotImplementedError` 再留一行不可达的
    `return bool(selftest_torch(...))`」—— `selftest_torch` 已不存在，那行是
    死代码（还会误导读者以为自检还能跑）。等价性对拍的新归属见 docstring 提示。
    """
    # selftest_torch 已随旧 torch 栈删除（P30）；等价性对拍改由
    # tests/verifiers/verify_accel_readout.py 承担。
    return False


def select_backend(prefer: str = "auto") -> BackendInfo:
    """按偏好选择后端。

    prefer:
      auto  —— 有加速器则用加速器（昇腾/ROCm/CUDA/DirectML），否则 numpy CPU
      numpy —— 强制 numpy(+numba) CPU（默认行为，零回归）
      torch —— 强制 torch 后端（用 probe() 选出的最佳设备）
      npu / rocm / cuda / cpu —— 指定种类（不可用时回退并说明）
    """
    info = probe()
    if prefer in ("auto", "", None):
        return info
    if prefer == "numpy":
        return BackendInfo("numpy-cpu", "cpu", "cpu", info.torch_version, info.hip,
                           False, info.numba, True, "显式指定 numpy CPU 路径")
    if prefer in ("torch", "rocm", "npu", "cuda", "dml", "cpu"):
        if prefer == "torch":
            return info
        if info.kind == prefer:
            return info
        # A1 修复：显式请求的加速器不可用，不再自称目标后端，明确标注 unavailable
        return BackendInfo(
            f"{prefer}-unavailable", "unavailable", info.device,
            info.torch_version, info.hip, False, info.numba, False,
            f"指定 {prefer} 但本机不可用（探测到 {info.kind}），请在具备对应硬件的环境运行；"
            f"当前回退 {info.name}")
    raise ValueError(f"未知后端偏好: {prefer}")


GUIDANCE = {
    "npu": ("昇腾 NPU：① 安装与固件匹配的 CANN 工具包；"
            "② pip install torch_npu（版本须与 torch 严格对应）；"
            "③ 运行 backend_probe.py 确认 torch.npu.is_available()。"),
    "rocm": ("ROCm（AMD GPU）：① Linux + ROCm 驱动（ROCm 官方不支持 Windows）；"
             "② 安装 ROCm 版 PyTorch（torch.version.hip 非空）；"
             "③ Windows 上的 AMD GPU 可改用 torch_directml 插件。"),
    "cuda": "CUDA：安装带 CUDA 的 PyTorch 版本（torch.version.cuda 非空）。",
    "dml": "DirectML：pip install torch-directml，适用于 Windows 上的 AMD/Intel GPU。",
    "cpu": "CPU：numpy + numba 路径（默认，零回归）；或 torch CPU 路径。",
}
