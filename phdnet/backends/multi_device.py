# -*- coding: utf-8 -*-
"""多卡自动适配（P14，2026-09-28）：设备自动解析 + 读出列并行 + 分片计划。

为什么不是 DDP / DataParallel（诚实定位，勿误读为通用多卡训练框架）
--------------------------------------------------------------------
PHD-Net 是**逐 token 事件驱动的稀疏网络**：无 batch 维、无注意力、全局状态
（WM 情景缓冲 / STDP 印迹 / LTM 长时记忆）跨样本连续携带。标准 DDP（按参数
分片 + all-reduce 梯度）在此**没有梯度可分**（无反向图、无 batch），DataParallel
亦无 batch 轴可切。因此本模块走**模型并行**路线，且只切**真正可分**的部分：

1. **读出列并行（主路线，embarrassingly parallel）**
   读出权重 W ∈ R^{V×H}，前向 y = W·h、更新 ΔW = -η·(p - t)⊗h。
   - 按**词表行（V 维）**切分到各卡：每行 dot 独立、分片更新互不依赖；
   - 每步通信仅两次小向量：全局 logits y（V×4B 下发）+ dp 分片（上行），
     V=9,219 时约 37 KB×2 ≈ 数 μs（PCIe ~25 GB/s），不构成瓶颈；
   - softmax 主回路在主卡 fp32（与单卡 P9 协议一致）。
   1B 预设读出占端到端 ~89% → N 卡理论上限 ≈ 1/(1-0.89+0.89/N)
   （N=4 → ≈3.3×）。**注意：这是读出部分的线性加速，不是全模型加速**，
   上游 PC 栈 / STDP / LTM 仍在主设备单卡计算。

2. **LTM / 印迹分片（计划层，1B 规模的自然模型并行）**
   big_ltm 2^24 神经元 × 60 边为稀疏 CSR，天然可按**神经元区间**切分：
   `shard_ranges(n_neurons, n_devices)` 给出均衡区间；跨神经元的 STDP 交互
   （共激活对）是唯一的跨卡通信需求。本机（无 GPU、256M 档）不可验证，
   仅提供计划与诚实限制说明，不提供假实现。

设备自动解析
------------
`resolve_devices("auto")`：按 **昇腾 NPU → ROCm → CUDA → DirectML → CPU** 择优
（与 `probe_devices` / `resolve_device` 同一优先级），返回**全部同型号设备**
（cuda:0..N-1），而非只取第 0 张——多卡适配的入口。显式设备不可用时
诚实报错（`allow_fallback=True` 才回退 CPU，口径与 `resolve_device` 一致）。
`max_devices>0` 时截断（多卡共享场景）。

自动选型
--------
`plan_parallel(devices, V, H)`：单卡 → 原生单卡路径（零行为变更）；
多卡 → 读出列并行（各卡行区间见 `MultiDeviceReadout.shards`），并如实报告
通信量、预期加速区间与未并行部分。`configure_host_threads()` 在多卡时把
OMP/MKL 线程数收敛到 1（避免 CPU 线程与卡内线程争抢；单卡不动）。

诚实边界
--------
- 跨设备判据是**容差一致**（不是逐位）：归约顺序不同。
- 但**同设备分片**（同一张 CPU/GPU 上按行切分再 cat）在本实现下是逐位一致的
  ——每行 dot 与分片顺序无关，softmax 在 cat 后的同一向量上计算。
  `tests/verifiers/verify_multi_device.py` 对此给逐位断言。
- fp8/fp4 档不参与列并行（码本量化路径为 CPU numba 单路，见 P9）。
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:                                            # pragma: no cover
    torch = None

from .torch_backend import probe_devices                    # noqa: E402

_DT = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
_PLATFORM_ORDER = ("npu", "rocm", "cuda", "dml")             # 择优顺序（与 P10 一致）
_PLATFORM_DEV = {"npu": "npu", "rocm": "cuda", "cuda": "cuda", "dml": "privateuseone"}


def _resolve_dtype(name: str):
    """精度名 → torch.dtype（未知精度不静默回退，诚实报错）。"""
    if torch is None:
        raise RuntimeError("未安装 torch")
    key = str(name).lower()
    if key not in _DT:
        raise ValueError(f"不支持 dtype={name!r}；可用取值：{sorted(_DT)}。")
    return getattr(torch, _DT[key])


def probe_multi() -> dict:
    """多卡视角探针：在 `probe_devices` 基础上补「同型号设备串列表」。

    返回 {平台: {ok, count, name, version, devices: [设备串, ...]}}；
    计数 < 2 的平台照常列出（单卡不算多卡，但信息完整便于报告）。
    """
    probes = probe_devices()
    out: dict = {}
    for key, info in probes.items():
        if key in ("cpu", "torch"):
            continue
        rec = dict(info)
        n = int(info.get("count") or 0)
        if info.get("ok") and n > 0:
            prefix = _PLATFORM_DEV.get(key)
            rec["devices"] = ([f"{prefix}:{i}" for i in range(n)]
                              if prefix else [])
        out[key] = rec
    out["cpu"] = {"ok": True, "count": 1, "name": "cpu",
                  "devices": ["cpu"], "note": "参考路径"}
    return out


def resolve_devices(spec: str = "auto", max_devices: int = 0,
                    allow_fallback: bool = False) -> list[str]:
    """把设备规格解析为**设备列表**（多卡入口；单卡 = 长度 1 的列表）。

    - "auto"：按 NPU → ROCm → CUDA → DirectML → CPU 择优，取**全部同型号设备**
      （cuda:0..N-1）；无可用加速器时返回 ["cpu"]（等价原单卡路径）；
    - "cpu" → ["cpu"]；"cuda:0,1" / "npu:0 1" → 显式列表（逐个校验可用）；
    - "max_devices>0" 截断（多任务共享场景）；
    - 显式设备不可用 → RuntimeError（`allow_fallback=True` 才回退 ["cpu"]）。
    """
    if torch is None:
        raise RuntimeError("未安装 torch，无法使用多卡适配")
    raw = str(spec).strip()
    probes = probe_multi()

    def _fallback(msg: str) -> list[str]:
        if allow_fallback:
            import warnings
            warnings.warn(msg + " 已按 allow_fallback=True 回退 ['cpu']。",
                          RuntimeWarning)
            return ["cpu"]
        raise RuntimeError(msg)

    if raw in ("", "auto"):
        for key in _PLATFORM_ORDER:
            rec = probes.get(key, {})
            if rec.get("ok") and rec.get("devices"):
                devs = list(rec["devices"])
                break
        else:
            return ["cpu"]
    else:
        devs = [d for d in raw.replace(" ", ",").split(",") if d]
        for d in devs:
            try:
                torch.zeros(1, device=d)
            except Exception as e:                           # noqa: BLE001
                return _fallback(
                    f"请求设备 '{d}' 不可用（{type(e).__name__}: {e}）；"
                    "多卡适配诚实报错，不做静默回退。")
    if max_devices and len(devs) > max_devices:
        devs = devs[:max_devices]
    return devs


def shard_ranges(n_items: int, n_devices: int) -> list[tuple[int, int]]:
    """把 n_items 均衡切成 n_devices 段，返回 [(start, end), ...]（空段剔除）。

    用于读出按词表行切分（V）与 LTM 按神经元区间切分（N）。
    """
    if n_devices < 1:
        raise ValueError(f"n_devices 必须 ≥1（收到 {n_devices}）")
    if n_items < 0:
        raise ValueError(f"n_items 必须 ≥0（收到 {n_items}）")
    base, rem = divmod(n_items, n_devices)
    out: list[tuple[int, int]] = []
    start = 0
    for i in range(n_devices):
        size = base + (1 if i < rem else 0)
        if size == 0:
            continue
        out.append((start, start + size))
        start += size
    return out


def configure_host_threads(n_threads: int | None = None) -> int:
    """多卡时收敛 CPU 线程数（OMP/MKL/NUMEXPR/OPENBLAS = 1），返回设置值。

    必须在 numpy/torch 初始化前调用才完全生效（对已初始化运行时为尽力而为）。
    单卡场景不调用（保持既有 BLAS 线程策略，零行为变更）。
    """
    import os
    n = 1 if n_threads is None else max(1, int(n_threads))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)
    if torch is not None:
        try:
            torch.set_num_threads(n)
        except Exception:                                    # pragma: no cover
            pass
    return n


def plan_parallel(devices: list[str], V: int, H: int,
                  readout_share: float = 0.89) -> dict:
    """自动选型 + 诚实预期报告（供训练/推理入口打印与文档引用）。

    readout_share = 读出占端到端耗时比（1B 预设实测 ≈0.89；其他配置按实测传入）。
    单卡 → 串行原生路径；多卡 → 读出列并行。
    """
    n = len(devices)
    shards = shard_ranges(V, n) if n > 1 else [(0, V)]
    per_step_bytes = V * 4 * 2                        # y 下发 + dp 上行（fp32）
    if n <= 1:
        strategy, speedup = "单设备原生路径（无并行，零行为变更）", 1.0
    else:
        strategy = f"读出列并行 ×{n}（按词表行切分；LTM 分片见 shard_ranges）"
        s = readout_share
        speedup = 1.0 / (1.0 - s + s / n)
    return {
        "devices": devices,
        "n_devices": n,
        "strategy": strategy,
        "readout_shards": shards,
        "comm_per_step_bytes": per_step_bytes if n > 1 else 0,
        "expected_speedup_readout_share": round(speedup, 3),
        "not_parallel": ["PC 栈（稀疏主循环）", "STDP 印迹", "LTM 长时记忆",
                         "分词/编码（CPU numpy）"],
        "limits": ("读出列为线性加速；上游 PC/STDP/LTM 仍在主设备单卡。"
                   "跨设备等价判据为容差一致（同设备分片为逐位一致）。"),
    }


class MultiDeviceReadout:
    """读出按 V 维分片的列并行封装（语义对齐单设备 `AccelReadout`）。

    单设备（len(devices)==1）时行为与 `AccelReadout` 完全一致。
    多设备：各卡持 W[s:e, :] 分片；前向 = 分片 matvec → 主设备 cat（无跨卡算子
    调用，行切分逐位等价）；更新 = 主卡 fp32 softmax 后把 dp 分片下发，各卡本地
    `add_(outer(dp_i, h), alpha=-eta)`（外积逐行独立，无通信依赖）。
    通信：每步 y（V×4B）下发 + dp 分片上行。
    """

    def __init__(self, W: np.ndarray, devices: list[str], dtype: str = "fp32",
                 w_clip: float = 0.0, main_device: str | None = None):
        if torch is None:
            raise RuntimeError("未安装 torch，无法使用多卡读出")
        if dtype not in _DT:
            raise ValueError(f"MultiDeviceReadout 不支持 dtype={dtype!r}；"
                             f"可用取值：{sorted(_DT)}。")
        Wc = np.ascontiguousarray(W, dtype=np.float32)
        if Wc.ndim != 2:
            raise ValueError(f"W 必须二维 (V,H)（收到 shape={Wc.shape}）")
        self.devices = list(devices)
        self.main = main_device or self.devices[0]
        self.w_clip = float(w_clip)
        self.tdtype = _resolve_dtype(dtype)
        self.V, self.H = Wc.shape
        self.shards = shard_ranges(self.V, len(self.devices))
        self.W = [torch.as_tensor(Wc[s:e, :], device=d, dtype=self.tdtype)
                  for (s, e), d in zip(self.shards, self.devices)]
        # 行区间（供分片计划与外部检查）
        self.ranges = list(self.shards)

    # ---------- 前向 ----------
    def forward(self, h) -> "torch.Tensor":
        """h：torch.Tensor（主设备）或 numpy/list；返回主设备上的 logits（fp32）。"""
        ht = self._as_h(h)
        ys = [Wi @ ht.to(Wi.device).to(Wi.dtype) for Wi in self.W]
        y = torch.cat(ys, dim=0) if len(ys) > 1 else ys[0]
        return y.float().to(self.main)

    def forward_np(self, h: np.ndarray) -> np.ndarray:
        return self.forward(h).cpu().numpy()

    # ---------- 学习 ----------
    def learn_softmax(self, h, target, eta: float,
                      y_pre: "torch.Tensor | None" = None) -> float:
        """softmax 交叉熵末端更新（返回 NLL）。target：onehot（主设备可感知形态）。"""
        ht = self._as_h(h)
        y = self.forward(ht) if y_pre is None else y_pre
        p = torch.softmax(y, dim=0)
        tgt = self._as_target(target, p.device)
        correct = int(torch.argmax(tgt).item())
        nll = float(-torch.log(p[correct] + 1e-12).item())
        # dp 分片：主卡算全局 p，各卡只更新自己的行区间
        parts = [(p[s:e] - tgt[s:e].to(p.dtype)).to(Wi.device).to(Wi.dtype)
                 for (s, e), Wi in zip(self.ranges, self.W)]
        for Wi, dp_i in zip(self.W, parts):
            Wi.add_(torch.outer(dp_i, ht.to(Wi.device).to(Wi.dtype)), alpha=-eta)
            if self.w_clip > 0.0:
                Wi.clamp_(-self.w_clip, self.w_clip)
        return nll

    def learn_softmax_np(self, h: np.ndarray, target: np.ndarray,
                         eta: float) -> float:
        ht = self._as_h(h)
        y = self.forward(ht)
        return self.learn_softmax(ht, self._as_target(target, y.device),
                                  eta, y_pre=y)

    # ---------- 权重取回（检查点兼容：拼回 (V,H) 矩阵）----------
    def to_numpy(self) -> np.ndarray:
        parts = [Wi.detach().float().cpu().numpy() for Wi in self.W]
        return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]

    @property
    def is_sharded(self) -> bool:
        return len(self.W) > 1

    def load_rows(self, W_np: np.ndarray) -> None:
        """从完整 (V,H) 矩阵灌入各分片（检查点恢复；行区间由分片计划固定）。"""
        arr = np.ascontiguousarray(W_np, dtype=np.float32)
        if arr.shape != (self.V, self.H):
            raise ValueError(f"形状不符：期望 {(self.V, self.H)}，收到 {arr.shape}")
        for (s, e), Wi in zip(self.ranges, self.W):
            Wi.copy_(torch.as_tensor(arr[s:e, :], device=Wi.device,
                                     dtype=Wi.dtype))

    # ---------- 内部工具 ----------
    def _as_h(self, h):
        if torch.is_tensor(h):
            return h.to(self.main)
        arr = np.ascontiguousarray(h, dtype=np.float32)
        return torch.as_tensor(arr, device=self.main, dtype=self.tdtype)

    def _as_target(self, target, dev):
        if torch.is_tensor(target):
            return target.to(dev)
        arr = np.ascontiguousarray(target, dtype=np.float32)
        t = torch.as_tensor(arr, device=self.main, dtype=torch.float32)
        return t.to(dev)


# ────────────────────── 能力矩阵（P18，2026-09-28）──────────────────────
# fhz 问「有 NPU 的环境为什么不用 NPU」——根因：生产训练/推理走的是 **numba**，
# 而 numba 只能编译到 **CPU 机器码**（物理限制，不支持 NPU/CUDA/ROCm）；支持
# 昇腾的是 **torch 栈**（torch_npu 插件）。此前生产入口静默走 CPU 不作提示，
# 易误判为「NPU 没被识别」。本函数给出显式矩阵与行动建议。

BACKEND_MATRIX = {
    "numba（生产：train/train.py、train/infer.py）": {
        "cpu": "yes",
        "npu": "no", "cuda": "no", "rocm": "no", "dml": "no",
        "note": "numba njit 只能编译到 CPU 机器码；分词核/读出融合核/稀疏主循环"
                "全部 CPU。NPU 上它表现为『已启用（numba nogil 线程）』但实际是 CPU。",
    },
    "加速读出（phdnet.backends.accel_readout.AccelReadout）": {
        "cpu": "yes", "npu": "yes", "cuda": "yes", "rocm": "yes",
        "dml": "yes",
        "note": "torch_npu / CUDA / ROCm / DirectML 均支持（--device auto 自动"
                "择优：昇腾 → ROCm → CUDA → DirectML → CPU）。限制：机制覆盖"
                "不全（M1–M6 + 部分 M7；big_ltm / 自适应 LR 等显式拒绝），"
                "且**权重与生产 npz 检查点不通用**。",
    },
}


def capability_report(verbose: bool = True) -> dict:
    """返回（并可打印）后端 × 设备能力矩阵 + 本机探测结果 + 建议动作。"""
    probes = probe_multi()
    avail = [k for k, v in probes.items() if v.get("ok") and v.get("count")]
    accel = [k for k in avail if k != "cpu"]
    report = {
        "matrix": BACKEND_MATRIX,
        "probed": {k: v.get("count") for k, v in probes.items()},
        "accelerators_present": accel,
        "production_backend": "numba/CPU",
        "accelerator_backend": "torch 栈",
        "action": (None if not accel else
                   f"检测到加速器 {accel}：生产入口（train_1b）仍走 numba/CPU，"
                   f"这是 numba 的物理限制，非探测失败。若要真正用上 {accel}，"
                   f"读出走 accel_readout（P19 起生产默认 auto）"
                   f"（注意机制覆盖与权重不通用，见下表 note）。"),
    }
    if verbose:
        print("[能力矩阵] 后端 × 设备：")
        for backend, row in BACKEND_MATRIX.items():
            cells = "  ".join(
                f"{d}={'✓' if row.get(d) == 'yes' else '✗'}"
                for d in ("cpu", "npu", "cuda", "rocm", "dml"))
            print(f"  · {backend}\n      {cells}")
        print(f"[能力矩阵] 本机探测：{report['probed']}")
        if report["action"]:
            print(f"[能力矩阵] {report['action']}")
        else:
            print("[能力矩阵] 本机无加速器，numba/CPU 为唯一路径（正常）")
    return report
