# -*- coding: utf-8 -*-
"""加速器读出（M6 读出头）——把计算密集部件放到 NPU/CUDA（P19，2026-09-28）。

fhz 要求「计算类型的东西就要跑在 NPU 上」。读出是唯一值得迁移的部件：
1B 预设里它占端到端 ~89%（V×H = 73,958×3,072 fp32 ≈ 908 MB），且**逐 token
只有一个 h 向量**参与计算 → 每步只需传 12 KB 上行 + 少量下行，通信可忽略。

**为什么只迁读出、不迁全部**（诚实边界）：
- 生产路径的 PC 栈 / STDP / LTM 是 numba（事件驱动稀疏 + 在线 CSR 生长），
  numba 只能编译 CPU；torch 栈虽有 M1–M6 实现，但缺 `big_ltm` 等 7 项机制，
  整体迁移会**丢机制**，且权重与生产 ckpt 不通用。
- 分词/编码是变长字符串处理，NPU 不擅长（要整段上传 + 变长输出），留 CPU。

因此本模块提供**混合执行**：数据侧（分词/编码）CPU + 读出 NPU。

精度判据：跨设备为**容差一致**（归约顺序不同，不宣称逐位）；同设备
（torch CPU vs torch CPU）逐位一致。验证见 `tests/verifiers/verify_accel_readout.py`。

接口与 `phdnet.readout.Readout` 对齐（forward / learn_softmax / W / n_synapses），
因此可作为 `PHDNetConfig.accel_readout` 开关的后端；**无加速器时自动回落原
numba 路径（默认路径逐位不变）**。
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:                                            # pragma: no cover
    torch = None

_DT = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def resolve_accel_device(spec: str = "auto") -> str:
    """解析加速器设备（auto = 昇腾 → ROCm → CUDA → DirectML → CPU）。

    复用 `phdnet.backends.torch_lm.resolve_device`（同一优先级、同一诚实报错
    口径）；torch 缺失时抛 RuntimeError（调用方回落 CPU 路径）。
    """
    if torch is None:
        raise RuntimeError("未安装 torch，加速读出不可用（应回落 CPU 路径）")
    from .torch_lm import resolve_device
    return resolve_device(spec)


class AccelReadout:
    """设备无关（cpu / npu / cuda / rocm）的稠密读出，接口对齐 `Readout`。

    W **常驻设备**（NPU 上 908 MB 级），每步只往返 h 与 y：
      - forward：h（CPU numpy）→ 设备 → y → CPU numpy；
      - learn_softmax：主回路 fp32（与 P9 协议一致），梯度更新在设备上完成。
    """

    def __init__(self, n_h: int, n_out: int, rng=None, device: str = "auto",
                 dtype: str = "fp32", w_clip: float = 0.0, w0=None):
        if torch is None:
            raise RuntimeError("未安装 torch，加速读出不可用")
        if dtype not in _DT:
            raise ValueError(f"不支持 dtype={dtype!r}；可用 {sorted(_DT)}")
        self.device = resolve_accel_device(device)
        self.tdtype = getattr(torch, _DT[dtype])
        self.n_out, self.n_h = int(n_out), int(n_h)
        self.w_clip = float(w_clip)
        if w0 is None:
            g = rng if rng is not None else np.random.default_rng(0)
            init = (g.normal(0.0, 0.05, (self.n_out, self.n_h)) if hasattr(g, "normal")
                    else np.asarray(g).reshape(self.n_out, self.n_h))
        else:
            init = np.asarray(w0, dtype=np.float32)
        self.W = torch.as_tensor(np.ascontiguousarray(init, dtype=np.float32),
                                 device=self.device, dtype=self.tdtype)

    # ---------- 前向 ----------
    def forward(self, h) -> np.ndarray:
        ht = self._to_dev(h)
        return (self.W @ ht).float().cpu().numpy()

    def forward_dev(self, h):
        """设备内前向（返回设备张量，省一次回传；供流水/批量场景）。"""
        return self.W @ self._to_dev(h)

    # ---------- 学习（softmax 感知器，局部梯度 p − t）----------
    def learn_softmax(self, h, target, eta: float, y_pre=None,
                      accumulate: int = 1) -> float:
        ht = self._to_dev(h)
        y = self.W @ ht if y_pre is None else y_pre
        y32 = y.float()
        p = torch.softmax(y32, dim=0)
        t = self._to_dev(target).float()
        correct = int(torch.argmax(t).item())
        nll = float(-torch.log(p[correct] + 1e-12).item())
        dp = (p - t).to(self.tdtype)
        self.W.add_(torch.outer(dp, ht), alpha=-eta)
        if self.w_clip > 0.0:
            self.W.clamp_(-self.w_clip, self.w_clip)
        return nll

    # ---------- 与 numba Readout 的接口兼容 ----------
    def n_synapses(self) -> int:
        """稠密读出的「连接数」= 元素数（与 Readout.dense 口径一致）。"""
        return int(self.W.numel())

    def W_cpu(self) -> np.ndarray:
        """权重拉回主机（检查点 / 统计用）。"""
        return self.W.detach().float().cpu().numpy()

    def load_W(self, arr: np.ndarray) -> None:
        """从主机矩阵灌入（检查点恢复；形状须匹配）。"""
        a = np.ascontiguousarray(arr, dtype=np.float32)
        if a.shape != (self.n_out, self.n_h):
            raise ValueError(f"形状不符：期望 {(self.n_out, self.n_h)}，"
                             f"收到 {a.shape}")
        self.W.copy_(torch.as_tensor(a, device=self.device, dtype=self.tdtype))

    # ---------- 内部 ----------
    def _to_dev(self, x):
        if torch.is_tensor(x):
            return x.to(device=self.device, dtype=self.tdtype)
        return torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32),
                               device=self.device, dtype=self.tdtype)


def pick_readout_backend(cfg, n_h: int, n_out: int, rng):
    """按 `cfg.accel_readout` 选读出后端；不可用时**回落 numba 原路径**。

    返回 (readout, 后端名)。回落必须静默安全：无加速器 / torch 缺失 /
    构造异常 → 原 `Readout`（默认路径逐位不变）。
    """
    spec = str(getattr(cfg, "accel_readout", "auto") or "auto").lower()
    if spec in ("", "cpu", "off", "numba"):
        from ..readout import Readout
        return Readout(n_h, n_out, rng, w_clip=cfg.readout_w_clip,
                       dtype=cfg.readout_dtype, conn_k=cfg.readout_conn_k,
                       lognormal_init=cfg.lognormal_init), "numba-cpu"
    if spec == "auto":
        try:
            from .multi_device import probe_multi
            probes = probe_multi()
            has_accel = any(v.get("ok") and v.get("count")
                            for k, v in probes.items() if k != "cpu")
        except Exception:                                    # noqa: BLE001
            has_accel = False
        if not has_accel:
            from ..readout import Readout
            return Readout(n_h, n_out, rng, w_clip=cfg.readout_w_clip,
                           dtype=cfg.readout_dtype, conn_k=cfg.readout_conn_k,
                           lognormal_init=cfg.lognormal_init), "numba-cpu"
        spec = "auto"                                       # 交给 resolve_device 择优
    try:
        return AccelReadout(n_h, n_out, rng, device=spec,
                            dtype=cfg.readout_dtype,
                            w_clip=cfg.readout_w_clip), f"accel:{spec}"
    except Exception as e:                                   # noqa: BLE001
        from ..readout import Readout
        fb = Readout(n_h, n_out, rng, w_clip=cfg.readout_w_clip,
                     dtype=cfg.readout_dtype, conn_k=cfg.readout_conn_k,
                     lognormal_init=cfg.lognormal_init)
        fb._accel_fallback_reason = f"{type(e).__name__}: {e}"   # 供日志如实报告
        return fb, "numba-cpu(回落)"
