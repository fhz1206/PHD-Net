"""STDP 关联核的 torch 实现 —— 同一份算子覆盖 CPU / CUDA / ROCm(HIP) / 昇腾 NPU。

要点：
  1. **算法语义与 `phdnet/plasticity.STDPCore` 完全一致**：predict 为 scatter-add
     （`index_add_`），学习为「先更新 pre 迹 → 用历史 post 迹算 LTD → 再更新 post 迹」；
     仅活跃突触前行参与更新，权重截断于 [0, w_max]。
  2. **后端只改 device/dtype**：昇腾用 `npu`、ROCm 走 `cuda` 接口（torch.version.hip）、
     DirectML 用 `privateuseone`；算子代码完全相同。
  3. **启用前必须过等价性自检**：`selftest_torch(device)` 与 numpy 参考逐元素比对
     （沿用 numba 自检 bug 的教训：不能只靠「权重有没有增长」判据）。
  4. 暂不支持逐突触自适应学习率（T4.1，需逐突触状态与循环）——该开关开启时
     请使用 numpy 后端，避免语义分歧。

已知限制（诚实）：目前仅 STDP 热点（M3）完成 torch 化；M1/M2/M4/M6 仍为 numpy/numba；
1B 大容量印迹表（dict 邻接表）仍为 CPU 结构，尚未迁移到设备端稀疏张量。
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _resolve_dtype(dtype):
    """把配置里的字符串（cfg.torch_dtype）映射为 torch.dtype；未知值回退 float32。"""
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).strip().lower()
    return {
        "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float64": torch.float64, "fp64": torch.float64, "double": torch.float64,
    }.get(name, torch.float32)


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


class TorchSTDPCore:
    """torch 版稀疏 STDP 关联核（接口对齐 STDPCore：predict / step）。"""

    def __init__(self, n: int, m_edges: int, lam: float, eta: float, w_max: float,
                 rng: np.random.Generator, device: str = "cpu",
                 dtype=None, dual: bool = False, lam_slow: float = 0.85,
                 beta_slow: float = 0.5):
        if torch is None:
            raise RuntimeError("未安装 torch，无法使用 torch 后端")
        self.n, self.m = n, m_edges
        self.lam, self.eta, self.w_max = lam, eta, w_max
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype)
        self.dual, self.lam_slow, self.beta_slow = dual, lam_slow, beta_slow

        # 确定性稀疏拓扑（与 numpy 版一致：每神经元 m 条随机出边）
        idx = np.empty((n, m_edges), dtype=np.int64)
        for i in range(n):
            idx[i] = rng.choice(n, size=m_edges, replace=False)
        self.post_idx = torch.as_tensor(idx, device=device)          # (n, m) Long
        self.W = torch.zeros((n, m_edges), dtype=self.dtype, device=device)
        self.t_pre = torch.zeros(n, dtype=self.dtype, device=device)
        self.t_post = torch.zeros(n, dtype=self.dtype, device=device)
        self.t_pre_slow = torch.zeros(n, dtype=self.dtype, device=device)
        self.t_post_slow = torch.zeros(n, dtype=self.dtype, device=device)

    # ---------- 工具 ----------
    def _t(self, x: np.ndarray) -> "torch.Tensor":
        return torch.as_tensor(np.asarray(x, dtype=np.float32),
                               dtype=self.dtype, device=self.device)

    # ---------- 预测：p[k] += W[i,j]·pre[i]（scatter 语义） ----------
    def predict(self, pre_rate: np.ndarray) -> np.ndarray:
        pre = self._t(pre_rate)
        p = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        contrib = (self.W * pre[:, None]).reshape(-1)
        p.index_add_(0, self.post_idx.reshape(-1), contrib)
        return torch.clamp(p, 0.0, self.w_max).detach().cpu().numpy()

    # ---------- 一步学习 ----------
    def step(self, pre_rate: np.ndarray, post_rate: np.ndarray,
             eta_scale: float = 1.0) -> None:
        pre = self._t(pre_rate)
        post = self._t(post_rate)
        self.t_pre = self.lam * self.t_pre + pre
        if self.dual:
            self.t_pre_slow = self.lam_slow * self.t_pre_slow + pre
            self.t_post_slow = self.lam_slow * self.t_post_slow + post
            tp = self.t_pre + self.beta_slow * self.t_pre_slow
            tp_hist = self.t_post + self.beta_slow * self.t_post_slow
        else:
            tp, tp_hist = self.t_pre, self.t_post

        k = self.post_idx                                   # (n, m)
        raw = 2.0 * tp[:, None] * post[k] - tp_hist[k] * pre[:, None]
        mask = (pre > 0.0).to(self.dtype)[:, None]          # 仅活跃突触前行
        self.W = torch.clamp(self.W + self.eta * eta_scale * raw * mask,
                             0.0, self.w_max)
        if self.dual:
            self.t_post_slow = self.lam_slow * self.t_post_slow + post
        self.t_post = self.lam * self.t_post + post


def _numpy_reference(n: int, m: int, idx: np.ndarray, lam: float, eta: float,
                     w_max: float, pre: np.ndarray, post: np.ndarray,
                     steps: int = 40) -> np.ndarray:
    """numpy 参考实现（与 plasticity.STDPCore 的 numpy 回退路径一致）。"""
    W = np.zeros((n, m))
    t_pre = np.zeros(n)
    t_post = np.zeros(n)
    for _ in range(steps):
        t_pre = lam * t_pre + pre
        for i in np.nonzero(pre > 0.0)[0]:
            k = idx[i]
            raw = 2.0 * t_pre[i] * post[k] - t_post[k] * pre[i]
            np.clip(W[i] + eta * raw, 0.0, w_max, out=W[i])
        t_post = lam * t_post + post
    return W


def selftest_torch(device: str = "cpu", n: int = 128, m: int = 8,
                   atol: float = 1e-5, dtype="float32") -> bool:
    """torch 后端 vs numpy 参考的等价性自检（启用任何非 numpy 后端前应通过）。

    判据：① 权重确实增长（行为级）；② 与 numpy 参考一致（数值级）。
      - 高精度（fp32/fp64）：逐元素 allclose（atol）。
      - 低精度（fp16/bf16）：允许饱和失真，但要求与 fp32 参考**强相关**
        （Pearson r ≥ 0.99），确保算法正确、只是精度受限（B2 修复：此前
        自检固定以 fp32 运行，配置的 dtype 精度下门禁形同虚设）。

    重要（2026-09-18 修复）：此前版本在测试里**手工重算**更新式，从不调用
    `TorchSTDPCore.step()` —— 被测代码未被覆盖，自检形同虚设（假阳性门禁）。
    现改为循环调用 `core.step()`，真正检验算子实现。
    """
    if torch is None:
        return False
    tdtype = _resolve_dtype(dtype)
    rng = np.random.default_rng(0)
    idx = np.array([rng.choice(n, m, replace=False) for _ in range(n)])
    pre = np.zeros(n)
    post = np.zeros(n)
    pre[:8] = 0.8
    post[8:16] = 0.8

    core = TorchSTDPCore(n, m, 0.35, 1.0, 1.0, np.random.default_rng(1),
                         device=device, dtype=tdtype)
    core.post_idx = torch.as_tensor(idx, device=device)   # 与参考实现同一拓扑
    for _ in range(40):
        core.step(pre, post, eta_scale=1.0)               # ← 真正调用被测算子
    W_torch = core.W.detach().cpu().numpy().astype(np.float64)
    W_ref = _numpy_reference(n, m, idx, 0.35, 1.0, 1.0, pre, post, steps=40)

    grown = float(W_torch.sum()) > 0.5
    if tdtype in (torch.float16, torch.bfloat16):
        # 低精度：算法须正确（与 fp32 参考强相关），数值饱和可放宽
        if not grown:
            return False
        sd, sr = W_torch.std(), W_ref.std()
        if sd < 1e-12 or sr < 1e-12:
            return bool(grown and np.allclose(W_torch, W_ref, atol=atol, rtol=1e-4))
        corr = float(np.mean((W_torch - W_torch.mean()) * (W_ref - W_ref.mean()))
                     / (sd * sr))
        return bool(corr >= 0.99)
    same = np.allclose(W_torch, W_ref, atol=atol, rtol=1e-4)
    return bool(grown and same)
