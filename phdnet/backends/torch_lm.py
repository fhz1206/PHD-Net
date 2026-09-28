"""词级 LM 网络全栈的 torch 化 —— M1–M6 同一份算子覆盖 CPU / CUDA / ROCm / 昇腾 NPU。

================================================================================
设计要点（诚实声明）
================================================================================
1. **device 无关**：`resolve_device()` 统一解析 auto/cpu/cuda/rocm/npu；
   显式请求不可用设备时**诚实报错**（不静默回退 CPU；`allow_fallback=True`
   才在告警后回退）。算子代码全部设备无关（仅 device/dtype 不同）。

2. **同 seed 等价的初始化**：torch 栈按 `PHDNet.__init__` 的**同一顺序**消费
   同一 `np.random.default_rng(cfg.seed)`（M1 → M2 CSR 拓扑+权重 → M3 STDP
   出边拓扑 → M6 读出 normal）。M1/M2 直接实例化 numpy 参考实现
   （`SparseEncoder` / `SparsePCStack`）抽取初始化数组后搬运到设备——
   保证第 0 步权重与 numpy 版逐位一致，等价性对比从零点起步。

3. **各机制的神经认知对应物**（铁律②，无自注意力 / 无位置编码 / 无堆叠层）：
     M1 V1 稀疏编码      —— k-WTA 竞争（`TorchSparseEncoder`）
     M2 皮层预测编码     —— CSR 稀疏主干 matvec / 误差驱动 ΔW / Oja（`TorchSparsePC`）
     M3 STDP 时序        —— 复用 `TorchSTDPCore`（同 rng 消耗序列）
     M4 海马/前额叶      —— 门控 WM（`TorchWM`）+ 快慢权重 LTM（`TorchLTM`）
     M5 蓝斑/胆碱能调制  —— surprise→Welford z→gate（直接复用 `Neuromodulator`，
                            标量状态设备无关、与 numpy 逐位一致）
     M6 IT→前运动读出    —— softmax 感知器末端梯度（`TorchReadoutDense`，
                            fp32 主回路；fp16/bf16 存储可选）

4. **数据侧保持 CPU numpy**：分词器（`WordTokenizer`，确定性整数哈希 SDR）
   留在 CPU——哈希是整数运算，与 numpy 逐位一致；每步仅传输 n_input 维向量。

5. **数值协议**：网络状态默认 fp32（cfg.torch_dtype）；softmax/NLL 在 fp32
   主回路（numpy 版为 fp64 主回路——这是跨实现数值差异的主来源，容差判据
   见 ci/verifiers/verify_torch_lm.py：PPL 相对差 <1%，权重范数轨迹强相关）。

已知限制（诚实）：
  - `readout_dtype` 支持 fp32/fp16/bf16 三档（2026-09-28 修复：此前 fp16/bf16
    构造成功但训练必崩——`TorchReadoutDense` 裸 `self.W @ h`，torch matmul 不做
    类型提升，`readout_dtype != torch_dtype` 即报 addmv dtype 不一致。现已在
    forward/learn_softmax 内显式统一 dtype，softmax 仍在 fp32 主回路算完再转回）；
    fp8 需 CUDA≥8.9 真机、fp4 无 torch 原生类型（fp4 另已在 config 层禁用）；
  - 主干恒为稀疏 CSR 语义（`TorchSparsePC` 只实现 CSR 边表示）；稠密 PC 栈
    已于 2026-09-28 按 fhz 指令删除（sparse_conn=False 在 config 层 fail-fast）；
  - T4.1 逐突触自适应 / STDP 稳态 / 元可塑性 / E-I 突触 / big_ltm 大容量表 /
    retrieval_topk / 两级读出 / 内容寻址 WM 未迁移（显式 NotImplementedError）；
  - M3 复用 `TorchSTDPCore` 的 numpy↔torch 每步互转（n_top 维小向量，CPU 上
    开销可忽略；CUDA 真机上为已知优化点）；
  - `w_max` 截断、k-WTA 阈值、Oja 更新等均为逐元素/scatter 语义，与 numpy 版
    **容差一致**（归约顺序不同，不宣称跨设备逐位）。
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .torch_backend import TorchSTDPCore, _resolve_dtype, probe_devices

if torch is not None:
    from ..config import PHDNetConfig
    from ..modulator import Neuromodulator
    from ..sparse_encoder import SparseEncoder
    from ..sparse_pc import SparsePCStack
    from ..word_encoder import WordSegmenter, WordTokenizer

__all__ = ["TorchWordLM", "TorchPHDNet", "resolve_device"]


# ---------------------------------------------------------------------------
# M3 STDP：复用 TorchSTDPCore 拓扑/迹存储，step 复刻 numpy 主路径（numba 热路径）语义
# ---------------------------------------------------------------------------
class _ProdSTDPCore(TorchSTDPCore):
    """生产语义 STDP —— 与 `plasticity.STDPCore.step` 的 **numba 热路径**一致。

    ⚠ 诚实标注（2026-09-27 对齐排查发现）：`model.py` 调用 numba 核时把
    `eta·eta_scale·tp` 折进核的 t_pre 参数、eta 位置传 1.0，故 numpy 主路径的
    实际更新式为
        dw = 2·(η·s·tp[i])·post[k] − tp_hist[k]·pre[i]   （clip 到 [0, w_max]）
    这与 `TorchSTDPCore.step` / `plasticity.STDPCore` 的 numpy 回退路径
    （dw = η·s·(2·tp[i]·post[k] − tp_hist[k]·pre[i])，三方自检 _numpy_reference
    亦按该式）**不一致**——是仓库内预先存在的实现分歧，非本文件引入。
    本类为对齐 numpy 主路径基线（4K 锚点 90.2480 口径）选择复刻 numba 热路径；
    predict（scatter + clamp）与迹更新顺序仍与 TorchSTDPCore 完全一致。
    """

    def step(self, pre_rate: np.ndarray, post_rate: np.ndarray,
             eta_scale: float = 1.0) -> None:
        pre = self._t(pre_rate)
        post = self._t(post_rate)
        self.t_pre = self.lam * self.t_pre + pre
        tp, tp_hist = self.t_pre, self.t_post
        k = self.post_idx                                   # (n, m)
        raw = 2.0 * (self.eta * eta_scale * tp[:, None]) * post[k] \
            - tp_hist[k] * pre[:, None]
        mask = (pre > 0.0).to(self.dtype)[:, None]          # 仅活跃突触前行
        self.W = torch.clamp(self.W + raw * mask, 0.0, self.w_max)
        self.t_post = self.lam * self.t_post + post


# ---------------------------------------------------------------------------
# 设备解析（诚实报错，不静默回退）
# ---------------------------------------------------------------------------
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
    if dev in ("", "auto"):
        probes = probe_devices()
        for key in ("npu", "rocm", "cuda"):
            if probes.get(key, {}).get("ok"):
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
class TorchSparseEncoder:
    """k-WTA 稀疏编码（脑对应 V1）：u = W·x + b → top-k 归一到 [0.1, 1.1]。"""

    def __init__(self, ref: "SparseEncoder", device: str, dtype: "torch.dtype"):
        self.k = ref.k
        self.W = torch.as_tensor(ref.W, dtype=dtype, device=device)
        self.b = torch.as_tensor(ref.b, dtype=dtype, device=device)
        self._hn = torch.as_tensor(ref._hn, dtype=dtype, device=device)

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        u = self.W @ x + self.b
        win, idx = torch.topk(u, self.k)
        s = torch.zeros_like(u)
        s[idx] = (win - win.min()) / (win.max() - win.min() + 1e-9) + 0.1
        return s

    def learn(self, x: "torch.Tensor", s: "torch.Tensor", eta: float) -> None:
        """T2.2 Foldiak 局部规则（默认关闭）：活跃行 Oja + 行范数稳态。"""
        act = torch.nonzero(s > 0.0).squeeze(1)
        if act.numel() == 0:
            return
        sa = s[act]
        Wact = self.W[act]
        self.W[act] = Wact + eta * (sa[:, None] * (x[None, :] - sa[:, None] * Wact))
        nrm = torch.linalg.vector_norm(self.W[act], dim=1).clamp_min(1e-12)
        self.W[act] *= (self._hn[act] / nrm)[:, None]


# ---------------------------------------------------------------------------
# M2 预测编码栈（CSR 稀疏主干 torch 化：前向 matvec / 误差 ΔW / Oja）
# ---------------------------------------------------------------------------
class TorchSparsePC:
    """结构性稀疏预测编码栈（语义对齐 `SparsePCStack`，边表示 CSR 扁平化）。

    每条存在的边 (row → idx) 存一个权重；前向 scatter-add（index_add_），
    学习向量化更新（val += eta·a[row]·b[idx]），只触达存在的突触。
    """

    def __init__(self, ref: "SparsePCStack", device: str, dtype: "torch.dtype"):
        self.eta_pc, self.eta_oja, self.w_max = ref.eta_pc, ref.eta_oja, ref.w_max
        self.n0, self.n1, self.n2 = ref.n0, ref.n1, ref.n2
        self.k0, self.k1 = ref.k0, ref.k1
        self.up0 = self._edges(ref.up0, device, dtype)
        self.up1 = self._edges(ref.up1, device, dtype)
        self.dn0 = self._edges(ref.dn0, device, dtype)
        self.dn1 = self._edges(ref.dn1, device, dtype)
        self._hn_up0 = torch.as_tensor(ref._hn_up0, dtype=dtype, device=device)
        self._hn_up1 = torch.as_tensor(ref._hn_up1, dtype=dtype, device=device)
        self._hn_dn0 = torch.as_tensor(ref._hn_dn0, dtype=dtype, device=device)
        self._hn_dn1 = torch.as_tensor(ref._hn_dn1, dtype=dtype, device=device)
        self.act_ema = None

    @staticmethod
    def _edges(csr, device, dtype) -> dict:
        indptr, idx, val = csr
        counts = np.diff(indptr)
        row = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        return {
            "idx": torch.as_tensor(idx, device=device),
            "row": torch.as_tensor(row, device=device),
            "val": torch.as_tensor(val, dtype=dtype, device=device),
            "n_rows": int(len(counts)),
        }

    def _matvec(self, e: dict, x: "torch.Tensor") -> "torch.Tensor":
        y = torch.zeros(e["n_rows"], dtype=x.dtype, device=x.device)
        y.index_add_(0, e["row"], e["val"] * x[e["idx"]])
        return y

    def n_synapses(self) -> int:
        return int(sum(len(e["val"]) for e in
                       (self.up0, self.up1, self.dn0, self.dn1)))

    # ---------- 推理（与 SparsePCStack.infer 逐条对应） ----------
    def infer(self, s0: "torch.Tensor", n_steps: int = 1) -> dict:
        r1 = torch.tanh(self._matvec(self.up0, s0))
        r2 = torch.tanh(self._matvec(self.up1, r1))
        for _ in range(n_steps):
            e1 = r1 - self._matvec(self.dn1, r2)
            d2 = torch.clamp(self._matvec(self.up1, e1), -0.5, 0.5)
            r2 = torch.tanh(r2 + 0.15 * d2)
            e0 = s0 - self._matvec(self.dn0, r1)
            d1 = torch.clamp(self._matvec(self.up0, e0), -0.5, 0.5)
            r1 = torch.tanh(r1 + 0.15 * d1)
        e0 = s0 - self._matvec(self.dn0, r1)
        e1 = r1 - self._matvec(self.dn1, r2)
        return {"s0": s0, "r1": r1, "r2": r2, "e0": e0, "e1": e1}

    # ---------- 学习（误差驱动 ΔW_dn + Oja ΔW_up，只更新存在的边） ----------
    def learn(self, cache: dict, eta_scale: float = 1.0,
              homeostasis: bool = False) -> None:
        if self.eta_pc == 0.0 and self.eta_oja == 0.0 and not homeostasis:
            return
        s0, r1, r2 = cache["s0"], cache["r1"], cache["r2"]
        e0, e1 = cache["e0"], cache["e1"]
        if homeostasis:
            e0 = e0 / torch.linalg.vector_norm(e0).clamp_min(1e-9)
            e1 = e1 / torch.linalg.vector_norm(e1).clamp_min(1e-9)
        eta = self.eta_pc * eta_scale
        for e, a, b in ((self.dn0, e0, r1), (self.dn1, e1, r2)):
            e["val"] += eta * a[e["row"]] * b[e["idx"]]
        eo = self.eta_oja * eta_scale
        for e, post, pre in ((self.up0, r1, s0), (self.up1, r2, r1)):
            pi = post[e["row"]]
            e["val"] += eo * pi * (pre[e["idx"]] - pi * e["val"])
        for e in (self.up0, self.up1, self.dn0, self.dn1):
            e["val"].clamp_(-self.w_max, self.w_max)
        if homeostasis:
            self._homeostatic_scale()

    def _homeostatic_scale(self) -> None:
        """Turrigiano 突触缩放：每行存在的权重范数拉回初始值。"""
        for e, target in ((self.up0, self._hn_up0), (self.up1, self._hn_up1),
                          (self.dn0, self._hn_dn0), (self.dn1, self._hn_dn1)):
            ss = torch.zeros(e["n_rows"], dtype=e["val"].dtype, device=e["val"].device)
            ss.index_add_(0, e["row"], e["val"] * e["val"])
            nrm = ss.sqrt().clamp_min(1e-12)
            e["val"] *= (target / nrm)[e["row"]]


# ---------------------------------------------------------------------------
# M4a 工作记忆 / M4b 长期记忆（torch 化）
# ---------------------------------------------------------------------------
class TorchWM:
    """门控工作记忆（脑对应 PFC 持续放电 + 基底核门控）。"""

    def __init__(self, n: int, n_slots: int, gamma: float,
                 device: str, dtype: "torch.dtype"):
        self.slots = torch.zeros(n_slots, n, dtype=dtype, device=device)
        self.strength = torch.zeros(n_slots, dtype=dtype, device=device)
        self.gamma = gamma

    def decay(self) -> None:
        self.slots *= self.gamma
        self.strength *= self.gamma

    def _pick_weakest(self) -> int:
        """最弱槽位选择 —— 复刻 numpy `WorkingMemory._pick_weakest` 的
        `np.argsort(strength)[0]` 语义（并列时 introsort 的顺序与 argmin 的
        首最小值可能不同，实测会导致槽位选择分歧，故直接用 numpy 对齐）。"""
        order = np.argsort(self.strength.detach().float().cpu().numpy())
        return int(order[0])

    def write(self, r: "torch.Tensor", gate: float, thresh: float) -> bool:
        if gate < thresh:
            return False
        slot = self._pick_weakest()
        self.slots[slot] = r
        self.strength[slot] = gate
        return True

    def read(self) -> "torch.Tensor":
        total = self.strength.sum()
        if float(total) < 1e-9:
            return torch.zeros_like(self.slots[0])
        return (self.strength[:, None] * self.slots).sum(dim=0) / total


class TorchLTM:
    """双速率长期记忆（脑对应海马快印迹 + 皮层慢巩固 + 吸引子检索）。"""

    def __init__(self, n: int, eta_hip: float, eta_cortex: float,
                 beta_fast: float, beta_slow: float, n_steps: int,
                 device: str, dtype: "torch.dtype"):
        self.n = n
        self.eta_hip, self.eta_cortex = eta_hip, eta_cortex
        self.beta_fast, self.beta_slow = beta_fast, beta_slow
        self.n_steps = n_steps
        self.W_fast = torch.zeros(n, n, dtype=dtype, device=device)
        self.W_slow = torch.zeros(n, n, dtype=dtype, device=device)

    def imprint(self, p: "torch.Tensor") -> None:
        self.W_fast += self.eta_hip * torch.outer(p, p)
        self.W_fast.fill_diagonal_(0.0)
        self.W_fast.clamp_(-1.0, 1.0)

    def recall(self, cue: "torch.Tensor") -> "torch.Tensor":
        W = self.beta_slow * self.W_slow + self.beta_fast * self.W_fast
        r = torch.where(torch.sign(cue) == 0, torch.ones_like(cue), torch.sign(cue))
        for _ in range(self.n_steps):
            r = torch.where(torch.sign(W @ r) == 0, torch.ones_like(r),
                            torch.sign(W @ r))
        return r

    def consolidate(self, forget: float = 1.0) -> None:
        self.W_slow = (1.0 - self.eta_cortex) * self.W_slow \
            + self.eta_cortex * self.W_fast
        self.W_fast *= forget


# ---------------------------------------------------------------------------
# M6 读出（softmax 感知器；fp32 主回路，fp16/bf16 存储可选）
# ---------------------------------------------------------------------------
class TorchReadoutDense:
    """稠密读出（语义对齐 `Readout.learn_softmax` 的稠密 fp32 路径）。"""

    def __init__(self, W: np.ndarray, device: str, dtype: "torch.dtype",
                 w_clip: float = 0.0):
        self.W = torch.as_tensor(np.ascontiguousarray(W, dtype=np.float32),
                                 device=device, dtype=dtype)
        self.w_clip = float(w_clip)

    def forward(self, h: "torch.Tensor") -> "torch.Tensor":
        # 显式统一 dtype：网络状态 dtype（cfg.torch_dtype）可与读出存储精度
        # （cfg.readout_dtype）不同，而 torch matmul 不做类型提升，直接
        # `self.W @ h` 会抛 "addmv input tensors must have the same dtype"。
        # fp32 默认路径下 .to() 为 no-op（返回同一张量），逐位不变。
        return self.W @ h.to(self.W.dtype)

    def learn_softmax(self, h: "torch.Tensor", target: "torch.Tensor",
                      eta: float, y_pre: "torch.Tensor | None" = None) -> float:
        ht = h.to(self.W.dtype)                  # 同上：统一到读出存储精度
        y = self.W @ ht if y_pre is None else y_pre
        p = torch.softmax(y.float(), dim=0)      # softmax 主回路恒为 fp32
        correct = int(torch.argmax(target).item())
        nll = float(-torch.log(p[correct] + 1e-12).item())
        dp = (p - target.float()).to(self.W.dtype)
        self.W.add_(torch.outer(dp, ht), alpha=-eta)
        if self.w_clip > 0.0:
            self.W.clamp_(-self.w_clip, self.w_clip)
        return nll


# ---------------------------------------------------------------------------
# 网络全栈：TorchPHDNet（语义对齐 PHDNet.step 的默认路径）
# ---------------------------------------------------------------------------
class TorchPHDNet:
    """PHD-Net 单步推理+局部学习的 torch 全栈（M1–M6）。

    未迁移机制（显式拒绝，不做静默近似）：adaptive_lr / stdp_homeostasis /
    metaplasticity / ei_synapses / big_ltm / retrieval_topk / readout_hidden /
    readout_conn_k / lognormal_init / wm_summary_every / readout_recurrence /
    segment_check / multi_modulation / dual_trace / learnable_encoder /
    critical_period / task_modulation / auto_development /
    pc_predictive_target / neuron_target_rate。

    ⚠ `sparse_conn` 单独处理（见下）：torch 栈**恒用稀疏 CSR 主干语义**
    （`TorchSparsePC` 只实现 CSR 边表示，无稠密路径），故 `sparse_conn=False`
    （要求稠密主干）显式 NotImplementedError；而库默认 `sparse_conn=True`
    与 torch 栈行为一致，正常放行。
    """

    def __init__(self, cfg: "PHDNetConfig", device: str = "cpu",
                 dtype=None, rng: "np.random.Generator | None" = None):
        if torch is None:
            raise RuntimeError("未安装 torch，无法使用 torch LM 栈")
        self.cfg = cfg
        self.device = device
        self.dtype = _resolve_dtype(dtype or cfg.torch_dtype)
        rng = rng or np.random.default_rng(cfg.seed)
        _unsupported = []
        for name in ("adaptive_lr", "stdp_homeostasis", "metaplasticity",
                     "ei_synapses", "big_ltm", "retrieval_topk", "readout_hidden",
                     "readout_conn_k", "lognormal_init", "wm_summary_every",
                     "readout_recurrence", "segment_check", "multi_modulation",
                     # 以下 7 项 numpy 端真实生效（见 model.py:325-398），
                     # 但 step() 从未实现 → 此前被静默忽略（构造成功、
                     # 训练照跑，但与 numpy 逐张量不同）。按模块 docstring
                     # 「显式拒绝，不做静默近似」的承诺改为构造时拒绝。
                     "dual_trace", "learnable_encoder", "critical_period",
                     "task_modulation", "auto_development",
                     "pc_predictive_target", "neuron_target_rate"):
            v = getattr(cfg, name, 0)
            if v and (v is True or (isinstance(v, (int, float)) and v > 0)):
                _unsupported.append(name)
        if _unsupported:
            raise NotImplementedError(
                f"torch LM 栈暂不支持机制：{_unsupported}（请关闭或使用 numpy 后端）")
        # sparse_conn：config.__post_init__ 已拦 False（稠密 PC 栈 2026-09-28 按
        # fhz 指令删除）；torch 栈恒用稀疏 CSR 主干语义（TorchSparsePC 只实现
        # CSR 边表示），与库默认 True 一致。
        if cfg.readout_dtype in ("fp8", "fp4"):
            raise NotImplementedError(
                "torch 栈 readout_dtype 暂不支持 fp8/fp4（fp8 需 CUDA≥8.9 真机；"
                "fp4 无 torch 原生类型），请用 fp32/fp16/bf16。")

        # ---- M1：numpy 参考实现抽取同 seed 初始化 → 搬运设备（逐位一致起点）
        enc = SparseEncoder(cfg.n_input, cfg.n_sdr, cfg.k_sparse, rng)
        self.encoder = TorchSparseEncoder(enc, device, self.dtype)
        # ---- M2：同 seed CSR 拓扑+权重 → 边表示搬运设备
        pc = SparsePCStack(cfg.n_sdr, cfg.n_mid, cfg.n_top, cfg.eta_pc, cfg.eta_oja,
                           rng, conn_k=cfg.conn_k,
                           lognormal_init=cfg.lognormal_init,
                           exc_ratio=cfg.exc_ratio)
        self.pc = TorchSparsePC(pc, device, self.dtype)
        # ---- M3：复用 TorchSTDPCore（同 rng 消耗序列）；step 用生产（numba
        #      热路径）语义，见 _ProdSTDPCore 的诚实标注
        self.stdp = _ProdSTDPCore(
            cfg.n_top, cfg.m_lateral, cfg.lambda_trace, cfg.eta_stdp, cfg.w_max,
            rng, device=device, dtype=self.dtype, dual=cfg.dual_trace,
            lam_slow=cfg.lambda_trace_slow, beta_slow=cfg.beta_slow_trace)
        # ---- M4a / M4b
        self.wm = TorchWM(cfg.n_top, cfg.n_wm_slots, cfg.gamma_wm, device, self.dtype)
        self.ltm = TorchLTM(cfg.n_top, cfg.eta_hip, cfg.eta_cortex, cfg.beta_fast,
                            cfg.beta_slow, cfg.n_recall_steps, device, self.dtype)
        # ---- M5：标量调制器（Welford z）设备无关，直接复用 numpy 语义
        self.modulator = Neuromodulator(cfg.mod_gain)
        # ---- M6：读出（同 seed normal 初始化；与 Readout 默认路径一致）
        if cfg.readout_dtype not in ("fp32", "fp16", "bf16"):
            raise NotImplementedError(f"readout_dtype={cfg.readout_dtype} 不支持")
        self.ro_dtype = _resolve_dtype(cfg.readout_dtype)
        n_h = cfg.n_top * (3 if cfg.pred_in_readout else 2)
        W0 = rng.normal(0.0, 0.05, (cfg.n_readout, n_h))
        self.readout = TorchReadoutDense(W0, device, self.ro_dtype,
                                         w_clip=cfg.readout_w_clip)
        self.n_out = cfg.n_readout if cfg.n_readout > 0 else cfg.n_input

        # ---- 状态（与 PHDNet 对齐）
        self._ro_eta = cfg.eta_readout
        self._prev_rate = np.zeros(cfg.n_top)
        self._last_rate = np.zeros(cfg.n_top)
        self._exp = 0.0
        self.learn_pc = True
        self.learn_stdp = True
        self.step_count = 0

    # ---------- k-WTA 发放率（与 PHDNet._rate 同语义） ----------
    def _rate(self, r2: "torch.Tensor") -> "torch.Tensor":
        full = (r2 + 1.0) * 0.5
        k = max(1, r2.numel() // 16)
        thresh = torch.kthvalue(full, full.numel() - k + 1).values
        return torch.where(full >= thresh, full, torch.zeros_like(full))

    @staticmethod
    def _sign01(v: "torch.Tensor") -> "torch.Tensor":
        s = torch.sign(v)
        return torch.where(s == 0, torch.ones_like(s), s)

    # ---------- 单步（readonly=True 冻结全部持久状态，供评估复现） ----------
    def step(self, x, target=None, learn: bool = True, readonly: bool = False,
             learn_scale: float = 1.0) -> dict:
        cfg = self.cfg
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32),
                                dtype=self.dtype, device=self.device)
        s0 = self.encoder.encode(x)                          # 1. M1
        cache = self.pc.infer(s0, cfg.n_infer_steps)         # 2. M2
        r2 = cache["r2"]
        rate = self._rate(r2)
        rate_np = rate.detach().float().cpu().numpy().astype(np.float64)

        pred = self.stdp.predict(self._prev_rate)            # 3. M3（诊断）
        pred_feat = self.stdp.predict(self._last_rate)       # T3.1 读出预测特征
        pred_t = torch.as_tensor(np.asarray(pred_feat, dtype=np.float32),
                                 dtype=self.dtype, device=self.device)
        cos = float((torch.as_tensor(pred, device=self.device) * rate).sum().item()
                    / (np.linalg.norm(pred) * float(rate.norm().item()) + 1e-9)) \
            if float(np.linalg.norm(pred)) > 0 else 0.0
        seq_err = 1.0 - cos

        e0 = cache["e0"]
        surprise = float(torch.linalg.vector_norm(e0).item()) / math.sqrt(e0.numel())
        if not readonly:                                     # 4. M5
            gate, mode = self.modulator.observe(surprise)
        else:
            gate, mode = 0.0, ""

        if not readonly:                                     # 5. M4a
            self.wm.decay()
            wm_written = self.wm.write(r2, gate, cfg.gate_thresh)
        else:
            wm_written = False

        recall_hit = False                                   # 6. M4b
        if not readonly:
            if learn and mode == "encode" and gate > cfg.ltm_imprint_gate:
                self.ltm.imprint(self._sign01(rate))
            elif mode == "retrieve" and self.step_count % 8 == 0:
                rec = self.ltm.recall(self._sign01(rate))
                recall_hit = True
                self.wm.write(rec, 0.5, 0.0)

        parts = [r2, self.wm.read()]                         # 7. M6
        if cfg.pred_in_readout:
            parts.append(pred_t)
        h = torch.cat(parts)
        y = self.readout.forward(h)
        nll = 0.0
        if target is not None and learn:
            tgt = torch.as_tensor(np.ascontiguousarray(target, dtype=np.float32),
                                  dtype=self.ro_dtype, device=self.device)
            nll = self.readout.learn_softmax(h, tgt, self._ro_eta * learn_scale,
                                             y_pre=y)
        if learn and cfg.eta_readout_anneal > 0.0:
            self._ro_eta = max(cfg.eta_readout_floor,
                               self._ro_eta * cfg.eta_readout_anneal)

        if learn:                                            # 8. 局部学习 ×调制门
            mod_scale = 0.3 + 0.7 * gate
            if self.learn_pc:
                self.pc.learn(cache, eta_scale=mod_scale,
                              homeostasis=cfg.homeostasis)
            if self.learn_stdp:
                self.stdp.step(self._prev_rate, rate_np, eta_scale=mod_scale)
            self._prev_rate = rate_np
            self._exp += 1.0
        if not readonly:
            self._last_rate = rate_np
            self.step_count += 1

        return {"seq_err": seq_err, "surprise": surprise, "gate": gate,
                "mode": mode, "wm_written": wm_written, "recall_hit": recall_hit,
                "nll": nll, "y": y, "r2": r2, "pred": pred, "rate": rate}

    def sleep(self, forget: float = 1.0) -> None:
        """睡眠巩固：快→慢 EMA + 可选遗忘（staged_sleep 未迁移）。"""
        self.ltm.consolidate(forget)

    # ---------- 权重快照（等价性验证 / 检查点） ----------
    def weight_norms(self) -> dict:
        return {
            "readout": float(torch.linalg.vector_norm(
                self.readout.W.float()).item()),
            "stdp_w_sum": float(self.stdp.W.float().sum().item()),
            "pc_up0": float(torch.linalg.vector_norm(
                self.pc.up0["val"].float()).item()),
        }

    def state_arrays(self) -> dict:
        """全部持久状态 → CPU numpy（npz 检查点用）。"""

        def npy(t) -> np.ndarray:
            return t.detach().float().cpu().numpy()

        d = {
            "enc_W": npy(self.encoder.W), "enc_b": npy(self.encoder.b),
            "pc_up0_val": npy(self.pc.up0["val"]),
            "pc_up1_val": npy(self.pc.up1["val"]),
            "pc_dn0_val": npy(self.pc.dn0["val"]),
            "pc_dn1_val": npy(self.pc.dn1["val"]),
            "stdp_W": self.stdp.W.detach().float().cpu().numpy(),
            "stdp_t_pre": self.stdp.t_pre.detach().float().cpu().numpy(),
            "stdp_t_post": self.stdp.t_post.detach().float().cpu().numpy(),
            "wm_slots": npy(self.wm.slots), "wm_strength": npy(self.wm.strength),
            "ltm_W_fast": npy(self.ltm.W_fast), "ltm_W_slow": npy(self.ltm.W_slow),
            "ro_W": self.readout.W.detach().float().cpu().numpy(),
            "prev_rate": self._prev_rate, "last_rate": self._last_rate,
            "mod_mu": np.array([self.modulator.mu]),
            "mod_m2": np.array([self.modulator.m2]),
            "mod_count": np.array([self.modulator.count]),
            "ro_eta": np.array([self._ro_eta]),
            "_exp": np.array([self._exp]),
            "step_count": np.array([self.step_count]),
        }
        return d

    def load_state_arrays(self, d: dict) -> None:
        def put(t, v):
            t.copy_(torch.as_tensor(np.asarray(v, dtype=np.float32),
                                    device=t.device, dtype=t.dtype))
        put(self.encoder.W, d["enc_W"]); put(self.encoder.b, d["enc_b"])
        put(self.pc.up0["val"], d["pc_up0_val"])
        put(self.pc.up1["val"], d["pc_up1_val"])
        put(self.pc.dn0["val"], d["pc_dn0_val"])
        put(self.pc.dn1["val"], d["pc_dn1_val"])
        put(self.stdp.W, d["stdp_W"])
        put(self.stdp.t_pre, d["stdp_t_pre"])
        put(self.stdp.t_post, d["stdp_t_post"])
        put(self.wm.slots, d["wm_slots"]); put(self.wm.strength, d["wm_strength"])
        put(self.ltm.W_fast, d["ltm_W_fast"]); put(self.ltm.W_slow, d["ltm_W_slow"])
        put(self.readout.W, d["ro_W"])
        self._prev_rate = np.asarray(d["prev_rate"], dtype=np.float64)
        self._last_rate = np.asarray(d["last_rate"], dtype=np.float64)
        self.modulator.mu = float(d["mod_mu"][0])
        self.modulator.m2 = float(d["mod_m2"][0])
        self.modulator.count = float(d["mod_count"][0])
        self._ro_eta = float(d["ro_eta"][0])
        # _exp（发育经验计数）此前漏出/漏恢复，round-trip 后归零；
        # 向后兼容旧检查点（无该键则不恢复）。
        if "_exp" in d:
            self._exp = float(d["_exp"][0])
        self.step_count = int(d["step_count"][0])


# ---------------------------------------------------------------------------
# 词级 LM（接口对齐 PHDWordLM：train_stream / evaluate / tokenize / generate）
# ---------------------------------------------------------------------------
class TorchWordLM:
    """PHD-Net 词级 LM 的 torch 全栈版（数据侧 CPU，网络计算在 device 上）。"""

    def __init__(self, vocab_text_or_tokenizer, cfg: "PHDNetConfig | None" = None,
                 device: str = "auto", seg_kwargs: dict | None = None,
                 allow_fallback: bool = False):
        if torch is None:
            raise RuntimeError("未安装 torch，无法使用 torch LM 栈")
        cfg = cfg or PHDNetConfig()
        if isinstance(vocab_text_or_tokenizer, WordTokenizer):
            self.tok = vocab_text_or_tokenizer
        else:
            self.tok = WordTokenizer(vocab_text_or_tokenizer, n_sdr=cfg.n_sdr,
                                     n_active=cfg.k_sparse, seed=cfg.seed,
                                     seg_kwargs=seg_kwargs)
        self.n_sdr = cfg.n_sdr
        self.device = resolve_device(device, allow_fallback=allow_fallback)
        inner = PHDNetConfig(**{**cfg.__dict__,
                                "n_input": 2 * cfg.n_sdr,     # T2.3 组合输入
                                "n_readout": len(self.tok),
                                "readout_softmax": True})
        self.cfg = inner
        self.net = TorchPHDNet(inner, device=self.device, dtype=cfg.torch_dtype)

    def tokenize(self, text: str) -> list[str]:
        return self.tok.seg.tokenize(text)

    # ---------- 一趟语料遍历（语义对齐 PHDWordLM._pass） ----------
    def _pass(self, toks: list[str], keep: list[bool], log_every: int,
              sleep_every: int, weights: list[float] | None = None) -> list[float]:
        stoi = self.tok.stoi
        nlls, seg = [], []
        prev: str | None = None
        for i in range(len(toks) - 1):
            if not keep[i]:                       # 跳过位置仍推进上下文
                prev = toks[i] if toks[i] in stoi else prev
                continue
            # OOV 安全（2026-09-28 修复，与 numpy 主实现同口径）：cur/nxt 任一端
            # OOV 整步跳过，prev 保持最后已知 token（旧实现只挡目标端 → cur OOV
            # 时 encode_composite KeyError）。
            if toks[i] not in stoi or toks[i + 1] not in stoi:
                continue
            ls = 1.0 if weights is None else weights[i]
            d = self.net.step(self.tok.encode_composite(toks[i], prev),
                              target=self.tok.onehot(stoi[toks[i + 1]]),
                              learn=True, learn_scale=ls)
            seg.append(d["nll"])
            prev = toks[i]
            if sleep_every and (i + 1) % sleep_every == 0:
                self.net.sleep()
            if len(seg) >= log_every:
                nlls.append(float(np.mean(seg)))
                seg = []
        if seg:
            nlls.append(float(np.mean(seg)))
        return nlls

    def train_stream(self, text: str, log_every: int = 1000, sleep_every: int = 0,
                     curriculum_top: float = 0.0,
                     valid_text: str | None = None) -> list[float]:
        """在线训练：逐 token 喂入，预测下一 token（接口对齐 PHDWordLM）。"""
        if valid_text is not None and self.net.cfg.plateau_sleep:
            raise NotImplementedError("torch 栈暂未迁移 plateau_sleep（T4.4）")
        toks = self.tokenize(text)
        all_keep = [True] * len(toks)
        if curriculum_top <= 0.0:
            return self._pass(toks, all_keep, log_every, sleep_every, None)
        freq = Counter(toks)
        ranks = sorted(freq.values(), reverse=True)
        thr = ranks[max(0, int(curriculum_top * len(ranks)) - 1)]
        weights = [1.0 if freq[toks[i + 1]] >= thr else 0.25
                   for i in range(len(toks) - 1)]
        return self._pass(toks, all_keep, log_every, sleep_every, weights)

    def evaluate(self, text: str) -> dict:
        """只推理。返回 token/字符归一 PPL 与 bpc（OOV 安全，与 CPU 版一致）。"""
        toks = self.tokenize(text)
        stoi = self.tok.stoi
        total_nll, n_chars, prev = 0.0, 0, None
        n_tok, oov = 0, 0
        for i in range(len(toks) - 1):
            cur, nxt = toks[i], toks[i + 1]
            if cur not in stoi or nxt not in stoi:   # OOV：不计 NLL、不计字符分母（同 numpy 口径）
                oov += 1
                continue
            y = self.net.step(self.tok.encode_composite(cur, prev),
                              learn=False, readonly=True)["y"]
            y32 = y.float()
            p = torch.softmax(y32 - y32.max(), dim=0)
            total_nll += -math.log(float(p[stoi[nxt]].item()) + 1e-12)
            n_chars += len(nxt)
            n_tok += 1
            prev = cur
        return {
            "ppl_token": float(math.exp(total_nll / max(1, n_tok))),
            "ppl_char": float(math.exp(total_nll / max(1, n_chars))),
            "bpc": float(total_nll / max(1, n_chars) / math.log(2)),
            "n_tok": n_tok,
            "oov": oov,
            "oov_rate": oov / max(1, oov + n_tok),
        }

    def generate(self, seed_token: str, n_tokens: int = 200, tau: float = 0.8,
                 rng: "np.random.Generator | None" = None) -> list[str]:
        """自回归采样（温度 τ；情景缓冲 T3.3 未迁移）。"""
        rng = rng or np.random.default_rng(0)
        if seed_token not in self.tok.stoi:
            raise KeyError(f"seed_token={seed_token!r} 不在词表中（词表规模 "
                           f"{len(self.tok)}；OOV 种子无法编码，诚实报错）")
        out = [seed_token]
        prev: str | None = None
        cur = seed_token
        for _ in range(n_tokens):
            y = self.net.step(self.tok.encode_composite(cur, prev),
                              learn=False)["y"] / max(tau, 1e-6)
            p = torch.softmax(y.float() - y.float().max(), dim=0).cpu().numpy()
            nxt = self.tok.tokens[int(rng.choice(len(p), p=p))]
            out.append(nxt)
            prev, cur = cur, nxt
        return out
