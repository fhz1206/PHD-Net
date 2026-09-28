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
        # 用**解析后**的 self.device 建张量（原用原始 device 参数，绕过了
        # _resolve_device 的回退契约：device='npu' 无硬件时直接 RuntimeError，
        # 而同一层的 TorchReadout 会正常回退 cpu + 告警）。
        dev = self.device
        self.post_idx = torch.as_tensor(idx, device=dev)             # (n, m) Long
        self.W = torch.zeros((n, m_edges), dtype=self.dtype, device=dev)
        self.t_pre = torch.zeros(n, dtype=self.dtype, device=dev)
        self.t_post = torch.zeros(n, dtype=self.dtype, device=dev)
        self.t_pre_slow = torch.zeros(n, dtype=self.dtype, device=dev)
        self.t_post_slow = torch.zeros(n, dtype=self.dtype, device=dev)

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
        # .float()：numpy 无 bfloat16 位型，bf16 存储下须先升到 fp32 再转 numpy；
        # fp32/fp16 路径下 .float() 为 no-op，逐位不变。
        return torch.clamp(p, 0.0, self.w_max).detach().float().cpu().numpy()

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


def _as_step_sequence(pre, post, steps: int):
    """把 (pre, post) + steps 归一成 [(pre, post), ...] 序列。

    既有调用方传「常量刺激 + 步数」；新增的自检可传逐步变化的序列
    （贴近真实 LM 训练：发放率逐帧变化）以获得更强判别力。
    """
    if isinstance(pre, (list, tuple)):
        seq = list(pre)
        if len(seq) != steps:
            raise ValueError(f"序列长度 {len(seq)} != steps {steps}")
        return seq
    return [(pre, post)] * steps


def _numpy_reference(n: int, m: int, idx: np.ndarray, lam: float, eta: float,
                     w_max: float, pre, post=None, steps: int = 40) -> np.ndarray:
    """numpy 参考实现（与 plasticity.STDPCore 的 numpy 回退路径一致）。

    对应 formB 更新式 dw = η·s·(2·tp·post − tp_hist·pre)。
    ⚠ 该式**不是** numpy 端的生产口径：`plasticity.STDPCore.step` 在
    NUMBA_OK 时走 numba 核，把 η·s 折进 t_pre 参数、eta 位置传 1.0，
    实际更新式为 formA（见 torch_lm._ProdSTDPCore）。
    验证生产类请用 `selftest_torch(cls="_ProdSTDPCore", reference="numba")`。
    """
    W = np.zeros((n, m))
    t_pre = np.zeros(n)
    t_post = np.zeros(n)
    for pre_i, post_i in _as_step_sequence(pre, post, steps):
        t_pre = lam * t_pre + pre_i
        for i in np.nonzero(pre_i > 0.0)[0]:
            k = idx[i]
            raw = 2.0 * t_pre[i] * post_i[k] - t_post[k] * pre_i[i]
            np.clip(W[i] + eta * raw, 0.0, w_max, out=W[i])
        t_post = lam * t_post + post_i
    return W


def _numba_path_reference(n: int, m: int, idx: np.ndarray, lam: float,
                          eta: float, w_max: float, pre, post=None,
                          steps: int = 40) -> np.ndarray:
    """numpy 端**生产主路径**参考（`plasticity.STDPCore.step`，含 numba 核）。

    formA：dw = 2·(η·s·tp)·post − tp_hist·pre。直接调用真实实现而非重算，
    因此该参考随 numpy 端演进而保持诚实（重算式会重新引入同样的分歧）。
    """
    from .. import plasticity as pl                                  # 惰性 import
    core = pl.STDPCore(n, m, lam, eta, w_max, np.random.default_rng(1))
    core.post_idx = idx.copy()
    for pre_i, post_i in _as_step_sequence(pre, post, steps):
        core.step(pre_i, post_i, eta_scale=1.0)
    return np.asarray(core.W, dtype=np.float64)


def _resolve_core_cls(cls):
    """把 selftest_torch 的 `cls` 参数解析为 STDP 类（支持惰性字符串引用）。"""
    if cls is None:
        return TorchSTDPCore
    if isinstance(cls, str):
        # 惰性字符串：避免 torch_lm ↔ torch_backend 循环导入
        if cls in ("_ProdSTDPCore", "ProdSTDPCore"):
            from .torch_lm import _ProdSTDPCore       # noqa: PLC0415
            return _ProdSTDPCore
        raise ValueError(f"未知的 STDP 类名：{cls!r}")
    return cls


def _make_stimulus(n: int, steps: int, kind: str):
    """构造自检刺激序列 [(pre, post), ...]（确定性，可复现）。

    - `"constant"`：pre/post 不重叠的常量发放率（历史默认，保持既有门禁行为）。
    - `"walk"`：随机稀疏发放率逐帧游走（贴近真实 LM 训练）。常量刺激在
      η<1 时 LTP 项被 η 压制、权重易全零退化为平凡解，判别力弱；游走刺激
      在真实 η 下能激发 LTP/LTD 竞争，是验证生产类的推荐刺激。
    """
    if kind == "constant":
        pre, post = np.zeros(n), np.zeros(n)
        pre[:8] = 0.8
        post[8:16] = 0.8
        return [(pre, post)] * steps
    if kind == "walk":
        rs = np.random.default_rng(7)
        seq = []
        for _ in range(steps):
            a = (rs.random(n) < 0.15).astype(np.float64) * rs.random(n) * 0.9
            b = (rs.random(n) < 0.15).astype(np.float64) * rs.random(n) * 0.9
            seq.append((a, b))
        return seq
    raise ValueError(f"未知的 stimulus 种类：{kind!r}（'constant'/'walk'）")


def selftest_torch(device: str = "cpu", n: int = 128, m: int = 8,
                   atol: float = 1e-5, dtype="float32", cls=None,
                   reference: str = "numpy", eta: float = 1.0,
                   stimulus: str = "constant", steps: int = 40) -> bool:
    """torch 后端 vs numpy 参考的等价性自检（启用任何非 numpy 后端前应通过）。

    判据：① 权重确实增长（行为级）；② 与 numpy 参考一致（数值级）。
      - 高精度（fp32/fp64）：逐元素 allclose（atol）。
      - 低精度（fp16/bf16）：允许饱和失真，但要求与 fp32 参考**强相关**
        （Pearson r ≥ 0.99），确保算法正确、只是精度受限（B2 修复：此前
        自检固定以 fp32 运行，配置的 dtype 精度下门禁形同虚设）。

    重要（2026-09-18 修复）：此前版本在测试里**手工重算**更新式，从不调用
    `TorchSTDPCore.step()` —— 被测代码未被覆盖，自检形同虚设（假阳性门禁）。
    现改为循环调用 `core.step()`，真正检验算子实现。

    `cls` / `reference`（2026-09-28 修复生产类零覆盖）：
      - `cls`：被测 STDP 类，默认 `TorchSTDPCore`；传 `"_ProdSTDPCore"`
        （字符串，惰性 import）或类本身可指向 `torch_lm` 的生产类。
      - `reference`：`"numpy"` = `_numpy_reference`（formB 口径，与
        `TorchSTDPCore` 同式）；`"numba"` = `_numba_path_reference`
        （formA 口径，numpy 端生产主路径）。**验证 `_ProdSTDPCore` 必须用
        `"numba"`**，否则参考式不同、必然失败。
      - `eta`：STDP 学习率，默认 1.0（保持既有门禁行为不变）。⚠ η=1 时
        formA 与 formB 数值重合，该自检**无法**区分两种更新式；要对生产类
        做有判别力的门禁，须传真实学习率（库默认 `eta_stdp=0.03`）。
      - `stimulus`：`"constant"`（历史默认）/`"walk"`（逐帧游走发放率，
        真实 η 下能激发 LTP/LTD 竞争，判别力强）。验证生产类推荐
        `cls="_ProdSTDPCore", reference="numba", eta=0.03, stimulus="walk"`。
    """
    if torch is None:
        return False
    core_cls = _resolve_core_cls(cls)
    tdtype = _resolve_dtype(dtype)
    rng = np.random.default_rng(0)
    idx = np.array([rng.choice(n, m, replace=False) for _ in range(n)])
    seq = _make_stimulus(n, steps, stimulus)

    core = core_cls(n, m, 0.35, eta, 1.0, np.random.default_rng(1),
                    device=device, dtype=tdtype)
    core.post_idx = torch.as_tensor(idx, device=device)   # 与参考实现同一拓扑
    for pre, post in seq:
        core.step(pre, post, eta_scale=1.0)               # ← 真正调用被测算子
    W_torch = core.W.detach().float().cpu().numpy().astype(np.float64)
    if reference == "numba":
        W_ref = _numba_path_reference(n, m, idx, 0.35, eta, 1.0, seq, steps=steps)
    elif reference == "numpy":
        W_ref = _numpy_reference(n, m, idx, 0.35, eta, 1.0, seq, steps=steps)
    else:
        raise ValueError(f"未知的 reference 口径：{reference!r}（'numpy'/'numba'）")

    # 行为级判据：权重确有变化。阈值取「相对参考总变化量」而非绝对 0.5，
    # 否则真实 η=0.03 下总增长仅 ~2，恒判失败。
    ref_change = float(np.abs(W_ref).sum())
    thr = max(0.5, 0.05 * ref_change)
    grown = float(np.abs(W_torch).sum()) > thr
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


# ---------------------------------------------------------------------------
# P10 硬件后端适配（2026-09-26，fhz 指令：适配 CUDA / CANN(NPU) / ROCm）
# ---------------------------------------------------------------------------
def probe_devices() -> dict:
    """统一加速器探针：返回 {平台: 可用信息}。

    - CUDA：torch.cuda.is_available()（NVIDIA，sm ≥ 7.0 建议）；
    - ROCm：torch.cuda.is_available() 且 torch.version.hip 非空（AMD，走 HIP 化的
      cuda 接口，算子代码与 CUDA 完全相同）；
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
    out["cuda" if not is_hip else "rocm"] = {
        "ok": cuda_ok,
        "count": torch.cuda.device_count() if cuda_ok else 0,
        "name": (torch.cuda.get_device_name(0) if cuda_ok else None),
        "version": (torch.version.hip if is_hip else torch.version.cuda),
    }
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


class TorchReadout:
    """词级读出热路径（M6）的 torch 化——前向 matvec + softmax 梯度更新。

    覆盖 CUDA / ROCm / NPU / CPU 四类设备（同一份算子代码，仅 device/dtype 不同）；
    精度：fp32（默认）/ fp16 / bf16 原生支持；fp8 需 CUDA ≥ 8.9（Ada/Hopper）且
    torch ≥ 2.1 的 float8_e4m3fn（CPU 无原生 fp8 → 该档在 CPU 上不可用，诚实降级）；
    fp4 无 torch 原生类型 → 走 CPU numba 量化码本路径（见 phdnet/readout.py P9）。

    语义对应 `phdnet.readout.Readout.learn_softmax`（softmax 交叉熵末端梯度）；
    数值协议：前向与梯度更新在设备 dtype 上执行，softmax 概率与 NLL 在 fp32
    计算（低精度存储 + 高精度主回路，与 P9 CPU 方案一致）。等价性判据为
    **容差一致**（设备归约顺序与 numpy 不同，逐位等价在跨设备场景不成立——
    这是与 CPU 内部优化的本质区别，须在报告中显式声明）。
    """

    _DT = {"fp32": "fp32", "fp16": "fp16", "bf16": "bf16"}

    def __init__(self, W: np.ndarray, device: str = "auto",
                 dtype: str = "fp32"):
        if torch is None:
            raise RuntimeError("torch 未安装，无法使用 torch 后端")
        self.device = _resolve_device("cpu" if device == "auto" else device)
        if dtype not in self._DT:      # 不再静默回退 fp32（掩盖拼写错误）
            raise ValueError(
                f"TorchReadout 不支持 dtype={dtype!r}；"
                f"可用取值：{sorted(self._DT)}。")
        self.tdtype = _resolve_dtype(self._DT[dtype])
        self.W = torch.as_tensor(np.ascontiguousarray(W, dtype=np.float32),
                                 device=self.device, dtype=self.tdtype)

    def forward(self, h: np.ndarray) -> np.ndarray:
        ht = torch.as_tensor(np.ascontiguousarray(h, dtype=np.float32),
                             device=self.device, dtype=self.tdtype)
        y = self.W @ ht
        return y.float().cpu().numpy()                       # 主回路升精度

    def learn_softmax(self, h: np.ndarray, target: np.ndarray,
                      eta: float) -> float:
        ht = torch.as_tensor(np.ascontiguousarray(h, dtype=np.float32),
                             device=self.device, dtype=self.tdtype)
        tt = torch.as_tensor(np.ascontiguousarray(target, dtype=np.float32),
                             device=self.device, dtype=self.tdtype)
        y = self.W @ ht
        y32 = y.float()
        p = torch.softmax(y32, dim=0)
        correct = int(torch.argmax(tt).item())
        nll = float(-torch.log(p[correct] + 1e-12).item())
        dp = (p - tt).to(self.tdtype)                        # 梯度降回存储精度
        g = torch.outer(dp, ht)                              # 存储精度上的外积
        if self.tdtype == torch.bfloat16:
            g = g.float(); self.W.add_(g.to(torch.bfloat16), alpha=-eta)
        else:
            self.W.add_(g, alpha=-eta)
        return nll

    def to_cpu(self) -> np.ndarray:
        """权重回读（fp32 快照，供检查点保存）。"""
        return self.W.float().cpu().numpy()


def bench_readout(device: str = "auto", V: int = 9219, H: int = 3072,
                  steps: int = 50) -> dict:
    """读出热路径基准（前向 + 更新，1B 预设真实读出规模），返回 ms/token。"""
    import time
    rng = np.random.default_rng(0)
    core = TorchReadout(rng.standard_normal((V, H)) * 0.01, device=device)
    h = np.abs(rng.standard_normal(H)) + 0.1
    t = np.zeros(V); t[V // 2] = 1.0
    core.forward(h)                                          # 预热（JIT/上下文）
    core.learn_softmax(h, t, 0.05)
    t0 = time.perf_counter()
    for _ in range(steps):
        core.forward(h)
    fwd = (time.perf_counter() - t0) / steps * 1000
    t0 = time.perf_counter()
    for _ in range(steps):
        core.learn_softmax(h, t, 0.05)
    upd = (time.perf_counter() - t0) / steps * 1000
    return {"device": core.device, "dtype": str(core.tdtype),
            "fwd_ms": fwd, "update_ms": upd, "total_ms": fwd + upd}
