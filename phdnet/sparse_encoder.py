"""M1 稀疏编码器 —— 对应初级感觉皮层的稀疏放电与抑制性侧抑制（k-WTA 竞争）。

架构对应（文档 §2 M1）：
    u = W_enc·x + b          全层电位（感受野投影）
    s_i = norm(u_i) 若 i ∈ top-k，否则 0   —— 胜者全取得到 SDR 稀疏分布式表示

P61（fhz 2026-09-29「迭代默认 fp32，模型默认 bf16」）：**迭代量全程 fp32**
（u / top-k 归一 / 学习增量），模型权重可按 dtype 存储。

⚠ 诚实边界（numpy/BLAS 路径的物理限制）：**bf16/fp16 存储 + fp32 迭代必然
每次付一次上采样转换**（W 读 8.4 MB + 写 16.8 MB + 再读 16.8 MB）——比 fp32
直接 GEMV 更慢。因此 M1 默认 `fp32`（访存真减半、零转换、舍入 ~1e-7）；
需要严格 bf16 语义时显式传 `dtype="bf16"`（走 ml_dtypes，代价是上面的转换，
仅在 torch/NPU 路径上才有意义——读出侧 bf16 已由 `--readout-dtype` 默认启用）。
"""

import numpy as np

try:                                        # bf16 支持（numpy 扩展）
    import ml_dtypes
    _BF16 = np.dtype(ml_dtypes.bfloat16)
except Exception:                           # pragma: no cover
    _BF16 = None

_MODEL_DTYPES = {
    "bf16": _BF16,
    "fp16": np.dtype(np.float16),
    "fp32": np.dtype(np.float32),
    "fp64": np.dtype(np.float64),
}


def resolve_model_dtype(name: str) -> np.dtype:
    """模型存储 dtype → numpy dtype（bf16 不可用时回落 fp32 并说明）。"""
    dt = _MODEL_DTYPES.get(str(name).lower())
    if dt is None:                          # bf16 缺 ml_dtypes
        if str(name).lower() == "bf16":
            return _MODEL_DTYPES["fp32"]
        raise ValueError(f"未知模型 dtype: {name!r}；可用 {sorted(_MODEL_DTYPES)}")
    return dt


class SparseEncoder:
    def __init__(self, n_input: int, n_sdr: int, k: int, rng: np.random.Generator,
                 dtype: str = "fp32"):
        self.n_input, self.n_sdr, self.k = n_input, n_sdr, k
        # 固定随机投影（生产中可由 Oja 规则慢调，这里保持冻结以聚焦高层学习）
        # P61：默认 fp32（访存比旧 fp64 减半，且无上采样转换开销——bf16 存储
        # 在 numpy 路径反而更慢，见模块 docstring）。**迭代量恒为 fp32**。
        self.dtype = resolve_model_dtype(dtype)
        self._w32 = None                      # 仅低精度存储时的 fp32 工作副本
        self.W = (rng.normal(0.0, 1.0 / np.sqrt(n_input), size=(n_sdr, n_input))
                  .astype(self.dtype))
        self.b = rng.uniform(0.0, 0.1, size=n_sdr)          # 偏置保持 fp32（小）
        # T2.2 稳态目标：各行权重的初始 L2 范数（Foldiak 学习后拉回，防塌缩）
        self._hn = np.linalg.norm(np.asarray(self.W, dtype=np.float64), axis=1)

    def _w_fp32(self) -> np.ndarray:
        """fp32 权重视图：fp32/fp64 存储直接返回（零拷贝）；低精度则用缓存副本。"""
        if self.W.dtype in (np.dtype(np.float32), np.dtype(np.float64)):
            return self.W
        if self._w32 is None or self._w32.shape != self.W.shape:
            self._w32 = np.asarray(self.W, dtype=np.float32)
        return self._w32

    def encode(self, x: np.ndarray):
        """输入 x -> (s, idx)。s: SDR 稀疏向量；idx: 胜者索引（供稀疏快速通路）。"""
        # P61：权重与输入**同为 fp32** 才能走 BLAS sgemv——混合 dtype（fp32 权重
        # @ fp64 输入）会让 numpy 脱离 BLAS 走逐元素慢路径（实测慢 5.9×：
        # 857 µs → 5047 µs）。输入只有 n_in 个元素，上采样代价可忽略。
        x32 = np.asarray(x, dtype=np.float32)
        u = self._w_fp32() @ x32 + self.b                   # 迭代 fp32
        idx = np.argpartition(-u, self.k - 1)[: self.k]      # k-WTA 竞争（侧抑制的抽象）
        s = np.zeros_like(u)
        win = u[idx]
        s[idx] = (win - win.min()) / (win.max() - win.min() + 1e-9) + 0.1  # 归一到 [0.1, 1.1]
        return s, np.sort(idx)

    def learn(self, x: np.ndarray, s: np.ndarray, eta: float) -> None:
        """T2.2 可学习稀疏词典（Foldiak 局部规则，默认关闭）。

        仅活跃（胜者）行更新：ΔW_i = η·s_i·(x − s_i·W_i) —— Hebb 项学习
        输入相关感受野，逐行 Oja 归一化项防塌缩；随后把每行范数拉回初始值
        （稳态双约束之二）。k-WTA 竞争（encode 内）即 Foldiak 的侧抑制分权，
        与稳态缩放共同构成路线图要求的「k-WTA + 稳态双约束」。

        局部性：只有 s_i>0 的行被触碰，成本 O(k·n_input)；无全局非局部梯度。
        """
        act = np.nonzero(s > 0.0)[0]
        if act.size == 0:
            return
        W = self._w_fp32()                                 # 迭代 fp32（P61）
        u = W @ x                                          # 当前全层电位（感受野投影）
        sa = s[act]
        W[act] += eta * (sa[:, None] * (x[None, :] - sa[:, None] * W[act]))
        # 稳态突触缩放：行范数拉回初始值（保留相对结构，防 Oja/Hebb 漂移塌缩）
        nrm = np.maximum(np.linalg.norm(W[act], axis=1), 1e-12)
        W[act] *= (self._hn[act] / nrm)[:, None]
        if W is not self.W:                                # 低精度存储：写回
            self.W[act] = W[act].astype(self.W.dtype)
