"""M6 读出头 —— 对应 IT → 前额叶/前运动皮层的决策读出。

监督信号仅存在于最末端（局部感知器规则）；自 model.py 拆分。

O1-3（2026-09-23）：**结构性稀疏读出**（`conn_k > 0`，默认关闭）。
皮层→输出投射同样是稀疏的（每个下游细胞只接收部分上游输入），而此前读出是
稠密矩阵（n_out×n_in，连接率 100%）。启用后读出权重以 CSR 承载（每输出单元
k 条**存在**的入边，不存在的连接不存储、不计算），复用 `sparse_pc` 的 numba 核。
学习规则**不变**（仍是 softmax 交叉熵的末端梯度）——变化的是连接的存在性结构。
"""

import numpy as np

from .sparse_pc import (_csr_add_outer, _csr_clip, _csr_matvec, _csr_oja_up,
                        _random_csr)


class Readout:
    """M6 读出头 —— 对应 IT→前运动皮层；感知器式局部规则（监督仅在此末端）。

    连接结构可选：稠密（默认，`conn_k=0`）或结构性稀疏 CSR（`conn_k=k>0`）。
    """

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator,
                 w_clip: float = 0.0, fp32: bool = False, conn_k: int = 0,
                 lognormal_init: bool = False, exc_ratio: float = 0.8,
                 hidden: int = 0, hid_k: int = 0, eta_hid: float = 0.002):
        self._n_in, self._n_out = int(n_in), int(n_out)
        self.conn_k = int(conn_k) if conn_k and conn_k > 0 else 0
        # A4（2026-09-23）：**两级群体读出**（`hidden > 0`，默认 0 = 单层线性）。
        # 脑对应：皮层→输出的投射是**多级（多突触中继）**且靠**群体编码**，
        # 不是单层线性分类器。第一级：h → 隐藏群体（稀疏投射 + k-WTA 侧抑制竞争，
        # 局部无监督 Oja 学习）；第二级：群体 → 输出（稀疏投射 + 任务监督）。
        self.hidden = int(hidden) if hidden and hidden > 0 else 0
        self.hid_k = int(hid_k)
        self.eta_hid = float(eta_hid)
        # P6b（2026-09-23，默认关闭）：读出 float32 化——词级读出权重达
        # ~9.5 MB（1540×768），其前向与稠密更新的耗时受内存带宽支配，
        # 降为 fp32 使带宽减半。属数值变化行为，须由 cfg.readout_fp32 显式启用。
        # 注：结构性稀疏模式使用 fp64 CSR（与 fp32 不叠加——组合无实测收益）。
        self.fp32 = bool(fp32) and self.conn_k == 0
        self.dtype = np.float32 if self.fp32 else np.float64
        # C8 修复：可选权重范数上限，防止长跑/高学习率下读出权重溢出
        # （默认 0.0 = 关闭，保持旧行为逐位不变）
        self.w_clip = float(w_clip)
        # B7（2026-09-22）：minibatch 梯度累积缓冲（accumulate=1 时永不启用）
        self._grad_acc: np.ndarray | None = None
        self._acc_n = 0

        if self.hidden > 0:                     # A4：两级群体读出
            k1 = self.hid_k if self.hid_k > 0 else max(1, n_in // 8)
            k2 = self.conn_k if self.conn_k > 0 else max(1, self.hidden // 8)
            self.conn_k = 0
            # ⚠ fan-in 补偿：稀疏投射的权重尺度 ×√(fan_in/k)，使其输出幅值与
            #   稠密版（尺度 0.05、fan_in 全量）可比；否则幅值偏低 √(k/fan_in) 倍。
            self._W1 = _random_csr(rng, self.hidden, n_in, k1,
                                   0.05 * np.sqrt(n_in / k1),
                                   lognormal_init, exc_ratio)
            self._W2 = _random_csr(rng, n_out, self.hidden, k2,
                                   0.05 * np.sqrt(self.hidden / k2),
                                   lognormal_init, exc_ratio)
            self._hid_keep = max(1, self.hidden // 8)    # k-WTA 保留数（群体竞争）
            self._W = None
            self._csr = None
        elif self.conn_k > 0:                   # O1-3：结构性稀疏读出（单级）
            self.conn_k = max(1, min(self.conn_k, n_in))
            self._csr = _random_csr(rng, n_out, n_in, self.conn_k,
                                    0.05 * np.sqrt(n_in / self.conn_k),   # fan-in 补偿
                                    lognormal_init, exc_ratio)
            self._W = None
        elif lognormal_init:                    # O1-4：皮层式（重尾 + E/I 比）
            from .inits import cortical_init
            self._W = cortical_init(rng, (n_out, n_in), 0.05, exc_ratio)
        else:
            W = rng.normal(0, 0.05, (n_out, n_in))
            self._W = W.astype(np.float32) if self.fp32 else W

    # ---------- 权重访问（稀疏模式返回只读稠密视图，兼容既有代码/统计） ----------
    @classmethod
    def from_dense(cls, W: np.ndarray, w_clip: float = 0.0, k: int = 0) -> "Readout":
        """由稠密权重构造（每输出单元取 |w| 最大的 k 列；k≤0 或 k≥列数 → 全列）。

        用途：(a) 可比性对拍——k = 全列时与稠密读出**权重完全相同**；
              (b) 稠密→稀疏蒸馏（保留最强连接）。
        """
        from .sparse_pc import _from_dense_csr
        n_out, n_in = W.shape
        self = cls.__new__(cls)
        self._n_in, self._n_out = n_in, n_out
        self.conn_k = min(k, n_in) if k > 0 else n_in
        self.fp32 = False
        self.dtype = np.float64
        self.w_clip = float(w_clip)
        self._grad_acc, self._acc_n = None, 0
        self._csr = _from_dense_csr(W, self.conn_k)
        self._W = None
        return self

    @property
    def W(self) -> np.ndarray:
        """输出层权重的只读稠密视图（两级模式返回第二级 W2）。"""
        if self.hidden > 0:
            ip, idx, val = self._W2
            W = np.zeros((self._n_out, self.hidden), dtype=val.dtype)
            for i in range(self._n_out):
                W[i, idx[ip[i]:ip[i + 1]]] = val[ip[i]:ip[i + 1]]
            return W
        if self.conn_k > 0:
            ip, idx, val = self._csr
            W = np.zeros((self._n_out, self._n_in), dtype=val.dtype)
            for i in range(self._n_out):
                W[i, idx[ip[i]:ip[i + 1]]] = val[ip[i]:ip[i + 1]]
            return W
        return self._W

    @W.setter
    def W(self, v) -> None:
        self._W = v

    def n_synapses(self) -> int:
        """实际存在的连接数（两级/稀疏模式 = CSR 条目；稠密模式 = 矩阵元素数）。"""
        if self.hidden > 0:
            return int(len(self._W1[2]) + len(self._W2[2]))
        return int(len(self._csr[2])) if self.conn_k > 0 else int(self._W.size)

    def _kwta(self, u: np.ndarray) -> np.ndarray:
        """群体竞争（侧抑制）：仅保留响应最强的 `hidden/8` 个单元并压缩幅值。

        脑对应：皮层群体编码的稀疏化竞争（抑制性中间神经元介导的侧抑制）。
        """
        k = min(self._hid_keep, len(u))
        idx = np.argpartition(-u, k - 1)[:k]
        a = np.zeros_like(u)
        a[idx] = np.tanh(u[idx])
        return a

    def stats(self) -> dict:
        dense = self._n_out * self._n_in
        n = self.n_synapses()
        out = {"synapses": n, "dense_equivalent": dense,
               "connectivity": n / dense if dense else 1.0, "k": self.conn_k}
        if self.hidden > 0:
            out.update({"levels": 2, "hidden": self.hidden,
                        "synapses_w1": int(len(self._W1[2])),
                        "synapses_w2": int(len(self._W2[2]))})
        return out

    # ---------- 前向 ----------
    def __call__(self, h: np.ndarray) -> np.ndarray:
        if self.hidden > 0:                     # 两级群体读出（稀疏投射 + 群体竞争）
            a = self._kwta(_csr_matvec(*self._W1, h))
            return _csr_matvec(*self._W2, a)
        if self.conn_k > 0:                     # 稀疏：只遍历存在的边
            return _csr_matvec(*self._csr, h)
        if self.fp32:
            return (self._W @ h.astype(np.float32, copy=False)).astype(np.float64,
                                                                       copy=False)
        return self._W @ h

    def _to_w(self, arr: np.ndarray) -> np.ndarray:
        """更新量按读出 dtype 转换（fp32 时降精度；fp64 时零拷贝直返）。"""
        return arr.astype(np.float32, copy=False) if self.fp32 else arr

    def _clip(self) -> None:
        if self.w_clip > 0.0:
            if self.hidden > 0:
                _csr_clip(self._W1[2], self.w_clip)
                _csr_clip(self._W2[2], self.w_clip)
            elif self.conn_k > 0:
                _csr_clip(self._csr[2], self.w_clip)
            else:
                np.clip(self._W, -self.w_clip, self.w_clip, out=self._W)

    def _check_contract(self, h: np.ndarray, target: np.ndarray | None) -> None:
        """维度契约校验（fail-fast）。

        此前维度不符时只抛 numpy 的模糊广播错误（如
        `operands could not be broadcast together with shapes (64,) (32,)`），
        调用方难以定位。注意 `step(x, target=...)` 的 target 维度是**读出输出
        维度**（`cfg.n_readout`，默认 = `cfg.n_input`），**不是** `n_top`。
        """
        if h.shape[0] != self._n_in:
            raise ValueError(
                f"Readout 输入维度不符：期望 {self._n_in}（h 的维度），实际 {h.shape[0]}")
        if target is not None and target.shape[0] != self._n_out:
            raise ValueError(
                f"Readout 目标维度不符：期望 {self._n_out}"
                f"（= cfg.n_readout，0 时取 cfg.n_input），实际 {target.shape[0]}。"
                "注意 PHDNet.step(x, target=...) 的 target 是**读出输出维度**，不是 n_top；"
                "可用 net.n_out 查询。")

    # ---------- 学习（感知器 / softmax） ----------
    def learn(self, h: np.ndarray, target: np.ndarray, eta: float) -> None:
        self._check_contract(h, target)
        if self.hidden > 0:                     # 两级：输出监督 + 中间层局部无监督
            a = self._kwta(_csr_matvec(*self._W1, h))
            y = _csr_matvec(*self._W2, a)
            _csr_add_outer(*self._W2, target - y, a, eta)
            if self.eta_hid > 0.0:
                _csr_oja_up(*self._W1, a, h, self.eta_hid)   # Hebbian/Oja（局部）
            self._clip()
            return
        y = self.__call__(h)
        if self.conn_k > 0:                     # ΔW = η·(t − y) ⊗ h（只更新存在的边）
            _csr_add_outer(*self._csr, target - y, h, eta)
        else:
            self.W += self._to_w(eta * np.outer(target - y, h))
        self._clip()

    def learn_softmax(self, h: np.ndarray, target: np.ndarray, eta: float,
                      y_pre: np.ndarray | None = None,
                      accumulate: int = 1) -> float:
        """softmax 感知器（交叉熵的局部梯度 ∂L/∂y = p − t，仅作用于末端）。

        返回当前样本的 −log p(correct)，供困惑度统计。

        P6 性能修复（2026-09-21，逐位等价）：
          ① y_pre：调用方（model.step）前向已算得 W@h，传入可省一次重复
             矩阵-向量乘。与内部重算逐位一致（同一 W、同一 h）；None 时
             自行计算（旧行为）。y_pre 本体不被修改（y = y_pre − max 新数组）。
          ② 梯度更新改为「外积后原地缩放」：tmp = outer(dp,h); tmp *= eta;
             W -= tmp —— 与 eta * outer(dp,h) 逐位一致（IEEE 乘法交换律），
             但少分配一个 (n_out, n_in) 临时数组（词级读出下 ~9.5 MB），
             消除训练热路径的主要内存流量。

        B7 minibatch（2026-09-22，默认关闭）：`accumulate=N>1` 时累积 N 步
        梯度、按**平均梯度**更新一次（ΔW = −η·mean(g)，标准 minibatch 语义）。

        O1-3（2026-09-23，默认关闭）：`conn_k>0` 时更新只触达存在的连接
        （学习规则与稠密版相同，仅连接结构不同）。
        """
        if target is not None:
            self._check_contract(h, target)
        if y_pre is not None:
            y = y_pre - y_pre.max()      # 新数组，不修改调用方持有的 y_pre
        elif self.hidden > 0:
            y = self.__call__(h)
            y = y - y.max()
        elif self.conn_k > 0:
            y = self.__call__(h)
            y = y - y.max()
        else:
            y = self.W @ h
            y -= y.max()                 # 数值稳定
        p = np.exp(y)
        p /= p.sum()
        correct = int(np.argmax(target))
        nll = float(-np.log(p[correct] + 1e-12))

        if accumulate > 1:                              # B7：minibatch 累积
            if self.hidden > 0:                         # 两级：按中间激活累积
                a_acc = self._kwta(_csr_matvec(*self._W1, h))
                g = np.outer(p - target, a_acc)
            else:
                a_acc = None
                g = np.outer(p - target, h)
            self._grad_acc = g if self._grad_acc is None else self._grad_acc + g
            self._acc_n += 1
            if self._acc_n >= accumulate:
                c = eta / float(accumulate)
                if self.hidden > 0:                     # 稀疏按行写回 W2 + W1 局部
                    ip, idx, val = self._W2
                    for i in range(self._n_out):
                        s = slice(ip[i], ip[i + 1])
                        val[s] -= c * self._grad_acc[i, idx[s]]
                    if self.eta_hid > 0.0:
                        _csr_oja_up(*self._W1, a_acc, h, self.eta_hid)
                elif self.conn_k > 0:                   # 稀疏按行写回累积梯度
                    ip, idx, val = self._csr
                    for i in range(self._n_out):
                        s = slice(ip[i], ip[i + 1])
                        val[s] -= c * self._grad_acc[i, idx[s]]
                else:
                    self.W -= self._to_w(c * self._grad_acc)
                self._grad_acc = None
                self._acc_n = 0
                self._clip()
            return nll

        if self.hidden > 0:                             # 两级：W2 监督 + W1 局部无监督
            a = self._kwta(_csr_matvec(*self._W1, h))
            _csr_add_outer(*self._W2, p - target, a, -eta)
            if self.eta_hid > 0.0:
                _csr_oja_up(*self._W1, a, h, self.eta_hid)
        elif self.conn_k > 0:                           # 稀疏梯度下降（存在的边）
            _csr_add_outer(*self._csr, p - target, h, -eta)
        else:
            tmp = self._to_w(np.outer(p - target, h))
            tmp *= eta                                  # 原地缩放（逐位等价）
            self.W -= tmp                               # 梯度下降（末端局部）
        self._clip()
        return nll
