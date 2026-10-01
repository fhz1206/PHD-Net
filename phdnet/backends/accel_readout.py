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

_DT = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16",
       "int8": "int8"}
# P105：旧名别名（与 readout.py 的 _RO_DTYPE_ALIASES 同口径）——"fp8" 本来就
# 是 1 字节码本，P100 已正名为 int8；加速后端同样接受旧名，避免配置里的
# dtype="fp8" 在这里被 ValueError 拒绝而静默回落。
_DTYPE_ALIASES = {"fp8": "int8"}

# int8 模式的**计算** dtype：存储是 int8 码本，但 ht/y 的 matmul 在 fp16 域做
# （昇腾/CUDA 的 fp16 矩阵乘是原生快路径）。构造期探测失败则整体回落 fp16。
_INT8_CDTYPE = torch.float16 if torch is not None else None
_INT8_QMAX = 127.0


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
    _is_accel = True          # P40：供 model.step 区分加速后端（设备直通 + target_idx）
    """设备无关（cpu / npu / cuda / rocm）的稠密读出，接口对齐 `Readout`。

    W **常驻设备**（NPU 上 908 MB 级），每步只往返 h 与 y：
      - forward：h（CPU numpy）→ 设备 → y → CPU numpy；
      - learn_softmax：主回路 fp32（与 P9 协议一致），梯度更新在设备上完成。
    """

    def __init__(self, n_h: int, n_out: int, rng=None, device: str = "auto",
                 dtype: str = "fp32", w_clip: float = 0.0, w0=None,
                 nll_sync_every: int = 1, compile: bool = False,
                 compile_mode: str = "default", conn_k: int = 0,
                 csr=None, lognormal_init: bool = False, exc_ratio: float = 0.8):
        if torch is None:
            raise RuntimeError("未安装 torch，加速读出不可用")
        dtype = _DTYPE_ALIASES.get(str(dtype), str(dtype))
        if dtype not in _DT:
            raise ValueError(f"不支持 dtype={dtype!r}；可用 {sorted(_DT)}")
        self.device = resolve_accel_device(device)
        # P105 int8 语义（fhz 2026-10-01 决策）——存储/计算/更新三层分开：
        #   存储：W 用 torch.int8 码本，per-tensor scale = 2*max|W|/127（P104 的
        #         2× 余量纪律：训练中权重增长 ≤2× 不需要重标定；越界被 clamp
        #         饱和，不回绕）。
        #   forward：每步反量化 W_int8 → fp16（codes.float()*scale）再 matmul
        #         ——1.6 亿元素 ≈ 0.5-1 ms（NPU 可接受）。**不缓存 fp16 副本**：
        #         W 每 step 都在更新，缓存必然过期。
        #   learn：在 fp32 域计算 dp⊗ht，然后 W_int8 重量化写回（见
        #         _int8_update）。精度约束（知情取舍）：int8 步长
        #         ≈ max|W|/127 ≈ 0.008 ≫ 非目标行 |dp|≈1e-6 → 非目标行更新
        #         会被量化吃掉（int8 存储降 4× 访存的代价，fhz 知情选择）；
        #         但目标行 |dp|≈1 远大于步长，必须完整保留。
        # tdtype 是**存储** dtype（int8）；ht/y 等 matmul 参与者的 dtype 跟随
        # _cdtype（int8 模式 = fp16）——所有原来用 self.tdtype 的地方都要检查。
        if dtype == "int8":
            self.tdtype = torch.int8
            self._cdtype = _INT8_CDTYPE
        else:
            self.tdtype = getattr(torch, _DT[dtype])
            self._cdtype = self.tdtype
        # int8 模式不走 torch.compile 融合核：_train_step_core 未覆盖反量化/
        # 重量化路径，融合会产出另一套计算图——宁可放弃融合收益（正确性优先）。
        if dtype == "int8" and compile:
            compile = False
        # P105 回落保护（参考 P85/P90 的 fp8 模式）：构造后做一次探测——int8 码本
        # 反量化 + fp16 matmul 在该设备是否真的可用；不可用则**永久**回落 fp16
        # （tdtype/_cdtype 跟随）+ 告警。⚠ P90 的 forward 级兜底（try/except
        # around matmul）仍然保留在 forward_dev：探测通过不代表运行期所有 shape
        # 都不炸，int8 matmul 在某些平台可能跑着跑着才失败。
        self._int8 = (self.tdtype == torch.int8)
        self._wscale = None
        if self._int8:
            try:
                _p8 = torch.ones(4, 4, dtype=torch.int8, device=self.device)
                _p16 = torch.ones(4, 4, dtype=self._cdtype, device=self.device)
                _ = (_p8.to(self._cdtype) @ _p16).sum().item()
            except Exception as e:                          # noqa: BLE001
                import warnings
                warnings.warn(
                    f"int8 探测在设备 {self.device} 上失败（{type(e).__name__}: "
                    f"{str(e)[:80]}）→ 永久回落 fp16（训练不中断）", RuntimeWarning)
                self.tdtype = torch.float16
                self._cdtype = torch.float16
                self._int8 = False
        self.n_out, self.n_h = int(n_out), int(n_h)
        # ── P111：稀疏（CSR）读出 ────────────────────────────────────────────
        # 此前`readout_conn_k>0` 命中 `_unsupported_reason` → 整个 NPU 读出
        # 回落 numba CPU（读出占端到端 ~89%，等于关掉了全部设备加速）。
        # 本实现在设备侧持有**均匀 k 的 (n_out, k) 稠密 val 张量 + (n_out, k)
        # 列索引张量**，前向/更新都按 gather 实现：
        #   前向  y[i] = Σⱼ Wv[i,j] ·h[Wi[i,j]]
        #   更新  Wv  -= η · dp[i] · h[Wi[i,:]]      （逐行 rank-1）
        # 流量口径（1B 档 vocab=73,958, k=128, n_h=3,072）：
        #   稠密 fp32 每步触达 866.7 MiB；稀疏 (val fp32 + idx int64) = 12 B/突触
        #   → 144.4 MiB，约 1/6。**idx 用 int64 是刻意的**：int32 会让 NPU 侧
        #   gather 走类型转换，实测口径以verify_accel_sparse 为准。
        #
        # ⚠ **只支持均匀 k**（`_random_csr` 的输出）。非均匀行宽（幂律分组）
        #   不在此路径——那种结构要用变长 CSR + segment sum，语义与实现都不同，
        #   命中时按P19 纪律回落 numba，不静默走错算法。
        self._sparse = False
        self._sparse_k = 0                                    # conn_k 的内部真值源
        self.Wi = None                                        # (n_out, k) int64 列索引
        #（`conn_k` 是只读 property —— 它必须与 numba Readout 同为只读，
        #   否则两边语义不对称，ckpt 的 `net.readout.conn_k` 读法会分叉。）
        if conn_k and int(conn_k) > 0:
            k = max(1, min(int(conn_k), self.n_h))
            if csr is not None:
                ip, idx, val = csr
                widths = np.diff(np.asarray(ip, dtype=np.int64))
                if widths.size != self.n_out or not np.all(widths == widths[0]):
                    raise ValueError(
                        f"稀疏读出只支持均匀 k（收到行宽 min={widths.min()} "
                        f"max={widths.max()}）")
                if int(widths[0]) != k:
                    raise ValueError(f"CSR 行宽 {int(widths[0])} != conn_k {k}")
                _idx = np.asarray(idx, dtype=np.int64).reshape(self.n_out, k)
                _val = np.asarray(val, dtype=np.float32).reshape(self.n_out, k)
            else:
                from ..sparse_pc import _random_csr
                # fan-in 补偿与 `Readout.__init__` 的稀疏分支同式（否则稀疏臂
                # 因幅值偏低 √(k/n_in) 被在 PPL 上无谓惩罚）。
                _ip, _idx, _val = _random_csr(
                    rng if rng is not None else np.random.default_rng(0),
                    self.n_out, self.n_h, k,
                    0.05 * np.sqrt(self.n_h / k), lognormal_init, exc_ratio)
                _idx = _idx.reshape(self.n_out, k)
                _val = _val.reshape(self.n_out, k).astype(np.float32)
            self._sparse_k = k
            self._sparse = True
            self.Wi = torch.tensor(_idx, device=self.device, dtype=torch.long)
            self.W = torch.tensor(_val, device=self.device, dtype=torch.float32)
            # 稀疏模式**只支持 fp32**：低精度会丢弃非目标行更新（P110 实测），
            # 而稀疏读出的可用性正是为了省流量，不能再叠加语义损失。
            self.tdtype = torch.float32
            self._cdtype = torch.float32
            self._int8 = False
            self._wscale = None
            # 融合核走的是稠密 addmm_，稀疏不适用 → 永久eager（P38/P55 纪律）。
            self._compiled = False
            self._fused = None
        else:
            self.Wi = None
        self.w_clip = float(w_clip)
        # P34：nll 设备侧累积（消除每步 .item() 同步 → CPU/NPU 重叠）
        self.nll_sync_every = max(1, int(nll_sync_every))
        self._nll_sum = None
        self._nll_n = 0
        self._last_nll = 0.0
        # P80/P82：correct 索引的 pinned 上传缓冲（避免 pageable 小拷贝的隐式同步）
        self._correct_pinned = None
        # P38/P44：`torch.compile` 融合读出热路径的 4 个小 kernel
        # （softmax / log / sub / addmm_；CANN 上每个 launch 开销 ~50-200 μs）。
        #
        # ⚠ **模式选择的关键取舍（服务器实测得到）**：`mode="reduce-overhead"`
        # 启用 CUDA Graph，而 `self.W.addmm_(...)` **原地修改 W** —— cudagraph
        # 捕获的张量不允许被后续 kernel mutate，故每次调用都会打印
        #     "skipping cudagraphs due to mutated inputs"
        # 并**静默退回**无 graph 模式（功能正确，但拿不到 graph 收益）。
        # 这是**本质冲突**：W 每步都在原地更新（908 MiB/step；改成非原地重新
        # 分配会让流量翻倍，更不可接受）。因此默认用 `mode="default"`
        # （只做 kernel 融合，不启用 cudagraphs）——融合收益保留，无 mutate 限制。
        # 需要图级优化时可选 reduce-overhead/max-autotune（接受上述回退）。
        self._compiled = False
        self._compile_mode = str(compile_mode or "default")
        # P58：pinned 暂存池（懒初始化；CPU-only torch 无 pin_memory → 回落）
        self._pin_bufs = None
        self._pin_ok = False
        self._pin_next = 0
        self._pin_cap = 4
        if compile:
            try:
                self._fused = torch.compile(self._train_step_core,
                                            mode=self._compile_mode,
                                            fullgraph=False)
                self._compiled = True
            except Exception as e:                       # noqa: BLE001
                import warnings
                warnings.warn(f"torch.compile 不可用，回落 eager：{e}",
                              RuntimeWarning)
        if w0 is None and not self._sparse:
            g = rng if rng is not None else np.random.default_rng(0)
            init = (g.normal(0.0, 0.05, (self.n_out, self.n_h)) if hasattr(g, "normal")
                    else np.asarray(g).reshape(self.n_out, self.n_h))
        elif self._sparse:
            # 稀疏模式下 `w0` 语义是 (n_out, k) 的 val矩阵（列由 Wi 决定）。
            # 给了就覆盖初始化值；不给则沿用上面 CSR/随机生成的 val。
            if w0 is not None:
                _w0s = np.asarray(w0, dtype=np.float32)
                if _w0s.shape != (self.n_out, self.conn_k):
                    raise ValueError(
                        f"稀疏 w0 形状不符：期望 {(self.n_out, self.conn_k)}，"
                        f"收到 {_w0s.shape}")
                self.W = torch.tensor(np.ascontiguousarray(_w0s),
                                      device=self.device, dtype=torch.float32)
            init = None
        else:
            init = np.asarray(w0, dtype=np.float32)
        # 稀疏路径的self.W 已在上面建好（fp32 (n_out,k)），不再走稠密初始化。
        if not self._sparse:
            # ⚠ 必须**复制**：torch.as_tensor / torch.from_numpy 会与传入数组共享内存，
            # 而 W 是原位更新的 → 调用方（例如比较用的参考权重）会被静默改掉。
            _init_t = torch.tensor(np.ascontiguousarray(init, dtype=np.float32),
                                   device=self.device, dtype=torch.float32)
            if self._int8:
                # int8 存储：per-tensor scale = 2*max|W|/127（2× 余量，见上方注释）。
                # 码本容量 ±127；初始码最大只到 ~63.5，权重涨到 2× 才触饱和（clamp
                # 裁剪，不回绕）。全零权重退化 wscale=1.0（与 fp 路径同等的退化行为）。
                _amax = float(_init_t.abs().max().item()) if _init_t.numel() else 0.0
                self._wscale = (2.0 * _amax / _INT8_QMAX) if _amax > 0.0 else 1.0
                self.W = torch.clamp(
                    torch.round(_init_t / self._wscale),
                    -_INT8_QMAX, _INT8_QMAX).to(torch.int8)
            else:
                self.W = _init_t.to(self.tdtype)
        # P28：设备侧 (h, y) 缓存，供 forward → learn_softmax 的热路径复用
        self._cache_h: np.ndarray | None = None
        self._cache_ht = None
        self._cache_y = None
        # P111：稀疏 gather 缓存（(n_out,k) 复用，见 _sp_gather）
        self._cache_g = None
        self._cache_g_epoch = -1                            # P112：主机侧 epoch 判据
        self._ht_epoch = 0                                  # 每次上传新 h 递增
        self._csr_val_host = None                           # P111：_csr 导出缓存

    # ---------- 前向 ----------
    def _matmul(self, ht):
        """y = W @ ht；int8 模式**每步**反量化后再乘。

        P105：反量化 = `codes.float() * scale`（fp32 域）→ cast 到 ht 的 dtype
        （fp16）做 matmul。**不缓存 fp16 副本**——W 每 step 都在更新，任何缓存
        下一步就过期（fhz 明确要求每步反量化；1.6 亿元素 ≈ 0.5-1 ms @NPU）。

        P111 稀疏：不做 matmul，改为 gather + 逐行求和
            y[i] = Σⱼ W[i,j] ·ht[Wi[i,j]]
        （`_gather` 复用 forward→learn 的缓存，避免同一h 被 gather 两次）。
        """
        if self._sparse:
            # y[i] = Σⱼ W[i,j] ·ht[Wi[i,j]] —— **必须逐元素乘 W 再求和**。
            # （曾经的 bug：写成 `self._sp_gather(ht).sum(dim=1)`，等于把
            #   权重全丢掉、只把 gather 到的 h 分量相加——数值完全错但形状/
            #   dtype 都对，只在数值对拍里才暴露。）
            return (self.W * self._sp_gather(ht)).sum(dim=1)
        if self._int8:
            Wq = (self.W.to(torch.float32) * self._wscale).to(ht.dtype)
            return Wq @ ht
        return self.W @ ht

    def _sp_gather(self, ht):
        """稀疏行内 gather：取每行 k 个 h 分量（(n_out,k)），供前向与更新共用。

        缓存判据（**P112 修复，这里原先是性能 bug**）
        ----------------------------------------------
        第一版用 `torch.equal(self._cache_g_src, ht)` 判「是不是同一个 h」。
        `torch.equal` 逐元素比较后返回 **Python bool** → 在 NPU 上这是一次
        **强制设备同步**：CPU 必须等所有已入队 kernel 跑完才能拿到结果。等于把
        P34 刚用 `nll_sync_every` 消除掉的那个同步又加回来，且发生在**每步**的
        最热路径上（服务器日志实测：读出 7.68 ms/tok，占 43.5%）。

        改为**主机侧 epoch 标记**：每次上传新 h 就递增 `_ht_epoch`，
        `_sp_gather` 只比这个整数。语义等价（同一 epoch = 同一个 h），
        **零设备交互**。
        ⚠ 不能简化成「盲目复用缓存」——那会在 h 变化时静默用错 gather 结果。
        """
        if self._cache_g is not None and self._cache_g_epoch == self._ht_epoch:
            return self._cache_g
        g = ht[self.Wi]                                  # (n_out, k) 高级索引 gather
        self._cache_g = g
        self._cache_g_epoch = self._ht_epoch
        return g

    def _int8_update(self, dp32, ht32, alpha: float) -> None:
        """int8 权重的 rank-1 更新：**fp32 域**计算 dp⊗ht，再重量化写回。

        P105（fhz 2026-10-01 决策）：直接在 int8 码上原地加更新毫无意义——
        int8 步长 ≈ max|W|/127 ≈ 0.008，而 softmax 梯度里非目标行的 |dp|≈1e-6，
        加上去 round 回来码不变（更新被量化吃掉，这是 int8 存储降 4× 访存的
        **知情取舍**，不是 bug）；目标行 |dp|≈1 远大于步长，走 fp32 域更新 +
        重量化后完整保留（verifier [F] 用 fp16 参考对拍把关）。

        流程：dequant(W) → fp32 域 addmm_（rank-1 AXPY，同 P28 口径）→
        可选 w_clip（int8 的 clamp_ 对码张量无意义，改为**重量化前裁剪**）→
        round(RNE)/clamp(±127)/cast int8 写回。scale 固定不重标定（2× 余量）。
        """
        W_fp = self.W.to(torch.float32) * self._wscale
        W_fp.addmm_(dp32.reshape(-1, 1), ht32.reshape(1, -1), alpha=float(alpha))
        if self.w_clip > 0.0:
            W_fp.clamp_(-self.w_clip, self.w_clip)
        codes = torch.clamp(torch.round(W_fp / self._wscale),
                            -_INT8_QMAX, _INT8_QMAX)
        self.W.copy_(codes.to(torch.int8))

    def forward(self, h) -> np.ndarray:
        """前向并返回 **numpy**（`Readout.__call__` 契约）。

        同时把设备侧的 (h, y) 缓存下来，供紧随其后的 `learn_softmax` 复用
        （P28：生产热路径是 `y = readout(h)` → `learn_softmax(..., y_pre=y)`，
        若不缓存就要 D2H 289 KiB 再 H2D 传回，纯往返）。
        """
        # P112：这也是一次「新 h 上传」，必须递增 epoch —— forward 走的是
        # `_to_dev` 而**不是** `_staged_to_dev`（前者不做 pinned 暂存），
        # 只在 `_staged_to_dev` 里递增会让本路径的 epoch 恒定不变，
        # `_sp_gather` 于是把上一步的 gather 结果错误地复用给新 h。
        # （第一版遗漏此处 → verify_accel_sparse A1 立刻报数值不一致。）
        self._ht_epoch += 1
        ht = self._to_dev(h)
        y = self._matmul(ht)
        self._cache_h = np.ascontiguousarray(h, dtype=np.float32)
        self._cache_ht = ht
        self._cache_y = y
        return y.float().cpu().numpy()

    def __call__(self, h) -> np.ndarray:
        """与 `Readout.__call__` 同协议（model.step 里是 `self.readout(h)`）。

        P19 修复：早先只提供 `forward()`，导致 `TypeError: 'AccelReadout' object
        is not callable`（生产训练首个 step 即崩）。`Readout` 的调用面共 4 个：
        `__call__` / `learn_softmax` / `learn` / `W`，此处全部对齐。
        """
        return self.forward(h)

    def forward_dev(self, h):
        """设备内前向（返回**设备张量**，零同步）。

        P28：这是消除「训练热路径上 y 白 D2H 再白 H2D」的正解——调用方拿设备
        张量直接进 `learn_softmax`，整条链路上没有主机↔设备往返。
        P58：H2D 走 pinned 暂存 + non_blocking（pageable 拷贝会阻塞 CPU；
        异步上传让 CPU 提前回去算下一步的 M1–M5）。
        """
        ht = self._staged_to_dev(h)
        try:
            y = self._matmul(ht)
        except Exception as e:                              # noqa: BLE001
            # P90 兜底（P105 沿用）：int8 路径的**任何**运行期失败（matmul 无算子、
            # 反量化/传输不支持）都在这里被吸收 → 永久回落 fp16 后重算，训练不中断。
            # 服务器实测（2026-09-30 19:17）证明这条必需：fp8 能建张量但 matmul 无
            # 实现会崩在 `@`——只靠构造期探测不够。非 int8 模式无可回落 → 真错误照抛。
            if not self._int8:
                raise
            self._int8 = False
            _scale = self._wscale
            self._wscale = None
            self.tdtype = torch.float16
            self._cdtype = torch.float16
            # 保值反量化到 fp16 稠密主副本（codes*scale），此后走普通 fp16 路径
            self.W = (self.W.to(torch.float32) * _scale).to(torch.float16)
            import warnings
            warnings.warn(
                f"int8 前向在设备 {self.device} 上失败（{type(e).__name__}: "
                f"{str(e)[:80]}）→ 永久回落 fp16（训练不中断）", RuntimeWarning)
            ht = self._staged_to_dev(h)
            y = self.W @ ht          # 已回落 fp16 稠密 → 直接乘（勿再走 _matmul）
        self._cache_h = np.ascontiguousarray(h, dtype=np.float32)
        self._cache_ht = ht
        self._cache_y = y
        return y

    def _train_step_core(self, y32, ht, t32, correct: int, eta: float):
        """读出更新一体步（softmax→nll→addmm_），供 `torch.compile` 融合。

        P38：调用方已算好 `y32 = (W@h).float()`（前向缓存），这里**不重复
        matvec**（否则 W 读两次，抵消 P28 收益）。编译器融合 softmax/log/sub/
        addmm_ 四个小 kernel、摊薄 launch 开销；eager 路径不走这里。

        ⚠ P55（fhz 服务器 2026-09-29 NPU 首次真跑即崩）：P45 把「target_idx
        路径就地改 p」实现在了 **eager 分支**，本融合核仍是旧的
        `dp = p - t32` → `t32=None` 时 `FakeTensor - None` 崩 dynamo。两条路径
        现在**逐行等价**（数值也等价：p − onehot ≡ p[c] −= 1）。
        """
        p = torch.softmax(y32, dim=0)
        nll_dev = -torch.log(p[correct:correct + 1] + 1e-12).reshape(())
        if t32 is None:                  # P45 就地改 p（target_idx 路径）
            dp = p.to(self.tdtype)
            dp[correct] -= 1.0
        else:
            dp = (p - t32).to(self.tdtype)
        self.W.addmm_(dp.reshape(-1, 1), ht.reshape(1, -1), alpha=-float(eta))
        if self.w_clip > 0.0:
            self.W.clamp_(-self.w_clip, self.w_clip)
        return nll_dev

    # ---------- 学习（softmax 感知器，局部梯度 p − t）----------
    def learn_softmax(self, h, target, eta: float, y_pre=None,
                      accumulate: int = 1, target_idx: int | None = None) -> float:
        """P6 感知器更新（softmax 交叉熵的局部梯度 ∂L/∂y = p − t）。

        P28 的三处性能修正（数值语义不变，仍属「容差一致」）：
          ① `W.addmm_(dp⊗ht, alpha=-eta)` 取代 `W.add_(torch.outer(dp, ht))`——
             后者会**物化一个与 W 同尺寸的临时张量**（1B 档 867 MiB），再被 add_
             读回来 → 单步多出 1.73 GiB 设备带宽（实测占总流量 40%）。addmm_ 是
             一步 rank-1 AXPY，不产生临时张量。
          ② y/h 复用：优先用 `forward`/`forward_dev` 缓存的设备张量，省掉
             「y D2H 289 KiB → H2D 传回」与 h 的重复上传。
          ③ 同步点 3 → 1：`correct` 在主机侧算（target 本来就在主机，零设备交互），
             nll 合并成唯一一次 `.item()`。
        """
        ht, cache_hit = self._lookup_ht(h)
        # y_pre 既可能是设备张量（推荐路径）、numpy（旧接口），或 None（自算）
        if y_pre is None:
            y = self._matmul(ht)     # P105：int8 模式每步反量化
        elif torch.is_tensor(y_pre):
            y = y_pre if y_pre.device == ht.device else y_pre.to(ht.device)
        elif cache_hit and self._cache_y is not None:
            # 缓存命中 = 本次 h 与 forward 时的 h 相同 → y_pre 就是那次的前向结果
            # （生产热路径 forward → learn_softmax 的形状），直接复用设备张量，
            # 省掉「D2H 289 KiB + H2D 传回 + 一次硬同步」。
            y = self._cache_y
        else:
            y = torch.as_tensor(np.ascontiguousarray(y_pre, dtype=np.float32),
                                device=ht.device, dtype=self._cdtype)
        y32 = y.float()
        # P40：target 两种来源——`target_idx`（int，推荐）在**设备上**构造 onehot，
        # 省掉每步 289 KiB 的 host onehot 构造 + H2D；`target`（host 数组）为
        # 兼容旧接口保留。
        if target_idx is not None:
            # P45：直接把 p 变成 dp（p[c] -= 1），**不新建 zeros_like 缓冲**
            # ——后者每步多一次 289 KiB 设备分配 + scatter kernel，比原 H2D 更贵。
            correct = int(target_idx)
            t = None                      # 标记：走「就地改 p」路径
        else:
            t = self._to_dev(target).float()
            # correct 在主机侧求（target 是 host 数组）——省一次 .item() 硬同步
            correct = int(np.argmax(np.asarray(target)))
        # nll：P34 双模式。sync_every=1 时每步 .item()（旧行为）；N>1 时
        # 累积到设备标量、每 N 步同步一次——中间步返回滞后均值，训练热路径
        # 上 CPU 与 NPU 从此可以重叠（这是 NPU 上最大的延迟收益）。
        # P38：`_compiled` 时 softmax/nll/addmm_ 四个 kernel 交给编译器融合
        #（eager 路径逐算子执行，语义一致）
        # P55：softmax/nll 只在 eager 路径算——compiled 路径由融合核内部算，
        # 此处重算会白付一次 51962 元素 softmax + 一次 log（纯浪费）。
        # P55 加固：融合核**首次执行**才真正触发 inductor 编译（构造期
        # torch.compile() 只是包装，不会失败）——编译失败若不捕获，生产训练
        # 直接崩（Windows 无 cl / NPU 上 inductor 异常均可能发生）。按 P19
        # 「回落 + 记原因」纪律：首次失败即**永久回落 eager** 并告警。
        if self._compiled:
            try:
                nll_dev = self._fused(y32, ht, t, correct, float(eta))
            except Exception as e:                        # noqa: BLE001
                import warnings
                warnings.warn(f"torch.compile 融合核执行失败，永久回落 eager"
                              f"（原因：{type(e).__name__}: {e}）",
                              RuntimeWarning)
                self._compiled = False
                self._fused = None
                nll_dev = self._eager_step(y32, ht, t, correct, eta)
        else:
            nll_dev = self._eager_step(y32, ht, t, correct, eta)
        if self.nll_sync_every <= 1:
            return float(nll_dev.item())
        self._nll_sum = (nll_dev if self._nll_sum is None
                         else self._nll_sum + nll_dev)
        self._nll_n += 1
        if self._nll_n >= self.nll_sync_every:
            self._last_nll = float(self._nll_sum.item()) / self._nll_n
            self._nll_sum = None
            self._nll_n = 0
        return self._last_nll

    def _eager_step(self, y32, ht, t, correct: int, eta: float):
        """eager 逐步执行（softmax→nll→addmm_）；compiled 回落时也走这里。

        P80：nll 改用 `cross_entropy` 单 kernel（替代 `softmax + log + 索引`
        的多 kernel 组合，并去掉 1e-12 加项）。数学等价：CE(logits, c) =
        logsumexp(y) − y[c] = −log(softmax(y)[c])。dp 仍由同一份 softmax 得出。
        """
        p = torch.softmax(y32, dim=0)
        # nll 先取（必须在改 p 之前）
        cp = getattr(self, "_correct_pinned", None)
        if cp is None:
            cp = torch.zeros(1, dtype=torch.long,
                             pin_memory=(ht.device.type == "npu"))
            self._correct_pinned = cp
        cp.fill_(correct)
        _ct = cp.to(y32.device, non_blocking=True)
        nll_dev = torch.nn.functional.cross_entropy(
            y32.reshape(1, -1), _ct).reshape(())
        if self._int8:
            # P105：int8 模式的更新走 **fp32 域** dp⊗ht + 重量化写回（精度约束
            # 与知情取舍见 _int8_update 的 docstring）。p 已用完（nll 先取），
            # clone 后就地改目标行，语义与 P45「p[c] -= 1」一致。
            dp32 = p.clone() if t is None else (p - t)
            if t is None:
                dp32[correct] -= 1.0
            self._int8_update(dp32, ht.float(), -eta)
            return nll_dev
        # P84：更新主副本是 fp16（低精度回落场景）→ dp/ht 必须同 dtype，否则
        # addmm_ 退回慢路径或直接报错。int8 模式已在上面提前返回，不会到这里。
        _upd_dtype = (torch.float16
                      if self.W.dtype == torch.float16 else self.tdtype)
        if t is None:                     # P45：p 就地变成 dp（p − t）
            dp = p.to(_upd_dtype)
            dp[correct] -= 1.0
        else:
            dp = (p - t).to(_upd_dtype)
        if ht.dtype != _upd_dtype:
            ht = ht.to(_upd_dtype)
        if self._sparse:
            # P111 稀疏 rank-1：W[i,j] -= η·dp[i]·h[Wi[i,j]]
            # `addmm_` 是稠密 (n_out,n_h) 的；稀疏下等价写法是 gather 后逐行
            # 外积累加，流量只有 (n_out,k) 而非 (n_out,n_h)。
            g = self._sp_gather(ht)
            self.W.add_(dp.reshape(-1, 1) * g, alpha=-float(eta))
            if self.w_clip > 0.0:
                self.W.clamp_(-self.w_clip, self.w_clip)
            return nll_dev
        self.W.addmm_(dp.reshape(-1, 1), ht.reshape(1, -1), alpha=-float(eta))
        if self.w_clip > 0.0:
            self.W.clamp_(-self.w_clip, self.w_clip)
        return nll_dev

    def _lookup_ht(self, h):
        """复用缓存的设备 h（`forward` 刚上传过同一个 h 时省一次 H2D）。

        返回 (设备张量, 是否命中缓存)。h 很小（n_h 个 fp32），逐元素比较成本
        可忽略，换来的是每步少一次 H2D + 一次分配。
        """
        if self._cache_ht is not None and self._cache_h is not None:
            ha = np.ascontiguousarray(h, dtype=np.float32)
            if ha.shape == self._cache_h.shape and np.array_equal(ha, self._cache_h):
                return self._cache_ht, True
        return self._staged_to_dev(h), False

    # ---------- 与 numba Readout 的接口兼容 ----------
    def learn(self, h, target, eta: float) -> None:
        """非 softmax 感知器更新：**W += η·(t − y) ⊗ h**（对齐生产实际路径）。

        P19 踩坑记录（两处符号不一致，已统一）：
        - `Readout.learn` 走 numba 融合核时是 `W += η·(t−y)⊗h`（dp = y−t，核内取负）；
        - 它的 numpy 回退分支却写成 `W − η·(t−y)⊗h` —— **与融合核符号相反**
          （P7 遗留，只在融合核不可用时暴露）。本实现以**融合核**（生产默认）为
          准；同时已把该回退分支改正，两条路径现语义一致。
        """
        ht, _ = self._lookup_ht(h)
        y = self._matmul(ht)
        t = self._to_dev(target)
        if self._int8:
            # P105：int8 模式 → fp32 域更新 + 重量化写回（语义同 learn_softmax）
            self._int8_update(t.float() - y.float(), ht.float(), eta)
            return
        if self._sparse:
            # P111 稀疏 rank-1（见 _eager_step 的同款实现）
            g = self._sp_gather(ht)
            self.W.add_((t - y).reshape(-1, 1) * g, alpha=float(eta))
            if self.w_clip > 0.0:
                self.W.clamp_(-self.w_clip, self.w_clip)
            return
        # P28：rank-1 AXPY（W += η·(t − y)⊗h），不物化 (n_out×n_in) 临时张量
        self.W.addmm_((t - y).reshape(-1, 1), ht.reshape(1, -1),
                      alpha=float(eta))
        if self.w_clip > 0.0:
            self.W.clamp_(-self.w_clip, self.w_clip)

    def n_synapses(self) -> int:
        """读出的「连接数」：稀疏 =实际存在的突触；稠密 = 元素数。"""
        return int(self.W.numel())

    # ---- 与 Readout 的属性/方法面对齐（P23：三次崩溃的根治）----
    # 全仓库对 `readout.X` 的访问面（见 tests/verifiers/verify_accel_readout.py
    # 的接口扫描用例）：W / learn_softmax / learn / __call__ / n_synapses /
    # **conn_k** / **hidden** / **stats()**。前五项首版已实现，conn_k（ckpt 保存
    # 会访问）与 stats（诊断脚本会访问）缺失 → 生产保存检查点时崩溃。
    @property
    def _csr(self):
        """稀疏 CSR 三元组 (indptr, idx, val) —— 与 `Readout._csr` 同口径。

        P111：检查点保存/恢复（`ckpt_1b`）直接吃这个三元组，故必须导出标准
        CSR 布局而不是内部的 (n_out,k) 张量。
        ⚠ **val 返回的是主机副本**（设备张量不能原地写）。恢复侧
        （`ckpt_1b` 的 `val[:] = ...`）写的是这个 numpy 数组 → 写完不会自动
        同步回设备张量，故另提供 `sync_csr_from_host()` 并在 ckpt 恢复后调用
        （已在 ckpt_1b 接入，见该文件稀疏分支）。
        """
        if not self._sparse:
            raise NotImplementedError(
                "稠密读出没有 CSR 结构（conn_k=0；本属性仅稀疏模式可用）")
        k = self.conn_k
        n_out = self.n_out
        indptr = np.arange(0, (n_out + 1) * k, k, dtype=np.int64)
        idx = self.Wi.detach().cpu().numpy().astype(np.int64).ravel()
        # ⚠ 必须**记住这次导出的数组**：ckpt 恢复侧做的是 `val[:] = z[...]`
        # （原地写这个 numpy 数组），随后才调`sync_csr_from_host()`。
        # 若 sync 里重新 D2H 取一份新拷贝，写入就落在无人引用的临时数组上，
        # 设备权重从未被更新——恢复「成功」但权重还是旧值（B3 抓到的就是这个）。
        val = self.W.detach().float().cpu().numpy().astype(np.float64).ravel()
        self._csr_val_host = val
        return indptr, idx, val

    def sync_csr_from_host(self) -> None:
        """把**上次 `_csr` 导出**的主机 val 写回设备张量（ckpt 恢复后必须调用）。

        ⚠ 不能在这里重新 `_csr`：那会取一份全新的 D2H 拷贝，调用方之前对
        `val[:]` 的原地写入就丢了。必须复用缓存的那一份。
        """
        if not self._sparse:
            return
        val = self._csr_val_host
        if val is None:
            # 没有可用的导出副本 → fail-fast 而不是静默什么都不做
            # （静默 no-op 会让「检查点恢复成功但权重没变」变成隐形 bug）。
            raise RuntimeError(
                "sync_csr_from_host() 在 _csr 之前被调用；请先取 _csr、写入 "
                "val[:]，再调用本方法（ckpt_1b 的稀疏恢复分支已按此顺序接入）")
        self.W.copy_(torch.tensor(
            np.ascontiguousarray(val, dtype=np.float32).reshape(
                self.n_out, self.conn_k),
            device=self.device, dtype=torch.float32))
        self._csr_val_host = None

    @property
    def conn_k(self) -> int:
        """每输出单元入边数（0 = 稠密）。稀疏模式下为实际 k。"""
        return int(self._sparse_k)

    @property
    def hidden(self) -> int:
        """加速后端只实现单级读出（两级在 pick_readout_backend 回落）。"""
        return 0

    @property
    def dtype_name(self) -> str:
        if self._int8:
            return "int8"          # 存储口径（回落 fp16 后 _int8=False → 走下面）
        return {torch.float32: "fp32", torch.float16: "fp16",
                torch.bfloat16: "bf16"}.get(self.W.dtype, str(self.W.dtype))

    def stats(self) -> dict:
        """统计信息（键与 `Readout.stats` 对齐，供诊断/表格使用）。"""
        n = self.n_synapses()
        dense = self.n_out * self.n_h
        st = {"synapses": n, "dense_equivalent": dense,
              "connectivity": n / dense if dense else 1.0,
              "k": self.conn_k, "dtype": self.dtype_name,
              "storage_MB": self.W.numel() * self.W.element_size() / 1e6}
        if self._sparse:
            # P111：稀疏存储还含列索引（int64，设备常驻）。单报 W 会让诊断表
            # 低估真实显存/内存占用 —— 这正是「稀疏省内存」最容易被高估的地方。
            st["index_MB"] = (self.Wi.numel()
                              * self.Wi.element_size() / 1e6)
            st["storage_MB_total"] = st["storage_MB"] + st["index_MB"]
            st["layout"] = "csr-uniform-k"
        return st

    def W_cpu(self) -> np.ndarray:
        """权重拉回主机（检查点 / 统计用）。int8 模式返回**反量化后的实值**。

        P111 稀疏：返回 **(n_out, k) 的 val 矩阵**（不是 (n_out, n_h) 稠密）——
        稠密化会凭空造出 n_out×(n_h−k)/2 个不存在的突触，语义错。ckpt 的稀疏
        分支走 `_csr`，不经过本方法。
        """
        W = self.W.detach().float()
        if self._int8:
            W = W * self._wscale   # 码本值 × scale = 实值（否则 ckpt 全是码）
        return W.cpu().numpy()

    def load_W(self, arr: np.ndarray) -> None:
        """从主机矩阵灌入（检查点恢复；形状须匹配）。int8 模式重量化。"""
        a = np.ascontiguousarray(arr, dtype=np.float32)
        if self._sparse:
            # P111 稀疏：期望 (n_out, k) 的 val 矩阵
            if a.shape != (self.n_out, self.conn_k):
                raise ValueError(f"稀疏形状不符：期望 {(self.n_out, self.conn_k)}，"
                                 f"收到 {a.shape}")
            self.W.copy_(torch.as_tensor(a, device=self.device,
                                         dtype=torch.float32))
            return
        if a.shape != (self.n_out, self.n_h):
            raise ValueError(f"形状不符：期望 {(self.n_out, self.n_h)}，"
                             f"收到 {a.shape}")
        if self._int8:
            # P105：实值 → 码本（同一 scale 语义；越界 clamp 饱和，不回绕）。
            # scale 不随灌入的 W 重标定——保持与训练期同一把尺子。
            t = torch.as_tensor(a, device=self.device, dtype=torch.float32)
            self.W.copy_(torch.clamp(torch.round(t / self._wscale),
                                     -_INT8_QMAX, _INT8_QMAX).to(torch.int8))
            return
        self.W.copy_(torch.as_tensor(a, device=self.device, dtype=self.tdtype))

    # ---------- 内部 ----------
    def _to_dev(self, x):
        # P105：matmul 参与者的 dtype 跟随 **_cdtype**（int8 模式 = fp16，
        # 绝不能是 int8——ht 是实值向量，跟码本 dtype 无关）。
        if torch.is_tensor(x):
            return x.to(device=self.device, dtype=self._cdtype)
        return torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32),
                               device=self.device, dtype=self._cdtype)

    def _staged_to_dev(self, h):
        """P58（fhz「CPU 预计算还要更加提前」）：pinned 暂存 + non_blocking H2D。

        pageable H2D 会阻塞 CPU 直到拷贝完成——CPU 算完 h 后干等传输。pinned
        暂存让 H2D 真异步：CPU 提交后立即回去算下一步的 M1–M5，NPU 流按 FIFO
        消化（设备内再 cast 到 _cdtype——int8 模式下是 fp16，与原 host-cast
        语义同为 RNE 舍入）。
        缓冲池轮转 + Event 覆写保护（pinned 复用前必须确认上次拷贝已完成；
        每步 CPU 几 ms ≫ H2D 几 µs，实际零等待）。pin 不可用（CPU-only torch
        / 驱动限制）时回落同步路径，数值逐位一致。

        P112：**每次上传新 h 都递增 `_ht_epoch`**（主机侧整数，零设备交互），
        供 `_sp_gather` 的缓存判据使用 —— 取代原先那个会强制设备同步的
        `torch.equal`。
        """
        self._ht_epoch += 1
        a = np.ascontiguousarray(h, dtype=np.float32)
        if self._pin_bufs is None:                      # 懒初始化（一次）
            self._pin_bufs = []
            try:
                probe = torch.empty(8, pin_memory=True)
                del probe
                self._pin_ok = True
            except Exception:                           # noqa: BLE001
                self._pin_ok = False
        if not self._pin_ok:
            return torch.as_tensor(a, device=self.device, dtype=self._cdtype)
        idx = self._pin_next % self._pin_cap
        if len(self._pin_bufs) <= idx:                  # 首轮：扩池
            self._pin_bufs.append((torch.empty(self.n_h, dtype=torch.float32,
                                               pin_memory=True), None))
        buf, ev = self._pin_bufs[idx]
        if ev is not None:
            ev.synchronize()                            # 覆写保护（常态零等待）
        buf.copy_(torch.from_numpy(a))
        if ev is None:
            ev = torch.Event()
            self._pin_bufs[idx] = (buf, ev)
        dev_t = buf.to(self.device, non_blocking=True).to(self._cdtype)
        ev.record()
        self._pin_next += 1
        return dev_t


def _unsupported_reason(cfg) -> str | None:
    """返回**不可用原因**（None = 加速读出可用）。

    P19 教训：换后端前必须核对配置——两级读出（`readout_hidden`）、结构性稀疏
    读出（`readout_conn_k`）、fp4 码本三条路径 `AccelReadout` **未实现**。强行上
    会「能跑但语义不同」——比回落到 numba 原路径更糟。命中任一项即回落，并记录
    原因供日志如实报告。

    ⚠ **fp8 已在 P84 实现**（forward 用 fp8 副本 + 更新用 fp16 主副本），曾被
    列在这里导致 `--readout-dtype fp8`（**现在的默认值**）被静默回落到 numba
    CPU ——加新 dtype 时务必同步这张能力表。
    """
    if int(getattr(cfg, "readout_hidden", 0) or 0) > 0:
        return "readout_hidden>0（两级读出未在加速后端实现）"
    # P111：稀疏读出**已在加速后端实现**（均匀 k 的 gather-GEMV），不再回落。
    # 仍需拒绝的只有非均匀行宽（幂律变长 CSR）——那是另一种结构，见AccelReadout
    # 的「只支持均匀 k」注释。若将来有人传入变长 CSR，构造期会fail-fast。
    # P86（fhz 2026-09-30：「针对昇腾设备禁用 fp4, fp8」）：加速后端**禁用
    # 量化码本**——昇腾实测 fp8 抛 "Float8_e4m3fn has not been supported"
    # （ERR01007），fp4 的 MX 块缩放同样没有算子。回落 numba CPU 路径，那里
    # P9/P12 的 fp8/fp4 位算法量化核是可用的（**不是能力缺失，只是没有加速
    # 算子**）。CPU 上想要量化码本可以直接 `--accel cpu --readout-dtype fp8`。
    _rd = str(getattr(cfg, "readout_dtype", "fp32"))
    if _rd in ("int4", "fp4"):
        return ("readout_dtype=int4（910B 无 INT4 矩阵乘单元，只能反量化→FP16 "
                "再算，省存储不省算力；int4 请用 --accel cpu 的 4-bit 打包核）")
    # P105：int8 已实现（存储 int8 + fp16 计算，构造期探测失败时在 AccelReadout
    # 内部**永久回落 fp16**，不走 numba 回落）。旧名 fp8 是 int8 的别名（P100
    # 正名），同样放行。加新 dtype 时务必同步这张能力表（P92 就是漏了 int8 才
    # 导致默认配置静默回落 numba CPU）。
    if _rd in ("int8", "fp8"):
        return None
    if bool(getattr(cfg, "lognormal_init", False)):
        return None                            # 初始化分布不同但结构兼容，不阻断
    return None


def pick_readout_backend(cfg, n_h: int, n_out: int, rng):
    """按 `cfg.accel_readout` 选读出后端；不可用时**回落 numba 原路径**。

    返回 (readout, 后端名)。回落必须静默安全：无加速器 / torch 缺失 /
    配置不兼容 / 构造异常 → 原 `Readout`（默认路径逐位不变），并把原因记在
    `readout._accel_fallback_reason` 上（不静默）。
    """
    def _fallback(reason: str):
        from ..readout import Readout
        ro = Readout(n_h, n_out, rng, w_clip=cfg.readout_w_clip,
                     dtype=cfg.readout_dtype, conn_k=cfg.readout_conn_k,
                     lognormal_init=cfg.lognormal_init)
        ro._accel_fallback_reason = reason
        return ro, "numba-cpu(回落)"

    spec = str(getattr(cfg, "accel_readout", "auto") or "auto").lower()
    if spec in ("", "cpu", "off", "numba"):
        from ..readout import Readout
        return Readout(n_h, n_out, rng, w_clip=cfg.readout_w_clip,
                       dtype=cfg.readout_dtype, conn_k=cfg.readout_conn_k,
                       lognormal_init=cfg.lognormal_init), "numba-cpu"
    bad = _unsupported_reason(cfg)
    if bad is not None:
        return _fallback(bad)
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
        return (AccelReadout(n_h, n_out, rng, device=spec,
                             dtype=cfg.readout_dtype,
                             w_clip=cfg.readout_w_clip,
                             nll_sync_every=int(getattr(cfg, "nll_sync_every", 1)),
                             compile=bool(getattr(cfg, "torch_compile", False)),
                             compile_mode=str(getattr(cfg,
                                                     "torch_compile_mode",
                                                     "default")),
                             conn_k=int(getattr(cfg, "readout_conn_k", 0) or 0),
                             lognormal_init=bool(getattr(cfg, "lognormal_init",
                                                         False)),
                             exc_ratio=float(getattr(cfg, "exc_ratio", 0.8))),
                f"accel:{spec}")
    except Exception as e:                                   # noqa: BLE001
        return _fallback(f"{type(e).__name__}: {e}")
