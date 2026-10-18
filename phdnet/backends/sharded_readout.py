# -*- coding: utf-8 -*-
"""P204：多卡读出（N 卡列并行）——训练热路径的多卡接入。

## 定位（先读这个，防止误读为通用多卡训练框架）

PHD-Net 是逐 token 事件驱动的稀疏网络：M1-M5 在 CPU（numba），只有 M6 读出
在加速器上。多卡的可并行部分就是**读出头的 V 维（词表行）**：
  - 前向 y = W·h：按行切分后各卡独立算分片 y，cat 回主卡（行切分逐位等价）；
  - 更新 ΔW = -η·dp⊗h：dp 按行分片下发，各卡本地 rank-1（行独立，零通信依赖）；
  - softmax 主回路在主卡 fp32（与单卡 P9 协议一致）。
上游 PC 栈 / STDP / LTM 仍是主设备单卡——它们的事件驱动全局状态
（WM/STDP 迹/LTM 印迹）跨样本连续携带，不可按 batch 切（P14 已论证）。

## 实现策略：组合而非侵入

`AccelReadout`（2523 行）承载 int8/fp8/master/graph 等十余种存储模式的语义，
`self.W` 被几十处直接操作。把多卡塞进去 = 每处都是语义陷阱。
本类**持有 N 个完整 `AccelReadout` 分片**（各持 V/N 行、各自设备、
各自的降级链/主权重/图捕获独立运作），对外暴露与 `AccelReadout` 相同的
核心调用面（`forward` / `forward_dev` / `learn_softmax` / `load_W` /
`stats`），由 `make_readout`（accel_readout.py 尾部工厂）在
`len(devices)>1` 时自动分发。

## 全局 softmax 的必要性（与 MultiDeviceReadout 的差异）

P204 版本**必须**用完整 y 做主卡 softmax：读出是 softmax 分类头，p[c] 依赖
**全部** V 行 logits（跨行归一化）——分卡各自 softmax 会得到 V/N 个互不
相干的分布（数值完全错）。P14 的 gate_multi_device D 例之所以能逐位对拍，
是因为它在**同一设备**上分片 + cat 后仍用完整 y 算 softmax——本类沿用同一
协议，只是把「cat 后的完整 y 的 rank-1 更新」按行区间**分发**回各分卡。

## 诚实边界
- **通信**：每步 y（V×4B 下发）+ dp 分片（上行），V=51,962 时 ≈ 208 KB×2；
  H2D 走各分卡自身的流，与主卡计算天然重叠。
- **等价判据**：同设备分片 = 逐位一致（gate_multi_device D 例协议）；
  跨设备 = 容差一致（各卡 matmul 归约顺序可能不同）。
- **不并行**：M1-M5（numba CPU 物理限制）、LTM/STDP/WM（全局事件状态）。
- master weight：各分卡独立持自己行的 fp32 master（行切分，语义不变）。
- 1B 档读出占端到端 ~19.6%（P202 后），N 卡理论上限 ≈ 1/(1-0.196+0.196/N)
  （N=4 → 1.15×）。**读出不再是瓶颈时多卡收益趋零**——诚实预期。
- **稀疏读出（--readout-conn-k 128，生产默认）暂不支持多卡**：CSR 的
  Wi 列索引是全局词表索引，行分片后仍指向全局列（无需列切分），但
  `_csr` 导出/ckpt 恢复的行区间拼接未实现 → P204 只放行稠密读出，
  稀疏多卡 fail-fast 提示。生产要吃多卡收益需 `--readout-conn-k 0`。
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:                                            # pragma: no cover
    torch = None

from .multi_device import resolve_devices, shard_ranges      # noqa: E402


class ShardedAccelReadout:
    """N 卡读出列并行：持 N 个 `AccelReadout` 分片，调用面对齐单卡。"""

    def __init__(self, n_h: int, n_out: int, rng=None, devices=None,
                 dtype: str = "fp32", w_clip: float = 0.0, w0=None,
                 conn_k: int = 0, **kw):
        if torch is None:
            raise RuntimeError("未安装 torch，多卡读出不可用")
        if conn_k and conn_k > 0:
            raise NotImplementedError(
                "多卡 × 稀疏读出（--readout-conn-k>0）在 P204 未实现："
                "CSR 的 _csr 导出/ckpt 恢复需要按行区间拼接 Wi/val。"
                "多卡请用稠密读出（--readout-conn-k 0）。")
        if n_out < 2:
            raise ValueError(f"多卡读出要求 n_out ≥ 2（收到 {n_out}）")
        # ⚠ 设备解析：**显式传入时不过滤**（否则 ["cpu","cpu"] 这种
        #   「同设备多分片」的等价性对拍会被误判为单卡 → fail-fast。
        #   过滤只在 auto 路径做（auto 解析出 ["cpu"] 说明无加速器）。
        if devices:
            from .multi_device import resolve_devices as _rd
            devs = _rd(",".join(devices), allow_fallback=False)
        else:
            devs = [d for d in resolve_devices("auto") if d != "cpu"]
        if len(devs) < 2:
            raise RuntimeError(
                f"多卡读出需要 ≥2 个设备（解析结果 {devs}）。"
                "单卡请直接用 AccelReadout（make_readout 已自动分发）；"
                "若在无加速器的机器上做等价性对拍，请显式传 "
                "devices=['cpu','cpu']（同设备多分片 = 逐位判据）。")
        self.devices = devs
        self.main = devs[0]
        self.n_h = int(n_h)
        self.n_out = int(n_out)
        self.dtype = str(dtype)
        self.w_clip = float(w_clip)
        self.conn_k = 0
        self.shards = shard_ranges(self.n_out, len(self.devices))
        # w0 按行区间切给各分片（外部传入完整 (V,H) 时保持行对应）
        w0_parts = None
        if w0 is not None:
            arr = np.asarray(w0)
            if arr.shape[0] != self.n_out:
                raise ValueError(f"w0 行数不符：期望 {self.n_out}，"
                                 f"收到 {arr.shape[0]}")
            w0_parts = [arr[s:e] for (s, e) in self.shards]
        # rng：各分片用独立子流（确定性拆分，与单卡行为可对拍）
        r = rng if rng is not None else np.random.default_rng(0)
        self.parts = []
        for (s, e), d in zip(self.shards, self.devices):
            sub_rng = np.random.default_rng(int(r.integers(0, 2**63 - 1)))
            kw2 = dict(kw)
            kw2.pop("multi", None)
            from .accel_readout import AccelReadout
            self.parts.append(AccelReadout(
                n_h, e - s, rng=sub_rng, device=d, dtype=dtype,
                w_clip=w_clip, w0=(w0_parts[self.parts.__len__()]
                                   if w0_parts is not None else None),
                conn_k=0, **kw2))
        self._cache_y: list | None = None
        self._cache_h: np.ndarray | None = None
        self._is_accel = all(getattr(p, "_is_accel", False) for p in self.parts)
        self._last_nll = 0.0
        # nll 同步协议取各分片最大值（保守：任一分片要求同步就同步）
        self.nll_sync_every = max(int(getattr(p, "nll_sync_every", 1))
                                  for p in self.parts)
        self._nll_sum = None
        self._nll_n = 0

    # ---------- 前向 ----------
    def forward_dev(self, h):
        """设备直通前向：各分卡并行提交，返回主卡上的**拼接 fp32 y**。"""
        ys = [p.forward_dev(h) for p in self.parts]
        self._cache_y = ys
        self._cache_h = np.asarray(h)
        if len(ys) == 1:
            return ys[0]
        # cat 到主卡：分片行区间 = shard_ranges，拼接序 = 全局行序
        # （与单卡全局 y 逐位对应的**前提**：行切分只切 V 维、每分片完整 H 列）
        return torch.cat([y.float().to(self.main) for y in ys], dim=0)

    def forward(self, h) -> np.ndarray:
        y = self.forward_dev(h)
        return y.detach().cpu().numpy()

    def __call__(self, h) -> np.ndarray:
        return self.forward(h)

    # ---------- 学习 ----------
    def learn_softmax(self, h, target, eta: float, y_pre=None,
                      accumulate: int = 1, target_idx: int | None = None):
        """主卡全局 softmax（完整 y）→ dp 行分片 → 各分卡 rank-1 更新。

        与单卡 `learn_softmax` 的**语义差异只有一处**：更新不在主卡一张 W 上
        完成，而是把 dp 按行区间切片、各分卡对自己的 W_i 做 rank-1——
        行间独立（dp_i⊗h 只写本片行），与「主卡整张 W rank-1」逐行等价
        （同设备分片逐位一致；跨设备容差一致，归约顺序不同）。
        nll：主卡 fp32 全局算（与单卡 P9 协议一致），同步节奏对齐
        nll_sync_every（取各分片最大值）。
        """
        if y_pre is not None and torch.is_tensor(y_pre):
            y_main = y_pre if y_pre.device.type == self.main.split(":")[0] \
                else y_pre.to(self.main)
        else:
            y_main = self.forward_dev(h)
            self._cache_y = [p._cache_y for p in self.parts] \
                if all(getattr(p, "_cache_y", None) is not None
                       for p in self.parts) else None
        y32 = y_main.float()
        # 主卡 softmax + nll（完整 y——softmax 是跨行归一化，不可分片）
        p = torch.softmax(y32, dim=0)
        if target_idx is None:
            target_idx = int(np.argmax(np.asarray(target)))
        correct = int(target_idx)
        nll_dev = (torch.logsumexp(y32, dim=0) - y32[correct]).reshape(())
        # dp 行分片 → 各分卡 rank-1（行独立，零通信依赖；同设备切片零拷贝）
        for pi, ((s, e), part) in enumerate(zip(self.shards, self.parts)):
            dp_i = p[s:e]
            if s <= correct < e:
                dp_i = p[s:e].clone()
                dp_i[correct - s] -= 1.0
            part.apply_dp(dp_i, h, eta)
        # nll 同步节奏（对齐单卡 P34 协议）
        if self.nll_sync_every <= 1:
            self._last_nll = float(nll_dev.item())
            return self._last_nll
        self._nll_sum = (nll_dev if self._nll_sum is None
                         else self._nll_sum + nll_dev)
        self._nll_n += 1
        if self._nll_n >= self.nll_sync_every:
            self._last_nll = float(self._nll_sum.item()) / self._nll_n
            self._nll_sum = None
            self._nll_n = 0
        return self._last_nll

    # ---------- 检查点兼容 ----------
    def load_W(self, a) -> None:
        """按行区间把完整 (V,H) 灌入各分片（ckpt 恢复路径）。"""
        arr = np.asarray(a)
        if arr.shape[0] != self.n_out:
            raise ValueError(f"形状不符：期望 ({self.n_out}, ·)，收到 {arr.shape}")
        for (s, e), part in zip(self.shards, self.parts):
            part.load_W(arr[s:e])

    def stats(self) -> dict:
        return {"multi_device": True, "devices": self.devices,
                "shards": self.shards, "n_out": self.n_out,
                "dtype": self.dtype}

    # 兼容属性（训练循环/工厂会读）
    @property
    def device(self):
        return self.main

    @property
    def _master_weight(self):
        return all(getattr(p, "_master_weight", False) for p in self.parts)

    @property
    def _sparse(self):
        return False
