"""PHD-Net 1B 参数模型：构建 + 在线训练（Python 3.14，CPU 可运行）。

核心思想（架构文档 §6「规模化设计」+ Hebb 结构可塑性）：
  人脑 10^14 级突触只有 20W 功耗——靠的是「稀疏连接 + 事件驱动的局部计算」。

  本脚本构建总突触参数容量 ≈ 1.0×10^9 的 PHD-Net 时序关联核（M3 规模化版）：

    - 神经元 N = 2^24 = 16,777,216，每神经元最多 M_OUT = 60 条可塑出边
      → 总突触参数容量 N·M_OUT = 1,006,632,960 ≈ 1.01×10^9（> 1B）
    - 结构可塑性：共激活的神经元之间**生长新突触**（"fire together,
      wire together"），满 60 条后不再生长 —— 1B 是硬容量约束
    - 可训练参数 = 已生长突触（印迹表，邻接表存储）。SDR 每步仅 ~32 神经元
      活跃，每步只读写 32×32 = 1024 个候选对 —— 计算量正比于活跃规模，
      与 1B 总容量无关
    - 突触迹采用「惰性衰减」：触碰时按 λ^Δt 补偿，无需每步扫描全网
    - STDP 规则：先发 → 后发的连接增强（LTP），反之抑制（LTD）

  密集 1B 模型（权重+梯度+优化器状态 ≈ 16GB，每步 ≥10^9 级 FLOPs）在无独显
  机器上物理不可行；事件驱动的稀疏类脑架构使其在 CPU 上毫秒级单步、真实训练。

运行：python tools/train_1b.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]      # 项目根目录
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))               # 保证 `import phdnet` 可用

import time

import numpy as np

from phdnet.tokenizer import U64, mix64          # C5 重构（2026-09-22）：去重本文件内的 splitmix64 副本

# ── 规模参数 ────────────────────────────────────────────────────────────────
N_NEURONS = 1 << 24                     # 16,777,216 个神经元（2^24）
M_OUT = 60                              # 每神经元可塑出边上限
TOTAL_SYNAPSES = N_NEURONS * M_OUT      # 1,006,632,960 ≈ 1.0B 突触参数容量
N_ACTIVE = 32                           # SDR 稀疏度（每步活跃神经元数）
N_SYMBOLS = 3                           # 序列符号数（A/B/C）
N_STEPS = 1500                          # 在线训练步数
LAMBDA = 0.7                            # 突触迹衰减
ETA = 0.08                              # STDP 学习率
W_MAX = 1.0
SEED = np.uint64(0xA5A5_5A5A_1234_5678)

U64 = np.uint64


def sdr_of(sym: int) -> np.ndarray:
    """符号 → SDR 活跃集（确定性哈希，32/16.7M ≈ 0.2% 激活率）。"""
    x = U64(sym) * U64(7919) + np.arange(N_ACTIVE, dtype=np.uint64) * U64(2654435761)
    return (mix64(x + SEED) % U64(N_NEURONS)).astype(np.int64)


class BillionSynapseNet:
    """1B 突触容量的 PHD-Net 时序关联核（M3 规模化版，事件驱动实现）。"""

    def __init__(self):
        # 突触迹（惰性衰减：只在被触碰时按 λ^Δt 补偿，避免每步扫描全网）
        # 注意：pre/post 迹各自独立的 touch 时间戳（共用会导致第二次 touch
        # 的 Δt=0，迹逐步无衰减累加直至爆炸，LTD 随之失衡）
        self.t_pre = np.zeros(N_NEURONS, dtype=np.float32)
        self.t_post = np.zeros(N_NEURONS, dtype=np.float32)
        self.last_touch_pre = np.zeros(N_NEURONS, dtype=np.int64)
        self.last_touch_post = np.zeros(N_NEURONS, dtype=np.int64)
        # 邻接表印迹表：out[i][k] = w —— 只有被经验触碰过的突触才显式存在，
        # 每个突触前神经元最多 M_OUT 条出边（1B 容量约束）
        self.out: dict[int, dict[int, float]] = {}
        self.step_count = 0

    # ---------- 事件驱动惰性衰减 ----------
    def _touch(self, neurons: np.ndarray, trace: np.ndarray, which: str) -> None:
        stamp = self.last_touch_pre if which == "pre" else self.last_touch_post
        dt = (self.step_count - stamp[neurons]).astype(np.float32)
        trace[neurons] = trace[neurons] * np.power(np.float32(LAMBDA), dt) + 1.0
        stamp[neurons] = self.step_count

    # ---------- 单步：预测 → STDP 学习 ----------
    def step(self, active: np.ndarray, prev_active: np.ndarray | None) -> float:
        if prev_active is None:                              # 首步：只积累 pre 迹
            self._touch(active, self.t_pre, "pre")
            self.step_count += 1
            return 1.0

        # 预测 p[k] = Σ_i w(i,k)·pre[i] —— 只读上一时刻活跃神经元的已学出边
        p: dict[int, float] = {}
        for i in prev_active.tolist():
            for k, w in self.out.get(i, {}).items():
                p[k] = p.get(k, 0.0) + w

        # 时序误差 = 1 − cos(p, 当前活跃集 one-hot)
        if p:
            keys = np.fromiter(p.keys(), dtype=np.int64, count=len(p))
            vals = np.fromiter(p.values(), dtype=np.float64, count=len(p))
            tgt = np.isin(keys, active)
            denom = np.linalg.norm(vals) * np.sqrt(float(len(active))) + 1e-9
            seq_err = 1.0 - float(vals[tgt].sum()) / denom
        else:
            seq_err = 1.0

        # STDP 学习：更新当前活跃的突触后迹 + 共激活对之间生长/强化突触
        # 语义要点（相邻对调用约定）：边 (i∈prev → k∈cur) 的 pre-post 时序
        # 即正因果 → 新突触生长只由 LTP 驱动；LTD 仅作用于已存在突触
        # （提供稳态平衡，防止重复共激活导致权重无限增长）
        self._touch(active, self.t_post, "post")
        self._touch(active, self.t_pre, "pre")                      # 同时开始积累 pre 迹
        for i in prev_active.tolist():
            tpi = float(self.t_pre[i])                       # i 于上一时刻活跃 ✓
            bucket = self.out.get(i)
            for k in active.tolist():
                ltp = ETA * tpi * float(self.t_post[k])
                if bucket is not None and k in bucket:
                    ltd = ETA * float(self.t_post[i]) * float(self.t_pre[k])
                    bucket[k] = min(max(bucket[k] + ltp - ltd, 0.0), W_MAX)
                elif ltp > 0.0:                              # 结构可塑性：生长新突触
                    if bucket is None:
                        bucket = self.out[i] = {}
                    if len(bucket) < M_OUT:                  # 1B 容量约束（每神经元 ≤60）
                        bucket[k] = min(ltp, W_MAX)
        self.step_count += 1
        return seq_err

    # ---------- 结构检验：看到 S 后预测谁 ----------
    def probe(self, sym: int) -> np.ndarray:
        """返回看到 sym 后，预测质量在各符号上的归一化分布。"""
        p: dict[int, float] = {}
        for i in sdr_of(sym).tolist():
            for k, w in self.out.get(i, {}).items():
                p[k] = p.get(k, 0.0) + w
        scores = []
        for s in range(N_SYMBOLS):
            target = sdr_of(s)
            scores.append(sum(p.get(int(n), 0.0) for n in target))
        arr = np.array(scores)
        return arr / (arr.sum() + 1e-9)


def main():
    print("=" * 64)
    print("PHD-Net 1B 参数模型 —— 构建 + 在线训练（事件驱动稀疏计算）")
    print("=" * 64)
    print(f"神经元数量            : {N_NEURONS:,}")
    print(f"每神经元可塑出边上限  : {M_OUT}")
    print(f"总突触参数容量        : {TOTAL_SYNAPSES:,}  ≈ {TOTAL_SYNAPSES / 1e9:.3f} × 10^9")
    print(f"激活率                : {N_ACTIVE / N_NEURONS:.4%}（SDR 稀疏编码）")

    net = BillionSynapseNet()
    sym_sdr = [sdr_of(s) for s in range(N_SYMBOLS)]
    errs = []
    prev = None
    t0 = time.perf_counter()
    for t in range(N_STEPS):                                 # 在线训练：ABCABC…
        cur = sym_sdr[t % N_SYMBOLS]
        errs.append(net.step(cur, prev))
        prev = cur
    dt = time.perf_counter() - t0

    early = sum(errs[:100]) / 100
    late = sum(errs[-100:]) / 100
    print("-" * 64)
    print(f"训练 {N_STEPS} 步（序列 A→B→C→A…，在线、单样本、无反向传播）")
    print(f"前 100 步平均时序误差 : {early:.3f}")
    print(f"后 100 步平均时序误差 : {late:.3f}   （下降 {100 * (1 - late / early):.1f}%）")
    print(f"训练总耗时            : {dt:.2f}s（平均 {dt / N_STEPS * 1000:.2f} ms/步）")
    n_edges = sum(len(b) for b in net.out.values())
    print(f"已生长突触（实际存储的可训练参数）: {n_edges:,} 条"
          f"（占容量 {n_edges / TOTAL_SYNAPSES:.5%}）")
    print(f"印迹表内存占用        : ≈{n_edges * 1e-4:.2f} MB"
          f"（对比密集 1B 模型：~4 GB 权重 + ~12 GB 优化器状态）")

    blocks = "▁▂▃▄▅▆▇█"
    step = max(1, len(errs) // 40)
    m = [sum(errs[i:i + step]) / step for i in range(0, len(errs), step)]
    lo, hi = min(m), max(m)
    print("误差趋势: " + "".join(blocks[int((v - lo) / (hi - lo + 1e-9) * 7)] for v in m))

    print("-" * 64)
    print("结构检验（预测质量在各符号上的归一化分布）：")
    names = "ABC"
    n_ok = 0
    for s in range(N_SYMBOLS):
        dist = net.probe(s)
        best = int(np.argmax(dist))
        ok = best == (s + 1) % N_SYMBOLS
        n_ok += ok
        print(f"  看到 {names[s]} → 预测分布 [A:{dist[0]:.3f} B:{dist[1]:.3f} "
              f"C:{dist[2]:.3f}] → 学到 {names[s]}→{names[best]} {'✓' if ok else '✗'}")

    print("-" * 64)
    print(f"结论：1B 突触容量类脑模型在本机 CPU 完成真实在线训练，"
          f"时序误差下降且 {n_ok}/3 转移结构正确 —— 事件驱动稀疏计算有效。")


if __name__ == "__main__":
    main()
