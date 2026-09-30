"""PHD-Net 1B 档配置与容量验算（train_1b 子项目专用）。

容量口径（与项目既有 1B 语义一致：tools/train.py、tools/bench_1b_migrate.py）
--------------------------------------------------------------------------------
「1B 模型」= **总突触参数容量 ≥ 1.0×10^9**，由两部分构成：

1. 事件驱动大空间长期记忆（M4b 容量栈，big_ltm）：
       容量 = big_ltm_N × big_ltm_m = 2^24 × 72 = 1,207,959,552 ≈ 1.208×10^9
       （fhz 2026-09-28 指令：突触生长为原本的 1.2×，原 2^24 × 60 ≈ 1.0066×10^9）
   这是 1B 的**主体**。突触不是构建即存在，而是随经验**生长**
   （结构可塑性："fire together, wire together"），每神经元 ≤60 条出边是硬容量约束。
   每步计算量只正比于活跃神经元数（事件驱动），与 1B 总容量无关。

2. 固定突触（构建即存在）：稀疏主干 CSR（M2）+ STDP 侧向核（M3）
   + 编码器（M1）+ 读出头（M6）。

总容量 ≈ 1.0066×10^9 + 固定突触 ≈ **1.01–1.07×10^9 ≥ 1B** ✓

诚实边界
--------
- count_params 口径计入的是「已生长突触数」；训练初期大空间表利用率低，
  随训练增长趋近容量上限（`--report` 可随时查看 stats）。
- 本机（纯 CPU、12.6 GB RAM）只能冒烟验证管线与小步数训练；
  10^9 token 生产训练需 GPU/集群（README.md「硬件需求」一节）。
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.config import PHDNetConfig          # noqa: E402

# 词级分词参数（与 tools/train_production.py 的 SEG 完全一致，保证可比性）
SEG_KWARGS = dict(max_len=6, min_count=5, min_entropy=1.0)

# 预设档：smoke = 管线验证（分钟级）；1b = 标准档（容量 ≥1B）；1b_max = 大主干档
PRESETS = {
    #             width  conn_k  big_N    big_m  k_hash
    "smoke": dict(width=256, conn_k=32, big_n=1 << 20, big_m=60, k_hash=4),
    "1b":    dict(width=1024, conn_k=128, big_n=1 << 24, big_m=72, k_hash=4),
    "1b_max": dict(width=4096, conn_k=512, big_n=1 << 24, big_m=60, k_hash=4),
    # P47（fhz「训练模型大小改为 30B」）：参数主体是 big_ltm（1B 档 2^24×72 =
    # 1.208B，占 99%），故 30B 档 = 2^29 神经元 × 56 突触 = 30.06B + 读出 0.23B
    # ≈ **30.3B**。m 从 72 降到 56 以控制总容量（2^29×72 = 38.6B 会超）。
    # ⚠ 诚实边界：① 稠密下 30.3B × fp32 = **121 GB** 权重（依赖大空间表的在线
    # CSR 稀疏存储才落得住；服务器内存需 ≥ 200 GB）；② 训练速度随 LTM 访存
    # 线性放大（1B 档 ~20 ms/tok → 30B 档预计数百 ms/tok），**单机 1M tokens
    # 不现实**，需要多机/长跑预算；③ 词表/读出规模不变（73,958×3,072）。
    "30b":   dict(width=1024, conn_k=128, big_n=1 << 29, big_m=56, k_hash=4),
}


def build_cfg(preset: str = "1b", width: int = 0, big_n: int = 0,
              csr_online: bool = False, readout_conn_k: int = 0,
              seed: int = 11) -> PHDNetConfig:
    """构造 1B 档配置。

    与生产基线（tools/train_production.py build_cfg）同配方：
    表征冻结（eta_pc=eta_oja=0）、pred_in_readout、结构性稀疏主干（conn_k）、
    big_ltm 1B 容量栈；另开放 csr_online / readout_conn_k 供长跑与大词表场景。
    """
    p = PRESETS[preset]
    w = width if width > 0 else p["width"]
    k = p["conn_k"] if width <= 0 else max(8, w // 8)      # 覆盖宽度时保持 12.5% 连接率
    n = big_n if big_n > 0 else p["big_n"]
    return PHDNetConfig(
        n_sdr=w, n_mid=w, n_top=w,
        k_sparse=max(8, w // 8),                 # 稀疏编码 6.25%（库默认 16，此处按档位放大）
        eta_pc=0.0, eta_oja=0.0,                 # 表征冻结（铁律：多轮实测最优解）
        eta_stdp=0.02, seed=seed,
        pred_in_readout=True,
        sparse_conn=True, conn_k=k,              # 结构性稀疏主干（库默认开启）
        readout_conn_k=readout_conn_k,           # 0 = 稠密读出（主路径）；>0 = CSR 稀疏读出
        big_ltm=True,                            # 1B 主体：大空间事件驱动容量栈
        big_ltm_N=n, big_ltm_m=p["big_m"], big_ltm_k=p["k_hash"],
        csr_online=csr_online,                   # O1：在线可写 CSR（长跑内存更优，逐位等价）
    )


def capacity_report(cfg: PHDNetConfig, vocab_size: int) -> dict:
    """按 count_params 同口径验算突触参数容量与内存预算（不构建网络，纯算术）。"""
    n_in = 2 * cfg.n_sdr                                   # T2.3 组合输入
    # M1 编码器（稠密 W + b）
    n_enc = n_in * cfg.n_sdr + cfg.n_sdr
    # M2 主干 CSR：4 层，每层 n_out × conn_k 条存在的边（conn_k=0 → n_in//8）
    k = cfg.conn_k if cfg.conn_k > 0 else 0
    layers = [(cfg.n_sdr, cfg.n_mid), (cfg.n_mid, cfg.n_top),
              (cfg.n_top, cfg.n_mid), (cfg.n_mid, cfg.n_sdr)]
    n_pc = sum(n_out * (k if k > 0 else max(1, n_in // 8)) for n_in, n_out in layers)
    # M3 STDP 侧向核
    n_stdp = cfg.n_top * cfg.m_lateral
    # M6 读出（稠密 = 元素数；稀疏 = n_out × conn_k）
    n_h = cfg.n_top * (3 if cfg.pred_in_readout else 2) + max(0, cfg.retrieval_topk)
    n_ro = vocab_size * (cfg.readout_conn_k if cfg.readout_conn_k > 0 else n_h)
    # M4b 大空间表：容量（事件驱动，已生长数 ≤ 容量）
    n_big_cap = cfg.big_ltm_N * cfg.big_ltm_m if cfg.big_ltm else 0
    fixed = n_enc + n_pc + n_stdp + n_ro
    total_cap = fixed + n_big_cap
    mem = {
        "encoder_MB": n_enc * 8 / 1e6,
        "pc_csr_MB": n_pc * 12 / 1e6,                      # int32 索引 + fp64 权重
        "stdp_MB": n_stdp * 8 / 1e6,
        "readout_MB": n_ro * 8 / 1e6,
        # dict 版 ≈100 B/条（boxed entry）；在线 CSR 版 ≈10 B/条
        "bigltm_grown_1M_MB": (1e6 * 100 / 1e6) if not cfg.csr_online else (1e6 * 10 / 1e6),
    }
    return {
        "fixed_synapses": fixed,
        "encoder": n_enc, "pc": n_pc, "stdp": n_stdp, "readout": n_ro,
        "bigltm_capacity": n_big_cap,
        "bigltm_neuron_map_MB": (cfg.big_ltm_N * cfg.big_ltm_k * 8 / 1e6) if cfg.big_ltm else 0,
        "total_capacity": total_cap,
        "total_capacity_B": total_cap / 1e9,
        "readout_dense_equiv": vocab_size * n_h,
        "mem": mem,
        "mem_static_MB": sum(v for k2, v in mem.items() if k2 != "bigltm_grown_1M_MB"),
        "cfg_dict": asdict(cfg),
    }


def print_capacity_report(rep: dict) -> None:
    """启动时打印容量验算表（1B 口径透明化）。"""
    m = rep["mem"]
    print("-" * 76)
    print("突触参数容量验算（1B 口径：固定突触 + 大空间表容量）")
    print(f"  M1 编码器（稠密）        : {rep['encoder']:>13,}")
    print(f"  M2 主干 CSR 稀疏连接     : {rep['pc']:>13,}")
    print(f"  M3 STDP 侧向核           : {rep['stdp']:>13,}")
    print(f"  M6 读出头                : {rep['readout']:>13,}"
          f"   （稠密等价 {rep['readout_dense_equiv']:,}）")
    print(f"  固定突触小计             : {rep['fixed_synapses']:>13,}")
    print(f"  M4b 大空间表容量（事件驱动）: {rep['bigltm_capacity']:>13,}")
    print(f"  总突触参数容量           : {rep['total_capacity']:>13,}"
          f"   ≈ {rep['total_capacity_B']:.3f} × 10^9"
          f"   {'≥ 1B ✓' if rep['total_capacity_B'] >= 1.0 else '⚠ < 1B（smoke 档仅验证管线）'}")
    print(f"  静态内存 ≈ {rep['mem_static_MB']:.0f} MB"
          f"（编码器 {m['encoder_MB']:.0f} + 主干 {m['pc_csr_MB']:.1f} + 读出 {m['readout_MB']:.0f}"
          f" + STDP {m['stdp_MB']:.1f}）")
    print(f"  大空间表索引映射 ≈ {rep['bigltm_neuron_map_MB']:.0f} MB"
          f"；已生长突触按 100 万条 ≈ {m['bigltm_grown_1M_MB']:.0f} MB")
    print("-" * 76)
