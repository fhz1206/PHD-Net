"""PHD-Net 性能基准测试 —— 优化前后定量对比（Python 3.14，Windows）。

测量内容：
  A. STDP 关联核：优化前（全网稠密更新）vs 优化后（事件驱动活跃行更新），
     在两个网络规模下对比 —— 呈现事件驱动的收益随规模/激活率的变化
  B. 本机 CPU 有效算力（numpy matmul GFLOPS）—— 用于推算密集 1B 对照物
     的单步时间（该对照物在本机不可运行，推算值报告中明确标注）
  C. 1B 突触模型实际运行：wall / CPU 时间、峰值 RAM（psutil）

说明：优化前基线 = 开发过程中初始可运行版本的计算形态（全网更新 +
稠密输入）；两者在相同数据、相同硬件、相同解释器下顺序测量。
"""

# --- 目录结构调整（2026-09-18）：脚本位于 tests/ 或 tools/ 子目录 ---
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]     # 项目根目录
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))              # 保证 `import phdnet` 可用
_DOC = _ROOT / "eval_corpus" / "internal_corpus.txt"    # 默认语料
# --- 引导结束 ---

import platform
import sys
import time

import numpy as np
import psutil

from phdnet.plasticity import STDPCore

PROC = psutil.Process()


def peak_rss_mb() -> float:
    return PROC.memory_info().peak_wset / (1024 * 1024)


# ── 优化前基线：全网稠密更新（初始实现的计算形态） ───────────────────────────
def legacy_stdp_step(W, post_idx, t_pre, t_post, pre, post, eta, w_max):
    """全网更新：对全部 n×m 条拓扑边计算 dw（含非活跃突触前神经元）。"""
    dw = eta * (t_pre[:, None] * post[post_idx] - t_post[:, None] * pre[post_idx])
    np.clip(W + dw, 0.0, w_max, out=W)


def legacy_predict(W, post_idx, pre):
    return (W * pre[post_idx]).sum(axis=1)


def bench_stdp(n, m, steps, active, lam=0.35, eta=0.12):
    rng = np.random.default_rng(0)
    post_idx = np.array([rng.choice(n, m, replace=False) for _ in range(n)])
    pre_seq = []
    for _ in range(3):
        v = np.zeros(n)
        v[rng.choice(n, active, replace=False)] = rng.uniform(0.5, 0.8, active)
        pre_seq.append(v)

    # 优化前：全网稠密
    W = np.zeros((n, m))
    t_pre = np.zeros(n)
    t_post = np.zeros(n)
    c0 = time.process_time()
    for t in range(steps):
        pre, post = pre_seq[t % 3], pre_seq[(t + 1) % 3]
        legacy_predict(W, post_idx, pre)
        t_pre = lam * t_pre + pre
        legacy_stdp_step(W, post_idx, t_pre, t_post, pre, post, eta, 1.0)
        t_post = lam * t_post + post
    t_legacy = time.process_time() - c0

    # 优化后：事件驱动活跃行（当前 STDPCore numpy 回退路径）
    core = STDPCore(n, m, lam, eta, 1.0, rng)
    c0 = time.process_time()
    for t in range(steps):
        pre, post = pre_seq[t % 3], pre_seq[(t + 1) % 3]
        core.predict(pre)
        core.step(pre, post, eta_scale=0.5)
    t_current = time.process_time() - c0

    return t_legacy, t_current


def bench_cpu_flops(n=1024, reps=20):
    a = np.random.random((n, n))
    b = np.random.random((n, n))
    a @ b
    c0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    dt = time.perf_counter() - c0
    return reps * 2 * n ** 3 / dt / 1e9


def bench_1b(steps=1500):
    import train_1b as T
    net = T.BillionSynapseNet()
    sdr = [T.sdr_of(s) for s in range(3)]
    prev = None
    c0_cpu = time.process_time()
    w0 = time.perf_counter()
    for t in range(steps):
        cur = sdr[t % 3]
        net.step(cur, prev)
        prev = cur
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0_cpu
    edges = sum(len(b) for b in net.out.values())
    return wall, cpu, peak_rss_mb(), edges


if __name__ == "__main__":
    print("=" * 64)
    print("测试环境")
    print("=" * 64)
    vm = psutil.virtual_memory()
    print(f"OS        : {platform.platform()}")
    print(f"Python    : {sys.version.split()[0]}  ({sys.executable})")
    print(f"numpy     : {np.__version__} / psutil {psutil.__version__}")
    print(f"CPU       : {platform.processor()}（8 线程）, GPU: 无（纯 CPU 测试）")
    print(f"系统内存  : {vm.total / 1e9:.1f} GB")

    print("=" * 64)
    print("测试 A：STDP 关联核 —— 优化前后 CPU 时间（process_time，3 次取最优语义"
          "为单次测量，两版本同序各测一次）")
    print("=" * 64)
    print(f"{'规模':<26}{'优化前 ms/步':>14}{'优化后 ms/步':>14}{'降幅':>10}")
    for n, m, active, steps in [(256, 16, 16, 600), (8192, 32, 512, 200),
                                (65536, 32, 512, 100)]:
        t_leg, t_cur = bench_stdp(n, m, steps, active)
        act_pct = active / n
        name = f"n={n}, m={m}（激活 {act_pct:.2%}）"
        print(f"{name:<26}{t_leg / steps * 1000:>14.3f}{t_cur / steps * 1000:>14.3f}"
              f"{100 * (1 - t_cur / t_leg):>9.1f}%")

    print("=" * 64)
    print("测试 B：本机 CPU 有效算力（numpy matmul，float64）")
    print("=" * 64)
    gf = bench_cpu_flops()
    print(f"实测吞吐: {gf:.1f} GFLOPS")
    calc_lb = 10 * 1e9 / (gf * 1e9)
    print(f"推算①（计算下限，10×N FLOPs）: ≥ {calc_lb:.2f} s/步")
    print(f"推算②（带宽下限，16 GB 状态读写 @~25 GB/s）: ≥ ~0.6 s/步")
    print(f"综合估计：密集 1B 单步训练 ≈ 1–5 s/步")
    print(f"对照：1500 步 ≈ {calc_lb * 1500 / 60:.0f}–{5 * 1500 / 60:.0f} 分钟，"
          f"且需 ≥16 GB 训练状态内存")

    print("=" * 64)
    print("测试 C：1B 突触模型实际运行（1500 步在线训练）")
    print("=" * 64)
    wall, cpu, rss, edges = bench_1b()
    print(f"wall 时间 : {wall:.2f} s（{wall / 1500 * 1000:.2f} ms/步）")
    print(f"CPU  时间 : {cpu:.2f} s（单核占用 {cpu / wall:.0%}）")
    print(f"进程峰值 RSS: {rss:.0f} MB")
    print(f"实际训练参数: {edges:,} 条（理论密集 1B 需 ~4.0 GB 权重存储）")
