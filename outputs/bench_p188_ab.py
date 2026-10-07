# P188 A/B 基准：_eager_step nll 计算新旧实现单步开销对比（CPU 臂）
# 新：logsumexp(y)-y[c]（零主机交互）
# 旧：pinned fill_ + H2D + cross_entropy（2 kernel + 1 copy）
# 纪律：同进程交错测 + best-of-3；计时脚本落盘 .py
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

torch.manual_seed(0); np.random.seed(0)

n_h, n_out = 3072, 73958          # 1b 档生产规模
y32 = torch.randn(n_out)
correct = 12345

def old_nll():
    cp = torch.zeros(1, dtype=torch.long, pin_memory=False)
    cp.fill_(correct)
    _ct = cp.to(y32.device, non_blocking=True)
    return torch.nn.functional.cross_entropy(y32.reshape(1, -1), _ct).reshape(())

def new_nll():
    return (torch.logsumexp(y32, dim=0) - y32[int(correct)]).reshape(())

def bench(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e3  # ms

R = 2000
best_old = min(bench(old_nll, R) for _ in range(3))
best_new = min(bench(new_nll, R) for _ in range(3))
print(f"n_out={n_out}  reps={R}  best-of-3")
print(f"旧 cross_entropy 路径: {best_old:.4f} ms/次")
print(f"新 logsumexp 路径:     {best_new:.4f} ms/次")
print(f"加速: {best_old / best_new:.2f}x  (CPU eager 部分，NPU 上收益主要来自"
      f"省 2 次 kernel launch + 1 次 H2D)")
