# P189 A/B 基准：fp8→int8 转换核 rust vs torch vs nogil（1b 档规模）
# 规模 = 51962×128 = 6,651,136 元素（6.34 MiB fp8 位模式）
# 纪律：同进程交错测 + best-of-3；计时脚本落盘 .py
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "phdnet_rs")
sys.path.insert(0, _RS_DIR)
import numpy as np
import torch

from phdnet_rs import load as rs_load
from phdnet.backends.fp8_int8_convert import fp8_to_int8_codes

N = 51962 * 128
rng = np.random.default_rng(0)
# 真实 fp8 位模式分布（±448 对数网格）
vals = (rng.normal(0, 0.05, N) * 100).astype(np.float32)
bits = np.ascontiguousarray(
    torch.from_numpy(vals).to(torch.float8_e4m3fn).view(torch.uint8).numpy())

k, why = rs_load()
assert k is not None and getattr(k, "_has_fp8_conv", False), why

codes_r = np.empty(N, dtype=np.int8)
sc_r = np.zeros(1, dtype=np.float32)
part = np.zeros(64, dtype=np.float32)

def bench_rust():
    k.fp8_to_int8(bits, codes_r, sc_r, part, 8)

def bench_torch():
    fp8_to_int8_codes(bits, conv="torch")

def bench_nogil():
    fp8_to_int8_codes(bits, conv="nogil")

def best3(fn, reps=10):
    out = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        out.append((time.perf_counter() - t0) / reps * 1e3)
    return min(out)

# 预热（numba JIT / 首次分配）
bench_torch(); bench_nogil(); bench_rust()

t_rust = best3(bench_rust)
t_torch = best3(bench_torch)
t_nogil = best3(bench_nogil, reps=3)

# 正确性复核（基准内再对拍一次）
ref, _ = fp8_to_int8_codes(bits, conv="torch")
assert np.array_equal(ref, codes_r), "rust 与 torch 不一致"

print(f"规模: {N:,} 元素 ({N/1e6:.2f} MiB fp8)  best-of-3")
print(f"  rust  (8 线程): {t_rust:8.2f} ms   {N/1e6/t_rust*1e3/1000:.2f} GB/s")
print(f"  torch (向量化): {t_torch:8.2f} ms   {N/1e6/t_torch*1e3/1000:.2f} GB/s")
print(f"  nogil (numba) : {t_nogil:8.2f} ms")
print(f"rust/torch = {t_torch/t_rust:.2f}x  rust/nogil = {t_nogil/t_rust:.2f}x")
