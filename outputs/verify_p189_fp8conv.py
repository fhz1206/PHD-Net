# P189 对拍：Rust fp8→int8 转换核 vs Python torch 路径（逐位一致判据）
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# phdnet_rs 是目录+同名绑定模块：必须把目录插到 sys.path 首位，
# 否则被当作 namespace package（`cannot import name 'load'`）
_RS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "phdnet_rs")
sys.path.insert(0, _RS_DIR)
import numpy as np
import torch

from phdnet_rs import load as rs_load
from phdnet.backends.fp8_int8_convert import fp8_to_int8_codes

k, why = rs_load()
assert k is not None and getattr(k, "_has_fp8_conv", False), why

rng = np.random.default_rng(42)
for n in (1, 7, 255, 4096, 1 << 20):
    # 真实 fp8 位模式：随机 fp32 → fp8 cast → uint8 位模式
    vals = (rng.normal(0, 0.05, n) * np.sqrt(n)).astype(np.float32)
    bits = torch.from_numpy(vals).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    # Python 参照（torch 路径）
    ref_codes, ref_sc = fp8_to_int8_codes(bits, conv="torch")
    # Rust 核
    codes = np.empty(n, dtype=np.int8)
    sc = np.zeros(1, dtype=np.float32)
    part = np.zeros(64, dtype=np.float32)
    k.fp8_to_int8(np.ascontiguousarray(bits), codes, sc, part, 8)
    same = np.array_equal(ref_codes, codes)
    # scale：Python 返回 fp64 float，Rust 接口是 f32 —— 差 ≤ 1 个 fp32 ULP
    # （codes 逐位一致是硬判据，scale 用相对容差）
    dsc = abs(float(sc[0]) - ref_sc) / max(abs(ref_sc), 1e-30)
    print(f"n={n:>8}: codes 逐位={'OK' if same else 'FAIL'} "
          f"scale={float(sc[0]):.6f} vs {ref_sc:.6f} (rel|Δ|={dsc:.2e})")
    assert same, f"n={n} codes 不一致"
    assert dsc < 1e-6, f"n={n} scale 不一致"
print("P189 对拍：Rust == Python（逐位）全部通过")
