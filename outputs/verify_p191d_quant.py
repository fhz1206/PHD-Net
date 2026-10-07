# P191d 对拍：_real_to_fp8_bits 设备端位运算核 vs torch CPU fp8 cast（逐位）
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from phdnet.backends.accel_readout import AccelReadout

torch.manual_seed(0); np.random.seed(0)
ro = AccelReadout(64, 64, None, device="cpu", dtype="fp8", conn_k=0,
                  fp8_conv="torch")

def ref_cast(t):
    return t.detach().float().cpu().to(torch.float8_e4m3fn).view(torch.uint8)

# ── 1. 随机大样本（覆盖正规全域）───────────────────────────────────────
for scale in (1e-3, 1e-2, 1.0, 100.0, 400.0):
    t = (torch.randn(1 << 20) * scale)
    got = ro._real_to_fp8_bits(t)
    want = ref_cast(t)
    mism = (got != want).sum().item()
    print(f"scale={scale:>7}: {1<<20} 样本 mismatch={mism}")
    assert mism == 0, f"scale={scale} 不一致"

# ── 2. 边界值（探针实测过的语义点）─────────────────────────────────────
edge = torch.tensor([448.0, 1000.0, 449.0, 447.9, 2.0**-6, 2.0**-9,
                     2.0**-10, 3*2.0**-10, 2.0**-6 + 2.0**-11,
                     0.0, -0.0, 1.0, -1.0, -300.0, float("nan")])
got = ro._real_to_fp8_bits(edge)
want = ref_cast(edge)
mism = (got != want).sum().item()
print("边界值 mismatch =", mism)
assert mism == 0, f"边界值不一致: got={got.tolist()} want={want.tolist()}"
print("P191d 量化核对拍：全部逐位一致")
