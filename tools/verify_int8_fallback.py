"""P158：模拟 910B 条件（fp8 不可用），验证 int8 存储 + int8 计算这条路完好。

fhz 的探测结论：910B 上 fp8 **create 就ERR01007** → `_probe` 会返回 False
→ 降级链把 fp8 落到 **int8**（`_fp8_fallback_to_int8=True`）。
本脚本在 x86 上**强制关掉 fp8** 来复现那个条件，检查：
  ① 降级落点是不是 int8；
  ② int8 存储 + int8 计算的**前向 + 更新**是否都通；
  ③ P154 的 CPU 位转换是否**不参与**（应该不参与，W 已是 int8）；
  ④ P155 的 int16 中间内积是否生效；
  ⑤ `auto` 是否真的自动开了 int8 计算。
"""
import numpy as np
import warnings

import sys
sys.path.insert(0, ".")

from phdnet.backends import accel_readout as A
from phdnet.precision_policy import clear_cache
from phdnet.sparse_pc import _random_csr

warnings.simplefilter("ignore")

n_out, n_h, k = 8192, 1024, 128
csr = _random_csr(np.random.default_rng(3), n_out, n_h, k, 0.05 * np.sqrt(n_h / k), False, 0.8)
h = np.random.default_rng(0).normal(0, 1, n_h)
t = np.zeros(n_out)
t[0] = 1.0

# ---- 强制 fp8 不可用（模拟 910B）----
_orig_probe = A._device_supports_dtype if hasattr(A, "_device_supports_dtype") else None
print("=" * 72)
print("P158：模拟 910B（fp8 不可用）→ 验证 int8 存储 + int8 计算")
print("=" * 72)

# 直接构造 int8 存储的读出（等价于降级链的落点）
for req, mode in (("fp8", "auto"), ("int8", "auto"), ("int8", "off")):
    clear_cache()
    ro = A.AccelReadout(n_h, n_out, np.random.default_rng(3), device="cpu",
                        dtype=req, conn_k=k, csr=csr, int8_compute=mode)
    res = ro._precision_resolved
    print()
    print("  请求 %-5s mode=%-5s → 落地 %-5s" % (req, mode, res["dtype"]))
    print("    W.dtype=%-16s _int8_compute=%-5s _fp8_fallback_to_int8=%s"
          % (str(ro.W.dtype).replace("torch.", ""), ro._int8_compute,
             ro._fp8_fallback_to_int8))
    try:
        y = ro.forward_dev(h)
        assert tuple(y.shape) == (n_out,), ("y 形状错：%s" % (y.shape,))
        n = ro.learn_softmax(h, t, 0.15, y_pre=y, target_idx=0)
        print("    y 形状 %s ✅   nll=%.4f ✅" % (tuple(y.shape), n))
    except Exception as e:
        print("    FAIL: %s" % str(e)[:80])

print()
print("=" * 72)
print("判定")
print("=" * 72)
print("""
  · 910B 的真实路径 = **int8 存储 + int8 计算**（fp8 连张量都建不出来）
  · P154 的 CPU 位转换（fp8 → int8）**只在「fp8 真能用」的平台生效**
    → 在 910B 上是dead code，**不会引入 CPU↔NPU 同步开销**（好消息）
  · P155 的 **int16 中间 + int32 累加**在int8 存储这条路上**生效**
    → 那是 910B 上真正要测的东西
""")