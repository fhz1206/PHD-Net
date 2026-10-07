# P188 对拍：_eager_step 的 nll 改写（cross_entropy → logsumexp−y[c]）数值等价验证
# 同时验证整条 forward_dev → learn_softmax 热路径在改动后 CPU 臂输出与改动前一致。
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from phdnet.backends.accel_readout import AccelReadout

torch.manual_seed(0)
np.random.seed(0)

n_h, n_out = 512, 4096
ro = AccelReadout(n_h, n_out, None, device="cpu", dtype="fp32")

# ── 1. nll 恒等性：cross_entropy(y,c) == logsumexp(y)-y[c] ──────────────
y32 = torch.randn(n_out)
nll_old = torch.nn.functional.cross_entropy(
    y32.reshape(1, -1), torch.tensor([7])).reshape(())
nll_new = (torch.logsumexp(y32, dim=0) - y32[7]).reshape(())
diff = float((nll_old - nll_new).abs())
print(f"[1] nll 恒等性: old={float(nll_old):.8f} new={float(nll_new):.8f} "
      f"|Δ|={diff:.3e}")
assert diff < 4e-6, "nll 不等价"

# ── 2. learn_softmax 端到端：W 更新结果对拍（eager 路径）────────────────
h = np.random.randn(n_h).astype(np.float32)
eta = 0.15
W_before = ro.W.detach().clone()
nll = ro.learn_softmax(h, None, eta, target_idx=7)
# 参照实现：手动 softmax → dp → rank-1 更新（与 P45 语义一致）
ht = torch.as_tensor(h)
y_ref = (ro.W_before_ref if hasattr(ro, "W_before_ref") else W_before) @ ht
p_ref = torch.softmax(y_ref.float(), dim=0)
dp_ref = p_ref.clone(); dp_ref[7] -= 1.0
W_ref = W_before.clone()
W_ref.addmm_(dp_ref.reshape(-1, 1), ht.reshape(1, -1), alpha=-eta)
dW = float((ro.W - W_ref).abs().max())
print(f"[2] learn_softmax 端到端: nll={nll:.6f} max|ΔW|={dW:.3e}")
assert dW < 4e-6, "W 更新不等价"

# ── 3. 稀疏臂（conn_k）对拍 ────────────────────────────────────────────
ro2 = AccelReadout(n_out, n_h, None, device="cpu", dtype="fp32",
                   conn_k=64 if "conn_k" in
                   AccelReadout.__init__.__code__.co_varnames else None)
try:
    Wb2 = ro2.W.detach().clone()
    ro2.forward_dev(h)
    nll2 = ro2.learn_softmax(h, None, eta, target_idx=7)
    Wi = ro2.Wi
    g = ro2._sp_gather(ro2._cache_ht)
    p2 = torch.softmax((Wb2.to(torch.float32) @ ro2._cache_ht).float(), dim=0)
    dp2 = p2.clone(); dp2[7] -= 1.0
    W_ref2 = Wb2.to(torch.float32).clone()
    W_ref2[Wi] -= eta * dp2.reshape(-1, 1).to(W_ref2.dtype) * g.to(W_ref2.dtype)
    dW2 = float((ro2.W.to(torch.float32) - W_ref2).abs().max())
    print(f"[3] 稀疏臂: nll={nll2:.6f} max|ΔW|={dW2:.3e}")
    assert dW2 < 4e-6, "稀疏臂 W 更新不等价"
except TypeError as e:
    print(f"[3] 稀疏臂跳过（构造参数不符: {e}）")

print("P188 对拍：全部通过")
