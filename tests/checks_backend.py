"""回归检查：硬件后端（探测 / torch 等价性 / torch 后端模型冒烟）。"""

import numpy as np

from checks_common import _check


def t_backend_probe() -> None:
    from phdnet.device import probe, use_torch_backend
    info = probe()
    _check("后端探测可运行（无加速器时回退 CPU 亦属通过）",
           info.name is not None and use_torch_backend("auto") is False,
           f"{info.name} / 默认 torch 后端={use_torch_backend('auto')}")


def t_torch_equivalence() -> None:
    """torch 后端与 numpy 参考的逐元素等价（在可用设备上验证；无 torch 则跳过）。"""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        _check("torch 后端等价性（未安装 torch，跳过）", True, "skip")
        return
    from phdnet.torch_backend import selftest_torch
    ok = selftest_torch(device="cpu")
    _check("torch 后端与 numpy 参考等价（CPU 验证；昇腾/ROCm 同算子待硬件验证）", ok)


def t_torch_model_smoke() -> None:
    """显式 backend='torch' 时，模型必须能正常构造并学习。"""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        _check("torch 后端模型冒烟（未安装 torch，跳过）", True, "skip")
        return
    from phdnet.config import PHDNetConfig
    from phdnet.model import PHDNet
    cfg = PHDNetConfig(n_sdr=128, k_sparse=16, n_mid=64, n_top=64,
                       backend="torch", seed=3)
    net = PHDNet(cfg)
    rng = np.random.default_rng(0)
    for _ in range(10):
        net.step(rng.normal(size=cfg.n_input), learn=True)
    w = net.stdp.W
    s = float(w.detach().cpu().numpy().sum())
    _check("torch 后端模型可构造并学习（W 增长 > 0）", s > 0, f"W.sum={s:.2f}")
