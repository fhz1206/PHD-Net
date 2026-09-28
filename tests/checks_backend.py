"""回归检查：硬件后端（探测 / torch 等价性 / torch 后端模型冒烟）。

关于 torch 缺失：CI 的 tests job 已安装 CPU 版 torch，两个 torch 检查必须真正执行。
万一运行环境缺 torch，本文件不会「静默记通过」——检查项名会带 `【SKIP-未覆盖】`
前缀并额外打印一条醒目告警，说明该覆盖点本次未被验证。
"""

import numpy as np

from checks_common import _check

_BANNER = "  " + "!" * 74


def _no_torch(check_name: str) -> None:
    """torch 缺失：记为通过（不算失败）但必须显眼，不得伪装成已覆盖。"""
    _check(f"{check_name}（未安装 torch，未覆盖）", True, "SKIP")
    print(_BANNER, flush=True)
    print(f"  !! SKIP（未覆盖）: {check_name} —— torch 不可用，本项未被真正验证", flush=True)
    print(_BANNER, flush=True)


def t_backend_probe() -> None:
    from phdnet.device import probe, use_torch_backend
    info = probe()
    # 只断言探针本身能跑通（info.name 非空）。「默认是否走 torch 后端」取决于本机
    # 是否存在 npu/rocm/cuda/dml 加速器（见 phdnet/device.py），是环境事实而非不变式，
    # 因此只作信息输出——否则在 GPU runner 上必然误红。
    _check("后端探测可运行", info.name is not None, f"kind={info.kind} name={info.name}")
    print(f"    · 探测结果: kind={info.kind}, name={info.name}, "
          f"默认 torch 后端={use_torch_backend('auto')}", flush=True)


def t_torch_equivalence() -> None:
    """torch 后端与 numpy 参考的逐元素等价（在可用设备上验证；无 torch 则跳过）。"""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        _no_torch("torch 后端与 numpy 参考等价")
        return
    from phdnet.torch_backend import selftest_torch
    ok = selftest_torch(device="cpu")
    _check("torch 后端与 numpy 参考等价（CPU 验证；昇腾/ROCm 同算子待硬件验证）", ok)


def t_prod_stdp_equivalence() -> None:
    """生产 STDP 类（`torch_lm._ProdSTDPCore`）与 numpy 端**主路径**等价。

    背景：LM 全栈（`torch_lm.TorchPHDNet`）用的是 `_ProdSTDPCore`（formA：
    `2·(η·s·tp)·post − tp_hist·pre`，对齐 numpy numba 热路径），而
    `selftest_torch` 历史上只测 `TorchSTDPCore`（formB：`η·s·(2·tp·post −
    tp_hist·pre)`）—— 生产类在 tests/ 中零覆盖。实测两式在真实 η=0.03 下
    maxdiff 可达 1.0。

    判据要点（否则该断言形同虚设）：
      - 必须用真实学习率 η=0.03：η=1 时 formA 与 formB 数值**重合**，
        任何参考式都判不出差异；
      - 必须用 `stimulus="walk"`：常量不重叠刺激在 η<1 时 LTP 被 η 压制，
        参考权重全零退化为平凡解；
      - 必须用 `reference="numba"`：`_numpy_reference` 是 formB 口径。
    """
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        _no_torch("生产 STDP 类等价 numpy numba 主路径")
        return
    from phdnet.backends.torch_backend import TorchSTDPCore, selftest_torch
    from phdnet.backends.torch_lm import _ProdSTDPCore

    kw = dict(device="cpu", reference="numba", eta=0.03, stimulus="walk")
    ok_prod = selftest_torch(cls="_ProdSTDPCore", **kw)
    _check("生产 STDP 类 _ProdSTDPCore 等价 numpy numba 主路径"
           "（η=0.03 / walk 刺激）", ok_prod,
           "惰性字符串引用 torch_lm._ProdSTDPCore")

    # 门禁须有证伪能力：formB 类在 formA 参考下必须失败，否则断言无意义
    ok_neg = not selftest_torch(cls=TorchSTDPCore, **kw)
    _check("自检门禁有证伪能力（formB 类在 formA 参考下应失败）", ok_neg,
           "若此项失败说明参考式/刺激无判别力，门禁形同虚设")

    # 生产类在 fp16/bf16 下同样须过（低精度走强相关判据）
    for dt in ("float16", "bfloat16"):
        _check(f"生产 STDP 类低精度门禁 {dt}", selftest_torch(
            cls="_ProdSTDPCore", device="cpu", reference="numba", eta=0.03,
            stimulus="walk", dtype=dt))
    _check("生产 STDP 类可由类对象直接引用（惰性 import 兜底）",
           selftest_torch(cls=_ProdSTDPCore, **kw))


def t_torch_model_smoke() -> None:
    """显式 backend='torch' 时，模型必须能正常构造并学习。"""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        _no_torch("torch 后端模型冒烟")
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
    s = float(w.detach().float().cpu().numpy().sum())
    _check("torch 后端模型可构造并学习（W 增长 > 0）", s > 0, f"W.sum={s:.2f}")


def t_torch_lm_switches() -> None:
    """torch LM 栈：未实现开关必须**显式拒绝**，且库默认配置仍可构造。

    背景：`TorchPHDNet` 接受 8 个开关但 `step()` 从不实现，构造成功、训练
    照跑，但与 numpy 端逐张量不同（静默 no-op，违反模块 docstring
    「显式拒绝，不做静默近似」的承诺）。

    `sparse_conn` 特殊：库默认 True，torch 栈恒用稀疏 CSR 语义（无稠密
    路径）→ True 放行、False 拒绝。故默认配置必须仍能构造。
    """
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        _no_torch("torch LM 栈未实现开关显式拒绝")
        return
    from phdnet.backends.torch_lm import TorchPHDNet
    from phdnet.config import PHDNetConfig

    SMALL = dict(n_sdr=64, k_sparse=8, n_mid=32, n_top=32, n_input=128,
                 n_readout=32, readout_softmax=True, seed=7)
    REJECTED = ("dual_trace", "learnable_encoder", "critical_period",
                "task_modulation", "auto_development", "pc_predictive_target",
                "neuron_target_rate")
    bad = []
    for name in REJECTED:
        try:
            TorchPHDNet(PHDNetConfig(**{**SMALL, name: True}), device="cpu")
            bad.append(f"{name}=True 未被拒绝")
        except NotImplementedError:
            pass
    _check("torch LM 栈 7 个未实现开关开启时显式 NotImplementedError",
           not bad, "; ".join(bad) or f"{len(REJECTED)} 项全部拒绝")

    # sparse_conn：True（库默认语义）放行；False（稠密主干已删除）config 期拒绝
    try:
        TorchPHDNet(PHDNetConfig(**{**SMALL, "sparse_conn": True}), device="cpu")
        ok_true = True
    except Exception as e:                                        # noqa: BLE001
        ok_true = False
        print(f"    · sparse_conn=True 意外失败: {e}", flush=True)
    try:
        PHDNetConfig(**{**SMALL, "sparse_conn": False})
        ok_false = False
    except ValueError:
        ok_false = True
    _check("sparse_conn=True 放行 / False（稠密已删，config 期拒绝）", ok_true and ok_false)

    # 库默认 PHDNetConfig（含 sparse_conn=True）必须仍可构造
    try:
        TorchPHDNet(PHDNetConfig(), device="cpu")
        ok_def = True
    except Exception as e:                                        # noqa: BLE001
        ok_def = False
        print(f"    · 库默认配置构造失败: {e}", flush=True)
    _check("库默认 PHDNetConfig 下 TorchPHDNet 仍可构造（默认路径不破坏）", ok_def)
