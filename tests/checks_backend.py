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



def main() -> None:
    """P30：torch 栈已删除，本文件只剩设备探测检查（其余用例随栈移除）。"""
    print("[checks_backend] 设备探测")
    t_backend_probe()
    print("结果：全部 PASS")


if __name__ == "__main__":
    main()
