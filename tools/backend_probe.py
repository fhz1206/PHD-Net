"""硬件后端探测：昇腾 NPU / ROCm(HIP) / CUDA / DirectML / CPU。

用法：
    python backend_probe.py            # 探测 + 本机可跑的等价性自检
    python backend_probe.py --torch    # 强制在 torch 后端上跑一次等价性自检

说明：PHD-Net 默认走 numpy(+numba) CPU 路径；只有在**探测到加速器**或
显式指定（cfg.backend="torch"/"npu"/"rocm"/"cuda"/"cpu"）时才启用 torch 后端。
昇腾与 ROCm 需要本机具备对应硬件与驱动，本脚本负责把环境事实说清楚。
"""

# --- 目录结构调整（2026-09-18）：脚本位于 tests/ 或 tools/ 子目录 ---
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]     # 项目根目录
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))              # 保证 `import phdnet` 可用
_DOC = _ROOT / "eval_corpus" / "internal_corpus.txt"    # 默认语料
# --- 引导结束 ---

import sys

from phdnet.device import GUIDANCE, probe, select_backend, use_torch_backend

if __name__ == "__main__":
    print("=" * 70)
    print("PHD-Net 硬件后端探测（昇腾 NPU / ROCm / CUDA / DirectML / CPU）")
    print("=" * 70)
    info = probe()
    print(f"后端名称      : {info.name}")
    print(f"后端种类      : {info.kind}")
    print(f"torch 设备    : {info.device}")
    print(f"torch 版本    : {info.torch_version}")
    print(f"ROCm/HIP      : {info.hip}")
    print(f"torch_npu     : {info.npu_plugin}")
    print(f"numba(CPU加速): {info.numba}")
    print(f"等价性自检    : {'✓ 通过' if info.verified else '— 未运行/不适用'}")
    print(f"说明          : {info.notes}")
    print(f"默认是否启用 torch 后端: {'是' if use_torch_backend('auto') else '否（numpy 路径，零回归）'}")

    print("\n--- 各生态安装指引 ---")
    for k in ("npu", "rocm", "cuda", "dml", "cpu"):
        print(f"  [{k}] {GUIDANCE[k]}")

    if "--torch" in sys.argv:
        print("\n--- 强制 torch 后端等价性自检 ---")
        from phdnet.backends.accel_readout import AccelReadout  # P30
        dev = select_backend("torch").device
        ok = selftest_torch(device=dev)
        print(f"  device={dev}: {'✓ torch 后端与 numpy 参考一致' if ok else '✗ 不一致/不可用'}")

    print("\n提示：本机无加速器时（如当前 Windows CPU 环境），昇腾/ROCm 路径"
          "仅完成代码适配与接口探测，需在有硬件的环境运行 backend_probe.py 完成实测验证。")
