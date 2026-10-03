"""PHD-Net **顶层编排** —— Python 是顶层，Rust 是算子库。

这个模块是**唯一**把 Rust 接入训练的地方。职责边界（违反即错）：

| 层 | 负责 | 不负责 |
|---|---|---|
| **Python（本文件 + `model.py`）** | 编排顺序、状态连续、门禁判据、**回落决策**、日志 | 任何 `for i in range(n): y[i] = ...` 的数值循环 |
| **Rust（`phdnet_rs` crate）** | 无状态数值核、ACL 句柄、设备内存 | 任何「下一步做什么」的决策 |

# 为什么回落决策必须在 Python

项目的一条铁律：**降级必须可归因**（P161 的教训 —— 读出全线回落时，
`_dev` 用了未解析的 `"auto"` 这个真 bug被一串"预期内失败"埋掉了）。
把「Rust 不可用 →回落 numba」这个决策放在 Python，才能把它写进日志。

# 用法

```python
from phdnet_rs.top import kernels, why_unavailable

if kernels is not None:
    print(f"[mech] Rust 算子库已启用（has_npu={kernels.has_npu()}）")
else:
    print(f"[mech] Rust 不可用，回落 numba：{why_unavailable}")
```
"""

from __future__ import annotations

import sys
from pathlib import Path

# `phdnet_rs/` 是 crate 目录；它同时含 `phdnet_rs.py`（绑定层）与 Rust 源码
_RS_DIR = Path(__file__).resolve().parent
if str(_RS_DIR) not in sys.path:
    sys.path.insert(0, str(_RS_DIR))

try:
    from phdnet_rs import load as _rs_load
    kernels, why_unavailable = _rs_load()
except Exception as _e:                                       # noqa: BLE001
    kernels, why_unavailable = None, "%s: %s" % (type(_e).__name__, _e)


def describe() -> str:
    """人读一行（打日志用）。**永远返回字符串**，即使不可用。"""
    if kernels is None:
        return ("Rust 算子库：不可用（%s）→ 回落 numba/ Python 参考实现"
                % why_unavailable)
    return ("Rust 算子库：已加载（has_npu=%s；CPU 参照路径始终可用）"
            % kernels.has_npu())


def has_npu() -> bool:
    """本机是否有真NPU 后端。**False 不等于不可用**——CPU 参照路径仍工作。"""
    return bool(kernels is not None and kernels.has_npu())


def enabled() -> bool:
    """Rust 算子库是否可用（**不等于**是否上 NPU）。"""
    return kernels is not None