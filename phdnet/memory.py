"""M4 双记忆系统 —— 结构拆分后的兼容再导出（2026-09-18）。

实现已按架构模块拆分：
    M4a 工作记忆   -> phdnet/wm.py
    M4b 长期记忆   -> phdnet/ltm.py
"""

from .ltm import LongTermMemory
from .wm import WorkingMemory

__all__ = ["WorkingMemory", "LongTermMemory"]

