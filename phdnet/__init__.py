"""PHD-Net：预测-赫布-双记忆网络（不依赖自注意力的类脑架构）。

官方版本号：v0.0.0（架构文档与评估文档统一使用此口径；
"M1–M6 基础架构 / M7–M12 认知层"为模块分层表述，不再使用 v1/v2 版本称呼）。

模块与架构文档（PHD-Net_架构设计.md）的对应关系：
    M1 稀疏编码器   -> sparse_encoder.py
    M2 预测编码层级 -> pc.py
    M3 时序关联核   -> plasticity.py
    M4 双记忆系统   -> memory.py
    M5 神经调制器   -> modulator.py
    M6 组装与读出   -> model.py
    M7–M12 认知层   -> cognition.py / generate.py
"""

__version__ = "0.0.0"

from .config import PHDNetConfig
from .model import PHDNet

__all__ = ["PHDNetConfig", "PHDNet", "__version__"]
