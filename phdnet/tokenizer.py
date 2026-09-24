"""P1 语言接口层 —— 字符级 tokenizer → SDR 确定性映射。

设计对应架构文档 M1：每个字符的 SDR 由确定性哈希生成（同字符恒同码、
异字符近似正交），保持稀疏分布式表示的区分性前提。
"""

import numpy as np

U64 = np.uint64
_K1, _K2, _K3 = U64(0x9E3779B97F4A7C15), U64(0xBF58476D1CE4E5B9), U64(0x94D049BB133111EB)


def mix64(z: np.ndarray) -> np.ndarray:
    """splitmix64 终混（确定性、可向量化）—— SDR / 大空间索引的伪随机来源。

    公开名；`_mix64` 为兼容旧引用的别名（历史上以私有名导出，多处 import）。
    """
    z = (z + _K1).astype(np.uint64)
    z = ((z ^ (z >> U64(30))) * _K2).astype(np.uint64)
    z = ((z ^ (z >> U64(27))) * _K3).astype(np.uint64)
    return z ^ (z >> U64(31))


_mix64 = mix64          # 兼容别名（C5 重构 2026-09-22：去重 tools/train_1b.py 的副本）


class CharTokenizer:
    """字符级分词器：字符表构建 + 字符 → 稀疏分布式表示（SDR）。"""

    def __init__(self, text: str, n_sdr: int = 256, n_active: int = 32,
                 seed: int = 0):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.n_sdr = n_sdr
        self.n_active = n_active
        self.seed = seed
        # 预生成全部字符的 SDR（字符表规模有限，一次性构建）
        self._sdrs = {}
        for i, c in enumerate(self.chars):
            x = U64(i) * U64(2654435761) + np.arange(n_active, dtype=np.uint64)
            idx = (_mix64(x + U64(seed)) % U64(n_sdr)).astype(np.int64)
            s = np.zeros(n_sdr)
            s[idx] = 1.0                      # 二值 SDR（事件驱动语义）
            self._sdrs[c] = s

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, ch: str) -> np.ndarray:
        return self._sdrs[ch]

    def onehot(self, idx: int) -> np.ndarray:
        t = np.zeros(len(self.chars))
        t[idx] = 1.0
        return t
