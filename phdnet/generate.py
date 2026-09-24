"""PHD-Net v2 文本生成器 —— M11 生成解码器。

机制（v2 文档 §2 M11）：
    主题锚定：prompt 先呈现（学习关闭），WM 槽位累积主题上下文并持续参与读出
    —— 条件化机制是 WM 锚定而非注意力。
    自回归采样：读出分布经温度 τ（神经噪声/随机变异性的抽象）与 top-k
    截断后采样下一字符，回填输入。
"""

import numpy as np


class Generator:
    def __init__(self, lm, tau: float = 0.8, topk: int = 8,
                 rng: np.random.Generator | None = None):
        self.lm = lm
        self.tau = tau
        self.topk = topk
        self.rng = rng or np.random.default_rng(0)

    def generate(self, prompt: str, n_chars: int = 120) -> str:
        for ch in prompt:                        # 锚定阶段：WM 累积主题上下文
            self.lm.net.step(self.lm.tok.encode(ch), learn=False)
        out = []
        last = prompt[-1] if prompt else self.lm.tok.chars[0]
        for _ in range(n_chars):
            d = self.lm.net.step(self.lm.tok.encode(last), learn=False)
            y = d["y"].astype(np.float64)
            y -= y.max()
            p = np.exp(y / self.tau)
            p /= p.sum()
            idx = np.argpartition(-p, self.topk - 1)[:self.topk]
            pp = p[idx] / p[idx].sum()
            j = int(self.rng.choice(idx, p=pp))
            out.append(self.lm.tok.chars[j])
            last = self.lm.tok.chars[j]
        return "".join(out)
