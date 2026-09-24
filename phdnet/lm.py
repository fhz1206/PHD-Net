"""P1 语言接口层 —— 字符级语言模型（下一字符预测）。

信息流（架构文档 §5）：字符 → SDR → M1/M2/M3 全链路 → M6 读出（softmax 感知器）。
学习全在线：训练即推理，无 epoch、无反向传播。
字符级 n-gram 基线已拆分至 phdnet/ngram.py，此处再导出以兼容旧引用。"""

import numpy as np

from .config import PHDNetConfig
from .model import PHDNet
from .ngram import NGramBaseline
from .tokenizer import CharTokenizer

__all__ = ["PHDNetLM", "NGramBaseline"]



class PHDNetLM:
    """PHD-Net 字符级语言模型。

    字符表由 vocab_text（全量文本）构建，训练只使用 train_text——
    保证评估段未见字符也有合法 SDR 与读出维度。
    """

    def __init__(self, vocab_text: str, cfg: PHDNetConfig | None = None,
                 lr: float = 0.05):
        self.tok = CharTokenizer(vocab_text, n_sdr=cfg.n_sdr,
                                 n_active=cfg.k_sparse, seed=cfg.seed)
        cfg = PHDNetConfig(**{**cfg.__dict__,
                              "n_input": cfg.n_sdr,          # SDR 直接作为输入
                              "n_readout": len(self.tok),
                              "readout_softmax": True})
        self.net = PHDNet(cfg)
        self.lr = lr

    def train_stream(self, text: str, log_every: int = 2000) -> list[float]:
        """在线训练：逐字符喂入，预测下一字符。返回逐段 NLL。"""
        nlls = []
        seg = []
        for t in range(len(text) - 1):
            x = self.tok.encode(text[t])
            target = self.tok.onehot(self.tok.stoi[text[t + 1]])
            d = self.net.step(x, target=target, learn=True)
            seg.append(d["nll"])
            if len(seg) >= log_every:
                nlls.append(float(np.mean(seg)))
                seg = []
        if seg:
            nlls.append(float(np.mean(seg)))
        return nlls

    def evaluate(self, text: str) -> float:
        """字符级困惑度（PPL = exp(平均 NLL)）。只推理、不学习。

        2026-09-18 修复：NLL 直接取自 step() 返回的读出输出 y —— 与训练同款特征、
        同款状态；旧实现每步都走 `_nll` 回退（在状态推进后重算一遍前向），
        属重复计算（评估耗时翻倍），且 `d["nll"] > 0` 是脆弱判据。
        数值等价：回退路径与主路径在相同状态上算出同一 h，故 PPL 不变。
        """
        total = 0.0
        n = max(1, len(text) - 1)
        for t in range(len(text) - 1):
            y = self.net.step(self.tok.encode(text[t]), learn=False)["y"]
            y = y - y.max()
            p = np.exp(y)
            p /= p.sum()
            total += float(-np.log(p[self.tok.stoi[text[t + 1]]] + 1e-12))
        return float(np.exp(total / n))

    def _nll(self, x: np.ndarray, target: np.ndarray) -> float:
        """（保留兼容，主评估路径已不再使用；见 evaluate 的说明。）"""
        s0, _ = self.net.encoder.encode(x)
        cache = self.net.pc.infer(s0, self.net.cfg.n_infer_steps)
        h_parts = [cache["r2"], self.net.wm.read()]
        if self.net.cfg.pred_in_readout:              # T3.1：与训练路径的三拼特征保持一致
            h_parts.append(self.net.stdp.predict(self.net._last_rate))
        h = np.concatenate(h_parts)
        y = self.net.readout(h)
        y -= y.max()
        p = np.exp(y)
        p /= p.sum()
        return float(-np.log(p[int(np.argmax(target))] + 1e-12))
