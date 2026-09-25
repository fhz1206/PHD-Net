"""M2 词级语言接口 —— 词级 LM（T2.1 词涌现 + T2.3 上下文绑定）。

词法（WordSegmenter / WordTokenizer）已拆分至 phdnet/word_encoder.py，
n-gram 基线拆分至 phdnet/ngram.py；此处仅保留 PHDWordLM 并再导出以兼容旧引用。"""

from collections import Counter
from math import log

import numpy as np

from .config import PHDNetConfig
from .model import PHDNet
from .ngram import WordNGram
from .word_encoder import WordSegmenter, WordTokenizer

__all__ = ["PHDWordLM", "WordSegmenter", "WordTokenizer", "WordNGram"]



class PHDWordLM:
    """PHD-Net 词级语言模型（T2.1 + T2.3；M1 三开关经 cfg 透传）。"""

    def __init__(self, vocab_text: str, cfg: PHDNetConfig | None = None,
                 seg_kwargs: dict | None = None,
                 tokenizer: "WordTokenizer | None" = None):
        """`tokenizer` 可选注入（2026-09-25，train_1b 流式词表扫描用）：
        传入时跳过由 vocab_text 构建 WordTokenizer（调用方保证其 seg/tokens/sdrs
        已就绪且 n_active/seed 与 cfg 一致），n_readout 按注入词表大小取。
        默认 None = 旧行为逐位不变。"""
        cfg = cfg or PHDNetConfig()
        if tokenizer is not None:
            self.tok = tokenizer
        else:
            self.tok = WordTokenizer(vocab_text, n_sdr=cfg.n_sdr,
                                     n_active=cfg.k_sparse, seed=cfg.seed,
                                     seg_kwargs=seg_kwargs)
        self.n_sdr = cfg.n_sdr
        cfg = PHDNetConfig(**{**cfg.__dict__,
                              "n_input": 2 * cfg.n_sdr,     # T2.3 组合输入
                              "n_readout": len(self.tok),
                              "readout_softmax": True})
        self.net = PHDNet(cfg)

    def tokenize(self, text: str) -> list[str]:
        return self.tok.seg.tokenize(text)

    def _recur_vector(self, y: np.ndarray) -> np.ndarray:
        """O5 再入线索：读出分布 → 期望 token SDR（top-8 加权组合）。

        脑对应：前额叶对"预期下一项"的预激活表征（predictive pre-activation），
        经反馈再入送回感觉/关联皮层，形成自预测条件化回路。
        返回 n_sdr 维单位范数向量（与 WM 槽同维度）；不做位置编码/顺序无关化。
        """
        yy = y - y.max()
        p = np.exp(yy)
        p /= p.sum()
        k = min(8, len(p))
        top = np.argpartition(-p, k - 1)[:k]
        out = np.zeros(self.n_sdr)
        for t in top:
            out += float(p[int(t)]) * self.tok.encode(self.tok.tokens[int(t)])
        nrm = float(np.linalg.norm(out))
        return out / nrm if nrm > 1e-12 else out

    def _pass(self, toks: list[str], keep: list[bool], log_every: int,
              sleep_every: int, weights: list[float] | None = None,
              valid_text: str | None = None,
              plateau_patience: int = 0) -> list[float]:
        """一趟语料遍历（keep 决定哪些位置参与学习；weights 为逐位置读出学习率缩放）。

        T4.4 验证驱动睡眠（valid_text 给定且 plateau_patience>0）：每个日志段末
        只读评估验证 PPL，连续 plateau_patience 段无改善 → 触发一次 sleep()
        巩固（睡眠时机由验证曲线科学化，替代定频）。

        O5（2026-09-22，默认关闭）：
          - `readout_recurrence=True`：每步把读出分布的期望 token SDR 作为
            下一步再入线索（predictive pre-activation 回灌 WM）；
          - `segment_check=N>0`：每 N 步做段内一致性校验，窗口平均 NLL 超过
            习得水平 1.5× 时判定"低置信段"，对读出学习率降权（级联误差治理）。
        两项默认关闭时不改变任何数值路径（seg_scale 恒 1.0、recur_cue 恒 None）。
        """
        cfg = self.net.cfg
        use_recur = bool(cfg.readout_recurrence)
        seg_check = int(cfg.segment_check)
        recur_cue: np.ndarray | None = None
        seg_scale = 1.0
        nll_ema: float | None = None
        nlls, seg = [], []
        prev: str | None = None
        best_ppl, bad = float("inf"), 0
        for i in range(len(toks) - 1):
            if not keep[i]:                       # 跳过位置仍推进上下文（prev）
                prev = toks[i]
                continue
            if toks[i + 1] not in self.tok.stoi:  # OOV 目标：跳过学习，上下文照常推进
                prev = toks[i]
                continue
            x = self.tok.encode_composite(toks[i], prev)
            target = self.tok.onehot(self.tok.stoi[toks[i + 1]])
            ls = (1.0 if weights is None else weights[i]) * seg_scale
            d = self.net.step(x, target=target, learn=True, learn_scale=ls,
                              recur_cue=recur_cue)
            if use_recur:
                recur_cue = self._recur_vector(d["y"])
            seg.append(d["nll"])
            # O5 分段校验：窗口平均 NLL 显著高于习得水平 → 低置信段降权
            if seg_check > 0 and len(seg) >= seg_check:
                w_mean = float(np.mean(seg[-seg_check:]))
                if nll_ema is None:
                    nll_ema = w_mean
                if w_mean > 1.5 * nll_ema:
                    seg_scale = 0.5
                else:
                    seg_scale = 1.0
                    nll_ema = 0.9 * nll_ema + 0.1 * w_mean
            prev = toks[i]
            if sleep_every and (i + 1) % sleep_every == 0:
                self.net.sleep()
            if len(seg) >= log_every:
                nlls.append(float(np.mean(seg)))
                seg = []
                if valid_text is not None and plateau_patience > 0:
                    ppl = self.evaluate(valid_text)["ppl_char"]
                    if ppl < best_ppl - 1e-9:
                        best_ppl, bad = ppl, 0
                    else:
                        bad += 1
                        if bad >= plateau_patience:
                            self.net.sleep()      # 验证平台期 → 巩固
                            bad = 0
        if seg:
            nlls.append(float(np.mean(seg)))
        return nlls

    def train_stream(self, text: str, log_every: int = 1000,
                     sleep_every: int = 0, curriculum_top: float = 0.0,
                     valid_text: str | None = None) -> list[float]:
        """在线训练：逐 token 喂入，预测下一 token。

        sleep_every > 0 时周期巩固（T4.4 的定频形态）；
        curriculum_top > 0 时启用 T4.3 高频词课程（B6 修复：单遍掩码加权，
        不再两遍遍历）：高频骨架 token 高权重、低频长尾低权重，整段只过一遍。
        valid_text 给定且 cfg.plateau_sleep 时启用 T4.4 验证驱动睡眠。
        """
        toks = self.tokenize(text)
        all_keep = [True] * len(toks)
        plateau = (valid_text is not None and self.net.cfg.plateau_sleep)
        if curriculum_top <= 0.0:
            return self._pass(toks, all_keep, log_every, sleep_every, None,
                              valid_text=valid_text if plateau else None,
                              plateau_patience=self.net.cfg.plateau_patience if plateau else 0)
        freq = Counter(toks)
        ranks = sorted(freq.values(), reverse=True)
        thr = ranks[max(0, int(curriculum_top * len(ranks)) - 1)]
        # 单遍加权（B6 修复）：高频位置权 1.0、低频位置权 0.25；整段只过一遍，
        # 替代旧的两遍遍历（两遍会破坏"在线单遍/同预算"的公平口径）。
        weights = [1.0 if freq[toks[i + 1]] >= thr else 0.25
                   for i in range(len(toks) - 1)]
        return self._pass(toks, all_keep, log_every, sleep_every, weights,
                          valid_text=valid_text if plateau else None,
                          plateau_patience=self.net.cfg.plateau_patience if plateau else 0)

    def evaluate(self, text: str) -> dict:
        """只推理。返回 token PPL / 字符归一 PPL / bpc。NLL 直接取自读出输出。

        2026-09-19 修复（OOV 安全）：词表未覆盖的 token（评估段新词）按标准做法
        **跳过不计入 NLL 与字符数**，并报告 OOV 数量与比例 —— 旧实现在此直接
        KeyError（文档语料时代词表覆盖全部文本，故未暴露）。
        """
        toks = self.tokenize(text)
        total_nll, n_chars, prev = 0.0, 0, None
        n_tok, oov = 0, 0
        stoi = self.tok.stoi
        for i in range(len(toks) - 1):
            cur, nxt = toks[i], toks[i + 1]
            # OOV 安全：cur/nxt 任一不在词表则整步跳过（不推进状态、不计入 NLL），
            # 且 prev 保持最后一个已知 token —— 避免 OOV 串污染上下文
            if cur not in stoi or nxt not in stoi:
                oov += 1
                n_chars += len(nxt)
                continue
            x = self.tok.encode_composite(cur, prev)
            y = self.net.step(x, learn=False, readonly=True)["y"]  # B5：评估冻结状态，保证可复现
            y = y - y.max()
            p = np.exp(y)
            p /= p.sum()
            total_nll += -log(float(p[stoi[nxt]]) + 1e-12)
            n_chars += len(nxt)
            n_tok += 1
            prev = cur
        return {
            "ppl_token": float(np.exp(total_nll / max(1, n_tok))),
            "ppl_char": float(np.exp(total_nll / max(1, n_chars))),
            "bpc": float(total_nll / max(1, n_chars) / log(2)),
            "n_tok": n_tok,
            "oov": oov,
            "oov_rate": oov / max(1, oov + n_tok),
        }

    def generate(self, seed_token: str, n_tokens: int = 200, tau: float = 0.8,
                 rng: np.random.Generator | None = None,
                 episodic_len: int | None = None) -> list[str]:
        """自回归采样：读出分布 + 温度 τ（M11 机制的词级版），返回 token 序列。

        T3.3 情景缓冲接入生成（episodic_len>0，默认取 cfg.episodic_len）：
        每步采样前，把最近 episodic_len 个已生成 token 连同其上下文
        作为「情节流」重放观察（学习关闭）——生成受 WM 锚定 + 情景缓冲链尾
        双重条件化（内言语受情景流约束），提升长程连贯性。
        """
        rng = rng or np.random.default_rng(0)
        if episodic_len is None:
            episodic_len = self.net.cfg.episodic_len
        out = [seed_token]
        prev: str | None = None
        cur = seed_token
        for _ in range(n_tokens):
            if episodic_len > 0 and len(out) > 1:
                tail = out[-(episodic_len + 1):]          # 最近情节（含前驱上下文）
                for j in range(1, len(tail)):
                    self.net.step(self.tok.encode_composite(tail[j], tail[j - 1]),
                                  learn=False)
            x = self.tok.encode_composite(cur, prev)
            y = self.net.step(x, learn=False)["y"] / max(tau, 1e-6)
            y -= y.max()
            p = np.exp(y)
            p /= p.sum()
            nxt = self.tok.tokens[int(rng.choice(len(p), p=p))]
            out.append(nxt)
            prev, cur = cur, nxt
        return out

    def bigram_validity(self, gen_tokens: list[str],
                        train_tokens: list[str]) -> float:
        """生成 bigram 合法率（在训练 bigram 集中出现的比例）。"""
        train_b = {(train_tokens[i], train_tokens[i + 1])
                   for i in range(len(train_tokens) - 1)}
        if len(gen_tokens) < 2:
            return 0.0
        hit = sum(1 for i in range(len(gen_tokens) - 1)
                  if (gen_tokens[i], gen_tokens[i + 1]) in train_b)
        return hit / (len(gen_tokens) - 1)
