"""M5 评测任务：语言建模（任务 1）与长程依赖（任务 2）。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import math
import time

import numpy as np

from eval_common import BASE, DOC, SEG, _tf_fit
from phdnet.config import PHDNetConfig
from phdnet.lm import PHDNetLM
from phdnet.word_lm import PHDWordLM


def task1_lm(train_txt: str, eval_txt: str, text: str) -> dict:
    """语言建模：字符归一 PPL + bpc + 学习曲线。"""
    print("[任务1] 语言建模（PHD-Net 词级）")
    lm = PHDWordLM(text, PHDNetConfig(**BASE), seg_kwargs=SEG)
    curve = []
    n = len(train_txt)
    # 2026-10-07 P2-8 修复：训练/评估**分别计时**。旧实现一个 dt 同时包住
    # 4 次 train_stream 与 4 次 evaluate（复核实测评估占 25.2%）→ 训练吞吐被
    # 高估 1.34×。现两个口径都打：
    #   ms_per_token / ms_per_token_train = 纯训练（真实训练吞吐）
    #   ms_per_token_eval                 = 纯评估
    #   ms_per_token_incl_eval            = 旧合并口径（含评估，兼容对照）
    # 门禁：tests/verifiers/verify_eval_readonly.py (e)。
    t_train = 0.0
    t_eval = 0.0
    for f in (0.25, 0.5, 0.75, 1.0):
        if f == 0.25:
            chunk = train_txt[:int(n * 0.25)]
        else:
            prev = int(n * ({0.25: 0.0, 0.5: 0.25, 0.75: 0.5, 1.0: 0.75}[f]))
            chunk = train_txt[prev:int(n * f)]
        t0 = time.perf_counter()
        lm.train_stream(chunk)
        t_train += time.perf_counter() - t0
        t0 = time.perf_counter()
        m = lm.evaluate(eval_txt)
        t_eval += time.perf_counter() - t0
        curve.append((f, m["ppl_char"]))
    n_tok = len(lm.tokenize(train_txt))
    ms_tr = t_train / n_tok * 1000
    ms_ev = t_eval / n_tok * 1000
    ms_all = (t_train + t_eval) / n_tok * 1000
    print(f"    吞吐两口径: 训练 {ms_tr:.3f} ms/token | 评估 {ms_ev:.3f} ms/token"
          f"（含评估旧口径 {ms_all:.3f} ms/token）")
    return {"ppl_char": curve[-1][1], "bpc": math.log(curve[-1][1]) / math.log(2),
            "curve": curve, "ms_per_token": ms_tr, "ms_per_token_train": ms_tr,
            "ms_per_token_eval": ms_ev, "ms_per_token_incl_eval": ms_all,
            "rss_delta": None}


def _gen_copy(n_sym: int = 12, n: int = 8, n_seq: int = 150, seed: int = 0) -> tuple[str, list[int]]:
    rng = np.random.default_rng(seed)
    alpha = "abcdefghijkl"
    seqs = []
    for _ in range(n_seq):
        s = rng.choice(list(alpha[:n_sym]), size=n)
        seqs.append("".join(s) + "#" + "".join(s) + ".")
    return "".join(seqs), []


def _copy_eval_positions(text: str) -> list[int]:
    """标记第二阶段（'#' 之后）的位置：这些位置需要长程复制。"""
    pos, phase = [], 0
    for i, ch in enumerate(text):
        if ch == "#":
            phase = 1
        elif ch == ".":
            phase = 0
        elif phase == 1:
            pos.append(i)
    return pos


def copy_eval_acc(lm: PHDNetLM, n_test: int, n_seq: int = 40, seed: int = 99) -> float:
    """延迟复制第二阶段 teacher-forcing 准确率（长度 n_test）。

    2026-10-07 P2-7 修复：评估 step 必须 readonly=True —— 旧实现只传
    learn=False，三个测试长度（4/8/16）共用同一个**非只读**模型 → 结果随执行
    顺序变化（实测：先跑 n=8 首测 0.0940，先跑 n=4 再测 0.0909，Δ=−0.0031）；
    readonly 后 0.0878 → 0.0878（Δ=0），顺序无关。
    门禁：tests/verifiers/verify_eval_readonly.py (d)。
    """
    eval_txt, _ = _gen_copy(n=n_test, n_seq=n_seq, seed=seed)
    pos = _copy_eval_positions(eval_txt)
    hit = tot = 0
    for i in pos[:-1]:
        y = lm.net.step(lm.tok.encode(eval_txt[i]), learn=False, readonly=True)["y"]
        if int(np.argmax(y)) == lm.tok.stoi[eval_txt[i + 1]]:
            hit += 1
        tot += 1
    return hit / max(1, tot)


def task2_longrange() -> dict:
    """延迟复制：训练 n=8，测试 n=4/8/16 的第二阶段 teacher-forcing 准确率。"""
    print("[任务2] 长程依赖（延迟复制）")
    train_txt, _ = _gen_copy(n=8, n_seq=150, seed=1)
    vocab = "abcdefghijkl#."
    lm = PHDNetLM(vocab, PHDNetConfig(**{**BASE, "pred_in_readout": False}))
    lm.train_stream(train_txt)
    out = {}
    for n_test in (4, 8, 16):
        out[n_test] = copy_eval_acc(lm, n_test)
    print(f"    复制准确率: {out}")
    # Transformer 对照（同语料、同字符预算；滑窗 32，2 层）
    from nano_gpt import NanoGPT
    import torch
    torch.manual_seed(11)
    m = NanoGPT(len(vocab), 64, 2, 4, 32)
    ids_tr = [vocab.index(c) for c in train_txt]
    _tf_fit(m, ids_tr, epochs=4, block_size=32)
    tf_out = {}
    for n_test in (4, 8, 16):
        eval_txt, _ = _gen_copy(n=n_test, n_seq=40, seed=99)
        ids_ev = [vocab.index(c) for c in eval_txt]
        pos = _copy_eval_positions(eval_txt)
        tf_out[n_test] = _tf_copy_acc(m, ids_ev, pos, block_size=32)
    print(f"    Transformer 复制准确率: {tf_out}")
    return {"phdnet": out, "transformer": tf_out}


def _tf_copy_acc(model, ids: list[int], pos: list[int], block_size: int) -> float:
    import torch
    model.eval()
    hit = tot = 0
    with torch.no_grad():
        for i in pos[:-1]:
            ctx = torch.tensor(ids[max(0, i - block_size + 1):i + 1], dtype=torch.long)
            logits = model(ctx[None, :])[0, -1]
            if int(torch.argmax(logits)) == ids[i + 1]:
                hit += 1
            tot += 1
    return hit / max(1, tot)
