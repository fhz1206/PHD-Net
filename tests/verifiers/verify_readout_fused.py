"""P7 融合并行读出核的对拍：与 numpy 三步路径**逐位等价**验证 + 加速比。

四层验证（分层门禁结构）：
  L1 单元：同一 Readout 实例，融合核 vs numpy 路径，同一输入 → W 逐位相同
  L2 序列：随机 500 步（含 eta=0 / 重复 h / 极端值），两条路径终态 W 逐位相同
  L3 端到端：两个 PHDWordLM（同 seed）跑同一语料，PPL 与权重哈希逐位相同
  L4 吞吐：融合核 vs numpy 的 ms/token
"""
from __future__ import annotations

import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np

import phdnet.readout as R
from phdnet.config import PHDNetConfig
from phdnet.word_lm import PHDWordLM

CORPUS = "eval_corpus/internal_corpus.txt"
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)
BASE = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
            eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
            pred_in_readout=True)


def _force(state: bool) -> None:
    R._RO_STATE["ok"] = state


def _bits(a: np.ndarray) -> str:
    return f"{a.view(np.uint64).sum()}"     # 逐位敏感摘要


def l1_unit() -> bool:
    rng = np.random.default_rng(7)
    n_in, n_out = 768, 1540
    h = rng.normal(size=n_in)
    tgt = np.zeros(n_out)
    tgt[123] = 1.0
    Wa = rng.normal(0, 0.05, (n_out, n_in))
    Wb = Wa.copy()
    ro = R.Readout(n_in, n_out, np.random.default_rng(1))
    ro._W = Wa
    y = Wa @ h
    y = y - y.max()
    p = np.exp(y)
    p /= p.sum()
    dp = p - tgt
    eta = 0.05

    _force(False)
    tmp = np.outer(dp, h)
    tmp *= eta
    Wb -= tmp

    _force(True)
    ok = R._ro_fused(Wa, dp, h, eta)
    same = np.array_equal(Wa, Wb) and np.array_equal(Wa.view(np.uint64),
                                                     Wb.view(np.uint64))
    print(f"[L1] 融合核启用={ok}  逐位相同={same}")
    if not same:
        d = np.abs(Wa - Wb)
        print(f"     最大偏差 {d.max():.3e}  不同元素 {int((d > 0).sum())}")
    return same


def l2_sequence() -> bool:
    rng = np.random.default_rng(11)
    n_in, n_out = 768, 1540
    Wa = rng.normal(0, 0.05, (n_out, n_in))
    Wb = Wa.copy()
    nlls = [[], []]
    # 预生成输入序列（两条路径必须消费**同一序列**，否则不可比）
    seq = []
    for t in range(500):
        if t % 37 == 0:
            h = np.zeros(n_in)                      # 全零输入
        elif t % 23 == 0:
            h = rng.normal(size=n_in) * 100         # 大幅值
        else:
            h = rng.normal(size=n_in)
        tgt = np.zeros(n_out)
        tgt[int(rng.integers(0, n_out))] = 1.0
        eta = 0.0 if t % 41 == 0 else 0.05          # 零学习率短路
        seq.append((h, tgt, eta))
    for mode, W in ((False, Wb), (True, Wa)):
        _force(mode)
        ro = R.Readout(n_in, n_out, np.random.default_rng(2))
        ro._W = W
        for h, tgt, eta in seq:
            nlls[1 if mode else 0].append(
                ro.learn_softmax(h, tgt, eta, y_pre=ro.W @ h))
    same_w = np.array_equal(Wa.view(np.uint64), Wb.view(np.uint64))
    same_nll = nlls[0] == nlls[1]
    print(f"[L2] 500 步序列：W 逐位相同={same_w}  NLL 序列相同={same_nll}")
    if not same_w:
        d = np.abs(Wa - Wb)
        print(f"     最大偏差 {d.max():.3e}  不同元素 {int((d > 0).sum())}")
    return bool(same_w and same_nll)


def l3_endtoend(n_step: int = 1500) -> tuple[bool, float, float]:
    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()
    res = {}
    for mode in (False, True):
        _force(mode)
        cfg = PHDNetConfig(**BASE)
        lm = PHDWordLM(text, cfg, seg_kwargs=SEG)
        toks = lm.tokenize(text)
        prev = None
        nll, n, chars = 0.0, 0, 0
        t0 = time.perf_counter()
        for i in range(len(toks) - 1):
            if toks[i + 1] not in lm.tok.stoi:
                prev = toks[i]
                continue
            x = lm.tok.encode_composite(toks[i], prev)
            target = lm.tok.onehot(lm.tok.stoi[toks[i + 1]])
            d = lm.net.step(x, target=target, learn=True)
            nll += d["nll"]
            n += 1
            chars += len(toks[i + 1])
            prev = toks[i]
            if n >= n_step:
                break
        dt = (time.perf_counter() - t0) / n * 1000
        res[mode] = (float(np.exp(nll / max(1, chars))), dt,
                     _bits(lm.net.readout.W))
    same = res[False][2] == res[True][2] and abs(res[False][0] - res[True][0]) < 1e-12
    print(f"[L3] 端到端 {n_step} 步：numpy PPL {res[False][0]:.4f} / "
          f"融合 PPL {res[True][0]:.4f}  权重逐位相同={same}")
    print(f"[L4] 吞吐：numpy {res[False][1]:.3f} ms/token → "
          f"融合 {res[True][1]:.3f} ms/token  "
          f"（{res[False][1]/max(1e-9, res[True][1]):.2f}×）")
    return bool(same), res[False][1], res[True][1]


if __name__ == "__main__":
    print("=" * 66)
    print("P7 融合并行读出核 —— 逐位等价对拍")
    print("=" * 66)
    ok1 = l1_unit()
    ok2 = l2_sequence()
    ok3, a, b = l3_endtoend()
    print("-" * 66)
    print(f"L1={ok1}  L2={ok2}  L3={ok3}   → {'PASS' if all([ok1, ok2, ok3]) else 'FAIL'}"
          f"   加速 {a/max(1e-9,b):.2f}×")
    _force(None)
