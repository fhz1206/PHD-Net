"""torch 词级 LM 等价性验证 —— torch 版 vs numpy 主实现（同 seed、冻结语料 4K 口径）。

判据（容差一致，不宣称跨设备逐位）：
  1. PPL 相对差：|ppl_torch − ppl_numpy| / ppl_numpy < 1%（fp32 协议）；
  2. 权重范数轨迹 Pearson 相关 r ≥ 0.99（读出 W 范数 / STDP W 和 / PC up0 范数，
     按 NCHUNK 段采样）。

口径：冻结语料 eval_corpus/internal_corpus.txt，80/20 划分，训练段前 4,000 字符
（≈2,780 token，本脚本取前 1,500 token 训练）；BASE 配置见 tests/eval_common.py
（eta_readout=0.15、读出 fp32）。numpy 主实现基线（全 4K token）ppl_char=90.2480，
本脚本 1,500 token 短训的绝对值与该锚点不可比——**只做 torch vs numpy 的同口径对比**。

用法：python tools/verify_torch_lm.py [--tokens 1500] [--device auto|all|cpu,cuda]
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import argparse

import numpy as np

from eval_common import BASE, DOC, SEG
from phdnet.backends.torch_backend import probe_devices
from phdnet.backends.torch_lm import TorchWordLM
from phdnet.config import PHDNetConfig
from phdnet.word_lm import PHDWordLM

NCHUNK = 6          # 权重轨迹采样段数
PPL_TOL = 0.01      # PPL 相对差容差（1%）
CORR_TOL = 0.99     # 轨迹相关系数下限


def _trajectory(lm, train_text: str) -> list[dict]:
    """分 NCHUNK 段训练并采样权重范数轨迹（numpy 与 torch 共用此逻辑）。"""
    n = len(train_text)
    bounds = [n * i // NCHUNK for i in range(NCHUNK + 1)]
    traj = []
    for c in range(NCHUNK):
        lm.train_stream(train_text[bounds[c]:bounds[c + 1]], log_every=10 ** 9)
        if not hasattr(lm.net, "weight_norms"):             # numpy 版
            ro = float(np.linalg.norm(lm.net.readout.W))
            sw = float(np.sum(lm.net.stdp.W))
            pc = float(np.linalg.norm(lm.net.pc.up0[2]))
        else:                                               # torch 版
            wn = lm.net.weight_norms()
            ro, sw, pc = wn["readout"], wn["stdp_w_sum"], wn["pc_up0"]
        traj.append({"readout": ro, "stdp": sw, "pc": pc})
    return traj


def _pearson(a: list[float], b: list[float]) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 1.0 if np.allclose(a, b, atol=1e-6) else 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1500)
    ap.add_argument("--device", type=str, default="all",
                    help="all=逐个探测到的设备 / 逗号分隔（cpu,cuda,npu）")
    args = ap.parse_args()

    with open(DOC, encoding="utf-8") as f:
        text = f.read()
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split], text[split:]
    cfg = PHDNetConfig(**BASE)

    # ---- numpy 主实现参考（同 seed）
    print("[1/2] numpy 主实现参考轨迹 ...", flush=True)
    lm_np = PHDWordLM(text, cfg, seg_kwargs=SEG)
    toks = lm_np.tokenize(train_txt[:4000])
    train_text = "".join(toks[:args.tokens])
    traj_np = _trajectory(lm_np, train_text)
    m_np = lm_np.evaluate(eval_txt)
    print(f"  numpy: ppl_char={m_np['ppl_char']:.4f}  n_tok={m_np['n_tok']}  "
          f"oov={m_np['oov']}", flush=True)

    # ---- 逐设备 torch 版
    probes = probe_devices()
    if args.device == "all":
        devices = ["cpu"] + [k for k in ("npu", "rocm", "cuda")
                             if probes.get(k, {}).get("ok")]
    else:
        devices = [d.strip() for d in args.device.split(",") if d.strip()]

    print(f"[2/2] torch 版逐设备对比（{devices}）...", flush=True)
    all_ok = True
    for dev in devices:
        try:
            lm_t = TorchWordLM(text, cfg, device=dev, seg_kwargs=SEG)
        except (RuntimeError, NotImplementedError) as e:
            print(f"  [{dev}] 跳过：{e}")
            if dev != "cpu":
                all_ok = False
            continue
        real = lm_t.device
        traj_t = _trajectory(lm_t, train_text)
        m_t = lm_t.evaluate(eval_txt)
        rel = abs(m_t["ppl_char"] - m_np["ppl_char"]) / m_np["ppl_char"]
        corrs = {k: _pearson([t[k] for t in traj_np], [t[k] for t in traj_t])
                 for k in ("readout", "stdp", "pc")}
        ok = rel < PPL_TOL and all(v >= CORR_TOL for v in corrs.values())
        all_ok &= ok
        print(f"  [{dev}]（实际 {real}） ppl_char={m_t['ppl_char']:.4f}  "
              f"PPL 相对差={rel * 100:.5f}%  轨迹相关 "
              f"readout={corrs['readout']:.6f} stdp={corrs['stdp']:.6f} "
              f"pc={corrs['pc']:.6f}  {'PASS' if ok else 'FAIL'}", flush=True)

    print("-" * 66)
    print("PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
