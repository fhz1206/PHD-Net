"""规模-吞吐实测：为「256M 参数生产模型」可行性判断提供实测依据。

方法：固定配方（稀疏主干 `sparse_conn`，conn_k = 各层 n_in//8，即 12.5% 连接），
      逐档放大主干宽度与词表规模，实测 (参数量, ms/token)，并外推 256M 档的
      训练时间。同时给出**内存需求**（权重 + 事件驱动记忆表）。

口径：真实语料（`datasets/sft/ultrainteract_sft.txt` 前若干行）+ 词级 LM 管线。
      每档跑固定 token 数（默认 400 tokens）测稳态 ms/token。

用法：
  python tools/estimate_scale.py                # 默认档位
  python tools/estimate_scale.py --tokens 800
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.config import PHDNetConfig          # noqa: E402
from phdnet.model import count_params           # noqa: E402
from phdnet.word_lm import PHDWordLM            # noqa: E402

SFT = _ROOT / "datasets" / "sft" / "ultrainteract_sft.txt"
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)
BASE = dict(k_sparse=0, eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
            pred_in_readout=True, sparse_conn=True)

# (标签, n_sdr=n_mid=n_top, k_sparse 比例)
SCALES = [
    ("S1_0.2M", 256, 0.125),
    ("S2_1M", 512, 0.125),
    ("S3_4M", 1024, 0.125),
    ("S4_16M", 2048, 0.125),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--chars", type=int, default=20000)
    args = ap.parse_args()

    text = SFT.read_text(encoding="utf-8")[: args.chars]
    print("=" * 92)
    print(f"规模-吞吐实测（稀疏主干 12.5% 连接）| 语料 {len(text):,} 字符（SFT）| "
          f"每档 {args.tokens} tokens")
    print("=" * 92)
    rows = []
    for tag, width, k_ratio in SCALES:
        cfg = PHDNetConfig(**{**BASE, "n_sdr": width, "n_mid": width, "n_top": width,
                              "k_sparse": max(8, int(width * 0.125)), "conn_k": 0})
        lm = PHDWordLM(text, cfg, seg_kwargs=SEG)
        n_par = count_params(lm.net)
        toks = lm.tokenize(text)
        n = min(args.tokens, len(toks) - 1)
        pc_st = lm.net.pc.stats()
        t0 = time.perf_counter()
        for i in range(n):
            lm.net.step(*_mk_input(lm, toks, i), learn=True)
        dt = time.perf_counter() - t0
        ms = dt / max(1, n) * 1000
        rows.append((tag, width, n_par, ms, pc_st["connectivity"]))
        print(f"{tag:<8} 宽度 {width:<5} 参数量 {n_par:>10,}  已测 {n} tokens  "
              f"{ms:>8.3f} ms/token  主干连接率 {pc_st['connectivity'] * 100:.1f}%",
              flush=True)

    print("-" * 92)
    # 拟合 ms/token 与参数量（幂律）
    N = np.array([r[2] for r in rows], dtype=float)
    T = np.array([r[3] for r in rows], dtype=float)
    slope, intercept = np.polyfit(np.log10(N), np.log10(T), 1)
    print(f"拟合: ms/token ≈ 10^{intercept:.2f} × N^{slope:.3f}（N = 参数量）")

    for target in (256e6,):
        ms_t = 10 ** (intercept + slope * np.log10(target))
        print(f"\n外推 256M 参数（{target / 1e6:.0f}M）：")
        print(f"  单 token 耗时 ≈ {ms_t:,.0f} ms（{ms_t / 1000:.1f} s）")
        print(f"  权重内存（fp64）≈ {target * 8 / 1e9:.2f} GB；"
              f"（fp32）≈ {target * 4 / 1e9:.2f} GB")
        for nt in (1e6, 1e7, 1e8):
            h = ms_t * nt / 1000 / 3600
            print(f"  训练 {nt:.0e} tokens ≈ {h:,.0f} 小时"
                  f"（{h / 24:,.1f} 天）")

    print("\n" + "=" * 92)
    print("注：本机为纯 CPU（无 GPU/NPU）。生产可用还需数据量（10^9 token 量级）与质量门槛。")


def _mk_input(lm, toks, i):
    """构造与 PHDWordLM 训练路径一致的 (x, target)。"""
    prev = toks[i - 1] if i > 0 else None
    x = lm.tok.encode_composite(toks[i], prev)
    tgt = lm.tok.onehot(lm.tok.stoi[toks[i + 1]])
    return x, tgt


if __name__ == "__main__":
    main()
