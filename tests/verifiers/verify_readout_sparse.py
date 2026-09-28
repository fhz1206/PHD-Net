"""O1-3 验证：读出头结构性稀疏连接（CSR）的正确性与端到端效果。

Part 1 对拍：用稠密读出的权重构造稀疏读出（k = 全列，**权重完全相同**），
           相同 h/target 序列下 softmax 前向与学习更新应在浮点容差内一致
           （CSR 求和顺序 vs BLAS）。
Part 2 端到端：词级 LM 训练+评估（4,000 字符口径，稠密基线 97.2596），
           读出连接率 3.1% / 6.2% / 12.5% 的 PPL / 速度 / 参数对比。

用法：python tests/verifiers/verify_readout_sparse.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.config import PHDNetConfig          # noqa: E402
from phdnet.model import count_params           # noqa: E402
from phdnet.readout import Readout              # noqa: E402
from phdnet.word_lm import PHDWordLM            # noqa: E402

CORPUS = _ROOT / "eval_corpus" / "internal_corpus.txt"
BASE = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
            eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11, pred_in_readout=True)
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)
TOL = 1e-10
_ok = True


def check(name: str, ok: bool, extra: str = "") -> None:
    global _ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))
    _ok &= ok


def part1_equiv() -> None:
    print("=" * 88)
    print("Part 1 · 稀疏读出 vs 稠密读出（k = 全列，权重完全相同）")
    print("=" * 88)
    n_in, n_out = 128, 64
    rng = np.random.default_rng(7)
    dense = Readout(n_in, n_out, rng)
    sparse = Readout.from_dense(dense.W, k=0)          # k=0 → 全列
    st = sparse.stats()
    print(f"稠密元素 {st['dense_equivalent']:,}；稀疏存在连接 {st['synapses']:,}"
          f"（连接率 {st['connectivity'] * 100:.1f}%）")
    check("初始权重一致", np.allclose(dense.W, sparse.W, atol=0, rtol=0),
          f"max|Δ|={np.abs(dense.W - sparse.W).max():.3e}")

    rng2 = np.random.default_rng(3)
    max_dy = 0.0
    for _ in range(120):
        h = rng2.normal(0, 1, n_in)
        tgt = np.zeros(n_out)
        tgt[int(rng2.integers(n_out))] = 1.0
        yd = dense(h)
        ys = sparse(h)
        max_dy = max(max_dy, float(np.abs(yd - ys).max()))
        dense.learn_softmax(h, tgt, 0.05)
        sparse.learn_softmax(h, tgt, 0.05)
    check("120 步前向 y 一致", max_dy < TOL, f"max|Δy|={max_dy:.3e}")
    check("120 步学习后权重一致", np.abs(dense.W - sparse.W).max() < TOL,
          f"max|Δ|={np.abs(dense.W - sparse.W).max():.3e}")

    print("\n稀疏读出的规模收益（n_out=64, n_in=128）：")
    total = n_out * n_in
    for k in (4, 8, 16):
        r = Readout(n_in, n_out, np.random.default_rng(7), conn_k=k)
        s = r.stats()
        print(f"  conn_k={k:<3} 存在连接 {s['synapses']:>6,}  连接率 {s['connectivity'] * 100:>5.1f}%"
              f"  存储压缩 {total / s['synapses']:.1f}×")


def part2_e2e() -> None:
    print("\n" + "=" * 88)
    print("Part 2 · 端到端（词级 LM，训练段前 4,000 字符）")
    print("=" * 88)
    text = CORPUS.read_text(encoding="utf-8")
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split][:4000], text[split:]
    rows = []
    for name, ov in (("dense_readout", {}),
                     ("readout_k8", dict(readout_conn_k=8)),
                     ("readout_k16", dict(readout_conn_k=16)),
                     ("readout_k32", dict(readout_conn_k=32))):
        cfg = PHDNetConfig(**{**BASE, **ov})
        lm = PHDWordLM(text, cfg, seg_kwargs=SEG)
        n_par = count_params(lm.net)
        n_tok = len(lm.tokenize(train_txt))
        t0 = time.perf_counter()
        lm.train_stream(train_txt)
        dt = time.perf_counter() - t0
        m = lm.evaluate(eval_txt)
        st = lm.net.readout.stats()
        rows.append((name, float(m["ppl_char"]), dt / max(1, n_tok) * 1000, n_par,
                     st["connectivity"]))
        print(f"{name:<16} ppl_char={m['ppl_char']:>9.4f}  "
              f"{dt / max(1, n_tok) * 1000:>6.3f} ms/tok  params={n_par:>9,}  "
              f"读出连接率 {st['connectivity'] * 100:>5.1f}%", flush=True)
    base = rows[0][1]
    print("-" * 88)
    for name, ppl, ms, n_par, conn in rows:
        print(f"{name:<16}{ppl:>10.4f}{(ppl - base) / base * 100:>+8.2f}%"
              f"{ms:>10.3f} ms {n_par:>10,} {conn * 100:>7.1f}%")


def main() -> None:
    part1_equiv()
    part2_e2e()
    print("\n=== " + ("全部通过" if _ok else "存在超容差差异") + " ===")
    sys.exit(0 if _ok else 1)


if __name__ == "__main__":
    main()
