"""第四轮审计 · 1B 生产配置性能剖析（模块级计时，2026-09-25）。

配置 = train 的 `1b` 预设（width=1024 / conn_k=128 / big_ltm 2^24），
词表取 sft 采样文本涌现词表（与生产管线同一路径）。
方法：对 net.step 管线各模块方法做 perf_counter 包装（累计挂钟），
热身后测 N 步；另报端到端 ms/token 与 readout 读出更新内存流量推算。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)

from phdnet.word_lm import PHDWordLM                     # noqa: E402
from phdnet.config import PHDNetConfig                   # noqa: E402

WIDTH = 1024
STEPS_WARM = 50
STEPS_MEAS = 300


def build(cfg_kwargs: dict, vocab_text: str):
    cfg = PHDNetConfig(
        n_sdr=WIDTH, n_mid=WIDTH, n_top=WIDTH,
        k_sparse=max(8, WIDTH // 8),
        eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
        pred_in_readout=True, sparse_conn=True, conn_k=128,
        big_ltm=True, big_ltm_N=1 << 24, big_ltm_m=60, big_ltm_k=4,
        **cfg_kwargs)
    lm = PHDWordLM(vocab_text, cfg)
    return lm, cfg


def wrap_timers(lm) -> dict:
    acc: dict = {}
    specs = [
        ("M1 encoder.encode", lm.net.encoder, "encode"),
        ("M2 pc.infer", lm.net.pc, "infer"),
        ("M2 pc.learn", lm.net.pc, "learn"),
        ("M3 stdp.predict", lm.net.stdp, "predict"),
        ("M3 stdp.step", lm.net.stdp, "step"),
        ("M4a wm.write", lm.net.wm, "write"),
        ("M4b ltm.imprint", lm.net.ltm, "imprint"),
        ("M4b ltm.recall", lm.net.ltm, "recall"),
        ("M6 readout.__call__", lm.net.readout, "__call__"),
        ("M6 readout.learn_softmax", lm.net.readout, "learn_softmax"),
    ]
    orig = {}
    for name, obj, meth in specs:
        fn = getattr(obj, meth)
        cnt = [0]
        t0 = [0.0]

        def make(fn, name):
            def wrapped(*a, **k):
                t = time.perf_counter()
                try:
                    return fn(*a, **k)
                finally:
                    dt = time.perf_counter() - t
                    acc.setdefault(name, [0.0, 0])
                    acc[name][0] += dt
                    acc[name][1] += 1
            return wrapped

        setattr(obj, meth, make(fn, name))
        orig[(id(obj), meth)] = fn
    return acc


def main() -> None:
    from phdnet.corpus import iter_texts
    it = iter_texts(_ROOT / "datasets" / "sft" / "sft_000.000.parquet")
    vocab_text = ""
    for t in it:
        vocab_text += t + "\n\n"
        if len(vocab_text) >= 200_000:
            break
    vocab_text = vocab_text[:200_000]

    lm, cfg = build({}, vocab_text)
    V = len(lm.tok)
    ro_bytes = V * (cfg.n_top * 3) * 8
    print(f"配置: width={WIDTH} conn_k={cfg.conn_k} big_N={cfg.big_ltm_N:,} "
          f"词表={V:,} 读出权重={ro_bytes / 1e6:.1f} MB（fp64）")

    toks = lm.tok.seg.tokenize(vocab_text)
    # 循环使用 token 流
    n = len(toks)
    print(f"采样 token 数（200K 字符）: {n:,} → 循环供 {STEPS_WARM + STEPS_MEAS} 步")

    acc = wrap_timers(lm)
    stoi = lm.tok.stoi

    def one(i, learn):
        p2 = toks[(i - 2) % n] if i >= 2 else None
        p1 = toks[(i - 1) % n]
        t0 = toks[i % n]
        x = lm.tok.encode_composite(p1, p2)
        tgt = lm.tok.onehot(stoi[t0])
        return lm.net.step(x, target=tgt, learn=learn)

    for i in range(2, 2 + STEPS_WARM):
        one(i, learn=True)
    t = time.perf_counter()
    for i in range(2 + STEPS_WARM, 2 + STEPS_WARM + STEPS_MEAS):
        one(i, learn=True)
    total = time.perf_counter() - t

    print("-" * 72)
    print(f"{'模块':34s} {'ms/token':>10s} {'占比':>7s} {'调用数':>7s}")
    rows = sorted(acc.items(), key=lambda kv: -kv[1][0])
    for name, (sec, cnt) in rows:
        ms = sec / STEPS_MEAS * 1000
        print(f"{name:34s} {ms:10.3f} {ms / (total / STEPS_MEAS * 1000) * 100:6.1f}% {cnt:7d}")
    ms_tok = total / STEPS_MEAS * 1000
    print("-" * 72)
    print(f"net.step 端到端: {ms_tok:.3f} ms/token（learn=True，含编排）")
    print(f"读出更新内存流量推算: ≈5×{ro_bytes / 1e6:.0f} MB = "
          f"{5 * ro_bytes / 1e6:.0f} MB/token → 理论带宽下限 ≈ "
          f"{5 * ro_bytes / 1e9 / 8.4 * 1000:.2f} ms/token（8.4 GB/s 单核）")
    grown = lm.net.ltm.table.stats()["grown_synapses"] if hasattr(lm.net.ltm, "table") else 0
    print(f"大空间表已生长: {grown:,}")


if __name__ == "__main__":
    main()
