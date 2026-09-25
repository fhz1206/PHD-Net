"""第四轮审计 · big_ltm 印迹门控实验（2026-09-25，配 docs/性能评估与迭代方案）。

背景（审计发现）：默认 `mem_gate > 0.8` 的印迹条件在 LM 训练中几乎从不触发
（调制器 z 分布 std≈0.32，gate>0.8 实测仅 ~0.4% 步）→ big_ltm 的 1B 容量
从未物化、检索通路空转。

本实验：新增开关 `ltm_imprint_gate`（默认 0.8 = 旧行为逐位不变），在冻结语料
BASE 口径（80/20 划分，big_ltm=True）下测各阈值对 PPL 与表生长的影响。
"""

from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from eval_common import BASE, DOC, SEG                 # noqa: E402
from phdnet.config import PHDNetConfig                 # noqa: E402
from phdnet.word_lm import PHDWordLM                   # noqa: E402


def run(gate: float, limit: int | None, big_ltm: bool = True) -> dict:
    with open(DOC, encoding="utf-8") as f:
        text = f.read()
    split = int(len(text) * 0.8)
    train_txt = text[:split][:limit] if limit else text[:split]
    eval_txt = text[split:]
    cfg = PHDNetConfig(**BASE, big_ltm=big_ltm,
                       big_ltm_N=1 << 20, big_ltm_m=60, big_ltm_k=4,
                       ltm_imprint_gate=gate)
    lm = PHDWordLM(text, cfg, seg_kwargs=SEG)
    counts = {"imprint": 0, "recall": 0}
    o_imp = lm.net.ltm.imprint
    o_rec = lm.net.ltm.recall
    lm.net.ltm.imprint = lambda x: (counts.__setitem__("imprint", counts["imprint"] + 1),
                                    o_imp(x))[1]
    lm.net.ltm.recall = lambda x: (counts.__setitem__("recall", counts["recall"] + 1),
                                   o_rec(x))[1]
    t0 = time.perf_counter()
    lm.train_stream(train_txt)
    dt = time.perf_counter() - t0
    m = lm.evaluate(eval_txt)
    n_tok = len(lm.tokenize(train_txt))
    st = (lm.net.ltm.table.stats() if big_ltm and hasattr(lm.net.ltm, "table")
          else {"grown_synapses": -1})
    return dict(ppl=m["ppl_char"], ms=dt / max(1, n_tok) * 1000,
                imprint=counts["imprint"], recall=counts["recall"],
                grown=st["grown_synapses"])


if __name__ == "__main__":
    full = "--full" in sys.argv
    print("=" * 88)
    lim = None if full else 4000
    print(f"big_ltm 印迹门控实验（冻结语料 80/20；BASE + big_ltm 2^20；"
          f"口径={'全语料' if full else '4,000 字符'}训练段）")
    print("=" * 88)
    print(f"{'gate':>6} {'ppl_char':>10} {'Δvs0.8':>8} {'ms/tok':>8} "
          f"{'imprint':>8} {'recall':>7} {'grown':>9}")
    base_ppl = None
    gates = (0.8, 0.5) if full else (0.8, 0.6, 0.5, 0.4)
    for gate in gates:
        r = run(gate, lim)
        if base_ppl is None:
            base_ppl = r["ppl"]
        d = (r["ppl"] - base_ppl) / base_ppl * 100
        print(f"{gate:>6.2f} {r['ppl']:>10.4f} {d:>+7.2f}% {r['ms']:>8.3f} "
              f"{r['imprint']:>8} {r['recall']:>7} {r['grown']:>9,}", flush=True)
    print("── 对照：big_ltm=False（稠密双速率 LTM，库默认路径锚点）──")
    r = run(0.8, lim, big_ltm=False)
    d = (r["ppl"] - base_ppl) / base_ppl * 100
    print(f"{'dense':>6} {r['ppl']:>10.4f} {d:>+7.2f}% {r['ms']:>8.3f} "
          f"{r['imprint']:>8} {r['recall']:>7} {'—':>9}", flush=True)
