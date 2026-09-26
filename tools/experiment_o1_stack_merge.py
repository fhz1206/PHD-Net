"""O1 容量栈合并实验：词级 LM 栈 × 1B 事件驱动容量栈（含在线 CSR）。

背景（路线图 §五 O1 / 瓶颈 B3）：LM 栈（~1.71M 稠密）与 1B 事件驱动栈
（`SparseLTM`：N=2^24 × m_out=60 ≈ 1.0B 突触容量）此前**未在评测路径上合并**，
且大容量表只有 dict 邻接。本轮打通接线并给出实测。

配置组：
  A dense      ：稠密双速率 LTM（历史基线）
  B big1B      ：1B 容量栈（dict 邻接，事件驱动生长）
  C big1B+csr  ：B + 在线可写 CSR 结构（`csr_online`）
  D big1B+int8 ：B + int8 量化 + 生长引导（T5.2/T5.3）
  E big1B+summ ：B + P5 摘要槽（`wm_summary_every`）—— 验证大容量表下的契约适配

口径：冻结语料，训练段前 4,000 字符（与 ablation 同口径，稠密基线 97.2596）；
      评估段后 20% 全量。报告 ppl_char / ms·token⁻¹ / 已生长突触 / 参数量。

用法：python tools/experiment_o1_stack_merge.py
"""
from __future__ import annotations

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

CORPUS = _ROOT / "eval_corpus" / "internal_corpus.txt"
BASE = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
            eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11, pred_in_readout=True)
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)
TRAIN_CHARS = 4000
BIG = dict(big_ltm=True, big_ltm_N=1 << 24, big_ltm_m=60, big_ltm_k=4)

CONFIGS = [
    ("A_dense", {}, "稠密双速率 LTM（基线，97.2596）"),
    ("B_big1B", dict(**BIG), "1B 容量栈（dict 邻接，事件驱动生长）"),
    ("C_big1B_csr", dict(**BIG, csr_online=True, csr_grow_chunk=8),
     "1B 容量栈 + 在线 CSR（定长行 + 预留槽）"),
    ("D_big1B_int8", dict(**BIG, sparse_int8=True, growth_guidance=True),
     "1B 容量栈 + int8 量化 + 生长引导"),
    ("E_big1B_summary", dict(**BIG, wm_summary_every=50),
     "1B 容量栈 + P5 摘要槽（验证契约适配）"),
]


def table_stats(net) -> dict:
    if not net.cfg.big_ltm:
        return {}
    t = net.ltm.table
    st = t.stats()
    return {"grown": st["grown_synapses"], "capacity": st["capacity"],
            "neurons_with_out": st["neurons_with_out"]}


def main() -> None:
    text = CORPUS.read_text(encoding="utf-8")
    split = int(len(text) * 0.8)
    train_txt = text[:split][:TRAIN_CHARS]
    eval_txt = text[split:]
    print(f"语料 {len(text)} | 训练段 {len(train_txt)} | 评估段 {len(eval_txt)}")
    print("=" * 96)
    rows = []
    for name, ov, note in CONFIGS:
        try:
            cfg = PHDNetConfig(**{**BASE, **ov})
            lm = PHDWordLM(text, cfg, seg_kwargs=SEG)
            n_par = count_params(lm.net)
            n_tok = len(lm.tokenize(train_txt))
            t0 = time.perf_counter()
            lm.train_stream(train_txt)
            dt = time.perf_counter() - t0
            m = lm.evaluate(eval_txt)
            ms = dt / max(1, n_tok) * 1000
            st = table_stats(lm.net)
            grown = st.get("grown", "-")
            cap = st.get("capacity", "-")
            rows.append((name, float(m["ppl_char"]), ms, n_par, grown, cap, dt, note))
            gs = f"{grown:,}" if isinstance(grown, int) else str(grown)
            cs = f"{cap:,}" if isinstance(cap, int) else str(cap)
            print(f"{name:<16} ppl_char={m['ppl_char']:>9.4f}  {ms:>6.3f} ms/tok  "
                  f"params={n_par:>10,}  生长突触={gs:>10} / 容量 {cs}", flush=True)
        except Exception as e:  # 单配置失败不影响其余
            import traceback
            print(f"{name:<16} 失败: {e!r}")
            traceback.print_exc()
            rows.append((name, None, None, None, None, None, 0.0, note))

    base = rows[0][1]
    print("=" * 96)
    print(f"{'配置':<16}{'ppl_char':>10}{'ΔPPL%':>9}{'ms/token':>11}{'参数量':>13}  说明")
    for name, ppl, ms, n_par, grown, cap, dt, note in rows:
        if ppl is None:
            print(f"{name:<16}{'FAIL':>10}{'--':>9}{'--':>11}{'--':>13}  {note}")
            continue
        print(f"{name:<16}{ppl:>10.4f}{(ppl - base) / base * 100:>+9.2f}"
              f"{ms:>11.3f}{n_par:>13,}  {note}")
    print("=" * 96)
    print("判读：A→B 即「容量栈合并」是否可用；C 验证在线 CSR 在 LM 栈的实际代价；")
    print("      D 验证量化/生长引导；E 验证大容量表下 P5 摘要槽契约适配。")


if __name__ == "__main__":
    main()
