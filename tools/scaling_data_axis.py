"""O7 补充实验：scaling 的**数据轴**（参数轴受单回合预算限制，此轴补足信息量）。

背景：参数轴实测（`tools/scaling_curve.py`）在**固定 4,000 字符训练预算**下得到
  1.71M 参数 → ppl_char 97.26；17M 参数 → 119.10（**参数越大越差**）
——这是典型的"固定数据预算下大模型欠训练"效应，α 无正区间含义。

本脚本固定 P1 架构（1.71M），只放大**训练数据量**（4,000 / 8,000 / 16,000 字符，
同一语料顺序截取），给出数据轴 PPL 曲线：
  - 若 PPL 随数据量显著下降 → 证实"瓶颈是数据预算而非参数规模"，
    即远域泛化与 scaling 的正确路径是扩语料（已就位 M7_Core 中文子集 ~75 万条 / ~1GB）；
  - 与参数轴合并可给出二维结论（参数↑ 需 数据↑ 配套）。

用法：python tools/scaling_data_axis.py
"""
from __future__ import annotations

import json
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

CORPUS = _ROOT / "datasets" / "eval" / "internal_corpus.txt"
LOG = _ROOT / "outputs" / "scaling_data_axis.log"
JSON = _ROOT / "outputs" / "scaling_data_axis.json"
BASE = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
            eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11, pred_in_readout=True)
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)
BUDGETS = [4000, 8000, 16000]


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main() -> None:
    text = CORPUS.read_text(encoding="utf-8")
    split = int(len(text) * 0.8)
    train_full, eval_txt = text[:split], text[split:]
    LOG.write_text("", encoding="utf-8")
    log(f"# O7 数据轴：固定 P1 架构（n_sdr=n_mid=n_top=256），放大训练数据量")
    log(f"# 语料 {len(text)} 字符 | 训练段可用 {len(train_full)} | 评估段 {len(eval_txt)}（全量）")
    log("")

    rows = []
    for nb in BUDGETS:
        chunk = train_full[:nb]
        t0 = time.perf_counter()
        lm = PHDWordLM(text, PHDNetConfig(**BASE), seg_kwargs=SEG)
        n_par = count_params(lm.net)
        n_tok = len(lm.tokenize(chunk))
        lm.train_stream(chunk)
        m = lm.evaluate(eval_txt)
        dt = time.perf_counter() - t0
        row = dict(train_chars=nb, n_params=n_par, n_tok_train=n_tok,
                   ppl_char=float(m["ppl_char"]), bpc=float(m["bpc"]), sec=round(dt, 1))
        rows.append(row)
        log(f"[{nb:>6} 字符] {n_tok:>5} tokens | ppl_char={row['ppl_char']:.4f} "
            f"bpc={row['bpc']:.4f} | {dt:.1f}s")
        JSON.write_text(json.dumps({"points": rows}, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    log("")
    log("汇总（固定参数量 %d）：" % rows[0]["n_params"])
    log(f"{'训练字符':>10}{'tokens':>10}{'ppl_char':>12}{'相对 4000':>12}")
    base_ppl = rows[0]["ppl_char"]
    for r in rows:
        log(f"{r['train_chars']:>10}{r['n_tok_train']:>10}{r['ppl_char']:>12.4f}"
            f"{(r['ppl_char'] - base_ppl) / base_ppl * 100:>11.1f}%")
    # 数据轴幂律（PPL = a·D^(−β)）
    D = np.log10([r["train_chars"] for r in rows])
    P = np.log10([r["ppl_char"] for r in rows])
    slope, intercept = np.polyfit(D, P, 1)
    log("")
    log(f"数据轴幂律: PPL = {10 ** intercept:.3f} · D^({slope:.4f})   β={-slope:.4f}")
    log("")
    log("对照（参数轴，固定 4,000 字符预算）: 1.71M→97.26 / 17M→119.10（参数↑反劣 = 欠训练）")
    log("判读：数据轴斜率 β>0 且显著 → 瓶颈在数据预算；配合已就位 M7_Core（中文 ~75 万条/~1GB）"
        "即可扩大预训练预算。")


if __name__ == "__main__":
    main()
