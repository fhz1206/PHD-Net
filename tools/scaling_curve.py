"""O7 —— PHD-Net 词级 LM 参数规模缩放曲线（幂律 α 拟合）。

任务：固定配方跑 3 个参数规模点，实测 (参数量, ppl_char, bpc, ms/token)，
     拟合幂律 PPL = a · N^(-α)，报告 α 与 R²。

============================================================================
训练口径（重要，务必先读）：
  - 词表由【全语料】(eval_corpus/internal_corpus.txt, 23,504 字符) 构建，
    三点共用同一 SEG（max_len=6, min_count=5, min_entropy=1.0）→ 词表大小
    对三点为共同项（不计入口径差异），符合"固定配方"。
  - 评测段 = 冻结语料后 20%（4,701 字符），三点共用、不变。
  - 训练段统一截断为【前 4,000 字符】单遍训练（三点同一截断口径）。
  - 截断理由：按实测 per-token 耗时线性外推，顶层 ~1.7 亿参数（=1.7M 点的 ~100×）
    在完整 80%（18,803 字符）训练段下单遍预计 > 3 小时；为让三点在可比口径下
    都能在预算内跑完，三点统一截到前 4,000 字符。这是 compute/数据受控的缩放
    协议（Kaplan 式），幂律反映"同数据预算下参数规模 → 困惑度"的关系。
  - 注意：因训练数据截断，本实验 ppl_char 绝对值高于文档基线 78.16（后者用
    完整 80% 训练）；三点彼此可比，幂律结论不受影响。

三点配置（仅宽度 n_sdr=n_mid=n_top 等比放大；k_sparse=n_sdr//8 维持 12.5% 激活率）：
  P1: n_sdr=256   → ~1.7M   （即文档已知单点 BASE）
  P2: n_sdr=1200  → ~17M
  P3: n_sdr=4350  → ~170M
参数规模一律用 phdnet.model.count_params 实测（非理论估算）。

输出：
  outputs/experiments/scaling_curve.log   每完成一点立即 append（flush=True）
  outputs/experiments/scaling_curve.json  已完成点 + 最终 α/R²（中断也可保留已得点）
============================================================================
"""
import sys, json, time, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from phdnet.config import PHDNetConfig
from phdnet.word_lm import PHDWordLM
from phdnet.model import count_params

DOC = str(ROOT / "eval_corpus" / "internal_corpus.txt")
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)
BASE = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
            eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
            pred_in_readout=True)

N_CHARS_TRAIN = 4000          # 三点统一训练截断口径（前 4000 字符）
LOG_PATH = ROOT / "outputs" / "scaling_curve.log"
JSON_PATH = ROOT / "outputs" / "scaling_curve.json"

# 三点：仅宽度等比放大；k_sparse = n_sdr//8 维持 12.5% 激活率
POINTS = [
    ("P1_1.7M",  dict(n_sdr=256,  n_mid=256,  n_top=256,  k_sparse=32)),
    ("P2_17M",   dict(n_sdr=1200, n_mid=1200, n_top=1200, k_sparse=150)),
    ("P3_170M",  dict(n_sdr=4350, n_mid=4350, n_top=4350, k_sparse=543)),
]


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


def write_json(rows, meta=None) -> None:
    payload = {"meta": meta or {}, "points": rows}
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()


def run_point(name: str, override: dict) -> dict:
    cfg = {**BASE, **override}
    log(f"\n[START] {name}  config={ {k: cfg[k] for k in ('n_sdr','n_mid','n_top','k_sparse')} }")
    lm = PHDWordLM(text, PHDNetConfig(**cfg), seg_kwargs=SEG)
    n_params = count_params(lm.net)
    n_tok_train = len(lm.tokenize(train_trunc))
    log(f"  vocab_size={len(lm.tok):,}  measured_params={n_params:,}  train_tokens={n_tok_train:,}")

    t0 = time.perf_counter()
    lm.train_stream(train_trunc)                 # 单遍，截断训练段
    dt_train = time.perf_counter() - t0
    ms_per_token = dt_train / max(1, n_tok_train) * 1000.0

    m = lm.evaluate(eval_txt)                    # 只读评估，冻结语料后 20%
    ppl_char, bpc = float(m["ppl_char"]), float(m["bpc"])

    row = {
        "name": name,
        "n_sdr": cfg["n_sdr"], "n_mid": cfg["n_mid"], "n_top": cfg["n_top"],
        "k_sparse": cfg["k_sparse"],
        "n_params": n_params,
        "ppl_char": ppl_char,
        "bpc": bpc,
        "ms_per_token": round(ms_per_token, 4),
        "n_tok_train": n_tok_train,
        "n_tok_eval": int(m["n_tok"]),
        "oov_eval": int(m["oov"]),
        "train_sec": round(dt_train, 2),
    }
    log(f"  -> ppl_char={ppl_char:.4f}  bpc={bpc:.4f}  "
        f"ms/token={ms_per_token:.3f}  (train {dt_train:.1f}s, eval n_tok={m['n_tok']}, oov={m['oov']})")
    log(f"[DONE] {name}")
    del lm
    gc.collect()
    return row


def fit_power_law(rows):
    """log10(PPL) = log10(a) - alpha * log10(N)  → 对 log 空间线性拟合。"""
    N = np.array([r["n_params"] for r in rows], dtype=float)
    P = np.array([r["ppl_char"] for r in rows], dtype=float)
    x = np.log10(N)
    y = np.log10(P)
    slope, intercept = np.polyfit(x, y, 1)
    alpha = -slope
    a = 10.0 ** intercept
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(1e-12, ss_tot)
    return alpha, a, r2


def main() -> None:
    global text, train_trunc, eval_txt
    text = open(DOC, encoding="utf-8").read()
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split], text[split:]
    train_trunc = train_txt[:N_CHARS_TRAIN]

    # 初始化日志
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("# O7 PHD-Net 词级 LM 参数规模缩放曲线\n")
        f.write(f"# corpus={len(text)}  vocab_from=full_corpus  "
                f"train_trunc={len(train_trunc)} chars (前{N_CHARS_TRAIN})  "
                f"eval={len(eval_txt)} chars (后20%)\n")
        f.write(f"# SEG={SEG}  seed={BASE['seed']}  "
                f"pred_in_readout={BASE['pred_in_readout']}\n")
        f.write("# 参数规模=count_params 实测；幂律 PPL=a·N^(-α) 对 log10 线性回归\n\n")
        f.flush()

    write_json([], meta={"status": "running"})
    rows = []
    for name, ov in POINTS:
        try:
            row = run_point(name, ov)
            rows.append(row)
            write_json(rows, meta={"status": "running", "done": [r["name"] for r in rows]})
        except Exception as e:  # 单点失败不丢已得点
            log(f"[ERROR] {name}: {e!r}")
            import traceback
            log(traceback.format_exc())

    if len(rows) >= 2:
        alpha, a, r2 = fit_power_law(rows)
        log("\n" + "=" * 64)
        log("汇总表（实测）")
        log(f"{'点':<10}{'参数量':>14}{'ppl_char':>12}{'bpc':>9}{'ms/token':>12}")
        for r in rows:
            log(f"{r['name']:<10}{r['n_params']:>14,}{r['ppl_char']:>12.4f}"
                f"{r['bpc']:>9.4f}{r['ms_per_token']:>12.3f}")
        log("-" * 64)
        log(f"幂律拟合: PPL = {a:.4f} · N^(-{alpha:.4f})    α={alpha:.4f}    R²={r2:.4f}")
        log("=" * 64)
        write_json(rows, meta={"status": "done", "alpha": alpha, "a": a,
                               "r2": r2, "n_points": len(rows)})
    else:
        log("\n[WARN] 已完成点不足 2 个，无法拟合幂律。")
        write_json(rows, meta={"status": "incomplete", "done": [r["name"] for r in rows]})


if __name__ == "__main__":
    main()
