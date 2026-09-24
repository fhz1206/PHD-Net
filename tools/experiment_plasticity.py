"""O2 实验：在什么条件下可以让 eta_pc>0（表征可塑）而不发生灾难性遗忘？

只新建脚本，不修改 phdnet/ 与 tests/ 下任何文件。

核心症结（O2）：表征可塑性（eta_pc>0, eta_oja>0）在本项目历史四轮实验全负，
根因是「表征漂移 × 恒定读出学习率 → 灾难性遗忘」（PPL 发散到 4.7e4 量级）。
当前配方永久冻结表征（eta_pc=0）。本脚本实测文档给出的两条对症思路：
  ① 生成式回放稳定读出（readout_replay / sleep 生成式回放）；
  ② 发育期表征巩固后彻底冻结读出并重训练（pc_dev_steps + eta_readout_anneal/floor）。

评测口径（严格照用）：冻结语料 datasets/eval/internal_corpus.txt（23,504 字符），
训练段 = 前 80% 的**前 4000 字符**，评估段 = 后 20%（4,701 字符）。
词级 LM：PHDWordLM，BASE + SEG 同 eval_common.py / 任务说明。

用法：
  python -X utf8 tools/experiment_plasticity.py                 # 全跑
  python -X utf8 tools/experiment_plasticity.py --only freeze,plastic_naive,...
  python -X utf8 tools/experiment_plasticity.py --bwt            # 可选 BWT 对比（见下）
日志：outputs/experiment_plasticity.log（脚本内追加写，调用方重定向亦可）。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.config import PHDNetConfig
from phdnet.word_lm import PHDWordLM

CORPUS = _ROOT / "datasets" / "eval" / "internal_corpus.txt"
LOG = _ROOT / "outputs" / "experiment_plasticity.log"

# 与 eval_common.BASE / 任务说明完全一致
BASE = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
            eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
            pred_in_readout=True)
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)

# 必测 + 自选变体。顺序即编号（1..10）。
# 前 6 个（freeze..plastic_dev_replay）本回合先跑；其余交主代理分批。
CONFIGS = [
    # 1 对照：冻结表征
    ("freeze", dict(eta_pc=0.0, eta_oja=0.0)),
    # 2 朴素可塑：复现历史失败
    ("plastic_naive", dict(eta_pc=0.02, eta_oja=0.02)),
    # 3 + PC 稳态突触缩放
    ("plastic_homeo", dict(eta_pc=0.02, eta_oja=0.02, homeostasis=True)),
    # 4 + 发育期后冻结 PC（前 2000 步可塑）
    ("plastic_homeo_dev", dict(eta_pc=0.02, eta_oja=0.02, homeostasis=True,
                               pc_dev_steps=2000)),
    # 5 生成式回放稳定读出
    ("plastic_replay", dict(eta_pc=0.02, eta_oja=0.02,
                            readout_replay=True, replay_every=4, replay_eta=0.5)),
    # 6 4+5 组合
    ("plastic_dev_replay", dict(eta_pc=0.02, eta_oja=0.02, homeostasis=True,
                                pc_dev_steps=2000,
                                readout_replay=True, replay_every=4, replay_eta=0.5)),
    # 7 思路②完整形态
    ("plastic_full", dict(eta_pc=0.02, eta_oja=0.02, homeostasis=True,
                          pc_dev_steps=2000, readout_replay=True,
                          eta_readout_anneal=0.9997, eta_readout_floor=0.004)),
    # 9 自选-A：自动发育调度（表征稳定后 PC 自动衰减，比固定 pc_dev_steps 更自适应）
    ("plastic_auto_dev", dict(eta_pc=0.02, eta_oja=0.02, auto_development=True)),
    # 10 自选-B：逐神经元目标发放率（内在可塑性，细粒度稳态补 k-WTA 全局稀疏）
    ("plastic_target_rate", dict(eta_pc=0.02, eta_oja=0.02, neuron_target_rate=True)),
    # 11 自选-C：STDP 出行权稳态缩放（防 STDP 侧发散，隔离表征漂移外的第二发散源）
    ("plastic_stdp_homeo", dict(eta_pc=0.02, eta_oja=0.02, stdp_homeostasis=True)),
]


def build_splits():
    """训练段 = 前 80% 的前 4000 字符；评估段 = 后 20%。"""
    full = CORPUS.read_text(encoding="utf-8")
    n = len(full)
    train_pool = full[: int(n * 0.8)]
    train_txt = train_pool[:4000]
    eval_txt = full[int(n * 0.8):]
    return full, train_txt, eval_txt


def run_cfg(name, overrides, full, train_txt, eval_txt, freeze_ppl):
    t0 = time.time()
    d = dict(BASE)
    d.update(overrides)
    cfg = PHDNetConfig(**d)
    # 词表以全语料构建（最大化覆盖，与任务说明示例一致：PHDWordLM(text, ...) 用全语料）
    lm = PHDWordLM(full, cfg, seg_kwargs=SEG)
    lm.train_stream(train_txt)
    m = lm.evaluate(eval_txt)
    ppl = float(m["ppl_char"])
    dt = time.time() - t0
    diverged = (not math.isfinite(ppl)) or (ppl > 1e4)
    if freeze_ppl and math.isfinite(freeze_ppl):
        rel = (ppl / freeze_ppl - 1.0) * 100.0  # 相对 freeze 变化率 %
    else:
        rel = float("nan")
    return {
        "name": name, "ppl": ppl, "diverged": diverged, "rel": rel,
        "dt": dt, "oov_rate": float(m["oov_rate"]), "n_tok": int(m["n_tok"]),
    }


def fmt_row(r):
    if r is None:
        return ""
    ppl = r["ppl"]
    ppl_s = "NaN" if not math.isfinite(ppl) else (f"{ppl:.2f}" if ppl < 1e4 else f"{ppl:.2e}")
    div = "发散" if r["diverged"] else "OK"
    rel = "—" if (r["rel"] != r["rel"]) else f"{r['rel']:+.1f}%"
    return (f"{r['name']:<20} ppl_char={ppl_s:<12} {div:<4} "
            f"Δfreeze={rel:<8} t={r['dt']:.1f}s oov={r['oov_rate']*100:.1f}%")


def run_bwt(overrides):
    """可选：复用 task4_continual 两任务协议测「遗忘」(BWT=accA2-accA1)。

    自包含实现（不依赖 torch/transformer 对照），字符级 PHDNetLM 上 A->B 顺序训练，
    比较训练 B 后任务 A 的保持率。BWT 越接近 0 越无遗忘，越负遗忘越严重。
    """
    from phdnet.lm import PHDNetLM
    vocab = "xyzpqr"
    A = "xyz" * 1000
    B = "pqr" * 1000
    d = dict(BASE)
    d.update(overrides)
    cfg = PHDNetConfig(**d)
    lm = PHDNetLM(vocab, cfg)

    def acc(text):
        hit = tot = 0
        for i in range(min(600, len(text) - 1)):
            y = lm.net.step(lm.tok.encode(text[i]), learn=False)["y"]
            if int(np.argmax(y)) == lm.tok.stoi[text[i + 1]]:
                hit += 1
            tot += 1
        return hit / max(1, tot)

    lm.train_stream(A)
    a1 = acc(A)
    lm.train_stream(B)
    a2, b = acc(A), acc(B)
    return {"A1": a1, "A2": a2, "B": b, "BWT": a2 - a1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="逗号分隔的配置名；省略=全跑")
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--bwt", action="store_true",
                    help="可选：对 freeze 与 plastic_full 各跑一次 BWT 对比")
    args = ap.parse_args()

    chosen = [(n, o) for (n, o) in CONFIGS
              if not args.only or n in set(args.only.split(","))]

    full, train_txt, eval_txt = build_splits()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as lf:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lf.write(f"\n=== O2 实验开始 {stamp} 配置数={len(chosen)} "
                 f"only='{args.only}' ===\n")
        lf.flush()

        print(f"[O2] 语料 {len(full)} 字符 | 训练 {len(train_txt)} | "
              f"评估 {len(eval_txt)}", flush=True)

        results = {}
        freeze_ppl = results.get("freeze", {}).get("ppl")
        for name, ov in chosen:
            print(f"[O2] 运行 {name} ...", end=" ", flush=True)
            try:
                r = run_cfg(name, ov, full, train_txt, eval_txt, freeze_ppl)
                results[name] = r
                if name == "freeze":
                    freeze_ppl = r["ppl"]
            except Exception as e:  # 单配置崩了不中断其余
                r = {"name": name, "ppl": float("nan"), "diverged": True,
                     "rel": float("nan"), "dt": 0.0, "oov_rate": 0.0, "n_tok": 0,
                     "error": repr(e)}
                results[name] = r
                print(f"ERROR {e!r}", flush=True)
            line = fmt_row(r)
            status = "ERROR" if "error" in r else ("发散" if r["diverged"] else "OK")
            print(line, flush=True)
            lf.write(f"{stamp} {line} {'ERROR' if 'error' in r else ''}\n")
            lf.flush()

        # ── 可选 BWT ───────────────────────────────────────────────
        if args.bwt:
            for name in ("freeze", "plastic_full"):
                ov = dict(next(o for n, o in CONFIGS if n == name))
                try:
                    bw = run_bwt(ov)
                    print(f"[BWT] {name}: {bw}", flush=True)
                    lf.write(f"{stamp} BWT {name}: {bw}\n")
                except Exception as e:
                    print(f"[BWT] {name} ERROR {e!r}", flush=True)
                    lf.write(f"{stamp} BWT {name} ERROR {e!r}\n")
                lf.flush()

        # ── 汇总表 + 结论 ─────────────────────────────────────────
        print("\n===== 汇总表 =====", flush=True)
        header = f"{'配置':<20}{'ppl_char':<14}{'状态':<6}{'Δfreeze':<10}{'耗时':<8}{'oov'}"
        print(header, flush=True)
        lf.write(header + "\n")
        for name, _ in chosen:
            r = results.get(name)
            if r is None:
                continue
            ppl = r["ppl"]
            ppl_s = ("NaN" if not math.isfinite(ppl)
                     else (f"{ppl:.2f}" if ppl < 1e4 else f"{ppl:.2e}"))
            div = "ERROR" if "error" in r else ("发散" if r["diverged"] else "OK")
            rel = "—" if (r["rel"] != r["rel"]) else f"{r['rel']:+.1f}%"
            row = (f"{name:<20}{ppl_s:<14}{div:<6}{rel:<10}"
                   f"{r['dt']:.1f}s{'':<3}{r['oov_rate']*100:.1f}%")
            print(row, flush=True)
            lf.write(row + "\n")

        # ── 结论（数据驱动）──────────────────────────────────────
        fz = results.get("freeze", {}).get("ppl")
        plastic = [(n, results[n]) for n, _ in chosen
                   if n != "freeze" and n in results
                   and "error" not in results[n]]
        ok = [(n, r) for n, r in plastic if (not r["diverged"])
              and math.isfinite(r["ppl"]) and fz and r["ppl"] < 2.0 * fz]
        conclusion = build_conclusion(fz, plastic, ok)
        print("\n===== 结论 =====", flush=True)
        print(conclusion, flush=True)
        lf.write("\n结论:\n" + conclusion + "\n")
        lf.flush()


def build_conclusion(fz, plastic, ok):
    lines = []
    if fz and math.isfinite(fz):
        lines.append(f"freeze 自检 ppl_char={fz:.2f}（参照 ~97.26，"
                     f"{'口径一致' if abs(fz-97.26)<15 else '偏差较大，见下方说明'}）。")
    else:
        lines.append("freeze 自检未得到有限值，脚本口径需检查。")
    if not plastic:
        return "\n".join(lines) + "\n（本批无可塑配置。）"
    if ok:
        best = min(ok, key=lambda x: x[1]["ppl"])
        lines.append(f"【部分可行】条件 {best[0]} 下表征可塑未发散且 "
                     f"ppl_char={best[1]['ppl']:.2f}（Δfreeze={best[1]['rel']:+.1f}%），"
                     f"相对 freeze 恶化 <2×。可行条件："
                     + "、".join(n for n, _ in ok) + "。")
        lines.append("代价：需叠加稳态/回放/发育期冻结等稳定机制，且 PPL 仍高于冻结基线，"
                     "即「可塑但更贵、略差」。")
    else:
        lines.append("【仍不可行】本批所有 eta_pc>0 配置要么发散（ppl>1e4/NaN）、"
                     "要么 ppl 远超 freeze 2× 以上。说明仅靠本批机制不足以解锁表征可塑——"
                     "根因「表征漂移 × 读出率」未被充分压制。")
        lines.append("可行方向（待剩余配置验证）：发育期彻底巩固后冻结 + 读出退火下限 "
                     "（plastic_full）、生成式回放（sleep n_replay）+ 自动发育门控 "
                     "（auto_development）等组合。")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
