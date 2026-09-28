"""泛化能力探针 —— 判断 PHD-Net 词级 LM 的分布外泛化（Request H / 任务 #77）。

四轴：
  A 域内（in-domain）：internal_corpus 80/20 held-out，同 eval_suite 口径
     （对照冻结基线 ppl_char 78.16 @23,504）。
  B 近域（near-domain）：docs/ 其余三份文档 + README.md —— 同项目领域、
     训练未见过（架构设计文档是语料来源，故排除本身）。
  C 远域（far-domain）：eval_corpus/ood_wiki.txt —— 中文维基 26 篇
     （ModelScope Range 抽取，与内部文档完全异构）。
  D 长度外推：copy 任务 n=8→16 已有冻结数据（eval_suite），此处只引用不重跑。

诚实口径（三条，全部落进输出）：
  1. PHDWordLM.evaluate 对 OOV 步**跳过 NLL 但字符照计**，高 OOV 文本的
     ppl_char 被系统性压低（分母虚大）→ 同时报告 ppl_char_pen：每个 OOV 步
     按 -log(1/V)（词表均匀分布）补记 NLL 的乐观惩罚版。
  2. WordNGram 对照给两版：
       full = 全 token 流、OOV 走独立 unk 类（加 1 平滑，自带 OOV 处理）；
       step = 与 PHD-Net 完全相同的有效步集（cur/nxt 均在词表）+ 相同字符
              分母 + 相同 prev 语义（仅隔离词表效应）。
     step 版排除 j=0（prev=None 无法构 n-gram 上下文）与 OOV 步；PHD 数值
     仍取 evaluate() 原样（含 j=0），单步差异可忽略。
  3. 网络内部状态跨目标延续（模型无状态重置接口；token 级 prev 每次评估
     置空）——影响限于每个目标开头数步；目标顺序固定：域内→近域→远域。

判定（本探针操作定义，非外部标准）：
  inflation = ppl_char_pen(target) / ppl_char_pen(域内)
  ≤2× 良好 ｜ 2–4× 中等 ｜ >4× 不足；另看远域 ratio（vs ngram full）>1
  为附加不足信号。

运行：python tests/demo_gen.py（后台 ~15–30 分钟）→ 建议重定向 outputs/test/demo_gen.log
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根

from eval_common import BASE, DOC, SEG

from phdnet.config import PHDNetConfig
from phdnet.ngram import WordNGram
from phdnet.word_lm import PHDWordLM

ROOT = Path(__file__).resolve().parents[1]
# ⚠ 治理约定（fhz 2026-09-26）：架构介绍文档（docs/*.md）不是数据集——
# 近域探针改用真实语料的跨来源样本（sft 训练域之外的同域文本）。
NEAR = []
FAR = ("远域·维基百科OOD", ROOT / "eval_corpus" / "ood_wiki.txt")


def ngram_step_aligned(ng: WordNGram, toks: list[str], stoi: dict) -> tuple[float, int]:
    """与 PHD-Net evaluate() 逐步对齐的 NGram NLL（加 1 平滑同 WordNGram.nll）。

    prev 语义复刻 evaluate()：仅在该步有效（cur/nxt 均在词表）后更新为 cur。
    返回 (总 NLL, 字符数)，字符只累计有效步的 nxt 长度。j=0 无 prev 跳过。
    """
    Vn = len(ng.stoi) + 1                      # +1 = unk 类（与 WordNGram.nll 一致）
    total, nch = 0.0, 0
    prev: str | None = None
    for j in range(len(toks) - 1):
        cur, nxt = toks[j], toks[j + 1]
        if cur in stoi and nxt in stoi and prev is not None:
            ctx = (ng.stoi.get(prev, ng.unk), ng.stoi.get(cur, ng.unk))
            cc = ng.counts.get(ctx, {})
            total += -math.log((cc.get(ng.stoi[nxt], 0) + 1)
                               / (sum(cc.values()) + Vn))
            nch += len(nxt)
        if cur in stoi and nxt in stoi:
            prev = cur                          # 仅有效步后推进（同 evaluate）
    return total, nch


def probe(lm: PHDWordLM, ng: WordNGram, name: str, txt: str, V: int) -> dict:
    """单目标全量指标。"""
    t0 = time.perf_counter()
    toks = lm.tokenize(txt)
    m = lm.evaluate(txt)
    dt = time.perf_counter() - t0

    # 由 ppl 反解总 NLL 与字符分母（evaluate 未直接返回 n_chars）
    total_nll = math.log(m["ppl_token"]) * m["n_tok"] if m["n_tok"] else 0.0
    if total_nll > 0 and m["ppl_char"] > 1.0:
        n_chars = total_nll / math.log(m["ppl_char"])
    else:
        n_chars = sum(len(t) for t in toks[1:])
    nll_pen = total_nll + m["oov"] * math.log(V)      # OOV 步按词表均匀补记
    ppl_pen = math.exp(nll_pen / max(1.0, n_chars))
    bpc_pen = nll_pen / max(1.0, n_chars) / math.log(2)

    # NGram full：全 token 流（n=2 即 2-token 上下文），OOV→unk 类
    ng_full = ng.nll(toks) * (len(toks) - 2)
    ng_full_chars = sum(len(t) for t in toks[2:])
    ppl_ng_full = math.exp(ng_full / max(1.0, ng_full_chars))

    # NGram step：与 PHD 相同有效步集 + 相同字符分母
    ng_step_nll, ng_step_chars = ngram_step_aligned(ng, toks, lm.tok.stoi)
    ppl_ng_step = math.exp(ng_step_nll / max(1.0, ng_step_chars))

    return {
        "name": name, "chars": len(txt), "tokens": len(toks),
        "n_tok": m["n_tok"], "oov": m["oov"],
        "oov_rate": m["oov_rate"],
        "ppl_char": m["ppl_char"], "ppl_char_pen": ppl_pen, "bpc_pen": bpc_pen,
        "ppl_ng_full": ppl_ng_full, "ppl_ng_step": ppl_ng_step,
        "ratio_pen_vs_ngfull": ppl_pen / ppl_ng_full,
        "sec": dt,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="泛化能力探针（域内/近域/远域）")
    ap.add_argument("--strength", action="store_true",
                    help="加强组：启用 T3.4 内容寻址 WM + ② 回放稳定读出（均默认关闭开关）")
    args = ap.parse_args(argv)
    base = dict(BASE)
    label = "BASE"
    if args.strength:
        base.update(wm_content_address=True, readout_replay=True)
        label = "BASE + T3.4 内容寻址 + ② 回放稳定读出"

    print("=" * 78)
    print(f"泛化能力探针：域内 / 近域 / 远域（词级 LM，冻结语料 80/20 训练一次）  配置: {label}")
    print("=" * 78)
    with open(DOC, encoding="utf-8") as f:
        text = f.read()
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split], text[split:]
    print(f"语料 {len(text):,} 字符（训练 {len(train_txt):,} / 域内评估 {len(eval_txt):,}）")

    lm = PHDWordLM(train_txt, PHDNetConfig(**base), seg_kwargs=SEG)
    V = len(lm.tok)
    toks_train = lm.tokenize(train_txt)
    ng = WordNGram(toks_train, n=2)
    print(f"词表 {V}（训练段构建，无评估泄漏）  训练 token {len(toks_train):,}  "
          f"压缩比 {len(train_txt) / max(1, len(toks_train)):.2f}×")

    t0 = time.perf_counter()
    lm.train_stream(train_txt)
    print(f"训练完成 {time.perf_counter() - t0:.1f}s\n")

    targets: list[tuple[str, str]] = [("域内·held-out 80/20", eval_txt)]
    for name, p in NEAR:
        targets.append((name, p.read_text(encoding="utf-8")))
    targets.append((FAR[0], FAR[1].read_text(encoding="utf-8")))

    rows = []
    for name, txt in targets:
        r = probe(lm, ng, name, txt, V)
        rows.append(r)
        print(f"[{r['name']}]  {r['chars']:,} 字符  {r['tokens']:,} token  "
              f"有效步 {r['n_tok']:,}  OOV {r['oov_rate'] * 100:.1f}%  ({r['sec']:.0f}s)")
        print(f"    PHD  ppl_char={r['ppl_char']:9.2f}  ppl_pen={r['ppl_char_pen']:9.2f}  "
              f"bpc_pen={r['bpc_pen']:.3f}")
        print(f"    NGram full={r['ppl_ng_full']:9.2f}  step对齐={r['ppl_ng_step']:9.2f}  "
              f"PHD_pen/full={r['ratio_pen_vs_ngfull']:.3f}")

    # ---- 判定 ----
    base = rows[0]["ppl_char_pen"]
    print("\n" + "=" * 78)
    print("泛化判定（inflation = ppl_pen(target)/ppl_pen(域内)；≤2×良好｜2–4×中等｜>4×不足）")
    print("=" * 78)
    results = []
    for r in rows[1:]:
        infl = r["ppl_char_pen"] / base
        level = "良好" if infl <= 2 else ("中等" if infl <= 4 else "不足")
        extra = "；且劣于统计基线(full)" if r["ratio_pen_vs_ngfull"] > 1 else ""
        print(f"  {r['name']:<14} inflation={infl:5.2f}×  → {level}{extra}")
        results.append({"name": r["name"], "inflation": infl, "level": level,
                        "ratio_vs_ngfull": r["ratio_pen_vs_ngfull"]})
    worst = max((x["inflation"] for x in results), default=1.0)
    verdict = ("泛化良好" if worst <= 2
               else "泛化中等（近/远域可控膨胀）" if worst <= 4 else "泛化不足，需加强")
    print(f"\n总判定: {verdict}（最差 inflation {worst:.2f}×）")
    print("长度外推（引用 eval_suite 冻结数据）: copy n=8→16 PHD-Net ~10.0–11.0% / "
          "TF ~5.7–9.7%，随机 8.3% —— 双方均接近随机，长度外推弱。")

    with open(ROOT / "outputs" / "demo_gen_result.json", "w", encoding="utf-8") as f:
        json.dump({"config": label, "rows": rows, "results": results,
                   "verdict": verdict, "worst_inflation": worst},
                  f, ensure_ascii=False, indent=1)
    print("结果已写入 outputs/test/demo_gen_result.json")


if __name__ == "__main__":
    main()
