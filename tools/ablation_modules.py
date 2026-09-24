"""PHD-Net O6 实验：逐模块关闭/打开，看 PPL 敏感度排序。

目标：回答「哪些机制真贡献、哪些是装饰」。每个配置只改一处开关，跑完整
训练 + 评估，输出 ppl_char 与相对基线的变化率（%）。

重要事实（已读源码确认，非臆造）：
- 评测口径严格照 tests/eval_common.py：
    BASE = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
                eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
                pred_in_readout=True)
    SEG  = dict(max_len=6, min_count=5, min_entropy=1.0)
  冻结语料 datasets/eval/internal_corpus.txt（23,504 字符），训练段=前 80%，评估段=后 20%。
- 词级 LM 用法（tests/eval_tasks_lm.py 的 task1_lm）：
    lm = PHDWordLM(full_text, PHDNetConfig(**BASE), seg_kwargs=SEG)
    lm.train_stream(train_txt);  m = lm.evaluate(eval_txt)  # m["ppl_char"]
  第一个参数是建词表用的全语料文本（评测套件口径）。
- 注意：PHDWordLM.__init__ 会把 readout_softmax 强制写成 True（word_lm.py:33），
  因此「M6 readout_softmax 关闭」在词级 LM 路径下是 no-op，本脚本如实记录但
  在解读里标注为「harness 覆盖，非真实无贡献」，避免误判 M6 为装饰。
- 注意：PHDWordLM 把 n_input 改写为 2*n_sdr、n_readout 改写为词表大小。
  其余 BASE 开关（k_sparse / eta_stdp / pred_in_readout / mod_gain / gamma_wm /
  eta_hip / eta_cortex / n_infer_steps / seg）均原样透传，本脚本逐一改之。

M4a WM：n_wm_slots=0 会因 _pick_weakest 在空槽数组上 IndexError 而崩
（wm.py 已确认），故改用 gamma_wm=0.0（每步完全清空 WM，退化为瞬时上下文）。

时间：每次「训练全语料 80% + 评估」约 2–3 分钟。本脚本共 14 个配置
（含基线），总计约 30–45 分钟，后台运行，日志落 outputs/ablation_modules.log。
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.config import PHDNetConfig          # noqa: E402
from phdnet.word_lm import PHDWordLM            # noqa: E402

CORPUS = _ROOT / "datasets" / "eval" / "internal_corpus.txt"
LOG = _ROOT / "outputs" / "ablation_modules.log"

# ── 评测口径（严格照 tests/eval_common.py）─────────────────────────────
BASE = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
            eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
            pred_in_readout=True)
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)

# 截断口径：None = 用全语料 80%；若想加速可设整数（如 6000），基线与所有
# 消融项会统一截断，保证可比。
# 2026-09-23 主代理改：全语料口径实测 155.5s/配置 × 14 = 36 分钟，跨回合会被
# 系统清理 → 统一改 4,000 字符截断口径（与 scaling_curve 同口径），单配置
# ≈25-35s，14 配置回合内可完成；相对变化率结论不受影响（绝对 PPL 会升高）。
TRUNCATE_TRAIN = 4000   # 训练段统一截断到前 4000 字符


def log(msg: str) -> None:
    print(msg, flush=True)


# ── 语料加载（模块级，供 run_one 直接引用）──────────────────────────────
_TEXT = CORPUS.read_text(encoding="utf-8")
_N = len(_TEXT)
_CUT = int(_N * 0.8)
_TRAIN_TXT = _TEXT[:_CUT]
_EVAL_TXT = _TEXT[_CUT:]
if TRUNCATE_TRAIN is not None:
    _TRAIN_TXT = _TRAIN_TXT[:TRUNCATE_TRAIN]


# ── 配置清单 ───────────────────────────────────────────────────────────
# 每组： (name, override_dict, seg_override_or_None, group, 解读要点)
#   group="close" 关闭/削弱机制（若有贡献，PPL 应上升，变化率 >0）
#   group="open"  打开默认关闭机制（若有增益，PPL 应下降，变化率 <0）
CONFIGS = [
    # ── 基线 ──
    ("baseline", {}, None, "base", "完整配置（不消融），同脚本重跑作为参照"),

    # ── A 组：关闭/削弱单机制（关闭看损失）──
    ("A1_M1_k_sparse_256", dict(k_sparse=256), None, "close",
     "k_sparse 32→256：等价取消 k-WTA 稀疏竞争，M1 稀疏编码失效"),
    ("A2_M2_n_infer_steps_0", dict(n_infer_steps=0), None, "close",
     "n_infer_steps 1→0：关闭误差驱动精炼，M2 顶层表示退化为单次前向"),
    ("A3_M2_pred_in_readout_off", dict(pred_in_readout=False), None, "close",
     "pred_in_readout True→False：关闭读出三拼，M2/M3 预测特征不再进读出头"),
    ("A4_M3_eta_stdp_0", dict(eta_stdp=0.0), None, "close",
     "eta_stdp 0.02→0.0：关闭 M3 STDP 时序学习，时序关联核冻结"),
    ("A5_M4a_gamma_wm_0", dict(gamma_wm=0.0), None, "close",
     "gamma_wm 0.85→0.0：每步清空 WM，M4a 工作记忆退化为瞬时上下文"
     "（n_wm_slots=0 会崩，故用此等价削弱）"),
    ("A6_M4b_ltm_off", dict(eta_hip=0.0, eta_cortex=0.0), None, "close",
     "eta_hip=eta_cortex=0.0：印迹/巩固无效，M4b 长期记忆为空（召回退化为常向量）"),
    ("A7_M5_mod_gain_0", dict(mod_gain=0.0), None, "close",
     "mod_gain 1.5→0.0：调制器 gate≡σ(0)=0.5 恒定，M5 惊奇门控失效"),
    ("A8_M6_readout_softmax_off", dict(readout_softmax=False), None, "close",
     "readout_softmax True→False：换回线性感知器读出。"
     "⚠ harness 覆盖——PHDWordLM 强制 readout_softmax=True，本项实为 no-op，"
     "0% 变化是脚本特性而非 M6 无贡献，解读时排除"),
    ("A9_M7_seg_max_len_1", {}, dict(max_len=1, min_count=5, min_entropy=1.0), "close",
     "seg max_len 6→1：不涌现多字词，退化为字符级（词表与 token 数改变，须看 n_tok）"),

    # ── B 组：打开默认关闭机制（打开看增益）──
    ("B1_multi_modulation", dict(multi_modulation=True), None, "open",
     "multi_modulation=True：M5 升级为 ACh/NE/DA/5-HT 四通道调制，看是否增益"),
    ("B2_error_triggered_retrieval", dict(error_triggered_retrieval=True), None, "open",
     "error_triggered_retrieval=True：任务误差高时额外检索（T3.2-lite），看是否增益"),
    ("B3_adaptive_lr", dict(adaptive_lr=True), None, "open",
     "adaptive_lr=True：STDP 逐突触自适应学习率（局部二阶矩归一），看是否增益"),
    ("B4_retrieval_topk_4", dict(retrieval_topk=4), None, "open",
     "retrieval_topk=4：每步把 top-k 召回线索拼入读出（T3.2），看是否增益"),
]


def run_one(name, override, seg_override, note):
    """跑单个配置，返回 (ppl_char, n_tok, oov, dt_sec, ok, err)。"""
    t0 = time.time()
    try:
        cfg = PHDNetConfig(**{**BASE, **override})
        seg = seg_override if seg_override is not None else SEG
        lm = PHDWordLM(_TEXT, cfg, seg_kwargs=seg)
        lm.train_stream(_TRAIN_TXT)
        m = lm.evaluate(_EVAL_TXT)
        ppl = float(m["ppl_char"])
        n_tok = int(m["n_tok"])
        oov = int(m["oov"])
        return ppl, n_tok, oov, time.time() - t0, True, ""
    except Exception:
        return None, None, None, time.time() - t0, False, traceback.format_exc()


def main():
    if not CORPUS.exists():
        raise SystemExit(f"语料不存在: {CORPUS}")
    LOG.parent.mkdir(parents=True, exist_ok=True)

    log(f"# PHD-Net O6 模块消融  | 语料 {_N} 字符, 训练段 {len(_TRAIN_TXT)}, 评估段 {len(_EVAL_TXT)}")
    log(f"# 截断口径: {'全语料 80%' if TRUNCATE_TRAIN is None else f'训练段截断到前 {TRUNCATE_TRAIN} 字符'}")
    log(f"# 配置总数: {len(CONFIGS)}（含基线）")
    log("")

    results = []          # (name, group, ppl, n_tok, oov, dt, ok, err, note)
    for i, (name, ov, seg, group, note) in enumerate(CONFIGS, 1):
        log(f"[{i}/{len(CONFIGS)}] 运行 {name} ... ({group})")
        ppl, n_tok, oov, dt, ok, err = run_one(name, ov, seg, note)
        if ok:
            log(f"      -> ppl_char={ppl:.4f}  n_tok={n_tok}  oov={oov}  "
                f"耗时={dt:.1f}s  [{note}]")
        else:
            log(f"      -> 失败! 耗时={dt:.1f}s\n{err}")
        results.append((name, group, ppl, n_tok, oov, dt, ok, err, note))

    # 基线 ppl
    base_row = next(r for r in results if r[0] == "baseline")
    base_ppl = base_row[2]
    log("")
    log(f"# 基线 ppl_char（同脚本实测）: {base_ppl:.4f}")
    if base_ppl is None:
        log("# 基线失败，无法计算变化率，终止。")
        return

    # 变化率
    log("")
    log("# 逐项结果（按 |变化率| 降序）：")
    log(f"# {'配置':<28} {'组':<6} {'ppl_char':>10} {'变化率%':>9} {'n_tok':>7} {'耗时':>7}")
    table = []
    for name, group, ppl, n_tok, oov, dt, ok, err, note in results:
        if not ok or ppl is None:
            table.append((name, group, None, None, n_tok, dt, note))
            continue
        change = (ppl - base_ppl) / base_ppl * 100.0
        table.append((name, group, ppl, change, n_tok, dt, note))
    # 排序：基线排首，其余按 |change| 降序
    others = [t for t in table if t[0] != "baseline" and t[3] is not None]
    others.sort(key=lambda t: abs(t[3]), reverse=True)
    ordered = [t for t in table if t[0] == "baseline"] + others + \
              [t for t in table if t[0] != "baseline" and t[3] is None]

    for name, group, ppl, change, n_tok, dt, note in ordered:
        if ppl is None:
            log(f"# {name:<28} {group:<6} {'FAIL':>10} {'--':>9} {str(n_tok):>7} {dt:>6.1f}s")
        else:
            chg = f"{change:+.2f}" if abs(change) >= 0.005 else f"{change:+.4f}"
            log(f"# {name:<28} {group:<6} {ppl:>10.4f} {chg:>9} {n_tok:>7} {dt:>6.1f}s")

    # 敏感度排序表
    log("")
    log("# ════════════════════════════════════════════════════════════════")
    log("# 敏感度排序（按 |变化率| 降序；关闭组上升=真贡献，打开组下降=真增益）")
    log("# ════════════════════════════════════════════════════════════════")
    rank = 1
    for name, group, ppl, change, n_tok, dt, note in ordered:
        if ppl is None:
            continue
        verdict = _verdict(name, group, change, note)
        chg = f"{change:+.2f}%" if abs(change) >= 0.005 else f"{change:+.3f}%"
        log(f"# {rank:>2}. {name:<30} {chg:>9}  [{group:<5}] {verdict}")
        rank += 1

    # 结论速览
    log("")
    log("# ── 结论速览 ──")
    closes = [(n, c) for n, g, p, c, t, d, note in ordered if g == "close" and c is not None]
    opens = [(n, c) for n, g, p, c, t, d, note in ordered if g == "open" and c is not None]
    real = [n for n, c in closes if c > 1.0]            # 关闭后明显变差
    deco = [n for n, c in closes if abs(c) <= 1.0]      # 关闭后几乎无影响
    better = [n for n, c in closes if c < -1.0]         # 关闭后反而更好
    gain = [n for n, c in opens if c < -1.0]            # 打开后明显变好
    nogain = [n for n, c in opens if c >= -1.0]         # 打开后无增益
    log(f"# 真贡献（关闭→PPL 升 >1%）: {real if real else '无'}")
    log(f"# 装饰/中性（关闭→|Δ|≤1%）: {deco if deco else '无'}")
    log(f"# 关闭后反而更好（疑似过拟合/噪声）: {better if better else '无'}")
    log(f"# 打开有增益（打开→PPL 降 >1%）: {gain if gain else '无'}")
    log(f"# 打开无增益: {nogain if nogain else '无'}")
    log(f"# 总耗时: {sum(r[5] for r in results):.1f}s")


def _verdict(name, group, change, note):
    if group == "close":
        if abs(change) <= 1.0:
            if name == "A8_M6_readout_softmax_off":
                return "装饰? 实为 harness no-op，排除"
            return "装饰/中性（关闭无影响）"
        return "真贡献（关闭后 PPL 上升）" if change > 0 else "关闭后反而更好（疑似噪声/过拟合）"
    else:  # open
        if change < -1.0:
            return "有增益（打开后 PPL 下降）"
        if change > 1.0:
            return "有损（打开后 PPL 上升）"
        return "中性（打开无明显变化）"


if __name__ == "__main__":
    main()
