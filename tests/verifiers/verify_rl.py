"""PHD-Net RL 训练闭环验证（P27）。

玩具任务：奖励 = 续写里「目标词」出现的次数。可验证 RL 是否真的在提升策略
（同一批 prompt，训练前后的平均奖励应上升）。

    python tests/verifiers/verify_rl.py

覆盖：
  A 数据集读取（JSONL，reward 内联 / 缺失时报错）
  B rollout：形状、logprob 合法、状态守卫（训练前后 step_count 不变）
  C REINFORCE 更新：权重确实变化、统计字段齐备
  D 学习有效性：玩具奖励上升（策略在朝奖励方向走）
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_ROOT), str(_ROOT / "train_1b")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                          # noqa: E402

_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" —— {detail}" if detail else ""), flush=True)
    if not ok:
        _FAILURES.append(name)


def _toy_lm(seed: int = 0):
    """小 LM：smoke 预设 + 玩具词表（保证有可生成的 token）。"""
    from config_1b import SEG_KWARGS
    from phdnet.word_encoder import WordSegmenter
    from phdnet.word_lm import PHDWordLM
    from phdnet.config import PHDNetConfig

    text = ("自然语言处理很有趣。机器学习需要数据。深度学习是机器学习的一部分。"
            "强化学习通过奖励学习策略。神经网络处理序列。"
            "中文分词是自然语言处理的基础。") * 40
    seg = WordSegmenter(text, **SEG_KWARGS)
    toks = sorted(set(seg.tokenize(text)))[:400] or ["的", "是", "了"]
    cfg = PHDNetConfig()
    tok = type("T", (), {})()
    from phdnet.word_encoder import WordTokenizer
    tk = WordTokenizer.__new__(WordTokenizer)
    tk.seg = seg
    tk.tokens = toks
    tk.stoi = {t: i for i, t in enumerate(toks)}
    tk.n_sdr, tk.n_active, tk.seed = cfg.n_sdr, cfg.k_sparse, cfg.seed
    from ckpt_1b import _rebuild_sdrs
    _rebuild_sdrs(tk, toks)
    lm = PHDWordLM(text, cfg, tokenizer=tk)
    return lm


def main() -> None:
    import os
    os.chdir(_ROOT)
    from phdnet.rl import (REINFORCETrainer, load_prompts, reward_from_dataset,
                           rollout, state_guard)

    lm = _toy_lm()
    target = "强化"          # 玩具目标词

    def reward_fn(prompt, completion, **kw):
        # 稠密 + 稀疏混合奖励：字符覆盖率给出稠密梯度（让策略有方向可学），
        # 整词命中给出额外加分。纯稀疏奖励（count==0 恒成立）会让优势恒为 0。
        chars = set(target)
        soft = sum(1 for c in chars if c in completion) / len(chars)
        return float(completion.count(target)) + soft

    prompts = [{"prompt": "机器学习"}, {"prompt": "自然语言"},
               {"prompt": "神经网络"}, {"prompt": "深度学习"}]

    # ── A 数据集 ──
    print("[A] 提示集 / 奖励来源")
    tmp = Path(tempfile.mkdtemp(prefix="rl_"))
    jl = tmp / "prompts.jsonl"
    with jl.open("w", encoding="utf-8") as f:
        for i, it in enumerate(prompts):
            f.write(json.dumps({**it, "reward": 1.0 if i % 2 == 0 else 0.0},
                               ensure_ascii=False) + "\n")
    items = load_prompts(jl)
    check("A JSONL 条数", len(items) == 4, f"{len(items)}")
    check("A prompt 字段", items[0]["prompt"] == "机器学习",
          repr(items[0]["prompt"]))
    check("A 内联 reward 解析", reward_from_dataset(items[0]) == 1.0,
          f"{reward_from_dataset(items[0])}")
    check("A 缺 reward → None", reward_from_dataset({"prompt": "x"}) is None)
    bad = tmp / "bad.jsonl"
    bad.write_text('{"no_prompt": 1}\n', encoding="utf-8")
    try:
        load_prompts(bad)
        check("A 缺 prompt 报错", False)
    except ValueError:
        check("A 缺 prompt 报错", True, "ValueError")

    # ── B rollout ──
    print("[B] rollout（采样不学习 + 状态守卫）")
    step0 = int(lm.net.step_count)
    r = rollout(lm, prompts[0]["prompt"], n_tokens=12, tau=0.9, topk=5, seed=1)
    check("B 轨迹长度", r.n == 12, f"tokens={r.n}")
    check("B logprob 合法（≤0 且有限）",
          all(np.isfinite(lp) and lp <= 1e-9 for lp in r.logprobs),
          f"min={min(r.logprobs):.2f} max={max(r.logprobs):.2f}")
    check("B 文本拼接一致", r.text == "".join(r.tokens))

    with state_guard(lm.net):
        before = int(lm.net.step_count)
        rollout(lm, prompts[1]["prompt"], n_tokens=8, seed=2)
        during = int(lm.net.step_count)
    after = int(lm.net.step_count)
    check("B 状态守卫恢复 step_count", before == after,
          f"{before} -> {during} -> {after}")

    # ── C 更新 ──
    print("[C] REINFORCE 更新")
    tr = REINFORCETrainer(lm, lr=0.02, reward_fn=reward_fn)
    W_before = lm.net.readout.W.copy()
    rs = tr.collect(prompts, n_tokens=10, tau=0.9, topk=5, seed=7)
    check("C 采样打分", all(x.reward >= 0 for x in rs),
          f"rewards={[x.reward for x in rs]}")
    st = tr.update(rs)
    check("C 更新步数 > 0", st["steps"] > 0, f"steps={st['steps']}")
    check("C 统计字段齐备",
          {"reward", "baseline", "adv_abs", "nll"} <= set(st),
          f"{ {k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()} }")
    dW = float(np.abs(lm.net.readout.W - W_before).max())
    check("C 权重确实变化", dW > 0, f"max|ΔW|={dW:.3e}")

    # ── D 学习有效性 ──
    print("[D] 学习有效性（玩具奖励应上升）")
    tr2 = REINFORCETrainer(lm, lr=0.05, reward_fn=reward_fn,
                           normalize_adv=True, baseline_decay=0.0)
    # 采样噪声大（每条轨迹只 12 token）→ 用更多提示 + 多轮均值判定
    many = [{"prompt": p} for p in
            ("机器学习", "自然语言", "神经网络", "深度学习", "分词", "语料",
             "训练", "模型", "算法", "数据") for _ in range(2)]
    iters = 12
    trace = []
    for it in range(iters):
        rs = tr2.collect(many, n_tokens=16, tau=1.0, seed=1000 + it)
        tr2.update(rs)
        trace.append(float(np.mean([x.reward for x in rs])))
    r1 = [x.reward for x in tr2.collect(many, n_tokens=16, tau=1.0, seed=999)]
    print(f"    奖励轨迹: {[round(v, 3) for v in trace]}")
    head = float(np.mean(trace[:3]))
    tail = float(np.mean(trace[-3:]))
    check("D 训练后期奖励 > 早期（3 轮均值）", tail > head,
          f"前三 {head:.3f} → 后三 {tail:.3f}")
    check("D 评估奖励为正", float(np.mean(r1)) > 0,
          f"eval reward={np.mean(r1):.3f} (目标 {target!r})")
    st = tr2.stats()
    check("D stats 可读", st["reward_mean"] is not None
          and st["nll_mean"] is not None,
          f"reward_mean={st['reward_mean']:.3f} nll={st['nll_mean']:.3f}")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("-" * 70)
    if _FAILURES:
        print(f"结果：{len(_FAILURES)} 例 FAIL → {_FAILURES}")
        sys.exit(1)
    print("结果：全部 PASS（RL 闭环可用：数据集 → rollout → 奖励 → 策略梯度更新）")


if __name__ == "__main__":
    main()
