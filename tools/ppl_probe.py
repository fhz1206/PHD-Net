"""PPL 评估机制 v2：**固定探针集 + 配对比较**（P146）。

为什么必须重做
================================================================================
fhz 指出：训练时 PPL 会因突触生长而**明显波动**。用真实日志核实（1b 档
`train_1b_1b_pretrain_20261002-130316.log` 的 40 个采样点）：

  范围          20,652 ~ 51,119     → **2.5×**
  变异系数      **19.6%**
  相邻变化中位  **7.0%**（最大 60.1%）
  首→尾趋势    −2.8%
  **信噪比**    **0.40**             ← 趋势被波动完全淹没

**后果**：本轮所有「跑 PPL A/B 决定」的指示（fp16 的 g 会不会伤学习、einsum
非逐位要不要紧、幂律有没有用…）在旧机制下**不可执行** —— 任何 <20% 的真实
差异都会被自然波动淹没。

旧机制的三个结构性问题
--------------------------------------------------------------------------------
1. **窗口是训练流上的最近 N 个**（`seg_nll[-log_every:]`）→ ① 覆盖的**语料片段
   每次都不同**（远程流式 + 混合域），词表/句式/长度分布的差异直接进 PPL，
   这是**混淆变量**而非噪声；② 连续 token 的 nll **逐步强相关**（共享语境、
   共享刚被 STDP/Oja 改过的权重），2000 个样本的**有效样本数远小于 2000**，
   均值标准误被严重低估。
2. **`exp(mean(nll))`（几何平均）对高频词敏感** → 突触生长/修剪改变 nll 的
   *分布形状*，而分布形状变化与「学得更好」无关。
3. **单点数值**无法区分「变好」与「碰巧低」。

v2 的做法
--------------------------------------------------------------------------------
**核心原则：比较必须建立在「同一批 token」上。**
  ① **固定探针集**：每 N 步在**同一份冻结文本**上跑**只读前向**（不训练、
    不更新权重、不写状态）→ 消除语料这个混淆变量。
  ② **配对比较**：探针 PPL 的**差值**（B−A）与「训练流 PPL 的差值」并列输出。
    探针差值的**噪声由探针集大小决定**（可加大到近乎零噪声），
    训练流差值则是「真信号 + 波动」的混合 → 两者一比就知道信号是否存在。
  ③ **三档输出**：
       - `probe_ppl`   固定集PPL（**判A/B 用这个**）
       - `train_ppl`  训练流 PPL（**看趋势**，不用于判A/B）
       - `delta`      B−A（配对差，噪声可控）
  ④ **可判定性**：直接输出「该差值是否超出噪声带」→ **A/B 结论可执行**。
  ⑤ **不污染训练**：探针是纯前向，`readonly=True` 路径（`model.step(learn=False)`），
     不改权重、不进 `_prev_*` 状态、不写 LTM/回放缓冲 → 与训练完全解耦。

⚠ **局限（必须知道）**
  - 探针集用的是 `eval_corpus/internal_corpus.txt`，它**也在训练语料里**
    （P100曾记录过）→ 探针不是严格 held-out。它消除的是「**逐次窗口的语料
    差异**」这个混淆变量，**不是**「训练/评测数据泄漏」。
  - 探针 PPL **绝对值**与训练流 PPL 不可比（不同文本），**只能用差值**。
  - 探针频率不能太高（每次要跑一遍前向）；默认 5000 步一次。
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
# ⚠ `train` 是**目录**不是包 → 必须把 `train/` 本身加进 sys.path，
#   否则 `import corpus_stream` 会失败（写 `import train.corpus_stream` 同样失败）
for _p in (str(_ROOT), str(_ROOT / "train")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── 探针集：冻结文本 + 固定偏移（**每次跑完全相同的 token 序列**）──────────
_PROBE_SRC = None
_PROBE_CACHE: dict = {}


def load_probe_text() -> str:
    """加载**冻结的**探针文本（一次性，之后缓存）。

    ⚠ 关键：文本与偏移都**固定不随训练步数变**，否则「探针 PPL 的下降」里
    会混入「换了语料」的成分 —— 那正是旧机制最大的混淆变量。
    ⚠ tokenizer **不用**这里建：探针必须用 `lm.tok`（同一份词表），
    否则量的不是同一个模型（P146 实测踩过：自建词表 512 vs LM 的 2610）。
    """
    global _PROBE_SRC
    if _PROBE_SRC is None:
        p = _ROOT / "eval_corpus" / "internal_corpus.txt"
        _PROBE_SRC = p.read_text(encoding="utf-8")
    return _PROBE_SRC


def probe_nll(lm, n_tokens: int = 2048) -> dict:
    """在**固定探针集**上跑只读前向，返回 {ppl, nll, n_tok, ms}。

    ⚠ `readonly=True`（`model.step(learn=False)`）→ 不更新权重、不进 `_prev_*`、
      不写 LTM/回放缓冲 → 对训练**零污染**。
    """
    # ⚠⚠ **P146 修正（设计缺陷）**：原实现自己建一个 WordTokenizer，
    # 那与 `lm.tok` **不是同一份词表**（本机实测 512 vs 2610）→
    # `target` 维度不匹配直接抛ValueError。
    #   → 探针**必须用 LM 自己的 tokenizer**，否则「探针」量的不是同一个模型。
    #     `lm` 已构造好，其 `tok` 词表固定；我们只用**同一份 tokenizer**
    #     去切**同一份冻结文本**。
    tok = getattr(lm, "tok", None)
    if tok is None:
        raise ValueError("lm 没有 tok 属性——探针必须用 LM 的 tokenizer")
    src = _PROBE_SRC
    if src is None:
        raise ValueError("探针文本未加载（先调 get_probe_tokens 或置 _PROBE_SRC）")
    toks = tok.seg.tokenize(src)[:n_tokens + 1]
    nlls = []
    t0 = time.perf_counter()
    with _suppress_stdout():
        for i in range(len(toks) - 1):
            a, b = toks[i], toks[i + 1]
            if a not in tok.stoi or b not in tok.stoi:
                continue
            x = tok.encode_composite(a, b)                    # noqa: F501
            y = tok.onehot(tok.stoi[b])
            d = lm.net.step(x, target=y, readonly=True,
                            target_idx=int(tok.stoi[b]))
            nlls.append(float(d["nll"]))
    dt = (time.perf_counter() - t0) * 1e3
    if not nlls:
        return {"ppl": float("nan"), "nll": float("nan"),
                "n_tok": 0, "ms": dt}
    m = float(np.mean(nlls))
    return {"ppl": float(np.exp(m)), "nll": m, "n_tok": len(nlls), "ms": dt}


class _suppress_stdout:
    def __enter__(self):
        import contextlib
        self._c = contextlib.redirect_stdout(io.StringIO())
        return self._c.__enter__()

    def __exit__(self, *a):
        return self._c.__exit__(*a)


# ── 配对比较：把「噪声」显式算出来，而不是靠肉眼看 ────────────────────────
def paired_delta(a_runs: list[float], b_runs: list[float]) -> dict:
    """配对差值 + **可判定性**判断。

    正确用法是**同一探针集**下跑 A 与 B 各若干次（不同随机种子仅影响
    初始化/采样顺序，探针集本身不变）→ 差值的噪声可用**成对差的标准差**
    直接估计，从而回答「观察到的差异是否只是噪声」。
    """
    out = {"n_pairs": 0, "mean_a": float("nan"), "mean_b": float("nan"),
           "delta": float("nan"), "sd": float("nan"), "t_like": float("nan"),
           "verdict": "样本不足"}
    if not a_runs or not b_runs:
        return out
    n = min(len(a_runs), len(b_runs))
    a = np.asarray(a_runs[:n], dtype=np.float64)
    b = np.asarray(b_runs[:n], dtype=np.float64)
    d = b - a
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    out.update(n_pairs=n, mean_a=float(a.mean()), mean_b=float(b.mean()),
               delta=float(d.mean()), sd=sd,
               t_like=float(abs(d.mean()) / sd) if sd > 0 else float("inf"))
    # 判定门槛：|mean| > 2×SD（≈ 95% 置信的粗略版；样本少时用 3×更稳）
    k = 3.0 if n < 5 else 2.0
    out["verdict"] = ("差异超出噪声（可判定）" if abs(out["delta"]) > k * sd
                      else "差异在噪声内（不可判定，需加样本）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="P146 PPL 评估机制 v2：固定探针集 + 配对比较")
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--probe-tokens", type=int, default=2048)
    ap.add_argument("--runs", type=int, default=3, help="配对重复次数")
    ap.add_argument("--compare", default="", help="A,B 两个标签（仅打印用）")
    args = ap.parse_args()

    import os
    os.environ.setdefault("NUMBA_CACHE_DIR",
                          str(_ROOT / "outputs" / "numba_cache"))

    print("=" * 78)
    print("P146 PPL 评估机制 v2 —— 固定探针集（**解决波动淹没 A/B 的问题**）")
    print("=" * 78)
    load_probe_text()
    print(f"探针集: eval_corpus/internal_corpus.txt 的前 {args.probe_tokens} token")
    print("⚠ 探针**绝对值**与训练流 PPL 不可比（不同文本）；**只用差值**。")
    print("⚠ 探针文本也在训练语料里→ 它消除的是「窗口语料差异」这个混淆变量，")
    print("   **不是**消除训练/评测泄漏。")

    # 独立自测：不接训练，直接展示「同一探针集的重复测量有多稳」
    print("\n--- 机制自检：同一探针集重复测量的噪声水平 ---")
    try:
        from phdnet.word_lm import PHDWordLM
        from config_1b import build_cfg
        cfg = build_cfg(args.preset)
        runs = []
        for r in range(args.runs):
            lm = PHDWordLM(load_probe_text(), cfg)
            res = probe_nll(lm, args.probe_tokens)
            runs.append(res["ppl"])
            print(f"  run{r+1}: probe_ppl = {res['ppl']:.4f}"
                  f"（n_tok={res['n_tok']}, {res['ms']:.0f} ms）")
        if len(runs) > 1:
            sd = float(np.std(runs, ddof=1))
            print(f"\n  同一探针集的重复测量标准差 = {sd:.4f}"
                  f"（相对 {sd/np.mean(runs)*100:.2f}%）")
            print("  → 这是**配对比较的噪声下限**。若A/B 差值大于它，"
                  "差异就可判定。")
            print()
            print("  ⚠⚠ **但 sd = 0 并不意味着机制完美**，三件事必须说清：")
            print("  ① 本自检的多次运行**用同一个 seed** → 初始化与采样顺序完全"
                  "相同 → 读数**必然相同**。真实 A/B 里 A 与 B 的差别**就是"
                  "配置本身**（如 gather 精度），那才是有意义的差值。")
            print("  ② 要估计「测量噪声」需**换 seed** 重跑（本机慢，建议服务器做）。")
            print("  ③ 探针 PPL **绝对值**与训练流 PPL 不可比（不同文本）→ "
                  "**只能用差值**。")
            print("  ④ 探针文本**也在训练语料里**（P100 记录过）→ 它消除的是"
                  "「窗口语料差异」这个混淆变量，**不是**训练/评测泄漏。")
    except Exception as e:                                     # noqa: BLE001
        print(f"  自检跳过（需要可构造的 LM）：{type(e).__name__}: "
              f"{str(e)[:120]}")
    print("\n接入方式：在训练循环里每 N 步调用 `probe_nll(lm, n_tokens)`，")
    print("把 `probe_ppl` 与 `train_ppl` **并列**打印；A/B 比较时用 probe 的差。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())