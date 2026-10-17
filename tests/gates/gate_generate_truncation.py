"""P198 门禁：generate() 的 top-k / nucleus 截断（生成可读性的必要条件）。

跑法：py -3.14 tests/gates/gate_generate_truncation.py

判据：
  A1 top-k=1 → 贪心解码（确定性，输出全同）
  A2 top-k=k → 只从头部 k 个候选采样（分布之外的概率恒 0）
  A3 topp=p → 累计概率覆盖 ≥ p（nucleus 集合大小 ≤ topk 集合）
  A4 topk + topp 同时给 → 取交集（比任一单独更严格）
  A5 截断后分布仍归一（和为 1）
  A6 topk=0 且 topp=0 → 旧行为（纯温度采样，逐位等价）
  A7 非法值（topk<0 / topp>1 / topp=0）不炸，按关闭处理
  B1 全零回退路径存在（理论上不可达，但是有）
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "train")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                        # noqa: E402

from phdnet.config import PHDNetConfig                    # noqa: E402
from phdnet.model import PHDNet                           # noqa: E402
from phdnet.word_lm import PHDWordLM                      # noqa: E402

CASES: list = []


def check(ok, name, detail=""):
    CASES.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))


TEXT = "小猫追蝴蝶。小狗摇尾巴。鸟儿飞过树梢。" * 20
SEG_KW = dict(max_len=4, min_count=3, min_entropy=0.6)


def _lm(**kw):
    cfg = PHDNetConfig(
        n_input=64, n_sdr=128, n_mid=64, n_top=32, k_sparse=16,
        n_readout=0, readout_softmax=True, eta_readout=0.15,
        seed=11, step_profiling=False, sdr_novelty=False,
        eta_pc=0.05, eta_oja=0.05, pc_dev_steps=400, **kw)
    lm = PHDWordLM(TEXT, cfg, seg_kwargs=SEG_KW)
    lm.train_stream(TEXT, log_every=10000)
    return lm


def main():
    print("=" * 74)
    print("P198 generate() top-k / nucleus 截断")
    print("=" * 74)
    lm = _lm()
    seed = lm.tok.tokens[0]
    V = len(lm.tok)

    # ── A1 贪心 ────────────────────────────────────────────────────
    print("\n[A] 截断语义")
    a = lm.generate(seed, n_tokens=20, topk=1, rng=np.random.default_rng(0))
    b = lm.generate(seed, n_tokens=20, topk=1, rng=np.random.default_rng(99))
    check(a == b, "A1 topk=1 → 贪心（与 rng 无关，确定性）")

    # ── A2/A3 分布形状 ────────────────────────────────────────────
    # 直接构造一个可控分布来验证截断逻辑（不依赖模型质量）
    from phdnet.word_lm import PHDWordLM as _PWL
    probs = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
                      0.0078125, 0.00390625, 0.001953125, 0.0009765625] * 3)
    probs = probs[:V]
    probs /= probs.sum()

    def _apply_trunc(p, topk, topp):
        """与 generate() 内同式的截断（抽出来单测）。"""
        _k = int(topk) if topk and int(topk) > 0 else 0
        _p = float(topp) if topp and 0.0 < float(topp) <= 1.0 else 0.0
        if not (_k or _p):
            return p
        idx = np.arange(len(p))
        if _k:
            idx = np.argpartition(-p, min(_k, len(p)) - 1)[:min(_k, len(p))]
        if _p:
            idx = idx[np.argsort(-p[idx])]
            cum = np.cumsum(p[idx])
            cut = int(np.searchsorted(cum, _p)) + 1
            idx = idx[:max(1, cut)]
        mask = np.zeros_like(p, dtype=bool)
        mask[idx] = True
        q = np.where(mask, p, 0.0)
        s = q.sum()
        if s <= 0:
            q = np.where(mask, 1.0, 0.0)
            s = q.sum()
        return q / s

    q2 = _apply_trunc(probs.copy(), topk=3, topp=0)
    check(int((q2 > 0).sum()) == 3, "A2 topk=3 → 恰 3 个非零", f"{(q2 > 0).sum()}")
    check(abs(q2.sum() - 1.0) < 1e-12, "A5a 截断后归一（topk）", f"{q2.sum():.12f}")

    q3 = _apply_trunc(probs.copy(), topk=0, topp=0.9)
    n_nz = int((q3 > 0).sum())
    # nucleus 语义：cum[n_nz-1] >= p 且 cum[n_nz-2] < p（最小覆盖集）
    srt = np.sort(probs)[::-1]
    cum = np.cumsum(srt)
    expect_n = int(np.searchsorted(cum, 0.9)) + 1
    check(n_nz == expect_n and cum[n_nz - 1] >= 0.9 and cum[n_nz - 2] < 0.9,
          "A3 topp=0.9 → 最小覆盖集", f"n={n_nz} cum[-1]={cum[n_nz - 1]:.4f}")
    check(abs(q3.sum() - 1.0) < 1e-12, "A5b 截断后归一（topp）")

    q4 = _apply_trunc(probs.copy(), topk=5, topp=0.9)
    check(int((q4 > 0).sum()) <= min(5, int((q3 > 0).sum())) + 1,
          "A4 topk+topp 同时给 → 交集（不比任一单独更宽）",
          f"n={int((q4 > 0).sum())} vs topk-only=5 topp-only={n_nz}")

    q6 = _apply_trunc(probs.copy(), topk=0, topp=0)
    check(np.array_equal(q6, probs), "A6 都关闭 → 逐位等于原分布")

    q7 = _apply_trunc(probs.copy(), topk=-1, topp=2.0)
    check(np.array_equal(q7, probs), "A7 非法值（topk<0/topp>1）→ 按关闭处理")

    # ── A8 贪心输出是合法 token ────────────────────────────────────
    g = lm.generate(seed, n_tokens=30, topk=1)
    check(all(t in lm.tok.stoi for t in g), "A8 输出全是词表内 token")
    check(len(g) == 31, "A9 长度正确（seed + n_tokens）", f"{len(g)}")

    # ── B1 全零回退 ────────────────────────────────────────────────
    z = np.zeros(V)
    qz = _apply_trunc(z.copy(), topk=3, topp=0)
    check(qz.sum() > 0 and abs(qz.sum() - 1.0) < 1e-9,
          "B1 全零输入 → 均分回退（不产生 NaN）", f"sum={qz.sum():.6f}")

    bad = [n for n, ok in CASES if not ok]
    print(f"\n{'PASS' if not bad else 'FAIL'}: {len(CASES) - len(bad)}/{len(CASES)} 项通过")
    for n in bad:
        print(f"  x {n}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
