"""P6 性能优化的逐位等价对拍（Task #78 / Request I）。

验证三项热路径修复与旧行为 bit 级一致：
  A) readout.learn_softmax：「外积后原地缩放」vs 旧「eta * np.outer」（随机值对拍）
  B) learn_softmax y_pre 通道：传入已算 W@h vs 内部重算（且 y_pre 本体不被修改）
  C) pc.learn 零学习率短路 vs 显式执行乘零更新（eta_pc=eta_oja=0）
  D) 端到端金标准：同 seed 两个模型，一个走新代码、一个 monkeypatch 回全部旧实现，
     各跑 800 token 训练 + 评估，对比逐段 NLL、评估 PPL 与全部持久权重逐位一致。

全部通过则证明：默认路径行为未变，仅消除无效计算。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests"))

from eval_common import BASE, DOC, SEG  # noqa: E402
from phdnet.config import PHDNetConfig  # noqa: E402
from phdnet.pc import PredictiveCodingStack  # noqa: E402
from phdnet.readout import Readout  # noqa: E402
from phdnet.word_lm import PHDWordLM  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"


# ---------- 旧实现（内联快照，与修改前源码逐行一致） ----------

def old_learn_softmax(self, h, target, eta, y_pre=None):
    y = self.W @ h
    y -= y.max()
    p = np.exp(y)
    p /= p.sum()
    self.W -= eta * np.outer(p - target, h)
    self._clip()
    correct = int(np.argmax(target))
    return float(-np.log(p[correct] + 1e-12))


def old_pc_learn(self, cache, eta_scale=1.0, homeostasis=False):
    s0, r1, r2 = cache["s0"], cache["r1"], cache["r2"]
    e0, e1 = cache["e0"], cache["e1"]
    if homeostasis:
        e0 = e0 / max(float(np.linalg.norm(e0)), 1e-9)
        e1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
    self.W_dn0 += self.eta_pc * eta_scale * np.outer(e0, r1)
    self.W_dn1 += self.eta_pc * eta_scale * np.outer(e1, r2)
    self.W_up0 += self.eta_oja * eta_scale * (r1[:, None] * (s0[None, :] - r1[:, None] * self.W_up0))
    self.W_up1 += self.eta_oja * eta_scale * (r2[:, None] * (r1[None, :] - r2[:, None] * self.W_up1))
    np.clip(self.W_up0, -self.w_max, self.w_max, out=self.W_up0)
    np.clip(self.W_up1, -self.w_max, self.w_max, out=self.W_up1)
    np.clip(self.W_dn0, -self.w_max, self.w_max, out=self.W_dn0)
    np.clip(self.W_dn1, -self.w_max, self.w_max, out=self.W_dn1)
    if homeostasis:
        self._homeostatic_scale()


def check(name: str, ok: bool) -> bool:
    print(f"[{PASS if ok else FAIL}] {name}")
    return ok


# ---------- A: learn_softmax 新旧对拍 ----------

def test_a() -> bool:
    rng = np.random.default_rng(0)
    n_out, n_in = 50, 40
    h = rng.normal(0, 1, n_in)
    target = np.zeros(n_out)
    target[7] = 1.0
    ok = True
    for eta in (0.05, 0.0, 0.3):
        W_old = rng.normal(0, 0.5, (n_out, n_in))
        W_new = W_old.copy()
        r_old = Readout.__new__(Readout)
        r_old.W, r_old.w_clip = W_old, 0.0
        r_new = Readout.__new__(Readout)
        r_new.W, r_new.w_clip = W_new, 0.0
        nll_old = old_learn_softmax(r_old, h, target, eta)
        nll_new = Readout.learn_softmax(r_new, h, target, eta)
        ok &= check(f"A eta={eta}: W 逐位一致", bool(np.array_equal(W_old, W_new)))
        ok &= check(f"A eta={eta}: NLL 逐位一致", nll_old == nll_new)
    return ok


# ---------- B: y_pre 通道对拍 ----------

def test_b() -> bool:
    rng = np.random.default_rng(1)
    n_out, n_in = 60, 40
    h = rng.normal(0, 1, n_in)
    target = np.zeros(n_out)
    target[3] = 1.0
    eta = 0.05
    W_ref = rng.normal(0, 0.5, (n_out, n_in))

    r_old = Readout.__new__(Readout)
    r_old.W, r_old.w_clip = W_ref.copy(), 0.0
    nll_ref = old_learn_softmax(r_old, h, target, eta)

    W_new = W_ref.copy()
    r_new = Readout.__new__(Readout)
    r_new.W, r_new.w_clip = W_new, 0.0
    y_pre = W_new @ h                       # 模拟 model.step 已算的前向
    y_pre_snapshot = y_pre.copy()
    nll_new = Readout.learn_softmax(r_new, h, target, eta, y_pre=y_pre)

    ok = check("B: W 逐位一致", bool(np.array_equal(r_old.W, W_new)))
    ok &= check("B: NLL 逐位一致", nll_ref == nll_new)
    ok &= check("B: y_pre 本体未被修改", bool(np.array_equal(y_pre, y_pre_snapshot)))
    return ok


# ---------- C: pc.learn 零学习率短路对拍 ----------

def test_c() -> bool:
    rng = np.random.default_rng(2)
    pc = PredictiveCodingStack(64, 48, 32, eta_pc=0.0, eta_oja=0.0, rng=rng)
    s0 = rng.normal(0, 1, 64)
    cache = pc.infer(s0, 1)
    cache["s0"] = s0

    W0 = {k: getattr(pc, k).copy() for k in ("W_up0", "W_up1", "W_dn0", "W_dn1")}
    old_pc_learn(pc, cache)                 # 显式执行旧乘零更新
    ok = True
    for k, w0 in W0.items():
        same = np.array_equal(w0, getattr(pc, k))
        ok &= check(f"C: {k} 乘零更新后逐位不变（短路等价）", same)
    return ok


# ---------- D: 端到端 monkeypatch 金标准 ----------

_ORIG_LS = Readout.learn_softmax
_ORIG_PL = PredictiveCodingStack.learn


def _run_e2e(patch_old: bool, chunk: str) -> tuple[list[float], float, dict]:
    text = Path(DOC).read_text(encoding="utf-8")
    lm = PHDWordLM(text, PHDNetConfig(**BASE), seg_kwargs=SEG)
    Readout.learn_softmax = old_learn_softmax if patch_old else _ORIG_LS
    PredictiveCodingStack.learn = old_pc_learn if patch_old else _ORIG_PL
    nlls = lm.train_stream(chunk, log_every=100)
    m = lm.evaluate(text[int(len(text) * 0.8):int(len(text) * 0.8) + 1500])
    net = lm.net
    weights = {
        "enc.W": net.encoder.W, "pc.W_up0": net.pc.W_up0, "pc.W_up1": net.pc.W_up1,
        "pc.W_dn0": net.pc.W_dn0, "pc.W_dn1": net.pc.W_dn1, "stdp.W": net.stdp.W,
        "ro.W": net.readout.W, "wm.slots": net.wm.slots, "wm.strength": net.wm.strength,
        "ltm.W_fast": net.ltm.W_fast, "ltm.W_slow": net.ltm.W_slow,
        "prev_rate": net._prev_rate, "last_rate": net._last_rate,
    }
    return nlls, m["ppl_char"], weights


def test_d() -> bool:
    text = Path(DOC).read_text(encoding="utf-8")
    chunk = text[:int(len(text) * 0.8)][:2200]     # ~1300 tokens
    # 新代码先跑（避免 reload 顺序问题：先 patch 后 reload 恢复）
    nlls_new, ppl_new, w_new = _run_e2e(False, chunk)
    nlls_old, ppl_old, w_old = _run_e2e(True, chunk)
    ok = check("D: 训练 NLL 序列逐位一致", nlls_new == nlls_old)
    ok &= check("D: 评估 ppl_char 逐位一致", ppl_new == ppl_old)
    for k in w_new:
        ok &= check(f"D: 权重 {k} 逐位一致", bool(np.array_equal(w_new[k], w_old[k])))
    print(f"    ppl_char: old={ppl_old!r}  new={ppl_new!r}")
    return ok


if __name__ == "__main__":
    all_ok = True
    all_ok &= test_a()
    all_ok &= test_b()
    all_ok &= test_c()
    all_ok &= test_d()
    print("\n=== 全部逐位等价 ===" if all_ok else "\n=== 存在不一致，禁止合入 ===")
    sys.exit(0 if all_ok else 1)
