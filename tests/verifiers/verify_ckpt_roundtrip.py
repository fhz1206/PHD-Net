"""临时验证：train_1b 检查点保存→恢复→续训逐位等价（跑完即删）。

流程：
  net1：训练 300 步 → save_model → 继续训练 100 步 → 权重状态 A
  net2：重建（同 seed 同语料）→ load_model → 训练 100 步 → 权重状态 B
  断言：A 与 B 的全部可塑状态逐位一致（权重 + 大空间表 + 词表 SDR）。
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_HERE), str(_ROOT), str(_ROOT / "train")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from ckpt_1b import load_model, save_model
from config_1b import SEG_KWARGS, build_cfg
from phdnet.corpus import load_text
from phdnet.word_lm import PHDWordLM

TMP = _HERE / "_ckpt_roundtrip.npz"
text = load_text(_ROOT / "eval_corpus" / "internal_corpus.txt", limit_chars=200000)


def build():
    cfg = build_cfg("smoke")
    return PHDWordLM(text, cfg, seg_kwargs=SEG_KWARGS), cfg


def run(lm, toks, i0, n):
    for i in range(i0, min(i0 + n, len(toks) - 1)):
        prev = toks[i - 1] if i > 0 else None
        x = lm.tok.encode_composite(toks[i], prev)
        tgt = lm.tok.onehot(lm.tok.stoi[toks[i + 1]])
        lm.net.step(x, target=tgt, learn=True)


def snapshot(lm):
    net = lm.net
    s = {
        "enc": np.asarray(net.encoder.W).copy(), "enc_b": np.asarray(net.encoder.b).copy(),
        "stdp": np.asarray(net.stdp.W).copy(), "ro": np.asarray(net.readout.W).copy(),
        "wm_s": np.asarray(net.wm.slots).copy(), "wm_t": np.asarray(net.wm.strength).copy(),
    }
    for nm in ("up0", "up1", "dn0", "dn1"):
        ip, idx, val = getattr(net.pc, nm)
        s[f"pc_{nm}"] = np.asarray(val).copy()
    t = net.ltm.table
    snap = t.compact_csr()
    s["ltm"] = (snap["row_ids"].copy(), snap["indptr"].copy(),
                snap["indices"].copy(), snap["data"].copy())
    s["ltm_trace"] = (dict(t.t_pre), dict(t.t_post), dict(t.stamp_pre),
                      dict(t.stamp_post), t.step_count)
    return s


def same(a, b, tag) -> bool:
    if isinstance(a, np.ndarray):
        ok = a.shape == b.shape and bool(np.array_equal(a, b))
    else:
        ok = a == b
    if not ok:
        print(f"  ✗ {tag} 不一致")
    return ok


# ── net1：300 步 → 存 → 再 100 步 → 状态 A ──
lm1, cfg1 = build()
toks = lm1.tokenize(text)
run(lm1, toks, 0, 300)
save_model(TMP, lm1, cfg1, 300)
run(lm1, toks, 300, 100)
A = snapshot(lm1)

# ── net2：重建 → 恢复 → 100 步 → 状态 B ──
lm2, _ = build()
meta = load_model(TMP, lm2)
assert meta["done"] == 300 and meta["vocab_size"] == len(lm2.tok)
toks2 = lm2.tokenize(text)
assert toks2 == toks, "恢复后分词不一致"
run(lm2, toks2, 300, 100)
B = snapshot(lm2)

# ── 逐位对比 ──
ok = True
for k in A:
    if k == "ltm":
        for j, nm in enumerate(("row_ids", "indptr", "indices", "data")):
            ok &= same(A[k][j], B[k][j], f"ltm.{nm}")
        ok &= same(A["ltm_trace"], B["ltm_trace"], "ltm.traces+step")
    else:
        ok &= same(A[k], B[k], k)

# 额外：net1 与 net2 的 tokenizer SDR 缓存逐位一致
for t in list(lm1.tok._sdrs)[:50] + list(lm1.tok._sdrs)[-50:]:
    ok &= same(lm1.tok._sdrs[t], lm2.tok._sdrs[t], f"sdr[{t}]")

TMP.with_suffix(".json").unlink(missing_ok=True)
TMP.unlink(missing_ok=True)
print("=" * 60)
print("检查点往返逐位等价: " + ("PASS ✓（全部状态一致）" if ok else "FAIL ✗"))
sys.exit(0 if ok else 1)
