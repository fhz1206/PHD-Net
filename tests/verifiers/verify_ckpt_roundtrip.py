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


def _gate_ckpt_dtype() -> bool:
    """门禁 (a)：--ckpt-dtype fp8/bf16 保存→恢复必须逐位还原、量级不变。

    2026-10-07 修复（审计 1/3）回归护栏：修复前 _down 存位模式、load 直接数值
    cast → fp8 读回 max|W|=166（原 0.2154）、bf16=4.87e4；且 _down 在 ro_W
    加入 arrs 之前执行 → --ckpt-dtype 从不压缩读出 W。"""
    import json as _json
    import tempfile as _tf
    import torch as _t
    ok = True
    tmp = Path(_tf.mkdtemp(prefix="phd_gate_dtype_"))
    lm1, cfg = build()
    W = np.asarray(lm1.net.encoder.W)
    assert W.size >= 4096, f"encoder_W {W.shape} 太小 → _down 不压缩，门禁失效"
    # 造 fp8(e4m3：[1,2) 步长 1/8) 与 bf16(步长 2^-7) 都**精确可表示**的权重，
    # 所以保存前/后必须逐位一致——修复前的数值 cast 读回垃圾必然不等。
    base = np.arange(W.size, dtype=np.float64).reshape(W.shape)
    W_repr = (1.0 + (base % 9) / 8.0).astype(np.float32)
    RW = np.asarray(lm1.net.readout.W)
    _ro_big = RW.size >= 4096
    ro_repr = (1.0 + (np.arange(RW.size, dtype=np.float64).reshape(RW.shape) % 9)
               / 8.0).astype(np.float32) if _ro_big else None
    W0 = W_repr.copy()
    m0 = float(np.abs(W0).max())
    for mode in ("fp8", "bf16"):
        lm1.net.encoder.W[:] = W_repr
        if _ro_big:
            lm1.net.readout.W[:] = ro_repr
        ro0 = np.asarray(lm1.net.readout.W).copy()
        p = tmp / f"gate_{mode}.npz"
        save_model(p, lm1, cfg, 1, ckpt_dtype=mode)
        want = np.uint8 if mode == "fp8" else np.uint16
        with np.load(str(p), allow_pickle=False) as z:
            files = list(z.files)
            stored = z["encoder_W"]
            ro_stored = z["ro_W"] if "ro_W" in files else None
        if stored.dtype != want:
            print(f"  ✗ [{mode}] 存盘 encoder_W dtype={stored.dtype}，应为位模式 {want}")
            ok = False
        if _ro_big and (ro_stored is None or ro_stored.dtype != want):
            print(f"  ✗ [{mode}] ro_W 存盘 dtype="
                  f"{None if ro_stored is None else ro_stored.dtype}（审计 3："
                  f"--ckpt-dtype 没压缩读出 W）")
            ok = False
        lm2, _ = build()
        load_model(p, lm2)
        W1 = np.asarray(lm2.net.encoder.W)
        ro1 = np.asarray(lm2.net.readout.W)
        if not np.array_equal(W1, W0):
            print(f"  ✗ [{mode}] encoder_W 恢复非逐位一致：max|W1|="
                  f"{np.abs(W1).max():.4g} max|W0|={m0:.4g}（审计 1：应为 166/4.9e4 "
                  f"这类垃圾）")
            ok = False
        if _ro_big and not np.array_equal(ro1, ro0):
            print(f"  ✗ [{mode}] readout.W 恢复非逐位一致 max|ro1|="
                  f"{np.abs(ro1).max():.4g}")
            ok = False
        ratio = float(np.abs(W1).max()) / m0
        if not (0.5 <= ratio <= 2.0):
            print(f"  ✗ [{mode}] max|W| 量级漂移 ratio={ratio:.4g}（应≈1）")
            ok = False
        # 非可表示权重：解码必须与保存端位模式编码**逐位互逆**
        np.random.seed(1)
        lm1.net.encoder.W[:] = (np.random.randn(*W0.shape) * 0.2).astype(np.float32)
        Wr = np.asarray(lm1.net.encoder.W).copy()
        p2 = tmp / f"gate_{mode}_rand.npz"
        save_model(p2, lm1, cfg, 1, ckpt_dtype=mode)
        lm3, _ = build()
        load_model(p2, lm3)
        Wd = np.asarray(lm3.net.encoder.W)
        tt = _t.from_numpy(Wr)
        exp = (tt.to(_t.bfloat16).float().numpy() if mode == "bf16"
               else tt.to(_t.float8_e4m3fn).float().numpy())
        if not np.array_equal(Wd, exp):
            print(f"  ✗ [{mode}] 随机权重解码 ≠ 位模式编码回读：max|Wd|="
                  f"{np.abs(Wd).max():.4g}（原 max={np.abs(Wr).max():.4g}）")
            ok = False
        print(f"  ✓ [{mode}] 逐位往返 ok | max|W|={np.abs(W1).max():.4g}/"
              f"{m0:.4g} | ro_W stored dtype="
              f"{None if ro_stored is None else ro_stored.dtype}"
              f"{'' if _ro_big else ' (读出过小未压)'}")
    for f in tmp.glob("*"):
        f.unlink(missing_ok=True)
    tmp.rmdir()
    return ok


def _gate_marker() -> bool:
    """门禁 (b)：marker 长于 seg.max_len 时 SFT 掩码状态机必须正确。

    2026-10-07 修复（审计 2）回归护栏：修复前 pending 无条件减法 → 纯回复文本
    trainable=0（静默零学习）、_pending=-20；marker 未出现却误开 12 个 token；
    失配分支（corpus_stream.py:318）因 buf.startswith(tok) 恒真而不可达。"""
    from corpus_stream import StreamingTokenizer

    class Seg:
        pass
    seg = Seg()
    seg.max_len = 6
    seg.vocab = {"Assist", "ant: 你", "你好", "世界", "abc"}
    mk = "Assistant: "                      # 11 > max_len=6
    ok = True

    # 用例 1：marker 出现两次的纯回复文本 → 回复段必须开学习（不能全 0）
    s = StreamingTokenizer(seg,
                           iter(["Assistant: 你好世界 Assistant: 你好世界"]),
                           assistant_marker=mk)
    items = list(s)
    n1 = sum(1 for _, f in items if f)
    if n1 == 0:
        print(f"  ✗ marker>max_len 时 trainable 全 0（静默零学习）tokens="
              f"{[t for t, _ in items]} pending={s._pending}")
        ok = False
    if not (0 <= s._pending <= len(mk)):
        print(f"  ✗ _pending={s._pending} 未钳制在 [0, {len(mk)}]")
        ok = False

    # 用例 2：完整 marker 从未出现 → 绝不能误开任何 token
    s2 = StreamingTokenizer(seg, iter(["Assistantly hello world"]),
                            assistant_marker=mk)
    n2 = sum(1 for _, f in s2 if f)
    if n2 != 0:
        print(f"  ✗ marker 未出现却误开 {n2} 个 trainable token")
        ok = False

    # 用例 3：失配分支必须可达（内容与 marker 后续不符 → 复位 pending）
    s3 = StreamingTokenizer(seg, iter(["Assistant: 你好世界"]), assistant_marker=mk)
    next(s3)                                 # 'Assist' → 部分命中，_pending>0
    if s3._pending <= 0:
        print(f"  ✗ 部分命中后 _pending={s3._pending}，状态机没进入待续")
        ok = False
    else:
        s3._consume("zz")                    # 与 marker 后续 'ant: ' 内容不符
        if s3._pending != 0 or s3._pending_mk is not None:
            print(f"  ✗ 失配分支不可达：_pending={s3._pending} "
                  f"_pending_mk={s3._pending_mk!r}")
            ok = False
    # 用例 4（返工 4 回归）：贪心 token **整段吞下 marker 且更长**
    #   （tok.startswith(mk) 且 len(tok) > len(mk)）→ 必须切模式；
    #   旧版该例落进分支③，marker 整个丢失、助手回复全程不计损失。
    s4 = StreamingTokenizer(seg, iter(["xyzzy"]), assistant_marker=mk)
    next(s4)                                  # 初始化内部缓冲（同用例 3）
    tok4 = mk + "你好"                        # 13 > 11 → 吞下整个 marker 且更长
    r4 = s4._consume(tok4)
    if s4._mode is not True:
        print(f"  ✗ token 整段吞掉 marker 未切模式：_mode={s4._mode} r4={r4}")
        ok = False
    if s4._pending != 0 or s4._pending_mk is not None:
        print(f"  ✗ 吞 marker 后 _pending={s4._pending} "
              f"_pending_mk={s4._pending_mk!r}（应清零）")
        ok = False
    r5 = s4._consume("世界")                  # marker 之后的回复 token
    if not (isinstance(r5, tuple) and r5[1] is True):
        print(f"  ✗ 切模式后回复 token 仍不计损失：r5={r5}")
        ok = False
    print(f"  {'✓' if ok else '✗'} marker gate: case1 trainable={n1} "
          f"(须>0) | case2 trainable={n2} (须=0) | 失配复位可达 | "
          f"case4 吞 marker 切模式 r4={r4} r5={r5}")
    return ok


def _gate_ckpt_formats() -> bool:
    """门禁 (c)：返工 3 回归 —— 两种回退判据边界的检查点都必须能载入。

    ① **新格式 + 空键**：meta 有 `ckpt_dtype_keys`（= []）而 ckpt_dtype 非空
       （= 请求了压缩但没有任何数组过压缩阈值）→ 键表合法为空，绝不能走
       旧格式回退去解码 float 数组（旧 bug：`not _dk` 判据 → 合法检查点被拒载）。
    ② **旧格式 + 未过阈值的 1 维数组**：meta **没有** `ckpt_dtype_keys` →
       回退必须**重放压缩谓词**（ndim>=2 且 size>=4096），1 维 encoder_b /
       wm_strength 存盘仍是 float32，不能被当位模式解码
       （旧 bug：硬编码 5 键 → _decode_bits 抛 ValueError → 旧检查点载不回）。
    两例都必须打印一行「按 X 格式载入」。"""
    import io as _io
    import contextlib as _ctx
    import json as _json
    import tempfile as _tf
    ok = True
    tmp = Path(_tf.mkdtemp(prefix="phd_gate_fmt_"))

    def _repatch(src: Path, dst: Path, mutate) -> None:
        """把 npz 原样重写、只改 meta（模拟旧/新格式的键表差异）。"""
        with np.load(str(src), allow_pickle=False) as z:
            arrs = {k: z[k] for k in z.files}
        meta = _json.loads(str(arrs["meta"][0]))
        mutate(meta)
        arrs["meta"] = np.array([_json.dumps(meta, ensure_ascii=False)])
        np.savez(str(dst), **arrs)

    lm1, cfg = build()
    W_ref = np.asarray(lm1.net.encoder.W).copy()
    # ① 新格式 + 空键
    pA0, pA = tmp / "a0.npz", tmp / "a.npz"
    save_model(pA0, lm1, cfg, 1, ckpt_dtype=None)     # 无压缩 → 数组全 float
    _repatch(pA0, pA, lambda m: (m.__setitem__("ckpt_dtype", "fp8"),
                                 m.__setitem__("ckpt_dtype_keys", [])))
    lmA, _ = build()
    buf = _io.StringIO()
    try:
        with _ctx.redirect_stdout(buf):
            load_model(pA, lmA)
        _load_ok, _err = True, ""
    except Exception as _e:                           # noqa: BLE001
        _load_ok, _err = False, f"{type(_e).__name__}: {_e}"
    _same = (_load_ok and np.array_equal(np.asarray(lmA.net.encoder.W),
                                         W_ref.astype(np.asarray(lmA.net.encoder.W).dtype)))
    if not _load_ok:
        print(f"  ✗ [新格式+空键] 载入被拒（合法检查点）：{_err}")
        ok = False
    elif "按新格式载入" not in buf.getvalue():
        print(f"  ✗ [新格式+空键] 未打印「按新格式载入」行：{buf.getvalue()!r}")
        ok = False
    elif not _same:
        print("  ✗ [新格式+空键] 数组被误当位模式解码（数值变了）")
        ok = False
    # ② 旧格式 + 未过阈值的 1 维数组
    pB0, pB = tmp / "b0.npz", tmp / "b.npz"
    save_model(pB0, lm1, cfg, 1, ckpt_dtype="fp8")    # 新格式：大矩阵 uint8、
    _repatch(pB0, pB, lambda m: m.pop("ckpt_dtype_keys", None))  # 退回旧格式
    with np.load(str(pB), allow_pickle=False) as z:
        if z["encoder_b"].dtype.kind == "u":
            print("  ✗ [旧格式] fixture 未生效：encoder_b 竟是位模式")
            ok = False
    lmB, _ = build()
    buf2 = _io.StringIO()
    try:
        with _ctx.redirect_stdout(buf2):
            load_model(pB, lmB)
        _ok2, _err2 = True, ""
    except Exception as _e:                           # noqa: BLE001
        _ok2, _err2 = False, f"{type(_e).__name__}: {_e}"
    if not _ok2:
        print(f"  ✗ [旧格式+1维数组] 载入被拒（旧检查点载不回）：{_err2}")
        ok = False
    elif "按旧格式载入" not in buf2.getvalue():
        print(f"  ✗ [旧格式] 未打印「按旧格式载入」行：{buf2.getvalue()!r}")
        ok = False
    else:
        # 对照：同一份数据按**新格式**载入，权重必须完全一致
        lmC, _ = build()
        with _ctx.redirect_stdout(_io.StringIO()):
            load_model(pB0, lmC)
        if not np.array_equal(np.asarray(lmB.net.encoder.W),
                              np.asarray(lmC.net.encoder.W)):
            print("  ✗ [旧格式] encoder.W 与新格式载入结果不一致")
            ok = False
    for f in tmp.glob("*"):
        f.unlink(missing_ok=True)
    tmp.rmdir()
    print(f"  {'✓' if ok else '✗'} ckpt 回退判据 gate: ①新格式+空键可载入 "
          f"②旧格式+1维数组可载入（均打印「按 X 格式载入」）")
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

# ── 门禁 (a)(b)：2026-10-07 修复（审计 1-4）的回归护栏（须全绿）──
ok &= _gate_ckpt_dtype()
ok &= _gate_marker()
ok &= _gate_ckpt_formats()

TMP.with_suffix(".json").unlink(missing_ok=True)
TMP.unlink(missing_ok=True)
print("=" * 60)
print("检查点往返逐位等价: " + ("PASS ✓（全部状态一致）" if ok else "FAIL ✗"))
sys.exit(0 if ok else 1)
