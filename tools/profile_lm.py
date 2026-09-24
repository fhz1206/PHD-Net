"""LM 热路径剖析（Task #78）：定位 PHDWordLM 训练/评估的性能热点。

方法：
  1) cProfile 端到端剖析 train_stream + evaluate（冻结语料切片，eval_suite 同款
     BASE 配置），按 tottime / cumulative 双排序输出热点排名；
  2) 手工包装计时：对 step 管线各相（编码 / PC 推理 / PC 学习 / STDP / WM / LTM /
     读出前向 / 读出学习 / onehot / composite）单独累计耗时，给出每 token 分解。

只读剖析，不修改任何模型与数据；结果写入 outputs/profile_lm.log。
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests"))

from eval_common import BASE, DOC, SEG  # noqa: E402
from phdnet.config import PHDNetConfig  # noqa: E402
from phdnet.word_lm import PHDWordLM  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "profile_lm.log"
N_TRAIN = 1000        # cProfile 剖析的训练 token 数（切片）
N_WARM = 200          # 预热（numba JIT / numpy 缓存）
N_EVAL = 500          # 剖析的评估 token 数


def main() -> None:
    text = Path(DOC).read_text(encoding="utf-8")
    print(f"语料: {DOC}  共 {len(text)} 字符")
    print(f"BASE 配置: {BASE}")
    print(f"SEG 配置: {SEG}")

    lm = PHDWordLM(text, PHDNetConfig(**BASE), seg_kwargs=SEG)
    V = len(lm.tok)
    n_train_chars = int(len(text) * 0.8)
    train_txt = text[:n_train_chars]

    # ---------- 分词计时 ----------
    t0 = time.perf_counter()
    toks = lm.tokenize(train_txt)
    t_tok = time.perf_counter() - t0
    print(f"\n[分词] {len(toks)} tokens / {n_train_chars} chars, "
          f"{t_tok * 1000:.1f} ms 全量 ({t_tok / max(1, len(toks)) * 1e6:.1f} us/token)")

    # ---------- 预热 ----------
    lm.train_stream(train_txt[:N_WARM])

    # ---------- cProfile：训练 ----------
    chunk = train_txt[N_WARM:N_WARM + N_TRAIN * 2]  # 足量切片
    pr = cProfile.Profile()
    pr.enable()
    lm.train_stream(chunk[:N_TRAIN])
    pr.disable()

    # ---------- cProfile：评估 ----------
    eval_txt = text[n_train_chars:n_train_chars + N_EVAL * 3]
    pr2 = cProfile.Profile()
    pr2.enable()
    lm.evaluate(eval_txt)
    pr2.disable()

    for tag, prof, sortkeys in (("训练 train_stream", pr, ("tottime", "cumulative")),
                                ("评估 evaluate", pr2, ("tottime", "cumulative"))):
        print(f"\n{'=' * 78}\n cProfile — {tag}\n{'=' * 78}")
        for sk in sortkeys:
            s = io.StringIO()
            pstats.Stats(prof, stream=s).sort_stats(sk).print_stats(22)
            out = s.getvalue()
            # 去掉头部冗余行，压缩输出
            lines = out.splitlines()
            keep = [ln for ln in lines if ln.strip()][:34]
            print("\n".join(keep))

    # ---------- 手工分相计时（wrap 关键方法） ----------
    print(f"\n{'=' * 78}\n 分相计时（wrap 实测，训练 {N_TRAIN} tokens）\n{'=' * 78}")
    net = lm.net

    def wrap(obj, name, bucket):
        orig = getattr(obj, name)
        st = {"t": 0.0, "n": 0}

        def wrapper(*a, **kw):
            t0 = time.perf_counter()
            r = orig(*a, **kw)
            st["t"] += time.perf_counter() - t0
            st["n"] += 1
            return r
        setattr(obj, name, wrapper)
        bucket[name] = st
        return orig

    phases: dict[str, dict] = {}
    wrap(lm.tok, "encode_composite", phases)
    wrap(lm.tok, "onehot", phases)
    wrap(net.encoder, "encode", phases)
    wrap(net.pc, "infer", phases)
    wrap(net.pc, "learn", phases)
    wrap(net.stdp, "predict", phases)
    wrap(net.stdp, "step", phases)
    wrap(net.wm, "decay", phases)
    wrap(net.wm, "write", phases)
    wrap(net.wm, "read", phases)
    wrap(net.ltm, "imprint", phases)
    wrap(net.ltm, "recall", phases)
    wrap(net.readout, "__call__", phases)
    wrap(net.readout, "learn_softmax", phases)
    wrap(net, "step", phases)

    lm.train_stream(chunk[N_TRAIN:N_TRAIN * 2])   # 再来 N_TRAIN tokens

    total = phases.pop("step")
    print(f"net.step 总计: {total['t'] * 1000:.1f} ms / {total['n']} 步 = "
          f"{total['t'] / max(1, total['n']) * 1000:.3f} ms/token\n")
    print(f"{'环节':<20}{'累计 ms':>12}{'次数':>10}{'us/次':>12}{'占 step%':>10}")
    rows = sorted(phases.items(), key=lambda kv: -kv[1]["t"])
    for name, st in rows:
        us_per = st["t"] / max(1, st["n"]) * 1e6
        pct = st["t"] / max(total["t"], 1e-9) * 100
        print(f"{name:<20}{st['t'] * 1000:>12.1f}{st['n']:>10}{us_per:>12.1f}{pct:>9.1f}%")

    covered = sum(st["t"] for st in phases.values())
    print(f"\n已覆盖环节合计: {covered * 1000:.1f} ms "
          f"({covered / max(total['t'], 1e-9) * 100:.1f}% of step) "
          f"—— 差值为 step 内 Python 编排开销")

    # ---------- 静态参数规模 ----------
    print(f"\n{'=' * 78}\n 参数规模（float64 字节数）\n{'=' * 78}")
    for name, arr in (("encoder.W", net.encoder.W), ("pc.W_up0", net.pc.W_up0),
                      ("pc.W_up1", net.pc.W_up1), ("pc.W_dn0", net.pc.W_dn0),
                      ("pc.W_dn1", net.pc.W_dn1), ("stdp.W", net.stdp.W),
                      ("readout.W", net.readout.W)):
        print(f"{name:<12} {str(arr.shape):<14} {arr.nbytes / 1e6:>8.2f} MB")

    OUT.write_text("", encoding="utf-8")  # 占位：本脚本以 stdout 为准


if __name__ == "__main__":
    main()
