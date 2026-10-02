"""读出**分段**基准：定位 10.99 ms 到底耗在哪一段（P132）。

为什么需要这个
================================================================================
P131 之前的所有读出分析都只有**一个**端到端数字（10.99 ms/tok），而它对应
的带宽下界是 ~0.28 ms（按 ~800 GB/s 估算）—— 中间有 **~39 倍**的差距，
无法判断差距来自哪一段。继续在 x86 上猜已经失败三次：

* 「einsum 快 21%」→ 实测是**单次测量噪声**（best-of-5 抖动 1.1×）
* 「分块能让 gather 进 L2」→ 实测 **0.83–0.91×（更慢）**
* 「idx 换 int32 省流量」→ x86 上**慢 0.58×**（要类型转换）

⚠ 这些都不是昇腾上的结论。本工具的**唯一目的**是：在**昇擎真机**上把
读出的每一步拆开测，让优化对准真正的瓶颈，而不是对准x86 的行为。

测什么
================================================================================
按数据依赖顺序逐段计时（每段 best-of-N，避免单次噪声）：

  ①H2D 上传 h         0.03 MiB，应远小于 0.05 ms
  ② gather  `h[Wi]`       665万次随机取 ← 理论最大单项
  ③ mulsum `(W*g).sum(1)`物化 25.37 MiB 中间
  ④ einsum  `einsum(...)`  同上但不物化
  ⑤ 前向合计（②+③ / ②+④）
  ⑥ softmax + cross_entropy
  ⑦ 更新 addmm_（读 W + 读 idx + 写 W）
  ⑧ 单步总计 ①..⑦

用法
================================================================================
    # 昇腾上（生产口径，1b 档）
    python tools/bench_readout_segments.py --preset 1b

    # 也可比两种前向算子
    python tools/bench_readout_segments.py --preset 1b --kernels mulsum,einsum

    # 本机 x86 也能跑（只作相对参考，不能当昇腾结论）
    python tools/bench_readout_segments.py --preset smoke

⚠ **每段都要看 best-of-N 的抖动**（`jit` 列）。抖动 >1.3× 说明该段被噪声
主导，那一段的结论不可用 —— 先加大 --reps 再看。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "train")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _bench(fn, reps: int, inner: int = 5) -> tuple[float, float]:
    """best-of-`reps` × `inner` 次 → (最小值 ms, 抖动倍数)。"""
    fn()                                    # 预热（含 JIT/首次分配）
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        out.append((time.perf_counter() - t0) / inner * 1e3)
    lo, hi = min(out), max(out)
    return lo, (hi / lo if lo > 1e-9 else float("inf"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="P132 读出分段基准（定位 10.99 ms 耗在哪一段）")
    ap.add_argument("--preset", default="smoke",
                    help="1b = 生产口径；smoke = 本机小档")
    ap.add_argument("--conn-k", type=int, default=0, help="0 = 用 preset 默认")
    ap.add_argument("--kernels", default="mulsum,einsum",
                    help="要测的前向算子，逗号分隔")
    ap.add_argument("--reps", type=int, default=9, help="best-of-N 的 N")
    ap.add_argument("--device", default="", help="留空=auto")
    args = ap.parse_args()

    os.environ.setdefault("NUMBA_CACHE_DIR", str(_ROOT / "outputs" / "numba_cache"))
    from config_1b import PRESETS, SEG_KWARGS, build_cfg
    from phdnet.word_lm import PHDWordLM
    from phdnet.config import PHDNetConfig

    # 构词表：用**已训练模型的 ckpt**（若存在）否则用快照词表 + 完整
    # WordTokenizer 路径。⚠ 不手工拼装 tokenizer（其私有字段 n_sdr/seed 等
    # 会随版本漂移 → P129 那类脆弱依赖）。
    import json
    from phdnet.word_lm import PHDWordLM
    snap = sorted((_ROOT / "outputs").rglob("vocab_*_pretrain.json"))
    if not snap:
        print("[bench] 找不到词表快照（outputs/**/vocab_*pretrain.json）；"
              "先跑一次训练或 tools/build_vocab.py")
        return 2
    meta = json.loads(snap[-1].read_text(encoding="utf-8"))
    words = meta["words"] if isinstance(meta, dict) else meta
    print(f"[bench] 词表快照 {snap[-1].name} → {len(words):,} 词")

    cfg = build_cfg("1b" if args.preset == "1b" else args.preset)
    if args.conn_k > 0:
        cfg = PHDNetConfig(**{**cfg.__dict__, "readout_conn_k": args.conn_k})
    # 最小语料：仅用于让 tokenizer 走完构造流程（段间字符）
    lm = PHDWordLM("".join(words[:4000]) or "test", cfg, seg_kwargs=SEG_KWARGS)
    tok = lm.tok
    n_out = len(tok)
    ro = lm.net.readout
    backend = getattr(lm.net, "_readout_backend", "?")

    n_h = cfg.n_top * (3 if cfg.pred_in_readout else 2)
    print(f"[bench] 后端 {backend} | n_out={n_out:,} n_h={n_h:,} "
          f"conn_k={getattr(ro, 'conn_k', 0)} dtype={getattr(ro, 'dtype_name', '?')}")

    dev = getattr(ro, "device", None)
    if dev is None:
        print("[bench] ⚠ 读出在 CPU（回落路径）—— 分段数据只作相对参考")
    torch = ro.__class__.__module__
    import torch as _t

    rng = np.random.default_rng(0)
    h = rng.normal(0, 1, n_h).astype(np.float32)
    W = ro.W
    # 加速后端 W/Wi 是 torch 张量，回落路径是 numpy → 统一算字节数
    def _nbytes(t) -> int:
        # torch 张量的 `nbytes` 是 **int 属性**；numpy 的 `nbytes` 是**方法**。
        # （先试element_size 的写法失败过：numpy 也有同名方法。）
        nb = getattr(t, "nbytes", None)
        return nb if isinstance(nb, int) else int(t.nbytes())

    # 稀疏臂才有 Wi；稠密回落路径（numba Readout）用 CSR 的 idx
    Wi = getattr(ro, "Wi", None)
    if Wi is None:
        _csr = getattr(ro, "_csr", None)
        Wi = _t.as_tensor(_csr[1]) if _csr is not None else None
    if Wi is not None:
        print(f"[bench] W{tuple(W.shape)} idx{tuple(Wi.shape)} "
              f"（每步流量：idx {_nbytes(Wi)/2**20:.1f} MiB "
              f"+ W×2 {_nbytes(W)*2/2**20:.1f} MiB）")
    else:
        print(f"[bench] W{tuple(W.shape)}（稠密回落，无列索引）")

    ht = ro._staged_to_dev(h) if hasattr(ro, "_staged_to_dev") else h
    g = ro._sp_gather(ht) if hasattr(ro, "_sp_gather") else None
    # 回落路径（numba Readout）没有 `_matmul`（那是加速后端的内部方法），
    # 用公共 `__call__`。它对两种后端都可用。
    y = _t.as_tensor(ro(h))
    y32 = y.float()
    dp = _t.softmax(y32, dim=0)

    rows = []

    def rec(name, ms, jit):
        rows.append((name, ms, jit))

    # 稀疏臂才有 Wi / W 布局差异；稠密回落路径 W 是 (n_out, n_in)
    _is_sparse = hasattr(ro, "_sp_gather")

    for kern in [k.strip() for k in args.kernels.split(",") if k.strip()]:
        if hasattr(ro, "_sp_fwd"):
            ro._sp_fwd = kern
        print(f"\n--- 前向算子: {kern} ---")
        if hasattr(ro, "_staged_to_dev"):
            t, j = _bench(lambda: ro._staged_to_dev(h), args.reps)
            rec(f"[{kern}] ① H2D 上传 h", t, j)
        if _is_sparse:
            t, j = _bench(lambda: ro._sp_gather(ht), args.reps)
            rec(f"[{kern}] ② gather h[Wi]", t, j)
            if kern == "einsum":
                t, j = _bench(lambda: _t.einsum("ij,ij->i", W, g), args.reps)
                rec(f"[{kern}] ④ einsum(不物化)", t, j)
            else:
                t, j = _bench(lambda: (W * g).sum(dim=1), args.reps)
                rec(f"[{kern}] ③ mul+sum(物化)", t, j)
        t, j = _bench(lambda: ro(h), args.reps)
        rec(f"[{kern}] ⑤ 前向合计", t, j)

    print("\n--- 后向段（与前向算子无关）---")
    ct = _t.zeros(1, dtype=_t.long)
    t, j = _bench(lambda: _t.nn.functional.cross_entropy(
        y32.reshape(1, -1), ct.to(y32.device)).reshape(()), args.reps)
    rec("⑥ softmax+CE", t, j)
    eta = 0.15
    if _is_sparse and hasattr(W, "addmm_"):
        t, j = _bench(lambda: W.addmm_(dp.reshape(-1, 1), ht.reshape(1, -1),
                                      alpha=-eta), args.reps)
        rec("⑦ 更新 addmm_（稀疏臂）", t, j)
    else:
        # 稠密臂的更新是 W -= eta·dp⊗h（外积），语义等价
        t, j = _bench(lambda: W.__setitem__(
            Ellipsis, W - eta * np.outer(dp.cpu().numpy(), h)), args.reps)
        rec("⑦ 更新（W -= eta·dp⊗h，稠密臂）", t, j)

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'段':<34}{'ms':>9}{'抖动':>8}  判读")
    print("-" * 78)
    for name, ms, jit in rows:
        tag = "噪声主导，不可据此优化" if jit > 1.3 else (
            "← 瓶颈" if ms > 1.0 else "")
        print(f"{name:<34}{ms:>9.3f}{jit:>7.1f}×  {tag}")
    print("=" * 78)
    big = [(n, m) for n, m, _ in rows if m > 1.0]
    if big:
        big.sort(key=lambda x: -x[1])
        print("最大段：" + ", ".join(f"{n}({m:.2f} ms)" for n, m in big[:3]))
        print("→ 优化应对准这些段；抖动 >1.3× 的段先加大 --reps 重测。")
    else:
        print("没有单段超过 1 ms → 瓶颈不在算子内部，"
              "更可能是 kernel 调度/下发（看 HBM-bw 与 AICore）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())