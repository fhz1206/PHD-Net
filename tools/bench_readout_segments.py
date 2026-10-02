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
    ap.add_argument("--ckpt", default="",
                    help="生产 ckpt 路径（含 tok_vocab + ro_ip/idx/val）。"
                         "默认取 outputs/models/*1b*pretrain*.npz。"
                         "P133：词表与 CSR 必须来自 ckpt，**不能**用快照重建")
    args = ap.parse_args()

    os.environ.setdefault("NUMBA_CACHE_DIR", str(_ROOT / "outputs" / "numba_cache"))
    from config_1b import PRESETS, SEG_KWARGS, build_cfg
    from phdnet.config import PHDNetConfig
    from phdnet.backends.accel_readout import pick_readout_backend

    # ⚠ P133：数据来源改为**生产 ckpt**，不再从词表快照重建。
    # 之前用 `PHDWordLM("".join(words[:4000]))` 重建 → tokenizer 会**重新分词**，
    # 得到一个与生产无关的小词表（服务器实测 n_out=**1021** vs 真实 51,962，
    # 且 conn_k 退化成 0）→ 整场测的是另一个模型，结论全无意义。
    # ckpt 里带齐了真实结构：`tok_vocab`（词表）、`ro_ip/ro_idx/ro_val`（稀疏 CSR）、
    # 以及各权重（据此可反推 n_sdr/n_top/pred_in_readout → n_h）。
    import json
    import numpy as _np
    ckpts = sorted((_ROOT / "outputs" / "models").glob("*1b*pretrain*.npz"))
    if not ckpts:
        ckpts = sorted((_ROOT / "outputs").rglob("*.npz"))
    if not ckpts:
        print("[bench] 找不到任何 ckpt（outputs/**/vocab_*.npz 旁的 *.npz）；"
              "先跑一次训练或用 --ckpt 指定")
        return 2
    ck_path = Path(args.ckpt) if args.ckpt else ckpts[-1]
    print(f"[bench] ckpt {ck_path.name}")
    z = _np.load(ck_path, allow_pickle=True)
    keys = set(z.keys())

    # ① 词表（决定 n_out）
    n_out = None
    if "tok_vocab" in keys:                # 项目权威口径（docs: tok_vocab≠tok_tokens）
        n_out = int(z["tok_vocab"].shape[0])
        print(f"[bench] 词表来自 ckpt['tok_vocab'] → n_out = {n_out:,}")
    else:
        snap = sorted((_ROOT / "outputs").rglob("vocab_*pretrain.json"))
        if not snap:
            print("[bench] ckpt 里没有 tok_vocab，且找不到词表快照")
            return 2
        meta = json.loads(snap[-1].read_text(encoding="utf-8"))
        words = meta["words"] if isinstance(meta, dict) else meta
        n_out = len(words)
        print(f"[bench] ckpt 无 tok_vocab → 用快照 {snap[-1].name}（n_out={n_out:,}）")

    # ② 由 CSR 反推 conn_k与 n_h（idx 的最大值上界即 n_h）
    conn_k = int(args.conn_k) if args.conn_k > 0 else 0
    n_h = None
    if "ro_ip" in keys and "ro_idx" in keys:
        _ip = z["ro_ip"].astype(_np.int64)
        _idx = z["ro_idx"].astype(_np.int64)
        n_rows = int(_ip.shape[0] - 1)
        if n_rows == n_out and _idx.size:
            conn_k = int(_idx.size // max(1, n_rows))
            n_h = int(_idx.max()) + 1
            print(f"[bench] 稀疏 CSR: rows={n_rows:,} conn_k={conn_k} "
                  f"nnz={_idx.size:,} → n_h>= {n_h}（由 idx 上界反推）")

    # ③ 若 ckpt 带栈配置，直接读；否则用 preset 补
    cfg = build_cfg("1b" if args.preset == "1b" else args.preset)
    if conn_k > 0:
        cfg = PHDNetConfig(**{**cfg.__dict__, "readout_conn_k": conn_k})
    if n_h:
        # pred_in_readout=2/3 通路，决定 n_h = n_top * (2 or 3)
        _ntop = max(1, n_h // 3)
        if n_h % 2 == 0 and (n_h // 2) % 3 != 0:
            _ntop = n_h // 2
        cfg = PHDNetConfig(**{**cfg.__dict__, "n_top": _ntop,
                              "pred_in_readout": bool(n_h % 3 == 0)})
    #⚠ n_h 的兜底：ckpt 的 ro_idx 反推只在「rows 恰等于 n_out」时成立。
    # 不成立（如本机 smoke ckpt n_out=637 而 CSR rows=2611）时n_h 仍为 None，
    # 后面用它造 h 会炸。→ 统一在此定为整数，并打印实际值供核对。
    _n_h_eff = int(n_h) if n_h else int(
        cfg.n_top * (3 if cfg.pred_in_readout else 2))
    n_h = _n_h_eff
    ro, backend = pick_readout_backend(cfg, n_h, n_out,
                                      _np.random.default_rng(cfg.seed))
    print(f"[bench] 后端 {backend} | n_out={n_out:,} n_h={n_h or cfg.n_top*3:,} "
          f"conn_k={getattr(ro, 'conn_k', 0)} dtype={getattr(ro, 'dtype_name', '?')}")
    if getattr(ro, "conn_k", 0) == 0:
        print("[bench] ⚠ conn_k=0（稠密）——分段数据与生产稀疏路径**不同口径**。")
        print("        若要测生产路径，请确认 ckpt 是 1b 预训练 ckpt（含 ro_ip/idx）。")


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

    # 稀疏臂才有 Wi；稠密回落路径（numba Readout）用 CSR 的 idx。
    # ⚠⚠ **不能用 `getattr(ro, "_csr", None)`**：`_csr` 是 **property**，稠密
    # 模式下它**抛 NotImplementedError**（见 accel_readout.py:651）→ getattr
    # 的默认值**不生效**，异常直接冒出来。这正是 2026-10-02 14:48 服务器报的错。
    # → 必须用 try/except 包住。
    Wi = getattr(ro, "Wi", None)
    if Wi is None:
        try:
            _csr = ro._csr                              # noqa: SLF001
        except (NotImplementedError, AttributeError):
            _csr = None
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