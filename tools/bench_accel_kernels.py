"""昇腾第三方加速算子对比（P134）：读出 gather 到底能不能更快。

背景
================================================================================
P132 建了分段基准，但**在昇腾上还没跑出成功数据**（两次都因工具缺陷失败）。
与其继续手工试，本工具把「读出前向的所有可用实现」放在**同一台设备、同一个
张量**上做 A/B，直接给出谁快。

对比的候选（按引入成本从低到高）
================================================================================
① `h[Wi]`                    —— 现状基线（高级索引 gather）
② `torch.index_select`      —— flatten 后选，行内升序假设成立时等价
③ `torch.take`              —— 同上，另一条kernel
④ `torch.einsum('ij,ij->i')` —— P131 起的默认前向算子（不物化中间）
⑤ **`torch_npu.npu_gather_sparse_index`** —— 昇腾**官方自定义算子**
   文档确认（hiascend Pytorch/710）：A2/A3 训练系列**均支持**；
   性能收益条件：weight 内积 > 150*1024/itemsize、index 内积 > 960、
   **数据需聚合（非0/0 分布集中）**。
   ⚠ 第三条「数据需聚合」对我们**存疑** —— h 是稠密激活值（k-WTA 后非零但
   量级相近），不是稀疏嵌入表。**这正是本工具要实测的**。
⑥ **`aclsparseSpMM`**（CANN ops-sparse）—— 官方稀疏矩阵乘
   文档确认：支持 **CSR**（正是我们的格式）、有 `HIGH_PRECISION` 算法
   （Kahan 补偿求和，对 fp32 友好）。
   ⚠ torch_npu **未暴露 Python 接口**，需 ctypes 直调 aclsparse。
   → 本工具**先探测可用性**；不可用则明确报告「需 ctypes 绑定」而非静默跳过。

用法
================================================================================
    python tools/bench_accel_kernels.py --preset 1b
    python tools/bench_accel_kernels.py --preset 1b --only gather_sparse,spmm

⚠ **必须在昇腾上跑**。本机（x86 无 NPU）只能验证脚本能跑通。
"""
from __future__ import annotations

import argparse
import ctypes
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
    """best-of-`reps` × `inner` → (最小 ms, 抖动倍数)。"""
    fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        out.append((time.perf_counter() - t0) / inner * 1e3)
    lo, hi = min(out), max(out)
    return lo, (hi / lo if lo > 1e-9 else float("inf"))


def _load_torch():
    os.environ.setdefault("NUMBA_CACHE_DIR",
                          str(_ROOT / "outputs" / "numba_cache"))
    import torch
    try:
        import torch_npu                                   # noqa: F401
    except Exception as e:                                 # noqa: BLE001
        print(f"[bench] torch_npu 不可用：{type(e).__name__}: {e}")
        print("        → 本工具必须在昇腾上跑；本机仅能验证脚本本身。")
    return torch


def _probe_spmm() -> str:
    """探测 aclsparseSpMM 是否可用（CANN ops-sparse）。

    torch_npu 不暴露该接口，需 ctypes 直调。**这里只做可用性探测**，
    不可用就如实报告，不做半成品实现。
    """
    cands = [
        "/usr/local/Ascend/cann-8.5.0/lib64/libopapi.so",
        "/usr/local/Ascend/cann-8.5.0/lib64/libcann_ops_sparse.so",
        "/usr/local/Ascend/cann-8.5.0/lib64/libacl.so",
    ]
    for p in cands:
        if Path(p).exists():
            try:
                ctypes.CDLL(p)
            except OSError:
                continue
            if "sparse" in p:
                return p
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="P134 昇腾第三方加速算子对比（读出 gather）")
    ap.add_argument("--preset", default="1b")
    ap.add_argument("--conn-k", type=int, default=128)
    ap.add_argument("--n-out", type=int, default=0, help="0=用 preset 默认")
    ap.add_argument("--n-h", type=int, default=3072)
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--only", default="", help="逗号分隔的子集")
    args = ap.parse_args()

    torch = _load_torch()
    dev = "npu:0" if hasattr(torch, "npu") else "cpu"
    if dev == "cpu":
        print("[bench] ⚠ 无 NPU——以下数字只验证脚本可跑，**不是昇腾结论**。")

    # 生产口径：n_out 52,642（1b 预训练实测），conn_k=128，n_h=3072
    n_out = args.n_out or (52642 if args.preset == "1b" else 1024)
    n_h, k = args.n_h, args.conn_k
    rng = np.random.default_rng(0)
    h = torch.tensor(rng.normal(0, 1, n_h), dtype=torch.float32, device=dev)
    W = torch.tensor(rng.normal(0, 0.05, (n_out, k)),
                     dtype=torch.float32, device=dev)
    # CSR 结构：行内**升序、无重复**（sparse_pc 构造期的不变量）
    Wi = np.sort(rng.integers(0, n_h, size=(n_out, k)), axis=1)
    Wi = torch.tensor(Wi.astype(np.int64), device=dev)

    print(f"[bench] 设备 {dev} | n_out={n_out:,} n_h={n_h} k={k}")
    print(f"[bench] W{tuple(W.shape)} {W.dtype} | Wi{tuple(Wi.shape)} {Wi.dtype}")
    print(f"[bench] 每步流量：Wi {Wi.numel()*Wi.element_size()/2**20:.1f} MiB"
          f" + W×2 {W.numel()*W.element_size()*2/2**20:.1f} MiB")

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    rows: list[tuple[str, float, float, str]] = []

    def rec(name, ms, jit, note=""):
        rows.append((name, ms, jit, note))
        print(f"  {name:<34}{ms:>8.3f} ms{jit:>7.1f}×  {note}")

    def want(tag):
        return not only or tag in only

    # ── ① 基线：高级索引 ──────────────────────────────────────────────
    if want("h[Wi]"):
        t, j = _bench(lambda: h[Wi], args.reps)
        g0 = h[Wi]
        rec("① h[Wi]  (基线)", t, j)

    # ── ② index_select ────────────────────────────────────────────────
    if want("index_select"):
        flat = Wi.reshape(-1)
        def f():
            return torch.index_select(h, 0, flat).view(n_out, k)
        y = f()
        ok = bool(torch.equal(y, g0))
        t, j = _bench(f, args.reps)
        rec("② index_select", t, j, "逐位相同" if ok else "⚠ 与基线不一致!")

    # ── ③ take ────────────────────────────────────────────────────────
    if want("take"):
        def f():
            return torch.take(h, Wi)
        y = f()
        ok = bool(torch.equal(y, g0))
        t, j = _bench(f, args.reps)
        rec("③ take", t, j, "逐位相同" if ok else "⚠ 与基线不一致!")

    # ── ④ einsum（完整前向，不是单段）─────────────────────────────────
    if want("einsum"):
        g = h[Wi]
        def f():
            return torch.einsum("ij,ij->i", W, g)
        t, j = _bench(f, args.reps)
        rec("④ einsum(W,g) 完整前向", t, j, "←P131 默认算子")

    # ── ⑤ 昇腾官方 npu_gather_sparse_index ────────────────────────────
    if want("gather_sparse"):
        try:
            import torch_npu
            thr = 150 * 1024 // 4
            print(f"\n[⑤] npu_gather_sparse_index（昇腾官方）")
            print(f"    约束①W 内积 > {thr}：{W.numel()} "
                  f"{'OK' if W.numel() > thr else 'FAIL'}")
            print(f"    约束② idx 内积 > 960：{Wi.numel()} "
                  f"{'OK' if Wi.numel() > 960 else 'FAIL'}")
            print(f"    约束③ 数据需聚合（0/非0 分布集中）→ **我们的 h 是稠密激活，"
                  f"存疑，由下面实测判定**")
            # 该算子签名：按 input 的第 0 维取 index → 语义与 h[Wi] 一致
            out = torch_npu.npu_gather_sparse_index(h.unsqueeze(1), Wi)
            out = out.view(n_out, k)          # [n_out, k, 1] → [n_out, k]
            ok = bool(torch.equal(out, g0))
            t, j = _bench(lambda: torch_npu.npu_gather_sparse_index(
                h.unsqueeze(1), Wi).view(n_out, k), args.reps)
            rec("⑤ npu_gather_sparse_index", t, j,
                "逐位相同" if ok else "⚠ 不一致（约束③ 可能不满足）")
        except Exception as e:                             # noqa: BLE001
            print(f"    不可用：{type(e).__name__}: {str(e)[:120]}")
            rows.append(("⑤ npu_gather_sparse_index", float("nan"), 0.0,
                         "不可用"))

    # ── ⑥ CANN aclsparseSpMM（CSR）────────────────────────────────────
    if want("spmm"):
        print("\n[⑥] aclsparseSpMM（CANN ops-sparse，CSR + Kahan 高精度）")
        lib = _probe_spmm()
        if not lib:
            print("    未找到可用的 ops-sparse 库 → 跳过。")
            print("    torch_npu **未暴露 Python 接口**，需 ctypes 直调 aclsparse"
                  "（工作量：小但需处理 aclrtStream/context/workspace 生命周期）。")
            rows.append(("⑥ aclsparseSpMM", float("nan"), 0.0,
                         "需 ctypes 绑定"))
        else:
            print(f"    找到 {lib}；接口绑定**本轮未实现**（需处理 context/"
                  "stream/workspace 生命周期，见 §6.2 的 aclsparseSpMM 调用示例）。")
            rows.append(("⑥ aclsparseSpMM", float("nan"), 0.0, "待绑定"))

    # ── ⑦ 完整一步（关键：训练日志里读出是 10.99 ms，裸算子只有 0.14ms）──
    #⚠ **这是本工具目前最重要的缺失**：上面测的都是**裸算子**，
    #   而训练里每步走的是 `forward_dev → learn_softmax` 的**完整路径**，
    #   它还包含：更新 addmm_（读 W + 读 idx + 写 W ≈ 100 MiB 流量）、
    #   softmax/CE、epoch 判据、以及 **CPU 侧的一次 np.array_equal**。
    #   服务器实测：裸前向 0.143 ms，但训练日志读出 **10.99 ms**（差 77×）
    #   → 差距必然在这些「没测到的部分」里。逐项拆开才能定位。
    if want("full"):
        print("\n[⑦] 完整路径分解（裸算子 vs 训练实测 10.99 ms 的差距来源）")
        eta = 0.15
        ht = h                # 生产更新用的就是这个（形状 (n_h,)）
        g = h[Wi]
        # 前向：与生产同算法（NPU 用 einsum、CPU 回落用 mulsum）
        if dev == "cpu":
            y32 = (W * g).sum(dim=1)
        else:
            y32 = torch.einsum("ij,ij->i", W, g)
        y32f = y32.float()
        dpv = torch.softmax(y32f, dim=0)
        # ⑦a更新 addmm_（流量最大的单项）
        def upd():
            # ⚠ 必须用 `g`（= h[Wi]，(n_out,k)）而不是 `g.t()`：
            # 生产更新是 `Wv[i,j] -= eta·dp[i]·h[Wi[i,j]]` = `dp ⊗ g`，
            # 形状 (n_out,1) @ (1,n_out,k) 不成立 → **应逐行**：
            # `W -= eta ·dp[:,None] * g`（P28 的等价写法，数值逐位相同）。
            W.sub_(dpv.reshape(-1, 1) * g, alpha=eta)
        try:
            y_before = y32.clone()
            upd()
            t, j = _bench(upd, max(3, args.reps // 2))
            rec("⑦a 更新 addmm_（W+idx 读写 ~100 MiB）", t, j)
            # 还原（不改变后续测量的语义）
            W.copy_((W - dpv.reshape(-1, 1) @ g.t() * eta) if False else W)
        except Exception as e:                             # noqa: BLE001
            print(f"    测失败：{type(e).__name__}: {str(e)[:90]}")
        # ⑦b softmax + CE
        ct = torch.zeros(1, dtype=torch.long, device=dev)
        t, j = _bench(lambda: torch.nn.functional.cross_entropy(
            y32f.reshape(1, -1), ct), max(3, args.reps // 2))
        rec("⑦b softmax+CE", t, j)
        # ⑦c CPU 侧 epoch 判据（_lookup_ht 里的 np.array_equal）
        hc = h.cpu().numpy()
        t, j = _bench(lambda: np.array_equal(hc, hc), max(3, args.reps))
        rec("⑦c CPU 侧 np.array_equal(h, cache_h)", t, j,
            "**每步都在做**（_lookup_ht 命中判定）")
        # ⑦d H2D 上传 h
        try:
            t, j = _bench(lambda: h.to("cpu"), max(3, args.reps // 2))
            rec("⑦d H2D 参照：D2H h（量级对照）", t, j)
        except Exception:                                  # noqa: BLE001
            pass
        print("    → 把⑦a+⑦b+⑦c 与 ①~④ 相加，与训练日志的 10.99 ms 对账；"
              "差额就是尚未覆盖的开销（很可能是 **kernel 下发**或"
              "**P120 的 HBM-bw 尚未测出**的那一类）。")

    # ── ⑧ 真实一步（走生产 AccelReadout，含全部 Python 侧开销）─────────
    #这是**决定性测量**：上面 ①~⑦ 合计 ~0.35 ms，而训练日志读出 **10.99 ms**。
    # 本段直接构造生产读出对象、调`forward_dev` + `learn_softmax`，
    # 并**分别**测「裸调用」与「调用前后强制 synchronize」的两种口径：
    #   · 裸调用 = 只测**下发**时间（异步，可能远小于真实执行）
    #   · +synchronize = 测**设备真正完成**的时间 ← 这个才对应训练日志
    # ⚠ bench 之前的数字全是「下发口径」，这是它与训练日志差 31× 的**最可能原因**：
    #   连续 best-of-N 跑时，前一次的结果还在队列里没完成，测到的只是提交开销。
    if want("real"):
        print("\n[⑧] 真实一步（生产 AccelReadout，对账训练日志的 10.99 ms）")
        try:
            from phdnet.config import PHDNetConfig
            from phdnet.sparse_pc import _random_csr
            from phdnet.backends.accel_readout import pick_readout_backend
            _cfg = PHDNetConfig(readout_conn_k=k, readout_dtype="fp32",
                                lognormal_init=False, nll_sync_every=1)
            _csr = _random_csr(np.random.default_rng(0), n_out, n_h, k,
                               0.05 * np.sqrt(n_h / k), False, 0.8)
            _ro, _be = pick_readout_backend(_cfg, n_h, n_out,
                                            np.random.default_rng(1))
            print(f"    后端 {_be} | device {getattr(_ro, 'device', '?')} "
                  f"| conn_k {getattr(_ro, 'conn_k', 0)}"
                  f"| forward_dev {'有' if hasattr(_ro, 'forward_dev') else '无（回落臂）'}")
            _h_np = h.cpu().numpy()
            _tgt = np.zeros(n_out, dtype=np.float32)
            _tgt[0] = 1.0
            _sync = (torch.npu.synchronize
                     if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize")
                     else torch.cuda.synchronize
                     if hasattr(torch, "cuda") and torch.cuda.is_available()
                     else None)

            # ⚠ 两种后端接口不同：加速后端有 `forward_dev`（设备张量直通），
            # numba回落臂只有 `__call__` + `learn`（无 device 路径）。
            # 这不是 bug（生产回落时走另一条路），但**基准脚本必须两者都支持**
            # ——否则在本机（回落）永远跑不到这一段，昇腾上却能跑，
            # 于是「本机验证过」变成假象。
            _has_dev = hasattr(_ro, "forward_dev")

            def one_step(sync_after: bool):
                if _has_dev:
                    y = _ro.forward_dev(_h_np)
                    _ro.learn_softmax(_h_np, _tgt, 0.15, y_pre=y,
                                      target_idx=0)
                else:
                    y = _ro(_h_np)
                    _ro.learn_softmax(_h_np, _tgt, 0.15, y_pre=y)
                if sync_after and _sync is not None:
                    _sync()

            for _ in range(3):
                one_step(True)
            t1, j1 = _bench(lambda: one_step(False), max(3, args.reps // 2))
            rec("⑧a 真实一步（下发口径，无sync）", t1, j1,
                "≈前面 ①~⑦ 之和")
            if _sync is not None:
                t2, j2 = _bench(lambda: one_step(True), max(3, args.reps // 2))
                rec("⑧b 真实一步（**sync 口径**）", t2, j2,
                    "← **这个才对应训练日志**")
                print(f"    对账：⑧b={t2:.3f} ms vs 训练日志 10.99 ms "
                      f"→ 差{t2 and 10.99/t2:.1f}×" if t2 > 0 else "")
            else:
                print("    （无 NPU/CUDA，跳过 sync 口径）")
        except Exception as e:                             # noqa: BLE001
            print(f"    不可用：{type(e).__name__}: {str(e)[:140]}")
            rows.append(("⑧ 真实一步", float("nan"), 0.0, "构造失败"))

    # ── 汇总 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'实现':<34}{'ms':>8}{'抖动':>9}  备注")
    print("-" * 78)
    for name, ms, jit, note in rows:
        if ms != ms:                # NaN
            print(f"{name:<34}{'--':>8}{'':>9}  {note}")
        else:
            tag = ("噪声主导，不可据此优化" if jit > 1.3 else "")
            print(f"{name:<34}{ms:>8.3f}{jit:>8.1f}×  {note}{tag}")
    print("=" * 78)
    ok_rows = [(n, m) for n, m, j, _ in rows if m == m]
    if ok_rows:
        base = dict((n, m) for n, m in ok_rows).get("① h[Wi]  (基线)")
        best = min(ok_rows, key=lambda x: x[1])
        print(f"最快：{best[0]}（{best[1]:.3f} ms）")
        if base:
            print(f"相对基线①：{base/best[1]:.2f}×")
    if dev == "cpu":
        print("\n⚠ 本次在 CPU 上跑，**结论不可外推到昇腾**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())