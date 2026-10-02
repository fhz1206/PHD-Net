"""P137：生产读出路径的**逐行**打点，定位那~14.7 ms。

为什么需要这个
================================================================================
P135/P136 的结论把问题逼到了一个点上（昇腾实测，1b 档真实尺寸）：

  ①~⑦ 裸算子合计                       ≈ 0.35 ms
  ⑧a 真实一步（下发口径）              15.147 ms   抖动 1.0×
  ⑧b 真实一步（sync 口径）             15.204 ms   抖动 1.0×
  训练日志 readout                     10.99 ms

**下发口径 ≈ 完成口径** → 瓶颈**不是**异步等待/ 下发队列（我 P135 的猜测被否）。
**⑧b ≈ 训练日志**（差 0.7×，bench 略高因为含更多 Python 开销、无 CPU/NPU 重叠）
→ ⑧ 测的就是真实成本。
→ **那 14.8 ms 花在 ①~⑦ 未覆盖的代码里**。裸算子测不到，只能在**生产函数内部**
   逐行打点。

本工具做什么
================================================================================
用 `sys.settrace` 只对**读出类的两个方法**做逐行计时（其余函数不trace，
开销可忽略），把每行归到「函数 × 行」并累加耗时。输出 top-N 最贵行，
带源码行文本，直接指出「哪一行吃掉了 14 ms」。

局限（必须诚实说）
================================================================================
- `settrace` 本身有开销（本机实测约 0.1~0.3 ms/行），所以**绝对值偏大**，
  只看**相对占比**与「哪行最大」；本机跑不出 15 ms（无 NPU），
  **必须在昇腾上跑**才有意义。
- 逐行计时会把「该行触发的异步下发」算在该行上，设备真正完成时间仍要靠
  末尾的 sync 口径对账。
- CPython 专用（依赖 `f_lineno`）。

用法
================================================================================
    python tools/bench_readout_lineprofile.py --steps 60
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="P137 读出路径逐行打点（定位那~14.7 ms）")
    ap.add_argument("--steps", type=int, default=60, help="打多少步")
    ap.add_argument("--conn-k", type=int, default=128)
    ap.add_argument("--n-out", type=int, default=52642)
    ap.add_argument("--n-h", type=int, default=3072)
    ap.add_argument("--top", type=int, default=18, help="显示前 N 行")
    ap.add_argument("--pin-cap", type=int, default=0,
                    help="P139：覆盖 pinned buffer 池大小（0=用默认4）。"
                         "P138 观测到 `_staged_to_dev` 的 `buf.copy_()` 要~5 ms"
                         "（12 KiB 的 CPU 拷贝本该微秒级）→ 怀疑是**CPU 写 pinned "
                         "buffer 时被 NPU 阻塞**（Event 只等H2D 传输完成、不等"
                         "kernel 消费完）。若成立，**增大 cap** 才是解法")
    ap.add_argument("--sync-every", type=int, default=1,
                    help="每 N 步 sync 一次（对齐生产的 nll_sync_every）")
    args = ap.parse_args()

    os.environ.setdefault("NUMBA_CACHE_DIR",
                          str(_ROOT / "outputs" / "numba_cache"))
    from phdnet.config import PHDNetConfig
    from phdnet.sparse_pc import _random_csr
    from phdnet.backends.accel_readout import pick_readout_backend
    import torch

    n_out, n_h, k = args.n_out, args.n_h, args.conn_k
    cfg = PHDNetConfig(readout_conn_k=k, readout_dtype="fp32",
                       lognormal_init=False, nll_sync_every=args.sync_every)
    csr = _random_csr(np.random.default_rng(0), n_out, n_h, k,
                      0.05 * np.sqrt(n_h / k), False, 0.8)
    ro, backend = pick_readout_backend(cfg, n_h, n_out, np.random.default_rng(1))
    if args.pin_cap > 0:
        ro._pin_cap = int(args.pin_cap)
    print(f"[pf] 后端 {backend} | device {getattr(ro, 'device', '?')} "
          f"| conn_k {getattr(ro, 'conn_k', 0)} "
          f"| nll_sync_every {cfg.nll_sync_every}")
    if not hasattr(ro, "forward_dev"):
        print("[pf] ⚠ 回落臂（无 forward_dev）——本工具针对**加速臂**，"
              "本机只能验证脚本可跑，**昇腾上才有意义**。")

    rng = np.random.default_rng(2)
    h = rng.normal(0, 1, n_h).astype(np.float32)
    tgt = np.zeros(n_out, dtype=np.float32)
    tgt[0] = 1.0

    cls = type(ro)
    watched = {f"{cls.__module__}.{cls.__name__}.{nm}"
               for nm in ("forward_dev", "learn_softmax", "_matmul",
                          "_staged_to_dev", "_lookup_ht", "_sp_gather",
                          "_eager_step", "_train_step_core")}
    acc: dict[tuple[str, int], list[float]] = {}

    def _tracer(frame, event, arg):
        if frame.f_code.co_name not in ("forward_dev", "learn_softmax",
                                        "_staged_to_dev", "_lookup_ht",
                                        "_sp_gather", "_matmul", "_eager_step"):
            return None
        key = f"{cls.__name__}.{frame.f_code.co_name}"
        # ⚠ **归因必须用「上一次事件 → 本次事件」**。第一版写成
        # `t0 - _last[0]`（t0 是**函数进入**的时刻、_last 在**行事件**里推进）
        # → 行事件早于函数进入时差值为**负**（本机实测整表负数、占比 -1%），
        # 数值完全无意义。改为：函数进入时只记时刻，**由行事件自身的时间戳
        # 减去上一个行事件的时间戳**，并在return 时补上「最后一行 → 返回」。
        state = {"last": time.perf_counter()}

        def _local(frame, event, arg):
            now = time.perf_counter()
            if event == "line":
                acc.setdefault((key, frame.f_lineno), []).append(
                    (now - state["last"]) * 1e3)
                state["last"] = now
            elif event == "return":
                acc.setdefault((key, -1), []).append(
                    (now - state["last"]) * 1e3)
                state["last"] = now
            return _local

        return _local

    # 先预热（JIT/内存池/首次分配都不该计入）
    for _ in range(5):
        y = ro.forward_dev(h) if hasattr(ro, "forward_dev") else ro(h)
        ro.learn_softmax(h, tgt, 0.15, y_pre=y,
                         **({"target_idx": 0} if hasattr(ro, "forward_dev") else {}))
    acc.clear()

    t_wall0 = time.perf_counter()
    sys.settrace(_tracer)
    try:
        for _ in range(args.steps):
            y = ro.forward_dev(h) if hasattr(ro, "forward_dev") else ro(h)
            ro.learn_softmax(h, tgt, 0.15, y_pre=y,
                             **({"target_idx": 0}
                                if hasattr(ro, "forward_dev") else {}))
    finally:
        sys.settrace(None)
    wall = (time.perf_counter() - t_wall0) * 1e3 / args.steps

    sync = getattr(getattr(torch, "npu", None), "synchronize", None)
    if sync is not None:
        sync()
    print(f"[pf] {args.steps} 步，平均墙钟（含 trace 开销）= {wall:.3f} ms/步")
    print(f"[pf] ⚠ settrace 本身有开销（本机约 0.1~0.3 ms/行），"
          f"故**绝对值偏大**，看相对占比与最大行")

    # acc 的键是 (函数名, 行号) → 行是 (均ms, 次数, 函数名, 行号)
    rows = [(float(np.mean(v)), len(v), k[0], k[1]) for k, v in acc.items()]
    # ── P139诊断：`buf.copy_()` 那5 ms 是「CPU 撞上NPU」还是真实 memcpy？──
    # 观测：`_staged_to_dev:822 buf.copy_(torch.from_numpy(a))` = 4.94 ms。
    # 那是**12 KiB 的 CPU→CPU 拷贝**，正常应是微秒级。两个候选解释：
    #  (a) 真实 memcpy 慢（不可能）；
    #  (b) **CPU 写pinned buffer 时被NPU 阻塞** —— Event 只等「H2D 传输完成」，
    #      **不等 NPU kernel 消费完这块 buffer**（P28 写Event 时的假设是错的）。
    #      CPU 跑得比NPU 快 → 队列积压 → 到第 4 个 buffer 时必须等。
    # 验证：让 CPU 每步多干一件事（模拟「跑得更快」）→ 若该行耗时**下降**，
    #      就证明是「等 NPU」而非「拷贝慢」。
    print("\n[diag] pinned buffer 拷贝的真实成本（区分「等NPU」vs「memcpy 慢」）")
    _pb = getattr(ro, "_pin_bufs", None)
    _use_pin = _pb is not None
    print(f"    pinned 池启用: {_use_pin}（cap={getattr(ro, '_pin_cap', '?')}）")
    _sink = 0.0
    _chunk = 2000

    def _spill(work: int):
        # 每步额外消耗 CPU 时间：若 _staged_to_dev 的耗时随CPU 变忙而**下降**，
        # 证明它是在**等 NPU**（队列有空间时就不用等）。
        acc_ = 0
        for i in range(work):
            acc_ += (i * 2654435761) & 0xFFFF
        return acc_

    for _work, _lbl in ((0, "CPU 不加活"), (400, "CPU 加活(模拟跑更快)")):
        if not _use_pin:
            print("    （本机无 pinned 池/非加速臂，跳过）")
            break
        _sink = 0.0
        for _ in range(4):
            _ro._staged_to_dev(h)
        _t0 = time.perf_counter()
        for _ in range(20):
            _sink += _spill(_work)
            _ro._staged_to_dev(h)
        _dt = (time.perf_counter() - _t0) / 20 * 1e3
        print(f"    {_lbl:<22} _staged_to_dev = {_dt:.4f} ms/次")
    print("    → 若「加活」后**下降**：说明是「等 NPU」→ 队列已满，"
          "**减小 pin_cap 无效、增大才有意义**；")
    print("       若「加活」后**上升**：说明是真 memcpy 成本（与 12 KiB 不符，需查驱动）。")

    rows = [r for r in rows if r[0] >= 0.0]      # 负值=归因错误，宁可丢弃
    rows.sort(key=lambda r: -r[0])
    total = sum(r[0] for r in rows) or 1.0

    print("\n" + "=" * 96)
    print(f"{'函数:行':<34}{'均ms':>9}{'占比':>7}{'次数':>7}  源码")
    print("-" * 96)
    try:
        import linecache
        src_path = cls.__module__.replace(".", "/") + ".py"
        for mean_ms, cnt, fname, lineno in rows[:args.top]:
            txt = (linecache.getline(src_path, lineno) or "").strip()
            if len(txt) > 44:
                txt = txt[:44] + "..."
            print(f"{fname + ':' + str(lineno):<34}{mean_ms:>9.4f}"
                  f"{mean_ms / args.steps * 100:>6.1f}%{cnt:>7}  {txt}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  （读源码失败：{type(e).__name__}: {e}）")
        for mean_ms, cnt, fname, lineno in rows[:args.top]:
            print(f"{fname + ':' + str(lineno):<34}{mean_ms:>9.4f}")
    print("=" * 96)
    print(f"合计{sum(r[0] for r in rows):.3f} ms/步（trace 口径，含开销）")
    if rows:
        top = rows[0]
        print(f"最贵：**{top[2]}:{top[3]}** 平均 {top[0]:.4f} ms/行"
              f"（占 {top[0]/args.steps*100:.1f}%）")
        print("→ 该行就是优化起点。若它是 `.item()` / `synchronize()` / "
              "H2D 拷贝 → 属**同步**成本；若是 Python 张量运算 → 属**主机**成本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())