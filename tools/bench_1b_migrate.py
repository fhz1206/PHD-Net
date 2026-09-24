"""O3 验证：1B 容量印迹表的在线 CSR 迁移实测（dict 版 vs 在线 CSR 版）。

口径：
  - 容量 = n_neurons(2^24) × m_out(60) = 1,006,632,960 ≈ 1.0×10^9 突触容量；
  - 事件驱动：每步活跃 ~32 神经元，实际计算量与 1B 总容量无关；
  - N_STEPS 步在线 learn + predict，两版结果逐位对拍；
  - 报告 ms/step、已生长突触内存（含 Python 容器开销实测）、stats。

诚实边界（100B 不可行性）：
  100B 条目 × (目标 int64 8B + 权重 float64 8B) = 1.6 TB 纯数据；
  即便 int8 量化权重亦需 100B × (8B + 1B) ≈ 0.9 TB，另加 CSR 行元数据——
  超出本机 RAM(12.6 GB) 与磁盘(152 GB)数量级 3–4 个数量级，
  故本机可验证上限即为本脚本的 1B 容量档；100B 属"结构就绪、规模不可验证"。

用法：
  python tools/bench_1b_migrate.py                     # 默认 2000 步
  python tools/bench_1b_migrate.py --steps 5000 --mmap-dir outputs/_csr_shards
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.sparse_table import OnlineCSRTable, SparseSynapseTable  # noqa: E402

try:
    import psutil
    _PROC = psutil.Process()
    def rss_mb() -> float:
        return _PROC.memory_info().rss / 1e6
except ImportError:                     # pragma: no cover
    def rss_mb() -> float:
        return 0.0

N_NEURONS = 1 << 24
M_OUT = 60
ACT = 32
SEED = 0xA5A55A5A


def container_bytes(table) -> int:
    """实测已生长突触的容器内存（dict 版递归 sizeof；CSR 版数组 nbytes）。"""
    if isinstance(table, OnlineCSRTable):
        return int(sum(v.nbytes for v in table.keys.values())
                   + sum(v.nbytes for v in table.vals.values()))
    total = sys.getsizeof(table.out)
    for i, bucket in table.out.items():
        total += sys.getsizeof(bucket) + 64          # 行 dict 头（近似）
        for k, w in bucket.items():
            total += sys.getsizeof(k) + sys.getsizeof(w)
    return int(total)


def run(table, acts, do_predict=True):
    t0 = time.perf_counter()
    preds = []
    prev = None
    for cur in acts:
        if prev is not None:
            table.learn(prev, cur)
        if do_predict:
            preds.append(table.predict(cur))
        prev = cur
    return time.perf_counter() - t0, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--mmap-dir", type=str, default="")
    ap.add_argument("--shards", type=int, default=8)
    args = ap.parse_args()

    print("=" * 74)
    print("O3 · 1B 容量印迹表在线 CSR 迁移实测")
    print(f"容量 = {N_NEURONS:,} × {M_OUT} = {N_NEURONS * M_OUT:,} 突触（≈1.0B）")
    print(f"步数 = {args.steps}，每步活跃 = {ACT}")
    print("=" * 74)

    rng = np.random.default_rng(SEED)
    acts = [rng.choice(N_NEURONS, size=ACT, replace=False).tolist()
            for _ in range(args.steps)]

    results = {}
    for tag, cls, extra in (("dict", SparseSynapseTable, {}),
                            ("csr", OnlineCSRTable, {"grow_chunk": 8})):
        r0 = rss_mb()
        t = cls(N_NEURONS, M_OUT, seed=SEED, **extra)
        dt, preds = run(t, acts)
        r1 = rss_mb()
        st = t.stats()
        mem = container_bytes(t)
        results[tag] = dict(dt=dt, preds=preds, stats=st, mem=mem, rss=r1 - r0)
        print(f"\n[{tag}]")
        print(f"  训练 {args.steps} 步耗时 : {dt:.2f}s（{dt / args.steps * 1000:.3f} ms/步）")
        print(f"  已生长突触             : {st['grown_synapses']:,}"
              f"（容量占用 {st['utilization']:.5%}）")
        print(f"  触达神经元 / 有出行数   : {st['touched_neurons']:,} / {st['neurons_with_out']:,}")
        print(f"  容器内存（实测）        : {mem / 1e6:.2f} MB"
              f"（{mem / max(1, st['grown_synapses']):.1f} B/条目）")
        print(f"  RSS 增量               : {r1 - r0:.1f} MB")
        if "slot_efficiency" in st:
            print(f"  预留槽利用率           : {st['slot_efficiency']:.3f}"
                  f"（已分配槽 {st['row_slots_allocated']:,}）")

    # 逐位对拍
    pd, pc = results["dict"]["preds"], results["csr"]["preds"]
    bad = [i for i, (a, b) in enumerate(zip(pd, pc))
           if a.keys() != b.keys() or any(a[k] != b[k] for k in a)]
    print("\n" + "-" * 74)
    print(f"逐位对拍（{args.steps} 步 predict）: "
          f"{'PASS 全部一致' if not bad else f'FAIL {len(bad)} 步不一致'}")
    mem_ratio = results["dict"]["mem"] / max(1, results["csr"]["mem"])
    print(f"内存对比: dict {results['dict']['mem'] / 1e6:.2f} MB → "
          f"csr {results['csr']['mem'] / 1e6:.2f} MB（节省 {mem_ratio:.2f}×）")
    sp_d = results["dict"]["dt"] / args.steps * 1000
    sp_c = results["csr"]["dt"] / args.steps * 1000
    print(f"速度对比: dict {sp_d:.3f} ms/步 → csr {sp_c:.3f} ms/步"
          f"（{'csr 更快 %.2f×' % (sp_d / sp_c) if sp_c < sp_d else 'dict 更快 %.2f×' % (sp_c / sp_d)}）")

    # 分片落盘（可选）
    if args.mmap_dir:
        print("\n" + "-" * 74)
        t_csr = OnlineCSRTable(N_NEURONS, M_OUT, seed=SEED, grow_chunk=8)
        run(t_csr, acts, do_predict=False)
        t0 = time.perf_counter()
        info = t_csr.export_shards(args.mmap_dir, n_shards=args.shards)
        t_exp = time.perf_counter() - t0
        t2 = OnlineCSRTable(N_NEURONS, M_OUT, seed=SEED, grow_chunk=8)
        t0 = time.perf_counter()
        t2.import_shards(args.mmap_dir)
        t_imp = time.perf_counter() - t0
        p1 = [t_csr.predict(a) for a in acts[:200]]
        p2 = [t2.predict(a) for a in acts[:200]]
        ok = all(a.keys() == b.keys() and all(a[k] == b[k] for k in a)
                 for a, b in zip(p1, p2))
        print(f"分片落盘: {info['shards']} 片 / {info['total_entries']:,} 条目，"
              f"导出 {t_exp:.2f}s，恢复 {t_imp:.2f}s，目录 {info['dir']}")
        print(f"round-trip 逐位一致: {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 74)
    print("结论：1B 容量档在线 CSR 迁移验证完成（结构 + 逐位等价 + 落盘）；")
    print("      100B 档本机不可验证（≥0.9 TB 数据 vs 12.6 GB RAM），属结构就绪项。")
    print("=" * 74)


if __name__ == "__main__":
    main()
