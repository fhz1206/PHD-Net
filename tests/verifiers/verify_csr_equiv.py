"""O1/O3 验证：在线可写 CSR 表 vs dict 邻接表 —— 逐位等价对拍。

对拍内容（全部要求逐位一致）：
  A) 同一 learn/predict 调用序列下，每步 predict 返回 dict 的键集与值；
  B) 最终 compact_csr() 的 row_ids / indptr / indices / data；
  C) stats() 的关键计数（grown_synapses / neurons_with_out）；
  D) 分片落盘 round-trip（export_shards → import_shards）后 predict 仍逐位一致。

覆盖 3 组配置：float 权重、int8 量化、growth_guidance + int8。
退出码 0 = 全部一致；非 0 = 存在不一致。

用法：python tests/verifiers/verify_csr_equiv.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.sparse_table import OnlineCSRTable, SparseSynapseTable  # noqa: E402

N_NEURONS = 1 << 14
M_OUT = 60
N_STEPS = 300
ACT = 32            # 每步活跃神经元数
SEED = 12345

CONFIGS = [
    ("float", dict(int8_store=False, growth_guidance=False)),
    ("int8", dict(int8_store=True, growth_guidance=False)),
    ("int8+guidance", dict(int8_store=True, growth_guidance=True, prune=2000)),
]

_ok_all = True


def check(name: str, ok: bool, detail: str = "") -> bool:
    global _ok_all
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    _ok_all &= ok
    return ok


def dicts_equal(p1: dict, p2: dict) -> bool:
    if p1.keys() != p2.keys():
        return False
    return all(p1[k] == p2[k] for k in p1)      # 精确浮点比较（逐位）


def run_sequence(table, act_lists: list[list[int]]) -> list[dict]:
    """按给定活跃序列跑 learn + predict，返回每步 predict 结果。"""
    preds = []
    prev = None
    for cur in act_lists:
        if prev is not None:
            table.learn(prev, cur)
        preds.append(table.predict(cur))
        prev = cur
    return preds


def gen_acts(rng: np.random.Generator, n_neurons: int, n_steps: int, act: int):
    return [rng.choice(n_neurons, size=act, replace=False).tolist()
            for _ in range(n_steps)]


def csr_equal(a: dict, b: dict) -> bool:
    return (np.array_equal(a["row_ids"], b["row_ids"])
            and np.array_equal(a["indptr"], b["indptr"])
            and np.array_equal(a["indices"], b["indices"])
            and np.array_equal(a["data"], b["data"]))


def main() -> None:
    for tag, kw in CONFIGS:
        print(f"\n=== 配置: {tag}  {kw} ===")
        rng = np.random.default_rng(SEED)
        acts = gen_acts(rng, N_NEURONS, N_STEPS, ACT)

        t_d = SparseSynapseTable(N_NEURONS, M_OUT, seed=SEED, **kw)
        t_c = OnlineCSRTable(N_NEURONS, M_OUT, seed=SEED, grow_chunk=8, **kw)
        p_d = run_sequence(t_d, acts)
        p_c = run_sequence(t_c, acts)

        # A) 每步 predict 逐位一致
        bad = [i for i, (a, b) in enumerate(zip(p_d, p_c)) if not dicts_equal(a, b)]
        check(f"{tag}: {N_STEPS} 步 predict 逐位一致", not bad,
              f"不一致步数 {len(bad)}" + (f"（首例 step={bad[0]}）" if bad else ""))

        # B) compact_csr 一致
        check(f"{tag}: compact_csr 逐位一致", csr_equal(t_d.compact_csr(), t_c.compact_csr()))

        # C) stats 关键计数一致
        sd, sc = t_d.stats(), t_c.stats()
        same_cnt = (sd["grown_synapses"] == sc["grown_synapses"]
                    and sd["neurons_with_out"] == sc["neurons_with_out"])
        check(f"{tag}: stats 计数一致", same_cnt,
              f"dict grown={sd['grown_synapses']} vs csr grown={sc['grown_synapses']}")

        # D) 分片落盘 round-trip（对**训练后**的表逐活跃集 predict 对拍）
        with tempfile.TemporaryDirectory() as td:
            info = t_c.export_shards(td, n_shards=4)
            t_c2 = OnlineCSRTable(N_NEURONS, M_OUT, seed=SEED, grow_chunk=8, **kw)
            t_c2.import_shards(td)
            p_final = [t_c.predict(a) for a in acts]        # 训练后（未落盘）
            p_rt = [t_c2.predict(a) for a in acts]          # 恢复后
            bad_rt = [i for i, (a, b) in enumerate(zip(p_final, p_rt)) if not dicts_equal(a, b)]
            check(f"{tag}: shard round-trip predict 逐位一致", not bad_rt,
                  f"entries={info['total_entries']}, 不一致步数 {len(bad_rt)}")

    print("\n=== 全部逐位等价 ===" if _ok_all else "\n=== 存在不一致，禁止合入 ===")
    sys.exit(0 if _ok_all else 1)


if __name__ == "__main__":
    main()
