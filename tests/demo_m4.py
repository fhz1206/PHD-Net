"""M4 验收：1B 容量栈接入（事件驱动稀疏印迹表）。

Part 1  容量层本体：SparseSynapseTable（N=2^24 × m=60 ≈ 1.007B 突触容量）
        在 CPU 上学 A→B→C 序列，检验因果结构、单步耗时、内存、已生长突触数。
Part 2  接入验证：词级 LM（n_top 256 稠密 LTM ←→ n_top 1024 大容量 LTM）
        同语料、同划分，比较字符归一 PPL / 单步耗时 / RSS / 印迹表统计。

注：完整"双栈统一"（顶层 4096 + PC 全链路在线学习）受限于
① numba 自检未过 → 稠密 PC 走 numpy 回退，在千维以上单步成本过高；
② 编译器修复（P2 遗留）未完成。本轮交付的是容量侧接口打通与实测。
"""

# --- 目录结构调整（2026-09-18）：脚本位于 tests/ 或 tools/ 子目录 ---
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]     # 项目根目录
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))              # 保证 `import phdnet` 可用
_DOC = _ROOT / "datasets" / "eval" / "internal_corpus.txt"    # 默认语料
# --- 引导结束 ---

import time

import numpy as np
import psutil

from phdnet.bigltm import SparseSynapseTable
from phdnet.config import PHDNetConfig
from phdnet.tokenizer import U64, _mix64
from phdnet.word_lm import PHDWordLM

N_NEURONS = 1 << 24
M_OUT = 60
N_ACTIVE = 32
SEED = 0xA5A55A5A


def sdr_of(sym: int, n_active: int = N_ACTIVE) -> list[int]:
    """符号 → 大空间 SDR 活跃集（确定性哈希）。"""
    x = U64(sym) * U64(7919) + np.arange(n_active, dtype=np.uint64) * U64(2654435761)
    return (_mix64(x + U64(SEED)) % U64(N_NEURONS)).astype(np.int64).tolist()


def part1_capacity(n_steps: int = 300) -> dict:
    print("=" * 70)
    print("Part 1  容量层：SparseSynapseTable（1B 突触容量，CPU 在线学习）")
    print("=" * 70)
    table = SparseSynapseTable(N_NEURONS, M_OUT, lam=0.7, eta=0.08, seed=SEED)
    seq = [0, 1, 2]
    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 1e6
    t0 = time.perf_counter()
    prev = None
    for step in range(n_steps):
        cur = sdr_of(seq[step % 3])
        if prev is not None:
            table.learn(prev, cur)
        table.step_count += 1
        prev = cur
    dt = time.perf_counter() - t0
    rss1 = proc.memory_info().rss / 1e6
    st = table.stats()

    def overlap(a: list[int], b: list[int]) -> float:
        return len(set(a) & set(b)) / max(1, len(a))

    probes = {}
    for name, sym in (("A", 0), ("B", 1), ("C", 2)):
        p = table.predict(sdr_of(sym))
        top = sorted(p.items(), key=lambda kv: -kv[1])[:N_ACTIVE]
        top_idx = [k for k, _ in top]
        probes[name] = {
            "→B": round(overlap(top_idx, sdr_of(1)), 3),
            "→C": round(overlap(top_idx, sdr_of(2)), 3),
            "→A": round(overlap(top_idx, sdr_of(0)), 3),
        }
    print(f"容量 {st['capacity']:,}  已生长突触 {st['grown_synapses']:,}"
          f"（利用率 {st['utilization'] * 100:.4f}%）")
    print(f"单步 {dt / n_steps * 1000:.2f} ms  总耗时 {dt:.2f}s  "
          f"RSS 增量 {rss1 - rss0:.1f} MB（含 numpy/dict 基线）")
    print("结构检验（给上一符号预测下一步，与真值 SDR 的重合度）:")
    for k, v in probes.items():
        print(f"  {k}: {v}")
    ok = probes["A"]["→B"] > probes["A"]["→C"] and probes["B"]["→C"] > probes["B"]["→A"]
    print(f"因果结构: {'✓ 学到 A→B→C' if ok else '✗ 未学到'}")
    return {"stats": st, "ms_per_step": dt / n_steps * 1000, "rss_delta": rss1 - rss0,
            "probes": probes, "ok": ok}


def part2_integration() -> dict:
    print("\n" + "=" * 70)
    print("Part 2  接入：词级 LM 稠密 LTM（n_top=256）vs 大容量 LTM（n_top=1024）")
    print("=" * 70)
    with open(str(_DOC), encoding="utf-8") as f:
        text = f.read()
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split], text[split:]
    seg_kwargs = dict(max_len=6, min_count=5, min_entropy=1.0)
    base = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
                eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
                pred_in_readout=True)
    big = dict(n_sdr=256, k_sparse=32, n_mid=512, n_top=1024,
               eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
               pred_in_readout=True, big_ltm=True)

    out = {}
    for name, cfg in (("稠密 LTM（n_top=256）", PHDNetConfig(**base)),
                      ("大容量 LTM（n_top=1024，2^24×60）", PHDNetConfig(**big))):
        proc = psutil.Process()
        rss0 = proc.memory_info().rss / 1e6
        lm = PHDWordLM(text, cfg, seg_kwargs=seg_kwargs)
        t0 = time.perf_counter()
        lm.train_stream(train_txt)
        dt = time.perf_counter() - t0
        m = lm.evaluate(eval_txt)
        rss1 = proc.memory_info().rss / 1e6
        n_tok = len(lm.tokenize(train_txt))
        extra = ""
        if cfg.big_ltm:
            st = lm.net.ltm.table.stats()
            extra = (f"  印迹表: 生长 {st['grown_synapses']:,} 突触 / "
                     f"容量 {st['capacity']:,}")
        print(f"{name}\n    训练 {dt:.1f}s（{dt / n_tok * 1000:.2f} ms/token）"
              f"  字符归一 PPL = {m['ppl_char']:.2f}  bpc = {m['bpc']:.3f}"
              f"  RSS 增量 {rss1 - rss0:.1f} MB{extra}")
        out[name] = {"ppl_char": m["ppl_char"], "bpc": m["bpc"],
                     "ms_per_token": dt / n_tok * 1000, "rss_delta": rss1 - rss0}
    return out


if __name__ == "__main__":
    r1 = part1_capacity()
    r2 = part2_integration()
    print("\n" + "-" * 70)
    print("M4 结论：容量层 1B 可训练（"
          f"{r1['ms_per_step']:.2f} ms/步，因果结构 {'✓' if r1['ok'] else '✗'}）；"
          f"接入后字符归一 PPL {r2['大容量 LTM（n_top=1024，2^24×60）']['ppl_char']:.2f}"
          f" vs 稠密 {r2['稠密 LTM（n_top=256）']['ppl_char']:.2f}")
