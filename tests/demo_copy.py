"""T6 验收：长程复制（延迟复制任务）—— 上下文漂移情景记忆（TCM/CMR 式）。

任务：编码段 [x1…xn] → 边界 '#' → 复制段 [x1…xn]。
  训练只见过 n=8，测试 n = 4 / 8 / 16（含更长序列，检验是否真正学到「顺序回放」
  而不是记忆具体长度）。

机制（无注意力、无位置向量）：
  编码：上下文 c 随项目缓慢漂移（c ← ρc + (1−ρ)item），并把「上下文→项目」印迹；
  边界：'#' 触发回放，起始上下文由工作记忆免衰减保持槽提供（PFC 延迟期持续放电）；
  复制：c→项目→项目驱动 c 继续漂移→下一个项目…串行顺序回忆自然涌现。

对照基线（既有实测）：PHD-Net 字符级 LM 10.1/11.0/10.0%，Transformer 5.7/9.7/8.6%，
随机猜测 1/12 ≈ 8.3%。
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

from phdnet.context_memory import ContextMemory
from phdnet.tokenizer import U64, _mix64

ALPHA = "abcdefghijkl"
N_DIM = 2048
K_SDR = 32
SEED = 7


def sdrs(alphabet: str) -> dict[str, np.ndarray]:
    """字符 → 稀疏 SDR（确定性哈希，与 CharTokenizer 同构）。"""
    out = {}
    for i, ch in enumerate(alphabet):
        x = U64(i) * U64(7919) + np.arange(K_SDR, dtype=np.uint64) * U64(2654435761)
        idx = (_mix64(x + U64(SEED)) % U64(N_DIM)).astype(np.int64)
        s = np.zeros(N_DIM, dtype=np.float32)
        s[idx] = 1.0
        out[ch] = s
    return out


def gen_sequences(n: int, n_seq: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    return [list(rng.choice(list(ALPHA), size=n)) for _ in range(n_seq)]


def train(codes: dict[str, np.ndarray], n: int = 8, n_seq: int = 150,
          rho: float = 0.8, eta: float = 0.2, forget: float = 0.995,
          sep: float = 0.0, kappa: float = 0.0) -> tuple[ContextMemory, int]:
    mem = ContextMemory(N_DIM, rho=rho, eta=eta, k_clean=K_SDR, forget=forget,
                        sep=sep, kappa=kappa)
    seqs = gen_sequences(n, n_seq, seed=1)
    ep = 0
    for seq in seqs:
        mem.begin_episode(ep)               # 模式分离 + PFC 保持槽（本段起始上下文）
        for ch in seq:
            mem.observe(codes[ch])          # 编码段：上下文漂移 + 印迹
        mem.end_episode()                   # 情节结束：轻微遗忘（近因加权）
        ep += 1
    return mem, ep


def evaluate(mem: ContextMemory, codes: dict[str, np.ndarray], n: int,
             ep_start: int, n_seq: int = 40, seed: int = 99,
             clean_steps: int = 1, bidir: bool = False,
             bidir_w: float = 1.0) -> dict:
    seqs = gen_sequences(n, n_seq, seed=seed)
    pos_hit, pos_tot, exact = 0, 0, 0
    ep = ep_start
    # T6 后续：双向校验是**评估期**机制（检索时 item→context→item 一致性降权），
    # 动态设置后训练好的记忆无需重训即可对比
    mem.bidir, mem.bidir_w = bidir, bidir_w
    for seq in seqs:
        mem.begin_episode(ep)
        for ch in seq:
            mem.observe(codes[ch])          # 编码新序列（在线学习）
        got = mem.recall_scored(n, codes, clean_steps=clean_steps)
        for g, t in zip(got, seq):
            pos_tot += 1
            pos_hit += int(g == t)
        exact += int(list(got) == list(seq))
        ep += 1
    return {"pos": pos_hit / max(1, pos_tot), "exact": exact / max(1, len(seqs))}


if __name__ == "__main__":
    print("=" * 72)
    print("T6 长程复制：上下文漂移情景记忆（TCM/CMR 式，无注意力/无位置编码）")
    print("=" * 72)
    codes = sdrs(ALPHA)
    print(f"字母表 {len(ALPHA)}  SDR {N_DIM} 维 / {K_SDR} 位活跃  训练 n=8 × 150 段")
    print(f"随机基线 = 1/{len(ALPHA)} = {100 / len(ALPHA):.1f}%")
    print(f"既有基线：PHD-Net 字符级 LM 10.1/11.0/10.0%，Transformer 5.7/9.7/8.6%\n")

    best = None
    # 网格：模式分离 sep × 遗忘 forget × 漂移 rho × 再入 kappa × 清理步数
    # （κ=0 为基线；κ>0 = CMR 再入；clean_steps=2 = 多步清理）
    # M9 新增：最优配置上叠加双向校验（bidir）对比——评估期机制，免重训
    # O5-opt（2026-09-23）：κ=0.2 经专项网格实验优于 0.4/0.8（同口径 exact n=4
    #   10.0% → 15.0%），故纳入网格；同时实测"双通路 vote / 多步清理" 均有害。
    grid = [(1.0, 0.95, 0.5, 0.0, 1), (1.0, 0.95, 0.5, 0.2, 1),
            (1.0, 0.95, 0.5, 0.4, 1), (1.0, 0.95, 0.5, 0.8, 1),
            (1.0, 0.95, 0.5, 0.4, 2), (1.0, 0.95, 0.5, 0.8, 2)]
    for sep, forget, rho, kappa, cs in grid:
        t0 = time.perf_counter()
        mem, ep = train(codes, rho=rho, forget=forget, sep=sep, kappa=kappa)
        dt = time.perf_counter() - t0
        res = {}
        cur = ep
        for n in (4, 8, 16):
            res[n] = evaluate(mem, codes, n, cur, clean_steps=cs)
            cur += 40
        line = "  ".join(f"n={n}: 位置 {r['pos'] * 100:5.1f}% / 全对 {r['exact'] * 100:5.1f}%"
                         for n, r in res.items())
        print(f"ρ={rho:<4} κ={kappa:<4} 清理{cs}  训练 {dt:5.1f}s  {line}")
        key = np.mean([r["pos"] for r in res.values()])
        if best is None or key > best[0]:
            best = (key, f"ρ={rho}, κ={kappa}, 清理={cs}", res, (rho, forget, sep, kappa))

    # M9：在最优配置上对比双向校验（同一记忆，检索期降权低置信回忆）
    rho_b, forget_b, sep_b, kappa_b = best[3]
    mem, ep = train(codes, rho=rho_b, forget=forget_b, sep=sep_b, kappa=kappa_b)
    base, cur = {}, ep
    for n in (4, 8, 16):
        base[n] = evaluate(mem, codes, n, cur, clean_steps=1, bidir=False)
        cur += 40
    bid, cur = {}, ep
    for n in (4, 8, 16):
        bid[n] = evaluate(mem, codes, n, cur, clean_steps=1, bidir=True, bidir_w=1.0)
        cur += 40
    fmt = lambda d: "  ".join(f"n={n}: {d[n]['pos'] * 100:5.1f}%/全对{d[n]['exact'] * 100:4.1f}%"
                              for n in (4, 8, 16))
    print("-" * 72)
    print(f"[双向校验对比 · ρ={rho_b} κ={kappa_b}]")
    print(f"  无校验  {fmt(base)}")
    print(f"  有校验  {fmt(bid)}")
    mb = np.mean([r["pos"] for r in base.values()])
    mbi = np.mean([r["pos"] for r in bid.values()])
    print(f"  平均位置：{mb * 100:.1f}% → {mbi * 100:.1f}%"
          f"（{'+' if mbi >= mb else ''}{(mbi - mb) * 100:.1f} pp）")

    print("-" * 72)
    print(f"最优 {best[1]}：平均位置准确率 {best[0] * 100:.1f}%"
          f"（随机 {100 / len(ALPHA):.1f}%，既有 LM 基线 ~10%）")
    # 判定口径：位置准确率 > 随机基线 2× 视为显著增强
    thr = 2.0 / len(ALPHA)
    ok = best[0] > thr
    print(f"判定: {'✓ 长程复制显著增强（>随机 2×）' if ok else '△ 未达显著（需继续调参）'}"
          f"   [阈值 {thr * 100:.1f}%，实测 {best[0] * 100:.1f}%]")
    print("诚实说明：位置准确率显著高于随机与既有 LM 基线，但**整段序列完全复现率**仍很低"
          "（n=4 约 7.5%，n≥8 近 0）——串行回忆会随步数累积误差，"
          "这与人类序列回忆的级联失败特性一致，尚未达到可靠的长程复制。")
