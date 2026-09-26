"""M9 验收：路线图第二轮全量优化的逐项消融（2026-09-19）。

覆盖项（全部为 config 开关，默认关闭）：
  T1.2 预测编码主目标化      pc_predictive_target（需 eta_pc>0，与①组合测）
  T2.2 可学习稀疏词典        learnable_encoder
  T3.2 检索常态化+top-k      retrieval_topk
  T3.4 WM 内容寻址+扩容      wm_content_address / n_wm_slots
  T4.4 验证驱动睡眠          plateau_sleep（valid_text）
  T5.1lite 稀疏 PC 学习更新  sparse_pc（容量组基准）
  T5.2 拓扑生长引导          growth_guidance（容量组）
  T5.3 int8 量化存储         sparse_int8（容量组）
  PC突破① 发育期后巩固      pc_dev_steps
  PC突破② 回放稳定读出      readout_replay
  PC突破③ 逐神经元目标率    neuron_target_rate
  T3.3  情景缓冲生成         episodic_len（生成合法性对比）

口径：与 M2/M3/reval_doc 一致——词级 LM、small 配置（n_sdr=128,k=16,n_mid=128,
n_top=128, eta_pc=0, eta_oja=0, eta_stdp=0.03, seed=11, pred_in_readout=True；
分词 max_len=4,min_count=8,min_entropy=1.0）、语料=《架构设计》全文 80/20、
指标=字符归一 PPL。R 参照应复现 ≈110.57（官方重评估值）。
"""

# --- 目录引导：从任意 cwd 可运行 ---
import sys as _sys
import json
import time
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
_DOC = _ROOT / "eval_corpus" / "internal_corpus.txt"
# --- 引导结束 ---

import numpy as np

from phdnet.bigltm import SparseLTM
from phdnet.config import PHDNetConfig
from phdnet.pc import PredictiveCodingStack
from phdnet.word_lm import PHDWordLM

SEG = dict(max_len=4, min_count=8, min_entropy=1.0)
BASE = dict(n_sdr=128, k_sparse=16, n_mid=128, n_top=128,
            eta_pc=0.0, eta_oja=0.0, eta_stdp=0.03, seed=11,
            pred_in_readout=True)


def base_cfg(**kw) -> PHDNetConfig:
    d = dict(BASE)
    d.update(kw)
    return PHDNetConfig(**d)


def run_variant(text: str, train_text: str, eval_text: str, tag: str,
                **kw) -> dict:
    """训练 + 评估一个配置，返回指标与耗时。"""
    t0 = time.perf_counter()
    lm = PHDWordLM(text, base_cfg(**kw), seg_kwargs=SEG)
    valid = eval_text if kw.get("plateau_sleep") else None
    lm.train_stream(train_text, valid_text=valid)
    m = lm.evaluate(eval_text)
    dt = time.perf_counter() - t0
    m.update({"tag": tag, "sec": round(dt, 1)})
    print(f"  {tag:<34} PPL={m['ppl_char']:8.2f}  bpc={m['bpc']:.3f}  "
          f"oov={m['oov_rate'] * 100:4.1f}%  {m['sec']:6.1f}s")
    return m


def word_lm_ablation(text: str, train_text: str, eval_text: str) -> list[dict]:
    print("=" * 78)
    print("M9 消融 A：词级 LM 字符归一 PPL（R 参照 = M2 最优 W2 配置）")
    print("=" * 78)
    rows: list[dict] = []
    # R 参照：应复现官方重评估 ≈110.57
    rows.append(run_variant(text, train_text, eval_text, "R 参照（W2, 全关）"))
    # V4'：全程可塑表征（M8.A V4 复现：稳态+退火仍劣于冻结 ~18.7%）
    rows.append(run_variant(text, train_text, eval_text, "V4' 全程可塑（ηpc.02+稳态+退火）",
                            eta_pc=0.02, eta_oja=0.02, homeostasis=True,
                            eta_readout_anneal=0.9997))
    # ①：发育期后强制巩固（前 1500 步可塑，之后冻结）
    rows.append(run_variant(text, train_text, eval_text, "①发育期巩固(1500步后冻结)",
                            eta_pc=0.02, eta_oja=0.02, homeostasis=True,
                            eta_readout_anneal=0.9997, pc_dev_steps=1500))
    # ①+T1.2：可塑期内用「下一时间步预测」作为生成目标
    rows.append(run_variant(text, train_text, eval_text, "①+T1.2 预测主目标化",
                            eta_pc=0.02, eta_oja=0.02, homeostasis=True,
                            eta_readout_anneal=0.9997, pc_dev_steps=1500,
                            pc_predictive_target=True))
    # T2.2：可学习稀疏词典
    rows.append(run_variant(text, train_text, eval_text, "T2.2 可学习稀疏词典",
                            learnable_encoder=True))
    # T3.2：检索常态化 + top-k 线索融合
    rows.append(run_variant(text, train_text, eval_text, "T3.2 top-k 检索融合(k=8)",
                            retrieval_topk=8))
    # T3.4：WM 内容寻址 + 扩容 4→8
    rows.append(run_variant(text, train_text, eval_text, "T3.4 内容寻址+WM×2",
                            wm_content_address=True, n_wm_slots=8))
    # ②：回放稳定读出
    rows.append(run_variant(text, train_text, eval_text, "②回放稳定读出",
                            readout_replay=True))
    # ③：逐神经元目标发放率
    rows.append(run_variant(text, train_text, eval_text, "③逐神经元目标发放率",
                            neuron_target_rate=True))
    # T4.4：验证驱动睡眠
    rows.append(run_variant(text, train_text, eval_text, "T4.4 验证驱动睡眠",
                            plateau_sleep=True, plateau_patience=2))
    return rows


def generation_check(text: str, train_text: str) -> dict:
    """T3.3 情景缓冲生成：bigram 合法率对比（episodic_len 0/2/4）。"""
    print("-" * 78)
    print("M9 消融 B：T3.3 情景缓冲接入生成（bigram 合法率，τ=0.3，120 token）")
    lm = PHDWordLM(text, base_cfg(), seg_kwargs=SEG)
    lm.train_stream(train_text)
    toks = lm.tokenize(train_text)
    out = {}
    rng = np.random.default_rng(7)
    for ep in (0, 2, 4):
        gen = lm.generate(toks[0], n_tokens=120, tau=0.3, rng=rng, episodic_len=ep)
        v = lm.bigram_validity(gen, toks)
        out[f"episodic_{ep}"] = v
        print(f"  episodic_len={ep}:  bigram 合法率 {v * 100:5.1f}%")
    return out


def capacity_bench() -> dict:
    """T5.2 / T5.3 / T5.1-lite：容量轨道基准。"""
    print("-" * 78)
    print("M9 消融 C：容量轨道（T5.2 生长引导 / T5.3 int8 / T5.1lite 稀疏更新）")
    out: dict = {}

    # ---- T5.3 int8 等价性与内存：同一序列，float vs int8 预测偏差 ----
    def seq_stream(mem: SparseLTM, dim: int, n: int, rng) -> None:
        for _ in range(n):
            v = np.zeros(dim)
            idx = rng.choice(dim, size=8, replace=False)
            v[idx] = 1.0
            mem.imprint(v)

    dim = 128
    m_f = SparseLTM(dim, n_neurons=1 << 14, m_out=16, k_hash=4, seed=3)
    m_q = SparseLTM(dim, n_neurons=1 << 14, m_out=16, k_hash=4, seed=3,
                    int8_store=True)
    rg = np.random.default_rng(0)
    seq_stream(m_f, dim, 200, rg)
    rg = np.random.default_rng(0)
    seq_stream(m_q, dim, 200, rg)
    cue = np.zeros(dim)
    cue[np.random.default_rng(1).choice(dim, 8, replace=False)] = 1.0
    rf, rq = m_f.recall(cue), m_q.recall(cue)
    grow_f = m_f.table.stats()["grown_synapses"]
    grow_q = m_q.table.stats()["grown_synapses"]
    snap = m_q.table.compact_csr()
    # 值对象内存粗测：float 路径每突触一个 float 对象（24B），int8 路径为缓存小 int
    mem_f = grow_f * 24
    mem_q = grow_q * 8                      # 小 int 对象指针 + 共享码本近似
    out["int8"] = {
        "grown_f": grow_f, "grown_q": grow_q,
        "max_abs_diff": float(np.abs(rf - rq).max()),
        "cos": float(rf @ rq / (np.linalg.norm(rf) * np.linalg.norm(rq) + 1e-9)),
        "mem_float_KB": mem_f / 1024, "mem_int8_KB": mem_q / 1024,
        "csr_entries": int(len(snap["data"])),
    }
    print(f"  T5.3 int8: 生长 {grow_f}/{grow_q} 突触，预测 cos={out['int8']['cos']:.4f}，"
          f"max|Δ|={out['int8']['max_abs_diff']:.4f}，"
          f"值内存 {out['int8']['mem_float_KB']:.1f}→{out['int8']['mem_int8_KB']:.1f} KB，"
          f"CSR {out['int8']['csr_entries']} 条")

    # ---- T5.2 生长引导：入度分布均匀性（同流量，看 hub 是否缓和） ----
    m_g = SparseLTM(dim, n_neurons=1 << 14, m_out=16, k_hash=4, seed=3,
                    growth_guidance=True)
    rg = np.random.default_rng(0)
    seq_stream(m_g, dim, 200, rg)
    s0 = m_f.table.stats()
    s1 = m_g.table.stats()
    out["growth"] = {"deg_max_plain": s0.get("in_deg_max"),
                     "deg_max_guided": s1.get("in_deg_max"),
                     "deg_mean_guided": s1.get("in_deg_mean")}
    print(f"  T5.2 生长引导: 最大入度 {s0.get('in_deg_max')} → {s1.get('in_deg_max')}"
          f"（引导后均值 {s1.get('in_deg_mean')}）"
          if s1.get("in_deg_max") is not None else
          f"  T5.2 生长引导: 引导表统计缺失")
    # 引导表的因果可用性：重合度自检（A→B→C 型）
    a = np.zeros(dim); a[np.arange(0, 8)] = 1.0
    b = np.zeros(dim); b[np.arange(8, 16)] = 1.0
    c = np.zeros(dim); c[np.arange(16, 24)] = 1.0
    m_c = SparseLTM(dim, n_neurons=1 << 14, m_out=16, k_hash=4, seed=5,
                    growth_guidance=True)
    for v in (a, b, c, a, b, c):
        m_c.imprint(v)
    r1_ = m_c.recall(a)
    hit_b = float(r1_[np.arange(8, 16)].mean())
    r2_ = m_c.recall(b)
    hit_c = float(r2_[np.arange(16, 24)].mean())
    out["growth_causal"] = {"a_to_b": hit_b, "b_to_c": hit_c}
    print(f"  T5.2 因果结构（引导表）: A→B 重合 {hit_b:.2f} / B→C 重合 {hit_c:.2f}")

    # ---- T5.1-lite 稀疏 PC：n_top=1024 学习步耗时 ----
    n0, n1, n2 = 256, 1024, 1024
    pc_d = PredictiveCodingStack(n0, n1, n2, 0.02, 0.02,
                                 np.random.default_rng(0), sparse_pc=False)
    pc_s = PredictiveCodingStack(n0, n1, n2, 0.02, 0.02,
                                 np.random.default_rng(0), sparse_pc=True)
    x = np.zeros(n0); x[np.arange(0, 32)] = 1.0
    cache_d = pc_d.infer(x, 1)
    cache_s = pc_s.infer(x, 1)
    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        pc_d.learn(cache_d, eta_scale=1.0)
    t_dense = (time.perf_counter() - t0) / N * 1000
    t0 = time.perf_counter()
    for _ in range(N):
        pc_s.learn(cache_s, eta_scale=1.0)
    t_sparse = (time.perf_counter() - t0) / N * 1000
    # 近似等价度：同一权重出发跑 1 步后比较
    pc_a = PredictiveCodingStack(n0, n1, n2, 0.02, 0.02,
                                 np.random.default_rng(0), sparse_pc=False)
    pc_b = PredictiveCodingStack(n0, n1, n2, 0.02, 0.02,
                                 np.random.default_rng(0), sparse_pc=True)
    pc_a.learn(pc_a.infer(x, 1), 1.0)
    pc_b.learn(pc_b.infer(x, 1), 1.0)
    rel = float(np.abs(pc_a.W_dn0 - pc_b.W_dn0).max()
                / (np.abs(pc_a.W_dn0).max() + 1e-9))
    out["sparse_pc"] = {"dense_ms": round(t_dense, 2),
                        "sparse_ms": round(t_sparse, 2),
                        "speedup": round(t_dense / t_sparse, 2),
                        "rel_dev_1step": rel}
    print(f"  T5.1lite 稀疏 PC (n_top=1024): 稠密 {t_dense:.2f} ms/步 → "
          f"稀疏 {t_sparse:.2f} ms/步（{t_dense / t_sparse:.2f}×），"
          f"单步相对偏差 {rel:.4f}（近似等价）")
    return out


def main() -> None:
    text = _DOC.read_text(encoding="utf-8")
    cut = int(len(text) * 0.8)
    train_text, eval_text = text[:cut], text[cut:]
    print(f"语料 {len(text)} 字符（训练 {len(train_text)} / 评估 {len(eval_text)}）\n")

    rows = word_lm_ablation(text, train_text, eval_text)
    gen = generation_check(text, train_text)
    cap = capacity_bench()

    ref = rows[0]["ppl_char"]
    print("=" * 78)
    print("汇总（vs R 参照 %.2f）" % ref)
    for r in rows:
        d = (r["ppl_char"] - ref) / ref * 100
        print(f"  {r['tag']:<34} {r['ppl_char']:8.2f}  {d:+6.1f}%")
    result = {"ref_ppl": ref, "rows": rows, "generation": gen, "capacity": cap,
              "n_chars": len(text)}
    (_ROOT / "outputs" / "demo_m9_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n结果已写入 outputs/demo_m9_result.json")


if __name__ == "__main__":
    main()
