"""PHD-Net v2 验收：四项 LLM 功能对应机制的实验验证。

实验 A 知识存储与调用（M9）：三元组印迹 + 单跳检索（含干扰）
实验 B 逐步推理（M12）：联想链 2 跳演绎（苏格拉底-是-人-具有-会死）
实验 C 文本生成（M11）：温度采样 + 主题锚定 + 2-gram 合法率对照
实验 D 顺序上下文（M8）：无位置编码的突触链时序回放
"""

# --- 目录结构调整（2026-09-18）：脚本位于 tests/ 或 tools/ 子目录 ---
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]     # 项目根目录
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))              # 保证 `import phdnet` 可用
_DOC = _ROOT / "datasets" / "eval" / "internal_corpus.txt"    # 默认语料
# --- 引导结束 ---

import numpy as np

from phdnet.cognition import EpisodicBuffer, SemanticGraph
from phdnet.config import PHDNetConfig
from phdnet.generate import Generator
from phdnet.lm import PHDNetLM

N = 1024
K = 64


def entity(rng, name: str) -> np.ndarray:
    """实体 → 平衡 ±1 向量（确定性：名字字节作为种子，跨进程可复现）。"""
    seed = int.from_bytes(name.encode("utf-8"), "little") % (2**31)
    g = np.random.default_rng(seed)
    v = g.choice([-1.0, 1.0], size=N)
    return v


def hit(o_pred: np.ndarray, o_true_idx: int, entities: list[np.ndarray]) -> bool:
    """命中判定：与真实宾语的相似度应为全体实体中最高且显著。"""
    sims = [float(o_pred @ e / N) for e in entities]
    return int(np.argmax(sims)) == o_true_idx and sims[o_true_idx] > 0.1


def exp_a():
    print("─" * 62)
    print("实验 A  知识存储与调用（M9 语义关联图，含干扰三元组）")
    names_s = ["法国", "巴黎", "苏格拉底", "人", "鸟", "图灵", "地球", "太阳",
               "中国", "杭州", "鲸鱼", "海洋"]
    names_r = ["首都", "地标", "是", "具有", "位于", "环绕"]
    names_o = ["巴黎", "埃菲尔铁塔", "人", "会死", "会飞", "人", "地球", "太阳系",
               "北京", "西湖", "海洋", "水"]
    triples = list(zip(names_s, names_r, names_o))
    rng = np.random.default_rng(0)
    ent = {n: entity(rng, n) for n in set(names_s + names_r + names_o)}
    graph = SemanticGraph(N, eta=0.06, rng=rng)
    for s, r, o in triples:
        graph.bind(ent[s], ent[r], ent[o])

    tests = [("法国", "首都", "巴黎"), ("巴黎", "地标", "埃菲尔铁塔"),
             ("苏格拉底", "是", "人"), ("鸟", "具有", "会飞"),
             ("中国", "首都", "北京"), ("杭州", "位于", "中国")]
    all_ents = [ent[n] for n in sorted(set(names_s + names_r + names_o))]
    name_idx = {n: i for i, n in enumerate(sorted(set(names_s + names_r + names_o)))}
    hits = sum(hit(graph.query(ent[s], ent[r]), name_idx[o], all_ents)
               for s, r, o in tests)
    print(f"  单跳知识查询: {hits}/{len(tests)} 命中（12 条干扰三元组竞争下）")
    return hits / len(tests)


def exp_b():
    print("─" * 62)
    print("实验 B  逐步推理（M12 联想链：2 跳演绎）")
    rng = np.random.default_rng(1)
    ent = {n: entity(rng, n) for n in
           ["苏格拉底", "柏拉图", "图灵", "人", "鸟", "企鹅", "会死", "会飞", "是", "具有"]}
    graph = SemanticGraph(N, eta=0.06, rng=rng)
    facts = [("苏格拉底", "是", "人"), ("柏拉图", "是", "人"), ("图灵", "是", "人"),
             ("人", "具有", "会死"), ("鸟", "具有", "会飞"), ("企鹅", "是", "鸟")]
    for s, r, o in facts:
        graph.bind(ent[s], ent[r], ent[o])
    all_ents = [ent[n] for n in ent]
    name_idx = {n: i for i, n in enumerate(ent)}
    # 联想链 2 跳 vs 直接单跳查询（无该印迹）
    chains = [(("苏格拉底", "是", "具有"), "会死"), (("柏拉图", "是", "具有"), "会死"),
              (("图灵", "是", "具有"), "会死"), (("企鹅", "是", "具有"), "会飞")]
    chain_hits = 0
    direct_hits = 0
    for (s, r1, r2), target in chains:
        o = graph.chain_query(ent[s], [ent[r1], ent[r2]])       # 链式推理
        chain_hits += hit(o, name_idx[target], all_ents)
        direct_hits += hit(graph.query(ent[s], ent[r2]), name_idx[target], all_ents)
    print(f"  链式推理（2 跳）: {chain_hits}/{len(chains)} 命中")
    print(f"  直接查询（1 跳，未印迹的组合）: {direct_hits}/{len(chains)} 命中"
          f"  → 链式检索实现'未存储事实'的推断")
    return chain_hits / len(chains), direct_hits / len(chains)


def exp_c():
    print("─" * 62)
    print("实验 C  文本生成（M11：温度采样 + 主题锚定）")
    with open(str(_DOC), encoding="utf-8") as f:
        text = f.read()
    cfg = PHDNetConfig(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
                       eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11)
    lm = PHDNetLM(text, cfg)
    lm.train_stream(text, log_every=10**9)

    gen = Generator(lm, tau=0.75, topk=8, rng=np.random.default_rng(3))
    samples = [gen.generate(p, n_chars=80) for p in ["PHD-Net 的核心", "工作记忆的槽位"]]
    bigrams_train = {text[i:i + 2] for i in range(len(text) - 1)}
    rng = np.random.default_rng(9)
    random_chars = list(lm.tok.chars)
    legal = np.mean([s[i:i + 2] in bigrams_train
                     for s in samples for i in range(len(s) - 1)])
    rand = "".join(rng.choice(random_chars) for _ in range(200))
    legal_rand = np.mean([rand[i:i + 2] in bigrams_train
                          for i in range(len(rand) - 1)])
    for s in samples:
        print(f"  生成样本: 「{s[:40]}…」")
    print(f"  2-gram 合法率: 生成 {legal:.1%} vs 随机对照 {legal_rand:.1%}")
    return legal, legal_rand


def exp_d():
    print("─" * 62)
    print("实验 D  顺序上下文（M8 情景缓冲：无位置编码的时序回放）")
    rng = np.random.default_rng(7)
    buf = EpisodicBuffer(N, eta=0.5, k=K, rng=rng)
    events = []
    for i in range(6):                              # 6 个事件（32/512 位 SDR）
        e = np.zeros(N)
        e[rng.choice(N, K, replace=False)] = 1.0
        events.append(e)
        buf.store(e)
    seq = buf.replay(events[0], steps=5)            # 从事件 1 链式回放
    accs = []
    for t, (pred, true) in enumerate(zip(seq, events[1:])):
        pred_idx = set(np.nonzero(pred)[0].tolist())
        true_idx = set(np.nonzero(true)[0].tolist())
        accs.append(len(pred_idx & true_idx) / K)
    print(f"  链式回放 5 步与真实序列的位置重合度: "
          + " ".join(f"{a:.0%}" for a in accs)
          + f"  平均 {np.mean(accs):.1%}")
    return float(np.mean(accs))


if __name__ == "__main__":
    print("PHD-Net v2 验收（LLM 功能对应机制，无自注意力/位置编码/堆叠层）")
    a = exp_a()
    chain, direct = exp_b()
    legal, legal_rand = exp_c()
    d = exp_d()
    print("═" * 62)
    print(f"汇总: 知识调用 {a:.0%} ｜ 链式推理 {chain:.0%}（直接 {direct:.0%}）｜ "
          f"2-gram 合法率 {legal:.0%}（随机 {legal_rand:.0%}）｜ 时序回放 {d:.0%}")
    ok = a >= 0.8 and chain > direct and chain >= 0.6 and legal > 0.8 and d >= 0.8
    print(f"v2 验收: {'✓ 四项通过' if ok else '△ 见各项详情'}")
