"""M6 稀疏读出 A/B 测量框架 —— 「读出稀疏度 ↔ PPL / 速度 / 访存」曲线。

================================================================================
为什么要有这个脚本（决策依据，不是调参工具）
================================================================================
M6 读出目前**默认 100% 稠密**。1B 档实测 51,962 × 3,072 = 1.596 亿权重
（bf16 320 MB），是全模型最大的**单段访存**：每步 forward 读一遍 + update
读写各一遍 ≈ 3× 权重字节数。脑同构要求稀疏（人脑连接率 ~2e-8），仓库里也
已有稀疏路径（`Readout(conn_k=k)`，均匀 k-conn，`_random_csr` 生成）。

归档实测（tests/verifiers/verify_readout_sparse.py，BASE 256 维档）k=8 时
**PPL +14.8%**、速度 13.3×。但那个口径只有 3 个点（k=8/16/32）、且是
BASE 档，**不足以支撑「要不要改默认值」这个决定**——问题在于：
  1. 稀疏度 ↔ PPL 的曲线形状未知（是单调恶化，还是某个 k 之后趋于平坦？
     脑科学里「稀疏度甜点」是常见现象，如果曲线在 k=256 就趋平，改默认值
     就是明显收益；如果到 k=512 仍在陡增，那默认值就该留在稠密）。
  2. 访存/速度收益与 PPL 代价的**交换率**未知。稀疏不是免费的：CSR 每条突触
     要存 idx(int64) + val(float64) = **16 B/突触**，而稠密 fp32 只要
     **4 B/权重**。所以 k/n_in > 0.25 时 CSR **反而更大**。这一点必须在表里
     显式呈现，否则「稀疏 = 省内存」是错觉。

本脚本就是量这条曲线用的。它**只测量、不推荐**：所有配置用同一 token 预算、
同一 seed、同一（fp32）读出精度，唯一变量是 conn_k。PPL 相对稠密的增幅如实
打印，不收敛/发散的配置如实标注。

================================================================================
口径（严格复用 tools/rebaseline.py，不另立口径）
================================================================================
rebaseline.py 没有 CLI，只有 `run(limit)`，且 limit 是**字符**数、无法注入
conn_k。所以本脚本：
  * import rebaseline，复用它暴露的 `DOC` / `BASE` / `SEG` / `ANCHOR`
    —— 语料、80/20 划分、评测口径（`lm.evaluate` 返回的 ppl_char）全部同源；
  * `--anchor-check` 会真调 `rebaseline.run(4000)` 与本脚本的 k=0 臂对拍，
    确认「本框架的稠密臂 == 官方基线口径」，避免框架自说自话。

**token 预算口径**：`--tokens` 是**真实 token 数**（不是字符数）。做法是对
训练段（80% 前缀）分词后取前 N 个 token 再拼回字符串——已验证拼回后重新分词
得到完全相同的 token 序列（分词器对拼接是幂等的），所以 token 数是精确的、
不受「字符↔token 比例」漂移影响。

================================================================================
诚实话术（为什么这些字段不能省）
================================================================================
  * **读出精度锁 fp32**：稀疏模式内部强制 fp32（`readout.py` 的
    `dtype_name = dtype if conn_k == 0 else "fp32"`），若让稠密臂走默认 bf16，
    就会变成「bf16 稠密 vs fp32 稀疏」的混淆对比。故全部臂锁 fp32，
    这是**为公平而施加的控制**，不是 tuned 选择。
  * **访存按实际 dtype 算**，不假设、不外推；同时给出「每步触达字节数」
    （forward 读 1 份 + update 读 1 写 1 = 3 份权重），这是访存口径而非
    FLOPs 口径——M6 的瓶颈是内存流量（见 readout.py 的 P7 注释）。
  * **commit / 平台 / 档位**写进 JSON 的 `meta`，任何数字脱离口径无意义。
  * **发散检测**：权重非有限、PPL 非有限、或 PPL > 20× 稠密臂 → 标
    `diverged`，而不是把一个爆掉的数字当正常结果打出去。

用法：
    python tools/bench_readout_sparse.py --tokens 300                 # 冒烟
    python tools/bench_readout_sparse.py --tokens 3000                # 正式
    python tools/bench_readout_sparse.py --preset rebaseline --anchor-check
    python tools/bench_readout_sparse.py --k-list 0,8,32 --out /tmp/x.json
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tests"))
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

# rebaseline：语料 / 基础配置 / 分词口径 / 官方锚点的唯一来源。
import rebaseline                                        # noqa: E402
from phdnet.config import PHDNetConfig                   # noqa: E402
from phdnet.model import count_params                     # noqa: E402
from phdnet.readout import Readout                        # noqa: E402
from phdnet.sparse_pc import _random_csr                  # noqa: E402,F401（文档引用：稀疏读出的 CSR 生成入口）
from phdnet.word_lm import PHDWordLM                     # noqa: E402

DOC, BASE, SEG, ANCHOR = rebaseline.DOC, rebaseline.BASE, rebaseline.SEG, rebaseline.ANCHOR

# 档位预设：**刻意不含 1B 档**。1B 档稠密读出单步往返 320 MB×3，本机（x86
# 笔记本）跑一轮的噪声会盖过 k 之间的差异；且 1B 档需要流式词表扫描（另一条
# 建库路径），与本框架的「固定冻结语料」口径不可比。4M 档是能在本机拿到
# 干净计时信号的**最大**档位。
PRESETS = {
    # 4M 档（默认）：n_sdr=n_mid=n_top=1024 → n_h = 1024×3 = 3072（pred_in_readout
    # 三拼），与 1B 档的 n_h=3072 **同列数**，故 k 的「相对稀疏度」口径可迁移。
    # k_sparse=128 维持主干 12.5% 连接率（沿用 tools/estimate_scale.py 的 S3_4M）。
    "4m": dict(n_sdr=1024, n_mid=1024, n_top=1024, k_sparse=128),
    # rebaseline 档：与 tools/rebaseline.py 的 BASE 完全一致（n_h=768）。
    # 用途是对拍官方锚点（--anchor-check），不是产曲线。
    "rebaseline": dict(),
    # 16M 档：留作「档位敏感性」检查；本机 3000 token 预算下噪声偏大。
    "16m": dict(n_sdr=2048, n_mid=2048, n_top=2048, k_sparse=256),
}

DEFAULT_K_LIST = (0, 8, 32, 64, 128, 256, 512)


# ────────────────────────────────────────────────────────────────────────────
# 幂律连接数分配器（预留接口；sparse_alloc.py 由另一位同事并行实现中）
# ────────────────────────────────────────────────────────────────────────────
# 首选路径（sparse_alloc 已就绪时）用它的公开 API：
#     assign_conn_counts(counts, alpha=, k_min=1, k_max=n_in,
#                        total_budget=budget, n_in=n_in) -> (k,) int64
# 契约：k_min <= k <= k_max 且 sum(k) <= total_budget；alpha=0 退化为均匀。
# `counts` 是**每个输出单元的频次**——本框架传词表频次（输出层每单元 = 一个词），
# 这正是幂律分组要回答的问题：同一突触预算下「按频次分配」vs「均匀分配」。
# 回退路径（模块未就绪 / 入口不匹配）：候选名探测 + 本文件内的
# `csr_from_widths`（与 `_random_csr` 同采样语义，只改行宽分布）。#
# 为什么要 try/except 包起来：该模块此刻**还没写完**。框架不能因为一个并行
# 分支的 ImportError 就整体不可用——回退到均匀 k-conn 仍能产出主曲线
# （幂律分组只是「同一突触预算下怎么分配」的另一维度，不影响主结论）。
SPARSE_ALLOC_CANDIDATES = ("alloc_widths", "powerlaw_widths", "assign_conn_counts",
                           "assign_widths", "allocate", "plan", "alloc")


def load_sparse_alloc():
    """→ (module|None, 说明字符串)。**不抛异常**：并行分支未就绪是正常状态。"""
    try:
        from phdnet import sparse_alloc                     # noqa: WPS433（有意延迟导入）
    except Exception as exc:                                 # noqa: BLE001
        return None, f"不可 import（{type(exc).__name__}: {exc}）→ 回退均匀 k-conn"
    return sparse_alloc, f"可 import（{getattr(sparse_alloc, '__file__', '?')}）"


def powerlaw_widths(mod, n_out: int, n_in: int, budget: int, alpha: float,
                    counts=None):
    """拿每行宽度；返回 (widths|None, 入口名|None)。契约不符 → None → 回退均匀。

    这里对「入口名」和「返回形状」都做**防御性**处理：本框架不能反过来成为
    耦合点——宁可跳过幂律这一行，也不能因对方签名不同而崩掉整轮扫描。"""
    if counts is None:
        counts = np.ones(n_out, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64).ravel()
    if counts.size != n_out:
        return None, None

    def _ok(arr) -> bool:
        """契约校验：长度 == n_out、宽度 ∈ [1, n_in]、**总和 ≤ 预算**。"""
        a = np.asarray(arr).ravel()
        if a.size != n_out:
            return False
        if not np.issubdtype(a.dtype, np.integer):
            if not np.all(np.isfinite(a)) or not np.allclose(a, np.round(a)):
                return False
            a = np.round(a).astype(np.int64)
        a = a.astype(np.int64)
        return bool(a.min() >= 1 and a.max() <= n_in and int(a.sum()) <= budget)

    # 首选：真实 API（幂律分配器的纯函数入口，counts = 每输出单元的频次）
    fn = getattr(mod, "assign_conn_counts", None)
    if callable(fn):
        try:
            k = fn(counts, alpha=alpha, k_min=1, k_max=n_in,
                   total_budget=int(budget), n_in=n_in)
        except Exception:                                    # noqa: BLE001
            k = None
        if k is not None and _ok(k):
            return np.asarray(k).ravel().astype(np.int64), "assign_conn_counts"

    # 回退：候选名探测（防御性签名适配，避免与并行分支的签名变更耦合）
    for name in SPARSE_ALLOC_CANDIDATES:
        f = getattr(mod, name, None)
        if not callable(f) or f is fn:
            continue
        for kwargs in (dict(n_rows=n_out, n_cols=n_in, budget=budget,
                            alpha=alpha),
                       dict(n_out=n_out, n_in=n_in, total_budget=budget,
                            alpha=alpha)):
            try:
                w = f(**kwargs)
            except Exception:                                # noqa: BLE001
                continue
            if _ok(w):
                return np.asarray(w).ravel().astype(np.int64), name
    return None, None


def csr_from_widths(rng: np.random.Generator, n_rows: int, n_cols: int,
                    widths: np.ndarray, scale: float):
    """按**变长**每行宽度生成 CSR（幂律分组用）。

    与 `_random_csr` 的唯一差别是行宽不等长，故 indptr 用 cumsum 而非 arange。
    数值语义（列索引升序、无放回、权重 N(0, scale)）与 `_random_csr` 严格一致
    —— 保证「均匀 vs 幂律」的差异只来自**宽度分布**，不来自采样方式。
    """
    widths = np.asarray(widths, dtype=np.int64)
    widths = np.clip(widths, 1, n_cols)
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(widths, out=indptr[1:])
    nnz = int(indptr[-1])
    idx = np.empty(nnz, dtype=np.int64)
    val = np.empty(nnz, dtype=np.float64)
    for r in range(n_rows):
        cols = np.sort(rng.choice(n_cols, size=int(widths[r]), replace=False))
        s = slice(indptr[r], indptr[r + 1])
        idx[s] = cols
        val[s] = rng.normal(0.0, scale, int(widths[r]))
    return indptr, idx, val


def attach_variable_csr(ro: Readout, csr) -> Readout:
    """把变长 CSR 装进一个 Readout（幂律分组专用）。

    `Readout.__init__` 只支持均匀 k（`conn_k` 标量），故这里先建稠密实例再
    换掉 CSR。`conn_k > 0` 在读出内部**只是「稀疏模式」的开关**（见
    readout.py 的 `if self.conn_k > 0` 分支），不参与任何按 k 的算术，故置为
    行宽最大值是安全的；`n_synapses()` 走的是 `len(csr[2])`，与 conn_k 无关。
    """
    ro._csr = csr                                            # noqa: SLF001（刻意：变长结构无公开构造入口）
    widths = np.diff(csr[0])
    ro.conn_k = int(widths.max())
    ro._W = None                                             # noqa: SLF001
    return ro


# ────────────────────────────────────────────────────────────────────────────
# 访存估算
# ────────────────────────────────────────────────────────────────────────────
def mem_profile(ro: Readout) -> dict:
    """读出段的**存储与每步访存**估算（口径写在字段名里，不含糊）。

    · `store_bytes`：权重实际占用的字节（稠密 fp32 = 4 B/权重；稀疏 CSR =
      val(float64) + idx(int64) = 16 B/突触 + indptr(int64)）。
    · `step_touch_bytes`：每步触达 = forward 读 1 份 + update 读 1 写 1 份
      = 3 × store_bytes。这是**访存**口径（M6 的瓶颈是内存流量而非 FLOPs，
      见 readout.py P7 注释：读出学习曾占 78.9% 的步时间，根因是流量）。
    · `store_B_per_param`：每参数/突触字节数——CSR 的 16 B vs 稠密 fp32 的
      4 B 是本表最重要的一列：k/n_in > 0.25 时 CSR 反而**更占内存**。
    """
    n_syn = ro.n_synapses()
    dense_elems = ro._n_out * ro._n_in                     # noqa: SLF001
    if ro.conn_k > 0 and getattr(ro, "_csr", None) is not None:
        ip, idx, val = ro._csr                              # noqa: SLF001
        store = int(idx.nbytes + val.nbytes + ip.nbytes)
        kind = "csr"
    elif ro._codes is not None:                             # noqa: SLF001
        store = int(ro._codes.nbytes)                       # noqa: SLF001
        kind = f"codebook-{ro.qfmt}"
    else:
        store = int(ro._W.nbytes)                           # noqa: SLF001
        kind = "dense"
    return {
        "layout": kind,
        "n_synapses": int(n_syn),
        "dense_equivalent_elems": int(dense_elems),
        "sparsity_ratio_syn_over_dense": n_syn / dense_elems if dense_elems else 0.0,
        "store_bytes": store,
        "store_MB": store / 1e6,
        "store_B_per_param": store / n_syn if n_syn else 0.0,
        "step_touch_bytes": 3 * store,
        "step_touch_MB": 3 * store / 1e6,
        "readout_dtype": ro.dtype_name,
    }


# ────────────────────────────────────────────────────────────────────────────
# 单个配置
# ────────────────────────────────────────────────────────────────────────────
def readout_segment_ms_per_tok(net, n_calls: int) -> float:
    """读出段 ms/tok（取自 model.step 内置的 P20 计时，`_ro_calls` 为实测调用数）。

    为什么要单独报：M6 在 1B 档是全模型最大单段访存，**总 ms/tok 的变化**会被
    其它段（M1/M2/M4/M5）稀释。归档里「稀疏 13.3× 加速」是总时长口径；本表
    同时给出读出段自身口径，两者不一致时以读出段为准（它才是稀疏直接作用的段）。
    取不到时返回 NaN 并在表中标 n/a——**不用 0 冒充**（0 会被读成「零耗时」）。
    """
    acc = float(getattr(net, "_ro_ms_accum", 0.0))
    calls = int(n_calls or getattr(net, "_ro_calls", 0))
    if calls <= 0:
        return float("nan")
    return acc / calls


def run_config(label: str, override: dict, text: str, train_txt: str,
               eval_txt: str, tokens: int, seed: int,
               conn_k: int = 0, widths=None, alloc_name: str | None = None
               ) -> dict:
    """跑一个 conn_k 配置，返回一行测量结果。

    固定项（唯一变量是 conn_k / widths）：
      训练段 = 80% 前缀的前 `tokens` 个 token；seed 固定；读出锁 fp32；
      不设 minibatch（accumulate=1，单 token 在线更新）——与 rebaseline 同。
    """
    cfg_d = {**BASE, **override, "seed": seed, "readout_dtype": "fp32"}
    # ⚠ 这一行是整个脚本的**唯一变量**所在：conn_k 必须真的进 cfg，否则
    #   pick_readout_backend 会构造稠密读出、所有 k 臂都退化成稠密对照
    #   （症状：k=8 与 k=0 的 PPL/存储逐位相同）。改动前先确认它在 cfg 里。
    cfg_d["readout_conn_k"] = int(conn_k)
    cfg = PHDNetConfig(**cfg_d)
    assert cfg.readout_conn_k == int(conn_k)
    lm = PHDWordLM(text, cfg, seg_kwargs=SEG)
    ro = lm.net.readout
    # 交叉断言：稀疏臂必须是 CSR 模式（conn_k>0 时 dtype 也被强制 fp32）
    if conn_k > 0:
        assert ro.conn_k > 0 and ro._csr is not None, (      # noqa: SLF001
            f"conn_k={conn_k} 未生效：读出未进入 CSR 稀疏模式")

    # 幂律分组：把变长 CSR 装上去（均匀 k 时 widths=None，即用 __init__ 的路径）
    if widths is not None:
        n_in, n_out = ro._n_in, ro._n_out                   # noqa: SLF001
        # fan-in 补偿与 Readout.__init__ 的稀疏分支同式：稀疏投射的权重尺度
        # ×√(fan_in/k) 才能让输出幅值与稠密版可比，否则稀疏臂会因为幅值偏低
        # 而在 PPL 上被无谓地惩罚（这是**公平性控制**，不是给稀疏调参）。
        scale = 0.05 * np.sqrt(n_in / max(1.0, float(np.mean(widths))))
        # 独立 rng（不用 net._replay_rng：那是被主模型占用的随机源，测量脚本
        # 不该伸手进别人的状态）。种子与 cfg.seed 绑定 → 幂律行也可复现。
        attach_variable_csr(ro, csr_from_widths(
            np.random.default_rng(seed + 0xA11CE), n_out, n_in, widths, scale))

    n_params = count_params(lm.net)
    mem = mem_profile(ro)
    n_h, n_out = ro._n_in, ro._n_out                        # noqa: SLF001

    # token 预算：分词取前 N 个再拼回（round-trip 幂等，见模块 docstring）
    toks = lm.tokenize(train_txt)
    budget = min(int(tokens), max(2, len(toks) - 1))
    seg = "".join(toks[:budget])
    n_tok = len(lm.tokenize(seg))

    # 复位读出段计时（P20），确保只统计本配置的**训练**步
    lm.net._ro_ms_accum = 0.0                               # noqa: SLF001
    lm.net._ro_calls = 0                                    # noqa: SLF001

    t0 = time.perf_counter()
    nlls = lm.train_stream(seg)
    dt = time.perf_counter() - t0
    ro_ms = readout_segment_ms_per_tok(lm.net, 0)
    m = lm.evaluate(eval_txt)
    ppl = float(m["ppl_char"])
    ms_per_tok = dt / max(1, n_tok) * 1000.0

    # 训练 NLL 轨迹尾段（发散/塌缩的早期信号，比终态 PPL 更早暴露问题）
    tail_nll = float(nlls[-1]) if nlls else float("nan")
    head_nll = float(nlls[0]) if nlls else float("nan")
    w = ro.W
    finite = bool(np.all(np.isfinite(w)))

    row = {
        "label": label,
        "conn_k": int(conn_k) if widths is None else 0,
        "widths_min": int(np.min(widths)) if widths is not None else int(conn_k),
        "widths_max": int(np.max(widths)) if widths is not None else int(conn_k),
        "widths_mean": float(np.mean(widths)) if widths is not None else float(conn_k),
        "alloc": alloc_name or ("uniform" if conn_k > 0 else "dense"),
        "n_h": int(n_h), "n_out": int(n_out),
        "ppl_char": ppl,
        "bpc": float(m["bpc"]),
        "nll_head": head_nll,
        "nll_tail": tail_nll,
        "ms_per_tok": ms_per_tok,
        "readout_ms_per_tok": ro_ms,
        "n_tok_train": int(n_tok),
        "n_tok_eval": int(m["n_tok"]),
        "oov_eval": int(m["oov"]),
        "n_params_total": int(n_params),
        "n_params_readout": int(mem["n_synapses"]),
        "params_ratio_vs_dense": None,          # 训练后回填（相对 k=0 臂）
        "weights_finite": finite,
        "diverged": False,                      # 训练后回填/判定
        "note": "",
        **mem,
    }
    del lm
    gc.collect()
    return row


def classify(rows: list[dict]) -> None:
    """回填 params_ratio_vs_dense 并标注发散（就地改 rows）。

    发散判据（任一命中即标 diverged=True，如实标注而不是把爆掉的数字打出去）：
      1. 权重含 NaN/Inf；
      2. PPL 非有限；
      3. PPL > 20× 稠密臂 —— 4K 字符这种小预算下正常稀疏配置都在 ±50% 内，
         20× 意味着训练确实崩了；
      4. 训练 NLL 尾段 > 首段且 > 2× 稠密臂尾段（学习发散而非「学得慢」）。
    """
    dense = next((r for r in rows if r["alloc"] == "dense"), None)
    d_ppl = dense["ppl_char"] if dense else None
    d_tail = dense["nll_tail"] if dense else None
    for r in rows:
        if dense is not None and dense is not r:
            r["params_ratio_vs_dense"] = (r["n_params_readout"]
                                          / dense["n_params_readout"])
        why = []
        if not r["weights_finite"]:
            why.append("权重含非有限值")
        if not np.isfinite(r["ppl_char"]):
            why.append("PPL 非有限")
        if d_ppl and np.isfinite(d_ppl) and r["ppl_char"] > 20 * d_ppl:
            why.append(f"PPL > 20× 稠密臂({d_ppl:.2f})")
        if (d_tail and np.isfinite(d_tail) and np.isfinite(r["nll_tail"])
                and r["nll_tail"] > d_tail and r["nll_tail"] > 2 * d_tail):
            why.append(f"训练 NLL 尾段 {r['nll_tail']:.3f} > 2× 稠密臂且高于首段")
        r["diverged"] = bool(why)
        r["note"] = "; ".join(why)


# ────────────────────────────────────────────────────────────────────────────
# 表格输出
# ────────────────────────────────────────────────────────────────────────────
def fmt(x, spec=".3f", dash="n/a") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return format(x, spec)


def print_table(rows: list[dict], meta: dict) -> None:
    d_ppl = next((r["ppl_char"] for r in rows if r["alloc"] == "dense"), None)
    print()
    print("=" * 132)
    print(f"M6 读出稀疏度 A/B —— {meta['preset']} 档 | {meta['tokens_requested']} token 预算 | "
          f"seed {meta['seed']} | 读出锁 {meta['readout_dtype']} | {meta['platform']}")
    print(f"口径: rebaseline.py（同语料 80/20 划分 + lm.evaluate 的 ppl_char）| "
          f"n_h={meta['n_h']} n_out={meta['n_out']} | commit {meta['commit']}")
    print("=" * 132)
    hdr = (f"{'配置':<14}{'k':>6}{'ΔPPL%':>9}{'ppl_char':>10}{'ms/tok':>9}"
           f"{'读出ms/tok':>12}{'读出段占比':>11}{'突触数':>11}{'参数比':>9}"
           f"{'存储MB':>10}{'B/突触':>9}{'每步访存MB':>12}  备注")
    print(hdr)
    print("-" * 132)
    for r in rows:
        if d_ppl and np.isfinite(d_ppl):
            dpct = (r["ppl_char"] - d_ppl) / d_ppl * 100
            dcol = f"{dpct:+.2f}%"
        else:
            dcol = "n/a"
        share = (r["readout_ms_per_tok"] / r["ms_per_tok"] * 100
                 if np.isfinite(r["readout_ms_per_tok"]) and r["ms_per_tok"] > 0
                 else float("nan"))
        ratio = fmt(r["params_ratio_vs_dense"], ".4f") if r["params_ratio_vs_dense"] else "1.0000"
        print(f"{r['label']:<14}{r['widths_max']:>6}{dcol:>9}"
              f"{fmt(r['ppl_char'], '.4f'):>10}{fmt(r['ms_per_tok'], '.3f'):>9}"
              f"{fmt(r['readout_ms_per_tok'], '.4f'):>12}{fmt(share, '.1f'):>10}% "
              f"{r['n_params_readout']:>11,}{ratio:>9}"
              f"{fmt(r['store_MB'], '.2f'):>10}{fmt(r['store_B_per_param'], '.1f'):>9}"
              f"{fmt(r['step_touch_MB'], '.2f'):>12}  {r['note'] or 'ok'}")
    print("-" * 132)
    print("列口径：")
    print("  k            = 每输出单元入边数（幂律行则为最大行宽）")
    print("  ΔPPL%        = 相对同轮 k=0（稠密）臂的 ppl_char 增幅；正=更差")
    print("  读出ms/tok   = model.step 内置 P20 计时（forward+update，含计时器自身开销）")
    print("  读出段占比   = 读出段 / 总 ms/tok；其余时间在 M1/M2/M3/M4/M5 段")
    print("  突触数       = count_params 口径的读出可塑参数（稀疏=存在连接，稠密=元素数）")
    print("  参数比       = 读出突触数 / 稠密臂读出突触数")
    print("  B/突触       = 存储字节 / 突触数。CSR = 16 B（idx int64 + val float64）"
          " vs 稠密 fp32 = 4 B/权重")
    print("  每步访存MB   = 3 × 存储（forward 读 1 + update 读 1 写 1）")
    print()
    # 欠训练告警：**必须打**，否则表会被误读。
    # 稠密臂的读出参数量（4M 档 ≈ 8.0M）比 token 预算（3000）大 3 个数量级，
    # 在这种预算下稠密读出**根本训不完**——它的 PPL 差不是「稠密更差」，而是
    # 「稠密欠训练」。于是稀疏臂（参数少 2~3 个数量级、能在预算内训完）会显得
    # 更好看。**这不是稀疏的胜利，是预算不足的伪影**，绝不能据此说「稀疏优于
    # 稠密」或据此改默认值。要得到可信的 PPL 曲线，token 预算必须远大于稠密
    # 臂参数量（或至少让两臂都进入「同预算、同欠训练程度」的口径）。
    d = next((r for r in rows if r["alloc"] == "dense"), None)
    if d is not None and d["n_params_readout"] > 0:
        ratio = d["n_params_readout"] / max(1, d["n_tok_train"])
        print(f"⚠ 欠训练口径告警：稠密臂读出参数 {d['n_params_readout']:,} / "
              f"训练 token {d['n_tok_train']:,} = {ratio:.1f} 参数每 token")
        if ratio > 10:
            print("  → 稠密臂在本预算下**严重欠训练**。稀疏臂 ΔPPL 为负（看似更优）"
                  "是预算不足的伪影，")
            print("     不是稀疏的结构优势。要判断 PPL 曲线的真实形状，"
                  "需把 --tokens 提到 ≥ 参数量的 10~100 倍，")
            print("     或改用 --preset rebaseline（小档位，参数与预算同量级）做定性判断。")
            print("     **本表的 PPL 列在小预算下只反映收敛程度，不反映稀疏度的质量代价。**")
        else:
            print("  → 参数/预算比合理（≤10），PPL 列在两臂间可比。")
    print()
    print(f"幂律分组：{meta['sparse_alloc']}")
    if not any(r["alloc"] not in ("dense", "uniform") for r in rows):
        print("  （未跑幂律行：sparse_alloc 不可用或 --no-powerlaw）")
    print("提醒：本脚本只测量不推荐。4M 档 + 小 token 预算下，PPL 绝对值不可外推到")
    print("      1B 档/长训练；可信的只有「同一口径内的相对趋势」与访存/参数比。")
    print("=" * 132)


# ────────────────────────────────────────────────────────────────────────────
def git_commit() -> str:
    """当前 commit 短哈希（读 .git 裸文件，避免在测量脚本里跑 git 命令）。"""
    head = _ROOT / ".git" / "HEAD"
    try:
        txt = head.read_text(encoding="utf-8").strip()
        if txt.startswith("ref:"):
            ref = _ROOT / ".git" / txt.split(" ", 1)[1].strip()
            return ref.read_text(encoding="utf-8").strip()[:8] if ref.exists() else "unknown"
        return txt[:8]
    except Exception:                                        # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="M6 读出稀疏度 A/B（conn_k ↔ PPL / 速度 / 访存）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--tokens", type=int, default=3000,
                    help="每个配置的**真实 token** 预算（不是字符数）")
    ap.add_argument("--k-list", type=str, default=",".join(map(str, DEFAULT_K_LIST)),
                    help="扫描的 conn_k 列表，逗号分隔；0 = 稠密对照臂。"
                         "超过 n_h 的值会被夹到 n_h 并在表里体现")
    ap.add_argument("--preset", type=str, default="4m", choices=sorted(PRESETS),
                    help="规模档位预设（刻意不含 1B 档，见 PRESETS 注释）")
    ap.add_argument("--out", type=str, default="",
                    help="JSON 输出路径；默认 outputs/experiments/readout_sparse_bench_<ts>.json")
    ap.add_argument("--seed", type=int, default=11,
                    help="seed（与 BASE 一致，保证与 rebaseline 锚点同种子）")
    ap.add_argument("--anchor-check", action="store_true",
                    help="额外跑一次 rebaseline.run(4000) 与本框架 k=0 臂对拍")
    ap.add_argument("--powerlaw", type=str, default="auto",
                    choices=("auto", "on", "off"),
                    help="幂律分组：auto = sparse_alloc 可 import 就跑")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="幂律指数（仅 --powerlaw 生效时）；0 = 均匀")
    ap.add_argument("--powerlaw-budget", type=int, default=0,
                    help="幂律组的突触预算；0 = 取均匀扫描里最小的非零 k 的预算"
                         "（保证「同预算不同分配」的对比干净）")
    args = ap.parse_args()

    k_list = []
    for piece in args.k_list.split(","):
        piece = piece.strip()
        if piece:
            k = int(piece)
            if k >= 0 and k not in k_list:
                k_list.append(k)
    k_list.sort()
    if 0 not in k_list:                      # 稠密对照臂是「相对增幅」的基准，必须有
        k_list.insert(0, 0)

    override = PRESETS[args.preset]
    n_h = (override.get("n_top", BASE.get("n_top", 256))
           * (3 if BASE.get("pred_in_readout") else 2))

    text = open(DOC, encoding="utf-8").read()   # 与 rebaseline.run 同一读取方式
    split = int(len(text) * 0.8)
    train_txt, eval_txt = text[:split], text[split:]

    alloc_mod, alloc_note = load_sparse_alloc()
    run_pl = alloc_mod is not None and args.powerlaw != "off"
    if args.powerlaw == "on" and alloc_mod is None:
        print(f"[warn] --powerlaw=on 但 sparse_alloc 不可 import：{alloc_note}")

    print("=" * 132)
    print(f"M6 读出稀疏度 A/B | preset={args.preset} {override} | "
          f"{args.tokens} token/配置 | seed {args.seed} | 读出锁 fp32")
    print(f"幂律接口：{alloc_note}")
    print("=" * 132, flush=True)

    rows: list[dict] = []
    for k in k_list:
        kk = min(k, n_h)                     # conn_k 会被 Readout 夹到 n_in
        clipped = kk != k
        label = "dense" if k == 0 else f"uniform_k{kk}"
        print(f"[run] {label} (n_h={n_h}, k={kk}) ...", flush=True)
        try:
            r = run_config(label, override, text, train_txt, eval_txt,
                           args.tokens, args.seed, conn_k=kk)
        except Exception as exc:                          # noqa: BLE001
            # 单配置失败不吞掉整轮：记一行失败原因，其余配置继续（诚实优先）
            rows.append({"label": label, "conn_k": kk, "widths_min": kk, "widths_max": kk,
                         "widths_mean": float(kk), "alloc": "dense" if k == 0 else "uniform",
                         "n_h": n_h, "n_out": 0, "ppl_char": float("nan"), "bpc": float("nan"),
                         "nll_head": float("nan"), "nll_tail": float("nan"),
                         "ms_per_tok": float("nan"), "readout_ms_per_tok": float("nan"),
                         "n_tok_train": 0, "n_tok_eval": 0, "oov_eval": 0,
                         "n_params_total": 0, "n_params_readout": 0,
                         "params_ratio_vs_dense": None, "weights_finite": False,
                         "diverged": True,
                         "note": f"运行失败：{type(exc).__name__}: {exc}",
                         "layout": "n/a", "n_synapses": 0, "dense_equivalent_elems": 0,
                         "sparsity_ratio_syn_over_dense": 0.0, "store_bytes": 0,
                         "store_MB": 0.0, "store_B_per_param": 0.0,
                         "step_touch_bytes": 0, "step_touch_MB": 0.0,
                         "readout_dtype": "n/a"})
            continue
        if clipped:
            r["note"] = (r["note"] + "; " if r["note"] else "") + \
                f"请求 k={k} 超过 n_h={n_h}，已夹到 {kk}"
        print(f"      ppl_char={r['ppl_char']:.4f}  {r['ms_per_tok']:.3f} ms/tok  "
              f"读出 {r['readout_ms_per_tok']:.4f} ms/tok  "
              f"存储 {r['store_MB']:.2f} MB ({r['store_B_per_param']:.1f} B/突触)"
              + (f"  ⚠ {r['note']}" if r["note"] else ""), flush=True)
        rows.append(r)

    # 幂律行：与「最小的非零均匀 k」同突触预算，只改变宽度的**分配方式**。
    if run_pl and rows:
        nz = [k for k in k_list if k > 0]
        dense_rows = [r for r in rows if r["alloc"] == "dense" and r["n_out"] > 0]
        if nz and dense_rows:
            budget = (args.powerlaw_budget if args.powerlaw_budget > 0
                      else min(nz) * n_h)
            # counts = 训练段词频（输出层每单元 = 一个词）。用**训练段**而非
            # 全语料的频次，避免评估段信息影响权重的**结构**（结构不是学出来
            # 的，但拿评估段频次定结构仍是泄漏）。
            from collections import Counter                  # noqa: PLC0415
            probe = PHDWordLM(text, PHDNetConfig(**{**BASE, **override,
                                                    "readout_dtype": "fp32"}),
                              seg_kwargs=SEG)
            cnt = Counter(probe.tokenize(train_txt))
            _word_counts = np.array([cnt.get(t, 0) for t in probe.tok.tokens],
                                    dtype=np.float64)
            del probe
            gc.collect()
            print(f"      幂律 counts: 非零频次单元 {int((_word_counts > 0).sum())}"
                  f"/{_word_counts.size}，预算 {budget:,} 突触")
            w, entry = powerlaw_widths(alloc_mod, dense_rows[0]["n_out"],
                                       dense_rows[0]["n_h"], budget,
                                       args.alpha, counts=_word_counts)
            if w is not None:
                label = f"powerlaw_a{args.alpha:g}"
                print(f"[run] {label} (预算 {int(w.sum()):,} 突触, 入口 {entry}) ...",
                      flush=True)
                r = run_config(label, override, text, train_txt, eval_txt,
                               args.tokens, args.seed, widths=w, alloc_name=entry)
                print(f"      ppl_char={r['ppl_char']:.4f}  {r['ms_per_tok']:.3f} ms/tok  "
                      f"读出 {r['readout_ms_per_tok']:.4f} ms/tok  "
                      f"存储 {r['store_MB']:.2f} MB", flush=True)
                rows.append(r)
            else:
                print(f"[skip] 幂律行：sparse_alloc 入口未匹配契约 {SPARSE_ALLOC_CANDIDATES} "
                      f"→ 本轮只有均匀 k 结果")

    classify(rows)

    if args.anchor_check:
        # 对拍 rebaseline.run(4000)：确认「本框架的稠密臂 == 官方口径」。
        # ⚠ 这里有个**必须说清的口径差异**（否则对拍结果会被误读）：
        #   rebaseline.run 用 BASE 构造 PHDNetConfig，而 cfg.readout_dtype 的
        #   **当前默认值已是 bf16**（P9 之后改的）；本框架为了与稀疏臂公平
        #   对比，把两臂都锁在 fp32。所以两者 PPL 天然不同——**这不是 bug**。
        #   与本框架同口径的历史值是 rebaseline.ANCHOR（fp32 时代测的）。
        print("\n[anchor] 对拍 rebaseline.run(4000) 与本框架 k=0 臂 ...", flush=True)
        a = rebaseline.run(4000)
        d = next((r for r in rows if r["alloc"] == "dense"), None)
        print(f"  rebaseline.run(4000) : ppl_char={a['ppl_char']:.4f}  "
              f"{a['ms_per_token']:.3f} ms/token   ← 走 cfg 默认 dtype"
              f"（当前默认 bf16）")
        print(f"  rebaseline.ANCHOR    : {ANCHOR[4000]}   ← fp32 时代的历史锚点")
        if d is not None:
            print(f"  本框架 k=0 臂        : ppl_char={d['ppl_char']:.4f}  "
                  f"{d['ms_per_tok']:.3f} ms/tok   ← 锁 fp32，与稀疏臂同口径")
            same_run = abs(d["ppl_char"] - a["ppl_char"]) < 1e-6
            same_anchor = abs(d["ppl_char"] - ANCHOR[4000]) < 1e-3
            print(f"  与 rebaseline.run(4000) 逐位一致 : "
                  f"{'是 ✅' if same_run else '否（预期：dtype 不同，见上）'}")
            print(f"  与 fp32 历史锚点 ANCHOR 一致   : "
                  f"{'是 ✅' if same_anchor else '否 ❌（同 dtype 下偏离，需排查）'}")
            if not same_anchor and abs(d["n_tok_train"] - 3137) < 20:
                print("     → token 预算已对齐 4,000 字符口径，但 PPL 与 fp32 锚点不符，"
                      "请检查是否有配置漂移。")
            elif not same_anchor:
                print(f"     → token 预算 {d['n_tok_train']} 与 4,000 字符口径"
                      f"（≈3137 token）不同，PPL 不可直接比较。")
            print("  结论：**本框架的稠密臂与稀疏臂同 fp32 口径，两两可比**；"
                  "与 rebaseline.run 的绝对值差异仅来自 dtype，不是回归。")

    meta = {
        "tool": "tools/bench_readout_sparse.py",
        "preset": args.preset,
        "preset_override": override,
        "tokens_requested": args.tokens,
        "seed": args.seed,
        "readout_dtype": "fp32",
        "readout_dtype_reason": "稀疏模式内部强制 fp32；稠密臂若走默认 bf16 会构成"
                                "「bf16 稠密 vs fp32 稀疏」的混淆对比，故全部锁 fp32",
        "n_h": int(rows[0]["n_h"]) if rows else n_h,
        "n_out": int(rows[0]["n_out"]) if rows else 0,
        "vocab_built_from": "full corpus (PHDWordLM 默认在传入全文上建词表)",
        "corpus": DOC,
        "corpus_chars": len(text),
        "split": "80/20（与 rebaseline.run 同一划分）",
        "ppl_metric": "ppl_char（lm.evaluate，与 rebaseline 同口径）",
        "k_list": k_list,
        "n_h_preset": n_h,
        "platform": f"{platform.system()} {platform.machine()} "
                    f"py{platform.python_version()} np{np.__version__}",
        "commit": git_commit(),
        "sparse_alloc": alloc_note,
        "powerlaw_ran": any(r["alloc"] not in ("dense", "uniform") for r in rows),
        "ts": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "honesty_note": "本表只测量不推荐；4M 档 + 小 token 预算的 PPL 绝对值不可外推到 "
                        "1B 档或长训练，可信的只有同口径内的相对趋势与访存/参数比。",
    }
    print_table(rows, meta)

    out = args.out or str(_ROOT / "outputs" / "experiments"
                          / f"readout_sparse_bench_{meta['ts']}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"\nJSON → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
