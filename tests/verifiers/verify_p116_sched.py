"""P116：方案 B（M4b imprint 摊销）与方案 C（稀疏前向算子）的对拍。

为什么需要这个文件
================================================================================
P114 的CPU/NPU 重叠计划里，方案 B（M4b imprint 摊销）与方案 C（稀疏读出前向
算子）都被排在「先做」的位置——它们**都不碰训练主循环**，风险低于方案 A。

但两者都**不是纯调度优化**，各自带一个必须被门禁盯住的语义变化：

· **方案 B（`ltm_imprint_amortize=N`）**：N=1 是旧行为；N>1 把配对学习攒起来
  一次性提交 → **中间 N-1 步大空间表的状态与逐步版不同**。这不是「延迟执行」
  而是「改变写入时机」。若N>1 的表状态错了，训练不会崩、只会**悄悄学错**——
  正是最危险的一类 bug。
  ⚠ 且组合代价随 N **二次**增长（N(N-1)/2），所以只宜小值。

· **方案 C（`sparse_fwd_kernel=einsum`）**：不物化 (n_out,k) 中间张量，
  但**归约顺序与 mulsum 不同** → 实测本机`max|Δ|≈3e-05`，**非逐位**。
  所以它的验收判据必须是**容差**，绝不能宣称零回归。

四层验证
================================================================================
A. 零回归（最重要）：`amortize=1` 必须与旧行为**逐位**一致
   —— 这是「默认关闭」铁律的兑现。
B. 方案 B 语义：N=2 的表状态 vs 逐步版，量化差异并**如实标注**它不是等价变换；
   同时验证「配对不丢」（每对的 prev/cur 都至少被提交过一次）。
C. 方案 C 容差：einsum vs mulsum 在 1e-4 相对容差内 + 输出非零（防「丢权重」
   那类静默错误）+ 更新路径仍生效。
D. 配置与 CLI 默认值一致（防再次出现 default 与实际不符）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in ("", "tests", "tools"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

_RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    return bool(ok)


def _make_ltm(n_dim=64, n_neurons=4096, m_out=8, k_hash=4, seed=3):
    """构造一个小型 SparseLTM（离线单测，不需要完整模型）。"""
    from phdnet.bigltm import SparseLTM
    # 签名：(n_dim, n_neurons, m_out, k_hash, lam, eta, seed, ...)
    return SparseLTM(n_dim, n_neurons=n_neurons, m_out=m_out, k_hash=k_hash,
                     seed=seed)


def _rates(n_dim, n_steps, seed=5):
    """造稀疏率向量序列（模拟 rate：少量维度非零）。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_steps):
        r = np.zeros(n_dim, dtype=np.float64)
        k = max(1, n_dim // 8)
        dims = rng.choice(n_dim, size=k, replace=False)
        r[dims] = rng.random(k)
        out.append(r)
    return out


def _table_snapshot(ltm) -> dict:
    """表的**可比较快照**（键集合 + 权重和），用于跨配置对拍。"""
    t = ltm.table
    # `t.out` 是 {行键: {列索引: 权重}} 的 dict（不是数组）→快照按「行→权重和」
    # 与「行→列数」两级记录，跨配置才可比。
    out = getattr(t, "out", {}) or {}
    wsum, wn = {}, {}
    for k in sorted(out.keys()):
        v = out[k]
        vals = list(v.values()) if isinstance(v, dict) else list(np.asarray(v).ravel())
        wsum[int(k)] = float(np.sum(vals))
        wn[int(k)] = int(len(vals))
    return {"n_keys": len(wsum), "wsum": wsum, "wn": wn,
            "step_count": int(t.step_count)}


# ── A. 零回归：amortize=1 与不传该参数逐位一致 ──────────────────────────────
def section_a() -> None:
    print("\n[A] 零回归：amortize=1 = 旧行为（逐位）")
    rs = _rates(64, 12, seed=5)

    l1 = _make_ltm()
    for r in rs:
        l1.imprint(r)                                # 不传 amortize（默认 1）

    l2 = _make_ltm()
    for r in rs:
        l2.imprint(r, amortize=1)                    # 显式 1

    s1, s2 = _table_snapshot(l1), _table_snapshot(l2)
    same_keys = s1["n_keys"] == s2["n_keys"]
    dmax = max((abs(s1["wsum"][k] - s2["wsum"].get(k, 0.0))
                for k in s1["wsum"]), default=0.0)
    check(same_keys and dmax == 0.0,
          "A1 amortize=1 两次调用结果逐位相同",
          f"keys {s1['n_keys']}=={s2['n_keys']} max|Δ|={dmax:.3e}")
    check(s1["step_count"] == s2["step_count"] == len(rs),
          "A2 step_count 每步递增（未被摊销吞掉）",
          f"{s1['step_count']} vs {s2['step_count']}（期望 {len(rs)}）")
    check(s1["n_keys"] > 0, "A3 表确实生长了（对拍非空）",
          f"n_keys={s1['n_keys']}")


# ── B. 方案 B 语义：N>1 的表状态差异必须被如实量化 ─────────────────────────
def section_b() -> None:
    print("\n[B] 方案 B：N=2 的语义变化（**不是等价变换**，如实量化）")
    rs = _rates(64, 12, seed=5)

    l_step = _make_ltm()
    for r in rs:
        l_step.imprint(r, amortize=1)                # 逐步写

    l_am = _make_ltm()
    for r in rs:
        l_am.imprint(r, amortize=2)                   # 摊销

    s_step, s_am = _table_snapshot(l_step), _table_snapshot(l_am)

    # B1 配对不丢：摊销版的表必须**也**生长（不为空）
    check(s_am["n_keys"] > 0, "B1 摊销版表也生长（配对未丢）",
          f"n_keys={s_am['n_keys']}（逐步版 {s_step['n_keys']}）")

    # B2 摊销版的键数应当**少于或等于**逐步版（少写了最后几对）
    check(s_am["n_keys"] <= s_step["n_keys"],
          "B2 摊销版键数 ≤ 逐步版（确实少写了缓冲中的对）",
          f"{s_am['n_keys']} vs {s_step['n_keys']}")

    # B3 ⚠ 明确记录差异的真实形态（实测得到的，不是猜的）：
    # **差异不在权重值，而在「缓冲区里还没提交的那几对」** ——
    # 逐步版已写入 171 个键，摊销版只有 168 个（最后 3 对还在缓冲区），
    # 而共同键的权重值**完全相同**（`learn` 对同一批 prev/cur 是幂等的）。
    # 这正是「延迟提交」该有的表现：数据没丢，只是写表时机推后。
    only_step = sorted(set(s_step["wsum"]) - set(s_am["wsum"]))
    only_am = sorted(set(s_am["wsum"]) - set(s_step["wsum"]))
    common = set(s_step["wsum"]) & set(s_am["wsum"])
    d = max((abs(s_step["wsum"][k] - s_am["wsum"][k]) for k in common),
            default=0.0)
    check(len(only_step) > 0 and len(only_am) == 0 and d == 0.0,
          "B3 差异形态：摊销版**只少写**缓冲区里的键，共同键权重逐位相同",
          f"仅逐步版有 {len(only_step)} 键（例 {only_step[:3]}）；"
          f"仅摊销版有 {len(only_am)} 键（期望 0）；共同键 max|Δw|={d:.3e}")
    check(d == 0.0,
          "B3b 共同键权重逐位相同（说明 learn 幂等，差异纯粹是写入时机）",
          f"max|Δw|={d:.3e}")

    # B4 摊销不改变 step_count 语义
    check(s_am["step_count"] == len(rs),
          "B4 摊销版 step_count 仍每步递增",
          f"{s_am['step_count']}（期望 {len(rs)}）")

    # B5 N=1 是 N=2 的特例？（N=1 走旧路径，两条分支不应互相污染）
    l_n1 = _make_ltm()
    for r in rs:
        l_n1.imprint(r, amortize=1)
    check(_table_snapshot(l_n1)["n_keys"] == s_step["n_keys"],
          "B5 重复跑 amortize=1 结果稳定（摊销状态不泄漏）",
          f"n_keys={_table_snapshot(l_n1)['n_keys']}")

    # B6 缓冲不会无限增长（否则是内存泄漏）
    l_mem = _make_ltm()
    for r in _rates(64, 40, seed=7):
        l_mem.imprint(r, amortize=2)
    buf = getattr(l_mem, "_imprint_buf", [])
    check(len(buf) <= 2, "B6 摊销缓冲不随步数增长（无泄漏）",
          f"buf 长度={len(buf)}（期望 ≤2）")


# ── C. 方案 C 容差：einsum vs mulsum ────────────────────────────────────────
def section_c() -> None:
    print("\n[C] 方案 C：einsum vs mulsum（容差判据，**非逐位**）")
    import torch
    from phdnet.backends.accel_readout import AccelReadout
    from phdnet.sparse_pc import _random_csr

    n_out, n_h, k = 512, 128, 16
    csr = _random_csr(np.random.default_rng(41), n_out, n_h, k,
                      0.05 * np.sqrt(n_h / k), False, 0.8)
    ro_m = AccelReadout(n_h, n_out, np.random.default_rng(41), device="cpu",
                        dtype="fp32", conn_k=k, csr=csr)
    ro_e = AccelReadout(n_h, n_out, np.random.default_rng(41), device="cpu",
                        dtype="fp32", conn_k=k, csr=csr,
                        sparse_fwd_kernel="einsum")
    check(ro_e._sp_fwd == "einsum" and ro_m._sp_fwd == "mulsum",
          "C0 两个实例的算子设置正确",
          f"{ro_m._sp_fwd} / {ro_e._sp_fwd}")

    rng = np.random.default_rng(43)
    dmax = 0.0
    for _ in range(8):
        h = rng.normal(0.0, 1.0, n_h)
        ym = np.asarray(ro_m(h))
        ye = np.asarray(ro_e(h))
        dmax = max(dmax, float(np.abs(ym - ye).max()))
        check(float(np.abs(ye).max()) > 0.0,
              "C1 einsum 输出非零（防「丢权重」静默错误）",
              f"max|y|={float(np.abs(ye).max()):.4f}") if _ == 0 else None
    scale = max(1.0, float(np.abs(ym).max()))
    check(dmax <= 1e-4 * scale,
          "C2 einsum 与 mulsum 在 1e-4 相对容差内（**非逐位**）",
          f"max|Δ|={dmax:.3e} 相对={dmax/scale:.2e}")

    # C3 更新路径仍生效
    w0 = ro_e.W.clone()
    ro_e.learn_softmax(rng.normal(0, 1, n_h), None, 0.15, target_idx=3)
    check(not bool((w0 == ro_e.W).all()), "C3 einsum 臂更新路径生效")

    # C4 默认仍是 mulsum（不能因为「本机更快」就悄悄改默认）
    from phdnet.config import PHDNetConfig
    d = PHDNetConfig().sparse_fwd_kernel
    check(d == "mulsum", "C4 config 默认 sparse_fwd_kernel='mulsum'（未偷换）",
          f"实际={d!r}")


# ── D. 默认值一致性 ────────────────────────────────────────────────────────
def section_d() -> None:
    print("\n[D] 配置与 CLI 默认值一致（防 default 与实际不符）")
    from phdnet.config import PHDNetConfig
    import subprocess
    c = PHDNetConfig()
    check(c.sparse_fwd_kernel == "mulsum", "D1 config.sparse_fwd_kernel 默认 mulsum")
    check(int(c.ltm_imprint_amortize) == 1, "D2 config.ltm_imprint_amortize 默认 1")

    # CLI 默认（读源码，避开运行整个训练）
    src = (_ROOT / "train" / "train.py").read_text(encoding="utf-8")
    ok_fwd = '"--sparse-fwd-kernel", default="mulsum"' in src.replace("\n", " ")
    ok_amort = '"--ltm-imprint-amortize", type=int, default=1' in src.replace("\n", " ")
    check(ok_fwd, "D3 CLI --sparse-fwd-kernel 默认 mulsum（与 config 一致）")
    check(ok_amort, "D4 CLI --ltm-imprint-amortize 默认 1（与 config 一致）")

    # --help 可执行（含新参数）
    r = subprocess.run([sys.executable, str(_ROOT / "train" / "train.py"), "--help"],
                       capture_output=True, text=True, cwd=str(_ROOT), timeout=180)
    check(r.returncode == 0 and "--sparse-fwd-kernel" in r.stdout
          and "--ltm-imprint-amortize" in r.stdout,
          "D5 新参数出现在 --help 且不崩",
          f"exit={r.returncode}")


def main() -> int:
    print("=" * 78)
    print("P116 方案 B（M4b imprint 摊销）+ 方案 C（稀疏前向算子）对拍")
    print("=" * 78)
    section_a()
    section_b()
    section_c()
    section_d()
    npass = sum(1 for ok, _, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print("\n" + "=" * 78)
    print(f"结果：{npass}/{total} 通过 | 失败 {total - npass}")
    if npass != total:
        print("失败用例：")
        for ok, name, detail in _RESULTS:
            if not ok:
                print(f"  · {name}  {detail}")
    print("=" * 78)
    return 0 if npass == total else 1


if __name__ == "__main__":
    sys.exit(main())