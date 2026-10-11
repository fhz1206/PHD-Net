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


# ── B. 方案 B 已移除（P122）：验证它被 fail-fast 拦住 ────────────────────────
def section_b() -> None:
    print("\n[B] 方案 B（M4b imprint 摊销）**已于 P122 移除** —— 验证 fail-fast")
    rs = _rates(64, 8, seed=5)

    # B1 默认（=1）必须正常工作
    l1 = _make_ltm()
    ok = True
    try:
        for r in rs:
            l1.imprint(r)
    except Exception as e:                                   # noqa: BLE001
        ok = False
        print(f"    {type(e).__name__}: {e}")
    check(ok and l1.table.step_count == len(rs),
          "B1 amortize=1（默认）正常工作，step_count 每步递增",
          f"step_count={l1.table.step_count}/{len(rs)}")

    # B2 传 >1 必须**报错退出**，不能静默走有害路径
    for n_bad in (2, 3, 8, 64):
        lb = _make_ltm()
        try:
            for r in rs:
                lb.imprint(r, amortize=n_bad)
            check(False, f"B2 amortize={n_bad} → fail-fast", "竟然没报错！")
        except ValueError:
            check(True, f"B2 amortize={n_bad} → ValueError fail-fast")
        except Exception as e:                               # noqa: BLE001
            check(False, f"B2 amortize={n_bad} → ValueError", f"抛的是 {type(e).__name__}")

    # B3 摊销缓冲区**不得残留**（旧实现在 buf[-1:] 里永久滞留数据）
    l3 = _make_ltm()
    for r in rs:
        l3.imprint(r)
    check(not hasattr(l3, "_imprint_buf"),
          "B3 无摊销缓冲区残留（旧实现会永久滞留尾部对 = 持续丢数据）")

    # B4 **生产尺寸**下逐位等于逐步版（用 n_dim=1024/m_out=72 —— 审计发现
    # 小尺寸测会漏掉污染，故这里必须用真实档位）
    from phdnet.bigltm import SparseLTM
    rs_big = _rates(1024, 20, seed=13)
    a = SparseLTM(1024, n_neurons=1 << 20, m_out=72, k_hash=4, seed=3)
    b = SparseLTM(1024, n_neurons=1 << 20, m_out=72, k_hash=4, seed=3)
    for r in rs_big:
        a.imprint(r, amortize=1)
        b.imprint(r)          # 不传 → 默认1
    d = 0.0
    common = set(a.table.out) & set(b.table.out)
    for k in common:
        va = list(a.table.out[k].values())
        vb = list(b.table.out[k].values())
        d = max(d, abs(float(sum(va)) - float(sum(vb))))
    check(d == 0.0, "B4 生产尺寸（n_dim=1024/m_out=72）两臂逐位相同",
          f"共同键 {len(common)}，max|Δw|={d:.3e}")
    check(len(a.table.out) == len(b.table.out), "B5 键数相同",
          f"{len(a.table.out)} vs {len(b.table.out)}")

    # B6 config 默认必须是 1
    from phdnet.config import PHDNetConfig
    check(int(PHDNetConfig().ltm_imprint_amortize) == 1,
          "B6 config.ltm_imprint_amortize 默认=1")


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
    # P131 起默认 = einsum，故「默认臂」也应是 einsum；mulsum 需显式传入。
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
    check(d == "einsum", "C4 config 默认 sparse_fwd_kernel='einsum'（P131 起）",
          f"实际={d!r}")


# ── D. 默认值一致性 ────────────────────────────────────────────────────────
def section_d() -> None:
    print("\n[D] 配置与 CLI 默认值一致（防 default 与实际不符）")
    from phdnet.config import PHDNetConfig
    import subprocess
    c = PHDNetConfig()
    check(c.sparse_fwd_kernel == "einsum",
          "D1 config.sparse_fwd_kernel 默认 einsum（P131）")
    check(int(c.ltm_imprint_amortize) == 1, "D2 config.ltm_imprint_amortize 默认 1")

    # CLI 默认（读源码，避开运行整个训练）
    src = (_ROOT / "train" / "train.py").read_text(encoding="utf-8")
    ok_fwd = '"--sparse-fwd-kernel", default="einsum"' in src.replace("\n", " ")
    ok_amort = '"--ltm-imprint-amortize", type=int, default=1' in src.replace("\n", " ")
    check(ok_fwd, "D3 CLI --sparse-fwd-kernel 默认 einsum（与 config 一致）")
    check(ok_amort, "D4 CLI --ltm-imprint-amortize 默认 1（与 config 一致）")

    # --help 可执行（含新参数）
    # ⚠ 2026-10-07：必须显式 encoding/errors —— `text=True` 默认按**区域编码**
    #   （中文 Windows = cp936）解码子进程输出；子进程一旦打出 UTF-8 字节就在
    #   reader 线程里抛 UnicodeDecodeError，`r.stdout` 留成 None，本断言随之
    #   变成 "argument of type 'NoneType' is not a container"（本机实测）。
    #   断言只比对 ASCII 参数名，两种编码下字节一致，故 decode 用 utf-8+replace 即可。
    r = subprocess.run([sys.executable, str(_ROOT / "train" / "train.py"), "--help"],
                       capture_output=True, text=True, cwd=str(_ROOT), timeout=180,
                       encoding="utf-8", errors="replace")
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