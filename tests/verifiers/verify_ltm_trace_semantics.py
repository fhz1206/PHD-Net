"""门禁：LTM 迹读侧补偿 / 快照保真 / 分片状态恢复 / 巩固钳位（exit 0 = PASS）。

四组检查：
 (a) 同一含断档序列：惰性迹表 vs 每步 flush 的急切参考表
     （dict / csr_py / csr_batch 三配置）→ max|ΔW| ≤ 1e-9，并打印 ratio
     （修复前 η_eff≈1.43η ⇒ ratio≈1.43；修复后 ≈1.0000）；
 (b) compact_csr：float 快照 np.array_equal(data, 表内原权重) 逐位保真、
     scale=1.0；int8 仍是 int8 量化码；
 (c) export→import：in_deg / t_pre / t_post / stamp / step_count 全部恢复、
     首轮 learn max|ΔW|>0（不恢复时 t_pre<=0 整行 continue）、随后 10 步逐位一致；
 (d) consolidate：dict/CSR × float/int8 × forget∈{0,50,100} →
     恒非负、0 不变、100 归零（float/int8 两路径一致）、50=w·f；
     forget>100（150/200）必须 **fail-fast 拒绝**（`bigltm.py` 的
     `consolidate` 已加该守卫：>100 会让 float 路径 `f=1-0.01f<0` 把权重
     整体变负，而 int8 路径 clip 到 0 —— **两路径语义相反**，是静默误用）。

⚠ **(a) 组在本机当前全红**（`NUMBA_LTM=True`）：惰性迹与本 verifier 自建的
  「每步 flush 急切参考」之间存在**真实的语义差**（`ratio = 0.939770`，期望
  ≈1.0000；`max|ΔW| = 8.832e-02`）。
  **已验证这是既有失败、与任何新改动无关** —— 把本会话全部改动 stash 后重跑，
  (a) 组**同样**失败。(b)(c)(d) 三组全绿 ⇒ 快照保真 / 分片状态恢复 / 巩固钳位
  这三条语义都是对的。
  **退出码因此保持 1**：不得为了「看起来全绿」把它降级成警告，那会让这个真实
  差异被永久掩盖。定位它需要一次独立立项（责任在 `phdnet/sparse_table.py`
  的迹补偿 vs 本参考实现的断档处理，不是别处）。

运行：py -3.14 tests/verifiers/verify_ltm_trace_semantics.py
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from phdnet.sparse_table import SparseSynapseTable, OnlineCSRTable    # noqa: E402
from phdnet.ltm_kernel import NUMBA_LTM                               # noqa: E402
from phdnet.bigltm import SparseLTM                                   # noqa: E402

CASES: list[tuple[str, bool]] = []
N = 1 << 12
M_OUT = 8
LAM, ETA, W_MAX, SEED = 0.7, 0.05, 1.0, 0xBEEF
UNIV = 48


def check(name: str, cond: bool, extra: str = "") -> None:
    CASES.append((name, bool(cond)))
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  [{extra}]" if extra else ""))


# ── 急切参考表：每步 learn 前把所有迹 flush 到当前 step_count ──
class _EagerMixin:
    def _flush(self) -> None:
        lam = self.lam
        for trace, stamp in ((self.t_pre, self.stamp_pre),
                             (self.t_post, self.stamp_post)):
            for i in list(trace.keys()):
                dt = self.step_count - stamp.get(i, self.step_count)
                if dt:
                    trace[i] = trace[i] * (lam ** dt)
                stamp[i] = self.step_count

    def learn(self, prev, cur):
        self._flush()
        return super().learn(prev, cur)


class EagerDict(_EagerMixin, SparseSynapseTable):
    pass


class EagerCSR(_EagerMixin, OnlineCSRTable):
    pass


def _mk(cfg: str, eager: bool = False, guidance: bool = False, int8: bool = False):
    if cfg == "dict":
        cls = EagerDict if eager else SparseSynapseTable
        t = cls(N, M_OUT, lam=LAM, eta=ETA, w_max=W_MAX, seed=SEED, prune=0,
                growth_guidance=guidance, int8_store=int8)
    else:
        cls = EagerCSR if eager else OnlineCSRTable
        t = cls(N, M_OUT, lam=LAM, eta=ETA, w_max=W_MAX, seed=SEED, prune=0,
                growth_guidance=guidance, int8_store=int8)
        t.BATCH_MIN_COMBOS = 1 if cfg == "csr_batch" else 10 ** 18
    return t


def _wmap(t) -> dict:
    """{(i, k): 解码后权重} —— dict 表与 CSR 表统一表示。"""
    d: dict = {}
    if isinstance(t, OnlineCSRTable):
        for i in t.keys:
            ks, vs = t.keys[i], t.vals[i]
            for s in range(t.size[i]):
                d[(int(i), int(ks[s]))] = float(t._dec(vs[s]))
    else:
        for i, bucket in t.out.items():
            for k, w in bucket.items():
                d[(int(i), int(k))] = float(t._dec(w))
    return d


def _seq(steps: int, seed: int) -> list:
    """含断档的序列：随机 prev/cur，含大量「只出现一次后隔几步再出现」的 id。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        np_prev = rng.choice(UNIV, size=int(rng.integers(3, 9)), replace=False)
        np_cur = rng.choice(UNIV, size=int(rng.integers(3, 9)), replace=False)
        out.append(([int(x) for x in np_prev], [int(x) for x in np_cur]))
    return out


def _drive(t, seq) -> None:
    for prev, cur in seq:                     # 与 bigltm.imprint 同序：先 learn 后自增
        t.learn(prev, cur)
        t.step_count += 1


def check_a(seq: list) -> None:
    print(f"\n=== (a) 惰性迹 vs 急切参考（NUMBA_LTM={NUMBA_LTM}） ===")
    for cfg in ("dict", "csr_py", "csr_batch"):
        lazy, eager = _mk(cfg), _mk(cfg, eager=True)
        _drive(lazy, seq)
        _drive(eager, seq)
        wl, we = _wmap(lazy), _wmap(eager)
        keys_ok = set(wl) == set(we)
        dmax = max((abs(wl[k] - we[k]) for k in set(wl) & set(we)), default=0.0)
        s_e = sum(we.values()) if we else float("nan")
        ratio = sum(wl.values()) / s_e          # ΣW 惰性 / ΣW 急切（修复后 ≈1）
        # 修复前行为回放（_decay_read 直读原值，无 λ^Δt 补偿）：
        # 稳态 dt=1、λ=0.7 ⇒ η_eff≈1.43η；本序列含更长断档 ⇒ 偏大更多。
        pre = _mk(cfg)
        pre._decay_read = lambda trace, stamp, i: trace.get(i, 0.0)
        _drive(pre, seq)
        wp = _wmap(pre)
        ratio_pre = sum(wp.values()) / s_e
        d_pre = max((abs(wp.get(k, 0.0) - we.get(k, 0.0))
                     for k in set(wp) | set(we)), default=0.0)
        check(f"(a) {cfg}: max|ΔW|<=1e-9 且键集一致", keys_ok and dmax <= 1e-9,
              f"max|ΔW|={dmax:.3e} ratio={ratio:.6f} edges={len(wl)}")
        # 门禁有效性（突变测试）：关掉补偿必须能被本组检出
        check(f"(a) {cfg}: 未补偿（修复前）可被识别", ratio_pre > 1.1 and d_pre > 1e-3,
              f"ratio_pre={ratio_pre:.4f} max|ΔW|_pre={d_pre:.4f}")


def _expected(t):
    """按 compact_csr 的行序/键序取出表内原值（不经过任何量化）。"""
    idx, data = [], []
    if isinstance(t, OnlineCSRTable):
        for i in sorted(t.keys):
            ks, vs, sz = t.keys[i], t.vals[i], t.size[i]
            for s in sorted(range(sz), key=lambda s: int(ks[s])):
                idx.append(int(ks[s]))
                data.append(vs[s])
    else:
        for i in sorted(t.out):
            for k in sorted(t.out[i]):
                idx.append(int(k))
                data.append(t.out[i][k])
    return np.asarray(idx, dtype=np.int64), data


def _seed(t, weights, pairs) -> None:
    for (i, k), w in zip(pairs, weights):
        if isinstance(t, OnlineCSRTable):
            if i not in t.keys:
                t._new_row(i)
            if t._find_slot(i, k) is None:
                t._append(i, k, w)
        else:
            t.out.setdefault(int(i), {})[int(k)] = t._enc(w)


FRACT = [0.12345678901234567, 0.9876543210987654, 0.33333333333333331,
         0.1, 0.7000000000000001]
PAIRS = [(0, 1), (0, 5), (3, 7), (9, 2), (12, 4)]


def check_b() -> None:
    print("\n=== (b) compact_csr 快照保真 ===")
    for mode in ("dict", "csr"):
        for int8 in (False, True):
            t = _mk(mode, int8=int8)
            _seed(t, FRACT, PAIRS)
            snap = t.compact_csr()
            exp_idx, exp_data = _expected(t)
            tag = f"(b) {mode}/{'int8' if int8 else 'float'}"
            idx_ok = np.array_equal(snap["indices"], exp_idx)
            if int8:
                codes = np.asarray([int(v) for v in exp_data], dtype=np.int8)
                ok = (idx_ok and snap["data"].dtype == np.int8
                      and np.array_equal(snap["data"], codes))
                check(f"{tag}: data 仍是 int8 量化码", ok,
                      f"dtype={snap['data'].dtype} n={snap['data'].size}")
            else:
                exp = np.asarray([float(v) for v in exp_data], dtype=np.float64)
                lossy = not np.array_equal(exp, np.round(exp * t._q) / t._q)
                ok = (idx_ok and snap["data"].dtype == np.float64
                      and np.array_equal(snap["data"], exp)
                      and float(snap["scale"]) == 1.0)
                check(f"{tag}: data 逐位保真且 scale=1.0", ok,
                      f"lossy_old_path={lossy}")


def check_c(seq1: list, seq2: list) -> None:
    print("\n=== (c) export → import 分片状态恢复 ===")
    mk = lambda: _mk("csr_py", guidance=True)
    t = mk()
    _drive(t, seq1)
    with tempfile.TemporaryDirectory(prefix="ltm_shard_") as td:
        info = t.export_shards(td, n_shards=4)
        t2 = mk()
        t2.import_shards(td)
    check("(c) 导出包含全局状态键", info["total_entries"] > 0, str(info["shards"]))
    check("(c) step_count 相等", t2.step_count == t.step_count and t.step_count > 0,
          f"{t2.step_count}")
    check("(c) in_deg 相等且非空",
          t2.in_deg == t.in_deg and len(t.in_deg) > 0, f"n={len(t.in_deg)}")
    check("(c) t_pre/t_post 相等且非空",
          t2.t_pre == t.t_pre and t2.t_post == t.t_post and len(t.t_pre) > 0,
          f"n={len(t.t_pre)}")
    check("(c) stamp_pre/stamp_post 相等",
          t2.stamp_pre == t.stamp_pre and t2.stamp_post == t.stamp_post)
    check("(c) keys/vals/size 相等", _wmap(t) == _wmap(t2),
          f"edges={len(_wmap(t))}")

    # 首轮 learn：迹未恢复时 prev 行 t_pre<=0 → 整行 continue ⇒ ΔW 恒为 0
    prev, cur = seq2[0]
    before = _wmap(t2)
    t2.learn(prev, cur)
    t2.step_count += 1
    t.learn(prev, cur)
    t.step_count += 1
    after = _wmap(t2)
    d1 = max((abs(after[k] - before.get(k, 0.0)) for k in after), default=0.0)
    check("(c) 首轮 learn max|ΔW|>0", d1 > 0.0, f"max|ΔW|={d1:.6g}")

    # 随后 10 步：与原表逐位一致
    for prev, cur in seq2[1:]:
        t.learn(prev, cur)
        t.step_count += 1
        t2.learn(prev, cur)
        t2.step_count += 1
    a, b = _wmap(t), _wmap(t2)
    same = set(a) == set(b) and all(a[k] == b[k] for k in a)
    check("(c) 随后 10 步与原表逐位一致", same and t2.step_count == t.step_count,
          f"max|Δ|={max((abs(a[k]-b[k]) for k in a), default=0.0):.3e}")


def check_d() -> None:
    print("\n=== (d) consolidate(forget) 语义 ===")
    # ⚠ `forget` 的有效区间是 **[0, 100]**（百分比口径）：`bigltm.py` 的
    #   `consolidate` 已加 fail-fast 拒绝 >100（此前 forget=150 会让 float 路径
    #   `f = 1 - 0.01*150 = -0.5` 把权重**整体变负**，而 int8 路径 clip 到 0 ——
    #   **两条路径语义相反**，是静默的 API 误用）。
    #   故本组不再把 150 当「钳 0」用例，而是**显式断言它被拒绝**。
    for mode in ("dict", "csr"):
        for int8 in (False, True):
            for forget in (0.0, 50.0, 100.0):
                t = _mk(mode, int8=int8)
                _seed(t, FRACT, PAIRS)
                b = _wmap(t)
                SparseLTM.consolidate(SimpleNamespace(table=t), forget=forget)
                a = _wmap(t)
                kb = sorted(b)
                arr_b = np.asarray([b[k] for k in kb], dtype=np.float64)
                arr_a = np.asarray([a[k] for k in kb], dtype=np.float64)
                tag = f"(d) {mode}/{'int8' if int8 else 'float'}/f={forget:g}"
                nonneg = bool((arr_a >= 0.0).all())
                if forget == 0.0:
                    ok, what = np.array_equal(arr_a, arr_b), "0 不变"
                elif forget == 100.0:
                    ok, what = bool((arr_a == 0.0).all()), "归零/钳 0"
                elif int8:
                    tol = 1.0 / t._q + 1e-12
                    ok = bool(np.max(np.abs(arr_a - arr_b * 0.5)) <= tol)
                    what = f"50=w*f (tol={tol:.2e})"
                else:
                    ok, what = np.array_equal(arr_a, arr_b * 0.5), "50=w*f 逐位"
                check(f"{tag}: 恒非负 且 {what}", nonneg and ok,
                      f"max|after|={float(np.max(np.abs(arr_a))):.6g}")

    # forget>100 必须**报错**（而不是静默变负/静默归零）
    for forget in (150.0, 200.0):
        t = _mk("dict", int8=False)
        _seed(t, FRACT, PAIRS)
        try:
            SparseLTM.consolidate(SimpleNamespace(table=t), forget=forget)
            check(f"(d) forget={forget:g} 越界被 fail-fast 拒绝", False,
                  "未抛 ValueError（API 误用被静默接受！）")
        except ValueError:
            check(f"(d) forget={forget:g} 越界被 fail-fast 拒绝", True,
                  "ValueError（防止 float 路径权重变负）")


def main() -> None:
    seq = _seq(40, 20240607)
    check_a(seq)
    check_b()
    check_c(_seq(15, 777), _seq(11, 778))
    check_d()
    bad = [n for n, ok in CASES if not ok]

    # ⚠⚠ **(a) 组当前在本机全红，且这是**既有问题**、不是任何新改动的回归**。
    #   已用「把本会话全部改动 stash 掉再跑」验证过：(a) 组在**干净工作树**上
    #   同样失败（`max|ΔW| = 8.832e-02`，`ratio = 0.939770`，期望 ≈1.0000）。
    #   即惰性迹表与本 verifier 自建的「每步 flush 急切参考」之间存在**真实的
    #   语义差**，责任在 `phdnet/sparse_table.py` / 本参考实现，不在别处。
    #   (b)(c)(d) 三组全绿 ⇒ 快照保真、分片状态恢复、巩固钳位三条语义都是对的。
    #   **本门禁的退出码因此仍为 1** —— 不得为了「全绿」而把它降级成警告：
    #   那样会把这个真实差异永久藏起来。它需要一次独立立项去定位
    #   （见文件头「已知失败」一节）。
    only_a = bad and all(n.startswith("(a)") for n in bad)
    print(f"\n{'PASS' if not bad else 'FAIL'}: {len(CASES) - len(bad)}/{len(CASES)} 项通过")
    for n in bad:
        print(f"  x {n}")
    if only_a:
        print("\n  [!] 失败**全部集中在 (a) 组** —— 惰性迹 vs 急切参考的对齐问题，")
        print("      已验证为**既有失败**（stash 掉全部新改动后同样失败）。")
        print("      (b)(c)(d) 全绿。退出码保持 1 以免这个差异被永久掩盖。")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
