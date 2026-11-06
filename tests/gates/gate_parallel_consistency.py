"""并行化零回归验证工具：① CSR 五核 numba 并行 vs 并行化前串行版逐位对拍；
② zh 多进程流（PrefetchChars lang='zh'）vs 串行 zh_char_chunks 一致性对拍。"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "train"))

import numpy as np

# ---------- Part A：CSR 核 逐位对拍 ----------
from phdnet.sparse_pc import (_csr_add_outer, _csr_matvec, _csr_oja_up,
                              _csr_row_norms, _csr_scale_rows)
from phdnet.plasticity import NUMBA_OK

if NUMBA_OK:
    from numba import njit

    @njit(cache=False, fastmath=True)
    def matvec_serial(indptr, idx, val, x):
        n = indptr.shape[0] - 1
        y = np.zeros(n)
        for i in range(n):
            s = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                s += val[p] * x[idx[p]]
            y[i] = s
        return y

    @njit(cache=False, fastmath=True)
    def add_outer_serial(indptr, idx, val, a, b, eta):
        n = indptr.shape[0] - 1
        for i in range(n):
            ai = a[i]
            if ai == 0.0:
                continue
            for p in range(indptr[i], indptr[i + 1]):
                val[p] += eta * ai * b[idx[p]]

    @njit(cache=False, fastmath=True)
    def oja_up_serial(indptr, idx, val, post, pre, eta):
        n = indptr.shape[0] - 1
        for i in range(n):
            pi = post[i]
            if pi == 0.0:
                continue
            for p in range(indptr[i], indptr[i + 1]):
                val[p] += eta * pi * (pre[idx[p]] - pi * val[p])

    @njit(cache=False, fastmath=True)
    def row_norms_serial(indptr, val):
        # ⚠ P215：与生产 `_csr_row_norms` **统一 fp32**（P175 起生产输出 fp32）。
        #   原参考版用 `np.zeros(n)` 默认 fp64 + fp64 累加 → 与生产差 ~1e-7，
        #   那不是「并行不一致」，只是**精度口径不同**，会污染判据。
        n = indptr.shape[0] - 1
        out = np.zeros(n, dtype=np.float32)
        for i in range(n):
            s = np.float32(0.0)
            for p in range(indptr[i], indptr[i + 1]):
                s += val[p] * val[p]
            out[i] = s ** np.float32(0.5)
        return out

    @njit(cache=False, fastmath=True)
    def scale_rows_serial(indptr, val, target):
        n = indptr.shape[0] - 1
        for i in range(n):
            s = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                s += val[p] * val[p]
            nrm = s ** 0.5
            if nrm < 1e-12:
                nrm = 1e-12
            f = target[i] / nrm
            for p in range(indptr[i], indptr[i + 1]):
                val[p] *= f


def part_a() -> bool:
    """Part A：5 个 CSR 并行核 vs 并行化前串行版，全部要求**逐位一致**。

    ⚠ P215 审计修复：原实现只 `print` 结果、不返回也不退出 →
      **业务断言失败仍 rc=0（假绿）**。现在收集结果并返回。
    """
    rng = np.random.default_rng(42)
    n_rows, n_cols, k = 512, 256, 16
    indptr = np.arange(0, (n_rows + 1) * k, k, dtype=np.int64)
    idx = np.concatenate([rng.choice(n_cols, k, replace=False) + np.zeros(1)
                          for _ in range(n_rows)]).astype(np.int64)
    for r in range(n_rows):                     # 列索引升序（真实 CSR 语义）
        s = slice(r * k, (r + 1) * k)
        idx[s] = np.sort(idx[s])
    val0 = rng.normal(0, 0.5, n_rows * k)
    a = rng.normal(0, 1, n_rows)
    a[:32] = 0.0                                # 含零行（触发 skip 分支）
    b = rng.normal(0, 1, n_cols)
    b[:16] = 0.0
    post = a.copy()
    target = rng.uniform(0.5, 2.0, n_rows)

    # matvec
    y_par = _csr_matvec(indptr, idx, val0, b)
    y_ser = matvec_serial(indptr, idx, val0, b)
    _ok = [np.array_equal(y_par, y_ser)]
    print(f"matvec     逐位: {_ok[-1]}")

    # add_outer
    v1 = val0.copy()
    v2 = val0.copy()
    _csr_add_outer(indptr, idx, v1, a, b, 0.05)
    add_outer_serial(indptr, idx, v2, a, b, 0.05)
    _ok.append(np.array_equal(v1, v2))
    print(f"add_outer  逐位: {_ok[-1]}")

    # oja_up
    v1, v2 = val0.copy(), val0.copy()
    _csr_oja_up(indptr, idx, v1, post, b, 0.02)
    oja_up_serial(indptr, idx, v2, post, b, 0.02)
    _ok.append(np.array_equal(v1, v2))
    print(f"oja_up     逐位: {_ok[-1]}")

    # row_norms
    # ⚠ P215：这一项**不能要求逐位** —— 生产的 `_csr_row_norms` 带
    #   `fastmath=True`，允许 FMA 收缩 / 浮点重结合，与串行参考的累加次序
    #   不同 → fp32 下差 1 ulp（实测 max|Δ| ≈ 2.4e-7）。
    #   按项目惯例（`gate_cython_kernels`）用**数值级容差**并如实打出 max|Δ|；
    #   其余 4 个核仍严格要求逐位一致。
    n1 = _csr_row_norms(indptr, val0)
    n2 = row_norms_serial(indptr, val0)
    _delta = float(np.max(np.abs(n1.astype(np.float64) - n2.astype(np.float64))))
    _ok.append(np.allclose(n1, n2, rtol=1e-6, atol=1e-6))
    print(f"row_norms  容差: {_ok[-1]}  max|Δ|={_delta:.3e}（fastmath 1 ulp，非逐位）")

    # scale_rows
    v1, v2 = val0.copy(), val0.copy()
    _csr_scale_rows(indptr, v1, target)
    scale_rows_serial(indptr, v2, target)
    _ok.append(np.array_equal(v1, v2))
    print(f"scale_rows 逐位: {_ok[-1]}")
    print(f"Part A: {sum(_ok)}/{len(_ok)} 逐位一致")
    return all(_ok)


# ---------- Part B：zh 多进程流 vs 串行 一致性 ----------
def part_b(n_chunks: int = 64) -> bool:
    from corpus_stream import PrefetchChars, SEP, zh_char_chunks

    glob = os.path.join(_ROOT, "datasets", "pretrain", "pretrain_*.parquet")
    import glob as _glob
    if not _glob.glob(glob):
        # CI（GitHub Actions）没有本地数据集：生成**合成中文语料 parquet**
        # （验证目标是「并行核一致性/分词等价」，数据本身只需稳定、含中文）。
        import tempfile
        import pyarrow as pa
        import pyarrow.parquet as _pq
        _tmpd = tempfile.mkdtemp(prefix="phd_ci_corpus_")
        rng_texts = []
        _zh = ("机器学习模型的训练需要大量语料与稳定的评测基准。"
               "并行一致性验证的目标是保证多核与单核路径逐位相同。")
        for k in range(8):
            rng_texts.append((_zh * 200) + f" 样本编号 {k}。")
        tbl = pa.table({"text": pa.array(rng_texts),
                        "lang": pa.array(["zh"] * len(rng_texts)),
                        "src": pa.array(["ci_synth"] * len(rng_texts))})
        _pq.write_table(tbl, os.path.join(_tmpd, "pretrain_000.000.parquet"))
        glob = os.path.join(_tmpd, "pretrain_*.parquet")
        print(f"[ci-fallback] 本地数据集缺失，使用合成语料: {glob}")
    ser = []
    for i, c in enumerate(zh_char_chunks(glob)):
        ser.append(c)
        if i + 1 >= n_chunks:
            break
    par = []
    pf = PrefetchChars(glob, SEP, depth=16, batch_samples=8, workers=4,
                       lang="zh")
    for c in pf:
        par.append(c)
        if len(par) >= n_chunks:
            break
    pf.close()          # P192：多进程改多线程后，清理 API 从 terminate() 变 close()
    same = (ser == par)
    print(f"zh 流 块数: 串行={len(ser)} 并行={len(par)}  逐位一致: {same}")
    if not same:
        for i, (x, y) in enumerate(zip(ser, par)):
            if x != y:
                print(f"  首个差异 @chunk{i}: 串行 len={len(x)} 并行 len={len(y)}")
                break
    else:
        print(f"  首块前 60 字符: {ser[0][:60]!r}")
    return bool(same)


if __name__ == "__main__":
    print("== Part A: CSR 并行核 vs 并行化前串行版（同 fastmath）==")
    ok_a = part_a()
    print("== Part B: zh 多进程流 vs 串行参考 ==")
    ok_b = part_b()
    print("-" * 66)
    print(f"Part A={'PASS' if ok_a else 'FAIL'}  Part B={'PASS' if ok_b else 'FAIL'}"
          f"   → {'PASS' if (ok_a and ok_b) else 'FAIL'}")
    # ⚠ P215 审计修复：必须把业务结果传播为进程退出码，
    #   否则 runner / CI 会把失败判成 PASS（假绿）。
    raise SystemExit(0 if (ok_a and ok_b) else 1)
