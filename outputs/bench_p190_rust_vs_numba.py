# P190 基准：M1/M2 CSR/M3/M4a 的 Rust vs numba 对比（1b 档真实规模）
# 纪律：同进程交错测 + best-of-3；计时脚本落盘 .py；x86 口径（不外推昇腾）
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "phdnet_rs")
sys.path.insert(0, _RS_DIR)
import numpy as np

from phdnet_rs import load as rs_load
from phdnet.sparse_encoder import _gemv_rows_ilp, _NUMBA_ENC
from phdnet.sparse_pc import _csr_matvec, _csr_add_outer, _csr_oja_up

k, why = rs_load()
assert k is not None, f"Rust 不可用: {why}"
T = 8  # 与生产 --numba-threads 一致

rng = np.random.default_rng(0)
# ── 1b 档维度（train/config_1b.py PRESETS["1b"]）──────────────────────
W_DIM = 1024          # n_sdr = n_mid = n_top
N_IN = 2 * W_DIM      # T2.3 组合输入
CONN_K = 128          # 12.5% 连接率
M_LAT = 16            # M3 每神经元出边

# ── M1：编码器 GEMV (n_out=1024, n_in=2048) + k-WTA ───────────────────
Wm = (rng.normal(0, 0.05, (W_DIM, N_IN))).astype(np.float32)
x1 = rng.normal(0, 1.0, N_IN).astype(np.float32)
b1 = np.zeros(W_DIM, dtype=np.float32)
out1 = np.empty(W_DIM, dtype=np.float32)

def m1_numba():
    _gemv_rows_ilp(Wm, x1, b1, out1)

def m1_rust():
    k.m1_gemv(Wm, x1, b1, out1, T)

# ── M2：CSR SpMV + 学习核（4 层主干最大的 1024×2048 层）────────────────
n_rows, n_cols = W_DIM, N_IN
nnz = n_rows * CONN_K
idx_flat = (rng.integers(0, n_cols, nnz)).astype(np.int64)
indptr = np.arange(0, nnz + 1, CONN_K, dtype=np.int64)
val = (rng.normal(0, 0.05, nnz)).astype(np.float32)
x2 = rng.normal(0, 1.0, n_cols).astype(np.float32)
y2 = np.zeros(n_rows, dtype=np.float32)
a2 = rng.normal(0, 1.0, n_rows).astype(np.float32)
b2 = rng.normal(0, 1.0, n_cols).astype(np.float32)

def m2_mv_numba():
    _csr_matvec(indptr, idx_flat, val, x2)

def m2_mv_rust():
    k.m2_matvec(indptr, idx_flat, val, x2, y2, T)

def m2_add_numba():
    _csr_add_outer(indptr, idx_flat, val, a2, b2, 0.01)

def m2_add_rust():
    k.m2_add_outer(indptr, idx_flat, val, a2, b2, 0.01, T)

def m2_oja_numba():
    _csr_oja_up(indptr, idx_flat, val, a2, b2, 0.01)

def m2_oja_rust():
    k.m2_oja_up(indptr, idx_flat, val, a2, b2, 0.01, T)

# ── M3：STDP 预测步（n_top=1024 × m_lateral=16 稀疏边，W 为二维 (n, m)）──
n3n, n3m = W_DIM, M_LAT
W3 = (rng.normal(0, 0.05, (n3n, n3m))).astype(np.float32)
post_idx3 = (rng.integers(0, W_DIM, (n3n, n3m))).astype(np.int64)
pre3 = rng.normal(0, 1.0, n3n).astype(np.float32)
post3 = rng.normal(0, 1.0, n3n).astype(np.float32)
p3 = np.zeros(n3n, dtype=np.float32)

def m3_numba():
    from phdnet.stdp_kernels import _predict_edges
    _predict_edges(W3, post_idx3, pre3, n3n)

def m3_rust():
    # ⚠ 口径：Rust `phdnet_m3_stdp` 是**稠密逐元素** Hebbian 核
    # （w[i] += eta·pre[i]·post[i]，clamp [0,w_max]，长度 n），
    # 与 numba 稀疏 `_predict_edges` 不是同一语义（后者是 M3 的稀疏拓扑
    # 前向）。这里按 Rust 核的语义对拍等效 numpy 逐元素操作（同访存量），
    # 衡量的是「Rust 绑定+遍历 vs numpy 向量化」的层级差。
    w = W3d.copy()
    k.m3_stdp(w, pre3.copy(), post3.copy(), 1.0, 0.02)

W3d = rng.normal(0, 0.05, n3n).astype(np.float32)

def m3_numba_dense():
    w = W3d + 0.02 * (pre3 * post3)
    np.clip(w, 0.0, 1.0, out=w)

# ── M4a：工作记忆 slots 衰减/写入（n = 槽位数，1b 档 = n_top）─────────
n4 = W_DIM
slots4 = rng.normal(0, 1.0, n4).astype(np.float32)
r4 = rng.normal(0, 1.0, n4).astype(np.float32)

def m4a_rust():
    s = slots4.copy()
    k.m4a(s, r4, 0.95, 1.0, 0.5)

def m4a_py():
    s = slots4 * 0.95
    if 1.0 >= 0.5:
        s = s + 1.0 * (r4 - s)   # 语义近似（衰减+写入），同量级访存

def best3(fn, reps=50):
    fn()  # 预热（numba JIT / 缓存）
    out = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        out.append((time.perf_counter() - t0) / reps * 1e3)
    return min(out)

ROWS = [
    ("M1 gemv  1024x2048", m1_numba, m1_rust),
    ("M2 SpMV  1024x2048 k128", m2_mv_numba, m2_mv_rust),
    ("M2 add_outer", m2_add_numba, m2_add_rust),
    ("M2 oja_up", m2_oja_numba, m2_oja_rust),
    ("M3 stdp dense-Hebb 1024", m3_numba_dense, m3_rust),
    ("M4a decay+write", m4a_py, m4a_rust),
]
print(f"1b 档规模  threads={T}  numba={_NUMBA_ENC}")
print(f"{'核':<28}{'numba':>10}{'Rust':>10}{'Rust/numba':>12}")
for name, fn_py, fn_rs in ROWS:
    t_py = best3(fn_py)
    t_rs = best3(fn_rs)
    print(f"{name:<28}{t_py:>8.3f}ms{t_rs:>8.3f}ms{t_py/t_rs:>10.2f}x")
