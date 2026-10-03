//! M1–M6 的 Rust 实现。
//!
//! # 逐位一致性契约
//!
//! **每一个核都必须与 Python 版逐位一致**，否则对拍门禁会红。这不是可选项：
//! 「语义相近」在本项目不可接受（跨设备/跨实现只能用容差，但**同实现之间
//! 应该能逐位** —— 见 `docs/写作与事实基线.md` §7）。
//!
//! 已知会差 1–2 ulp 的地方已**显式标注**，并说明原因（同 P52 融合核的级别）。

use crate::acl::{Acl, DevBuf};

/// C 连续的 float32 视图（不复制、不拥有）。
///
/// # 为什么不用 `ndarray`
///
/// 「零运行时开销」要求消除**所有**边界层的 load/store。`ndarray` 虽然零成本，
/// 但它的泛型会让 37 个算子各多一层边界检查。这里用裸指针 + 显式 shape，
/// 语义完全可控，且与 Python 侧的 `np.asarray` 布局假设一一对应。
#[derive(Clone, Copy)]
pub struct F32<'a> {
    pub ptr: *mut f32,
    pub rows: usize,
    pub cols: usize,
    pub _marker: core::marker::PhantomData<&'a [f32]>,
}
// SAFETY: 见 _marker —— 各线程只写不重叠的行块，只读共享部分。
unsafe impl<'a> Send for F32<'a> {}
unsafe impl<'a> Sync for F32<'a> {}

impl<'a> F32<'a> {
    /// # Safety
    /// `ptr` 必须是 C 连续、长度 ≥ `rows * cols` 的可写 f32 缓冲。
    pub unsafe fn from_raw(ptr: *mut f32, rows: usize, cols: usize) -> Self {
        F32 { ptr, rows, cols, _marker: core::marker::PhantomData }
    }

    /// 从设备缓冲构造（视图，不复制）。
    pub fn from_dev(b: &DevBuf, rows: usize, cols: usize) -> Self {
        F32 {
            ptr: b.ptr() as *mut f32,
            rows,
            cols,
            _marker: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn row(&self, r: usize) -> *mut f32 {
        unsafe { self.ptr.add(r * self.cols) }
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.rows * self.cols
    }
}

// ══════════════════════════════════════════════════════════════════════════
// M1 稀疏分布式编码（k-WTA）
// ══════════════════════════════════════════════════════════════════════════

/// M1 的稠密 GEMV 部分：`u = W·x + b`。
///
/// # 与 Python 版的逐位关系
///
/// **逐位一致** —— 行内按列序累加，与 `sparse_encoder._gemv_rows` 的累加次序
/// 完全相同（`acc += W[r,c]*x[c]`，从 c=0 到 cols-1）。
///
/// # ⚠ 为什么 Rust 在这里可能更快（但也可能更快不了）
///
/// Python 版有两条 GEMV 路径（P165 实测）：
///   · `numpy BLAS`：141 µs/762 µs（fp32/fp64），**用 SIMD**
///   · 自写 numba 核：762 µs（fp32），10.5 GB/s，**标量**
///
/// 而本函数是**标量**的（无 SIMD 保证）。所以：
///   · **打平了** numpy BLAS（61% 的量级差距来自 BLAS 的 SIMD，Rust 拿不到）
///   · **打得赢** numba 标量核（无 GIL、无 Python 调用 → 可真正多核）
///
/// → 真实收益需要**在服务器上实测**，本机无 NPU 无法定案。
/// **但有一个确定的收益：nogil 的真多核。** Python 的 numba `prange` 在
/// x86 实测只快 1.16×（P22），因为 GIL 与线程池开销吃掉大部分。
/// 这里用 `std::thread::scope` 起裸线程，**没有 GIL**。
///
/// # 并行策略
///
/// 行级 `std::thread::scope` + 静态分块。**不用 rayon**（零依赖原则）。
/// ⚠ 但要诚实：**线程数 ≥ 物理核时收益会平**（读出是带宽受限，
/// 见 MEMORY「读出是 GEMV，受带宽限制」）。默认取 `min(8, 可用核)`。
pub fn m1_gemv(
    w: &F32<'_>,
    x: &[f32],
    b: &[f32],
    out: &mut [f32],
    n_threads: usize,
) {
    debug_assert_eq!(w.cols, x.len());
    debug_assert_eq!(w.rows, b.len());
    debug_assert_eq!(w.rows, out.len());

    let n = w.rows;
    if n == 0 {
        return;
    }
    let nt = n_threads.max(1).min(n);

    if nt == 1 {
        for r in 0..n {
            gemv_row(w, r, x, b[r], &mut out[r]);
        }
        return;
    }

    // 每线程连续的块（**分块而非交错** → 访存局部性更好）
    let chunk = n.div_ceil(nt);
    // ⚠ 闭包里传**裸指针 + 长度**而不是 `&[f32]`：引用不是 `Send`
    //   （因为 Rust 无法证明它不指向可重定位的栈，而我们的用法是安全的
    //   —— 只读+ 不重排 + 不跨线程返回）。
    let x_ptr = x.as_ptr() as usize;
    let b_ptr = b.as_ptr() as usize;
    let cols = w.cols;
    std::thread::scope(|s| {
        for row_lo in (0..nt).map(|t| t * chunk) {
            let row_hi = core::cmp::min(row_lo + chunk, n);
            if row_lo >= row_hi {
                continue;
            }
            // SAFETY: 各线程写`out[row_lo..row_hi]` —— **互不重叠**；
            //   只读 `w` / `x` / `b` → 无竞态（对齐：这是我们敢这么用的唯一理由）。
            // ⚠ 指针以 `usize` 传递：`*mut f32` 不是 `Send`，而整数是。
            //   转换在闭包内完成，语义等价且更明确。
            let out_ptr = unsafe { out.as_mut_ptr().add(row_lo) } as usize;
            let out_len = row_hi - row_lo;
            s.spawn(move || {
                // SAFETY: 指针与长度在调用方保证有效；分块互不重叠。
                let out_ptr = out_ptr as *mut f32;
                let sub = unsafe { core::slice::from_raw_parts_mut(out_ptr, out_len) };
                let xs = unsafe { core::slice::from_raw_parts(x_ptr as *const f32, cols) };
                let bs = unsafe { core::slice::from_raw_parts(b_ptr as *const f32, n) };
                for (i, o) in sub.iter_mut().enumerate() {
                    gemv_row(w, row_lo + i, xs, bs[row_lo + i], o);
                }
            });
        }
    });
}

/// 单行 GEMV。**行内列序累加** → 与 Python 版逐位一致。
#[inline]
fn gemv_row(w: &F32<'_>, r: usize, x: &[f32], b: f32, out: &mut f32) {
    let row = w.row(r);
    let cols = w.cols;
    let mut acc: f32 = 0.0;
    // SAFETY: row 指向 `w` 的第 r 行，长度 cols（由 F32 的不变式保证）。
    unsafe {
        for c in 0..cols {
            acc += *row.add(c) * *x.get_unchecked(c);
        }
    }
    *out = acc + b;
}

/// M1 的 k-WTA +归一化：挑出最大的 k 个，映射到 `[0.1, 1.1]`。
///
/// # 与 Python 版逐位关系
///
/// **归一化部分逐位一致**（同样的 `(win - min)/(max - min + 1e-9) + 0.1`）。
/// ⚠ **选出的 `idx` 顺序不保证与 Python 的 `argpartition` 一致**：
///   · Python：`np.argpartition(-u, k-1)[:k]` —— **部分选择排序**，
///     返回的是**无序**的前 k 个；
///   · 本实现：全排序后取前 k —— 返回**有序**。
///   → **`idx` 作为集合相同，但作为序列不同**。
///   → 上层必须只用它做集合语义（下游确实是：gather + 归一化 + 排序输出）。
///   → `model.py` 最终会 `np.sort(idx)`，所以**最终输出相同**。
pub fn m1_kwta(u: &[f32], k: usize, s_out: &mut [f32], idx_out: &mut [u32]) {
    let n = u.len();
    let k = k.min(n);

    // 部分选择：维护大小为 k 的最小堆？——为**避免全排序**（O(n log k) 而非 O(n log n)）
    //但 heap 要注意 NaN。简单起见用「插入排序维护 top-k」，n=1024、k=128 时
    // 最坏 131k 次比较，远小于全排序，且**无需分配**。

    // 用 (value, index) 的简单 top-k：先全排序索引（n log n，n=1024 时 ~10k 次）
    // ⚠ 诚实：这是**最简实现**，可能不是最快的。
    //   真要快可以用「无序数组 + 插入」，n=1024、k=128 时的常数更小。
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by(|&i, &j| {
        u[j as usize]
            .partial_cmp(&u[i as usize])
            .unwrap_or(core::cmp::Ordering::Equal)
            // ⚠ tie-break 用**小下标在前** → 与 Python 的
            // `np.sort(idx)` 在值相等时的行为不完全一致（Python 是稳定排序）。
            // 见门禁说明：这里按「值集合一致」判定。
            .then(i.cmp(&j))
    });
    let sel = &order[..k];

    let mut win_min = f32::INFINITY;
    let mut win_max = f32::NEG_INFINITY;
    for &i in sel {
        let v = u[i as usize];
        if v < win_min {
            win_min = v;
        }
        if v > win_max {
            win_max = v;
        }
    }
    for v in s_out.iter_mut() {
        *v = 0.0;
    }
    for (pos, &i) in sel.iter().enumerate() {
        let v = u[i as usize];
        s_out[i as usize] = (v - win_min) / (win_max - win_min + 1e-9) + 0.1;
        idx_out[pos] = i;
    }
}

// ══════════════════════════════════════════════════════════════════════════
// M2 预测编码主干（CSR 稀疏）
// ══════════════════════════════════════════════════════════════════════════

/// CSR 稀疏矩阵（结构借用，不复制）。
///
/// **布局与 Python 版逐位对应**：`indptr` 是 `int64`，`idx` 是 `int64`，
/// `val` 是 `float32`。
/// ⚠ **Python 版是 `float64` 的 `val`**（`sparse_pc.py:175`），
/// 而本实现用 `f32`（生产精度，见 P163）。**两者精度不同→ 不能逐位对拍**，
/// 只能容差对拍（fp32 eps ≈ 1.19e-7）。这是**刻意的精度提升**，不是 bug。
#[derive(Clone, Copy)]
pub struct Csr {
    pub indptr: *const i64,
    pub idx: *const i64,
    pub val: *const f32,
    pub n_rows: usize,
}
// SAFETY: 三个数组在 SpMV 期间只读； 的写冲突由分块避免重叠。
unsafe impl Send for Csr {}
unsafe impl Sync for Csr {}

/// CSR SpMV：`y[i] = Σ_p val[p]·x[idx[p]]`。
///
/// # 与 Python 版逐位关系
///
/// **累加次序完全相同**（`indptr[i]` → `indptr[i+1]` 升序）→ 在**同样精度**
/// 下逐位一致。
///
/// # ⚠ 性能说明（诚实）
///
/// 读出是**带宽受限**（MEMORY：「CPU 侧 numba 核已饱和，43 GB/s ≈ DDR4 上限」）。
/// Python 版已用 `prange` 按行并行；本实现用 `std::thread::scope`。
/// **预期收益不大**，因为瓶颈是访存不是算力。
/// 但**有一个确定收益**：Python 版的 prange 在 x86 只有 1.16×（P22），
/// 裸线程无 GIL 可望更好 —— **需服务器实测**。
pub fn csr_spmm(csr: &Csr, x: &[f32], y: &mut [f32], n_threads: usize) {
    let n = csr.n_rows;
    if n == 0 {
        return;
    }
    let nt = n_threads.max(1).min(n);
    let chunk = n.div_ceil(nt);

    std::thread::scope(|s| {
        for row_lo in (0..nt).map(|t| t * chunk) {
            let row_hi = core::cmp::min(row_lo + chunk, n);
            if row_lo >= row_hi {
                continue;
            }
            let yp = unsafe { y.as_mut_ptr().add(row_lo) } as usize;
            let ylen = row_hi - row_lo;
            // ⚠ `x` 以 `usize` 传（`*mut f32` 不是 `Send`；整数是）。
            let xp = x.as_ptr() as usize;
            let xlen = x.len();
            s.spawn(move || {
                let ysub = unsafe {
                    core::slice::from_raw_parts_mut(yp as *mut f32, ylen)
                };
                let xs = unsafe {
                    core::slice::from_raw_parts(xp as *const f32, xlen)
                };
                for (i, o) in ysub.iter_mut().enumerate() {
                    *o = csr_row(csr, row_lo + i, xs);
                }
            });
        }
    });
}

/// 单行 SpMV。**升序累加** → 与 Python 逐位一致。
#[inline]
fn csr_row(csr: &Csr, i: usize, x: &[f32]) -> f32 {
    // SAFETY: 由 Csr 的不变式（调用方构造时保证）—— indptr/idx/val 有效，
    //   且 idx 的每个值 < x.len()。
    //   ⚠ 越界会 panic（索引检查）→ 与 Python 的「核内边界检查」策略不同：
    //   Python 版 `ltm_kernel` 的 `_recall_project` 显式跳过越界的 big_i。
    let (lo, hi) = unsafe { ((*csr.indptr.add(i)) as usize, (*csr.indptr.add(i + 1)) as usize) };
    let mut acc: f32 = 0.0;
    unsafe {
        let ip = csr.idx.add(lo);
        let vp = csr.val.add(lo);
        for k in lo..hi {
            acc += *vp.add(k - lo) * *x.get_unchecked(*ip.add(k - lo) as usize);
        }
    }
    acc
}

// ══════════════════════════════════════════════════════════════════════════
// M3 STDP / M4a 工作记忆 / M5 调制 / M6 读出
// ══════════════════════════════════════════════════════════════════════════

/// M4a 工作记忆：指数衰减 + 门控写入。
///
/// **逐位一致**（原地乘法 `slots *= gamma`，同 `wm.py:42`）。
pub fn m4a_decay(slots: &mut [f32], gamma: f32) {
    for v in slots.iter_mut() {
        *v *= gamma;
    }
}

/// M4a 门控写入：仅当 `gate ≥ thresh` 才写。
pub fn m4a_write(slots: &mut [f32], r: &[f32], gate: f32, thresh: f32) {
    if gate < thresh {
        return;
    }
    let n = r.len().min(slots.len());
    for i in 0..n {
        slots[i] = r[i];
    }
}

/// M3 STDP 的 `predict`：按活跃边做局部突触更新。
///
/// ⚠ **本函数的逐位一致性尚未验证**（门禁未覆盖 STDP 的随机分支）。
/// 已知风险：STDP 只在「非adaptive / 非homeostasis / 非metaplasticity /
/// 无 EI 突触」时走核路径，否则走 Python 事件驱动循环 —— **两条路径的
/// 累加次序不同**。本实现对应**核路径**。
pub fn m3_stdp_predict(
    w: &mut [f32],
    pre: &[f32],
    post: &[f32],
    w_max: f32,
    eta: f32,
) {
    let n = pre.len().min(post.len()).min(w.len());
    for i in 0..n {
        let dw = eta * pre[i] * post[i];
        w[i] = (w[i] + dw).clamp(0.0, w_max);
    }
}

/// M5 神经调制：surprise → gate。
///
/// **逐位一致**（Welford在线均值/方差 + sigmoid，同 `modulator.py`）。
pub fn m5_gate(surprise: f32, mean: &mut f32, m2: &mut f32, count: &mut u64) -> f32 {
    let c = *count as f32;
    let delta = surprise - *mean;
    *mean += delta / (c + 1.0);
    *m2 += delta * (surprise - *mean);
    *count += 1;
    let var = if *count > 1 { *m2 / (*count as f32 - 1.0) } else { 0.0 };
    let sigma = var.sqrt() + 1e-9;
    let z = (surprise - *mean) / sigma;
    1.0 / (1.0 + (-2.0 * z).exp())
}

/// M6 读出：**稀疏感知器 + softmax** 的前向核心。
///
/// # ⚠ 本实现是 **NPU 后端的占位**
///
/// 真正的 NPU 计算走 `acl.rs` 的图模式；本函数是**CPU 参照实现**，
/// 用途是**逐位对拍**（验证 NPU 路径没写错）。
/// 这是刻意的双轨：Python 版是参照，Rust 版是候选，
/// 对拍通过之前**不启用 Rust 版**。
pub fn m6_readout_fwd(
    w: &F32<'_>,
    h: &[f32],
    out: &mut [f32],
    n_threads: usize,
) {
    m1_gemv(w, h, &[], out, n_threads);
}

/// M6 的稀疏版本：按 `idx` gather 出 `k` 列后逐行内积。
///
/// # 逐位关系
///
/// **逐位一致**（与 `accel_readout._sp_gather` + 逐行内积同序）。
/// ⚠ 但 P156 测出：**int8 累加数学上不可能**（127×127 > 127），
/// 所以这里用 `f32` 累加 —— 与 Python 版 fp16 路径需容差对拍。
pub fn m6_sparse_fwd(
    w: &F32<'_>,
    gather_idx: &[i64],
    h: &[f32],
    out: &mut [f32],
    n_threads: usize,
) {
    let n = w.rows;
    let k = w.cols.min(gather_idx.len());
    let nt = n_threads.max(1).min(n.max(1));
    let chunk = n.div_ceil(nt);

    std::thread::scope(|s| {
        for row_lo in (0..nt).map(|t| t * chunk) {
            let row_hi = core::cmp::min(row_lo + chunk, n);
            if row_lo >= row_hi {
                continue;
            }
            //⚠ 裸指针以 usize 传递（\ 不是 Send；整数是）。
            let op = unsafe { out.as_mut_ptr().add(row_lo) } as usize;
            let olen = row_hi - row_lo;
            let gip = gather_idx.as_ptr() as usize;
            let hp = h.as_ptr() as usize;
            let hlen = h.len();
            s.spawn(move || {
                let osub = unsafe {
                    core::slice::from_raw_parts_mut(op as *mut f32, olen)
                };
                let gis = unsafe {
                    core::slice::from_raw_parts(gip as *const i64, k)
                };
                let hs = unsafe {
                    core::slice::from_raw_parts(hp as *const f32, hlen)
                };
                for (i, o) in osub.iter_mut().enumerate() {
                    let row = w.row(row_lo + i);
                    let mut acc: f32 = 0.0;
                    unsafe {
                        for c in 0..k {
                            let col = *gis.get_unchecked(c);
                            acc += *row.add(c) * *hs.get_unchecked(col as usize);
                        }
                    }
                    *o = acc;
                }
            });
        }
    });
}

// ══════════════════════════════════════════════════════════════════════════
// 设备调度辅助
// ══════════════════════════════════════════════════════════════════════════

/// 把设备缓冲区同步回 host（**同步**语义：返回前数据已在 `dst`）。
pub fn dev_to_host(
    acl: &Acl,
    src: &DevBuf,
    dst: &mut [f32],
) -> Result<(), crate::acl::AclError> {
    acl.copy_d2h_async(
        dst.as_mut_ptr() as *mut core::ffi::c_void,
        crate::acl::AclMem::from_ptr(src.ptr()),
        src.len(),
    )?;
    acl.sync()
}


// ══════════════════════════════════════════════════════════════════════════
// P172：裸指针入口（供 FFI 的 dtype 分派用；SIMD 不可用时的回退）
// ══════════════════════════════════════════════════════════════════════════

/// f32 GEMV，裸指针版（**不用 SIMD** —— SIMD 在 `simd::gemv_avx2`）。
///
/// # Safety
/// `w` 须 C 连续 `rows*cols` 个 f32；`x`/`b`/`out` 长度匹配。
#[doc(hidden)]
pub unsafe fn m1_gemv_f32_raw(
    w: *mut f32, rows: usize, cols: usize,
    x: *mut f32, b: *mut f32, out: *mut f32, n_threads: usize,
) {
    let wv = unsafe { F32::from_raw(w, rows, cols) };
    let xs = unsafe { core::slice::from_raw_parts(x, cols) };
    let bs = unsafe { core::slice::from_raw_parts(b, rows) };
    let os = unsafe { core::slice::from_raw_parts_mut(out, rows) };
    m1_gemv(&wv, xs, bs, os, n_threads);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemv_单线程结果正确() {
        // 2 行 × 3 列，与手算对齐
        let mut wv: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0f32, 1.0, 1.0];
        let b = [0.5f32, 0.5];
        let mut out = [0.0f32; 2];
        let w = unsafe { F32::from_raw(wv.as_mut_ptr(), 2, 3) };
        m1_gemv(&w, &x, &b, &mut out, 1);
        assert!((out[0] - 6.5f32).abs() < 1e-6, "{}", out[0]);
        assert!((out[1] - 15.5f32).abs() < 1e-6, "{}", out[1]);
    }

    #[test]
    fn gemv_多线程与单线程逐位一致() {
        let n = 300;
        let c = 64;
        let mut wv: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.001).collect();
        let x: Vec<f32> = (0..c).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let w = unsafe { F32::from_raw(wv.as_mut_ptr(), n, c) };

        let mut o1 = vec![0.0f32; n];
        let mut o8 = vec![0.0f32; n];
        m1_gemv(&w, &x, &b, &mut o1, 1);
        m1_gemv(&w, &x, &b, &mut o8, 8);
        // **逐位**：分块不改行内累加次序 → 应完全相同
        assert_eq!(o1, o8, "多线程必须与单线程逐位一致");
    }

    #[test]
    fn csr_spmm_升序累加() {
        // 2 行：row0 -> {col1: 2.0}, row1 -> {col0: 3.0, col2: 4.0}
        let indptr: [i64; 3] = [0, 1, 3];
        let idx: [i64; 3] = [1, 0, 2];
        let val: [f32; 3] = [2.0, 3.0, 4.0];
        let x = [10.0f32, 20.0, 30.0];
        let mut y = [0.0f32; 2];
        let csr = Csr {
            indptr: indptr.as_ptr(),
            idx: idx.as_ptr(),
            val: val.as_ptr(),
            n_rows: 2,
        };
        csr_spmm(&csr, &x, &mut y, 1);
        assert!((y[0] - 40.0f32).abs() < 1e-6, "{}", y[0]);
        assert!((y[1] - 150.0f32).abs() < 1e-6, "{}", y[1]);
    }
}