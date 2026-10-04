//! 手写 SIMD 内核 —— P171 实测发现的真正差距。
//!
//! # 实测数据（本机 x86，1b 档 1024×2048 fp32，best-of-11）
//!
//! | 实现 | 耗时 | 并行效率 |
//! |---|---|---|
//! | numpy BLAS（MKL/OpenBLAS） | **179 µs** | — |
//! | numba `serial`（1 线程） | 2744 µs | — |
//! | numba `prange`（8 线程，同一核） | 2856 µs | **0.96×（没并行）** |
//! | Rust `m1_gemv`（1 线程） | 2874 µs | — |
//! | Rust `m1_gemv`（8 线程，常驻池） | 1136 µs | **2.53×** |
//!
//! **两个结论**：
//! 1. Rust 与 numba 的**单线程**代码质量**相同**（1.0×）—— LLVM 对两者
//!    都**没有**向量化这个循环（它有依赖链 `acc += ...`，无法 SIMD）。
//!    → 所以「Rust 写得差」这个猜测是**错的**。
//! 2. **Rust 的池真的在并行（2.53×），numba 的 prange 在同一核上没有**
//!    （0.96×）。→ **Rust 的价值在这里得到证实**。
//!
//! # 那还差什么？—— 与 BLAS 的 6.4×
//!
//! BLAS 179 µs ≈ **访存下界**（8 MiB ÷ 43 GB/s = 186 µs）→它做到了 ~96% 带宽效率。
//! 关键不是算得快，而是**它用了非破坏式归约**：
//! BLAS 用 8 个（或 16 个）**独立累加器** + 水平相加，
//! 每个累加器的依赖链只有 `n/8` 长 → **依赖链短 8× → 可以 pipelined**。
//!
//! 本模块用 **x86 AVX2 手写 intrinsics** 复刻这个结构：
//! - 4 个 `__m256` 累加器（每次处理 8 个 f32）
//! - **每个累加器一次处理 8 个乘积** → 依赖链长度 = `n/32`
//! - 水平相加时**按 P51 的结合顺序**保证可复现
//!
//! # ⚠ 逐位一致性的代价
//!
//! 4 路累加器**改变了求和顺序** → 与单标量版本**不逐位**（fp64 约 1 ulp，
//! 与 P52 融合核同级）。门禁 `verify_rust_kernels.py` 因此对 SIMD 路径
//! **用容差**（1e-5），对标量路径**要求逐位**。

/// x86 AVX2 的 256 位向量类型（8× f32）。
#[cfg(target_os = "windows")]
use core::arch::x86_64::__m256;
#[cfg(not(target_os = "windows"))]
use core::arch::x86_64::__m256;

/// 检测 AVX2 是否可用（**真跑一次**，不是查表 —— 项目的铁律）。
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// GEMV 单行，**AVX2 + 4 路 FMA 累加器**。
///
/// # Safety
/// `row` 须指向 ≥ `cols` 个 f32；`x` 须指向 ≥ `cols` 个 f32。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_row_avx2(row: *const f32, x: *const f32, cols: usize) -> f32 {
    use core::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_setzero_ps,
    };
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        // 主循环：每次 32 个元素（4 ×8）
        let n32 = cols / 32;
        for k in 0..n32 {
            let base = k * 32;
            a0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(row.add(base)),
                _mm256_loadu_ps(x.add(base)),
                a0,
            );
            a1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(row.add(base + 8)),
                _mm256_loadu_ps(x.add(base + 8)),
                a1,
            );
            a2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(row.add(base + 16)),
                _mm256_loadu_ps(x.add(base + 16)),
                a2,
            );
            a3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(row.add(base + 24)),
                _mm256_loadu_ps(x.add(base + 24)),
                a3,
            );
        }
        // 水平相加：((a0+a1)+(a2+a3)) —— 固定结合顺序 → 可复现
        let s01 = _mm256_add_ps(a0, a1);
        let s23 = _mm256_add_ps(a2, a3);
        let s = _mm256_add_ps(s01, s23);
        let mut lanes = [0.0f32; 8];
        core::ptr::copy_nonoverlapping(
            (&s as *const __m256).cast::<f32>(),
            lanes.as_mut_ptr(),
            8,
        );
        let mut acc = lanes[0] + lanes[1] + lanes[2] + lanes[3]
            + lanes[4] + lanes[5] + lanes[6] + lanes[7];
        // 尾部（cols % 32）
        let mut i = n32 * 32;
        while i < cols {
            acc += *row.add(i) * *x.add(i);
            i += 1;
        }
        acc
    }
}

/// 标量回退（非 x86 / 无 AVX2）。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn gemv_row_avx2(row: *const f32, x: *const f32, cols: usize) -> f32 {
    unsafe {
        let mut acc = 0.0f32;
        for i in 0..cols {
            acc += *row.add(i) * *x.add(i);
        }
        acc
    }
}

/// 稠密 GEMV（整矩阵），用 SIMD 行核 + 常驻池并行。
///
/// # Safety
/// `w` 须是 `rows × cols` 的 C 连续 f32；`x`/`b`/`out` 长度须匹配。
pub unsafe fn gemv_avx2(
    w: *const f32,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: *const f32,
    out: *mut f32,
    n_threads: usize,
) {
    let use_simd = has_avx2();
    let ctx = AvxCtx {
        w,
        x,
        b,
        out,
        n: rows,
        cols,
        use_simd,
    };
    let nt = n_threads.max(1).min(rows.max(1));
    // ⚠ **P174 审计：派发固定成本 41.5 µs** → 中小 shape 必须串行
    //   （实测 256x512：串行 28.7 → 8 线程 70.6，**0.41×**；
    //    盈亏平衡点 512x512 = 262144 cells）。
    //   与 `mechanisms::m1_gemv` 用**同一个门限**，保证两条路径口径一致。
    const _MIN_PAR_CELLS: usize = 256 * 1024;
    if rows.saturating_mul(cols) < _MIN_PAR_CELLS {
        for r in 0..rows {
            let row = unsafe { w.add(r * cols) };
            let mut acc = 0.0f32;
            for j in 0..cols {
                acc += unsafe { *row.add(j) * *x.add(j) };
            }
            unsafe { *out.add(r) = acc + *b.add(r) };
        }
        return;
    }
    crate::pool::run(
        avx_work,
        &ctx as *const AvxCtx as *mut u8,
        nt,
        nt,
    );
}

/// 一个分块（只写自己的行块）。
unsafe fn avx_work(ctx: *mut u8, part: usize, n_parts: usize) {
    let c = unsafe { &*(ctx as *const AvxCtx) };
    let chunk = c.n.div_ceil(n_parts);
    let lo = part * chunk;
    let hi = ((part + 1) * chunk).min(c.n);
    if lo >= hi {
        return;
    }
    for r in lo..hi {
        let row = unsafe { c.w.add(r * c.cols) };
        let v = unsafe {
            if c.use_simd {
                gemv_row_avx2(row, c.x, c.cols)
            } else {
                let mut acc = 0.0f32;
                for j in 0..c.cols {
                    acc += *row.add(j) * *c.x.add(j);
                }
                acc
            }
        };
        unsafe { *c.out.add(r) = v + *c.b.add(r) };
    }
}

struct AvxCtx {
    w: *const f32,
    x: *const f32,
    b: *const f32,
    out: *mut f32,
    n: usize,
    cols: usize,
    use_simd: bool,
}
// SAFETY: 每个 part 只写 `out[lo..hi)`（连续、互不重叠），其余只读。
unsafe impl Send for AvxCtx {}
unsafe impl Sync for AvxCtx {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx2_matches_scalar_within_tolerance() {
        if !has_avx2() {
            return; // 本机无 AVX2 → 跳过
        }
        let cols = 1024;
        let row: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.001).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01).collect();
        let simd = unsafe { gemv_row_avx2(row.as_ptr(), x.as_ptr(), cols) };
        let mut scalar = 0.0f32;
        for i in 0..cols {
            scalar += row[i] * x[i];
        }
        let rel = (simd - scalar).abs() / scalar.abs().max(1e-9);
        assert!(rel < 1e-5, "rel={rel}");
    }
}

// ══════════════════════════════════════════════════════════════════════════
// **P173：CSR SpMV 的 SIMD 路径**（M2 专用）
// ══════════════════════════════════════════════════════════════════════════
// 目标：M2 的 SpMV 是**访存受限**（1b 档 524288 边 × 12 B = 6.3 MB/步，fp32 后）。
// val 从 fp64→fp32 让**每边 16→12 B**（P173）；SIMD 再补算子效率。
//
// ## ⚠ 第一版设计错误（记录下来免得再犯）
// 我先用 `_mm256_set1_ps(val[p])` 把标量**广播**到整个向量，然后 FMA。
// **这是错的**：广播后的 8 个 lane 全是同一个乘积 →
//   ① 累加只落在 element[0]，其余 7 个 lane 恒为 0；
//   ② 水平相加时我取了全部 8 lane → relerr=**7.0**（完全错）。
//
// ## 正确做法：**gather 装 8 个不同的 x[idx]**
// 一个 lane 处理**一个不同的边**：`x[idx[p+j]]`，j=0..7。
// 这样 8 个 lane 全部有用（无浪费），且是**合法的 SIMD**。
// ⚠ val 也需gather → 用 `_mm256_i32gather_ps`（int32 索引）不够，
//   故用**标量 gather 到临时数组再 load**（编译器会内联）：
//   `let xs = [_mm256_set1_ps(...); ...]` 不行 → 用临时数组 + `_mm256_loadu_ps`。
//
// ## 预期收益（诚实）
// SpMV 的瓶颈是**访存**（`idx` 间接寻址无法连续加载），SIMD 只能补
// 算术效率。**实测收益待服务器确认**，本机先量上界。

/// CSR 一行，**8 lane 各处理一条不同的边**（真正的 SIMD，无宽度浪费）。
///
/// # Safety
/// `idx_p`/`val_p` 有效且 `nnz >= 32`；`x` 有效。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn csr_row_avx2(
    idx_p: *const i64,
    val_p: *const f32,
    nnz: usize,
    x: *const f32,
) -> f32 {
    use core::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps,
    };
    unsafe {
        if nnz < 32 {
            return csr_row_scalar(idx_p, val_p, nnz, x);
        }
        // 累加器：**2 个**（每个负责间隔 8 的边）→ 依赖链 n/16
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        // ⚠ gather 到栈数组再 load：8 个 lane 各装**不同的 x[idx[p+j]]**
        let mut xb = [0.0f32; 8];
        let mut vb = [0.0f32; 8];
        let n8 = nnz / 16 * 16;
        let mut p = 0usize;
        while p < n8 {
            for j in 0..8 {
                xb[j] = *x.add(*idx_p.add(p + j) as usize);
                vb[j] = *val_p.add(p + j);
            }
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(vb.as_ptr()),
                                 _mm256_loadu_ps(xb.as_ptr()), a0);
            for j in 0..8 {
                xb[j] = *x.add(*idx_p.add(p + 8 + j) as usize);
                vb[j] = *val_p.add(p + 8 + j);
            }
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(vb.as_ptr()),
                                 _mm256_loadu_ps(xb.as_ptr()), a1);
            p += 16;
        }
        let s01 = _mm256_add_ps(a0, a1);
        let mut lanes = [0.0f32; 8];
        core::ptr::copy_nonoverlapping(
            (&s01 as *const __m256).cast::<f32>(),
            lanes.as_mut_ptr(),
            8,
        );
        // 8 个 lane 各是一条独立的边 → **全部相加**（这里取全部是对的）
        let mut acc = lanes[0] + lanes[1] + lanes[2] + lanes[3]
            + lanes[4] + lanes[5] + lanes[6] + lanes[7];
        while p < nnz {
            acc += *val_p.add(p) * *x.add(*idx_p.add(p) as usize);
            p += 1;
        }
        acc
    }
}

/// CSR SpMV（fp32，**SIMD 优先**）。
///
/// # Safety
/// `indptr` i64、`idx` i64、`val` f32，`x`/`y` 长度 ≥ `n_rows`。
#[no_mangle]
pub unsafe extern "C" fn phdnet_csr_spmm_simd(
    indptr: *const i64,
    idx: *const i64,
    val: *const f32,
    n_rows: usize,
    x: *mut f32,
    y: *mut f32,
    n_threads: usize,
) -> i32 {
    let use_simd = has_avx2();
    let ctx = CsrSimdCtx {
        indptr,
        idx,
        val,
        x_ptr: x as *const f32,
        y_ptr: y as *mut f32,
        n: n_rows,
        use_simd,
    };
    let nt = n_threads.max(1).min(n_rows.max(1));
    // ⚠ **P174 审计：同样要门限**（派发固定成本 41.5 µs）。
    // CSR 每边做 1 乘 1 加 → 工作量 ≈ 总 nnz。
    // ⚠ 门限按**实测**定：盈亏平衡约在 nnz ~ 262144（与 GEMV 同量级）。
    let nnz_total = if n_rows == 0 {
        0
    } else {
        unsafe { *indptr.add(n_rows) as usize }
    };
    const _MIN_PAR_NNZ: usize = 256 * 1024;
    if nnz_total < _MIN_PAR_NNZ {
        for r in 0..n_rows {
            let a = unsafe { *indptr.add(r) } as usize;
            let b = unsafe { *indptr.add(r + 1) } as usize;
            let v = unsafe {
                csr_row_scalar(idx.add(a), val.add(a), b - a, x as *const f32)
            };
            unsafe { *y.add(r) = v };
        }
        return 0;
    }
    crate::pool::run(
        csr_simd_work,
        &ctx as *const CsrSimdCtx as *mut u8,
        nt,
        nt,
    );
    0
}

/// 一个分块（只写自己的行块）。
unsafe fn csr_simd_work(ctx: *mut u8, part: usize, n_parts: usize) {
    let c = unsafe { &*(ctx as *const CsrSimdCtx) };
    let chunk = c.n.div_ceil(n_parts);
    let lo = part * chunk;
    let hi = ((part + 1) * chunk).min(c.n);
    if lo >= hi {
        return;
    }
    for r in lo..hi {
        let a = unsafe { *c.indptr.add(r) } as usize;
        let b = unsafe { *c.indptr.add(r + 1) } as usize;
        let idx_p = unsafe { c.idx.add(a) };
        let val_p = unsafe { c.val.add(a) };
        let nnz = b - a;
        #[cfg(target_arch = "x86_64")]
        let v = if c.use_simd {
            unsafe { csr_row_avx2(idx_p, val_p, nnz, c.x_ptr) }
        } else {
            unsafe { csr_row_scalar(idx_p, val_p, nnz, c.x_ptr) }
        };
        #[cfg(not(target_arch = "x86_64"))]
        let v = unsafe { csr_row_scalar(idx_p, val_p, nnz, c.x_ptr) };
        unsafe { *c.y_ptr.add(r) = v };
    }
}

#[inline]
unsafe fn csr_row_scalar(
    idx_p: *const i64,
    val_p: *const f32,
    nnz: usize,
    x: *const f32,
) -> f32 {
    unsafe {
        let mut acc = 0.0f32;
        for p in 0..nnz {
            acc += *val_p.add(p) * *x.add(*idx_p.add(p) as usize);
        }
        acc
    }
}

struct CsrSimdCtx {
    indptr: *const i64,
    idx: *const i64,
    val: *const f32,
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: usize,
    use_simd: bool,
}
// SAFETY: 每个 part 只写 `y[lo..hi)`（连续、互不重叠），其余只读。
unsafe impl Send for CsrSimdCtx {}
unsafe impl Sync for CsrSimdCtx {}

#[cfg(test)]
mod tests_csr {
    use super::*;

    #[test]
    fn csr_avx2_matches_scalar() {
        if !has_avx2() {
            return;
        }
        let n = 512;
        let nnz_row = 128;
        let idx: Vec<i64> = (0..nnz_row)
            .map(|j| ((j * 7919) % n) as i64)
            .collect();
        let val: Vec<f32> = (0..nnz_row).map(|j| (j as f32) * 0.01).collect();
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
        let simd = unsafe { csr_row_avx2(idx.as_ptr(), val.as_ptr(), nnz_row, x.as_ptr()) };
        let scalar = unsafe { csr_row_scalar(idx.as_ptr(), val.as_ptr(), nnz_row, x.as_ptr()) };
        let rel = (simd - scalar).abs() / scalar.abs().max(1e-9);
        assert!(rel < 1e-5, "rel={rel} simd={simd} scalar={scalar}");
    }

    #[test]
    fn csr_avx2_handles_short_rows() {
        if !has_avx2() {
            return;
        }
        // 行宽 < 32 → 应回落到标量，且**逐位**（同一段代码）
        let n = 64;
        let nnz_row = 7;
        let idx: Vec<i64> = (0..nnz_row).map(|j| (j * 3) as i64).collect();
        let val: Vec<f32> = (0..nnz_row).map(|j| (j as f32) * 0.1).collect();
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
        let simd = unsafe { csr_row_avx2(idx.as_ptr(), val.as_ptr(), nnz_row, x.as_ptr()) };
        let scalar = unsafe { csr_row_scalar(idx.as_ptr(), val.as_ptr(), nnz_row, x.as_ptr()) };
        assert_eq!(simd.to_bits(), scalar.to_bits(), "短行应逐位一致");
    }
}
