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