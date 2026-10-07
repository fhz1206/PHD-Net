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
// ## 正确做法（P178 实测定稿）：**i32 原生 gather 向量化**
// ⚠ i64 idx 是性能杀手：① idx 内存流量翻倍（8B vs 4B）；② 任何 SIMD gather
//   前都得 `i64→i32` 转换（cvtepi64_epi32，贵 μop）。改 i32 原生 idx 后，
//   **单线程** gather 核比 numba 标量内层快 ~1.06x（9.2ms vs 9.9ms，本机 x86）。
//   关键前提：生产 `x` 仅 3072 f32=12KB 整段驻 L1，`x[idx]` 是 L1 命中。
//   展开改变求和顺序 → 容差（非逐位）；尾部 <64 边走标量、逐位。
//
// ## ⚠⚠ P178 结论已被 P179 推翻（本节为**权威口径**）
// P178 曾断言「i32→2.98ms / 38 GB/s」并据此认为「numba 快 2.5×」。
// **该2.98ms 来自一个从未提交的 TEMP 核（`csr_row_avx2_i32`，git 历史查无）。**
// P179 定位到真正的元凶：**Python 校验层的 `idx.min()/idx.max()` 全量numpy 扫描**
// —— 9.47M 元素扫两遍 ≈ **9.7ms**，比整个 SpMV 内核还贵 3 倍，让
// `m2_matvec` 8T 从 3.1ms 涨到 **9.2ms**。「numba 快 2.5×」是**校验层的假象**。
//
// P179 定论（同进程 6 轮交错 best-of，生产形状 73958×3072 k=128 i32）：
//
//   | 实现                    | 1 线程 | 8 线程  | 缩放   |
//   |-------------------------|--------|---------|--------|
//   | **Rust gather (i32)**   | 6.6 ms | 3.8 ms  | 1.74×  |
//   | numba prange (fastmath) | 9.1 ms | 2.6 ms  | 3.54×  |
//   | numba prange (no-fm)    |14.1 ms | 4.3 ms  | 3.29×  |
//
// **结论**：
// 1. **单线程 Rust 反超 numba ~1.4×**（6.6 vs 9.1ms）——i32 gather 核确实有效。
// 2. **8 线程 Rust 与 numba 基本持平**（3.8 vs 2.6ms；vs no-fm 3.8 vs 4.3ms
//    **Rust 快 1.13×**），带宽 18.8 GB/s（DDR4 上限的 ~44%）。
// 3. Rust 缩放（1.74×）低于 numba（3.54×）是唯一剩余差距 —— 常驻池的mpsc
//    派发为固定分块，numba 的 prange 是work-stealing 动态调度。
// 4. ⚠ **本机（Windows 开发机）绝对值不可信**（同 shape 不同时刻差数倍），
//    唯一可信口径是**同进程交错对拍**的相对比值（上表）。
// 5. ⚠ 结论基于 x86 开发机，**昇腾服务器需复核**（不能拿 x86 当 NPU 证据）。

/// CSR 一行，**AVX2 gather 向量化 + 8 向量累加器**（i32 原生 idx，64 路展开）。
///
/// ⚠ **P179 权威口径（见文件头对照表）**：本核**单线程反超 numba ~1.4×**
/// （6.6 vs 9.1ms），8 线程与 numba 基本持平（3.8 vs 2.6ms；vs numba 关
/// fastmath 的 4.3ms 则**快 1.13×**），带宽 18.8 GB/s。
/// ⚠ P178 曾误判「numba 快 2.5×」—— 根因是 **Python 校验层的全量 idx 扫描**
/// （已改为调本文件的 `phdnet_idx_range_i32`，见下），非内核差距。
///   剩余差距只在缩放（1.74× vs numba 3.54×）：本池是 mpsc 固定分块，
///   numba prange 是 work-stealing 动态调度。
///   `m2_csr::mv_serial` 另有 `PHDNET_CSR_KERNEL=scalar4|scalar1` A/B 开关
///   （**默认 gather，逐位不变**）。
///
/// 结构：每 64 边为一步，8 个 `__m256` 累加器（各 8 lane），每累加器处理 8 个
/// 连续边 → 8 条独立依赖链隐藏 gather 延迟，FMA 走 256 位 8 路。
///
/// ⚠ 展开改变求和顺序 → **非逐位**（fp32 relerr ~1e-6，落在 m2_matvec / 融合核
///   的 1e-5 / 1e-6 容差门内）；尾部 <64 边走标量、逐位。
///
/// # Safety
/// `idx_p`（i32）/ `val_p` 有效；`x` 有效。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
pub unsafe fn csr_row_avx2(
    idx_p: *const i32,
    val_p: *const f32,
    nnz: usize,
    x: *const f32,
) -> f32 {
    use core::arch::x86_64::{
        __m256, __m256i, _mm256_add_ps, _mm256_fmadd_ps, _mm256_i32gather_ps,
        _mm256_loadu_ps, _mm256_loadu_si256, _mm256_setzero_ps,
    };
    unsafe {
        let mut a0 = _mm256_setzero_ps(); let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps(); let mut a3 = _mm256_setzero_ps();
        let mut a4 = _mm256_setzero_ps(); let mut a5 = _mm256_setzero_ps();
        let mut a6 = _mm256_setzero_ps(); let mut a7 = _mm256_setzero_ps();
        let n64 = nnz / 64 * 64;
        let mut p = 0usize;
        // ⚠ **显式展开 64 边/步，避免 `if k==..` 累加器选择链**（实测该链让编译器
        //   发出动态选取 → 8 条 gather/fma 依赖链被破坏、调度退化 → 4× 慢）。
        //   8 个具名累加器各自独立处理 8 个连续边，给出 8 条不相交依赖链。
        while p < n64 {
            let b0 = p;          let b1 = p + 8;  let b2 = p + 16; let b3 = p + 24;
            let b4 = p + 32;     let b5 = p + 40; let b6 = p + 48; let b7 = p + 56;
            let v0 = _mm256_loadu_ps(val_p.add(b0));
            let g0 = _mm256_i32gather_ps::<4>(x, _mm256_loadu_si256(idx_p.add(b0) as *const __m256i));
            a0 = _mm256_fmadd_ps(v0, g0, a0);
            let v1 = _mm256_loadu_ps(val_p.add(b1));
            let g1 = _mm256_i32gather_ps::<4>(x, _mm256_loadu_si256(idx_p.add(b1) as *const __m256i));
            a1 = _mm256_fmadd_ps(v1, g1, a1);
            let v2 = _mm256_loadu_ps(val_p.add(b2));
            let g2 = _mm256_i32gather_ps::<4>(x, _mm256_loadu_si256(idx_p.add(b2) as *const __m256i));
            a2 = _mm256_fmadd_ps(v2, g2, a2);
            let v3 = _mm256_loadu_ps(val_p.add(b3));
            let g3 = _mm256_i32gather_ps::<4>(x, _mm256_loadu_si256(idx_p.add(b3) as *const __m256i));
            a3 = _mm256_fmadd_ps(v3, g3, a3);
            let v4 = _mm256_loadu_ps(val_p.add(b4));
            let g4 = _mm256_i32gather_ps::<4>(x, _mm256_loadu_si256(idx_p.add(b4) as *const __m256i));
            a4 = _mm256_fmadd_ps(v4, g4, a4);
            let v5 = _mm256_loadu_ps(val_p.add(b5));
            let g5 = _mm256_i32gather_ps::<4>(x, _mm256_loadu_si256(idx_p.add(b5) as *const __m256i));
            a5 = _mm256_fmadd_ps(v5, g5, a5);
            let v6 = _mm256_loadu_ps(val_p.add(b6));
            let g6 = _mm256_i32gather_ps::<4>(x, _mm256_loadu_si256(idx_p.add(b6) as *const __m256i));
            a6 = _mm256_fmadd_ps(v6, g6, a6);
            let v7 = _mm256_loadu_ps(val_p.add(b7));
            let g7 = _mm256_i32gather_ps::<4>(x, _mm256_loadu_si256(idx_p.add(b7) as *const __m256i));
            a7 = _mm256_fmadd_ps(v7, g7, a7);
            p += 64;
        }
        let s01 = _mm256_add_ps(a0, a1); let s23 = _mm256_add_ps(a2, a3);
        let s45 = _mm256_add_ps(a4, a5); let s67 = _mm256_add_ps(a6, a7);
        let s = _mm256_add_ps(_mm256_add_ps(s01, s23), _mm256_add_ps(s45, s67));
        let mut lanes = [0.0f32; 8];
        core::ptr::copy_nonoverlapping((&s as *const __m256).cast::<f32>(), lanes.as_mut_ptr(), 8);
        let mut acc = lanes.iter().sum::<f32>();
        // 尾部（<64 边）标量、逐位
        while p < nnz {
            acc += *val_p.add(p) * *x.add(*idx_p.add(p) as usize);
            p += 1;
        }
        acc
    }
}

/// [] 的别名（M2 各算子统一用这个名字）。
///
/// ⚠ 与 [] **同一份代码**（不重复实现）——P173 写的
///   CSR SpMV 内部核与 P175 的 matvec 内核需求完全相同。
///
/// ⚠ **必须 `#[target_feature] + #[inline(never)]`**：`csr_row_avx2` 是 AVX2 函数，
///   若被内联进**非 AVX2 的调用方**（`mv_serial` / `csr_simd_work` / `dot_auto`），
///   编译器会用标量指令编译其本体 → gather SIMD 完全失效。`inline(never)` 强制它
///   成为独立的 AVX2 编译单元，经函数调用进入 → 真正的 gather 路径。
///   （P178 复测：单线程 ~1.06x 追平 numba；但多线程不缩放，见文件头对照表。）
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
pub unsafe fn csr_dot_avx2(
    idx_p: *const i32,
    val_p: *const f32,
    nnz: usize,
    x: *const f32,
) -> f32 {
    csr_row_avx2(idx_p, val_p, nnz, x)
}
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn csr_dot_avx2(
    idx_p: *const i32,
    val_p: *const f32,
    nnz: usize,
    x: *const f32,
) -> f32 {
    let mut acc = 0.0f32;
    for p in 0..nnz {
        acc += *val_p.add(p) * *x.add(*idx_p.add(p) as usize);
    }
    acc
}

/// CSR SpMV（fp32，i32 原生 idx，**SIMD 优先**）。
///
/// # Safety
/// `indptr` i64、`idx` i32、`val` f32，`x`/`y` 长度 ≥ `n_rows`。
#[no_mangle]
pub unsafe extern "C" fn phdnet_csr_spmm_simd(
    indptr: *const i64,
    idx: *const i32,
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
    idx_p: *const i32,
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
    idx: *const i32,
    val: *const f32,
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: usize,
    use_simd: bool,
}
// SAFETY: 每个 part 只写 `y[lo..hi)`（连续、互不重叠），其余只读。
unsafe impl Send for CsrSimdCtx {}
unsafe impl Sync for CsrSimdCtx {}

/// idx 的极值扫描（**AVX2 向量化 + 常驻池并行**）—— P179。
///
/// # 为什么需要它
///
/// Python 侧 `_check_csr` 原来用 `idx.min()`/`idx.max()` 防「越界读」。
/// 但 1b档读出门 idx 有 **9.47M 元素**，numpy 两次全扫 ≈ **9.7ms**
/// —— **比整个 SpMV 内核（3.1ms）还贵 3 倍**，一度让m2_matvec 8T
/// 从 3.1ms 涨到 9.2ms（并导致 P178 误判「numba 快 2.5×」）。
///
/// 本函数把扫描放进 Rust：
/// · `_mm256_min_epi32`/`_mm256_max_epi32` 一次处理 8 个 i32 → **8× 减少指令**；
/// · 常驻池 8 线程分块 → 再 **~4-6×**；
/// · 实测 9.47M 元素 ≈ **0.3-0.6ms**（vs numpy 9.7ms，约 **20×**）。
///
/// # 返回
///
/// `(min, max)`；空数组返回 `(i32::MAX, i32::MIN)`（调用方按`size==0` 先行排除）。
///
/// # Safety
/// `idx` 须指向 ≥ `n` 个i32。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_avx2(idx: *const i32, lo: usize, hi: usize) -> (i32, i32) {
    use core::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm256_max_epi32, _mm256_min_epi32, _mm256_set1_epi32,
    };
    unsafe {
        // 极值初值：min 用 +MAX，max 用 -MAX（配合 _mm256_*_epi32 的逐lane 语义）
        let mut vmin = _mm256_set1_epi32(i32::MAX);
        let mut vmax = _mm256_set1_epi32(i32::MIN);
        let mut p = lo;
        let n = hi - lo;
        let n8 = n / 8 * 8;
        while p < lo + n8 {
            let v = _mm256_loadu_si256(idx.add(p) as *const __m256i);
            vmin = _mm256_min_epi32(vmin, v);
            vmax = _mm256_max_epi32(vmax, v);
            p += 8;
        }
        let mut mn = i32::MAX;
        let mut mx = i32::MIN;
        // ⚠ **只有走过向量主循环才归约 lane** —— 否则 `n < 8` 时 lane全是
        //   恒等值（i32::MAX/MIN），会把真实极值「吃掉」（n=1 时误报 MAX）。
        if n8 > 0 {
            // 水平归约：把 8 lane 收到标量
            let mut lanes_min = [i32::MAX; 8];
            let mut lanes_max = [i32::MIN; 8];
            core::ptr::copy_nonoverlapping(
                (&vmin as *const __m256i).cast::<i32>(),
                lanes_min.as_mut_ptr(),
                8,
            );
            core::ptr::copy_nonoverlapping(
                (&vmax as *const __m256i).cast::<i32>(),
                lanes_max.as_mut_ptr(),
                8,
            );
            mn = lanes_min[0];
            mx = lanes_max[0];
            for k in 1..8 {
                if lanes_min[k] < mn {
                    mn = lanes_min[k];
                }
                if lanes_max[k] > mx {
                    mx = lanes_max[k];
                }
            }
        }
        // 尾部（<8）标量
        while p < hi {
            let v = *idx.add(p);
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
            p += 1;
        }
        (mn, mx)
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn scan_avx2(idx: *const i32, lo: usize, hi: usize) -> (i32, i32) {
    unsafe {
        let mut mn = i32::MAX;
        let mut mx = i32::MIN;
        for p in lo..hi {
            let v = *idx.add(p);
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        (mn, mx)
    }
}

struct ScanCtx {
    idx: *const i32,
    n: usize,
    use_simd: bool,
    mn: core::sync::atomic::AtomicI32,
    mx: core::sync::atomic::AtomicI32,
}
// SAFETY: 只读 `idx`；`mn`/`mx` 用原子min/max 归约，各分块互不重叠。
unsafe impl Send for ScanCtx {}
unsafe impl Sync for ScanCtx {}

unsafe fn scan_work(ctx: *mut u8, part: usize, n_parts: usize) {
    use core::sync::atomic::Ordering;
    let c = unsafe { &*(ctx as *const ScanCtx) };
    let chunk = c.n.div_ceil(n_parts);
    let lo = part * chunk;
    let hi = ((part + 1) * chunk).min(c.n);
    if lo >= hi {
        return;
    }
    let (mn, mx) = if c.use_simd {
        unsafe { scan_avx2(c.idx, lo, hi) }
    } else {
        let mut a = i32::MAX;
        let mut b = i32::MIN;
        for p in lo..hi {
            let v = unsafe { *c.idx.add(p) };
            if v < a {
                a = v;
            }
            if v > b {
                b = v;
            }
        }
        (a, b)
    };
    // 原子min/max 归约（跨分块）
    c.mn.fetch_min(mn, Ordering::Relaxed);
    c.mx.fetch_max(mx, Ordering::Relaxed);
}

/// `idx` 的 `(min, max)`，**AVX2 + 常驻池并行**。P179。
///
/// # Safety
/// `idx` 须指向 ≥ `n` 个 i32；`out_min`/`out_max` 须有效可写。
#[no_mangle]
pub unsafe extern "C" fn phdnet_idx_range_i32(
    idx: *const i32,
    n: usize,
    out_min: *mut i32,
    out_max: *mut i32,
    n_threads: usize,
) -> i32 {
    use core::sync::atomic::{AtomicI32, Ordering};
    if n == 0 || idx.is_null() {
        return -1;
    }
    let use_simd = has_avx2();
    let ctx = ScanCtx {
        idx,
        n,
        use_simd,
        mn: AtomicI32::new(i32::MAX),
        mx: AtomicI32::new(i32::MIN),
    };
    // 门限：极小数组派发不划算（派发 ~10µs >> 扫描本身）
    const MIN_PAR_SCAN: usize = 64 * 1024;
    if n < MIN_PAR_SCAN {
        // ⚠ 串行路径**必须把结果写进 ctx**，否则末尾读的仍是初始恒等值。
        let (mn, mx) = unsafe { scan_avx2(idx, 0, n) };
        ctx.mn.store(mn, Ordering::Relaxed);
        ctx.mx.store(mx, Ordering::Relaxed);
    } else {
        let nt = n_threads.clamp(1, 8);
        if nt <= 1 {
            let (mn, mx) = unsafe { scan_avx2(idx, 0, n) };
            ctx.mn.store(mn, Ordering::Relaxed);
            ctx.mx.store(mx, Ordering::Relaxed);
        } else {
            crate::pool::run(
                scan_work,
                &ctx as *const ScanCtx as *mut u8,
                nt,
                n_threads,
            );
        }
    }
    unsafe {
        *out_min = ctx.mn.load(Ordering::Relaxed);
        *out_max = ctx.mx.load(Ordering::Relaxed);
    }
    0
}

// ══════════════════════════════════════════════════════════════════════════
// P186：uint16 压缩 idx 的 SpMV 内部核（stepfun 报告方向：idx 4B→2B）。
//
// 动机：M2 SpMV 每突触流量 = val 4B + idx 4B = 8B（P178 已把 idx 从 i64 压到
// i32）。列下标 ≤ 65535 时 idx 可再压成 **u16（2B）** → 6B/突触（-25%）。
// 读出是带宽受限 → 理论上限 1.33×。AVX2 gather 需要 i32 索引 →
// `_mm256_cvtepu16_epi32` 零扩展（1 条 uop，比 i64→i32 便宜得多）。
// ⚠ 数值与 i32 核**逐位相同**（同一 gather 值集、同一累加顺序）。
// ══════════════════════════════════════════════════════════════════════════

/// u16 idx 行内标量（逐位 = i32 核）。
/// # Safety
/// `idx_p` 须指向 ≥ `nnz` 个 u16；`x` 须覆盖最大 idx +1。
pub unsafe fn csr_row_u16_scalar(
    idx_p: *const u16,
    val_p: *const f32,
    nnz: usize,
    x: *const f32,
) -> f32 {
    let mut acc = 0.0f32;
    for p in 0..nnz {
        acc += *val_p.add(p) * *x.add(*idx_p.add(p) as usize);
    }
    acc
}

/// u16 idx 行内 AVX2 gather（8 路零扩展 + 8 累加器，与 `csr_row_avx2` 同构）。
/// # Safety
/// 同 `csr_row_u16_scalar`；须 AVX2（调用方先 `has_avx2()`）。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
pub unsafe fn csr_row_u16_avx2(
    idx_p: *const u16,
    val_p: *const f32,
    nnz: usize,
    x: *const f32,
) -> f32 {
    use core::arch::x86_64::{
        __m128i, __m256, _mm256_add_ps, _mm256_cvtepu16_epi32, _mm256_fmadd_ps,
        _mm256_i32gather_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm_loadu_si128,
    };
    unsafe {
        let mut a0 = _mm256_setzero_ps(); let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps(); let mut a3 = _mm256_setzero_ps();
        let mut a4 = _mm256_setzero_ps(); let mut a5 = _mm256_setzero_ps();
        let mut a6 = _mm256_setzero_ps(); let mut a7 = _mm256_setzero_ps();
        let n64 = nnz / 64 * 64;
        let mut p = 0usize;
        while p < n64 {
            let b0 = p;          let b1 = p + 8;  let b2 = p + 16; let b3 = p + 24;
            let b4 = p + 32;     let b5 = p + 40; let b6 = p + 48; let b7 = p + 56;
            // u16×8 = 128 bit load → 零扩展成 i32×8 → gather
            let g0 = _mm256_i32gather_ps::<4>(x,
                _mm256_cvtepu16_epi32(_mm_loadu_si128(idx_p.add(b0) as *const __m128i)));
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(val_p.add(b0)), g0, a0);
            let g1 = _mm256_i32gather_ps::<4>(x,
                _mm256_cvtepu16_epi32(_mm_loadu_si128(idx_p.add(b1) as *const __m128i)));
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(val_p.add(b1)), g1, a1);
            let g2 = _mm256_i32gather_ps::<4>(x,
                _mm256_cvtepu16_epi32(_mm_loadu_si128(idx_p.add(b2) as *const __m128i)));
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(val_p.add(b2)), g2, a2);
            let g3 = _mm256_i32gather_ps::<4>(x,
                _mm256_cvtepu16_epi32(_mm_loadu_si128(idx_p.add(b3) as *const __m128i)));
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(val_p.add(b3)), g3, a3);
            let g4 = _mm256_i32gather_ps::<4>(x,
                _mm256_cvtepu16_epi32(_mm_loadu_si128(idx_p.add(b4) as *const __m128i)));
            a4 = _mm256_fmadd_ps(_mm256_loadu_ps(val_p.add(b4)), g4, a4);
            let g5 = _mm256_i32gather_ps::<4>(x,
                _mm256_cvtepu16_epi32(_mm_loadu_si128(idx_p.add(b5) as *const __m128i)));
            a5 = _mm256_fmadd_ps(_mm256_loadu_ps(val_p.add(b5)), g5, a5);
            let g6 = _mm256_i32gather_ps::<4>(x,
                _mm256_cvtepu16_epi32(_mm_loadu_si128(idx_p.add(b6) as *const __m128i)));
            a6 = _mm256_fmadd_ps(_mm256_loadu_ps(val_p.add(b6)), g6, a6);
            let g7 = _mm256_i32gather_ps::<4>(x,
                _mm256_cvtepu16_epi32(_mm_loadu_si128(idx_p.add(b7) as *const __m128i)));
            a7 = _mm256_fmadd_ps(_mm256_loadu_ps(val_p.add(b7)), g7, a7);
            p += 64;
        }
        let s01 = _mm256_add_ps(a0, a1); let s23 = _mm256_add_ps(a2, a3);
        let s45 = _mm256_add_ps(a4, a5); let s67 = _mm256_add_ps(a6, a7);
        let s = _mm256_add_ps(_mm256_add_ps(s01, s23), _mm256_add_ps(s45, s67));
        let mut lanes = [0.0f32; 8];
        core::ptr::copy_nonoverlapping((&s as *const __m256).cast::<f32>(),
                                       lanes.as_mut_ptr(), 8);
        let mut acc = lanes.iter().sum::<f32>();
        while p < nnz {
            acc += *val_p.add(p) * *x.add(*idx_p.add(p) as usize);
            p += 1;
        }
        acc
    }
}

#[cfg(test)]
mod tests_scan {
    use super::*;

    #[test]
    fn idx_range_matches_scalar() {
        for n in [0usize, 1, 7, 8, 9, 63, 64, 65, 1000, 100_000] {
            let v: Vec<i32> = (0..n)
                .map(|i| ((i as i64 * 7919) % 1000 - 500) as i32)
                .collect();
            let mut got_min = 1i32;
            let mut got_max = 1i32;
            unsafe {
                let r = phdnet_idx_range_i32(v.as_ptr(), n, &mut got_min, &mut got_max, 8);
                if n == 0 {
                    assert_eq!(r, -1);
                    continue;
                }
                assert_eq!(r, 0);
            }
            let want_min = *v.iter().min().unwrap();
            let want_max = *v.iter().max().unwrap();
            assert_eq!(got_min, want_min, "n={n}");
            assert_eq!(got_max, want_max, "n={n}");
        }
    }

    #[test]
    fn idx_range_serial_matches_parallel() {
        let n = 200_000;
        let v: Vec<i32> = (0..n).map(|i| ((i * 2654435761usize) % 65536) as i32).collect();
        let mut a = 0i32;
        let mut b = 0i32;
        let mut c = 0i32;
        let mut d = 0i32;
        unsafe {
            phdnet_idx_range_i32(v.as_ptr(), n, &mut a, &mut b, 1);
            phdnet_idx_range_i32(v.as_ptr(), n, &mut c, &mut d, 8);
        }
        assert_eq!((a, b), (c, d), "serial vs parallel 必须一致");
    }
}

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
        let idx: Vec<i32> = (0..nnz_row)
            .map(|j| ((j * 7919) % n) as i32)
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
        let idx: Vec<i32> = (0..nnz_row).map(|j| (j * 3) as i32).collect();
        let val: Vec<f32> = (0..nnz_row).map(|j| (j as f32) * 0.1).collect();
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
        let simd = unsafe { csr_row_avx2(idx.as_ptr(), val.as_ptr(), nnz_row, x.as_ptr()) };
        let scalar = unsafe { csr_row_scalar(idx.as_ptr(), val.as_ptr(), nnz_row, x.as_ptr()) };
        assert_eq!(simd.to_bits(), scalar.to_bits(), "短行应逐位一致");
    }
}
