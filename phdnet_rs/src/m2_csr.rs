//! **M2 CSR PC 主干 —— Rust 算子全集**（P175）
//!
//! # 为什么是「全集」而不是只做 SpMV
//!
//! 项目铁律（P170）：换后端必须**对齐完整调用面**，不能只做最热的那个。
//! M2 的 Python 侧有 **6 个 numba 算子** + 2 个融合核：
//!
//! | Python 算子 | 作用 | 本模块 |
//! |---|---|---|
//! | `_csr_matvec` | SpMV（推理） | [`csr_matvec`] |
//! | `_csr_add_outer` | 稀疏外积累加（Hebbian） | [`csr_add_outer`] |
//! | `_csr_oja_up` | 稀疏 Oja（学习） | [`csr_oja_up`] |
//! | `_csr_clip` | 权重裁剪 | [`csr_clip`] |
//! | `_csr_row_norms` | 逐行 L2 范数 | [`csr_row_norms`] |
//! | `_csr_scale_rows` | 逐行缩放到目标范数 | [`csr_scale_rows`] |
//!
//! ⚠ **融合核（`_pc_infer_fused` / `_pc_learn_fused`）刻意不做**——
//!   它们是「多算子串起来」的编排层，**属Python 侧职责**
//!   （见 `phdnet_rs/README.md` 的架构边界：Rust 侧不许有编排）。
//!   Rust 只提供可组合的**原子算子**，融合由 Python 决定。
//!
//! # 数值契约（**逐条对着 Python 实现核**）
//!
//! · **fp32**（P173 把 Python 侧从 fp64 改成了 fp32）。
//! · `matvec` / `row_norms` 的行内累加**按p顺序**（与 Python `for p in ...` 同序）
//!   → 标量路径**可逐位**；SIMD 路径改变求和顺序 → **容差**。
//! · `add_outer` / `oja_up` 是**原地更新**，逐边独立无依赖 → **逐位**。
//! · ⚠ `fastmath=True` 的 Python 版允许重结合；Rust 版**不开** fastmath
//!   → 更严格，差异只来自 SIMD 的求和次序。

use crate::simd::has_avx2;

// ══════════════════════════════════════════════════════════════════════════
//上下文（各算子共用）
// ══════════════════════════════════════════════════════════════════════════

/// CSR + 行/列向量的**只读**视图。**零拷贝**（不持有所有权）。
pub struct CsrView {
    pub indptr: *const i64,
    pub idx: *const i32,
    pub val: *const f32,
    pub n_rows: usize,
}

impl CsrView {
    /// # Safety
    /// `indptr` 须有 `n_rows+1` 个 i64；`idx`/`val` 须有 `indptr[n_rows]` 个元素。
    #[inline]
    pub unsafe fn new(
        indptr: *const i64,
        idx: *const i32,
        val: *const f32,
        n_rows: usize,
    ) -> Self {
        CsrView { indptr, idx, val, n_rows }
    }
}

/// CSR 的**可写**视图（供 `add_outer` / `oja_up` / `scale_rows` 原地更新）。
///
/// # Safety
/// `val` 须**可写**（C 连续 fp32）。
pub struct CsrMut<'a> {
    pub v: CsrView,
    pub val_w: *mut f32,
    /// PhantomData 让编译器知道我们**独占**这个可写指针 → 不当 `Send` 用。
    pub _own: core::marker::PhantomData<&'a mut f32>,
}

// ══════════════════════════════════════════════════════════════════════════
// 1. SpMV：y[i] = Σ val[p]·x[idx[p]]
// ══════════════════════════════════════════════════════════════════════════

/// 标量行内（**按p 顺序累加** → 与 Python 逐位）。
#[inline]
unsafe fn row_dot(idx: *const i32, val: *const f32, lo: usize, hi: usize,
                  x: *const f32) -> f32 {
    unsafe {
        let mut s = 0.0f32;
        for p in lo..hi {
            s += *val.add(p) * *x.add(*idx.add(p) as usize);
        }
        s
    }
}

/// 写 `y`（**已分配好**，长度 ≥ n_rows）。
///
/// # Safety
/// `y` 须C 连续且长度 ≥ `n_rows`；`x` 长度 ≥ 最大 idx +1。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m2_matvec(
    indptr: *const i64,
    idx: *const i32,
    val: *const f32,
    n_rows: usize,
    x: *const f32,
    y: *mut f32,
    n_threads: usize,
) -> i32 {
    let v = unsafe { CsrView::new(indptr, idx, val, n_rows) };
    let ctx = MvCtx { v, x, y };
    let nt = unsafe { crate::parts_for_csr(indptr, n_rows, n_threads) };
    if nt <= 1 {
        unsafe { mv_serial(&ctx as *const MvCtx as *mut u8, 0, 1) };
    } else {
        crate::pool::run(
            mv_serial,
            &ctx as *const MvCtx as *mut u8,
            nt,
            n_threads,
        );
    }
    0
}

/// 标量行内，**4 路累加器**（打断依赖链，但**不用 gather 指令**）。
///
/// P178 诊断：AVX2 `vgatherdps` 在本机**不缩放**（微码共享执行端口），
/// 8 线程只到 1.29×；而标量循环（numba 内层）能到 2.78×。本函数用于
/// 验证「标量 + 短依赖链」能否在多线程下追平 numba。
/// 展开改求和顺序 → **非逐位**（与 gather 核同级，落在 1e-5 容差门内）。
#[inline]
unsafe fn row_dot4(idx_p: *const i32, val_p: *const f32, lo: usize, hi: usize,
                   x: *const f32) -> f32 {
    unsafe {
        let n4 = (hi - lo) / 4 * 4;
        let mut a0 = 0.0f32; let mut a1 = 0.0f32; let mut a2 = 0.0f32; let mut a3 = 0.0f32;
        let mut p = lo;
        while p < lo + n4 {
            a0 += *val_p.add(p) * *x.add(*idx_p.add(p) as usize);
            a1 += *val_p.add(p + 1) * *x.add(*idx_p.add(p + 1) as usize);
            a2 += *val_p.add(p + 2) * *x.add(*idx_p.add(p + 2) as usize);
            a3 += *val_p.add(p + 3) * *x.add(*idx_p.add(p + 3) as usize);
            p += 4;
        }
        let mut acc = (a0 + a1) + (a2 + a3);
        while p < hi { acc += *val_p.add(p) * *x.add(*idx_p.add(p) as usize); p += 1; }
        acc
    }
}

/// SpMV 内层策略（`PHDNET_CSR_KERNEL` 环境变量，A/B 用；**默认 gather**）。
/// · `gather`（默认）= AVX2 `vgatherdps`（单线程最快，多线程不缩放）
/// · `scalar4` = 标量 4 路累加器（可缩放）
/// · `scalar1` = 标量单累加器（**与 numba 内层逐位同构**）
fn csr_kernel_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("PHDNET_CSR_KERNEL") {
            Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
                "scalar4" => 1,
                "scalar1" => 2,
                _ => 0,
            },
            Err(_) => 0,
        }
    })
}

unsafe fn mv_serial(ctx: *mut u8, part: usize, n_parts: usize) {
    let c = unsafe { &*(ctx as *const MvCtx) };
    let chunk = c.v.n_rows.div_ceil(n_parts);
    let lo = part * chunk;
    let hi = ((part + 1) * chunk).min(c.v.n_rows);
    let use_simd = has_avx2();
    let mode = csr_kernel_mode();
    for r in lo..hi {
        let a = unsafe { *c.v.indptr.add(r) } as usize;
        let b = unsafe { *c.v.indptr.add(r + 1) } as usize;
        // ⚠ 行宽≥ 32 且本机有 AVX2 → 走 `csr_dot_avx2`（P178：AVX2 gather 向量化 +
        //   8 向量累加器展开，**非逐位**，fp32 relerr ~1e-6；落在 m2_matvec 的 1e-5
        //   容差门内）。短行（<64）走尾部标量、逐位。
        #[cfg(target_arch = "x86_64")]
        let s = match mode {
            1 => unsafe { row_dot4(c.v.idx, c.v.val, a, b, c.x) },
            2 => unsafe { row_dot(c.v.idx, c.v.val, a, b, c.x) },
            _ if use_simd && b - a >= 32 => unsafe {
                crate::simd::csr_dot_avx2(c.v.idx.add(a), c.v.val.add(a), b - a, c.x)
            },
            _ => unsafe { row_dot(c.v.idx, c.v.val, a, b, c.x) },
        };
        #[cfg(not(target_arch = "x86_64"))]
        let s = unsafe { row_dot(c.v.idx, c.v.val, a, b, c.x) };
        unsafe { *c.y.add(r) = s };
    }
}

struct MvCtx {
    v: CsrView,
    x: *const f32,
    y: *mut f32,
}
// SAFETY: 每个 part 只写 `y[lo..hi)`（连续、互不重叠），其余只读。
unsafe impl Send for MvCtx {}
unsafe impl Sync for MvCtx {}

// ══════════════════════════════════════════════════════════════════════════
// 2. 稀疏外积累加：val[p] += eta·a[i]·b[idx[p]]
// ══════════════════════════════════════════════════════════════════════════

/// **原地**更新 `val`。逐边独立 → **可逐位**。
///
/// # Safety
/// `val` 须可写（C 连续 fp32）；`a`/`b` 只读。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m2_add_outer(
    indptr: *const i64,
    idx: *const i32,
    val: *mut f32,
    n_rows: usize,
    a: *const f32,
    b: *const f32,
    eta: f32,
    n_threads: usize,
) -> i32 {
    let ctx = AoCtx {
        v: unsafe { CsrView::new(indptr, idx, val, n_rows) },
        val_w: val,
        a,
        b,
        eta,
    };
    let nt = unsafe { crate::parts_for_csr(indptr, n_rows, n_threads) };
    if nt <= 1 {
        unsafe { ao_serial(&ctx as *const AoCtx as *mut u8, 0, 1) };
    } else {
        crate::pool::run(
            ao_serial,
            &ctx as *const AoCtx as *mut u8,
            nt,
            n_threads,
        );
    }
    0
}

unsafe fn ao_serial(ctx: *mut u8, part: usize, n_parts: usize) {
    let c = unsafe { &*(ctx as *const AoCtx) };
    let chunk = c.v.n_rows.div_ceil(n_parts);
    let lo = part * chunk;
    let hi = ((part + 1) * chunk).min(c.v.n_rows);
    for r in lo..hi {
        // ⚠ **复刻 Python 的 `if ai == 0.0: continue`** —— 不能省！
        //   省掉会让 ai=0 的行仍写 val（乘0 结果不变，但会**破坏逐位**
        //   在val 含 -0.0 或 NaN 时的行为）。
        let ai = unsafe { *c.a.add(r) };
        if ai == 0.0 {
            continue;
        }
        let lo_p = unsafe { *c.v.indptr.add(r) } as usize;
        let hi_p = unsafe { *c.v.indptr.add(r + 1) } as usize;
        for p in lo_p..hi_p {
            let j = unsafe { *c.v.idx.add(p) } as usize;
            unsafe { *c.val_w.add(p) += c.eta * ai * *c.b.add(j) };
        }
    }
}

struct AoCtx {
    v: CsrView,
    /// 可写别名（`v.val` 是只读的，原地更新必须走这个）。
    val_w: *mut f32,
    a: *const f32,
    b: *const f32,
    eta: f32,
}
unsafe impl Send for AoCtx {}
unsafe impl Sync for AoCtx {}

// ══════════════════════════════════════════════════════════════════════════
// 3. 稀疏 Oja：val[p] += eta·post[r]·(pre[idx[p]] − post[r]·val[p])
// ══════════════════════════════════════════════════════════════════════════

/// **原地**更新 `val`。逐边独立 → **可逐位**。
///
/// # Safety
/// 同 [`phdnet_m2_add_outer`]。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m2_oja_up(
    indptr: *const i64,
    idx: *const i32,
    val: *mut f32,
    n_rows: usize,
    post: *const f32,
    pre: *const f32,
    eta: f32,
    n_threads: usize,
) -> i32 {
    let ctx = OjaCtx {
        v: unsafe { CsrView::new(indptr, idx, val, n_rows) },
        val_w: val,
        post,
        pre,
        eta,
    };
    let nt = unsafe { crate::parts_for_csr(indptr, n_rows, n_threads) };
    if nt <= 1 {
        unsafe { oja_serial(&ctx as *const OjaCtx as *mut u8, 0, 1) };
    } else {
        crate::pool::run(
            oja_serial,
            &ctx as *const OjaCtx as *mut u8,
            nt,
            n_threads,
        );
    }
    0
}

unsafe fn oja_serial(ctx: *mut u8, part: usize, n_parts: usize) {
    let c = unsafe { &*(ctx as *const OjaCtx) };
    let chunk = c.v.n_rows.div_ceil(n_parts);
    let lo = part * chunk;
    let hi = ((part + 1) * chunk).min(c.v.n_rows);
    for r in lo..hi {
        let pi = unsafe { *c.post.add(r) };
        if pi == 0.0 {
            continue;                    // 同上：复刻 Python 的 continue
        }
        let lo_p = unsafe { *c.v.indptr.add(r) } as usize;
        let hi_p = unsafe { *c.v.indptr.add(r + 1) } as usize;
        for p in lo_p..hi_p {
            let j = unsafe { *c.v.idx.add(p) } as usize;
            let vp = unsafe { *c.val_w.add(p) };
            unsafe { *c.val_w.add(p) += c.eta * pi * (*c.pre.add(j) - pi * vp) };
        }
    }
}

struct OjaCtx {
    v: CsrView,
    val_w: *mut f32,
    post: *const f32,
    pre: *const f32,
    eta: f32,
}
unsafe impl Send for OjaCtx {}
unsafe impl Sync for OjaCtx {}

// ══════════════════════════════════════════════════════════════════════════
// 4. 权重裁剪：val[p] = clip(val[p], ±w_max)
// ══════════════════════════════════════════════════════════════════════════

/// **原地**裁剪。逐元素独立 → **可逐位**。
///
/// ⚠ 复刻 Python 的 `if v > w_max ... elif v < -w_max`（**不是** `f32::clamp`——
///   `clamp` 对 NaN 的行为与 Python 版不同）。
///
/// # Safety
/// `val` 须可写；`nnz` = 元素数。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m2_clip(
    val: *mut f32,
    nnz: usize,
    w_max: f32,
    n_threads: usize,
) -> i32 {
    // ⚠ `clip` **不用行分块**（没有 indptr）→ 用**连续区间分块**。
    let ctx = ClipCtx { val, w_max, val_len: nnz };
    let nt = crate::parts_for_len(nnz, n_threads);
    if nt <= 1 {
        unsafe { clip_serial(&ctx as *const ClipCtx as *mut u8, 0, 1) };
    } else {
        crate::pool::run(
            clip_serial,
            &ctx as *const ClipCtx as *mut u8,
            nt,
            n_threads,
        );
    }
    0
}

unsafe fn clip_serial(ctx: *mut u8, part: usize, n_parts: usize) {
    let c = unsafe { &*(ctx as *const ClipCtx) };
    let chunk = c.val_len.div_ceil(n_parts);
    let lo = part * chunk;
    let hi = ((part + 1) * chunk).min(c.val_len);
    for p in lo..hi {
        let v = unsafe { *c.val.add(p) };
        if v > c.w_max {
            unsafe { *c.val.add(p) = c.w_max };
        } else if v < -c.w_max {
            unsafe { *c.val.add(p) = -c.w_max };
        }
    }
}

struct ClipCtx {
    val: *mut f32,
    w_max: f32,
    /// 元素总数（`clip` 没有 indptr，只能按连续区间分块）。
    val_len: usize,
}
unsafe impl Send for ClipCtx {}
unsafe impl Sync for ClipCtx {}

// ══════════════════════════════════════════════════════════════════════════
// 5. 逐行 L2 范数：out[r] = ‖val[indptr[r]:indptr[r+1]]‖₂
// ══════════════════════════════════════════════════════════════════════════

/// 写 `out`（长度 ≥ n_rows）。
///
/// ⚠ `s ** 0.5`（Python）与 `f32::sqrt()`（Rust）**应逐位相同**
///   （都是 IEEE 精确 sqrt），故这条**可逐位**。
///
/// # Safety
/// `out` 须 C 连续且长度 ≥ `n_rows`。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m2_row_norms(
    indptr: *const i64,
    idx: *const i32,
    val: *const f32,
    n_rows: usize,
    out: *mut f32,
    n_threads: usize,
) -> i32 {
    let ctx = RnCtx {
        v: unsafe { CsrView::new(indptr, idx, val, n_rows) },
        out,
    };
    let nt = unsafe { crate::parts_for_csr(indptr, n_rows, n_threads) };
    if nt <= 1 {
        unsafe { rn_serial(&ctx as *const RnCtx as *mut u8, 0, 1) };
    } else {
        crate::pool::run(
            rn_serial,
            &ctx as *const RnCtx as *mut u8,
            nt,
            n_threads,
        );
    }
    0
}

unsafe fn rn_serial(ctx: *mut u8, part: usize, n_parts: usize) {
    let c = unsafe { &*(ctx as *const RnCtx) };
    let chunk = c.v.n_rows.div_ceil(n_parts);
    let lo = part * chunk;
    let hi = ((part + 1) * chunk).min(c.v.n_rows);
    for r in lo..hi {
        let a = unsafe { *c.v.indptr.add(r) } as usize;
        let b = unsafe { *c.v.indptr.add(r + 1) } as usize;
        let mut s = 0.0f32;
        for p in a..b {
            let vp = unsafe { *c.v.val.add(p) };
            s += vp * vp;
        }
        unsafe { *c.out.add(r) = s.sqrt() };
    }
}

struct RnCtx {
    v: CsrView,
    out: *mut f32,
}
unsafe impl Send for RnCtx {}
unsafe impl Sync for RnCtx {}

// ══════════════════════════════════════════════════════════════════════════
// 6. 逐行缩放：val[p] *= target[r] / max(‖row‖₂, 1e-12)
// ══════════════════════════════════════════════════════════════════════════

/// **原地**缩放。**原地更新 → 逐位**（同 Python 的两遍：先求 norm 再缩放）。
///
/// ⚠ `nrm < 1e-12` 的保护必须复刻 —— 少了会**除零**。
///
/// # Safety
/// `val` 须可写；`target` 长度 ≥ n_rows。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m2_scale_rows(
    indptr: *const i64,
    idx: *const i32,
    val: *mut f32,
    n_rows: usize,
    target: *const f32,
    n_threads: usize,
) -> i32 {
    let ctx = SrCtx {
        v: unsafe { CsrView::new(indptr, idx, val, n_rows) },
        val_w: val,
        target,
    };
    let nt = unsafe { crate::parts_for_csr(indptr, n_rows, n_threads) };
    if nt <= 1 {
        unsafe { sr_serial(&ctx as *const SrCtx as *mut u8, 0, 1) };
    } else {
        crate::pool::run(
            sr_serial,
            &ctx as *const SrCtx as *mut u8,
            nt,
            n_threads,
        );
    }
    0
}

unsafe fn sr_serial(ctx: *mut u8, part: usize, n_parts: usize) {
    let c = unsafe { &*(ctx as *const SrCtx) };
    let chunk = c.v.n_rows.div_ceil(n_parts);
    let lo = part * chunk;
    let hi = ((part + 1) * chunk).min(c.v.n_rows);
    for r in lo..hi {
        let a = unsafe { *c.v.indptr.add(r) } as usize;
        let b = unsafe { *c.v.indptr.add(r + 1) } as usize;
        let mut s = 0.0f32;
        for p in a..b {
            let vp = unsafe { *c.val_w.add(p) };
            s += vp * vp;
        }
        let mut nrm = s.sqrt();
        if nrm < 1e-12 {
            nrm = 1e-12;                   // ⚠ 必须复刻（否则除零）
        }
        let f = unsafe { *c.target.add(r) } / nrm;
        for p in a..b {
            unsafe { *c.val_w.add(p) *= f };
        }
    }
}

struct SrCtx {
    v: CsrView,
    val_w: *mut f32,
    target: *const f32,
}
unsafe impl Send for SrCtx {}
unsafe impl Sync for SrCtx {}

#[cfg(test)]
mod tests_m2 {
    use super::*;

    /// 造一个小的随机 CSR（确定性种子）。
    fn make(n_rows: usize, k: usize, seed: u64) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as usize) % n_rows
        };
        let indptr: Vec<i64> = (0..=n_rows)
            .map(|i| (i * k) as i64)
            .collect();
        let mut idx = Vec::with_capacity(n_rows * k);
        let mut val = Vec::with_capacity(n_rows * k);
        for _ in 0..n_rows * k {
            idx.push(next() as i32);
            let t = next() as f32 / (n_rows as f32);
            val.push(t - 0.5);
        }
        (indptr, idx, val)
    }

    /// Python 参考（fp32，按 p 顺序累加）。
    fn ref_matvec(
        indptr: &[i64], idx: &[i32], val: &[f32], x: &[f32], n: usize,
    ) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mut s = 0.0f32;
                for p in indptr[i] as usize..indptr[i + 1] as usize {
                    s += val[p] * x[idx[p] as usize];
                }
                s
            })
            .collect()
    }

    #[test]
    fn matvec_matches_python_bitwise() {
        let (n, k) = (64usize, 16usize);
        let (ip, idx, val) = make(n, k, 7);
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let mut y = vec![0.0f32; n];
        unsafe {
            phdnet_m2_matvec(
                ip.as_ptr(), idx.as_ptr(), val.as_ptr(), n,
                x.as_ptr(), y.as_mut_ptr(), 1,
            );
        }
        let r = ref_matvec(&ip, &idx, &val, &x, n);
        for i in 0..n {
            assert_eq!(
                y[i].to_bits(), r[i].to_bits(),
                "row {i}: got {} want {}",
                y[i], r[i]
            );
        }
    }

    #[test]
    fn add_outer_matches_python_bitwise() {
        let (n, k) = (64usize, 16usize);
        let (ip, idx, val0) = make(n, k, 11);
        let a: Vec<f32> = (0..n).map(|i| if i % 5 == 0 { 0.0 } else { i as f32 * 0.1 }).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.02).collect();
        let eta = 0.05f32;
        // Python 参考
        let mut want = val0.clone();
        for i in 0..n {
            let ai = a[i];
            if ai == 0.0 {
                continue;
            }
            for p in ip[i] as usize..ip[i + 1] as usize {
                want[p] += eta * ai * b[idx[p] as usize];
            }
        }
        let mut got = val0.clone();
        unsafe {
            phdnet_m2_add_outer(
                ip.as_ptr(), idx.as_ptr(), got.as_mut_ptr(), n,
                a.as_ptr(), b.as_ptr(), eta, 1,
            );
        }
        for p in 0..want.len() {
            assert_eq!(got[p].to_bits(), want[p].to_bits(), "edge {p}");
        }
    }

    #[test]
    fn oja_up_matches_python_bitwise() {
        let (n, k) = (64usize, 16usize);
        let (ip, idx, val0) = make(n, k, 13);
        let post: Vec<f32> = (0..n).map(|i| if i % 7 == 0 { 0.0 } else { i as f32 * 0.01 }).collect();
        let pre: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03).collect();
        let eta = 0.02f32;
        let mut want = val0.clone();
        for i in 0..n {
            let pi = post[i];
            if pi == 0.0 {
                continue;
            }
            for p in ip[i] as usize..ip[i + 1] as usize {
                want[p] += eta * pi * (pre[idx[p] as usize] - pi * want[p]);
            }
        }
        let mut got = val0.clone();
        unsafe {
            phdnet_m2_oja_up(
                ip.as_ptr(), idx.as_ptr(), got.as_mut_ptr(), n,
                post.as_ptr(), pre.as_ptr(), eta, 1,
            );
        }
        for p in 0..want.len() {
            assert_eq!(got[p].to_bits(), want[p].to_bits(), "edge {p}");
        }
    }

    #[test]
    fn clip_matches_python_bitwise() {
        let mut v: Vec<f32> = vec![-2.0, -0.5, 0.0, 0.5, 2.0, 1.5, -1.5];
        let original = v.clone();
        let w = 1.0f32;
        let mut want = original.clone();
        for x in want.iter_mut() {
            let val = *x;
            if val > w {
                *x = w;
            } else if val < -w {
                *x = -w;
            }
        }
        unsafe { phdnet_m2_clip(v.as_mut_ptr(), v.len(), w, 1) };
        for i in 0..original.len() {
            assert_eq!(v[i].to_bits(), want[i].to_bits(), "idx {i}");
        }
    }

    #[test]
    fn row_norms_matches_python_bitwise() {
        let (n, k) = (64usize, 16usize);
        let (ip, idx, val) = make(n, k, 17);
        let mut want = vec![0.0f32; n];
        for i in 0..n {
            let mut s = 0.0f32;
            for p in ip[i] as usize..ip[i + 1] as usize {
                s += val[p] * val[p];
            }
            want[i] = s.sqrt();
        }
        let mut got = vec![0.0f32; n];
        unsafe {
            phdnet_m2_row_norms(
                ip.as_ptr(), idx.as_ptr(), val.as_ptr(), n, got.as_mut_ptr(), 1,
            );
        }
        for i in 0..n {
            assert_eq!(got[i].to_bits(), want[i].to_bits(), "row {i}");
        }
    }

    #[test]
    fn scale_rows_matches_python_bitwise() {
        let (n, k) = (64usize, 16usize);
        let (ip, idx, val0) = make(n, k, 19);
        let target: Vec<f32> = (0..n).map(|i| 0.01 * (i % 13) as f32).collect();
        let mut want = val0.clone();
        for i in 0..n {
            let mut s = 0.0f32;
            for p in ip[i] as usize..ip[i + 1] as usize {
                s += want[p] * want[p];
            }
            let mut nrm = s.sqrt();
            if nrm < 1e-12 {
                nrm = 1e-12;
            }
            let f = target[i] / nrm;
            for p in ip[i] as usize..ip[i + 1] as usize {
                want[p] *= f;
            }
        }
        let mut got = val0.clone();
        unsafe {
            phdnet_m2_scale_rows(
                ip.as_ptr(), idx.as_ptr(), got.as_mut_ptr(), n,
                target.as_ptr(), 1,
            );
        }
        for p in 0..want.len() {
            assert_eq!(got[p].to_bits(), want[p].to_bits(), "edge {p}");
        }
    }

    #[test]
    fn scale_rows_handles_zero_norm() {
        // 全零行 → nrm 被夹到 1e-12，**不得除零/产生 NaN**
        let n = 4usize;
        let ip: Vec<i64> = vec![0, 2, 4, 6, 8];
        let idx: Vec<i32> = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let mut val = vec![0.0f32; 8];
        let target = vec![1.0f32, 1.0, 1.0, 1.0];
        unsafe {
            phdnet_m2_scale_rows(
                ip.as_ptr(), idx.as_ptr(), val.as_mut_ptr(), n,
                target.as_ptr(), 1,
            );
        }
        for v in val.iter() {
            assert!(v.is_finite(), "zero-norm row must not produce NaN/Inf: {v}");
        }
    }
}