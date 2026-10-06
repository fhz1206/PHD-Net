//! **M2 融合核的 Rust 实现**（P176）
//!
//! # ⚠ 我在 P175 的判断是错的，这里更正
//!
//! P175 我写过「融合核属编排，Rust 不做」。**读代码后发现不对**：
//! `_pc_infer_fused` / `_pc_learn_fused` 是**纯数值核**——
//! 8 次 SpMV + `tanh` / `clip` / 原地更新，**没有任何「下一步做什么」的决策**。
//! 真正的编排（`model.step` 的顺序、`--nll-sync-every` 调度、门禁判据）
//! 仍然留在 Python（见 `phdnet_rs/README.md` 的架构边界）。
//!
//! # 为什么融合核值得进Rust
//!
//! 融合核的收益来自「**中间数组只分配一次、循环内复用**」。
//! Rust 版把这个收益做得更彻底：**线程内**分配一次缓冲区，跨 8 个子核复用，
//! 且每个子核的行级循环走 **AVX2**。
//!
//! # 精度（P176，fhz 拍板「统一改 fp32」）
//!
//! ⚠ **改动前**融合核是「**fp32 存储 + fp64 累加 + fp64 中间数组**」的混合精度
//!   （`s = 0.0` 让 numba 提升 → 输出全 fp64），而 M2 其他算子已是 fp32
//!   → **同一份数据两种精度**。
//! ⚠ **改动后**两侧统一为**全 fp32**。代价：累加精度降低 → **需重新 rebaseline**。
//!   收益：① 能用 fp32 SIMD（fp64 无 FMA）；② 与 `m2_matvec` **逐位一致**
//!        → 融合路径与非融合路径**数值等价**（P175 时做不到这一点）。
//!
//! # 数值契约
//!
//! · 行内累加按 `p` 顺序（同 Python `for p in ...`）
//!   → 标量路径**可逐位**；SIMD 路径（行宽 ≥ 32）改求和顺序 → **容差**。
//! · `tanh` / `clip` 必须与 Python 的 `np.tanh` / `min-max` **同语义**。

use crate::m2_csr::CsrView;
use crate::simd::has_avx2;

// ══════════════════════════════════════════════════════════════════════════
// 行内点积（与 m2_csr 相同的策略）
// ══════════════════════════════════════════════════════════════════════════

/// 一行的 `Σ val[p]·x[idx[p]]`（**按 p 顺序**，标量）。
#[inline]
unsafe fn dot_seq(v: &CsrView, r: usize, x: *const f32) -> f32 {
    unsafe {
        let a = *v.indptr.add(r) as usize;
        let b = *v.indptr.add(r + 1) as usize;
        let mut s = 0.0f32;
        for p in a..b {
            s += *v.val.add(p) * *x.add(*v.idx.add(p) as usize);
        }
        s
    }
}

/// 一行的点积，**长行走 AVX2**（改求和顺序）。
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn dot_auto(v: &CsrView, r: usize, x: *const f32) -> f32 {
    unsafe {
        let a = *v.indptr.add(r) as usize;
        let b = *v.indptr.add(r + 1) as usize;
        if has_avx2() && b - a >= 32 {
            crate::simd::csr_dot_avx2(v.idx.add(a), v.val.add(a), b - a, x)
        } else {
            dot_seq(v, r, x)
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
unsafe fn dot_auto(v: &CsrView, r: usize, x: *const f32) -> f32 {
    unsafe { dot_seq(v, r, x) }
}

/// `min(hi, max(lo, s))` —— 复刻 Python 的 `min(0.5, max(-0.5, s))`。
#[inline]
fn clip05(s: f32) -> f32 {
    // ⚠ **不用 `f32::clamp`** —— 它对 NaN 的行为与 Python 的
    //   `min(hi, max(lo, x))` 不同（clamp 遇 NaN 返回 NaN，
    //   Python 的 `max(-0.5, NaN)` 返回 NaN 但 `min(0.5, NaN)` 也返回 NaN，
    //   实际一致；但 `f32::clamp` 在 lo>hi 时行为未定义）。
    //   这里显式写，**不依赖库的 clamp 语义**。
    if s < -0.5 {
        -0.5
    } else if s > 0.5 {
        0.5
    } else {
        s
    }
}

// 并行基础设施
// ══════════════════════════════════════════════════════════════════════════
//
// ⚠⚠ **为什么不用闭包 + `thread::scope`**（第一次尝试编译失败的原因）：
//   闭包捕获 `CsrView`（含 `*const i64` / `*const f32`）→ 裸指针**不是 `Sync`**
//   → 编译器直接拒绝（6 处 E0277）。`pool.rs` 用 `usize` 间接绕过，
//   这里沿用**同一手法**：`#[repr(C)] struct` + `unsafe impl Send/Sync`，
//   并把指针字段存成 `usize`（整数是 `Send`）。
//
// ⚠ **语义安全性**：每个 part 只写自己的**连续行块**，其余只读 → 无竞态。
//   这是我们敢这么用的**唯一理由**，已写在 SAFETY 注释里。

/// `Send` 包装：**只把 ctx 指针跨线程传**，不复制内容。
///
/// ⚠ 安全性靠**调用方的行块划分**保证（每个 part 只写自己的连续行块）。
///   裸指针不是 `Send`，但我们**只传指针本身**、数据在共享只读/分块写 —— 这是
///   与 `pool.rs` 同一手法（那里传 `usize`，这里传一个 `#[repr(transparent)]` 包装）。
#[repr(transparent)]
struct CtxPtr<T>(*const T);
// ⚠ **手写 `Copy`，不用 `#[derive]`** —— `derive(Clone, Copy)` 会加 `T: Copy`
//   约束，而 `T` 是含裸指针的 ctx（**不是** `Copy`）→ derive 生成的 impl
//   不适用，编译器仍报 `use of moved value: p`（实测踩过）。
impl<T> Clone for CtxPtr<T> {
    #[inline]
    fn clone(&self) -> Self {
        CtxPtr(self.0)
    }
}
impl<T> Copy for CtxPtr<T> {}
// SAFETY: 见上——数据只读，或各 part 写互不重叠的连续行块。
unsafe impl<T> Send for CtxPtr<T> {}
unsafe impl<T> Sync for CtxPtr<T> {}

/// 并行上下文（指针存 `usize` 以获得 `Send`）。
#[repr(C)]
struct InferCtxC {
    // 4 个 CSR：每 CSR 三个指针 + 行数
    up0_ip: usize, up0_ix: usize, up0_v: usize, up0_n: usize,
    up1_ip: usize, up1_ix: usize, up1_v: usize, up1_n: usize,
    dn0_ip: usize, dn0_ix: usize, dn0_v: usize, dn0_n: usize,
    dn1_ip: usize, dn1_ix: usize, dn1_v: usize, dn1_n: usize,
    s0: usize,
    n1: usize, n2: usize, n_steps: usize,
    /// `e0` 的长度 = **n0**（`dn0` 的行数 / `up0` 的列空间）。
    /// ⚠ P176：Python 侧原分配为 `n1` → `d1 = clip(up0 @ e0)` 越界读
    ///   （实测 n0=256/n1=192 时越界 64 个元素）。已按 fhz 拍板修正为 `n0`。
    n0: usize,
    r1: usize, r2: usize, e0: usize, e1: usize,
    /// 线程内暂存（**只分配一次**，8 阶段 x n_steps 轮全部复用）。
    /// ⚠ `te0` 长 **n0**（= `e0` 长度），但 `td1` 长 **n1**（= `up0` 行数）。
    te1: usize, td2: usize, te0: usize, td1: usize,
    // ── P177：常驻池派发所需 ────────────────────────────────────
    /// 当前 stage 的**函数指针位模式**。
    /// ⚠ 为什么存 ctx 而不用 `static`：池的 `WorkFn` 是**裸 fn 指针**
    ///   （不收闭包）。若用全局 `static` 存 stage → **共享写 + 竞态**。
    ///   存ctx → 每个 worker 读**自己的** ctx（只读），**无共享写**。
    stage_fn: usize,
    /// 本次要遍历的行数（worker 按它分块）。
    n_rows_hint: usize,
}
// SAFETY: 每个 part 只写自己的行块（见 `infer_stage_*` 的 SAFETY 注释）。
unsafe impl Send for InferCtxC {}
unsafe impl Sync for InferCtxC {}

impl InferCtxC {
    #[inline]
    unsafe fn up0(&self) -> CsrView {
        CsrView::new(self.up0_ip as *const i64, self.up0_ix as *const i64,
                     self.up0_v as *const f32, self.up0_n)
    }
    #[inline]
    unsafe fn up1(&self) -> CsrView {
        CsrView::new(self.up1_ip as *const i64, self.up1_ix as *const i64,
                     self.up1_v as *const f32, self.up1_n)
    }
    #[inline]
    unsafe fn dn0(&self) -> CsrView {
        CsrView::new(self.dn0_ip as *const i64, self.dn0_ix as *const i64,
                     self.dn0_v as *const f32, self.dn0_n)
    }
    #[inline]
    unsafe fn dn1(&self) -> CsrView {
        CsrView::new(self.dn1_ip as *const i64, self.dn1_ix as *const i64,
                     self.dn1_v as *const f32, self.dn1_n)
    }
}

/// 行块范围 `[lo, hi)`。
#[inline]
fn row_range(n: usize, part: usize, n_parts: usize) -> (usize, usize) {
    let chunk = n.div_ceil(n_parts);
    let lo = part * chunk;
    let hi = ((part + 1) * chunk).min(n);
    (lo, hi)
}

// ── infer 的 8 个子核（每个是一个「阶段」的可并行单元）────────────

unsafe fn stage_r1_from_s0(c: *const InferCtxC, lo: usize, hi: usize) {
    unsafe {
        let c = &*c;
        let up0 = c.up0();
        let s0 = c.s0 as *const f32;
        for i in lo..hi {
            *((c.r1) as *mut f32).add(i) = dot_auto(&up0, i, s0).tanh();
        }
    }
}

unsafe fn stage_r2_from_r1(c: *const InferCtxC, lo: usize, hi: usize) {
    unsafe {
        let c = &*c;
        let up1 = c.up1();
        let r1 = c.r1 as *const f32;
        for i in lo..hi {
            *((c.r2) as *mut f32).add(i) = dot_auto(&up1, i, r1).tanh();
        }
    }
}

/// `e1 = r1 - dn1@r2`（写入 ctx 外的暂存数组 `te1`）。
unsafe fn stage_e1(c: *const InferCtxC, lo: usize, hi: usize) {
    unsafe {
        let c = &*c;
        let dn1 = c.dn1();
        let r1 = c.r1 as *const f32;
        let r2 = c.r2 as *const f32;
        for i in lo..hi {
            *((c.te1) as *mut f32).add(i) = *r1.add(i) - dot_auto(&dn1, i, r2);
        }
    }
}

/// `d2 = clip05(up1@e1)`
unsafe fn stage_d2(c: *const InferCtxC, lo: usize, hi: usize) {
    unsafe {
        let c = &*c;
        let up1 = c.up1();
        let e1 = c.te1 as *const f32;
        for i in lo..hi {
            *((c.td2) as *mut f32).add(i) = clip05(dot_auto(&up1, i, e1));
        }
    }
}

/// `r2 = tanh(r2 + 0.15*d2)`（**原地**）
unsafe fn stage_r2_update(c: *const InferCtxC, lo: usize, hi: usize) {
    unsafe {
        let c = &*c;
        let r2 = c.r2 as *mut f32;
        let d2 = c.td2 as *const f32;
        for i in lo..hi {
            *r2.add(i) = (*r2.add(i) + 0.15f32 * *d2.add(i)).tanh();
        }
    }
}

/// `e0 = s0 - dn0@r1`
unsafe fn stage_e0(c: *const InferCtxC, lo: usize, hi: usize) {
    unsafe {
        let c = &*c;
        let dn0 = c.dn0();
        let r1 = c.r1 as *const f32;
        let s0 = c.s0 as *const f32;
        for i in lo..hi {
            *((c.te0) as *mut f32).add(i) = *s0.add(i) - dot_auto(&dn0, i, r1);
        }
    }
}

/// `d1 = clip05(up0@e0)`
unsafe fn stage_d1(c: *const InferCtxC, lo: usize, hi: usize) {
    unsafe {
        let c = &*c;
        let up0 = c.up0();
        let e0 = c.te0 as *const f32;
        for i in lo..hi {
            *((c.td1) as *mut f32).add(i) = clip05(dot_auto(&up0, i, e0));
        }
    }
}

/// `r1 = tanh(r1 + 0.15*d1)`（**原地**）
unsafe fn stage_r1_update(c: *const InferCtxC, lo: usize, hi: usize) {
    unsafe {
        let c = &*c;
        let r1 = c.r1 as *mut f32;
        let d1 = c.td1 as *const f32;
        for i in lo..hi {
            *r1.add(i) = (*r1.add(i) + 0.15f32 * *d1.add(i)).tanh();
        }
    }
}

/// 末尾误差 `e0 = s0 - dn0@r1`（写到输出缓冲）
unsafe fn stage_out_e0(c: *const InferCtxC, lo: usize, hi: usize) {
    unsafe {
        let c = &*c;
        let dn0 = c.dn0();
        let r1 = c.r1 as *const f32;
        let s0 = c.s0 as *const f32;
        for i in lo..hi {
            *((c.e0) as *mut f32).add(i) = *s0.add(i) - dot_auto(&dn0, i, r1);
        }
    }
}

/// 末尾误差 `e1 = r1 - dn1@r2`（写到输出缓冲）
unsafe fn stage_out_e1(c: *const InferCtxC, lo: usize, hi: usize) {
    unsafe {
        let c = &*c;
        let dn1 = c.dn1();
        let r1 = c.r1 as *const f32;
        let r2 = c.r2 as *const f32;
        for i in lo..hi {
            *((c.e1) as *mut f32).add(i) = *r1.add(i) - dot_auto(&dn1, i, r2);
        }
    }
}

/// P177：常驻池桥接（infer）。`pool::run` 把 `(ctx, part, n_parts)` 交给本函数，
/// 本函数从 ctx 读出 `stage_fn`（已存好的 stage 函数指针位模式）与 `n_rows_hint`，
/// 分块后调用真正的 stage 函数。
///
/// SAFETY：ctx 由 `run_stage` 在派发前写入 `stage_fn`/`n_rows_hint`，且
/// `pool::run` 同步返回（所有 worker 完成后才返回）→ 无并发写、无竞态。
unsafe fn infer_bridge(ctx: *mut u8, part: usize, n_parts: usize) {
    unsafe {
        let c = ctx as *const InferCtxC;
        let fp = (*c).stage_fn as *const ();
        if fp.is_null() {
            return;
        }
        let f: unsafe fn(*const InferCtxC, usize, usize) = core::mem::transmute(fp);
        let (lo, hi) = row_range((*c).n_rows_hint, part, n_parts);
        if lo >= hi {
            return;
        }
        f(c, lo, hi);
    }
}

/// P177：常驻池桥接（learn）。结构与 [`infer_bridge`] 完全相同，仅 ctx 类型不同。
unsafe fn learn_bridge(ctx: *mut u8, part: usize, n_parts: usize) {
    unsafe {
        let c = ctx as *const LearnCtxC;
        let fp = (*c).stage_fn as *const ();
        if fp.is_null() {
            return;
        }
        let f: unsafe fn(*const LearnCtxC, usize, usize) = core::mem::transmute(fp);
        let (lo, hi) = row_range((*c).n_rows_hint, part, n_parts);
        if lo >= hi {
            return;
        }
        f(c, lo, hi);
    }
}

/// 并行跑一个阶段。
///
/// ⚠ **P177：已改用常驻线程池**（`crate::pool::run`）—— 不再每步建/销线程。
///   P171 实测「每步新建线程」让 Rust 慢 numba 1.8×；融合核每 step 调本函数
///   **2 + 6·n_steps + 2** 次（`n_steps=1` 时 10 次/step，每次 8 线程 →
///   **80 次线程创建/step**），常驻池把这部分降为微秒级派发。
///   ⚠ **门限**：派发固定成本实测 41.5 µs（P174）→ 行数太小时仍串行。
#[inline]
unsafe fn run_stage(
    ctx: CtxPtr<InferCtxC>,
    n: usize,
    n_threads: usize,
    f: unsafe fn(*const InferCtxC, usize, usize),
) {
    if n == 0 {
        return;
    }
    // ⚠ 派发成本门限：nnz 小于阈值就串行（P174 实测平衡点 256K nnz）。
    let nnz_est = n * 128;              // 典型 k=128
    let nt = if nnz_est < 256 * 1024 {
        1
    } else {
        n_threads.clamp(1, 8)
    };
    if nt <= 1 {
        unsafe { f(ctx.0, 0, n) };
        return;
    }
    let nt = nt.min(n);
    // ⚠ **P177：改用常驻线程池**（`crate::pool::run`）—— 不再每步建/销线程
    //   （P171 实测这部分开销让 Rust 慢 numba 1.8×）。
    //   池的 `WorkFn` 是 `unsafe fn(*mut u8, usize, usize)`，不收闭包 →
    //   把当前 stage 的函数指针存进 ctx（`stage_fn`），并把行数存 `n_rows_hint`，
    //   由 `infer_bridge` 读出来分块派发。
    unsafe {
        let cm = ctx.0 as *mut InferCtxC;
        (*cm).stage_fn = f as *const () as usize;
        (*cm).n_rows_hint = n;
        crate::pool::run(infer_bridge, cm as *mut u8, nt, n_threads);
    }
}

/// **融合推理**（对应 `_pc_infer_fused`，P176 全 fp32）。
///
/// # Safety
/// 所有指针须有效且**长度匹配**；`r1`/`e0`/`e1` 长度 ≥ `n1`，`r2` 长度 ≥ `n2`；
/// 全部 **fp32 且 C 连续**（Python 侧 `_check_csr_fused` 已校验）。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m2_infer_fused(
    up0_indptr: *const i64, up0_idx: *const i64, up0_val: *const f32, n1: usize,
    up1_indptr: *const i64, up1_idx: *const i64, up1_val: *const f32, n2: usize,
    // `dn0` 的行数 = `e0` 的长度（P176 修正：原为 n1 → 越界）
    n0: usize,
    dn0_indptr: *const i64, dn0_idx: *const i64, dn0_val: *const f32,
    dn1_indptr: *const i64, dn1_idx: *const i64, dn1_val: *const f32,
    s0: *const f32,
    r1: *mut f32, r2: *mut f32, e0: *mut f32, e1: *mut f32,
    n_steps: usize,
    n_threads: usize,
) -> i32 {
    // ⚠ **中间数组只分配一次**，8 个阶段 ×n_steps 轮全部复用
    //（这正是融合核的核心收益，也是它值得进 Rust 的原因）。
    let mut te1 = vec![0.0f32; n1];
    let mut td2 = vec![0.0f32; n2];
    // ⚠ P176：`e0`/`d1` 长 **n0**（不是 n1）—— 见 InferCtxC::n0 的说明
    let mut te0 = vec![0.0f32; n0];
    let mut td1 = vec![0.0f32; n0];
    let ctx = InferCtxC {
        up0_ip: up0_indptr as usize, up0_ix: up0_idx as usize,
        up0_v: up0_val as usize, up0_n: n1,
        up1_ip: up1_indptr as usize, up1_ix: up1_idx as usize,
        up1_v: up1_val as usize, up1_n: n2,
        dn0_ip: dn0_indptr as usize, dn0_ix: dn0_idx as usize,
        dn0_v: dn0_val as usize, dn0_n: n0,   // ⚠ P176：dn0 有 n0 行（原误写 n1）
        dn1_ip: dn1_indptr as usize, dn1_ix: dn1_idx as usize,
        dn1_v: dn1_val as usize, dn1_n: n1,   // ⚠ P176：dn1 有 n1 行（原误写 n2）
        s0: s0 as usize,
        n1, n2, n_steps, n0,
        r1: r1 as usize, r2: r2 as usize,
        e0: e0 as usize, e1: e1 as usize,
        te1: te1.as_mut_ptr() as usize,
        td2: td2.as_mut_ptr() as usize,
        te0: te0.as_mut_ptr() as usize,
        td1: td1.as_mut_ptr() as usize,
        stage_fn: 0, n_rows_hint: 0,   // P177：run_stage 派发前覆写
    };
    let p = CtxPtr(&ctx as *const InferCtxC);
    unsafe {
        run_stage(p, n1, n_threads, stage_r1_from_s0);
        run_stage(p, n2, n_threads, stage_r2_from_r1);
        for _ in 0..n_steps {
            run_stage(p, n1, n_threads, stage_e1);
            run_stage(p, n2, n_threads, stage_d2);
            run_stage(p, n2, n_threads, stage_r2_update);
            run_stage(p, n0, n_threads, stage_e0);   // ⚠ P176：n1 -> n0
            run_stage(p, n1, n_threads, stage_d1);   // ⚠ P176：遍历 **up0 的 n1 行**
            //   （不是 n0 —— n0 是 e0 的长度，up0 只有 n1 行）
            run_stage(p, n1, n_threads, stage_r1_update);
        }
        run_stage(p, n0, n_threads, stage_out_e0);  // ⚠ P176：n1 -> n0
        run_stage(p, n1, n_threads, stage_out_e1);
    }
    0
}

// ══════════════════════════════════════════════════════════════════════════
// learn 融合核
// ══════════════════════════════════════════════════════════════════════════

/// learn 的上下文（4 个**可写** CSR + 只读向量）。
#[repr(C)]
struct LearnCtxC {
    dn0_ip: usize, dn0_ix: usize, dn0_v: usize, dn0_n: usize,
    dn1_ip: usize, dn1_ix: usize, dn1_v: usize, dn1_n: usize,
    up0_ip: usize, up0_ix: usize, up0_v: usize, up0_n: usize,
    up1_ip: usize, up1_ix: usize, up1_v: usize, up1_n: usize,
    e0: usize, e1: usize, r1: usize, r2: usize, s0: usize,
    /// 向量长度（**P176 钳制用** —— 见`lrn_dn0` 的越界说明）。
    e0_len: usize, e1_len: usize, r1_len: usize, r2_len: usize,
    eta_pc: f32, eta_oja: f32, w_max: f32,
    // ── P177：常驻池派发所需（理由同 `InferCtxC::stage_fn`）──
    stage_fn: usize,
    n_rows_hint: usize,
}
// SAFETY: 每个阶段只写自己 CSR 的行块；不同阶段写不同数组（或不同块）。
unsafe impl Send for LearnCtxC {}
unsafe impl Sync for LearnCtxC {}

/// `dn0.val += eta_pc·e0[i]·r1[idx[p]]`
unsafe fn lrn_dn0(c: *const LearnCtxC, lo: usize, hi_in: usize) {
    unsafe {
        let c = &*c;
        let ip = c.dn0_ip as *const i64;
        let ix = c.dn0_ix as *const i64;
        let v = c.dn0_v as *mut f32;
        let e0 = c.e0 as *const f32;
        let r1 = c.r1 as *const f32;
        // ⚠⚠⚠ **钳制到 e0 的长度**（P176 实测发现的**既有 bug**）。
        //
        // Python 的 `_pc_learn_fused` 里`n = dn0[0].shape[0] - 1`，
        // 而 `e0` 只有 `n1` 长、`dn0` 有 `n0` 行（`dn0 = transpose(up0)`）
        //   → 当 n0 > n1（**常见**）时Python 会**越界读** e0 的相邻内存。
        //   numba `prange` **不做边界检查** → 静默读到垃圾（不报错）。
        //
        //   实测：n0=64/n1=48 时越界 **16** 个元素。
        //
        //   Rust **不能复刻这个行为**（会段错误）→ 这里钳制到 e0 长度。
        //   ⚠ **这不是等价的**（Python 会更新那 16 行，Rust 不更新）。
        //   → 已在 `verify_m2_rust_kernels.py` 里作为**已知语义差异**记录，
        //     需 fhz 拍板是否修 Python 侧（P177 待办）。
        let hi = core::cmp::min(hi_in, c.e0_len);
        for i in lo..hi {
            let ai = *e0.add(i);
            if ai == 0.0 {                      // ⚠ 复刻 Python 的 continue
                continue;
            }
            let a = *ip.add(i) as usize;
            let b = *ip.add(i + 1) as usize;
            for p in a..b {
                let j = *ix.add(p) as usize;
                *v.add(p) += c.eta_pc * ai * *r1.add(j);
            }
        }
    }
}

/// `dn1.val += eta_pc·e1[i]·r2[idx[p]]`
unsafe fn lrn_dn1(c: *const LearnCtxC, lo: usize, hi: usize) {
    unsafe {
        let c = &*c;
        let ip = c.dn1_ip as *const i64;
        let ix = c.dn1_ix as *const i64;
        let v = c.dn1_v as *mut f32;
        let e1 = c.e1 as *const f32;
        let r2 = c.r2 as *const f32;
        for i in lo..hi {
            // ⚠ 同 lrn_dn0：钳制到 e1 长度（numba 不做边界检查 → 会越界读）
            if i >= c.e1_len {
                break;
            }
            let ai = *e1.add(i);
            if ai == 0.0 {
                continue;
            }
            let a = *ip.add(i) as usize;
            let b = *ip.add(i + 1) as usize;
            for p in a..b {
                let j = *ix.add(p) as usize;
                *v.add(p) += c.eta_pc * ai * *r2.add(j);
            }
        }
    }
}

/// `up0.val += eta_oja·r1[i]·(s0[idx[p]] − r1[i]·val[p])`
unsafe fn lrn_up0(c: *const LearnCtxC, lo: usize, hi: usize) {
    unsafe {
        let c = &*c;
        let ip = c.up0_ip as *const i64;
        let ix = c.up0_ix as *const i64;
        let v = c.up0_v as *mut f32;
        let r1 = c.r1 as *const f32;
        let s0 = c.s0 as *const f32;
        for i in lo..hi {
            let pi = *r1.add(i);
            if pi == 0.0 {
                continue;
            }
            let a = *ip.add(i) as usize;
            let b = *ip.add(i + 1) as usize;
            for p in a..b {
                let j = *ix.add(p) as usize;
                let vp = *v.add(p);
                *v.add(p) += c.eta_oja * pi * (*s0.add(j) - pi * vp);
            }
        }
    }
}

/// `up1.val += eta_oja·r2[i]·(r1[idx[p]] − r2[i]·val[p])`
unsafe fn lrn_up1(c: *const LearnCtxC, lo: usize, hi: usize) {
    unsafe {
        let c = &*c;
        let ip = c.up1_ip as *const i64;
        let ix = c.up1_ix as *const i64;
        let v = c.up1_v as *mut f32;
        let r1 = c.r1 as *const f32;
        let r2 = c.r2 as *const f32;
        for i in lo..hi {
            let pi = *r2.add(i);
            if pi == 0.0 {
                continue;
            }
            let a = *ip.add(i) as usize;
            let b = *ip.add(i + 1) as usize;
            for p in a..b {
                let j = *ix.add(p) as usize;
                let vp = *v.add(p);
                *v.add(p) += c.eta_oja * pi * (*r1.add(j) - pi * vp);
            }
        }
    }
}

/// clip ×4 中的**一个**（其余三个只换CSR）。
unsafe fn lrn_clip0(c: *const LearnCtxC, lo: usize, hi: usize) {
    unsafe { clip_one((*c).dn0_ip as *const i64, (*c).dn0_v as *mut f32, (*c).w_max, lo, hi) }
}
unsafe fn lrn_clip1(c: *const LearnCtxC, lo: usize, hi: usize) {
    unsafe { clip_one((*c).dn1_ip as *const i64, (*c).dn1_v as *mut f32, (*c).w_max, lo, hi) }
}
unsafe fn lrn_clip2(c: *const LearnCtxC, lo: usize, hi: usize) {
    unsafe { clip_one((*c).up0_ip as *const i64, (*c).up0_v as *mut f32, (*c).w_max, lo, hi) }
}
unsafe fn lrn_clip3(c: *const LearnCtxC, lo: usize, hi: usize) {
    unsafe { clip_one((*c).up1_ip as *const i64, (*c).up1_v as *mut f32, (*c).w_max, lo, hi) }
}

/// clip `[lo,hi)` 行的 val（⚠ 复刻 `if/elif`，**不用 `f32::clamp`**）。
#[inline]
unsafe fn clip_one(ip: *const i64, v: *mut f32, w_max: f32, lo: usize, hi: usize) {
    unsafe {
        for i in lo..hi {
            let a = *ip.add(i) as usize;
            let b = *ip.add(i + 1) as usize;
            for p in a..b {
                let x = *v.add(p);
                if x > w_max {
                    *v.add(p) = w_max;
                } else if x < -w_max {
                    *v.add(p) = -w_max;
                }
            }
        }
    }
}

/// 并行跑 learn 的一个阶段（门限同 [`run_stage`]）。
#[inline]
unsafe fn run_lrn(
    ctx: CtxPtr<LearnCtxC>,
    n: usize,
    n_threads: usize,
    f: unsafe fn(*const LearnCtxC, usize, usize),
) {
    if n == 0 {
        return;
    }
    let nt = if n * 128 < 256 * 1024 {
        1
    } else {
        n_threads.clamp(1, 8)
    };
    if nt <= 1 {
        unsafe { f(ctx.0, 0, n) };
        return;
    }
    let nt = nt.min(n);
    // ⚠ **P177：同 `run_stage`，改用常驻线程池**（`crate::pool::run`）。
    //   把当前 stage 的函数指针存进 ctx（`stage_fn`），行数存 `n_rows_hint`，
    //   由 `learn_bridge` 读出来分块派发。
    unsafe {
        let cm = ctx.0 as *mut LearnCtxC;
        (*cm).stage_fn = f as *const () as usize;
        (*cm).n_rows_hint = n;
        crate::pool::run(learn_bridge, cm as *mut u8, nt, n_threads);
    }
}

/// **融合学习**（对应 `_pc_learn_fused`，P176 全 fp32）。
///
/// ⚠ 4 个权重矩阵**原地更新**且**互不重叠** → 阶段顺序不影响语义，
///   与 Python 版 `prange` 的语义一致。
///
/// # Safety
/// 4 组 `val` 须**可写**（fp32 C 连续）；向量只读 fp32。
///
/// ⚠⚠ **`e0_len` / `e1_len` 必须传**真实长度 —— Python 的
///   `_pc_learn_fused` 对 `dn0`（n0 行）读 `e0`（n1 长）会**越界**
///   （numba `prange` 不做边界检查 → 静默读垃圾）。
///   Rust **钳制**到这两个长度 → **不等价**（见 `lrn_dn0` 注释）。
///   ⚠ 这是 P176 实测发现的**既有 bug**（P52 遗留），待 fhz 拍板。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m2_learn_fused(
    dn0_indptr: *const i64, dn0_idx: *const i64, dn0_val: *mut f32,
    n_dn0: usize,
    dn1_indptr: *const i64, dn1_idx: *const i64, dn1_val: *mut f32,
    n_dn1: usize,
    up0_indptr: *const i64, up0_idx: *const i64, up0_val: *mut f32,
    n_up0: usize,
    up1_indptr: *const i64, up1_idx: *const i64, up1_val: *mut f32,
    n_up1: usize,
    e0: *const f32, e1: *const f32,
    r1: *const f32, r2: *const f32, s0: *const f32,
    eta_pc: f32, eta_oja: f32, w_max: f32,
    e0_len: usize, e1_len: usize, r1_len: usize, r2_len: usize,
    n_threads: usize,
) -> i32 {
    let ctx = LearnCtxC {
        dn0_ip: dn0_indptr as usize, dn0_ix: dn0_idx as usize,
        dn0_v: dn0_val as usize, dn0_n: n_dn0,
        dn1_ip: dn1_indptr as usize, dn1_ix: dn1_idx as usize,
        dn1_v: dn1_val as usize, dn1_n: n_dn1,
        up0_ip: up0_indptr as usize, up0_ix: up0_idx as usize,
        up0_v: up0_val as usize, up0_n: n_up0,
        up1_ip: up1_indptr as usize, up1_ix: up1_idx as usize,
        up1_v: up1_val as usize, up1_n: n_up1,
        e0: e0 as usize, e1: e1 as usize,
        r1: r1 as usize, r2: r2 as usize, s0: s0 as usize,
        e0_len: e0_len, e1_len: e1_len, r1_len: r1_len, r2_len: r2_len,
        eta_pc, eta_oja, w_max,
        stage_fn: 0, n_rows_hint: 0,   // P177：run_lrn 派发前覆写
    };
    let p = CtxPtr(&ctx as *const LearnCtxC);
    unsafe {
        run_lrn(p, n_dn0, n_threads, lrn_dn0);
        run_lrn(p, n_dn1, n_threads, lrn_dn1);
        run_lrn(p, n_up0, n_threads, lrn_up0);
        run_lrn(p, n_up1, n_threads, lrn_up1);
        run_lrn(p, n_dn0, n_threads, lrn_clip0);
        run_lrn(p, n_dn1, n_threads, lrn_clip1);
        run_lrn(p, n_up0, n_threads, lrn_clip2);
        run_lrn(p, n_up1, n_threads, lrn_clip3);
    }
    0
}


#[cfg(test)]
mod tests_fused {
    use super::*;

    /// 造确定性 CSR（**方阵**：`n_rows == n_cols`）。
    fn make(n: usize, k: usize, seed: u64) -> (Vec<i64>, Vec<i64>, Vec<f32>) {
        make_rect(n, n, k, seed)
    }

    /// 造确定性 CSR（**矩形**：`n_rows` 行 × `n_cols` 列）。
    ///
    /// ⚠⚠ **必须有这个版本** —— 我第一版只有方阵 helper，而 `SparsePCStack`
    ///   的 4 个 CSR **全是矩形**：
    ///   · `up0`: n1 行 × n0 列   · `up1`: n2 行 × n1 列
    ///   · `dn0 = transpose(up0)`: **n0 行 × n1 列**
    ///   · `dn1 = transpose(up1)`: **n1 行 × n2 列**
    ///   拿方阵 helper 造矩形 → **idx 越界 → 段错误**（实测踩到**两次**：
    ///   第一次行数错、第二次列数错）。
    fn make_rect(n_rows: usize, n_cols: usize, k: usize, seed: u64)
        -> (Vec<i64>, Vec<i64>, Vec<f32>)
    {
        let mut s = seed;
        let nc = n_cols.max(1);
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as usize) % nc
        };
        let indptr: Vec<i64> = (0..=n_rows).map(|i| (i * k) as i64).collect();
        let mut idx = Vec::with_capacity(n_rows * k);
        let mut val = Vec::with_capacity(n_rows * k);
        for _ in 0..n_rows * k {
            idx.push(next() as i64);
            let t = next() as f32 / (nc as f32);
            val.push(t - 0.5);
        }
        (indptr, idx, val)
    }

    #[test]
    fn clip05_matches_python_semantics() {
        assert_eq!(clip05(0.7), 0.5);
        assert_eq!(clip05(-0.7), -0.5);
        assert_eq!(clip05(0.3), 0.3);
        // Python `min(0.5, max(-0.5, NaN))` → NaN
        assert!(clip05(f32::NAN).is_nan());
    }

    #[test]
    fn dot_seq_matches_manual() {
        let (n, k) = (64usize, 16usize);
        let (ip, ix, val) = make(n, k, 3);
        let v = unsafe { CsrView::new(ip.as_ptr(), ix.as_ptr(), val.as_ptr(), n) };
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03).collect();
        for r in [0usize, 1, 17, n - 1] {
            let got = unsafe { dot_seq(&v, r, x.as_ptr()) };
            let mut want = 0.0f32;
            for p in ip[r] as usize..ip[r + 1] as usize {
                want += val[p] * x[ix[p] as usize];
            }
            assert_eq!(got.to_bits(), want.to_bits(), "row {r}");
        }
    }

    /// 短行（<32）→ 全标量 → **应逐位**。
    #[test]
    fn infer_fused_short_row_is_bitwise() {
        let (n1, n2, k) = (48usize, 32usize, 16usize);
        let (u0i, u0x, u0v) = make_rect(n1, n1, k, 11);
        let (u1i, u1x, u1v) = make_rect(n2, n1, k, 13);
        let (d0i, d0x, d0v) = make_rect(n1, n2, k, 17);
        let (d1i, d1x, d1v) = make_rect(n1, n2, k, 19);
        let s0: Vec<f32> = (0..n1).map(|i| (i as f32) * 0.02 - 0.3).collect();
        let mut r1 = vec![0.0f32; n1];
        let mut r2 = vec![0.0f32; n2];
        let mut e0 = vec![0.0f32; n1];
        let mut e1 = vec![0.0f32; n1];
        unsafe {
            phdnet_m2_infer_fused(
                u0i.as_ptr(), u0x.as_ptr(), u0v.as_ptr(), n1,
                u1i.as_ptr(), u1x.as_ptr(), u1v.as_ptr(), n2,
                n1,                        // ⚠ P176: n0（= dn0 行数）
                d0i.as_ptr(), d0x.as_ptr(), d0v.as_ptr(),
                d1i.as_ptr(), d1x.as_ptr(), d1v.as_ptr(),
                s0.as_ptr(), r1.as_mut_ptr(), r2.as_mut_ptr(),
                e0.as_mut_ptr(), e1.as_mut_ptr(), 2, 1,
            );
        }
        let mv = |ip: &[i64], ix: &[i64], vl: &[f32], x: &[f32], n: usize| -> Vec<f32> {
            (0..n).map(|i| {
                let mut s = 0.0f32;
                for p in ip[i] as usize..ip[i + 1] as usize {
                    s += vl[p] * x[ix[p] as usize];
                }
                s
            }).collect()
        };
        let mut pr1 = mv(&u0i, &u0x, &u0v, &s0, n1);
        for v in pr1.iter_mut() { *v = v.tanh(); }
        let mut pr2 = mv(&u1i, &u1x, &u1v, &pr1, n2);
        for v in pr2.iter_mut() { *v = v.tanh(); }
        for _ in 0..2 {
            // ⚠ dn1 有 **n1 行**（transpose 语义），不是 n2
            let m1 = mv(&d1i, &d1x, &d1v, &pr2, n1);
            let pe1: Vec<f32> = (0..n1).map(|i| pr1[i] - m1[i]).collect();
            let pd2: Vec<f32> = mv(&u1i, &u1x, &u1v, &pe1, n2)
                .into_iter().map(clip05).collect();
            for i in 0..n2 { pr2[i] = (pr2[i] + 0.15 * pd2[i]).tanh(); }
            let m0 = mv(&d0i, &d0x, &d0v, &pr1, n1);   // dn0: n1 行
            let pe0: Vec<f32> = (0..n1).map(|i| s0[i] - m0[i]).collect();
            let pd1: Vec<f32> = mv(&u0i, &u0x, &u0v, &pe0, n1)
                .into_iter().map(clip05).collect();
            for i in 0..n1 { pr1[i] = (pr1[i] + 0.15 * pd1[i]).tanh(); }
        }
        let m0 = mv(&d0i, &d0x, &d0v, &pr1, n1);   // dn0: n1 行
        let m1 = mv(&d1i, &d1x, &d1v, &pr2, n1);   // ⚠ dn1: n1 行
        for i in 0..n1 {
            assert_eq!(r1[i].to_bits(), pr1[i].to_bits(), "r1[{i}]");
            assert_eq!(e0[i].to_bits(), (s0[i] - m0[i]).to_bits(), "e0[{i}]");
            assert_eq!(e1[i].to_bits(), (pr1[i] - m1[i]).to_bits(), "e1[{i}]");
        }
        for i in 0..n2 {
            assert_eq!(r2[i].to_bits(), pr2[i].to_bits(), "r2[{i}]");
        }
    }

    #[test]
    fn learn_fused_matches_python_bitwise() {
        let (n1, n2, k) = (32usize, 24usize, 8usize);
        // 矩形：dn0 n1×n2 · dn1 n1×n2 · up0 n1×n1 · up1 n2×n1
        let (d0i, d0x, d0v0) = make_rect(n1, n2, k, 23);
        let (d1i, d1x, d1v0) = make_rect(n1, n2, k, 29);
        let (u0i, u0x, u0v0) = make_rect(n1, n1, k, 31);
        let (u1i, u1x, u1v0) = make_rect(n2, n1, k, 37);
        let e0: Vec<f32> = (0..n1)
            .map(|i| if i % 4 == 0 { 0.0 } else { (i as f32) * 0.03 }).collect();
        let e1: Vec<f32> = (0..n2).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let r1: Vec<f32> = (0..n1).map(|i| (i as f32) * 0.04).collect();
        let r2: Vec<f32> = (0..n2).map(|i| (i as f32) * 0.05).collect();
        let s0: Vec<f32> = (0..n1).map(|i| (i as f32) * 0.01).collect();
        let (epc, eoj, wm) = (0.05f32, 0.02f32, 2.0f32);

        let mut p_d0 = d0v0.clone();
        for i in 0..n1 {
            let ai = e0[i];
            if ai == 0.0 { continue; }
            for p in d0i[i] as usize..d0i[i + 1] as usize {
                p_d0[p] += epc * ai * r1[d0x[p] as usize];
            }
        }
        let mut p_d1 = d1v0.clone();
        for i in 0..n2 {
            let ai = e1[i];
            if ai == 0.0 { continue; }
            for p in d1i[i] as usize..d1i[i + 1] as usize {
                p_d1[p] += epc * ai * r2[d1x[p] as usize];
            }
        }
        let mut p_u0 = u0v0.clone();
        for i in 0..n1 {
            let pi = r1[i];
            if pi == 0.0 { continue; }
            for p in u0i[i] as usize..u0i[i + 1] as usize {
                p_u0[p] += eoj * pi * (s0[u0x[p] as usize] - pi * p_u0[p]);
            }
        }
        let mut p_u1 = u1v0.clone();
        for i in 0..n2 {
            let pi = r2[i];
            if pi == 0.0 { continue; }
            for p in u1i[i] as usize..u1i[i + 1] as usize {
                p_u1[p] += eoj * pi * (r1[u1x[p] as usize] - pi * p_u1[p]);
            }
        }
        for t in [&mut p_d0, &mut p_d1, &mut p_u0, &mut p_u1] {
            for v in t.iter_mut() {
                if *v > wm { *v = wm; } else if *v < -wm { *v = -wm; }
            }
        }

        let mut r_d0 = d0v0.clone();
        let mut r_d1 = d1v0.clone();
        let mut r_u0 = u0v0.clone();
        let mut r_u1 = u1v0.clone();
        unsafe {
            phdnet_m2_learn_fused(
                d0i.as_ptr(), d0x.as_ptr(), r_d0.as_mut_ptr(), n1,
                d1i.as_ptr(), d1x.as_ptr(), r_d1.as_mut_ptr(), n1,
                u0i.as_ptr(), u0x.as_ptr(), r_u0.as_mut_ptr(), n1,
                u1i.as_ptr(), u1x.as_ptr(), r_u1.as_mut_ptr(), n2,
                e0.as_ptr(), e1.as_ptr(), r1.as_ptr(), r2.as_ptr(), s0.as_ptr(),
                epc, eoj, wm,
                // ⚠ P176：**传真实长度**（dn1 有 n1=32 行但 e1 只有 n2=24 长
                //   → 越界读。Rust 钳制到这些长度，与 Python 的越界行为**不等价**，
                //   但Rust 不能越界 → 这是P176 记录的**已知语义差异**）。
                e0.len(), e1.len(), r1.len(), r2.len(), 1,
            );
        }
        for (i, (a, b)) in [&r_d0, &r_d1, &r_u0, &r_u1]
            .iter().zip([&p_d0, &p_d1, &p_u0, &p_u1].iter()).enumerate()
        {
            for j in 0..a.len() {
                assert_eq!(a[j].to_bits(), b[j].to_bits(),
                           "matrix {i} edge {j}: {} vs {}", a[j], b[j]);
            }
        }
    }

    /// 长行（≥32）走 SIMD → 与标量参考**容差内**（非逐位）。
    ///
    /// ⚠ **实测末端 relerr = 4.6e-07**（fp32 eps 量级）。
    ///   ⚠ 我一度写成「误差沿迭代放大到 ~0.29」—— **那是我的参照 bug**
    ///   （dn1 行数按 n2 造但实际是 n1 → 越界读到垃圾值），**不是真实性质**。
    ///   修正后的实测：**放大很轻微**，6 个阶段 × n_steps 轮串联仍在 1e-06 量级。
    ///   → 融合核的 SIMD 路径**可用于回归**（判据 1e-5）。
    #[test]
    fn infer_fused_long_row_avx2_within_tolerance() {
        let (n1, n2, k) = (256usize, 192usize, 64usize);
        // ⚠ 4 个 CSR 全是**矩形**（见 make_rect 说明）
        let (u0i, u0x, u0v) = make_rect(n1, n1, k, 41);
        let (u1i, u1x, u1v) = make_rect(n2, n1, k, 43);
        let (d0i, d0x, d0v) = make_rect(n1, n2, k, 47);
        let (d1i, d1x, d1v) = make_rect(n1, n2, k, 53);
        let s0: Vec<f32> = (0..n1).map(|i| (i as f32) * 0.002 - 0.3).collect();
        let mut r1 = vec![0.0f32; n1];
        let mut r2 = vec![0.0f32; n2];
        let mut e0 = vec![0.0f32; n1];
        let mut e1 = vec![0.0f32; n1];
        unsafe {
            phdnet_m2_infer_fused(
                u0i.as_ptr(), u0x.as_ptr(), u0v.as_ptr(), n1,
                u1i.as_ptr(), u1x.as_ptr(), u1v.as_ptr(), n2,
                n1,                        // ⚠ P176: n0（= dn0 行数）
                d0i.as_ptr(), d0x.as_ptr(), d0v.as_ptr(),
                d1i.as_ptr(), d1x.as_ptr(), d1v.as_ptr(),
                s0.as_ptr(), r1.as_mut_ptr(), r2.as_mut_ptr(),
                e0.as_mut_ptr(), e1.as_mut_ptr(), 2, 1,
            );
        }
        let mv = |ip: &[i64], ix: &[i64], vl: &[f32], x: &[f32], n: usize| -> Vec<f32> {
            (0..n).map(|i| {
                let mut s = 0.0f32;
                for p in ip[i] as usize..ip[i + 1] as usize {
                    s += vl[p] * x[ix[p] as usize];
                }
                s
            }).collect()
        };
        let mut pr1 = mv(&u0i, &u0x, &u0v, &s0, n1);
        for v in pr1.iter_mut() { *v = v.tanh(); }
        let mut pr2 = mv(&u1i, &u1x, &u1v, &pr1, n2);
        for v in pr2.iter_mut() { *v = v.tanh(); }
        for _ in 0..2 {
            let m1 = mv(&d1i, &d1x, &d1v, &pr2, n1);   // ⚠ dn1: n1 行
            let pe1: Vec<f32> = (0..n1).map(|i| pr1[i] - m1[i]).collect();
            let pd2: Vec<f32> = mv(&u1i, &u1x, &u1v, &pe1, n2)
                .into_iter().map(clip05).collect();
            for i in 0..n2 { pr2[i] = (pr2[i] + 0.15 * pd2[i]).tanh(); }
            let m0 = mv(&d0i, &d0x, &d0v, &pr1, n1);
            let pe0: Vec<f32> = (0..n1).map(|i| s0[i] - m0[i]).collect();
            let pd1: Vec<f32> = mv(&u0i, &u0x, &u0v, &pe0, n1)
                .into_iter().map(clip05).collect();
            for i in 0..n1 { pr1[i] = (pr1[i] + 0.15 * pd1[i]).tanh(); }
        }
        let denom = pr1.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
        let worst = pr1.iter().zip(r1.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs() / denom));
        assert!(r1.iter().all(|v| v.is_finite()), "输出含 NaN/Inf");
        println!("长行 SIMD 末端 relerr = {worst:e}");
        assert!(worst < 1e-5, "worst_relerr={worst:e}");
    }
}
