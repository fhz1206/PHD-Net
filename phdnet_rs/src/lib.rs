//! PHD-Net Rust **算子库** —— M1–M6 的无状态数值核 + CANN/ACL 后端。
//!
//! # 架构约定：**Python 是顶层，Rust 是算子**（fhz 2026-10-03）
//!
//! ```text
//!   Python（顶层）                        Rust（本 crate，cdylib）
//!   ─────────────────────────             ──────────────────────────
//!   model.step 的编排顺序                 extern "C" 的无状态数值核
//!   状态跨 token/shard/epoch 的连续性│ ACL 句柄 / 设备内存（RAII）
//!   门禁判据、开关、回落决策              GE 图模式的载体
//!   日志（segments(win) 等）              ── 不允许出现编排逻辑 ──
//!   --nll-sync-every 等性能调度
//! ```
//!
//! **边界的硬规则**（违反即错）：
//! · **Rust 侧不许有编排** —— 没有 `step()`、没有「下一token 做什么」的决策、
//!   没有日志。Rust 只回答「给我数组，算出结果」。
//! · **Python 侧不许有数值循环** —— `for i in range(n): y[i] = ...`
//!   属于 Rust。这是**性能边界**，不是风格偏好。
//!
//! # 为什么这么分
//!
//! · **编排要动态**（门禁 / 开关 / 回落 / 可归因的日志）→ Python 的表达力够；
//!   项目现有的全部机制（四条铁律、逐 token 语义、状态连续）都在那一层，
//!   **移走它们会破坏可审计性**。
//! · **算子要零开销 + nogil 多核** → Rust 的裸指针 + `thread::scope` 够。
//!   Python 的 numba `prange` 在 x86 只快 1.16×（GIL + 线程池吃掉大部分）。
//! · **可独立对拍**：Rust 不依赖 Python 语义 → 门禁能要求**逐位**
//!   （见 `tests/verifiers/verify_rust_kernels.py`）。
//!
//! # ⚠ 必须先说清的一件事（诚实边界）
//!
//!「NPU 算力没用尽」这个判断**是对的** —— 实测读出 7.83 ms 是搬运下界
//! 0.300 ms 的 **26.1×**。但**本 crate 不能独立解决那 26×**，原因：
//!
//! | 证据 | 数值 |
//! |---|---|
//! | 裸算子提交耗时 | **0.35 ms**（Python 侧全部开销） |
//! | CANN launch 开销 | **~50–200 μs / 算子**（项目自己的注释记录） |
//! | 读出路径算子数 | **37 个** |
//! | 37 × 50–200 μs | **1.9–7.4 ms 的纯下发开销** |
//!
//! → **那 26× 的主体是「算子逐个下发」**，不是「Python 慢」。
//! → Rust 把 37 次下发变成 37 次 **FFI** 调用，**下发次数不变**。
//!
//! **真正对症的是 GE 图模式**（昇腾官方调优文档的第一原则：
//! 「减少 Host 算子下发时间」）→ 见 [`acl::Graph`]。
//! **本 crate 的价值是给图模式提供载体**：图要一次构建、静态 shape、复用，
//! Python 的动态属性访问无法保证，Rust 能提供稳定的 C ABI。
//!
//! # 与 Python 版的关系
//!
//! **Python 版是参照实现，Rust 版是候选实现。对拍通过之前不启用。**

pub mod acl;
pub mod ffi;
pub mod mechanisms;
pub mod pool;
pub mod dispatch;
pub mod m2_csr;
pub mod m2_fused;
pub mod simd;
pub mod fp8_conv;

/// M2 判并行的工作量门限（**总非零数 nnz**）。
///
/// # 为什么需要它（P174 审计发现）
/// 线程池派发固定成本 **41.5 µs**（本机实测，机器噪声仅 4%），
/// 而 M2 的 CSR 算子**每元素工作量极小** → 中小规模并行**反而更慢**。
///
/// # 门限依据（实测盈亏平衡点，本机 x86）
/// · 256x512（131072 nnz）：串行 28.7 µs → 8 线程 70.6 µs（**0.41×**）
/// · 512x512（262144 nnz）：36.8 → 37.2 µs（**0.99×，持平**）← 平衡点
/// · 1024x2048（2097152 nnz）：355.5 → 283.4 µs（**1.25×，并行赚**）
///
/// ⚠ **1b 档 M2**：`big_ltm 2^24 × 60` 的 nnz 量级 ≫ 门限 → 仍并行，正确。
/// ⚠ **本机（Windows 开发机）性能测量不可信**（同一 shape 串行耗时在不同时刻
///   差 7 倍：28.7 → 200.5 µs）→ 门限取**保守值**，需 Ascend 服务器复核。
///
/// P190：262144 → **32768**。旧值是 `std::thread::scope` 时代的实测（每次调用
/// 新起线程，派发 ~41.5µs）；M2 各核现走**常驻线程池**，派发成本大幅摊薄——
/// 实测 131k nnz 裸核 2 线程 0.065ms **快于**串行 0.080ms（并行已能赚）。
/// 32768 nnz ≈ 0.13MB val+idx 流量，约 20–40µs 工作量，仍够覆盖池派发。
pub const MIN_PAR_NNZ: usize = 32 * 1024;

/// 按「总工作量」决定分块数：nnz 太小就**串行**（省派发成本）。
///
/// # Safety
/// `indptr` 须至少有 `n_rows+1` 个元素（读 `indptr[n_rows]` 作总 nnz）。
#[inline]
pub unsafe fn parts_for_csr(indptr: *const i64, n_rows: usize, want: usize) -> usize {
    let nnz = if n_rows == 0 {
        0
    } else {
        unsafe { *indptr.add(n_rows) as usize }
    };
    let n = want.clamp(1, n_rows.max(1));
    if nnz < MIN_PAR_NNZ {
        1
    } else {
        n
    }
}

/// 按「元素总数」决定分块数（给没有 indptr 的算子如 `clip` 用）。
#[inline]
pub fn parts_for_len(n: usize, want: usize) -> usize {
    let nt = want.clamp(1, n.max(1));
    if n < MIN_PAR_NNZ {
        1
    } else {
        nt
    }
}

pub use acl::{Acl, AclError, DevBuf};
pub use mechanisms::*;

/// crate 版本（供Python 侧 sanity check —— 版本不匹配时应拒绝加载）。
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Python 侧会先调它：返回 1 表示「本库有 NPU 后端」，0 表示「应回落 CPU」。
///
/// ⚠ **回落不是错误** —— 与 P161 的教训一致：不支持要**明确返回**，
/// 不能抛异常让上层以为是崩溃。
#[no_mangle]
pub extern "C" fn phdnet_has_npu() -> i32 {
    i32::from(acl::available())
}

/// 返回本机 SoC 名（`Ascend910B4` / `None`= 无 CANN）。
#[no_mangle]
pub extern "C" fn phdnet_soc_name() -> *mut core::ffi::c_char {
    match acl::Acl::soc_name() {
        Some(s) => std::ffi::CString::new(s)
            .map(|c| c.into_raw())
            .unwrap_or(std::ptr::null_mut()),
        None => std::ptr::null_mut(),
    }
}