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
pub mod simd;

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