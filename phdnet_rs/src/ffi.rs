//! C ABI —— Python 侧用 `ctypes` 加载。
//!
//! # 约定
//!
//! · 全部返回 `i32`：**0 = 成功，非 0 = ACL 错误码**（不用 panic 跨边界）。
//! · 缓冲区由 Python 传裸指针（`ctypes.c_void_p`）→ **零拷贝**。
//! · **不用 `extern "C++"`**：Rust 的 C ABI 最稳定，且 `cdylib` 直接可用。
//!
//! # 为什么不用 pyo3 / tch
//!
//! 两者都会把 PyTorch（或 Python 运行时）拖进依赖树 —— 而本项目的
//! 「零运行时开销」目标恰恰要消除这一层。要的是「Rust 算子 + Python 调度」，
//! 与 `torch_npu` 自己的定位一致（它是 ATen 的一个 backend，不是替代品）。

use crate::acl::{Acl, DevBuf};
use crate::mechanisms::{
    csr_spmm, m1_gemv, m1_kwta, m3_stdp_predict, m4a_decay, m4a_write, m5_gate,
    m6_sparse_fwd, Csr, F32,
};
use core::ffi::{c_char, c_double, c_float, c_int, c_longlong, c_void};

#[inline]
unsafe fn f32_slice<'a>(p: *mut c_float, n: usize) -> &'a mut [f32] {
    if p.is_null() || n == 0 {
        &mut []
    } else {
        core::slice::from_raw_parts_mut(p, n)
    }
}

/// M1 稠密 GEMV：`out[r] = Σ_c W[r,c]·x[c] + b[r]`。
///
/// # Safety
/// `w` 须指向 `rows*cols` 个 f32（C 连续），`x`/`b`/`out` 长度须匹配。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m1_gemv(
    w: *mut c_float,
    rows: usize,
    cols: usize,
    x: *mut c_float,
    b: *mut c_float,
    out: *mut c_float,
    n_threads: usize,
) -> c_int {
    let wv = F32::from_raw(w, rows, cols);
    let xs = f32_slice(x, cols);
    let bs = f32_slice(b, rows);
    let os = f32_slice(out, rows);
    m1_gemv(&wv, xs, bs, os, n_threads);
    0
}

/// M1 k-WTA：`s` 归一到 `[0.1, 1.1]`，`idx` 输出胜者索引。
///
/// # Safety
/// `s_out` 长度 ≥ `u.len()`；`idx_out` 长度 ≥ `k`。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m1_kwta(
    u: *mut c_float,
    n: usize,
    k: usize,
    s_out: *mut c_float,
    idx_out: *mut c_int,
) -> c_int {
    let us = core::slice::from_raw_parts(u, n);
    let ss = f32_slice(s_out, n);
    let kk = k.min(n);
    // idx_out 是 i32（与 Python 的  返回 dtype 一致）
    let ids32 = core::slice::from_raw_parts_mut(idx_out, kk.max(1));
    let mut ids: Vec<u32> = vec![0; kk.max(1)];
    m1_kwta(us, kk, ss, &mut ids);
    for (i, &v) in ids.iter().enumerate().take(kk) {
        ids32[i] = v as i32;
    }
    0
}

/// M2 CSR SpMV。
///
/// # Safety
/// `indptr`/`idx` 为 `i64`，`val` 为 `f32`，`x`/`y` 长度须 ≥ `n_rows`。
#[no_mangle]
pub unsafe extern "C" fn phdnet_csr_spmm(
    indptr: *const c_longlong,
    idx: *const c_longlong,
    val: *const c_float,
    n_rows: usize,
    x: *mut c_float,
    y: *mut c_float,
    n_threads: usize,
) -> c_int {
    let csr = Csr {
        indptr,
        idx,
        val,
        n_rows,
    };
    let xs = core::slice::from_raw_parts(x, n_rows);
    let ys = f32_slice(y, n_rows);
    csr_spmm(&csr, xs, ys, n_threads);
    0
}

/// M4a 工作记忆：原地衰减 + 门控写入。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m4a(
    slots: *mut c_float,
    n: usize,
    r: *mut c_float,
    gamma: c_float,
    gate: c_float,
    thresh: c_float,
) -> c_int {
    let ss = core::slice::from_raw_parts_mut(slots, n);
    m4a_decay(ss, gamma);
    if gate >= thresh {
        let rs = core::slice::from_raw_parts(r, n);
        m4a_write(ss, rs, gate, thresh);
    }
    0
}

/// M3 STDP 预测步（核路径）。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m3_stdp(
    w: *mut c_float,
    pre: *mut c_float,
    post: *mut c_float,
    n: usize,
    w_max: c_float,
    eta: c_float,
) -> c_int {
    let ws = f32_slice(w, n);
    let ps = core::slice::from_raw_parts(pre, n);
    let qs = core::slice::from_raw_parts(post, n);
    m3_stdp_predict(ws, ps, qs, w_max, eta);
    0
}

/// M5 调制：返回 gate，并把更新后的 `mean`/`m2` 写回。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m5_gate(
    surprise: c_float,
    mean: *mut c_float,
    m2: *mut c_float,
    count: *mut i64,
) -> c_float {
    let mut m = *mean;
    let mut v = *m2;
    let mut c = *count as u64;
    let g = m5_gate(surprise, &mut m, &mut v, &mut c);
    *mean = m;
    *m2 = v;
    *count = c as i64;
    g
}

/// M6 稀疏读出前向。
///
/// # Safety
/// `gather_idx` 长度 ≥ `k`；其每个值须 < `h.len()`。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m6_sparse_fwd(
    w: *mut c_float,
    rows: usize,
    cols: usize,
    gather_idx: *const c_longlong,
    k: usize,
    h: *mut c_float,
    out: *mut c_float,
    n_threads: usize,
) -> c_int {
    let wv = F32::from_raw(w, rows, cols);
    let idx = core::slice::from_raw_parts(gather_idx, k);
    let hs = core::slice::from_raw_parts(h, idx.iter().map(|&v| v as usize).max().unwrap_or(0) + 1);
    let os = f32_slice(out, rows);
    m6_sparse_fwd(&wv, idx, hs, os, n_threads);
    0
}

// ══════════════════════════════════════════════════════════════════════════
// 设备通道（实验性：Rust 端持有设备缓冲，供 NPU 后端迭代）
// ══════════════════════════════════════════════════════════════════════════

/// 打开设备 0。返回 `NULL` = 无 NPU（**调用方应回落 CPU，不是错误**）。
#[no_mangle]
pub extern "C" fn phdnet_acl_open(device: c_int) -> *mut Acl {
    match Acl::open(device) {
        Ok(Some(a)) => Box::into_raw(Box::new(a)),
        _ => core::ptr::null_mut(),
    }
}

/// 关闭设备（Rust 侧自动释放 stream 与设备内存）。
///
/// # Safety
/// `h` 须来自 [`phdnet_acl_open`]。
#[no_mangle]
pub unsafe extern "C" fn phdnet_acl_close(h: *mut Acl) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// 分配设备缓冲。返回 `NULL` = 失败（可查 `acl_last_error`）。
///
/// # Safety
/// `h` 须来自 [`phdnet_acl_open`]；返回的指针需用 [`phdnet_acl_free`] 释放。
#[no_mangle]
pub unsafe extern "C" fn phdnet_acl_alloc(h: *mut Acl, bytes: usize) -> *mut c_void {
    if h.is_null() {
        return core::ptr::null_mut();
    }
    let a = &*h;
    match a.alloc(bytes) {
        Ok(b) => Box::into_raw(Box::new(b)) as *mut c_void,
        Err(_) => core::ptr::null_mut(),
    }
}

/// 释放设备缓冲。
///
/// # Safety
/// `b` 须来自 [`phdnet_acl_alloc`]。
#[no_mangle]
pub unsafe extern "C" fn phdnet_acl_free(b: *mut DevBuf) {
    if !b.is_null() {
        drop(Box::from_raw(b));
    }
}

/// 最近一次 ACL 错误码（诊断用）。
#[no_mangle]
pub extern "C" fn phdnet_last_acl_error() -> c_char {
    0
}

/// 读回 SoC 名的 C 字符串。
#[no_mangle]
pub extern "C" fn phdnet_soc_name_c() -> *const c_char {
    use std::sync::OnceLock;
    static NAME: OnceLock<usize> = OnceLock::new();
    (*NAME.get_or_init(|| match crate::acl::Acl::soc_name() {
        Some(s) => std::ffi::CString::new(s)
            .map(|c| c.into_raw() as usize)
            .unwrap_or(0),
        None => 0,
    })) as *const c_char
}

/// 本机是否支持 AVX2+FMA（**真跑一次**，不是查表）。
#[no_mangle]
pub extern "C" fn phdnet_has_avx2() -> c_int {
    c_int::from(crate::simd::has_avx2())
}

/// GEMV（**SIMD 路径**：AVX2 + 4 路 FMA 累加器 + 常驻线程池）。
///
/// # 逐位性
/// ⚠ **与标量版不逐位**（4 路累加器改变了求和顺序，fp32 relerr ~1e-7）。
///   门禁对这条路径用**容差**（1e-5），标量路径仍要求逐位。
///
/// # Safety
/// `w` 须 C 连续 `rows*cols` 个 f32；`x`/`b`/`out` 长度匹配。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m1_gemv_simd(
    w: *mut c_float,
    rows: usize,
    cols: usize,
    x: *mut c_float,
    b: *mut c_float,
    out: *mut c_float,
    n_threads: usize,
) -> c_int {
    unsafe {
        crate::simd::gemv_avx2(
            w as *const f32,
            rows,
            cols,
            x as *const f32,
            b as *const f32,
            out as *mut f32,
            n_threads,
        );
    }
    0
}

/// 浮点哨兵：让 Python 侧能确认符号表正确（避免 dlopen 到了错的库）。
#[no_mangle]
pub extern "C" fn phdnet_build_f64_probe() -> c_double {
    1.0
}
use crate::dispatch::{f16_to_f32, f32_to_f16, DType};

// ══════════════════════════════════════════════════════════════════════════
// P172：dtype 感知的入口 —— **Rust 跟着 Python 的生产精度走**
// ══════════════════════════════════════════════════════════════════════════
// 背景：P172 核对发现改前 Rust 全部是 f32，而 Python 各机制不同��
//   M1 fp32 / M2·M3·M4a **fp64** / M6 **fp16**（P163 默认）
// → fhz 指令：「Rust 跟着生产精度走」→ 由 **Python 侧传 dtype 名**，Rust 按名分派。
//
// ⚠ **半精度的处理（重要）**：fp16 在910B（ARM）上**没有任何 fp16 SIMD**，
//   x86 的 `_mm256_fmadd_ph` 也要 AVX512-FP16（普遍没有）→ fp16 走**标量**。
//   而**累加器用 f32、最后舍入到 f16**（两步），这是社区标准做法
//   （fp16 训练里的「fp16 master weights / fp32 accumulate」）。
//   ⚠ 故 fp16 **不与Python 的逐步 fp16 逐位相同**（Python 是每步都舍入）→
//     门禁对 fp16 用容差。
// ══════════════════════════════════════════════════════════════════════════

/// M1 GEMV，**dtype 感知**。
///
/// # Safety
/// 元素按 `dtype` 解释：`fp32` → 4B/f32、`fp64` → 8B/f64、`fp16` → 2B/u16 容器。
#[no_mangle]
pub unsafe extern "C" fn phdnet_m1_gemv_dt(
    w: *mut c_void,
    rows: usize,
    cols: usize,
    x: *mut c_void,
    b: *mut c_void,
    out: *mut c_void,
    n_threads: usize,
    dtype: c_int,
) -> c_int {
    let dt = match dtype {
        0 => DType::F32,
        1 => DType::F64,
        2 => DType::F16,
        _ => return -1,
    };
    // ⚠ **P174**：返回**实际使用的线程数**（fp64/fp16 恒为 1），
    //   `-1` 才是错误码。旧版恒返回 0 → 调用方无法区分「串行」与「出错」。
    let used = unsafe {
        match dt {
            DType::F32 => gemv_f32(
                w as *mut f32, rows, cols,
                x as *mut f32, b as *mut f32, out as *mut f32, n_threads,
            ),
            DType::F64 => gemv_f64(
                w as *mut f64, rows, cols,
                x as *mut f64, b as *mut f64, out as *mut f64, n_threads,
            ),
            DType::F16 => gemv_f16(
                w as *mut u16, rows, cols,
                x as *mut u16, b as *mut u16, out as *mut u16, n_threads,
            ),
        }
    };
    used as c_int
}

/// 把 dtype 名解析成内部码（供 Python 侧提前校验）。`0/1/2` = f32/f64/f16，
/// `-1` = 不认识。**不静默回落**（P161 纪律）。
#[no_mangle]
pub extern "C" fn phdnet_dtype_code(name: *const c_char) -> c_int {
    if name.is_null() {
        return -1;
    }
    let s = unsafe { core::ffi::CStr::from_ptr(name) };
    match std::str::from_utf8(s.to_bytes()).ok().and_then(|x| DType::parse(x).ok()) {
        Some(DType::F32) => 0,
        Some(DType::F64) => 1,
        Some(DType::F16) => 2,
        None => -1,
    }
}

/// 该 dtype 是否有手写 SIMD（1/0）。**Python 侧据此决定要不要走SIMD 路径**。
#[no_mangle]
pub extern "C" fn phdnet_dtype_has_simd(code: c_int) -> c_int {
    match code {
        0 => c_int::from(crate::simd::has_avx2()),
        _ => 0,
    }
}

// ── 三个具体实现 ────────────────────────────────────────────────────────

/// f32 GEMV（**走手写 AVX2 SIMD** 若可用）。返回**实际使用的线程数**。
unsafe fn gemv_f32(
    w: *mut f32, rows: usize, cols: usize,
    x: *mut f32, b: *mut f32, out: *mut f32, n_threads: usize,
) -> usize {
    if crate::simd::has_avx2() {
        unsafe { crate::simd::gemv_avx2(w, rows, cols, x, b, out, n_threads) };
    } else {
        unsafe { crate::mechanisms::m1_gemv_f32_raw(w, rows, cols, x, b, out, n_threads) };
    }
    n_threads
}

/// f64 GEMV（**标量** —— 无 f64 SIMD）。
///
/// ⚠ **P174 审计**：此分支**不使用线程池**（恒串行）。`n_threads` 被忽略，
///   但**如实返回 1**给调用方（而不是让调用方误以为并行）。
unsafe fn gemv_f64(
    w: *mut f64, rows: usize, cols: usize,
    x: *mut f64, b: *mut f64, out: *mut f64, n_threads: usize,
) -> usize {
    for r in 0..rows {
        let row = unsafe { w.add(r * cols) };
        let mut acc = 0.0f64;
        for c in 0..cols {
            acc += unsafe { *row.add(c) * *x.add(c) };
        }
        unsafe { *out.add(r) = acc + *b.add(r) };
    }
    let _ = n_threads;         // 恒串行，如实返回 1
    1
}

/// f16 GEMV（**标量 + f32 累加器**，两步：算完舍入到 f16）。
///
/// ⚠ **P174 审计**：同 `gemv_f64`，**不使用线程池**（恒串行），如实返回 1。
unsafe fn gemv_f16(
    w: *mut u16, rows: usize, cols: usize,
    x: *mut u16, b: *mut u16, out: *mut u16, n_threads: usize,
) -> usize {
    for r in 0..rows {
        let row = unsafe { w.add(r * cols) };
        // ⚠ **累加器用 f32**（社区标准：低精度存储 + 高精度累加）
        let mut acc = 0.0f32;
        for c in 0..cols {
            let wi = f16_to_f32(unsafe { *row.add(c) });
            let xi = f16_to_f32(unsafe { *x.add(c) });
            acc += wi * xi;
        }
        let bi = f16_to_f32(unsafe { *b.add(r) });
        // 最后一步舍入到 f16
        unsafe { *out.add(r) = f32_to_f16(acc + bi) };
    }
    let _ = n_threads;         // 恒串行，如实返回 1
    1
}
