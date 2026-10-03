"""P171：把 mechanisms.rs 的分块执行改用常驻线程池。"""
import io

P = "phdnet_rs/src/mechanisms.rs"
s = io.open(P, encoding="utf-8").read()

# ── 1) m1_gemv：用池 + work 函数 ──
old_gemv_start = s.index("    let x_ptr = x.as_ptr() as usize;")
old_gemv_end = s.index("/// 单行 GEMV。**行内列序累加** → 与 Python 版逐位一致。")
new_gemv = '''    let x_ptr = x.as_ptr() as usize;
    let b_ptr = b.as_ptr() as usize;
    let cols = w.cols;
    // ⚠ **P171：用常驻线程池**（不再每步 `thread::scope` 建线程）。
    //   实测：scope 版 8 线程 1096 µsvs numba prange 599 µs —— 差距全在
    //   「建 8 线程 + join + 销毁」。池化后派发是微秒级。
    // SAFETY（两个 unsafe impl 的依据）：见 F32/Csr 的注释 ——
    //   各part 写连续行块，**互不重叠**；`w`/`x`/`b` 只读。
    let ctx = JobCtx {
        w,
        x_ptr,
        b_ptr,
        out_ptr: out.as_mut_ptr(),
        n,
        cols,
    };
    crate::pool::run(m1_gemv_work, &ctx as *const JobCtx as *mut u8, nt, nt);
}

/// GEMV 的一个分块。**只写自己那段行**（连续、不重叠）。
///
/// # Safety
/// `ctx` 指向的 `w`/`x`/`b` 有效；本函数只写 `out[lo..hi)`。
unsafe fn m1_gemv_work(ctx: *mut u8, part: usize, n_parts: usize) {
    let c = unsafe { &*(ctx as *const JobCtx) };
    let chunk = c.n.div_ceil(n_parts);
    let lo = part * chunk;
    let hi = ((part + 1) * chunk).min(c.n);
    if lo >= hi {
        return;
    }
    let xs = unsafe { core::slice::from_raw_parts(c.x_ptr as *const f32, c.cols) };
    let bs = unsafe { core::slice::from_raw_parts(c.b_ptr as *const f32, c.n) };
    let os = unsafe {
        core::slice::from_raw_parts_mut(c.out_ptr.add(lo), hi - lo)
    };
    for (i, o) in os.iter_mut().enumerate() {
        gemv_row(c.w, lo + i, xs, bs[lo + i], o);
    }
}

/// GEMV 的线程间上下文。**含裸指针** → 显式 `unsafe impl Send/Sync`
/// （安全性见注释：各 part 写不重叠的行块）。
pub struct JobCtx {
    w: F32<'static>,
    x_ptr: usize,
    b_ptr: usize,
    out_ptr: *mut f32,
    n: usize,
    cols: usize,
}
// SAFETY: 每个 part 只写 `out[lo..hi)`（连续、互不重叠），其余只读。
unsafe impl Send for JobCtx {}
unsafe impl Sync for JobCtx {}

'''
s = s[:old_gemv_start] + new_gemv + s[old_gemv_end:]

# ── 2) csr_spmm：同样改池 ──
old_csr_start = s.index("    let chunk = n.div_ceil(nt);\n\n    std::thread::scope(|s| {")
old_csr_end = s.index("/// 单行 SpMV。**升序累加** → 与 Python 逐位一致。")
new_csr = '''    let chunk = n.div_ceil(nt);
    let ctx = CsrCtx {
        csr: csr.clone(),
        x_ptr: x.as_ptr() as usize,
        x_len: x.len(),
        y_ptr: y.as_mut_ptr(),
        n,
    };
    crate::pool::run(csr_spmm_work, &ctx as *const CsrCtx as *mut u8, nt, nt);
}

/// SpMV 的一个分块（只写自己的行块）。
unsafe fn csr_spmm_work(ctx: *mut u8, part: usize, n_parts: usize) {
    let c = unsafe { &*(ctx as *const CsrCtx) };
    let chunk = c.n.div_ceil(n_parts);
    let lo = part * chunk;
    let hi = ((part + 1) * chunk).min(c.n);
    if lo >= hi {
        return;
    }
    let xs = unsafe {
        core::slice::from_raw_parts(c.x_ptr as *const f32, c.x_len)
    };
    let ys = unsafe {
        core::slice::from_raw_parts_mut(c.y_ptr.add(lo), hi - lo)
    };
    for (i, o) in ys.iter_mut().enumerate() {
        *o = csr_row(&c.csr, lo + i, xs);
    }
}

/// SpMV 的线程间上下文。
pub struct CsrCtx {
    csr: Csr,
    x_ptr: usize,
    x_len: usize,
    y_ptr: *mut f32,
    n: usize,
}
// SAFETY: 每个 part 只写 `y[lo..hi)`（连续、互不重叠），CSR 三个数组只读。
unsafe impl Send for CsrCtx {}
unsafe impl Sync for CsrCtx {}

'''
s = s[:old_csr_start] + new_csr + s[old_csr_end:]

# ── 3) m6_sparse_fwd：同样改池 ──
old6_start = s.index("    let chunk = n.div_ceil(nt);\n\n    std::thread::scope(|s| {")
# 找 m6_sparse_fwd 里的那个（第二个出现）
idx2 = s.index("    let chunk = n.div_ceil(nt);\n\n    std::thread::scope(|s| {", old6_start + 10)
# 找到它所属函数的结束（下一个 "/// M6 的稀疏版本" 之后）
old6_end = s.index("}\n", s.index("^", 0)) if False else s.index("#[cfg(test)]", idx2) if s[idx2:].find("#[cfg(test)]") > 0 else len(s)
# 更稳：找到 m6_sparse_fwd 函数的结尾（下一个顶层注释）
marker = "// ══════════════════════════════════════════════════════════════════════════\n// 设备调度辅助"
old6_end = s.index(marker)
new6 = '''    let chunk = n.div_ceil(nt);
    let ctx = M6Ctx {
        w: w.clone(),
        g_ptr: gather_idx.as_ptr() as usize,
        h_ptr: h.as_ptr() as usize,
        h_len: h.len(),
        out_ptr: out.as_mut_ptr(),
        n,
        k,
    };
    crate::pool::run(m6_sparse_work, &ctx as *const M6Ctx as *mut u8, nt, nt);
}

/// M6 稀疏读出的一个分块。
unsafe fn m6_sparse_work(ctx: *mut u8, part: usize, n_parts: usize) {
    let c = unsafe { &*(ctx as *const M6Ctx) };
    let chunk = c.n.div_ceil(n_parts);
    let lo = part * chunk;
    let hi = ((part + 1) * chunk).min(c.n);
    if lo >= hi {
        return;
    }
    let gis = unsafe {
        core::slice::from_raw_parts(c.g_ptr as *const i64, c.k)
    };
    let hs = unsafe {
        core::slice::from_raw_parts(c.h_ptr as *const f32, c.h_len)
    };
    let ys = unsafe {
        core::slice::from_raw_parts_mut(c.out_ptr.add(lo), hi - lo)
    };
    for (i, o) in ys.iter_mut().enumerate() {
        let row = c.w.row(lo + i);
        let mut acc: f32 = 0.0;
        unsafe {
            for j in 0..c.k {
                let col = *gis.get_unchecked(j);
                acc += *row.add(j) * *hs.get_unchecked(col as usize);
            }
        }
        *o = acc;
    }
}

/// M6 稀疏读出的线程间上下文。
pub struct M6Ctx {
    w: F32<'static>,
    g_ptr: usize,
    h_ptr: usize,
    h_len: usize,
    out_ptr: *mut f32,
    n: usize,
    k: usize,
}
// SAFETY: 每个 part 只写 `out[lo..hi)`（连续、互不重叠），其余只读。
unsafe impl Send for M6Ctx {}
unsafe impl Sync for M6Ctx {}

'''
s = s[:old6_start] + new6 + s[old6_end:]

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("mechanisms.rs switched to the resident pool")