"""修Rust 编译错误：Send/Sync + 类型。"""
import io

P = "phdnet_rs/src/mechanisms.rs"
s = io.open(P, encoding="utf-8").read()

# ── 1. F32 / Csr 加 Send+Sync（裸指针跨线程的正当理由：我们保证分块不重叠）──
s = s.replace(
    """#[derive(Clone, Copy)]
pub struct F32<'a> {
    pub ptr: *mut f32,
    pub rows: usize,
    pub cols: usize,
}""",
    """#[derive(Clone, Copy)]
pub struct F32<'a> {
    pub ptr: *mut f32,
    pub rows: usize,
    pub cols: usize,
    /// ⚠ 只为满足`thread::scope` 的 `Send` 要求而存在。
    ///   **安全性由调用方保证**：各线程只访问 `[row_lo, row_hi)` 的行，
    ///   互不重叠；只读共享部分不写。违反即数据竞态（UB）。
    pub _marker: core::marker::PhantomData<&'a [f32]>,
}
// SAFETY: 见 `_marker` 的说明。分块写 `out[row_lo..row_hi]` 互不重叠。
unsafe impl<'a> Send for F32<'a> {}
unsafe impl<'a> Sync for F32<'a> {}""")

s = s.replace(
    """    pub unsafe fn from_raw(ptr: *mut f32, rows: usize, cols: usize) -> Self {
        F32 { ptr, rows, cols }
    }""",
    """    pub unsafe fn from_raw(ptr: *mut f32, rows: usize, cols: usize) -> Self {
        F32 {
            ptr,
            rows,
            cols,
            _marker: core::marker::PhantomData,
        }
    }""")

s = s.replace(
    """        F32 {
            ptr: b.ptr() as *mut f32,
            rows,
            cols,
        }
    }""",
    """        F32 {
            ptr: b.ptr() as *mut f32,
            rows,
            cols,
            _marker: core::marker::PhantomData,
        }
    }""")

# ── 2. Csr 加 Send+Sync ──
s = s.replace(
    """#[derive(Clone, Copy)]
pub struct Csr {
    pub indptr: *const i64,
    pub idx: *const i64,
    pub val: *const f32,
    pub n_rows: usize,
}""",
    """#[derive(Clone, Copy)]
pub struct Csr {
    pub indptr: *const i64,
    pub idx: *const i64,
    pub val: *const f32,
    pub n_rows: usize,
}
// SAFETY: CSR 的三个数组在 SpMV 期间**只读**，多线程共享读是安全的
//   （真正的竞态只可能来自 `y` 写冲突，而那由分块保证不重叠）。
unsafe impl Send for Csr {}
unsafe impl Sync for Csr {}""")

# ── 3. 删掉 m1_kwta 里的死代码（占位数组）──
s = s.replace(
    """    // 部分选择：维护大小为 k 的最小堆？——为**避免全排序**（O(n log k) 而非 O(n log n)）
    // 但 heap 要注意 NaN。简单起见用「插入排序维护 top-k」，n=1024、k=128 时
    // 最坏 131k 次比较，远小于全排序，且**无需分配**。
    let mut top: [u32; 0] = []; // 占位，真正的实现用 SmallVec 风格的固定数组
    let _ = &mut top;

    // 用 (value, index) 的简单 top-k：先全排序索引（n log n，n=1024 时 ~10k 次）""",
    """    // top-k：先对索引按「值降序、值同则小下标在前」全排序，再取前 k。
    // ⚠ n=1024、k=128 时约 10k 次比较 —— 诚实：**这不是最快的实现**
    //   （无序数组+ 插入是 O(n·k/2) 但常数更小）。
    //   门禁只锁「值集合与 Python 的 argpartition 一致」，不锁速度。""")

# ── 4. dev_to_host 的类型链错误 → 改成直接返回 Result ──
s = s.replace(
    """/// 把设备缓冲区同步回 host。
pub fn dev_to_host(acl: &Acl, src: &DevBuf, dst: &mut [f32]) -> crate::acl::AclStatusChain {
    acl.copy_d2h_async(
        dst.as_mut_ptr() as *mut core::ffi::c_void,
        crate::acl::AclMem::from_ptr(src.ptr()),
        src.len(),
    )
    .and_then(|()| acl.sync())
    .into()
}

/// 状态链（`Result<(), AclError>` → 无错的哨兵，便于 FFI 边界返回单个整数）。
pub trait AclStatusChain {
    fn into(self) -> i32;
}

impl AclStatusChain for Result<(), crate::acl::AclError> {
    fn into(self) -> i32 {
        match self {
            Ok(()) => 0,
            Err(e) => e.status,
        }
    }
}""",
    """/// 把设备缓冲区同步回 host（**同步**语义：返回前数据已在 `dst`）。
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
}""")

# ── 5. "if may be missing else" —— m1_kwta 的 idx 赋值改成块 ──
s = s.replace(
    """    for &i in sel {
        let v = u[i as usize];
        s_out[i as usize] = (v - win_min) / (win_max - win_min + 1e-9) + 0.1;
        idx_out[sel
            .iter()
            .position(|&j| j == i)
            .unwrap_or(0)] = i;
    }""",
    """    for (pos, &i) in sel.iter().enumerate() {
        let v = u[i as usize];
        s_out[i as usize] = (v - win_min) / (win_max - win_min + 1e-9) + 0.1;
        idx_out[pos] = i;
    }""")

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("patched mechanisms.rs")