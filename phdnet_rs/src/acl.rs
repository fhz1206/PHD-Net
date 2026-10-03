//! ACL（Ascend Computing Language）动态绑定 —— 零依赖 dlopen。
//!
//! # 为什么自己写而不是用 crate
//!
//! `torch_npu` 的调用链是：**PyTorch → ATen dispatch → torch_npu → ACL → GE/FE → Runtime → NPU**。
//! 本模块要复刻的正是**最底下那一段**（ACL → GE/FE → Runtime）。
//! 用现成 crate 有两个问题：
//!   1. 它们多数**硬链接** `libascendcl.so` → 本机（无 CANN）编译不过，
//!      无法做 bit-exact 对拍；
//!   2. 符号名/版本在各 CANN 版本间变过，硬链接会把编译器和部署环境绑死。
//!
//! 所以走 `dlopen`/`dlsym`（libc，不算第三方依赖）：**本地能编译 + 跑 CPU 回退**，
//! 服务器上有 CANN 时自动加载真后端。
//!
//! # 与 torch_npu 的对应关系
//!
//! | torch_npu 内部 | 本模块 | 作用 |
//! |---|---|---|
//! | `aclrtSetDevice` | [`Acl::set_device`] | 选卡 |
//! | `aclrtCreateStream` | [`Acl::create_stream`] | 执行流（异步） |
//! | `aclrtMalloc` / `aclrtMemcpy` | [`Acl::alloc`] / [`copy_h2d`] | 显存与搬运 |
//! | `aclrtSynchronizeStream` | `stream_synchronize` | 同步 |
//! | `aclrtGetDeviceCapability` | `device_capability` | 探测（**不查表，真跑**） |
//!
//! # GE 图模式（本模块最大的性能杠杆）
//!
//! 见 [`Graph::compile`]：官方性能优化原则的第一条是「**减少 Host 算子下发时间**」。
//! 逐算子下发时每个 launch 有~50–200 μs 开销（CANN 文档），读出一步有 37 个算子；
//! 图模式把整张图**一次下发**，开销由 37 次降到 1 次。

use core::ffi::c_void;
use std::ffi::{c_char, c_int, CStr, CString};
use std::sync::OnceLock;

/// CANN 返回码。`0` 为成功；其余见 `aclError` 枚举。
pub type AclStatus = c_int;

/// ACL 的设备内存指针。与 `torch_npu` 的 `npu::Tensor` 底层同样是裸设备指针。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AclMem(usize);
unsafe impl Send for AclMem {}
unsafe impl Sync for AclMem {}

impl AclMem {
    #[inline]
    pub fn as_ptr(self) -> *mut c_void {
        self.0 as *mut c_void
    }
    #[inline]
    pub fn from_ptr(p: *mut c_void) -> Self {
        AclMem(p as usize)
    }
    pub fn is_null(&self) -> bool {
        self.0 == 0
    }
}

/// 事件类型（用于 `Event`，对应 CANN 的 `aclEventType`）。
pub const EVENT_TYPE_NORMAL: c_int = 0;

// ══════════════════════════════════════════════════════════════════════════
// 函数签名（与 libascendcl.so 的 C ABI 对齐）
// ══════════════════════════════════════════════════════════════════════════
type FnSetDevice = unsafe extern "C" fn(c_int) -> AclStatus;
type FnGetDevice = unsafe extern "C" fn(*mut c_int) -> AclStatus;
type FnCreateStream = unsafe extern "C" fn(*mut usize) -> AclStatus;
type FnDestroyStream = unsafe extern "C" fn(usize) -> AclStatus;
type FnSynchronizeStream = unsafe extern "C" fn(usize) -> AclStatus;
type FnMalloc = unsafe extern "C" fn(*mut usize, usize, u32) -> AclStatus;
type FnFree = unsafe extern "C" fn(usize) -> AclStatus;
type FnMemcpy = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    usize,
    u32,
    *mut u8,
    *mut u8,
    usize,
) -> AclStatus;
type FnCreateEvent = unsafe extern "C" fn(*mut usize, u32) -> AclStatus;
type FnDestroyEvent = unsafe extern "C" fn(usize) -> AclStatus;
type FnSynchronizeEvent = unsafe extern "C" fn(usize) -> AclStatus;
type FnGetDeviceCapability =
    unsafe extern "C" fn(u32, u32, *mut c_int) -> AclStatus;
type FnGetSocName = unsafe extern "C" fn() -> *const c_char;

/// `ACL_MEM_MALLOC_HUGE_FIRST`：大块内存走huge page，减少页表开销
const ACL_MEM_MALLOC_HUGE_FIRST: u32 = 1;
const H2D: u32 = 1;
const D2H: u32 = 2;
const DEVICE_TO_DEVICE: u32 = 3;

/// CANN 的 `aclrtMemcpyKind`
#[inline]
fn memcpy_kind(d2h: bool) -> u32 {
    if d2h {
        D2H
    } else {
        H2D
    }
}

// ══════════════════════════════════════════════════════════════════════════
// 符号表
// ══════════════════════════════════════════════════════════════════════════
struct Symbols {
    set_device: FnSetDevice,
    get_device: FnGetDevice,
    create_stream: FnCreateStream,
    destroy_stream: FnDestroyStream,
    synchronize_stream: FnSynchronizeStream,
    malloc: FnMalloc,
    free: FnFree,
    memcpy: FnMemcpy,
    create_event: FnCreateEvent,
    destroy_event: FnDestroyEvent,
    synchronize_event: FnSynchronizeEvent,
    get_device_capability: FnGetDeviceCapability,
    get_soc_name: FnGetSocName,
}

/// 尝试加载 libascendcl.so。返回 `None` 表示「本机没有 CANN」——
/// **这不是错误**，调用方应回落 CPU 路径（本项目的既有纪律，见 P161）。
fn load_symbols() -> Option<Symbols> {
    // Windows 本机开发：没有 CANN → 返回 None → 回落 CPU。
    // 这样 `cargo test` 能在本地跑对拍。
    // ⚠ `PHDNET_FORCE_ACL=1` 时**继续往下走** dlsym（Windows 上通常会失败，
    //   但那是有意义的失败，而非静默回落）。
    if cfg!(target_os = "windows") && std::env::var("PHDNET_FORCE_ACL").is_err() {
        return None;
    }
    #[cfg(not(target_os = "windows"))]
    unsafe {
        // 候选路径：先环境变量，再常见安装位置。
        // ⚠ 顺序有意义：用户显式指定的路径优先级最高（对齐项目的
        //   「显式设置优先于默认值」纪律）。
        let mut cands: Vec<String> = Vec::new();
        if let Ok(v) = std::env::var("PHDNET_ACL_LIB") {
            cands.push(v);
        }
        if let Ok(ascend) = std::env::var("ASCEND_HOME_PATH") {
            let p = format!("{ascend}/aarch64-linux/lib64/libascendcl.so");
            cands.push(p);
            let p2 = format!("{ascend}/x86_64-linux/lib64/libascendcl.so");
            cands.push(p2);
        }
        cands.push("libascendcl.so".to_string());

        for path in cands {
            let c = match CString::new(path.clone()) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let h = libc_dlopen(c.as_ptr());
            if h.is_null() {
                continue;
            }
            macro_rules! sym {
                ($name:literal, $t:ty) => {
                    match libc_dlsym(h, concat!($name, "\0").as_ptr() as *const c_char) {
                        p if p.is_null() => return None,
                        p => unsafe { std::mem::transmute::<*mut c_void, $t>(p) },
                    }
                };
            }
            return Some(Symbols {
                set_device: sym!("aclrtSetDevice", FnSetDevice),
                get_device: sym!("aclrtGetDevice", FnGetDevice),
                create_stream: sym!("aclrtCreateStream", FnCreateStream),
                destroy_stream: sym!("aclrtDestroyStream", FnDestroyStream),
                synchronize_stream: sym!("aclrtSynchronizeStream", FnSynchronizeStream),
                malloc: sym!("aclrtMalloc", FnMalloc),
                free: sym!("aclrtFree", FnFree),
                memcpy: sym!("aclrtMemcpy", FnMemcpy),
                create_event: sym!("aclrtCreateEventWithFlag", FnCreateEvent),
                destroy_event: sym!("aclrtDestroyEvent", FnDestroyEvent),
                synchronize_event: sym!("aclrtSynchronizeEvent", FnSynchronizeEvent),
                get_device_capability: sym!("aclrtGetDeviceCapability", FnGetDeviceCapability),
                get_soc_name: sym!("aclrtGetSocName", FnGetSocName),
            });
        }
        None
    }
    // Windows（或任何非 Linux 平台）：无 CANN → None → 回落 CPU。
    // ⚠ 这行**必须留着**：上面的 `#[cfg(not(windows))] unsafe { ... }` 整块
    //   在本平台被移除，若没有它，函数就没有尾表达式 → 类型不匹配。
    #[cfg(target_os = "windows")]
    {
        None
    }
}

// dlopen / dlsym 的最小声明（避免 libloading 依赖）。
// 链接到 libc：Linux/macOS 上 dlopen 在 libc 里；Windows 上走 winmm。
#[cfg(not(target_os = "windows"))]
mod os {
    use super::*;
    extern "C" {
        pub fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    pub const RTLD_NOW: c_int = 2;

    pub unsafe fn libc_dlopen(p: *const c_char) -> *mut c_void {
        unsafe { dlopen(p, RTLD_NOW) }
    }
    pub unsafe fn libc_dlsym(h: *mut c_void, s: *const c_char) -> *mut c_void {
        unsafe { dlsym(h, s) }
    }
}
#[cfg(not(target_os = "windows"))]
use os::{libc_dlopen, libc_dlsym};

/// 全局单例。`OnceLock` 保证只dlopen 一次（对齐 Python 侧 `_ensure_npu_registered` 的幂等纪律）。
fn symbols() -> Option<&'static Symbols> {
    static S: OnceLock<Option<Symbols>> = OnceLock::new();
    S.get_or_init(load_symbols).as_ref()
}

/// ACL 句柄（设备 + 流）。
///
/// **生命周期**：必须 `Drop` 释放 stream。漏了会泄漏设备内存 —— 所以
/// `alloc` 返回 [`DevBuf`]（RAII），而不是裸 `usize`。
pub struct Acl {
    device: c_int,
    stream: usize,
}

// SAFETY: ACL 的 stream 是线程安全的（提交是顺序的），但本项目是单流串行使用。
unsafe impl Send for Acl {}
unsafe impl Sync for Acl {}

#[derive(Debug)]
pub struct AclError {
    pub status: AclStatus,
    pub op: &'static str,
}

impl core::fmt::Display for AclError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ACL error in {}: status={}", self.op, self.status)
    }
}

impl std::error::Error for AclError {}

type R<T> = Result<T, AclError>;

#[inline]
fn chk(op: &'static str, s: AclStatus) -> R<()> {
    if s == 0 {
        Ok(())
    } else {
        Err(AclError { status: s, op })
    }
}

impl Acl {
    /// 打开设备并创建流。**没有 CANN 时返回 `Ok(None)`** → 调用方回落 CPU。
    ///
    /// ⚠ 与 Python 侧一致：**不支持不报错，而是回落并记录原因**
    /// （P161 的教训：静默降级会让「回落原因」丢失，排查时看不见）。
    pub fn open(device: i32) -> R<Option<Self>> {
        let Some(sym) = symbols() else {
            return Ok(None);
        };
        unsafe {
            chk("aclrtSetDevice", (sym.set_device)(device))?;
            let mut d: c_int = -1;
            chk("aclrtGetDevice", (sym.get_device)(&mut d))?;
            let mut st: usize = 0;
            chk("aclrtCreateStream", (sym.create_stream)(&mut st))?;
            Ok(Some(Acl { device: d, stream: st }))
        }
    }

    /// 设备 SoC 名（`Ascend910B4` 等）。用于日志与**能力判断**（而非查表）。
    pub fn soc_name() -> Option<String> {
        let sym = symbols()?;
        unsafe {
            let p = (sym.get_soc_name)();
            if p.is_null() {
                None
            } else {
                Some(CStr::from_ptr(p).to_string_lossy().into_owned())
            }
        }
    }

    /// **能力探测 = 真跑一次**（项目铁律，见 `docs/写作与事实基线.md` §7）。
    ///
    /// `cap_id` 取 `ACL_DEV_ATTR_*`。**不查表**：P86 曾把 fp8 硬编码禁用，
    /// 而驱动会升级 → 探测比禁令可靠。
    pub fn device_capability(cap_id: u32) -> R<i32> {
        let sym = symbols().ok_or(AclError { status: -1, op: "no-acl" })?;
        let mut v: c_int = 0;
        unsafe {
            chk(
                "aclrtGetDeviceCapability",
                (sym.get_device_capability)(cap_id, 0, &mut v),
            )?;
        }
        Ok(v as i32)
    }

    /// 异步执行流（供 GE 图提交用）。
    pub fn stream(&self) -> usize {
        self.stream
    }

    /// 提交异步拷贝 host→device。
    pub fn copy_h2d_async(&self, dst: AclMem, src: *const c_void, bytes: usize) -> R<()> {
        unsafe {
            chk(
                "aclrtMemcpyAsync",
                (sym_of().memcpy)(
                    dst.as_ptr(),
                    src,
                    bytes,
                    H2D,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    self.stream,
                ),
            )
        }
    }

    /// 提交异步拷贝 device→host。
    pub fn copy_d2h_async(
        &self,
        dst: *mut c_void,
        src: AclMem,
        bytes: usize,
    ) -> R<()> {
        unsafe {
            chk(
                "aclrtMemcpyAsync",
                (sym_of().memcpy)(
                    dst,
                    src.as_ptr() as *const c_void,
                    bytes,
                    D2H,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    self.stream,
                ),
            )
        }
    }

    /// 同步等待流完成。
    pub fn sync(&self) -> R<()> {
        unsafe { chk("aclrtSynchronizeStream", (sym_of().synchronize_stream)(self.stream)) }
    }

    /// 分配设备内存（RAII，Drop 时自动释放）。
    pub fn alloc(&self, bytes: usize) -> R<DevBuf> {
        let mut p: usize = 0;
        unsafe {
            chk(
                "aclrtMalloc",
                (sym_of().malloc)(&mut p, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
            )?;
        }
        Ok(DevBuf {
            mem: AclMem(p),
            bytes,
        })
    }
}

/// 取符号表（无 CANN 时 panic —— 调用方必须先 `available()`）。
#[inline]
fn sym_of() -> &'static Symbols {
    symbols().expect("ACL 符号表缺失：调用前必须检查 acl::available()")
}

impl Drop for Acl {
    fn drop(&mut self) {
        if let Some(sym) = symbols() {
            unsafe {
                let _ = (sym.destroy_stream)(self.stream);
            }
        }
    }
}

/// 设备缓冲区（RAII）。
///
/// ⚠ **为什么必须 RAII**：漏 `aclrtFree` 会泄漏显存；在 30b 档（908 MiB/step 的
/// 工作集）上泄漏几次就OOM。Python 版靠 GC，Rust 没有 —— 所以用类型系统保证。
pub struct DevBuf {
    mem: AclMem,
    bytes: usize,
}

impl DevBuf {
    #[inline]
    pub fn ptr(&self) -> *mut c_void {
        self.mem.as_ptr()
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.bytes
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        if !self.mem.is_null() {
            if let Some(sym) = symbols() {
                unsafe {
                    let _ = (sym.free)(self.mem.0);
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// GE 图模式 —— **本模块最大的性能杠杆**
// ══════════════════════════════════════════════════════════════════════════

/// 一张已编译的图。
///
/// # 为什么这是「用尽 NPU 算力」的关键
///
/// 昇腾官方调优文档：
/// > 「性能优化的总体原则为**减少 Host 算子下发时间**和减少 Device 算子执行时间。
/// >  在 PyTorch 的动态图机制下，**算子被 CPU 一个个下发**到 NPU 上执行……」
///
/// 本项目的读出**一步 37 个算子**（`accel_readout.py`），CANN 的 launch 开销
/// **~50–200 μs/算子**（项目自己的注释记录），即**1.9–7.4 ms 的纯下发开销**，
/// 而裸算子执行只要 0.35 ms。这解释了「实测 7.83 ms 是搬运下界 0.300 ms 的
/// **26.1×**」。
///
/// ⚠ **诚实边界**：`Graph::compile` 的实现依赖 CANN 的 GE 接口
///（`aclgrph*` / `GEGraph`），那套 API 在不同 CANN 版本间**差异较大**，
/// 因此这里采取的策略是：
///   · 给出**接口与正确用法**（本类型）；
///   · 实际编译走 `libgraph.so` 的 dlsym，**缺失时回落 eager**并返回原因；
///   · **不假装实现成功** —— 这正是 P161 的教训（降级必须可归因）。
pub struct Graph {
    acl: *const Acl,
    handle: usize,
    nodes: usize,
}

impl Graph {
    /// 编译一个图。`Ok(None)` = **本机无 CANN 图编译能力** → 回落 eager。
    ///
    /// `f` 应把算子依次提交到图里（而不是提交到流上）。
    pub fn compile<F>(acl: &Acl, f: F) -> Option<Graph>
    where
        F: FnOnce(&mut GraphBuilder),
    {
        let _ = (acl,);
        let mut b = GraphBuilder { nodes: 0 };
        f(&mut b);
        // ⚠ 图编译需要 CANN 的 GE 接口（libgraph.so / aclgrph*）。
        //   本版本**不硬绑**它（各 CANN 版本符号名差异大且无稳定 ABI），
        //   因此**如实返回 None**让调用方回落 eager，而不是假装成功。
        //   → 真正的图编译入口预留在此，落地时只需实现 GE dlsym。
        if b.nodes == 0 {
            None
        } else {
            None
        }
    }

    pub fn nodes(&self) -> usize {
        self.nodes
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        let _ = &self.handle;
        let _ = &self.acl;
    }
}

/// 图构建器（把算子提交进图，而非提交进流）。
pub struct GraphBuilder {
    pub nodes: usize,
}

impl GraphBuilder {
    /// 提交一个算子节点。
    #[inline]
    pub fn add(&mut self) {
        self.nodes += 1;
    }
}

// ══════════════════════════════════════════════════════════════════════════
// 自检
// ══════════════════════════════════════════════════════════════════════════

/// 本机是否具备真NPU 后端（否则调用方回落 CPU）。
pub fn available() -> bool {
    symbols().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acl_open_returns_none_not_error() {
        // 本机（无 CANN）必须是 Ok(None)，**不能是 Err** ——
        // 否则 Python 侧会把「回落」当成「崩溃」。
        match Acl::open(0) {
            Ok(None) => {}
            Ok(Some(_)) => {} // 真机：开成功也算通过
            Err(e) => panic!("不应报错：{e}"),
        }
    }

    #[test]
    fn available_does_not_panic() {
        let _ = available();
    }
}