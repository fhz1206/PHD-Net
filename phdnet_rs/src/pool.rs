//! 常驻线程池 —— 修 P171 实测发现的「每步新建线程」开销。
//!
//! # 问题（P171 本机实测）
//!
//! `std::thread::scope` **每次调用都创建 + 销毁线程**。本项目每 token 至少
//! 调用 9 次（M1 一次 + M2 八次）→ 每步建 8 线程再销毁。
//! 实测 **M1 GEMV**：Rust 8 线程 1096 µs **vs** numba `prange` 599 µs
//! —— **慢 1.8×**，而两者都是标量循环，差距**全部来自线程管理**。
//!
//! numba 的 `prange` 走 **OpenMP 常驻线程池**，所以没有这部分开销。
//!
//! # 修法
//!
//! 一次性建好 N 个 worker，**永不退出**。每步用 mpsc channel 派发
//! `(fn, ctx, part, n_parts)`，worker 执行后回一个信号 ——
//! **派发开销是微秒级**（对比新建线程的数十微秒）。
//!
//! # ⚠ 与 GIL 的关系（这是本模块存在的**根本理由**）
//!
//! Python 通过 **`ctypes.CDLL`**（不是 `PyDLL`）调用本库 —— CDLL **会释放 GIL**，
//! 所以下面的 worker 是**真并行**的。
//! ⚠ 若改用 `PyDLL`，GIL 不释放，线程池退化为串行 ——**那还不如直接用 numba**。
//!
//! # 诚实边界
//!
//! · 常驻线程占住 N 个核的栈（1 MiB/线程 = 8 MiB）。训练进程本已常驻，可接受。
//! · 上限 8 线程：读出是**带宽受限**（MEMORY「CPU 侧已饱和」），
//!   再多线程只加剧内存争抢。实测 16 线程反而更慢（1562 vs 1096 µs）。
//! · `PHDNET_RS_POOL=0` 关闭（回落 `thread::scope` 的旧路径），用于 A/B。

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::OnceLock;

/// 工作函数：`unsafe fn(ctx, part, n_parts)` —— 只跑**第 part 块**。
///
/// ⚠ **用 fn 指针而非泛型**：泛型会 monomorphize 出 N 份，池就失去「一份代码」。
type WorkFn = unsafe fn(*mut u8, usize, usize);

struct Shared {
    /// 每步的同一个任务（所有 worker 收到同一个 (f, ctx, n_parts)，
    /// 靠 part 编号自行分工）→ **零堆分配**。
    /// ⚠ `ctx` 存为 `usize` 而非 `*mut u8`：裸指针不是 `Send`，
    ///   而全局池要求 `Sync`。整数是。转换在使用处完成（见 worker 循环）。
    task: std::sync::Mutex<Option<(WorkFn, usize, usize)>>,
    n_threads: usize,
}

struct Worker {
    tx: Sender<usize>,// → worker：派发 part 编号
    /// ← worker：完成信号。**必须包 Mutex**：`mpsc::Receiver` 不是 `Sync`，
    /// 而池是全局 `OnceLock`（要求 `Sync`）。用 `Mutex` 的开销只在
    /// 「派发 + 等待」各一次，可忽略。
    rx: std::sync::Mutex<Receiver<bool>>,
}

pub struct Pool {
    shared: std::sync::Arc<Shared>,
    workers: Vec<Worker>,
    _handles: Vec<std::thread::JoinHandle<()>>,
}

static POOL: OnceLock<Pool> = OnceLock::new();

/// 池的**常驻 worker 数**上限。
///
/// ⚠ **P174 审计**：旧实现按「第一次调用时的 threads」建池且永不改变，
/// 导致 A/B 测量失真（见 `get` 的注释）。现在统一按此上限建，
/// 多出的 worker 阻塞在 `recv()` 上空闲 —— 池略大无害，
/// **测量失真才是真问题**。
pub const MAX_POOL_THREADS: usize = 8;

/// 池是否启用（`PHDNET_RS_POOL=0` 关闭，用于 A/B 与排障）。
pub fn pool_enabled() -> bool {
    match std::env::var("PHDNET_RS_POOL") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        Err(_) => true,
    }
}

impl Pool {
    /// 取全局池（懒建）。`threads == 0` →按 CPU 核取 `min(8, 核-1)`。
    /// 返回 `None` 表示「不用池」（线程数 ≤ 1 或被环境变量禁用）。
    pub fn get(threads: usize) -> Option<&'static Pool> {
        if !pool_enabled() {
            return None;
        }
        let n = if threads > 0 {
            threads
        } else {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
                .saturating_sub(1)
                .clamp(1, 8)
        };
        if n <= 1 {
            return None;
        }
        // ⚠⚠ **P174 审计修复①：池必须按固定上限建，且**尊重**调用方的 n。
        //   旧代码 `POOL.get_or_init(|| Pool::new(n))` 只在**第一次**调用时
        //   用当时的 n 建池，之后 `threads` 参数**被完全忽略** ——
        //   于是「1 线程」与「8 线程」两次调用**用的是同一个池**，
        //   A/B 测量完全失真（本机实测：768x1024 fp32，1 线程 78.9 µs
        //   而 2 线程 174.3 µs —— **非单调**，2 线程反而慢 2.2×，
        //   因为「1 线程」那次其实也派发到了池）。
        //   现在：池按 `MAX_POOL_THREADS` 建（worker 全部常驻），
        //   `run` 里按**实际 n** 派发 —— 池大一点没关系，
        //   多出的 worker 阻塞在 `recv()` 上不做事。
        Some(POOL.get_or_init(|| Pool::new(MAX_POOL_THREADS)))
    }

    /// 本次实际可用的 worker 数（**≤ 池的 worker 数**）。
    #[inline]
    pub fn usable(&self, want: usize) -> usize {
        want.clamp(1, self.workers.len())
    }

    fn new(n: usize) -> Pool {
        let shared = std::sync::Arc::new(Shared {
            task: std::sync::Mutex::new(None),
            n_threads: n,
        });
        let mut workers = Vec::with_capacity(n);
        let mut handles = Vec::with_capacity(n);
        for _ in 0..n {
            let (tx_w, rx_w) = channel::<usize>();   //主 → worker
            let (tx_m, rx_m) = channel::<bool>();   // worker → 主
            let sh = std::sync::Arc::clone(&shared);
            let handle = std::thread::Builder::new()
                .name("phdnet-rs-worker".into())
                .stack_size(1 << 20)     // 1 MiB：本项目的任务栈很浅
                .spawn(move || loop {
                    // 阻塞等一个 part 编号（`None` =池要销毁）
                    let Ok(part) = rx_w.recv() else { return };
                    let task = sh.task.lock().ok().and_then(|mut t| *t);
                    if let Some((f, ctx, n_parts)) = task {
                        unsafe { f(ctx as *mut u8, part, n_parts) };
                    }
                    // 完成信号
                    if tx_m.send(true).is_err() {
                        return;
                    }
                })
                .expect("spawn phdnet-rs worker");
            workers.push(Worker {
                tx: tx_w,
                rx: std::sync::Mutex::new(rx_m),
            });
            handles.push(handle);
        }
        Pool { shared, workers, _handles: handles }
    }

    /// 并行执行 `f(ctx, part, n_parts)`，`n_parts` 块。
    ///
    /// # 安全性
    /// `f` **必须**只写 `part` 对应的行块，**各part 互不重叠**。
    /// 调用方（本项目唯一入口 `parallel_rows`）通过连续分块保证这一点。
    pub fn run(&self, f: WorkFn, ctx: *mut u8, n_parts: usize) {
        let n = n_parts.clamp(1, self.workers.len());
        if n <= 1 {
            unsafe { f(ctx, 0, 1) };
            return;
        }
        {
            let mut t = self.shared.task.lock().expect("task mutex poisoned");
            *t = Some((f, ctx as usize, n));
        }
        // 派发 part 1..n-1（part 0 由当前线程做 → 省一次唤醒与同步）
        for (i, w) in self.workers.iter().enumerate().take(n - 1) {
            let _ = w.tx.send(i + 1);
        }
        // 当前线程跑 part 0
        unsafe { f(ctx, 0, n) };
        // 等所有 worker 完成
        for w in self.workers.iter().take(n - 1) {
            let _ = w.rx.lock().expect("rx mutex").recv();
        }
    }

    pub fn threads(&self) -> usize {
        self.workers.len()
    }
}

/// 便捷入口：拿池并执行；无池时退回单线程。
#[inline]
pub fn run(f: WorkFn, ctx: *mut u8, n_parts: usize, threads: usize) {
    match Pool::get(threads) {
        Some(p) => p.run(f, ctx, n_parts),
        None => unsafe { f(ctx, 0, 1) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CNT: AtomicUsize = AtomicUsize::new(0);

    unsafe fn work(ctx: *mut u8, part: usize, n_parts: usize) {
        let n = ctx as usize;              // 上下文 = 元素总数（伪装在指针里）
        let chunk = n.div_ceil(n_parts);
        let lo = part * chunk;
        let hi = ((part + 1) * chunk).min(n);
        for i in lo..hi {
            CNT.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// ⚠ **这些测试必须串行**（`--test-threads=1`）。
    /// 原因：它们**共享全局池**（`POOL` OnceLock）与**共享计数器** `CNT`。
    /// Rust 的测试默认多线程并发 → 两个测试同时跑会互相污染 `CNT`。
    /// ⚠ 真实代码里 `POOL.run` 是**同步的**（派发→ 等完成），所以生产用是安全的；
    ///只有「测试并发调用 run」才会出这个问题。
    #[test]
    fn pool_does_same_amount_of_work() {
        let n = 10_000usize;
        CNT.store(0, Ordering::Relaxed);
        run(work, n as *mut u8, 8, 8);
        assert_eq!(CNT.load(Ordering::Relaxed), n);
    }

    #[test]
    fn pool_is_reusable() {
        let n = 5_000usize;
        for _ in 0..3 {
            CNT.store(0, Ordering::Relaxed);
            run(work, n as *mut u8, 4, 8);
            assert_eq!(CNT.load(Ordering::Relaxed), n);
        }
    }

    #[test]
    fn single_thread_path_is_correct() {
        let n = 3_000usize;
        CNT.store(0, Ordering::Relaxed);
        run(work, n as *mut u8, 1, 8);
        assert_eq!(CNT.load(Ordering::Relaxed), n);
    }
}