//! fp8(e4m3fn) 位模式 → int8 码本的 CPU 转换核（P189）。
//!
//! 背景：910B4 无 fp8 算子但有 int8 算子 →「fp8 位模式存储 + int8 计算」
//! 每步要做一次位数转换（P154）。原实现在 CPU 上走 torch 向量化（18.6ms @1b
//! 档）或 numba nogil 标量（105ms）。本核用 Rust 多核并行：
//!   · 第 1 趟分块求 amax（per-tensor scale = 2·amax/127，P105 口径）
//!   · 第 2 趟分块量化（round half-to-even + clamp ±127）
//! 数值契约与 `phdnet/backends/fp8_int8_convert.py` 逐位一致（门禁把关）：
//!   · e4m3fn：bias=7，subnormal = man/8·2^-6，0xFF/NaN 槽 → 0
//!   · 舍入 = **half-to-even**（Rust `f32::round` 是 half-away-from-zero，
//!     不能直接用——这里手写 RNE）
//!   · scale = 0 时退化 1.0
//!
//! # Safety（extern "C" 约定）
//! `bits`/`codes` 须各有 ≥ `n` 字节；`scale_out` 须指向 1 个 f32。
//! `n_threads` 仅是提示；实际并行度由 `pool::run` 决定。

use crate::pool;

struct ConvCtx {
    bits: *const u8,
    codes: *mut i8,
    n: usize,
    scale: f32,
    /// 分块 amax 的槽位（n_parts 个 f32，由调用方分配）
    partial: *mut f32,
    /// P189 性能修正：趟 2 的 **量化码 LUT**（256 项）——
    /// scale 定了之后每个输入位模式的码唯一确定，趟 2 从
    /// 「值 LUT + f32 除法 + RNE 分支」变成纯字节查表（带宽受限）。
    qlut: *const i8,
}

/// 单个 fp8 e4m3fn 位模式 → f32 值（与 Python `_e4m3_bits_to_f64` 同语义）。
#[inline]
fn e4m3_val(b: u8) -> f32 {
    // P191 修正：NaN 槽 = exp=0b1111 且 man=0b111（**两个符号都有**：
    // 0x7F 与 0xFF）——首版只判了 0xFF，正号 NaN 落进了规格化分支
    // （exp=15 → 2^8×(1+7/8) = 480）。与 Python `_e4m3_bits_to_f64`
    // 的判定对齐（同样查 exp/man 位，不只查字节值）。
    let exp = (b >> 3) & 0x0F;
    let man = b & 0x07;
    if (exp == 0x0F) && (man == 0x07) {
        return 0.0; // NaN 槽位（本项目约定 → 0）
    }
    let sign = (b >> 7) & 0x01;
    let man = man as f32;
    let v = if exp == 0 {
        (man / 8.0) * (2.0f32.powi(-6))
    } else {
        (1.0 + man / 8.0) * (2.0f32.powi(exp as i32 - 7))
    };
    if sign == 1 { -v } else { v }
}

/// **P189 性能修正：256 项 LUT**（与 Python `FP8_VAL_LUT` 同思路）。
/// 首版逐元素走 `powi`+分支（无法向量化）实测 147ms/6.65MiB，
/// 比 torch 向量化（25ms）慢 5.8×——查表把值计算换成一次内存读。
fn val_lut() -> &'static [f32; 256] {
    use core::sync::atomic::{AtomicPtr, Ordering};
    static LUT: AtomicPtr<f32> = AtomicPtr::new(core::ptr::null_mut());
    let p = LUT.load(Ordering::Acquire);
    if !p.is_null() {
        return unsafe { &*(p as *const [f32; 256]) };
    }
    let boxed: Box<[f32; 256]> =
        Box::new(core::array::from_fn(|i| e4m3_val(i as u8)));
    let leak = Box::leak(boxed);
    LUT.store(leak.as_ptr() as *mut f32, Ordering::Release);
    leak
}

/// half-to-even 舍入（与 torch.round / np.rint 逐位一致）。
#[inline]
fn rne(q: f32) -> f32 {
    let f = q.floor();
    let d = q - f;
    if d > 0.5 {
        f + 1.0
    } else if d < 0.5 {
        f
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

unsafe fn conv_work(ctx: *mut u8, part: usize, n_parts: usize) {
    let c = unsafe { &*(ctx as *const ConvCtx) };
    let chunk = c.n.div_ceil(n_parts);
    let lo = part * chunk;
    let hi = ((part + 1) * chunk).min(c.n);
    // 趟 1：本块 amax（scale 未定 → 先只求幅值；走 256 项 LUT）
    if c.scale == 0.0 {
        let lut = val_lut();
        let mut m = 0.0f32;
        for i in lo..hi {
            let v = unsafe { lut[*c.bits.add(i) as usize] };
            let a = if v < 0.0 { -v } else { v };
            if a > m {
                m = a;
            }
        }
        unsafe { *c.partial.add(part) = m };
    } else {
        // 趟 2：量化 = **纯字节查表**（qlut 在入口已按当前 scale 预计算，
        // 每元素只剩 1 次随机读 + 1 次顺序写；语义与逐元素
        // rne(v/scale)+clamp 逐位一致——qlut 的构建走的就是同一套公式）。
        for i in lo..hi {
            let b = unsafe { *c.bits.add(i) };
            unsafe { *c.codes.add(i) = *c.qlut.add(b as usize) };
        }
    }
}

/// fp8 位模式 → int8 码本 + per-tensor scale（两趟，分块并行）。
///
/// # Safety
/// 见模块注释。`partial` 须有 ≥ `n_threads`（或 n_parts）个 f32 槽位。
#[no_mangle]
pub unsafe extern "C" fn phdnet_fp8_to_int8(
    bits: *const u8,
    n: usize,
    codes: *mut i8,
    scale_out: *mut f32,
    partial: *mut f32,
    n_threads: usize,
) -> i32 {
    if n == 0 {
        unsafe { *scale_out = 1.0 };
        return 0;
    }
    let n_parts = crate::parts_for_len(n, n_threads);
    // 趟 1：amax（qlut 在此趟未用 → 哑指针；scale 哨兵 0.0 = 求 amax）
    let mut ctx = ConvCtx {
        bits, codes, n, scale: 0.0, partial,
        qlut: core::ptr::null(),
    };
    pool::run(conv_work, &mut ctx as *mut ConvCtx as *mut u8, n_parts, n_threads);
    let mut amax = 0.0f32;
    for p in 0..n_parts {
        let v = unsafe { *partial.add(p) };
        if v > amax {
            amax = v;
        }
    }
    // ⚠ scale 须在 **f64** 里算再落 f32（与 Python 参照同构）：
    // Python 侧 `2.0*amax/127.0` 是 Python float（fp64）算术，torch 元素除法
    // 时才把 scale cast 回 fp32。Rust 直接在 f32 里 /127 会引入不同的单次
    // 舍入（fp32 一次舍入 vs fp64→fp32 双重舍入），个别元素会差 1 码。
    let scale = if amax > 0.0 {
        ((2.0f64 * amax as f64) / 127.0) as f32
    } else {
        1.0
    };
    // 趟 2 前置：按当前 scale 预计算 **量化码 LUT**（256 项，栈上）。
    // 每个输入位模式的码 = rne(lut[b]/scale) clamp ±127——与逐元素公式
    // 逐位一致（同一套运算，只是提前算好）。
    let lut = val_lut();
    let mut qlut = [0i8; 256];
    for b in 0..256usize {
        let q = rne(lut[b] / scale);
        qlut[b] = if q > 127.0 {
            127
        } else if q < -127.0 {
            -127
        } else {
            q as i8
        };
    }
    // 趟 2：量化（纯字节查表）
    ctx.scale = scale;
    ctx.qlut = qlut.as_ptr();
    pool::run(conv_work, &mut ctx as *mut ConvCtx as *mut u8, n_parts, n_threads);
    unsafe { *scale_out = scale };
    0
}

#[cfg(test)]
mod tests_fp8 {
    use super::*;

    #[test]
    fn rne_matches_bankers() {
        assert_eq!(rne(2.5), 2.0); // tie → even
        assert_eq!(rne(3.5), 4.0); // tie → even
        assert_eq!(rne(1.5), 2.0);
        assert_eq!(rne(2.4), 2.0);
        assert_eq!(rne(2.6), 3.0);
    }

    #[test]
    fn e4m3_known_values() {
        assert_eq!(e4m3_val(0x00), 0.0);
        assert_eq!(e4m3_val(0x38), 1.0); // exp=7,man=0 → 2^0
        assert_eq!(e4m3_val(0xB8), -1.0);
        assert_eq!(e4m3_val(0x08), 2.0f32.powi(-6)); // min normal
        assert_eq!(e4m3_val(0x01), (1.0 / 8.0) * 2.0f32.powi(-6)); // subnormal
        assert_eq!(e4m3_val(0x7E), 448.0); // max finite
        assert_eq!(e4m3_val(0x7F), 0.0); // NaN 槽 → 0
    }
}
