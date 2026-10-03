//! 精度分派 —— P172：**Rust 跟着 Python 的生产精度走**。
//!
//! # 为什么要这个（P172 定位到的真实不一致）
//!
//! | 机制 | Python 生产精度 | 改前 Rust | 一致？ |
//! |---|---|---|---|
//! | M1 编码器 | `fp32`（`encoder_dtype`） | `f32` | ✅ |
//! | M2 CSR 主干 | **`fp64`**（`sparse_pc.py:175`） | `f32` | ❌ |
//! | M3 STDP | **`fp64`**（`plasticity.py:44`） | `f32` | ❌ |
//! | M4a 工作记忆 | **`fp64`**（`wm.py:13`） | `f32` | ❌ |
//! | M5 调制 | 标量 `f64` | `f32` | ⚠ |
//! | M6 读出 | **`fp16`**（P163 默认） | `f32` | ❌ |
//!
//! ⚠ **M6 那条最危险**：P110 实测 fp16 的非目标行更新保留率只有 **26.67%**
//! （学习规则部分退化为纯 Hebbian），而 `f32` 是全保留
//! → **Rust 版跑出的 PPL 会比 Python 版好看，但「不是同一个模型」**。
//!
//! # 做法：**泛型 + 单态化**（Rust 的零成本抽象）
//!
//! - 每个核泛型化：`<T: Acc> fn gemv(..., x: &[T], ...)`
//! - `trait Acc` 提供 `zero()` / `add()` / `mul()`，**由编译器内联掉**
//!   → 运行时零开销，仍然是标量代码（**SIMD 另走 `simd.rs` 的 f32 特化**）。
//! - Python 侧按 `str(dtype)` 选具体类型 → fhz 要的「跟着生产精度走」。
//!
//! # ⚠ 诚实边界
//!
//! · **SIMD 只有 f32 特化**（AVX2 的 `_mm256_fmadd_ps` 是 f32 指令）。
//!   f64 走标量；**f16 没有真 SIMD**（`_mm256_fmadd_ph` 只在 AVX512-FP16 上有，
//!   910B 是 ARM → 更没有）→ f16 用 `f32` 计算再舍入到 f16（**两步**）。
//! · 单态化会**增加代码体积**（每个 dtype 一份）—— 对本项目可忽略。

use core::ops::{Add, Mul};

/// 浮点标量的最小抽象。**只为了泛型化，不是为了抽象层次**。
pub trait Acc:
    Copy
    + Mul<Output = Self>
    + Add<Output = Self>
    + Default
{
    fn zero() -> Self {
        Self::default()
    }
}

impl Acc for f32 {}
impl Acc for f64 {}

/// f16 的软件表示（**只为对齐 Python 的 fp16 语义**）。
///
/// ⚠ 这是**半精度容器**（IEEE 754 binary16 的位模式），不是「f16 算术」。
/// 真正的算术用 f32 做再舍入 —— 见 [`to_f16_bits`] / [`from_f16_bits`]。
#[derive(Clone, Copy)]
pub struct F16(pub u16);

impl Default for F16 {
    fn default() -> Self {
        F16(0)                    // +0.0 的位模式
    }
}

impl Mul for F16 {
    type Output = F16;
    /// ⚠ **半精度乘法在 f32 域做再舍入**（软件实现，无硬件支持）。
    fn mul(self, o: F16) -> F16 {
        from_f32_bits_bits(to_f32_bits_bits(self) * to_f32_bits_bits(o))
    }
}
impl Add for F16 {
    type Output = F16;
    fn add(self, o: F16) -> F16 {
        from_f32_bits_bits(to_f32_bits_bits(self) + to_f32_bits_bits(o))
    }
}

#[inline]
fn to_f32_bits_bits(h: F16) -> f32 {
    f16_to_f32(h.0)
}
#[inline]
fn from_f32_bits_bits(v: f32) -> F16 {
    F16(f32_to_f16(v))
}

/// IEEE 754 binary16 → f32（**精确**，half 的所有值都可被 f32 表示）。
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = (sign << 31) | if exp == 0 {
        // subnormal 或 zero
        if man == 0 {
            sign << 31
        } else {
            // 归一化：half 的 subnormal 1.0×2^-24 .. 1.0×2^-14
            let shift = man.leading_zeros();          // 0..=9
            let e = 127 - 15 - shift;
            let m = (man << (shift + 1)) & 0x7ff_ffff;
            (sign << 31) | (e << 23) | m
        }
    } else if exp == 0x1f {
        // inf / NaN
        (sign << 31) | (0xff << 23) | (man << 13)
    } else {
        // normal：指数重映射 15→127
        ((sign) << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// f32 → IEEE 754 binary16（**就近舍入到偶数**，与 numpy 的 `.astype(np.float16)` 一致）。
///
/// ⚠ 溢出（|v| > 65504）→ 返回 ±inf（与 numpy 相同）。
/// ⚨ **NaN → NaN**（保持quiet）。
#[inline]
pub fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mut exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        // inf / NaN
        return if man != 0 { sign | 0x7e00 } else { sign | 0x7c00 };
    }
    exp = exp - 127 + 15;
    if exp >= 0x1f {
        return sign | 0x7c00;    // 溢出 → inf
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;           // 下溢 → ±0
        }
        // subnormal：加隐含位再右移
        let m = man | 0x80_0000;
        let shift = (14 - exp) as u32;
        let mut h = (m >> shift) as u16;
        // 就近舍入到偶数
        let round_bit = 1 << (shift - 1);
        if (m & round_bit) != 0 && ((m & (round_bit - 1)) != 0 || (h & 1) != 0) {
            h += 1;
        }
        return sign | h;
    }
    // normal：加隐含位、舍入
    let mut h = ((exp as u16) << 10) | ((man >> 13) as u16);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) != 0) {
        h += 1;                     // 可能进位到 exp（甚至 inf）—— 正确
    }
    sign | h
}

/// Python 的 dtype 名 → 本 crate 的枚举（**分派的唯一入口**）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    F16,
}

impl DType {
    /// 解析 Python 的 dtype 名。**不认识的报错**（不静默回落 —— P161 纪律）。
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "fp32" | "float32" | "f32" | "32" => Ok(DType::F32),
            "fp64" | "float64" | "f64" | "64" => Ok(DType::F64),
            "fp16" | "float16" | "f16" | "16" => Ok(DType::F16),
            other => Err(format!(
                "Rust 算子库不支持 dtype={other:?}；可用：fp32 / fp16 / fp64"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "fp32",
            DType::F64 => "fp64",
            DType::F16 => "fp16",
        }
    }

    /// 该 dtype 是否有**手写 SIMD** 路径。
    ///
    /// ⚠ **只有 f32 有**（AVX2 `_mm256_fmadd_ps` 是 f32 指令）。
    ///   · f64 → 无 f64 SIMD（FMA 是 f32 的）
    ///   · f16 → **910B 是 ARM，连 fp16 SIMD 都没有**；x86 的
    ///     `_mm256_fmadd_ph` 也要 AVX512-FP16 → 普遍没有。
    ///   → 非 f32 一律**标量**（正确但慢）。
    pub fn has_simd(self) -> bool {
        matches!(self, DType::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_exact_for_normal_values() {
        // half 能精确表示的 f32 → 往返应**逐位**一致
        for bits in [0x3f800000u32, 0x40000000, 0x3f000000, 0xbf800000] {
            let v = f32::from_bits(bits);
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert_eq!(v.to_bits(), back.to_bits(), "bits={bits:#x}");
        }
    }

    #[test]
    fn f16_matches_numpy_rne() {
        // 已知边界：numpy 的 astype(float16) 是 RNE
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);       // 最大 finite
        assert_eq!(f32_to_f16(65520.0), 0x7c00);       // 溢出 → inf
        assert_eq!(f32_to_f16(0.0), 0x0000);
    }

    #[test]
    fn dtype_parse_and_reject() {
        assert_eq!(DType::parse("fp32").unwrap(), DType::F32);
        assert_eq!(DType::parse("fp64").unwrap(), DType::F64);
        assert_eq!(DType::parse("FP16").unwrap(), DType::F16);
        // ⚠ 不认识的要**报错**而不是静默回落
        assert!(DType::parse("bf16").is_err());
        assert!(DType::parse("int8").is_err());
    }

    #[test]
    fn only_fp32_has_simd() {
        assert!(DType::F32.has_simd());
        assert!(!DType::F64.has_simd());
        assert!(!DType::F16.has_simd());
    }
}