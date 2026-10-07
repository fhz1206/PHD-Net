"""Python 侧加载 Rust 库 —— **唯一**的集成点。

设计约束（来自 `docs/写作与事实基线.md`）：
· Python 版是**参照实现**，Rust 版是**候选实现**；
  **对拍通过之前不启用 Rust 版**。
· 回落必须**可归因**（P161的教训：静默降级会让原因丢失）。
  → `load()` 返回 `(lib, reason)`，调用方能把reason 写进日志。
"""

from __future__ import annotations

import ctypes
import numpy as _np

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
# 候选文件名：Linux 是 .so，Windows 是 .dll，macOS 是 .dylib
_NAMES = ("libphdnet_rs.so", "phdnet_rs.dll", "libphdnet_rs.dylib")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_flag(name: str) -> bool:
    return (os.environ.get(name, "") or "").strip().lower() in (
        "1", "true", "yes", "on")


# ⚠ **P179：CSR 越界检查的成本控制**
# `_check_csr` 的越界判据原为 `idx.min()`/`idx.max()` **numpy 全量扫描**。
# 1b 档读出门 idx 有**9.47M 元素** → 每次调用两次全扫 ≈ **9.7ms**，
# 实测让 `m2_matvec` 8T 从 3.1ms 涨到 **9.2ms**（比内核还贵 3 倍），
# 并导致 P178 误判「numba 快 2.5×」（实为校验层假象）。
#
# 现走**Rust 侧 SIMD+并行全量扫描** `phdnet_idx_range_i32`：
# **仍是全检（零语义降级）**，只是快~20×。旧 DLL 缺该符号时自动回落 numpy。
_CSR_CHECK_FULL = _env_flag("PHDNET_RS_CHECK_FULL")


def _fast_idx_range(lib, idx) -> tuple:
    """`idx` 的 `(min, max)` —— **全量**，走 Rust SIMD+并行（P179）。

    numpy 全扫 9.47M 元素 ≈ 9.7ms（比整个 SpMV 内核还贵 3 倍）；
    Rust 侧 `phdnet_idx_range_i32`（AVX2 一次比 8 个 i32 + 常驻池 8 线程）
    实测 ≈0.3-0.6ms。**语义相同（仍全检），只是快~20×。**

    P190：**模块级缓存**（键 = (id(idx), size)）——idx 的内容在生产热路径
    不变（拓扑只在重建时替换数组对象），首次全扫后命中即零开销；
    数组被替换（id 变）自动失效。校验语义不降级：仍是「每个新数组全检一次」。

    ⚠ 旧 DLL 无该符号 / 调用失败 → 回落 numpy（**不静默失效**）。
    """
    try:
        _key = (id(idx), idx.size)
        _hit = _IDX_RANGE_CACHE.get(_key)
        if _hit is not None:
            return _hit
    except TypeError:
        _key = None
    p = idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    mn = ctypes.c_int(0)
    mx = ctypes.c_int(0)
    fn = getattr(lib, "phdnet_idx_range_i32", None)
    if fn is None:
        return int(idx.min()), int(idx.max())
    if fn(p, ctypes.c_size_t(idx.size),
          ctypes.byref(mn), ctypes.byref(mx), ctypes.c_size_t(8)) != 0:
        return int(idx.min()), int(idx.max())
    _res = (int(mn.value), int(mx.value))
    if _key is not None:
        if len(_IDX_RANGE_CACHE) >= 64:
            _IDX_RANGE_CACHE.clear()
        _IDX_RANGE_CACHE[_key] = _res
    return _res


# 最近的 `RustKernels` 实例（供@staticmethod 的校验器复用其 lib 句柄）。
# ⚠ 单例假设与本模块的用法一致：进程内只load 一个库。
_LIB_HOLDER: list = [None]

# P190：`_fast_idx_range` 的模块级缓存（键 = (id(idx), size)）。
_IDX_RANGE_CACHE: dict = {}


class RustKernels:
    """Rust 核的薄封装。**全部方法在无 NPU 时也可用**（CPU 参照实现）。"""

    def __init__(self, lib: ctypes.CDLL):
        self.lib = lib
        self._bind()
        # P179：供 @staticmethod 的校验器复用 lib 句柄（走 Rust 侧快速 idx 扫描）
        if _LIB_HOLDER[0] is None:
            _LIB_HOLDER[0] = self
        # P190：**绑定层 int32 idx 缓存**——生产热路径每步把同一批 CSR idx
        # 反复传进来，`_idx_i32` 的 int64→int32 转换实测 0.052ms/次
        # （比 Rust 裸核 0.055ms 还贵，是 Rust 落后 numba 的主因）。
        # 键 = (id(idx), size)：数组被替换（id 变）自动失效；val 数组共享
        # 同一对象（sparse_pc 的 CSR 三元组结构不变）→ 与 sparse_pc.py
        # 的 `_csr_i32_cache`（P179）同口径。
        self._idx_i32_cache = {}
        # P190：u16 副本缓存（m2_matvec 自动路由 u16 核时用）
        self._idx_u16_cache = {}

    def _bind(self) -> None:
        L = self.lib
        f32p = ctypes.POINTER(ctypes.c_float)
        i64p = ctypes.POINTER(ctypes.c_longlong)
        i32p = ctypes.POINTER(ctypes.c_int)

        L.phdnet_m1_gemv.restype = ctypes.c_int
        L.phdnet_m1_gemv.argtypes = [f32p, ctypes.c_size_t, ctypes.c_size_t,
                                     f32p, f32p, f32p, ctypes.c_size_t]
        # P189：fp8(e4m3fn 位模式 uint8) → int8 码本 + per-tensor scale
        # （「fp8 存储 + int8 计算」每步的位数转换，CPU 多核；见 fp8_conv.rs）
        self._has_fp8_conv = hasattr(L, "phdnet_fp8_to_int8")
        if self._has_fp8_conv:
            u8p = ctypes.POINTER(ctypes.c_uint8)
            i8p = ctypes.POINTER(ctypes.c_int8)
            L.phdnet_fp8_to_int8.restype = ctypes.c_int
            L.phdnet_fp8_to_int8.argtypes = [u8p, ctypes.c_size_t, i8p,
                                             f32p, f32p, ctypes.c_size_t]
        L.phdnet_m1_kwta.restype = ctypes.c_int
        L.phdnet_m1_kwta.argtypes = [f32p, ctypes.c_size_t, ctypes.c_size_t,
                                     f32p, i32p]
        # ── M2 CSR 全集（P175）─────────────────────────────────────
        L.phdnet_m2_matvec.restype = ctypes.c_int
        L.phdnet_m2_matvec.argtypes = [i64p, i32p, f32p, ctypes.c_size_t,
                                       f32p, f32p, ctypes.c_size_t]
        # P186：u16 压缩 idx 的 SpMV（列数 ≤65535 时 idx 4B→2B，带宽 -25%）
        self._has_matvec_u16 = hasattr(L, "phdnet_m2_matvec_u16")
        if self._has_matvec_u16:
            u16p = ctypes.POINTER(ctypes.c_uint16)
            L.phdnet_m2_matvec_u16.restype = ctypes.c_int
            L.phdnet_m2_matvec_u16.argtypes = [i64p, u16p, f32p,
                                               ctypes.c_size_t,
                                               f32p, f32p, ctypes.c_size_t]
        L.phdnet_m2_add_outer.restype = ctypes.c_int
        L.phdnet_m2_add_outer.argtypes = [i64p, i32p, f32p, ctypes.c_size_t,
                                          f32p, f32p, ctypes.c_float,
                                          ctypes.c_size_t]
        L.phdnet_m2_oja_up.restype = ctypes.c_int
        L.phdnet_m2_oja_up.argtypes = [i64p, i32p, f32p, ctypes.c_size_t,
                                       f32p, f32p, ctypes.c_float,
                                       ctypes.c_size_t]
        L.phdnet_m2_clip.restype = ctypes.c_int
        L.phdnet_m2_clip.argtypes = [f32p, ctypes.c_size_t, ctypes.c_float,
                                     ctypes.c_size_t]
        # ── M2 融合核（P176）──────────────────────────────────────
        # ⚠ 全部用 c_void_p（真实 dtype 已由 `_check_fused_csr` 校验）
        _vp = ctypes.c_void_p
        _sz = ctypes.c_size_t
        # infer: up0(ip,ix,v,n1) up1(ip,ix,v,n2) dn0(3) dn1(3) s0 r1 r2 e0 e1(5)
        #       + n_steps n_threads = **21**（⚠ 数错会静默传错位置）
        L.phdnet_m2_infer_fused.argtypes = (
            [_vp] * 3 + [_sz] + [_vp] * 3 + [_sz] + [_sz]
            + [_vp] * 11 + [_sz, _sz])
        # learn: dn0(3+sz) dn1(3+sz) up0(3) up1(3) + 5向量 + 3f32 + 1sz
        # learn: dn0(3+sz) dn1(3+sz) up0(3) up1(3) + 5 向量
        #        + 3 f32 + **4 个向量长度（P176越界钳制）** + 1 sz = 27
        L.phdnet_m2_learn_fused.argtypes = (
            [_vp] * 3 + [_sz] + [_vp] * 3 + [_sz] + [_vp] * 3 + [_sz]
            + [_vp] * 3 + [_sz] + [_vp] * 5
            + [ctypes.c_float] * 3 + [_sz] * 4 + [_sz])
        L.phdnet_m2_row_norms.restype = ctypes.c_int
        L.phdnet_m2_row_norms.argtypes = [i64p, i32p, f32p, ctypes.c_size_t,
                                           f32p, ctypes.c_size_t]
        L.phdnet_m2_scale_rows.restype = ctypes.c_int
        L.phdnet_m2_scale_rows.argtypes = [i64p, i32p, f32p, ctypes.c_size_t,
                                           f32p, ctypes.c_size_t]
        L.phdnet_csr_spmm_simd.restype = ctypes.c_int
        L.phdnet_csr_spmm_simd.argtypes = [i64p, i32p, f32p, ctypes.c_size_t,
                                            f32p, f32p, ctypes.c_size_t]
        L.phdnet_csr_spmm.restype = ctypes.c_int
        L.phdnet_csr_spmm.argtypes = [i64p, i32p, f32p, ctypes.c_size_t,
                                      f32p, f32p, ctypes.c_size_t]
        L.phdnet_m4a.restype = ctypes.c_int
        L.phdnet_m4a.argtypes = [f32p, ctypes.c_size_t, f32p,
                                 ctypes.c_float, ctypes.c_float, ctypes.c_float]
        L.phdnet_m3_stdp.restype = ctypes.c_int
        L.phdnet_m3_stdp.argtypes = [f32p, f32p, f32p, ctypes.c_size_t,
                                     ctypes.c_float, ctypes.c_float]
        L.phdnet_m5_gate.restype = ctypes.c_float
        L.phdnet_m5_gate.argtypes = [ctypes.c_float, f32p, f32p,
                                     ctypes.POINTER(ctypes.c_longlong)]
        L.phdnet_m6_sparse_fwd.restype = ctypes.c_int
        L.phdnet_m6_sparse_fwd.argtypes = [f32p, ctypes.c_size_t, ctypes.c_size_t,
                                           i32p, ctypes.c_size_t, f32p, f32p,
                                           ctypes.c_size_t]
        L.phdnet_has_npu.restype = ctypes.c_int
        L.phdnet_has_npu.argtypes = []
        L.phdnet_has_avx2.restype = ctypes.c_int
        L.phdnet_has_avx2.argtypes = []
        # P179：idx 极值扫描（AVX2 + 常驻池并行）—— 替代 numpy 全量 min/max
        L.phdnet_idx_range_i32.restype = ctypes.c_int
        L.phdnet_idx_range_i32.argtypes = [i32p, ctypes.c_size_t,
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.c_size_t]
        L.phdnet_dtype_code.restype = ctypes.c_int
        L.phdnet_dtype_code.argtypes = [ctypes.c_char_p]
        L.phdnet_dtype_has_simd.restype = ctypes.c_int
        L.phdnet_dtype_has_simd.argtypes = [ctypes.c_int]
        L.phdnet_m1_gemv_dt.restype = ctypes.c_int
        L.phdnet_m1_gemv_dt.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                         ctypes.c_size_t, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_size_t, ctypes.c_int]
        L.phdnet_m1_gemv_simd.restype = ctypes.c_int
        L.phdnet_m1_gemv_simd.argtypes = [f32p, ctypes.c_size_t, ctypes.c_size_t,
                                           f32p, f32p, f32p, ctypes.c_size_t]

    # ── M1 ──────────────────────────────────────────────────────────────
    def m1_gemv(self, W, x, b, out, n_threads: int = 1) -> None:
        """`out = W @ x + b`。**要求 W C 连续 fp32**。"""
        import numpy as np
        assert W.flags["C_CONTIGUOUS"] and x.dtype == np.float32
        r, c = W.shape
        self.lib.phdnet_m1_gemv(
            W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), r, c,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n_threads))

    # -- dtype 分派（P172：Rust 跟着生产精度走）------------------------
    def dtype_code(self, dtype) -> int:
        """Python dtype 名 -> 内部码（0=fp32 / 1=fp64 / 2=fp16）。

        ⚠ **不认识就抛异常**，不静默回落（P161 纪律）。
        """
        import numpy as _np
        try:
            #np.dtype(...) 能把「类 / 实例 / 名字」统一成dtype 对象
            name = _np.dtype(dtype).name
        except TypeError:
            name = str(dtype)
        code = int(self.lib.phdnet_dtype_code(name.encode()))
        if code < 0:
            raise ValueError(
                "Rust 算子库不支持 dtype=%s；可用：fp32 / fp16 / fp64" % name)
        return code

    def dtype_has_simd(self, dtype) -> bool:
        """该 dtype 是否有手写 SIMD（**只有 fp32 有**，见 simd.rs）。"""
        return bool(self.lib.phdnet_dtype_has_simd(self.dtype_code(dtype)))

    def m1_gemv_dt(self, W, x, b, out, dtype, n_threads: int = 8) -> None:
        """dtype 感知的 GEMV：**按 Python 的 dtype 分派**（P172）。

        - fp32 -> **走手写 AVX2 SIMD**（若本机可用）
        - fp64 -> 标量（无 f64 SIMD）
        - fp16 -> 标量+ **f32 累加器**（社区标准：低精度存储 + 高精度累加），
          最后舍入到 fp16 -> **不与 Python 的逐步 fp16 逐位相同**
        """
        # ── 审计修复（P174）：三处**静默算错**的入口必须 fail-fast ──
        # ① dtype 参数与数组实际 dtype 不符 → Rust 会按错误精度解读字节
        #    → 结果全错（实测 relerr 8.2e+305）且无任何报错。
        # ② 非连续数组 → Rust 按连续布局算 → 结果全错（实测 relerr 1.04）。
        #    ⚠ 旧接口 m1_gemv 有 assert，新接口原先没有 —— **回归**。
        # ③ fp64/fp16 分支忽略 n_threads（恒串行）→ 调用者误以为并行。
        #    这里显式告知调用方真实并行度，不静默。
        want = _np.dtype(dtype)
        for nm, arr in (("W", W), ("x", x), ("b", b), ("out", out)):
            got = _np.dtype(arr.dtype)
            if got != want:
                raise ValueError(
                    "m1_gemv_dt: %s 的 dtype=%s，但 dtype 参数给的是 %s"
                    % (nm, got.name, want.name))
        for nm, arr in (("W", W), ("x", x), ("b", b), ("out", out)):
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    "m1_gemv_dt: %s 非 C 连续（strides=%s）—— "
                    "Rust 按连续布局计算会全错，请先 np.ascontiguousarray()"
                    % (nm, arr.strides))
        code = self.dtype_code(dtype)
        r, c = W.shape
        if r * c != W.size or x.size != c or b.size != r or out.size != r:
            raise ValueError(
                "m1_gemv_dt: 形状不匹配 —— W%s·x%s+b%s -> out%s"
                % (W.shape, x.shape, b.shape, out.shape))
        rc = self.lib.phdnet_m1_gemv_dt(
            ctypes.c_void_p(W.ctypes.data),
            ctypes.c_size_t(r), ctypes.c_size_t(c),
            ctypes.c_void_p(x.ctypes.data),
            ctypes.c_void_p(b.ctypes.data),
            ctypes.c_void_p(out.ctypes.data),
            ctypes.c_size_t(n_threads),
            ctypes.c_int(code))
        # ③ **Rust 侧如实回报**实际线程数（fp64/fp16 恒为 1）。
        #    ⚠ 不能只看 Python 传了什么 —— 之前 fp64 静默串行而调用者
        #    以为并行（P174 审计发现）。返回 -1 才是错误。
        if rc < 0:
            raise RuntimeError("phdnet_m1_gemv_dt 失败（dtype 码非法？）")
        self._last_effective_threads = int(rc)
        if want != _np.float32 and int(rc) != n_threads:
            self._last_effective_threads = int(rc)

    def has_avx2(self) -> bool:
        """本机是否支持 AVX2+FMA（**真跑一次**探测）。"""
        return bool(self.lib.phdnet_has_avx2())

    def m1_gemv_simd(self, W, x, b, out, n_threads: int = 8) -> None:
        """SIMD 版 GEMV（AVX2 + 4 路 FMA）。

        ⚠ **与 `m1_gemv` 不逐位**（累加器分组改变求和顺序，fp32 ~1e-7）。
        无 AVX2 时会静默回落到标量路径 —— 用 `has_avx2()` 预先确认。
        """
        r, c = W.shape
        self.lib.phdnet_m1_gemv_simd(
            W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), r, c,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n_threads))

    def m1_kwta(self, u, k: int):
        """返回 `(s, idx)`。⚠ `idx` 是**按值降序**（Python 的 argpartition 无序）。"""
        import numpy as np
        n = u.size
        s = np.zeros(n, dtype=np.float32)
        idx = np.zeros(k, dtype=np.int32)
        self.lib.phdnet_m1_kwta(
            u.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, k,
            s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        return s, idx

    # ── M2 ──────────────────────────────────────────────────────────────
    # ── 校验helper（P174 审计）──────────────────────────────────────
    # ── M2 融合核（P176）──────────────────────────────────────────
    # ⚠ 精度已统一为 **全 fp32**（P176）—— Python 融合核原本是
    #   「fp32 存储 + fp64 累加 + fp64 中间数组」的混合精度。

    def m2_infer_fused(self, up0, up1, dn0, dn1, s0, r1, r2, e0, e1,
                       n_steps: int, n_threads: int = 0) -> None:
        """**融合推理**（对应 `_pc_infer_fused`，P176 全 fp32）。

        ⚠ **矩形 CSR**：up0 是 n1×n0、up1 是 n2×n1、
          dn0 = transpose(up0)、dn1 = transpose(up1)。
          ⚠ `dn0`/`dn1` 的**行数不是 n2** —— 传错会越界**段错误**（实测踩过）。
        """
        for nm, csr in (("up0", up0), ("up1", up1),
                        ("dn0", dn0), ("dn1", dn1)):
            self._check_fused_csr(csr, nm)
        # ⚠ P176：`e0` 必须是 **n0** 长（`up0` 的列空间），`r1`/`e1` 是 n1
        n1 = up0[0].size - 1
        n2 = up1[0].size - 1
        n0 = dn0[0].size - 1     # ⚠ P176：e0 的正确长度（原误用 n1）
        for nm, arr, want in (("s0", s0, n0),
                              ("r1", r1, n1), ("r2", r2, n2),
                              ("e0", e0, n0), ("e1", e1, n1)):
            if arr.size != want:
                raise ValueError(
                    "infer_fused: %s 长度应为 %d（= dn0/up0 行数），实得 %d"
                    % (nm, want, arr.size))
            if _np.dtype(arr.dtype) != _np.float32:
                raise ValueError("infer_fused: %s 必须 fp32，实得 %s"
                                 % (nm, arr.dtype))
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError("infer_fused: %s 非 C 连续" % nm)
        _p = ctypes.c_void_p
        _sz = ctypes.c_size_t
        # ⚠⚠ **P182修复（真 bug，relerr≈1.9 的根因）**：融合核的 `idx` 走 `c_void_p`，
        #   **从未做 int64→int32 转换**，而 Rust 侧按 `*const i32` 读 →
        #   int64 数组被当成 int32 交错读 → **读到垃圾，数值全错**。
        #   （P178 落地 i32 时只改了原子算子`_idx_i32`，**漏了融合核**。）
        #   典型触发：`SparsePCStack` 的 CSR idx **是 int64**（见 `_random_csr`），
        #   故 `m2_infer_fused`/`m2_learn_fused` 一直不可用（门禁 N2/N3 FAIL）。
        # ✅ 修法：与原子算子一致，idx 统一走 `_idx_i32`（int32 零拷贝 / int64 转换）。
        # ⚠⚠ **必须用局部变量持有转换结果**：`c_void_p` 只传地址、**不持引用**，
        #   若写成 `_p(self._idx_i32(up0[1]).ctypes.data)`，临时数组会在
        #   ctypes 实际使用前被 GC 回收 → **access violation**（已实测）。
        #   故先把4 个转换结果存进列表，用完后再释放。
        _i32 = [self._idx_i32(c[1]) for c in (up0, up1, dn0, dn1)]
        self.lib.phdnet_m2_infer_fused(
            _p(up0[0].ctypes.data), _p(_i32[0].ctypes.data),
            _p(up0[2].ctypes.data), _sz(n1),
            _p(up1[0].ctypes.data), _p(_i32[1].ctypes.data),
            _p(up1[2].ctypes.data), _sz(n2),
            _sz(n0),                     # ⚠ P176：dn0 行数 = e0 长度
            _p(dn0[0].ctypes.data), _p(_i32[2].ctypes.data),
            _p(dn0[2].ctypes.data),
            _p(dn1[0].ctypes.data), _p(_i32[3].ctypes.data),
            _p(dn1[2].ctypes.data),
            _p(s0.ctypes.data),
            _p(r1.ctypes.data), _p(r2.ctypes.data),
            _p(e0.ctypes.data), _p(e1.ctypes.data),
            _sz(n_steps), _sz(n_threads))
        del _i32

    def m2_learn_fused(self, dn0, dn1, up0, up1, e0, e1, r1, r2, s0,
                       eta_pc: float, eta_oja: float, w_max: float,
                       n_threads: int = 0) -> None:
        """**融合学习**（对应 `_pc_learn_fused`，P176 全 fp32）。**原地**更新 4 个 val。"""
        for nm, csr in (("dn0", dn0), ("dn1", dn1),
                        ("up0", up0), ("up1", up1)):
            self._check_fused_csr(csr, nm, writable=True)
        for nm, arr in (("e0", e0), ("e1", e1), ("r1", r1),
                        ("r2", r2), ("s0", s0)):
            if _np.dtype(arr.dtype) != _np.float32:
                raise ValueError("learn_fused: %s 必须是 fp32，实得 %s"
                                 % (nm, arr.dtype))
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError("learn_fused: %s 非 C 连续" % nm)
        _p = ctypes.c_void_p
        _sz = ctypes.c_size_t
        # ⚠⚠ **P182**：同 `m2_infer_fused`，idx 必须 int64→int32 转换，
        #   否则 Rust 按 i32 读 int64 数组 → 读到垃圾（门禁 N3 FAIL 根因）。
        _i32 = [self._idx_i32(c[1]) for c in (dn0, dn1, up0, up1)]
        self.lib.phdnet_m2_learn_fused(
            _p(dn0[0].ctypes.data), _p(_i32[0].ctypes.data),
            _p(dn0[2].ctypes.data), _sz(dn0[0].size - 1),
            _p(dn1[0].ctypes.data), _p(_i32[1].ctypes.data),
            _p(dn1[2].ctypes.data), _sz(dn1[0].size - 1),
            _p(up0[0].ctypes.data), _p(_i32[2].ctypes.data),
            _p(up0[2].ctypes.data), _sz(up0[0].size - 1),
            _p(up1[0].ctypes.data), _p(_i32[3].ctypes.data),
            _p(up1[2].ctypes.data), _sz(up1[0].size - 1),
            _p(e0.ctypes.data), _p(e1.ctypes.data),
            _p(r1.ctypes.data), _p(r2.ctypes.data), _p(s0.ctypes.data),
            ctypes.c_float(eta_pc), ctypes.c_float(eta_oja),
            ctypes.c_float(w_max),
            # ⚠ P176：传真实长度 —— Rust **不能**像 numba 那样越界读。
            #Python 的 `_pc_learn_fused` 对 dn0(n0行) 读 e0(n1长) 会越界
            #   → 静默读相邻内存。Rust 钳制到这些长度（**语义不等价**，
            #   已在 `verify_m2_rust_kernels.py` 记录为已知差异）。
            _sz(e0.size), _sz(e1.size), _sz(r1.size), _sz(r2.size),
            _sz(n_threads))
        del _i32

    @staticmethod
    def _check_fused_csr(csr, name, writable: bool = False) -> None:
        """融合核的 CSR 校验（**矩形合法**，不要求方阵 —— P174曾误加方阵断言）。"""
        ip, ix, vl = csr
        if _np.dtype(ip.dtype) != _np.int64:
            raise ValueError("%s: indptr 必须 int64，实得 %s"
                             % (name, ip.dtype))
        if _np.dtype(ix.dtype) not in (_np.int64, _np.int32):
            raise ValueError("%s: idx 必须 int64/int32（Rust 核 P178 原生 i32），实得 %s"
                             % (name, ix.dtype))
        if _np.dtype(vl.dtype) != _np.float32:
            raise ValueError("%s: val 必须 fp32（Rust 按 f32 读），实得 %s"
                             % (name, vl.dtype))
        for nm, arr in (("indptr", ip), ("idx", ix), ("val", vl)):
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError("%s: %s 非 C 连续" % (name, nm))
        if ip.size < 1 or ip[-1] != ix.size or ix.size != vl.size:
            raise ValueError(
                "%s: indptr[-1]=%d / idx.size=%d / val.size=%d 不一致"
                % (name, ip[-1], ix.size, vl.size))
        if ix.size:
            # P179：全扫 min/max 改走 Rust SIMD+并行（语义相同，快 ~20×）
            lo, _hi = _LIB_HOLDER[0]._idx_range(ix) if _LIB_HOLDER[0] else (int(ix.min()), 0)
            if lo < 0:
                raise ValueError("%s: idx 有负值（min=%d）" % (name, lo))
        if writable and not vl.flags["WRITEABLE"]:
            raise ValueError("%s: val 不可写（learn 需原地更新）" % name)

    # ── M2 CSR 算子全集（P175）────────────────────────────────────
    #⚠ 每个都先过 `_check_csr`（dtype/连续/形状/越界）——
    #   P174 审计发现这些校验缺失会导致**静默算错**。

    def _idx_i32(self, idx):
        """把 CSR 的 `idx` 转成 Rust 核要的 **int32**（P178 原生 i32 落地）。

        · 已是 int32 → 零拷贝视图（若非连续则 `ascontiguousarray`）。
        · int64 → **P190：走实例级缓存**（键 = (id(idx), size)）——生产热路径
          每步重复转换同一批 idx 实测 0.052ms/次，比 Rust 裸核还贵；
          命中即零转换。数组被替换（id 变）自动失效。
        · 其余 dtype → 直接报错（不静默降级，P161 纪律）。
        """
        dt = _np.dtype(idx.dtype)
        if dt == _np.int32:
            return _np.ascontiguousarray(idx) if not idx.flags["C_CONTIGUOUS"] \
                else idx
        if dt == _np.int64:
            key = (id(idx), idx.size)
            hit = self._idx_i32_cache.get(key)
            if hit is not None:
                return hit
            out = _np.ascontiguousarray(idx, dtype=_np.int32)
            # 缓存上限护栏：异常多的不同 idx 数组时清空（防内存增长）
            if len(self._idx_i32_cache) >= 64:
                self._idx_i32_cache.clear()
            self._idx_i32_cache[key] = out
            return out
        raise ValueError(
            "CSR idx 的 dtype=%s 不支持；Rust 核只接受 int64/int32（P178）"
            % dt)

    def _idx_u16(self, idx):
        """P190：int64 idx → uint16 副本（仅当已确认 max<65536 时调用），
        走 (id, size) 实例缓存（与 `_idx_i32` 同口径）。"""
        key = (id(idx), idx.size)
        hit = self._idx_u16_cache.get(key)
        if hit is not None:
            return hit
        out = _np.ascontiguousarray(idx, dtype=_np.uint16)
        if len(self._idx_u16_cache) >= 64:
            self._idx_u16_cache.clear()
        self._idx_u16_cache[key] = out
        return out

    def m2_matvec_u16(self, indptr, idx_u16, val, x, out,
                      n_threads: int = 8) -> None:
        """P186：u16 压缩 idx 的 SpMV（列数 ≤65535 时每突触 6B vs 8B）。

        ⚠ 调用方须保证 idx 内容 ≤65535（越界即未定义——Rust 侧零扩展 gather）。
        数值与 i32 核逐位一致（同一 gather 值集、同一累加顺序）。
        """
        n = indptr.size - 1
        self.lib.phdnet_m2_matvec_u16(
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            idx_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            val.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n_threads))

    def m2_matvec(self, indptr, idx, val, x, out, n_threads: int = 8) -> None:
        """`out[i] = Σ val[p]·x[idx[p]]`（M2 推理 SpMV，行宽≥32 走 AVX2）。

        ⚠ **不与Python 逐位**（SIMD 改求和顺序）→ 门禁用容差 1e-5。
        ⚠ idx 经边界转 **int32**（P178 原生 i32 落地；列下标在 vocab 范围内
          ≤ 65535，int32 无损；已是 int32 则零拷贝）。
        P190：列上界 ≤65535 时**自动路由 u16 核**（idx 4B→2B，每突触 6B vs 8B，
        带宽受限负载实测再快 ~1.4×）——范围用 `_idx_range` 的缓存结果判定
        （校验层本来就全扫，零额外开销），u16 副本同样走 (id, size) 缓存。
        """
        self._check_csr(indptr, idx, val, x, out)
        n = indptr.size - 1
        _k = _LIB_HOLDER[0]
        hi = (_k._idx_range(idx)[1] if _k is not None else int(idx.max())) \
            if idx.size else 0
        if (hi < 65536 and getattr(self, "_has_matvec_u16", False)):
            idx16 = self._idx_u16(idx)
            self.lib.phdnet_m2_matvec_u16(
                indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
                idx16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                val.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_size_t(n),
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_size_t(n_threads))
            return
        idx32 = self._idx_i32(idx)
        self.lib.phdnet_m2_matvec(
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            idx32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            val.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n_threads))

    def fp8_to_int8(self, bits, codes, scale, partial,
                    n_threads: int = 8) -> None:
        """fp8(e4m3fn) 位模式 → int8 码本 + per-tensor scale（P189）。

        `bits` uint8 (N,)；`codes` int8 (N,)（就地写）；`scale` f32[1]（就地写）；
        `partial` f32 (n_threads,) 工作缓冲。数值与
        `fp8_int8_convert.fp8_to_int8_codes` **逐位一致**（RNE + clamp ±127）。
        """
        n = bits.size
        self.lib.phdnet_fp8_to_int8(
            bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(n),
            codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            partial.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n_threads))

    def m2_add_outer(self, indptr, idx, val, a, b, eta: float,
                     n_threads: int = 8) -> None:
        """**原地** `val[p] += eta·a[i]·b[idx[p]]`（稀疏外积/Hebbian）。

        ⚠ 逐边独立无依赖 → **可逐位**（含 `a[i]==0` 的跳过语义）。
        """
        self._check_csr_w(indptr, idx, val, a, b, None)
        n = indptr.size - 1
        idx32 = self._idx_i32(idx)
        self.lib.phdnet_m2_add_outer(
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            idx32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            val.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n),
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(eta),
            ctypes.c_size_t(n_threads))

    def m2_oja_up(self, indptr, idx, val, post, pre, eta: float,
                  n_threads: int = 8) -> None:
        """**原地** `val[p] += eta·post[i]·(pre[idx[p]] − post[i]·val[p])`。

        ⚠ 逐边独立 → **可逐位**。
        """
        self._check_csr_w(indptr, idx, val, post, pre, None)
        n = indptr.size - 1
        idx32 = self._idx_i32(idx)
        self.lib.phdnet_m2_oja_up(
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            idx32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            val.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n),
            post.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pre.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(eta),
            ctypes.c_size_t(n_threads))

    def m2_clip(self, val, w_max: float, n_threads: int = 8) -> None:
        """**原地** `val[p] = clip(val[p], ±w_max)`。

        ⚠ 复刻 Python的 `if v > w_max / elif v < -w_max`（**不是** `f32::clamp`
          —— 它对 NaN 的行为不同）→ **可逐位**。
        """
        if val.dtype != _np.float32:
            raise ValueError("m2_clip: val 必须是 fp32，实得 %s" % val.dtype)
        if not val.flags["C_CONTIGUOUS"]:
            raise ValueError("m2_clip: val 非 C 连续")
        self.lib.phdnet_m2_clip(
            val.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(val.size),
            ctypes.c_float(w_max),
            ctypes.c_size_t(n_threads))

    def m2_row_norms(self, indptr, idx, val, out, n_threads: int = 8) -> None:
        """`out[i] = ‖val[indptr[i]:indptr[i+1]]‖₂`。

        ⚠ `s**0.5`（Python）与 `f32::sqrt()`（Rust）都是 IEEE 精确 sqrt
          → **可逐位**。
        ⚠ **矩形 CSR**（`n_cols != n_rows`，本项目是常态：up0 是 n1×n0）——
          故**不能**用 `_check_csr`（它按方阵校验 idx 上界，会误报）。
        """
        self._check_csr_rect(indptr, idx, val, out)
        n = indptr.size - 1
        idx32 = self._idx_i32(idx)
        self.lib.phdnet_m2_row_norms(
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            idx32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            val.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n_threads))

    def m2_scale_rows(self, indptr, idx, val, target, n_threads: int = 8) -> None:
        """**原地** `val[p] *= target[i]/max(‖row‖₂, 1e-12)`。

        ⚠ 逐位（两遍：先求 norm 再缩放，与 Python 同序）；
          `1e-12` 保护复刻 → **不会除零**。
        """
        self._check_csr_w(indptr, idx, val, target, None, None)
        n = indptr.size - 1
        idx32 = self._idx_i32(idx)
        self.lib.phdnet_m2_scale_rows(
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            idx32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            val.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n),
            target.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n_threads))

    @staticmethod
    def _check_csr_w(indptr, idx, val, v1, v2, out) -> None:
        """**原地更新**算子的校验（P175：`add_outer` / `oja_up` / `scale_rows`）。

        与 `_check_csr` 的差别：
        · `val` 须**可写**（`argtypes` 是 `f32p`，但语义上要写）
        · 支持**方形/矩形**（`v1`/`v2` 的长度不必等于行数）
        · `out=None`（原地算子无输出数组）
        """
        if indptr.dtype != _np.int64:
            raise ValueError("indptr 必须是 int64，实得 %s" % indptr.dtype)
        if idx.dtype not in (_np.int64, _np.int32):
            raise ValueError("idx 必须是 int64/int32（Rust 核 P178 原生 i32），实得 %s" % idx.dtype)
        if val.dtype != _np.float32:
            raise ValueError(
                "val 必须是 fp32（Rust 按 f32 读写），实得 %s" % val.dtype)
        for nm, arr in (("indptr", indptr), ("idx", idx), ("val", val)):
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError("%s 非 C 连续（strides=%s）"
                                 % (nm, arr.strides))
        n = indptr.size - 1
        if indptr[-1] != idx.size or idx.size != val.size:
            raise ValueError(
                "indptr[-1]=%d / idx.size=%d / val.size=%d 不一致"
                % (indptr[-1], idx.size, val.size))
        for nm, arr in (("v1", v1), ("v2", v2)):
            if arr is None:
                continue
            if arr.dtype != _np.float32:
                raise ValueError("%s 必须是 fp32，实得 %s" % (nm, arr.dtype))
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError("%s 非 C 连续" % nm)
        # P179：合并成**一次** Rust 侧全扫（原来 numpy 扫 2~3 遍）
        rng_lo = rng_hi = None
        if idx.size:
            _k = _LIB_HOLDER[0]
            rng_lo, rng_hi = (_k._idx_range(idx) if _k is not None
                              else (int(idx.min()), int(idx.max())))
        if rng_lo is not None and rng_lo < 0:
            raise ValueError("idx 有负值（min=%d）" % rng_lo)
        # ⚠ **P175 修正**：`add_outer` 查 `b`（源侧），`oja_up` 查 `pre`（源侧）——
        #   二者是**列索引空间**，不是 `a`/`post`（行侧）。之前用 v1 判上界
        #   会在矩形 CSR 上误报（实测 idx.max()=255 vs v1(post).size=192）。
        #   `scale_rows` 不访问任何向量 → 跳过。
        bound_arr = v2
        if rng_hi is not None and bound_arr is not None and rng_hi >= bound_arr.size:
            raise ValueError("idx 越界 [0,%d) —— 实得 max=%d"
                             % (bound_arr.size, rng_hi))

    @staticmethod
    def _check_csr_rect(indptr, idx, val, out) -> None:
        """**矩形** CSR 的校验（`n_cols != n_rows`）—— 用于**不访问 x** 的算子。

        ⚠ 为什么需要（P175 实测踩到）：`_check_csr` 按**方阵**假设校验
          `idx.max() < x.size`。但 `row_norms`/`scale_rows` **根本不读 x**，
          且本项目的 CSR **普遍是矩形**（`up0` 是 n1×n0、n1≠n0）
          → 方阵校验会**误报**「idx 越界」。

        ⚠ idx 上界**无法在此校验**（没有 n_cols 信息）→ 由调用方保证
          （Python 侧 CSR 由 `_random_csr` 生成，天然合法）。
        """
        for nm, arr, dt in (("indptr", indptr, _np.int64),
                            ("idx", idx, None),
                            ("val", val, _np.float32),
                            ("out", out, _np.float32)):
            if nm == "idx":
                if _np.dtype(arr.dtype) not in (_np.int64, _np.int32):
                    raise ValueError(
                        "%s 的 dtype=%s，Rust 侧按 int64/int32 读 —— 不符会**静默算错**（P175）"
                        % (nm, arr.dtype))
                if not arr.flags["C_CONTIGUOUS"]:
                    raise ValueError("%s 非 C 连续（strides=%s）"
                                     % (nm, arr.strides))
                continue
            if _np.dtype(arr.dtype) != _np.dtype(dt):
                raise ValueError(
                    "%s 的 dtype=%s，Rust 侧按 %s 读 —— 不符会**静默算错**（P175）"
                    % (nm, arr.dtype, _np.dtype(dt).name))
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError("%s 非 C 连续（strides=%s）"
                                 % (nm, arr.strides))
        if indptr.size < 1 or indptr.size - 1 != out.size:
            raise ValueError("indptr.size-1=%d 应等于 out.size=%d"
                             % (indptr.size - 1, out.size))
        if indptr[-1] != idx.size or idx.size != val.size:
            raise ValueError("indptr[-1]=%d / idx.size=%d / val.size=%d 不一致"
                             % (indptr[-1], idx.size, val.size))
        if idx.size:
            # P179：全扫 min 改走 Rust SIMD+并行（语义相同，快 ~20×）
            _k = _LIB_HOLDER[0]
            lo = _k._idx_range(idx)[0] if _k is not None else int(idx.min())
            if lo < 0:
                raise ValueError("idx 有负值（min=%d）" % lo)

    def _idx_range(self, idx) -> tuple:
        """`idx` 的 `(min, max)` —— **全量**，但走 Rust SIMD+并行（P179）。

        原实现是 `int(idx.min()), int(idx.max())`（numpy 全扫 9.47M 元素
        ≈ 9.7ms，比整个 SpMV 内核还贵 3 倍）。现调 `phdnet_idx_range_i32`
        （AVX2 一次比 8 个 i32 + 常驻池 8 线程），实测 ≈0.3-0.6ms。
        """
        return _fast_idx_range(self.lib, idx)

    @staticmethod
    def _check_csr(indptr, idx, val, x, out) -> None:
        """CSR SpMV 的前置校验 —— **缺了会静默算错**（P174 审计发现）。

        ⚠ 重点：**dtype 必须是 fp32**。`argtypes` 声明的是 `f32p`，
        若传 fp64（**P173 之前 `sparse_pc.py` 正是 fp64！**），
        ctypes仍会按指针传过去 → Rust 按 fp32 解读 fp64 字节 → **结果全错**。
        ⚠ 另：`idx` 必须是 **int64**（Rust 侧按 i64 读）。
        """
        for nm, arr, dt in (("indptr", indptr, _np.int64),
                            ("idx", idx, None),
                            ("val", val, _np.float32),
                            ("x", x, _np.float32),
                            ("out", out, _np.float32)):
            got = _np.dtype(arr.dtype)
            if nm == "idx":
                if got not in (_np.int64, _np.int32):
                    raise ValueError(
                        "csr_spmm: idx 的 dtype=%s，Rust 侧按 int64/int32 读 —— "
                        "不符会**静默算错**（P174）" % got.name)
                if not arr.flags["C_CONTIGUOUS"]:
                    raise ValueError(
                        "csr_spmm: %s 非 C 连续（strides=%s）→ 请先 "
                        "np.ascontiguousarray()" % (nm, arr.strides))
                continue
            if got != _np.dtype(dt):
                raise ValueError(
                    "csr_spmm: %s 的 dtype=%s，Rust 侧按 %s 读 —— "
                    "不符会**静默算错**（P174）"
                    % (nm, got.name, _np.dtype(dt).name))
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    "csr_spmm: %s 非 C 连续（strides=%s）→ 请先 "
                    "np.ascontiguousarray()" % (nm, arr.strides))
        if indptr.size < 1 or indptr.size - 1 != out.size:
            raise ValueError(
                "csr_spmm: indptr.size-1=%d 应等于 out.size=%d"
                % (indptr.size - 1, out.size))
        # ⚠⚠ **P175 修正：这里曾写 `x.size == out.size`（方阵假设）—— 是错的**。
        #   本项目的 CSR **普遍是矩形**（`up0` 是 n1×n0、n1≠n0）→
        #   方阵校验会让**生产默认场景直接报错**（实测 x.size=256 vs out.size=192）。
        #   正确不变式（**两条都要**）：
        #     · `x.size >= idx.max() + 1`（下面那一行在查 —— 这才是真正的越界判据）
        #     · `x.size >= 1`（空 x 时下面那行不报错，这里兜住）
        if x.size < 1:
            raise ValueError("csr_spmm: x 为空（size=0）—— 无法解释 idx")
        if indptr[-1] != idx.size or idx.size != val.size:
            raise ValueError(
                "csr_spmm: indptr[-1]=%d / idx.size=%d / val.size=%d 不一致"
                % (indptr[-1], idx.size, val.size))
        # ⚠⚠ **P179 修正：这里曾用 `idx.min()`/`idx.max()` 做全量 numpy 扫描**。
        #   意图是防「idx 越界 → Rust 越界读 → 段错误」。但 1b 档读出门 idx 有
        #   **9.47M 元素**，numpy 两次全扫 ≈ **9.7ms** —— **比整个 SpMV 内核
        #   （3.1ms）还贵 3 倍**，让 `m2_matvec` 8T 从 3.1ms 涨到 9.2ms。
        #   （P178 曾据此得出「numba 快 2.5×」—— 那是**校验层的假象**，非内核差距。）
        #
        #   现改为**调Rust 侧的 SIMD+并行全量扫描** `phdnet_idx_range_i32`：
        #   · **仍是全量检查**（不是抽样）→ 越界检测语义**零降级**；
        #   · AVX2 一次比8 个 i32 + 常驻池 8 线程 → 实测 ≈0.3-0.6ms（~20× 加速）。
        #   · 旧 DLL 无此符号 → 自动回落numpy 全扫（保持兼容，不静默失效）。
        if idx.size:
            _k = _LIB_HOLDER[0]
            lo, hi = (_k._idx_range(idx) if _k is not None
                      else (int(idx.min()), int(idx.max())))
            if lo < 0 or hi >= x.size:
                raise ValueError(
                    "csr_spmm: idx 越界 [0,%d) —— 实得 [%d,%d]"
                    % (x.size, lo, hi))

    def csr_spmm(self, indptr, idx, val, x, out, n_threads: int = 1) -> None:
        self._check_csr(indptr, idx, val, x, out)
        import numpy as np
        n = indptr.size - 1
        idx32 = self._idx_i32(idx)
        self.lib.phdnet_csr_spmm(
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            idx32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            val.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n_threads))

    # ── M4a / M3 / M5 ───────────────────────────────────────────────────
    def csr_spmm_simd(self, indptr, idx, val, x, out, n_threads: int = 8) -> None:
        """M2 CSR SpMV 的 **SIMD 路径**（P173，fp32）。

        ⚠ **同样需要校验**（P174 审计）—— `csr_spmm_simd` 原本**零校验**。

        ⚠ **收益有限**（与 GEMV 不同）：SpMV 瓶颈在**访存**且 `x[idx[p]]` 是
        **间接寻址** → 无法连续加载，SIMD 只能提供 4 路 ILP。
        ⚠ **不与标量逐位**（4 路累加器改求和顺序）→ 门禁用容差。
        """
        self._check_csr(indptr, idx, val, x, out)
        n = indptr.size - 1
        idx32 = self._idx_i32(idx)
        self.lib.phdnet_csr_spmm_simd(
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            idx32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            val.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n_threads))

    def m4a(self, slots, r, gamma: float, gate: float, thresh: float) -> None:
        import numpy as np
        n = slots.size
        self.lib.phdnet_m4a(
            slots.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
            r.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(gamma), ctypes.c_float(gate), ctypes.c_float(thresh))

    def m3_stdp(self, w, pre, post, w_max: float, eta: float) -> None:
        import numpy as np
        n = w.size
        self.lib.phdnet_m3_stdp(
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pre.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            post.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n), ctypes.c_float(w_max), ctypes.c_float(eta))

    def m5_gate(self, surprise: float, mean: float, m2: float, count: int):
        m = ctypes.c_float(mean)
        v = ctypes.c_float(m2)
        c = ctypes.c_longlong(count)
        g = self.lib.phdnet_m5_gate(ctypes.c_float(surprise),
                                    ctypes.byref(m), ctypes.byref(v),
                                    ctypes.byref(c))
        return float(g), float(m.value), float(v.value), int(c.value)

    # ── M6 ──────────────────────────────────────────────────────────────
    def m6_sparse_fwd(self, W, gather_idx, h, out, n_threads: int = 1) -> None:
        import numpy as np
        r, c = W.shape
        g32 = self._idx_i32(gather_idx)
        self.lib.phdnet_m6_sparse_fwd(
            W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), r, c,
            g32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_size_t(c),
            h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n_threads))

    # ── 设备 ─────────────────────────────────────────────────────────────
    def has_npu(self) -> bool:
        return bool(self.lib.phdnet_has_npu())


def _candidate_paths():
    env = os.environ.get("PHDNET_RS_LIB")
    if env:
        yield Path(env)
    for n in _NAMES:
        # release 与 debug 都找（对齐「显式设置优先」纪律）
        yield _ROOT / "target" / "release" / n
        yield _ROOT / "target" / "debug" / n
        # 也允许从仓库根引用（部署时把 .so 放phdnet_rs/ 或项目根）
        yield _ROOT.parent / n


def load() -> tuple[RustKernels | None, str]:
    """加载 Rust 库。返回 `(lib, reason)`：
    · `(lib, "")` → 成功
    · `(None, reason)` → **不可用**（reason 要写进日志，P161 的纪律）
    """
    tried = []
    for p in _candidate_paths():
        if not p.is_file():
            tried.append(str(p))
            continue
        try:
            lib = ctypes.CDLL(str(p))
        except OSError as e:
            return None, "加载失败 %s: %s" % (p, e)
        try:
            k = RustKernels(lib)
        except AttributeError as e:
            return None, "符号缺失（版本不匹配？）%s: %s" % (p, e)
        return k, ""
    return None, ("未找到 libphdnet_rs（已试 %d 个路径；"
                  "请先 `cd phdnet_rs && cargo build --release`）" % len(tried))


if __name__ == "__main__":  # pragma: no cover
    k, why = load()
    if k is None:
        print("[phdnet_rs] 不可用：", why, file=sys.stderr)
        raise SystemExit(1)
    print("[phdnet_rs] 已加载；has_npu =", k.has_npu())