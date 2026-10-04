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


class RustKernels:
    """Rust 核的薄封装。**全部方法在无 NPU 时也可用**（CPU 参照实现）。"""

    def __init__(self, lib: ctypes.CDLL):
        self.lib = lib
        self._bind()

    def _bind(self) -> None:
        L = self.lib
        f32p = ctypes.POINTER(ctypes.c_float)
        i64p = ctypes.POINTER(ctypes.c_longlong)
        i32p = ctypes.POINTER(ctypes.c_int)

        L.phdnet_m1_gemv.restype = ctypes.c_int
        L.phdnet_m1_gemv.argtypes = [f32p, ctypes.c_size_t, ctypes.c_size_t,
                                     f32p, f32p, f32p, ctypes.c_size_t]
        L.phdnet_m1_kwta.restype = ctypes.c_int
        L.phdnet_m1_kwta.argtypes = [f32p, ctypes.c_size_t, ctypes.c_size_t,
                                     f32p, i32p]
        L.phdnet_csr_spmm_simd.restype = ctypes.c_int
        L.phdnet_csr_spmm_simd.argtypes = [i64p, i64p, f32p, ctypes.c_size_t,
                                            f32p, f32p, ctypes.c_size_t]
        L.phdnet_csr_spmm.restype = ctypes.c_int
        L.phdnet_csr_spmm.argtypes = [i64p, i64p, f32p, ctypes.c_size_t,
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
                                           i64p, ctypes.c_size_t, f32p, f32p,
                                           ctypes.c_size_t]
        L.phdnet_has_npu.restype = ctypes.c_int
        L.phdnet_has_npu.argtypes = []
        L.phdnet_has_avx2.restype = ctypes.c_int
        L.phdnet_has_avx2.argtypes = []
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
    @staticmethod
    def _check_csr(indptr, idx, val, x, out) -> None:
        """CSR SpMV 的前置校验 —— **缺了会静默算错**（P174 审计发现）。

        ⚠ 重点：**dtype 必须是 fp32**。`argtypes` 声明的是 `f32p`，
        若传 fp64（**P173 之前 `sparse_pc.py` 正是 fp64！**），
        ctypes仍会按指针传过去 → Rust 按 fp32 解读 fp64 字节 → **结果全错**。
        ⚠ 另：`idx` 必须是 **int64**（Rust 侧按 i64 读）。
        """
        for nm, arr, dt in (("indptr", indptr, _np.int64),
                            ("idx", idx, _np.int64),
                            ("val", val, _np.float32),
                            ("x", x, _np.float32),
                            ("out", out, _np.float32)):
            got = _np.dtype(arr.dtype)
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
        if x.size != out.size:
            raise ValueError(
                "csr_spmm: x.size=%d 应等于 out.size=%d（方阵 SpMV）"
                % (x.size, out.size))
        if indptr[-1] != idx.size or idx.size != val.size:
            raise ValueError(
                "csr_spmm: indptr[-1]=%d / idx.size=%d / val.size=%d 不一致"
                % (indptr[-1], idx.size, val.size))
        if idx.size and (idx.min() < 0 or idx.max() >= x.size):
            raise ValueError(
                "csr_spmm: idx 越界 [0,%d) —— 实得 [%d,%d]"
                % (x.size, idx.min(), idx.max()))

    def csr_spmm(self, indptr, idx, val, x, out, n_threads: int = 1) -> None:
        self._check_csr(indptr, idx, val, x, out)
        import numpy as np
        n = indptr.size - 1
        self.lib.phdnet_csr_spmm(
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            idx.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
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
        self.lib.phdnet_csr_spmm_simd(
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            idx.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
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
        self.lib.phdnet_m6_sparse_fwd(
            W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), r, c,
            gather_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
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