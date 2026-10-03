"""Python 侧加载 Rust 库 —— **唯一**的集成点。

设计约束（来自 `docs/写作与事实基线.md`）：
· Python 版是**参照实现**，Rust 版是**候选实现**；
  **对拍通过之前不启用 Rust 版**。
· 回落必须**可归因**（P161的教训：静默降级会让原因丢失）。
  → `load()` 返回 `(lib, reason)`，调用方能把reason 写进日志。
"""

from __future__ import annotations

import ctypes
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
    def csr_spmm(self, indptr, idx, val, x, out, n_threads: int = 1) -> None:
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