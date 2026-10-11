"""Cython 核的加载与回落（P192，2026-10-08）—— 仓库里**唯一**的构建入口。

分派规则（`--cython-kernels`）
================================================================================
| 取值 | 行为 |
|---|---|
| `off`（**默认**，铁律④） | 完全不碰 Cython。走现行 numba/BLAS 路径，**逐位不变** |
| `auto` | 尝试加载已编译扩展 → 没有则**就地构建** → 构建失败回落 numba |
| `force` | 同 auto，但**加载/构建失败时 raise**（用于门禁与 CI 强制覆盖） |

为什么默认 `off`
================================================================================
铁律④「行为变更以 config 开关承载且默认关闭」，且本仓库对新增加速后端
的既有教训是：Rust 臂（`phdnet_rs/`）曾以「收益只在大 shape 上成立、
代价却是 dtype/idx 口径必须处处对齐」被整体删除（P182/P188）。
Cython 臂要避免同样的结局，就必须**先证明收益、再开默认**。

⚠ **诚实声明**：本机（x86 Windows）**无昇腾/NPU**，因此 Cython 核的
**端到端收益没有任何昇腾实测**。这里能做的是**结构/容差等价门禁**
（`tests/verifiers/verify_cykernels.py`）+ 本机微基准；不是浮点逐位保证。
收益数字必须等服务器 A/B，见 `docs/PHD-Net_性能评估与迭代方案.md` §八。
"""
from __future__ import annotations

import importlib
import os
import sys
import threading

__all__ = [
    "get_kernels", "cykernels_available", "cykernels_status",
    "build_kernels", "KERNELS",
]

# get_kernels 持锁后可进入 build_kernels；必须可重入，避免首次构建自死锁。
_lock = threading.RLock()
_kernels = None            # 加载成功的模块
_status: dict = {}         # 状态诊断（启动日志用）
_build_attempted = False


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mod_name() -> str:
    return "phdnet._cykernels"


def _try_import() -> object | None:
    """尝试 import 已编译的扩展；失败返回 None（不抛）。"""
    try:
        mod = importlib.import_module(_mod_name())
    except Exception as e:                          # noqa: BLE001
        _status["import_error"] = f"{type(e).__name__}: {e}"
        return None
    # ⚠ 运行时自检：扩展能被 import ≠ 算得对。历史上 numba 曾长期「静默错
    # 编译」被自检抓住（`phdnet/stdp_kernels.py::_selftest_numba` 的 docstring
    # 记录了那次事故：自检参数传错导致 numba 与 numpy **同时**返回 0，
    # 被误判成 numba 坏，其实是自检坏）。故这里对 Cython 也做真值对拍。
    try:
        if not mod.selftest():
            _status["selftest"] = "FAILED（真值对拍未过）"
            return None
    except Exception as e:                          # noqa: BLE001
        _status["selftest"] = f"raised {type(e).__name__}: {e}"
        return None
    return mod


def build_kernels(verbose: bool = False, force: bool = False) -> object | None:
    """就地构建 Cython 扩展（等价于 `python setup_cython.py build_ext --inplace`）。

    构建脚本自行读取 numpy include 路径，没有早于 numpy import 的限制；
    BLAS 线程环境需另在库初始化前设置。本函数不全局 chdir，setup 脚本
    使用绝对源码/包路径，支持从仓库外启动训练入口。

    返回构建后的模块；失败返回 None（`force=True` 时 raise）。
    """
    global _build_attempted, _kernels
    with _lock:
        if _kernels is not None:
            return _kernels
        if _build_attempted and not force:
            return None
        _build_attempted = True
        root = _repo_root()
        setup_py = os.path.join(root, "setup_cython.py")
        if not os.path.isfile(setup_py):
            _status["build"] = f"找不到 {setup_py}"
            if force:
                raise RuntimeError(_status["build"])
            return None
        # 抑制构建期的 stdout 噪声（Cython 警告会盖住训练日志）
        import io
        import contextlib
        buf = io.StringIO()
        old_argv = sys.argv
        try:
            sys.argv = ["setup_cython.py", "build_ext", "--inplace",
                        "--build-lib", root, "--build-temp",
                        os.path.join(root, "build", "cython_temp")]
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                import runpy
                runpy.run_path(setup_py, run_name="__main__")
        except (Exception, SystemExit) as e:          # setuptools 可通过 SystemExit 退出
            _status["build"] = f"构建失败 {type(e).__name__}: {e}"
            if verbose:
                print(f"[cykernels] 构建失败，回落 numba：{e}\n{buf.getvalue()[-2000:]}")
            if force:
                raise RuntimeError(_status["build"]) from e
            return None
        finally:
            sys.argv = old_argv
        importlib.invalidate_caches()
        mod = _try_import()
        if mod is None and force:
            raise RuntimeError(f"Cython 扩展构建后仍不可用：{cykernels_status()}")
        _kernels = mod
        return mod


def get_kernels(mode: str = "off", verbose: bool = False) -> object | None:
    """按 `--cython-kernels` 取核模块（None = 走 numba 路径）。

    ⚠ 幂等：重复调用只真正加载一次。
    """
    global _kernels
    mode = (mode or "off").strip().lower()
    with _lock:
        if mode == "off":
            _status["mode"] = "off"
            return None
        if mode not in ("auto", "force"):
            raise ValueError(f"--cython-kernels 取值非法：{mode!r}（可用 off/auto/force）")
        _status["mode"] = mode
        if _kernels is not None:
            return _kernels
        mod = _try_import()
        if mod is None:
            # 没有编译好的 → 尝试就地构建
            mod = build_kernels(verbose=verbose, force=(mode == "force"))
        if mod is None and mode == "force":
            raise RuntimeError(f"Cython 扩展不可用：{cykernels_status()}")
        _kernels = mod
        return _kernels


def cykernels_available() -> bool:
    """扩展是否**已加载可用**（诊断用，不触发构建）。"""
    return _kernels is not None


def cykernels_status() -> str:
    """一行状态串（启动日志 / verifier 用）。

    ⚠ 与 BUGS A8 族「设了但没生效」斗智：**实际报出加载/自检/构建的
    真实状态**，而不是「开关打开了吗」。
    """
    mod = _kernels
    if mod is None:
        bits = [f"mode={_status.get('mode', 'off')}", "ext=未加载"]
        for k in ("import_error", "selftest", "build"):
            if k in _status:
                bits.append(f"{k}={_status[k]}")
        return " | ".join(bits)
    try:
        info = mod.build_info()
    except Exception:                                # noqa: BLE001
        info = "build_info 不可用"
    return f"mode={_status.get('mode', 'auto')} | ext=已加载 | {info}"


class Kernels:
    """薄门面：把 `mode` 与模块封在一起，调用点读起来像普通属性。

    该门面仅提供 active/mode/status 与扩展属性转发，缺失的核会抛
    AttributeError，不隐式实现 numba 回退。生产机制由实例句柄决定分派，
    并在扩展不可用时调用原 numba/numpy 核。
    """

    __slots__ = ("_mod", "_mode")

    def __init__(self, mode: str = "off", verbose: bool = False):
        self._mode = (mode or "off").strip().lower()
        self._mod = get_kernels(self._mode, verbose=verbose) if self._mode != "off" else None

    @property
    def active(self) -> bool:
        """是否真的在用 Cython 核（`off` 或构建失败都返回 False）。"""
        return self._mod is not None

    @property
    def mode(self) -> str:
        return self._mode

    def status(self) -> str:
        return cykernels_status()

    # ── 各机制的转发（M1–M4a）────────────────────────────────────────
    # 无核时抛 AttributeError；回退由生产机制分派层负责。
    def __getattr__(self, name):
        if self._mod is not None and hasattr(self._mod, name):
            return getattr(self._mod, name)
        raise AttributeError(f"cykernels: 无核 {name!r}（mode={self._mode}）")


# 模块级默认门面（mode=off），不触发构建。
# PHDNet 使用 get_kernels 返回的原始模块句柄，不使用此门面。
KERNELS = Kernels("off")