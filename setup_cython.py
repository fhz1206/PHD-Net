"""PHD-Net Cython 扩展构建（P192，2026-10-08）。

为什么**不**放进 `pyproject.toml` 的常规 build
================================================================================
仓库的其余部分是**纯 Python**，用户 clone 下来就能 `python train/train.py`
跑训练（README §5「最小可跑」）。若把 Cython 写进 PEP 517 的
`build-system.requires`，则：

* 没有 C 编译器的机器（很多昇腾用户是纯容器/CI）**连 import 都失败**；
* `pip install -e .` 会因为需要编译器而卡住，破坏现有零构建流程。

故采用**两级策略**（`phdnet/cykernels.py` 是唯一的自动入口）：

1. **已有编译好的 `.so`/`.pyd`** → 直接 import，零构建开销。
2. **没有** → 显式请求 `auto/force` 时由 `cykernels.build_kernels()` 就地构建；
   `off` 不触发加载/构建，`auto` 失败回落并记原因，`force` 失败抛错。

构建命令（手动，与自动路径完全等价）：

```bash
# Linux / macOS
python setup_cython.py build_ext --inplace

# Windows（MSVC）
python setup_cython.py build_ext --inplace
```

构建需要 numpy 头文件，由构建函数调用 `np.get_include()`；
没有「必须早于 numpy import」的限制，BLAS 线程环境另须在库初始化前设置。

OpenMP
================================================================================
Cython 官方文档明确警告：**忘了传 OpenMP 编译参数时 `prange` 照样编译通过、
但退化为串行**（「will usually still compile but will not run in parallel」）。
因此：

* MSVC → `/openmp`，**不要**加进 link 参数（官方 parallelism 指南的原话）；
* gcc/clang → `-fopenmp`，编译与链接**都要**加。

本项目按 `--numba-threads` 的同一上限思路（默认 8）限制线程数，
理由见 `docs/PHD-Net_性能评估与迭代方案.md` §8.1（191 核不设限必然空转）。
线程上限在运行时由 `OMP_NUM_THREADS` / `omp_set_num_threads` 控制，
**不在编译期硬编码** —— 这样一台机器改一次环境变量即可，无需重新编译。

⚠ **Cython 3.3 的 OpenMP 限制**：`cython.parallel` 只能从**主线程**或
并行区内调用（OpenMP 自身限制）。本项目的核全部由主训练线程调用，
符合该约束。若将来把核搬到 worker 线程，只能用 `prange(..., use_threads_if=False)`
的串行版本；当前 loader 未提供 worker 自动串行封装，不应由 worker 调并行核。
"""
from __future__ import annotations

import os
import sys

# ── 语言/平台参数 ────────────────────────────────────────────────────────
# ⚠ 这里**不 import numpy**：只在需要 include 路径时才由 setuptools 触发
#   导入。该可选构建入口需要调用方事先安装 numpy/Cython/setuptools。
_IS_WIN = sys.platform == "win32"
_ROOT = os.path.dirname(os.path.abspath(__file__))


def _extra_compile_args() -> list[str]:
    """OpenMP 编译参数（MSVC 与 gcc/clang 分开）。"""
    if _IS_WIN:
        # 官方（docs.cython.org parallelism 指南）：MSVC 用 /openmp，
        # 且**不要**加到 link 参数。
        return ["/openmp", "/O2"]
    return ["-fopenmp", "-O3", "-funroll-loops"]


def _extra_link_args() -> list[str]:
    # ⚠ 官方 parallelism 指南（docs.cython.org，**当前版本**）明写：
    #   MSVC 只在 extra_compile_args 里加 /openmp，**不要**加进 link 参数。
    #   （旧版 tutorial 页说两边都加，那是过时页面，别照抄。）
    if _IS_WIN:
        return []
    return ["-fopenmp"]               # gcc/clang：编译 + 链接都要


def _ext_modules():
    import numpy as np
    from Cython.Build import cythonize
    from setuptools import Extension

    # ⚠ numpy 的 include 必须在 Cython 生成 C 之前拿到；setuptools 的
    #   Extension(include_dirs=…) 正是为此。cythonize 用 language_level=3。
    ext = Extension(
        "phdnet._cykernels",
        sources=[os.path.join(_ROOT, "phdnet", "_cykernels.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=_extra_compile_args(),
        extra_link_args=_extra_link_args(),
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    # compiler_directives 同时写进 Cython 与 C 编译器（生成 .c 时生效）。
    return cythonize(
        [ext],
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
            "nonecheck": False,
        },
    )


if __name__ == "__main__":
    from setuptools import setup

    setup(
        name="phdnet-cython-kernels",
        ext_modules=_ext_modules(),
        # --inplace 的源包定位不能依赖启动 cwd；从仓库外加载也写回本仓库。
        packages=["phdnet"],
        package_dir={"phdnet": os.path.join(_ROOT, "phdnet")},
        # 仅 build_ext 就地扩展；packages 用来确定扩展的源包路径，
        # 不改变项目的安装流程。我们只构建一个就地扩展，
        # 打包元数据是 noise。install 路径由 `phdnet/cykernels.py` 兜住。
    )