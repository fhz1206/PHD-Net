"""解释器探测：**找到能 import numpy 的那个 Python**。

## 为什么需要这个模块

本机（Windows）同时存在多个 Python：

| 命令 | 实际路径 | numpy |
|---|---|---|
| `py` | `C:\\Windows\\py` → Python **3.14.5** | ✅ 2.5.2 |
| `python` | WorkBuddy 内置 **3.13.12** | ❌ 无 |

后果（实测踩到过）：

- `python tests/gates/xxx.py` → `ModuleNotFoundError: No module named 'numpy'`
- 用 `sys.executable` 起子进程 → 若本进程本身是用裸 `python` 起的，
  子进程**继承同一个坏解释器** → 整批子进程全灭（`bench_server_threads.py`
  首次运行就是这样失败的：三轮全部 `No module named 'numpy'`）。

所以：**任何要起子进程的工具（门禁 runner / bench 脚本 / A/B 脚本）都必须
先确定一个"好解释器"**，不能盲信 `sys.executable`。

## 用法

```python
from phdnet._python_exe import python_exe

exe = python_exe()          # 返回能 import numpy 的解释器路径
subprocess.run([exe, "train/train.py", ...])
```

或在命令行：

```bash
python -c "from phdnet._python_exe import python_exe; print(python_exe())"
```
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["python_exe", "has_numpy"]


def has_numpy(exe: str | None = None, timeout: float = 60.0) -> bool:
    """该解释器能否 `import numpy`。

    ⚠ **必须用「脚本模式」探测，不能只用 `-c` 模式** ——
      Windows 的 `py.exe` 是**启动器**：`py -c "import numpy"` 会按**默认
      解释器**跑（成功），但 `py some_script.py` 按**文件关联**选解释器
      （实测：关联到 WorkBuddy 内置的无 numpy 3.13 → `No module named
      'numpy'`）。两者结果**不一致** → 只测 `-c` 会误判启动器可用。
      故这里用 `-c` 测**真解释器**、用「临时脚本」测**启动器**。
    """
    exe = exe or sys.executable
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "-c", "import numpy"],
                           capture_output=True, timeout=timeout)
        return r.returncode == 0
    except Exception:                                       # noqa: BLE001
        return False


def _script_mode_ok(exe: str, timeout: float = 60.0) -> bool:
    """用**脚本模式**验证（专门针对启动器：`-c` 成功不代表跑脚本也成功）。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "_probe_np.py"
        f.write_text("import numpy\n", encoding="utf-8")
        try:
            r = subprocess.run([exe, str(f)], capture_output=True,
                               timeout=timeout)
            return r.returncode == 0
        except Exception:                                   # noqa: BLE001
            return False


def python_exe(required: str = "numpy") -> str:
    """返回一个**能 import `required`** 的解释器路径。

    优先级：
      1. 当前解释器（`sys.executable`）——与本进程同环境，最可靠；
      2. PATH 里的 `python3` / `python`（**要求是真实解释器，不是启动器**）；
      3. PATH 里的 `py`（Windows 启动器，**最低优先级**）。

    ⚠ **为什么 `py` 放最后**：Windows 的 `C:\\Windows\\py.EXE` 是**启动器**
      （按文件关联选解释器），不是解释器本身 → 用它起子进程时，
      `import numpy` 的成败取决于关联配置，**极不稳定**
      （实测：同一批门禁，用 `py` 启动有的过有的不过）。
      真实解释器路径（如 `...\\Python314\\python.exe`）才是稳定的。

    ⚠ Windows 上 `python` 可能指向 WorkBuddy 内置的无 numpy 解释器，
      故必须**实测**而非看名字。

    全部不可用时**退回 `sys.executable`**（让调用方的报错信息更直接，
    而不是在这里抛一个掩盖真相的异常）。
    """
    def _is_launcher(p: str) -> bool:
        """Windows 的 `py.exe` / `pyw.exe` 是启动器，不是解释器。

        ⚠ 启动器的解释器选择**依赖被运行文件的路径/关联**（实测：临时目录里的
        `.py` 用对解释器，仓库 `tests/gates/*.py` 却关联到无 numpy 的 3.13）
        → **无法可靠预判** → 干脆**不选启动器**（除非没有别的候选）。
        """
        base = os.path.basename(p).lower()
        return base in ("py.exe", "pyw.exe")

    cands: list[str] = []
    # ── 第一梯队：**真实解释器**（按文件关联可靠）────────────────────────
    if sys.executable and not _is_launcher(sys.executable):
        cands.append(sys.executable)
    for name in ("python3", "python"):
        p = shutil.which(name)
        if p and not _is_launcher(p):
            cands.append(p)
    # ── 第一梯队补充：**常见安装位置**（PATH 里的 python 可能是「壳」
    #   —— 实测 WorkBuddy 内置的 3.13 无 numpy，而真解释器在
    #   %LOCALAPPDATA%\Programs\Python\PythonXY\python.exe）──────────────
    for pat in (
        r"%LOCALAPPDATA%\Programs\Python\Python*\python.exe",
        r"C:\Python*\python.exe",
        r"%PROGRAMFILES%\Python*\python.exe",
        "/usr/bin/python3",
        "/usr/local/bin/python3",
    ):
        try:
            import glob
            for p in sorted(glob.glob(os.path.expandvars(pat)), reverse=True):
                if os.path.isfile(p):
                    cands.append(p)
        except Exception:                                   # noqa: BLE001
            continue
    # ── 第二梯队：启动器（**只作兜底**）─────────────────────────────────
    #   它的解释器选择依赖被运行文件的路径/关联 → 不可靠。
    if sys.executable:
        cands.append(sys.executable)
    p = shutil.which("py")
    if p:
        cands.append(p)

    seen: set[str] = set()
    for exe in cands:
        key = (os.path.normcase(os.path.abspath(exe))
               if os.path.isfile(exe) else exe)
        if key in seen:
            continue
        seen.add(key)
        if has_numpy(exe):
            return exe
    return sys.executable or "python"
