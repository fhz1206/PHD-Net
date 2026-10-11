"""回归检查共享件 —— 常量、结果收集与断言辅助（自 run_tests.py 拆分）。"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # 项目根目录
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))                   # 保证 `import phdnet` 可用
TESTS = ROOT / "tests"
# 子目录按功能分类后仍用平铺导入（`from eval_common import …`、
# `from demo_delayed_copy import …`），故把三个子目录一并放进 sys.path。
# 这是分类布局**唯一**的跨模块契约点：改这里即可，不必改每个调用方。
for _sub in ("demos", "evals", "gates"):
    _p = str(TESTS / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
DOC = ROOT / "eval_corpus" / "internal_corpus.txt"
PY = sys.executable
# 子进程也要能平铺导入 → PYTHONPATH 必须带上同样的目录
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(
    [str(ROOT), str(TESTS / "demos"), str(TESTS / "evals"), str(TESTS / "gates")]),
    "PYTHONUTF8": "1"}

_results: list[tuple[str, bool, str]] = []


def reset_results() -> None:
    """C7 修复：清空累积结果，防止同进程重复运行（如测试框架复用）时重复计数。"""
    _results.clear()


def _check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f"  [{detail}]" if detail else ""))
