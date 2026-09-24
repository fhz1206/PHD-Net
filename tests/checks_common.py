"""回归检查共享件 —— 常量、结果收集与断言辅助（自 run_tests.py 拆分）。"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # 项目根目录
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))                   # 保证 `import phdnet` 可用
TESTS = ROOT / "tests"
DOC = ROOT / "datasets" / "eval" / "internal_corpus.txt"
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUTF8": "1"}

_results: list[tuple[str, bool, str]] = []


def reset_results() -> None:
    """C7 修复：清空累积结果，防止同进程重复运行（如测试框架复用）时重复计数。"""
    _results.clear()


def _check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f"  [{detail}]" if detail else ""))
