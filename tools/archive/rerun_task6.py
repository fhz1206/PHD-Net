"""单任务重跑：eval_suite task6 效率（P6 性能修复后同口径复测，2026-09-21）。

仅跑 task6_efficiency（训练全语料 80% 段），口径与 eval_suite 完全一致：
ms/token = train_stream 墙钟 / tokenize token 数。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests"))

from eval_common import DOC  # noqa: E402
from eval_tasks_perf import task6_efficiency  # noqa: E402

with open(DOC, encoding="utf-8") as f:
    text = f.read()
split = int(len(text) * 0.8)
print(f"语料 {len(text):,} 字符（训练段 {split:,}）")
task6_efficiency(text[:split], text)
