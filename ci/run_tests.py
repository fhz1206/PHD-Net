"""CI 测试流水线入口（GitHub Actions / GitCode 流水线共用）。

用法：bash ci/run_tests.sh [--quick]
  完整档 = fast 回归 + 全部逐位对拍 + 精度验证（L1/L2/L3-4K）
  --quick = 仅 fast 回归（≈1–2 min，供 PR 预检）
退出码非 0 即失败。所有测试不依赖 GPU/网络；冻结评测语料随仓库分发
（eval_corpus/，为评测基准而非训练数据集）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

STEPS_FULL = [
    ["python", "tests/run_tests.py", "fast"],
    ["python", "tools/verify_seg_equiv.py"],
    ["python", "train_1b/verify_vocab_parallel.py"],
    ["python", "tools/audit_precision.py"],
]
STEPS_QUICK = [["python", "tests/run_tests.py", "fast"]]


def main() -> int:
    quick = "--quick" in sys.argv
    steps = STEPS_QUICK if quick else STEPS_FULL
    for cmd in steps:
        print(f"\n===== $ {' '.join(cmd)} =====", flush=True)
        r = subprocess.run(cmd, cwd=_ROOT)
        if r.returncode != 0:
            print(f"\n[CI] FAILED: {' '.join(cmd)}", flush=True)
            return r.returncode
    print("\n[CI] ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
