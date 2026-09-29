"""分层回归总入口：python tests/run_tests.py [--fast | --full]

fast（默认，< 1 分钟）：版本一致性、核心三实验、容量层小规模、词级 LM 冒烟、
                       大容量 LTM 冒烟、长程复制、后端三项、开关可构造。
full：叠加全部耗时验收脚本（demo_lm / demo_m1 … demo_m4 / demo_strength / eval_suite）。

向后兼容：旧的位置参数 `fast`（`python tests/run_tests.py fast`）仍等价于
`--fast`；未给任何参数时默认 fast。

检查项按类目拆分（结构整理 2026-09-18）：
    tests/checks_common.py   常量、结果收集与断言辅助
    tests/checks_core.py     核心行为检查
    tests/checks_backend.py  硬件后端检查
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS = ROOT / "tests"
PY = sys.executable

from checks_backend import t_backend_probe
from checks_common import ENV, TESTS, _check, _results, reset_results
from checks_core import (t_bigltm_smoke, t_brain_homologues, t_capacity_small,
                         t_core_demo, t_copy_task, t_switches_construct,
                         t_version, t_word_lm_smoke)

FULL_SCRIPTS = ["demo_lm.py", "demo_m1.py", "demo_m2.py", "demo_m3.py",
                "demo_m4.py", "demo_m9.py", "demo_strength.py", "eval_suite.py"]


def run_full() -> None:
    """全量层：逐个子进程运行耗时验收脚本，返回码非 0 即判定失败。"""
    for s in FULL_SCRIPTS:
        r = subprocess.run([PY, str(TESTS / s)], capture_output=True, text=True,
                           cwd=str(ROOT), env=ENV, encoding="utf-8",
                           errors="replace")
        _check(f"full: {s}", r.returncode == 0,
               (r.stdout or "").strip().splitlines()[-1][:60] if r.stdout else "")


def parse_mode(argv: list[str]) -> str:
    """解析 fast/full 模式。

    显式标志 `--fast` / `--full` 优先；未给标志时回落到旧的位置参数
    （`fast` / `full` 皆可，忽略之），两者都没有则默认 fast。
    """
    ap = argparse.ArgumentParser(
        prog="tests/run_tests.py",
        description="PHD-Net 分层回归测试（fast 默认；--full 叠加耗时验收脚本）")
    ap.add_argument("--fast", action="store_true", help="仅快速回归（默认）")
    ap.add_argument("--full", action="store_true", help="快速回归 + 全部耗时验收脚本")
    ap.add_argument("legacy", nargs="*", metavar="LEGACY",
                    help="旧版位置参数（fast/full），仅为向后兼容保留")
    ns = ap.parse_args(argv)
    if ns.full:
        return "full"
    if ns.fast:
        return "fast"
    return "full" if "full" in ns.legacy else "fast"


def main() -> None:
    mode = parse_mode(sys.argv[1:])
    reset_results()                                  # C7：清空遗留累积结果
    print("=" * 66)
    print("PHD-Net 回归测试（" + mode + "）")
    print("=" * 66)
    t_version()
    t_core_demo()
    t_capacity_small()
    t_word_lm_smoke()
    t_bigltm_smoke()
    t_copy_task()
    t_backend_probe()
    t_switches_construct()
    t_brain_homologues()
    if mode == "full":
        run_full()
    n_ok = sum(1 for _, ok, _ in _results if ok)
    print("-" * 66)
    print(f"通过 {n_ok}/{len(_results)}")
    if n_ok != len(_results):
        for name, ok, detail in _results:
            if not ok:
                print(f"  失败项: {name} [{detail}]")
    sys.exit(0 if n_ok == len(_results) else 1)


if __name__ == "__main__":
    main()
