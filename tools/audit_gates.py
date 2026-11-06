#!/usr/bin/env python3
"""门禁有效性静态审计：找出**可能假绿**的 `gate_*.py`。

## 为什么需要

P215 审计实测坐实过 4 个假绿：业务断言失败只 `print`、进程仍 `rc=0`，
`tools/run_gates.py` 与 CI 都判 PASS。其中 `gate_readout_fused` 甚至会打印
`→ FAIL`，`gate_cann_probe` 会打印 `失败 ['...']`，但退出码仍是 0。
这类门禁**等于没有把关**。

⚠ 检测方法本身踩过的坑（别重犯）：
- 把注入后的临时脚本写到系统 Temp → 门禁里 `Path(__file__).parents[2]`
  定位仓库根失效 → **所有门禁都因路径错误 rc=1**，看起来「都能检出」，
  是彻头彻尾的假阳性。动态注入必须把临时文件放在 `tests/gates/` 内。
- 「强制抛异常能检出」≠「业务断言失败能检出」。必须测**后者**。

## 本工具做什么（纯静态，秒级）

逐个扫描 `tests/gates/gate_*.py`，按下面 3 条**高置信度**规则打标：

| 规则 | 可疑信号 |
|---|---|
| `NO_EXIT` | 全文没有 `sys.exit` / `SystemExit` → 业务失败无法变成退出码 |
| `BARE_ASSERT` | 用裸 `assert` 且无显式退出码（`python -O` 会整段跳过） |
| `PRINT_ONLY` | 出现 `FAIL` 字样但同一文件无退出机制 |
| `NO_ASSERTION` | 既没有 `check(` / `_RESULTS` / `assert` / `raise`，也没有失败计数 |

⚠ **刻意不用**「`main()` 里有 `return 0`」当信号：项目大量门禁用
   `return 0 if n_ok == len(_RESULTS) else 1` + `raise SystemExit(main())`，
   粗正则会把条件表达式误读成「恒返回 0」→ 实测 11 个假阳性。

输出可疑清单 + 建议；`--strict` 时有问题返回 1（可挂 CI）。

## 用法

```bash
python tools/audit_gates.py                 # 审计全部
python tools/audit_gates.py --strict       # 有问题 → rc=1（CI 用）
python tools/audit_gates.py --json         # 机器可读
```

⚠ 静态扫描是**初筛**（会有假阳性，例如某些门禁用 `check()` 汇总且
`_finish()` 内部 `sys.exit`）。命中后必须**人工动态复核**：
把业务断言强制改成 False，确认 `rc != 0`。修好的 4 个门禁已在
`tests/gates/` 里带 `P215 审计修复` 注释，可作范例。

真正的运行时回归保障是 `tools/run_gates.py`（它按退出码判定）。
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_GATE_DIR = _ROOT / "tests" / "gates"


def audit(path: Path) -> dict:
    src = path.read_text(encoding="utf-8", errors="replace")
    flags: list[str] = []
    notes: list[str] = []

    has_exit = bool(re.search(r"sys\.exit|SystemExit", src))
    has_assert = bool(re.search(r"^\s*assert\s", src, flags=re.M))
    has_check = bool(re.search(r"def check\(|_RESULTS", src))
    has_raise = bool(re.search(r"^\s*raise\s", src, flags=re.M))
    # ✅ 有效的汇总模式（这些会让 CONST_RETURN / NO_ASSERTION 变成假阳性）：
    #   ① `_finish()` / `check()` 模板，且退出码由汇总结果决定
    #   ② 手工计数 `n_fail` + `sys.exit(1 if n_fail else 0)`
    has_fail_counter = bool(re.search(r"n_fail|num_fail|\bFAIL\b.*\+=", src))
    has_conditional_exit = bool(
        re.search(r"sys\.exit\([^)]*(?:if|else)", src)
        or re.search(r"SystemExit\([^)]*(?:if|else)", src))
    # `_finish` 函数体内是否带退出
    fin = re.search(r"def _finish\([\s\S]{0,400}", src)
    finish_exits = bool(fin and re.search(r"sys\.exit|SystemExit",
                                          fin.group(0)))
    summary_ok = bool(finish_exits or has_conditional_exit
                      or (has_fail_counter and has_exit))
    main_block = src[src.rfind("if __name__"):] if "if __name__" in src else ""

    if not has_exit:
        flags.append("NO_EXIT")
        notes.append("全文无 sys.exit/SystemExit → 业务失败无法变成退出码")
    if has_assert and not has_exit:
        flags.append("BARE_ASSERT")
        notes.append("裸 assert 且无显式退出码：python -O 下会被整段跳过")

    # ⚠ 不用「main() 里有 return 0」当信号：项目大量门禁用
    #   `return 0 if n_ok == len(_RESULTS) else 1` + `raise SystemExit(main())`，
    #   粗正则会把条件表达式误读成「恒返回 0」→ 11 个假阳性（P215 实测）。
    #   真假绿的特征是**全文无退出机制**或**打印 FAIL 却无退出**，不是 return 0。

    if not (has_check or has_assert or has_raise or has_fail_counter
            or re.search(r"\ball_ok\b|\bn_ok\b|\bok_all\b", src)):
        flags.append("NO_ASSERTION")
        notes.append("找不到任何断言/结果收集 → 可能只是打印型记录工具")
    if "FAIL" in src and not has_exit:
        flags.append("PRINT_ONLY")
        notes.append("打印 FAIL 但没有退出机制 → 典型的「假绿」")

    return {
        "gate": path.stem,
        "flags": flags,
        "suspicious": bool(flags),
        "notes": notes,
        "has_exit": has_exit,
        "has_check": has_check,
        "summary_ok": summary_ok,
        "main_block": main_block.strip().splitlines()[-1][:70]
                       if main_block.strip() else "(无 __main__)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="发现可疑项时返回 1（CI 用）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    files = sorted(_GATE_DIR.glob("gate_*.py"))
    rows = [audit(p) for p in files]
    bad = [r for r in rows if r["suspicious"]]

    if args.json:
        print(json.dumps({"total": len(rows), "suspicious": len(bad),
                          "rows": rows}, ensure_ascii=False, indent=2))
        return 1 if (args.strict and bad) else 0

    print("=" * 76)
    print("门禁有效性静态审计（初筛）")
    print("=" * 76)
    print(f"  扫描 {len(rows)} 个门禁，可疑 {len(bad)} 个")
    if bad:
        print("\n  可疑清单（⚠ 需人工动态复核，静态扫描会有假阳性）：")
        for r in bad:
            print(f"    · {r['gate']:42s} {','.join(r['flags'])}")
            for n in r["notes"]:
                print(f"        {n}")
    else:
        print("\n  ✅ 未发现无退出机制的门禁")

    print("\n  复核方法（必须做，静态不算数）：")
    print("    把门禁的业务断言强制改成 False，跑一遍，确认 rc != 0。")
    print("    ⚠ 临时脚本要放在 tests/gates/ 内，否则 __file__ 定位仓库根会失效。")
    print("=" * 76)
    return 1 if (args.strict and bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
