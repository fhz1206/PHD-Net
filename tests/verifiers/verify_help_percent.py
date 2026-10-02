"""门禁：argparse 的 `help=` 里**不得有裸 `%`**（BUGS #2 家族，已犯 5 次）。

症状：`ValueError: badly formed help string`（argparse 对 help 做 %-格式化）。

为什么值得单列一条
================================================================================
这个项目我已经**犯了 5 次**：每次新加 help 文案时顺手打了个 `%`（如
「占 81.0%」「保留率 26.67%」），`--help` 直接崩。2026-10-02 的 P146 又犯一次。
`verify_ms_stream` 已有「跑一遍所有入口的 --help」的门禁，**能抓到**，
但它只在有人真跑时有效，且报错信息（ValueError）**不指向具体参数**，
排查成本高。

本门禁做的是**静态定位**：用 AST 找出每个 `add_argument(help=...)` 的字符串，
报告「哪一个参数、哪一行、哪个 %」。比跑 --help 更快更精确。

⚠ 只检查 **Python 字符串常量**的 help（`--help` 崩的都是这种）；
   拼接出来的动态 help 查不了 —— 那种靠 --help 门禁兜。
"""
from __future__ import annotations

import ast
import io
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
# 裸 % = 不是 %%、也不是 %s/%d/%r/%f/%e/%g/宽度格式
_BARE = re.compile(r"%(?![%sdfreEgGxXoc%])")


def _help_consts(tree: ast.AST):
    """产出 (参数名, 行号, help 字符串) —— 只取字符串常量。"""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        fname = getattr(fn, "attr", getattr(fn, "id", ""))
        if fname not in ("add_argument", "add_parser"):
            continue
        # 参数名（第一个位置参数）
        pname = "?"
        if node.args and isinstance(node.args[0], ast.Constant):
            pname = str(node.args[0].value)
        for kw in node.keywords:
            if kw.arg == "help" and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str):
                yield pname, kw.value.lineno, kw.value.value


def main() -> int:
    files = []
    for sub in ("train", "tools", "phdnet", "phdnet/backends", "tests/verifiers"):
        d = _ROOT / sub
        if d.is_dir():
            files += sorted(d.glob("*.py"))
    bad = []
    checked = 0
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for pname, lineno, s in _help_consts(tree):
            checked += 1
            probe = s.replace("%%", "\x00")      # 已转义的先占位
            for m in _BARE.finditer(probe):
                bad.append((f.relative_to(_ROOT), lineno, pname,
                            s[max(0, m.start() - 30):m.start() + 12]))
    print("=" * 78)
    print("门禁：argparse help 里的裸 %（BUGS #2 家族，已犯 5 次）")
    print("=" * 78)
    if not bad:
        print(f"  PASS  扫描 {len(files)} 个文件 / {checked} 个 help 字符串：**无裸 %**")
        return 0
    print(f"  FAIL  发现 {len(bad)} 处裸 %（`--help` 会 ValueError）：")
    for rel, lineno, pname, ctx in bad[:10]:
        print(f"    {rel}:{lineno}  参数 {pname}")
        print(f"        ...{ctx}...")
    print("\n  修法：把该 % 写成 %%（argparse 的 help 会做 %-格式化）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())