"""门禁：终端 i18n（phdnet/i18n.py）—— exit 0 = PASS。

为什么要有这条门禁（2026-10-07 审计，均已实测复现过）：
  ① `_TABLE` 里有**重复键**（读出计算精度/读出后端/完成 各两次；「平均」两次且值不同
     "mean" vs "avg"）→ dict 后写覆盖前写，前面的是死条目；
  ② `_CN` 的正则**取第一条匹配而非最长匹配**，而注释声称「按长串优先排序」——
     实际没排序 → 短串遮蔽长串：「总计用时」→"total用时"、「缓存目录」→"cachedirectory"、
     「已加载」→"already load"；
  ③ 诊断行白名单**必然腐烂**：运行期的 `[precision] ...` 不在名单里，被逐词翻成
     "run时提升"（P147 记录过的失败模式复发）；
  ④ 中英粘连：「读出后端不可用」→"readout backendunavailable"；
  ⑤ 全角标点残留：「警告：…，…」→"warning：...,..."；
  ⑥ `sys.stdout.writelines` 未被包装 → 半翻半不翻；
  ⑦ 每次 write 都重建 dict（27.97 µs/次，train.py 每步多次 print）。
这七条都曾**没有任何门禁**拦住（verify_lang_semantics 只测「切语言不影响数值」）。

用法：py -3.14 tests/verifiers/verify_i18n.py
"""
from __future__ import annotations

import io
import sys
import timeit
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet import i18n                                    # noqa: E402

_RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def main() -> int:
    print("=" * 78)
    print("门禁：终端 i18n（最长匹配 / 诊断行 / 粘连 / 标点 / 性能）")
    print("=" * 78)

    i18n.set_lang("en")

    # ── A. 最长匹配优先（修前被短串遮蔽）────────────────────────────────
    print("\n[A] 最长匹配优先")
    cases = [
        ("总计用时 12.3 秒", "total time 12.3 s"),
        ("缓存目录不存在", "cache dir missing"),
        ("已加载检查点 3 个", "loaded checkpoint 3"),
    ]
    for src, want in cases:
        got = i18n.translate(src)
        check(f"A  {src}", want in got, f"→ {got!r}")

    # ── B. 无重复键（修前 平均 两条值不同）──────────────────────────────
    print("\n[B] 词表健康")
    from collections import Counter
    dup = {k: v for k, v in Counter(k for k, _ in i18n._TABLE).items() if v > 1}
    check("B1 _TABLE 无重复键", not dup, str(dup))
    check("B2 _MAP 与 _TABLE 键数一致（去重后无丢失）",
          len(i18n._MAP) == len(set(k for k, _ in i18n._TABLE)),
          f"{len(i18n._MAP)} vs {len(set(k for k, _ in i18n._TABLE))}")
    # 最长优先的**行为不变量**：单独把每个词条丢进去，必须命中它**自己**的译文，
    # 而不是被某个更短的键先吃掉（修前「总计用时」会被「总计」遮蔽 → "total用时"）。
    # 注：不检查 pattern 字面顺序 —— re.escape 会转义空格等，前缀比较是脆弱断言。
    shadowed = [k for k in i18n._MAP if i18n.translate(k) != i18n._MAP[k]]
    check("B3 每个词条单独输入都命中自身译文（无短串遮蔽）",
          not shadowed, f"被遮蔽 {len(shadowed)} 个：{shadowed[:6]}")

    # ── C. 诊断行整行不翻（修前被逐词翻烂）──────────────────────────────
    print("\n[C] 诊断行")
    diag = [
        "[precision] 读出权重 fp16 请求 → 运行时提升为 fp32",
        "[ltm-diag] recalls=10 active=16",
        "[rocm-env] 设置失败（不致命）：OSError: x",
        "  [probe] 稀疏路径 torch.compile 不可用",
    ]
    for s in diag:
        check(f"C  {s[:34]!r}… 原样", i18n.translate(s) == s, "")
    # 但普通句子里的中括号标签之外的中文要正常翻
    check("C  非诊断行仍翻译", i18n.translate("训练完成") == "training finished",
          repr(i18n.translate("训练完成")))

    # ── D. 粘连与标点 ──────────────────────────────────────────────────
    print("\n[D] 粘连与标点")
    out = i18n.translate("读出后端不可用，已回退")
    check("D1 无中英粘连（backend unavailable）", "backend unavailable" in out, repr(out))
    check("D2 全角逗号转半角", "，" not in out and "," in out, repr(out))
    out2 = i18n.translate("警告：训练完成，最终 PPL 123.4")
    check("D3 全角冒号转半角且无双空格", "warning: training finished" in out2
          and "  " not in out2, repr(out2))
    out3 = i18n.translate("3 次迭代后仍未收敛")
    check("D4 「次」译为 times 而非 x", " times" in out3 and "x迭" not in out3, repr(out3))
    # —— 2026-10-07 交叉验证追加的三条边界 ——
    out4 = i18n.translate("已加载 3 个检查点")
    check("D5 空译文（个→""）不再产生双空格", "  " not in out4
          and "loaded 3 checkpoint" in out4, repr(out4))
    out5 = i18n.translate("训练完成。")
    check("D6 句末句号不带行尾空格", out5.endswith(".") and not out5.endswith(" "),
          repr(out5))
    out6 = i18n.translate("训练完成。下一段开始")
    check("D7 句号后紧跟汉字时补一个空格", ". " in out6, repr(out6))
    out7 = i18n.translate("迭代 3 次完成")
    check("D8 「迭代」有词条（不残留中文）", "iteration" in out7 and "迭代" not in out7,
          repr(out7))

    # ── E. 不该被动的东西 ──────────────────────────────────────────────
    print("\n[E] 保真（对齐表格 / 纯中文 / 数值）")
    aligned = "readout_k16       465.7766  -15.27%     1.401 ms"
    check("E1 纯 ASCII 对齐表格逐字节不变", i18n.translate(aligned) == aligned, "")
    pure = "纯中文没有任何英文字母的句子"
    check("E2 纯中文未命中词条时原样返回", i18n.translate(pure) == pure, "")
    num = "PPL=439.6 tokens=20829"
    check("E3 数值串不变", i18n.translate(num) == num, "")

    # ── F. writelines / 幂等 / 性能 ────────────────────────────────────
    print("\n[F] 包装器与性能")
    buf = io.StringIO()
    f = i18n._StreamFilter(buf)
    f.writelines(["训练完成\n", "最终 PPL 123.4\n"])
    check("F1 writelines 也被翻译（修前绕过）",
          "training finished" in buf.getvalue() and "最终" not in buf.getvalue(),
          repr(buf.getvalue()))

    saved = (sys.stdout, sys.stderr)
    try:
        i18n.install_stream_filter()
        i18n.install_stream_filter()          # 幂等：二次安装不得套两层
        check("F2 install_stream_filter 幂等",
              isinstance(sys.stdout, i18n._StreamFilter)
              and type(sys.stdout._w) is not i18n._StreamFilter, "")
    finally:
        sys.stdout, sys.stderr = saved

    ascii_t = timeit.timeit(lambda: i18n.translate(aligned), number=2000) / 2000
    cn_t = timeit.timeit(lambda: i18n.translate("训练完成，最终 PPL 123.4"), number=2000) / 2000
    check("F3 纯 ASCII 快速通道 < 10µs（修前 27.97µs）", ascii_t < 10e-6,
          f"{ascii_t * 1e6:.2f}µs")
    check("F4 中文行 < 60µs", cn_t < 60e-6, f"{cn_t * 1e6:.2f}µs")

    # ── G. 语言切换语义 ────────────────────────────────────────────────
    print("\n[G] set_lang 边界")
    check("G1 默认 en", i18n.set_lang(None) == "en", i18n.get_lang())
    check("G2 zh-CN → zh", i18n.set_lang("zh-CN") == "zh", i18n.get_lang())
    check("G3 未知语言回 en（不回 zh）", i18n.set_lang("不存在的语言") == "en", i18n.get_lang())
    i18n.set_lang("zh")
    check("G4 zh 模式原样返回", i18n.translate("训练完成") == "训练完成", "")
    i18n.set_lang("en")

    ok = sum(1 for r in _RESULTS if r[0])
    bad = [r[1] for r in _RESULTS if not r[0]]
    print("\n" + "=" * 78)
    print(f"结果：{ok}/{len(_RESULTS)} 通过" + (f" | 失败 {bad}" if bad else ""))
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
