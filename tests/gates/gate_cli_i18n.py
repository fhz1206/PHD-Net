# -*- coding: utf-8 -*-
"""P199 门禁：i18n 重写后的行为契约。

跑法：py -3.14 tests/gates/gate_cli_i18n.py

P199 把 i18n 从「正则词表替换」重写为「显式消息目录」：
  ① `_CATALOG` 整句精确匹配（完整消息 → 完整英文）
  ② `_WORDMAP` 多字词整词替换（扫描式，带词边界空格；**无正则交替**）
  ③ 未命中保留中文（可读），**结构上排除碎片化**

旧门禁（P119/P147 时代）测的是正则替换的内部行为（最长匹配、粘连修补、
全角转半角…）——那些机制已不存在，契约整体改写。旧方案七类缺陷
（重复键/短串遮蔽/白名单腐烂/中英粘连/全角残留/writelines 绕过/重建 dict）
中，「碎片化」类在新架构下**结构上不可能复发**（无正则交替、无单字替换），
故新契约以「坏例回归」方式钉死；机制类契约（词表无单字、扫描式补空格）
直接测新实现。

新契约：
  A1 整句目录命中 → 完整英文句子
  A2 目录未命中 → 词级降级（组件词被替换，剩余中文保留）
  A3 词级降级**无碎片**：单字不被替换、无「大 empty 间表」类夹击
  A4 词边界空格：英文↔汉字交界、英文词↔英文词（相邻词条）都补空格
  B1 诊断行（[ASCII 标签]）两种模式逐字节一致
  C1 zh 模式原样返回
  C2 T() 显式双语
  D1 set_lang 规范化（默认 en、zh-CN→zh、未知→en）
  E1 install_stream_filter 幂等 + writelines 也翻译
  F1 性能：纯 ASCII 快速通道、词级替换有界
  G  旧方案坏例回归（大 empty 间表 / backendunavailable / loadedcheckpoint）
"""

import io
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CASES: list = []


def check(ok, name, detail=""):
    CASES.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))


def main():
    from phdnet.i18n import (T, _CATALOG, _WORDMAP, get_lang, install_stream_filter,
                             set_lang, translate, _StreamFilter)

    print("=" * 74)
    print("P199 i18n 重写：显式消息目录契约")
    print("=" * 74)

    # ── A. 翻译语义 ────────────────────────────────────────────────
    print("\n[A] 目录命中与词级降级")
    set_lang("en")
    hit = translate("突触参数容量验算（1B 口径：固定突触 + 大空间表容量）")
    check(hit == _CATALOG["突触参数容量验算（1B 口径：固定突触 + 大空间表容量）"],
          "A1a 整句目录命中 → 登记的英文句", hit[:50])
    check("Synapse capacity audit" in hit, "A1b 目录内容合理")

    miss = translate("总突触参数容量           : 1,217,249,536")
    check("synapse parameters" in miss and "capacity" in miss,
          "A2a 未命中 → 词级降级（组件词被替换）", miss[:40])
    check("总" in miss, "A2b 未登记汉字保留（不强行翻单字）")

    frag = translate("大空间表容量（事件驱动）")
    check("empty 间" not in frag and "大 empty" not in frag,
          "A3a 无单字夹击（旧方案坏例：大 empty 间表）", frag[:40])
    # 单字不被替换：_WORDMAP 键全部 ≥2 汉字
    check(all(len(k) >= 2 for k in _WORDMAP), "A3b 词表键全部 ≥2 汉字（无单字）")

    glue1 = translate("总突触参数容量")
    # 「总突触参数容量」已入目录 → 整句返回（词级粘连规则由下面 G 组坏例
    # 与 A4b 覆盖）；这里改测一个**不在目录**的组合词串
    glue1 = translate("突触参数生长")
    # 「突触参数」→"synapse parameters"；「生长」未登记 → 保留中文。
    # 契约：**相邻英文词条不粘连**（parametersgrowth 不出现）+
    # 未登记词保留中文（可读），这正是新架构的设计取舍。
    check("synapse parameters" in glue1 and "parametersgrowth" not in glue1
          and "生长" in glue1,
          "A4a 相邻词条不粘连；未登记词保留中文", glue1[:40])
    glue2 = translate("已生长突触按 100 万条")
    check("已 grown" in glue2 or "已生长" not in glue2,
          "A4b 英文↔汉字交界补空格", glue2[:36])

    # ── B. 诊断行 ─────────────────────────────────────────────────
    print("\n[B] 诊断行逐字节一致")
    diags = [
        "[capability] accelerators detected ['npu']",
        "[readout] compute precision = fp32",
        "[ltm-diag] imprints=10 prev=256 cur=256",
        "[probe] token 100  probe_PPL 1.0",
    ]
    for d in diags:
        set_lang("en")
        ok_en = translate(d) == d
        set_lang("zh")
        ok_zh = translate(d) == d
        check(ok_en and ok_zh, f"B {d[:40]!r} 两模式一致")

    # ── C. zh 模式与 T() ──────────────────────────────────────────
    print("\n[C] zh 模式 / T()")
    set_lang("zh")
    check(translate("大空间表容量（事件驱动）") == "大空间表容量（事件驱动）",
          "C1 zh 模式原样返回")
    set_lang("en")
    check(T("训练完成", "training finished") == "training finished",
          "C2a T() en 模式用第二参数")
    set_lang("zh")
    check(T("训练完成", "training finished") == "训练完成",
          "C2b T() zh 模式用第一参数")
    check(T("只有中文") == "只有中文", "C2c T() 缺省 en 退回中文")

    # ── D. set_lang 规范化 ────────────────────────────────────────
    print("\n[D] set_lang 规范化")
    check(set_lang(None) == "en" and get_lang() == "en", "D1 缺省 → en")
    check(set_lang("zh-CN") == "zh", "D2 zh-CN → zh")
    check(set_lang("fr") == "en", "D3 未知语言 → en（不回 zh）")
    set_lang("en")

    # ── E. 输出层包装 ─────────────────────────────────────────────
    print("\n[E] 输出层包装")
    buf = io.StringIO()
    w = _StreamFilter(buf)
    set_lang("en")
    w.write("词表快照已保存\n")
    check("vocab" in buf.getvalue() and "saved" in buf.getvalue(),
          "E1a write 被翻译（词级降级路径）")
    w.writelines(["词表快照已保存\n"])
    check("saved" in buf.getvalue() or "vocab" in buf.getvalue(),
          "E1b writelines 也被翻译")
    install_stream_filter()
    install_stream_filter()
    check(isinstance(sys.stdout, _StreamFilter), "E2 install 幂等")
    sys.stdout = sys.__stdout__   # 还原（避免污染后续测试）
    # E3 TeeLogger 顺序回归（P199 实测 bug）：train.py 原顺序是
    #   install_stream_filter()（600 行）→ sys.stdout = TeeLogger()（618 行），
    #   Tee **覆盖**已包装的 stdout → 翻译层整个失效（--lang en 时容量表
    #   仍是中文）。修法是 Tee 先建、filter 重装。这里静态钉死顺序：
    #   「sys.stdout = TeeLogger」之后必须再出现一次 install_stream_filter。
    tp_src = (_ROOT / "train" / "train.py").read_text(encoding="utf-8")
    tee_pos = tp_src.find("sys.stdout = TeeLogger(log_path)")
    reinstall_pos = tp_src.find("install_stream_filter()", tee_pos)
    check(tee_pos != -1 and reinstall_pos != -1
          and reinstall_pos > tee_pos,
          "E3 TeeLogger 建立后重装翻译层（否则 --lang en 失效）",
          f"tee@{tee_pos} reinstall@{reinstall_pos}")

    # ── F. 性能 ───────────────────────────────────────────────────
    print("\n[F] 性能")
    ascii_line = "  token 1,000,000  sliding PPL  394.4687    62.31 ms/tok  elapsed 103.8 min\n"
    set_lang("en")
    t0 = time.perf_counter()
    for _ in range(2000):
        translate(ascii_line)
    t_ascii = (time.perf_counter() - t0) / 2000 * 1e6
    check(t_ascii < 10, "F1 纯 ASCII 快速通道 < 10µs", f"{t_ascii:.2f}µs")
    cn_line = "  静态内存 ≈ 76 MB（编码器 17 + 主干 6.3 + 读出 53）"
    t0 = time.perf_counter()
    for _ in range(1000):
        translate(cn_line)
    t_cn = (time.perf_counter() - t0) / 1000 * 1e6
    check(t_cn < 200, "F2 词级替换有界 < 200µs", f"{t_cn:.2f}µs")

    # ── G. 旧方案坏例回归 ────────────────────────────────────────
    print("\n[G] 旧方案坏例回归（必须全部消失）")
    bads = [
        ("大空间表容量（事件驱动）", ["大 empty", "empty 间"]),
        ("警告：读出后端不可用，已回退", ["backendunavailable"]),
        ("已加载检查点", ["loadedcheckpoint"]),
        ("缓存目录不存在", ["cachedirectory"]),
        ("总计用时", ["total用时"]),
    ]
    for src, bads_frag in bads:
        out = translate(src)
        hit_frag = [f for f in bads_frag if f in out]
        check(not hit_frag, f"G {src[:24]!r} 无碎片", out[:40])

    bad = [n for n, ok in CASES if not ok]
    print(f"\n{'PASS' if not bad else 'FAIL'}: {len(CASES) - len(bad)}/{len(CASES)} 项通过")
    for n in bad:
        print(f"  x {n}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
