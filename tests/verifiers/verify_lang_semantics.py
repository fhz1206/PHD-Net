"""P119：`--lang` 语义重定义的门禁 —— **只管终端文案，不影响训练结果**。

背景
================================================================================
P119 之前 `--lang` 同时背两个职责：
  ① 终端输出语言（`set_lang`）
  ② **训练语料语言过滤**（按 parquet 的 `lang` 列筛训练流）——**会改变训练结果**
两者语义完全无关，却共用一个参数名 → 极其危险：想改输出语言却改了训练数据。

P119（fhz 2026-10-01）重新定义：`--lang` **只管终端文案**，
**默认英文**（此前默认中文），且**对训练结果零影响**；
数据过滤职责拆到独立参数 `--data-lang`。

四层验证
================================================================================
A. 零回归（最重要）：`--lang` 不改变任何模型数值
   同一 seed、同一输入下跑两种 `--lang`，比对**权重与 PPL 轨迹逐位相同**。
   ⚠ 这是本门禁的核心断言——若它不成立，「lang 不影响训练结果」就是空话。
B. 输出确实切换了：默认英文、`--lang zh` 中文。
   ⚠ 注意必须用**不会被翻译的数字**做探针，否则测的是翻译器而非语言设置。
C. `--data-lang` 仍能真正过滤语料（拆出去的职责没丢）。
D. 三处入口（train/infer/convert_deepctrl）默认值一致且都是 `en`；
   `--help` 可执行。
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in ("", "tests", "tools", "train"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

_RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    return bool(ok)


# ── A. --lang 不影响任何数值 ────────────────────────────────────────────────
def _run_tiny(lang: str, n_steps: int = 120):
    """用给定终端语言跑一小段训练，返回 (逐步 NLL 列表, 读出权重指纹)。

    ⚠ 为什么不直接用 `lm.train_stream`：它**每段只返回一个聚合 NLL**
    （实测 200 token → 1 个值），那样的「轨迹逐位相同」断言几乎没有强度
    ——第一版就踩了这个坑（1 步也算PASS，等于什么都没验）。
    这里改为**直接逐 token驱动 `net.step`**，每步的 nll 都进轨迹，
    指纹也改成覆盖读出 + 主干两组权重。

    语言切换只经`phdnet.i18n`，不触碰训练路径 —— 本用例断言的正是
    「切换终端语言后训练结果不变」。
    """
    import numpy as np
    from eval_common import BASE, SEG, DOC
    from phdnet.i18n import install_stream_filter, set_lang
    from phdnet.word_lm import PHDWordLM
    from phdnet.config import PHDNetConfig

    set_lang(lang)
    install_stream_filter()
    txt = io.open(DOC, encoding="utf-8").read()
    cfg = PHDNetConfig(**{**BASE, "readout_dtype": "fp32"})
    lm = PHDWordLM(txt, cfg, seg_kwargs=SEG)
    toks = lm.tokenize(txt)[:n_steps + 2]

    #逐 token 驱动（不经 train_stream，拿到逐步 nll）
    ids = [lm.tok.stoi.get(t) for t in toks]
    nll_trace: list[float] = []
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):    # 只留数值，屏蔽文案
        for i in range(len(ids) - 1):
            a, b = ids[i], ids[i + 1]
            if a is None or b is None:
                continue
            x = lm.tok.encode_composite(toks[i], toks[i + 1])
            tgt = lm.tok.onehot(b)
            d = lm.net.step(x, target=tgt, target_idx=b)
            nll_trace.append(float(d["nll"]))
    # 指纹：读出 + 主干 CSR 的位敏感摘要
    Wr = np.asarray(lm.net.readout.W, dtype=np.float64)
    Wp = np.asarray(lm.net.pc.up0[2], dtype=np.float64)      # CSR val
    digest = [int(np.sum(Wr * 1e6)), int(Wr.size), float(Wr.sum()),
              int(np.sum(Wp * 1e6)), int(Wp.size), float(Wp.sum())]
    return nll_trace, digest


def section_a() -> None:
    print("\n[A] 零回归：--lang 不改变任何训练数值")
    ppl_en, dig_en = _run_tiny("en")
    ppl_zh, dig_zh = _run_tiny("zh")
    print(f"    （逐步 NLL 轨迹：en {len(ppl_en)} 步 / zh {len(ppl_zh)} 步）")

    same_trace = len(ppl_en) == len(ppl_zh) and all(
        a == b for a, b in zip(ppl_en, ppl_zh))
    check(len(ppl_en) >= 50, "A0 轨迹长度足够（避免「只 1 步也算通过」）",
          f"{len(ppl_en)} 步")
    check(same_trace, "A1 两种 --lang 下逐步 NLL 轨迹**逐位相同**",
          f"{len(ppl_en)} 步，首步 {ppl_en[0]:.6f} vs {ppl_zh[0]:.6f}")

    check(dig_en == dig_zh, "A2 读出 + 主干权重指纹逐位相同",
          f"digest={dig_en}")
    check(ppl_en[:5] == ppl_zh[:5], "A3 前 5 步 NLL 逐位相同（放大看细节）",
          f"{[round(x, 6) for x in ppl_en[:3]]}")


# ── B. 输出确实切换 ─────────────────────────────────────────────────────────
def section_b() -> None:
    print("\n[B] 输出语言确实切换（默认英文 / --lang zh 中文）")
    from phdnet.i18n import get_lang, install_stream_filter, set_lang

    set_lang(None)
    check(get_lang() == "en", "B1 默认（不传）=英文", get_lang())
    set_lang("en")
    check(get_lang() == "en", "B2 --lang en = 英文", get_lang())
    set_lang("zh")
    check(get_lang() == "zh", "B3 --lang zh = 中文", get_lang())
    set_lang("EN")
    check(get_lang() == "en", "B4 大写 EN 也识别为英文", get_lang())
    set_lang("zh-CN")
    check(get_lang() == "zh", "B5 zh-CN 识别为中文", get_lang())
    set_lang("不存在的语言")
    check(get_lang() == "en", "B6 未知值回落**英文**（不意外变中文）",
          get_lang())

    # 实际输出探针：用不会被翻译的字符串（数字/英文 token）
    set_lang("en")
    install_stream_filter()
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("MARKER 12345")
    out_en = buf.getvalue().strip()
    set_lang("zh")
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        print("MARKER 12345")
    out_zh = buf2.getvalue().strip()
    check(out_en == out_zh == "MARKER 12345",
          "B7 纯 ASCII 探针在两种语言下**都不被翻译**（翻译器不误伤）",
          f"en={out_en!r} zh={out_zh!r}")


# ── C. --data-lang 仍能过滤（拆出去的职责没丢）───────────────────────────────
def section_c() -> None:
    print("\n[C] --data-lang 语料过滤职责仍在（与 --lang 分离）")
    src = (_ROOT / "train" / "train.py").read_text(encoding="utf-8")
    flat = src.replace("\n", " ")
    check('"--lang", choices=["en", "zh"], default="en"' in flat,
          "C1 --lang 是终端语言且默认 en")
    check('"--data-lang", choices=["all", "zh", "en"], default="all"' in flat,
          "C2 --data-lang 是语料过滤且默认 all（不过滤）")
    check("elif args.data_lang != \"all\":" in flat,
          "C3 过滤逻辑读 args.data_lang（不再读 args.lang）")
    # 确认没有残留的 args.lang 用于数据过滤
    bad = [ln.strip() for ln in src.split("\n")
           if "args.lang" in ln and "set_lang" not in ln
           and "_lang_filter" not in ln]
    check(not bad, "C4 无残留：args.lang 只出现在 set_lang 调用处",
          f"发现 {len(bad)} 处：{bad[:2]}" if bad else "")


# ── D. 三处入口默认值一致 ───────────────────────────────────────────────────
def section_d() -> None:
    print("\n[D] 三处入口默认值一致 + --help 可执行")
    for rel, label in (("train/train.py", "train"),
                       ("train/infer.py", "infer"),
                       ("tools/convert_deepctrl.py", "convert_deepctrl")):
        txt = (_ROOT / rel).read_text(encoding="utf-8")
        flat = txt.replace("\n", " ")
        ok = '"--lang", choices=["en", "zh"], default="en"' in flat
        check(ok, f"D --{label} 的 --lang 默认 en", rel if not ok else "")

    # ⚠ 2026-10-07：显式 encoding/errors（同 verify_p116_sched D5 的说明）——
    #   区域编码解码失败会让 r.stdout 变 None，断言直接 TypeError。
    r = subprocess.run([sys.executable, str(_ROOT / "train" / "train.py"), "--help"],
                       capture_output=True, text=True, cwd=str(_ROOT), timeout=180,
                       encoding="utf-8", errors="replace")
    check(r.returncode == 0 and "--lang" in r.stdout and "--data-lang" in r.stdout,
          "D4 train.py --help 可执行且两个参数都在",
          f"exit={r.returncode}")

    r2 = subprocess.run([sys.executable, str(_ROOT / "train" / "infer.py"), "--help"],
                        capture_output=True, text=True, cwd=str(_ROOT), timeout=180,
                        encoding="utf-8", errors="replace")
    check(r2.returncode == 0, "D5 infer.py --help 可执行", f"exit={r2.returncode}")


def main() -> int:
    print("=" * 78)
    print("P119 门禁：--lang 只管终端文案（默认英文），不影响训练结果")
    print("=" * 78)
    section_a()
    section_b()
    section_c()
    section_d()
    npass = sum(1 for ok, _, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print("\n" + "=" * 78)
    print(f"结果：{npass}/{total} 通过 | 失败 {total - npass}")
    if npass != total:
        print("失败用例：")
        for ok, name, detail in _RESULTS:
            if not ok:
                print(f"  · {name}  {detail}")
    print("=" * 78)
    return 0 if npass == total else 1


if __name__ == "__main__":
    sys.exit(main())