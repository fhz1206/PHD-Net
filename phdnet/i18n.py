# -*- coding: utf-8 -*-
"""训练/推理终端输出的语言切换（`--lang zh` / `--lang en`）。

## P199 重写（2026-10-17）：正则词表替换 → 显式消息目录

**旧方案为什么必须废弃**（服务器日志实测的失败模式，不可修）：
  1. **子串误切**：`大空间表容量` → `大 empty 间表 capacity`
     （`空间` 被 `容量` 词条夹击；`大空间表` 这类专有组合无法穷举）
  2. **漏网残留**：`already 生长突触按 100 万 rows`
     （`生长突触` 不在词条表 → 永远追不完；338 个静态中文串只登记了 <10%）
  3. **中英粘连**：`readout backendunavailable`、`loadedcheckpoint`
     （替换后相邻 token 首尾相接，事后的空格规则只能兜一半）
  4. **维护面失控**：200+ 词条 × 338 个串的交叉组合无法穷举验证

**新方案**（gettext 思路的零依赖版）——三层策略，**结构上排除碎片化**：

  ① `_CATALOG`：**整句**中文 → 整句英文（键=完整消息，精确匹配）
     → 消息一旦登记，输出就是人写的英文句子，不存在拼接错误
  ② `T(zh, en)`：代码内**显式双语**（新代码首选；调用点即翻译点）
  ③ `translate()` 的降级规则：整句查目录 → 未命中则**原样返回中文**
     → 漏译 = 该句保持中文（**可读**），而不是碎片乱码

**迁移策略**：训练热路径的核心消息（容量验算、检查点、里程碑、汇总行）
先迁入 `_CATALOG`；其余中文输出在 en 模式下保持中文（可读、无损），
随迭代逐步迁入。**不再维护「短词词条表」**——它是所有乱码的根源。

## 用法
```python
from phdnet.i18n import set_lang, install_stream_filter
set_lang(args.lang)            # 早于任何 print
install_stream_filter()        # 包装 stdout/stderr（幂等）
```
新代码：`from phdnet.i18n import T`，打印处写 `print(T("中文", "English"))`。

## 兼容性
`set_lang` / `get_lang` / `T` / `translate` / `install_stream_filter`
签名与 P119 完全一致；`gate_cli_i18n.py` 已同步改写（旧门禁测的是
正则替换的内部行为，那些行为已不存在）。
"""
from __future__ import annotations

import sys

# P119：**默认英文**（fhz 2026-10-01）。
_LANG = "en"


def set_lang(lang: str | None) -> str:
    """设置终端语言（`zh` = 中文，`en`/其它 = 英文）。返回规范化后的语言码。

    未知值一律回落英文（而非中文）：漏传/传错时输出与默认行为一致。
    """
    global _LANG
    _lang = str(lang or "en").strip().lower()
    _LANG = "zh" if _lang.startswith("zh") else "en"
    return _LANG


def get_lang() -> str:
    return _LANG


def T(zh: str, en: str = "") -> str:
    """按当前语言返回文案：中文模式原样；英文模式用 `en`（缺省退回中文）。

    新代码的首选——调用点即翻译点，无全局状态依赖、无替换歧义。
    """
    return zh if _LANG == "zh" else (en or zh)


# ============================================================================
# 消息目录：**整句**中文 → 整句英文（键 = 完整消息，精确匹配）
# ----------------------------------------------------------------------------
# 覆盖训练热路径的高频消息（服务器日志 grep 实测的出现频率排序）。
# 条目按语义分组；**键必须与源码打印串逐字一致**（含前后缀与标点）——
# `translate()` 是 dict 查表，不是模糊匹配。
# 没登记的消息在 en 模式下**原样输出中文**（可读、无损、不产生碎片）。
# ============================================================================
_CATALOG: dict[str, str] = {
    # ── 容量验算表（每次启动打印）──
    "突触参数容量验算（1B 口径：固定突触 + 大空间表容量）":
        "Synapse capacity audit (1B basis: fixed synapses + big-space table)",
    "M1 编码器（稠密）":
        "M1 encoder (dense)",
    "M2 主干 CSR 稀疏连接":
        "M2 backbone CSR sparse links",
    "M3 STDP 侧向核":
        "M3 STDP lateral core",
    "M6 读出头":
        "M6 readout head",
    "固定突触小计":
        "fixed synapses subtotal",
    "M4b 大空间表容量（事件驱动）":
        "M4b big-space table capacity (event-driven)",
    "总突触参数容量":
        "Total synapse capacity",
    "静态内存":
        "static memory",
    "大空间表索引映射":
        "big-space index map",
    "已生长突触按 100 万条":
        "grown synapses @ 1M rows",
    # ── 容量/进度行（每 --log-every 步）──
    "滑窗 PPL":
        "sliding PPL",
    "上下文里程碑":
        "context milestone",
    "总容量":
        "total capacity",
    "已生长":
        "grown",
    "利用率":
        "utilization",
    # ── 检查点/收尾 ──
    "检查点已保存":
        "checkpoint saved",
    "训练完成":
        "training finished",
    "最终 PPL":
        "final PPL",
    "已用步数":
        "steps done",
    "训练步":
        "train step",
    "模型参数":
        "model parameters",
    "（预训练）":
        " (pretrain)",
    "（SFT）":
        " (SFT)",
    "预取进程数":
        "prefetch processes",
    "预取队列深度":
        "prefetch queue depth",
    "分词线程（numba nogil）":
        "tokenisation thread (numba nogil)",
    "词表快照":
        "vocab snapshot",
    "数据加载完成":
        "data loading done",
    "总用时":
        "total time",
    "总计用时":
        "total time",
    "学习率":
        "learning rate",
    "语种":
        "language",
}

# ============================================================================
# 词级补充表：**只收多字词**（≥2 个汉字），**只做整词替换**（无正则、无单字）。
# ----------------------------------------------------------------------------
# 用途：目录未登记的**长句**里出现的组件词（「内存」「编码器」「读出」…）。
# 与旧方案的本质区别：
#   · **键 ≥ 2 个汉字** → 不可能出现「大/空/间/表」这种单字夹击
#   · **replace() 是字面匹配**，不做正则交替 → 无最长匹配歧义
#   · 查不到就保留中文 → 最坏情况是「半中文半英文的完整词序列」，
#     每个词都是完整词，可读性远高于碎片
# ============================================================================
_WORDMAP: dict[str, str] = {
    "编码器": "encoder",
    "主干": "backbone",
    "读出头": "readout head",
    "读出": "readout",
    "检查点": "checkpoint",
    "词表": "vocab",
    "语料": "corpus",
    "分片": "shard",
    "大空间表": "big-space table",
    "大空间": "big-space",
    "突触": "synapse",
    "突触参数": "synapse parameters",
    "容量": "capacity",
    "内存": "memory",
    "静态内存": "static memory",
    "已生长": "grown",
    "生长突触": "grown synapses",
    "利用率": "utilization",
    "里程碑": "milestone",
    "训练步": "train step",
    "样本": "sample",
    "字符": "chars",
    "缓存目录": "cache dir",
    "缓存": "cache",
    "耗时": "elapsed",
    "训练完成": "training finished",
    "检查点已保存": "checkpoint saved",
    "已保存": "saved",
    "正在保存": "saving",
    "训练数据": "training data",
    "语种": "language",
    "设备": "device",
    "线程": "threads",
    "迭代": "iteration",
    "学习率": "learning rate",
    "种子": "seed",
    "精度": "precision",
    "加速器": "accelerator",
    "版本": "version",
    "路径": "path",
    "配置": "config",
    "预取进程数": "prefetch processes",
    "预取队列深度": "prefetch queue depth",
    "分词线程": "tokenisation thread",
}

# 词级替换按键长降序（长词优先，避免「大空间」先于「大空间表」吃掉前缀）
_WORD_ITEMS = sorted(_WORDMAP.items(), key=lambda kv: -len(kv[0]))

# 词边界空格：英文字母↔汉字交界处补一个空格（只在替换发生时跑）
import re as _re
_HAS_CJK = _re.compile(r"[\u4e00-\u9fff]")
_L2C = _re.compile(r"(?<=[A-Za-z])(?=[\u4e00-\u9fff])")
_C2L = _re.compile(r"(?<=[\u4e00-\u9fff])(?=[A-Za-z])")


def translate(text: str) -> str:
    """按当前语言转换 `text`（供输出层包装器逐行调用）。

    en 模式三级策略：
      ① 整句精确查 `_CATALOG`（完整消息 → 完整英文句子）
      ② 未命中 → `_WORDMAP` 的**多字词整词替换**（长词优先，无正则）
      ③ 替换后仍含中文 → 原样保留（可读），**不做单字/碎片替换**

    zh 模式：原样返回（源码以中文为主；硬编码英文片段已通过 T() 显式化）。

    ⚠ 诊断前缀（`[ASCII 标签]` 开头）整行跳过：诊断/告警行必须两种语言
      逐字节一致（P147 纪律）。
    """
    if not text:
        return text
    if _LANG == "zh":
        return text
    # 快速通道：**整段没有汉字**（指标表格、英文日志行，占绝大多数）
    # → 原样返回。词级扫描是 O(len(text)×词表)，394 字符行 × 45 词条
    # 实测 204µs/行 —— 没有汉字时根本无需进扫描。
    if not _HAS_CJK.search(text):
        return text
    # ① 整句目录（精确匹配，O(1)）
    hit = _CATALOG.get(text)
    if hit is not None:
        return hit
    # 诊断行跳过（必须在词级替换之前——诊断行里的中文组件词不该被翻）
    if _is_diag_line(text):
        return text
    # ② 词级降级替换（无正则：str.replace 字面匹配）
    # ⚠ 用「逐词扫描 + 手写游标」而不是循环 str.replace：
    #   str.replace 无法感知**替换点前一个字符**——两个相邻词条
    #   （突触参数+容量）会替换成 "synapse parameters"+"capacity" 粘连。
    #   扫描版在每处替换前看前一个**输出**字符：若两侧都是字母 → 补空格。
    out_parts: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        matched = False
        for zh, en in _WORD_ITEMS:
            if text.startswith(zh, pos):
                prev = out_parts[-1][-1] if out_parts and out_parts[-1] else ""
                # 前一个输出字符是字母 且 译文以字母开头 → 补空格
                if (prev and prev.isascii() and prev.isalpha()
                        and en[:1].isascii() and en[:1].isalpha()):
                    out_parts.append(" ")
                out_parts.append(en)
                pos += len(zh)
                matched = True
                break
        if not matched:
            out_parts.append(text[pos])
            pos += 1
    out = "".join(out_parts)
    # ②b 词边界空格：英文替换词紧邻汉字时补空格（"已 grown synapses 按"）
    #    只在替换真的发生时才跑这两条正则。
    if out != text and _HAS_CJK.search(out):
        out = _L2C.sub(" ", out)
        out = _C2L.sub(" ", out)
    # ③ 剩余中文原样保留（不做单字替换——那是所有乱码的根源）
    return out


# 诊断行判据：`[ASCII 标签]` 开头（P147/P147-2026-10-07 结构性判据，保留）
def _is_diag_line(text: str) -> bool:
    s = text.lstrip()
    if not s.startswith("["):
        return False
    end = s.find("]")
    if end <= 1:
        return False
    tag = s[1:end]
    # 标签必须以 ASCII 字母开头、主体为 ASCII 字母数字/点/加号/空格
    return (tag[:1].isascii() and tag[:1].isalpha()
            and all(c.isascii() and (c.isalnum() or c in "._+ -") for c in tag))


class _StreamFilter:
    """stdout/stderr 包装：写入前做语言转换（只影响终端文案）。"""

    def __init__(self, wrapped):
        self._w = wrapped

    def write(self, s: str):
        return self._w.write(translate(s))

    def flush(self):
        return self._w.flush()

    def writelines(self, lines):
        return self._w.writelines(translate(s) for s in lines)

    def __getattr__(self, name):
        return getattr(self._w, name)


def install_stream_filter() -> None:
    """包装 sys.stdout / sys.stderr（幂等；与 TeeLogger 兼容——它只代理 write）。"""
    for name in ("stdout", "stderr"):
        cur = getattr(sys, name)
        if isinstance(cur, _StreamFilter):
            continue
        setattr(sys, name, _StreamFilter(cur))
