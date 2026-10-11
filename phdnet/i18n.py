"""训练/推理终端输出的语言切换（`--lang zh` / `--lang en`）。

P119（fhz 2026-10-01 重新定义）：`--lang` **只管终端文案，不碰数据**。
- **默认英文**（`en`）；`--lang zh` 时终端输出全中文。
- 该参数**对训练结果没有任何影响** —— 语料选择、词表、模型状态、数值全部不变，
  改的只是「打印出来的那句话是什么语言」。
- ⚠ **P119 之前的 `--lang` 含义完全不同**：它同时做**语料语言过滤**
  （`all`/`zh`/`en`，按 parquet 的 `lang` 列筛训练流，**会改变训练结果**）。
  该职责已拆到独立参数 `--data-lang`，`--lang` 不再过滤数据。

## 设计
- 两条路并存：
  1. `T("中文", "English")` —— **精确路径**，新写的输出应该用它；
  2. **输出层兜底**：`install_stream_filter()` 包装 `sys.stdout/stderr`，
     把历史代码里遗留的中文串按 `_TABLE` 替换。这样**一处安装即可覆盖全仓库**
     所有输出点，不必逐个改造旧代码。
- 只影响**终端文案**；数据（词表、语料、日志里的数值）**一律不翻译**。

## 用法
```python
from .i18n import set_lang, install_stream_filter
set_lang(args.lang)            # 早于任何 print
install_stream_filter()        # 包装 stdout/stderr（重复安装安全）
```
`phdnet/` 内部用 `from .i18n import T`；`train/` 用
`from phdnet.i18n import T, set_lang, install_stream_filter`。
"""
from __future__ import annotations

import re
import sys

# P119：**默认英文**（fhz 2026-10-01）。此前默认中文。
_LANG = "en"


def set_lang(lang: str | None) -> str:
    """设置终端语言（`zh` = 中文，`en`/其它 = 英文）。返回规范化后的语言码。

    P119：默认**英文**；`--lang zh` 才是中文。未知值一律回落英文（而非中文），
    这样「漏传/传错」时的输出与默认行为一致，不会意外变中文。
    """
    global _LANG
    _lang = str(lang or "en").strip().lower()
    _LANG = "zh" if _lang.startswith("zh") else "en"
    return _LANG


def get_lang() -> str:
    return _LANG


def T(zh: str, en: str = "") -> str:
    """按当前语言返回文案：中文模式原样；英文模式用 `en`（缺省退回中文）。"""
    return zh if _LANG == "zh" else (en or zh)


# 兜底替换表：按「长串优先」排序，保证最长匹配生效
# 结构：(中文键, 英文值)。正向（en 模式）：中文键 → 英文值；
# 反向（zh 模式）：由下方 _REV_ZH 程序化构建 英文值 → 中文键。
_TABLE: list[tuple[str, str]] = [
    # —— 训练主循环 ——
    ("预取进程数", "prefetch processes"),
    ("预取队列深度", "prefetch queue depth"),
    ("解码核/进程", "decode cores/proc"),
    ("数据侧并行不与分词线程争用", "data-side parallelism does not contend with tokenisation"),
    ("分词线程（numba nogil）", "tokenisation thread (numba nogil)"),
    ("训练完成", "training finished"),
    ("最终 PPL", "final PPL"),
    ("OOV 跳过", "OOV skipped"),
    ("已用步数", "steps done"),
    ("滑窗 PPL", "sliding PPL"),
    ("上下文里程碑", "context milestone"),
    ("检查点已保存", "checkpoint saved"),
    ("未提交本地改动", "uncommitted local changes"),
    ("轨迹", "trajectory"),
    ("模型参数", "model parameters"),
    ("容量", "capacity"),
    ("可塑", "plastic"),
    ("（预训练）", " (pretrain)"),
    ("（SFT）", " (SFT)"),
    # —— 精度 / 后端 ——
    ("读出计算精度", "readout compute precision"),
    ("读出后端", "readout backend"),
    ("检查点存储精度", "checkpoint storage"),
    ("不可用，回落", "unavailable, falling back"),
    ("永久回落", "permanent fallback"),
    ("训练不中断", "training continues"),
    ("编译", "compile"),
    ("融合", "fusion"),
    ("加速器", "accelerator"),
    ("设备", "device"),
    ("线程", "threads"),
    ("主副本", "master copy"),
    # —— 通用提示 ——
    ("原因：", "reason: "),
    ("不支持", "unsupported"),
    ("已禁用", "disabled"),
    ("不可用", "unavailable"),
    ("失败", "failed"),
    ("警告", "warning"),
    ("错误", "error"),
    ("完成", "done"),
    ("开始", "start"),
    ("结束", "end"),
    ("加载", "load"),
    ("保存", "save"),
    ("跳过", "skipped"),
    ("超时", "timeout"),
    ("回退", "reverted"),
    ("生效", "applied"),
    ("逐位", "bit-exact"),
    ("默认值", "default"),
    ("预设", "preset"),
    ("档位", "tier"),
    ("量化", "quantised"),
    ("稀疏", "sparse"),
    ("稠密", "dense"),
    ("连接率", "connectivity"),
    ("步时", "step time"),
    ("遥测", "telemetry"),
    ("缓存", "cache"),
    ("目录", "directory"),
    ("文件", "file"),
    ("大小", "size"),
    ("耗时", "elapsed"),
    # 2026-10-07：删掉本条重复键 `("平均", "mean")` —— 表尾还有一条
    # `("平均", "avg")`，dict 后写覆盖前写 ⇒ "mean" 一直是死条目（保留实际
    # 生效的 "avg"，行为不变）。
    ("总计", "total"),
    ("示例", "example"),
    ("用法", "usage"),
    ("说明", "description"),
    ("注意", "note"),
    ("警告：", "warning: "),
    # —— 高频运行时文案 ——
    # 2026-10-07：删掉与上面重复的「读出计算精度 / 读出后端」两条（值相同）。
    # 「长串优先」不再靠表的顺序，由下方 _CN 按长度降序排正则保证。
    ("语种", "language"),
    ("数据加载完成", "data loading done"),
    ("已保存", "saved"),
    ("正在保存", "saving"),
    ("词表快照", "vocab snapshot"),
    ("远程", "remote"),
    ("分片", "shard"),
    ("语料", "corpus"),
    ("样本", "sample"),
    ("字符", "chars"),
    ("行", "lines"),
    ("个", ""),
    ("秒", "s"),
    ("分钟", "min"),
    ("毫秒", "ms"),
    ("总计用时", "total time"),
    ("学习率", "learning rate"),
    ("打乱", "shuffle"),
    ("种子", "seed"),
    ("确定性", "deterministic"),
    ("未收敛", "not converged"),
    ("已收敛", "converged"),
    ("检查点", "checkpoint"),
    ("已写入", "written"),
    ("写入", "write"),
    ("读取", "read"),
    ("过滤", "filter"),
    ("总用时", "total time"),
    ("当前", "current"),
    ("已用", "used"),
    ("正在", "running"),
    ("已", "already "),
    ("新的", "new "),
    ("与", " with "),
    ("张量", "tensor"),
    ("缓存目录", "cache dir"),
    ("内存", "memory"),
    ("占用", "usage"),
    ("版本", "version"),
    ("路径", "path"),
    ("存在", "exists"),
    ("不存在", "missing"),
    ("需要", "requires"),
    ("使用", "using"),
    ("配置", "config"),
    ("初始化", "init"),
    ("已加载", "loaded"),
    ("训练步", "train step"),
    ("迭代", "iteration"),   # 2026-10-07：交叉验证发现该高频词无词条，"迭代 3 次" 残留中文
    ("每步", "per step"),
    ("共", "total "),
    ("条", " rows"),
    # 2026-10-07：原为 ("次", "x") —— 那是把「倍数记号」和「次数」混了，
    # 实测 "3 次迭代" → "3 x迭代"（不可读）。改成 " times"；
    # 再补一条带空格的键，避免 "3 次"（原文已带空格）变成双空格。
    (" 次", " times"),
    ("次", " times"),
    ("批", "batch"),
    ("首", "first"),
    ("末", "last"),
    ("平均", "avg"),
    ("最大", "max"),
    ("最小", "min"),
    ("是否", "whether"),
    ("无", "none"),
    ("空", "empty"),
    ("全部", "all"),
    ("部分", "partial"),
    ("默认开启", "enabled by default"),
    ("关闭", "off"),
    ("开启", "on"),
    ("运行", "run"),
    ("会", "will "),
    ("一次", "once"),
    ("这", "this "),
    # —— 标点：英文模式下全角 → 半角（2026-10-07 新增）——
    # 不加这条，翻译结果会残留全角标点，实测：
    #   translate("警告：读出后端不可用，已回退") → "warning：readout backendunavailable，already reverted"
    # 有了它 → "warning: readout backendunavailable, already reverted"。
    # 只作用于终端输出层（_StreamFilter），不碰日志文件与任何数据。
    ("，", ", "),
    ("：", ": "),
    ("。", "."),        # 2026-10-07：原为 ". " → 行尾多一个空格；尾随空格改由下面的 _P2C 按需补
    ("；", "; "),
    ("、", ", "),
    ("（", " ("),
    ("）", ")"),
    ("！", "! "),
    ("？", "? "),
]

# ⚠⚠ 2026-10-07 修复（两处真 bug，均已实测复现）：
#  ① **表里有重复键**：「读出计算精度」「读出后端」「完成」各出现两次，
#    「平均」两次且值不同（"mean" vs "avg"）→ dict 后写覆盖前写，前面的是死条目。
#  ② **正则交替取「第一条匹配」，不是「最长匹配」**：上面注释声称「按长串优先排序」，
#    但 _TABLE 根本没排序 → 短串遮蔽长串，实测：
#       总计用时 → "total用时"   （被「总计」遮蔽）
#       缓存目录 → "cachedirectory"（被「缓存」遮蔽，且粘连没空格）
#       已加载   → "already load"  （被「已」遮蔽）
#    → 先按去重键表、再**按长度降序**排正则，保证最长优先。
_MAP: dict[str, str] = dict(_TABLE)          # 重复键：后者生效（与历史行为一致）
_KEYS = sorted(_MAP, key=lambda k: (-len(k), k))
_CN = re.compile("|".join(re.escape(k) for k in _KEYS))


# P147：**诊断前缀白名单** —— 这些前缀开头的行**不做翻译**。
# 起因：P147 的 fp8 警告行里有「会/非目标/行/学习规则」等词，逐个命中
#   `_TABLE` 的条目 → 打印成「半精度will 舍入丢弃感知器 p − t 的**非目标lines**」
#   这种中英混杂，**完全无法阅读**（实测 2026-10-02）。
# 诊断/告警行必须**两种语言下完全一致** → 干脆不翻译（它们本来就是给
# 人看的技术信息，不是面向用户的宣传文案）。
_DIAG_PREFIXES = (
    "[readout] compute precision",
    "[fp8]", "[probe]", "[cann-env]", "[telemetry]",
    "[capability]", "[readout-]", "[m2-kernel]",
)

# ⚠ 2026-10-07：白名单**必然腐烂** —— 实测运行期的
#   `[precision] 读出权重 fp16 请求 → 运行时提升为 fp32` 不在名单里，
#   被逐词翻成 `[precision] ... run时提升为 fp32`（正是 P147 记录过的复发）。
#   → 改成**结构性判据**：凡是以 `[ASCII 标签]` 开头的行一律不翻译。
#   `[1/3]`、`[预设]` 这类（数字/中文开头）不在本规则内，仍照常翻译。
_DIAG_TAG = re.compile(r"^\s*\[[A-Za-z][A-Za-z0-9._+ -]*\]")

# ⚠ 2026-10-07：**中英粘连**是逐词替换的固有产物 —— 源中文没有空格，替换成
#   英文后首尾相接就成一个怪词，实测：
#     "读出后端不可用"   → "readout backendunavailable"
#     "已加载检查点"     → "loadedcheckpoint"
#     "缓存目录不存在"   → "cache dirmissing"
#   逐条给词条加前导空格既漏又会引入双空格 → **主修在 translate 的替换回调里**
#   （相邻两个词条、或原文前一个字符是字母且本次译文以字母开头 → 补一个空格），
#   事后的「字母↔汉字」规则只兜**只译出一半**的边界（如 "reverted到 default"）。
#   只在英文模式、只在终端输出层生效。
_L2C = re.compile(r"(?<=[A-Za-z])(?=[\u4e00-\u9fff])")
_C2L = re.compile(r"(?<=[\u4e00-\u9fff])(?=[A-Za-z])")
# 半角标点后紧跟汉字时补一个空格（"。"改译成 "." 后靠它补位）；
# 只在**后面真有内容**时补 → 行尾不会多出空格。
_P2C = re.compile(r"(?<=[.,;:!?])(?=[\u4e00-\u9fff])")
_LATIN_OK = re.compile(r"[A-Za-z]")      # 快速预筛：没有字母就不用补空格
# 快速预筛：没有「需要翻译的字符」才走快速通道。
# ⚠ 2026-10-07：范围必须**含 CJK 标点与全角字母数字**（\u3000-\u303f 注释符号、
#   \uff00-\uffef 全角（）：，；、与 Ａ-Ｚ），否则「只有标点没有汉字」的串会
#   被快速通道原样返回 —— 门禁 gate_cli_i18n 的 B3 抓到过（'（SFT）'→ 未译）。
_CJK_OK = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uf900-\ufaff\uff00-\uffef]")

# ============================================================================
# zh 模式：**英 → 中** 反向兜底（2026-10-08 新增）
# ----------------------------------------------------------------------------
# 背景：此前 translate() 只有「源码中文串 → en 模式译成英文」一个方向；运行期
# 却有大量**硬编码英文**片段（"(requested fp8"、"checkpoint storage = fp16"、
# "prefetch processes"、"warning: " …，多数恰是 `_TABLE` 的英文值）在 zh 模式
# 原样透出 → 终端中英混杂。→ zh 模式把 `_TABLE` 的英文值**反向译回中文键**。
#
# 构建规则（保守优先，宁可漏译不可误翻）：
#   ① `_REV_ZH`：英文值（strip 后）→ 中文键。**一键多值**（多个中文键映射到
#      同一英文值）时取**较长的中文键**（信息更完整；同长保留先出现的，保证
#      确定性）。实测碰撞例："total time" ← 总计用时/总用时、"usage" ←
#      用法/占用、"warning" ← 警告/警告：——任取其一均可读。
#   ② **只收长度 ≥ 4 的英文值**：滤掉 "on"/"off"/"s"/"ms"/"run"/"avg"/"min"
#      这类短词 —— 短词即使加词边界也极易误伤代码标识符、路径、数字单位。
#      短词（含 "on/off"）一律放弃，不翻。
#   ③ 边界用显式环视 `(?<![A-Za-z0-9_]) … (?![A-Za-z0-9_])` 而非裸 `\b`：
#      对标识符字符的判定二者等价，但 `\b` 在值以空格/标点结尾时（如
#      "warning: "）必然失配，环视对任意值稳健。下划线归标识符侧，
#      "probe_PPL"、"save_dir"、"step_time" 等不会被拆开误翻。
#   ④ **区分大小写**精确匹配（保守起步，实测够用：源码硬编码片段均为小写，
#      放宽会把 FP8/Fp16 等专有名词卷进来）。
#   ⑤ 值被 strip 掉的**尾部空白**同步从键上剥掉（"警告："对应值 "warning: "
#      → core "warning:"：冒号在匹配内被消费掉，键必须保留全角冒号，
#      否则 "警告：读出后端…" 会丢冒号变成 "警告读出后端…"）。
#   ⑥ 快速通道：zh 模式下**没有 ASCII 字母**的串直接原样返回
#      （纯中文 / 纯数字 / 对齐表格零开销）。
_REV_ZH: dict[str, str] = {}
for _zh_k, _en_v in _TABLE:
    _core = _en_v.strip()
    if len(_core) < 4 or not _core.isascii():
        continue                                  # ② 短词 / ③ 非英文值 / 空译文
    _zc = _zh_k.strip()
    if _en_v != _core:                            # ⑤ 值有被剥的尾部空白 → 键同步剥
        _zc = _zc.rstrip(" \t")
    if _core not in _REV_ZH or len(_zc) > len(_REV_ZH[_core]):
        _REV_ZH[_core] = _zc                      # ① 更长的中文键优先

# 补充表：**只存在于 zh 反向**的零星硬编码英文 —— 源码直接打印、`_TABLE`
# 里没有对应中文键（无法程序化推出）。刻意维持极小集合，逐条注明出处；
# `setdefault` 不覆盖 `_TABLE` 派生的映射。
_ZH_EXTRA: dict[str, str] = {
    "requested": "请求",      # train.py 精度声明行 "(requested fp8 …)"
}
for _xk, _xv in _ZH_EXTRA.items():
    _REV_ZH.setdefault(_xk, _xv)

# 最长优先的交替正则（与 en 模式 _KEYS/_CN 同一策略：排序后 re.escape 拼接）
_REV_ITEMS = sorted(_REV_ZH.items(), key=lambda kv: (-len(kv[0]), kv[0]))
_ZH_EN = re.compile("(?<![A-Za-z0-9_])(?:"
                    + "|".join(re.escape(k) for k, _ in _REV_ITEMS)
                    + ")(?![A-Za-z0-9_])")

# zh 译后修整（只在 zh 模式、只作用于终端输出层）：
#   _ZH_SPACE  汉字之间的残留空格折叠（"读出后端 不可用" → "读出后端不可用"）；
#              只在两侧都是汉字时折叠，数字/英文前后的空格不动，保住对齐；
#   _ZH_COLON  汉字后的半角 ": " 补成全角 "："（与 "警告：" 类词条风格一致）；
#   _ZH_COMMA  逗号后紧跟汉字时补成全角 "，"（"8, 警告" → "8，警告"）；
#              "1, 2, 3" 这类列表后面不是汉字，不受影响。
_ZH_SPACE = re.compile(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])")
_ZH_COLON = re.compile(r"(?<=[\u4e00-\u9fff])\s*:\s*(?![\u4e00-\u9fff])")
# 汉字全角冒号后**仍残留**的空格（后跟英文/数字时 _ZH_COLON 不吞），
# 折叠成无空格 —— 中文冒号后照习惯不加空格。
_ZH_COLON_GAP = re.compile(r"(?<=[\u4e00-\u9fff]：)[ \t]+")
_ZH_COMMA = re.compile(r",\s*(?=[\u4e00-\u9fff])")


def translate(text: str) -> str:
    """按当前语言转换 `text`：en 模式译出中文串，zh 模式译回硬编码英文。

    ⚠ P147：**诊断类前缀整行跳过翻译**（见 `_DIAG_PREFIXES` / `_DIAG_TAG`）——
      逐词替换会把技术告警拼成中英混杂的不可读文本。**两种模式都跳过**：
      诊断行必须两种语言下逐字节一致。
    ⚠ 2026-10-07（en 模式）：**粘连处理**（逐词替换的固有产物，实测）：
        "读出后端不可用" → "readout backendunavailable"
        "已加载检查点"   → "loadedcheckpoint"
      根因是「两个相邻词条各自译成英文后首尾相接」，此时源字符串里**已经没有
      汉字**，靠事后的「字母↔汉字」规则根本看不到 → 必须在**替换回调里**判断：
      相邻词条、或原文前一个字符是字母且本次译文也以字母开头 → 补一个空格。
    ⚠ 2026-10-08（zh 模式）：反向兜底 —— `_REV_ZH`（由 `_TABLE` 程序化构建，
      英文值 → 中文键，只收长度 ≥ 4 的值）+ 显式环视词边界 + 区分大小写。
      短词（"on"/"off"/"s"/"ms"/"run" …）与未收录的英文一律原样保留；
      fp8/fp16/PPL 等专有名词自然不受影响（不在表内）。
    """
    if not text:
        return text
    if _LANG == "zh":
        return _translate_zh(text)
    # —— en 模式（原有逻辑，行为不变）——
    # 快速通道：**整段没有汉字**（英式日志里绝大多数行，含全部指标表格）
    # → 原样返回，零正则开销；顺带保证对齐用的连续空格绝不被动过。
    if not _CJK_OK.search(text):
        return text
    if text.lstrip().startswith(_DIAG_PREFIXES) or _DIAG_TAG.match(text):
        return text
    return _translate_en(text)


def _translate_en(text: str) -> str:
    """en 模式：中文片段 → 英文（2026-10-07 前的 translate 正文，原样搬入）。"""
    # _MAP 是模块级去重表：**每次 write 不再重建 dict**（原来 27.97 µs/次，
    # train.py 每步多次 print，纯浪费）。
    prev_end = -1                      # 上一个匹配的结尾（源串坐标）
    tail = ""                          # **输出流**的最后一个字符（间隙拷贝 或 上一段译文末字符）

    def _rep(m: "re.Match") -> str:
        nonlocal prev_end, tail
        rep = _MAP[m.group(0)]
        if m.start() > prev_end:
            # 与上一匹配之间有源文本被原样拷贝 → 输出尾字符 = 这段的最后一个字符。
            # 源里已有的空格、汉字、数字都靠这条拦住补空格（"已加载 3 个检查点"
            # 曾因为「上一段译文是空串」补出双空格，交叉验证抓到）。
            tail = text[m.start() - 1] if m.start() > 0 else ""
        if rep:
            first = rep[0]
            # ⚠ 只在「输出尾字符是英文字母 且 本段译文也以字母开头」时补空格 ——
            #   一类覆盖**相邻两词条**（"后端"→"不可用"），一类覆盖**源串本就粘连**
            #   （"PPL不可用"）。三条不补：译文以空格/标点开头、尾字符不是字母
            #   （空格、汉字、数字、行首）、译文为空串。
            if (not first.isspace() and first not in ",.;:!?)]}"
                    and tail and _LATIN_OK.fullmatch(tail)
                    and first.isascii() and first.isalpha()):
                rep = " " + rep
            tail = rep[-1]             # 空译文不更新尾字符（保留上一段的）
        prev_end = m.end()
        return rep

    out = _CN.sub(_rep, text)
    if not _LATIN_OK.search(out):        # 没有英文字母 → 无需补空格（省一次扫描）
        return out
    out = _C2L.sub(" ", _L2C.sub(" ", out))
    return _P2C.sub(" ", out)


def _translate_zh(text: str) -> str:
    """zh 模式：硬编码英文片段 → 中文（反向兜底，保守规则见 `_REV_ZH` 注释）。"""
    # 快速通道 ⑥：整段没有 ASCII 字母 → 原样返回（纯中文/纯数字/表格零开销）。
    if not _LATIN_OK.search(text):
        return text
    # 诊断前缀：两种模式下都整行跳过（与 en 模式同一判据、同一理由）。
    if text.lstrip().startswith(_DIAG_PREFIXES) or _DIAG_TAG.match(text):
        return text

    def _rep(m: "re.Match") -> str:
        return _REV_ZH[m.group(0)]

    out = _ZH_EN.sub(_rep, text)
    if out == text:                      # 没有任何命中 → 三个修整正则都白跑
        return out
    out = _ZH_SPACE.sub("", out)
    out = _ZH_COLON_GAP.sub("", _ZH_COLON.sub("：", out))
    return _ZH_COMMA.sub("，", out)


class _StreamFilter:
    """stdout/stderr 包装：写入前做语言转换（只影响终端文案）。"""

    def __init__(self, wrapped):
        self._w = wrapped

    def write(self, s: str):
        return self._w.write(translate(s))

    def flush(self):
        return self._w.flush()

    def writelines(self, lines):
        # 2026-10-07：原来只包装了 write()，`sys.stdout.writelines(...)`
        # 会经 __getattr__ 直接落到**底层流**、绕过翻译（半翻半不翻）。
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
