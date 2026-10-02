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
    ("平均", "mean"),
    ("总计", "total"),
    ("示例", "example"),
    ("用法", "usage"),
    ("说明", "description"),
    ("注意", "note"),
    ("警告：", "warning: "),
    # —— 高频运行时文案（长串优先）——
    ("读出计算精度", "readout compute precision"),
    ("读出后端", "readout backend"),
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
    ("完成", "done"),
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
    ("每步", "per step"),
    ("共", "total "),
    ("条", " rows"),
    ("次", "x"),
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
]

_CN = re.compile("|".join(re.escape(k) for k, _ in _TABLE))


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


def translate(text: str) -> str:
    """把 `text` 里的中文片段按表替换（英文模式）。未知片段原样保留。

    ⚠ P147：**诊断类前缀整行跳过翻译**（见 `_DIAG_PREFIXES`）——
      逐词替换会把技术告警拼成中英混杂的不可读文本。
    """
    if _LANG != "en" or not text:
        return text
    if text.lstrip().startswith(_DIAG_PREFIXES):
        return text
    return _CN.sub(lambda m: dict(_TABLE)[m.group(0)], text)


class _StreamFilter:
    """stdout/stderr 包装：写入前做语言转换（只影响终端文案）。"""

    def __init__(self, wrapped):
        self._w = wrapped

    def write(self, s: str):
        return self._w.write(translate(s))

    def flush(self):
        return self._w.flush()

    def __getattr__(self, name):
        return getattr(self._w, name)


def install_stream_filter() -> None:
    """包装 sys.stdout / sys.stderr（幂等；与 TeeLogger 兼容——它只代理 write）。"""
    for name in ("stdout", "stderr"):
        cur = getattr(sys, name)
        if isinstance(cur, _StreamFilter):
            continue
        setattr(sys, name, _StreamFilter(cur))
