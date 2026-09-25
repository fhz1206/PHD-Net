"""流式语料分词 —— 任意长度语料不截断训练的基础设施（train_1b 专用）。

为什么需要（fhz 2026-09-25：「训练数据无论多长不要截断」「1M context」）
------------------------------------------------------------------------
既有路径 `load_text()` 把语料**整段载入内存**再分词：
  - 4.6 GB pretrain 分片 → 全量 tokenize 后 token 列表数十 GB，不可行；
  - `--max-chars 4_000_000` 的默认值本质上是**截断训练数据**。
本模块把「读语料 → 分词 → 训练」改为**字符级流式**：
任意长度语料逐样本流过，token 边产边训，内存占用与语料总长无关。

逐位等价性（关键正确性声明）
----------------------------
`StreamingTokenizer` 与 `WordSegmenter.tokenize`（贪心最长匹配）**逐位一致**：

  全量版在位置 i 的决策：L = max_len … 2 中首个 `text[i:i+L] ∈ vocab`，否则单字。
  该决策只依赖 `text[i:i+max_len]`（贪心不回溯、不向左看）。
  在线版保证**每次决策前缓冲 ≥ max_len 字符（或源已耗尽 = 缓冲即全部剩余）**，
  因此 L 的上界、命中判断、回退行为与全量完全相同。
  `verify_stream_tokenize.py` 对拍验证（多块切分 vs 全量，逐位一致 PASS）。

语料流定义：`concat(样本_i + SEP)`（SEP="\n\n"，与 load_text 的样本分隔约定一致，
每条样本尾部显式携带分隔符 → 样本边界成为 token 流的普通字符）。

词表与 OOV
----------
词表（涌现词表 seg.vocab + token 映射）来自**采样文本**（默认前
`vocab_sample_chars` 字符，可 `--vocab-scan full` 全量扫一遍构建零 OOV 词表）。
训练流中遇到词表外 token（OOV）时**跳过该训练步**并计数——永不截断、永不崩溃，
OOV 率打印在日志与 [METRIC] 尾行（4M 采样 + 中文语料实测应 ≪0.1%）。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

SEP = "\n\n"


def char_chunks(path, sep: str = SEP) -> Iterator[str]:
    """语料字符块流：逐样本 yield（样本 + 分隔符）；path 支持单文件或 glob。"""
    from phdnet.corpus import iter_texts
    for t in iter_texts(path):
        yield t + sep


def _prefetch_producer(path, sep: str, q) -> None:
    """生产者进程体：把样本字符块推入有界队列（数据加载多核，fhz 2026-09-25）。"""
    try:
        for t in char_chunks(path, sep):
            q.put(t)
        q.put(None)
    except BaseException as e:                   # 生产者异常透传给消费者
        try:
            q.put(e)
        except Exception:
            pass


class PrefetchChars:
    """多核数据加载：独立生产者进程预取样本字符块（iterable，可直接喂给
    StreamingTokenizer，与 char_chunks 产出逐位一致——同一来源同一顺序）。

    「训练过程不要存在等待代码」实现方式：主进程训练循环只做 queue.get()
    （OS 级阻塞，无 sleep/轮询/忙等代码）；parquet 解码与训练计算在两个
    核心上重叠。队列有界（maxsize=depth）→ 背压自动限内存
    （内存上界 ≈ depth × 平均样本长度，缺省 2048 × 数 KB ≈ 数十 MB）。
    """

    def __init__(self, path, sep: str = SEP, depth: int = 2048):
        import multiprocessing as mp
        self._q: "mp.Queue" = mp.Queue(maxsize=depth)
        self._proc = mp.Process(target=_prefetch_producer, args=(path, sep, self._q),
                                daemon=True)
        self._proc.start()

    def __iter__(self) -> "PrefetchChars":
        return self

    def __next__(self) -> str:
        item = self._q.get()
        if item is None:
            raise StopIteration
        if isinstance(item, BaseException):
            if isinstance(item, KeyboardInterrupt):
                # 生产者进程被 Ctrl+C 打断：按流结束处理（主循环 _STOP 已置位，
                # 收尾仍会保存检查点）
                raise StopIteration
            raise item                            # 真实异常必须浮出
        return item


def build_vocab_text(path, max_chars: int, sep: str = SEP) -> str:
    """取语料流前 max_chars 字符（与训练流的**开头逐字符一致**）供词表构建。"""
    parts: list[str] = []
    n = 0
    for c in char_chunks(path, sep):
        parts.append(c)
        n += len(c)
        if n >= max_chars:
            break
    return "".join(parts)[:max_chars]


class StreamingTokenizer:
    """贪心最长匹配的在线版（与 WordSegmenter.tokenize 逐位一致，内存 O(max_len)）。

    用法：
        seg = WordSegmenter(vocab_text, **seg_kwargs)
        for tok in StreamingTokenizer(seg, char_chunks(path)):
            ...  # 逐 token 训练，语料任意长
    """

    def __init__(self, seg, chunks: Iterable[str]):
        self.seg = seg
        self._src = iter(chunks)
        self._cur = ""            # 当前字符块未消费部分
        self.buf = ""             # 分词缓冲
        self.src_done = False     # 源耗尽标志

    def _ensure(self, need: int) -> bool:
        """把缓冲补到 ≥ need 字符；源耗尽则尽力填并返回 False。"""
        while len(self.buf) < need:
            if self._cur:
                take = min(len(self._cur), need - len(self.buf))
                self.buf += self._cur[:take]
                self._cur = self._cur[take:]
                continue
            try:
                self._cur = next(self._src)
            except StopIteration:
                self.src_done = True
                return False
        return True

    def __iter__(self) -> "StreamingTokenizer":
        return self

    def __next__(self) -> str:
        max_len = self.seg.max_len
        if not self.src_done:
            self._ensure(max_len)                 # 决策前保证完整右侧上下文
        if not self.buf:
            raise StopIteration
        for L in range(min(max_len, len(self.buf)), 1, -1):
            if self.buf[:L] in self.seg.vocab:    # 与全量版同一贪心判定
                tok = self.buf[:L]
                self.buf = self.buf[L:]
                return tok
        # 单字符回退：源未耗尽且缓冲不满 max_len 时，先补满再决策
        # （否则 L 上界与全量版不一致，可能过早回退）
        if len(self.buf) < max_len and not self.src_done and self._ensure(max_len):
            return self.__next__()
        tok = self.buf[0]
        self.buf = self.buf[1:]
        return tok
