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

# 预取进程数（P16，2026-09-28 fhz「默认一个进程」）：
# ① parquet 解码在 pyarrow 内**自身多线程且释放 GIL**，单进程即可吃满核；
# ② 进程数越多，每进程一份 Python+pyarrow 上下文（~100–200 MB）线性增长；
# ③ 主进程的分词/扫描是 numba nogil 线程（真正的多核来源）。
# P37：默认 auto = min(8, 核数, 文件数) 多进程（跨文件并行 + 队列囤积，
# 数据供给可到 1M tokens 级）；--prefetch-workers 显式优先。
# P37（fhz「预取提升到 1M tokens」）：多生产者是**必要条件**——单生产者受
# 「按文件序归并」约束，囤积≈0，depth 调多大都不增加实际在途量。8 进程 ×
# depth 8192 批 → 在途可达 52 万样本 ≈ 数百万 tokens（每进程 ~150 MB 上下文，
# 服务器内存充裕）。训练稳态默认仍由 train.py 传 workers 控制。
PREFETCH_MAX_PROCS = 8


def char_chunks(path, sep: str = SEP) -> Iterator[str]:
    """语料字符块流：逐样本 yield（样本 + 分隔符）；path 支持单文件或 glob。"""
    from phdnet.corpus import iter_texts
    for t in iter_texts(path):
        yield t + sep


def iter_texts_lang(path, lang: str | None = None, decode_threads: int = 0):
    """按 lang 字段过滤的流式读取（lang=None 时等价 iter_texts；仅 parquet 支持过滤）。"""
    if lang is None:
        from phdnet.corpus import iter_texts
        yield from iter_texts(path)
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    if decode_threads > 0:
        pa.set_cpu_count(decode_threads)     # P16：按进程数分摊解码核
        pa.set_io_thread_count(max(1, decode_threads // 2))

    from phdnet.corpus import expand_paths
    for p in expand_paths(path):
        pf = pq.ParquetFile(str(p))
        cols = [c for c in ("text", "lang") if c in pf.schema_arrow.names]
        if "lang" not in cols:
            for b in pf.iter_batches(batch_size=2048, columns=["text"]):
                for v in b.column(0).to_pylist():
                    if v:
                        yield v
            continue
        for b in pf.iter_batches(batch_size=2048, columns=cols):
            for t, l in zip(b.column("text").to_pylist(), b.column("lang").to_pylist()):
                if t and l == lang:
                    yield t


def zh_char_chunks(path, lang: str = "zh", sep: str = SEP) -> Iterator[str]:
    """按 `lang` 字段过滤的 parquet 字符块流（串行参考；多进程版见 PrefetchChars(lang=...)）。"""
    for t in iter_texts_lang(path, lang):
        yield t + sep


def _prefetch_producer(tasks, sep: str, batch_samples: int, q,
                       lang: str | None = None,
                       decode_threads: int = 0) -> None:
    """生产者进程体：顺序读取分派给本进程的文件，按批推入有界队列。

    tasks = [(全局文件序号 gfi, 路径)]（连续片段）；
    每批推送 (gfi, [样本+sep, ...])；文件读完推 (gfi, None)（文件耗尽标记）；
    进程结束推 None（退出哨兵）。异常透传给消费者（Ctrl+C 按流结束处理）。
    """
    from phdnet.corpus import iter_texts
    try:
        for gfi, path in tasks:
            buf: list[str] = []
            for t in iter_texts_lang(path, lang,
                                     decode_threads=decode_threads):
                buf.append(t + sep)
                if len(buf) >= batch_samples:
                    q.put((gfi, buf))
                    buf = []
            if buf:
                q.put((gfi, buf))
            q.put((gfi, None))
        q.put(None)
    except BaseException as e:
        try:
            q.put(("__err__", e))
        except Exception:
            pass


class PrefetchChars:
    """多进程数据加载（fhz 2026-09-25：「数据加载也用多核」「能不能多进程」）。

    W 个生产者进程（**P16：默认 min(8, 核数×0.8, 文件数)**——解码在 pyarrow
    内多线程，进程数过多只增内存/调度）按**连续文件片段**并行
    解码 parquet/txt，各批带全局文件序号推入有界队列；主进程按文件顺序
    reorder 归并 → 产出与 char_chunks **逐位一致**（同一 iter_texts、同一
    文件顺序、批内同序）。

    「训练过程不要存在等待代码」：主进程只做 queue.get()（OS 级阻塞，
    无 sleep/轮询/忙等）；队列有界（depth 批 × batch_samples 样本）→
    背压自动限内存。

    **depth 64 → 256 → 8192（P24/P25，fhz「内存没咋占用，提高数据提供量」→
    「提升到 8192」）**。

    ⚠ **depth 的有效性与 reorder 直接相关（重要，别被数字误导）**：生产者按
    **连续文件片段**推批，主进程按**全局文件序**消费（`_buffer` 无上限，但只有
    下一个期待文件的批会被取出）。因此真正能"提前囤积"的批数上限 = **生产者数**
    （每个生产者只推自己的文件，多个生产者时消费者等待某文件时其他文件可囤积）。
    - 单生产者（当前默认）：几乎零囤积 → **depth 调多大都不增加在途量**；
    - 多生产者：有效在途 ≈ 生产者数 × 每生产者已推批数。
    真正提升"数据提供量"的两个杠杆（都不靠 depth）：
      ① `workers`（生产者进程数，多文件并发解码）——注意内存线性增长；
      ② `batch_samples`（每批样本数，增大单次 IPC 传输量，减少唤醒次数）。
    队列仍为有界（`maxsize=depth`）→ 背压始终成立，不会无界吃内存。
    单文件语料自动退化为 1 个生产者（与旧单进程版等价）。
    """

    def __init__(self, path, sep: str = SEP, depth: int = 8192,
                 batch_samples: int = 64, workers: int = 0,
                 lang: str | None = None):
        import multiprocessing as mp
        from phdnet.corpus import expand_paths

        from vocab_parallel import auto_workers
        files = [str(p) for p in expand_paths(path)]
        cpu = auto_workers()
        w = min(PREFETCH_MAX_PROCS, cpu, len(files)) if workers in (0, None) \
            else int(workers)
        w = max(1, min(w, len(files)))
        self._n_files = len(files)
        self._n_workers = w
        # 解码核在 W 个进程间分摊：W × decode_threads ≈ cpu（避免超订）
        self._decode_threads = max(1, cpu // w)

        self._q: "mp.Queue" = mp.Queue(maxsize=depth)
        self._procs = []
        bounds = [self._n_files * i // w for i in range(w + 1)]   # 连续均分
        for i in range(w):
            tasks = [(gfi, files[gfi]) for gfi in range(bounds[i], bounds[i + 1])]
            p = mp.Process(target=_prefetch_producer,
                           args=(tasks, sep, batch_samples, self._q, lang,
                                 self._decode_threads),
                           daemon=True)
            p.start()
            self._procs.append(p)

        # 归并状态（按文件顺序消费）
        self._expect = 0               # 期待的文件序号
        self._buffer: dict = {}        # gfi -> deque[批样本列表]
        self._file_done: set = set()   # 已耗尽文件
        self._done_workers = 0
        self._cur: list = []
        self._cur_i = 0

    def _refill(self) -> bool:
        """装下一批到 self._cur；语料流结束返回 False（OS 级阻塞，无忙等）。"""
        from collections import deque
        while True:
            if self._expect >= self._n_files:
                return False
            file_q = self._buffer.get(self._expect)
            if file_q:
                self._cur = file_q.popleft()
                self._cur_i = 0
                return True
            if self._expect in self._file_done:
                self._expect += 1                  # 该文件完 → 期待下一文件
                continue
            item = self._q.get()
            if item is None:
                self._done_workers += 1
                continue
            head = item[0]
            if isinstance(head, str) and head == "__err__":
                e = item[1]
                if isinstance(e, KeyboardInterrupt):
                    return False                   # 生产者被 Ctrl+C：按流结束
                raise e
            gfi, buf = item
            if buf is None:
                self._file_done.add(gfi)
            else:
                self._buffer.setdefault(gfi, deque()).append(buf)

    def __iter__(self) -> "PrefetchChars":
        return self

    def __next__(self) -> str:
        while self._cur_i >= len(self._cur):
            if not self._refill():
                raise StopIteration
        t = self._cur[self._cur_i]
        self._cur_i += 1
        return t


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

    def __init__(self, seg, chunks: Iterable[str],
                 assistant_marker: str | None = None):
        """`assistant_marker`（P26，如 "助手："）：启用 **SFT 回复掩码**。

        开启后 `__next__` 返回 `(token, trainable)`：
        - 文本中**最后一个** assistant_marker 之前的部分（系统/用户 prompt）
          → `trainable=False`（用 `learn=False` 推进状态，不更新权重）；
        - marker 之后的助手回复 → `trainable=True`（正常计算损失）。
        无 marker 时行为与原来**逐位一致**（仍 yield 纯 str）。
        语义正确性：SFT（instruction tuning）只应对**回复**计损失，对 prompt
        计损失会把模型往「复读用户问题」的方向拉。
        """
        self.seg = seg
        self._src = iter(chunks)
        self._cur = ""            # 当前字符块未消费部分
        self.buf = ""             # 分词缓冲
        self.src_done = False     # 源耗尽标志
        self.assistant_marker = assistant_marker
        # P26：用户侧标记（多轮对话里用于**退出**可训练段）
        self.user_marker = "用户："
        self._mode = False         # 当前是否处于助手回复段（计损失）
        self._pending = 0          # 正在匹配的 marker 剩余字符数
        self._pending_mode = False # 该 marker 匹配完成后的模式

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

    def _consume(self, tok: str):
        """消费一个 token，返回 (token, trainable)（P26 零滞后状态机）。

        之前的「rfind + 游标」实现会**滞后 1~2 个 token**（marker 跨 token 边界时），
        导致 SFT 掩码错位（实测首个可训练 token 是回复中间的「诗。」而不是开头）。
        现在改为**前缀匹配**：任何 token 若落在 marker 字符序列内 → 不可训练；
        marker 刚被消费完的那一步立即切换模式。规则：

            用户标记 → mode=False（不计损失）   助手标记 → mode=True（计损失）

        未设置 assistant_marker 时行为与旧版**逐位一致**（yield 纯 str）。
        """
        if not self.assistant_marker:
            self.buf = self.buf[len(tok):]   # 无掩码模式也必须推进缓冲
            return tok
        buf = self.buf
        # ① 正处于某个 marker 的字符序列中（marker 跨 token 边界）
        if self._pending:
            if buf.startswith(tok):
                self._pending -= len(tok)
                if self._pending == 0:
                    self._mode = self._pending_mode      # marker 消费完 → 切模式
                self.buf = buf[len(tok):]
                return (tok, False)                     # marker 本身不计损失
            self._pending = 0                           # 失配 → 回到普通判定
        # ② 待消费的 buf 前缀是否命中某个 marker（完整或前缀）
        for mk, mode in ((self.assistant_marker, True),
                         (self.user_marker, False)):
            head = buf[:len(mk)]
            if mk.startswith(head):
                self.buf = buf[len(head):]
                if len(head) == len(mk):
                    self._mode = mode                    # 完整命中 → 立即切换
                    return (head, False)                 # marker 字符不计损失
                self._pending = len(mk) - len(head)      # 部分命中 → 待续
                self._pending_mode = mode
                return (head, False)
        # ③ 普通 token：按当前模式
        self.buf = buf[len(tok):]
        return (tok, self._mode)

    def __next__(self):
        max_len = self.seg.max_len
        if not self.src_done:
            self._ensure(max_len)                 # 决策前保证完整右侧上下文
        if not self.buf:
            raise StopIteration
        for L in range(min(max_len, len(self.buf)), 1, -1):
            if self.buf[:L] in self.seg.vocab:    # 与全量版同一贪心判定
                return self._consume(self.buf[:L])
        # 单字符回退：源未耗尽且缓冲不满 max_len 时，先补满再决策
        # （否则 L 上界与全量版不一致，可能过早回退）
        if len(self.buf) < max_len and not self.src_done and self._ensure(max_len):
            return self.__next__()
        return self._consume(self.buf[0])
