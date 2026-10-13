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
  `gate_stream_tokenize.py` 对拍验证（多块切分 vs 全量，逐位一致 PASS）。

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

# 预取**线程**数（P192，2026-10-11「训练时总是只有一个主进程在跑，其余进程
# 都没有用处，删掉，部分地方用线程代替进程」）：
# ① **零序列化**：mp.Queue 要 pickle 每个批次（单批 64 条长文本 = 几十 KB~MB
#    的序列化+反序列化，全在 GIL 下白烧 CPU）；thread + queue.Queue 传引用。
# ② **零常驻内存**：每进程一份 Python+pyarrow 上下文 ~100–200 MB（16 进程 =
#    1.6–3.2 GB，大部分时间 0% CPU —— 用户观察到的「一堆进程在睡觉」）。
# ③ **GIL 不是障碍**：真重活是 pyarrow 的 C++ 解码（`iter_batches`/
#    `to_pylist` **都释放 GIL**）→ 线程的并行度与多进程版相同。
# 保留上限 8：跨文件并行 + 队列囤积（数据供给可到 1M tokens 级）。
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

    from phdnet.corpus import expand_paths, open_parquet_source
    for p in expand_paths(path):
        pf = open_parquet_source(p)
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
    """生产者**线程**体：顺序读取分派给本线程的文件，按批推入有界队列。

    tasks = [(全局文件序号 gfi, 路径)]（连续片段）；
    每批推送 (gfi, [样本+sep, ...])；文件读完推 (gfi, None)（文件耗尽标记）；
    线程结束推 None（退出哨兵）。异常透传给消费者（Ctrl+C 按流结束处理）。

    ⚠ **P192：进程 → 线程**。理由（三条，均已实测/可推）：
      ① **零拷贝**：mp.Queue 要 pickle 整个批次（本项目单批 64 条长文本，
         每步几十 KB~MB 的序列化 + 反序列化，全在 GIL 下白烧 CPU）；
         thread + queue.Queue 直接传 Python 对象引用。
      ② **零进程上下文**：每进程一份 Python+pyarrow 上下文 ~100–200 MB
         （191 核机器上 16 进程 = 1.6–3.2 GB 常驻，且大部分时间 0% CPU ——
         这正是用户观察到的「一堆进程在睡觉」）。
      ③ **GIL 不是障碍**：真正的重活在 pyarrow 的 C++ 解码与
         `iter_batches`/`to_pylist`，两者**都释放 GIL**；线程真正并行的
         部分与多进程版完全相同。
    ⚠ 保留 `daemon` 语义：线程随主进程退出，不阻止解释器关闭。
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
    """多**线程**数据加载（P192 从多进程改造，2026-10-11）。

    W 个生产者**线程**（默认 min(8, 核数×0.8, 文件数)）按**连续文件片段**并行
    解码 parquet/txt，各批带全局文件序号推入有界队列；主线程按文件顺序
    reorder 归并 → 产出与 char_chunks **逐位一致**（同一 iter_texts、同一
    文件顺序、批内同序）。

    **为什么抛弃多进程**（P192，用户 2026-10-11 指令「训练时总是只有一个主进程
    在跑，其余的进程都没有用处，删掉，部分地方用线程代替进程」）：
      ① **零序列化**：`mp.Queue` 要 pickle 每个批次（单批 64 条长文本 =
         几十 KB~MB 的序列化 + 反序列化，全在 GIL 下白烧 CPU，且占主线程）；
         `queue.Queue` 直接传 Python 对象引用，**零拷贝**。
      ② **零常驻内存**：每进程一份 Python + pyarrow 上下文 ~100–200 MB；
         16 进程 = 1.6–3.2 GB，且**大部分时间 0% CPU**（生产者等网络阻塞时
         进程空转）——这正是「一堆进程在睡觉」的成因。线程共享同一份解释器
         与 pyarrow 上下文，没有这层开销。
      ③ **并行度不损失**：真重活是 pyarrow 的 C++ 解码（`iter_batches` /
         `to_pylist` **都释放 GIL**），故多个解码线程真并行。串行的只有
         Python 层的 `to_pylist` 装箱，而它在多进程版里**同样**存在
         （只是发生在子进程里，还额外付一次 IPC 序列化）。
      ④ **关闭可控**：进程版 `terminate()` 有 SIGTERM 竞态（用户观察到的
         僵尸进程）；线程版是 daemon + 关闭标志，训练结束即随解释器退出。

    「训练过程不要存在等待代码」：主线程只做 `queue.get(timeout=0.5)`
    （条件变量阻塞，无 sleep/轮询/忙等；超时仅为响应 `close()`）；队列有界
    （depth 批 × batch_samples 样本）→ 背压自动限内存。

    ⚠ **depth 的有效性与 reorder 直接相关**：生产者按连续文件片段推批，
    主线程按全局文件序消费。因此"提前囤积"的批数上限 = **生产者数**。
      ① `workers`（生产者线程数，多文件并发解码）；
      ② `batch_samples`（每批样本数，增大单次传输量、减少唤醒次数）。
    队列仍为有界（`maxsize=depth`）→ 背压始终成立。
    单文件语料自动退化为 1 个生产者（与旧串行版等价）。
    """

    def __init__(self, path, sep: str = SEP, depth: int = 8192,
                 batch_samples: int = 64, workers: int = 0,
                 lang: str | None = None):
        import queue as _queue
        import threading as _threading
        from phdnet.corpus import expand_paths

        from vocab_parallel import auto_workers
        files = [str(p) for p in expand_paths(path)]
        cpu = auto_workers()
        w = min(PREFETCH_MAX_PROCS, cpu, len(files)) if workers in (0, None) \
            else int(workers)
        w = max(1, min(w, len(files)))
        self._n_files = len(files)
        self._n_workers = w
        # 解码核在 W 个**线程**间分摊：W × decode_threads ≈ cpu（避免超订）。
        # ⚠ pyarrow 的 `set_cpu_count` 是**进程级全局**（不是线程级），多线程
        #   并发设置会互相覆盖 → 只在第一个线程里设一次，其余沿用。
        self._decode_threads = max(1, cpu // w)

        self._depth = int(depth) if depth and int(depth) > 0 else 8192
        self._q: "_queue.Queue" = _queue.Queue(maxsize=self._depth)
        self._threads: list = []
        bounds = [self._n_files * i // w for i in range(w + 1)]   # 连续均分
        for i in range(w):
            tasks = [(gfi, files[gfi]) for gfi in range(bounds[i], bounds[i + 1])]
            th = _threading.Thread(
                target=_prefetch_producer,
                args=(tasks, sep, batch_samples, self._q, lang,
                      self._decode_threads),
                name=f"prefetch-{i}", daemon=True)
            th.start()
            self._threads.append(th)

        # 归并状态（按文件顺序消费）
        self._expect = 0               # 期待的文件序号
        self._buffer: dict = {}        # gfi -> deque[批样本列表]
        self._file_done: set = set()   # 已耗尽文件
        self._done_workers = 0
        self._cur: list = []
        self._cur_i = 0
        self._closed = False          # P192：close() 置位，供 _refill 的超时分支识别

    def _refill(self) -> bool:
        """装下一批到 self._cur；语料流结束返回 False。

        ⚠ **P192：OS 级阻塞退化为「带超时的条件变量等待」**。多进程版的
        `mp.Queue.get()` 是**无限阻塞**（真正的「无等待代码」）；线程版的
        `queue.Queue.get()` 同样是无限阻塞 —— 但线程被 GIL 抢占时**不会
        丢 CPU**，且生产者线程的 pyarrow 解码释放 GIL，故并发度不损失。
        这里加 timeout 是为了能响应 `close()`（否则 daemon 线程被 terminate
        后消费者永久阻塞）。超时不影响吞吐：正常路径每步都立即拿到数据。
        """
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
            try:
                item = self._q.get(timeout=0.5)
            except Exception:                       # Empty / 关闭
                if self._closed:
                    return False
                continue
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

    def close(self) -> None:
        """显式清理：置关闭标志并清空队列（线程是 daemon，随主进程退出）。

        ⚠ **P192 语义变更**：进程版的 `terminate()` 对线程**无效**（Python 没有
        杀线程的 API）。故改为「置 `_closed` 标志 → 消费者在 0.5 s 超时后
        正常返回 → 主循环退出 → daemon 线程随解释器退出」。代价是关闭最多
        延迟 0.5 s（仅在训练结束时，不在热路径上）。
        """
        self._closed = True
        try:
            while True:                            # 排空队列，释放引用
                self._q.get_nowait()
        except Exception:                           # Empty
            pass
        self._threads.clear()

    def __enter__(self) -> "PrefetchChars":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


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
                 assistant_marker: str | None = None,
                 user_marker: str = "用户："):
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
        # 2026-10-07 修复（审计 2）：原 :271 硬编码「用户：」→ 参数化 user_marker
        # （默认值不变；英文语料可传 "User: "，多语言数据不再被中文标记绑死）。
        self.user_marker = user_marker or "用户："
        self._mode = False         # 当前是否处于助手回复段（计损失）
        # 2026-10-07 修复（审计 2）：pending 原来只存「剩余字符数」并对 token
        # 无条件减法 → marker 长于 seg.max_len 时越减越负（实测 -20）、
        # :318 失配分支不可达、算术凑 0 会误切模式（marker 未出现却开学习）。
        # 现在存「已消费字符数 + 所属 marker」，消费前按 marker 实际前缀校验。
        self._pending = 0          # 已消费的 marker 字符数，钳制在 [0, len(marker)]
        self._pending_mk: str | None = None   # 正在匹配哪个 marker（None=未在匹配）
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
        # 2026-10-07 修复（审计 2）：改按 marker 的**实际前缀内容**校验——原实现
        # 用 buf.startswith(tok)，而 tok 恒为 buf[:L] → 条件恒真、:318 失配分支
        # 不可达，且 pending 对 token 长度无条件减法会在 marker 长于 seg.max_len
        # 时越减越负（实测 -20 → 纯回复文本 trainable=0），算术凑 0 还会误切模式
        # （marker 从未出现却开学习）。现在 pending 钳制在 [0, len(marker)]。
        if self._pending:
            mk = self._pending_mk
            if not mk:
                self._pending = 0                        # 陈旧 pending（无 marker 记录）→ 复位
            else:
                p = min(max(self._pending, 0), len(mk))  # 钳制到 [0, len(marker)]
                rest = mk[p:]                            # marker 还差的字符
                if rest.startswith(tok):                 # token 完全落在 marker 内
                    self._pending = p + len(tok)
                    self.buf = buf[len(tok):]
                    if self._pending >= len(mk):         # 恰好消费完 → 切模式
                        self._mode = self._pending_mode
                        self._pending = 0
                        self._pending_mk = None
                    return (tok, False)                  # marker 字符不计损失
                if tok.startswith(rest):                 # marker 在本 token 内部结束
                    self._pending = 0
                    self._pending_mk = None
                    self._mode = self._pending_mode      # marker 完成 → 立即切模式
                    self.buf = buf[len(tok):]
                    return (tok, False)                  # token 含 marker 尾巴 → 本步不计损失
                self._pending = 0                        # 失配 → 复位（审计 2：现已真正可达）
                self._pending_mk = None
                # 复位后落到 ②/③ 对**同一 buf** 重新判定（不预先消费）
        # ② 待消费的 token 是否命中某个 marker（完整或前缀）
        # 2026-10-07 修复（审计 2）：以 token 本身判定并只消费 token——原 head =
        # buf[:len(mk)] 会一次吞掉至多 len(mk) 字符，越过分词边界。
        for mk, mode in ((self.assistant_marker, True),
                         (self.user_marker, False)):
            if not mk:
                continue
            if tok == mk:                                # 完整命中 → 立即切换
                self.buf = buf[len(tok):]
                self._mode = mode
                return (tok, False)
            if len(tok) < len(mk) and mk.startswith(tok):  # 部分命中 → 待续
                self.buf = buf[len(tok):]
                self._pending = len(tok)                 # 已消费前缀长度 ∈ (0, len(mk))
                self._pending_mk = mk
                self._pending_mode = mode
                return (tok, False)
            if tok.startswith(mk) and len(tok) > len(mk):
                # ── 返工 4：贪心 token **整段吞下 marker 且更长** ────────────
                #   例：vocab 里有 "Assistant: 你好"（13 > len("Assistant: ")=11）。
                #   旧版只有 `tok == mk` 与 `len(tok) < len(mk)` 两条 → 本例
                #   落进分支③，marker **整个丢失**、模式永不切换、助手回复全程
                #   不计损失（静默零学习）。现在：**切模式** + 按 marker 长度
                #   处理可训练位（token 超出 marker 的 len(tok)-len(mk) 个字符
                #   属于新模式的正文 → 按新模式计损失）。
                #   ⚠ 无 marker 语料走不到这里：函数首部 `if not
                #   self.assistant_marker` 已原样返回，路径逐字节不变。
                self.buf = buf[len(tok):]
                self._mode = mode
                return (tok, mode)
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
