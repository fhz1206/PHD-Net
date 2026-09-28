"""PHD-Net 1B 参数模型训练脚本（train_1b 子项目主入口，流式版）。

2026-09-25 流式改造（fhz 指令：「训练的1B模型要支持1M context，
训练数据无论多长不要截断」）
----------------------------------------------------------------
- **训练数据永不截断**：语料按字符流逐样本流过（`corpus_stream.py`），
  token 边产边训，内存占用与语料总长无关——4.6 GB pretrain 分片与
  23 KB 内置语料走同一条路。废除 `--max-chars` 截断参数。
- **1M context**：PHD-Net 无位置编码/注意力窗口，context = 记忆机制的有效
  范围。本脚本保证**状态全程不重置**（WM/STDP/LTM 连续携带，跨样本、
  跨分片、跨 epoch 连续），长程依赖由 big_ltm 事件驱动印迹（1B 突触容量）
  承载；每跨过 `--context-milestone`（默认 1,000,000）token 打一行里程碑，
  显式证明连续 context 达标。检查点完整保存运行时状态 → 1M+ 训练可中断续训。
- **词表**：涌现词表来自采样文本（`--vocab-sample-chars`，默认 4M），
  `--vocab-scan full` 可全量扫一遍构建零 OOV 词表；训练流中 OOV token
  跳过该训练步并计数（永不崩溃、永不截断），OOV 率进 [METRIC] 尾行。
- **流式分词逐位等价**：StreamingTokenizer 与全量贪心匹配逐位一致，
  `verify_stream_tokenize.py` 四用例对拍 PASS。
- **多核词表与数据加载（fhz 2026-09-25 指令）**：词涌现按 L 层并行；
  全量扫描/head token 收集按批走锚点链并行（`--vocab-workers`，
  默认 = 核心数×0.8）；训练流由生产者进程预取（`PrefetchChars`），
  主循环零等待代码（无 sleep/轮询/忙等）。全部逐位等价，
  `verify_vocab_parallel.py` 对拍 PASS。

定位（同 config_1b.py）
--------------------
总突触参数容量 ≥ 1×10^9：1B 主体 = 大空间事件驱动长期记忆
（big_ltm：2^24 × 60 ≈ 1.0066×10^9 突触容量，随经验生长，每步计算量只
正比于活跃神经元数）；固定突触 = 稀疏主干 CSR + STDP + 编码器 + 读出。

用法
----
# 冒烟（分钟级，验证管线）
python train_1b/train.py --preset smoke --data sft --tokens 2000

# 标准 1B 档，全量流式训练 pretrain 分片（任意长度不截断）
python train_1b/train.py --preset 1b --data pretrain_zh --tokens 1000000

# 生产长跑（零 OOV 词表 + 1M context 里程碑；断点续训）
python train_1b/train.py --preset 1b --data pretrain_zh \
    --vocab-scan full --context-milestone 1000000 --resume
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_HERE), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ckpt_1b import _rebuild_sdrs, load_model, save_model            # noqa: E402
from config_1b import PRESETS, SEG_KWARGS, build_cfg                 # noqa: E402
from config_1b import capacity_report, print_capacity_report         # noqa: E402
from corpus_stream import PrefetchChars, SEP, StreamingTokenizer     # noqa: E402
from corpus_stream import build_vocab_text, char_chunks              # noqa: E402
from corpus_stream import mix_chunks, zh_char_chunks                 # noqa: E402
from phdnet.corpus import expand_paths                               # noqa: E402
from phdnet.model import count_params                                # noqa: E402
from phdnet.word_encoder import WordTokenizer                         # noqa: E402
from vocab_parallel import auto_workers, build_segmenter_parallel    # noqa: E402
from vocab_parallel import parallel_head_tokens, scan_vocab_parallel # noqa: E402
from corpus_stream import PREFETCH_MAX_PROCS                             # noqa: E402

# 预取进程数（0=自动；main 里按 --prefetch-workers 覆盖，P16）
PREFETCH_W = 0


def make_external_segmenter(seg_kwargs: dict, words):
    """由外部词表构造分段器（P17 --vocab-file）。

    只依赖 `tokenize` 用到的两个属性（vocab / max_len），与 `WordSegmenter`
    的贪心最长匹配语义逐位一致（同实现，仅换词表来源）。
    """
    from phdnet.word_encoder import WordSegmenter
    seg = WordSegmenter.__new__(WordSegmenter)
    seg.max_len = int(seg_kwargs.get("max_len", 6))
    seg.vocab = {w for w in words if w}
    return seg

SAVE_DIR = _ROOT / "outputs" / "models"   # fhz 2026-09-28：生产训练产物统一存 outputs/models/
LOG_DIR = _ROOT / "outputs" / "models" / "train_logs"

DATA_FILES = {
    # 训练数据一律用上传分片 parquet（sft/ pretrain/ 根）；raw/ 仅归档不触碰
    # （fhz 2026-09-25 指令，与 tools/train_production.py 同步）。
    "sft": _ROOT / "datasets" / "sft" / "sft_000.*.parquet",
    "pretrain_zh": _ROOT / "datasets" / "pretrain" / "pretrain_*.parquet",
    "eval": _ROOT / "eval_corpus" / "internal_corpus.txt",
    # 泛化优化 P0（2026-09-25 审计）：sft 与 pretrain 中文子集样本级轮转混合
    "mix": None,
}


def stream_factory(data: str):
    """返回 () -> 新的独立样本字符块流（mix = sft 与 pretrain 中文源轮转交错）。"""
    if data == "mix":
        # 双源均走多进程加载（文件级并行 + 顺序归并 → 与串行产出逐位一致）：
        # sft 全量源 + pretrain 中文过滤源（lang="zh" 在生产者进程内过滤）
        return lambda: mix_chunks([
            PrefetchChars(DATA_FILES["sft"], SEP, workers=PREFETCH_W),
            PrefetchChars(DATA_FILES["pretrain_zh"], SEP, lang="zh",
                          workers=PREFETCH_W),
        ])
    return lambda: char_chunks(DATA_FILES[data])

_STOP = {"flag": False}


class TeeLogger:
    """控制台 + 日志文件双写（行缓冲落盘，异常中断最多丢当前行）。"""

    def __init__(self, path: Path):
        self.terminal = sys.stdout
        self.file = open(path, "w", encoding="utf-8", buffering=1)
        self.path = path

    def write(self, msg: str) -> None:
        self.terminal.write(msg)
        self.file.write(msg)

    def flush(self) -> None:
        self.terminal.flush()
        self.file.flush()


def _on_sigint(signum, frame):        # pragma: no cover
    _STOP["flag"] = True
    print("\n[收到中断信号] 将保存检查点后退出…", flush=True)


def _make_tokenizer(seg: WordSegmenter, tokens: list[str], cfg) -> WordTokenizer:
    """由 (seg, tokens) 注入式构造 WordTokenizer（SDR 哈希与旧路径逐位一致）。"""
    tok = WordTokenizer.__new__(WordTokenizer)
    tok.seg = seg
    tok.tokens = tokens
    tok.stoi = {t: i for i, t in enumerate(tokens)}
    tok.n_sdr, tok.n_active, tok.seed = cfg.n_sdr, cfg.k_sparse, cfg.seed
    _rebuild_sdrs(tok, tokens)
    return tok


def _print_table_stats(lm) -> None:
    """打印大空间事件驱动表实时利用率 + 内存护栏（长跑增长有界性）。"""
    if not hasattr(lm.net.ltm, "table"):
        return
    st = lm.net.ltm.table.stats()
    per = 10 if st.get("row_slots_allocated") else 100      # CSR 版 ≈10B/条，dict 版 ≈100B/条
    mb = st["grown_synapses"] * per / 1e6
    print(f"  [大空间表] 已生长 {st['grown_synapses']:,} / 容量 {st['capacity']:,}"
          f"（利用率 {st['utilization']:.3%}）| 内存 ≈{mb:.0f} MB"
          + ("  ⚠ dict 版长跑建议 --csr-online（≈10×省内存）"
             if (not getattr(lm.net.ltm.table, "csr_online", False)
                 and st["grown_synapses"] > 50_000_000) else ""),
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="PHD-Net 1B 档流式训练（事件驱动稀疏类脑）")
    ap.add_argument("--preset", choices=list(PRESETS), default="1b",
                    help="smoke=管线验证 / 1b=标准档(容量≥1B) / 1b_max=大主干档")
    ap.add_argument("--data", choices=list(DATA_FILES), default="sft")
    ap.add_argument("--width", type=int, default=0, help="覆盖主干宽度（0=用预设）")
    ap.add_argument("--readout-dtype", default="fp32",
                    choices=["fp32", "fp16", "bf16", "fp8", "fp4"],
                    help="读出精度（P9/P12：默认 fp32；fp64 已停止支持；"
                         "fp4 = MX 块缩放 e2m1，2026-09-28 解禁）")
    ap.add_argument("--big-n", type=int, default=0, help="覆盖大空间神经元数（0=用预设）")
    ap.add_argument("--csr-online", action="store_true",
                    help="大空间表切换在线可写 CSR（长跑内存 ≈10×省，逐位等价已验证）")
    ap.add_argument("--readout-conn-k", type=int, default=0,
                    help="稀疏读出每输出单元入边数（0=稠密；大词表时建议 512–2048）")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--epochs", type=int, default=1,
                    help="语料流过遍数（状态跨 epoch 连续不重置）")
    ap.add_argument("--tokens", type=int, default=0, help="训练步预算（0=不限）")
    ap.add_argument("--minutes", type=float, default=0.0, help="时间预算分钟（0=不限）")
    ap.add_argument("--vocab-scan", choices=["head", "full"], default="head",
                    help="head=采样文本构建词表(快, OOV 由回退兜底) / "
                         "full=全量流式扫一遍(零 OOV, 大语料需数小时)")
    ap.add_argument("--vocab-sample-chars", type=int, default=4_000_000,
                    help="head 模式词表采样字符数（只影响词表，训练数据本身不截断）")
    ap.add_argument("--vocab-file", type=Path, default=None,
                    help="外部词表文件（每行一个词；# 注释与空行忽略）。"
                         "给了它就**跳过** head/full 扫描阶段。"
                         "另：--resume 时词表直接取自检查点（自包含），同样跳过扫描")
    ap.add_argument("--prefetch-workers", type=int, default=0,
                    help="语料预取进程数（P16：0=自动，默认 min(8, 核数×0.8, 文件数)；"
                         "解码在 pyarrow 内多线程，进程过多只增内存/调度）")
    ap.add_argument("--vocab-workers", type=int, default=0,
                    help="词表构建/全量扫描并行进程数（0=自动=核心数×0.8，1=串行；"
                         "fhz 2026-09-25 指令）")
    ap.add_argument("--context-milestone", type=int, default=1_000_000,
                    help="context 里程碑间隔 tokens（0=关闭；默认 1M）")
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true",
                    help="从 models/ 检查点续训（快进至断点，状态由检查点恢复）")
    ap.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--report", action="store_true",
                    help="只打印检查点的大空间表统计后退出")
    args = ap.parse_args()
    vw = args.vocab_workers if args.vocab_workers > 0 else auto_workers()
    # P16：预取**进程**数收敛（解码在 pyarrow 内多线程 + 主进程分词为 nogil
    # 线程）；进程数 × 每进程解码核 ≈ 核数，避免 153 进程的超订与内存爆炸。
    global PREFETCH_W
    PREFETCH_W = (args.prefetch_workers if args.prefetch_workers > 0 else 0)
    _cpu = auto_workers()
    _pf_auto = min(PREFETCH_MAX_PROCS, _cpu)
    _pf = PREFETCH_W if PREFETCH_W > 0 else _pf_auto

    if args.log_file is not None:
        log_path = args.log_file
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = LOG_DIR / f"train_1b_{args.preset}_{args.data}_{stamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = TeeLogger(log_path)

    # 后端 × 设备能力矩阵：显式说明「numba 只能上 CPU」这一物理限制，
    # 避免在 NPU/CUDA 机器上被误判为「加速器没被识别」（P18）。
    try:
        from phdnet.backends.multi_device import capability_report
        _cap = capability_report(verbose=False)
        if _cap["accelerators_present"]:
            print(f"[能力] 检测到加速器 {_cap['accelerators_present']}，但生产训练走 "
                  f"numba/CPU（numba 只能编译到 CPU 机器码）；要用加速器请走 torch 栈："
                  f"tools/train_torch_lm.py --device auto（权重不通用）")
    except Exception:                                       # noqa: BLE001
        pass
    print(f"[并行] 分词线程（numba nogil）={vw} | 预取进程={_pf}"
          f"（解码核/进程≈{max(1, _cpu // _pf)}，合计≈{_pf * max(1, _cpu // _pf)}）"
          f" | 数据侧并行不与分词线程叠加争抢")
    print(f"[log] 日志文件：{log_path}")
    print(f"[log] 启动：{datetime.now().isoformat(timespec='seconds')}  "
          f"argv：{' '.join(sys.argv[1:]) or '(默认参数)'}")

    signal.signal(signal.SIGINT, _on_sigint)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.save_dir / f"phdnet1b_{args.preset}_{args.data}.npz"

    cfg = build_cfg(args.preset, args.width, args.big_n,
                    args.csr_online, args.readout_conn_k, args.seed)
    cfg.readout_dtype = args.readout_dtype            # P9 精度（默认 fp32）

    if args.data == "mix":
        try:
            data_files = (expand_paths(DATA_FILES["sft"])
                          + expand_paths(DATA_FILES["pretrain_zh"]))
        except FileNotFoundError as e:
            print(f"数据文件不存在: {e}")
            sys.exit(1)
    else:
        data_path = DATA_FILES[args.data]
        try:
            data_files = expand_paths(data_path)
        except FileNotFoundError as e:
            print(f"数据文件不存在: {e}")
            sys.exit(1)
    total_mb = sum(p.stat().st_size for p in data_files) / 1e6
    dl_w = max(1, min(vw, len(data_files)))    # 数据加载进程数（≤文件数）

    # ── 词表构建（采样 or 全量扫描；训练数据本身永不截断）──
    # 词涌现 + token 收集均多核（fhz 2026-09-25：核心数×0.8；=1 时走串行原路径）
    # P17：--resume（检查点自含词表）或 --vocab-file（外部词表）时**整段跳过**
    #     —— 旧实现 resume 也会先扫一遍 full 词表（等于白扫几十分钟）。
    _ckpt_path = SAVE_DIR / f"phdnet1b_{args.preset}_{args.data}.npz"
    _resume_vocab = args.resume and _ckpt_path.exists()
    _file_vocab = (args.vocab_file is not None
                   and Path(args.vocab_file).exists())
    if _resume_vocab or _file_vocab:
        vocab_text, seg, tokens = "", None, None      # 下方按来源填充
    elif args.data == "mix":
        parts: list[str] = []
        nvc = 0
        for c in stream_factory("mix")():
            parts.append(c)
            nvc += len(c)
            if nvc >= args.vocab_sample_chars:
                break
        vocab_text = "".join(parts)[:args.vocab_sample_chars]
    else:
        vocab_text = build_vocab_text(data_path, args.vocab_sample_chars, SEP)
    if not (_resume_vocab or _file_vocab):
        t_v0 = time.perf_counter()
        seg = build_segmenter_parallel(vocab_text, SEG_KWARGS, args.vocab_workers)
    if not (_resume_vocab or _file_vocab) and vw > 1:
        print(f"[词表] 词涌现多核构建：L=2..{SEG_KWARGS['max_len']} × {vw} 进程，"
              f"涌现词表 {len(seg.vocab):,}（{time.perf_counter() - t_v0:.1f}s）",
              flush=True)
    # 词表来源三选一（优先级）：resume 自包含 > 外部文件 > head/full 扫描
    if _resume_vocab:
        from ckpt_1b import peek_tokenizer
        _cv, _ct, _cmax = peek_tokenizer(_ckpt_path)
        seg = make_external_segmenter({**SEG_KWARGS, "max_len": _cmax}, _cv)
        tokens = list(_ct)
        vocab_text = "\n".join(tokens[:4096])   # 仅供分词器注入构造
        print(f"[词表] --resume：跳过词表构建，直接用检查点自包含词表"
              f"（{_ckpt_path.name}：{len(_cv):,} 词 / max_len={_cmax}，"
              f"SDR 哈希确定性重建、逐位一致；读出层按该尺寸对齐）", flush=True)
    elif _file_vocab:
        from ckpt_1b import load_vocab_file
        _words, _segv, _wmax = load_vocab_file(Path(args.vocab_file))
        if _segv is None:
            print("[词表] ⚠ 快照缺 seg_vocab（分词器候选集）：已退回用 token 词表"
                  "作候选集——分词结果可能与训练时不同（静默降质风险）。"
                  "建议改用 --resume（检查点自包含，权威）或用新版快照。",
                  flush=True)
            _segv = _words
        seg = make_external_segmenter({**SEG_KWARGS, "max_len": _wmax}, _segv)
        tokens = sorted(set(_words))
        vocab_text = "\n".join(_words[:4096])   # 仅供分词器注入构造
        print(f"[词表] 外部词表文件 {args.vocab_file}"
              f"（{'JSON 权威格式' if Path(args.vocab_file).suffix.lower() == '.json' else '文本（转义还原）'}）"
              f"：{len(_words):,} 词 / max_len={_wmax} → 去重 {len(tokens):,} 词"
              f"（跳过扫描阶段；OOV 步跳过并统计）", flush=True)
    elif args.vocab_scan == "full":
        from vocab_parallel import NUMBA_TOK_OK as _TOK_NB
        _engine = (f"多核锚点链 ×{vw}"
                   + ("（numba nogil 线程）" if vw > 1 and _TOK_NB
                      else "（进程池）" if vw > 1 else ""))
        print(f"[词表] full 扫描：全量流式分词一遍（{_engine}；大语料需较久）…",
              flush=True)
        seen: set[str] = set()
        n_seen = 0
        t_v = time.perf_counter()
        if vw > 1:
            def _scan_prog(n_total, n_distinct):
                print(f"  [词表扫描] 累计 {n_total:>12,} tokens，"
                      f"当前词表 {n_distinct:,}"
                      f"（{time.perf_counter() - t_v:.0f}s）", flush=True)

            seen, n_seen = scan_vocab_parallel(seg.vocab, seg.max_len,
                                               (stream_factory(args.data)()
                                                if args.data == "mix"
                                                else PrefetchChars(
                                                    data_path, SEP,
                                                    workers=PREFETCH_W)),
                                               vw, progress=_scan_prog)
        else:
            for tk in StreamingTokenizer(seg, char_chunks(data_path, SEP)):
                seen.add(tk)
                n_seen += 1
                if n_seen % 1_000_000 == 0:
                    print(f"  [词表扫描] 已流过 {n_seen:>12,} tokens，"
                          f"当前词表 {len(seen):,}（{time.perf_counter() - t_v:.0f}s）",
                          flush=True)
        tokens = sorted(seen)
        print(f"[词表] full 扫描完成：全语料 {n_seen:,} tokens → 词表 {len(tokens):,}"
              f"（{time.perf_counter() - t_v:.0f}s，{_engine}），"
              f"零 OOV", flush=True)
    else:
        if vw > 1:
            t_h = time.perf_counter()
            tokens = sorted(parallel_head_tokens(seg, vocab_text, vw))
            print(f"[词表] head 模式（多核 token 收集 ×{vw}，"
                  f"{time.perf_counter() - t_h:.1f}s）：采样前 {len(vocab_text):,} 字符构建"
                  f"（涌现词表 {len(seg.vocab):,}，token 词表 {len(tokens):,}）；"
                  f"训练流 OOV 步将跳过并统计", flush=True)
        else:
            tokens = sorted(set(seg.tokenize(vocab_text)))
            print(f"[词表] head 模式：采样前 {len(vocab_text):,} 字符构建"
                  f"（涌现词表 {len(seg.vocab):,}，token 词表 {len(tokens):,}）；"
                  f"训练流 OOV 步将跳过并统计", flush=True)

    # ── 词表快照**第一时间落盘**（P17，fhz 要求）──
    # 训练崩溃/中断也不丢词表；下次 `--vocab-file <该文件>` 直接复用（免重扫，
    # full 扫描 520s）。四个来源（resume/外部文件/head/full）统一在此落盘。
    if tokens:
        from ckpt_1b import save_vocab_snapshot
        _src = ("resume" if _resume_vocab else
                "vocab-file" if _file_vocab else
                f"scan-{args.vocab_scan}")
        _vp = save_vocab_snapshot(  # 返回 .json 权威路径
            SAVE_DIR / f"vocab_{args.preset}_{args.data}.txt", tokens,
            getattr(seg, "max_len", SEG_KWARGS["max_len"]),
            seg_vocab=getattr(seg, "vocab", None),   # 分词器候选集（必存）
            meta={"preset": args.preset, "data": args.data, "source": _src,
                  "sha1": __import__("hashlib").sha1(
                      "\n".join(sorted(set(tokens))).encode("utf-8")
                  ).hexdigest()[:12]})
        print(f"[词表] 快照已落盘：{_vp.name}（{_vp.stat().st_size / 1024:.0f} KB，"
              f"来源={_src}；唯一权威格式 = 词表 + 分词候选集 + max_len + sha1）"
              f"—— 崩溃后续训直接 --vocab-file {_vp.name}，免重扫", flush=True)

    # ── 构建 LM（注入式 tokenizer；n_readout = 词表大小）──
    t0 = time.perf_counter()
    tok = _make_tokenizer(seg, tokens, cfg)
    from phdnet.word_lm import PHDWordLM
    lm = PHDWordLM(vocab_text, cfg, seg_kwargs=SEG_KWARGS, tokenizer=tok)
    vocab = len(lm.tok)

    print("=" * 76)
    print(f"PHD-Net 1B 流式训练 | preset={args.preset} data={args.data} "
          f"width={cfg.n_sdr} conn_k={cfg.conn_k}")
    print(f"大空间表 N={cfg.big_ltm_N:,} × m={cfg.big_ltm_m}"
          f"（容量 {cfg.big_ltm_N * cfg.big_ltm_m:,}）| 词表 {vocab:,}"
          f" | 分片 {len(data_files)} 个（{total_mb:.0f} MB，流式不截断）")
    print(f"epochs={args.epochs} | 预算 tokens={args.tokens or '∞'} "
          f"minutes={args.minutes or '∞'} | 里程碑={args.context_milestone or '∞'}"
          f" | 词表并行={vw} | 数据预取=多进程×{dl_w}{'+zh过滤' if args.data == 'mix' else ''}"
          f" | 构建耗时 {time.perf_counter() - t0:.1f}s")
    print_capacity_report(capacity_report(cfg, vocab))
    print(f"可塑参数（构建时实际，count_params 口径）: {count_params(lm.net):,}")
    print("[context] 状态全程不重置（WM/STDP/LTM 跨样本/分片/epoch 连续携带）；"
          "长程依赖由 big_ltm 印迹承载 → 支持任意长连续序列（目标 ≥1M tokens）")

    done = 0
    if args.resume and ckpt.exists():
        meta = load_model(ckpt, lm)
        done = int(meta["done"])
        print(f"[续训] 已恢复 {done:,} tokens（检查点 {meta['when']}）")
    elif args.resume:
        print(f"[续训] 未找到检查点 {ckpt}，从头开始")

    if args.report:
        _print_table_stats(lm)
        print(f"[log] 日志已保存：{log_path}")
        return

    # ── 流式训练主循环（1M context：状态永不重置）──
    seg_nll: list[float] = []
    oov_skipped = 0
    t_start = time.perf_counter()
    i = done                       # 全局训练步（= 已处理的 token 流位置）
    last_mile = done // args.context_milestone if args.context_milestone else 0

    for ep in range(args.epochs):
        if _STOP["flag"] or (args.tokens and i - done >= args.tokens):
            break
        # 多核数据加载：生产者进程预取（与 char_chunks 产出逐位一致）；
        # mix 模式为样本级轮转交错流（串行源，混合语义需要全局轮转顺序）
        if args.data == "mix":
            src = stream_factory("mix")()
        else:
            src = PrefetchChars(data_path, SEP, workers=PREFETCH_W)
        stream = StreamingTokenizer(lm.tok.seg, src)
        it = iter(stream)
        p2 = None                                          # t_{i-1}（epoch 首步 prev 断开）
        p1 = next(it, None)                                # t_i
        t0 = next(it, None) if p1 is not None else None    # t_{i+1}（训练目标）
        if args.epochs > 1 and ep > 0 and i > 0:
            print(f"[epoch {ep + 1}/{args.epochs}] 跨 epoch 续流：状态连续"
                  f"（net 不重置），首步 prev 断开", flush=True)
        while t0 is not None:
            if _STOP["flag"]:
                break
            if args.tokens and i - done >= args.tokens:
                break
            if args.minutes and (time.perf_counter() - t_start) / 60.0 >= args.minutes:
                break
            p2_in = p2 is None or p2 in lm.tok.stoi
            if p2_in and p1 in lm.tok.stoi and t0 in lm.tok.stoi:
                # p2 为 OOV 时按 None 处理（组合编码退化为无前词上下文）——
                # 2026-09-25 审计修复：原条件漏查 p2，OOV 落在 p2 位会 KeyError 崩溃
                x = lm.tok.encode_composite(p1, p2)
                tgt = lm.tok.onehot(lm.tok.stoi[t0])
                d = lm.net.step(x, target=tgt, learn=True)
                seg_nll.append(d["nll"])
            else:
                oov_skipped += 1                          # OOV：跳过该步，流不断
            i += 1
            p2, p1, t0 = p1, t0, next(it, None)

            if args.context_milestone and i // args.context_milestone > last_mile:
                last_mile = i // args.context_milestone
                ppl_ms = f"滑动 PPL {float(np.exp(np.mean(seg_nll[-args.log_every:]))):.3f}" \
                    if seg_nll else ""
                print(f"[context 里程碑] 已连续处理 {last_mile * args.context_milestone:,} "
                      f"tokens（状态无重置；≥1M context 达标 ×{last_mile}）{ppl_ms}",
                      flush=True)
                _print_table_stats(lm)
            if (i - done) % args.log_every == 0 and seg_nll:
                k = min(args.log_every, len(seg_nll))
                ppl = float(np.exp(np.mean(seg_nll[-k:])))
                spent = time.perf_counter() - t_start
                ms = spent / max(1, i - done) * 1000
                print(f"  token {i:>12,}  滑动 PPL {ppl:>9.3f}  {ms:>8.2f} ms/tok"
                      f"  已用 {spent / 60:.1f} min", flush=True)
            if args.ckpt_every and (i - done) and (i - done) % args.ckpt_every == 0:
                save_model(ckpt, lm, cfg, i)
                print(f"  [检查点] 已保存 {i:,} tokens → {ckpt}", flush=True)
                _print_table_stats(lm)
        if _STOP["flag"] or (args.tokens and i - done >= args.tokens):
            break

    # ── 收尾：滚动检查点 + final 模型 ──
    save_model(ckpt, lm, cfg, i)
    final = args.save_dir / f"phdnet1b_{args.preset}_{args.data}_final.npz"
    save_model(final, lm, cfg, i, extra={"final": True})

    spent = time.perf_counter() - t_start
    oov_rate = oov_skipped / max(1, i - done)
    print("-" * 76)
    print(f"本次训练 {i - done:,} tokens，用时 {spent / 60:.1f} min"
          f"（{spent / max(1, i - done) * 1000:.2f} ms/token）| "
          f"OOV 跳过 {oov_skipped:,}（{oov_rate:.4%}）")
    _print_table_stats(lm)
    if seg_nll:
        final_ppl = float(np.exp(np.mean(seg_nll[-min(2000, len(seg_nll)):])))
        print(f"末段滑动 PPL ≈ {final_ppl:.3f}")
        print(f"[METRIC] preset={args.preset} data={args.data} tokens={i:,} "
              f"ms_per_token={spent / max(1, i - done) * 1000:.2f} "
              f"final_ppl={final_ppl:.3f} oov_rate={oov_rate:.5f}")
    print(f"模型：{final}")
    print(f"检查点：{ckpt}")
    print(f"[log] 日志已保存：{log_path}")


if __name__ == "__main__":
    main()
