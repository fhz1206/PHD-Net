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
python train/train.py --preset smoke --data sft --tokens 2000

# 标准 1B 档，全量流式训练 pretrain 分片（任意长度不截断）
python train/train.py --preset 1b --data pretrain_zh --tokens 1000000

# 生产长跑（零 OOV 词表 + 1M context 里程碑；断点续训）
python train/train.py --preset 1b --data pretrain_zh \
    --vocab-scan full --context-milestone 1000000 --resume
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# 审计 H4：**必须在 import numpy 之前**限 BLAS 线程——OpenBLAS 初始化后
# 再设环境变量无效。191 核机器上 OpenBLAS 默认拉 191 线程去跑
# 1024×2048 的 sgemv（M1 编码器），是「只用 1.1–3.2 核 + 250 万 CS/s」的
# 头号嫌疑（P62 只限了 numba prange 线程，漏了 BLAS）。
_DEFAULT_THREADS = str(max(1, min(8, os.cpu_count() or 1)))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _DEFAULT_THREADS)
# P97：线程**自旋等待**是开关而非默认——本机实测（x86 8 核、M2 plain 1 ms 级任务）
# `OMP_WAIT_POLICY=PASSIVE` 让进程 CPU 时间 −21%（11.52 s → 9.08 s，自旋确实少
# 了），但**墙钟 +90%**（1073 µs → 2044 µs）：小任务上休眠/唤醒成本远高于自旋。
# 因此**默认保持 ACTIVE（原行为）**，改为 `--omp-wait passive` 按需开启——
# 191 核 + 每步数毫秒的大规模环境里结论可能相反，应由实测决定而非想当然。

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_HERE), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault(  # P39：numba 缓存持久化（不被 __pycache__ 清理波及）
    "NUMBA_CACHE_DIR", str(_ROOT / "outputs" / "numba_cache"))

from ckpt_1b import (_rebuild_sdrs, load_model, save_model,            # noqa: E402
                     save_model_async, wait_pending_saves)
from config_1b import PRESETS, SEG_KWARGS, build_cfg                 # noqa: E402
from config_1b import capacity_report, print_capacity_report         # noqa: E402
from corpus_stream import PrefetchChars, SEP, StreamingTokenizer     # noqa: E402
from corpus_stream import build_vocab_text, char_chunks              # noqa: E402
from phdnet.corpus import expand_paths                              # noqa: E402
from phdnet.telemetry import Telemetry                              # noqa: E402
from phdnet.corpus import expand_paths                               # noqa: E402
from phdnet.model import count_params                                # noqa: E402
from phdnet.word_encoder import WordTokenizer                         # noqa: E402
from vocab_parallel import auto_workers, build_segmenter_parallel    # noqa: E402
from vocab_parallel import parallel_head_tokens, scan_vocab_parallel # noqa: E402
from corpus_stream import PREFETCH_MAX_PROCS                             # noqa: E402

# 预取进程数（0=自动；main 里按 --prefetch-workers 覆盖，P16）
PREFETCH_W = 0
# 预取队列深度 / 每批样本数（0=类缺省 8192 / 64；P24-P25）
PREFETCH_DEPTH = 0
PREFETCH_BATCH = 0


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

# ModelScope 数据集仓库（--remote-data 时经 HTTP Range 流式直读，零本地落盘）
MS_DATA_REPO = "fhzfhz/Mixture-General-Mini"

DATA_FILES = {
    # 训练数据一律用上传分片 parquet（sft/ pretrain/ 根）；raw/ 仅归档不触碰
    # （fhz 2026-09-25 指令，与 tools/train_production.py 同步）。
    "sft": _ROOT / "datasets" / "sft" / "sft_000.*.parquet",
    # P43（fhz「数据要中英文全部都拿来训练」）：`pretrain` = **全量**（中文 3708 万
    # 块 + 英文 Magpie-R1 201 万块，无过滤）；`pretrain_zh` = 仅中文（lang 过滤，
    # 保留作对照/消融）。两者指向同一目录，过滤逻辑在 PrefetchChars(lang=) 区分。
    "pretrain": _ROOT / "datasets" / "pretrain" / "pretrain_*.parquet",
    "pretrain_zh": _ROOT / "datasets" / "pretrain" / "pretrain_*.parquet",
    "eval": _ROOT / "eval_corpus" / "internal_corpus.txt",
}


_STOP = {"flag": False}


class TeeLogger:
    """控制台 + 日志文件双写（行缓冲落盘，异常中断最多丢当前行）。

    P57：同时接管 sys.stderr——warnings.warn（如读出融合核回落 eager 的
    告警）与 torch inductor 日志此前不落盘，服务器排障时丢失关键原因。
    """

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
    print("\n[interrupt] Interrupt received; will save checkpoint then exit…", flush=True)


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
    print(f"  [big-ltm] grown {st['grown_synapses']:,} / capacity {st['capacity']:,}"
          f" (utilization {st['utilization']:.3%}) | memory ≈{mb:.0f} MB"
          + ("  ⚠ dict backend: for long runs prefer --csr-online (≈10× memory saving)"
             if (not getattr(lm.net.ltm.table, "csr_online", False)
                 and st["grown_synapses"] > 50_000_000) else ""),
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="PHD-Net 1B 档流式训练（事件驱动稀疏类脑）")
    ap.add_argument("--preset", choices=list(PRESETS), default="1b",
                    help="smoke=管线验证 / 1b=标准档(容量≥1B) / 1b_max=大主干档")
    ap.add_argument("--data", choices=list(DATA_FILES), default="sft")
    ap.add_argument("--remote-data", action="store_true",
                    help="数据直读 ModelScope（%s，HTTP Range 流式零落盘，"
                         "仅支持 ModelScope）；默认读本地 datasets/（逐位不变）"
                         % MS_DATA_REPO)
    ap.add_argument("--remote-fraction", type=float, default=0.3,
                    help="--remote-data 时取排序后前多少比例的分片"
                         "（默认 0.3 = fhz 2026-09-29「数据集只取30%%」；"
                         "前缀子集，词表扫描与训练流开头一致）")
    ap.add_argument("--lang", choices=["all", "zh", "en"], default="all",
                    help="训练语料语言过滤（fhz 2026-09-29）：all=全量（默认，"
                         "与旧逐位一致）；zh/en=按 parquet 的 lang 列过滤"
                         "（--data pretrain --lang zh 等价旧 --data pretrain_zh；"
                         "sft 分片同样适用）。词表扫描不受影响（词表是训练流的"
                         "超集，OOV 恒 0）")
    ap.add_argument("--width", type=int, default=0, help="覆盖主干宽度（0=用预设）")
    ap.add_argument("--readout-dtype", default="fp32",
                    choices=["fp32", "fp16", "bf16", "fp8", "fp4", "int8", "int4",
                             "int16", "int32"],
                    help="读出计算精度。**默认 fp32**（P110, 2026-10-01：实测 "
                         "低精度会丢弃感知器 p − t 的非目标行更新 → 学习退化为"
                         "纯 Hebbian，见 tools/probe_readout_precision.py）。"
                         "fp16/bf16 保留率仅 26.7%%/5.8%%（|dp|~1e-6 档），"
                         "提速有限而语义有损；int8 更差。需低精度请显式指定")
    ap.add_argument("--torch-compile", dest="torch_compile",
                    action="store_true", default=False,
                    help="P58（fhz 2026-09-29「图优化关了吧」）：默认 OFF——"
                         "inductor 编译在服务器上不稳定（debug trace 干扰 + "
                         "编译耗时不可控）；eager 在 NPU 上为 4 个小 kernel "
                         "异步流提交，实际差距待 --step-profiling 量化。"
                         "P45 本机实测曾 +15%%（26.34 vs 30.98 ms/tok，"
                         "无 per-step target buffer 分配）")
    ap.add_argument("--no-torch-compile", dest="torch_compile",
                    action="store_false",
                    help="disable the P38 kernel fusion（现已是默认）")
    ap.add_argument("--torch-compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"],
                    help="P44: default='default' fuses kernels WITHOUT cudagraphs "
                         "(required, because the readout updates W in place and "
                         "cudagraphs reject mutated inputs); reduce-overhead / "
                         "max-autotune enable graph capture but will log 'skipping "
                         "cudagraphs due to mutated inputs' and lose that benefit")
    ap.add_argument("--step-profiling", action="store_true",
                    help="P35：step 分段计时（诊断 CPU 侧耗时分布；日志按段打印）")
    ap.add_argument("--omp-wait", default="active", choices=["active", "passive"],
                    help="P97: OpenMP 线程池空闲等待策略。active=自旋（默认，"
                         "小任务更快）；passive=让出 CPU（省 CPU 空转与 CS/s，"
                         "但本机实测墙钟 +90%%）。191 核大规模下请 A/B 后再定。")
    ap.add_argument("--omp-proc-bind", dest="omp_proc_bind",
                    action="store_true", default=True,
                    help="P71/P74：设 OMP_PROC_BIND=close 把 OpenMP 线程绑到"
                         "物理核；⚠ 不再设 OMP_PLACES=cores（191 核 place 表会让"
                         "线程池每次同步遍历，实测 M2 慢 8-13×）；--no-omp-proc-bind 关闭")
    ap.add_argument("--no-omp-proc-bind", dest="omp_proc_bind",
                    action="store_false",
                    help="关闭 OpenMP 核心绑定")
    ap.add_argument("--fp8-refresh", type=int, default=8,
                    help="P84：读出 fp8 forward 副本的重建间隔（步）。"
                         "量化 1.6 亿元素是一次设备算子，摊到 N 步；N 越大越省，"
                         "但 forward 用的 fp8 副本越旧")
    ap.add_argument("--m2-kernel", default="plain",
                    choices=["serial", "fused", "plain"],
                    help="P76/P99/P113：M2 推理核。**默认 plain**——"
                         "P113 服务器实测（1b 档，昇腾 191 核 + NPU）："
                         "M2_infer **6.92 → 1.28 ms/tok（5.41×）**，"
                         "端到端 **17.63 → 12.09 ms/tok（1.46×）**；"
                         "四个采样点 sliding PPL 与 serial 运行**逐位相同**"
                         "（纯调度变化、零语义变化）。"
                         "  · plain：走 _csr_matvec（**有**行级 prange）"
                         "  · serial：单核融合核（P99，昇腾上prange 屏障主导开销）"
                         "  · fused：prange 融合核（昇腾实测慢 3-4×，已否决）"
                         "⚠ **x86 结论不构成昇腾证据**：本机上fused 比 plain 快，"
                         "跨平台训练请按机器选择；对拍见 "
                         "tests/verifiers/verify_m2_kernels.py（11 例）")
    ap.add_argument("--encoder-dtype", default="fp32",
                    choices=["fp32", "fp64", "fp16", "bf16"],
                    help="P74/P75/P107：M1 编码器权重存储精度（迭代量恒 fp32）。"
                         "**默认 fp32**：x86 上 fp32 比 fp64 快 1.80×；昇腾 aarch64 "
                         "上两者都不慢——P77 的平台自适应 GEMV 核（_gemv_rows numba "
                         "核）绕开了BLAS，故与 dtype 无关。"
                         "注：库config.encoder_dtype 仍默认 fp64（形状参数默认），"
                         "生产 CLI 覆盖为 fp32，两者不一致是事实")
    ap.add_argument("--numba-threads", type=int, default=8,
                    help="P62：numba prange 线程上限（0=用 numba 默认=全部核）。"
                         "服务器实测 191 核上主循环只用 1.3 核、CS/s 250 万+"
                         "（百万级上下文切换 = 线程空转等锁）；P22 实测核内 "
                         "1→6 线程仅 1.16×（访存带宽饱和）→ 8 线程足够，"
                         "默认从「全部核」降到 8")
    ap.add_argument("--nll-sync-every", type=int, default=8,
                    help="读出 nll 同步周期（P34）：1=每步同步（旧行为）；N>1 时 "
                         "nll 累积到设备、每 N 步同步一次 → CPU/NPU 重叠，"
                         "NPU 场景端到端约 -30~40%%（PPL 统计滞后 N 步，滑动均值下可忽略）"
                         "｜P62 默认 8：服务器实测（2026-09-29 读出 12.9 ms/tok、"
                         "CPU 仅 1.3-3.2/191 核、CS/s 250 万+ = 线程空转等同步）"
                         "确认每步 .item() 是主要暴露点")
    ap.add_argument("--accel", default="auto",
                    help="读出计算设备（P19，fhz「有 cuda/cann(npu)/rocm 就跑"
                         "对应设备」）：auto = 有加速器就用（昇腾→ROCm→CUDA→"
                         "DirectML），否则回落 numba CPU 原路径（逐位不变）；"
                         "亦可显式 cpu/npu/cuda/rocm/dml")
    ap.add_argument("--big-n", type=int, default=0, help="覆盖大空间神经元数（0=用预设）")
    ap.add_argument("--csr-online", dest="csr_online",
                    action="store_true", default=True,
                    help="大空间表使用在线可写 CSR（**默认开**，P50）。"
                         "每条突触 ~16 B（int8 权重 + CSR 索引）vs dict 的 "
                         "~100+ B → 1B 档实测 5.74× 省（91.8→16.0 B/条目）；"
                         "内存只随**已生长**突触数增长，与容量无关 → 32 GB "
                         "机器上长跑/30B 档的安全前提")
    ap.add_argument("--no-csr-online", dest="csr_online",
                    action="store_false",
                    help="关闭在线 CSR，回退 dict 邻接表（仅短跑/调试用）")
    ap.add_argument("--readout-conn-k", type=int, default=128,
                    help="P108（fhz 2026-10-01）：M6 读出稀疏化——每输出单元的"
                         "入边数（0=稠密）。1B 档 128/3072 = 4.2%% 连接率，"
                         "访存 320→39 MB（省 8×）。默认 128 取自 4M 档 A/B："
                         "k=128 时 PPL 477 vs 稠密 608 且速度持平。")
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
    ap.add_argument("--assistant-marker", type=str, default="",
                    help="SFT 回复掩码标记（P26，如 '助手：' / 'Assistant:'）。"
                         "设置后只对该标记之后的**助手回复**计算损失；"
                         "其前的系统/用户 prompt 用 learn=False 推进状态、"
                         "不更新权重。留空 = 全 token 计损失（旧行为）")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="两阶段微调（P26）：从该检查点**初始化权重**但步数归零"
                         "（= 在预训练权重上做 SFT）；与 --resume（继续同一状态"
                         "并保留步数）不同")
    ap.add_argument("--prefetch-depth", type=int, default=0,
                    help="预取队列深度（批数，P25：缺省 8192）。注意深度受"
                         "**按文件序归并**约束——单生产者时囤积≈0，depth 不是"
                         "数据供给的主杠杆（见 --prefetch-workers / --prefetch-batch）")
    ap.add_argument("--prefetch-batch", type=int, default=0,
                    help="每批样本数（P25，batch_samples；0=用类缺省 64）。"
                         "增大它可提高单次 IPC 传输量、减少唤醒次数——"
                         "这是提升数据提供量的有效杠杆之一")
    ap.add_argument("--prefetch-workers", type=int, default=0,
                    help="语料预取进程数（P16：0=自动，默认 min(8, 核数×0.8, 文件数)；"
                         "解码在 pyarrow 内多线程，进程过多只增内存/调度）")
    ap.add_argument("--vocab-workers", type=int, default=0,
                    help="词表构建/全量扫描并行进程数（0=自动=核心数×0.8，1=串行；"
                         "fhz 2026-09-25 指令）")
    ap.add_argument("--context-milestone", type=int, default=1_000_000,
                    help="context 里程碑间隔 tokens（0=关闭；默认 1M）")
    ap.add_argument("--ckpt-dtype", default="fp16",
                    choices=["", "fp8", "bf16", "fp16", "int8", "int16", "int32"],
                    help="P46: checkpoint storage precision for the big matrices "
                         "(readout W etc). fp8/bf16 are stored as raw bit patterns "
                         "and decoded losslessly on load. NOT the training "
                         "precision: torch cannot do fp8 matmul, and fp8 would "
                         "flush the perceptron's non-target-row updates to zero.")
    ap.add_argument("--ckpt-every", type=int, default=50000,
                    help="每 N token 存一次检查点（P36：默认 50,000；"
                         "每次保存含 867 MiB 读出权重 D2H + 写盘，约数秒）")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true",
                    help="从 models/ 检查点续训（快进至断点，状态由检查点恢复）")
    ap.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--report", action="store_true",
                    help="只打印检查点的大空间表统计后退出")
    args = ap.parse_args()
    # P94：终端语言随 `--lang` 切换（zh=全中文默认 / en=全英文），必须
    # **早于任何 print**（zh 时是空操作；en 时输出层翻译历史中文文案）。
    from phdnet.i18n import set_lang, install_stream_filter
    set_lang(args.lang)
    install_stream_filter()
    vw = args.vocab_workers if args.vocab_workers > 0 else auto_workers()
    # P16：预取**进程**数收敛（解码在 pyarrow 内多线程 + 主进程分词为 nogil
    # 线程）；进程数 × 每进程解码核 ≈ 核数，避免 153 进程的超订与内存爆炸。
    global PREFETCH_W, PREFETCH_DEPTH, PREFETCH_BATCH
    PREFETCH_W = (args.prefetch_workers if args.prefetch_workers > 0 else 0)
    PREFETCH_DEPTH = (args.prefetch_depth if args.prefetch_depth > 0 else 0)
    PREFETCH_BATCH = (args.prefetch_batch if args.prefetch_batch > 0 else 0)
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
    sys.stderr = sys.stdout          # P57：warnings / inductor 日志也落盘

    # 后端 × 设备能力矩阵：显式说明「numba 只能上 CPU」这一物理限制，
    # 避免在 NPU/CUDA 机器上被误判为「加速器没被识别」（P18）。
    # P20：同时把**读出设备的实际解析结果**打出来（探测到 ≠ 能用），
    # 失败时给一行可复制的诊断命令。
    try:
        from phdnet.backends.multi_device import capability_report
        _cap = capability_report(verbose=False)
        if _cap["accelerators_present"]:
            print(f"[capability] accelerator(s) detected {_cap['accelerators_present']}, "
                  f"but production training runs on numba/CPU for M1-M5 (numba only "
                  f"compiles to CPU machine code); the readout (M6, ~89% of step "
                  f"time) does run on the accelerator - see the [readout] backend "
                  f"line below. No full-torch training stack exists (P30 removed "
                  f"tools/train_torch_lm.py: it lacked 7 mechanisms incl. big_ltm)")
            try:
                from phdnet.backends.accel_readout import resolve_accel_device
                _rd = resolve_accel_device("auto")
                print(f"[capability] readout device resolved (auto) = {_rd} — "
                      f"readout will run on this device (rest remains numba CPU)")
            except Exception as _e:                          # noqa: BLE001
                print(f"[capability] readout device resolution failed: {type(_e).__name__}: "
                      f"{str(_e)[:120]}")
                print("[capability] diagnostics: python tools/accel_doctor.py "
                      "(env matrix / trial allocation / real-load timing)")
    except Exception:                                       # noqa: BLE001
        pass
    _inflight_samples = min(PREFETCH_DEPTH or 8192, 8192) * (PREFETCH_BATCH or 64)
    print(f"[parallel] tokenisation thread (numba nogil)={vw} | prefetch processes={_pf}"
          f" | prefetch depth={PREFETCH_DEPTH or 8192}, batch={PREFETCH_BATCH or 64} samples"
          f" (queue capacity≈{_inflight_samples:,} samples ≈ millions of tokens;"
          f" in-flight ≈ producers × lead batches — multiple producers needed to fill the queue)"
          f" (decode cores/process≈{max(1, _cpu // _pf)}, total≈{_pf * max(1, _cpu // _pf)})"
          f" | data-side parallelism does not contend with the tokenisation thread")
    print(f"[log] log file: {log_path}")
    print(f"[log] start: {datetime.now().isoformat(timespec='seconds')}  "
          f"argv: {' '.join(sys.argv[1:]) or '(defaults)'}")

    signal.signal(signal.SIGINT, _on_sigint)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.save_dir / f"phdnet1b_{args.preset}_{args.data}.npz"

    # P62：numba prange 线程上限（必须在任何核首次执行**之前**设置）。
    # 服务器 191 核实测：主循环只用 1.3–3.2 核、CS/s 250 万–600 万——大量
    # 上下文切换来自「用 191 线程跑千行级 prange」的线程空转。P22 实测核内
    # 1→6 线程仅 1.16×（访存带宽饱和），8 线程足够。
    # P71（fhz「核心绑定用了吗」）：`OMP_PROC_BIND` 让 OpenMP 把线程绑到物理核
    # 而非到处迁移。必须在 numba 初始化（首次调用核）之前设进环境。
    # ⚠ P74（2026-09-30 回滚一半）：`OMP_PLACES=cores` 在 191 核机器上要为
    # 191 个核建 place 表，OpenMP/numba 每次线程池同步都要遍历它 → 撤掉。
    # 服务器实测（12:50 日志）：带 OMP_PLACES 时 M2_infer 2.4 → 19-32 ms/tok
    # （涨 8-13×）、CS/s 反而升到 600 万。只保留 PROC_BIND。
    if args.omp_wait == "passive":
        # P97：必须**在 numba 初始化之前**设置才生效
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        os.environ.setdefault("KMP_BLOCKTIME", "0")
    if args.omp_proc_bind:
        os.environ.setdefault("OMP_PROC_BIND", "close")
    # P88：加速器**启动期健康检查**。服务器 2026-09-30 18:59 报
    # `error code 507033 / Failed to start the device / open device 0 failed`
    # → 回落 numba CPU（机制正常，但**原因只出现在 CANN 日志里**，训练日志
    # 看不出是「设备被上次残留进程占用」还是「驱动异常」）。这里在建 net 之前
    # 真正建一个设备张量，把原因翻译成人能懂的一句话。
    def _npu_health():
        spec = str(getattr(args, "accel", "auto") or "auto").lower()
        if spec in ("cpu", "off", "numba"):
            return
        try:
            import torch
            if not hasattr(torch, "npu"):
                return
            _ = torch.zeros(4, device="npu")
            _ = torch.zeros(4, device="npu") + 1
        except Exception as e:                                # noqa: BLE001
            msg = str(e).replace("\n", " ")[:160]
            print("[device] ⚠️ NPU 初始化失败 → 读出将回落到 numba CPU。"
                  f"\n         原因：{msg}"
                  "\n         常见：①上一次训练的 python 进程还在（查 `npu-smi info`"
                  " 的进程列表 / `ps aux | grep train_1b` 并 kill）；"
                  "\n              ②容器未映射设备或驱动状态异常（重进容器 / `npu-smi info`"
                  " 看设备是否 Healthy）；"
                  "\n              ③设备被占满（507033 = device retain 失败）。"
                  "\n         想强制 CPU：加 --accel cpu", flush=True)
    _npu_health()

    # P79：numba 缓存状态可见化（cache=True 的核是否真的命中持久缓存）
    try:
        import numba
        import numba.core.caching as _nbc
        _cd = os.environ.get("NUMBA_CACHE_DIR", "")
        _sz = 0
        if _cd and os.path.isdir(_cd):
            for _rt, _, _fs in os.walk(_cd):
                for _f in _fs:
                    try:
                        _sz += os.path.getsize(os.path.join(_rt, _f))
                    except OSError:
                        pass
        print(f"[numba] cache dir = {_cd or '(module-local __pycache__)'}"
              f" | size = {_sz / 1e6:.1f} MB"
              f" | 首次运行会全量编译，此后命中缓存（预期启动 ~2.6 s）", flush=True)
    except Exception:                                   # noqa: BLE001
        pass
    if args.numba_threads > 0:
        try:
            import numba
            numba.set_num_threads(min(args.numba_threads, _cpu))
            print(f"[parallel] numba prange threads = {numba.get_num_threads()}"
                  f" (cap {args.numba_threads}; P22: 1→6 threads only 1.16x)"
                  f" | OMP_PROC_BIND={os.environ.get('OMP_PROC_BIND', '-')}"
                  f" BLAS/OpenMP threads={_DEFAULT_THREADS}",
                  flush=True)
        except Exception as e:                          # noqa: BLE001
            print(f"[parallel] numba thread cap not applied: {e}")

    cfg = build_cfg(args.preset, args.width, args.big_n,
                    args.csr_online, args.readout_conn_k, args.seed)
    cfg.readout_dtype = args.readout_dtype            # P9 精度（默认 fp32）
    cfg.encoder_dtype = args.encoder_dtype            # P75：M1 权重精度（平台相关）
    # P99：三态（serial=单核融合 / fused=并行融合 / plain=原始多核调用）
    cfg.pc_fused_kernel = (False if args.m2_kernel == "plain"
                           else args.m2_kernel)        # P75/P99：M2 核选择
    cfg.fp8_refresh = max(1, int(args.fp8_refresh))   # P84：fp8 副本刷新间隔
    cfg.accel_readout = args.accel                     # P19 读出设备（默认 auto）
    cfg.nll_sync_every = args.nll_sync_every           # P34 nll 同步周期（默认 1）
    cfg.step_profiling = args.step_profiling           # P35 step 分段计时（默认关）
    cfg.torch_compile = args.torch_compile             # P38 kernel 融合（默认开）
    cfg.torch_compile_mode = args.torch_compile_mode   # P44 模式（default=无 cudagraph）
    _tel = Telemetry()                                 # P41：系统/设备遥测

    data_path = DATA_FILES[args.data]
    remote_active = bool(args.remote_data and args.data != "eval")
    if remote_active:
        # fhz 2026-09-29（服务器「glob 无匹配」）：数据集托管 ModelScope，
        # --remote-data 切换为 HTTP Range 流式直读（零落盘，仅支持 ModelScope）；
        # 分片级抽样取前 remote_fraction 比例（前缀子集，顺序语义不变）。
        # 与本地同名分片模式 → 文件顺序与本地一致。
        _pat = DATA_FILES[args.data].name
        data_path = f"ms://{MS_DATA_REPO}/{args.data if args.data != 'pretrain_zh' else 'pretrain'}/{_pat}"
        os.environ["PHDNET_REMOTE_FRACTION"] = str(args.remote_fraction)
        if args.vocab_scan == "full":
            print("[data] WARN: --vocab-scan full 会把远程分片整读两遍"
                  "（建词表 + 训练）；建议 --vocab-file 复用词表快照")
        print(f"[data] remote source: {data_path} "
              f"(fraction={args.remote_fraction:g})")
    try:
        data_files = expand_paths(data_path)
    except FileNotFoundError as e:
        print(f"data file not found: {e}")
        sys.exit(1)
    if not remote_active:
        total_mb = sum(p.stat().st_size for p in data_files) / 1e6
    else:                                   # 远程：MsFile 自带 API Size
        from phdnet.ms_stream import ms_total_size
        total_mb = ms_total_size(data_files) / 1e6
        print(f"[data] {len(data_files)} remote shards, ~{total_mb:.0f} MB "
              f"(ModelScope HTTP Range streaming, zero local copy)")
    # 数据口径随检查点落盘（审计 D2：--remote-fraction 变化 + --resume 会静默
    # 改变数据分布 → 恢复训练时须能看出这次续训用的是哪份数据）
    # 语言过滤（fhz 2026-09-29）：--lang zh/en 按 parquet 的 lang 列过滤训练流；
    # 旧 --data pretrain_zh 等价 --data pretrain --lang zh（保留兼容）。
    # 词表扫描不受影响（词表是训练流超集 → OOV 恒 0）。
    _lang_filter = None
    _lang_tail: list[int] = []          # P69：语种自检（最近 8 个真实 token）
    if args.data == "pretrain_zh":
        _lang_filter = "zh"
    elif args.lang != "all":
        _lang_filter = args.lang
        print(f"[data] lang filter: {_lang_filter} (parquet `lang` column)")
    _data_provenance = {
        "data": args.data,
        "remote": bool(remote_active),
        "remote_fraction": (float(args.remote_fraction) if remote_active else None),
        "lang": (_lang_filter or "all"),
        "data_shards": len(data_files),
        "data_spec": str(data_path),
    }
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
    else:
        vocab_text = build_vocab_text(data_path, args.vocab_sample_chars, SEP)
    if not (_resume_vocab or _file_vocab):
        t_v0 = time.perf_counter()
        seg = build_segmenter_parallel(vocab_text, SEG_KWARGS, args.vocab_workers)
    if not (_resume_vocab or _file_vocab) and vw > 1:
        print(f"[vocab] word-induction multi-core build: L=2..{SEG_KWARGS['max_len']} × "
              f"5-thread pool (serial within core under nogil, parallel across levels; P22)"
              f" (induced vocab {len(seg.vocab):,}, {time.perf_counter() - t_v0:.1f}s)",
              flush=True)
    # 词表来源三选一（优先级）：resume 自包含 > 外部文件 > head/full 扫描
    if _resume_vocab:
        from ckpt_1b import peek_tokenizer
        _cv, _ct, _cmax = peek_tokenizer(_ckpt_path)
        seg = make_external_segmenter({**SEG_KWARGS, "max_len": _cmax}, _cv)
        tokens = list(_ct)
        vocab_text = "\n".join(tokens[:4096])   # 仅供分词器注入构造
        print(f"[vocab] --resume: skipping vocab build, using checkpoint's self-contained vocab"
              f" ({_ckpt_path.name}: {len(_cv):,} words / max_len={_cmax}, "
              f"deterministic SDR hash rebuild, bit-identical; readout layer aligned to this size)", flush=True)
    elif _file_vocab:
        from ckpt_1b import load_vocab_file
        _words, _segv, _wmax = load_vocab_file(Path(args.vocab_file))
        if _segv is None:
            print("[vocab] ⚠ snapshot missing seg_vocab (tokenizer candidate set): "
                  "fell back to the token vocab as candidate set — tokenisation results "
                  "may differ from training (silent degradation risk). "
                  "Prefer --resume (checkpoint self-contained, authoritative) or a newer snapshot.",
                  flush=True)
            _segv = _words
        seg = make_external_segmenter({**SEG_KWARGS, "max_len": _wmax}, _segv)
        tokens = sorted(set(_words))
        vocab_text = "\n".join(_words[:4096])   # 仅供分词器注入构造
        print(f"[vocab] external vocab file {args.vocab_file}"
              f" ({'JSON authoritative format' if Path(args.vocab_file).suffix.lower() == '.json' else 'text (escape restored)'})"
              f": {len(_words):,} words / max_len={_wmax} → dedup {len(tokens):,} words"
              f" (scan stage skipped; OOV steps skipped and counted)", flush=True)
    elif args.vocab_scan == "full":
        from vocab_parallel import NUMBA_TOK_OK as _TOK_NB
        _engine = (f"multi-core anchor chain x{vw}"
                   + ("（numba nogil 线程）" if vw > 1 and _TOK_NB
                      else "（进程池）" if vw > 1 else ""))
        print(f"[vocab] full scan: streaming tokenisation over the full corpus ({_engine}; slow on large corpora)…",
              flush=True)
        seen: set[str] = set()
        n_seen = 0
        t_v = time.perf_counter()
        if vw > 1:
            def _scan_prog(n_total, n_distinct):
                print(f"  [vocab-scan] total {n_total:>12,} tokens, "
                      f"current vocab {n_distinct:,}"
                      f" ({time.perf_counter() - t_v:.0f}s)", flush=True)

            # P33（fhz「核心绑定到进程」）：扫描是一次性构建阶段，追求吞吐 →
            # 解码默认**多生产者**（每核心一个进程，跨文件并行；显式 workers
            # 不受 PREFETCH_MAX_PROCS=1 钳制）。训练稳态预取仍默认 1 进程不变。
            # 内存：队列有界（depth 上限）→ 背压成立；每进程仅局部缓冲。
            _scan_files = len(expand_paths(data_path)) if '*' in str(data_path) \
                else 1
            _scan_w = PREFETCH_W if PREFETCH_W > 0 else \
                max(1, min(16, _scan_files, (os.cpu_count() or 1)))
            print(f"[vocab] scan decode processes: {_scan_w} (core-pinned: one dedicated core per process;"
                  f" training steady-state prefetch still {PREFETCH_W or 1})", flush=True)
            seen, n_seen = scan_vocab_parallel(seg.vocab, seg.max_len,
                                               PrefetchChars(
                                                   data_path, SEP,
                                                   lang=("zh"
                                                        if args.data == "pretrain_zh"
                                                        else None),
                                                   depth=PREFETCH_DEPTH,
                                                   batch_samples=PREFETCH_BATCH or 64,
                                                   workers=_scan_w),
                                               vw, progress=_scan_prog)
        else:
            for tk in StreamingTokenizer(seg, char_chunks(data_path, SEP)):
                seen.add(tk)
                n_seen += 1
                if n_seen % 1_000_000 == 0:
                    print(f"  [vocab-scan] streamed {n_seen:>12,} tokens, "
                          f"current vocab {len(seen):,} ({time.perf_counter() - t_v:.0f}s)",
                          flush=True)
        tokens = sorted(seen)
        print(f"[vocab] full scan done: corpus {n_seen:,} tokens → vocab {len(tokens):,}"
              f" ({time.perf_counter() - t_v:.0f}s, {_engine}), "
              f"zero OOV", flush=True)
    else:
        if vw > 1:
            t_h = time.perf_counter()
            tokens = sorted(parallel_head_tokens(seg, vocab_text, vw))
            print(f"[vocab] head mode (multi-core token collection ×{vw}, "
                  f"{time.perf_counter() - t_h:.1f}s): built from first {len(vocab_text):,} sampled chars"
                  f" (induced vocab {len(seg.vocab):,}, token vocab {len(tokens):,});"
                  f" OOV steps in the training stream will be skipped and counted", flush=True)
        else:
            tokens = sorted(set(seg.tokenize(vocab_text)))
            print(f"[vocab] head mode: built from first {len(vocab_text):,} sampled chars"
                  f" (induced vocab {len(seg.vocab):,}, token vocab {len(tokens):,});"
                  f" OOV steps in the training stream will be skipped and counted", flush=True)

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
        print(f"[vocab] snapshot saved: {_vp.name} ({_vp.stat().st_size / 1024:.0f} KB, "
              f"source={_src}; the single authoritative format = vocab + tokenizer candidates + max_len + sha1)"
              f" — after a crash resume directly with --vocab-file {_vp.name}, no rescan needed", flush=True)

    # ── 构建 LM（注入式 tokenizer；n_readout = 词表大小）──
    t0 = time.perf_counter()
    tok = _make_tokenizer(seg, tokens, cfg)
    from phdnet.word_lm import PHDWordLM
    lm = PHDWordLM(vocab_text, cfg, seg_kwargs=SEG_KWARGS, tokenizer=tok)
    vocab = len(lm.tok)

    print("=" * 76)
    print(f"PHD-Net 1B streaming training | preset={args.preset} data={args.data} "
          f"width={cfg.n_sdr} conn_k={cfg.conn_k}")
    print(f"big LTM N={cfg.big_ltm_N:,} × m={cfg.big_ltm_m}"
          f" (capacity {cfg.big_ltm_N * cfg.big_ltm_m:,}) | vocab {vocab:,}"
          f" | shards: {len(data_files)} ({total_mb:.0f} MB, streaming, never truncated)")
    print(f"epochs={args.epochs} | token budget={args.tokens or '∞'} "
          f"minutes={args.minutes or '∞'} | milestone={args.context_milestone or '∞'}"
          f" | vocab workers={vw} | data prefetch=multiproc×{dl_w}{'+zh filter' if args.data == 'pretrain_zh' else ''}"
          f" | build time {time.perf_counter() - t0:.1f}s")
    _rb = getattr(lm.net, "_readout_backend", "numba-cpu")
    if cfg.readout_dtype in ("bf16", "fp16", "fp8", "fp4", "int8", "int4"):
        print(f"[readout] compute precision = {cfg.readout_dtype}"
              f" (checkpoint storage = {args.ckpt_dtype or 'fp32'}); "
              f"WARNING: 半精度会舍入丢弃感知器 p − t 的非目标行更新 "
              f"(实测保留率 fp16 26.7% / bf16 5.8% @ |dp|~1e-6)，"
              f"学习规则退化为纯 Hebbian。生产请用 --readout-dtype fp32。",
              flush=True)
    else:
        print(f"[readout] compute precision = {cfg.readout_dtype}"
              f" (checkpoint storage = {args.ckpt_dtype or 'fp32'})"
              f" — 精确 p − t 规则", flush=True)
    print(f"[readout] backend={_rb}"
          + (f" (device {lm.net.readout.device})"
             if _rb.startswith("accel:") else "")
          + (f" | fallback reason: {getattr(lm.net.readout, '_accel_fallback_reason', '')}"
             if hasattr(lm.net.readout, "_accel_fallback_reason") else ""))
    print_capacity_report(capacity_report(cfg, vocab))
    print(f"plastic params (as built, count_params basis): {count_params(lm.net):,}")
    print("[context] state never reset (WM/STDP/LTM carried continuously across samples/shards/epochs);"
          " long-range dependency carried by big_ltm imprints → supports arbitrarily long continuous sequences (target ≥1M tokens)")

    if args.init_from is not None and Path(args.init_from).exists():
        meta = load_model(Path(args.init_from), lm)
        print(f"[stage-2] weights initialised from {args.init_from}"
              f" ({meta.get('when', '?')}), steps reset to zero → entering "
              f"{'SFT fine-tuning' if args.assistant_marker else 'continued training'}"
              f" (unlike --resume: step count not restored)")
    elif args.init_from is not None:
        print(f"[stage-2] {args.init_from} not found, treating as from scratch (reported as-is)")

    done = 0
    if args.resume and ckpt.exists():
        meta = load_model(ckpt, lm)
        done = int(meta["done"])
        print(f"[resume] restored {done:,} tokens (checkpoint {meta['when']})")
    elif args.resume:
        print(f"[resume] checkpoint {ckpt} not found, starting from scratch")

    if args.report:
        _print_table_stats(lm)
        print(f"[log] log saved: {log_path}")
        return

    # ── 流式训练主循环（1M context：状态永不重置）──
    seg_nll: list[float] = []
    oov_skipped = 0
    prompt_masked = 0                        # P26：prompt 段（未计损失）步数
    t_start = time.perf_counter()
    i = done                       # 全局训练步（= 已处理的 token 流位置）
    last_mile = done // args.context_milestone if args.context_milestone else 0

    # ── P64：GC 调优（长跑训练的标准做法）──────────────────────────────
    # 现象（fhz 服务器 2026-09-29）：ms/tok 从 20 单调升到 100（token 1500 →
    # 10000），而读出耗时反而下降 → 非读出段随步数**超线性**恶化。
    # 最可能机制：Python GC —— 大空间表在线 CSR 每次生长都新建 dict/数组，
    # 对象数线性增长 → gen2 扫描频率与耗时随之上升。
    # 措施：①freeze 掉导入期的常量对象（模型/配置/词表句柄，之后不再参与扫描）；
    # ②大幅提高 gen0/gen1 阈值（长跑循环几乎不产生真循环引用，回收收益低、
    #    扫描成本高）。遥测新增 `GC <对象数>M/gen2 <次数>` 可验证效果。
    # P106：gc.freeze 恢复（fhz 明确要求）。freeze 把当前所有对象移到永久代、
    # 不再参与 GC 扫描——长跑训练（对象数只增不减、无循环引用）的理想配置。
    # ⚠ `tracked objects = 0` 是 **freeze 的正常行为**（get_objects 不返回
    # 永久代对象），不是 bug——P105 曾据此误回滚。统计移到 freeze 之前。
    import gc as _gc
    _gc.collect()
    _n = len(_gc.get_objects())
    _gc.freeze()
    _gc.set_threshold(50_000, 200, 200)
    print(f"[gc] freeze() + threshold(50000, 200, 200); "
          f"tracked objects = {_n:,}（freeze 后 get_objects 返回 0 属正常）",
          flush=True)

    for ep in range(args.epochs):
        if _STOP["flag"] or (args.tokens and i - done >= args.tokens):
            break
        # 多核数据加载：生产者进程预取（与 char_chunks 产出逐位一致）
        # 远程审计 B2：每进程持有独立 aiohttp 连接池 + 8 MB block 缓存，
        # 8 进程并发 Range 请求可能触发服务端限流 → 远程上限 REMOTE_MAX_PROCS
        _pw = PREFETCH_W
        if remote_active:
            from phdnet.ms_stream import REMOTE_MAX_PROCS
            _pw = min(PREFETCH_W or REMOTE_MAX_PROCS, REMOTE_MAX_PROCS)
        src = PrefetchChars(data_path, SEP,
                            lang=_lang_filter,
                            depth=PREFETCH_DEPTH,
                            batch_samples=PREFETCH_BATCH or 64,
                            workers=_pw)
        stream = StreamingTokenizer(lm.tok.seg, src,
                                   assistant_marker=(args.assistant_marker
                                                     or None))
        it = iter(stream)
        # P26：掩码模式下 next() 返回 (token, trainable)，否则是纯 token
        def _next_tok():
            item = next(it, None)
            if item is None:
                return None, True
            if isinstance(item, tuple):
                return item
            return item, True

        p2, _ = None, True                                 # t_{i-1}（epoch 首步 prev 断开）
        p1, trainable = _next_tok()                        # t_i
        t0, _ = _next_tok() if p1 is not None else (None, True)   # t_{i+1}（训练目标）
        if args.epochs > 1 and ep > 0 and i > 0:
            print(f"[epoch {ep + 1}/{args.epochs}] continuing stream across epochs: state continuous"
                  f" (net not reset), first step's prev link severed", flush=True)
        _stoi = lm.tok.stoi                   # 局部绑定（省每步属性链解析）
        # P72：主循环分段计时（此前只有 net.step 内部九段，**主循环开销
        # 从未被测量**——实测分段之和 < 总耗时，差额就落在这里）。
        _lp = {"tokenize": 0.0, "encode_onehot": 0.0, "step": 0.0}
        _lpn = 0
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
                _eo0 = time.perf_counter() if args.step_profiling else 0.0
                x = lm.tok.encode_composite(p1, p2)
                _i0 = _stoi[t0]                    # 复用：省两次 dict 查找
                tgt = lm.tok.onehot(_i0)
                if args.step_profiling:
                    _lp["encode_onehot"] += time.perf_counter() - _eo0
                    _lpn += 1
                # P69：语种自检用——保留最近 8 个真实 token（每 log_every 反查词表
                # 拼文本片段 + CJK 占比，一眼看出 --lang 是否生效）
                _lang_tail.append(int(_i0))
                if len(_lang_tail) > 8:
                    _lang_tail.pop(0)
                # P26 SFT：p1（当前 token）属 prompt 段 → 只推进状态不学习。
                # 损失由 (p1, p2) → p0 这一步产生，故用 **p0** 的可训练标记。
                _st0 = time.perf_counter() if args.step_profiling else 0.0
                d = lm.net.step(x, target=tgt, learn=trainable,
                                target_idx=_i0)
                if args.step_profiling:
                    _lp["step"] += time.perf_counter() - _st0
                if trainable:
                    seg_nll.append(d["nll"])
                else:
                    prompt_masked += 1
            else:
                oov_skipped += 1                          # OOV：跳过该步，流不断
            i += 1
            _tk0 = time.perf_counter() if args.step_profiling else 0.0
            _nxt, _flag = _next_tok()
            if args.step_profiling:
                _lp["tokenize"] += time.perf_counter() - _tk0
            p2, p1, t0, trainable = p1, t0, _nxt, _flag

            if args.context_milestone and i // args.context_milestone > last_mile:
                last_mile = i // args.context_milestone
                ppl_ms = f"sliding PPL {float(np.exp(np.mean(seg_nll[-args.log_every:]))):.3f}" \
                    if seg_nll else ""
                print(f"[context milestone] {last_mile * args.context_milestone:,} "
                      f"tokens processed continuously (state never reset; ≥1M context achieved ×{last_mile}){ppl_ms}",
                      flush=True)
                _print_table_stats(lm)
            if (i - done) % args.log_every == 0 and seg_nll:
                k = min(args.log_every, len(seg_nll))
                ppl = float(np.exp(np.mean(seg_nll[-k:])))
                spent = time.perf_counter() - t_start
                ms = spent / max(1, i - done) * 1000
                # P20：读出分项（后端 + 设备 + 占比）——直接看出加速是否生效、
                # 以及耗时是否已转移到 PC 栈等其余部件
                _ro_ms = float(getattr(lm.net, "_ro_ms_accum", 0.0))
                _ro_n = max(1, int(getattr(lm.net, "_ro_calls", 0)))
                _ro_dev = (getattr(lm.net.readout, "device", "cpu")
                           if _rb.startswith("accel:") else "cpu")
                print(f"  token {i:>12,}  sliding PPL {ppl:>9.3f}  {ms:>8.2f} ms/tok"
                      f"  elapsed {spent / 60:.1f} min"
                      f"  | readout {_ro_ms / _ro_n:>7.3f} ms/tok"
                      f" ({_rb.split('(')[0]}@{_ro_dev}, {_ro_ms / 1000 / max(1e-9, spent) * 100:>5.1f}% of total)",
                      flush=True)
                if args.step_profiling and _lpn:
                    _dt = max(1, _lpn)
                    print("      loop: " + "  ".join(
                        f"{k} {v * 1000 / _dt:.2f}" for k, v in _lp.items())
                          + " ms/tok", flush=True)
                if getattr(lm.net, "_prof_on", False) and lm.net._prof:
                    # P115：`_prof` 是**从 token 0 起全程累加**（代码事实：
                    # `_prof[name] = _prof.get(name,0)+dt`，从不清零），
                    # 所以这一行是「历史平均」，会被启动期（numba 首次编译、
                    # 设备 kernel 首次编译）系统性高估。并行打印**滑窗**口径
                    #（上一个打印间隔内的真实平均），两个都给出。
                    _pr = sorted(lm.net._prof.items(), key=lambda kv: -kv[1])[:6]
                    print("segments(cum): " + "  ".join(
                        f"{k} {v * 1000 / max(1, i - done):.2f}" for k, v in _pr)
                          + " ms/tok", flush=True)
                    _wn = int(getattr(lm.net, "_prof_win_n", 0) or 0)
                    if _wn > 0:
                        _pw = sorted(lm.net._prof_win.items(),
                                     key=lambda kv: -kv[1])[:6]
                        print("      segments(win): " + "  ".join(
                            f"{k} {v * 1000 / _wn:.2f}" for k, v in _pw)
                              + f" ms/tok  [{_wn} steps]", flush=True)
                    # 打印后清空滑窗，下一个间隔重新统计
                    lm.net.reset_prof_window()
                # P41：系统/设备遥测（CPU/RAM/NPU/HBM；IPC 需外部 perf）
                try:
                    print("      " + _tel.fmt(_tel.sample()), flush=True)
                except Exception:                       # noqa: BLE001
                    pass
                # P69：语种自检——用**最近若干步的真实 target token** 反查词表拼出
                # 文本片段 + CJK 占比。`--lang zh/en` 是否真生效，此前只能靠肉眼看
                # PPL 猜（fhz 2026-09-29：「--lang zh 后日志还是英文」——实测那次
                # argv 里根本没带 --lang）。
                try:
                    if _lang_tail:
                        txt = "".join(str(tokens[t]) for t in _lang_tail
                                      if 0 <= int(t) < len(tokens))
                        cjk = (sum(1 for ch in txt if "\u4e00" <= ch <= "\u9fff")
                               / max(1, len(txt)))
                        print(f"      [sample lang={_lang_filter or 'all'}] "
                              f"CJK={cjk:.0%} | {txt[:40]!r}", flush=True)
                except Exception:                       # noqa: BLE001
                    pass
            if args.ckpt_every and (i - done) and (i - done) % args.ckpt_every == 0:
                save_model_async(ckpt, lm, cfg, i, ckpt_dtype=args.ckpt_dtype,
                           extra=_data_provenance)
                print(f"  [checkpoint] saved {i:,} tokens → {ckpt}", flush=True)
                _print_table_stats(lm)
        if _STOP["flag"] or (args.tokens and i - done >= args.tokens):
            break

    # ── 收尾：滚动检查点 + final 模型 ──
    wait_pending_saves()          # P83：先等后台队列写完（滚动 ckpt 的数据一致性）
    save_model(ckpt, lm, cfg, i, ckpt_dtype=args.ckpt_dtype, extra=_data_provenance)
    final = args.save_dir / f"phdnet1b_{args.preset}_{args.data}_final.npz"
    save_model(final, lm, cfg, i, extra={**_data_provenance, "final": True})
    wait_pending_saves()          # P83：确保 final 落盘后再打印完成

    spent = time.perf_counter() - t_start
    oov_rate = oov_skipped / max(1, i - done)
    print("-" * 76)
    if args.assistant_marker:
        print(f"[SFT] reply masking active (marker {args.assistant_marker!r}): "
              f"{prompt_masked:,} prompt steps advance state only without weight updates;"
              f" loss counted on assistant replies only")
    print(f"this run {i - done:,} tokens in {spent / 60:.1f} min"
          f" ({spent / max(1, i - done) * 1000:.2f} ms/token) | "
          f"OOV skipped {oov_skipped:,} ({oov_rate:.4%})")
    _print_table_stats(lm)
    if seg_nll:
        final_ppl = float(np.exp(np.mean(seg_nll[-min(2000, len(seg_nll)):])))
        print(f"tail sliding PPL ≈ {final_ppl:.3f}")
        print(f"[METRIC] preset={args.preset} data={args.data} tokens={i:,} "
              f"ms_per_token={spent / max(1, i - done) * 1000:.2f} "
              f"final_ppl={final_ppl:.3f} oov_rate={oov_rate:.5f}")
    print(f"model: {final}")
    print(f"checkpoint: {ckpt}")
    print(f"[log] log saved: {log_path}")


if __name__ == "__main__":
    main()
