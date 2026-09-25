"""PHD-Net 1B 参数模型训练脚本（train_1b 子项目主入口）。

定位
----
在**事件驱动稀疏类脑语义**下训练总突触参数容量 ≥ 1×10^9 的 PHD-Net 词级 LM：
  - 1B 主体 = 大空间事件驱动长期记忆（big_ltm：2^24 × 60 ≈ 1.0066×10^9 突触容量，
    结构可塑性随经验生长，每步计算量只正比于活跃神经元数）；
  - 固定突触 = 稀疏主干 CSR（M2）+ STDP 侧向核（M3）+ 编码器（M1）+ 读出（M6）。
容量验算启动时自动打印（config_1b.capacity_report，1B 口径透明化）。

与 tools/train_production.py 的关系
------------------------------------
同一训练语义（词级 LM、表征冻结、结构性稀疏主干、pred_in_readout），
差异：① 1B 档预设与容量验算；② **完整检查点**（大空间表 + 迹/时间戳 +
词表 + 稀疏读出 CSR，生产版不保存这些）；③ 模型产物统一保存到
`models/` 目录（fhz 2026-09-25 指令）。

用法
----
# 冒烟（分钟级，验证管线；容量 <1B，仅功能验证）
python train_1b/train.py --preset smoke --data sft --tokens 2000

# 标准 1B 档（容量 ≈1.01–1.07×10^9；本机 CPU 每步约 0.1–0.5 s，小预算可跑）
python train_1b/train.py --preset 1b --data sft --tokens 100000

# 生产长跑（大内存/GPU 机器；断点续训）
python train_1b/train.py --preset 1b --data pretrain_zh --tokens 1000000000 --resume

# 查看当前检查点的大空间表利用率
python train_1b/train.py --preset 1b --data sft --report
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

from ckpt_1b import load_model, save_model                       # noqa: E402
from config_1b import PRESETS, SEG_KWARGS, build_cfg             # noqa: E402
from config_1b import capacity_report, print_capacity_report     # noqa: E402
from phdnet.corpus import expand_paths, load_text                # noqa: E402
from phdnet.model import count_params                            # noqa: E402
from phdnet.word_lm import PHDWordLM                             # noqa: E402

SAVE_DIR = _ROOT / "models"          # fhz 2026-09-25：训练模型统一存 models/
LOG_DIR = _ROOT / "outputs" / "train_logs"

DATA_FILES = {
    # 训练数据一律用上传分片 parquet（sft/ pretrain/ 根）；raw/ 仅归档不触碰
    # （fhz 2026-09-25 指令，与 tools/train_production.py 同步）。
    "sft": _ROOT / "datasets" / "sft" / "sft_000.*.parquet",
    "pretrain_zh": _ROOT / "datasets" / "pretrain" / "pretrain_*.parquet",
    "eval": _ROOT / "datasets" / "eval" / "internal_corpus.txt",
}

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


def _resolve_corpus(p: Path) -> Path:
    s = str(p)
    return p if any(c in s for c in "*?[") else p


def _print_table_stats(lm) -> None:
    """打印大空间事件驱动表实时利用率（1B 主体的生长进度）。"""
    if not hasattr(lm.net.ltm, "table"):
        return
    st = lm.net.ltm.table.stats()
    print(f"  [大空间表] 已生长 {st['grown_synapses']:,} / 容量 {st['capacity']:,}"
          f"（利用率 {st['utilization']:.3%}，触碰神经元 {st['touched_neurons']:,}）",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="PHD-Net 1B 档训练（事件驱动稀疏类脑）")
    ap.add_argument("--preset", choices=list(PRESETS), default="1b",
                    help="smoke=管线验证 / 1b=标准档(容量≥1B) / 1b_max=大主干档")
    ap.add_argument("--data", choices=list(DATA_FILES), default="sft")
    ap.add_argument("--width", type=int, default=0, help="覆盖主干宽度（0=用预设）")
    ap.add_argument("--big-n", type=int, default=0, help="覆盖大空间神经元数（0=用预设）")
    ap.add_argument("--csr-online", action="store_true",
                    help="大空间表切换在线可写 CSR（内存更优，与 dict 版逐位等价）")
    ap.add_argument("--readout-conn-k", type=int, default=0,
                    help="稀疏读出每输出单元入边数（0=稠密；大词表时建议 512–2048）")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tokens", type=int, default=0, help="token 预算（0=不限）")
    ap.add_argument("--minutes", type=float, default=0.0, help="时间预算分钟（0=不限）")
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--max-chars", type=int, default=4_000_000, help="语料载入上限")
    ap.add_argument("--resume", action="store_true",
                    help="从 models/ 下最近检查点续训（需与原训练同 --data/--max-chars）")
    ap.add_argument("--save-dir", type=Path, default=SAVE_DIR,
                    help=f"模型保存目录（默认 {SAVE_DIR}）")
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--report", action="store_true",
                    help="只打印检查点的大空间表统计后退出")
    args = ap.parse_args()

    # ── 日志落盘 ──
    if args.log_file is not None:
        log_path = args.log_file
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = LOG_DIR / f"train_1b_{args.preset}_{args.data}_{stamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = TeeLogger(log_path)

    print(f"[log] 日志文件：{log_path}")
    print(f"[log] 启动：{datetime.now().isoformat(timespec='seconds')}  "
          f"argv：{' '.join(sys.argv[1:]) or '(默认参数)'}")

    signal.signal(signal.SIGINT, _on_sigint)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.save_dir / f"phdnet1b_{args.preset}_{args.data}.npz"

    cfg = build_cfg(args.preset, args.width, args.big_n,
                    args.csr_online, args.readout_conn_k, args.seed)

    # ── 语料（parquet 分片优先，glob 多分片顺序拼接；不触碰 raw/）──
    data_path = _resolve_corpus(DATA_FILES[args.data])
    try:
        data_files = expand_paths(data_path)
    except FileNotFoundError as e:
        print(f"数据文件不存在: {e}")
        sys.exit(1)
    total_mb = sum(p.stat().st_size for p in data_files) / 1e6
    text = load_text(data_path, limit_chars=args.max_chars)

    t0 = time.perf_counter()
    lm = PHDWordLM(text, cfg, seg_kwargs=SEG_KWARGS)
    vocab = len(lm.tok)

    print("=" * 76)
    print(f"PHD-Net 1B 训练 | preset={args.preset} data={args.data} "
          f"width={cfg.n_sdr} conn_k={cfg.conn_k}")
    print(f"大空间表 N={cfg.big_ltm_N:,} × m={cfg.big_ltm_m}"
          f"（容量 {cfg.big_ltm_N * cfg.big_ltm_m:,}）| 词表 {vocab:,}"
          f" | 语料 {len(text):,} 字符 × {len(data_files)} 分片（{total_mb:.0f} MB）")
    print(f"预算 tokens={args.tokens or '∞'} minutes={args.minutes or '∞'}"
          f" | 构建耗时 {time.perf_counter() - t0:.1f}s")
    print_capacity_report(capacity_report(cfg, vocab))
    print(f"可塑参数（当前实际，count_params 口径）: {count_params(lm.net):,}")

    done = 0
    if args.resume and ckpt.exists():
        meta = load_model(ckpt, lm)
        done = int(meta["done"])
        print(f"[续训] 已恢复 {done:,} tokens（检查点 {meta['when']}）")
        _print_table_stats(lm)
    elif args.resume:
        print(f"[续训] 未找到检查点 {ckpt}，从头开始")

    if args.report:
        _print_table_stats(lm)
        print(f"[log] 日志已保存：{log_path}")
        return

    toks = lm.tokenize(text)
    print(f"数据集 token 数 {len(toks):,} | 开始训练", flush=True)

    n_total = len(toks) - 1
    seg_nll: list[float] = []
    t_start = time.perf_counter()
    i = done
    while i < n_total:
        if _STOP["flag"]:
            break
        if args.tokens and (i - done) >= args.tokens:
            break
        if args.minutes and (time.perf_counter() - t_start) / 60.0 >= args.minutes:
            break
        prev = toks[i - 1] if i > 0 else None
        x = lm.tok.encode_composite(toks[i], prev)
        tgt = lm.tok.onehot(lm.tok.stoi[toks[i + 1]])
        d = lm.net.step(x, target=tgt, learn=True)
        seg_nll.append(d["nll"])
        i += 1
        if (i - done) % args.log_every == 0:
            k = min(args.log_every, len(seg_nll))
            ppl = float(np.exp(np.mean(seg_nll[-k:])))
            spent = time.perf_counter() - t_start
            ms = spent / max(1, i - done) * 1000
            print(f"  token {i:>11,}  滑动 PPL {ppl:>9.3f}  {ms:>8.2f} ms/tok"
                  f"  已用 {spent / 60:.1f} min", flush=True)
            _print_table_stats(lm)
        if args.ckpt_every and (i - done) % args.ckpt_every == 0:
            save_model(ckpt, lm, cfg, i)
            print(f"  [检查点] 已保存 {i:,} tokens → {ckpt}", flush=True)

    # ── 收尾：滚动检查点 + final 模型 ──
    save_model(ckpt, lm, cfg, i)
    final = args.save_dir / f"phdnet1b_{args.preset}_{args.data}_final.npz"
    save_model(final, lm, cfg, i, extra={"final": True})

    spent = time.perf_counter() - t_start
    print("-" * 76)
    print(f"本次训练 {i - done:,} tokens，用时 {spent / 60:.1f} min"
          f"（{spent / max(1, i - done) * 1000:.2f} ms/token）")
    _print_table_stats(lm)
    if seg_nll:
        final_ppl = float(np.exp(np.mean(seg_nll[-min(2000, len(seg_nll)):])))
        print(f"末段滑动 PPL ≈ {final_ppl:.3f}")
        print(f"[METRIC] preset={args.preset} data={args.data} tokens={i:,} "
              f"ms_per_token={spent / max(1, i - done) * 1000:.2f} "
              f"final_ppl={final_ppl:.3f}")
    print(f"模型：{final}")
    print(f"检查点：{ckpt}")
    print(f"[log] 日志已保存：{log_path}")


if __name__ == "__main__":
    main()
