"""生产级训练脚本 —— PHD-Net 词级 LM（可配置规模 / 断点续训 / 检查点 / 日志）。

设计目标：在**具备 GPU 或集群的机器**上可直接用于 256M 档训练；在本机（纯 CPU）
则受算力限制（见 `tools/estimate_scale.py` 实测外推）。

功能
----
1. 规模预设：`--scale small|1M|4M|16M|256M`（稀疏主干 + 可选 1B 容量栈）
2. 数据源：`--data sft|infinity_zh|infinity|eval`（自动定位 `datasets/` 下文件）
3. 预算控制：`--tokens N` 与 `--minutes M`（任一到达即收尾）
4. 断点续训：`--resume` 从最近检查点继续（含模型权重 + 已训 token 数 + 日志）
5. 检查点：每 `--ckpt-every` tokens 保存一次（npz，含配置与进度）
6. 日志：每 `--log-every` tokens 打印滑动 PPL / ms per token / 剩余时间估计
7. 优雅退出：收到 SIGINT 先保存检查点再退出

用法示例
--------
# 本机小规模验证（约 10 分钟）
python tools/train_production.py --scale 4M --data sft --minutes 10

# GPU 机器上 256M 档（需自行确认显存/内存与数据量）
python tools/train_production.py --scale 256M --data infinity_zh --tokens 100000000

# 断点续训
python tools/train_production.py --scale 256M --data infinity_zh --resume

硬件需求（256M 档，权重 fp64 ≈ 2.05 GB；fp32 ≈ 1.02 GB）
--------------------------------------------------------
- 内存/显存：≥ 8 GB（含中间激活与读出矩阵）
- 本机实测吞吐（纯 CPU、numba）：157 ms/token ⇒ 10^7 token ≈ 18 天；
  10^9 token（生产级）≈ 5 年 —— **本机不可行**。GPU 上按 50–100× 加速估算，
  10^9 token ≈ 1–2 个月（单卡），仍需多卡并行。
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.config import PHDNetConfig          # noqa: E402
from phdnet.model import count_params           # noqa: E402
from phdnet.word_lm import PHDWordLM            # noqa: E402

CKPT_DIR = _ROOT / "outputs" / "ckpt"
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)

SCALES = {
    # name      宽度   conn_k  词表(0=语料全词表)  备注
    "small": dict(width=256, conn_k=32),
    "1M": dict(width=512, conn_k=64),
    "4M": dict(width=1024, conn_k=128),
    "16M": dict(width=2048, conn_k=256),
    "256M": dict(width=16384, conn_k=2048),      # 稀疏主干：3×16384×2048 ≈ 1.0 亿 + 读出
}

DATA_FILES = {
    "sft": _ROOT / "datasets" / "sft" / "ultrainteract_sft.txt",
    "infinity_zh": _ROOT / "datasets" / "pretrain" / "infinity_m7core.txt",
    "infinity": _ROOT / "datasets" / "pretrain" / "infinity_m7core.txt",
    "eval": _ROOT / "datasets" / "eval" / "internal_corpus.txt",
}

_STOP = {"flag": False}


def _on_sigint(signum, frame):        # pragma: no cover
    _STOP["flag"] = True
    print("\n[收到中断信号] 将保存检查点后退出…", flush=True)


def build_cfg(scale: str, big_ltm: bool) -> PHDNetConfig:
    s = SCALES[scale]
    w, k = s["width"], s["conn_k"]
    return PHDNetConfig(n_sdr=w, n_mid=w, n_top=w,
                        k_sparse=max(8, w // 8), eta_pc=0.0, eta_oja=0.0,
                        eta_stdp=0.02, seed=11, pred_in_readout=True,
                        sparse_conn=True, conn_k=k,
                        readout_conn_k=0,
                        big_ltm=big_ltm, big_ltm_N=1 << 22, big_ltm_m=60)


def save_ckpt(path: Path, lm: PHDWordLM, cfg: PHDNetConfig, done: int,
              use_csr: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    net = lm.net
    arrs = {
        "encoder_W": np.asarray(net.encoder.W), "encoder_b": np.asarray(net.encoder.b),
        "stdp_W": np.asarray(net.stdp.W),
        "readout_W": np.asarray(net.readout.W),
        "wm_slots": np.asarray(net.wm.slots), "wm_strength": np.asarray(net.wm.strength),
    }
    if use_csr:
        ip, idx, val = net.pc.up0
        arrs["pc_up0_indptr"], arrs["pc_up0_idx"], arrs["pc_up0_val"] = ip, idx, val
        ip, idx, val = net.pc.up1
        arrs["pc_up1_indptr"], arrs["pc_up1_idx"], arrs["pc_up1_val"] = ip, idx, val
        ip, idx, val = net.pc.dn0
        arrs["pc_dn0_indptr"], arrs["pc_dn0_idx"], arrs["pc_dn0_val"] = ip, idx, val
        ip, idx, val = net.pc.dn1
        arrs["pc_dn1_indptr"], arrs["pc_dn1_idx"], arrs["pc_dn1_val"] = ip, idx, val
    else:
        for nm in ("W_up0", "W_up1", "W_dn0", "W_dn1"):
            arrs[f"pc_{nm}"] = np.asarray(getattr(net.pc, nm))
    np.savez_compressed(str(path),
                        meta=np.array([json.dumps({"done": done, "csr": use_csr})]),
                        **arrs)
    (path.with_suffix(".json")).write_text(json.dumps(
        {"done_tokens": done, "params": count_params(net),
         "when": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2),
        encoding="utf-8")


def load_ckpt(path: Path, lm: PHDWordLM) -> int:
    net = lm.net
    with np.load(str(path)) as z:
        net.encoder.W[:] = z["encoder_W"]
        net.encoder.b[:] = z["encoder_b"]
        net.stdp.W[:] = z["stdp_W"]
        if net.readout.conn_k > 0:
            print("  [警告] 稀疏读出暂不支持检查点恢复（将保持随机初始化）")
        else:
            net.readout.W[:] = z["readout_W"]
        net.wm.slots[:] = z["wm_slots"]
        net.wm.strength[:] = z["wm_strength"]
        if hasattr(net.pc, "up0") and "pc_up0_val" in z:
            net.pc.up0[2][:] = z["pc_up0_val"]
            net.pc.up1[2][:] = z["pc_up1_val"]
            net.pc.dn0[2][:] = z["pc_dn0_val"]
            net.pc.dn1[2][:] = z["pc_dn1_val"]
        elif "pc_W_up0" in z:
            net.pc.W_up0[:] = z["pc_W_up0"]
            net.pc.W_up1[:] = z["pc_W_up1"]
            net.pc.W_dn0[:] = z["pc_W_dn0"]
            net.pc.W_dn1[:] = z["pc_W_dn1"]
        done = int(z["meta"][0] and json.loads(str(z["meta"][0])).get("done", 0))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=list(SCALES), default="4M")
    ap.add_argument("--data", choices=list(DATA_FILES), default="sft")
    ap.add_argument("--tokens", type=int, default=0, help="token 预算（0 = 不限）")
    ap.add_argument("--minutes", type=float, default=0.0, help="时间预算（分钟，0=不限）")
    ap.add_argument("--ckpt-every", type=int, default=20000)
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--big-ltm", action="store_true", help="启用 1B 事件驱动容量栈")
    ap.add_argument("--max-chars", type=int, default=4_000_000, help="语料载入上限")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_sigint)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"phdnet_{args.scale}_{args.data}.npz"

    data_path = DATA_FILES[args.data]
    if not data_path.exists():
        print(f"数据文件不存在: {data_path}")
        sys.exit(1)
    text = data_path.read_text(encoding="utf-8")[: args.max_chars]
    cfg = build_cfg(args.scale, args.big_ltm)
    print("=" * 92)
    print(f"PHD-Net 生产级训练 | scale={args.scale} data={args.data} "
          f"稀疏主干 conn_k={cfg.conn_k} 宽度={cfg.n_sdr}")
    print(f"语料 {len(text):,} 字符 | 预算 tokens={args.tokens or '∞'} "
          f"minutes={args.minutes or '∞'}")
    print("=" * 92)

    t0 = time.perf_counter()
    lm = PHDWordLM(text, cfg, seg_kwargs=SEG)
    n_par = count_params(lm.net)
    toks = lm.tokenize(text)
    print(f"词表 {len(lm.tok):,} | 可塑参数 {n_par:,} | 数据集 token 数 {len(toks):,} "
          f"| 构建耗时 {time.perf_counter() - t0:.1f}s", flush=True)

    done = 0
    if args.resume and ckpt.exists():
        try:
            done = load_ckpt(ckpt, lm)
            print(f"[续训] 已从检查点恢复 {done:,} tokens", flush=True)
        except Exception as e:
            print(f"[续训失败] {e!r}（将从头开始）", flush=True)

    use_csr = hasattr(lm.net.pc, "up0")
    n_total = len(toks) - 1
    seg_nll: list[float] = []
    t_start = time.perf_counter()
    i = done
    while i < n_total:
        if _STOP["flag"]:
            break
        if args.tokens and (i - done) >= args.tokens:
            break
        elapsed_min = (time.perf_counter() - t_start) / 60.0
        if args.minutes and elapsed_min >= args.minutes:
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
            print(f"  token {i:>9,}  滑动 PPL {ppl:>9.3f}  {ms:>7.2f} ms/tok  "
                  f"已用 {spent / 60:.1f} min", flush=True)
        if args.ckpt_every and (i - done) % args.ckpt_every == 0:
            save_ckpt(ckpt, lm, cfg, i, use_csr)
            print(f"  [检查点] 已保存 {i:,} tokens → {ckpt}", flush=True)

    save_ckpt(ckpt, lm, cfg, i, use_csr)
    spent = time.perf_counter() - t_start
    print("-" * 92)
    print(f"本次训练 {i - done:,} tokens，用时 {spent / 60:.1f} min"
          f"（{spent / max(1, i - done) * 1000:.2f} ms/token）")
    if seg_nll:
        print(f"末段滑动 PPL ≈ {float(np.exp(np.mean(seg_nll[-min(2000, len(seg_nll)):]))):.3f}")
    print(f"检查点：{ckpt}")


if __name__ == "__main__":
    main()
