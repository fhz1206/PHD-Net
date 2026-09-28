"""torch 词级 LM 独立训练入口（不改动 train_1b/train.py 主流程，避免回归风险）。

数据流复用 train_1b.corpus_stream（char_chunks / build_vocab_text /
StreamingTokenizer）——数据侧保持 CPU numpy（分词/哈希编码），只有网络计算在
torch（--device auto|cpu|cuda|rocm|npu）。

示例：
  python tools/train_torch_lm.py --device cpu --preset smoke --data eval --tokens 2000
  python tools/train_torch_lm.py --device auto --preset base --data eval \\
      --tokens 50000 --ckpt outputs/smoke/torch_lm_base.npz --resume

检查点：npz（权重 + 标量状态 + 词表文本 + 配置 JSON），--resume 支持断点续训。
"""
from __future__ import annotations

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))
sys.path.insert(0, os.path.join(_ROOT, "train_1b"))

import argparse

import numpy as np

from corpus_stream import StreamingTokenizer, build_vocab_text, char_chunks
from eval_common import BASE, DOC, SEG                       # noqa: E402
from phdnet.backends.torch_lm import TorchWordLM
from phdnet.config import PHDNetConfig

PRESETS = {
    # 冒烟：1/2 维度栈，快速验证链路与 PPL 下降
    "smoke": {**BASE, "n_sdr": 128, "n_mid": 128, "n_top": 128},
    # 基线口径：与 tests/eval_common.BASE 一致（现行锚点见 tools/rebaseline.py）
    "base": dict(BASE),
}


def evaluate_text(lm: TorchWordLM, eval_txt: str) -> float:
    m = lm.evaluate(eval_txt)
    return m["ppl_char"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="auto",
                    help="auto / cpu / cuda / rocm / npu")
    ap.add_argument("--devices", type=str, default="auto",
                    help="多卡自动适配（P14）：auto=探测全部同型号设备；"
                         "或 'cuda:0,1' / 'npu:0 1' / 'cpu'。多卡时读出按词表行列并行")
    ap.add_argument("--max-devices", type=int, default=0,
                    help=">0 时截断设备数（多任务共享场景）")
    ap.add_argument("--preset", type=str, default="smoke",
                    choices=sorted(PRESETS))
    ap.add_argument("--data", type=str, default="eval",
                    help="'eval'（冻结语料 internal_corpus.txt）或路径/glob")
    ap.add_argument("--tokens", type=int, default=2000)
    ap.add_argument("--vocab-sample-chars", type=int, default=4000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500,
                    help="每 N 个学习步评估一次 PPL（--data eval 时）")
    ap.add_argument("--ckpt", type=str, default="",
                    help="检查点 npz 路径（训练中周期保存 + 结束保存）")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true",
                    help="ckpt 存在时从断点续训")
    args = ap.parse_args()

    data = DOC if args.data == "eval" else args.data
    cfg = PHDNetConfig(**PRESETS[args.preset])

    # ---- 评估文本（--data eval：80/20 划分的尾段）
    eval_txt = None
    if args.data == "eval":
        with open(DOC, encoding="utf-8") as f:
            full = f.read()
        eval_txt = full[int(len(full) * 0.8):]

    # ---- P14 多卡自动适配：解析设备（单设备 = 原生路径，零行为变更）
    from phdnet.backends.multi_device import configure_host_threads, resolve_devices
    devices = resolve_devices(args.devices, max_devices=args.max_devices,
                              allow_fallback=True)
    if len(devices) > 1:
        configure_host_threads(1)              # 避免 CPU 线程与卡内线程争抢
    print(f"[device] devices={devices}"
          + ("（多卡：读出列并行，host 线程已收敛为 1）" if len(devices) > 1
             else "（单设备原生路径）"), flush=True)

    # ---- 检查点 / 断点续训
    start_step = 0
    if args.resume and args.ckpt and os.path.exists(args.ckpt):
        with np.load(args.ckpt, allow_pickle=False) as z:
            vocab_text = str(z["vocab_text"])
            meta = json.loads(str(z["meta"]))
            cfg = PHDNetConfig(**meta["cfg"])
            start_step = int(z["step_count"][0]) if "step_count" in z else 0
            # 状态数组必须在 with 块内取出（块外访问已关闭的 npz 会失败）
            state = {k: z[k] for k in z.files
                     if k not in ("vocab_text", "meta")}
        lm = TorchWordLM(vocab_text, cfg, device=args.device,
                         seg_kwargs=meta.get("seg"), devices=devices)
        # 用重建的模型逐字重放词表文本并无害（只为拿到 seg），状态来自检查点
        lm.net.load_state_arrays(state)
        start_step = int(lm.net.step_count)
        print(f"[resume] 从 {args.ckpt} 恢复：step={start_step}", flush=True)
    else:
        # ---- 词表（与训练流开头逐字符一致，train_1b 同款）
        vocab_text = build_vocab_text(data, args.vocab_sample_chars)
        lm = TorchWordLM(vocab_text, cfg, device=args.device, seg_kwargs=SEG,
                         devices=devices)
        print(f"[init] 词表 {len(lm.tok)} token（采样 {len(vocab_text)} 字符）"
              f"  device={lm.device}", flush=True)

    def save_ckpt(step_done: int) -> None:
        if not args.ckpt:
            return
        os.makedirs(os.path.dirname(os.path.abspath(args.ckpt)), exist_ok=True)
        d = lm.net.state_arrays()           # 已含 step_count（模型内部计数）
        np.savez_compressed(
            args.ckpt, vocab_text=np.array(vocab_text),
            meta=np.array(json.dumps(
                {"cfg": cfg.__dict__, "seg": SEG, "preset": args.preset})), **d)

    # ---- 流式训练（token 边产边训；OOV 步跳过并计数，与 CPU 版语义一致）
    toks_stream = StreamingTokenizer(lm.tok.seg, char_chunks(data))
    stoi = lm.tok.stoi
    prev: str | None = None
    cur: str | None = None
    n_learn = n_oov = 0
    seg_nll: list[float] = []
    curve: list[tuple[int, float]] = []
    t0 = time.perf_counter()
    stop = False
    for tok in toks_stream:
        if cur is None:
            cur = tok
            continue
        nxt = tok
        if cur in stoi and nxt in stoi:
            if n_learn >= args.tokens:
                stop = True
                break
            d = lm.net.step(lm.tok.encode_composite(cur, prev),
                            target=lm.tok.onehot(stoi[nxt]), learn=True)
            seg_nll.append(d["nll"])
            n_learn += 1
            step = lm.net.step_count
            if args.eval_every > 0 and eval_txt and step % args.eval_every == 0:
                ppl = evaluate_text(lm, eval_txt)
                curve.append((step, ppl))
                print(f"[eval ] step={step}  ppl_char={ppl:.4f}", flush=True)
            if args.log_every > 0 and step % args.log_every == 0:
                dt = time.perf_counter() - t0
                print(f"[train] step={step}  nll_mean={np.mean(seg_nll[-args.log_every:]):.4f}  "
                      f"{dt / max(1, step - start_step) * 1000:.2f} ms/token", flush=True)
            if args.ckpt and args.ckpt_every > 0 and step % args.ckpt_every == 0:
                save_ckpt(step)
            prev = cur                      # prev 只推进到已知 token（OOV 安全）
        else:
            n_oov += 1                      # OOV：跳过学习，prev 保持最后已知 token（与 _pass 同口径）
        cur = nxt

    # ---- 收尾：最终评估 + 检查点
    if eval_txt:
        ppl_final = evaluate_text(lm, eval_txt)
        curve.append((lm.net.step_count, ppl_final))
        print(f"[final] step={lm.net.step_count}  ppl_char={ppl_final:.4f}  "
              f"(learned={n_learn}, oov_skipped={n_oov})", flush=True)
    save_ckpt(lm.net.step_count)
    if args.ckpt:
        print(f"[ckpt ] 已保存 {args.ckpt}", flush=True)

    # ---- PPL 趋势判定（供验收：持续下降）
    if len(curve) >= 2:
        declining = all(curve[i + 1][1] <= curve[i][1] * 1.02
                        for i in range(len(curve) - 1))
        better = curve[-1][1] < curve[0][1]
        print(f"[curve] {[(s, round(p, 2)) for s, p in curve]}")
        print(f"[curve] 下降趋势: {'是' if better else '否'}"
              f"（末点 vs 首点 {'↓' if better else '↑'}）；"
              f"平滑下降: {'是' if declining else '否（允许 ±2% 抖动）'}")


if __name__ == "__main__":
    main()
