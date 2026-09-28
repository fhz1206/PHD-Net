"""PHD-Net 1B 模型推理 / 对话入口（train_1b 子项目，与流式训练配套）。

能力（2026-09-25，配合 1M context 流式训练补齐推理侧）
----------------------------------------------------
1. **检查点自包含加载**：从 `outputs/models/phdnet1b_*.npz` 恢复模型，**不需要语料**
   ——词表（seg.vocab/tokens/SDR 哈希）已完整序列化在检查点内（ckpt_1b），
   cfg 由检查点元数据重建。训练与推理的词表逐位一致（同一哈希逻辑）。
2. **流式长 prompt**：prompt 经 StreamingTokenizer 流式预热（learn=False），
   任意长（≥1M token）不截断；预热进度按 `--milestone` 打点。网络状态全程
   连续携带（WM/STDP/LTM），与训练同一 context 语义。
3. **自回归采样**：温度 τ + top-k 截断（M11 机制的词级版统一实现）。
4. **两种模式**：`--prompt` 单次续写 / 交互式对话（`用户：…\n助手：` 模板，
   多轮共享网络状态——对话历史就是 context）。

诚实边界：模型是**词级联想续写器**（PHD-Net 无注意力/位置编码，条件化 =
WM 锚定 + LTM 印迹 + 情景缓冲），生成的是格式上连贯的文本，不保证事实
正确或真正的语义问答。`phdnet/generate.py::Generator` 是字符级 M11 遗留
接口，与词级 LM 不兼容，请勿混用。

用法
----
python train_1b/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz \
    --prompt "用户：什么是机器学习？\n助手：" --n 200
python train_1b/infer.py --model outputs/models/phdnet1b_1b_sft_final.npz --chat
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_HERE), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ckpt_1b import _rebuild_sdrs, load_model                      # noqa: E402
from corpus_stream import SEP, StreamingTokenizer, char_chunks     # noqa: E402
from phdnet.config import PHDNetConfig                             # noqa: E402
from phdnet.word_encoder import WordSegmenter, WordTokenizer       # noqa: E402
from phdnet.word_lm import PHDWordLM                               # noqa: E402

CHAT_TEMPLATE = "用户：{q}\n助手："


def load_from_ckpt(model_path: Path) -> PHDWordLM:
    """从检查点自包含重建 PHDWordLM（词表 + cfg + 全部权重；不需要语料）。"""
    import json
    with np.load(str(model_path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"][0]))
        ckpt_vocab = {str(w) for w in z["tok_vocab"]}
        ckpt_tokens = [str(t) for t in z["tok_tokens"]]
    cfg = PHDNetConfig(**meta["cfg"])

    # 词表重建（与 ckpt_1b._restore_tokenizer 同一注入逻辑，逐位一致）
    seg = WordSegmenter.__new__(WordSegmenter)
    seg.vocab = ckpt_vocab
    seg.max_len = 6
    tok = WordTokenizer.__new__(WordTokenizer)
    tok.seg = seg
    tok.tokens = ckpt_tokens
    tok.stoi = {t: i for i, t in enumerate(tok.tokens)}
    tok.n_sdr, tok.n_active, tok.seed = cfg.n_sdr, cfg.k_sparse, cfg.seed
    _rebuild_sdrs(tok, tok.tokens)

    lm = PHDWordLM("", cfg, tokenizer=tok)
    got = load_model(model_path, lm)
    print(f"[infer] 模型已加载：{model_path}")
    print(f"[infer] 词表 {len(lm.tok):,} | done={got['done']:,} tokens | "
          f"检查点 {got['when']}")
    return lm


def warmup(lm: PHDWordLM, tokens_iter, milestone: int) -> tuple[int, list[str]]:
    """把 prompt token 流入网络（learn=False，WM/STDP/LTM 状态累积）。

    单遍流式：边消化边保留最后 2 个已消化 token（自回归衔接用）。
    返回 (OOV 计数, 尾部 token 列表 ≤2)。"""
    oov = 0
    prev = None
    tail: list[str] = []
    t_m = time.perf_counter()
    for n_seen, tk in enumerate(tokens_iter, 1):
        if tk in lm.tok.stoi:
            lm.net.step(lm.tok.encode_composite(tk, prev), learn=False)
            prev = tk
            tail.append(tk)
            if len(tail) > 2:
                tail.pop(0)
        else:
            oov += 1                     # OOV：跳过（与训练侧同一回退语义）
        if milestone and n_seen % milestone == 0:
            print(f"  [预热] 已消化 {n_seen:,} prompt tokens"
                  f"（{time.perf_counter() - t_m:.0f}s）", flush=True)
    return oov, tail


def stream_prompt_tokens(lm: PHDWordLM, prompt: str):
    """prompt → token 流（分词器与训练侧逐位一致；长 prompt 流式不截断）。"""
    seg = lm.tok.seg
    for tk in StreamingTokenizer(seg, iter([prompt + SEP])):
        yield tk


def sample_next(lm: PHDWordLM, prev: str | None, cur: str,
                tau: float, topk: int, rng: np.random.Generator) -> str:
    """自回归一步：读出分布 → 温度 τ → top-k 截断 → 采样下一 token。"""
    x = lm.tok.encode_composite(cur, prev)
    y = lm.net.step(x, learn=False)["y"].astype(np.float64) / max(tau, 1e-6)
    y -= y.max()
    p = np.exp(y)
    p /= p.sum()
    if 0 < topk < len(p):
        idx = np.argpartition(-p, topk - 1)[:topk]
        pp = p[idx] / p[idx].sum()
        j = int(rng.choice(idx, p=pp))
    else:
        j = int(rng.choice(len(p), p=p))
    return lm.tok.tokens[j]


def generate(lm: PHDWordLM, prompt: str, n_tokens: int = 200,
             tau: float = 0.7, topk: int = 8, seed: int = 0,
             milestone: int = 0) -> str:
    """流式预热 + 自回归续写。prompt 任意长（含 1M 级）；返回生成的文本。"""
    rng = np.random.default_rng(seed)
    oov, tail = warmup(lm, stream_prompt_tokens(lm, prompt), milestone)
    if oov:
        print(f"[infer] prompt 中 OOV token 跳过 {oov:,} 个", flush=True)
    prev, cur = None, None
    if len(tail) == 2:                    # 预热后衔接自回归：最后两个已消化 token
        prev, cur = tail[0], tail[1]
    elif len(tail) == 1:
        cur = tail[0]
    else:
        cur = lm.tok.tokens[0]
    out = []
    for _ in range(n_tokens):
        nxt = sample_next(lm, prev, cur, tau, topk, rng)
        out.append(nxt)
        prev, cur = cur, nxt
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="PHD-Net 1B 推理 / 对话")
    ap.add_argument("--model", type=Path, required=True,
                    help="outputs/models/phdnet1b_*.npz（检查点或 final）")
    ap.add_argument("--prompt", type=str, default="",
                    help="单次续写 prompt（空 + --chat 进入交互）")
    ap.add_argument("--chat", action="store_true",
                    help="交互式对话（多轮共享网络状态）")
    ap.add_argument("--n", type=int, default=200, help="生成 token 数")
    ap.add_argument("--tau", type=float, default=0.7, help="温度（越高越多样）")
    ap.add_argument("--topk", type=int, default=8, help="top-k 截断（0=全分布）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--milestone", type=int, default=1_000_000,
                    help="预热进度打点间隔 tokens（0=关闭）")
    args = ap.parse_args()

    lm = load_from_ckpt(args.model)

    if args.chat:
        print("[infer] 交互对话（逐行输入，Ctrl-C 退出；多轮状态连续共享）")
        while True:
            try:
                q = input("\n用户：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[infer] 退出")
                break
            if not q:
                continue
            prompt = CHAT_TEMPLATE.format(q=q)
            print("助手：", end="", flush=True)
            print(generate(lm, prompt, args.n, args.tau, args.topk,
                           args.seed + len(q), args.milestone))
    else:
        t0 = time.perf_counter()
        text = generate(lm, args.prompt, args.n, args.tau, args.topk,
                        args.seed, args.milestone)
        print(text)
        print(f"[infer] 生成 {args.n} tokens 用时 {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
