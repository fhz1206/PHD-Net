"""M5 公平评测对照组：nanoGPT 级 Transformer（PyTorch，CPU 可训）。

用途：为 PHD-Net 提供**同参数规模、同数据、同预算**的稠密 Transformer 对照。
实现刻意保持最小（嵌入 + 学习位置编码 + N×(自注意力+MLP) + 输出头），
参数量由 d_model/n_layer 控制，便于与 PHD-Net 的参数口径对齐。

公平口径：
  - 同语料、同划分、同字符数；
  - 报告**字符归一 PPL**（exp(总 NLL / 字符数)）与 bpc，与词级 PHD-Net 可比；
  - 训练预算以墙钟与 CPU 时间双口径记录。
"""

from __future__ import annotations

# --- 目录结构调整（2026-09-18）：脚本位于 tests/ 或 tools/ 子目录 ---
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]     # 项目根目录
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))              # 保证 `import phdnet` 可用
# --- 引导结束 ---

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NanoGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 96, n_layer: int = 2,
                 n_head: int = 4, block_size: int = 64, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_head, dropout=dropout,
                                              batch_first=True),
                "ln2": nn.LayerNorm(d_model),
                "mlp": nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                     nn.Linear(4 * d_model, d_model)),
            }) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), 1)
        for blk in self.blocks:
            h = blk["ln1"](x)
            a, _ = blk["attn"](h, h, h, attn_mask=causal, need_weights=False)
            x = x + a
            x = x + blk["mlp"](blk["ln2"](x))
        return self.head(self.ln_f(x))


def train_eval(train_ids: list[int], eval_ids: list[int], vocab_size: int,
               d_model: int = 96, n_layer: int = 2, n_head: int = 4,
               block_size: int = 64, batch_size: int = 32, epochs: int = 3,
               lr: float = 3e-3, seed: int = 11, chars_per_token: float = 1.0,
               verbose: bool = True) -> dict:
    """训练并在评估段上算字符归一 PPL / bpc。返回指标与预算记录。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cpu")
    model = NanoGPT(vocab_size, d_model, n_layer, n_head, block_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    x = torch.tensor(train_ids, dtype=torch.long)
    B_, T_ = batch_size, block_size
    n_batch = max(1, (len(x) - T_ - 1) // (B_ * T_))
    t0, c0 = time.perf_counter(), time.process_time()
    for ep in range(epochs):
        model.train()
        idx = torch.randint(0, max(1, len(x) - T_ - 1), (n_batch * B_,))
        inputs = torch.stack([x[i:i + T_] for i in idx])
        targets = torch.stack([x[i + 1:i + 1 + T_] for i in idx])
        perm = torch.randperm(inputs.size(0))
        for b in range(0, inputs.size(0), B_):
            j = perm[b:b + B_]
            if j.numel() == 0:
                continue
            logits = model(inputs[j].to(dev))
            loss = F.cross_entropy(logits.reshape(-1, vocab_size),
                                   targets[j].reshape(-1).to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    train_s = time.perf_counter() - t0
    cpu_s = time.process_time() - c0

    # 评估：滑动窗口，逐 token 取最后一个位置的分布（与 PHD-Net 在线评估口径一致）
    model.eval()
    xe = torch.tensor(eval_ids, dtype=torch.long)
    total_nll, n_tok = 0.0, 0
    with torch.no_grad():
        for i in range(len(xe) - 1):
            ctx = xe[max(0, i - T_ + 1):i + 1]
            if ctx.numel() < 1:
                continue
            logits = model(ctx[None, :].to(dev))[0, -1]
            lp = torch.log_softmax(logits, dim=-1)
            total_nll += float(-lp[xe[i + 1]])
            n_tok += 1
    n_char = n_tok * chars_per_token
    ppl_char = math.exp(total_nll / max(1, n_char))
    if verbose:
        print(f"    Transformer({d_model}d×{n_layer}L, {model.params():,} 参数): "
              f"token PPL = {math.exp(total_nll / max(1, n_tok)):.2f}  "
              f"字符归一 PPL = {ppl_char:.2f}  "
              f"bpc = {total_nll / max(1, n_char) / math.log(2):.3f}  "
              f"训练 {train_s:.1f}s（CPU {cpu_s:.1f}s）")
    return {
        "params": model.params(),
        "ppl_token": math.exp(total_nll / max(1, n_tok)),
        "ppl_char": ppl_char,
        "bpc": total_nll / max(1, n_char) / math.log(2),
        "train_s": train_s,
        "cpu_s": cpu_s,
    }
