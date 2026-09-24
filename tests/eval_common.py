"""M5 评测共享件 —— 语料路径、公共配置与跨任务辅助（自 eval_suite.py 拆分）。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import psutil

DOC = str(_ROOT / "datasets" / "eval" / "internal_corpus.txt")    # 默认语料（同 run_tests/验收脚本）
BASE = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
            eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
            pred_in_readout=True)
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def _tf_fit(model, ids: list[int], epochs: int = 3, block_size: int = 32,
            batch_size: int = 32, lr: float = 3e-3) -> None:
    import torch
    import torch.nn.functional as F
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    x = torch.tensor(ids, dtype=torch.long)
    n_batch = max(1, (len(x) - block_size - 1) // (batch_size * block_size))
    model.train()
    for _ in range(epochs):
        idx = torch.randint(0, max(1, len(x) - block_size - 1), (n_batch * batch_size,))
        inputs = torch.stack([x[i:i + block_size] for i in idx])
        targets = torch.stack([x[i + 1:i + 1 + block_size] for i in idx])
        perm = torch.randperm(inputs.size(0))
        for b in range(0, inputs.size(0), batch_size):
            j = perm[b:b + batch_size]
            if j.numel() == 0:
                continue
            logits = model(inputs[j])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets[j].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def _tf_acc(model, ids: list[int], block_size: int = 32, n: int = 600) -> float:
    import torch
    model.eval()
    hit = tot = 0
    with torch.no_grad():
        for i in range(min(n, len(ids) - 1)):
            ctx = torch.tensor(ids[max(0, i - block_size + 1):i + 1], dtype=torch.long)
            logits = model(ctx[None, :])[0, -1]
            if int(torch.argmax(logits)) == ids[i + 1]:
                hit += 1
            tot += 1
    return hit / max(1, tot)
