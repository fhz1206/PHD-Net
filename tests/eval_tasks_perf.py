"""M5 评测任务：效率（6）—— ms/token、RSS 增量、参数量。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import time

import numpy as np
import psutil

from eval_common import BASE, DOC, SEG, _rss_mb
from phdnet.config import PHDNetConfig
from phdnet.model import count_params
from phdnet.word_lm import PHDWordLM


def task6_efficiency(train_txt: str, text: str) -> dict:
    """效率：ms/token、RSS 增量、参数字节数。"""
    print("[任务6] 效率")
    r0 = _rss_mb()
    lm = PHDWordLM(text, PHDNetConfig(**BASE), seg_kwargs=SEG)
    t0 = time.perf_counter()
    lm.train_stream(train_txt)
    dt = time.perf_counter() - t0
    r1 = _rss_mb()
    n_tok = len(lm.tokenize(train_txt))
    net = lm.net
    params = count_params(net)   # C6 修复：集中统计，避免手工统计漂移；口径含 WM/LTM/迹状态除外
    print(f"    PHD-Net: {dt / n_tok * 1000:.2f} ms/token，RSS 增量 {r1 - r0:.1f} MB，"
          f"可塑参数 {params:,}（{params * 8 / 1e6:.1f} MB 等价；口径：编码/PC/STDP/读出/WM/LTM）")
    return {"ms_per_token": dt / n_tok * 1000, "rss_delta": r1 - r0,
            "params": params}
