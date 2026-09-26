"""P9 精度体系验证（2026-09-26，fhz 指令：停止 fp64；fp16/bf16/fp8/fp4；默认 fp32）。

L1 逐位：各量化 dtype 的位算法融合核 vs numpy 参考
        （fp16=IEEE RNE astype / bf16=RNE 位截断 / fp8·fp4=格点 searchsorted，
        内核与参考的舍入约定逐格式对齐）→ 码本逐元素一致；
L2 带宽：V=9219 × H=3072（1B 预设真实读出规模）各 dtype 更新核耗时/等效带宽/存储；
L3 端到端：冻结语料 BASE 口径（4,000 字符训练段）各 dtype 的 ppl_char 与 ms/token；
        fp32 / fp16 另测全语料口径。
"""

from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import numpy as np                                        # noqa: E402
import numba                                              # noqa: E402

from eval_common import BASE, DOC, SEG                    # noqa: E402
from phdnet.config import PHDNetConfig                    # noqa: E402
from phdnet.readout import (Readout, RO_DTYPES, _ro_dense_update,   # noqa: E402
                            _ro_q_update_fp16, _ro_q_update_bf16,
                            _ro_q_update_fp8, _ro_q_update_fp4,
                            _q_scratch, quantize_to, dequantize_from)
from phdnet.word_lm import PHDWordLM                      # noqa: E402

UPD = {"fp32": _ro_dense_update, "fp16": _ro_q_update_fp16,
       "bf16": _ro_q_update_bf16, "fp8": _ro_q_update_fp8,
       "fp4": _ro_q_update_fp4}


def np_reference(fmt: str, codes, wscale: float, dp, h, eta) -> np.ndarray:
    """numpy 参考路径（dequant → f32 更新 → 按格式约定重量化）。"""
    V, H = dp.shape[0], h.shape[0]
    w = dequantize_from(fmt, codes, V * H).reshape(V, H) * wscale
    w = w - (dp[:, None] * h[None, :]) * eta
    if fmt == "fp16":
        h = np.ascontiguousarray(w, dtype=np.float32).astype(np.float16)
        h = np.nan_to_num(h, nan=np.float16(0), posinf=np.float16(65504),
                          neginf=np.float16(-65504))
        return h.view(np.uint16).ravel()
    if fmt == "bf16":
        u = np.ascontiguousarray(w, dtype=np.float32).view(np.uint32)
        sign = u & np.uint32(0x80000000)
        r = u & np.uint32(0x7FFFFFFF)
        c = (r + np.uint32(0x7FFF) + ((r >> 16) & np.uint32(1))) >> 16
        c = np.where(c >= np.uint32(0x7F80), np.uint32(0x7F7F), c)
        return (c | (sign >> 16)).astype(np.uint16).ravel()
    flat = w.ravel()
    if fmt == "fp4":
        flat = flat / wscale
    return quantize_to(fmt, flat)


def l1() -> bool:
    print("[L1] 位算法融合核 vs numpy 参考（码本逐元素一致）")
    rng = np.random.default_rng(7)
    ok_all = True
    scr = _q_scratch()
    for fmt in ("fp16", "bf16", "fp8", "fp4"):
        ok_fmt = True
        for trial in range(3):
            V, H = 300, 256
            W0 = (rng.standard_normal((V, H)) * 0.05).astype(np.float32)
            dp = (rng.random(V) * 0.01).astype(np.float32)
            h = rng.standard_normal(H).astype(np.float32)
            eta = np.float32(0.05)
            wscale = (float(np.abs(W0).max()) / 6.0 or 1.0) if fmt == "fp4" else 1.0
            codes_a = quantize_to(fmt, W0 / wscale).reshape(-1).copy()
            codes_b = quantize_to(fmt, W0 / wscale).reshape(-1).copy()
            if fmt == "fp4":
                UPD[fmt](codes_a, scr, dp, h, eta, H, wscale)
            else:
                UPD[fmt](codes_a, scr, dp, h, eta, H)
            ref = np_reference(fmt, codes_b, wscale, dp, h, eta)
            ok_fmt &= bool(np.array_equal(codes_a.astype(np.int64),
                                          np.asarray(ref).astype(np.int64)))
        ok_all &= ok_fmt
        print(f"  {fmt}: {'PASS' if ok_fmt else 'FAIL'}", flush=True)
    return ok_all


def l2() -> None:
    print("[L2] 更新核带宽（V=9219 × H=3072，1B 预设真实读出规模）")
    V, H = 9219, 3072
    scr = _q_scratch()
    for fmt in ("fp32", "fp16", "bf16", "fp8", "fp4"):
        rng = np.random.default_rng(11)
        if fmt == "fp32":
            W = (rng.standard_normal((V, H)) * 0.01).astype(np.float32)
            dp = (rng.random(V) * 0.001).astype(np.float32)
            h = rng.standard_normal(H).astype(np.float32)
            eta = np.float32(0.05)
            args = (W, dp, h, eta)
            nbytes = 4

            def K(a, b, c, d):
                _ro_dense_update(a, b, c, d)
        else:
            codes = quantize_to(fmt, rng.standard_normal((V, H)) * 0.01)
            dp = (rng.random(V) * 0.001).astype(np.float32)
            h = rng.standard_normal(H).astype(np.float32)
            eta = np.float32(0.05)
            wscale = (float(np.abs(rng.standard_normal((V, H))).max()) / 6.0
                      if fmt == "fp4" else 1.0)
            K = UPD[fmt]
            args = ((codes, scr, dp, h, eta, H, wscale) if fmt == "fp4"
                    else (codes, scr, dp, h, eta, H))
            nbytes = {"fp16": 2, "bf16": 2, "fp8": 1, "fp4": 0.5}[fmt]
        K(*args)                                          # JIT 预热
        t = time.perf_counter()
        for _ in range(5):
            K(*args)
        dt = (time.perf_counter() - t) / 5
        traffic = 2 * V * H * nbytes + (V * 2 + H) * 4
        print(f"  {fmt}: {dt * 1000:7.2f} ms  等效带宽={traffic / 1e9 / dt:6.1f} GB/s"
              f"  存储={V * H * nbytes / 1e6:6.1f} MB", flush=True)


def l3_e2e(limit: int | None, dtype: str) -> dict:
    with open(DOC, encoding="utf-8") as f:
        text = f.read()
    split = int(len(text) * 0.8)
    train_txt = text[:split][:limit] if limit else text[:split]
    eval_txt = text[split:]
    cfg = PHDNetConfig(**BASE, readout_dtype=dtype)
    lm = PHDWordLM(text, cfg, seg_kwargs=SEG)
    t0 = time.perf_counter()
    lm.train_stream(train_txt)
    dt = time.perf_counter() - t0
    m = lm.evaluate(eval_txt)
    n_tok = len(lm.tokenize(train_txt))
    return dict(ppl=m["ppl_char"], ms=dt / max(1, n_tok) * 1000)


def l3(dtype_list) -> None:
    print("[L3] 端到端（冻结语料 80/20，BASE 口径，默认读出精度=fp32）")
    base = None
    for lim, dts in ((4000, dtype_list), (None, ("fp32", "fp16"))):
        for dt in dts:
            r = l3_e2e(lim, dt)
            if lim == 4000:
                if base is None:
                    base = r["ppl"]
                d = (r["ppl"] - base) / base * 100
                tag = f"Δvs fp32 {d:+.2f}%"
            else:
                tag = "全语料"
            print(f"  [{lim or 'full'}] {dt}: ppl {r['ppl']:.4f}  "
                  f"{r['ms']:.3f} ms/tok  {tag}", flush=True)


if __name__ == "__main__":
    print(f"numba threads = {numba.get_num_threads()}")
    ok = l1()
    l2()
    l3(("fp32", "fp16", "bf16", "fp8", "fp4"))
    print("-" * 76)
    print(f"L1 {'PASS' if ok else 'FAIL'}；L2/L3 数据见上。"
          "默认 readout_dtype=fp32（fhz 指令）；fp64 已停止支持。")
