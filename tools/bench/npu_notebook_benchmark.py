#!/usr/bin/env python3
"""AtomGit NPU Notebook 自包含 benchmark（P217）。

验证两件事：
  1. 把 Vector `Index` 换成 Cube 友好的等价算子能快多少；
  2. Yan「分区激活」（只算部分输出行）能省多少单 token 计算量。

用法（1×NPU 910B / CANN 8.5 / Python 3.11）：
    python npu_notebook_benchmark.py
结果自动写到同目录：
    npu_gather_bench.json   （结构化，供下载）
    npu_gather_bench.txt    （人读）

⚠ 不依赖 PHD-Net 仓库任何模块，可整段独立运行。
⚠ 判数值等价用 max|Δ|（容差级），不要求逐位相同。
⚠ 本机 x86 数字不构成昇腾证据。
"""

from __future__ import annotations

import json
import os
import platform
import time

import torch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import torch_npu  # noqa: F401
    HAS_NPU_LIB = True
except Exception:
    HAS_NPU_LIB = False

NPU_OK = bool(HAS_NPU_LIB and hasattr(torch, "npu") and torch.npu.is_available())
DEV = "npu:0" if NPU_OK else "cpu"

V, K, H = 51962, 128, 3072          # 1b 档真实尺寸
DTS = [torch.float16, torch.float32]
WARMUP, REPEAT = 8, 50


def sync():
    if DEV.startswith("npu"):
        torch.npu.synchronize()
    elif DEV.startswith("cuda"):
        torch.cuda.synchronize()


def bench(name, fn, repeat=REPEAT):
    try:
        for _ in range(WARMUP):
            y = fn()
        sync()
        t0 = time.perf_counter()
        for _ in range(repeat):
            y = fn()
        sync()
        ms = (time.perf_counter() - t0) / repeat * 1000
        return {"name": name, "ok": True, "ms": round(ms, 4), "_y": y}
    except Exception as e:
        return {"name": name, "ok": False, "ms": None,
                "error": f"{type(e).__name__}: {e}"}


def delta(a, b):
    try:
        return float((a.float() - b.float()).abs().max())
    except Exception:
        return None


env = {
    "device": DEV,
    "torch": torch.__version__,
    "has_torch_npu": HAS_NPU_LIB,
    "npu_available": NPU_OK,
    "npu_device_count": int(torch.npu.device_count()) if NPU_OK else 0,
    "npu_name": (str(torch.npu.get_device_name(0)) if NPU_OK else None),
    "python": platform.python_version(),
    "V": V, "K": K, "H": H,
    "elements_per_step": V * K,
    "MiB_per_step_fp16": round(V * K * 2 / 2 ** 20, 2),
    "warmup": WARMUP, "repeat": REPEAT,
}
lines = ["=" * 78, "NPU 读出 gather 等价实现 benchmark (P217)", "=" * 78]
for k, v in env.items():
    lines.append(f"  {k:24s} = {v}")
lines.append("")
print("\n".join(lines))
all_lines = list(lines)

torch.manual_seed(0)
Wi = torch.randint(0, H, (V, K), device=DEV, dtype=torch.long)

results = {"env": env, "runs": []}

for DT in DTS:
    dtname = "fp16" if DT == torch.float16 else "fp32"
    W = torch.randn(V, K, device=DEV, dtype=DT) * 0.1
    ht = torch.randn(H, device=DEV, dtype=DT)

    block = [f"\n--- dtype = {dtname} ---"]
    print("\n".join(block)); all_lines += block

    variants = [
        ("A ht[Wi]*W -> sum(1)  [生产]", lambda: (ht[Wi] * W).sum(dim=1)),
        ("B torch.take(ht,Wi)*W -> sum(1)", lambda: (torch.take(ht, Wi) * W).sum(dim=1)),
        ("C einsum('ij,ij->i',W,ht[Wi])", lambda: torch.einsum("ij,ij->i", W, ht[Wi])),
        ("D index_select(ht,0,Wi.ravel())",
         lambda: (torch.index_select(ht, 0, Wi.reshape(-1)).view(V, K) * W).sum(dim=1)),
    ]
    for Kcand in (512, 2048):
        variants.append((
            f"E 分区激活 只算前 {Kcand} 行",
            (lambda k=Kcand: (ht[Wi[:k]] * W[:k]).sum(dim=1)),
        ))

    VS = 4096
    try:
        S = torch.zeros(VS, H, device=DEV, dtype=DT)
        S.scatter_(1, Wi[:VS], W[:VS])
        variants.append(("F 等价稠密 SpMM  S@ht (V=4096)", lambda: S @ ht))
    except Exception as e:
        line = f"  (SpMM 构造失败: {type(e).__name__}: {e})"
        print(line); all_lines.append(line)

    runs = []
    base = None
    for name, fn in variants:
        r = bench(name, fn)
        y = r.pop("_y", None)
        runs.append(r)
        if r["ok"] and base is None and name.startswith("A "):
            base = y
        if r["ok"] and y is not None and base is not None:
            n = min(base.shape[0], y.shape[0])
            r["max_abs_delta_vs_A"] = delta(base[:n], y[:n])
        line = (f"  {name:38s} " +
                (f"{r['ms']:9.3f} ms/step" if r["ok"] else f"FAILED {r['error']}"))
        if r.get("max_abs_delta_vs_A") is not None:
            line += f"   max|Δ|={r['max_abs_delta_vs_A']:.3e}"
        print(line); all_lines.append(line)

    tA = next((r["ms"] for r in runs if r["name"].startswith("A ") and r["ok"]), None)
    if tA:
        block = [f"\n  相对 A 的加速（{dtname}）:"]
        print("\n".join(block)); all_lines += block
        for r in runs:
            if r["ok"] and r["ms"]:
                r["speedup_vs_A"] = round(tA / r["ms"], 3)
                line = f"    {r['name'][:38]:38s} {r['speedup_vs_A']:6.2f}×"
                print(line); all_lines.append(line)

    results["runs"].append({"dtype": dtname, "variants": runs})

summary = [
    "", "=" * 78, "判读要点",
    "  1. B/C/D 若明显快且 max|Δ|=0 → 算子层面等价；生产换默认仍需 A/B",
    "     （`--sp-gather take` / `--sparse-fwd-kernel einsum` 开关已存在）。",
    "  2. E 是「分区激活」的理论上限；真用要改学习规则，必须 probe_PPL 判。",
    "  3. F 是等价 SpMM 的算子信号；若 Cube 明显快，值得做真 CSR SpMM。",
    "  4. ⚠ 别拿 x86 数字下结论；P110 教训：少算行会让非目标行更新",
    "     保留率暴跌（fp32 99.95% → bf16 5.79%）。",
    "=" * 78,
]
print("\n".join(summary)); all_lines += summary

jpath = os.path.join(OUT_DIR, "npu_gather_bench.json")
tpath = os.path.join(OUT_DIR, "npu_gather_bench.txt")
with open(jpath, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
with open(tpath, "w", encoding="utf-8") as f:
    f.write("\n".join(all_lines) + "\n")

print(f"\n[结果已保存] {jpath}")
print(f"[结果已保存] {tpath}")
