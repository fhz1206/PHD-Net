#!/usr/bin/env python3
"""NPU notebook 自包含 benchmark：读出前向 gather 的等价实现对比。

## 为什么有这个脚本

P214 msprof 实测（昇腾 1b 档）：`aclnnIndex_IndexAiCore_Index` 占 NPU
Task Duration 的 **80.7%**（21,245 次、均 6.5 ms），跑在 Vector 单元；
同机 `BatchMatMul` 在 Cube 上快 4.7 倍。前向是

    y[i] = Σ_j W[i,j] · ht[Wi[i,j]]      Wi: (V, k), W: (V, k), ht: (H,)

它数学上等价于稀疏矩阵乘 `S @ ht`（S ∈ R^(V×H)，每行 k 个非零）。
本脚本在**真实 NPU**上对比几种实现的耗时与数值差异，回答：
「把 Vector gather 换成 Cube 友好的等价算子，到底能快多少」。

## 用法

**自包含，不依赖 PHD-Net 仓库** —— 直接整段粘贴进 AtomGit Notebook
（1 × NPU 910B / CANN 8.5 / Python 3.11）的一个 cell 运行即可。

    exec(open('bench_readout_gather_npu.py').read())   # 若已上传文件
    # 或直接粘贴全文后运行

⚠ 本机 x86 结果**不构成昇腾证据**（项目已有四次方向相反）。
⚠ 判数值等价用 **max|Δ|**（容差级即可），不要要求逐位相同：
   累加顺序不同会有 fp16/fp32 的 ulp 级差异。
"""

import time
import torch

DEV = "npu:0" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
print(f"device = {DEV} | torch {torch.__version__}")

# 1b 档真实尺寸（phdnet/config.py + 生产日志）
V, K, H = 51962, 128, 3072
DT = torch.float16          # 生产读出计算 dtype

torch.manual_seed(0)
Wi = torch.randint(0, H, (V, K), device=DEV, dtype=torch.long)
W = torch.randn(V, K, device=DEV, dtype=DT) * 0.1
ht = torch.randn(H, device=DEV, dtype=DT)


def sync():
    if DEV.startswith("npu"):
        torch.npu.synchronize()
    elif DEV.startswith("cuda"):
        torch.cuda.synchronize()


def bench(name, fn, n=50, warmup=8):
    for _ in range(warmup):
        y = fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        y = fn()
    sync()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"  {name:34s} {ms:8.3f} ms/step")
    return y, ms


print(f"\n规模 V={V} k={K} H={H} | ht[Wi] = {V*K:,} 元素 "
      f"= {V*K*2/2**20:.1f} MiB (fp16)\n")

# ── A：当前生产实现（Vector Index + mulsum） ──────────────────────
yA, tA = bench("A  ht[Wi] * W  → sum(1)  [生产]",
               lambda: (ht[Wi] * W).sum(dim=1))

# ── B：torch.take（P134 已有的另一条 gather kernel） ───────────────
yB, tB = bench("B  torch.take(ht, Wi) * W → sum(1)",
               lambda: (torch.take(ht, Wi) * W).sum(dim=1))

# ── C：einsum（不物化 (V,k) 中间张量） ────────────────────────────
yC, tC = bench("C  einsum('ij,ij->i', W, ht[Wi])",
               lambda: torch.einsum("ij,ij->i", W, ht[Wi]))

# ── D：等价稠密 SpMM：S @ ht（S 每行 k 个非零） ────────────────────
# 只在缩小版上测：构造完整 (V,H) 稠密矩阵要 51962*3072*2 = 304 MiB，
# 本身就不公平；这里给出「算子是否可用 + 数值是否等价」的信号。
VS = 4096
S = torch.zeros(VS, H, device=DEV, dtype=DT)
src = torch.arange(VS, device=DEV)[:, None]
S.scatter_(1, Wi[:VS], W[:VS])


def spmm_dense():
    return S @ ht


yD, tD = bench(f"D  S@ht (稠密, V={VS}, 仅算子信号)", spmm_dense, n=30)

# ── 数值一致性 ────────────────────────────────────────────────────
def delta(a, b):
    return float((a.float() - b.float()).abs().max())

print("\n数值差异 (max|Δ|, fp16 容差级):")
print(f"  A vs B (take)     : {delta(yA, yB):.3e}")
print(f"  A vs C (einsum)   : {delta(yA, yC):.3e}")
print(f"  A vs D (S@ht,前{VS}行): {delta(yA[:VS], yD):.3e}")

print("\n相对 A 的加速:")
for nm, t in (("B take", tB), ("C einsum", tC)):
    print(f"  {nm:12s} {tA / t:6.2f}×")

print("\n" + "=" * 70)
print("判读要点:")
print("  1. 若 B/C 明显快且 max|Δ| 在 1e-3 量级内 → 换算子即可，零语义风险")
print("     （`--sp-gather take` / `--sparse-fwd-kernel einsum` 开关已经存在）。")
print("  2. 若都差不多 → 瓶颈是 Vector 单元本身，必须走真 SpMM / 自定义算子")
print("     （`S @ ht`，Wi 静态可预构建 CSR 常驻设备）。")
print("  3. ⚠ 别用本机 x86 数字下结论：项目已有四次 x86/昇腾方向相反。")
print("  4. 任何要改学习规则的方案（分区激活 / 批内延迟更新）必须 probe_PPL 判。")
print("=" * 70)
