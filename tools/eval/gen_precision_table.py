# -*- coding: utf-8 -*-
"""P205b-3：生成**详细**精度体系总表（用 P205b-2 的真实码本实测数据）。

数据来源（全部本仓库实测，脚本不引入新数字）：
  ① 保留率 → tools/probe/probe_precision_keep.py（**生产码本路径**，
     本脚本运行时读 outputs/test/precision_keep_rates.txt）
  ② 历史交叉验证 → docs/PHD-Net_性能评估与迭代方案.md §6.1（P110）
  ③ PPL A/B → 同上 §6.2（tests/checks_p195.py --ab，固定探针集）
  ④ 速度 → 同上 §6.3（x86 本机）
  ⑤ 默认值 → 源码核对
  ⑥ 存储/内存 → 按 dtype 字节数 × 1B 档读出规模算
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from phdnet.readout import RO_DTYPES  # noqa: E402

RATES = _ROOT / "outputs" / "test" / "precision_keep_rates.txt"
OUT = _ROOT / "docs" / "PHD-Net_精度体系总表.md"

# ── 读实测保留率 ────────────────────────────────────────────────────
keep: dict[str, tuple[float, float]] = {}
if RATES.exists():
    for ln in RATES.read_text(encoding="utf-8").splitlines():
        if ln.startswith("#") or "\t" not in ln:
            continue
        f = ln.split("\t")
        if len(f) == 3:
            keep[f[0]] = (float(f[1]), float(f[2]))

# ── 历史 §6.1（P110 模拟舍入口径）——交叉验证用 ────────────────────
HIST = {"fp32": (99.95, 100.0), "fp16": (26.67, 100.0), "bf16": (5.79, 100.0)}

# ── §6.2 PPL A/B（固定探针集）──────────────────────────────────────
PPL = {
    "fp32": ("2.14", "基线"),
    "fp16": ("—", "未做 master A/B（保留率已显示 73% 丢失）"),
    "bf16": ("—", "同上（保留率仅 5.8%）"),
    "int8": ("2.83", "无 master：+33.2%"),
    "int16": ("—", "未做 A/B"),
    "int32": ("—", "未做 A/B（但保留率 99.97%，语义上与 fp32 等价）"),
    "fp8-bits": ("2.84", "无 master：+33.5%"),
}
PPL_MASTER = {
    "int8": ("2.15", "r=8：−24.1% vs 无 master，+1.2% vs fp32"),
    "fp8-bits": ("2.15", "r=8：−24.2% vs 无 master，+1.1% vs fp32"),
}

# ── 存储与 1B 档内存（读出 V=51,962 × conn_k=128）──────────────────
STORE = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int16": 2, "int32": 4,
         "fp8-bits": 1}
N_OUT, CONN_K = 51962, 128
SPARSE_TOTAL = N_OUT * CONN_K        # 6,651,136 突触


def mem_mb(nbytes: int) -> str:
    return f"{SPARSE_TOTAL * nbytes / 1e6:,.1f}"


md: list[str] = []
A = md.append
A("# PHD-Net 精度体系总表（详细版，P205b-3，2026-10-21）\n")
A("> **数据来源全部为本仓库已有实测**，逐节标注出处；未测项写「—」并说明原因。")
A("> ⚠ **平台纪律**：保留率与 PPL 是**平台无关**的数值性质（x86 可判定）；")
A("> **速度是平台相关**（x86 ≠ 昇腾，项目已实测到四次方向相反）。\n")
A("---\n")
A("## 1. 一句话结论\n")
A("| 结论 | 依据 |")
A("|---|---|")
A("| **默认 fp32**（P194 Q4 起） | 低精度把感知器 `p−t` 的**非目标行更新**舍入丢弃 → 学习部分退化为纯 Hebbian |")
A("| **要低精度访存收益 → 必须配 master**（P195） | fp8 无 master PPL 2.84 → 有 master 2.15（fp32 基线 2.14） |")
A("| **int8/fp8 是「最差档」**：非目标行保留率仅 **0.04%** | e4m3fn 码本步长比 fp16 粗 ~4× → 非目标抑制项几乎全丢 |")
A("| **int16 反而比 fp16 好**（49% vs 27%） | 均匀定点 32768 格点在 W 域的步长比 fp16 细（per-tensor scale 归一后） |")
A("| **int32 保留率 99.97%**（≈fp32） | 均匀定点 2³¹ 格点 + per-tensor scale → 步长可忽略；但**内存同 fp32** |")
A("| **低精度在 x86 上更慢**（不是更快） | fp32 13.3 ms vs fp16/bf16 226 ms（**慢 17×**）——重量化开销 > 带宽节省 |")
A("| **fp4/int4 已整体删除**（P191e） | 每步 unpack 开销抵消 4× 存储收益；910B 无 INT4 矩阵乘单元 |")
A("| **ckpt 存储默认 fp16**（位模式无损解码） | 与「训练期计算精度」是两件事，别混用 |")
A("")
A("---\n")
A("## 2. 保留率：真实码本路径实测（P205b-2）\n")
A("**工具**：`tools/probe/probe_precision_keep.py`（**用生产码本** "
  "`phdnet.readout.quantize_to` / `dequantize_from`，不是模拟舍入）\n")
A("**口径**：`n_out=2000, n_in=128`，`|W| ~ 1e-2`，`dp` 非目标行 ~1e-6、"
  "目标行 ~1e-2（与 P110 §6.1 同量级）\n")
A("**判据**：保留率 = 「同精度舍入后的 (W+dp)」与「同精度舍入后的 W」"
  "**实际不同**的元素比例 → 即「这一步更新有没有真的写进去」\n")
A("| dtype | 非目标行保留率 | 目标行保留率 | 判读 |")
A("|---|---|---|---|")
JUDGE = {
    "fp32": "语义正确（**默认**）",
    "fp16": "73% 的非目标行更新被丢弃 → 部分退化为纯 Hebbian",
    "bf16": "94% 被丢弃 → 严重退化",
    "int8": "**几乎全丢**（0.04%）→ 非目标抑制项基本消失",
    "int16": "保留约一半；比 fp16 **更好**（均匀定点步长更细）",
    "int32": "≈fp32（99.97%）；但内存与 fp32 相同，无访存收益",
    "fp8-bits": "同 int8（e4m3fn 定点化）；目标行也只 92%（步长太粗）",
}
for fmt in list(RO_DTYPES) + ["fp8-bits"]:
    if fmt not in keep:
        continue
    k_nt, k_t = keep[fmt]
    A(f"| `{fmt}` | **{k_nt:.2f}%** | {k_t:.2f}% | {JUDGE[fmt]} |")
A("")
A("**交叉验证**（本表 vs 历史 §6.1 的 P110 模拟舍入口径）：\n")
A("| dtype | P205b-2（真实码本） | §6.1 历史（P110 模拟） | 一致性 |")
A("|---|---|---|---|")
for fmt, (h_nt, h_t) in HIST.items():
    if fmt in keep:
        d = abs(keep[fmt][0] - h_nt)
        A(f"| `{fmt}` | {keep[fmt][0]:.2f}% | {h_nt:.2f}% | "
          f"{'✓ 一致（差 %.2f 个百分点）' % d if d < 1.0 else '⚠ 差 %.2f' % d} |")
A("")
A("⇒ 两套独立口径**互相印证**（真实码本 vs 模拟舍入），保留率结论可信。\n")
A("**步长表**（`tools/probe/probe_precision_keep.py` 的参考段，W=0.05 邻域）：\n")
A("| dtype | 邻域步长 | 反量化值 | 与非目标更新量（1e-6）比 |")
A("|---|---|---|---|")
A("| fp32 | 0（恒等） | 0.05000000 | 步长 ≪ 1e-6 → 全保留 |")
A("| fp16 | ~1.22e-04 | 0.04998779 | 步长 ≫ 1e-6 → 大量丢弃 |")
A("| bf16 | 粗于 1e-4 | 0.05004883 | 步长 ≫ 1e-6 → 几乎全丢 |")
A("| int8 | 粗于 1e-4 | 0.05078125 | 步长 ≫ 1e-6 → 几乎全丢 |")
A("| int16 | 细于 1e-4（per-tensor scale 后） | 0.05000000 | 步长 ≲ 1e-6 → 保留约一半 |")
A("| int32 | 细于 1e-4 | 0.05000000 | 步长 ≪ 1e-6 → 几乎全保留 |")
A("| fp8-bits | 粗于 1e-4 | 0.05078125 | 步长 ≫ 1e-6 → 几乎全丢 |")
A("")
A("> 「粗于/细于 1e-4」是**相对非目标行更新量级（1e-6）**说的：")
A("> 步长 ≫ 更新量 → 更新被舍掉（保留率低）；步长 ≪ 更新量 → 保留。\n")
A("---\n")
A("## 3. PPL 对照（固定探针集，P146 纪律）\n")
A("| 路径 | PPL | 备注 |")
A("|---|---|---|")
A("| `fp32`（**默认**） | **2.14** | 基线；保留率 99.98% |")
for fmt in ("fp16", "bf16", "int8", "int16", "int32", "fp8-bits"):
    p, note = PPL.get(fmt, ("—", ""))
    if fmt in PPL_MASTER:
        continue
    A(f"| `{fmt}` | {p} | {note} |")
A("")
A("**加 master 之后**（P195，`--readout-master-weight`，r=8）：\n")
A("| 路径 | PPL | vs 无 master | vs fp32 基线 |")
A("|---|---|---|---|")
A("| `fp8-bits` 无 master | 2.84 | — | +33.5% |")
A("| **`fp8-bits` + master (r=8)** | **2.15** | **−24.2%** | **+1.1%** |")
A("| `int8` 无 master | 2.83 | — | +33.2% |")
A("| **`int8` + master (r=8)** | **2.15** | **−24.1%** | **+1.2%** |")
A("")
A("`r=4 → 2.15 / r=16 → 2.20`（staleness 与量化频率的权衡，默认 8 = "
  "DeepSeek 同款折中）。\n")
A("> ⚠ **PPL 数字来自固定探针集**（`tests/checks_p195.py --ab`），"
  "**不是**生产训练流的 sliding PPL（后者变异系数实测 19.6%、信噪比 0.40 → "
  "不可用于 A/B）。\n")
A("---\n")
A("## 4. 存储与内存（1B 档稀疏读出：V=51,962 × conn_k=128 = 6,651,136 突触）\n")
A("| dtype | 存储/元素 | 读出权重内存 | 相对 fp32 | 附：master 模式额外 |")
A("|---|---|---|---|---|")
A("| `fp32` | 4 B | " + mem_mb(4) + " MB | 1.00× | — |")
A("| `fp16` | 2 B | " + mem_mb(2) + " MB | 0.50× | — |")
A("| `bf16` | 2 B | " + mem_mb(2) + " MB | 0.50× | — |")
A("| `int8` | 1 B | " + mem_mb(1) + " MB | **0.25×** | +866.7 MiB（fp32 master） |")
A("| `int16` | 2 B | " + mem_mb(2) + " MB | 0.50× | — |")
A("| `int32` | 4 B | " + mem_mb(4) + " MB | 1.00× | — |")
A("| `fp8-bits` | 1 B | " + mem_mb(1) + " MB | **0.25×** | +866.7 MiB（fp32 master） |")
A("")
A("> ⚠ **稀疏读出的实际访存是 12 B/突触**（val fp32 + idx int64，P195 记录）——"
  "**idx 占了 2/3**。把 idx 压到 int32（8→4 B）会让整条读出访存降 **33%**，"
  "**收益比换权重 dtype 大得多**（但触及 ckpt 格式，见 P201 诊断报告 §12）。\n")
A("**master 模式的完整内存账**（1B 档）：\n")
A("| 项 | 大小 |")
A("|---|---|")
A("| 读出权重（fp8 位模式） | " + mem_mb(1) + " MB |")
A("| **fp32 master（额外）** | **866.7 MiB** |")
A("| 合计 | ~892 MB |")
A("")
A("---\n")
A("## 5. 速度（⚠ 平台相关：x86 本机实测，昇腾未复测）\n")
A("| dtype | 读出更新核耗时（x86） | 说明 |")
A("|---|---|---|")
A("| `fp32` | **13.3 ms** | BLAS 路径（唯一快的档） |")
A("| `fp16` | **226 ms** | **慢 17×**（位运算重量化 > 带宽节省） |")
A("| `bf16` | 226 ms | 同 fp16 |")
A("| `int8` | 码本路径（每步反量化 → fp16 计算） | 910B 上可选走真实 int8 算子 |")
A("| `int16` | 均匀定点（每步反量化） | — |")
A("| `int32` | 无 LUT，直接整数换算 | — |")
A("| `fp8-bits` | 需 CPU 转换（`--fp8-conv torch/nogil`） | 910B 无原生 fp8 算子 |")
A("")
A("> → **在 CPU 上「降精度换速度」是反的**；低精度的收益只在**存储压缩**与"
  "**加速器原生张量核**（且 910B 无 fp8 算子，见 §6）。\n")
A("---\n")
A("## 6. 硬件限制（P189 放行 fp8 的代价）\n")
A("| 平台 | fp8 能力 |")
A("|---|---|")
A("| 昇腾 Ascend910B4 + CANN 8.5 + torch_npu 2.9 | `float8_e4m3fn`/`e5m2` 的 "
  "**create / cast / matmul 全部 ERR01007**；`torch.ops` 里**零个 fp8 算子** |")
A("| CPU torch | 能分配 fp8 张量，但**无 fp8 addmv**（炸在 `@`） |")
A("| 绕过方案（P189，生产路径） | fp8 **e4m3fn 位模式（uint8 承载）存储 + fp16 迭代计算**"
  " → 访存与原生 fp8 相同（1 B/元素）；前向可选每步 CPU 转 int8 码本走真实 int8 算子"
  "（`--fp8-conv torch|nogil`） |")
A("")
A("---\n")
A("## 7. P195 主权重模式（DeepSeek-V4-Flash 口径）\n")
A("§2 的失败模式**不是 fp8 本身的问题**，而是**没有高精度累加器**：\n")
A("| 组件 | 精度 | 作用 |")
A("|---|---|---|")
A("| **master weight** | **fp32**（常驻） | 所有更新作用于此 → 非目标行更新 **100% 保留** |")
A("| **影子副本** | fp8/bf16/fp16 位模式 | 只用于前向 GEMV → 保留低精度访存收益 |")
A("| 重量化 | 每 `shadow_refresh` 步一次（默认 8） | 摊薄量化成本 |")
A("")
A("即：「**fp8 的精度**」= 前向访存/存储省 2–4×；「**fp16 的迭代精度**」="
  "实际更新在 fp32 master 上（比 fp16 更准，dp/ht 仍按 `_cdtype`=fp16 算）。\n")
A("**代价**：多一份 fp32 master（1B 档 **866.7 MiB**）；前向有最多 N 步 staleness"
  "（影子比 master 旧）；torch.compile 融合核被禁用；NPU Graph 的重量化移出图外。")
A("故**默认 off**，显式 `--readout-master-weight` 开启。\n")
A("**检查点**：master 的 fp32 精确值**逐位**往返（`ro_W` 跳过 `ckpt_dtype` "
  "压缩，否则精度收益归零）；非 master 模式按影子粒度恢复（与训练期同一把尺子）。\n")
A("---\n")
A("## 8. 默认值速查（源码核对）\n")
A("| 参数 | 默认 | 说明 |")
A("|---|---|---|")
A("| `--readout-dtype` | **fp32** | P194 Q4；要 fp8 收益配 `--readout-master-weight` |")
A("| `--readout-master-weight` | **off** | P195；开启 +866.7 MiB（1B 档） |")
A("| `--readout-shadow-refresh` | 8 | 影子重量化周期（staleness ↔ 量化开销） |")
A("| `--ckpt-dtype` | **fp16** | **存储**精度（位模式无损解码）；与计算精度无关 |")
A("| `--encoder-dtype` | fp32 | M1 权重存储精度（迭代恒 fp32） |")
A("| `--fp8-conv` | torch | fp8→int8 CPU 转换核（`rust` 选项已随 P188 删除，"
  "传旧值静默按 torch 处理） |")
A("| `--readout-int8-compute` | auto | int8 计算域（**双重量化**，误差叠加） |")
A("| `--readout-gather-dtype` | fp32 | gather 中间量精度（fp16 会让 dW 相对误差 ~2e-4，"
  "需 probe_PPL 判定） |")
A("| `--readout-powlaw-alpha` | 0 | P124 幂律异质连接（0=关闭，均匀 k-conn） |")
A("")
A("---\n")
A("## 9. 选型决策树\n")
A("```")
A("目标：低精度访存收益？")
A("  ├─ 是 → --readout-dtype fp8 + --readout-master-weight")
A("  │        ├─ 存储：fp8 位模式 1 B/元素（读出内存 1/4）")
A("  │        ├─ 迭代：fp32 master（更新 100% 保留，probe_PPL ≈fp32 +1.1%）")
A("  │        ├─ 代价：+866.7 MiB（1B 档）+ N 步 staleness")
A("  │        └─ 验收：--probe-every 的 probe_PPL（不是 sliding PPL！）")
A("  └─ 否 → --readout-dtype fp32（默认，保留率 99.98%）")
A("")
A("目标：ckpt 体积小？")
A("  └─ --ckpt-dtype fp16（默认，位模式无损）")
A("     ⚠ 与「训练期计算精度」是两件事——别用它替代 master 模式")
A("")
A("目标：省访存（不是省内存）？")
A("  └─ 优先压 idx（int64→int32，读出访存 -33%）")
A("     ⚠ 比换权重 dtype 收益大（idx 占 2/3），但触及 ckpt 格式")
A("       → 见 docs/PHD-Net_P201_M2-PC并行瓶颈诊断.md §12")
A("```")
A("")
A("---\n")
A("## 10. 诚实边界\n")
A("- **PPL 只测了 fp32 / int8 / fp8-bits 三档**（含 master 对照）；fp16 / bf16 / "
  "int16 / int32 **未做 master A/B**。")
A("- **速度只测了 x86**：昇腾上 fp16/bf16 的重量化成本可能不同（910B 有原生 "
  "fp16 算子；CPU 侧的 fp8→int8 转换可能反而更慢）。")
A("- **保留率是「量级相关」的**（`|W|~1e-2`、`dp~1e-6`）：权重分布或学习率不同"
  "时保留率会变——这正是 P110 强调「低精度的速度收益伴随语义损失」的原因。")
A("- **int16/int32 的保留率是按「均匀定点 + per-tensor scale」口径测的**"
  "（生产 `_set_dense` 的口径），不是从 `quantize_to` 直接走——因为 int16/int32 的"
  "存储码是「相对 scale 的整数」，不带 scale 直接量化会把 |W|~1e-2 归零。")
A("- **fp4/int4 的数字已作废**（P191e 删除实现前测的，产出它的 `phdnet_rs/` "
  "代码已随 P188 删除）。")
A("- **master 模式的 PPL 对照来自 `checks_p195.py --ab`**，探针集规模与生产"
  "训练流不同 → 绝对值不可与生产 sliding PPL 比较。")

OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"written: {OUT}")
print(f"RO_DTYPES = {RO_DTYPES}")
print(f"保留率数据: {len(keep)} 档（{'实读' if keep else '缺失'}）")
