# -*- coding: utf-8 -*-
"""P205b：生成精度体系总表（读出各精度的实测保留率 + PPL + 存储 + 速度）。

数据来源全部是**本仓库已有的实测**（不在此处编造）：
  §6.1 保留率  → docs/PHD-Net_性能评估与迭代方案.md:561-578（P110 实测）
  §6.2 PPL A/B → 同上 :594-603（checks_p195.py --ab，固定探针集）
  §6.3 速度    → 同上 :652（x86 本机读出更新核实测）
  RO_DTYPES    → phdnet/readout.py:383
  master 语义  → phdnet/backends/accel_readout.py:890-921

输出：docs/PHD-Net_精度体系总表.md
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from phdnet.readout import RO_DTYPES  # noqa: E402

OUT = _ROOT / "docs" / "PHD-Net_精度体系总表.md"

# ── §6.1 实测保留率（P110，x86 本机，|W|~1e-2、|dp|~1e-6）────────────
KEEP = {
    "fp32": (99.95, "语义正确（P194 起为默认）"),
    "fp16": (26.67, "73% 的非目标行更新被丢弃"),
    "bf16": (5.79, "94% 被丢弃"),
    "int8": (None, "e4m3fn 定点化码本；步长与 fp8 同级（比 fp16 粗 ~4×）"),
    "int16": (None, "均匀定点 16 位（格点 65536）；步长比 fp16 细"),
    "int32": (None, "均匀定点 32 位（无 LUT，17 GB 表不建）；步长最细"),
}

# ── §6.2 PPL A/B（固定探针集，P146 纪律）──────────────────────────────
PPL = {
    "fp32": ("2.14", "基线"),
    "fp16": ("—", "未做 master A/B（低精度下 §6.1 的丢失显著）"),
    "bf16": ("—", "同上"),
    "int8": ("2.83", "无 master：+33.2%"),
    "int8+master": ("2.15", "r=8：−24.1% vs 无 master，+1.2% vs fp32"),
    "fp8": ("2.84", "无 master：+33.5%"),
    "fp8+master": ("2.15", "r=8：−24.2% vs 无 master，+1.1% vs fp32"),
}

# ── 存储（1B 档读出 V=51,962 × conn_k=128）─────────────────────────
def mem_mb(nbytes: int) -> str:
    return f"{51962 * 128 * nbytes / 1e6:,.1f}"


STORE = {
    "fp32": ("4", mem_mb(4)),
    "fp16": ("2", mem_mb(2)),
    "bf16": ("2", mem_mb(2)),
    "int8": ("1", mem_mb(1)),
    "int16": ("2", mem_mb(2)),
    "int32": ("4", mem_mb(4)),
}

# ── §6.3 速度（x86 本机读出更新核）──────────────────────────────────
SPEED = {
    "fp32": "13.3 ms（BLAS 路径）",
    "fp16": "226 ms（**慢 17×**）",
    "bf16": "226 ms（同 fp16）",
    "int8": "码本路径（每步反量化 → fp16 计算）",
    "int16": "均匀定点（每步反量化）",
    "int32": "无 LUT，直接整数换算",
}

md = []
A = md.append
A("# PHD-Net 精度体系总表（数据截止 2026-10-20，P205b）\n")
A("> **本表所有数字都是本仓库已有实测**，出处逐行标注；未测项写「—」并说明原因。")
A("> ⚠ **平台纪律**：保留率与 PPL 是**平台无关**的数值性质（可在 x86 判定）；")
A("> **速度是平台相关**（x86 ≠ 昇腾，项目已实测到四次方向相反）。\n")
A("## 1. 一句话结论\n")
A("| 结论 | 依据 |")
A("|---|---|")
A("| **默认用 fp32**（P194 Q4 起） | 低精度把感知器 `p−t` 的**非目标行更新**舍入丢弃 → 学习退化为纯 Hebbian |")
A("| **想要低精度访存收益 → 必须配 master**（P195） | fp8 无 master PPL 2.84 → 有 master 2.15（fp32 基线 2.14） |")
A("| **低精度在 x86 上更慢**（不是更快） | fp32 13.3 ms vs fp16/bf16 226 ms（**慢 17×**）——位运算重量化 > 带宽节省 |")
A("| **fp4/int4 已整体删除**（P191e） | 每步 unpack 开销抵消 4× 存储收益；910B 无 INT4 矩阵乘单元 |")
A("| **ckpt 存储默认 fp16**（位模式，无损解码） | 与训练期计算精度是两件事 |")
A("")
A("## 2. 读出精度矩阵（`RO_DTYPES` 现存六档，`phdnet/readout.py:383`）\n")
A("| dtype | 非目标行保留率 | 目标行保留率 | 存储/元素 | 1B 档读出内存 | PPL（固定探针集） | 判读 |")
A("|---|---|---|---|---|---|---|")
rows = [
    ("fp32", "**99.95%**", "100%", "语义正确；无 master 需求（**默认**）"),
    ("fp16", "26.67%", "100%", "73% 的非目标行更新被丢弃 → 部分退化为纯 Hebbian"),
    ("bf16", "5.79%", "100%", "94% 被丢弃 → 严重退化"),
    ("int8", "更低（e4m3fn 码本，步长比 fp16 粗 ~4×）", "100%",
     "P100 起的 e4m3fn 定点化；正确姿势 = 配 master"),
    ("int16", "（均匀定点 16 位，步长比 fp16 细；未单独实测）", "100%",
     "65536 格点；P101"),
    ("int32", "（均匀定点 32 位，步长最细；未单独实测）", "100%",
     "无 LUT（17 GB 表不建）→ 直接整数换算"),
]
for name, keep, tgt, note in rows:
    b, mb = STORE[name]
    ppl, ppl_note = PPL.get(name, ("—", "未做 master A/B"))
    A(f"| `{name}` | {keep} | {tgt} | {b} B | {mb} MB | {ppl} | {note} |")
A("")
A("**fp8 位模式**（P189 引入，**不在 `RO_DTYPES` 里**——它是「uint8 承载的 "
  "e4m3fn 位模式」+ fp16 迭代计算的组合路径，910B 无原生 fp8 算子）：\n")
A("| 路径 | 非目标行保留率 | PPL（固定探针集） | vs fp32 基线 |")
A("|---|---|---|---|")
A("| fp8 位模式，无 master | 更低（步长比 fp16 粗 ~4×） | 2.84 | +33.5% |")
A("| **fp8 位模式 + master (r=8)** | **100%**（更新落 fp32 master） | **2.15** | **+1.1%** |")
A("| int8 码本，无 master | 同上 | 2.83 | +33.2% |")
A("| **int8 码本 + master (r=8)** | **100%** | **2.15** | **+1.2%** |")
A("")
A("**来源**：保留率 = §6.1（`tools/probe/probe_readout_precision.py`，P110 实测，"
  "判据=同精度舍入后元素是否真的变了，量级 `|W|~1e-2`、`|dp|~1e-6`）；"
  "PPL = §6.2（`tests/checks_p195.py --ab`）；"
  "int16/int32 的保留率**未单独实测**（P101 只测了与稠密版的逐位等价）。\n")
A("## 3. P195 主权重模式（DeepSeek-V4-Flash 口径）\n")
A("§6.1 的失败模式**不是 fp8 本身的问题**，而是**没有高精度累加器**：\n")
A("| 组件 | 精度 | 作用 |")
A("|---|---|---|")
A("| **master weight** | **fp32**（常驻） | 所有更新作用于此 → 非目标行更新 **100% 保留** |")
A("| **影子副本** | fp8/bf16/fp16 位模式 | 只用于前向 GEMV → 保留低精度访存收益 |")
A("| 重量化 | 每 `shadow_refresh` 步一次（默认 8） | 摊薄量化成本 |")
A("")
A("**本机 A/B**（固定探针集，P146 纪律）：\n")
A("| 配置 | PPL | vs 无 master | vs fp32 基线 |")
A("|---|---|---|---|")
A("| accel fp8 无 master | 2.84 | — | +33.5% |")
A("| **accel fp8 + master (r=8)** | **2.15** | **−24.2%** | **+1.1%** |")
A("| numba int8(=e4m3) 无 master | 2.83 | — | +33.2% |")
A("| **numba int8 + master (r=8)** | **2.15** | **−24.1%** | **+1.2%** |")
A("")
A("`r=4 → 2.15 / r=16 → 2.20`（staleness 与量化频率的权衡，默认 8 = DeepSeek 同款折中）。\n")
A("**代价**：多一份 fp32 master（1B 档 **866.7 MiB**）；前向有最多 N 步 staleness；"
  "torch.compile 融合核被禁用；NPU Graph 的重量化移出图外。"
  "故**默认 off**，显式 `--readout-master-weight` 开启。\n")
A("**检查点**：master 的 fp32 精确值**逐位**往返（`ro_W` 跳过 `ckpt_dtype` 压缩，"
  "否则精度收益归零）；非 master 模式按影子粒度恢复。\n")
A("## 4. 速度（⚠ 平台相关：x86 本机实测，昇腾未复测）\n")
A("| dtype | 读出更新核耗时（x86） |")
A("|---|---|")
for k in ("fp32", "fp16", "bf16", "int8", "int16", "int32"):
    A(f"| `{k}` | {SPEED[k]} |")
A("")
A("> **x86 上低精度比 fp32 慢 17×** —— 死因是**位运算重量化 > 带宽节省**。")
A("> → 「降精度换速度」在 CPU 上是**反的**；低精度的收益只在**存储压缩**与"
  "**加速器原生张量核**（且 910B 无 fp8 算子，见 §6.2）。\n")
A("## 5. 硬件限制（P189 放行 fp8 的代价）\n")
A("| 平台 | fp8 能力 |")
A("|---|---|")
A("| 昇腾 Ascend910B4 + CANN 8.5 + torch_npu 2.9 | `float8_e4m3fn`/`e5m2` 的 "
  "**create / cast / matmul 全部 ERR01007**；`torch.ops` 里**零个 fp8 算子** |")
A("| CPU torch | 能分配 fp8 张量，但**无 fp8 addmv**（炸在 `@`） |")
A("| 绕过方案（P189） | fp8 **e4m3fn 位模式（uint8 承载）存储 + fp16 迭代计算** "
  "→ 访存与原生 fp8 相同（1 B/元素）；前向可选每步 CPU 转 int8 码本走真实 "
  "int8 算子（`--fp8-conv`） |")
A("")
A("## 6. 默认值速查（源码核对）\n")
A("| 参数 | 默认 | 说明 |")
A("|---|---|---|")
A("| `--readout-dtype` | **fp32** | P194 Q4；要 fp8 收益配 `--readout-master-weight` |")
A("| `--readout-master-weight` | **off** | P195；开启 +866.7 MiB（1B 档） |")
A("| `--readout-shadow-refresh` | 8 | 影子重量化周期（staleness ↔ 量化开销） |")
A("| `--ckpt-dtype` | **fp16** | **存储**精度（位模式，无损解码）；与计算精度无关 |")
A("| `--encoder-dtype` | fp32 | M1 权重存储精度（迭代恒 fp32） |")
A("| `--fp8-conv` | torch | fp8→int8 CPU 转换核（`rust` 选项已随 P188 删除，"
  "传旧值静默按 torch 处理） |")
A("| `--readout-int8-compute` | auto | int8 计算域（**双重量化**，误差叠加，默认关） |")
A("| `--readout-gather-dtype` | fp32 | gather 中间量精度（fp16 会让 dW 相对误差 "
  "~2e-4，需 probe_PPL 判定） |")
A("")
A("## 7. 选型决策树\n")
A("```")
A("要低精度访存收益？")
A("  ├─ 是 → 配 --readout-master-weight（P195）")
A("  │        └─ 存储 fp8/bf16/fp16 位模式 + fp32 master 迭代")
A("  │        └─ 验收：--probe-every 的 probe_PPL（≈fp32 基线 +1.1%）")
A("  │        └─ 代价：+866.7 MiB / 1B 档 + N 步 staleness")
A("  └─ 否 → --readout-dtype fp32（默认）")
A("         └─ 保留率 99.95%，语义正确，无需额外内存")
A("")
A("要 ckpt 体积小？")
A("  └─ --ckpt-dtype fp16（默认，位模式无损）")
A("     ⚠ 与「训练期计算精度」是两件事——别用它替代 master 模式")
A("```")
A("")
A("## 8. 诚实边界\n")
A("- **int16 / int32 的保留率未单独实测**（表里标「—」）：P101 只验了与稠密版的"
  "**逐位等价**（码本路径正确），未测「低精度存储下的更新保留率」。")
A("- **速度只测了 x86**：昇腾上 fp16/bf16 的重量化成本可能不同（910B 有原生 "
  "fp16 算子，CPU 侧的 fp8→int8 转换可能反而更慢）。")
A("- **PPL 数字来自固定探针集**（`tests/checks_p195.py --ab`），不是生产训练流的 "
  "sliding PPL（后者的变异系数实测 19.6%，信噪比 0.40 → 不可用于 A/B）。")
A("- **fp4/int4 的数字已作废**（P191e 删除实现前测的，产出它的 `phdnet_rs/` "
  "代码已随 P188 删除）。")

OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"written: {OUT}")
print(f"RO_DTYPES = {RO_DTYPES}")
