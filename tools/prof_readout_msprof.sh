#!/usr/bin/env bash
# P141：用 CANN msprof 采集读出profile —— **唯一**能回答
#「设备 9.2~14 ms 在执行什么」的工具。
#
# 为什么必须用它（外部测量已到极限）
#================================================================================
# P135~P140 的全部外部测量把读出拆到了这个程度（昇腾 1b 档，n_out=52,642/k=128）：
#   ①~⑦ **裸算子**合计  ≈ 0.35 ms   （gather 0.020 / einsum 0.155 /
#                                更新 0.061 / CE 0.104 / H2D 0.038 …）
#   ⑧b 真实一步(sync)     15.2 ms     ← 与训练日志 10.99 ms 对得上
#   逐行打点：14.25 ms 全在 `return float(nll_dev.item())`
#   sync=8 联立：设备**单步真实执行 ≈ 9.2 ms**（cap=4）~ 14.1 ms（cap=16）
#   `[diag]`：`_staged_to_dev` 脱离循环只要 **0.09 ms**，嵌在循环里要 **4.94 ms**
#            → 同一函数差 54× → 那4.94 ms 是**等 NPU 消费完 pinned buffer**
#            → pin_cap 调大只是把等待「攒少一点、攒多一点」，总墙钟不变
#   → **裸算子 0.35 ms vs 设备 9.2 ms，差 26×。**
#
# ⚠ 这 26 倍**在 Python 侧无法解释**：逐行打点显示除 `.item()` 外所有行
#   都 < 0.2 ms。所以设备上确实在执行**我们没数到的kernel**。
#   可能性（本工具会验证）：
#     (a) 隐式的 dtype 转换 / `to(device)` / `contiguous` 等每步算子
#     (b) 内存分配（每步 25 MiB 临时张量的 alloc/free）
#     (c) H2D 相关的搬运算子（我们只测了 12 KiB 的 h，没测W/idx 是否被搬）
#     (d) CANN 的隐式同步点（每步都有 `.item()`）
#   **msprof 是唯一能看到「设备上真实执行了什么」的工具。**
#
# 用法
#================================================================================
#   bash tools/prof_readout_msprof.sh
#   # 1) 用 msprof 跑一个**极短**训练（profiler 会让程序慢 10~100×）
#   # 2) 打印 top 算子 + 与「我们已知的 8 个」做差集
#
# ⚠ **必须在昇腾上跑**（需要 msprof 与 CANN 环境）。
set -u
OUT="${OUT:-outputs/prof/msprof_readout}"
STEPS="${STEPS:-600}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "============================================================"
echo "P141 msprof 采集（读出 9.2 ms 到底在执行什么）"
echo "============================================================"
command -v msprof >/dev/null 2>&1 || { echo "✗ msprof 不在 PATH —— 请在昇腾环境跑"; exit 2; }

# 关键：**步数必须极少**。profiler 开销极大（本项目 1 档token 10.99 ms，
# 插桩后可能变成 200~1000 ms），跑多了没意义且会占满磁盘。
echo "[prof] 采集目录 $OUT，训练步数=$STEPS（故意极小）"
rm -rf "$OUT"; mkdir -p "$OUT"

# 只对读出做 profile：MSPROF_FILTER 可限定关注范围，但 msprof 的过滤语法
# 在不同 CANN 版本差异大，先全采（步数极少，体积可控）。
MSPROF_OUTPUT="$OUT" \
timeout 1800 msprof \
  --application="python3 train/train.py --preset 1b --data pretrain \
      --remote-data --step-profiling --lang zh --tokens $STEPS" \
  --output="$OUT" 2>&1 | tail -20

echo
echo "---- 采集完成，检查产物 ----"
find "$OUT" -maxdepth 3 -type f \( -name "*.json" -o -name "*.csv" -o -name "*.txt" \) \
  2>/dev/null | head -20
echo
echo "下一步：用 msprof 的导出工具把 timeline 转成可分析格式"
echo "  msprof --analyse --output=$OUT/timeline $OUT   （版本相关，见 msprof --help）"
echo
echo "若要找「设备 top 算子」，最快的方式是直接看 timeline CSV："
echo "  find $OUT -name '*.csv' -exec head -2 {} \;   # 看列名"
echo "  然后按 Device 侧的 Duration 求 top-N，与我们已知的 8 个算子做差集。"
echo "我们已知的（来自 P135 的裸算子测量，都远小于 9.2ms）："
echo "  h[Wi] 0.020 | index_select 0.025 | take 0.016 |"
echo "  einsum 0.142 | npu_gather_sparse_index 0.034 |"
echo "  addcmul_/addmm_ 0.061 | cross_entropy 0.104 | H2D h 0.038"
