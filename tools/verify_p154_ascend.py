"""P158：核查 fhz 的昇腾探测结果对 P154 实现的影响（本机x86 侧静态审查）。

fhz 2026-10-03 12:52 的探测结论（Ascend910B4 / torch 2.9.0+cpu / torch_npu 2.9.0）：
  - fp8 e4m3/e5m2：**create@npu / roundtrip@npu / matmul@npu 全部 ERR01007**
  - fp8 在 CPU 上 create/roundtrip **ok**，但 matmul **NotImplementedError**
  - npu_quant_* 系列：**全部因参数签名不符而报错**（不是「不支持」）
  - npu_anti_quant：`Input x must be Int8 or Int32`
  - npu_format_cast：三dtype 全ok
  - accel_doctor：设备可用，0.13 ms/token（vs numba 生产基线 158.30 → 1184×）

**关键推论**（本脚本逐条核对代码）：
  ① fp8 在 NPU 上**连create 都不行**→ `self.W` 永远不会真是 fp8 dtype
     → P154 我写的 `if self._int8_compute and self.W.dtype == torch.float8_e4m3fn`
       这条**更新分支在昇腾上永远不进**（dead code，不是 bug，但收益也没了）
  ②降级链：fp8 不可用 → 落 **int8 存储** → 所以昇腾上实际是
     **int8 存储 + int8 计算**，而**不是** fhz 要的「fp8 存储 + int8 计算」
  ③ 我加的 `_fp8_to_int8_cpu`（CPU 位转换）在昇腾上**永远不会被调用**
     → CPU↔NPU 同步拷贝的担忧在 fp8 不可用时**不存在**
  ④ 真正要在昇腾上验的是 **int8 存储 + int8 计算**这条路，即 P153 的auto
"""
import re

SRC = "phdnet/backends/accel_readout.py"

print("=" * 72)
print("P158：昇腾探测结果 × P154 实现的静态核查")
print("=" * 72)

with open(SRC, encoding="utf-8") as f:
    src = f.read()
lines = src.split("\n")

print()
print("【①】fp8 更新分支是否会在昇腾生效？")
print("     条件：self.W.dtype == torch.float8_e4m3fn")
print("     昇腾实况：fp8 **create 就ERR01007** → W 永远不是 fp8")
print("     → 该分支是 **dead code**（无害，但 P154 的收益在昇腾上不存在）")
hits = [i + 1 for i, l in enumerate(lines)
        if "float8_e4m3fn" in l and "_int8_compute" in l]
print("     位置：", hits or "无")

print()
print("【②】降级链会把fp8 落到什么？")
# 找降级顺序
for i, l in enumerate(lines):
    if "_unsupported" in l or "降级" in l or "fallback" in l.lower():
        if "fp8" in l or "int8" in l:
            print("     L%d: %s" % (i + 1, l.strip()[:88]))

print()
print("【③】_fp8_to_int8_cpu 在昇腾上会被调用吗？")
c1 = src.count("_fp8_to_int8_cpu")
print("     调用点%d 处；条件是 W.dtype==fp8 → 昇腾恒不成立 → **0 次调用**" % c1)

print()
print("【④】那昇腾上真正跑的是哪条路？")
print("     fp8 不可用 → **int8 存储** + int8 计算（P153 auto 已默认开）")
print("     → 要测的是「int8 存储 + int8 计算」，不是「fp8 存储 + int8 计算」")

print()
print("=" * 72)
print("结论：fhz 的指令在 910B 上**无法按字面实现**")
print("=" * 72)
print("""
  fhz 指令：「fp8不行就默认降级到 int8 计算，**fp8 存储**」
  实测：910B 上 fp8 **完全不可用**（连张量都建不出来）
  → 「fp8 存储」这半句在 910B 上**物理上不成立**

  ⚠ 需fhz 拍板：是否接受 **int8 存储 + int8 计算** 作为 910B 的落地形态？
     （它与「fp8 存储 + int8 计算」的**访存量完全相同**，都是 1 字节/元素）
""")