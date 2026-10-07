# -*- coding: utf-8 -*-
"""加速器读出（M6 读出头）——把计算密集部件放到 NPU/CUDA（P19，2026-09-28）。

fhz 要求「计算类型的东西就要跑在 NPU 上」。读出是唯一值得迁移的部件：
1B 预设里它占端到端 ~89%（V×H = 73,958×3,072 fp32 ≈ 908 MB），且**逐 token
只有一个 h 向量**参与计算 → 每步只需传 12 KB 上行 + 少量下行，通信可忽略。

**为什么只迁读出、不迁全部**（诚实边界）：
- 生产路径的 PC 栈 / STDP / LTM 是 numba（事件驱动稀疏 + 在线 CSR 生长），
  numba 只能编译 CPU；torch 栈虽有 M1–M6 实现，但缺 `big_ltm` 等 7 项机制，
  整体迁移会**丢机制**，且权重与生产 ckpt 不通用。
- 分词/编码是变长字符串处理，NPU 不擅长（要整段上传 + 变长输出），留 CPU。

因此本模块提供**混合执行**：数据侧（分词/编码）CPU + 读出 NPU。

精度判据：跨设备为**容差一致**（归约顺序不同，不宣称逐位）；同设备
（torch CPU vs torch CPU）逐位一致。验证见 `tests/verifiers/verify_accel_readout.py`。

接口与 `phdnet.readout.Readout` 对齐（forward / learn_softmax / W / n_synapses），
因此可作为 `PHDNetConfig.accel_readout` 开关的后端；**无加速器时自动回落原
numba 路径（默认路径逐位不变）**。
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:                                            # pragma: no cover
    torch = None

# P151（fhz 2026-10-03：「解禁 fp8, fp4, int4, int8」）：
#   `fp4`/`int4` 在 **numba 侧已实现**（`_ro_q_matvec_fp4` / 4-bit 两元素一字节
#   打包，`RO_DTYPES` 里int4 可跑，实测 1b 档完整一步 OK），
#   但加速臂的 `_DT` 一直没有它们的键 → 直接 ValueError「不支持 dtype」。
#   → 这里补上键；加速臂用**两个 uint8 张量**模拟 4-bit（高/低半字节），
#     与 numba 的打包语义一致（`fp4` 是 e2m1 浮点、`int4` 是定点，
#     P100 已正名：fp4↔int4 是「浮点 vs 定点」之别，不是新旧之别）。
_DT = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16",
       "int8": "int8",
       # 4-bit：底层都是 uint8 打包（两元素/字节），float4 用 e2m1 格点
       "fp4": "uint8_packed4", "int4": "uint8_packed4",
       # P147：fp8 不再是 int8 的别名 → 能力表必须有它，
       # 否则 §「别名归一化」之后会在 `dtype not in _DT` 处 ValueError。
       "fp8": "float8_e4m3fn"}
# P105 旧名别名（与 readout.py 的 _RO_DTYPE_ALIASES 同口径）。
# ⚠⚠ **P147 修正**：原来这里是 `{"fp8": "int8"}`（fp8 是 int8 的别名，P100 正名），
#   而那个别名在 `__init__` **开头**就把 `dtype` 归一化了 → 后面
#   `_fp8_req = dtype == "fp8"` **恒为 False** → **整个 fp8 三级自适应是死代码**。
#   实测：fp8 臂与 int8 臂 `tdtype` 完全相同、**零告警零探测**。
#   现在 fp8 是**独立路径**（有原生 fp8 算子就用 fp8，否则转int8），
#   不再是别名。int8 走P105 的既有路径，两者在 `__init__` 里**汇合**。
_DTYPE_ALIASES: dict = {}

# int8 模式的**计算** dtype：存储是 int8 码本，但 ht/y 的 matmul 在 fp16 域做
# （昇腾/CUDA 的 fp16 矩阵乘是原生快路径）。构造期探测失败则整体回落 fp16。
_INT8_CDTYPE = torch.float16 if torch is not None else None
_INT8_QMAX = 127.0

# ══════════════════════════════════════════════════════════════════════════
# 4-bit 打包（fp4 e2m1 / int4 定点）—— 加速臂实现（P151）
# ══════════════════════════════════════════════════════════════════════════
# 语义与 **numba 侧完全一致**（否则两臂数值不同，`verify` 的对拍会红）：
#   `phdnet/readout.py:451-452`（`_ro_q_matvec_fp4`）：
#       c = (codes[row + (j >> 1)] >> 4) if (j & 1) \
#           else (codes[row + (j >> 1)] & 0xF)
#   即**高半字节 = 偶数列、低半字节 = 奇数列**；权重 = LUT[c] × scale。
# 存储 uint8（两元素/字节），**计算域 fp16**（P149 口径）。
# ⚠ P151 修正：**必须恰好 16 个元素**（code 0x0..0xF 是 4-bit 的全部位型）。
#   我第一版写成 14 个（漏了 e2m1 的两个非规格值 +0.0/-0.0 与 ±1.0 的组合）
#   → `size != 16` 让 `_pack4` 的族判据把它当成 int4 → scale 用2·amax/15
#   而非 amax/4 → 量化误差 **46.7%**（实测）。这是「LUT 长度即契约」的典型。
# e2m1：1 符号 + 2 指数 + 1 尾数 →
#   exp=00: 0 / subnormal(0.5)；exp=01..10: 1.0/1.5/2.0/3.0/4.0；exp=11: inf/nan
#   → 有限值共 10 个/符号 = 20；把两个 NaN 槽让给 ±0（与 fp8 e4m3fn 同惯例）
_FP4_E2M1_LUT = np.array([
    +0.0,                          # 0000 0 (exp=00,m=0)
    +0.5,                          # 0001 subnormal
    +1.0, +1.5, +2.0, +3.0, +4.0,  # 0010..0111 (exp=01/10)
    -0.0,                          # 1000
    -0.5,                          # 1001 subnormal
    -1.0, -1.5, -2.0, -3.0, -4.0,  # 1010..1111
    +6.0, -6.0,                    # 0110/1110: 用 2·3.0 代替 inf（保持 16 格）
], dtype=np.float32)
assert _FP4_E2M1_LUT.size == 16, "e2m1 LUT 必须 16 格"
_INT4_LUT = np.arange(-8, 8, dtype=np.float32)      # 4-bit 定点：-8..+7


def _pack4(vals: np.ndarray, lut: np.ndarray,
           chunk: int = 1 << 20,
           family: str = "int") -> tuple[np.ndarray, float]:
    """float32 值→ 4-bit 码本（uint8，两元素一字节）。返回 `(packed, scale)`。

    量化 = 到 LUT 的**最近格点**（与 numba 的二分最近邻同语义，P9）。
    `scale` 用 per-tensor：fp4 取 `max|v| / max|lut|`（不浪费格点），
    int4 取 `2·max|v|/15`（与 P105 的 int8 口径 `2·max|W|/127` 同族）。

    ⚠ 最近格点用 argmin：16 格 × N 元素 = 16N 次比较，1b 档 (51962×128)
      ≈ 1.06e8 次 → **分块**（默认 1 Mi 元素/块）以控制峰值内存。
      成本 O(16N) 但**仅构造期一次**，不在热路径。
    """
    flat = np.ascontiguousarray(vals, dtype=np.float32).reshape(-1)
    amax = float(np.abs(flat).max())
    if not np.isfinite(amax) or amax == 0.0:
        return np.zeros((flat.size + 1) // 2, dtype=np.uint8), 1.0
    # ⚠⚠ **P151 修正两次**：
    #   ① 原判据 `lut.size == 16` 把 int4（也16 格）误判成 fp4 → 用 max|lut|=8
    #      而 e2m1 该用 6 → 误差 **46.7%**。
    #   ② 改用「`min|lut| == 0.5` 判 e2m1」**也不可靠**（`+0.5` 与 `-0.5` 的
    #      组合方式在不同LUT 写法下会变）→ 实测仍走错分支。
    #   → 最终用**显式 `family` 参数**（调用点自己知道是 fp4 还是 int4），
    #     并**断言 LUT 长度 == 16**（4-bit 的位型契约）。
    assert lut.size == 16, f"4-bit LUT 必须 16 格，实际 {lut.size}"
    if family == "fp":
        scale = amax / float(np.max(np.abs(lut)))   # e2m1：按最大格点归一
    else:
        scale = 2.0 * amax / 15.0# int4 定点：-8..+7，per-tensor
    lut_b = lut.reshape(1, -1)
    codes = np.empty(flat.size, dtype=np.uint8)
    step = max(1, int(chunk))
    for i in range(0, flat.size, step):
        seg = flat[i:i + step].reshape(-1, 1)
        codes[i:i + step] = np.argmin(np.abs(seg / scale - lut_b),
                                      axis=1).astype(np.uint8)
    if codes.size % 2:                # 奇数个 → 补一个 0（numba 侧同样容许）
        codes = np.append(codes, np.uint8(0))
    hi = (codes[0::2] & np.uint8(0x0F)).astype(np.uint8)
    lo = (codes[1::2] & np.uint8(0x0F)).astype(np.uint8)
    return ((hi << np.uint8(4)) | lo), float(scale)


def _unpack4(packed, n_cols: int, lut: np.ndarray, scale: float,
             device=None):
    """uint8 码本 → fp16 计算张量 `(n_rows, n_cols)`。**每步调用**（热路径）。

    ⚠ 这是 4-bit 的**代价**：每步 unpack 出 fp16 副本（n_out×k×2 字节），
    与 fp8 的反量化同量级。**能否省要看昇腾 msprof** —— 若 unpack 抵消了
    4-bit 省下的访存，4-bit 就不如 fp8（那时应诚实告诉用户，而不是吹）。
    """
    import torch as _t
    p = packed if isinstance(packed, _t.Tensor) else _t.as_tensor(packed)
    if p.dim() == 1:                      # 接受扁平的 (n_rows*n_cols/2,) 码本
        p = p.reshape(-1, 1) if n_cols <= 2 else p.reshape(-1, 2)
    lo = (p & 0x0F).to(_t.int32)
    hi = ((p >> 4) & 0x0F).to(_t.int32)
    inter = _t.empty((p.shape[0], p.shape[1] * 2), dtype=_t.int32,
                     device=p.device)
    inter[:, 0::2] = hi                  # 偶数列 = 高半字节
    inter[:, 1::2] = lo                  # 奇数列 = 低半字节
    if n_cols < inter.shape[1]:
        inter = inter[:, :n_cols]
    lut_t = _t.as_tensor(lut * float(scale), dtype=_t.float32,
                         device=p.device)
    return lut_t[inter.long()].to(_t.float16)




def _ensure_npu_registered():
    """确保 torch_npu 已把 npu 注册进 torch 设备表，返回错误原因或 None。

    P160：实现收敛到 `torch_lm.ensure_npu_registered`（**单一来源**），
    这里只包一层，让 `_probe` 的调用点自解释。
    """
    from .torch_lm import ensure_npu_registered as _f
    return _f()


def resolve_accel_device(spec: str = "auto") -> str:
    """解析加速器设备（auto = 昇腾 → ROCm → CUDA → DirectML → CPU）。

    复用 `phdnet.backends.torch_lm.resolve_device`（同一优先级、同一诚实报错
    口径）；torch 缺失时抛 RuntimeError（调用方回落 CPU 路径）。
    """
    if torch is None:
        raise RuntimeError("未安装 torch，加速读出不可用（应回落 CPU 路径）")
    from .torch_lm import resolve_device
    return resolve_device(spec)


class AccelReadout:
    _is_accel = True          # P40：供 model.step 区分加速后端（设备直通 + target_idx）
    """设备无关（cpu / npu / cuda / rocm）的稠密读出，接口对齐 `Readout`。

    W **常驻设备**（NPU 上 908 MB 级），每步只往返 h 与 y：
      - forward：h（CPU numpy）→ 设备 → y → CPU numpy；
      - learn_softmax：主回路 fp32（与 P9 协议一致），梯度更新在设备上完成。
    """

    def __init__(self, n_h: int, n_out: int, rng=None, device: str = "auto",
                 dtype: str = "fp32", w_clip: float = 0.0, w0=None,
                 nll_sync_every: int = 1, compile: bool = False,
                 compile_mode: str = "default", conn_k: int = 0,
                 csr=None, lognormal_init: bool = False, exc_ratio: float = 0.8,
                 gather_impl: str = "index", gather_dtype: str = "fp32",
                 int8_compute: str = "auto",
                 fp8_conv: str = "torch",
                 sparse_fwd_kernel: str = "mulsum"):
        if torch is None:
            raise RuntimeError("未安装 torch，加速读出不可用")
        # P148：fp8 三级自适应的结果。**必须在此先初始化**——稀疏分支
        # （`if conn_k and conn_k>0`）在 fp8 分支**之前**执行，会读这两个标志。
        self._fp8_native = False
        self._fp8_fallback_to_int8 = False
        self._fp8_cap = None
        self._int4 = False                   # P151：4-bit 打包标记
        # P151：**「码本+scale」型存储**的标记（int8 / int4 / fp4）。
        #   它们反量化时必须乘 `_wscale`；原生浮点张量（fp8/fp16/bf16/fp32）
        #   不能乘。这个标志是 P149 那个「一律 `W.to(cdtype)`」的补丁。
        self._int8_like = False
        # P152：`int8_compute` —— fp8/int8 **存储** + int8 **计算域**。
        # 默认 False（社区做法是反量化到 fp16 计算；int8 计算是双重量化，见
        # `_matmul` 里的代价说明）。开启须经探针 PPL 验证。
        self._int8_compute = False   # P153：由降级链 + int8_compute 参数决定
        self._fp8_wscale = None      # fp8 存储的 per-tensor scale（若有）
        # P152：int8 计算域的 scale。`_int8_compute_scale` 是量化用的分母，
        #   `_int8_out_scale` 是**脱 scale 系数**（= 分母，结果要乘回）。
        #   少了后者 → y 量级差 1/scale → nll 爆掉（实测 10.84 → 1071）。
        self._int8_compute_scale = None
        self._int8_out_scale = 1.0
        # P154：fp8 → int8 的**位数转换在 CPU 上做**（fhz 指令），走哪条路径：
        #   "torch"（默认，实测快 5.1×：27 vs 137 ms @1b 档），多 76 MiB 中间张量
        #   "nogil"（0 中间张量 + 释放 GIL，慢 5.1×）
        #   "rust"（P189：phdnet_rs 转换核，多核并行；不可用时回落 torch）
        self._fp8_conv = str(fp8_conv or "torch").lower()
        dtype = _DTYPE_ALIASES.get(str(dtype), str(dtype))
        if dtype not in _DT:
            raise ValueError(f"不支持 dtype={dtype!r}；可用 {sorted(_DT)}")
        self.device = resolve_accel_device(device)
        # P105 int8 语义（fhz 2026-10-01 决策）——存储/计算/更新三层分开：
        #   存储：W 用 torch.int8 码本，per-tensor scale = 2*max|W|/127（P104 的
        #         2× 余量纪律：训练中权重增长 ≤2× 不需要重标定；越界被 clamp
        #         饱和，不回绕）。
        #   forward：每步反量化 W_int8 → fp16（codes.float()*scale）再 matmul
        #         ——1.6 亿元素 ≈ 0.5-1 ms（NPU 可接受）。**不缓存 fp16 副本**：
        #         W 每 step 都在更新，缓存必然过期。
        #   learn：在 fp32 域计算 dp⊗ht，然后 W_int8 重量化写回（见
        #         _int8_update）。精度约束（知情取舍）：int8 步长
        #         ≈ max|W|/127 ≈ 0.008 ≫ 非目标行 |dp|≈1e-6 → 非目标行更新
        #         会被量化吃掉（int8 存储降 4× 访存的代价，fhz 知情选择）；
        #         但目标行 |dp|≈1 远大于步长，必须完整保留。
        # tdtype 是**存储** dtype（int8）；ht/y 等 matmul 参与者的 dtype 跟随
        # _cdtype（int8 模式 = fp16）——所有原来用 self.tdtype 的地方都要检查。
        # ══════════════════════════════════════════════════════════════
        # P151（fhz 2026-10-03）：**统一精度降级链**，取代 P147 的 fp8 专用三级
        # 指令原文：「fp 该多少是多少，如果 fp 的不支持就自动转为 int，int 如果
        #   不支持自动转为 fp，如果两者都不行再报错；解禁 fp8/fp4/int4/int8」
        # 链的顺序由 `phdnet/precision_policy.candidate_order` 决定：
        #   fp8 → fp16 → bf16 → **int8** → int16 → int32 → fp32
        # ⚠ fp32 放**链尾**（不在同族兄弟里）—— 否则 fp8 在 fp16 不可用时会先
        #   撞上 fp32、**永远轮不到 int8**，违背「先试 int」的指令。
        # 判据=**真跑一次**（不是查表）：P86 的硬编码禁令会被驱动升级推翻，
        #   而 P148 我写的`a.to(fp16) @ b` 探测**全程没跑 fp8 kernel**，
        #   在任何设备上都成功 → 等于没有探测。
        # ══════════════════════════════════════════════════════════════
        from ..precision_policy import (candidate_order, resolve_precision_cached,
                                       unsupported_message)
        _req = str(dtype or "fp32").lower()
        #⚠⚠⚠ **P161 根因修复（fhz 2026-10-03 13:19 生产日志）**
        # 之前这里写的是 `_dev = str(device)` —— 即**未解析的原始入参**。
        # 而上一行 `self.device = resolve_accel_device(device)` **已经把
        # "auto" 解析成具体设备**（昇腾上 = "npu"）。
        #   → `_probe` 拿到 "auto" 后执行 `torch.ones(8, 8, device="auto")`
        #     → torch 把 "auto" 当**未知设备名**去查表 → 报
        #       「Expected one of cpu, cuda, ipu, xpu, mkldnn, ...」
        #     → **列表里没有 npu**（那是 torch 核心的设备表，与 torch_npu 无关）
        #     → **fp8 / fp16 / int8 / bf16 / fp32 全部失败**（全是这一个原因）
        #     → 降级链走到尽头 → `[readout] backend=numba-cpu(回落)`。
        # ⚠ 所以之前我误判成「torch_npu 未注册」（P160），那是**错的诊断**：
        #   fhz 的日志 STEP3 显示 `import torch_npu` 本来就成功、device_count=1、
        #   且 STEP5 在 npu 上建张量**成功** —— 环境一直是好的。
        #   错的只有这一行变量取值。
        # ✅ 正确做法：用**已解析**的 `self.device`。
        _dev = str(self.device)

        def _probe(dt: str):
            """真跑一次，判断 `dt` 在该设备上是否可用（构造 + 前向 + 更新）。"""
            # ⚠ 先让 torch_npu 注册 npu（副作用式 import）。
            #   注P160：当时我以为这是读出回落的**根因**，**那是误判** ——
            #   真因是 `_dev` 用了未解析的 `"auto"`（P161 已修）。
            #   这行 import 本身**仍然必要**（`torch_lm` 从不import torch_npu，
            #   不注册则 torch 设备表里确实没有 npu），但它不是那次回落的原因。
            #   保留它顺带让失败原因**可见**（此前被 `except: pass` 吞掉）。
            _npu_err = _ensure_npu_registered()
            try:
                import torch as _t
                if _npu_err is not None and str(_dev).startswith("npu"):
                    raise RuntimeError(
                        f"torch_npu 未注册 npu 设备（{_npu_err}）；"
                        f"torch 设备表里没有 npu，无法建张量"
                        f"（当前 device={_dev!r}）")
                if dt in ("fp8",):
                    # 原生 fp8：**必须让 fp8 张量真正进 matmul**（不预 cast）
                    a = _t.ones(8, 8, dtype=_t.float8_e4m3fn, device=_dev)
                    b = _t.ones(8, 8, dtype=_t.float16, device=_dev)
                    try:
                        _ = (a @ b).float().sum().item()
                    except (RuntimeError, TypeError):
                        a2 = _t.ones(8, 8, dtype=_t.float8_e4m3fn, device=_dev)
                        _ = (a2 @ a2).float().sum().item()
                    return True, "原生 fp8 kernel"
                if dt in ("fp4", "int4"):
                    # 4-bit：加速臂用 uint8 打包，**计算域 fp16**
                    #   → 只需验证 fp16 路径可用（打包是纯 CPU 侧，无算子依赖）
                    _t.ones(8, 8, dtype=_t.uint8, device=_dev)
                    _t.ones(8, 8, dtype=_t.float16, device=_dev)
                    _ = (_t.ones(8, 8, dtype=_t.float16, device=_dev)
                         @ _t.ones(8, 8, dtype=_t.float16,
                                   device=_dev)).sum().item()
                    return True, "uint8 打包 + fp16 计算"
                # 常规：真跑一次目标 dtype 的 matmul（不预 cast）
                _dt_map = {"fp32": _t.float32, "fp16": _t.float16,
                           "bf16": _t.bfloat16, "int8": _t.int8}
                tdt = _dt_map[dt]
                _ = (_t.ones(8, 8, dtype=tdt, device=_dev)
                     @ _t.ones(8, 8, dtype=tdt, device=_dev)
                     ).float().sum().item()
                return True, f"{dt} matmul"
            except Exception as e:                              # noqa: BLE001
                return False, f"{type(e).__name__}: {str(e)[:70]}"

        # fhz 2026-10-03：「训练启动时判断一次即可」→ 进程级缓存
        #   （同设备能力不会变；探测要真跑 matmul，不该重复做）
        _res = resolve_precision_cached(_req, _probe, _dev)
        if _res["dtype"] is None:
            # 指令：「如果两者都不行再报错」
            raise RuntimeError(unsupported_message(_res, _dev))
        if _res["downgraded"]:
            import warnings as _w
            _w.warn(f"[precision] 设备 {_dev} 不支持 {_req}（或链上更高精度），"
                    f"**已自动降级为 {_res['dtype']}**。降级链：{_res['chain']}。"
                    f"（原请求的存储收益可能达不到）",
                    RuntimeWarning)
        dtype = _res["dtype"]                     # 用**实际可用**的那个
        self._precision_resolved = _res
        self._dtype_requested = _req
        # P151：降级链选中了 fp8 → 稀疏/稠密两臂都要「fp8 存储 + fp16 计算」。
        #   （原先由 P147 的 `resolve_fp8_request` 单独设置，P151 统一到链里，
        #     那段已被链取代，但稀疏分支仍在读这个标志。）
        # P153（fhz 2026-10-03）：**fp8 不可用 + int8 可用 → 自动开启 int8 计算**。
        #   这正是 910B 的情形（P86：ERR01007，无 fp8 算子），也是 int8 算子
        #   **唯一**能被用上的时机（fp16 计算走的是 fp16 单元）。
        #   可用 `--readout-int8-compute auto|on|off` 覆盖：
        #     auto（默认）= 降级到 int8 时自动开；on = 强制开；off = 强制关。
        _i8m = str(int8_compute).lower()
        _forced_on = _i8m in ("true", "1", "yes", "on")
        _auto = _i8m == "auto"
        # auto：降级到 **int8** 时自动开（910B 的情形：fp8 不可用 → 链落 int8）。
        #on：强制开，**包括存储仍是 fp8 的情形**（fp8 → int8 网格也算 int8 计算）。
        # ⚠ P153：**只在「存储也是 int8」时启用 int8 计算**。
        #   不做「fp8 存储 + int8 计算」—— 那是**双重量化**（fp8 再量化到int8），
        # ⚠ P154（fhz 2026-10-03 明确指令）：
        #   「fp8不行就**默认降级到 int8 计算，fp8 存储**」
        #   → 所以 **auto 模式下只要落到 fp8**（即设备**有** fp8 算子，
        #     但我们要用 int8 算子算），也启用 int8 计算。
        #   ⚠ 与「fp8 不可用 → 落 int8」的区别：后者是**存储也变int8**，
        #     而这里是**存储仍 fp8、计算走 int8**（位数转换在 CPU 上做，
        #     见 `phdnet/backends/fp8_int8_convert.py`）。
        #   两条路的int8 计算都启用，因为**int8 算子只有这两条路能用到**。
        self._int8_compute = (_forced_on or _auto)
        self._fp8_native = (_res["dtype"] == "fp8")
        self._fp8_fallback_to_int8 = (_res["requested"] == "fp8"
                                      and _res["dtype"] == "int8")
        # ════════════════════════════════════════════════════════════════
        # **P189（fhz 2026-10-07 指令「模型 fp8，迭代 fp16」）**────────────
        # 显式请求 fp8 且设备**无原生 fp8 算子**时：**保持 fp8 请求语义**，
        # 走「fp8 位模式（uint8）存储 + 每步 CPU 转换 int8 计算」——
        #   · 存储 1 B/元素 = fp8 的访存收益完整保留（vs fp16 降级的 2 B）；
        #   · 计算走 910B **真实存在**的 int8 算子（P154/P155 的 W8A8 路径）；
        #   · 迭代（dp⊗ht 更新）走 fp16 域：int8 码本反量化 → fp16 addcmul_
        #     → 重量化写回 uint8 位模式（fhz：「迭代用 fp16」）。
        # ⚠ 仅此一条降级路恢复 int8 算子的使用；**显式请求 int8 仍被
        #   `candidate_order` 直接 raise**（P163 禁令不动）。
        # ⚠ 与 P163「落到 fp16」旧口径的区别：fp16 降级丢掉一半访存收益，
        #   而本路径访存与原生 fp8 完全相同（都 1 B）。
        # ════════════════════════════════════════════════════════════════
        self._fp8_bits = bool(_req == "fp8" and _res["dtype"] in ("fp8", "fp16")
                              and _res["dtype"] != "fp8")
        if self._fp8_bits:
            # P191d（服务器 2026-10-07 实测 PPL 34354 + 读出 313ms 的根因）：
            # fp8_bits 模式下**强制关 int8 计算域**——`--readout-int8-compute
            # auto` 会把 auto 判 True，稀疏臂 `_matmul` 随即把 uint8 位模式
            # 当 0..255 实数参与运算（数值全错）；且 int8 计算需每步 D2H 转
            # 码（这正是 313ms 的来源）。P189 语义 = fp8 位模式存储 + **fp16
            # 前向/迭代**（LUT 反量化，零主机交互）；int8 计算仍可显式
            # --readout-int8-compute on 走 P154 路径（知情选择）。
            self._int8_compute = False
        if dtype == "int8":
            self.tdtype = torch.int8
            self._cdtype = _INT8_CDTYPE
            self._int8_like = True      # P151：码本+scale 型（反量化要 ×scale）
        elif dtype in ("fp4", "int4"):
            # P151：4-bit 在加速臂用**uint8 打包**（两元素一字节），与 numba
            #   的 `_ro_q_matvec_fp4` 语义一致。**计算域仍是 fp16**（P149 口径：
            #   低精度只管存储，算子要吃 fp16）。
            self.tdtype = torch.uint8
            self._cdtype = torch.float16
            self._int4 = True                # 4-bit 需要 pack/unpack 辅助
            self._int8_like = True           # P151：码本+scale 型
            self._wscale = None
        else:
            self.tdtype = getattr(torch, _DT[dtype])
            self._cdtype = self.tdtype
        if self._fp8_bits:
            # P189：存储域 = fp8 e4m3fn 位模式（**uint8 承载**——910B 连 fp8
            # 张量都建不出来（P86 ERR01007），只能用 uint8 存位模式）；
            # 计算域 fp16（fhz「迭代用 fp16」）；前向走 int8 算子（见 _matmul）。
            self.tdtype = torch.uint8
            self._cdtype = torch.float16
            self._int8 = False
            self._int8_like = False
            self._wscale = None
            self._fp8_wscale = None
            self._last_int8_scales = (1.0, 1.0)
        # int8 模式不走 torch.compile 融合核：_train_step_core 未覆盖反量化/
        # 重量化路径，融合会产出另一套计算图——宁可放弃融合收益（正确性优先）。
        if dtype == "int8" and compile:
            compile = False
        # P105 回落保护（参考 P85/P90 的 fp8 模式）：构造后做一次探测——int8 码本
        # 反量化 + fp16 matmul 在该设备是否真的可用；不可用则**永久**回落 fp16
        # （tdtype/_cdtype 跟随）+ 告警。⚠ P90 的 forward 级兜底（try/except
        # around matmul）仍然保留在 forward_dev：探测通过不代表运行期所有 shape
        # 都不炸，int8 matmul 在某些平台可能跑着跑着才失败。

        self._int8 = (self.tdtype == torch.int8)
        self._wscale = None
        if self._int8:
            try:
                _p8 = torch.ones(4, 4, dtype=torch.int8, device=self.device)
                _p16 = torch.ones(4, 4, dtype=self._cdtype, device=self.device)
                _ = (_p8.to(self._cdtype) @ _p16).sum().item()
            except Exception as e:                          # noqa: BLE001
                import warnings
                warnings.warn(
                    f"int8 探测在设备 {self.device} 上失败（{type(e).__name__}: "
                    f"{str(e)[:80]}）→ 永久回落 fp16（训练不中断）", RuntimeWarning)
                self.tdtype = torch.float16
                self._cdtype = torch.float16
                self._int8 = False
        self.n_out, self.n_h = int(n_out), int(n_h)
        # ── P111：稀疏（CSR）读出 ────────────────────────────────────────────
        # 此前`readout_conn_k>0` 命中 `_unsupported_reason` → 整个 NPU 读出
        # 回落 numba CPU（读出占端到端 ~89%，等于关掉了全部设备加速）。
        # 本实现在设备侧持有**均匀 k 的 (n_out, k) 稠密 val 张量 + (n_out, k)
        # 列索引张量**，前向/更新都按 gather 实现：
        #   前向  y[i] = Σⱼ Wv[i,j] ·h[Wi[i,j]]
        #   更新  Wv  -= η · dp[i] · h[Wi[i,:]]      （逐行 rank-1）
        # 流量口径（1B 档 vocab=73,958, k=128, n_h=3,072）：
        #   稠密 fp32 每步触达 866.7 MiB；稀疏 (val fp32 + idx int64) = 12 B/突触
        #   → 144.4 MiB，约 1/6。**idx 用 int64 是刻意的**：int32 会让 NPU 侧
        #   gather 走类型转换，实测口径以verify_accel_sparse 为准。
        #
        # ⚠ **只支持均匀 k**（`_random_csr` 的输出）。非均匀行宽（幂律分组）
        #   不在此路径——那种结构要用变长 CSR + segment sum，语义与实现都不同，
        #   命中时按P19 纪律回落 numba，不静默走错算法。
        self._sparse = False
        self._sparse_k = 0                                    # conn_k 的内部真值源
        self.Wi = None                                        # (n_out, k) int64 列索引
        #（`conn_k` 是只读 property —— 它必须与 numba Readout 同为只读，
        #   否则两边语义不对称，ckpt 的 `net.readout.conn_k` 读法会分叉。）
        if conn_k and int(conn_k) > 0:
            k = max(1, min(int(conn_k), self.n_h))
            if csr is not None:
                ip, idx, val = csr
                widths = np.diff(np.asarray(ip, dtype=np.int64))
                if widths.size != self.n_out or not np.all(widths == widths[0]):
                    raise ValueError(
                        f"稀疏读出只支持均匀 k（收到行宽 min={widths.min()} "
                        f"max={widths.max()}）")
                if int(widths[0]) != k:
                    raise ValueError(f"CSR 行宽 {int(widths[0])} != conn_k {k}")
                _idx = np.asarray(idx, dtype=np.int64).reshape(self.n_out, k)
                _val = np.asarray(val, dtype=np.float32).reshape(self.n_out, k)
            else:
                from ..sparse_pc import _random_csr
                # fan-in 补偿与 `Readout.__init__` 的稀疏分支同式（否则稀疏臂
                # 因幅值偏低 √(k/n_in) 被在 PPL 上无谓惩罚）。
                _ip, _idx, _val = _random_csr(
                    rng if rng is not None else np.random.default_rng(0),
                    self.n_out, self.n_h, k,
                    0.05 * np.sqrt(self.n_h / k), lognormal_init, exc_ratio)
                _idx = _idx.reshape(self.n_out, k)
                _val = _val.reshape(self.n_out, k).astype(np.float32)
            self._sparse_k = k
            self._sparse = True
            # P116：前向算子（"mulsum" | "einsum"）。⚠ einsum 非逐位（归约顺序
            # 不同，本机 max|Δ|≈3e-05），且昇腾收益未实测 → 默认 mulsum。
            self._sp_fwd = str(sparse_fwd_kernel or "mulsum").lower()
            self.Wi = torch.tensor(_idx, device=self.device, dtype=torch.long)
            # ⚠ P148：`W` 的 dtype 必须在**下面的 tdtype 决定之后**才能建
            #   （此前这里硬写 float32，即使 tdtype 被设成 fp8/int8 也不会生效）。
            #   故先留 None，等dtype 定下来再量化/转换。
            self.W = None
            # ⚠⚠ **P148（fhz 2026-10-02 决策「稀疏也用 fp8」）**：
            #   此前这里**硬置 fp32**（P110 的理由：低精度会丢非目标行更新），
            #   于是「模型本体 fp8」在**生产默认路径（conn_k=128）上完全不生效**。
            #   fhz 明确选择「稀疏也用 fp8」→ 改为**遵从 fp8 探测的结果**。
            # ⚠ 代价必须说清（P110 实测）：fp32 非目标行更新保留率 **99.95%**、
            #   fp16 **26.67%**、bf16 5.79%；**fp8 的量化步长比 fp16 粗约 4 倍**
            #   （mantissa 3bit vs 10bit）→ 非目标行更新可能**几乎全丢**，
            #   学习规则将大幅退化为纯 Hebbian（"use it or lose it" 变成全丢）。
            #   → 故**必须告警**（不静默），且验收只能靠固定探针 PPL
            #     （`--probe-every`；训练流 PPL 波动 ±20% 不可用，见 P146）。
            if self._fp8_native:
                # ⚠⚠ **P149（fhz 2026-10-02：「模型原生 fp8，迭代 fp16」）**：
                #   **存储 fp8、计算 fp16** —— 这是本项目的第三种精度布局
                #   （P105 的 int8 是「存储 int8、计算 fp16」；fp32 是两者同）。
                #   ⚠ 我 P148 在这里把 `_cdtype` 也设成 fp8 → **训练直接崩**：
                #     torch 算子**不做 fp8↔fp16 的隐式提升**
                #     （"Promotion for Float8 Types is not supported"），
                #     且 `softmax`/`addcmul_`/`einsum` 全都不接受 fp8 张量。
                #   → 故 `_cdtype` **必须**留在 fp16（计算域），`tdtype` 才是 fp8
                #     （存储域）。两个域的用途：
                #       tdtype  = 决定 `self.W` 的 dtype（访存量）
                #       _cdtype = 决定算子的输入 dtype（计算精度）
                #     每次用 W 算之前**显式反量化**（见 `_matmul` / `_eager_step`）。
                self.tdtype = torch.float8_e4m3fn
                self._cdtype = torch.float16        # ← 不是 fp8！
                # P152：记下 fp8 的 per-tensor scale，供 int8 计算域用
                #（fp8 张量本身的值域由 dtype 决定，但若将来加 scale 需在此挂）。
                self._fp8_wscale = None
                self._int8 = False
                self._wscale = None
                import warnings as _w8
                _w8.warn(
                    f"[fp8] 稀疏读出（conn_k={k}）**已启用 fp8 存储**"
                    f"（fhz 2026-10-02 决策）。⚠ P110 实测低精度会丢弃"
                    f"感知器 p−t 的**非目标行**更新（fp32 99.95% / fp16 26.67%），"
                    f"**fp8 步长比 fp16 粗约 4 倍** → 学习规则将大幅退化为"
                    f"纯 Hebbian。**请用 --probe-every 的 probe_PPL 验证"
                    f"是否还在学习**。",
                    RuntimeWarning)
            else:
                # fp8 不可用而 int8 可用 → 稀疏臂的 W 走 int8 码本
                # （P105 语义：存储 int8、per-tensor scale=2·max|W|/127）
                if self._fp8_fallback_to_int8:
                    self.tdtype = torch.int8
                    self._cdtype = torch.float16
                    self._int8 = True
                    self._int8_like = True   # P151：码本+scale 型
                    self._wscale = None      # 由下方 int8 量化块按 _val 计算
                # ⚠⚠ **P150 修正**：`else` 分支原来**无条件压成 fp32**，
                #   于是 `--readout-dtype fp16` / `bf16` 在**稀疏臂被静默吞掉**
                #   （实测：请求 fp16 → `W.dtype` 竟是 float32）。
                #   → 改为**尊重 `__init__` 开头已经按 `_DT[dtype]` 定好的
                #     `tdtype`**（那里 P149 之前就正确处理了 fp16/bf16/fp32）。
                #     稀疏分支只负责「int8/fp8 需要码本化」，其余保持原样。
                else:
                    self._int8 = (self.tdtype == torch.int8)
                    self._wscale = None
            # P148：dtype 已定，按它建 W（稀疏臂的存储 = tdtype）
            if self._int4:
                # P151：4-bit 打包（两元素一字节，列数需向上取偶）
                _lut = (_FP4_E2M1_LUT if _req == "fp4" else _INT4_LUT)
                _k4 = _val.shape[1]
                _pk, self._wscale = _pack4(_val, _lut,
                                            family=("fp" if _req == "fp4"
                                                    else "int"))
                self._w4_cols = _k4
                self.W = torch.tensor(_pk.reshape(_val.shape[0], -1),
                                      device=self.device, dtype=torch.uint8)
            elif self.tdtype in (torch.int8,):
                # P105 语义：per-tensor scale = 2·max|W|/127
                _amax = float(np.abs(_val).max())
                self._wscale = (2.0 * _amax / _INT8_QMAX) if _amax > 0 else 1.0
                _q = np.clip(np.rint(_val / self._wscale),
                             -_INT8_QMAX, _INT8_QMAX).astype(np.int8)
                self.W = torch.tensor(_q, device=self.device,
                                      dtype=torch.int8)
            else:
                if self._fp8_bits:
                    # P189：实值 → fp8 e4m3fn 位模式（uint8 承载）。
                    # cast 在 CPU 做（910B 无 fp8 算子，设备 cast 会 ERR01007）；
                    # RNE 舍入与 torch fp8 cast 同。
                    _t8 = torch.from_numpy(np.ascontiguousarray(
                        _val, dtype=np.float32)).to(torch.float8_e4m3fn)
                    self.W = _t8.view(torch.uint8).to(self.device)
                else:
                    self.W = torch.tensor(_val, device=self.device,
                                          dtype=self.tdtype)
            # 融合核走的是稠密 addmm_，稀疏不适用 → 永久eager（P38/P55 纪律）。
            self._compiled = False
            self._fused = None
        else:
            self.Wi = None
        self.w_clip = float(w_clip)
        # P34：nll 设备侧累积（消除每步 .item() 同步 → CPU/NPU 重叠）
        self.nll_sync_every = max(1, int(nll_sync_every))
        self._nll_sum = None
        self._nll_n = 0
        self._last_nll = 0.0
        # P80/P82：correct 索引的 pinned 上传缓冲（避免 pageable 小拷贝的隐式同步）
        self._correct_pinned = None
        # P38/P44：`torch.compile` 融合读出热路径的 4 个小 kernel
        # （softmax / log / sub / addmm_；CANN 上每个 launch 开销 ~50-200 μs）。
        #
        # ⚠ **模式选择的关键取舍（服务器实测得到）**：`mode="reduce-overhead"`
        # 启用 CUDA Graph，而 `self.W.addmm_(...)` **原地修改 W** —— cudagraph
        # 捕获的张量不允许被后续 kernel mutate，故每次调用都会打印
        #     "skipping cudagraphs due to mutated inputs"
        # 并**静默退回**无 graph 模式（功能正确，但拿不到 graph 收益）。
        # 这是**本质冲突**：W 每步都在原地更新（908 MiB/step；改成非原地重新
        # 分配会让流量翻倍，更不可接受）。因此默认用 `mode="default"`
        # （只做 kernel 融合，不启用 cudagraphs）——融合收益保留，无 mutate 限制。
        # 需要图级优化时可选 reduce-overhead/max-autotune（接受上述回退）。
        self._compiled = False
        self._compile_mode = str(compile_mode or "default")
        # P58：pinned 暂存池（懒初始化；CPU-only torch 无 pin_memory → 回落）
        self._pin_bufs = None
        self._pin_ok = False
        self._pin_next = 0
        self._pin_cap = 4
        # P128：融合核的启用**不再排除稀疏路径**。
        # 此前 `if w0 is None and not self._sparse:` 把稀疏读出排在编译之外 →
        # 稀疏（= 生产默认）永远跑逐算子 eager：softmax / log / rank-1 更新
        # 各自成一个 kernel。2026-10-02 服务器日志显示读出 **10.99 ms**、
        # 是带宽下界（0.300 ms）的 36×，而 P115/P120 的判读规则指向
        #「瓶颈在 kernel 下发」——逐算子 eager 正是该结论的直接成因。
        # `_train_step_core` 本身与稀疏无关（它只吃 y32/ht/t/correct/eta），
        # 故放开是安全的；运行期失败已有「永久回落 eager + 告警」（P19）。
        if compile:
            try:
                self._fused = torch.compile(self._train_step_core,
                                            mode=self._compile_mode,
                                            fullgraph=False)
                self._compiled = True
            except Exception as e:                       # noqa: BLE001
                import warnings
                warnings.warn(f"torch.compile 不可用，回落 eager：{e}",
                              RuntimeWarning)
        if w0 is None and not self._sparse:
            g = rng if rng is not None else np.random.default_rng(0)
            init = (g.normal(0.0, 0.05, (self.n_out, self.n_h)) if hasattr(g, "normal")
                    else np.asarray(g).reshape(self.n_out, self.n_h))
        elif self._sparse:
            # 稀疏模式下 `w0` 语义是 (n_out, k) 的 val矩阵（列由 Wi 决定）。
            # 给了就覆盖初始化值；不给则沿用上面 CSR/随机生成的 val。
            if w0 is not None:
                _w0s = np.asarray(w0, dtype=np.float32)
                if _w0s.shape != (self.n_out, self.conn_k):
                    raise ValueError(
                        f"稀疏 w0 形状不符：期望 {(self.n_out, self.conn_k)}，"
                        f"收到 {_w0s.shape}")
                self.W = torch.tensor(np.ascontiguousarray(_w0s),
                                      device=self.device, dtype=torch.float32)
            init = None
        else:
            init = np.asarray(w0, dtype=np.float32)
        # 稀疏路径的self.W 已在上面建好（fp32 (n_out,k)），不再走稠密初始化。
        if not self._sparse:
            # ⚠ 必须**复制**：torch.as_tensor / torch.from_numpy 会与传入数组共享内存，
            # 而 W 是原位更新的 → 调用方（例如比较用的参考权重）会被静默改掉。
            _init_t = torch.tensor(np.ascontiguousarray(init, dtype=np.float32),
                                   device=self.device, dtype=torch.float32)
            if self._int8:
                # int8 存储：per-tensor scale = 2*max|W|/127（2× 余量，见上方注释）。
                # 码本容量 ±127；初始码最大只到 ~63.5，权重涨到 2× 才触饱和（clamp
                # 裁剪，不回绕）。全零权重退化 wscale=1.0（与 fp 路径同等的退化行为）。
                _amax = float(_init_t.abs().max().item()) if _init_t.numel() else 0.0
                self._wscale = (2.0 * _amax / _INT8_QMAX) if _amax > 0.0 else 1.0
                self.W = torch.clamp(
                    torch.round(_init_t / self._wscale),
                    -_INT8_QMAX, _INT8_QMAX).to(torch.int8)
            elif self._fp8_bits:
                # P189：实值 → fp8 e4m3fn 位模式（uint8 承载），cast 在 CPU
                _t8 = _init_t.cpu().to(torch.float8_e4m3fn)
                self.W = _t8.view(torch.uint8).to(self.device)
            else:
                self.W = _init_t.to(self.tdtype)
        # P28：设备侧 (h, y) 缓存，供 forward → learn_softmax 的热路径复用
        self._cache_h: np.ndarray | None = None
        self._cache_ht = None
        self._cache_y = None
        # P111：稀疏 gather 缓存（(n_out,k) 复用，见 _sp_gather）
        self._cache_g = None
        self._cache_g_epoch = -1                            # P112：主机侧 epoch 判据
        self._ht_epoch = 0                                  # 每次上传新 h 递增
        self._ht_epoch_at_cache = -1# P122：`_cache_ht` 上传时的 epoch
        # P134：gather 实现（"index" = 高级索引基线 / "take" = 另一条 device kernel）
        # ⚠ P145 清理：这里原本有**三份重复赋值**（P134 那次脚本失败重试的残留，
        #   功能上无害但属死代码，且说明「重试式插入」的纪律漏洞）。
        self._gather_impl = str(gather_impl or "index").lower()
        # P145：gather 输出的精度（"fp32" | "fp16" | "bf16"），**默认 fp32=保持现状**。
        # 动机（msprof 昇腾实测，600 步）：
        #   aclnnIndex_IndexAiCore_Index  占 81.0%，耗时 3,975,871 μs，
        #   其中 **aiv_time = 99.6%** → 它在 **Vector 单元**上真执行（非排队），
        #   流量 50.8 MiB → 有效带宽仅 **8.1 GB/s**；
        #   而同流量的 BatchMatMul 在 Cube 上只需 1.41 ms（**38 GB/s**，快 4.7×）。
        #   → `g` 若降到 fp16，**流量减半 → 理论 AIV 6.63 → ~3.3 ms**，
        #     每步总量 8.18 → **~4.9 ms**（接近 5 ms 目标）。
        # ⚠⚠ **但精度代价必须由 fhz 的 PPL A/B 判定，我不替他拍板**：
        #   本机实测（8192×64×128，含稀疏激活 h）fp16 的 g 带来
        #     前向 y 相对误差 ~1.6e-4~1.9e-4
        #     **更新量dW 相对误差 ~2.1e-4~2.3e-4**（≈1800× fp32 eps）
        #   对比参照：P131 的 einsum 非逐位是 3e-5（fp16 的 g **大一个量级**）；
        #   而 P110 的 fp16 **权重**是破坏性的（保留率 26.67%）。
        #   **关键区别**：W 是**累积状态**（每步读回都丢精度→累积），而 g 是
        #   **中间量**（算完即弃、不累积）→ P110 的结论**不能直接套到 g**。
        #   但 2e-4 的相对误差是否会侵蚀学习效果，**只有 PPL 能回答**。
        self._gather_dtype = str(gather_dtype or "fp32").lower()
        self._csr_val_host = None                           # P111：_csr 导出缓存
        # ── P187（fhz 2026-10-06 指令）：**模型权重 fp16、迭代全 fp32** ──────
        # fp16/bf16 请求的读出权重**统一提升为 fp32 常驻**。理由（P110 实测）：
        #   更新写回落在 W 的存储精度上——fp16 下非目标行更新保留率仅 26.67%，
        #   「迭代全 fp32」要求 W 本身是 fp32，否则每次 addcmul_ 的 fp32 结果
        #   写回 fp16 时就被舍入丢弃（半 ULP ~6e-5 ≫ |dp|~1e-6）。
        #   检查点存储仍是 fp16（--ckpt-dtype 默认，P107b）→「模型权重 fp16」
        #   在存储层成立，运行时为保精度驻留 fp32。
        # ⚠ 仅作用于 fp16/bf16 显式请求；int8/int4/fp8 码本路径保持各自语义
        #   （反量化→更新→重量化），不动。
        if (self.W is not None
                and getattr(self, "tdtype", None) in (torch.float16,
                                                      torch.bfloat16)):
            self.W = self.W.to(torch.float32)
            self.tdtype = torch.float32
            self._cdtype = torch.float32
            self._int8 = False
            import warnings as _w187
            _w187.warn(
                "[precision] 读出权重 fp16/bf16 请求 → 运行时提升为 fp32"
                "（迭代全 fp32，fhz 2026-10-06 决策；P110：fp16 写回丢失 73%"
                " 非目标行更新。检查点存储仍为 fp16）", RuntimeWarning)

    # ---------- 前向 ----------
    def _matmul(self, ht):
        """y = W @ ht；int8 模式**每步**反量化后再乘。

        P105：反量化 = `codes.float() * scale`（fp32 域）→ cast 到 ht 的 dtype
        （fp16）做 matmul。**不缓存 fp16 副本**——W 每 step 都在更新，任何缓存
        下一步就过期（fhz 明确要求每步反量化；1.6 亿元素 ≈ 0.5-1 ms @NPU）。

        P111 稀疏：不做 matmul，改为 gather + 逐行求和
            y[i] = Σⱼ W[i,j] ·ht[Wi[i,j]]
        （`_gather` 复用 forward→learn 的缓存，避免同一h 被 gather 两次）。
        """
        if self._sparse:
            # y[i] = Σⱼ W[i,j] ·ht[Wi[i,j]] —— **必须逐元素乘 W 再求和**。
            # （曾经的 bug：写成 `self._sp_gather(ht).sum(dim=1)`，等于把
            #   权重全丢掉、只把 gather 到的 h 分量相加——数值完全错但形状/
            #   dtype 都对，只在数值对拍里才暴露。）
            g = self._sp_gather(ht)
            # ⚠⚠ **P149（fhz 指令「模型原生 fp8，迭代 fp16」）**：
            #   `W` 以 **fp8 存储**（省 4× 访存），但**参与算术前必须显式
            #   反量化成 fp16** ——torch **不做 fp8↔fp16 的隐式提升**
            #   （实测报 "Promotion for Float8 Types is not supported"）。
            #   → 存储/计算分离，与 P105 的 int8 完全同构。
            # ⚠⚠ **P151 修正**：int8/4-bit 是**码本 + per-tensor scale**
            #   （P105），反量化必须 **× `_wscale`**；而 fp8/fp16/bf16 是
            #   **原生浮点张量**（dtype 本身就是值），**不能乘** scale。
            #   我 P149 只写了 `W.to(_cdtype)` → int8 稀疏臂的权重全成了
            #   ±127 附近的整数 → 实测 nll **540**（对照 fp8 的 10.8）。
            if self._int4:
                # P151：4-bit → 先 unpack 成 fp16 再算（**每步**，见 `_unpack4`
                #   docstring 的代价说明）。
                _Wm = _unpack4(self.W, getattr(self, "_w4_cols",
                                               self.W.shape[1]),
                               _FP4_E2M1_LUT if self._dtype_requested == "fp4"
                               else _INT4_LUT,
                               float(self._wscale or 1.0),
                               device=self.device)
            elif self.W.dtype == self._cdtype:
                _Wm = self.W
            elif self._wscale is not None and self._int8_like:
                _Wm = self.W.to(self._cdtype) * float(self._wscale)
            elif getattr(self, "_fp8_bits", False):
                # P191d：fp8 位模式（uint8 承载）→ 设备端 LUT 反量化成 fp16
                # 再算（零 D2H）。⚠ 必须排在 `_int8_compute` 之前——uint8 码
                # 不是实数，落进 int8/通用分支就是 PPL 34354（服务器实测）。
                _Wm = self._fp8_bits_to_real()
            elif self._int8_compute:
                # ⚠⚠ **P152（fhz 2026-10-03）：fp8 存储 + int8 计算域**。
                #   动机：910B **无 fp8 算子**（P86ERR01007）但**有 int8 算子**，
                #   而 msprof 证明访存是瓶颈（Index 81%）→ int8 计算既走硬件快路径
                #   又与 fp8 **访存完全相同**（都1 字节/元素）。
                #   做法：fp8 →（反量化到 fp32）→ 再量化到 **int8 网格** →
                #   int8 计算。累加仍回 fp32（社区标准：低精度乘 + 高精度累加）。
                #   ⚠⚠ **代价：这是「双重量化」**，误差 = fp8 量化 + int8 量化
                #   **叠加**（社区没有这个组合，torchao 的 float8_weight_only 是
                #   直接反量化到 fp16 再算）。因为 fp8 的对数间距在±448 处很粗，
                #   再压到 int8 的±127 均匀网格会**丢掉 fp8 的动态范围优势**
                #   ——社区正是因此说「INT8 在 Transformer 里易溢出」。
                #   → 故**默认关闭**（`_int8_compute=False`），需显式开启，
                #     且必须用探针 PPL 验证（见 --readout-int8-compute）。
                _wf = self.W.to(torch.float32)
                if self._fp8_wscale is not None:
                    _wf = _wf * float(self._fp8_wscale)
                elif self._int8_like and self._wscale is not None:
                    _wf = _wf * float(self._wscale)
                # ── **真正的 int8 GEMM（W8 × A8 → int32 累加）**────────────
                # P153（fhz 2026-10-03 的追问暴露了 P152 的无效）：
                #   P152 只转 W、g 仍是 fp16 → `(_Wm * g)` 还是 **fp16 matmul**，
                #   **根本没走 int8 算子**，纯亏（P152 实测 nll 10.8376→10.8701）。
                #   社区标准（W8A8：torchao `int8_weight_only` / TE int8）要求
                #   **两边都是 int8** → 故这里把**激活 g 也动态量化**。
                # ⚠ 代价（必须说清）：动态量化激活会引入额外误差；且 M1 k-WTA
                #   的稀疏激活**动态范围很宽**（少数值远大于其余）→ per-tensor
                #   scale 被大值拉高、小值被压扁。这正是社区说「INT8 在
                #   Transformer 里易溢出」的根因，**本项目的稀疏性可能更严重**
                #   → 必须用探针 PPL 判定（--probe-every）。
                # ⚠ 也正因如此，**默认关闭**；仅在 fp8 不可用且 int8 可用时
                #   由降级链自动开启（--readout-int8-compute auto）。
                # ⚠ P154（fhz 指令「位数转化要在 cpu 上」）：
                #   W 的 fp8 → int8 **位数转换走 CPU 核**
                #   （`phdnet/backends/fp8_int8_convert.py`），默认 torch
                #   向量化路径（实测比 nogil 标量核**快 5.1 倍**：27 vs 137 ms，
                #   1b 档 51962×128；两者**逐位一致**）。
                #   → 注意 fp8 张量在**设备**上，要先 D2H 回CPU 再转换。
                _Wq = self._fp8_to_int8_cpu(self.W)
                _ws = float(self._last_int8_scales[0])
                _gf = g.to(torch.float32) if g.dtype != torch.float32 else g
                _gmax = float(_gf.abs().max())
                _gs = (_gmax / 127.0) if _gmax > 0 else 1.0
                # ⚠ P155 **结论：这两步都省不掉**，我试过并实测推翻自己的假设：
                #   · `.to(torch.int8)` 对大值**回绕而非饱和**（1e6 → 64）→
                #     **clamp 必须保留**；
                #   · `.to(torch.int8)` 的舍入是 **round-half-away-from-zero**
                #     而 `torch.round` 是 **RNE（banker's）**：
                #     1.5 → to=2 / round=2✓；**2.5 → to=2 / round=2 ✓**；
                #     但 -2.5 → to=-2 / round=-2 ✓……仍有个别样本不同
                #     （300 组随机**每组都有差异**）→ 不能省。
                # → 保留 clamp+round+to 的原始写法。**记下来免得再试**。
                _Gq = torch.clamp(torch.round(_gf / _gs),
                                  -_INT8_QMAX, _INT8_QMAX).to(torch.int8)
                # int8 × int8 → int32 累加 → fp32 → 脱两个 scale
                # ⚠⚠ **P154 修正形状**：`matmul((n,k), (k,n))` 返回 **(n, n)**
                #   （矩阵乘的定义），而 `einsum("ij,ij->i")` 返回的是 **(n,)**
                #   （逐行内积）。二者语义不同！前者算的是 `W @ Gᵀ` 的**交叉**
                #   组合，不是同一行的内积。
                #   → int8 GEMM **无法直接表达逐行内积**：硬件 int8 GEMM 单元
                #     算的是 (n,n) 交叉矩阵，**不支持行内积**（P155 定案）。
                # ── **P155 优化：int16 乘 + int32 累加**────────────────────
                # 我P154 用的 `(W_i32 * G_i32).sum(1)` 语义正确但**慢1.9×**
                # （1b 档实测 21.8 → 11.5 ms），因为它物化 **两个 (n,k) int32
                # = 2×4×6.34 = 50.7 MiB**。
                # ⚠ 试过但**错误**的两条快路（实测都回绕，**不能用**）：
                #   · `torch.einsum("ij,ij->i", W8, G8)` 5.0 ms —— dtype 是
                #     **int8** → |sum| 上限 127·127·128 = 2.06e6 远超 int8 → 回绕；
                #   · `torch.sum(W8*G8, 1, dtype=int32)` 5.8 ms —— `W8*G8`
                #     **先在 int8 里算完**再提升，同样回绕（实测逐位不一致）。
                # ✅ **int16 乘 + int32 归约**：`W8.to(int16)*G8.to(int16)`
                #   的每个乘积 |≤ 127×127 = 16129 < int16 上限 32767 → **不回绕**，
                #   而 `sum(dtype=int32)` 的累加仍在 **int32**（社区标准做法）。
                #   实测 **11.5 ms，比 int32 版快 1.9×**（中间张量 2 字节而非 4）。
                #   ⚠ 上界：k ≤ 32767²/16129 = 65535 → 本项目 k ≤ 65535 **恒安全**。
                #     若将来 k 超过 65535，须改回 int32 或分块。
                _y32 = (torch.mul(_Wq.to(torch.int16),
                                  _Gq.to(torch.int16))
                        ).sum(dim=1, dtype=torch.int32)   # **int32 累加**
                self._int8_out_scale = 1.0        # 已在此脱掉
                self._last_int8_scales = (_ws, _gs)
                return _y32.to(torch.float32) * (_ws * _gs)
                # ⚠⚠ **P152 修正**：算出 scale 之后**必须把 scale 乘回结果**，
                #   否则 y 的量级差 1/_sc（实测 542×）→ nll 从 10.84 爆到 **1071**。
                #   社区的标准做法正是「**低精度乘 + 高精度累加 + 最后脱 scale**」。
                self._int8_out_scale = float(_sc)
                # ⚠⚠⚠ **P153 修正（fhz 追问暴露：P152 这条路径其实无效）**：
                #   上面只把 **W** 转成 int8，而 **g（激活）仍是 fp16**
                #   → `(_Wm * g)` 走的是 **fp16 matmul**，**根本没用到 int8 算子**，
                #   只是白白多了一次重量化（实测 nll 10.8376 → 10.8701，纯损失）。
                #   真正的 int8 GEMM 要求**两边都是 int8**（社区标准 W8A8：
                #   torchao `int8_weight_only` / TE int8 = int8 × int8 → int32）。
                # → 因此在下面 `int8_compute` 分支里**改用真 int8 GEMM**
                #   （W 与 g 各自量化 + int32 累加 + 脱两个 scale），见该分支。
                self._int8_p152_deprecated = True
            else:
                _Wm = self.W.to(self._cdtype)
            if self._sp_fwd == "einsum":
                # P116：不物化 (n_out,k) 中间张量（mulsum 每步两处各25.37 MiB）。
                # ⚠ 归约顺序与 mulsum 不同 → **非逐位**（本机 max|Δ|≈3e-05）。
                _y = torch.einsum("ij,ij->i", _Wm, g)
            else:
                _y = (_Wm * g).sum(dim=1)
            # P152：int8 计算域要**脱 scale**
            if self._int8_out_scale != 1.0:
                _y = _y * self._int8_out_scale
            return _y
        if self._int8:
            Wq = (self.W.to(torch.float32) * self._wscale).to(ht.dtype)
            return Wq @ ht
        if self._fp8_bits:
            # P189：uint8 位模式存储 → 查表反量化到 fp16 → fp16 GEMV
            # （910B 无 fp8 算子；fp16 matmul 是设备上真实可用的路径）
            return self._fp8_bits_to_real() @ ht
        return self.W @ ht

    def _fp8_to_int8_cpu(self, w_fp8):
        """fp8 存储的 W → int8 码本，**位数转换在 CPU 上**（P154）。

        为什么不在设备上转：fp8 的位布局要在 CPU 上按bit 拆解，
        而 torch 的设备算子会物化 fp32 中间张量（1b 档 3×25.4 MiB）。
        本实现  → D2H → CPU 核量化 → 回设备 int8。

        ⚠ **D2H/H2D 是同步点**（每步 2 次，6.34 MiB）。msprof 里应能看到
        这两笔；昇腾上它们可能比转换本身还贵 → **待实测**。

        P189：`--fp8-conv rust` 走 phdnet_rs 多核转换核（数值逐位一致），
        Rust 不可用时回落 torch 路径并告警一次（不静默）。
        """
        from .fp8_int8_convert import fp8_to_int8_codes
        if w_fp8.dtype == torch.float8_e4m3fn:
            _bits = w_fp8.view(torch.uint8)
        elif w_fp8.dtype == torch.uint8:
            _bits = w_fp8                       # P189：fp8-bits 模式原样
        else:
            _bits = w_fp8.to(torch.uint8)
        _host = _bits.cpu().numpy().reshape(-1)
        # conv 路径：torch（默认，快 5.1x）| nogil（0 中间张量 + 释放 GIL）
        #           | rust（P189：多核并行，见 fp8_conv.rs）
        _conv = str(getattr(self, "_fp8_conv", "torch"))
        if _conv == "rust":
            try:
                # P189：按 sparse_pc.py 的同款模式导入——先把 phdnet_rs/ 目录
                # 插到 sys.path 首位，让 `phdnet_rs` 解析到 **phdnet_rs.py
                # 绑定模块**（否则目录被当作 namespace package，load 拿不到）。
                import os as _os, sys as _sys
                _rs_dir = _os.path.join(
                    _os.path.dirname(_os.path.dirname(
                        _os.path.abspath(__file__))), os.pardir, "phdnet_rs")
                _rs_dir = _os.path.abspath(_rs_dir)
                if _rs_dir not in _sys.path:
                    _sys.path.insert(0, _rs_dir)
                from phdnet_rs import load as _rs_load
                _k, _why = _rs_load()
                if _k is not None and getattr(_k, "_has_fp8_conv", False):
                    _codes = np.empty(_host.size, dtype=np.int8)
                    _sc = np.zeros(1, dtype=np.float32)
                    _part = np.zeros(64, dtype=np.float32)   # ≥ n_parts 即可
                    _k.fp8_to_int8(_host, _codes, _sc, _part, 8)
                    self._last_int8_scales = (float(_sc[0]), 1.0)
                    return torch.from_numpy(_codes).to(self.device).reshape(
                        w_fp8.shape)
            except Exception:                        # noqa: BLE001
                pass
            import warnings as _wr
            if not getattr(self, "_fp8_conv_rust_warned", False):
                _wr.warn("[fp8-conv] rust 转换核不可用 → 回落 torch 路径"
                         "（--fp8-conv rust 未生效；phdnet_rs 未加载或缺"
                         " phdnet_fp8_to_int8 符号）", RuntimeWarning)
                self._fp8_conv_rust_warned = True
        _codes, _sc = fp8_to_int8_codes(_host, conv=_conv)
        self._last_int8_scales = (_sc, 1.0)
        return torch.from_numpy(_codes).to(self.device).reshape(
            w_fp8.shape)

    def _fp8_bits_to_real(self):
        """P189：uint8 位模式存储的 W → 实值 fp16 设备张量（256 项查表）。

        用于迭代（learn）与前向稠密臂的反量化。fp8 的值域由 dtype 语义
        决定（无 per-tensor scale），查表即精确反量化。

        ⚠⚠ **P191d 修正（服务器实测 313ms/tok 负优化的根因之一）**：
        首版实现 `self.W.cpu() → CPU 查表 → H2D`——**每步 2×6.65 MiB 的
        D2H/H2D 往返 + 全量 CPU 查表**，NPU 全程空转等 CPU。改为
        **设备端 gather**：LUT 常驻设备（1 KB），`LUT[bits]` 一次
        fancy-index kernel 完成，零主机交互。
        """
        lut = self._fp8_dev_lut()
        # ⚠ uint8 张量做索引会被 torch 当成 bool mask（歧义，实测 IndexError）
        # → 必须先转 int64 再 fancy-index。
        return lut[self.W.long()].to(self._cdtype)   # (n_out,k) 设备内查表

    def _fp8_dev_lut(self):
        """fp8 e4m3fn 位模式 → 实值 的 **设备端** 256 项 LUT（懒建，1 KB）。"""
        lut = getattr(self, "_fp8_lut_dev", None)
        if lut is None:
            from .fp8_int8_convert import FP8_VAL_LUT
            lut = torch.from_numpy(np.ascontiguousarray(FP8_VAL_LUT)).to(
                self.device)
            self._fp8_lut_dev = lut
        return lut

    def _real_to_fp8_bits(self, t_real):
        """P189：实值张量 → fp8 e4m3fn 位模式（uint8）写回存储。

        ⚠⚠ **P191d 重写（服务器实测 313ms/tok 负优化的根因之一）**：
        首版 `t.cpu() → .to(float8_e4m3fn) → .view(uint8) → H2D`——每步
        2×6.65 MiB D2H/H2D 往返 + CPU cast，且 910B 上 fp8 cast 本身
        ERR01007（P86：create/cast/matmul 全禁）。改为**设备端位运算核**：
        纯 int64 位操作向量化，零主机交互、不依赖任何 fp8 算子。

        数值语义（与 torch CPU cast **逐位对齐**，边界已实测探针确认）：
          · RNE（half-to-even）舍入到 3 位 mantissa；
          · |v| ≥ 448 → **饱和**到 0x7E（max finite，不进 NaN 槽 0x7F）；
          · 次正规 = man×2^-9（RNE 到整数 man∈0..8，man=8 自然进正规 0x08）；
          · ±0 带符号位（-0.0 → 0x80）。
        """
        t = t_real.detach().float()
        a = torch.nan_to_num(t, nan=0.0).abs()
        sign = torch.signbit(t).to(torch.int64) << 7      # -0.0 → 0x80 ✓
        a = torch.clamp(a, max=448.0)                     # 饱和语义（先夹再舍）
        u = a.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        # ── 正规数：f32 位 RNE 到 3 位 mantissa ──────────────────────────
        # 保留 bit22..20，在 bit19 上半进位 + guard=(保留段 LSB)（标准
        # RNE 加法技巧）；(un>>20) = f32 的 [sign|E|m3]，减 120<<3 = 960
        # 得 e4m3 复合码（e4m3 bias 7 vs f32 bias 127 → 差 120）。
        un = u + 0x7FFFF + ((u >> 20) & 1)
        comp_n = ((un >> 20) - 960).clamp(8, 0x7E)
        # ── 次正规：man = RNE(a×2^9)，man=8 自然等于正规 0x08 ────────────
        man = torch.round(a * 512.0).to(torch.int64)      # torch.round = RNE
        normal_mask = a >= 2.0 ** -6
        code = torch.where(normal_mask, comp_n, man).clamp(0, 0x7E)
        out = sign | code
        # NaN → 0x7F（NaN 槽，与 torch fp8 cast 逐位一致——对拍抓到的最后 1 例）
        return torch.where(torch.isnan(t),
                           torch.full_like(out, 0x7F), out).to(torch.uint8)

    def _sp_gather(self, ht):
        """稀疏行内 gather：取每行 k 个 h 分量（(n_out,k)），供前向与更新共用。

        缓存判据（**P112 修复，这里原先是性能 bug**）
        ----------------------------------------------
        第一版用 `torch.equal(self._cache_g_src, ht)` 判「是不是同一个 h」。
        `torch.equal` 逐元素比较后返回 **Python bool** → 在 NPU 上这是一次
        **强制设备同步**：CPU 必须等所有已入队 kernel 跑完才能拿到结果。等于把
        P34 刚用 `nll_sync_every` 消除掉的那个同步又加回来，且发生在**每步**的
        最热路径上（服务器日志实测：读出 7.68 ms/tok，占 43.5%）。

        改为**主机侧 epoch 标记**：每次上传新 h 就递增 `_ht_epoch`，
        `_sp_gather` 只比这个整数。语义等价（同一 epoch = 同一个 h），
        **零设备交互**。
        ⚠ 不能简化成「盲目复用缓存」——那会在 h 变化时静默用错 gather 结果。
        """
        # ⚠⚠ **P143 修复（msprof 实测定位，性能级）**：
        #  msprof（昇腾，600 步，op_summary）给出：
        #    aclnnIndex_IndexAiCore_Index  ×1200  占 **89.3%**（每步 13.11 ms）
        #    aclnnBatchMatMul_V2           × 600  占  9.6%
        #  → **每步 gather 被执行了两次**（1200 = 600×2），而它是全部开销的主项。
        #  实测复现（本机CPU臂）：`forward_dev` + `learn_softmax` 一步内
        #  `_sp_gather` 被调 **2 次**，`_cache_g_epoch=-1 != _ht_epoch`。
        #
        #  根因：`_lookup_ht` 命中缓存时（P122 加的）把 `_ht_epoch` **回退**到
        #  `_ht_epoch_at_cache`，但 `forward_dev` 走的是 `_to_dev`（**不是**
        #  `_staged_to_dev`），它递增 epoch 后**从不更新** `_ht_epoch_at_cache`
        #  → 该字段停在 -1 → 判据永远不成立 → 缓存形同虚设。
        #  （P122 修的是「不同 h 时用错 gather」；P143 是**同一 h 时重复 gather**。
        #    两者方向相反，必须同时成立。）
        #
        #  修法：判据改为「**ht 的内容身份**」而非「epoch 数字」——
        #  用 `ht` 的**对象身份**（`is`）判定同一个设备张量复用，这是**零设备
        #  交互**的（P112 消同步的要求），且不受epoch 数字怎么变的影响。
        #  跨调用时 `forward_dev` 的 `ht` 与 `learn_softmax` 的 `ht` 是**同一
        #  个张量对象**（`_cache_ht` 命中）→ 命中缓存。
        if (self._cache_g is not None
                and getattr(self, "_cache_g_src", None) is ht):
            return self._cache_g
        # ⚠⚠ **P148 修正一个「方向完全相反」的实现**（P145 写的）：
        #   原代码是 `g = ht[Wi].to(fp16).to(fp32)` —— gather **已经物化出
        #   fp32**（1b 档 25.37 MiB），再cast 成 fp16 又立刻转回 fp32，
        #   **两次 cast 各自读+写那 25.37 MiB** → 纯负优化。
        #   流量核算（1b 档 51962×128）：
        #     fp32（现状）              写 25.37 MiB
        #     **P145 的实现**  写 25.4 + 读25.4 + 写12.7 + 读12.7 + 写25.4
        #                          = **101.5 MiB（比现状多 76 MiB/步）**
        #     正确做法（本次）让 gather **直接产出 fp16** → 写出 12.69 MiB
        #   实测墙钟（P148 审计，本机）：Index 8.35 ms → 带往返 cast 14.83 ms
        #   （**慢 1.78×**），而 P145 的原意是省流量。
        #   → 现在把 cast **移到 gather 之前**（对 `ht` 做，而不是对 `g`）。
        if self._gather_dtype == "fp32":
            if self._gather_impl == "take":
                # P186：`_gather_impl` 此前**只存不用**（P134 接线断了）——
                # 热路径永远走下面的 `ht[self.Wi]`。这里补上真正的 take 路径：
                # `torch.take(ht, Wi)` 数值与 `ht[Wi]` **逐位相同**（P134 本机
                # x86 实测快 1.7×，昇腾待实测——可用 --readout-gather-impl A/B）。
                g = torch.take(ht, self.Wi)              # (n_out, k) fp32
            else:
                g = ht[self.Wi]                          # (n_out, k) fp32
        else:
            # 先把 **h 本身**（n_h 个数，极小）降到目标精度，**再 gather**
            # → `Index` 的**写出量**直接减半/减四，这是真正的省流量。
            _dt = {"fp16": torch.float16,
                   "bf16": torch.bfloat16}[self._gather_dtype]
            _ht_l = ht.to(_dt)
            # P186：take 路径对 fp16 gather 同样适用（数值逐位相同）
            g = (torch.take(_ht_l, self.Wi) if self._gather_impl == "take"
                 else _ht_l[self.Wi])
        self._cache_g = g
        self._cache_g_src = ht                # 记住张量**对象**（P143）
        self._cache_g_epoch = self._ht_epoch  # 保留供诊断
        return g

    def _int8_update(self, dp32, ht32, alpha: float) -> None:
        """（稠密 + 稀疏皆支持；P151 修了稀疏臂的形状错误。）"""
        """int8 权重的 rank-1 更新：**fp32 域**计算 dp⊗ht，再重量化写回。

        P105（fhz 2026-10-01 决策）：直接在 int8 码上原地加更新毫无意义——
        int8 步长 ≈ max|W|/127 ≈ 0.008，而 softmax 梯度里非目标行的 |dp|≈1e-6，
        加上去 round 回来码不变（更新被量化吃掉，这是 int8 存储降 4× 访存的
        **知情取舍**，不是 bug）；目标行 |dp|≈1 远大于步长，走 fp32 域更新 +
        重量化后完整保留（verifier [F] 用 fp16 参考对拍把关）。

        流程：dequant(W) → fp32 域 addmm_（rank-1 AXPY，同 P28 口径）→
        可选 w_clip（int8 的 clamp_ 对码张量无意义，改为**重量化前裁剪**）→
        round(RNE)/clamp(±127)/cast int8 写回。scale 固定不重标定（2× 余量）。
        """
        W_fp = self.W.to(torch.float32) * self._wscale
        W_fp.addmm_(dp32.reshape(-1, 1), ht32.reshape(1, -1), alpha=float(alpha))
        if self.w_clip > 0.0:
            W_fp.clamp_(-self.w_clip, self.w_clip)
        codes = torch.clamp(torch.round(W_fp / self._wscale),
                            -_INT8_QMAX, _INT8_QMAX)
        self.W.copy_(codes.to(torch.int8))

    def forward(self, h) -> np.ndarray:
        """前向并返回 **numpy**（`Readout.__call__` 契约）。

        同时把设备侧的 (h, y) 缓存下来，供紧随其后的 `learn_softmax` 复用
        （P28：生产热路径是 `y = readout(h)` → `learn_softmax(..., y_pre=y)`，
        若不缓存就要 D2H 289 KiB 再 H2D 传回，纯往返）。
        """
        # P112：这也是一次「新 h 上传」，必须递增 epoch —— forward 走的是
        # `_to_dev` 而**不是** `_staged_to_dev`（前者不做 pinned 暂存），
        # 只在 `_staged_to_dev` 里递增会让本路径的 epoch 恒定不变，
        # `_sp_gather` 于是把上一步的 gather 结果错误地复用给新 h。
        # （第一版遗漏此处 → verify_accel_sparse A1 立刻报数值不一致。）
        self._ht_epoch += 1
        ht = self._to_dev(h)
        y = self._matmul(ht)
        self._cache_h = np.ascontiguousarray(h, dtype=np.float32)
        self._cache_ht = ht
        self._ht_epoch_at_cache = self._ht_epoch   # P122：记下这份 ht 的 epoch
        # P143：`forward_dev` 是**生产热路径**（训练每步都走它），而
        # `_lookup_ht` 命中时会把 `_ht_epoch` 回退到 `_ht_epoch_at_cache`。
        # 二者必须一致，否则回退到一个陈旧值 → `_sp_gather`的判据失效
        # → **每步重复 gather**（msprof 实测：Index 占 89.3%、每步 13.11 ms）。
        self._cache_y = y
        return y.float().cpu().numpy()

    def __call__(self, h) -> np.ndarray:
        """与 `Readout.__call__` 同协议（model.step 里是 `self.readout(h)`）。

        P19 修复：早先只提供 `forward()`，导致 `TypeError: 'AccelReadout' object
        is not callable`（生产训练首个 step 即崩）。`Readout` 的调用面共 4 个：
        `__call__` / `learn_softmax` / `learn` / `W`，此处全部对齐。
        """
        return self.forward(h)

    def forward_dev(self, h):
        """设备内前向（返回**设备张量**，零同步）。

        P28：这是消除「训练热路径上 y 白 D2H 再白 H2D」的正解——调用方拿设备
        张量直接进 `learn_softmax`，整条链路上没有主机↔设备往返。
        P58：H2D 走 pinned 暂存 + non_blocking（pageable 拷贝会阻塞 CPU；
        异步上传让 CPU 提前回去算下一步的 M1–M5）。
        """
        ht = self._staged_to_dev(h)
        try:
            y = self._matmul(ht)
        except Exception as e:                              # noqa: BLE001
            # P90 兜底（P105 沿用）：int8 路径的**任何**运行期失败（matmul 无算子、
            # 反量化/传输不支持）都在这里被吸收 → 永久回落 fp16 后重算，训练不中断。
            # 服务器实测（2026-09-30 19:17）证明这条必需：fp8 能建张量但 matmul 无
            # 实现会崩在 `@`——只靠构造期探测不够。非 int8 模式无可回落 → 真错误照抛。
            if not self._int8:
                # P191c：fp8 **原生**模式的同款兜底——构造期探测可能假通过
                # （小张量建得出来），真实规模 matmul 才炸
                # （`addmv_impl_cpu not implemented for Float8_e4m3fn`，
                #   2026-10-07 服务器/CPU 实测）。→ 永久转 **fp8_bits 模式**
                # （uint8 位模式存储 + fp16 计算），**保持 fp8 请求语义**、
                # 训练不中断；非 fp8/int8 模式无可回落 → 真错误照抛。
                if not self._fp8_native:
                    raise
                self._fp8_native = False
                self._fp8_bits = True
                self.tdtype = torch.uint8
                self._cdtype = torch.float16
                self._int8 = False
                self._int8_like = False
                self._wscale = None
                self._fp8_wscale = None
                self._last_int8_scales = (1.0, 1.0)
                # fp8 张量 → uint8 位模式（fp8 的值域就是位模式，零损转置）
                self.W = self.W.view(torch.uint8) if self.W.dtype == \
                    torch.float8_e4m3fn else self.W.to(torch.uint8)
                import warnings
                warnings.warn(
                    f"fp8 原生前向在设备 {self.device} 上失败（{type(e).__name__}: "
                    f"{str(e)[:80]}）→ 永久转 fp8 位模式存储 + fp16 计算"
                    f"（P189 语义保持，训练不中断）", RuntimeWarning)
                ht = self._staged_to_dev(h)
                y = self._matmul(ht)
                self._cache_h = np.ascontiguousarray(h, dtype=np.float32)
                self._cache_ht = ht
                self._cache_y = y
                return y
            self._int8 = False
            _scale = self._wscale
            self._wscale = None
            self.tdtype = torch.float16
            self._cdtype = torch.float16
            # 保值反量化到 fp16 稠密主副本（codes*scale），此后走普通 fp16 路径
            self.W = (self.W.to(torch.float32) * _scale).to(torch.float16)
            import warnings
            warnings.warn(
                f"int8 前向在设备 {self.device} 上失败（{type(e).__name__}: "
                f"{str(e)[:80]}）→ 永久回落 fp16（训练不中断）", RuntimeWarning)
            ht = self._staged_to_dev(h)
            y = self.W @ ht          # 已回落 fp16 稠密 → 直接乘（勿再走 _matmul）
        self._cache_h = np.ascontiguousarray(h, dtype=np.float32)
        self._cache_ht = ht
        self._cache_y = y
        return y

    def _train_step_core(self, y32, ht, t32, correct: int, eta: float):
        """读出更新一体步（softmax→nll→addmm_），供 `torch.compile` 融合。

        P38：调用方已算好 `y32 = (W@h).float()`（前向缓存），这里**不重复
        matvec**（否则 W 读两次，抵消 P28 收益）。编译器融合 softmax/log/sub/
        addmm_ 四个小 kernel、摊薄 launch 开销；eager 路径不走这里。

        ⚠ P55（fhz 服务器 2026-09-29 NPU 首次真跑即崩）：P45 把「target_idx
        路径就地改 p」实现在了 **eager 分支**，本融合核仍是旧的
        `dp = p - t32` → `t32=None` 时 `FakeTensor - None` 崩 dynamo。两条路径
        现在**逐行等价**（数值也等价：p − onehot ≡ p[c] −= 1）。
        """
        p = torch.softmax(y32, dim=0)
        nll_dev = -torch.log(p[correct:correct + 1] + 1e-12).reshape(())
        if t32 is None:                  # P45 就地改 p（target_idx 路径）
            dp = p.to(self.tdtype)
            dp[correct] -= 1.0
        else:
            dp = (p - t32).to(self.tdtype)
        self.W.addmm_(dp.reshape(-1, 1), ht.reshape(1, -1), alpha=-float(eta))
        if self.w_clip > 0.0:
            self.W.clamp_(-self.w_clip, self.w_clip)
        return nll_dev

    # ---------- 学习（softmax 感知器，局部梯度 p − t）----------
    def learn_softmax(self, h, target, eta: float, y_pre=None,
                      accumulate: int = 1, target_idx: int | None = None) -> float:
        """P6 感知器更新（softmax 交叉熵的局部梯度 ∂L/∂y = p − t）。

        P28 的三处性能修正（数值语义不变，仍属「容差一致」）：
          ① `W.addmm_(dp⊗ht, alpha=-eta)` 取代 `W.add_(torch.outer(dp, ht))`——
             后者会**物化一个与 W 同尺寸的临时张量**（1B 档 867 MiB），再被 add_
             读回来 → 单步多出 1.73 GiB 设备带宽（实测占总流量 40%）。addmm_ 是
             一步 rank-1 AXPY，不产生临时张量。
          ② y/h 复用：优先用 `forward`/`forward_dev` 缓存的设备张量，省掉
             「y D2H 289 KiB → H2D 传回」与 h 的重复上传。
          ③ 同步点 3 → 1：`correct` 在主机侧算（target 本来就在主机，零设备交互），
             nll 合并成唯一一次 `.item()`。
        """
        ht, cache_hit = self._lookup_ht(h)
        # y_pre 既可能是设备张量（推荐路径）、numpy（旧接口），或 None（自算）
        if y_pre is None:
            y = self._matmul(ht)     # P105：int8 模式每步反量化
        elif torch.is_tensor(y_pre):
            y = y_pre if y_pre.device == ht.device else y_pre.to(ht.device)
        elif cache_hit and self._cache_y is not None:
            # 缓存命中 = 本次 h 与 forward 时的 h 相同 → y_pre 就是那次的前向结果
            # （生产热路径 forward → learn_softmax 的形状），直接复用设备张量，
            # 省掉「D2H 289 KiB + H2D 传回 + 一次硬同步」。
            y = self._cache_y
        else:
            y = torch.as_tensor(np.ascontiguousarray(y_pre, dtype=np.float32),
                                device=ht.device, dtype=self._cdtype)
        y32 = y.float()
        # P40：target 两种来源——`target_idx`（int，推荐）在**设备上**构造 onehot，
        # 省掉每步 289 KiB 的 host onehot 构造 + H2D；`target`（host 数组）为
        # 兼容旧接口保留。
        if target_idx is not None:
            # P45：直接把 p 变成 dp（p[c] -= 1），**不新建 zeros_like 缓冲**
            # ——后者每步多一次 289 KiB 设备分配 + scatter kernel，比原 H2D 更贵。
            correct = int(target_idx)
            t = None                      # 标记：走「就地改 p」路径
        else:
            t = self._to_dev(target).float()
            # correct 在主机侧求（target 是 host 数组）——省一次 .item() 硬同步
            correct = int(np.argmax(np.asarray(target)))
        # nll：P34 双模式。sync_every=1 时每步 .item()（旧行为）；N>1 时
        # 累积到设备标量、每 N 步同步一次——中间步返回滞后均值，训练热路径
        # 上 CPU 与 NPU 从此可以重叠（这是 NPU 上最大的延迟收益）。
        # P38：`_compiled` 时 softmax/nll/addmm_ 四个 kernel 交给编译器融合
        #（eager 路径逐算子执行，语义一致）
        # P55：softmax/nll 只在 eager 路径算——compiled 路径由融合核内部算，
        # 此处重算会白付一次 51962 元素 softmax + 一次 log（纯浪费）。
        # P55 加固：融合核**首次执行**才真正触发 inductor 编译（构造期
        # torch.compile() 只是包装，不会失败）——编译失败若不捕获，生产训练
        # 直接崩（Windows 无 cl / NPU 上 inductor 异常均可能发生）。按 P19
        # 「回落 + 记原因」纪律：首次失败即**永久回落 eager** 并告警。
        if self._compiled:
            try:
                nll_dev = self._fused(y32, ht, t, correct, float(eta))
            except Exception as e:                        # noqa: BLE001
                import warnings
                warnings.warn(f"torch.compile 融合核执行失败，永久回落 eager"
                              f"（原因：{type(e).__name__}: {e}）",
                              RuntimeWarning)
                self._compiled = False
                self._fused = None
                nll_dev = self._eager_step(y32, ht, t, correct, eta)
        else:
            nll_dev = self._eager_step(y32, ht, t, correct, eta)
        if self.nll_sync_every <= 1:
            return float(nll_dev.item())
        self._nll_sum = (nll_dev if self._nll_sum is None
                         else self._nll_sum + nll_dev)
        self._nll_n += 1
        if self._nll_n >= self.nll_sync_every:
            self._last_nll = float(self._nll_sum.item()) / self._nll_n
            self._nll_sum = None
            self._nll_n = 0
        return self._last_nll

    def _eager_step(self, y32, ht, t, correct: int, eta: float):
        """eager 逐步执行（softmax→nll→addmm_）；compiled 回落时也走这里。

        P80：nll 改用 `cross_entropy` 单 kernel（替代 `softmax + log + 索引`
        的多 kernel 组合，并去掉 1e-12 加项）。数学等价：CE(logits, c) =
        logsumexp(y) − y[c] = −log(softmax(y)[c])。dp 仍由同一份 softmax 得出。
        """
        p = torch.softmax(y32, dim=0)
        # nll 先取（必须在改 p 之前）
        # P188：`correct` 本来就在主机侧（target_idx 是 int / target 是 host 数组
        # 的 argmax）——旧实现却把它 `fill_` 进 pinned 张量再 H2D 上传，
        # 每步多付 2 次 kernel 交互（fill + copy）+ 1 次 H2D，正是 msprof
        # 「NPU 空转等 CPU」的 launch 开销成分。改用 CE 的恒等展开
        #   cross_entropy(y, c) = logsumexp(y) − y[c]
        # 纯设备内标量运算、零主机交互，数值恒等（同 fp32 域）。
        nll_dev = (torch.logsumexp(y32, dim=0)
                   - y32[int(correct)]).reshape(())
        if self._int8:
            # P105：int8 模式的更新走 **fp32 域** dp⊗ht + 重量化写回（精度约束
            # 与知情取舍见 _int8_update 的 docstring）。p 已用完（nll 先取），
            # clone 后就地改目标行，语义与 P45「p[c] -= 1」一致。
            dp32 = p.clone() if t is None else (p - t)
            if t is None:
                dp32[correct] -= 1.0
            # P151：稀疏臂**不能**调 `_int8_update` —— 它用
            #   `ht.reshape(1, -1)` 假设 `ht` 是 (1, n_h)，而稀疏臂的 `ht`
            #   是 **(n_h,)** →  `Bad in-place call: input [8192,128] vs
            #   output [8192, 1024]`（实测真实规模必崩）。
            #   → 稀疏臂复用 P149 为 fp8 写的那条路：gather → fp16 域
            #     addcmul_ → 重量化写回。语义与稠密一致。
            if self._sparse:
                _g = self._sp_gather(ht.to(self._cdtype))
                # ⚠⚠ **P151 修正**：`self.W.to(_cdtype)` 只把**码值**转成
                #   浮点，**没有乘 `_wscale`** → 权重全成了 ±127 附近的整数
                #   → 实测 nll **540**（对照 fp8 的 10.8，彻底坏掉）。
                #   正确：先乘 scale 反量化到真实值域，再更新、再重量化。
                _Wq = self.W.to(self._cdtype) * float(self._wscale)
                _Wq.addcmul_(dp32.reshape(-1, 1).to(self._cdtype), _g,
                             value=float(eta))
                if self.w_clip > 0.0:
                    _Wq.clamp_(-self.w_clip, self.w_clip)
                _sc = float(self._wscale) or 1.0
                _codes = torch.clamp(
                    torch.round(_Wq.to(torch.float32) / _sc),
                    -_INT8_QMAX, _INT8_QMAX)
                self.W.copy_(_codes.to(torch.int8))
            else:
                self._int8_update(dp32, ht.float(), -eta)
            return nll_dev
        # P84：更新主副本是 fp16（低精度回落场景）→ dp/ht 必须同 dtype，否则
        # addmm_ 退回慢路径或直接报错。int8 模式已在上面提前返回，不会到这里。
        # ⚠⚠ **P149（fhz「模型原生 fp8，迭代 fp16」）**：`_upd_dtype` 原本在
        #   `W.dtype` 非 fp16 时直接取 `self.tdtype` —— 而 fp8 模式下 `tdtype`
        #   就是 fp8 → dp/ht 也被转成 fp8 → 算子全部崩
        #   （"Promotion for Float8 Types is not supported"）。
        # → 正确：**计算精度一律用 `_cdtype`**（fp8 存储时它 = fp16），
        #   **存储精度**才是 `tdtype`。两者必须分开判。
        # P151：4-bit（`tdtype = uint8`）**也必须用 fp16 更新域**——
        #   否则 `dp` 会变成 uint8 → `dp[correct] -= 1.0` 报
        #   "result type Float can't be cast to the desired output type Byte"。
        _upd_dtype = (torch.float16
                      if self.W.dtype in (torch.float16,
                                          torch.float8_e4m3fn,
                                          torch.uint8)
                      else self.tdtype)
        if t is None:                     # P45：p 就地变成 dp（p − t）
            dp = p.to(_upd_dtype)
            dp[correct] -= 1.0
        else:
            dp = (p - t).to(_upd_dtype)
        if ht.dtype != _upd_dtype:
            ht = ht.to(_upd_dtype)
        if getattr(self, "_fp8_bits", False):
            # P189（fhz「模型 fp8，迭代 fp16」）：fp8 位模式（uint8 承载）存储
            # 的更新走 fp16 域：查表反量化 → fp16 addcmul_/addmm_（dp⊗h rank-1，
            # 不物化临时张量）→ RNE cast 回 fp8 位模式写回。与 int8 的
            # 「反量化→更新→重量化」同构；P110 代价照旧（非目标行更新大多
            # 被舍入丢弃，目标行 |dp|~1 完整保留）。
            # ⚠ 位置：必须在 dp/ht 计算之后（P191c 修正——首版插在 dp 前
            #   → UnboundLocalError，2026-10-07 服务器实测炸过）。
            _real = self._fp8_bits_to_real()              # fp16 (n_out,k)
            if self._sparse:
                _g = self._sp_gather(ht)
                _real.addcmul_(dp.to(self._cdtype).reshape(-1, 1),
                               _g.to(self._cdtype), value=-float(eta))
            else:
                _real.addmm_(dp.to(self._cdtype).reshape(-1, 1),
                             ht.to(self._cdtype).reshape(1, -1),
                             alpha=-float(eta))
            if self.w_clip > 0.0:
                _real.clamp_(-self.w_clip, self.w_clip)
            self.W.copy_(self._real_to_fp8_bits(_real))
            return nll_dev
        if self._sparse:
            # P111 稀疏 rank-1：W[i,j] -= η·dp[i]·h[Wi[i,j]]
            # `addmm_` 是稠密 (n_out,n_h) 的；稀疏下等价写法是 gather 后逐行
            # 外积累加，流量只有 (n_out,k) 而非 (n_out,n_h)。
            g = self._sp_gather(ht)
            # ⚠⚠ **P138：这里曾用 `W.add_(dp.reshape(-1,1) * g, ...)`，它会
            # **物化一个 (n_out, k) 的临时张量**（1b 档 52,642×128 = 6,738,176
            # 元素 = **25.4 MiB**）。P28 早就明确禁止过这个写法（记录：物化版单步
            # 多 1.73 GiB 设备流量、占总流量 40%），稠密臂用的正是
            # `W.addmm_(dp.reshape(-1,1), ht.reshape(1,-1))` —— **稀疏臂漏改了**。
            #
            # 正确写法：稀疏下 `ht` 是 gather 前的 (n_h,)，没有对应的
            # (1, n_out) 矩阵可做 addmm_ → 用 **rank-1 AXPY 逐行**实现，
            # 数值逐位相同（`W[i,j] -= eta·dp[i]·g[i,j]`）且**不物化**：
            #     W[i, j] -= eta * dp[i] * g[i, j]
            # 用 `addcmul_`（W 是 torch 张量，支持逐元素 + 张量系数）。
            # ⚠ 数值契约：`add_(alpha*x)` 与 `addcmul_` 的**乘法次序**不同
            #   （前者 `(dp[i]*g)*alpha`、后者 `dp[i]*g` 先算再整体乘），
            #   fp32 下**可能差 1 ulp**。故下方 verifier 会用容差判据，
            #   而非宣称逐位。**若必须逐位**，退回 `add_`（但那带物化）。
            # ⚠⚠ **P149**：fp8/int8 存储时`self.W` 不能直接参与算子
            #   （torch 不做 fp8 的隐式提升）。→ **显式反量化到 `_cdtype`**，
            #   在 fp16 域算完再**重量化写回** fp8（保持 4× 存储收益）。
            #   这一步与 P105 的 `_int8_update`（稠密路径）是同一套语义。
            if self._int4:
                # P151：4-bit 更新 = unpack → fp16 域 addcmul_ → 重pack 写回
                _lut = (_FP4_E2M1_LUT
                        if self._dtype_requested == "fp4" else _INT4_LUT)
                _fam = ("fp" if self._dtype_requested == "fp4" else "int")
                _Wq = _unpack4(self.W, getattr(self, "_w4_cols",
                                               self.W.shape[1]),
                               _lut, float(self._wscale or 1.0),
                               device=self.device)
                _Wq.addcmul_(dp.reshape(-1, 1).to(self._cdtype),
                             g.to(self._cdtype), value=-float(eta))
                if self.w_clip > 0.0:
                    _Wq.clamp_(-self.w_clip, self.w_clip)
                # 重pack（`_pack4` 是 numpy 路径 → 一次 D2H；见 P151 代价说明）
                _npW = _Wq.detach().float().cpu().numpy()
                _pk, self._wscale = _pack4(_npW, _lut, family=_fam)
                self.W.copy_(torch.tensor(
                    _pk.reshape(self.W.shape), device=self.device,
                    dtype=torch.uint8))
                return nll_dev
            if self._int8_compute and self.W.dtype == torch.float8_e4m3fn:
                #⚠ P154：**fp8 存储 + int8 计算**的更新路径。
                #   必须与前向同源（都用 `_fp8_to_int8_cpu` 的转换结果），
                #   否则前向走 int8、更新走另一套 → 形状/精度都不一致。
                # ⚠⚠ **torch 不支持 `fp8_tensor.to(torch.float32)`**（P149 在
                #   `_matmul` 踩过：报 "Promotion for Float8 Types is not
                #   supported"），所以**必须经 uint8 位模式**走 CPU 核反量化。
                _gi = self._sp_gather(ht.to(self._cdtype))
                # 真实值域 = int8 码 × scale（_fp8_to_int8_cpu 已算出 scale）
                self._fp8_to_int8_cpu(self.W)              # 更新 _last_int8_scales
                _ws2 = float(self._last_int8_scales[0]) or 1.0
                _Wreal = (self.W.view(torch.uint8).cpu().numpy()
                          .astype(np.float32).reshape(self.W.shape) * _ws2)
                _Wt = torch.from_numpy(_Wreal).to(self.device)
                _Wt.addcmul_(dp.reshape(-1, 1).to(torch.float32),
                             _gi.to(torch.float32), value=-float(eta))
                if self.w_clip > 0.0:
                    _Wt.clamp_(-self.w_clip, self.w_clip)
                # 重量化回 fp8（**除以 scale**，fp8 的 dtype 本身就是值域）
                self.W.copy_((_Wt / _ws2).to(torch.float8_e4m3fn))
                return nll_dev
            if self.W.dtype == self._cdtype:
                self.W.addcmul_(dp.reshape(-1, 1), g, value=-float(eta))
                if self.w_clip > 0.0:
                    self.W.clamp_(-self.w_clip, self.w_clip)
                return nll_dev
            _Wq = self.W.to(self._cdtype)          # 反量化（fp8 → fp16）
            # 形状：`_Wq` 与 `g` 都是 (n_out, k)，而 `dp.reshape(-1, 1)` 是
            # (n_out, 1) → 广播逐行，**这是 P138 的 addcmul_ 形式**，
            # **不物化 (n_out, k) 临时张量**（P138 当初就是为了禁掉那个物化）。
            # ⚠ P153 记录：调试此路径时曾临时改成 `add_(dp.reshape(-1,1) * g)`，
            #   那会物化 25.4 MiB 临时 → 门禁 C4e1 立刻报红，已回退。
            _Wq.addcmul_(dp.reshape(-1, 1).to(self._cdtype),
                         g if g.dtype == self._cdtype else g.to(self._cdtype),
                         value=-float(eta))
            if self.w_clip > 0.0:
                _Wq.clamp_(-self.w_clip, self.w_clip)
            # 重量化写回存储精度。⚠ fp8 用 torch 的 dtype cast（其舍入由
            #   dtype 语义决定）；int8沿用 P105 的 ±127 clamp + round(RNE)。
            if self.W.dtype == torch.int8:
                _sc = float(getattr(self, "_wscale", 1.0)) or 1.0
                _codes = torch.clamp(torch.round(_Wq.to(torch.float32) / _sc),
                                     -_INT8_QMAX, _INT8_QMAX)
                self.W.copy_(_codes.to(torch.int8))
            else:
                self.W.copy_(_Wq.to(self.W.dtype))
            return nll_dev
        self.W.addmm_(dp.reshape(-1, 1), ht.reshape(1, -1), alpha=-float(eta))
        if self.w_clip > 0.0:
            self.W.clamp_(-self.w_clip, self.w_clip)
        return nll_dev

    def _lookup_ht(self, h):
        """复用缓存的设备 h（`forward` 刚上传过同一个 h 时省一次 H2D）。

        返回 (设备张量, 是否命中缓存)。h 很小（n_h 个 fp32），逐元素比较成本
        可忽略，换来的是每步少一次 H2D + 一次分配。

        ⚠⚠ **P122 修复（审计抓到的静默数值错误）**：命中缓存时**必须同时把
        `_ht_epoch` 回退到该 ht 上传时的那一个**，否则 `_sp_gather` 的 epoch
        判据会错判。

        原 bug 的调用序列（实测复现，`max|Δ|` 不是舍入级而是整个 gather 用错）：
            ro.forward(hA)              # epoch=1，缓存 gather=G(hA)
            ro.learn_softmax(hB, ...)   # 未命中 → _staged_to_dev → epoch=2，
                                        #   gather 缓存变成 G(hB)
            ro.learn_softmax(hA, ...)   # ← 命中缓存，返回**旧 epoch 的 htA**，
                                        #   但 epoch 停在 2 → 判据 2==2 命中
                                        #   → 用 G(hB) 去更新 hA 的输出。
        生产热路径（每步 forward_dev 紧接 learn_softmax 同一 h）碰不到，
        但任何「一次 forward + 多次不同 h 学习」的调用方会踩到，
        且失败模式是**静默的数值错误**（形状/dtype 全对）。

        修法：用 `_ht_epoch_at_cache` 记录「当前 `_cache_ht` 上传时的 epoch」，
        命中时把 `_ht_epoch` 置回它—— 这样 `_sp_gather` 的判据重新成立。
        """
        if self._cache_ht is not None and self._cache_h is not None:
            ha = np.ascontiguousarray(h, dtype=np.float32)
            if ha.shape == self._cache_h.shape and np.array_equal(ha, self._cache_h):
                # 命中：epoch 回退到「这份 ht 上传时」的值，使 gather 判据有效
                self._ht_epoch = getattr(self, "_ht_epoch_at_cache",
                                         self._ht_epoch)
                # P143：若这份 ht 与上次 gather 用的是**同一个张量对象**
                # （生产热路径就是这样：`_cache_ht` 命中返回同一对象），
                # 把 gather 缓存的 epoch 也对齐 → 下游 `_sp_gather` 命中缓存，
                # **省掉每步 13.11 ms 的重复 Index**。
                # 反之（不同对象，即使内容相同）**必须不命中**，
                # 否则 P122 那个「用错 gather」的静默错误会回来。
                if self._cache_ht is not None:
                    self._cache_g_epoch = self._ht_epoch \
                        if getattr(self, "_cache_g_src", None) is self._cache_ht \
                        else getattr(self, "_cache_g_epoch", -1)
                return self._cache_ht, True
        return self._staged_to_dev(h), False

    # ---------- 与 numba Readout 的接口兼容 ----------
    def learn(self, h, target, eta: float) -> None:
        """非 softmax 感知器更新：**W += η·(t − y) ⊗ h**（对齐生产实际路径）。

        P19 踩坑记录（两处符号不一致，已统一）：
        - `Readout.learn` 走 numba 融合核时是 `W += η·(t−y)⊗h`（dp = y−t，核内取负）；
        - 它的 numpy 回退分支却写成 `W − η·(t−y)⊗h` —— **与融合核符号相反**
          （P7 遗留，只在融合核不可用时暴露）。本实现以**融合核**（生产默认）为
          准；同时已把该回退分支改正，两条路径现语义一致。
        """
        ht, _ = self._lookup_ht(h)
        y = self._matmul(ht)
        t = self._to_dev(target)
        if self._int8:
            # P105：int8 模式 → fp32 域更新 + 重量化写回（语义同 learn_softmax）
            self._int8_update(t.float() - y.float(), ht.float(), eta)
            return
        if self._sparse:
            # P111 稀疏 rank-1（见 _eager_step 的同款实现）
            g = self._sp_gather(ht)
            self.W.add_((t - y).reshape(-1, 1) * g, alpha=float(eta))
            if self.w_clip > 0.0:
                self.W.clamp_(-self.w_clip, self.w_clip)
            return
        # P28：rank-1 AXPY（W += η·(t − y)⊗h），不物化 (n_out×n_in) 临时张量
        self.W.addmm_((t - y).reshape(-1, 1), ht.reshape(1, -1),
                      alpha=float(eta))
        if self.w_clip > 0.0:
            self.W.clamp_(-self.w_clip, self.w_clip)

    def n_synapses(self) -> int:
        """读出的「连接数」：稀疏 =实际存在的突触；稠密 = 元素数。"""
        return int(self.W.numel())

    # ---- 与 Readout 的属性/方法面对齐（P23：三次崩溃的根治）----
    # 全仓库对 `readout.X` 的访问面（见 tests/verifiers/verify_accel_readout.py
    # 的接口扫描用例）：W / learn_softmax / learn / __call__ / n_synapses /
    # **conn_k** / **hidden** / **stats()**。前五项首版已实现，conn_k（ckpt 保存
    # 会访问）与 stats（诊断脚本会访问）缺失 → 生产保存检查点时崩溃。
    @property
    def _csr(self):
        """稀疏 CSR 三元组 (indptr, idx, val) —— 与 `Readout._csr` 同口径。

        P111：检查点保存/恢复（`ckpt_1b`）直接吃这个三元组，故必须导出标准
        CSR 布局而不是内部的 (n_out,k) 张量。
        ⚠ **val 返回的是主机副本**（设备张量不能原地写）。恢复侧
        （`ckpt_1b` 的 `val[:] = ...`）写的是这个 numpy 数组 → 写完不会自动
        同步回设备张量，故另提供 `sync_csr_from_host()` 并在 ckpt 恢复后调用
        （已在 ckpt_1b 接入，见该文件稀疏分支）。
        """
        if not self._sparse:
            raise NotImplementedError(
                "稠密读出没有 CSR 结构（conn_k=0；本属性仅稀疏模式可用）")
        k = self.conn_k
        n_out = self.n_out
        indptr = np.arange(0, (n_out + 1) * k, k, dtype=np.int64)
        idx = self.Wi.detach().cpu().numpy().astype(np.int64).ravel()
        # ⚠ 必须**记住这次导出的数组**：ckpt 恢复侧做的是 `val[:] = z[...]`
        # （原地写这个 numpy 数组），随后才调`sync_csr_from_host()`。
        # 若 sync 里重新 D2H 取一份新拷贝，写入就落在无人引用的临时数组上，
        # 设备权重从未被更新——恢复「成功」但权重还是旧值（B3 抓到的就是这个）。
        # P189：fp8-bits 模式下 uint8 是位模式 → 查表反量化成实值再导出。
        if getattr(self, "_fp8_bits", False):
            from .fp8_int8_convert import FP8_VAL_LUT
            val = FP8_VAL_LUT[self.W.detach().cpu().numpy().reshape(
                -1)].astype(np.float64)
        else:
            val = self.W.detach().float().cpu().numpy().astype(
                np.float64).ravel()
        self._csr_val_host = val
        return indptr, idx, val

    def sync_csr_from_host(self) -> None:
        """把**上次 `_csr` 导出**的主机 val 写回设备张量（ckpt 恢复后必须调用）。

        ⚠ 不能在这里重新 `_csr`：那会取一份全新的 D2H 拷贝，调用方之前对
        `val[:]` 的原地写入就丢了。必须复用缓存的那一份。
        """
        if not self._sparse:
            return
        val = self._csr_val_host
        if val is None:
            # 没有可用的导出副本 → fail-fast 而不是静默什么都不做
            # （静默 no-op 会让「检查点恢复成功但权重没变」变成隐形 bug）。
            raise RuntimeError(
                "sync_csr_from_host() 在 _csr 之前被调用；请先取 _csr、写入 "
                "val[:]，再调用本方法（ckpt_1b 的稀疏恢复分支已按此顺序接入）")
        if getattr(self, "_fp8_bits", False):
            # P189：实值 → fp8 位模式（uint8）写回
            _t8 = torch.as_tensor(
                np.ascontiguousarray(val, dtype=np.float32).reshape(
                    self.n_out, self.conn_k),
                device="cpu", dtype=torch.float32).to(torch.float8_e4m3fn)
            self.W.copy_(_t8.view(torch.uint8).to(self.device))
            self._csr_val_host = None
            return
        self.W.copy_(torch.tensor(
            np.ascontiguousarray(val, dtype=np.float32).reshape(
                self.n_out, self.conn_k),
            device=self.device, dtype=torch.float32))
        self._csr_val_host = None

    @property
    def conn_k(self) -> int:
        """每输出单元入边数（0 = 稠密）。稀疏模式下为实际 k。"""
        return int(self._sparse_k)

    @property
    def hidden(self) -> int:
        """加速后端只实现单级读出（两级在 pick_readout_backend 回落）。"""
        return 0

    @property
    def dtype_name(self) -> str:
        if getattr(self, "_fp8_bits", False):
            return "fp8"           # P189：位模式存储，口径报请求的 fp8
        if self._int8:
            return "int8"          # 存储口径（回落 fp16 后 _int8=False → 走下面）
        return {torch.float32: "fp32", torch.float16: "fp16",
                torch.bfloat16: "bf16"}.get(self.W.dtype, str(self.W.dtype))

    def stats(self) -> dict:
        """统计信息（键与 `Readout.stats` 对齐，供诊断/表格使用）。"""
        n = self.n_synapses()
        dense = self.n_out * self.n_h
        st = {"synapses": n, "dense_equivalent": dense,
              "connectivity": n / dense if dense else 1.0,
              "k": self.conn_k, "dtype": self.dtype_name,
              "storage_MB": self.W.numel() * self.W.element_size() / 1e6}
        if self._sparse:
            # P111：稀疏存储还含列索引（int64，设备常驻）。单报 W 会让诊断表
            # 低估真实显存/内存占用 —— 这正是「稀疏省内存」最容易被高估的地方。
            st["index_MB"] = (self.Wi.numel()
                              * self.Wi.element_size() / 1e6)
            st["storage_MB_total"] = st["storage_MB"] + st["index_MB"]
            st["layout"] = "csr-uniform-k"
        return st

    def W_cpu(self) -> np.ndarray:
        """权重拉回主机（检查点 / 统计用）。int8 模式返回**反量化后的实值**。

        P111 稀疏：返回 **(n_out, k) 的 val 矩阵**（不是 (n_out, n_h) 稠密）——
        稠密化会凭空造出 n_out×(n_h−k)/2 个不存在的突触，语义错。ckpt 的稀疏
        分支走 `_csr`，不经过本方法。
        """
        W = self.W.detach().float()
        if self._int8:
            W = W * self._wscale   # 码本值 × scale = 实值（否则 ckpt 全是码）
        elif getattr(self, "_fp8_bits", False):
            # P189：uint8 是 fp8 位模式，不是实值 → 查表反量化成实值
            # （否则 ckpt 存的是位模式整数，恢复/统计全错）
            from .fp8_int8_convert import FP8_VAL_LUT
            _bits = self.W.detach().cpu().numpy().reshape(-1)
            W = torch.from_numpy(FP8_VAL_LUT[_bits].reshape(
                self.W.shape))
        return W.cpu().numpy()

    def load_W(self, arr: np.ndarray) -> None:
        """从主机矩阵灌入（检查点恢复；形状须匹配）。int8 模式重量化。"""
        a = np.ascontiguousarray(arr, dtype=np.float32)
        if self._sparse:
            # P111 稀疏：期望 (n_out, k) 的 val 矩阵
            if a.shape != (self.n_out, self.conn_k):
                raise ValueError(f"稀疏形状不符：期望 {(self.n_out, self.conn_k)}，"
                                 f"收到 {a.shape}")
            if getattr(self, "_fp8_bits", False):
                # P189：实值 → fp8 位模式（uint8）写回
                _t8 = torch.as_tensor(a, device="cpu",
                                      dtype=torch.float32).to(torch.float8_e4m3fn)
                self.W.copy_(_t8.view(torch.uint8).to(self.device))
                return
            self.W.copy_(torch.as_tensor(a, device=self.device,
                                         dtype=torch.float32))
            return
        if a.shape != (self.n_out, self.n_h):
            raise ValueError(f"形状不符：期望 {(self.n_out, self.n_h)}，"
                             f"收到 {a.shape}")
        if self._int8:
            # P105：实值 → 码本（同一 scale 语义；越界 clamp 饱和，不回绕）。
            # scale 不随灌入的 W 重标定——保持与训练期同一把尺子。
            t = torch.as_tensor(a, device=self.device, dtype=torch.float32)
            self.W.copy_(torch.clamp(torch.round(t / self._wscale),
                                     -_INT8_QMAX, _INT8_QMAX).to(torch.int8))
            return
        self.W.copy_(torch.as_tensor(a, device=self.device, dtype=self.tdtype))

    # ---------- 内部 ----------
    def _to_dev(self, x):
        # P105：matmul 参与者的 dtype 跟随 **_cdtype**（int8 模式 = fp16，
        # 绝不能是 int8——ht 是实值向量，跟码本 dtype 无关）。
        if torch.is_tensor(x):
            return x.to(device=self.device, dtype=self._cdtype)
        return torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32),
                               device=self.device, dtype=self._cdtype)

    def _staged_to_dev(self, h):
        """P58（fhz「CPU 预计算还要更加提前」）：pinned 暂存 + non_blocking H2D。

        pageable H2D 会阻塞 CPU 直到拷贝完成——CPU 算完 h 后干等传输。pinned
        暂存让 H2D 真异步：CPU 提交后立即回去算下一步的 M1–M5，NPU 流按 FIFO
        消化（设备内再 cast 到 _cdtype——int8 模式下是 fp16，与原 host-cast
        语义同为 RNE 舍入）。
        缓冲池轮转 + Event 覆写保护（pinned 复用前必须确认上次拷贝已完成；
        每步 CPU 几 ms ≫ H2D 几 µs，实际零等待）。pin 不可用（CPU-only torch
        / 驱动限制）时回落同步路径，数值逐位一致。

        P112：**每次上传新 h 都递增 `_ht_epoch`**（主机侧整数，零设备交互），
        供 `_sp_gather` 的缓存判据使用 —— 取代原先那个会强制设备同步的
        `torch.equal`。
        """
        self._ht_epoch += 1
        # P122：本函数**不**更新 `_cache_ht`（它是"取ht"而非"缓存 ht"），
        # 故不动 `_ht_epoch_at_cache`——缓存那份 ht 的 epoch 仍有效。
        a = np.ascontiguousarray(h, dtype=np.float32)
        if self._pin_bufs is None:                      # 懒初始化（一次）
            self._pin_bufs = []
            try:
                probe = torch.empty(8, pin_memory=True)
                del probe
                self._pin_ok = True
            except Exception:                           # noqa: BLE001
                self._pin_ok = False
        if not self._pin_ok:
            return torch.as_tensor(a, device=self.device, dtype=self._cdtype)
        idx = self._pin_next % self._pin_cap
        if len(self._pin_bufs) <= idx:                  # 首轮：扩池
            self._pin_bufs.append((torch.empty(self.n_h, dtype=torch.float32,
                                               pin_memory=True), None))
        buf, ev = self._pin_bufs[idx]
        if ev is not None:
            ev.synchronize()                            # 覆写保护（常态零等待）
        buf.copy_(torch.from_numpy(a))
        if ev is None:
            ev = torch.Event()
            self._pin_bufs[idx] = (buf, ev)
        dev_t = buf.to(self.device, non_blocking=True).to(self._cdtype)
        ev.record()
        self._pin_next += 1
        return dev_t


def _unsupported_reason(cfg) -> str | None:
    """返回**不可用原因**（None = 加速读出可用）。

    P19 教训：换后端前必须核对配置——两级读出（`readout_hidden`）、结构性稀疏
    读出（`readout_conn_k`）、fp4 码本三条路径 `AccelReadout` **未实现**。强行上
    会「能跑但语义不同」——比回落到 numba 原路径更糟。命中任一项即回落，并记录
    原因供日志如实报告。

    ⚠ **fp8 已在 P84 实现**（forward 用 fp8 副本 + 更新用 fp16 主副本），曾被
    列在这里导致 `--readout-dtype fp8`（**现在的默认值**）被静默回落到 numba
    CPU ——加新 dtype 时务必同步这张能力表。
    """
    if int(getattr(cfg, "readout_hidden", 0) or 0) > 0:
        return "readout_hidden>0（两级读出未在加速后端实现）"
    # P124：幂律异质连接 → **行宽不等** → 需要变长 CSR + segment sum，
    # 与本路径的「均匀 k 稠密张量gather-GEMV」是**不同的算法**（P19 纪律：
    # 不能「能跑但语义不同」）。必须在能力表**提前**拒绝，否则会一路走到
    # 构造期抛 ValueError → 被 pick_readout_backend 的兜底 except 吞掉 →
    # **回落原因丢失**（`_accel_fallback_reason=None`，与 B12 同类静默失效）。
    if float(getattr(cfg, "readout_powlaw_alpha", 0.0) or 0.0) > 0.0:
        return ("readout_powlaw_alpha>0（幂律异质连接 = 行宽不等，需变长 CSR "
                "+ segment sum，加速后端未实现；回落 numba 路径——"
                "它在 numba 下是正确且已验证的）")
    # P111：稀疏读出**已在加速后端实现**（均匀 k 的 gather-GEMV），不再回落。
    # 仍需拒绝的只有非均匀行宽（幂律变长 CSR）——那是另一种结构，见AccelReadout
    # 的「只支持均匀 k」注释。若将来有人传入变长 CSR，构造期会fail-fast。
    # ⚠⚠ **P86 的 fp8 禁用已按 fhz 2026-10-02 指令解除**：
    #   原 P86 记录「昇腾实测 fp8 抛 Float8_e4m3fn has not been supported
    #   (ERR01007)」→ 一度硬编码禁用。但**驱动/算子库会升级**，「这台机器不
    #   支持」≠「昇腾都不支持」，硬编码会在支持 fp8 的机器上白白浪费2 倍
    #   存储与带宽。
    #   → 改为**运行时探测**（`phdnet/backends/fp8_capability.py`）：
    #     原生 fp8 → 直接用（**不告警**）；只有 int8 → **告警 + 自动转 int8**
    #     （转换在 CPU，效率已实测并记录）；两者都无 → **报错**并给可选方案。
    #   ⚠ 探测**不在这里**做（那是运行期的事，见 `AccelReadout.__init__`），
    #     能力表只负责「不能静态判定」的配置。
    _rd = str(getattr(cfg, "readout_dtype", "fp32"))
    # ⚠ P151（fhz 2026-10-03：「解禁 fp8, fp4, int4, int8」）：4-bit 的禁令**解除**。
    #   P86 当初禁它是因为「910B 无 INT4 矩阵乘单元」—— 但那个理由对
    #   **稀疏 gather-GEMV** 不成立：4-bit 在本实现里是**uint8 打包 + fp16 计算域**
    #   （`_pack4`/`_unpack4`，见 P151），**不需要 int4 矩阵乘单元**。
    #   存储省 8×（vs fp32）是真实的，算力与 fp16 相同。
    #   ⚠ 代价：**每步要 unpack 出 fp16 副本** → 是否真省流量要看昇腾 msprof。
    # P105：int8 已实现（存储 int8 + fp16 计算，构造期探测失败时在 AccelReadout
    # 内部**永久回落 fp16**，不走 numba 回落）。旧名 fp8 是 int8 的别名（P100
    # 正名），同样放行。加新 dtype 时务必同步这张能力表（P92 就是漏了 int8 才
    # 导致默认配置静默回落 numba CPU）。
    if _rd in ("int8", "fp8"):
        return None
    # P112 修复（能力表与 `_DT` 错位）：int16/int32 在 **CPU 路径**是真实
    # 实现的定点码本（`readout.py` P101，最近格点 = 四舍五入到最近整数），
    # 但加速后端的 `_DT` 没有这两种 → 原来会一路走到构造期抛
    # `ValueError: 不支持 dtype=...`，由 `pick_readout_backend` 的
    # `except Exception` 兜底回落 —— **回落原因丢失（`_accel_fallback_reason`
    # 为 None）**，日志上看不出为什么没上设备。
    # 正确做法：在能力表里**显式拒绝并说明原因**（P19 纪律「回落 + 记原因」）。
    if _rd in ("int16", "int32"):
        return (f"readout_dtype={_rd}（加速后端未实现该定点码本；"
                f"CPU 的 numba 路径可用——用 --accel cpu 走 P101 的整数核）")
    if bool(getattr(cfg, "lognormal_init", False)):
        return None                            # 初始化分布不同但结构兼容，不阻断
    return None


def pick_readout_backend(cfg, n_h: int, n_out: int, rng):
    """按 `cfg.accel_readout` 选读出后端；不可用时**回落 numba 原路径**。

    返回 (readout, 后端名)。回落必须静默安全：无加速器 / torch 缺失 /
    配置不兼容 / 构造异常 → 原 `Readout`（默认路径逐位不变），并把原因记在
    `readout._accel_fallback_reason` 上（不静默）。
    """
    def _powlaw_counts():
        """取词频向量（P124）。构造期拿不到 → 由训练循环用 `set_powlaw_counts`
        注入；未注入时返回 `None`，`build_powlaw_csr` 会退化成均匀分配。

        ⚠ **这是「词频从哪来」的设计缺口**（必须如实记下）：
        理想的 counts 是**训练语料的词频**（Zipf 分布），但读出在
        `pick_readout_backend` 时就构造好了，那时词表刚建好、语料尚未流过
        → **拿不到真实词频**。当前只有两条可行路径：
          (a) 用**词表序**当代理（`tok.stoi` 的插入序≈ 频次序，因为词是按
              训练流里出现顺序induced 的）—— 零成本、可立即用，但是**近似**；
          (b) 两阶段：先用均匀 k-conn 跑一段收集词频，再重建读出（复杂，且
              会改变 checkpoint 结构）。
        故本函数先返回注入值，没有则 None（=均匀）。**这个近似是否可接受，
        必须由 PPL A/B 判定，不能想当然。**
        """
        return getattr(cfg, "_powlaw_counts_runtime", None)

    def _mk_readout():
        """构造 numba `Readout`（**三条分支共用**，P124）。

        幂律参数在这里统一下发，避免「回落路径 / cpu 路径 / auto 无加速路径」
        三处漏传 —— 那会导致「同一份配置在不同后端得到不同连接结构」。
        ⚠ `powlaw_counts` **不由 cfg 携带**（它是运行时的词频统计，见
        `set_powlaw_counts`）：构造期拿不到，须由训练循环在第一次 step 前注入。
        """
        from ..readout import Readout
        return Readout(n_h, n_out, rng, w_clip=cfg.readout_w_clip,
                       dtype=cfg.readout_dtype, conn_k=cfg.readout_conn_k,
                       lognormal_init=cfg.lognormal_init,
                       powlaw_alpha=float(getattr(cfg, "readout_powlaw_alpha", 0.0)),
                       powlaw_counts=_powlaw_counts(),
                       powlaw_density=getattr(cfg, "readout_powlaw_density", None),
                       powlaw_kmin=int(getattr(cfg, "readout_powlaw_kmin", 1)),
                       powlaw_kmax=int(getattr(cfg, "readout_powlaw_kmax", 0)))

    def _fallback(reason: str):
        from ..readout import Readout
        ro = _mk_readout()
        ro._accel_fallback_reason = reason
        return ro, "numba-cpu(回落)"

    spec = str(getattr(cfg, "accel_readout", "auto") or "auto").lower()
    if spec in ("", "cpu", "off", "numba"):
        return _mk_readout(), "numba-cpu"
    bad = _unsupported_reason(cfg)
    if bad is not None:
        return _fallback(bad)
    if spec == "auto":
        try:
            from .multi_device import probe_multi
            probes = probe_multi()
            has_accel = any(v.get("ok") and v.get("count")
                            for k, v in probes.items() if k != "cpu")
        except Exception:                                    # noqa: BLE001
            has_accel = False
        if not has_accel:
            return _mk_readout(), "numba-cpu"
        spec = "auto"                                       # 交给 resolve_device 择优
    try:
        return (AccelReadout(n_h, n_out, rng, device=spec,
                             dtype=cfg.readout_dtype,
                             w_clip=cfg.readout_w_clip,
                             gather_impl=str(
                                 getattr(cfg, "readout_gather_impl", "index")),
                             # ⚠⚠ **P148 修正（审计 BUG-3）**：这个参数之前**漏传**，
                             #   于是 `cfg.readout_gather_dtype` / CLI 的 fp16
                            #   一路走到后端又退回签名默认 "fp32"
                             #   → 「P147唯一真正生效的改动」完全落空。
                             #   实测：cfg 是 fp16 而 `ro._gather_dtype` 是 fp32。
                             gather_dtype=str(
                                 getattr(cfg, "readout_gather_dtype", "fp32")),
                             int8_compute=str(
                                 getattr(cfg, "readout_int8_compute",
                                         "auto")),
                             fp8_conv=str(getattr(cfg, "fp8_conv", "torch")),
                             nll_sync_every=int(getattr(cfg, "nll_sync_every", 1)),
                             compile=bool(getattr(cfg, "torch_compile", False)),
                             compile_mode=str(getattr(cfg,
                                                     "torch_compile_mode",
                                                     "default")),
                             conn_k=int(getattr(cfg, "readout_conn_k", 0) or 0),
                             lognormal_init=bool(getattr(cfg, "lognormal_init",
                                                         False)),
                             exc_ratio=float(getattr(cfg, "exc_ratio", 0.8)),
                             sparse_fwd_kernel=str(
                                 getattr(cfg, "sparse_fwd_kernel", "mulsum"))),
                f"accel:{spec}")
    except Exception as e:                                   # noqa: BLE001
        return _fallback(f"{type(e).__name__}: {e}")
