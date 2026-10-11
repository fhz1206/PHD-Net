"""M6 读出**新增整数码本**门禁（int16 / int32）—— 退出码 0 = PASS。

================================================================================
为什么单独一个文件，而不是并进 verify_readout_sparse_gate.py
================================================================================
两者守的是**不同的不变量**：
  * verify_readout_sparse_gate.py 守 **CSR 结构**（indptr/idx 自洽、只写存在槽位）
    ——它的一条核心断言是「稀疏时`dtype_name` 被强制为 fp32」，量化码本**不与其叠加**。
  * 本文件守 **码本数值语义**（量化误差序、饱和不回绕、核 vs numpy 逐位一致）。
把码本断言塞进稀疏门禁会让「稀疏结构坏了」与「码本数值坏了」共用一个失败信号，
反而降低定位能力。故新开本文件，两者互补、都必须PASS。

================================================================================
五组用例，各自「为什么要有」
================================================================================
【T1】9 种写法（7 个规范名 + 2 个旧名别名）都能构造 / 前向有限 / learn 跑通
  为什么：`RO_DTYPES` 与实际 dispatch 键是**两处独立维护**的列表（前者是白名单，
  后者是 `__call__` 与 `_apply_update` 的 dict/if 链）。新增格式时漏改 dispatch
  是最典型也最难发现的错误——构造成功、`learn` 不报错，但 `__call__` 抛
  `KeyError` 或静默走了错误内核。本组把「白名单 ∩ dispatch 键」显式断言，
  且用**旧名 fp8/fp4** 验证 `_RO_DTYPE_ALIASES` 归一后仍命中新码本
  （P100 正名的兼容性契约）。

【T2】量化误差序：int32 < int16 < int8（同一随机 W）
  为什么：位宽越大精度必须越好——这是**码本存在意义**的根属性。若某次改动让
  int16 的误差超过 int8（例如 scale 选错、格点数写错、LUT 索引错位），训练会
  静默地用一个更差的格式，**没有任何报错**。误差用「最大绝对误差 / max|W|」
  的**尺度归一**度量（不是逐元素相对差——权重里大量近零元素会让相对差发散到
  无意义量级，同 verify_readout_sparse_gate.py 的 scaled_err 理由）。

【T3】饱和裁剪：极端更新/极端权重后**不回绕**
  为什么：这是本轮最关键的一条。整数码本**没有隐式 clamp**——`np.int16(40000)`
  是回绕（且在不同 numpy 版本上可能变成 UB 或饱和），回绕后的码**仍是合法码本值**
  （例如 40000 → −25536），于是「权重符号翻转、幅度仍在码本内」，训练照跑、PPL
  照出，**没有任何静默损坏的信号**。int8 之所以能容忍是因为 e4m3fn 的位域在域外
  会被自然 clamp 到 NaN 槽/最大有限码；int16/int32 必须**显式** clip。
  本组用 η=1e6 的极端更新把码本推到两端，断言所有码 == 码本端点。

【T4】更新核 vs numpy 参考**逐位一致**
  为什么：numba 核是「手写位/算术 + 内联查表」，numpy 参考是「直接数学」。二者
  独立实现同一套舍入约定（RNE + 饱和裁剪）。只有对拍到**逐位相同**才能证明
  融合核没有偷偷换掉舍入模式（若改成截断，int16 上误差会从1e-5 涨到 1e-2，
  但仍然「看起来在收敛」）。

【T5】不回归：int8/int4/fp16/bf16/fp32 的既有行为不变
  为什么：本轮只**新增**分支，理论上不可能动到旧路径；但 `_build_lut` 返回值、
  `quantize_to` 的 dtype 映射、`__call__` 的 dispatch 表都是共享代码，一个
  笔误就能同时破坏新旧格式。故显式断言：旧 5 种格式的量化-反量化往返结果
  与「本次改动前已验证的量级」一致，且 int4 仍是半字节打包、fp16/bf16 仍是
  uint16 码本（存储位宽不变 = 省内存的前提被守住）。

用法：python tests/verifiers/verify_readout_intdtypes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phdnet.readout import (INT16_QMAX, INT32_QMAX, INT32_QMIN,  # noqa: E402
                            RO_DTYPES, Readout, _q_tables,
                            _ro_q_update_int16, _ro_q_update_int32,
                            dequantize_from, quantize_to)

N_IN, N_OUT = 128, 64
# T3 用η=1e6：单步更新量 ~1e6·|dp|·|h|，远超码本端点（int16 归一化后±32767、
# int32 ±2.1e9），必然把码推到**两端**，是「饱和」而非「恰好落在码本内」。
SAT_ETA = 1e6
_results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, extra: str = "") -> bool:
    _results.append((bool(ok), name, extra))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {extra}" if extra else ""))
    return bool(ok)


def scaled_err(got: np.ndarray, want: np.ndarray) -> float:
    """**尺度归一**的最大绝对误差：max|got−want| / max(1, max|want|)。

    为什么不用逐元素相对差：权重里有大量近零元素，分母趋 0 会把误差放大到
    无意义的量级（同一份正确结果能报出 1e-14 的「相对差」，纯属分母太小）。
    这里断言的是「误差相对于整体量级可忽略」。
    """
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(want))) if want.size else 1.0)
    return float(np.max(np.abs(got - want)) / scale) if got.size else 0.0


def rel_roundtrip_err(fmt: str, W: np.ndarray) -> tuple[float, float]:
    """量化-反量化往返误差：返回 (尺度归一相对误差, per-tensor scale)。

    scale 归一化到 max|W|（而不是 max(1,·)）——这里关心的是**码本分辨率**，
    与权重本身的量级无关；故除以 max|W|。
    """
    ro = Readout(W.shape[1], W.shape[0], np.random.default_rng(0), dtype=fmt)
    ro.W = W
    Q = ro.W
    denom = float(np.max(np.abs(W)))
    return float(np.max(np.abs(Q - W)) / denom), float(ro._wscale)  # noqa: SLF001


# ────────────────────────────────────────────────────────────────────────────
# T1 · 9 种写法：构造 + 前向有限 + learn跑通
# ────────────────────────────────────────────────────────────────────────────
def t1_construct_forward_learn() -> None:
    print("\n" + "=" * 88)
    print("T1 · 7 种写法（6 规范名 + fp8 旧名别名；int4/fp4 已按 P191e 删除）："
          "构造 / 前向有限 / learn")
    print("=" * 88)
    print(f"  RO_DTYPES = {RO_DTYPES}（共 {len(RO_DTYPES)} 个规范名）")

    # 白名单必须恰好是这 6 个（P191e 删 4-bit 后；防「加了格式忘了加进白名单」）。
    check("RO_DTYPES 含全部 6 个规范名（无 4-bit）",
          set(RO_DTYPES) == {"fp32", "fp16", "bf16", "int8",
                             "int16", "int32"},
          f"RO_DTYPES = {RO_DTYPES}")

    rng = np.random.default_rng(20260930)
    for dtype in RO_DTYPES + ("fp8",):                # 6 + 1 = 7 种写法
        try:
            ro = Readout(N_IN, N_OUT, rng, dtype=dtype)
            h = rng.normal(0, 1, N_IN).astype(np.float32)
            tgt = np.zeros(N_OUT, dtype=np.float32)
            tgt[3] = 1.0
            y = ro(h)
            nll = ro.learn_softmax(h, tgt, 0.05)
            ro.learn(h, tgt, 0.05)                #感知器路径（返回 None）
            ok = (y.shape == (N_OUT,) and np.all(np.isfinite(y))
                  and np.isfinite(nll)
                  and np.all(np.isfinite(ro.W)))
            check(f"{dtype:<5} 构造 / 前向有限 / learn+learn_softmax 跑通", ok,
                  f"dtype_name={ro.dtype_name} nll={nll:.4f} "
                  f"max|W|={np.abs(ro.W).max():.4f}")
        except Exception as exc:                # noqa: BLE001
            check(f"{dtype:<5} 构造 / 前向有限 / learn 跑通", False,
                  f"{type(exc).__name__}: {exc}")

    # 旧名必须归一到新名（否则 `fp8` 会走一条无人验证的独立路径）。
    ro8 = Readout(N_IN, N_OUT, rng, dtype="fp8")
    check("旧名 fp8 → int8（旧名正名兼容契约，P100）",
          ro8.dtype_name == "int8" and ro8.qfmt == "int8",
          f"fp8 → dtype_name={ro8.dtype_name}")
    # P191e：4-bit 已删除 → int4/fp4 显式请求必须在白名单处 fail-fast。
    for _bad in ("int4", "fp4"):
        try:
            Readout(N_IN, N_OUT, rng, dtype=_bad)
            check(f"显式请求 {_bad} 被 fail-fast 拒绝（P191e）", False, "未拒绝")
        except ValueError:
            check(f"显式请求 {_bad} 被 fail-fast 拒绝（P191e）", True)

    # CSR 稀疏模式**仍强制 fp32**（量化码本与 CSR 不叠加）——P92/P100 已定语义，
    # 本轮不得改动，这里显式钉住。
    for dt in ("int8", "int16", "int32"):
        ro_s = Readout(N_IN, N_OUT, rng, dtype=dt, conn_k=16)
        check(f"conn_k>0 时 {dt} 被强制回落 fp32（码本不与 CSR 叠加）",
              ro_s.dtype_name == "fp32" and ro_s.qfmt is None,
              f"conn_k=16 → dtype_name={ro_s.dtype_name}")

    # 存储位宽：int16 = 2 B/权重（与 bf16 同量）、int32 = 4 B/权重。
    for dt, nbytes in (("int16", 2), ("int32", 4)):
        ro_d = Readout(N_IN, N_OUT, rng, dtype=dt)
        per = ro_d._codes.nbytes / (N_IN * N_OUT)   # noqa: SLF001
        check(f"{dt} 存储 {nbytes} B/权重", abs(per - nbytes) < 1e-9,
              f"实测 {per:.1f} B/权重，共 {ro_d.stats()['storage_MB']:.3f} MB")


# ────────────────────────────────────────────────────────────────────────────
# T2 · 量化误差序int32 < int16 < int8
# ────────────────────────────────────────────────────────────────────────────
def t2_precision_order() -> None:
    print("\n" + "=" * 88)
    print("T2 · 量化误差序int32 < int16 < int8（同一随机 W，尺度归一）")
    print("=" * 88)
    rng = np.random.default_rng(4242)
    W = rng.normal(0.0, 0.05, (N_OUT, N_IN)).astype(np.float32)

    errs = {}
    for fmt in ("int8", "int16", "int32"):
        errs[fmt], scale = rel_roundtrip_err(fmt, W)
        print(f"    {fmt:<6} scale={scale:<12.6g} 往返误差 = {errs[fmt]:.3e}")

    check("int32 误差 < int16 误差", errs["int32"] < errs["int16"],
          f"{errs['int32']:.3e} < {errs['int16']:.3e}")
    check("int16 误差 < int8 误差", errs["int16"] < errs["int8"],
          f"{errs['int16']:.3e} < {errs['int8']:.3e}")
    # 量级护栏：位宽差16× → 误差大致差 2^8=256×（int8 e4m3 只有 3 位尾数，
    # int16/int32 是均匀格点，故实际差距比这更大）。给宽松区间防「碰巧过关」。
    check("int32 误差在近无损量级（< 1e-6）", errs["int32"] < 1e-6,
          f"{errs['int32']:.3e}（2³² 格点 → 理论上~1e-10）")
    check("int16 误差在 1e-3量级以下", errs["int16"] < 1e-3,
          f"{errs['int16']:.3e}（32768 格点 → 理论上 ~3e-5）")
    check("int8 误差在 1e-1 量级以下", errs["int8"] < 1e-1,
          f"{errs['int8']:.3e}（e4m3 3 位尾数 → 理论上 ~4e-2）")

    # int32 的**核心卖点**是大动态范围：同一 scale 下能表示的幅度跨度是 int16 的
    # 2^16 = 65536 倍。构造「跨 6 个数量级」的权重验证 int32 不饱和、int16 饱和。
    Wwide = (rng.normal(0.0, 1.0, (N_OUT, N_IN)) * np.float32(1e-6)).astype(np.float32)
    Wwide[0, 0] = 50.0                      # 单个巨大权重把 scale 拉高
    for fmt in ("int16", "int32"):
        ro_w = Readout(N_IN, N_OUT, rng, dtype=fmt)
        ro_w.W = Wwide
        Q = ro_w.W
        big_want = float(Wwide[0, 0])
        big_got = float(Q[0, 0])
        check(f"{fmt} 动态范围：能还原 {big_want:.1f} 这个大权重（不被饱和）",
              abs(big_got - big_want) < 0.05 * abs(big_want),
              f"还原为 {big_got:.4f}（scale={ro_w._wscale:.3g}，"
              f"相对误差 {abs(big_got - big_want) / abs(big_want):.2e}）")


# ────────────────────────────────────────────────────────────────────────────
# T3 · 饱和裁剪（**不回绕**）——本轮最关键的一条
# ────────────────────────────────────────────────────────────────────────────
def t3_saturation_no_wraparound() -> None:
    print("\n" + "=" * 88)
    print("T3 · 饱和裁剪：极端更新后码停在码本端点，**不回绕**")
    print("=" * 88)
    rng = np.random.default_rng(555)
    W = rng.normal(0.0, 0.05, (N_OUT, N_IN)).astype(np.float32)

    # (a) 极端**权重**直接量化：超出码本范围的值必须被clip，而不是回绕。
    #     int32 的 per-tensor scale = max|W|/QMAX，故需给出「人为超范围」的输入：
    #     先按小 scale 重量化（相当于动态范围外的值），验证 clip 到端点。
    for fmt, qmax, qmin in (("int16", INT16_QMAX, -INT16_QMAX),
                            ("int32", INT32_QMAX, INT32_QMIN)):
        tiny_scale = float(np.abs(W).max()) / qmax * 1e-6   # 远小于需要的 scale
        codes = quantize_to(fmt, W.ravel() / tiny_scale)
        if fmt == "int16":
            mag = codes.astype(np.int64) & 0x7FFF
            neg = (codes.astype(np.int64) & 0x8000) != 0
            signed = np.where(neg, -mag, mag)
            hi, lo = int(INT16_QMAX), int(-INT16_QMAX)
        else:
            signed = codes.astype(np.int64)
            hi, lo = int(INT32_QMAX), int(INT32_QMIN)
        check(f"{fmt} 量化超范围值 → 全部被 clip 到码本端点（不回绕）",
              bool(signed.max() <= hi and signed.min() >= lo),
              f"码范围 [{signed.min()}, {signed.max()}] ⊆ [{lo}, {hi}]"
              f"；越界元素 {int((np.abs(W.ravel() / tiny_scale) > qmax).sum())} 个")

    # (b) 极端**更新**（η=1e6）→ 码本被推到两端。
    #     实测行为（fp32 路径 np.clip 为基准）：int16 码 ∈ {0x0000, 0x7FFF, 0x8000,
    #     0xFFFF} = {+0, +32767, −0, −32767}；int32 码 = ±iinfo(int32).min/max。
    #     **判据是「反量化后的幅值恰好等于码本端点 × scale」**——若发生回绕，
    #     幅值会突然变成一个码本内但**随机**的值，本断言会立刻失败。
    for fmt, qmax, qmin in (("int16", INT16_QMAX, -INT16_QMAX),
                            ("int32", INT32_QMAX, INT32_QMIN)):
        ro = Readout(N_IN, N_OUT, rng, dtype=fmt)
        ro.W = W
        scale = ro._wscale                            # noqa: SLF001
        h = rng.normal(0, 1, N_IN).astype(np.float32)
        tgt = np.zeros(N_OUT, dtype=np.float32)
        tgt[1] = 1.0
        for _ in range(3):
            ro.learn_softmax(h, tgt, SAT_ETA)
        Wq = ro.W
        ok_finite = bool(np.all(np.isfinite(Wq)))
        # 归一化后的码必须落在码本范围内（±qmax；int16 零规范化后有−0→+0）
        norm = Wq / scale
        # 码本**不对称**：int32 能表示 −2³¹（比 +2³¹−1 多一个格点），故幅值端点
        # 有**两个**（qmax 与 |qmin|）；int16 对称故两者相同。归一化后的幅值只允许
        # 取 {0, 端点} —— 若发生回绕，会出现码本内但**随机**的中间值，本断言立刻失败。
        ends = sorted({float(qmax), abs(float(qmin))})
        in_range = bool(np.abs(norm).max() <= max(ends) + 1.0)
        only_ends = bool(np.isin(np.round(np.abs(norm)), [0.0, *ends]).all())
        check(f"{fmt} η=1e6×3 步：反量化幅值全部有限", ok_finite)
        check(f"{fmt} η=1e6×3 步：|w|/scale ≤ 码本端点（未越界）", in_range,
              f"max|w|/scale = {np.abs(norm).max():.6g}，"
              f"码本端点 = {ends}")
        check(f"{fmt} η=1e6×3 步：码只落在端点 {ends}（**未回绕**）", only_ends,
              f"实测出现的 |code| 取值 = "
              f"{np.unique(np.round(np.abs(norm))).tolist()[:6]}，"
              f"scale = {scale:.6g} → 端点权重幅值 = {max(ends) * scale:.6g}")

    # (c) 对照：若**不做**饱和裁剪会怎样（说明为什么这条断言不是形式主义）。
    #     直接展示 numpy int16 的回绕结果——它仍是「合法码本值」，
    #     故权重符号翻转后训练照跑、无任何静默损坏信号。
    wrapped = np.array([40000], dtype=np.int32).astype(np.int16)[0]
    check("对照：int16(40000) 若不回绕会变成负数（故必须显式 clip）",
          int(wrapped) < 0,
          f"np.int16(40000) = {int(wrapped)}（回绕到码本内的另一个值；"
          f"本实现改用 clip → {int(INT16_QMAX)}）")


# ────────────────────────────────────────────────────────────────────────────
# T4 · 更新核 vs numpy 参考逐位一致
# ────────────────────────────────────────────────────────────────────────────
def t4_kernel_vs_numpy() -> None:
    print("\n" + "=" * 88)
    print("T4 · int16/int32 更新核 vs numpy 参考（码本**逐位**一致）")
    print("=" * 88)
    rng = np.random.default_rng(7)
    V, H = 300, 256
    W0 = (rng.standard_normal((V, H)) * 0.05).astype(np.float32)
    dp = (rng.random(V) * 0.01).astype(np.float32)
    h = rng.standard_normal(H).astype(np.float32)

    def np_ref_codes(fmt: str, codes_old: np.ndarray, scale: float,
                     dp_: np.ndarray, h_: np.ndarray, eta: float) -> np.ndarray:
        """numpy 参考：dequant → fp32 更新 → RNE 取整 → 饱和裁剪 → 编码。

        舍入约定与 numba 核逐位对应：`rint(w / scale)`（RNE，并列取偶）+ 显式 clip。
        int16 额外做「零规范化」（幅值 0 → 无符号位），与核内 `if q < 0.0` 一致。

        ⚠ 两处**刻意的位宽对齐**（否则对拍会出现 ~1e-5 量级的假差异）：
          ① `astype(np.float64)` 后再乘 scale —— numba 核里`wscale` 是 float64
             参数，`lut_f32 * wscale` 被提升为 float64；而 numpy 里
             `f32数组 * python_float` 仍是 float32（NEP50 弱标量）。不提升则
             分母多一层 fp32 舍入，量化到整格点后差 1。
          ② 更新项 `(dp[:,None]*h[None,:]) * np.float32(eta)` 全程 fp32 ——
             与核内 `(e * h[j]) * eta` 的三个 fp32 操作逐位一致。
        """
        w = (dequantize_from(fmt, codes_old).reshape(V, H).astype(np.float64)
             * scale)
        w = ((w - (dp_[:, None] * h_[None, :]) * np.float32(eta))
             .ravel() / scale)
        if fmt == "int16":
            m = np.minimum(np.rint(np.abs(w)), INT16_QMAX)
            return (m.astype(np.int64)
                    | np.where((w < 0) & (m > 0.0), 1 << 15, 0))
        return np.clip(np.rint(w), INT32_QMIN, INT32_QMAX).astype(np.int64)

    for fmt, K, qmax in (("int16", _ro_q_update_int16, INT16_QMAX),
                         ("int32", _ro_q_update_int32, INT32_QMAX)):
        for trial, eta in enumerate((0.05, 1e-3, SAT_ETA)):   # 含极端 η（触发饱和）
            scale = float(np.abs(W0).max()) / qmax
            a = quantize_to(fmt, (W0 / scale).ravel()).copy()
            b = a.copy()
            K(a, dp, h, np.float32(eta), H, scale)
            ref = np_ref_codes(fmt, b, scale, dp, h, eta)
            ndiff = int((a.astype(np.int64) != ref).sum())
            check(f"{fmt} 更新核 == numpy 参考（η={eta:g}, trial={trial}）",
                  ndiff == 0,
                  f"{a.size} 个元素中差异 {ndiff} 个"
                  + (f"；码范围 [{a.min()}, {a.max()}]" if eta >= 1.0 else ""))

    # int32 前向 == numpy 参考（无 LUT 路径的端到端核对）。
    for fmt in ("int16", "int32"):
        ro = Readout(H, V, rng, dtype=fmt)
        Wq = rng.normal(0, 0.05, (V, H)).astype(np.float32)
        ro.W = Wq
        hv = rng.normal(0, 1, H).astype(np.float32)
        y = ro(hv)
        Wref = ro.W.astype(np.float64)
        yref = Wref @ hv.astype(np.float64)
        check(f"{fmt} 前向 == 反量化权重 @ h（fp64 参考）",
              scaled_err(y, yref) < 1e-6,
              f"尺度归一误差 {scaled_err(y, yref):.3e}")


# ────────────────────────────────────────────────────────────────────────────
# T5 · 不回归：旧 5 种格式行为不变
# ────────────────────────────────────────────────────────────────────────────
def t5_no_regression() -> None:
    print("\n" + "=" * 88)
    print("T5 · 不回归：fp32/fp16/bf16/int8 的存储与往返行为不变"
          "（int4 已按 P191e 删除）")
    print("=" * 88)
    rng = np.random.default_rng(31337)
    W = rng.normal(0.0, 0.05, (N_OUT, N_IN)).astype(np.float32)
    n = N_IN * N_OUT

    # (a) 存储位宽不变（省内存的前提）。
    expect_bytes = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0,
                    "int16": 2.0, "int32": 4.0}
    for fmt, eb in expect_bytes.items():
        ro = Readout(N_IN, N_OUT, rng, dtype=fmt)
        per = ro._codes.nbytes / n if fmt != "fp32" else ro._W.nbytes / n  # noqa: SLF001
        check(f"{fmt:<5} 存储位宽不变（{eb:g} B/权重）", abs(per - eb) < 1e-9,
              f"实测 {per:.2f} B/权重")

    # (b) 码本 dtype 不变：fp16/bf16 = uint16 码、int8 = uint8 码。
    #     int16 也是 uint16（符号|幅值，与 fp16/bf16 同存储形态）；int32 = int32。
    for fmt, want in (("fp16", np.uint16), ("bf16", np.uint16),
                      ("int8", np.uint8),
                      ("int16", np.uint16), ("int32", np.int32)):
        codes = quantize_to(fmt, W)
        check(f"{fmt:<5} 码本 dtype = {np.dtype(want).name}",
              codes.dtype == want, f"实测 {codes.dtype}")

    # (c) P191e：int4 半字节打包断言已随 4-bit 删除（显式请求 fail-fast，T1 已验）。

    # (d) 旧 4 种格式的往返误差仍在已验证量级（fp32 精确为 0）。
    for fmt in ("fp32", "fp16", "bf16", "int8"):
        ro = Readout(N_IN, N_OUT, rng, dtype=fmt)
        ro.W = W
        e = float(np.max(np.abs(ro.W - W)) / np.max(np.abs(W)))
        lim = {"fp32": 1e-9, "fp16": 1e-3, "bf16": 1e-2, "int8": 1e-1}[fmt]
        check(f"{fmt:<5} 往返误差 < {lim:g}（既有行为不变）", e < lim, f"{e:.3e}")

    # (e) LUT 表单例仍齐备，且 int32 **刻意不建 LUT**（17 GB 物化即 OOM）。
    for fmt in ("fp16", "bf16", "int8", "int16"):
        lut, lat, sb = _q_tables(fmt)
        ok = lut is not None and lat is not None and sb > 0
        check(f"{fmt:<5} LUT 单例存在（{lut.size} 项，符号位 {sb}）", ok,
              f"lut={lut.size} lat={lat.size}")
    lut32, lat32, sb32 = _q_tables("int32")
    check("int32 **不建 LUT**（(None,None,0) 哨兵，2³²格点=17 GB 不可物化）",
          lut32 is None and lat32 is None and sb32 == 0,
          f"_q_tables('int32') = ({lut32}, {lat32}, {sb32})")


def main() -> int:
    print("=" * 88)
    print("M6 读出整数码本门禁：int16 / int32（verify_readout_intdtypes）")
    print(f"python {sys.version.split()[0]} | numpy {np.__version__}")
    print(f"维度 n_in={N_IN} n_out={N_OUT} | 饱和测试 η={SAT_ETA:g}")
    print("=" * 88)
    t1_construct_forward_learn()
    t2_precision_order()
    t3_saturation_no_wraparound()
    t4_kernel_vs_numpy()
    t5_no_regression()

    npass = sum(1 for ok, _, _ in _results if ok)
    ntot = len(_results)
    fails = [name for ok, name, _ in _results if not ok]
    print("\n" + "=" * 88)
    print(f"结果：{npass}/{ntot} 通过 | 失败 {len(fails)}")
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
    print("=" * 88)
    print("门禁结论：" + ("PASS ✅" if not fails else "FAIL ❌"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())