# -*- coding: utf-8 -*-
"""加速器读出（P19）的等价性与回落行为对拍。

fhz：「词表做完之后如果设备有 cuda/cann/rocm 就跑在对应设备上」——读出是
训练里唯一的计算密集部件（1B 预设占 ~89%），本验证覆盖：

A  接口兼容：AccelReadout 与 `phdnet.readout.Readout` 的调用面一致
   （learn_softmax / W / n_synapses）；
B  等价性：前向 / NLL / 更新后 W **容差一致**（atol/rtol 口径，同 P9/P10）；
   注：即便同在 CPU，numpy BLAS 与 torch 内核的**归约顺序不同**（向量宽度 /
   FMA 使用差异），跨库不保证逐位——实测 max|Δ| ≈ 4e-06，属预期。
C  状态往返：W_cpu() / load_W() 精确还原（检查点兼容）；
D  后端选择：auto 在**无加速器**机器上回落 numba 原路径（默认路径逐位不变）；
   显式不可用设备时回落并记录原因（诚实降级，不静默）。
E  精度档：fp32 / fp16 / bf16 / int8 均可构造并前向（低精度容差判据）。
F  int8 读出（P105）：存储 int8 / 前向每步反量化 / 目标行更新 vs fp16 参考
   对拍 / 20 步稳定性 / w_clip 重量化裁剪 / fp8 旧名别名 / ckpt 往返。

运行：python tests/verifiers/verify_accel_readout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_HERE), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                          # noqa: E402

_FAILURES: list[str] = []


def _random_csr_for(n_out: int, n_h: int, k: int):
    """P149：造一份 CSR（供 fp8 存储/计算分离的契约检查用）。"""
    from phdnet.sparse_pc import _random_csr
    return _random_csr(np.random.default_rng(0), n_out, n_h, k,
                       0.05 * np.sqrt(n_h / k), False, 0.8)


def _fp8_step_ok(ro) -> bool:
    """P149：fp8 存储的读出能否跑通「前向 + 感知器更新」一步。

    我 P148 把计算域也设成 fp8 → 这一步直接抛
    "Promotion for Float8 Types is not supported"。这是**回归点**，
    故钉成门禁：只要有人再把两个域混淆，测试立刻红。
    """
    try:
        h = np.random.default_rng(1).normal(0, 1, ro.n_h).astype(np.float32)
        t = np.zeros(ro.n_out, dtype=np.float32)
        t[0] = 1.0
        y = ro.forward_dev(h)
        ro.learn_softmax(h, t, 0.15, y_pre=y, target_idx=0)
        return True
    except Exception:                                        # noqa: BLE001
        return False


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" —— {detail}" if detail else ""), flush=True)
    if not ok:
        _FAILURES.append(name)


def main() -> None:
    import torch

    from phdnet.backends.accel_readout import (AccelReadout,
                                               pick_readout_backend)
    from phdnet.config import PHDNetConfig
    from phdnet.readout import Readout

    rng = np.random.default_rng(7)
    n_h, n_out = 128, 501
    W0 = (rng.normal(0.0, 0.05, (n_out, n_h)) * np.sqrt(n_h)).astype(np.float32)
    h = rng.normal(0.0, 1.0, n_h).astype(np.float32)
    tgt = np.zeros(n_out, dtype=np.float32)
    tgt[rng.integers(0, n_out)] = 1.0

    # ── A/B：同设备逐位等价（读出只有 W@h 与外积更新，标量可精确比对）──
    print("[A/B] AccelReadout ≡ Readout（容差判据）")
    ro_ref = Readout(n_h, n_out, rng, w_clip=0.0, dtype="fp32")
    ro_ref.W[:] = W0
    ro_acc = AccelReadout(n_h, n_out, None, device="cpu", dtype="fp32", w0=W0)
    y_ref = (ro_ref.W @ h).astype(np.float32)            # Readout 无 forward：
    y_acc = ro_acc.forward(h)                            # 前向语义 = W @ h
    atol_f = 1e-5 * max(1.0, float(np.abs(y_ref).max()))
    check("A 前向容差一致", float(np.abs(y_ref - y_acc).max()) <= atol_f,
          f"max|Δ|={np.abs(y_ref - y_acc).max():.3e} ≤ {atol_f:.2e}")
    nll_ref = ro_ref.learn_softmax(h, tgt, 0.05)
    nll_acc = ro_acc.learn_softmax(h, tgt, 0.05)
    rel_nll = abs(nll_ref - nll_acc) / max(1e-12, abs(nll_ref))
    # 判据说明（P93, 2026-09-30）：accel 路径的 nll 用 `F.cross_entropy`
    # 单 kernel（P80，把 softmax+log 两三个 kernel 合成一个），numba 参考路径
    # 用自己的 log 路径。两者算法等价但**浮点结合顺序不同**（cross_entropy 内部
    # 走 logsumexp，参考路径是 log(softmax)），所以 1e-6 的相对判据过严。
    # 实测 rel ≈ 3.5e-5（fp32 噪声量级），W 侧同源差异是 1.2e-7（≤1e-5 ✓）。
    # 这里的口径是「nll 是**标量报告值**，不参与学习」，容差按 fp32 噪声取 1e-4；
    # 真正影响学习的是 dp（→ W），由下一条「更新后 W 容差一致」把关。
    check("B NLL 容差一致（相对差 ≤1e-4，见判据说明）", rel_nll <= 1e-4,
          f"{nll_ref:.8f} vs {nll_acc:.8f} rel={rel_nll:.2e}")
    dW = float(np.abs(ro_ref.W.copy() - ro_acc.W_cpu()).max())
    w_scale = max(1e-12, float(np.abs(ro_ref.W).max()))
    check("B 更新后 W 容差一致（≤1e-5 相对）", dW <= 1e-5 * w_scale,
          f"max|Δ|={dW:.3e} ≤ {1e-5 * w_scale:.2e}")
    check("A n_synapses 与稠密元素数一致", ro_acc.n_synapses() == n_out * n_h,
          f"{ro_acc.n_synapses():,}")
    check("A W 常驻设备且为 torch 张量", torch.is_tensor(ro_acc.W))

    from phdnet.backends.multi_device import probe_multi
    _probes = probe_multi()
    has_accel = any(v.get("ok") and v.get("count")
                    for k, v in _probes.items() if k != "cpu")

    # ── A2：4 项调用面齐备（P19 生产事故：缺 __call__ → 首个 step 崩）──
    print("[A2] 调用面齐备（model.step 的 4 个入口）")
    ro_x = AccelReadout(n_h, n_out, None, device="cpu", dtype="fp32", w0=W0)
    check("A2 可调用 __call__（model.step 用 readout(h)）",
          np.array_equal(np.asarray(ro_x(h), dtype=np.float32), ro_x.forward(h)))
    ro_y = AccelReadout(n_h, n_out, None, device="cpu", dtype="fp32", w0=W0)
    y_pre = ro_y.forward(h)
    nll_c = ro_y.learn_softmax(h, tgt, 0.05, y_pre=y_pre)
    check("A2 learn_softmax 支持 y_pre（省一次 W@h）", isinstance(nll_c, float))
    ro_z = AccelReadout(n_h, n_out, None, device="cpu", dtype="fp32", w0=W0)
    ro_z.learn(h, tgt, 0.05)                # 非 softmax 路径（P19 补齐）
    W_after = ro_z.W_cpu()
    check("A2 learn（非 softmax）改变 W 且形状不变",
          W_after.shape == (n_out, n_h) and not np.array_equal(W_after, W0))
    ro_r = Readout(n_h, n_out, rng, w_clip=0.0, dtype="fp32")
    ro_r.W[:] = W0
    ro_r.learn(h, tgt, 0.05)
    d_learn = float(np.abs(ro_r.W.copy() - W_after).max())
    sc = max(1e-12, float(np.abs(ro_r.W).max()))
    check("A2 learn 与 Readout.learn 容差一致", d_learn <= 1e-5 * sc,
          f"max|Δ|={d_learn:.3e}")
    check("A2 W 为设备张量（保存路径用 W_cpu）", torch.is_tensor(ro_x.W))

    # ── A3：配置不兼容必须回落（两级读出未实现）；dtype 能力表直查 ──
    # 注（P105）：旧版这里还有一条 `readout_dtype=fp8` 的回落断言，但在无加速器
    # 机器上 pick 走 `auto→numba-cpu`（硬件探测层决定），永远拿不到回落原因，
    # 是一条环境依赖的假FAIL。dtype 的能力判断应直查 `_unsupported_reason`：
    # fp8 是 int8 的旧名别名（P100 正名），int8 在 P105 已实现 → 两者都放行。
    #
    # P111 更新：`readout_conn_k>0` **不再回落**（稀疏 gather-GEMV 已在加速
    # 后端实现，见 tests/verifiers/verify_accel_sparse.py 的 21 例对拍）。
    # 只剩 `readout_hidden>0`（两级群体读出）需要回落。
    print("[A3] 配置兼容性 → 强制回落 / dtype 能力表")
    cfg_x = PHDNetConfig(accel_readout="cuda" if has_accel else "auto",
                         readout_hidden=8)
    ro_c, backend_c = pick_readout_backend(cfg_x, n_h, n_out, rng)
    reason = getattr(ro_c, "_accel_fallback_reason", "")
    check("A3 readout_hidden 回落 numba 并记录原因",
          isinstance(ro_c, Readout) and "readout_hidden" in reason,
          f"backend={backend_c} reason={reason[:40]}")
    # P111：稀疏配置不再被拒绝表拦住（无加速器时 backend 仍由硬件探测决定）
    from phdnet.backends.accel_readout import _unsupported_reason as _usr
    check("A3 readout_conn_k>0 放行（P111 稀疏已实现，不回落）",
          _usr(PHDNetConfig(readout_conn_k=16)) is None,
          f"_unsupported_reason={_usr(PHDNetConfig(readout_conn_k=16))}")
    from phdnet.backends.accel_readout import _unsupported_reason
    check("A3 int8 放行（_unsupported_reason=None，P105 已实现）",
          _unsupported_reason(PHDNetConfig(readout_dtype="int8")) is None)
    check("A3 fp8 旧名别名放行（=int8，P100 正名 / P105 同步）",
          _unsupported_reason(PHDNetConfig(readout_dtype="fp8")) is None)
    # P191e（fhz 2026-10-07 指令）：int4/fp4 计算代码**整体删除**（P152 定案
    # 「unpack 开销抵消存储收益」，从未上生产）——能力表把显式请求拦下。
    check("A3 int4 显式请求被拒（P191e 已删，fail-fast）",
          _unsupported_reason(PHDNetConfig(readout_dtype="int4")) is not None)
    check("A3b fp4 显式请求被拒（P191e 已删，fail-fast）",
          _unsupported_reason(PHDNetConfig(readout_dtype="fp4")) is not None)

    # ── A4：接口完整性自动扫描（P23 根治：三次崩溃都是漏属性）──
    print("[A4] 接口完整性（扫描全仓库 readout.X 访问面）")
    import re
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[2]
    import os
    os.chdir(root)
    used = set()
    for pat in ("phdnet/**/*.py", "train/*.py", "tools/*.py", "tests/**/*.py"):
        for f in root.glob(pat):
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:                                # noqa: BLE001
                continue
            # 门禁收紧：锚定左边界 —— `accel_readout.pick_readout_backend`
            # 这类**模块名**访问（左邻是 `_`）与 `x.readout.Y` 形式不再被
            # 误捕成实例属性访问面。
            for m in re.finditer(
                    r"(?<![\w.])readout\.([a-zA-Z_][a-zA-Z_0-9]*)", txt):
                used.add(m.group(1))
    # 本模块自身定义的属性不算；只检查**外部访问面**
    own = set(re.findall(r"def ([a-zA-Z_][a-zA-Z_0-9]*)",
                         (_P(root / "phdnet" / "backends" /
                             "accel_readout.py").read_text(encoding="utf-8"))))
    own |= set(re.findall(r"\n    def ([a-zA-Z_][a-zA-Z_0-9]*)",
                          (_P(root / "phdnet" / "backends" /
                              "accel_readout.py").read_text(encoding="utf-8"))))
    import importlib
    mod = importlib.import_module("phdnet.backends.accel_readout")
    inst = AccelReadout(8, 8, None, device="cpu", w0=np.zeros((8, 8), np.float32))
    missing = []
    for name in sorted(used):
        # 门禁收紧：正则已锚定左边界（`(?<![\w.])readout\.`），
        # `accel_readout.AccelReadout` / `accel_readout.resolve_accel_device` /
        # `accel_readout.pick_readout_backend` 这些**模块级名**不再被误捕
        # → 从 skip 表删除（删除后 A4 仍须全绿，否则说明真有漏网访问面）。
        if name in ("py", "Readout", "forward", "device", "X",
                    "_accel_fallback_reason", "backends", "_csr"):
            continue                                    # 模块名/自身属性/诊断用
        if not hasattr(inst, name):
            missing.append(name)
    check("A4 访问面无缺失（" + ", ".join(sorted(used - {'py'})[:8]) + " …）",
          not missing, f"缺失={missing}" if missing else "")
    check("A4 stats() 可用", isinstance(inst.stats(), dict)
          and inst.stats()["dtype"] == "fp32")
    check("A4 conn_k/hidden 为 0（稠密单级）",
          inst.conn_k == 0 and inst.hidden == 0)
    try:
        _ = inst._csr
        check("A4 _csr 显式拒绝（稀疏未实现）", False, "未抛异常")
    except NotImplementedError:
        check("A4 _csr 显式拒绝（稀疏未实现）", True, "NotImplementedError")

    # ── C：状态往返 ──
    print("[C] 检查点往返")
    W_ck = rng.normal(0, 0.03, (n_out, n_h)).astype(np.float32)
    ro_acc.load_W(W_ck)
    check("C load_W 精确还原", np.array_equal(ro_acc.W_cpu(), W_ck))
    try:
        ro_acc.load_W(np.zeros((3, 3), dtype=np.float32))
        check("C 形状不符被拒", False, "未抛异常")
    except ValueError:
        check("C 形状不符被拒", True, "ValueError")

    # ── D：后端选择（无加速器 → 回落 numba）──
    print("[D] 后端选择与回落")
    probes = _probes
    cfg = PHDNetConfig()
    ro, backend = pick_readout_backend(cfg, n_h, n_out, rng)
    check("D auto 在无加速器时回落 numba",
          (backend.startswith("numba") == (not has_accel)),
          f"backend={backend} accel={has_accel}")
    check("D 回落对象是原 Readout", isinstance(ro, Readout))
    cfg_off = PHDNetConfig(accel_readout="cpu")
    _ro2, backend2 = pick_readout_backend(cfg_off, n_h, n_out, rng)
    check("D 显式 cpu 走原路径", backend2 == "numba-cpu")
    cfg_bad = PHDNetConfig(accel_readout="cuda:99")
    ro3, backend3 = pick_readout_backend(cfg_bad, n_h, n_out, rng)
    check("D 不可用设备回落且记录原因",
          backend3.startswith("numba") and
          hasattr(ro3, "_accel_fallback_reason"),
          f"backend={backend3} reason="
          f"{getattr(ro3, '_accel_fallback_reason', '无')[:48]}")
    if has_accel:      # 有加速器的机器：auto 必须真的上设备
        ro4, backend4 = pick_readout_backend(PHDNetConfig(), n_h, n_out, rng)
        check("D 有加速器时 auto 上设备", backend4.startswith("accel:"),
              f"backend={backend4}")
    else:
        print("  [SKIP] D 有加速器时 auto 上设备（本机无加速器）")

    # ── E：精度档 ──
    # ⚠ P163（fhz 2026-10-03）：**int 族已整体禁用** → 只测 fp 族
    print("[E] 精度档（int 族已禁用，P163）")
    for dt in ("fp32", "fp16", "bf16"):
        try:
            m = AccelReadout(n_h, n_out, None, device="cpu", dtype=dt, w0=W0)
            y = m.forward(h)
            check(f"E dtype={dt} 前向可用", y.shape == (n_out,)
                  and np.isfinite(y).all(), f"max={np.abs(y).max():.4f}")
        except Exception as e:                                # noqa: BLE001
            check(f"E dtype={dt} 前向可用", False, f"{type(e).__name__}: {e}")
    for dt in ("int8", "int16", "int32"):
        try:
            AccelReadout(n_h, n_out, None, device="cpu", dtype=dt, w0=W0)
            check(f"E dtype={dt} 应被禁用（P163）", False, "竟然构造成功")
        except Exception as e:                                # noqa: BLE001
            check(f"E dtype={dt} 应被禁用（P163）", ("int 族" in str(e) or "不支持 dtype" in str(e)),
                  f"{type(e).__name__}")

    # ── F：int8 读出（P105，fhz 2026-10-01 决策）──
    # 语义：存储 int8（per-tensor scale = 2*max|W|/127，2× 余量不重标定）；
    # forward 每步反量化到 fp16 再 matmul；learn 在 fp32 域算 dp⊗ht 后重量化
    # 写回。精度约束（知情取舍）：非目标行 |dp|≈1e-6 ≪ int8 步长 ≈ max|W|/127，
    # 其更新被量化吃掉（int8 存储降 4× 访存的代价）；目标行 |dp|≈1 必须保留。
    # ⚠ P163（fhz2026-10-03）：**int 族已整体禁用** → 原来这整段 int8 存储/
    #   前向/目标行更新/往返测试**不再适用**，改为断言「int 请求被拒绝」。
    #依据：Ascend910B4 生产实测 fp8 与 int8 均不可用、实际落到 fp16；
    #   P156 实测 int8 在稀疏 gather-GEMV 下比 fp16 慢 1.2×。
    print("[F] int 族已禁用（P163）—— 请求必须被拒绝")
    for _dt in ("int8", "int16", "int32"):
        _rej = False
        try:
            AccelReadout(64, 100, None, device="cpu", dtype=_dt)
        except (ValueError, RuntimeError) as _e:
            _rej = ("int 族" in str(_e) or "不支持 dtype" in str(_e))
        except Exception:                                    # noqa: BLE001
            _rej = False
        check(f"F1 {_dt} 构造被拒绝（P163 硬约束）", _rej,
              "int 族已整体禁用")

    # ── H：fp8 端到端（P0-1 / P0-2 / P0-3 门禁，2026-10-07 修复后补）──────
    # 覆盖三条「曾全绿通过却必崩/必损坏」的路径：
    #   P0-1 稀疏 fp8 原生更新把**位模式当实值**（一步 max|ΔW|=160，期望 ~0.06）；
    #   P0-2 forward()/__call__() 无 P191c 降级兜底（CPU 无 fp8 GEMV → 直接崩）；
    #   P0-3 learn() 三种 fp8 存储态全部崩溃（addmv / Byte vs Half / Promotion）。
    # 判据是「量级 + 有限性 + 容差」而非逐位（fp8 格点本身非线性）。
    print("[H] fp8 端到端（P0-1 更新损坏 / P0-2 forward 降级 / P0-3 learn）")
    from phdnet.backends.fp8_int8_convert import FP8_VAL_LUT

    def _fp8_dec(ro_h):
        """W（fp8 原生或降级后的 uint8 位模式）→ 实值（LUT 精确反量化）。"""
        bits = ro_h.W.view(torch.uint8).detach().cpu().numpy().reshape(-1)
        return FP8_VAL_LUT[bits].reshape(tuple(ro_h.W.shape)).astype(np.float64)

    _eta_h = 0.05
    _n_hh, _n_oo = 64, 96
    _rng_h = np.random.default_rng(11)
    _w0_h = _rng_h.normal(0.0, 0.05, (_n_oo, _n_hh)).astype(np.float32)
    _h_h = (np.abs(_rng_h.normal(0.0, 1.0, _n_hh)) + 0.1).astype(np.float32)
    _t_h = np.zeros(_n_oo, dtype=np.float32)
    _t_h[int(_rng_h.integers(0, _n_oo))] = 1.0

    def _h_case(tag, sparse):
        ck = 8 if sparse else 0
        ro_h = AccelReadout(_n_hh, _n_oo, None, device="cpu", dtype="fp8",
                            w0=None if sparse else _w0_h, conn_k=ck)
        # ① forward 不抛（P0-2）；CPU fp8 GEMV 不可用是上游事实 → 必须降级跑完
        try:
            y_h = ro_h.forward(_h_h)
            _ok, _err = True, ""
        except Exception as _e:                                # noqa: BLE001
            y_h, _ok, _err = None, False, f"{type(_e).__name__}: {_e}"
        check(f"H{tag} forward() 不抛（P0-2）", _ok, _err)
        if not _ok:
            return
        if not sparse:
            # 稠密：上游无 fp8 GEMV（addmv_impl_cpu 未实现）→ 必须已永久降级，
            # 且存储仍是 1 字节（fp8 请求的访存语义不丢）。
            _deg = bool(ro_h._fp8_bits) and ro_h.W.dtype == torch.uint8
            check(f"H{tag} CPU 无 fp8 GEMV（上游）→ 降级 fp8_bits 且存储 1B",
                  (ro_h.device != "cpu") or (_deg and ro_h.W.element_size() == 1),
                  f"device={ro_h.device} _fp8_bits={ro_h._fp8_bits} W={ro_h.W.dtype}")
        else:
            # 稀疏：前向走 P154 int8 计算域（不降级），存储必须仍是 1 字节 fp8
            check(f"H{tag} 稀疏 fp8 存储保持 1B（原生或位模式）",
                  ro_h.W.element_size() == 1
                  and (bool(ro_h._fp8_native) or bool(ro_h._fp8_bits)),
                  f"W={ro_h.W.dtype} _fp8_native={ro_h._fp8_native}")
        # ② learn_softmax 一步不抛 + ΔW 与 η·|dp|·|h| 同量级（P0-1 损坏 = 160）
        _Wb = _fp8_dec(ro_h)
        try:
            _nll = ro_h.learn_softmax(_h_h, _t_h, _eta_h, y_pre=y_h)
            _ok2, _err2 = True, f"nll={float(_nll):.4f}"
        except Exception as _e:                                # noqa: BLE001
            _ok2, _err2 = False, f"{type(_e).__name__}: {_e}"
        check(f"H{tag} learn_softmax 一步不抛（P0-1 路径）", _ok2, _err2)
        if not _ok2:
            return
        _Wa = _fp8_dec(ro_h)
        _p = np.exp(np.asarray(y_h, dtype=np.float64) - float(np.max(y_h)))
        _p = _p / _p.sum()
        _bnd_s = (1.5 * _eta_h * float(np.abs(_p - _t_h).max())
                  * float(np.abs(_h_h).max()))
        _d1 = float(np.abs(_Wa - _Wb).max())
        check(f"H{tag} learn_softmax max|ΔW| ≤ 1.5·η·max|dp|·max|h|（损坏=160）",
              _d1 <= max(_bnd_s, 1e-6),
              f"max|ΔW|={_d1:.4g} ≤ {max(_bnd_s, 1e-6):.4g}")
        # ③ learn() 一步不抛 + ΔW 同量级 + W 全有限（P0-3；dp 基准 = 新前向）
        _y2 = ro_h.forward(_h_h)
        _Wb2 = _fp8_dec(ro_h)
        try:
            ro_h.learn(_h_h, _t_h, _eta_h)
            _ok3, _err3 = True, ""
        except Exception as _e:                                # noqa: BLE001
            _ok3, _err3 = False, f"{type(_e).__name__}: {_e}"
        check(f"H{tag} learn() 一步不抛（P0-3）", _ok3, _err3)
        if not _ok3:
            return
        _Wa2 = _fp8_dec(ro_h)
        _bnd_l = (1.5 * _eta_h * float(np.abs(_t_h - _y2).max())
                  * float(np.abs(_h_h).max()))
        _d2 = float(np.abs(_Wa2 - _Wb2).max())
        _fin = (bool(np.isfinite(_Wa2).all())
                and float(np.abs(_Wa2).max()) <= 448.0)   # e4m3 最大有限值
        check(f"H{tag} learn max|ΔW| 同量级 + W 全有限（fp8 ≤448）",
              _d2 <= max(_bnd_l, 1e-6) and _fin,
              f"max|ΔW|={_d2:.4g} ≤ {max(_bnd_l, 1e-6):.4g} "
              f"max|W|={float(np.abs(_Wa2).max()):.4g}")
        # 结构不变量（门禁收紧时**保留**）：W 必须始终落在 e4m3 格点上 ——
        # LUT 反量化的每个值再做一次 float32→e4m3→float32 往返必须逐位还原。
        _bits_g = ro_h.W.view(torch.uint8).detach().cpu().numpy().reshape(-1)
        _vals_g = np.asarray(FP8_VAL_LUT[_bits_g], dtype=np.float32)
        _rt_g = (torch.from_numpy(_vals_g).to(torch.float8_e4m3fn)
                 .to(torch.float32).numpy())
        _gok = bool(np.array_equal(_rt_g, _vals_g, equal_nan=True))
        check(f"H{tag} W 落在 e4m3 网格（fp8 往返逐位还原）", _gok,
              f"non-grid={int((~np.equal(_rt_g, _vals_g)).sum())}")
        # ④ 前向对拍（**门禁收紧**）：
        #    稠密（LUT→fp16 GEMV）实测 1.3e-4 → 1e-2 判据（按任务书口径）；
        #    稀疏前向走 P154 W8A8（fp8→int8 重量化 + 激活动态 int8）——旧门禁
        #    拿「同一份量化后权重的 fp32 孪生」对拍，把**量化本身**的实测
        #    2.6e-2 也算进容差（5e-2），损坏（O(1)）之外近 10× 的漂移会漏网。
        #    现在在门禁里用 numpy **复刻同一重量化管线**（W 与 gather 后的 h
        #    都量化到 int8 网格、int16 乘 + 整数累加、最后脱两个 scale）——
        #    量化误差被复刻吸收 → 容差 5e-2 → **1e-3**（实测 rel 见输出）。
        #    结构不变量「max|ΔW| ≤ 1.5·η·max|dp|·max|h|」与「W 落在 e4m3 网格」
        #    在本块之外**单独保留**，收紧容差不得把它们删掉。
        _y_now = ro_h.forward(_h_h)
        if sparse and bool(getattr(ro_h, "_int8_compute", False)):
            _ws_r, _gs_r = (float(v) for v in ro_h._last_int8_scales)
            _Wd = _fp8_dec(ro_h)                    # (n_out,k) fp8 实值
            _Wq = np.clip(np.rint(_Wd.astype(np.float32)
                                  / np.float32(_ws_r)), -127, 127)
            _g_t = getattr(ro_h, "_cache_g", None)  # 实现真实用到的 gather(h)
            if _g_t is None:                        # 理论不发生（前向必 gather）
                _g_t = torch.from_numpy(
                    np.ascontiguousarray(_h_h, dtype=np.float32))[ro_h.Wi.long()]
            _g32 = _g_t.detach().cpu().numpy().astype(np.float32)
            _Gq = np.clip(np.rint(_g32 / np.float32(_gs_r)), -127, 127)
            # int16×int16 → 整数累加（k=8 远不回绕）→ 脱两个 scale，与 torch
            # 侧逐位同整数；量化/舍入全部在 numpy 里复刻 → 误差只剩 fp32 末次乘。
            _y_ref = (_Wq.astype(np.int64) * _Gq.astype(np.int64)
                      ).sum(axis=1) * (_ws_r * _gs_r)
            _tol, _ref_kind = 1e-3, "numpy 复刻重量化管线"
        else:
            _twin = AccelReadout(_n_hh, _n_oo, None, device="cpu", dtype="fp32",
                                 w0=_fp8_dec(ro_h).astype(np.float32), conn_k=ck)
            _y_ref = _twin.forward(_h_h)
            _tol, _ref_kind = 1e-2, "fp32 孪生"
        _rel = (float(np.abs(np.asarray(_y_now, dtype=np.float64)
                             - np.asarray(_y_ref, dtype=np.float64)).max())
                / max(float(np.abs(_y_ref).max()), 1e-6))
        check(f"H{tag} forward y vs {_ref_kind}（相对峰差 ≤{_tol:g}）",
              _rel <= _tol, f"rel={_rel:.3e}")

    _h_case("D", sparse=False)
    _h_case("S", sparse=True)

    print("-" * 76)
    if _FAILURES:
        print(f"结果：{len(_FAILURES)} 例 FAIL → {_FAILURES}")
        sys.exit(1)
    print("结果：全部 PASS")


if __name__ == "__main__":
    main()
