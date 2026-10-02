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
    check("A3 int4 仍拒绝（910B 无 INT4 矩阵乘单元）",
          _unsupported_reason(PHDNetConfig(readout_dtype="int4")) is not None)

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
            for m in re.finditer(r"readout\.([a-zA-Z_][a-zA-Z_0-9]*)", txt):
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
        if name in ("py", "Readout", "forward", "device", "X",
                    "_accel_fallback_reason", "backends", "_csr",
                    "AccelReadout",          # from .accel_readout import AccelReadout
                    "resolve_accel_device"):  # from .accel_readout import resolve_accel_device

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
    print("[E] 精度档")
    for dt in ("fp32", "fp16", "bf16", "int8"):
        try:
            m = AccelReadout(n_h, n_out, None, device="cpu", dtype=dt, w0=W0)
            y = m.forward(h)
            check(f"E dtype={dt} 前向可用", y.shape == (n_out,)
                  and np.isfinite(y).all(), f"max={np.abs(y).max():.4f}")
        except Exception as e:                                # noqa: BLE001
            check(f"E dtype={dt} 前向可用", False, f"{type(e).__name__}: {e}")

    # ── F：int8 读出（P105，fhz 2026-10-01 决策）──
    # 语义：存储 int8（per-tensor scale = 2*max|W|/127，2× 余量不重标定）；
    # forward 每步反量化到 fp16 再 matmul；learn 在 fp32 域算 dp⊗ht 后重量化
    # 写回。精度约束（知情取舍）：非目标行 |dp|≈1e-6 ≪ int8 步长 ≈ max|W|/127，
    # 其更新被量化吃掉（int8 存储降 4× 访存的代价）；目标行 |dp|≈1 必须保留。
    print("[F] int8 存储 / 前向 / 目标行更新 / 多步稳定性")
    _n8, _h8 = 100, 64
    _rng8 = np.random.default_rng(20261001)
    _W08 = (_rng8.normal(0.0, 0.05, (_n8, _h8)) * np.sqrt(_h8)).astype(np.float32)
    _h8v = _rng8.normal(0.0, 1.0, _h8).astype(np.float32)
    _tgt8 = np.zeros(_n8, dtype=np.float32)
    _tgt8[7] = 1.0
    try:
        ro_i = AccelReadout(_h8, _n8, None, device="cpu", dtype="int8", w0=_W08)
        check("F1 int8 构造且 W.dtype == torch.int8", ro_i.W.dtype == torch.int8,
              f"dtype={ro_i.W.dtype}")
        _amax = float(np.abs(_W08).max())
        check("F1 scale = 2*max|W|/127（2× 余量纪律）",
              abs(ro_i._wscale - 2.0 * _amax / 127.0) < 1e-9,         # noqa: SLF001
              f"wscale={ro_i._wscale:.6g}（码余量 = {127 - _amax / ro_i._wscale:.1f}）")
        check("F1 存储 1 B/权重",
              ro_i.stats()["storage_MB"] * 1e6 == ro_i.n_synapses(),
              f"{ro_i.stats()['storage_MB'] * 1e6:.0f} B / {ro_i.n_synapses():,} 权重")
        _y8 = ro_i.forward_dev(_h8v)
        check("F2 forward_dev 返回有限值", torch.is_tensor(_y8)
              and bool(torch.isfinite(_y8).all()),
              f"max|y|={float(_y8.abs().max()):.4f}")
        # 目标行更新 vs fp16 参考对拍：同 W0、同 (h, target, η)。
        ro_r16 = AccelReadout(_h8, _n8, None, device="cpu", dtype="fp16", w0=_W08)
        _W0q = ro_i.W_cpu().copy()             # int8 反量化后的实值（共同起点）
        ro_r16.load_W(_W0q)
        _nll_i = ro_i.learn_softmax(_h8v, _tgt8, 0.05, target_idx=7)
        _nll_r = ro_r16.learn_softmax(_h8v, _tgt8, 0.05, target_idx=7)
        _d_i = ro_i.W_cpu()[7] - _W0q[7]
        _d_r = ro_r16.W_cpu()[7] - _W0q[7]
        _cos = float(np.dot(_d_i, _d_r)
                     / (np.linalg.norm(_d_i) * np.linalg.norm(_d_r) + 1e-30))
        _mag = float(np.abs(_d_i).max()) / max(1e-30, float(np.abs(_d_r).max()))
        check("F3 目标行更新方向与 fp16 参考一致（cos ≥ 0.99）",
              0.99 <= _cos <= 1.0 + 1e-9, f"cos={_cos:.4f}")
        check("F3 目标行更新幅度合理（0.5 ≤ |Δ|int8/|Δ|fp16 ≤ 1.5）",
              0.5 <= _mag <= 1.5, f"|Δ_i|={np.abs(_d_i).max():.4f} "
              f"|Δ_r|={np.abs(_d_r).max():.4f} 比值={_mag:.3f}")
        check("F3 NLL 与 fp16 参考容差一致（≤1e-4 相对）",
              abs(_nll_i - _nll_r) <= 1e-4 * max(1e-12, abs(_nll_r)),
              f"{_nll_i:.6f} vs {_nll_r:.6f}")
        # 非目标行：更新被量化吃掉是**知情取舍**——绝大多数码应纹丝不动
        # （个别元素恰跨半步长边界 ±1 码属 RNE 正常舍入，值变 ≤1 步长）。
        _dn = np.abs(ro_i.W[3].cpu().numpy().astype(np.int64)
                     - np.round(_W0q[3] / ro_i._wscale).astype(np.int64))  # noqa: SLF001
        check("F3 非目标行更新被量化吸收（码变化 ≤1 步，知情取舍）",
              int(_dn.max()) <= 1,
              f"非目标行最大码变化 = {int(_dn.max())}"
              f"（≤1 码 = {ro_i._wscale:.3g} 实值）")                      # noqa: SLF001
        for _k in range(20):
            _yk = ro_i.forward_dev(_h8v)
            ro_i.learn_softmax(_h8v, _tgt8, 0.05, target_idx=_k % _n8)
        _yk = ro_i.forward_dev(_h8v)
        check("F4 20 步 learn+forward 无崩溃无 NaN",
              bool(torch.isfinite(_yk).all())
              and int(ro_i.W.min()) >= -127 and int(ro_i.W.max()) <= 127,
              f"码范围 [{int(ro_i.W.min())}, {int(ro_i.W.max())}] ⊆ [-127, 127]")
        # w_clip：int8 的 clamp_ 对码张量无意义 → 改为重量化前裁剪。
        _ro_c = AccelReadout(_h8, _n8, None, device="cpu", dtype="int8",
                             w0=_W08, w_clip=0.05)
        _ro_c.learn_softmax(_h8v, _tgt8, 0.05, target_idx=7)
        _lim = int(np.floor(0.05 / _ro_c._wscale + 1e-9))                    # noqa: SLF001
        check("F5 w_clip 在重量化时裁剪（码 ≤ w_clip/scale）",
              int(_ro_c.W.abs().max()) <= _lim,
              f"max|码|={int(_ro_c.W.abs().max())} ≤ {_lim}（w_clip=0.05）")
        # P148：`dtype="fp8"` 已**不再是 int8 的别名**（P147 起它是独立路径：
        # 有原生 fp8 算子就用 fp8，否则转int8），且 fp8 会被**运行时探测**。
        # → 断言改为「fp8 走的是 fp8 或 int8，且**探测被真正执行过**」
        #    （审计 BUG-1：此前 fp8 被别名归一成 int8，探测是死代码）。
        _ro_a = AccelReadout(_h8, _n8, None, device="cpu", dtype="fp8", w0=_W08)
        _is8 = str(_ro_a.W.dtype) in ("torch.int8", "torch.float8_e4m3fn")
        check("F6a dtype=fp8 构造出 8-bit 码本（fp8 或 int8）",
              _is8, f"W.dtype={_ro_a.W.dtype}")
        check("F6b fp8 **运行时探测被真正执行**"
              + ("（原生 fp8）" if getattr(_ro_a, "_fp8_native", False)
                 else "（回落 int8）" if getattr(
                     _ro_a, "_fp8_fallback_to_int8", False) else "（未执行←回归!)"),
              getattr(_ro_a, "_fp8_native", False)
              or getattr(_ro_a, "_fp8_fallback_to_int8", False)
              or getattr(_ro_a, "_fp8_cap", None) is not None)
        # 检查点往返：W_cpu 给实值、load_W 重量化，一次往返后稳定（不再漂移）。
        _v1 = ro_i.W_cpu()
        ro_i.load_W(_v1)
        _v2 = ro_i.W_cpu()
        check("F7 W_cpu/load_W 实值往返稳定", np.array_equal(_v1, _v2),
              f"max|Δ|={float(np.abs(_v1 - _v2).max()):.2e}")
    except Exception as e:                                    # noqa: BLE001
        check("F1 int8 构造", False, f"{type(e).__name__}: {e}")

    print("-" * 76)
    if _FAILURES:
        print(f"结果：{len(_FAILURES)} 例 FAIL → {_FAILURES}")
        sys.exit(1)
    print("结果：全部 PASS")


if __name__ == "__main__":
    main()
