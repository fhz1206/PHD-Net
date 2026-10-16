"""P195 门禁：DeepSeek-V4-Flash 主权重 + 低精度影子副本的不变量。

跑法：py -3.14 tests/gates/gate_readout_master_weight.py

判据（全部**可执行**，不是静态断言）：
  A1 主权重是 fp32，影子是请求的低精度（fp8 → uint8 位模式）
  A2 更新落在 master 上（影子只是快照）
  A3 影子按 refresh 间隔刷新（不是每步）
  A4 非目标行小更新被保留（P110 反证的解除）
  B1 **所有存储 dtype** 下 master 都被更新（BugC 回归：分派与 dtype 无关）
  B2 master 建成后影子与 master 一致（同尺度）
  C1 ckpt 往返：master fp32 **逐位**恢复（不被 ckpt_dtype 压缩）
  C2 非 master 模式：往返后按影子粒度恢复（与训练期同尺子）
  C3 原生 fp8 的 ckpt 恢复不崩（BugA 回归）
  C4 meta.ro_master 记录正确；npz 里**没有**同名数组（BugE）
  D1 torch.compile 在主权重模式下被禁用（否则静默更新影子）
  D2 NPU Graph 捕获前 master 已建（否则 replay 用错地址）
  D3 micro_batch 的 commit_grad 走 master（不绕过去）
  E1 master off（默认）时行为与旧版一致（无 master 属性被创建）
"""

import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "train")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                        # noqa: E402
import torch                                              # noqa: E402

from phdnet.config import PHDNetConfig                    # noqa: E402
from phdnet.model import PHDNet                           # noqa: E402
from phdnet.word_lm import PHDWordLM                      # noqa: E402

CASES: list = []


def check(ok: bool, name: str, detail: str = "") -> None:
    CASES.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))


TEXT = "小猫追蝴蝶。小狗摇尾巴。鸟儿飞过树梢。" * 20
SEG_KW = dict(max_len=4, min_count=2, min_entropy=0.5)


def _mk(master=True, dtype="fp8", refresh=8, mb=1, **kw):
    cfg = PHDNetConfig(
        n_input=64, n_sdr=128, n_mid=64, n_top=32, k_sparse=16,
        n_readout=0, readout_softmax=True, eta_readout=0.15,
        seed=11, step_profiling=False, accel_readout="cpu:0",
        readout_dtype=dtype, readout_master_weight=master,
        readout_shadow_refresh=refresh, micro_batch=mb,
        # P197：readout_replay 默认 True 会额外调 learn_softmax（每次 replay
        # 也推进 _shadow_age）→ A3 的「refresh 间隔内影子不变」计数被污染。
        # 本门禁测的是**主权重机制本身**，回放是正交机制 → 关掉以隔离。
        readout_replay=False,
        nll_sync_every=1, eta_pc=0.05, eta_oja=0.05, pc_dev_steps=400,
        big_ltm=True, big_ltm_N=1 << 16, big_ltm_m=8, big_ltm_k=2, **kw)
    lm = PHDWordLM(TEXT, cfg, seg_kwargs=SEG_KW)
    return lm


def _drive(lm, n=12):
    rng = np.random.default_rng(123)
    for _ in range(n):
        x = rng.normal(0, 1, lm.net.cfg.n_input).astype(np.float32)
        t = np.eye(lm.net.n_out)[int(rng.integers(lm.net.n_out))]
        lm.net.step(x, target=t)


# ══════════════════════════════════════════════════════════════════════
print("=" * 74)
print("P195 主权重 + 低精度影子（DeepSeek-V4-Flash 模式）不变量")
print("=" * 74)

# ── A. 基本结构 ────────────────────────────────────────────────────────
print("\n[A] 结构：master fp32 + 低精度影子")
lm = _mk(master=True, dtype="fp8", refresh=4)
ro = lm.net.readout
check(isinstance(ro, type(lm.net.readout)) and hasattr(ro, "_W_master"),
      "A0 加速后端（含 master 属性）", f"{type(ro).__name__}")
check(ro._master_weight is True, "A1a master_weight 标志为真")
_drive(lm, 8)
check(ro._W_master is not None and ro._W_master.dtype == torch.float32,
      "A1b master 是 fp32", f"dtype={getattr(ro._W_master, 'dtype', None)}")
check(str(ro.W.dtype) in ("torch.uint8", "torch.float8_e4m3fn"),
      "A1c 影子是 fp8 存储", f"{ro.W.dtype}")

_m0 = ro._W_master.detach().clone()
_drive(lm, 3)
_dm = float((ro._W_master.detach() - _m0).abs().max())
check(_dm > 0, "A2 更新落在 master 上", f"max|Δ|={_dm:.3e}")

# 影子刷新间隔：refresh=4。先清零计数，再跑 3 步（<4）→ 不该变
ro._shadow_age = 0
ro._master_to_shadow()          # 先同步一次，排除「已积压」的干扰
_w_before = ro.W.clone()
_drive(lm, 3)
_unchanged = bool((ro.W == _w_before).all().item())
check(_unchanged, "A3a 影子在 refresh 间隔内不变（不每步重量化）")
_drive(lm, 2)                       # 累计 5 步 > 4 → 应刷新
_changed = not bool((ro.W == _w_before).all().item())
check(_changed, "A3b 影子到间隔后刷新")

# ── A4：非目标行小更新被保留（P110 反证的解除）──────────────────────
_mst = ro._W_master
_small = int(((_mst.abs() > 0) & (_mst.abs() < 2.0 ** -6)).sum().item())
check(_small > 0, "A4 master 保有亚 fp8 粒度的值（非目标行更新未被舍入）",
      f"{_small}/{_mst.numel()} 个 |w|<2^-6")

# ── B. 所有存储 dtype 下 master 均被更新（BugC 回归）─────────────────
print("\n[B] 更新分派与存储 dtype 无关（BugC 回归）")
for _dt in ("fp8", "fp32", "fp16", "bf16"):
    _lm = _mk(master=True, dtype=_dt, refresh=4)
    _drive(_lm, 6)
    _m = getattr(_lm.net.readout, "_W_master", None)
    if _m is None:
        check(False, f"B1 {_dt}: master 已建成")
        continue
    _b = _m.detach().clone()
    _drive(_lm, 3)
    _c = float((_m.detach() - _b).abs().max())
    check(_c > 0, f"B1 {_dt}: master 被更新", f"max|Δ|={_c:.3e}")

# ── C. 检查点往返 ─────────────────────────────────────────────────────
print("\n[C] 检查点往返（master fp32 逐位恢复）")
from ckpt_1b import load_model, save_model              # noqa: E402


def _roundtrip(lm, ckpt_dtype=""):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ck.npz"
        meta = save_model(p, lm, lm.net.cfg, done=10, ckpt_dtype=ckpt_dtype)
        cfg = PHDNetConfig(**{**lm.net.cfg.__dict__})
        lm2 = PHDWordLM(TEXT, cfg, seg_kwargs=SEG_KW)
        load_model(p, lm2)
        return lm2, meta, p


lm = _mk(master=True, dtype="fp8", refresh=4)
_drive(lm, 8)
_m0 = lm.net.readout._W_master.detach().cpu().numpy().copy()
_td = tempfile.mkdtemp()
_p = Path(_td) / "ck.npz"
meta = save_model(_p, lm, lm.net.cfg, done=10, ckpt_dtype="fp8")
cfg = PHDNetConfig(**{**lm.net.cfg.__dict__})
lm2 = PHDWordLM(TEXT, cfg, seg_kwargs=SEG_KW)
load_model(_p, lm2)
_m1 = lm2.net.readout._W_master.detach().cpu().numpy()
check(float(np.abs(_m0 - _m1).max()) == 0.0,
      "C1 master 往返逐位一致（不被 ckpt_dtype 压缩）",
      f"max|Δ|={float(np.abs(_m0 - _m1).max()):.3e}")

lm_nb = _mk(master=False, dtype="fp8")
_drive(lm_nb, 8)
_w0 = lm_nb.net.readout.W_cpu().copy()
lm2nb, _, _ = _roundtrip(lm_nb, ckpt_dtype="fp8")
_w1 = lm2nb.net.readout.W_cpu()
_dw = float(np.abs(_w0 - _w1).max())
check(_dw <= 2.0 ** -6 * 1.001,
      "C2 非 master 往返按影子粒度（与训练期同尺子）", f"max|Δ|={_dw:.3e}")

try:
    _h = np.random.default_rng(0).normal(
        0, 1, lm2.net.readout.n_h).astype(np.float32)
    _y = lm2.net.readout(_h)
    check(_y is not None and len(_y) > 0,
          "C3 fp8 检查点恢复后前向正常（BugA 回归）",
          f"n_h={lm2.net.readout.n_h}")
except TypeError as e:
    check(False, "C3 fp8 检查点恢复后前向正常（BugA 回归）", str(e)[:80])

check(meta.get("ro_master") is True, "C4a meta.ro_master=True")
with np.load(_p, allow_pickle=False) as _z:
    check("ro_master" not in _z.files,
          "C4b npz 里无同名 ro_master 数组（BugE：单一来源）")
    check(np.asarray(_z["ro_W"]).dtype == np.float32,
          "C4c ro_W 存为 fp32（未被压缩）",
          f"{np.asarray(_z['ro_W']).dtype}")

# ── D. 与其它机制的交互 ──────────────────────────────────────────────
print("\n[D] 与 torch.compile / NPU Graph / micro_batch 的交互")
_lm = _mk(master=True, dtype="fp8", refresh=4)
check(_lm.net.readout._compiled is False,
      "D1 torch.compile 在主权重模式下禁用（否则静默更新影子）")

_lm2 = _mk(master=True, dtype="fp8", refresh=4, beta={"p0"})
_ro2 = _lm2.net.readout
check(getattr(_ro2, "_W_master", None) is not None or not _ro2._npu_graph_ready,
      "D2 NPU Graph 捕获前 master 已建（或图未启用）",
      f"graph_ready={_ro2._npu_graph_ready}")

_lm3 = _mk(master=True, dtype="fp8", refresh=100, mb=4)
_drive(_lm3, 8)                    # 2 个完整 micro_batch
_m3 = _lm3.net.readout._W_master
check(_m3 is not None and float(_m3.abs().sum()) > 0,
      "D3 micro_batch 的 commit_grad 走 master", f"Σ|W|={float(_m3.abs().sum()):.3f}")

# ── E. 默认关闭时零参与 ──────────────────────────────────────────────
print("\n[E] 默认（master off）零参与")
_lm4 = _mk(master=False, dtype="fp8")
check(getattr(_lm4.net.readout, "_W_master", None) is None,
      "E1 master off 时不建 master")
_drive(_lm4, 4)
check(getattr(_lm4.net.readout, "_W_master", None) is None,
      "E2 master off 时训练后仍无 master（默认路径不受影响）")


def main() -> None:
    bad = [n for n, ok in CASES if not ok]
    print(f"\n{'PASS' if not bad else 'FAIL'}: {len(CASES) - len(bad)}/{len(CASES)} 项通过")
    for n in bad:
        print(f"  x {n}")
    shutil.rmtree(_td, ignore_errors=True)
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
