"""AccelReadout learn_softmax 双路径矩阵验证（P55，2026-09-29）。

**背景**：fhz 服务器 NPU 首次真跑生产训练即崩——P45 的「target_idx 路径就地改
p」只实现在 **eager 分支**，`_train_step_core`（torch.compile 融合核）仍是旧的
`dp = p - t32`，`t32=None` 时 `FakeTensor - None` 崩 dynamo。既有
`verify_accel_readout.py` 的 learn_softmax 用例**只走 target 数组路径**，从未
传 `target_idx` → 恰好漏掉崩的那条。

本脚本补齐 2×2 矩阵（target 数组 / target_idx × eager / compiled），
判据 = 容差一致（跨库归约不同，max|Δ| ≤ 1e-5；fp32 CPU）。
运行：``python tests/verifiers/verify_accel_readout_p55.py``（退出码 0 = PASS）
"""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "train"))

import torch                                                     # noqa: E402
from phdnet.backends.accel_readout import AccelReadout           # noqa: E402

CASES: list[tuple[str, bool]] = []
TOL = 1e-5                                        # fp32 跨路径容差


def check(name: str, cond: bool) -> None:
    CASES.append((name, bool(cond)))
    print(f"  {'✓' if cond else '✗'} {name}")


def _make(n_h=64, n_out=1024, seed=0, **kw):
    g = np.random.default_rng(seed)
    w0 = g.normal(0.0, 0.05, (n_out, n_h)).astype(np.float32)
    h = np.ascontiguousarray(g.normal(0.0, 1.0, n_h).astype(np.float32))
    return w0, h


def _onehot(n_out, idx):
    t = np.zeros(n_out, dtype=np.float32)
    t[idx] = 1.0
    return t


def _max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _main() -> int:
    print("P55：learn_softmax target_idx × compiled 2×2 矩阵")

    # ── 1. 崩溃回归：target_idx 路径在 compiled 与 eager 下都能跑 ──────────
    for compiled in (False, True):
        w0, h = _make()
        ro = AccelReadout(w0.shape[1], w0.shape[0], w0=w0.copy(),
                          compile=compiled)
        try:
            nll = ro.learn_softmax(h, None, 0.05, y_pre=None,
                                   target_idx=123)
            check(f"target_idx + compiled={compiled} 不崩且返回 float",
                  isinstance(nll, float))
        except Exception as e:                            # noqa: BLE001
            check(f"target_idx + compiled={compiled} 不崩且返回 float", False)
            print(f"    ! {type(e).__name__}: {e}")

    # ── 2. 数值等价：target_idx 与 target 数组（onehot）结果一致 ───────────
    #    （同一 W 初值、同 h、同 correct、同 eta，两路径更新应逐元素一致）
    w0, h = _make(seed=1)
    n_out = w0.shape[0]
    correct = 777
    ro_idx = AccelReadout(w0.shape[1], n_out, w0=w0.copy(),
                          compile=False)
    nll_idx = ro_idx.learn_softmax(h, None, 0.05, target_idx=correct)
    ro_arr = AccelReadout(w0.shape[1], n_out, w0=w0.copy(),
                          compile=False)
    nll_arr = ro_arr.learn_softmax(h, _onehot(n_out, correct), 0.05)
    check("eager：target_idx ≡ onehot 数组（nll 一致）",
          abs(nll_idx - nll_arr) <= TOL)
    check("eager：target_idx ≡ onehot 数组（W 一致）",
          _max_abs_diff(ro_idx.W.detach().cpu().numpy(),
                        ro_arr.W.detach().cpu().numpy()) <= TOL)

    # ── 3. compiled vs eager：两种路径下 W 更新一致 ────────────────────────
    for tag, tgt, idx in (("target_idx", None, 555),
                          ("target 数组", _onehot(w0.shape[0], 555), None)):
        ro_c = AccelReadout(w0.shape[1], w0.shape[0], w0=w0.copy(),
                            compile=True)
        ro_e = AccelReadout(w0.shape[1], w0.shape[0], w0=w0.copy(),
                            compile=False)
        nll_c = ro_c.learn_softmax(h, tgt, 0.05, target_idx=idx)
        nll_e = ro_e.learn_softmax(h, tgt, 0.05, target_idx=idx)
        dW = _max_abs_diff(ro_c.W.detach().cpu().numpy(),
                           ro_e.W.detach().cpu().numpy())
        check(f"compiled ≡ eager（{tag}，W 更新一致）", dW <= TOL)
        check(f"compiled ≡ eager（{tag}，nll 一致）",
              abs(nll_c - nll_e) <= TOL)

    # ── 4. 多步训练烟测：compiled 路径连续 10 步不崩（dynamo 重编译路径）───
    ro = AccelReadout(w0.shape[1], w0.shape[0], w0=w0.copy(),
                      compile=True, nll_sync_every=4)
    g = np.random.default_rng(2)
    ok = True
    try:
        for i in range(10):
            hh = np.ascontiguousarray(
                g.normal(0.0, 1.0, w0.shape[1]).astype(np.float32))
            nll = ro.learn_softmax(hh, None, 0.05, target_idx=int(g.integers(0, w0.shape[0])))
            if not np.isfinite(nll):
                ok = False
                break
    except Exception as e:                                # noqa: BLE001
        ok = False
        print(f"    ! {type(e).__name__}: {e}")
    check("compiled 连续 10 步多 target 烟测（nll 有限）", ok)

    n_ok = sum(1 for _, ok2 in CASES if ok2)
    print(f"\n通过 {n_ok}/{len(CASES)}")
    return 0 if n_ok == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
