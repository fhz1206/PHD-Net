"""P3 验收：自动发育调度（表征稳定度门控）。

场景 A：冻结表征 + 自动门控 —— 应回归手工模式结果（支撑稳定 → STDP 全开）。
场景 B：表征在线发育 + 自动门控 —— 发育期 STDP 自动受抑，稳定后放开；
        对照手工调度不可行（发育时长未知），这正是自动化的价值。
"""

# --- 目录结构调整（2026-09-18）：脚本位于 tests/ 或 tools/ 子目录 ---
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]     # 项目根目录
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))              # 保证 `import phdnet` 可用
_DOC = _ROOT / "datasets" / "eval" / "internal_corpus.txt"    # 默认语料
# --- 引导结束 ---

import numpy as np

from phdnet.config import PHDNetConfig
from phdnet.model import PHDNet


def sparse_symbol(n, k, rng):
    s = np.zeros(n)
    idx = rng.choice(n, k, replace=False)
    s[idx] = rng.uniform(0.3, 1.0, k)
    return s


def run(tag, cfg_overrides, n_steps=600):
    rng = np.random.default_rng(0)
    sym = [sparse_symbol(64, 16, rng) for _ in range(3)]
    cfg = PHDNetConfig(n_input=64, n_sdr=256, k_sparse=32,
                       n_mid=256, n_top=256, eta_stdp=0.12, seed=11,
                       **cfg_overrides)
    net = PHDNet(cfg)
    errs, stabs = [], []
    for t in range(n_steps):
        d = net.step(sym[t % 3], learn=True)
        errs.append(d["seq_err"])
        stabs.append(1.0 - net._instab_ema)
    early, late = sum(errs[:30]) / 30, sum(errs[-100:]) / 100

    def top_rate(x):
        s0, _ = net.encoder.encode(x)
        return net._rate(net.pc.infer(s0, net.cfg.n_infer_steps)["r2"])

    rates = [top_rate(s) for s in sym]
    pred = net.stdp.predict(rates[0])
    cos = [float(pred @ rt / (np.linalg.norm(pred) * np.linalg.norm(rt) + 1e-9))
           for rt in rates]
    best = int(np.argmax(cos))
    ok = best == 1
    print(f"[{tag}] 误差 {early:.3f}→{late:.3f}  稳定度(末值) {stabs[-1]:.2f}  "
          f"结构检验 {'✓ A→B' if ok else '✗ best=' + 'ABC'[best]} "
          f"cos=[{cos[0]:.2f},{cos[1]:.2f},{cos[2]:.2f}]")
    return ok


if __name__ == "__main__":
    print("=" * 62)
    print("P3 自动发育调度验收")
    print("=" * 62)
    # 场景 A：冻结表征 + 自动门控（对照手工模式的 0.28 / cos 0.81）
    okA = run("A 冻结表征+自动门控", dict(eta_pc=0.0, eta_oja=0.0,
                                          auto_development=True))
    # 场景 B：表征在线发育 + 自动门控（PC 学习率取 eta_dev_pc）
    okB = run("B 表征发育+自动门控", dict(auto_development=True,
                                          eta_pc=0.02, eta_oja=0.02))
    print("-" * 62)
    print(f"P3 结论: {'✓ 通过' if (okA and okB) else '△ 部分通过' if (okA or okB) else '✗ 未通过'}"
          f"（A={'✓' if okA else '✗'} B={'✓' if okB else '✗'}）")
