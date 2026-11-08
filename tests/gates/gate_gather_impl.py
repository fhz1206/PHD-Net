#!/usr/bin/env python3
"""门禁：稀疏读出 gather 三种实现的**逐位等价**契约（P217）。

背景：P217 在 AtomGit NPU Notebook（Ascend910B4）实测
`torch.index_select(ht, 0, Wi.ravel()).view(Wi.shape)` 比 `ht[Wi]`
快 52.80×（fp16）/ 29.63×（fp32），且 max|Δ| = 0。
已把 `index_select` 接成 `--readout-gather-impl` 的第三个选项
（**默认仍为 index**，需服务器 A/B + probe_PPL 后才可改默认）。

本门禁验证：
  1. 三种实现在 fp32 / fp16 / bf16 下**逐位相同**（torch.equal，不是容差）；
  2. 含重复列索引（Wi 每行可有重复）—— 这是 scatter_ 类实现会出错的地方；
  3. 含 Wi 取到 H-1 / 0 的边界；
  4. shape / dtype 一致；
  5. CLI choices 确实包含三个选项，且默认仍是 index（防止误改默认）。

⚠ 只测算子等价（不构造 AccelReadout 实例，避免 NPU/CANN 依赖）。
   生产分支接线由 `gate_readout_sparse.py` 等覆盖。

运行：python tests/gates/gate_gather_impl.py（退出码 0 = PASS）
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

_RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def variants(ht, Wi):
    return {
        "index": ht[Wi],
        "take": torch.take(ht, Wi),
        "index_select": torch.index_select(ht, 0, Wi.reshape(-1)).view(Wi.shape),
    }


def main() -> int:
    print("=" * 74)
    print("门禁：gather 三实现逐位等价（P217 index_select）")
    print("=" * 74)

    torch.manual_seed(7)
    H, V, K = 3072, 1024, 128

    for dname, dt in (("fp32", torch.float32),
                      ("fp16", torch.float16),
                      ("bf16", torch.bfloat16)):
        ht = torch.randn(H, dtype=dt) * 0.5
        Wi = torch.randint(0, H, (V, K), dtype=torch.long)
        # 强制边界 + 重复列索引
        Wi[:, 0] = 0
        Wi[:, 1] = H - 1
        Wi[:, 2] = Wi[:, 0]                 # 重复
        Wi[::4, 3] = Wi[::4, 4]             # 行内重复

        vs = variants(ht, Wi)
        a = vs["index"]
        for name in ("take", "index_select"):
            b = vs[name]
            check(torch.equal(a, b), f"{dname}: index == {name}（逐位）",
                  f"max|Δ|={float((a.float()-b.float()).abs().max()):.3e}")
            check(b.shape == a.shape and b.dtype == a.dtype,
                  f"{dname}: {name} shape/dtype 一致",
                  f"{tuple(b.shape)} {b.dtype}")

    # 负向对照：确认这个门禁真的能检出不等价的实现
    # （scatter_ 会把重复列索引覆盖而非累加 → 与 index 分歧，
    #   正是 P217 benchmark 里 F 等价稠密 SpMM max|Δ|≈1.05 的原因）
    ht = torch.randn(H, dtype=torch.float32)
    Wi = torch.randint(0, H, (64, K), dtype=torch.long)
    Wi[:, 3] = Wi[:, 4]                     # 制造重复列
    W = torch.randn(64, K)
    S = torch.zeros(64, H)
    S.scatter_(1, Wi, W)
    y_scatter = S @ ht
    y_index = (ht[Wi] * W).sum(dim=1)
    check(not torch.equal(y_scatter, y_index),
          "负向对照：scatter_ 构造的稠密 SpMM 与 index 不等价（应被检出）",
          f"max|Δ|={float((y_scatter-y_index).abs().max()):.3e}")

    # CLI / config 契约
    src = io.open(_ROOT / "train" / "train.py", encoding="utf-8").read()
    seg = src[src.find('"--readout-gather-impl"'):]
    seg = seg[:400]
    check('"index_select"' in seg and '"index"' in seg and '"take"' in seg,
          "CLI choices 含三个实现")
    check('default="index"' in seg,
          "CLI 默认仍是 index（未擅自改默认）")

    cfg = io.open(_ROOT / "phdnet" / "config.py", encoding="utf-8").read()
    check('readout_gather_impl: str = "index"' in cfg,
          "config 默认仍是 index")

    body = io.open(_ROOT / "phdnet" / "backends" / "accel_readout.py",
                   encoding="utf-8").read()
    check(body.count('_gather_impl == "index_select"') >= 2,
          "accel_readout 两条精度臂都接了 index_select 分支")

    ok = sum(1 for r in _RESULTS if r[0])
    print("\n" + "=" * 74)
    print(f"结果：{ok}/{len(_RESULTS)} 通过 | 失败 {len(_RESULTS)-ok}")
    for r in _RESULTS:
        if not r[0]:
            print(f"  FAIL {r[1]}  [{r[2]}]")
    print("=" * 74)
    return 0 if ok == len(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
