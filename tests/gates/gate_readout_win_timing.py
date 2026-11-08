#!/usr/bin/env python3
"""门禁：readout 滑窗计时 + 开关生效值自证的契约（P218）。

背景：2026-11-08 生产 A/B 时，服务器代码是旧版（choices 里没有
`index_select`），日志又**只打印 cum 口径的 readout ms/tok**，且启动日志
不打印生效的 `gather_impl` → 无法区分「改了没效果」和「根本没改」。

本门禁验证：
  1. `PHDNetNet.step` 同时维护 cum 与 win 两份 readout 计时；
  2. `reset_prof_window()` 会**同步清零 win**（否则 win 退化成 cum）；
  3. `train.py` 的打印行确实输出 `(win)`，且启动日志打印生效的
     `gather_impl` / `gather_dtype` / `conn_k`；
  4. CLI choices 含 `index_select`（防止旧版代码被误当成新版）。

运行：python tests/gates/gate_readout_win_timing.py（退出码 0 = PASS）
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def main() -> int:
    print("=" * 74)
    print("门禁：readout 滑窗计时 + 开关自证（P218）")
    print("=" * 74)

    model = io.open(_ROOT / "phdnet" / "model.py", encoding="utf-8").read()
    train = io.open(_ROOT / "train" / "train.py", encoding="utf-8").read()

    check("_ro_ms_win" in model and "_ro_calls_win" in model,
          "model.step 维护 win 计时字段")
    check("_ro_ms_accum" in model and "_ro_calls" in model,
          "cum 计时字段仍在（未破坏旧口径）")

    # reset_prof_window 里必须清零 win
    seg = model[model.find("def reset_prof_window"):]
    seg = seg[:seg.find("\n    @")] if "\n    @" in seg else seg[:1200]
    check("self._ro_ms_win = 0.0" in seg and
          "self._ro_calls_win = 0" in seg,
          "reset_prof_window 同步清零 win")

    # 打印行 + 启动自证
    check("_ro_wms / _ro_wn" in train and "(win)" in train,
          "训练日志打印 readout (win)")
    check("[readout] gather_impl=" in train and
          "cfg.readout_gather_impl" in train,
          "启动日志打印生效的 gather_impl（requested vs actual）")
    check("gather_dtype=" in train and "conn_k=" in train,
          "启动日志同时打印 gather_dtype / conn_k")

    # CLI choices
    cli = train[train.find('"--readout-gather-impl"'):]
    cli = cli[:400]
    check('"index_select"' in cli, "CLI choices 含 index_select（新版标志）")

    # 负向对照：确认「只打印 cum」会被检出
    check("(cum)" in train and "(win)" in train,
          "负向对照：cum 与 win 两个口径都标注，不会混淆")

    ok = sum(1 for r in _RESULTS if r[0])
    print("\n" + "=" * 74)
    print(f"结果：{ok}/{len(_RESULTS)} 通过 | 失败 {len(_RESULTS) - ok}")
    for r in _RESULTS:
        if not r[0]:
            print(f"  FAIL {r[1]}  [{r[2]}]")
    print("=" * 74)
    return 0 if ok == len(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
