"""门禁：phdnet/device.py::probe() 的后端择优链 —— exit 0 = PASS。

为什么要有这条（2026-10-07 真实事故）：
  审计 P1-7 指出「装了 torch_npu 但没有 NPU 硬件时，elif hip 把同机 CUDA 跳过」，
  修的时候把 `elif hip` 改成 `if hip` —— **但 new_string 漏掉了整块 npu 检测**，
  结果昇腾机器上 probe() 再也不会返回 torch-npu（NPU 最高优先级分支凭空消失），
  use_torch_backend("auto") / select_backend("npu") / backend_probe 全部跟着错。
  这类「改 if 链时把整块删掉」的回归，当时**所有门禁都是绿的** —— 因为没有任何用例
  构造过「NPU 可用」「NPU 不可用但同机 CUDA 可用」这两种机器形态。本文件补齐它们。

覆盖四种机器形态（注入假 torch_npu，不依赖真硬件）：
  ① 装了 torch_npu 且 NPU 可用        → 必须返回 torch-npu（最高优先级）
  ② 装了 torch_npu 但 NPU 不可用 + 同机 CUDA 可用 → 必须返回 torch-cuda（原 P1-7）
  ③ 没装 torch_npu + CUDA 可用         → torch-cuda
  ④ 没装 torch_npu + 无加速器          → cpu 路径
另加源码级结构断言：torch-npu 返回分支必须真实存在、npu_plugin 变量不得是死变量。

用法：py -3.14 tests/verifiers/verify_device_npu_probe.py
"""
from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch                                             # noqa: E402
import phdnet.device as D                                # noqa: E402

_RESULTS: list[tuple[bool, str, str]] = []
_SAVED_NPU = getattr(torch, "npu", None)
_SAVED_CUDA_AVAIL = torch.cuda.is_available


def _probe():
    """每种机器形态都要**先清 lru_cache**再 probe —— probe() 带 @lru_cache，
    不清缓存的话四种形态拿到的都是第一种的结果（本门禁第一版就踩了这个坑）。"""
    D.probe.cache_clear()
    return D.probe(verify=False)


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def _inject_fake_npu() -> None:
    """把假的 torch_npu 放进 sys.modules，并给 torch 装上 npu 命名空间。"""
    mod = types.ModuleType("torch_npu")
    mod.__spec__ = importlib.machinery.ModuleSpec("torch_npu", None)
    mod.__version__ = "0.0-test"
    sys.modules["torch_npu"] = mod
    torch.npu = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda i=0: "Ascend910B4-test",
        synchronize=lambda: None,
        empty_cache=lambda: None,
    )


def _remove_fake_npu() -> None:
    sys.modules.pop("torch_npu", None)
    if _SAVED_NPU is None:
        if hasattr(torch, "npu"):
            delattr(torch, "npu")
    else:
        torch.npu = _SAVED_NPU


def _set_npu_available(flag: bool) -> None:
    torch.npu.is_available = lambda: flag


def _set_cuda(flag: bool) -> None:
    torch.cuda.is_available = lambda: flag


def main() -> int:
    print("=" * 78)
    print("门禁：device.probe 后端择优链（NPU 最高优先级 + 不可用时不得短路）")
    print("=" * 78)

    try:
        print("\n[形态①] 装了 torch_npu 且 NPU 可用")
        _inject_fake_npu()
        _set_npu_available(True)
        _set_cuda(False)
        info = _probe()
        check("①  NPU 可用时返回 torch-npu（npu 分支不得被删）",
              info.name == "torch-npu" and info.kind == "npu",
              f"name={info.name} kind={info.kind}")
        check("①  BackendInfo.npu_plugin 标记为已装",
              bool(info.npu_plugin), str(info.npu_plugin))
        check("①  use_torch_backend('npu') 为 True",
              D.use_torch_backend("npu") is True,
              str(D.use_torch_backend("npu")))
        sel = D.select_backend("npu")
        check("①  select_backend('npu') 选中 npu",
              sel.kind == "npu", f"{sel.name}/{sel.kind}")

        print("\n[形态②] 装了 torch_npu 但 NPU 不可用 + CUDA 可用")
        _set_npu_available(False)
        _set_cuda(True)
        info = _probe()
        check("②  不再被 npu 短路，落到 torch-cuda（P1-7 原始命题）",
              info.kind == "cuda", f"name={info.name} kind={info.kind}")
        check("②  notes 保留「已装 torch_npu 但不可用」的附注",
              "torch_npu" in (info.notes or ""), (info.notes or "")[:70])

        print("\n[形态③] 未装 torch_npu + CUDA 可用")
        _remove_fake_npu()
        _set_cuda(True)
        info = _probe()
        check("③  无插件 + CUDA → torch-cuda", info.kind == "cuda",
              f"name={info.name} kind={info.kind}")

        print("\n[形态④] 未装 torch_npu + 无加速器")
        _set_cuda(False)
        info = _probe()
        check("④  无插件 + 无加速器 → cpu 路径", info.kind == "cpu",
              f"name={info.name} kind={info.kind}")

        print("\n[结构] 源码级防回归（防止再次「改 if 链删掉整块」）")
        src = (_ROOT / "phdnet" / "device.py").read_text(encoding="utf-8")
        check("S1 源码含 torch-npu 返回分支", 'BackendInfo("torch-npu"' in src, "")
        check("S2 源码含 torch.npu.is_available 判据", "torch.npu.is_available()" in src, "")
        check("S3 npu_plugin 变量被真正使用（非死变量）",
              src.count("npu_plugin") >= 3, f"出现 {src.count('npu_plugin')} 次")
        check("S4 非 npu 分支不是 elif（不可用时不得短路 hip/cuda）",
              "elif torch_npu" not in src and "elif hip" in src, "")
    finally:
        _remove_fake_npu()
        torch.cuda.is_available = _SAVED_CUDA_AVAIL

    ok = sum(1 for r in _RESULTS if r[0])
    bad = [r[1] for r in _RESULTS if not r[0]]
    print("\n" + "=" * 78)
    print(f"结果：{ok}/{len(_RESULTS)} 通过" + (f" | 失败 {bad}" if bad else ""))
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
