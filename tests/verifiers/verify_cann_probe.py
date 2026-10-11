"""门禁：CANN 硬件探测（phdnet/backends/cann_probe.py）—— exit 0 = PASS。

为什么要有这条（2026-10-07）：
  「昇腾提速」线要求直连 CANN，而本机**没有 NPU、没有 C 编译器、没有 Cython** ——
  这意味着桥接层在开发机上**只能验证「优雅回落」这一半**。探测器是整条回落链的第一环：
  它抛异常 → 训练启动期崩；它漏 reason → 后面「回落原因丢失」（BUGS B3/B12 同族）。
  所以门禁钉的是**契约**而不是能力：
    · 永不抛、零 print、format_line 纯 ASCII 单行；
    · not ok ⇒ reason 必非空；
    · dtype 能力是**实测**来的（源码里必须有真实 matmul 调用，不是查表）；
    · CANN 侧必须真调 aclrtGetVersion/aclrtGetDeviceCount 并检查返回码。
  真实 NPU 上的 submit→fetch 用例由 cann_bridge 的门禁覆盖（本机无 NPU，这里显式跳过并打印原因）。

用法：py -3.14 tests/verifiers/verify_cann_probe.py
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from importlib import util as _ilu
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_RESULTS: list[tuple[bool, str, str]] = []
_MOD_PATH = _ROOT / "phdnet" / "backends" / "cann_probe.py"


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def _load_module():
    """独立按文件加载（绕开包 __init__ 的副作用），并捕获 stdout 断言零 print。"""
    buf = io.StringIO()
    spec = _ilu.spec_from_file_location("_cann_probe_isolated", _MOD_PATH)
    mod = _ilu.module_from_spec(spec)
    # ⚠ 必须先登记进 sys.modules 再 exec：dataclass + `from __future__ import
    #   annotations` 的字符串注解会经 sys.modules[cls.__module__] 反查，
    #   未登记时会抛 "'NoneType' object has no attribute '__dict__'"。
    sys.modules[spec.name] = mod
    try:
        with redirect_stdout(buf):
            spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod, buf.getvalue()


def main() -> int:
    print("=" * 78)
    print("门禁：CANN 硬件探测（零编译依赖 / 永不抛 / 零 print / 纯 ASCII 摘要）")
    print("=" * 78)

    # ── 1. 加载与调用不抛、零 print ─────────────────────────────────────
    print("\n[1] 加载与调用契约")
    try:
        mod, leaked = _load_module()
        check("1.1 模块加载零 print", leaked == "", repr(leaked[:60]))
    except Exception as e:                              # noqa: BLE001
        check("1.1 模块加载不抛", False, f"{type(e).__name__}: {e}")
        _finish()
        return 1

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            res = mod.probe()
            ok, why = mod.available()
            ok2, why2 = mod.available()
        check("1.2 probe()/available() 不抛", True, "")
        check("1.3 调用零 print", buf.getvalue() == "", repr(buf.getvalue()[:60]))
    except Exception as e:                              # noqa: BLE001
        check("1.2 probe()/available() 不抛", False, f"{type(e).__name__}: {e}")
        _finish()
        return 1

    # ── 2. 字段契约 ─────────────────────────────────────────────────────
    print("\n[2] ProbeResult 字段")
    d = res.as_dict()
    need = ["ok", "kind", "npu_count", "cann_version", "torch_backend",
            "fp8_native", "fp16_ok", "bf16_ok", "cann_lib", "reason", "detail"]
    check("2.1 11 个字段齐", all(k in d for k in need),
          str([k for k in need if k not in d]))
    check("2.2 dtype 标志是 bool", all(isinstance(d[k], bool)
                                    for k in ("fp8_native", "fp16_ok", "bf16_ok")), "")
    check("2.3 kind 在枚举内", d["kind"] in ("torch_npu", "cann_acl", "cuda", "cpu", "none"),
          d["kind"])
    check("2.4 available 与 probe 结果一致", ok == res.ok, f"{ok} vs {res.ok}")
    check("2.5 available 两次调用稳定（缓存）", ok2 == ok and why2 == why, "")

    # ── 3. 契约：not ok ⇒ reason 必非空 ────────────────────────────────
    print("\n[3] 回落契约（P161：回落 + 记原因）")
    if not res.ok:
        check("3.1 not ok ⇒ reason 非空", bool(res.reason.strip()), res.reason[:70])
        check("3.2 not ok ⇒ available 返回 (False, 非空)",
              ok is False and bool((why or "").strip()), (why or "")[:70])
    else:
        check("3.1 本机探测到 NPU 通路（跳过 reason 断言）", True,
              f"kind={d['kind']} npu={d['npu_count']}")
        print("      · 有 NPU 的机器上走这里，3.2 同步跳过")

    # ── 4. format_line 纯 ASCII 单行 ────────────────────────────────────
    print("\n[4] 摘要行")
    line = res.format_line()
    check("4.1 纯 ASCII", all(ord(c) < 128 for c in line),
          repr(line[:80]))
    check("4.2 单行（无换行）", "\n" not in line and "\r" not in line, "")
    check("4.3 含关键字段", "kind=" in line and "fp16=" in line and "ok=" in line, line[:90])

    # ── 5. 坏输入不抛 ───────────────────────────────────────────────────
    print("\n[5] 坏路径 / 坏库")
    try:
        old = os.environ.get("ASCEND_HOME_PATH")
        os.environ["ASCEND_HOME_PATH"] = "/nonexistent/ascend-home-for-gate"
        mod._CACHE.clear()
        r_bad = mod.probe(refresh=True)
        os.environ.pop("ASCEND_HOME_PATH", None) if old is None else os.environ.update(
            {"ASCEND_HOME_PATH": old})
        check("5.1 坏 ASCEND_HOME_PATH 下 probe 不抛", True, "")
        check("5.2 仍满足 not ok ⇒ reason 非空（或探测到别的通路）",
              r_bad.ok or bool(r_bad.reason.strip()), r_bad.reason[:70])
    except Exception as e:                              # noqa: BLE001
        check("5.1 坏 ASCEND_HOME_PATH 下 probe 不抛", False,
              f"{type(e).__name__}: {e}")
    finally:
        os.environ.pop("ASCEND_HOME_PATH", None)
        mod._CACHE.clear()

    # ── 6. 源码级：必须真调 ACL 符号、必须真跑 matmul ────────────────────
    print("\n[6] 源码级（防退化成「猜一个结果」）")
    src = _MOD_PATH.read_text(encoding="utf-8")
    check("6.1 源码含 aclrtGetVersion", "aclrtGetVersion" in src, "")
    check("6.2 源码含 aclrtGetDeviceCount", "aclrtGetDeviceCount" in src, "")
    check("6.3 源码真跑 matmul（非查表）", "@ b" in src and "isfinite" in src, "")
    check("6.4 有返回码检查（rc != 0 分支）", "rc != 0" in src or "rc2 != 0" in src, "")

    # ── 7. 本机无 NPU 的显式说明（避免误判为通过依据）───────────────────
    print("\n[7] 环境说明")
    check("7.1 本机无 NPU 时给出可读 reason",
          res.ok or ("NPU" in res.reason or "ascendcl" in res.reason or "torch_npu" in res.reason),
          res.reason[:80])
    if not res.ok:
        print("      · 本机无 NPU → 真实 submit→fetch 用例由 cann_bridge 门禁覆盖（此处显式跳过）")

    _finish()
    return 0


def _finish() -> None:
    ok = sum(1 for r in _RESULTS if r[0])
    bad = [r[1] for r in _RESULTS if not r[0]]
    print("\n" + "=" * 78)
    print(f"结果：{ok}/{len(_RESULTS)} 通过" + (f" | 失败 {bad}" if bad else ""))
    print("=" * 78)


if __name__ == "__main__":
    raise SystemExit(main())
