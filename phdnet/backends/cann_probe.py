"""CANN / 昇腾 NPU 硬件探测 —— 零编译依赖、**永不抛异常、零 print**。

设计约束（2026-10-07，"昇腾提速"线）：
  * 主实现走 **ctypes + libascendcl.so**（本机无 C 编译器、未装 Cython，
    任何要编译的方案在这台机器上都无法构建/验证）；
  * 探测是**回落链的第一环**：拿不到就结构化返回 reason（P161「回落 + 记原因」纪律），
    绝不让异常穿到调用方 —— 训练启动期崩在探测上是最糟的失败模式；
  * dtype 能力**真跑 8×8 matmul**，不是查表（P86/P148 的教训：能力表会过期）；
  * 模块零 print：诊断只经 format_line() 交给调用方决定打不打。

用法：
    from phdnet.backends.cann_probe import probe, available
    res = probe()            # 带进程级缓存
    print(res.format_line()) # 纯 ASCII 单行
    ok, why = available()    # 永不抛
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import asdict, dataclass

# 常见 AscendCL 安装位置（服务器上未必在 ld 缓存里，逐个试）
_CANN_CANDIDATES = (
    "/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so",
    "/usr/local/Ascend/driver/lib64/libascendcl.so",
)


@dataclass
class ProbeResult:
    ok: bool = False
    kind: str = "none"              # torch_npu | cann_acl | cuda | cpu | none
    npu_count: int = 0
    cann_version: str = ""
    torch_backend: str = ""
    fp8_native: bool = False
    fp16_ok: bool = False
    bf16_ok: bool = False
    cann_lib: str = ""
    reason: str = ""
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def format_line(self) -> str:
        """单行纯 ASCII 摘要（诊断用；控制字符一律替换，保证不破日志版式）。"""
        s = (f"ok={self.ok} kind={self.kind} npu={self.npu_count} "
             f"cann={self.cann_version or '-'} torch={self.torch_backend or '-'} "
             f"fp8={self.fp8_native} fp16={self.fp16_ok} bf16={self.bf16_ok} "
             f"lib={self.cann_lib or '-'} reason={self.reason or '-'}")
        s = s.replace("\r", " ").replace("\n", " ")
        s = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in s)
        return s.encode("ascii", "replace").decode("ascii")


_CACHE: dict = {}


def _matmul_ok(torch, a_dt, b_dt, require_finite: bool = True) -> bool:
    """真跑一次 8×8 matmul（能力必须实测，不查表）。"""
    try:
        a = torch.ones(8, 8, dtype=a_dt)
        b = torch.ones(8, 8, dtype=b_dt)
        out = (a @ b).float()
        if require_finite and not bool(torch.isfinite(out).all()):
            return False
        return True
    except Exception:                                   # noqa: BLE001
        return False


def _probe_cann() -> tuple[str, str, str]:
    """返回 (lib_path, version_str, reason)。找不到/调不通都只回 reason，不抛。"""
    names = []
    found = ctypes.util.find_library("ascendcl")
    if found:
        names.append(found)
    env = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if env:
        names.append(os.path.join(env, "lib64", "libascendcl.so"))
    names.extend(_CANN_CANDIDATES)

    tried = []
    for cand in names:
        if not cand:
            continue
        if not (os.path.isabs(cand) or os.path.sep in cand or cand.endswith(".so")):
            # find_library 可能返回裸名字，交给 CDLL 自己解析
            pass
        elif not os.path.exists(cand):
            tried.append(f"{cand}:不存在")
            continue
        try:
            lib = ctypes.CDLL(cand)
        except Exception as e:                          # noqa: BLE001
            tried.append(f"{cand}:{type(e).__name__}")
            continue
        ver = ctypes.c_int32(0)
        try:
            lib.aclrtGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int32)]
            lib.aclrtGetVersion.restype = ctypes.c_int
            rc = lib.aclrtGetVersion(ctypes.byref(ver))
        except AttributeError:
            tried.append(f"{cand}:缺 aclrtGetVersion 符号")
            continue
        except Exception as e:                          # noqa: BLE001
            tried.append(f"{cand}:{type(e).__name__}")
            continue
        if rc != 0:
            tried.append(f"{cand}:aclrtGetVersion rc={rc}")
            continue
        cnt = ctypes.c_uint32(0)
        try:
            lib.aclrtGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
            lib.aclrtGetDeviceCount.restype = ctypes.c_int
            rc2 = lib.aclrtGetDeviceCount(ctypes.byref(cnt))
            if rc2 != 0:
                tried.append(f"{cand}:aclrtGetDeviceCount rc={rc2}")
                continue
        except Exception as e:                          # noqa: BLE001
            tried.append(f"{cand}:{type(e).__name__}")
            continue
        # 主版本 = ver >> 16（AscendCL 版本号编码：major<<16 | minor<<8 | patch）
        major = (int(ver.value) >> 16) & 0xFFFF
        minor = (int(ver.value) >> 8) & 0xFF
        patch = int(ver.value) & 0xFF
        return cand, f"{major}.{minor}.{patch}", ""
    return "", "", ("ascendcl 未找到（" + "; ".join(tried[:4]) + "）"
                    if tried else "ascendcl 未找到（无候选路径）")


def _probe_uncached() -> ProbeResult:
    res = Detail = None  # noqa: F841  （占位，保证下面异常路径也有对象）
    res = ProbeResult()
    notes = []

    # ── torch 侧 ────────────────────────────────────────────────────────
    torch = None
    try:
        import torch as _t
        torch = _t
        res.torch_backend = getattr(getattr(_t, "version", None), "hip", None) or "cpu"
    except Exception as e:                              # noqa: BLE001
        notes.append(f"torch 导入失败 {type(e).__name__}")

    if torch is not None:
        # 昇腾：先注册插件再看设备（torch_npu 的 import 副作用就是注册 npu）
        try:
            import torch_npu  # noqa: F401
            if hasattr(torch, "npu") and torch.npu.is_available():
                res.kind = "torch_npu"
                res.npu_count = int(torch.npu.device_count())
                res.torch_backend = "npu"
            else:
                notes.append("已装 torch_npu 但 NPU 不可用")
        except ImportError:
            notes.append("torch_npu 未安装")
        except Exception as e:                          # noqa: BLE001
            notes.append(f"torch_npu 探测失败 {type(e).__name__}: {str(e)[:60]}")

        if res.kind != "torch_npu":
            try:
                if torch.cuda.is_available():
                    res.kind = "cuda"
                    res.npu_count = int(torch.cuda.device_count())
                    res.torch_backend = "cuda"
            except Exception as e:                      # noqa: BLE001
                notes.append(f"cuda 探测失败 {type(e).__name__}")

        # dtype 能力：真跑 matmul
        res.fp16_ok = _matmul_ok(torch, torch.float16, torch.float16)
        res.bf16_ok = _matmul_ok(torch, torch.bfloat16, torch.bfloat16)
        if hasattr(torch, "float8_e4m3fn"):
            for other in ("float8_e4m3fn", "float16", "float32"):
                try:
                    od = getattr(torch, other)
                except AttributeError:
                    continue
                if _matmul_ok(torch, torch.float8_e4m3fn, od, require_finite=False):
                    res.fp8_native = True
                    break

    # ── CANN 侧（ctypes，零编译依赖）────────────────────────────────────
    lib, ver, why = _probe_cann()
    if lib:
        res.cann_lib = lib
        res.cann_version = ver
        if res.kind in ("none", "cpu"):
            res.kind = "cann_acl"
            res.npu_count = max(res.npu_count, 1)
    else:
        notes.append(why)

    if res.kind == "none":
        res.kind = "cpu"

    # ── 归一化：ok 只表示「NPU 通路可用」；not ok ⇒ reason 必非空 ────────
    res.ok = res.kind in ("torch_npu", "cann_acl")
    if not res.ok:
        res.reason = "; ".join(notes) or "本机无可用 NPU 通路（无 torch_npu / libascendcl）"
    else:
        res.reason = "; ".join(notes)
    res.detail = "|".join(notes)
    return res


def probe(refresh: bool = False) -> ProbeResult:
    """带进程级缓存的探测；**任何异常都被兜底成 ok=False + reason**，永不抛。"""
    key = "1"
    if not refresh and key in _CACHE:
        return _CACHE[key]
    try:
        res = _probe_uncached()
    except Exception as e:                              # noqa: BLE001
        res = ProbeResult(ok=False, kind="none",
                          reason=f"probe 兜底：{type(e).__name__}: {str(e)[:120]}")
    if not res.ok and not res.reason:
        res.reason = "探测未通过且未给出原因（内部异常）"
    _CACHE[key] = res
    return res


def available() -> tuple[bool, str]:
    """(是否可用, 原因) —— 永不抛；供 cann_bridge 等调用方做回落判断。"""
    try:
        r = probe()
        return bool(r.ok), ("" if r.ok else r.reason)
    except Exception as e:                              # noqa: BLE001
        return False, f"available 兜底：{type(e).__name__}: {str(e)[:120]}"
