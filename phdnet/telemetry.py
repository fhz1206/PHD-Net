# -*- coding: utf-8 -*-
"""训练日志的系统级遥测（P41，fhz「日志加入 CPU/NPU 占用、RAM、HBM、IPC」）。

全部采样**非阻塞**（psutil 的 interval=None 模式 + torch 内存查询），可安全
放进每 log_every 一次的训练日志。

| 指标 | 来源 | 说明 |
|---|---|---|
| CPU 系统占用 | `psutil.cpu_percent(None)` | 上次采样以来的系统平均（多核合计，可 >100%？否——percent 归一到 0-100/总核数口径见 psutil 文档） |
| CPU 进程占用 | `time.process_time` 差 / 墙钟差 | 本进程（含 numba nogil 线程）吃掉的 CPU 核数（>1 = 多线程并行中） |
| RAM | `psutil.virtual_memory` + 进程 RSS | 系统占用百分比 + 进程常驻内存 |
| NPU/GPU 利用率 | `torch.npu.utilization()` / `torch.cuda.utilization()` | 设备忙碌百分比；不可用时 None |
| HBM/显存 | `torch.<dev>.memory_allocated/reserved` + `get_device_properties().total_memory` | 设备内存占用；**HBM 带宽利用率无法从 torch 读**——需 `npu-smi info`（昇腾）或 `nvidia-smi dmon`（NVIDIA）外部工具 |
| IPC | 硬件计数器 | Python 层**拿不到**（需 `perf stat -p <pid>`，Linux + perf 权限）。诚实返回 None；替代指标 = 进程 CPU 占用（>1 即多线程吃满）与上下文切换率 `psutil.cpu_stats().ctx_switches` 差值 |

采样成本：psutil 两次调用 + 若干属性读取 ≈ 50-200 μs，每 log_once 一次可忽略。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time

import numpy as np

try:
    import psutil
except Exception:                                        # pragma: no cover
    psutil = None


class Telemetry:
    """训练遥测采样器（非阻塞；所有字段容错，不可用即 None）。"""

    def __init__(self, device: str | None = None, npu_smi: bool = True):
        self._proc = psutil.Process() if psutil else None
        self._last_wall = time.perf_counter()
        self._last_ptime = time.process_time()
        self._last_ctx = None
        if psutil:
            try:
                psutil.cpu_percent(interval=None)        # 首次调用建立基线
                self._last_ctx = psutil.cpu_stats().ctx_switches
            except Exception:                            # noqa: BLE001
                pass
        # P57（fhz 2026-09-29「NPU 和 HBM 的数据读不出来」）：生产训练一直
        # 用 Telemetry() 无参构造 → device="" → sample() 的加速器分支永远
        # 不触发，日志 NPU/HBM 恒 `--`（P41 遗留：P19 读出上 NPU 后没人回补
        # 遥测）。现在**自动探测**：显式传入的 device 优先，否则 torch.npu /
        # torch.cuda 谁可用用谁。
        self.device = device or self._auto_probe_device()
        self._acc_err: str | None = None        # 首次加速器采样失败原因
        self._acc_err_reported = False
        self._smi_path = shutil.which("npu-smi") if npu_smi else None

    def _auto_probe_device(self) -> str:
        """无显式 device 时自动探测（torch.npu → torch.cuda → 无）。"""
        if not torch_available():
            return ""
        import torch
        try:
            if hasattr(torch, "npu") and torch.npu.is_available():
                return "npu"
        except Exception:                                # noqa: BLE001
            pass
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                return "cuda"
        except Exception:                                # noqa: BLE001
            pass
        return ""

    def sample(self) -> dict:
        out: dict = {"cpu_sys": None, "cpu_proc_cores": None,
                     "ram_pct": None, "ram_proc_gb": None,
                     "acc_util": None, "hbm_alloc_gb": None,
                     "hbm_total_gb": None, "ipc": None,
                     "ctx_switches": None}
        # ── CPU / RAM ──
        if psutil:
            try:
                out["cpu_sys"] = psutil.cpu_percent(interval=None)
            except Exception:                            # noqa: BLE001
                pass
            try:
                vm = psutil.virtual_memory()
                out["ram_pct"] = vm.percent
                if self._proc:
                    out["ram_proc_gb"] = self._proc.memory_info().rss / 2 ** 30
            except Exception:                            # noqa: BLE001
                pass
            try:
                cs = psutil.cpu_stats().ctx_switches
                if self._last_ctx is not None:
                    out["ctx_switches"] = int(cs - self._last_ctx)
                self._last_ctx = cs
            except Exception:                            # noqa: BLE001
                pass
        # ── 进程 CPU 核数（process_time / 墙钟；>1 = nogil 线程并行中）──
        wall = time.perf_counter()
        ptime = time.process_time()
        dw = wall - self._last_wall
        if dw > 1e-6:
            out["cpu_proc_cores"] = (ptime - self._last_ptime) / dw
        self._last_wall = wall
        self._last_ptime = ptime
        # ── 加速器 ──
        if torch_available() and self.device:
            dev = self.device.lower()
            try:
                if dev.startswith("npu") and hasattr(torch, "npu"):
                    if hasattr(torch.npu, "utilization"):
                        out["acc_util"] = float(torch.npu.utilization())
                    out["hbm_alloc_gb"] = torch.npu.memory_allocated() / 2 ** 30
                    out["hbm_total_gb"] = (torch.npu.get_device_properties()
                                           .total_memory) / 2 ** 30
                elif dev.startswith("cuda") and hasattr(torch, "cuda"):
                    out["acc_util"] = float(torch.cuda.utilization())
                    out["hbm_alloc_gb"] = torch.cuda.memory_allocated() / 2 ** 30
                    out["hbm_total_gb"] = (torch.cuda.get_device_properties(
                        torch.cuda.current_device()).total_memory) / 2 ** 30
            except Exception as e:                       # noqa: BLE001
                # P57：不再静默——首次失败把原因带出来（此前 except: pass 吞掉
                # 全部错误，NPU/HBM 恒 `--` 且无法区分「没有加速器」与「读失败」）
                self._note_acc_err(e)
            # torch 层空缺时用 npu-smi 补（AI Core% / HBM 用量；HBM 带宽
            # 利用率任何软件层都读不到，需 profiler）
            if out["acc_util"] is None and out["hbm_alloc_gb"] is None:
                self._npu_smi_fill(out)
        # ── IPC：硬件计数器，Python 层不可得（诚实 None）──
        # 外部测量：perf stat -p <pid> -- sleep 5（Linux + perf 权限）
        return out

    def _note_acc_err(self, e: Exception) -> None:
        """首次加速器采样失败时打印原因（只报一次，不刷屏）。"""
        if not self._acc_err_reported:
            self._acc_err_reported = True
            self._acc_err = f"{type(e).__name__}: {e}"
            print(f"[telemetry] ⚠ 加速器采样失败（本次运行不再重试上报，"
                  f"NPU/GPU 与 HBM 列将显示 --）：{self._acc_err}", flush=True)

    def _npu_smi_fill(self, out: dict) -> None:
        """`npu-smi info` 兜底：解析 AI Core(%) 与 HBM-Usage(MB)（best-effort）。

        解析失败静默（torch 层已兜底过一次）。
        """
        if not self._smi_path:
            return
        try:
            r = subprocess.run([self._smi_path, "info"], capture_output=True,
                               text=True, timeout=10)
            got = _parse_npu_smi(r.stdout)
        except Exception:                               # noqa: BLE001
            return
        if got is not None:
            util, used, total = got
            out["acc_util"] = util
            out["hbm_alloc_gb"] = used / 1024
            out["hbm_total_gb"] = total / 1024

    @staticmethod
    def fmt(d: dict) -> str:
        """单行日志格式（None 的字段显示 `--`）。"""
        def g(k, f="{:.0f}"):
            v = d.get(k)
            return f.format(v) if v is not None else "--"
        parts = [
            f"CPU {g('cpu_sys')}%",
            f"proc {g('cpu_proc_cores', '{:.1f}')}核",
            f"RAM {g('ram_pct')}%/{g('ram_proc_gb', '{:.1f}')}GB",
            f"NPU/GPU {g('acc_util')}%",
            f"HBM {g('hbm_alloc_gb', '{:.1f}')}/{g('hbm_total_gb', '{:.0f}')}GB",
        ]
        if d.get("ctx_switches") is not None:
            parts.append(f"CS/s {d['ctx_switches'] / max(1e-9, 1):.0f}")
        return " | " + "  ".join(parts)


def _parse_npu_smi(text: str):
    """解析 `npu-smi info` 输出 → (AI Core%, HBM used MB, HBM total MB)。

    npu-smi 表格为「两行一设备」：NPU 行（Power/Temp/Huge-Pages）与 Chip 行
    （Bus-Id 0x… | AICore(%) | AI-Real(%) | HBM-Usage used / total）。**只认
    Chip 行**（特征 = 含 0x 总线号）；AICore = HBM `used / total` 之前的
    第一个整数（`0x0000` 会被 findall 误读成 0，勿用 Bus-Id 列）。
    单卡训练取首个设备；解析失败返回 None。
    """
    for line in text.splitlines():
        if "0x" not in line or "/" not in line:
            continue
        segs = [s.strip() for s in line.split("|")]
        if len(segs) < 4:
            continue
        m_hbm = re.search(r"(\d+)\s*/\s*(\d+)", segs[3])
        if not m_hbm:
            continue
        before = re.findall(r"\d+", segs[3][:m_hbm.start()])
        if not before:
            continue
        return float(before[0]), int(m_hbm.group(1)), int(m_hbm.group(2))
    return None


def torch_available() -> bool:
    try:
        import torch                                   # noqa: F401
        return True
    except Exception:                                  # noqa: BLE001
        return False


try:
    import torch                                       # noqa: F401,E402
except Exception:                                      # noqa: BLE001
    pass
