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
import time

import numpy as np

try:
    import psutil
except Exception:                                        # pragma: no cover
    psutil = None


class Telemetry:
    """训练遥测采样器（非阻塞；所有字段容错，不可用即 None）。"""

    def __init__(self, device: str | None = None):
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
        self.device = device or ""

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
            try:
                if self.device.startswith("npu") and hasattr(torch, "npu"):
                    if hasattr(torch.npu, "utilization"):
                        out["acc_util"] = float(torch.npu.utilization())
                    out["hbm_alloc_gb"] = torch.npu.memory_allocated() / 2 ** 30
                    out["hbm_total_gb"] = (torch.npu.get_device_properties()
                                           .total_memory) / 2 ** 30
                elif self.device.startswith("cuda") and hasattr(torch, "cuda"):
                    out["acc_util"] = float(torch.cuda.utilization())
                    out["hbm_alloc_gb"] = torch.cuda.memory_allocated() / 2 ** 30
                    out["hbm_total_gb"] = (torch.cuda.get_device_properties(
                        torch.cuda.current_device()).total_memory) / 2 ** 30
            except Exception:                            # noqa: BLE001
                pass
        # ── IPC：硬件计数器，Python 层不可得（诚实 None）──
        # 外部测量：perf stat -p <pid> -- sleep 5（Linux + perf 权限）
        return out

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
