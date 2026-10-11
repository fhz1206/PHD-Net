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


# P63：npu-smi 定位（环境变量 → PATH → 常见安装路径）。训练进程常没有
# source 过 Ascend 的 set_env.sh，`which npu-smi` 落空 → AI Core% 恒 `--`。
_SMI_ENV = "NPU_SMI_PATH"
_SMI_CANDIDATES = (
    "/usr/local/Ascend/driver/tools/npu-smi",
    "/usr/local/Ascend/driver/tools/npu-smi",
    "/usr/local/bin/npu-smi",
    "/usr/sbin/npu-smi",
    "/opt/ascend/driver/tools/npu-smi",
)


def _find_npu_smi() -> str | None:
    """定位 npu-smi 可执行文件（None = 找不到，AI Core% 只能显示 --）。"""
    p = os.environ.get(_SMI_ENV)
    if p and os.path.exists(p):
        return p
    w = shutil.which("npu-smi")
    if w:
        return w
    for c in _SMI_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


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
        # P63：`npu-smi` 在非交互 shell 里常常不在 PATH（能跑 `watch npu-smi`
        # 的那个 shell source 过 set_env.sh，训练是直接 python 启动的）→
        # AI Core% 恒 `--`。按「环境变量 → PATH → 常见安装路径」三级探测。
        self._smi_path = _find_npu_smi() if npu_smi else None
        self._smi_warned = False
        self._smi_reported = False

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
                     "ctx_switches": None, "gc_objs": None, "gc_gen2": None}
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
        # ── GC 指标（P64）：大空间表长跑时对象数线性增长（CSR 行 dict/数组、
        #    迹字典…），gen2 扫描频率随之上升 → ms/tok 缓慢恶化。这是**假设**，
        #    指标化后才能证实/证伪。
        try:
            import gc
            out["gc_objs"] = len(gc.get_objects())
            st = gc.get_stats()
            out["gc_gen2"] = int(st[2]["collections"]) if len(st) > 2 else None
        except Exception:                                # noqa: BLE001
            pass
        # ── 加速器 ──
        if torch_available() and self.device:
            dev = self.device.lower()
            if dev.startswith("npu"):
                # P58：AI Core% 改走 npu-smi——torch.npu.utilization() 内部会
                # 同步设备流，每 log_every 采样一次就在训练热路径上打一次全局
                # 同步（与「CPU 预计算提前 / NPU 流水」直接冲突）。HBM 占用两
                # 路都可用：memory_allocated 是 host 侧计数器（不同步）。
                self._npu_smi_fill(out)
                try:
                    if hasattr(torch, "npu"):
                        out["hbm_alloc_gb"] = torch.npu.memory_allocated() / 2 ** 30
                        out["hbm_total_gb"] = (torch.npu.get_device_properties()
                                               .total_memory) / 2 ** 30
                except Exception as e:                   # noqa: BLE001
                    self._note_acc_err(e)
            else:
                try:
                    if dev.startswith("cuda") and hasattr(torch, "cuda"):
                        # ⚠ P167：**ROCm 没有 `torch.cuda.utilization()`**
                        #   （NVIDIA 专属，走 NVML；AMD 对应的是 rocm_smi）。
                        #   原代码直接调 → ROCm 上抛异常 → 被下面的 except
                        #   吞掉 → **GPU 利用率恒为 `--`**，且只在首次打一次
                        #   警告，看起来像「没采到」而非「不支持」。
                        # → 显式区分：ROCm 只填**显存**（torch 的
                        #   memory_allocated 在 ROCm 上**可用**，属 HIP 化接口），
                        #   利用率留None 并标注原因。
                        _is_hip = bool(getattr(torch.version, "hip", None))
                        out["hbm_alloc_gb"] = torch.cuda.memory_allocated() / 2 ** 30
                        out["hbm_total_gb"] = (torch.cuda.get_device_properties(
                            torch.cuda.current_device()).total_memory) / 2 ** 30
                        if _is_hip:
                            out["acc_util"] = None
                            out["acc_util_note"] = (
                                "ROCm 无 torch.cuda.utilization()（NVML 专属）；"
                                "需 rocm-smi 或按 msprof 口径采。显存字段不受影响。")
                        else:
                            out["acc_util"] = float(torch.cuda.utilization())
                except Exception as e:                   # noqa: BLE001
                    self._note_acc_err(e)
            # torch 层空缺时用 npu-smi 补（HBM 带宽利用率任何软件层都读不到，
            # 需 profiler）
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
            if not self._smi_warned:                    # P63：只提示一次
                self._smi_warned = True
                print("[telemetry] npu-smi 未找到 → AI Core% 显示 --（HBM 仍走 "
                      f"torch 计数）。设 {_SMI_ENV}=/path/to/npu-smi 或把 "
                      "Ascend set_env.sh source 进环境即可。", flush=True)
            return
        if not self._smi_reported:                      # 首次成功：报一次路径
            self._smi_reported = True
            print(f"[telemetry] npu-smi = {self._smi_path}", flush=True)
        try:
            # ⚠ 2026-10-07：显式 UTF-8 解码（text=True 默认按区域编码解码，
            #   中文 Windows/GBK 环境下 npu-smi 输出会在 reader 线程抛
            #   UnicodeDecodeError → 这里被 except 吞掉 → **遥测静默失效**（本行
            #   加 encoding 就是为堵它；下面的 parse FAILED 分支另有独立开关，
            #   2026-10-07 已把「只报一次」的 flag 从 _smi_reported 拆出），
            #   正是 B6/B7 那类「工具静默失效」的入口）。
            r = subprocess.run([self._smi_path, "info"], capture_output=True,
                               text=True, timeout=10,
                               encoding="utf-8", errors="replace")
            got = _parse_npu_smi(r.stdout)
            # ⚠ 2026-10-07 修复（审计 P2-9）：这里原来**复用** `_smi_reported`
            #   当「解析失败只报一次」的开关，但上面 220 行已经把它置 True →
            #   `not self._smi_reported` **恒 False**，这条诊断**永不可达**：
            #   npu-smi 输出格式一变，AI Core%/HBM 就悄悄回 `--` 而日志零线索，
            #   正是本段注释声称修掉的「工具静默失效」。→ 拆独立开关。
            #   另：原文 `raw head://n` 是错误转义残留（本意是换行 \n）。
            if got is None and not getattr(self, "_smi_parse_failed_once", False):
                # 解析失败：把原始输出打出来一次，便于按真实格式修解析器
                self._smi_parse_failed_once = True
                head = "\n".join(r.stdout.splitlines()[:14])
                print(f"[telemetry] npu-smi parse FAILED; raw head:\n{head}",
                      flush=True)
        except Exception:                               # noqa: BLE001
            return
        if got is not None:
            util, used, total = got
            out["acc_util"] = util
            out["hbm_alloc_gb"] = used / 1024
            out["hbm_total_gb"] = total / 1024
        # P120：追加 `-t usages` —— **带宽利用率只在这个子命令里**。
        # 它是判断「NPU 到底有没有发挥出来」的关键：若 HBM 带宽占用很低
        # 而 AICore 很高 → 瓶颈在 kernel 下发/调度，不在算力也不在带宽。
        # ⚠ 每 tick 多一次子进程（~10 ms）。故按 `_smi_usages_every` 抽样，
        # 不每个 tick 都跑；带宽指标变化慢，抽样的代表性足够。
        self._smi_ticks = getattr(self, "_smi_ticks", 0) + 1
        _every = int(getattr(self, "_smi_usages_every", 4) or 4)
        if self._smi_ticks % _every == 0:
            # P127 修复：**必须带 `-i <device_id>`**。官方文档明确
            # `npu-smi info -t usages -i id`，且第三来源实测 910B 上无 `-i`
            # 的行为不稳定（可能报错、也可能输出多设备混杂）→ 解析必然失败，
            # 于是 `HBM-bw` 字段**从不出现**（2026-10-02 服务器日志证实：
            # 跑满59k token，`HBM-bw` 出现 **0** 次，而 `[telemetry] npu-smi=`
            # 已打印且**无 parse FAILED** → 说明是「成功执行但没解析到字段」，
            # 正是本bug 的signature）。
            _dev = self._accl_device_id()
            _cmd = [self._smi_path, "info", "-t", "usages"]
            if _dev is not None:
                # ⚠ **P130 加 `-c <chip_id>`**：多个实测样例与官方快速查询都用
                # `-i0 -c 0`（如 `watch npu-smi info -t usages -i 0 -c 0`）。
                # chip id 恒取 0（AI 芯片在卡内的编号，见 `npu-smi info -m`）。
                _cmd += ["-i", str(_dev)]
                _cid = os.environ.get("PHD_NPU_CHIP_ID", "0").strip() or "0"
                _cmd += ["-c", _cid]
            try:
                r2 = subprocess.run(_cmd, capture_output=True, text=True,
                                    timeout=10, encoding="utf-8",
                                    errors="replace")
                u = _parse_npu_smi_usages(r2.stdout)
                if not u and not getattr(self, "_usages_dbg", False):
                    # P127：**解析不到字段时必须报一次**。此前是完全静默的，
                    # 于是「`HBM-bw` 从不出现」这件事在日志里毫无痕迹——
                    # 这类「工具静默失效」比工具报错更难发现。
                    self._usages_dbg = True
                    print("[telemetry] `npu-smi -t usages` 未解析到字段；"
                          f"cmd={' '.join(_cmd)} rc={r2.returncode}；"
                          f"原始输出前 6 行：\n"   # ← 2026-10-07：原为字面 ":/" + "n"（错误转义残留）
                          + "\n".join(r2.stdout.splitlines()[:6]), flush=True)
                if u:
                    out.update(u)
                    if not getattr(self, "_usages_reported", False):
                        self._usages_reported = True
                        print("[telemetry] npu-smi -t usages 字段: "
                              + ", ".join(sorted(u)), flush=True)
            except Exception:                           # noqa: BLE001
                pass

    def _accl_device_id(self):
        """解析加速器设备号（供 `npu-smi ... -i <id>` 用）。

        来源：训练入口把设备字符串（如 `npu:0`）存进 `_accel_device`，这里取
        其数字部分。**取不到就返回 None**（不带 `-i`，行为由npu-smi 决定）——
        但那正是 P127 修复前 `HBM-bw` 恒空的原因，故启动时会打印一次提示。
        """
        #优先级 1：**环境变量显式指定**（最可靠，用户知道自己的卡号）
        _env = os.environ.get("PHD_NPU_ID", "").strip()
        if _env.isdigit():
            return int(_env)
        # 优先级 2：训练入口传入的设备串
        dev = getattr(self, "_accel_device", None)
        if dev:
            d = str(dev)
            m = re.search(r"(\d+)$", d)
            if m:
                return int(m.group(1))
            # P130：后端描述串形如 `accel:auto@npu` / `accel:npu` / `npu` —— **没有
            # 数字**，但它证明「当前确实跑在 NPU 上」→ 按单卡最常见情形取 0。
            # 风险（必须诚实标出）：多卡机器上若进程实际绑在 2 号卡，这里会读错卡
            # → 那是**别的卡的指标**（读数仍有效但不对应本进程）。
            # → 故设`PHD_NPU_ID` 可显式纠正，且本函数会打印实际采用的值。
            if re.search(r"npu|cuda|rocm", d, re.I):
                if not getattr(self, "_devid_assumed", False):
                    self._devid_assumed = True
                    print(f"[telemetry] 设备串 {d!r} 无编号 → 假定 card 0"
                          "（多卡若绑到其它卡请设 PHD_NPU_ID）", flush=True)
                return 0
        # 退化：从 `npu-smi info -l` 读第一个 NPU ID。
        # ⚠ **P130 修**：原正则 `^\s*(\d+)\s` 匹配不到真实格式——该命令输出是
        # Key-Value 式而非表格行，两种实测格式都要认：
        #   `NPU ID : 0`（华为文档样例）/ `NPU : 0`（实测样例），
        #   另有`Total : 8` / `Card Count : 8` 这类**计数行**必须排除。
        # 服务器 2026-10-02 14:0x 的证据：两条路径都失败 → 不带 `-i` →
        # `npu-smi` 直接 rc=215 "This command must input card id."。
        try:
            r = subprocess.run([self._smi_path, "info", "-l"],
                               capture_output=True, text=True, timeout=10,
                               encoding="utf-8", errors="replace")
            for ln in r.stdout.splitlines():
                m = re.match(r"\s*(?:npu\s*id|npu)\s*[:：]\s*(\d+)",
                             ln, re.I)
                if m:
                    return int(m.group(1))
        except Exception:                                   # noqa: BLE001
            pass
        if not getattr(self, "_devid_warned", False):
            self._devid_warned = True
            print("[telemetry] 未能确定 NPU 设备号 → `npu-smi -t usages` "
                  "不带 -i，`HBM-bw` 可能仍为空（设 PHD_NPU_ID 环境变量可指定）",
                  flush=True)
        return None

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
        # P120：带宽利用率（仅在 `-t usages` 抽样命中时出现）。
        # **高 AICore + 低带宽 = 瓶颈在 kernel 下发，不是算力/带宽**——
        # 这个组合是本项目读出段 7.83 ms 的最可能形态（实测是带宽下界的
        # 26.1 倍，故必然不是带宽瓶颈）。
        if d.get("hbm_bw_pct") is not None:
            parts.append(f"HBM-bw {d['hbm_bw_pct']:.0f}%"
                         f"/vec {g('aivector_pct')}%"
                         f"/aicpu {g('aicpu_pct')}%")
        if d.get("ctx_switches") is not None:
            parts.append(f"CS/s {d['ctx_switches'] / max(1e-9, 1):.0f}")
        if d.get("gc_objs") is not None:
            parts.append(f"GC {d['gc_objs']/1e6:.1f}M/gen2 {d.get('gc_gen2')}")
        return " | " + "  ".join(parts)


def _parse_npu_smi(text: str):
    """解析 `npu-smi info` → (AI Core%, HBM used MB, HBM total MB)；失败 None。

    **按表头定位列**（fhz 服务器 2026-09-29 实测：先前的「找含 0x 总线号的
    数据行」启发式在该机输出上匹配不到 → NPU% 恒 `--`；不同型号/版本的
    Bus-Id 列并不通用）。做法：先找到含 `AICore` 的**表头行**，切列得到
    AICore / HBM-Usage 的列序，再逐行按列号取值 → 与型号无关。
    """
    lines = text.splitlines()
    hdr = None
    for i, ln in enumerate(lines):
        if "AICore" in ln:
            hdr = i
            break
    if hdr is None:                                  # 表头可能叫 Aicore
        for i, ln in enumerate(lines):
            if "aicore" in ln.lower():
                hdr = i
                break
    if hdr is None:
        return None
    cols = [c.strip() for c in lines[hdr].split("|")]
    ai_col = next((k for k, c in enumerate(cols) if "aicore" in c.lower()), None)
    hbm_col = next((k for k, c in enumerate(cols) if "hbm" in c.lower()), None)
    if ai_col is None or hbm_col is None:
        return None
    for ln in lines[hdr + 1:]:
        if "|" not in ln:
            continue
        segs = [s.strip() for s in ln.split("|")]
        if len(segs) <= max(ai_col, hbm_col):
            continue
        m = re.search(r"(\d+)\s*/\s*(\d+)", segs[hbm_col])
        if not m:
            continue
        ai = re.findall(r"\d+", segs[ai_col])
        if not ai:
            continue
        return float(ai[0]), int(m.group(1)), int(m.group(2))
    return None


def _parse_npu_smi_usages(text: str) -> dict:
    """解析 `npu-smi info -t usages` → 关键指标字典（失败返回 {}）。

    P120：**`npu-smi info`（表格式）里没有带宽利用率**，必须用 `-t usages`
    子命令才有。这些字段是判断「NPU 有没有真正发挥出来」的**直接证据**：

    ==========================  ============================================
    字段含义（官方口径）
    ==========================  ============================================
    ``Aicore Usage Rate(%)``    AI Core 占用率。⚠ **它高≠算力用满**：
                                 大量小算子（launch 开销）也能把它顶到
                                 99%，而真正的大矩阵可能没在跑。
    ``Memory Bandwidth Usage     **HBM 带宽占用率** —— 判断 memory-bound
     Rate(%)`` 的**唯一**可靠指标。实测口径（910B）：memory-bound 算子会让
                                 它接近 100%；若它很低而 AICore 很高，
                                 说明瓶颈**不在算力也不在带宽**，而在
                                 kernel 调度/下发（见 TASK_QUEUE_ENABLE）。
    ``Aivector Usage Rate(%)``  Vector 单元占用率。
    ``Aicpu Usage Rate(%)``     AI CPU（控制流 + 非矩阵算子）占用率。
    ``Ctrlcpu Usage Rate(%)``   管理 CPU 占用率。
    ``Memory Usage Rate(%)``    显存占用率。
    ==========================  ============================================

    输出样例（每行 `Key : value`，故按 `:` 切分而非按列）：
        NPU ID : 0
        Aicore Usage Rate(%) : 99
        Memory Bandwidth Usage Rate(%) : 4
    """
    out: dict = {}
    for ln in text.splitlines():
        if ":" not in ln:
            continue
        k, _, v = ln.partition(":")
        k = k.strip().lower()
        v = v.strip()
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if not m:
            continue
        try:
            num = float(m.group(0))
        except ValueError:
            continue
        if "aicore" in k:
            out["aicore_pct"] = num
        elif "bandwidth" in k:
            out["hbm_bw_pct"] = num
        elif "aivector" in k:
            out["aivector_pct"] = num
        elif "aicpu" in k:
            out["aicpu_pct"] = num
        elif "ctrlcpu" in k:
            out["ctrlcpu_pct"] = num
        elif "memory usage" in k or "hbm usage" in k:
            out["mem_usage_pct"] = num
    return out


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
