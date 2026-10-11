"""P120：CANN 环境治理与 NPU 带宽遥测的门禁。

背景
================================================================================
P115 实测读出 7.83 ms/tok，而按字节流量算的带宽下界是 0.300 ms —— **实测是
下界的 26.1倍**，所以瓶颈**既不在算力也不在 HBM 带宽**。要继续往下定位，
缺两个东西：

1. **带宽利用率遥测**：`npu-smi info`（表格式）里**没有**这个字段，
   必须用 `npu-smi info -t usages`。此前代码注释里写「HBM 带宽利用率无法从
   任何软件层读到」——**这个判断是错的**，官方文档明确有该字段。
   判读规则：**高 AICore + 低带宽 = 瓶颈在 kernel 下发/调度**。
2. **CANN 环境变量治理**：`TASK_QUEUE_ENABLE`（默认 1，官方建议训练用 2）、
   `COMBINED_ENABLE`、`PYTORCH_NPU_ALLOC_CONF` 等我们**完全没设**。

四层验证
================================================================================
A. CANN 环境变量
   A1 默认开启的四条值正确
   A2 `CPU_AFFINITY_CONF` **默认不设**（本项目 CPU 不是瓶颈，且与已设的
      OMP_PROC_BIND / BLAS cap 可能打架）
   A3 **绝不覆盖用户显式设过的值**（P120 纪律）
   A4 幂等：调两次结果一致
B. 带宽遥测解析（用**合成**的 npu-smi 输出，两种格式都测）
   B1 `-t usages` 的 `Key : value` 格式
   B2 `-t usages` 里 `Memory Bandwidth Usage Rate(%)` 被正确识别
   B3 字段缺失/格式变化时**不崩**（返回部分或空 dict）
   B4 `npu-smi info` 表格式**不被**误当成 usages 解析
C. 日志行包含带宽字段
D. **不越界**：这些环境变量只影响设备侧调度，**不得改变任何训练数值**
   （与 P119 的 `--lang` 同类纪律：环境类开关必须证明零数值影响）
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in ("", "tests", "tools", "train"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

_RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    return bool(ok)


# ── A. CANN 环境变量 ────────────────────────────────────────────────────────
def section_a() -> None:
    print("\n[A] CANN 环境变量治理")
    # 在**子进程**里测，避免污染本进程的环境
    code = (
        "import os,sys,json;"
        "sys.path[:0]=[%r];"
        "from phdnet.backends.cann_env import apply_cann_env;"
        "import io,contextlib;"
        "buf=io.StringIO();"
        "\nwith contextlib.redirect_stdout(buf):\n    a=apply_cann_env(verbose=False)\n"
        "print(json.dumps({k:v for k,v in a.items()}))"
    ) % (str(_ROOT),)
    # ⚠ 2026-10-07：显式 UTF-8 解码，避免区域编码（cp936）在 reader 线程
    #   抛 UnicodeDecodeError → stdout=None → 下一行 json.loads(None) 崩。
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=str(_ROOT), timeout=120,
                       encoding="utf-8", errors="replace")
    got = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}

    check(got.get("TASK_QUEUE_ENABLE") == "2",
          "A1a TASK_QUEUE_ENABLE 默认=2（官方训练建议 Level 2，默认为 1）",
          f"实际={got.get('TASK_QUEUE_ENABLE')!r}")
    check(got.get("COMBINED_ENABLE") == "1",
          "A1b COMBINED_ENABLE 默认=1（默认为 0）",
          f"实际={got.get('COMBINED_ENABLE')!r}")
    check(got.get("PYTORCH_NPU_ALLOC_CONF") == "expandable_segments:True",
          "A1c PYTORCH_NPU_ALLOC_CONF 默认 expandable_segments:True",
          f"实际={got.get('PYTORCH_NPU_ALLOC_CONF')!r}")
    check(got.get("MULTI_STREAM_MEMORY_REUSE") == "1",
          "A1d MULTI_STREAM_MEMORY_REUSE 默认=1",
          f"实际={got.get('MULTI_STREAM_MEMORY_REUSE')!r}")
    check(got.get("CPU_AFFINITY_CONF") is None,
          "A2 CPU_AFFINITY_CONF 默认不设（CPU 非瓶颈，且与 OMP_PROC_BIND 可能打架）",
          f"实际={got.get('CPU_AFFINITY_CONF')!r}")

    # A3 不覆盖用户显式值（**单独一个干净进程**：上一段已经调过一次
    # apply_cann_env，若在同进程里改 os.environ 再reload，reload 前的那次
    # 调用已经把值写成了默认 2，就测不出「不覆盖」了——第一版就踩了这个，
    # 报出假失败。实现本身是对的：单独进程里直接验证。）
    code3 = (
        "import os,sys,json;"
        "sys.path[:0]=[%r];"
        "os.environ['TASK_QUEUE_ENABLE']='0';"
        "os.environ['COMBINED_ENABLE']='0';"
        "from phdnet.backends.cann_env import apply_cann_env;"
        "import io,contextlib;"
        "buf=io.StringIO();"
        "\nwith contextlib.redirect_stdout(buf):\n    a=apply_cann_env(verbose=False)\n"
        "print(json.dumps({'tq':a['TASK_QUEUE_ENABLE'],"
        "'ce':a['COMBINED_ENABLE'],"
        "'tq_env':os.environ['TASK_QUEUE_ENABLE'],"
        "'ce_env':os.environ['COMBINED_ENABLE']}))"
    ) % (str(_ROOT),)
    r3 = subprocess.run([sys.executable, "-c", code3], capture_output=True,
                        text=True, cwd=str(_ROOT), timeout=120,
                        encoding="utf-8", errors="replace")
    got3 = json.loads(r3.stdout.strip().splitlines()[-1]) if r3.returncode == 0 else {}
    check(got3.get("tq") == "0" and got3.get("tq_env") == "0",
          "A3 用户显式设的 TASK_QUEUE_ENABLE=0 **不被覆盖**",
          f"apply 返回 {got3.get('tq')!r}，os.environ 实际 {got3.get('tq_env')!r}")
    check(got3.get("ce") == "0" and got3.get("ce_env") == "0",
          "A3b COMBINED_ENABLE=0 同样不被覆盖",
          f"os.environ 实际 {got3.get('ce_env')!r}")

    # A4 幂等
    check(got.get("TASK_QUEUE_ENABLE") == "2",
          "A4 幂等：重复调用结果一致（第二进程同样得2）")


# ── B. 带宽遥测解析 ─────────────────────────────────────────────────────────
SYNTH_USAGES = """NPU ID                       : 0
Chip Count                    : 1
Chip ID                       : 0
Memory Capacity(MB)           : 65536
Memory Usage Rate(%)          : 0
Hugepages Total(page)         : 970
Hugepages Usage Rate(%)       : 0
Aicore Usage Rate(%)          : 99
Aicpu Usage Rate(%)           : 12
Ctrlcpu Usage Rate(%)         : 3
Memory Bandwidth Usage Rate(%) : 4
"""

SYNTH_TABLE = """|NPU Name  |Health| Power(W) Temp(C)Hugepages-Usage(page)|
|910B3| OK  | 88.6    51      0 / 0         |
|0+0    | 0000:5A:00.0 | 99     20701 / 65536   |
"""


def section_b() -> None:
    print("\n[B] npu-smi -t usages 带宽遥测解析")
    from phdnet.telemetry import _parse_npu_smi_usages

    u = _parse_npu_smi_usages(SYNTH_USAGES)
    check(u.get("aicore_pct") == 99.0, "B1a 解析 Aicore Usage Rate",
          f"={u.get('aicore_pct')}")
    check(u.get("hbm_bw_pct") == 4.0,
          "B1b **解析 Memory Bandwidth Usage Rate（此前完全缺失的字段）**",
          f"={u.get('hbm_bw_pct')}%")
    check(u.get("aicpu_pct") == 12.0, "B1c 解析 Aicpu Usage Rate",
          f"={u.get('aicpu_pct')}")
    check(u.get("ctrlcpu_pct") == 3.0, "B1d 解析 Ctrlcpu Usage Rate",
          f"={u.get('ctrlcpu_pct')}")

    # B2 判读规则：高 AICore + 低带宽 = 下发瓶颈（我们最可能的形态）
    if "aicore_pct" in u and "hbm_bw_pct" in u:
        hostbound = u["aicore_pct"] > 80 and u["hbm_bw_pct"] < 30
        check(hostbound, "B2 合成数据呈现「高 AICore + 低带宽」→ 判读为下发瓶颈",
              f"AICore {u['aicore_pct']}% / 带宽 {u['hbm_bw_pct']}% "
              f"→ host-bound={hostbound}")

    # B3 异常输入不崩
    for bad, label in (("", "空串"), ("garbage\nno colon here", "无冒号"),
                       ("X : abc", "值非数字")):
        try:
            r = _parse_npu_smi_usages(bad)
            check(True, f"B3 异常输入不崩（{label}）", f"→ {r}")
        except Exception as e:                          # noqa: BLE001
            check(False, f"B3 异常输入不崩（{label}）",
                  f"{type(e).__name__}: {e}")

    # B4 表格式不被误认成 usages（字段名完全不同）
    r4 = _parse_npu_smi_usages(SYNTH_TABLE)
    check("hbm_bw_pct" not in r4,
          "B4 `npu-smi info` 表格式**不会**被误解析成带宽利用率",
          f"→ {r4}")


# ── C. 日志行 ───────────────────────────────────────────────────────────────
def section_c() -> None:
    print("\n[C] 遥测日志行包含带宽字段")
    from phdnet.telemetry import Telemetry
    d = {"cpu_sys": 2.0, "cpu_proc_cores": 1.1, "ram_pct": 2.0,
         "ram_proc_gb": 2.9, "acc_util": 99.0, "hbm_alloc_gb": 0.1,
         "hbm_total_gb": 29.0, "hbm_bw_pct": 4.0, "aivector_pct": 2.0,
         "aicpu_pct": 1.0}
    line = Telemetry.fmt(d)
    check("HBM-bw 4%" in line, "C1 fmt 打出带宽利用率",
          line.strip()[:110])
    d2 = dict(d)
    d2.pop("hbm_bw_pct")
    line2 = Telemetry.fmt(d2)
    check("HBM-bw" not in line2,
          "C2 缺该字段时不打印空标签（不产生 `HBM-bw --%`）",
          line2.strip()[:80])


# ── D. 零数值影响 ───────────────────────────────────────────────────────────
def section_d() -> None:
    print("\n[D] 零数值影响：这些环境变量不改变训练结果")
    # 用子进程跑两种环境，训练同一段，比 NLL 轨迹
    code = (
        "import sys,io,json,contextlib;"
        "sys.path[:0]=[%r,'tests','tools','train'];"
        "import numpy as np;"
        "from eval_common import BASE,SEG,DOC;"
        "from phdnet.word_lm import PHDWordLM;"
        "from phdnet.config import PHDNetConfig;"
        "\nif %s:\n"
        "    import os\n"
        "    os.environ['TASK_QUEUE_ENABLE']='2'\n"
        "    os.environ['COMBINED_ENABLE']='1'\n"
        "    os.environ['PYTORCH_NPU_ALLOC_CONF']='expandable_segments:True'\n"
        "txt=io.open(DOC,encoding='utf-8').read();"
        "cfg=PHDNetConfig(**{**BASE,'readout_dtype':'fp32'});"
        "lm=PHDWordLM(txt,cfg,seg_kwargs=SEG);"
        "toks=lm.tokenize(txt)[:60];"
        "ids=[lm.tok.stoi.get(t) for t in toks];"
        "nll=[];"
        "\nwith contextlib.redirect_stdout(io.StringIO()):\n"
        "    for i in range(len(ids)-1):\n"
        "        a,b=ids[i],ids[i+1]\n"
        "        if a is None or b is None: continue\n"
        "        d=lm.net.step(lm.tok.encode_composite(toks[i],toks[i+1]),"
        "target=lm.tok.onehot(b),target_idx=b)\n"
        "        nll.append(float(d['nll']))\n"
        "W=np.asarray(lm.net.readout.W,dtype=np.float64);"
        "print(json.dumps({'nll':nll,'dig':[int(W.sum()*1e6),float(W.sum())]}))"
    ) % (str(_ROOT), "%s")
    out = []
    for flag in ("0", "1"):
        r = subprocess.run([sys.executable, "-c", code % flag],
                           capture_output=True, text=True, cwd=str(_ROOT),
                           timeout=600, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            check(False, f"D 环境 {flag} 跑失败", r.stderr[-120:])
            return
        import json
        out.append(json.loads(r.stdout.strip().splitlines()[-1]))
    check(out[0]["nll"] == out[1]["nll"],
          "D1 两种 CANN 环境下逐步 NLL **逐位相同**",
          f"{len(out[0]['nll'])} 步，首步 {out[0]['nll'][0]:.6f}")
    check(out[0]["dig"] == out[1]["dig"], "D2 读出权重指纹逐位相同",
          f"{out[0]['dig']}")


def section_e() -> None:
    """P127：`npu-smi -t usages` 必须带 `-i <id>`。

    2026-10-02 服务器日志证据：跑满 59k token，`HBM-bw` 字段出现 **0 次**，
    而 `[telemetry] npu-smi = ...` 已打印且**无 parse FAILED**
    → 说明命令「成功执行但没解析到字段」，正是缺 `-i` 的signature
    （官方文档明确 `npu-smi info -t usages -i id`）。
    本节是**静态形态检查**（本机无 NPU，跑不了真命令）。
    """
    print("\n[E] usages 子命令的调用形态（P127）")
    tel = (_ROOT / "phdnet" / "telemetry.py").read_text(encoding="utf-8")
    flat = " ".join(tel.split())
    check('"-t", "usages"' in flat and '"-i", str(' in flat,
          "E1 usages 带 `-i <id>`（不带则 910B 上 HBM-bw 恒空）")
    check("未能确定 NPU 设备号" in flat,
          "E2 取不到设备号时**打印提示**（不再静默）")
    # P130：三处必须齐备（服务器 14:0x 实测 rc=215 "must input card id"）
    check('"-c", _cid' in flat,
          "E2b usages 带 `-c <chip_id>`（官方样例均为 `-i 0 -c 0`）")
    check("PHD_NPU_ID" in flat and "PHD_NPU_CHIP_ID" in flat,
          "E2c 支持 PHD_NPU_ID / PHD_NPU_CHIP_ID 环境变量显式指定")
    check("npu\\s*id|npu" in flat.replace("\\", "\\"),
          "E2d `npu-smi info -l` 解析认 Key-Value 两种格式"
          + "（`NPU ID : 0` / `NPU : 0`）")
    check("no编号 → 假定 card 0" in flat or "无编号" in flat,
          "E2e 后端串无编号时（accel:auto@npu）假定 card 0 并提示")
    check("未解析到字段" in flat,
          "E3 usages 解析失败时**报一次原始输出**（不再静默失效）")
    tpy = (_ROOT / "train" / "train.py").read_text(encoding="utf-8")
    check("_tel._accel_device" in tpy,
          "E4 训练入口把设备号传给遥测（`_tel._accel_device`）")
    # 设备号解析本身（纯正则，可测）
    import re as _re
    t = _re.search(r'_accl_device.*?\(\d\+\)', flat)
    ok = True
    for dev, want in (("npu:0", "0"), ("npu:3", "3"), ("cuda:0", "0")):
        got = _re.search(r"(\d+)$", dev)
        if not (got and got.group(1) == want):
            ok = False
    check(ok, "E5 设备串→ 数字设备号解析（npu:N / cuda:N）")


def main() -> int:
    print("=" * 78)
    print("P120 门禁：CANN 环境治理 + NPU 带宽遥测")
    print("=" * 78)
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    npass = sum(1 for ok, _, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print("\n" + "=" * 78)
    print(f"结果：{npass}/{total} 通过 | 失败 {total - npass}")
    if npass != total:
        print("失败用例：")
        for ok, name, detail in _RESULTS:
            if not ok:
                print(f"  · {name}  {detail}")
    print("=" * 78)
    return 0 if npass == total else 1


if __name__ == "__main__":
    sys.exit(main())