"""P192 CPU/NPU 重叠的**事实核对**门禁 —— 不是「证明它变快了」，是
「把重叠的真实前提逐条钉死，避免以后基于错误前提去优化」。

这个门禁回答一个很具体的问题
================================================================================
既然 `phdnet/model.py:369-382`（P117）**已经把读出改成异步提交**
（`forward_dev` 返回设备张量、零同步；`nll_sync_every=8` 摊薄 `.item()`），
那么「让 NPU 别等 CPU」这件事**还剩多少空间**？

结论（逐条有代码依据）：
  ① 常规 forward_dev 不回读完整 logits；NLL 同步有门控；
  ② 静态子串检查无法排除调用链内其他同步；运行时下发/队列背压
     是候选原因，需要服务器 msprof 才能证实，
     **本机无法验证**；
  ③ 所以「再加一个读出流水」很可能是**在已有异步之上叠第二层异步**，
     收益未知、风险（时序语义）实在。**故本轮不加**（见 §结论）。

⚠ **诚实边界**：本机（x86，**无昇腾/NPU**）跑本门禁**只能验证代码契约**
  （哪些分支同步、哪些不同步、开关默认值、跨 token 依赖顺序），
  **不能**给出任何端到端收益数字。

运行：``python tests/verifiers/verify_overlap_contract.py``
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

CASES: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CASES.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'} {name}"
          + (f"  [{detail}]" if detail else ""), flush=True)


def _src(path: str) -> str:
    return (_ROOT / path).read_text(encoding="utf-8")


def _main() -> int:
    print("P192 CPU/NPU overlap contract gate")

    # ── A. 开关默认必须全关（铁律④）──────────────────────────────────────
    from phdnet.config import PHDNetConfig
    c = PHDNetConfig()
    check("A1 readout_pipeline 默认 False",
          c.readout_pipeline is False, f"实际 {c.readout_pipeline!r}")
    check("A2 cann_dispatch 默认 False",
          c.cann_dispatch is False, f"实际 {c.cann_dispatch!r}")

    # ── B. 读出已异步：训练步设备路径**不应**有阻塞式 D2H ────────────────
    m = _src("phdnet/model.py")
    check("B1 训练步走 forward_dev（返回设备张量、零同步）",
          "self.readout.forward_dev(h)" in m)
    check("B2 训练步不需要 numpy 化 y（y=None）",
          "y = None" in m and "训练步不需要 numpy" in m)
    # 静态子串仅核对 NLL 门控存在，不证明所有调用链都无其他同步。
    check("B3 NLL 同步门控存在（不证明它是唯一同步点）",
          "nll_sync_every <= 1" in _src("phdnet/backends/accel_readout.py"))

    # ── C. 「不存在隐式同步」是**代码契约**，不是猜测 ─────────────────────
    # 判据：训练步设备路径上不得出现 .cpu() / np.asarray(y) 之类 D2H。
    # 这条一旦被破坏（有人加了 `y_dev.cpu()`），门禁立刻红。
    step_body = m[m.index("    def step(self, x"):m.index("    def step(self, x") + 20000]
    bad = [t for t in ("y_dev.cpu()", "y_dev.numpy()", "float(y_dev)")
           if t in step_body]
    check("C1 step() 内无 y_dev 的显式 D2H（异步契约未被破坏）",
          not bad, f"发现 {bad}" if bad else "forward_dev 直通")

    # ── D. 学习段**不读 y** —— 这是「可重叠」的前提 ──────────────────────
    # PC_learn 吃 cache + _prev_pc；STDP_learn 吃 _prev_rate/rate。
    # 若将来有人让它们依赖 y，本门禁必须红（否则重叠会算错值）。
    pc_learn = step_body[step_body.index("pc.learn("):step_body.index("pc.learn(") + 400] \
        if "pc.learn(" in step_body else ""
    stdp_call = step_body[step_body.index("self.stdp.step("):
                          step_body.index("self.stdp.step(") + 200] \
        if "self.stdp.step(" in step_body else ""
    check("D1 PC_learn 不引用 y/y_dev",
          "y_dev" not in pc_learn and "y)" not in pc_learn.replace("y_pre)", ""),
          pc_learn.strip().split("\n")[0][:60] if pc_learn else "n/a")
    check("D2 STDP_learn 不引用 y/y_dev",
          "y_dev" not in stdp_call and " y" not in stdp_call,
          stdp_call.strip().split("\n")[0][:60] if stdp_call else "n/a")

    # ── E. 跨 token 的真实 y 依赖只有一处，且**默认关闭** ─────────────────
    # readout_recurrence：word_lm 用 d['y'] 构造下一 token 的 recur_cue。
    check("E1 readout_recurrence 默认 False",
          c.readout_recurrence is False, f"实际 {c.readout_recurrence!r}")
    check("E2 该依赖确实存在（word_lm 从 d['y'] 取线索）",
          'd["y"]' in _src("phdnet/word_lm.py"))

    # ── F. P117 已明确拒绝「把学习段提到读出之前」，且理由成立 ───────────
    # 理由：STDP_learn(self._prev_rate, rate) 读**上一步**的 rate，
    # 前移等于换了一套时序规则 —— 这是**学习规则语义变更**，非调度优化。
    check("F1 P117 的拒绝理由仍在代码里（防止后人误加）",
          "本轮不实现顺序前移" in m)
    check("F2 STDP_learn 确实用 _prev_rate（上一步的 rate）",
          "self.stdp.step(self._prev_rate, rate" in m)

    # ── G. 结论：为什么本轮**不加**读出流水（写进门禁，防遗忘）────────────
    print("\n  [结论] 读出流水（跨 token 提前提交）本轮**不实现**，理由三条：")
    print("    ① 训练已有 forward_dev 异步提交（P117/P36），额外流水是在已有"
          "\n       异步之上重新调度**，收益未测，不能静态断言为零；")
    print("    ② 12.09 = 4.26 + 7.83 的**完全串行**若真存在，其成因是运行时"
          "\n       下发/队列背压，不是显式同步 —— 要证实它需服务器 msprof，"
          "\n       **本机无法验证**（不得凭 x86 结果下结论）；")
    print("    ③ 唯一的跨 token y 依赖 readout_recurrence 默认关，但一旦打开，"
          "\n       流水会导致 recur_cue 拿到**上一步**的 y —— 静默的语义错误，"
          "\n       比它可能带来的收益更危险。")
    print("    → 真正值得做的是 CANN 侧下发（--cann-dispatch）与 Cython 核"
          "\n       （--cython-kernels），两者都不动时序语义。")
    check("G1 readout_pipeline 保留为配置项但**未接线**（显式留白，非半实现）",
          not _wired(c, "readout_pipeline"),
          "config 有字段、model 未消费")
    # ⚠⚠ **绝不能是「开了没反应」的开关** —— 用户会以为自己在跑一个优化。
    #   故 train.py 对它 fail-fast：传了就报错退出。
    tr = _src("train/train.py")
    check("G2 传 --readout-pipeline 会 fail-fast（不静默接受）",
          "readout_pipeline" in tr and "SystemExit" in tr
          and "已被 P192 关闭" in tr,
          "raise SystemExit → 退出码 1，不假装生效")
    # 该报错必须在**任何训练副作用之前**触发（否则会先建模型/开日志）
    i_flag = tr.index("cfg.readout_pipeline = args.readout_pipeline")
    i_raise = tr.index("已被 P192 关闭")
    i_build = tr.index("PHDNet(") if "PHDNet(" in tr else 10 ** 9
    check("G3 fail-fast 发生在模型构造之前（无副作用）",
          i_raise < i_build,
          f"报错行 {i_raise} < 建模行 {i_build}")

    # ── I. 推理路径：与训练**结构不同**，「CPU/NPU 重叠」在推理侧不成立 ───
    # ⚠ 这一组纠正的是一个**提法错误**，不是「还没做」。自回归生成里，下一个
    #   token 的输入就是上一个 token 采样出来的结果 —— CPU 与 NPU 天然串行。
    # 当前实现会完整回读 logits，但这不证明 host 等待不可优化。证据链：
    #   · `model.py:391` `_dev_readout = (learn and not readonly and _is_accel)`
    #     —— 推理是 `learn=False` → 恒 False → 走 `readout(h)`（第 398 行）；
    #   · `accel_readout.forward()` 末行 `y.float().cpu().numpy()`
    #     （`accel_readout.py:1089`）—— 一次**硬 D2H + 同步**。
    # 相邻生成 token 的依赖须保留；完整 logits D2H 则是实现选择。
    # 本轮不为推理加流水，设备端采样仍需独立正确性/性能验证。
    acc = _src("phdnet/backends/accel_readout.py")
    check("I1 推理步走 readout(h) 硬 D2H（learn=False → _dev_readout 恒 False）",
          "learn and not readonly" in m and "y = None" in m,
          "model.py:391 判定含 learn → 推理不取设备直通")
    check("I2 AccelReadout.forward 末尾是硬 D2H（推理的同步点）",
          "return y.float().cpu().numpy()" in acc,
          "accel_readout.py:1089")
    check("I3 forward_dev 是零同步版本，且只被训练步调用",
          "def forward_dev(self, h)" in acc
          and "self.readout.forward_dev(h)" in m,
          "forward_dev 返回设备张量，仅 _dev_readout 分支用")
    check("I4 推理的采样数学在 host 上跑（sample_next 消费 numpy y）",
          '["y"]' in _src("train/infer.py"),
          "infer.py:172 sample_next 拿 y.astype(float64)")
    print("  [结论] 推理**不加**流水：自回归下 token t+1 的输入依赖 token t 的"
          "\n       采样结果；当前实现走硬 D2H，设备端采样仍是未验证候选。"
          "\n       P192 的 CPU/NPU 工作只针对**训练**路径。")

    # ── H. `--cann-dispatch`：核对器存在，且**非昇腾时不算失败** ──────────
    from phdnet.backends import cann_env
    r = cann_env.verify_dispatch(report=False)
    check("H1 verify_dispatch() 可调用且返回结构完整",
          {"effective", "checks", "notes"} <= set(r.keys()))
    if r["effective"] is None:
        check("H2 不适用或未知状态不被宣称已生效",
              bool(r["notes"]) or any(ok is None for _, ok, _ in r["checks"]),
              r["notes"][0][:50] if r["notes"] else "")
    else:
        check("H2 昇腾环境下逐项核对 task_queue 失效条件",
              len(r["checks"]) >= 4, f"{len(r['checks'])} 项")
    # 它必须**不修改**环境变量（可训练中途安全调用）
    before = cann_env.describe()
    cann_env.verify_dispatch(report=False)
    check("H3 verify_dispatch 不修改任何环境变量（幂等只读）",
          before == cann_env.describe())
    # TASK_QUEUE_ENABLE / COMBINED_ENABLE 本来就默认开（P120）——
    # 这条防止有人误以为 --cann-dispatch 是「新加的优化」
    check("H4 TASK_QUEUE_ENABLE/COMBINED_ENABLE 本就默认开（--cann-dispatch 只核对不新增）",
          cann_env._CANN_VARS["TASK_QUEUE_ENABLE"][0]
          and cann_env._CANN_VARS["COMBINED_ENABLE"][0],
          "P120 起默认开启")

    n_ok = sum(1 for c_ in CASES if c_[1])
    print(f"\n{'=' * 62}\noverlap contract: {n_ok}/{len(CASES)} passed")
    if n_ok != len(CASES):
        for nm, ok, det in CASES:
            if not ok:
                print(f"  FAIL {nm}  [{det}]")
    print("=" * 62)
    return 0 if n_ok == len(CASES) else 1


def _wired(cfg, field: str) -> bool:
    """该配置项是否已被 `phdnet/model.py` 真正消费（不只是声明）。"""
    src = _src("phdnet/model.py")
    return f"cfg.{field}" in src


if __name__ == "__main__":
    sys.exit(_main())