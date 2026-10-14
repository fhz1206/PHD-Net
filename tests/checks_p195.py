"""P195 检查：DeepSeek-V4-Flash 主权重 + 低精度影子副本（加速臂 + numba 臂）。

跑法：python tests/checks_p195.py [--ab]
  默认：结构/行为/往返的快速断言（<30 s）。
  --ab ：追加 PPL A/B 表（fp8 ± master vs fp32 基线，~10 s）。

设计纪律（P146）：**训练流 PPL 波动 ±20% 不可用**，故 A/B 一律用
**固定探针集**（同一语料的 evaluate）而非训练流均值。
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ROOT_TRAIN = ROOT / "train"
if str(ROOT_TRAIN) not in sys.path:
    sys.path.insert(0, str(ROOT_TRAIN))

import numpy as np                                        # noqa: E402
import torch                                              # noqa: E402

from phdnet.config import PHDNetConfig                    # noqa: E402
from phdnet.model import PHDNet                           # noqa: E402
from phdnet.readout import Readout                        # noqa: E402
from phdnet.word_lm import PHDWordLM                      # noqa: E402

TEXT = "小猫追蝴蝶。小狗摇尾巴。鸟儿飞过树梢。" * 30
SEG_KW = dict(max_len=4, min_count=3, min_entropy=0.6)


def _mk_accel(master=True, dtype="fp8", refresh=4, mb=1):
    cfg = PHDNetConfig(
        n_input=64, n_sdr=128, n_mid=64, n_top=32, k_sparse=16,
        n_readout=0, readout_softmax=True, eta_readout=0.15,
        seed=11, step_profiling=False, accel_readout="cpu:0",
        readout_dtype=dtype, readout_master_weight=master,
        readout_shadow_refresh=refresh, micro_batch=mb, nll_sync_every=1,
        eta_pc=0.05, eta_oja=0.05, pc_dev_steps=400,
        big_ltm=True, big_ltm_N=1 << 16, big_ltm_m=8, big_ltm_k=2)
    return PHDWordLM(TEXT, cfg, seg_kwargs=SEG_KW)


def _drive(lm, n=8, seed=123):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x = rng.normal(0, 1, lm.net.cfg.n_input).astype(np.float32)
        t = np.eye(lm.net.n_out)[int(rng.integers(lm.net.n_out))]
        lm.net.step(x, target=t)


def t_master_structure():
    """A：master 是 fp32、影子是低精度、更新落在 master、影子按间隔刷新。"""
    lm = _mk_accel(master=True, refresh=4)
    ro = lm.net.readout
    assert type(ro).__name__ == "AccelReadout", type(ro).__name__
    _drive(lm, 6)
    assert ro._W_master is not None and ro._W_master.dtype == torch.float32
    assert str(ro.W.dtype) in ("torch.uint8", "torch.float8_e4m3fn"), ro.W.dtype
    m0 = ro._W_master.detach().clone()
    _drive(lm, 3)
    assert float((ro._W_master.detach() - m0).abs().max()) > 0, "master 未更新"
    # 影子刷新间隔
    ro._shadow_age = 0
    ro._master_to_shadow()
    w_before = ro.W.clone()
    _drive(lm, 3)
    assert bool((ro.W == w_before).all().item()), "影子在间隔内被刷新（应不刷）"
    _drive(lm, 2)
    assert not bool((ro.W == w_before).all().item()), "影子到间隔未刷新"
    # P110 反证解除
    m = ro._W_master
    small = int(((m.abs() > 0) & (m.abs() < 2.0 ** -6)).sum().item())
    assert small > 0, "master 未保有亚 fp8 粒度的值"
    return f"master fp32 + fp8 影子；亚粒度值 {small}/{m.numel()}"


def t_master_all_dtypes():
    """B：更新分派与存储 dtype 无关（BugC 回归）。"""
    out = []
    for dt in ("fp8", "fp32", "fp16", "bf16"):
        lm = _mk_accel(master=True, dtype=dt, refresh=4)
        _drive(lm, 6)
        m = getattr(lm.net.readout, "_W_master", None)
        assert m is not None, f"{dt}: master 未建成"
        b = m.detach().clone()
        _drive(lm, 3)
        c = float((m.detach() - b).abs().max())
        assert c > 0, f"{dt}: master 未被更新（BugC 回归）"
        out.append(f"{dt}:{c:.1e}")
    return "各 dtype master 均更新 → " + " ".join(out)


def t_master_ckpt_roundtrip():
    """C：master fp32 逐位恢复；非 master 按影子粒度恢复。"""
    from ckpt_1b import load_model, save_model
    td = tempfile.mkdtemp()
    p = Path(td) / "ck.npz"
    try:
        lm = _mk_accel(master=True, refresh=4)
        _drive(lm, 8)
        m0 = lm.net.readout._W_master.detach().cpu().numpy().copy()
        meta = save_model(p, lm, lm.net.cfg, done=10, ckpt_dtype="fp8")
        lm2 = _mk_accel(master=True, refresh=4)
        load_model(p, lm2)
        m1 = lm2.net.readout._W_master.detach().cpu().numpy()
        d = float(np.abs(m0 - m1).max())
        assert d == 0.0, f"master 往返非逐位：{d}"
        assert meta.get("ro_master") is True, "meta.ro_master 应为 True"
        with np.load(p, allow_pickle=False) as z:
            assert "ro_master" not in z.files, "npz 不应有同名 ro_master"
            assert np.asarray(z["ro_W"]).dtype == np.float32, "ro_W 应未被压缩"
        # 非 master 路径
        lm3 = _mk_accel(master=False, refresh=4)
        _drive(lm3, 8)
        w0 = lm3.net.readout.W_cpu().copy()
        p2 = Path(td) / "ck2.npz"
        save_model(p2, lm3, lm3.net.cfg, done=10, ckpt_dtype="fp8")
        lm4 = _mk_accel(master=False, refresh=4)
        load_model(p2, lm4)
        w1 = lm4.net.readout.W_cpu()
        assert float(np.abs(w0 - w1).max()) <= 2.0 ** -6 * 1.001, "影子粒度不符"
        # BugA：fp8 ckpt 恢复后前向不崩
        h = np.random.default_rng(0).normal(
            0, 1, lm2.net.readout.n_h).astype(np.float32)
        assert lm2.net.readout(h) is not None
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)
    return "master 逐位往返；影子粒度往返；fp8 恢复前向正常"


def t_master_numba_arm():
    """D：numba Readout 也有主权重（GAP-4：--accel cpu 不 silent 失效）。"""
    rng = np.random.default_rng(0)
    ro = Readout(64, 32, rng, dtype="int8", master_weight=True,
                 shadow_refresh=4)
    assert ro.qfmt == "int8", ro.qfmt
    r2 = np.random.default_rng(1)
    for _ in range(6):
        ro.learn_softmax(r2.normal(0, 1, 64).astype(np.float32),
                         np.eye(32)[int(r2.integers(32))], 0.15)
    assert ro._W_master is not None and ro._W_master.dtype == np.float32
    m0 = ro._W_master.copy()
    for _ in range(3):
        ro.learn_softmax(r2.normal(0, 1, 64).astype(np.float32),
                         np.eye(32)[int(r2.integers(32))], 0.15)
    assert float(np.abs(ro._W_master - m0).max()) > 0, "numba master 未更新"
    small = int(((np.abs(ro._W_master) > 0)
                 & (np.abs(ro._W_master) < 2.0 ** -6)).sum())
    assert small > 0, "numba master 未保有亚粒度值"
    assert np.array_equal(ro.W_authoritative(), ro._W_master)
    # 默认关闭零参与
    ro2 = Readout(64, 32, np.random.default_rng(0), dtype="int8")
    assert ro2._W_master is None
    for _ in range(3):
        ro2.learn_softmax(r2.normal(0, 1, 64).astype(np.float32),
                          np.eye(32)[int(r2.integers(32))], 0.15)
    assert ro2._W_master is None, "master off 时不应建 master"
    return f"numba master fp32；亚粒度值 {small}；off 时零参与"


def t_master_interactions():
    """E：与 torch.compile / NPU Graph / micro_batch / 降级的交互。"""
    lm = _mk_accel(master=True, refresh=4)
    assert lm.net.readout._compiled is False, "master 模式应禁用 torch.compile"
    lm2 = _mk_accel(master=True, refresh=100, mb=4)
    _drive(lm2, 8)                       # 2 个完整 micro_batch
    m = lm2.net.readout._W_master
    assert m is not None and float(m.abs().sum()) > 0, "micro_batch 未走 master"
    # 降级：图应被废弃
    ro = lm.net.readout
    ro._npu_graph_ready = True           # 模拟已捕获
    ro._fp8_native = True
    try:
        ro._degrade_after_matmul_failure(RuntimeError("addmv not implemented"))
    except Exception:                    # noqa: BLE001
        pass
    assert not ro._npu_graph_ready, "降级后 NPU Graph 应被废弃（Bug-2）"
    return "compile 禁用；micro_batch 走 master；降级废图"


def t_ppl_ab():
    """F：PPL A/B（固定探针集，P146 纪律）。"""
    rows = []

    def _run(label, accel, dtype, master, refresh=8, mb=1):
        cfg = PHDNetConfig(
            n_input=64, n_sdr=128, n_mid=64, n_top=32, k_sparse=16,
            n_readout=0, readout_softmax=True, eta_readout=0.15,
            seed=11, step_profiling=False, accel_readout=accel,
            readout_dtype=dtype, readout_master_weight=master,
            readout_shadow_refresh=refresh, micro_batch=mb,
            eta_pc=0.05, eta_oja=0.05, pc_dev_steps=400,
            big_ltm=True, big_ltm_N=1 << 16, big_ltm_m=8, big_ltm_k=2)
        lm = PHDWordLM(TEXT, cfg, seg_kwargs=SEG_KW)
        lm.train_stream(TEXT, log_every=200)
        ppl = lm.evaluate(TEXT)["ppl_token"]
        rows.append((label, ppl))
        return ppl

    a_no = _run("accel fp8 无 master", "cpu:0", "fp8", False)
    a_yes = _run("accel fp8 + master", "cpu:0", "fp8", True)
    a_f32 = _run("accel fp32 基线", "cpu:0", "fp32", False)
    n_no = _run("numba int8 无 master", "cpu", "fp8", False)
    n_yes = _run("numba int8 + master", "cpu", "fp8", True)
    n_f32 = _run("numba fp32 基线", "cpu", "fp32", False)
    lines = []
    for lbl, v, base_nm, base_f32 in [
            ("accel fp8 无 master", a_no, a_no, a_f32),
            ("accel fp8 + master", a_yes, a_no, a_f32),
            ("numba int8 无 master", n_no, n_no, n_f32),
            ("numba int8 + master", n_yes, n_no, n_f32)]:
        lines.append(f"{lbl:<24} PPL={v:6.2f}  "
                     f"{(v / base_nm - 1) * 100:+6.1f}% vs 无master  "
                     f"{(v / base_f32 - 1) * 100:+5.1f}% vs fp32")
    # 断言：master 模式必须显著优于无 master，且接近 fp32 基线
    assert a_yes < a_no * 0.9, f"accel master 未改善：{a_yes} vs {a_no}"
    assert n_yes < n_no * 0.9, f"numba master 未改善：{n_yes} vs {n_no}"
    assert abs(a_yes / a_f32 - 1) < 0.10, f"accel master 偏离 fp32 过大：{a_yes}"
    assert abs(n_yes / n_f32 - 1) < 0.10, f"numba master 偏离 fp32 过大：{n_yes}"
    return "\n      ".join(lines)


TESTS = [
    ("master 结构（fp32 master + fp8 影子）", t_master_structure),
    ("各存储 dtype 下 master 均更新（BugC）", t_master_all_dtypes),
    ("检查点往返（master 逐位 / 影子粒度）", t_master_ckpt_roundtrip),
    ("numba 臂主权重（GAP-4）", t_master_numba_arm),
    ("机制交互（compile / graph / micro_batch / 降级）", t_master_interactions),
]


def main():
    ab = "--ab" in sys.argv
    tests = list(TESTS)
    if ab:
        tests.append(("PPL A/B（固定探针集）", t_ppl_ab))
    ok = 0
    for name, fn in tests:
        try:
            detail = fn()
            print(f"  PASS {name}" + (f"\n      {detail}" if detail else ""))
            ok += 1
        except Exception as e:                        # noqa: BLE001
            import traceback
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{ok}/{len(tests)} 项通过")
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
