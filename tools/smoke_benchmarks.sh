#!/usr/bin/env bash
# P140：所有 benchmark 工具的**短跑冒烟**。
# 动机：P139 我在bench 里写错变量名（`_ro` vs `ro`）→ 服务器 NameError 崩。
# 静态 AST 扫描误报 15/16，**不可用**；短跑冒烟是零误报且真正能抓到
# 「导入即炸 / 拼错名 / 参数不合法」的唯一可靠手段。
# 用法：bash tools/smoke_benchmarks.sh   （提交前跑；本机无 NPU 也能跑）
set -u
PY="${PY:-python3}"
fail=0
run() {
  printf '%-42s' "$1"
  if timeout 300 "$PY" "$@" >/tmp/_smoke.log 2>&1; then
    echo "OK"
  else
    rc=$?
    # 昇腾相关工具在本机会因无 NPU 而非零退出—— 区分「环境缺失」与「真错」
    if grep -qE "torch_npu 不可用|无 NPU|回落臂|npu:0.*NotImplemented|ModuleNotFoundError: No module named 'torch_npu'" /tmp/_smoke.log; then
      echo "SKIP（本机无 NPU，退出码 $rc）"
    else
      echo "FAIL（退出码 $rc）"; tail -5 /tmp/_smoke.log; fail=1
    fi
  fi
}
run tools/bench_accel_kernels.py     --preset smoke --reps 2
run tools/bench_readout_segments.py  --preset smoke --conn-k 32
run tools/bench_readout_lineprofile.py --steps 2 --n-out 2048 --top 3
run tools/bench_readout_sparse.py    --preset rebaseline --tokens 64
run tools/probe_readout_precision.py
echo "----"
[ "$fail" = 0 ] && echo "全部通过" || echo "有失败项（见上）"
exit $fail
