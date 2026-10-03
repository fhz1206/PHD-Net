#!/usr/bin/env bash
# P164：工作区清理（fhz 2026-10-03「没用的日志、构建件、修复用代码就删除掉」）
#
# ⚠️ **只删「可重建的中间产物」**，三类必须保留（项目纪律）：
#   ① **证据日志**：文档/记忆引用过的 *.log（尤其 outputs/experiments/ 下的
#      历史实验记录 —— 那是 P 系列结论的原始凭据）。
#   ② **数据集**：datasets/ 与远程分片缓存。
#   ③ **训练产物**：outputs/models/*.npz（检查点，删除需fhz 确认）。
#
# 用法：
#   bash tools/cleanup_workspace.sh# 预演（只报不删）
#   bash tools/cleanup_workspace.sh --yes           # 真删
#   bash tools/cleanup_workspace.sh --yes --aggressive   # 连旧日志/产物一起删
set -uo pipefail

DRY=1
AGGRESSIVE=0
for a in "$@"; do
case "$a" in
    --yes|-y) DRY=0 ;;
    --aggressive) AGGRESSIVE=1 ;;
  esac
done

cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

run() {
    # run "<说明>" <cmd...>
    local desc="$1"; shift
    if [[ $DRY -eq 1 ]]; then
        local n
        n="$("$@" 2>/dev/null | wc -l)"
        printf '  [dry] %-52s %s 项\n' "$desc" "$n"
    else
        "$@" >/dev/null 2>&1
        printf '  [删]  %s\n' "$desc"
    fi
}

echo "======================================================================"
echo "P164 工作区清理${DRY:+  （预演模式，加 --yes 真删）}"
echo "根目录：$ROOT"
echo "======================================================================"
echo
echo "【A】Python 字节码（可重建）"
run "find . -name '__pycache__' -type d -not -path './.git/*' -exec rm -rf {} +" \
    bash -c "find . -name '__pycache__' -type d -not -path './.git/*'"
run "find . -name '*.pyc' / '*.pyo' -delete" \
    bash -c "find . -name '*.pyc' -o -name '*.pyo' | grep ."

echo
echo "【B】numba 编译缓存（可重建，但会牺牲 ~2.6 s 启动）"
#⚠ 默认**保留** NUMBA_CACHE_DIR：P22 记录首次全量编译要几十秒，
#   删了每次启动都重来。只有 --aggressive 才清。
if [[ $AGGRESSIVE -eq 1 ]]; then
    run "find . -name '*.nbc' -o -name '*.nbi' -delete" \
        bash -c "find . \( -name '*.nbc' -o -name '*.nbi' \) -not -path './.git/*'"
else
    n=$(find . \( -name '*.nbc' -o -name '*.nbi' \) -not -path './.git/*' 2>/dev/null | wc -l)
    printf '  [keep] numba 缓存 %s 项（加 --aggressive 才清）\n' "$n"
fi

echo
echo "【C】临时诊断日志（outputs/_*.log，下划线前缀 = 一次性）"
run "rm -f outputs/_*.log" bash -c "ls outputs/_*.log 2>/dev/null"
if [[ $AGGRESSIVE -eq 0 ]]; then
    printf '  [note] **证据日志保留**（outputs/experiments/*.log 等）\n'
fi

echo
echo "【D】我这一轮的一次性修复脚本（_*.py / _*.txt）"
run "rm -f _*.py _*.txt" bash -c "ls _*.py _*.txt 2>/dev/null"

echo
echo "【E】后台任务残留（.tmp_* / *.tmp）"
run "rm -rf .tmp_* *.tmp" bash -c "ls -d .tmp_* *.tmp 2>/dev/null"

if [[ $AGGRESSIVE -eq 1 ]]; then
    echo
    echo "【F --aggressive】旧实验产物与测试产物"
    printf '  ⚠  outputs/experiments/ 是 **P系列结论的原始凭据**\n'
    printf '     outputs/test/ 192K — 确认无用才删\n'
    run "rm -rf outputs/test/*" bash -c "ls outputs/test/* 2>/dev/null"
    run "rm -rf outputs/numba_cache" bash -c "ls outputs/numba_cache 2>/dev/null"
fi

echo
echo "【保留清单（绝不动）】"
for p in outputs/models datasets outputs/experiments eval_corpus; do
    if [[ -e "$p" ]]; then
        printf '  ✓ %-24s %s\n' "$p" "$(du -sh "$p" 2>/dev/null | cut -f1)"
    fi
done
echo
echo "======================================================================"
if [[ $DRY -eq 1 ]]; then
    echo "预演结束。确认无误后加 --yes 执行。"
else
    echo "清理完成。建议跑一次门禁确认：python tests/run_tests.py fast"
fi
echo "======================================================================"