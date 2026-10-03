#!/usr/bin/env bash
# P170：构建 Rust 库（cdylib），供 Python 通过 ctypes 加载。
#
# # 架构约定（fhz 2026-10-03）
# # ------------------------------------------------------------------
# # **Python 是顶层**：编排（step顺序、状态连续、门禁判据、回落决策）全在Python。
# # **Rust 是算子库**：只暴露 `extern "C"` 的无状态数值核 + ACL 句柄。
# # → Rust 侧**不允许**出现 `model.step` 这类编排逻辑；
# #   **Python 侧不允许**出现 `for` 循环算数值（那是 Rust 的活）。
# #
# # 为什么要这么分：
# #  · 编排需要**动态性**（门禁/开关/回落/日志）→ Python 的表达力够
# #  · 算子需要**零开销 + nogil 多核** → Rust 的裸指针 + thread::scope 够
# #  · 且**可独立对拍**：Rust 不依赖 Python 语义 → 门禁能逐位验证（见verify_rust_kernels.py）
#
# 用法：
#   bash phdnet_rs/build.sh              # release 构建
#   bash phdnet_rs/build.sh --debug      # debug 构建
#   bash phdnet_rs/build.sh --test# 跑 Rust 自带单测
set -euo pipefail

cd "$(dirname "$0")"
MODE="release"
RUN_TEST=0
for a in "$@"; do
    case "$a" in
        --debug) MODE="debug" ;;
        --test)RUN_TEST=1 ;;
    esac
done

echo "[build] crate = $(basename "$PWD")  profile = $MODE"
if ! command -v cargo >/dev/null 2>&1; then
    echo "[build] ❌ 找不到 cargo。安装 Rust：https://rustup.rs/" >&2
    exit 1
fi
echo "[build] cargo = $(cargo --version)"
echo "[build] rustc = $(rustc --version)"

if [[ $MODE == "release" ]]; then
    cargo build --release
    PROFILE_DIR="target/release"
else
    cargo build
    PROFILE_DIR="target/debug"
fi

# 产物名（Linux=lib*.so / macOS=*.dylib / Windows=*.dll）
SO=""
for cand in "libphdnet_rs.so" "phdnet_rs.dll" "libphdnet_rs.dylib"; do
    if [[ -f "$PROFILE_DIR/$cand" ]]; then SO="$PROFILE_DIR/$cand"; break; fi
done
if [[ -z "$SO" ]]; then
    echo "[build] ❌ 未找到 cdylib 产物（$PROFILE_DIR）" >&2
    exit 1
fi
SIZE=$(du -h "$SO" | cut -f1)
echo "[build] ✅ 产物：$SO（$SIZE）"

# Python 侧是否能找到它（把load() 的真实结果打出来，避免「编译了但加载不到」）
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    PY=""
fi
if [[ -n "$PY" ]]; then
    echo "[build] 用 $PY 验证加载……"
    $PY - <<'PYEOF' || { echo "[build] ❌ Python 加载失败" >&2; exit 1; }
import sys
sys.path.insert(0, ".")
from phdnet_rs import load
k, why = load()
if k is None:
    print("[build] ❌", why); raise SystemExit(1)
print("[build] ✅ Python 加载成功；has_npu =", k.has_npu(),
      "（False 是正确的：CPU 参照路径必须可用）")
PYEOF
else
    echo "[build] ⚠未找到 python，跳过加载验证"
fi

if [[ $RUN_TEST -eq 1 ]]; then
    echo "[build]跑 Rust 单测……"
    if [[ $MODE == "release" ]]; then cargo test --release; else cargo test; fi
fi

echo "[build] 完成。下一步：python tests/verifiers/verify_rust_kernels.py（对拍门禁）"