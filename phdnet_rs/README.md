# PHD-Net Rust 算子库

**Python 是顶层，Rust 是算子库。**本 crate 只提供**无状态数值核**与
**CANN/ACL 后端**，不含任何编排逻辑。

---

## 安装 Rust

Linux / macOS（服务器）：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

安装后**重开一个 shell**（或 `source "$HOME/.cargo/env"`），然后确认：

```bash
cargo --version
rustc --version
```

### 国内网络镜像（可选）

`sh.rustup.rs` 拉不动时用清华镜像：

```bash
RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup \
RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup \
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Windows（开发机）

访问 <https://rustup.rs> 下载 `rustup-init.exe`，或用 winget：

```bash
winget install Rustlang.Rustup
```

⚠ 本项目**零第三方依赖**，所以不需要 `protobuf-compiler`、`cmake` 之类。

### 最低版本

Rust **1.75+**（用到 `div_ceil`、`c_char` in `core::ffi`）。
实测环境：**1.97.1**。

---

## 构建

```bash
cd phdnet_rs
bash build.sh            # release 构建 + 验证 Python 能加载
bash build.sh --debug    # debug 构建
bash build.sh --test     # 附带跑 Rust 自带单测（须串行，见下）
```

产物（`target/release/`）：

| 平台 | 文件名 |
|---|---|
| Linux | `libphdnet_rs.so` |
| macOS | `libphdnet_rs.dylib` |
| Windows | `phdnet_rs.dll` |

### ⚠ Rust 单测必须串行

```bash
cargo test --release -- --test-threads=1
```

原因：单测**共享全局线程池**（`OnceLock`）与**共享计数器** → 并发跑会互相污染。
生产代码里 `Pool::run` 是**同步的**（派发 → 等完成），所以生产用是安全的；
只有「测试并发调用 run」才有这个问题。

---

## 对拍门禁

```bash
python tests/verifiers/verify_rust_kernels.py
```

⚠ **对拍通过之前不启用 Rust 版**（Python 版是参照实现）。

---

## 架构边界

```text
Python（顶层）                      Rust（本 crate）
─────────────────────                ──────────────────────────
model.step 的编排顺序               extern "C" 的无状态数值核
状态跨 token/shard/epoch 连续       ACL 句柄 / 设备内存（RAII）
门禁判据、开关、回落决策GE 图模式的载体
日志（segments(win) 等）            ── 不允许出现编排逻辑 ──
--nll-sync-every 等调度
```

**硬规则**：

- **Rust 侧不许有编排** —— 没有 `step()`、没有「下一步做什么」的决策、没有日志。
- **Python 侧不许有数值循环** —— `for i in range(n): y[i] = ...` 属于 Rust。
  这是**性能边界**，不是风格偏好。

---

## 现状与诚实边界

### CPU 侧：已跑通，对拍 13/13 + SIMD 3条

| 路径 | M1 GEMV（1024×2048 fp32，本机 x86） |
|---|---|
| Rust 标量 1 线程 | 3485 µs |
| Rust 标量 8 线程（常驻池） | 1302 µs |
| **Rust SIMD 8 线程** | **241 µs**（14.5× vs 标量） |
| numpy BLAS | 214 µs（SIMD 参照） |
| numba `serial` 1 线程 | 2744 µs |

**两个实测结论**：

1. Rust 与 numba 的**单线程**代码质量**相同**（1.0×）—— LLVM 对两者都**没有**
   自动向量化这个循环（`acc += ...` 有依赖链）。所以「Rust 写得差」是错的猜测。
2. **手写 AVX2 SIMD 后**：8 线程 241 µs，与 BLAS 的 214 µs **只差 12%**。

### ⚠ 三条限制

1. **NPU 计算尚未接通**。`acl.rs` 只绑了 13 个 ACL **资源管理**函数
   （设备/流/内存/事件/能力探测），**零个算子下发**。`Graph::compile`
   如实返回 `None` 回落 eager —— 不假装成功。
2. **Rust 不碰设备内存**。`torch_npu` 没有「从外部 ACL 指针包装成 Tensor」
   的公开API（CUDA 有 `from_blob`，NPU 侧无等价物）→ 外部分配的内存
   **无法**交给 torch 用。所以设备内存归torch_npu，Rust 只做CPU 算子。
3. **SIMD 路径与标量路径不逐位**（4 路FMA 累加器改变求和顺序，fp32 ~1e-7）
   → 门禁对 SIMD 用**容差 1e-5**，对标量仍要求**逐位**。

---

## 环境变量

| 变量 | 作用 |
|---|---|
| `PHDNET_RS_POOL=0` | 关闭常驻线程池（回落单线程），用于 A/B 与排障 |
| `PHDNET_ACL_LIB` | 指定 `libascendcl.so` 路径（优先于 `ASCEND_HOME_PATH`） |
| `PHDNET_FORCE_ACL=1` | 即使本机无 CANN 也尝试 dlsym（会失败，但失败有意义） |

---

## 文件

| 文件 | 内容 |
|---|---|
| `src/acl.rs` | ACL dlopen 绑定（13 个符号）+ RAII 设备内存 + `Graph`（GE 接口占位） |
| `src/mechanisms.rs` | M1 GEMV/k-WTA、M2 CSR SpMV、M3 STDP、M4a 工作记忆、M5 调制、M6 读出 |
| `src/pool.rs` | 常驻线程池（修「每步新建线程」的开销） |
| `src/simd.rs` | 手写 AVX2 + 4 路 FMA GEMV |
| `src/ffi.rs` | C ABI（`ctypes` 加载），全部返回 `i32`，零拷贝 |
| `phdnet_rs.py` | ctypes 薄封装，`load()` 返回 `(lib, reason)` —— **回落可归因** |
| `top.py` | **唯一**接入点：暴露 `kernels` / `why_unavailable` / `describe()` |
| `build.sh` | 构建 + 验证加载 + 可选单测 |