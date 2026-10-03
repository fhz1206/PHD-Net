"""P169 记忆：Rust 版 M1–M6 落地 + 那26× 的正确归因。"""
import io

NOTE = '''
## P169：Rust 版 M1–M6（`phdnet_rs/`）+ **「26×」的正确归因**
### ⚠① 先纠正一个可能的误判：Rust **不能**解决那26×
fhz：「我觉得你没有把昇腾 NPU 的算力用尽，把部分转移到 Rust」。
**「没用尽」判断是对的**（读出 7.83 ms是搬运下界 0.300 ms 的 **26.1×**），
**但 Rust 解决不了**，证据链：
| 项 | 数值 | 出处 |
|---|---|---|
| 裸算子提交耗时 | **0.35 ms**（Python 侧全部开销） | P113 |
| CANN launch 开销 | **~50–200 μs/算子** | 项目自己的注释 |
| 读出路径算子数 | **37 个** | 实测 grep |
| 37 × 50–200 μs | **1.9–7.4 ms 纯下发开销** | 计算 |
→ **那 26× 的主体是「算子逐个下发」**。Rust 把 37 次下发变成 37 次 **FFI**，
**下发次数不变**。
**对症的解法是 GE 图模式**（昇腾官方调优文档第一原则：「减少 Host 算子下发时间」；
`torch.compile(backend="ascend")`）。项目**已在用** `torch.compile`（`accel_readout.py:579`）
但只融合 4 个小 kernel，且是 `mode="default"`（cudagraphs 与「W 每步原地更新」冲突）。
### ② Rust 版的真实价值（4 条，都有依据）
1. **GE 图模式的载体** —— 图要一次构建+ 静态 shape + 复用；Python 的动态属性
   访问无法保证，Rust 能给**稳定 C ABI**。
2. **真nogil 多核** —— Python numba `prange` 在 x86 只快 **1.16×**（GIL+线程池），
   Rust `thread::scope` 没这个税。
3. **无 GC 停顿** —— 30b 档 908 MiB/step，任何 GC 停顿都是 ms 级。
4. **架构与 torch_npu 同构** —— 它的调用链是
   `PyTorch → ATen dispatch → torch_npu → ACL → GE/FE → Runtime → NPU`，
   本 crate 复刻**最底下那一段**（`acl.rs`，dlopen 零依赖）。
### ③ 交付内容（`phdnet_rs/`，Rust 1.97.1，**零第三方依赖**）
| 文件 | 内容 |
|---|---|
| `src/acl.rs` | ACL dlopen 绑定（`aclrtSetDevice/CreateStream/Malloc/Memcpy/GetDeviceCapability`）+ RAII 设备内存 + `Graph`（GE 图模式接口） |
| `src/mechanisms.rs` | M1 GEMV + k-WTA、M2 CSR SpMV、M3 STDP、M4a 工作记忆、M5 调制、M6 读出（稠密+稀疏） |
| `src/ffi.rs` | C ABI（ctypes 加载），全部返回 `i32`，**零拷贝** |
| `phdnet_rs.py` | Python 侧薄封装，`load()` 返回 `(lib, reason)` ——**回落可归因**（P161 纪律） |
| `tests/verifiers/verify_rust_kernels.py` | 对拍门禁 |
### ④ 门禁结果：**13/13**，其中 5 处**逐位一致（Δ=0）**
| 对拍 | 判据 | 结果 |
|---|---|---|
| Rust GEMV vs numpy BLAS | 容差 1e-5 | relerr **2.14e-06** |
| **Rust 8/16 线程 vs 1 线程** | **逐位** | ✅ |
| k-WTA 的 s（归一化） | **逐位** | Δ=0 |
| **CSR SpMV vs Python 标量升序** | **逐位** | Δ=0 |
| **M6 稀疏读出 vs Python 标量** | **逐位** | Δ=0 |
| M4a 衰减+门控 | **逐位** | Δ=0 |
| M5 gate | 容差 1e-6 | Δ=0 |
⚠ **idx 的顺序不保证一致**（Rust 按值降序、Python `argpartition` 无序）→
**集合相同即可**，上层最终 `np.sort`。已在门禁 B3 显式记录。
⚠ **M2 的 val 精度不同**：Python 是 float64，Rust 是 float32（生产精度）→不能逐位。
### ⑤ 编译期踩的坑（Rust 特有）
1. **`*mut f32` 不是 `Send`** → `thread::scope` 闭包不能捕获裸指针。
   解法：**以 `usize` 传递**，闭包内转回（比 `unsafe impl Send` 更明确）。
2. **`#[cfg]` 属性块在函数体里会破坏尾表达式** → Windows 分支下函数没有尾表达式
   → E0317。解法：给每个平台都写显式尾表达式。
3. **`PhantomData` + `unsafe impl Send/Sync`**：裸指针类型必须显式声明，
   且要在注释里写清安全性依据（分块互不重叠）。
4. **测试里字面量默认 `{float}`** → `assert!((y[0]-40.0).abs()<1e-6)` 编译失败，
   必须写 `40.0f32`。
### ⑥ 状态与下一步
- **本地**：编译 ✅ / 单测 5/5 ✅ / 对拍 13/13 ✅ / `has_npu=False`（正确）。
- **服务器**：需`cd phdnet_rs && cargo build --release`（本机产物是 .dll，服务器要 .so）。
- ⚠ **尚未实现**：GE 图编译（`Graph::compile` 如实返回 `None` 回落 eager，
  **不假装成功** —— 这是 P161 的教训）。真正落地需实现 `libgraph.so` 的 dlsym。
- ⚠ **未接入训练**：`model.py` 仍是纯 Python。接入前必须有 msprof 的
  `aiv_time` vs `cube_time` 分项，证明 Rust 版+图模式确实吃到了设备算力。
'''
with io.open(".workbuddy/memory/2026-10-01.md", "a", encoding="utf-8") as f:
    f.write(NOTE)
print("appended P169 note")