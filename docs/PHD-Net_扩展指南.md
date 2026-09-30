# PHD-Net 扩展指南

> 面向要**扩展 PHD-Net** 的开发者：加机制、加后端、加精度、加数据通路、加训练阶段。
> 每节都给出「改哪里 → 怎么验证 → 门禁在哪」，并标注 2026-09-29~30 实战中
> 反复踩到的坑（标 ⚠）。
> 配套：`BUGS.md`（缺陷台账）、`docs/BUGS.md#结构性教训汇总`（过程与数据）。

## 0. 五分钟先读

| 你要做的事 | 改哪里 | 必过门禁 |
|---|---|---|
| 加一个新机制（如 M7） | `phdnet/` 新模块 + `phdnet/model.py::step` 接入 | fast 9/9 + 新增逐位对拍 |
| 加一个新设备后端 | `phdnet/backends/` + `resolve_device()` | `verify_accel_readout*.py` |
| 加一种 dtype | `_DT`（accel_readout）+ `resolve_model_dtype`（sparse_encoder） | **dtype 两端都要改**（⚠ 见 §3.1） |
| 加一个数据源 | `phdnet/corpus.py`（统一接口）+ `corpus_stream.py` | `verify_stream_tokenize.py` |
| 加一个训练阶段 | `train_1b/train.py`（复用 `lm.net.step`） | fast 9/9 + 阶段语义对拍 |
| 调一个已有开关 | 见 §6 速查表 | 对应 verifier |

**架构铁律（不可违反）**：① 禁自注意力 / 位置编码 / 堆叠层；② 每个机制必须有神经
认知对应物；③ 逐 token 语义（状态跨 token/shard/epoch 连续演化，`W` 每步原地更新
→ **不允许批处理**）；④ 行为变更以 config 开关承载且**默认关闭**，唯一例外是
`csr_online`/`k_sparse`（项目特例，default-on）。

## 1. 代码地图

```
phdnet/
  model.py          PHDNet.step —— 唯一的机制编排入口（六段 + 状态演化）
  sparse_encoder.py M1 稀疏分布式编码（k-WTA）
  sparse_pc.py      M2 预测编码主干（CSR 稀疏 + 可选融合核）
  plasticity.py     M3 STDP 侧向连接
  wm.py             M4a 工作记忆（PFC 漏整合）
  bigltm.py         M4b 大容量事件驱动表（适配器；存储在 sparse_table.py）
  readout.py        M6 读出（numba CPU 原路径）
  sparse_table.py   M4b 存储本体（dict 版 + 在线 CSR 版）
  ltm_kernel.py     M4b 的 numba 核（learn 批量 / recall 投影）
  backends/         加速后端（torch / 多设备 / 读出 accel）+ README 有踩坑表
train_1b/
  train.py          唯一生产训练入口（双轨：生产 torch 栈 = tools/）
  tokenizer_core.py 分词器热路径（CSR-trie + numba nogil）
  corpus_stream.py  流式数据（PrefetchChars 多进程 + StreamingTokenizer）
  ckpt_1b.py        检查点（同步 + 异步写盘）
tests/
  run_tests.py fast 零回归门槛（9 项）
  verifiers/        专项对拍（逐位/容差各自有约定）
```

## 2. 加一个新机制（最常见）

### 2.1 接入点：`PHDNet.step`
```python
# 1) 写模块：phdnet/<your>.py
class YourMechanism:
    def __init__(self, cfg, rng): ...
    def step(self, cache: dict, *, learn: bool) -> dict:   # 纯函数式：不跨步持有张量
        ...
    def learn(self, cache: dict) -> None: ...
```
```python
# 2) 接入 model.py：构造期建实例，step 里按顺序调用
self.your = YourMechanism(cfg, rng)
...
_p = self._prof_t('M7_your')        # ① 分段计时（--step-profiling 可见）
ctx = self.your.step(cache, learn=learn)
self._prof_end('M7_your', _p)
```

### 2.2 硬性要求
1. **`step` 里不做设备同步**：不 `.item()` / `.cpu()` / `.numpy()`。需要标量时
   走「设备侧累积 + 每 N 步取一次」（照 `accel_readout._nll_sum` 写）。
2. **状态演化跨 token 连续**：跨步持有的东西只能是 numpy/设备张量，**不要缓存
   设备张量的 numpy 视图**（P80 的 `_cache_h` 用 `np.ascontiguousarray` 拷贝）。
3. **逐位一致优先**：能用相同表达式就不用重排（BLAS 归约顺序、prange 分块都会
   改 1–2 ulp）。若必须改，**在 verifier 里写明容差口径与原因**。
4. **numba 核必须 `cache=True`**（`@njit(cache=True, nogil=True, ...)`），
   否则每次启动重编译；`inline="always"` 与 `cache` 互斥。
5. **分段计时**：`self._prof_t/_prof_end` 包住，这样服务器上 `--step-profiling`
   能直接看到你的段耗时。⚠ 别让计算落在计时缝隙里（曾有一段 30 µs 的代码漏在
   段外，浪费了一轮排查）。

### 2.3 门禁
```bash
python tests/run_tests.py fast                    # 必须 9/9
python tests/verifiers/<your>_equiv.py           # 新写：逐位对拍
```
对拍脚本要覆盖：**fp64 / fp16(int8 量化) / 关键开关两态** + **多步序列**（P67
就是在多步上失败的）+ **满规模行/槽** + **键顺序打乱**（否则槽位查找总命中首项，
把优化测成负收益）。

## 3. 加一种 dtype（最容易出事的地方）

### 3.1 ⚠ dtype 必须**两端同时**改
2026-09-30 连续踩了三次「混合 dtype 脱离 BLAS」：fp32 权重 @ fp64 输入（慢 5.9×）、
fp64 权重 @ fp32 输入（昇腾慢 70×）。规则：
```python
# 错：只转一边
u = self._w_fp32() @ np.asarray(x, dtype=np.float32)
# 对：输入跟随权重
u = self._w_fp32() @ np.asarray(x, dtype=self._w_fp32().dtype)
```
**自检**：任何 `A @ b` 前断言 `A.dtype == b.dtype`（至少在门禁里断言）。

### 3.2 三处要同步
| 位置 | 内容 |
|---|---|
| `phdnet/backends/accel_readout.py::_DT` | dtype 名 → torch dtype |
| `phdnet/sparse_encoder.py::resolve_model_dtype` | 存储 dtype（含 bf16/fp8 不可用时的回落） |
| `phdnet/model.py::to_numpy` | **检查点**：`numpy` 没有 bf16/fp8 → 必须走位模式（`view(uint16/uint8)`）并与加载侧 `ckpt_1b.py` 的 `view()` 解码闭环 |

⚠ **torch 缺失是常态**：`phdnet/model.py` **不 import torch**（只有 accel 读出路径
才有张量）。任何用到 torch 的代码都要惰性 import，否则纯 numpy 训练会 NameError
（P83b 实战）。

## 4. 加一个设备后端

```python
# phdnet/backends/resolve_accel_device(spec) —— 优先级：昇腾 → ROCm → CUDA → … → CPU
```
要求：
1. **`--accel auto` 必须安全回落**到 numba CPU，并把原因记在
   `readout._accel_fallback_reason`（P19 纪律：回落 + 记原因，不静默）。
2. **对齐完整调用面**（P23 教训）：`__call__` / `learn` / `learn_softmax` / `W` /
   `n_synapses` / `conn_k` / `hidden` / `stats`——只实现主方法会在生产第一个 step 崩。
3. **未实现的配置要显式拒绝**（如 `readout_hidden>0`、两级读出），不要「能跑但语义
   不同」；跨后端行为分叉比报错更危险。
4. **torch.compile 是惰性的**：构造期 `torch.compile()` 不编译，**首次调用**才编译
   → 失败要捕获并**永久回落 + 告警**（P58）。
5. 新后端要有 `verify_<backend>.py`，覆盖 eager/compile × target/target_idx 等组合。

## 5. 加数据源 / 训练阶段

- **数据源**：实现 `phdnet/corpus.py` 的统一接口（txt / parquet / jsonl 一套），
  在 `corpus_stream.py` 挂上；若涉及分词动 `tokenizer_core.py` 的热路径，必须
  保持 `nogil` + 线程安全（P13/P18）。
- **SFT**：用 `assistant_marker` 做**双标记状态机**（prompt 段 `learn=False` 只推进
  状态；只取「最后一个 marker 之后」会漏 99.8% 的步）。
- **RL**：REINFORCE 直接复用读出感知器（`η ← rl_lr × advantage`），零新增算子；
  注意奖励稀疏 → 优势恒 0 → 无学习，入口要有诊断。

## 6. 已有扩展点速查（CLI）

| 开关 | 默认 | 说明 |
|---|---|---|
| `--readout-dtype {fp32,fp16,bf16,fp8,fp4}` | **fp8** | fp8 = forward 用 fp8 副本 + 更新用 fp16 主副本（保住微小更新）；非昇腾/CUDA 回落 fp16 |
| `--fp8-refresh N` | 8 | fp8 forward 副本的重建间隔（步） |
| `--readout-conn-k K` | 0 | 读出 CSR 稀疏化（>0 时每输出单元 K 条入边，PPL 会涨） |
| `--m2-kernel {fused,plain}` | **plain** | M2 融合核 vs 原始多核调用（昇腾上 fused 退化 3–4×） |
| `--encoder-dtype {fp32,fp64,fp16,bf16}` | **fp64** | M1 权重存储；aarch64 走自写 numba GEMV，x86 走 BLAS |
| `--accel {auto,npu,cuda,rocm,cpu}` | auto | 读出后端 |
| `--nll-sync-every N` | 8 | nll 设备侧累积周期（消除每步 `.item()` 同步） |
| `--numba-threads N` / `--omp-proc-bind` | 8 / on | 线程治理（⚠ 不要设 `OMP_PLACES`，超多核机器反噬） |
| `--torch-compile` | **off** | 读出热路径融合（inductor 在部分平台不稳） |
| `--lang {all,zh,en}` | all | 按 parquet `lang` 列过滤训练流（`[sample lang=…]` 日志自检） |
| `--remote-data --remote-fraction F` | off | 直读 ModelScope（HTTP Range 流式，零落盘） |
| `--step-profiling` | off | 九段 + 主循环三段耗时分解 |
| `--ckpt-every N` | 50000 | 异步保存（主线程快照 + 后台写盘） |

## 7. 提交前检查单（⚠ 都是实战踩出来的）

- [ ] `python tests/run_tests.py fast` = 9/9
- [ ] 改动文件 `py_compile`；`train_1b/*.py` 另跑 `python train_1b/train.py --help`
      （argparse 的裸 `%` 会让 `--help` 崩）
- [ ] **改动真的落盘**：`Edit`/字符串 replace 后 `grep` 断言新文本存在
      （`str.replace` 无匹配是**静默成功**；P25/P31/P84 连续踩）
- [ ] **两端 dtype 对齐**（§3.1）；新 dtype 已加 `to_numpy` 位模式 + 加载侧解码
- [ ] **可选依赖边界**：用到 torch/np 的分支在「缺该依赖」时仍可导入
- [ ] **平台假设**：不要把 x86 的性能结论当证据——昇腾上 prange/fastmath/BLAS
      行为都可能相反（P76/P77/P74）
- [ ] **计时口径**：满规模 + 键顺序打乱 + best-of-N（否则测出最优而非平均）
- [ ] **异步/并发路径**：快照数据要 `.copy()`（共享视图会被后台线程序列化时撕裂）
- [ ] `BUGS.md` 记一条（症状→根因→修复→门禁缺口）
- [ ] 提交前清 `__pycache__` / `*.nbc,*.nbi` / 临时日志

## 8. 性能定位速查（先测再改）

```bash
python train_1b/train.py … --step-profiling      # 九段 + loop: 三段
```
- 段之和 ≠ 总耗时 → 先查「主循环三段」和**落在计时缝隙里的代码**；
- 单段随训练变长 → 该段的数据结构在长（用 `[ltm-diag]` 之类诊断看规模）；
- 「只用 1 核」→ 多半是纯 Python 热点（GIL），考虑 numba 化（gather+prange
  在**小规模**常是负收益，先在**服务器规模**上测）；
- 「NPU 闲逛」→ 看 `nll_sync_every`、是否有隐式 `.item()`/pageable H2D。
