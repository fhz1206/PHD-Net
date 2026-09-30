# PHD-Net 扩展指南

> 面向要**扩展 PHD-Net** 的开发者：加机制、加后端、加精度、加数据通路、加训练阶段。
> 每节给出「改哪里 → 怎么验证 → 门禁在哪」，并标注 2026-09-29~30 实战中反复踩到的坑（标 ⚠）。
> 配套：`BUGS.md`（缺陷台账，本文所有 ⚠ 都能在那里查到完整症状与根因）、
> `docs/PHD-Net_硬件后端适配报告.md`（后端矩阵与昇腾真机踩坑）。

## 0. 五分钟速查表

| 你要做的事 | 改哪个文件 | 必过哪个门禁 |
|---|---|---|
| 加一个新机制（如 M7） | `phdnet/<新模块>.py` + `phdnet/model.py::step` 接入 | fast 9/9 + **新写逐位对拍**（§2.3） |
| 加一个新设备后端 | `phdnet/backends/` + `resolve_accel_device()` | `verify_accel_readout.py` + `verify_accel_readout_p55.py`（2×2 矩阵） |
| 加一种 dtype | **三处**：`accel_readout.py::_DT` + `sparse_encoder.py::resolve_model_dtype` + `model.py::to_numpy` | 对拍 + `verify_ms_stream.py`（检查点往返） |
| 加一个数据源 | `phdnet/corpus.py`（统一接口）+ `corpus_stream.py` | `verify_stream_tokenize.py`、`verify_ms_stream.py` |
| 加一个训练阶段 | `train/train.py`（复用 `lm.net.step`） | fast 9/9 + 阶段语义对拍 |
| 改一个 numba 核 | `phdnet/ltm_kernel.py` 等 | 对拍（**逐位优先**）+ `verify_ltm_kernels.py` |
| 动读出 / 精度 | `phdnet/backends/accel_readout.py` | ⚠ **必须同步 `_unsupported_reason()` 能力表**（§3.3） |
| 调一个已有开关 | 见 §6 速查表 | 对应 verifier |

**架构铁律（不可违反）**：① 禁自注意力 / 位置编码 / 堆叠层；② 每个机制必须有神经认知对应物；
③ 逐 token 语义（状态跨 token/shard/epoch 连续演化，`W` 每步原地更新 → **不允许批处理**）；
④ 行为变更以 config 开关承载且**默认关闭**，唯一例外是 `csr_online`/`k_sparse`（项目特例，default-on）。

## 1. 代码地图与 M1–M6 接入点

```
phdnet/
  model.py           PHDNet.step —— 唯一的机制编排入口（六段 + 状态演化）；to_numpy 在文件末尾
  sparse_encoder.py  M1 稀疏分布式编码（k-WTA）；resolve_model_dtype / _gemv_rows
  sparse_pc.py       M2 预测编码主干（CSR 稀疏 + 可选融合核）
  plasticity.py      M3 STDP 侧向连接
  wm.py              M4a 工作记忆（PFC 漏整合）
  bigltm.py          M4b 大容量事件驱动表（适配器；存储在 sparse_table.py）
  readout.py         M6 读出（numba CPU 原路径）
  sparse_table.py    M4b 存储本体（dict 版 + 在线 CSR 版）
  ltm_kernel.py      M4b 的 numba 核（learn 批量 / recall 投影）
  telemetry.py       设备/GC 遥测；i18n.py 终端语言
  backends/          加速后端（torch / 多设备 / 读出 accel）+ README 有踩坑表
train/
  train.py           唯一生产训练入口（argparse 全部开关都在这里）
  tokenizer_core.py  分词器热路径（CSR-trie + numba nogil）
  corpus_stream.py   流式数据（PrefetchChars 多进程 + StreamingTokenizer）
  ckpt_1b.py         检查点（同步 save_model + 异步 save_model_async）
tests/
  run_tests.py fast  零回归门槛（9 项）
  verifiers/         18 个专项对拍（逐位 / 容差各自有约定）
```

**M1–M6 接入点对照**：M1 `SparseEncoder.encode` ｜M2 `SparsePC.step`（+ `learn` 侧融合核）｜
M3 `Plasticity.step` ｜M4a `WM.step` ｜M4b `SparseLTM.{encode,recall,learn}` + `OnlineCSRTable`｜
M6 `Readout`（CPU）/ `AccelReadout`（设备）。**新机制一律加在 `PHDNet.step` 里**，按 M1→M6 顺序调用。

## 2. 加一个新机制

### 2.1 接入模板
```python
# 1) 写模块：phdnet/<your>.py
class YourMechanism:
    def __init__(self, cfg, rng): ...
    def step(self, cache: dict, *, learn: bool) -> dict:   # 不跨步持有设备张量
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

### 2.2 四条硬性要求
1. **不做设备同步**：`step` 里不 `.item()` / `.cpu()` / `.numpy()`。需要标量时走
   「设备侧累积 + 每 N 步取一次」（照 `accel_readout._nll_sum` 写，默认 `nll_sync_every=8`）。
2. **跨 token 状态**：跨步持有的东西只能是 numpy/设备张量，**不要缓存设备张量的 numpy 视图**
   （`accel_readout._cache_h` 用 `np.ascontiguousarray` 拷贝）。异步交接前必须 `.copy()` ——
   `to_numpy` 返回的是**共享视图**，`addmm_`/`learn` 会原地写它，后台写盘线程会序列化到撕裂数据。
3. **逐位优先**：能用相同表达式就不重排（BLAS 归约顺序、prange 分块都会改 1–2 ulp）。
   若必须改，**在 verifier 里写明容差口径与原因**。
   ⚠ **逐位对拍的基线必须是同一库、同一 fastmath 口径的串行实现** —— 用纯 Python 当基线会产生
   满屏假阳性，而假阳性比没测试更危险（它会让人去「修」正确代码）。
4. **numba 核必须 `cache=True`**：`@njit(cache=True, nogil=True, ...)`，否则每次启动重编译。
   `inline="always"` 与 `cache` 互斥（`readout.py` 里 7 个内联 helper 是设计上的例外）。
   另：⚠ **别让计算落在计时缝隙里**（曾有一段 30 µs 的代码漏在段外，浪费一轮排查）。

### 2.3 对拍必须覆盖的六种情形
写 `tests/verifiers/verify_<your>_equiv.py`，逐位或容差对拍，且必须覆盖：

| 维度 | 必覆盖 | 为什么 |
|---|---|---|
| 精度 | **fp64 / int8 量化**（或你的机制涉及的各档） | 量化路径的舍入顺序与浮点路径不同 |
| 开关 | **关键开关两态**（on/off） | 开关分支常年只测默认值 |
| 序列 | **多步序列**（≥6 步，含状态增长/衰减） | ⚠ P67 的首版就是**只在单步逐位、多步不一致** |
| 规模 | **满规模行/槽**（不是小表） | 核的固定开销、容量边界只在满规模暴露 |
| 键序 | **键顺序打乱** | ⚠ 按 append 顺序填键会让槽位查找总命中首项，把优化**测成负收益** |
| 形状/顺序 | 若改了 CSR 段序或索引布局 | 段序错位**纯计时看不出来**，结果仍「跑得动」 |

## 3. 加一种 dtype（最容易出事的地方）

### 3.1 ⚠ dtype 必须**两端同时**对齐
2026-09-30 连续踩了两次「混合 dtype 脱离 BLAS」：fp32 权重 @ fp64 输入（慢 5.9×，`c8148b4`）、
fp64 权重 @ fp32 输入（昇腾慢约 70×，`77f9b78`）。两次的成因一模一样：**各自修对了一侧，没人对齐两端**。
```python
# 错：只转一边
u = self._w_fp32() @ np.asarray(x, dtype=np.float32)
# 对：输入跟随权重
u = self._w_fp32() @ np.asarray(x, dtype=self._w_fp32().dtype)
```
**自检**：任何 `A @ b` 前断言 `A.dtype == b.dtype`。⚠ 混合 dtype **不报错、结果正确，只是慢** ——
它在所有正确性门禁下都是隐形的，只能靠断言或 review 抓。

### 3.2 三处必须同步
| 位置 | 内容 |
|---|---|
| `phdnet/backends/accel_readout.py::_DT` | dtype 名 → torch dtype（⚠ P92 后只有 `{fp32, fp16, bf16}`；fp8/fp4 连构造都不允许） |
| `phdnet/sparse_encoder.py::resolve_model_dtype` | 存储 dtype（含 bf16 缺 `ml_dtypes` 时的回落） |
| `phdnet/model.py::to_numpy` | **检查点**：numpy 没有 bf16/fp8 → 必须走**位模式**（`view(uint16/uint8)`）并与加载侧 `ckpt_1b.py` 的 `view()` 解码闭环 |

⚠ **torch 缺失是常态**：`phdnet/model.py` **不 import torch**（只有 accel 读出路径才有张量）。
任何用到 torch 的代码都要**惰性 import**，否则纯 numpy 训练会 NameError（P83b，`4666870`）。

### 3.3 ⚠ 加 dtype 必须同步**能力表**
`accel_readout.py::_unsupported_reason()` 是一张「加速读出未实现项」黑名单。
P84 把 fp8 实现出来并设为默认，**却忘了从这张表里删掉它** → 生产日志一直是
`numba-cpu(回落)`，**一整轮改造没生效且不报任何错**（BUGS B3，本台账最严重一条）。

**规则**：加 dtype 时同步 `_DT` + `_unsupported_reason()` + docstring；并在 verifier 里加一条
「声明支持 ⇒ `_unsupported_reason()` 返回 None」的断言。

### 3.4 ⚠ 选默认精度的判据不是「更省访存」
bf16 半 ULP ≈ 2e-4 是非目标行更新 `|dp| ≈ 1e-6` 的约 200 倍 → 更新被舍入，训练**退化为纯
Hebbian**（PPL 震荡不降的根因）。fp8 forward + fp16 更新买到的是**学习精度**，不是带宽
（更新侧反而让总访存 +42%）。⚠ 这类「数值语义退化」**不崩、门禁不报警**，比崩溃难查得多。

## 4. 加一个设备后端

```python
# phdnet/backends/resolve_accel_device(spec) —— 优先级：昇腾 → ROCm → CUDA → … → CPU
```
1. **`--accel auto` 必须安全回落**到 numba CPU，并把原因记在 `readout._accel_fallback_reason`，
   启动日志打印（P19 纪律：回落 + 记原因，不静默）。**以启动日志的实际行为为准，不以「探测到设备」为准。**
2. **对齐完整调用面**（P23 教训）：`__call__` / `learn` / `learn_softmax` / `W` / `n_synapses` /
   `conn_k` / `hidden` / `stats()` / `W_cpu` / `load_W` —— 只实现主方法会在生产第一个 step 崩。
3. **未实现的配置要 fail-fast**：`readout_hidden>0`（两级读出）、`readout_conn_k>0`（稀疏读出）
   在加速后端未实现，显式回落并记原因，绝不「能跑但语义不同」。跨后端行为分叉比报错更危险。
4. **`torch.compile` 是惰性的**：构造期不编译、**首次调用**才编译 → 失败要捕获并**永久回落 +
   告警**（P58）。`--torch-compile` 默认关。
5. **能力探测要用真实算子**：`torch.float8_e4m3fn` 在 CPU 上**能分配但不能乘** —— 只测创建会得到
   假通过（P85）。要测 matmul。
6. **失败要在使用点兜底**：探测是乐观的、使用点兜底是悲观的，两手都要有。旧机器上的旧代码
   不会自动获得新防护（P90：直接崩在 `W.to(float8_e4m3fn)`）。
7. 新后端要有 `verify_<backend>.py`，覆盖 **eager/compile × target/target_idx** 的 2×2 矩阵
   （P55 的崩就藏在 target_idx × compiled 这一格）。

## 5. 数据源与训练阶段

- **数据源**：实现 `phdnet/corpus.py` 的统一接口（txt / parquet / jsonl 一套），在 `corpus_stream.py`
  挂上；远程源走 `ms://`（`phdnet/ms_stream.py`，HTTP Range seekable 流）。若涉及分词动
  `tokenizer_core.py` 热路径，必须保持 `nogil` + 线程安全（P13/P18）。
  ⚠ 走远程时把预取进程数**封顶**（`REMOTE_MAX_PROCS=4`）：8 个进程各持独立 aiohttp 池会被限流。
- **SFT 双标记状态机**：用 `assistant_marker` 分段，**prompt 段 `learn=False` 只推进状态**。
  ⚠ 只取「最后一个 marker 之后」会**漏掉 99.8% 的步** —— 必须是双标记（段内逐 token 判定）。
- **RL（REINFORCE）**：直接复用读出感知器（`η ← rl_lr × advantage`），**零新增算子**。
  ⚠ 奖励稀疏 → 优势恒 0 → 无学习，入口必须有诊断（否则「RL 不学」无法与「RL 写错了」区分）。
- **两阶段训练**：预训练（zh+en）→ SFT，走 `--remote-fraction` 分片前缀采样。
  ⚠ `--lang` 过滤**依赖 parquet 的 `lang` 列**，过滤是否真生效看日志里的 `[sample lang=…]`。

## 6. 已有扩展点速查（CLI）

| 开关 | 默认 | 说明 |
|---|---|---|
| `--readout-dtype {fp32,fp16,bf16,fp8,fp4}` | **fp16** | ⚠ fp8/fp4 在**加速后端全禁**（P92）；`--accel cpu --readout-dtype fp8` 仍可用（位算法量化核） |
| `--readout-conn-k K` | 0 | 读出 CSR 稀疏化（>0 时加速后端未实现 → fail-fast 回落） |
| `--m2-kernel {fused,plain}` | **plain** | 昇腾上 fused 退化 3–4×；两者对拍 1–2 ulp（fused 用 fastmath，不逐位） |
| `--encoder-dtype {fp32,fp64,fp16,bf16}` | **fp64** | M1 权重存储；aarch64 走自写 numba GEMV，x86 走 BLAS |
| `--accel {auto,npu,cuda,rocm,cpu}` | auto | 读出后端；auto 优先级 昇腾 → ROCm → CUDA → DirectML → CPU |
| `--nll-sync-every N` | **8** | nll 设备侧累积周期（消除每步 `.item()` 同步；PPL 统计滞后 N 步） |
| `--numba-threads N` | **8** | numba prange 上限（0=用 numba 默认=全部核） |
| `--omp-proc-bind` / `--no-omp-proc-bind` | **on** | 只设 `OMP_PROC_BIND=close`；⚠ **不要设 `OMP_PLACES`**（191 核上反噬 8×，P74 已回滚） |
| `--torch-compile` | **off** | 读出热路径融合（inductor 在部分平台不稳；惰性编译，失败永久回落） |
| `--devices auto` | — | 多卡：读出列并行 + host 线程收敛 |
| `--lang {all,zh,en}` | all | 按 parquet `lang` 列过滤训练流（`[sample lang=…]` 日志自检） |
| `--remote-data --remote-fraction F` | off / 0.3 | 直读 ModelScope（HTTP Range 流式，零落盘） |
| `--step-profiling` | off | 九段 + 主循环三段耗时分解 |
| `--ckpt-every N` | 50000 | 异步保存（主线程快照 `.copy()` + 后台写盘） |
| `--fp8-refresh N` | 8 | ⚠ 仅 numba CPU 路径有意义（加速后端已禁 fp8） |

## 7. 提交前检查单（⚠ 全是实战踩出来的）

- [ ] `python tests/run_tests.py fast` = 9/9
- [ ] 改动文件 `py_compile`；改 `train/*.py` 另跑 `python train/train.py --help`
      （⚠ fast 门禁**从不 import** 训练入口；argparse help 里的裸 `%` 会让 `--help` 崩）
- [ ] **断言补丁真的落盘**：`Edit`/字符串 replace 后 `grep` 断言新文本存在
      —— ⚠ `str.replace` **无匹配是静默成功**，`print('patched')` 照常打印（本组已连踩 4 次）
- [ ] **「不确定就从 git 历史逐字取回」** —— ⚠ 凭记忆重写历史实现是错的（P75 的 plain 版）
- [ ] **dtype 两端对齐**（§3.1）；新 dtype 已加 `_DT` + `resolve_model_dtype` + `to_numpy` 位模式
      **+ 加载侧解码 + 能力表**（§3.2/§3.3）
- [ ] **可选依赖边界**：用到 torch 的分支在「缺 torch」时仍可导入（惰性 import）
- [ ] **平台假设**：不要把 x86 的性能结论当昇腾证据 —— 已实测三个方向相反的案例（§8）
- [ ] **计时口径**：满规模 + 键顺序打乱 + best-of-N；⚠ 本机与目标机差一个数量级时本机结论作废
- [ ] **异步/并发路径**：交给后台线程的数据先 `.copy()`（`to_numpy` 是共享视图，会被撕裂）
- [ ] **诊断代码的成本**：诊断也在热路径上（曾有一次 `np.diff` 每次物化 134 MB 只为打一行数）；
      门槛要考虑**触发频率**（条件触发的事件按「每 1000 次」永远打不出来）
- [ ] **numba 核有 `cache=True`**；启动日志的 `[numba] cache dir/size` 确认真被命中
- [ ] **门禁红了要归因**，不要当成「已知噪声」放着（一条 NLL 容差断言长期红着没人问它何时开始红）
- [ ] `BUGS.md` 记一条（症状→根因→修复→门禁缺口）
- [ ] 提交前清 `__pycache__` / `*.nbc,*.nbi` / 临时日志

## 8. 性能定位速查（先测再改）

```bash
python train/train.py … --step-profiling      # 九段 + loop: 三段
python tools/bench_local.py --compare             # 本机回归基准（不产出文档数字）
python tools/accel_doctor.py                      # 设备探针 + 试分配 + 一次前向
```
- **段之和 ≠ 总耗时** → 先查「主循环三段」和**落在计时缝隙里的代码**（P72 就是为此加的）。
- **单段随训练变长** → 该段的数据结构在长；用 `[ltm-diag]` 看规模。
- **「只用 1 核」** → 多半是纯 Python 热点（GIL）。考虑 numba 化，但 ⚠ **小规模上 gather+prange
  常是负收益**（实测 0.96×、0.84×、0.91×、0.73× 四次）：**先算核的固定开销，再在服务器规模上测**。
- **「NPU 闲逛」** → 看 `nll_sync_every`、是否有隐式 `.item()` / pageable H2D。
- **「读出占比 < 50%」** → 瓶颈已转移到 CPU 侧（PC 栈 / STDP / `big_ltm` 随机访问），继续优化读出精度收益有限。
- ⚠ **x86 与昇腾方向相反的三个案例**：M1 fp32（x86 快 1.80× / 昇腾慢约 70×）、M2 融合核
  （x86 快 1.15–2.16× / 昇腾慢 3–8×）、`OMP_PLACES=cores`（x86 无感 / 昇腾慢 8×）。
  → 任何优化都要留一个 A/B 开关，让目标机器上一个 flag 就能复测。
