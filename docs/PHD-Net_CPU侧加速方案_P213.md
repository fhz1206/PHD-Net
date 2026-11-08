# PHD-Net CPU 侧加速方案（P213）

> **适用范围**：CPU 侧（M1–M5 跑在 numba/CPU，读出 M6 跑在 NPU）的时间预算与提速手段。
> **数据截止**：2026-11-01（P213）。
> **数据来源**：服务器日志 `train_1b_1b_pretrain_20261101-140007.log`
> （1b 档、昇腾 aarch64 + CANN 8.5 + NPU、`--remote-data`、vocab 51,962、
> `--readout-conn-k 128`、`--m2-kernel plain`（auto 选出）、24.5k token 稳态）。
> **相关文档**：[性能评估与迭代方案](PHD-Net_性能评估与迭代方案.md)（生产档性能唯一出处）、
> [并行与加速架构分析](PHD-Net_并行与加速架构分析.md)（为什么吃不满）、
> [现状速查](PHD-Net_现状速查.md)（当前事实基线）。
> 与代码冲突时**一律以代码为准**。

---

## 0. 一句话

服务器 1b 档当前 **step 10.8 ms/tok**，其中 **CPU 侧约 6.5 ms（60%）**、
NPU 读出 3.5 ms（32%）—— **CPU 是瓶颈**。本轮确认了三件事：

1. **CPU/NPU 重叠这条路已被 P192 排除**（训练步本就没有阻塞同步，
   `--readout-pipeline` 是 fail-fast 的死开关）→ 「CPU 赶不上 NPU」**不是同步问题，
   是纯算力问题**；
2. **`--numba-threads` 的服务器 A/B 从 P204 至今从未做过**（help 与性能文档都写着
   「服务器须 A/B」）→ 本轮交付可直接执行的 A/B 脚本
   （`tools/bench/bench_server_threads.py`）；
3. **CPU 侧最大的可优化项是 `M4b_ltm`（LTM imprint）**，其内部
   **64% 是 Python 逐行循环**（gather/scatter/decay_read_arr/touch），
   而核本身只占 28%。

---

## 1. 当前时间预算（服务器 1b 档稳态）

| 项 | ms/tok | 占比 | 归属 |
|---|---|---|---|
| **端到端 step** | **10.8** | 100% | |
| `M6_replay` | **1.5–1.8** | 14–17% | CPU（含一次 NPU 读出学习） |
| `M4b_ltm` | **0.9–2.2**（波动大） | 8–20% | **CPU 纯计算** |
| `M2_infer` | 0.65 | 6% | CPU |
| `M1_encode` | 0.30 | 3% | CPU |
| `PC_learn` / `FO_learn` | ~0.9（cum） | 8% | CPU |
| `other` 内未归因 | **~1.5** | 14% | CPU |
| **CPU 侧合计** | **~6.5** | **60%** | |
| NPU 读出（M6） | 3.5 | 32% | NPU |

⚠ `M4b_ltm` 的窗口值在 0.26–2.24 之间**剧烈波动**（取决于该窗口内 LTM 表的
生长/饱和程度）→ 单个窗口的数字**不可直接比较**，A/B 要看**中位**。

---

## 2. 「CPU 赶不上 NPU」不是同步问题（P192 已定案）

`docs/PHD-Net_CPU-NPU重叠执行计划.md` 顶部（P192 状态更新）已核对：

- 训练步的设备路径**本来就没有阻塞式同步** ——
  `phdnet/model.py` 走 `forward_dev`（返回**设备张量、零同步**），
  训练步把 `y` 置 `None`；`step()` 内唯一的显式 D2H 是 `nll_dev.item()`，
  被 `nll_sync_every=8` 门控；
- `--readout-pipeline` **已实现为开关但强制 fail-fast**（传了直接报错退出），
  理由是「额外流水尚无保持语义的实现或 NPU 收益实测，贸然前移学习会改变
  铁律③的逐 token 时序」；
- 证据门禁：`tests/gates/gate_cpu_npu_overlap_contract.py`（19/19）。

**结论**：端到端 10.8 ≈ CPU 6.5 + NPU 3.5 的**相加**，不是「等 NPU」。
CPU 慢 → NPU 在等 CPU 喂数据 → **CPU 侧提速会 1:1 变成端到端收益**。

---

## 3. `--numba-threads`：服务器 A/B 从未做过（本轮交付）

### 3.1 现状

| 项 | 值 | 出处 |
|---|---|---|
| CLI 默认 | **16** | `train/train.py`（P204 从 8 改成 16） |
| 上限逻辑 | `numba.set_num_threads(min(args.numba_threads, _cpu))` | `train.py:839` |
| 服务器实测生效 | **16**（cap 16） | 日志：`[parallel] numba prange threads = 16 (cap 16)` |
| 机器 | **191 核** | 日志：`[parallel] tokenisation thread (numba nogil)=153` |

**191 核的机器上，numba 只用 16 个线程** —— 看起来像浪费，但**不必然是**：

### 3.2 为什么「加线程」很可能不划算（先算账再测）

所有 CPU 侧 prange 核的**并行元素数**（本轮实测 `prange` 循环边界）：

| 核 | prange 元素数（1b 档） |
|---|---|
| `_csr_matvec` | n0/n1/n2 ≈ **341** |
| `_pc_infer_fused` | n1/n2 ≈ **341** |
| `_pc_fo_fused` | n1/n0 ≈ **341** |
| `_ltm_learn_rows` | R ≈ **256** |

**每个 prange 段只有 ~300 个元素。** 而 fork/join 屏障的固定成本随线程数上升
（P201 本机实测：4.7 → 24.7 → 64.3 µs/区）。

本轮在本机（8 核）实测**屏障成本**（341 元素、空核体）：

| 线程数 | 1 段 | 8 段 | 每段额外开销 |
|---|---|---|---|
| 1 | 1.20 µs | 5.70 µs | 0.64 µs |
| 4 | 3.50 µs | 18.40 µs | **2.13 µs** |
| 8 | 4.70 µs | 30.80 µs | **3.73 µs** |

→ **线程越多，每段固定开销越大**；而段的工作量只有 ~300 元素。
32 线程分 300 个元素 = 每线程仅 9–10 个 → 屏障成本极可能吃掉全部并行收益。

⚠ **这是本机（x86 8 核）的账，昇腾 191 核的数字会不同**（屏障更贵、并行度更高），
**所以结论必须由服务器实测给出** —— 而这正是 P204 遗留的空白。

### 3.3 A/B 脚本（新增）

```bash
# 服务器上直接跑
python tools/bench/bench_server_threads.py --threads 8,16,32

# 更细的曲线（含 1 线程基线）
python tools/bench/bench_server_threads.py --threads 1,2,4,8,16,32,64

# 与生产同参（1b 档 + 远程数据），跑久一点让稳态段出现
python tools/bench/bench_server_threads.py --threads 8,16,32 \
    --preset 1b --data pretrain --remote-data --tokens 20000
```

输出（末窗中位，自动排除 JIT/预热噪声）：

```
 threads  eff    ms/tok   vs 1st  readout(NPU)     CPU侧合计   final_ppl
       8    8      10.21   1.000×          3.51       6.70     13017.9
      16   16       9.88   0.968×          3.49       6.39     13017.9
      32   32      10.55   1.033×          3.53       7.02     13017.9
```

**判据**（与 README §4 一致）：
1. 端到端 `ms/tok`（主判据，看中位）；
2. `segments(win)` 的 CPU 侧各段（定位哪一段变快/变慢）；
3. ⚠ **PPL 只报不判**（训练流 sliding PPL 信噪比仅 0.40，<20% 的差异测不出来）。

⚠ **改默认前必须把表贴进 `docs/PHD-Net_性能评估与迭代方案.md`**（性能唯一出处）。

### 3.4 本机（8 核）预跑结果 —— 只证明「有最优点」，不构成昇腾证据

```
 threads  eff    ms/tok   vs 1st  readout(NPU)     CPU侧合计   final_ppl
       2    2     12.38   1.000×          8.85       3.53     14012.5
       4    4      9.07   0.733×          5.88       3.19     14012.5
       8    6     10.18   0.822×          5.51       4.67     14012.5
```

→ 2→4 线程快 **1.37×**，8 线程反而略慢（`eff=6`，被 `min(threads, cpu核数)` 截断）。
**说明线程数存在最优点**，但**本机数字不构成昇腾证据**（项目已有四次方向相反）。

---

## 4. CPU 侧真正的算法瓶颈：`M4b_ltm`（LTM imprint）

### 4.1 分解（本机 x86，1b 档真实形态：n_dim=1024 / m_out=72 / prev=cur=256）

生产走 **在线 CSR 表**（`--csr-online` 默认开）→ `OnlineCSRTable._learn_batch`
（批量核路径）。本轮实测 **table.learn 2.5 ms**（dict 版 11.3 ms，慢 4.5× → 生产确实走对了）。

| 部分 | 耗时 | 占比 | 性质 |
|---|---|---|---|
| gather 循环（256 行逐行切片） | 0.51 ms | 22% | **Python** |
| 核 `_ltm_learn_rows` | 0.64–0.80 ms | 28% | numba prange |
| scatter 循环（256 行逐行写回） | 0.54 ms | 23% | **Python** |
| 3× `_decay_read_arr` | 0.44 ms | 19% | **Python** |
| `_touch` ×2 | 0.15 ms | 6% | **Python** |

**Python 循环合计 1.64 ms = 66%** → 这就是 CPU 侧最值得下手的地方。

### 4.2 为什么不能简单向量化（已有的契约）

- `_decay_read_arr` **必须逐个调 `_decay_read`** ——
  `gate_ltm_trace_semantics` 的突变测试契约（P205）：门禁用
  `t._decay_read = lambda ...` 注入「未补偿」负向对照，**要求能被检出**。
  内联展开会让突变注入变成空操作 → **门禁检出能力静默消失**。
  （P205 实测：内联版触发 (a) 组 FAIL，`ratio_pre=1.0000`。）
- gather/scatter 的逐行循环**不能简单换成 `np.stack`**：行宽**不均**
  （实测本轮 256 行全是 72，但那是稳态；生长期行宽不同）。
  `np.stack` 版在满宽时逐位一致且快 0.56→0.46 ms，**但满宽不是普遍情况**
  → 需按行宽分支，慢路径未必更快。

### 4.3 已试过并**否决**的方向（记录以免重踩）

| 方案 | 结果 |
|---|---|
| `np.stack` 批量 gather | 满宽时 0.56→0.46 ms；**行宽不均时需分支**，收益不稳 |
| `np.copyto` 替代切片赋值 scatter | **0.54 → 0.69 ms（更慢）** —— numpy 调用开销大于拷贝本身 |
| 残差归一化（替代稳态缩放） | 与 homeostasis 语义冲突，且 RL 门禁仍 FAIL（见 P212 文档 §2） |
| `dict.fromkeys` → numba 编码 | P68 实测 **0.84× 负收益**，已回滚（残留死核本轮已删） |
| `np.unique` 替代 `dict.fromkeys` | P95b/P98 实测 94.3 µs **更慢** |

---

## 5. 其它 CPU 侧的既有事实（别重复踩）

| 项 | 结论 | 出处 |
|---|---|---|
| **M2 推理核** | 昇腾用 `plain`（`--m2-kernel auto` 自动选）；`fused` 在昇腾**慢 3–4×**（prange 屏障主导）；**x86 上 fused 快 1.34–2.05×**（P212 新增的按平台择优） | P76/P99/P113/P212 |
| `learn_predictive` | 保持**多核调用**（P59 实测融合版 0.91–1.36× **负收益**） | P59 |
| 读出前向 | 走 **BLAS**（实测比 numba 快，且逐位一致） | `readout.py` |
| fp8 转换 | 默认走 **torch**（P154 实测快 5.6×）；`nogil` 查表版本轮已接线（**3.06×**） | P154/P212 |
| `--readout-pipeline` | **死开关**（fail-fast）——P192 已定案不做 | P192 |
| `--enable-cpu-parallel` | **死配置**（config 有、CLI 有，但**代码里从未被读取**） | 本轮发现 |

---

## 6. 顺带修掉的两个环境问题（本轮）

### 6.1 `gate_cli_lang_semantics` 长期失败 —— 真路径缺陷

`eval_common` 实际在 **`tests/evals/`**，而门禁的 `sys.path` 只加了 `tests/`
→ `ModuleNotFoundError: No module named 'eval_common'`。
已修（加 `tests/evals`）→ **20/20 通过**。

### 6.2 另两个门禁的「失败」是**启动器问题**，不是代码问题

本机同时存在两个 Python：

| 命令 | 实际解释器 | numpy |
|---|---|---|
| `python` / `python3` | WorkBuddy 内置 **3.13.12** | ❌ 无 |
| `py`（Windows 启动器） | 按**文件关联**选 → 跑 `tests/gates/*.py` 时关联到 3.13 | ❌ 无 |
| `py -c "..."` | 按**默认解释器**选 → Python 3.14.5 | ✅ 2.5.2 |

⚠ **`py <脚本>` 与 `py -c` 走不同解释器** —— 这是所有「numpy 找不到」现象的根源。

已交付 `tools/run_gates.py`（统一门禁 runner）：
- 用 `phdnet/_python_exe.py::python_exe()` **实测**选一个能 `import numpy` 的解释器
  （含扫描 `%LOCALAPPDATA%\Programs\Python\Python*\python.exe` 等常见安装位置）；
- ⚠ **启动器（`py.exe`）只作兜底** —— 它的解释器选择依赖被运行文件的路径/关联，
  **无法可靠预判**（临时目录的 `.py` 用对解释器，仓库里的却用错的）；
- **性能敏感门禁串行跑**（`gate_cli_i18n` 含 200µs 级时限断言，并发下会假失败）。

**结果**：`python tools/run_gates.py --jobs 4` → **55/55 全绿**。

```bash
python tools/run_gates.py --jobs 4        # 全量（并发）
python tools/run_gates.py --show-failures  # 看失败详情
python tools/run_gates.py --match ltm      # 按关键字筛选
```

---

## 7. 建议的下一步（按性价比排序）

| # | 动作 | 预期收益 | 风险 | 成本 |
|---|---|---|---|---|
| 1 | **跑服务器线程 A/B**（§3.3） | 若 32 线程更快 → 端到端直接降；若有最优点 → 调到那个值 | 低（只改默认） | 一次跑 |
| 2 | **跑 `--numba-threads` + `--omp-threads` 组合 A/B** | `omp_threads` 默认 0=自动（=min(核数,32)），与 numba 线程**存在互相挤占** | 低 | 一次跑 |
| 3 | **M6_replay 摊销再评估** | `replay_every=16` → 32 可省 ~0.8ms/tok；但 P204 实测「replay 少反而略差」（P197 锚定有效）→ **必须用 probe_PPL 判**，不能只看速度 | 中 | 服务器 A/B |
| 4 | **M4b 的 Python 循环**（§4.1） | 理论上限 ~1.0ms/tok（66% 中的 gather/scatter 部分） | **高**（触及突变测试契约） | 大工程 |
| 5 | **启用 `--enable-cpu-parallel`** | ⚠ **死配置**（从未被读取）→ 要么接线要么删 | — | 先决策 |

⚠ **第 3、4 项都必须先有服务器 probe_PPL A/B 判据**（README §4：
训练流 sliding PPL 信噪比仅 0.40，<20% 的差异测不出来）。

---

## 8. 变更清单（P213）

| 文件 | 改动 |
|---|---|
| `tools/bench/bench_server_threads.py` | **新增**：服务器端到端 numba 线程 A/B（判据表 + JSON） |
| `tools/run_gates.py` | **新增**：统一门禁 runner（自动选解释器 / 性能敏感门禁串行 / JSON 明细） |
| `phdnet/_python_exe.py` | **新增**：解释器探测（实测 import numpy；启动器只作兜底） |
| `tests/gates/gate_cli_lang_semantics.py` | **修**：`sys.path` 加 `tests/evals`（`eval_common` 的真实位置）→ 20/20 |
| `docs/PHD-Net_CPU侧加速方案_P213.md` | **新增**：本文 |
