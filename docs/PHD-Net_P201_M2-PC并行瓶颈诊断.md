# P201 诊断：M2_infer 与 PC_learn 的并行瓶颈 + NPU 假占用

> **测量机器**：本机 = x86 Windows 10.0.26200、8 逻辑核、Python 3.13.14、numba 0.66.0、torch-CPU（**无 NPU**）。
> **被测尺寸**：1b 档 = `n0=n1=n2=1024`、`k=conn_k=128`、`nnz=524,288`、`val` fp32、`idx` int64（`SparsePCStack.__init__`，`phdnet/sparse_pc.py`）。
> **⚠ 纪律声明（项目铁律）**：本报告所有**性能**数字均为 **x86 本机实测**。服务器是昇腾 191/152 核 aarch64，本项目已有**四次方向相反**的教训（`phdnet/sparse_pc.py` L55、`docs/PHD-Net_性能评估与迭代方案.md` §"x86≠昇腾"）。**任何优化在昇腾生效前必须复测**。NPU 部分（问题 4）本机无设备，**全部基于代码分析 + 项目已有服务器 msprof 记录**，不是本机实测。
> **本报告的现状**：结论已按当前源码复核（`--m2-kernel auto`、`--numba-threads 16`、`--cython-kernels auto`）。诊断提出的改动中：建议 1、4 **已落地**，建议 2 **部分落地**，建议 3、5、6 **仍未做**（见 §6 各条的「现状」）。

---

## 0. 一句话结论

**numba 线程数的 8 线程上限不是真瓶颈；服务器 token 500 那行的 `M2_infer 19.59 / PC_learn 15.28` 主要是 `segments(cum)` 口径污染（numba JIT 冷编译 + LTM 生长期）的产物，不是稳态并行瓶颈。** 可优化项按「收益 × 风险」排序见 §6。

---

## 1. 问题 1：numba 线程封顶到底影响多大？

### 1.1 答案

**在 1b 尺寸（1024 行 × 128 边 = 131,072 边/矩阵）下，线程数 2~8 已到平台；8 → 16/32 无收益甚至负收益。** 本机 x86 数据**不支持**「8 线程封顶是 M2_infer/PC_learn 的真瓶颈」。

### 1.2 本机实测（x86，min-of-40-rounds，n_steps=1）

| numba 线程 | 1 | 2 | 4 | 8 | 16 | 32 |
|---:|---:|---:|---:|---:|---:|---:|
| infer+learn 加速比（min-of-40） | 1.00× | 1.87× | 1.78× | **2.08×** | 1.19× ↓ | 0.92× ↓ |
| 同上（绝对最小口径，压外部负载噪声） | 1.00× | 1.98× | 1.98× | **2.40×** | 1.39× ↓ | 1.10× ↓ |

- `_csr_matvec`：114.8 µs(1T) → 62.8 µs(8T) → 100.0(16T) / 162.7 µs(32T)。
- **8 线程 = 本机平台点**；16/32 线程在两种口径下都掉速。

### 1.3 证据链（为什么 8 就够）

1. **带宽饱和**：单次 `_csr_matvec` 遍历 131,072 条边、访存 ≈ **1.59 MiB**；本机 8 核实测有效带宽仅 **5.3 GB/s（1T）→ 6.1 GB/s（8T）** → 加线程只增加对同一份 `val`/`idx` 的争抢；核内 1→6 线程仅 **1.16×**。
2. **fork/join 屏障超线性**：8 个极短 prange 区的每区边际成本 **1T 0.74 → 8T 4.70 → 16T 24.74 → 32T 64.29 µs**（16/32 是 8T 的 5~14 倍）。`_pc_infer_fused` 内 8 个 prange 区 → 8T 下 infer 屏障 ≈ 38 µs（占 0.515 ms 的 **7%**），16T 仅屏障 ≈ 198 µs —— **这就是 16/32 反而更慢的直接机制**（昇腾 191 核上只会更强）。
3. **项目内同族证据**：`phdnet/sparse_pc.py` 记 P123 实测「本机 x86 8 核 1B 档尺寸，有/无 nogil 并发场景慢 14%」；`docs/PHD-Net_并行与加速架构分析.md` 记「同一条 CSR 行级 prange，在 x86 上无可测差别、在昇腾上赢 5.41× —— **x86 连「无差别」都不足以推断昇腾**」。
4. **去掉并行不是灾难**：单核 `_pc_infer_fused_serial` 本机 1.12~1.42 ms，与 8T 的 0.515 ms 只差 ~2× → 「去掉并行」在本机不是灾难；昇腾上 `fused` 曾比 `plain` 慢 3-4×，那是**平台相关**，不能从本机推。

### 1.4 判读

- **8 线程封顶在本机不是瓶颈**：8 → 2/4 已拿到 1.8~2.4×，8 是平台起点；8 → 16 反而掉到 1.2×。
- **但「本机 8 线程就够」不构成「昇腾 8 线程就够」的证据。** 昇腾 191 核 + 更高总带宽，`_csr_matvec` 的带宽饱和点可能出现在更高线程数。**要确认必须做服务器 A/B**（§6 建议 3）：判据是 `segments(win)` 的 `M2_infer` + 端到端 ms/tok + CS/s。
- ⚠ **本机 ≠ 昇腾**（本项目已有**四次方向相反**）—— 见 §8。

---

## 2. 问题 2：负载均衡与 learn 里的串行段

### 2.1 行并行本身是均衡的 + 短路边界是死分支

| 矩阵 | 行宽 min/mean/max | 行宽 std/mean |
|---|---|---|
| `up0`/`up1`（1024×128） | 128 / 128.0 / 128 | **0.0%** |
| `dn0`/`dn1`（转置） | 92~95 / 128.0 / 158~167 | 8.0~8.4% |

`_random_csr`（`phdnet/sparse_pc.py:243`）构造的 `up0/up1` 是**严格均匀行宽**（`indptr = arange(0, (n+1)*k, k)`）；`dn0/dn1` 由 `_transpose_csr`（同文件 L335）生成，行宽是泊松涨落（std/mean 8%）。**8% 的行宽不均对 8 线程 prange 完全不是问题**（每线程 ~128 行，统计涨落 <1%）。
稀疏性上：`e0`/`e1`/`r1`/`r2` **100% 非零**（`r1/r2` 是 tanh 输出恒非零），只有 `s0`（M1 k-WTA 输出，k_sparse=128/1024）为 12.5% 非零 → `_csr_add_outer` / `_csr_oja_up` 里的 `if ai == 0.0: continue` 短路**永不触发 → 是死分支**，负载均衡无损失。

### 2.2 `learn` 里的所有非并行段（源码逐行）

`SparsePCStack.learn(homeostasis=True)`（生产 1b 走的正是这条路：`train/config_1b.py:112` `homeostasis=True` → `phdnet/model.py:745` `pc.learn(..., homeostasis=cfg.homeostasis)` → `phdnet/sparse_pc.py:1065`）**P201 诊断时的调用分解**：

| # | 调用 | 落到核 | 并行？ |
|---|---|---|---|
| 1-2 | `_add_outer(dn0/e0/r1, dn1/e1/r2, eta)` | `_csr_add_outer` | prange |
| 3-4 | `_oja(up0/r1/s0, up1/r2/r1)` | `_csr_oja_up` | prange |
| 5-8 | `_clip(up0/up1/dn0/dn1, w_max)` | `_csr_clip` | **❌ 串行**（当时无 prange） |
| 9-12 | `_scale_rows(up0/up1/dn0/dn1, _hn)` | `_csr_scale_rows` | prange（含 sqrt） |

**⇒ 诊断结论：生产路径 `PC_learn` = 12 次独立 numba 核调用/步，其中 4 次是完全串行的 O(nnz) 全扫描。** `_csr_clip` 的线程缩放实测（1b 尺寸，131,072 个 fp32）：1T 34.5 µs → 8T 26.3 µs = **1.31×**；同尺寸有 prange 的核是 `_csr_add_outer` **3.96×**、`_csr_scale_rows` **3.21×**、`_csr_oja_up` **2.69×**、`_csr_matvec` 1.59× → **`_csr_clip` 是 learn 里唯一真正不并行的段**（1.31× 只是内存子系统的自然差异）。

> **现状：已修**（见 §6 建议 1）—— `_csr_clip` 加了 prange+`nogil`，并新增 `_pc_learn_fused_homeo` 把 homeostasis 路径的 12 步融为 1 次核调用。
> ⚠ `_csr_scale_rows` 的行范数归一化是 fp64 sqrt + fp64 除法（`s ** 0.5`），在 prange 内、不改归约顺序，实测 1T/8T = 3.21×，不是瓶颈。

### 2.3 learn 的流量账（1b 尺寸）

融合核的 8 个 prange 区各扫 131,072 边 → 合计 **1,048,576 边 = nnz × 2.00**；其中 **4 段 clip 占 50% 的边遍历量**。生产路径（当时 12 次独立核调用）里 clip 4 次串行 ≈ 4 × 35 µs ≈ **140 µs**，而整个 learn ≈ 1.5 ms → 约 **9%**。

---

## 3. 问题 3：`_pc_infer_fused` 内部结构

### 3.1 `n_steps` 循环在核内

`phdnet/sparse_pc.py:379` 的 `for _ in range(n_steps):` 位于 `@njit(cache=True, nogil=True, parallel=True, fastmath=True)` 的 `_pc_infer_fused`（同文件 L357）**函数体内** → 循环在编译后的机器码里，**`n_steps` 增加不产生任何 Python 层调用开销**。非融合路径（`SparsePCStack.infer`，L1053）的循环在 **Python 层** → 每步 6 次 `_csr_matvec` 调用。

### 3.2 每次 infer 的 SpMV 次数（n_steps=1）

逐段数得 **8 次行扫描（SpMV）**：步前 `up0@s0`/`up1@r1`，步内 `dn1@r2`/`up1@e1`/`dn0@r1`/`up0@e0`，尾部 `dn0@r1`/`dn1@r2`；每次 131,072 边，**合计 1,048,576 = nnz × 2.00**。
⇒ **每次 `infer` = 8 次 SpMV**（融合核内共 8 个 prange 区；`_pc_learn_fused` 也是 8 个）。

### 3.3 有没有可合并的冗余计算？

**逐段分析：无冗余计算可合并。**
- **步内 `dn0@r1` vs 尾部 `dn0@r1`**：**不是纯冗余**。步内用**更新前**的 `r1`，尾部用**更新后**的 `r1`（`r1[i] = tanh(r1[i] + 0.15*d1[i])` 之后）—— 语义上尾部是「最终 `r1` 的自由能」，是 PC 的正确返回定义（门禁 `gate_m2_infer_kernels.py` C4 验 `e0 = s0 − dn0·r1`）。**不能合并。**
- **`up0@s0` vs `up0@e0`**、`up1@r1` vs `up1@e1`、`dn1@r2`（步内）vs `dn1@r2`（尾部）：输入不同 → 不同 SpMV，不能合并。

**⇒ 真正的结构性浪费在别处（见 §3.4：`idx` 是 int64，占访存 67%）。**

### 3.4 访存账（1b 尺寸，每次 infer）

| 项 | 字节 |
|---|---:|
| `val`（fp32，1,048,576 条） | 4.19 MiB |
| `idx`（**int64**，1,048,576 条） | **8.39 MiB** |
| 中间向量（~8 个 1024 fp64） | 0.20 MiB |
| **合计** | **12.78 MiB / infer** |

> ⚠ **`idx` 占流量的 67%，比 `val`（fp32）还大 2×。** `idx` 改 int32 可省 4.19 MiB（约 33% 总流量）。这与 `docs/PHD-Net_CPU-NPU重叠执行计划.md` §5.4 的外部 AI 建议同向，但那条已记录两个坑：① **值不能降到 bf16**（P110）；② **昇腾 gather 对 uint16 支持需实测**。落地现状见 §6 建议 2 与 §12。

### 3.5 融合核的净收益（本机 x86）

- 8 次独立 `_csr_matvec` + numpy `tanh`/`clip` = 1493.2 µs；`_pc_infer_fused`（1 次调用）= 1100.9 µs → **1.36×**（其中 numpy `tanh`/`clip` 占独立调用的 **32%**）。
- ⚠ 融合核**内部仍是 8 个 prange 区**（fork/join 一个没少），收益全部来自「Python 层调用次数 ↓ + 中间数组不落 numpy」。
- **⇒ 本机 1.36× 不能外推昇腾**：昇腾上 fused 曾比 plain 慢 3-4×。生产默认现为 `--m2-kernel auto`（按平台择优：昇腾 plain / x86 fused），**不应据本机数据改**。

---

## 4. 问题 4：NPU 20-50% 占用的可能原因

> ⚠ **本机无 NPU（torch-CPU），本节全部基于代码分析 + 项目已有服务器 msprof/日志记录。不是本机实测。**

### 4.1 先纠正一个前提：服务器日志记的是 **AI Core 99%**，不是 20-50%

`docs/PHD-Net_性能评估与迭代方案.md` §2.3：「NPU AI Core% = **99%**（满载。⚠ 这是瞬时值，**不等于带宽用满**）」。
`docs/PHD-Net_CPU-NPU重叠执行计划.md` §4：「CS/s 稳态只有 ~90 万而 NPU 显示 99%」。

**所以「NPU 20-50%」这个数字在本项目已有记录里没有出处。** 若新日志确实显示 20-50%，它更可能是下面 (c) 的**平均口径**（`npu-util` 是采样平均，含等 CPU 的空窗）。

### 4.2 读出路的结构（代码事实）

`phdnet/model.py:594-660` 的生产热路径：

```
→ h = _build_h(r2, pred_feat, fused)          # CPU numpy 拼接
→ y_dev = readout.forward_dev(h)              # NPU：H2D(pinned,async) + gather + Σ
→ nll   = readout.learn_softmax(h, target, eta, y_pre=y_dev, target_idx=...)
                                            # NPU：softmax + nll + rank-1 addmm_
→ nll_sync_every = 8                          # → 8 步才一次 .item() 硬同步
```

关键设计（`phdnet/backends/accel_readout.py`）：`forward_dev`（L1283）返回**设备张量、零同步**，H2D 走 pinned 暂存 + `non_blocking`（`_staged_to_dev` L2381）；`learn_softmax`（L1577）复用 `forward_dev` 缓存的 `ht`/`y`（`_lookup_ht` L2007），**不重复 H2D**；`nll_sync_every=8`（`phdnet/config.py:540`）让 nll 累积到设备标量、**每 8 步才 `.item()` 一次**；`correct` 在主机侧算、**省一次 `.item()`**。P205a 的 `--beta p2` 双流已实现：H2D 走独立流 `_h2d_stream`、nll 链走 `_nll_stream`（L595-606）→ CPU 只等 nll 链、不等主流排空。

**⇒ 代码里每步的显式同步点已被压到 1/8 步一次。**

### 4.3 三个候选原因，按证据强度排序

#### (a) ⚠⚠ 最可能：**等 CPU（每步 2 ms 计算 + 等待）—— 但已被 `nll_sync_every=8` + `forward_dev` 部分缓解**

证据：`docs/PHD-Net_CPU-NPU重叠执行计划.md` §1 记的实测吻合 —— 「`4.26 + 7.83 = 12.09 ms`，与日志端到端**逐位一致** → 确认是完全串行，没有任何隐含的重叠」。即：CPU 侧 4.26 ms（M1-M4b + PC_learn）+ NPU 侧 7.83 ms，**串行相加**。同文档的核对结论补充：「训练步的设备路径**本来就没有阻塞式同步**……若真完全串行，成因是**运行时下发/队列背压**（与 msprof 的 `aclnnIndex` 89.3% / `aicore_time=0` / `wait` 6.8× 同源），**不是**代码里的显式同步 —— 要证实需服务器 msprof，**本机无法验证**」。

⇒ **判读：这是最可能的主因，但已被 `nll_sync_every=8` + `forward_dev` 缓解过一轮。若 20-50% 是新的退步，第一嫌疑是「CPU 侧某段变长 → NPU 等更久」。**

#### (b) ⚠ kernel launch 开销占比高 —— **有 msprof 实测支持**

`docs/PHD-Net_并行与加速架构分析.md` 记「读出 7.83 ms 是带宽下界（0.300 ms）的 **26.1 倍**」；`phdnet/config.py:505-508`（P145）记：

> msprof（昇腾 600 步）实测：`Index` 占 **81.0%**，其中 **aiv_time = 99.6%** → 它在 **Vector 单元**上**真执行**（不是排队），50.8 MiB 只跑出 **8.1 GB/s**；而同流量的 `BatchMatMul` 在 **Cube** 上达 **38 GB/s**（**快 4.7×**）。

→⇒ **同一份访存量，Vector 单元效率比 Cube 低 4.7×。** `Index`（= `_sp_gather` 的 `ht[Wi]`，`accel_readout.py:1012`）跑在 Vector 上，这就是「AI Core 99% 但有效吞吐低」的机制：**单元忙、带宽没用满**。另 `phdnet/config.py:550` 记：「CANN 上 launch ~50-200µs/kernel，每步 5 个 → 纯 launch 开销约 0.5 ms/step」；稀疏路径 `torch_compile` 默认 **False**（`:566`，P58 指令「图优化关了吧」）→ 逐算子 eager → 每步 softmax/log/sub/addmm_ 各自一个 kernel。

#### (c) `torch_compile` 默认关 = 稀疏路径永远 eager

`phdnet/config.py:566` `torch_compile: bool = False`（P58 指令保持不变），`train/train.py:909` `cfg.torch_compile = args.torch_compile`。
⇒ **稀疏读出（生产默认 `--readout-conn-k 128`）在 P58 下永远走 eager 逐算子路径**。`accel_readout.py:519` 明确记录：P58 当时「稀疏路径根本用不上融合核（构造期被排除）」，后已放开且「本机实测融合与 eager **逐位相同**」，但**默认仍是 False**。

### 4.4 判断与证据

**判断：20-50% 更可能是 (a) 等 CPU + (b) Vector 单元低效的叠加，而不是 (c) launch 开销单独造成。**

1. **launch 开销的量级不够**：项目记录「每步 5 个 kernel × 50-200 µs ≈ 0.5 ms/step」。即使全免，也只是读出 7.83 ms 的 6%。**不足以把 99% 拖到 20-50%。**
2. **等 CPU 的量级够**：CPU 侧 4.26 ms vs NPU 7.83 ms，若完全串行则 NPU 有 4.26/12.09 = **35% 的时间在等 CPU** —— **与「20-50%」量级吻合**。这是唯一一个量级对得上的解释。
3. **Vector/Cube 效率差 4.7× 是「利用率数字失真」的机制**：`aiv_time=99.6%` 说明 Vector 真在算，但 `aicore_time=0` + `wait` 6.8× 说明有大量等待。**NPU 利用率的口径（AI Core vs Vector vs aicore）本身就是混淆源**，引用时必须说清是哪个计数器。

**要在服务器上证实的判据**（本机无法验证）：
- `--nll-sync-every=1` vs `8` A/B：若两者读出耗时相同 → 确认无隐藏同步；若 1 明显更慢 → 还有同步点（§6 建议 6）。
- msprof 对比 `aicore_time` / `aiv_time` / `wait` 三项占比。
- 若确认是等 CPU → 唯一有语义保持空间的路径是 `docs/PHD-Net_CPU-NPU重叠执行计划.md` 方案 C（稀疏前向改 CSR SpMM，重评 `--torch-compile`），**收益未测，需服务器 A/B**。

---

## 5. 问题 5：numba 线程与预取进程的争用

### 5.1 核数账（服务器口径，cpu=152）

| 资源 | 取值 | 源码 |
|---|---:|---|
| `auto_workers()` | **121** | `train/vocab_parallel.py:77-79` `cpu*8//10` |
| numba prange 线程 | **16** | `train/train.py:839` `min(args.numba_threads, _cpu)`，`--numba-threads` 默认 **16**（L496） |
| BLAS/OpenMP 线程 | **8** | `train/train.py:58` `min(8, cpu)` |
| 预取进程数 | **8** | `train/corpus_stream.py:180` `min(PREFETCH_MAX_PROCS=8, cpu, len(files))` |
| 每进程 pyarrow 解码线程 | **15** | `train/corpus_stream.py:188` `max(1, cpu//w)` = 121//8 |
| pyarrow io 线程 | **7** | `train/corpus_stream.py:68` `decode_threads//2` |
| 分词线程池 | **121**（head 模式） | `train/train.py:675` `auto_workers()` |

**合计潜在并行 ≈ 16 + 8 + 8×15 + 7 + 121 ≈ 272 线程竞争 152 核。**

### 5.2 它们是否真的互相争抢？

**答：会，但争的不是同一类资源，且主循环里真正与 numba 同时跑的只有预取解码。**

| 消费者 | 时间窗 | 资源类型 |
|---|---|---|
| 分词线程池（121） | **启动期**（`scan_vocab_parallel`）+ 主循环每步 `_next_tok()` | CPU（nogil 真多核） |
| 预取进程（8）+ pyarrow 解码（8×15） | **全程流式**（与主循环重叠） | CPU 带宽 + 内存带宽 |
| numba 16 线程 | **主循环内** M1-M4b/PC_learn 段 | CPU + 内存带宽 |
| BLAS 8 线程 | 主循环内 M1 GEMV（`sgemv`） | CPU + 内存带宽 |

关键机制（`train/corpus_stream.py:212-224` 注释）：「生产者线程的 pyarrow 解码释放 GIL，故并发度不损失」→ 预取解码是**真并行**的后台负载，与 numba 线程**时间上重叠、带宽上争抢**。日志证据（`train/train.py:756` 打印）：`[parallel] tokenisation thread (numba nogil)=121 | prefetch processes=8 | decode cores/process≈15, total≈120`。

### 5.3 「剩下的核在干什么？」

**答：在跑预取解码（120 个解码线程）+ 分词线程池（启动期），不是在等锁、也不是全空闲。**
`docs/PHD-Net_性能评估与迭代方案.md` §2.3 记录：进程实际用核 **1.1 / 191**、CPU 系统占用 **1-2%**、上下文切换 CS/s **~0.9 M**（早期曾 11.2 M）。
**⇒ 「进程用核 1.1」+「CPU 1-2%」+「CS/s 0.9M」三者合起来说明：numba 线程确实没把核用满（这是 §1 的带宽饱和），而预取/分词的并行度虽高但受 I/O 与 GIL 释放点限制，也没把 152 核铺满。**

⚠ **反直觉点**（性能文档原文）：「`M2_infer` 5.41× 的提速**没有**把进程用核数从 1.1 抬到 8 或更多。原因是 `--numba-threads` 与 CPU 侧其余段的 1.16× 上界 —— 并行度上去了，但 CPU 侧本来就只有 4.26 ms，抬到 0.6% 之外的核数**总量不变**。」

### 5.4 有没有值得做的调整？

1. **`--omp-wait passive`**（`train/train.py:430`）：本机实测 `OMP_WAIT_POLICY=PASSIVE` 让进程 CPU 时间 −21%（自旋确实少了）但**墙钟 +90%**（小任务上休眠/唤醒成本远高于自旋）。→ **191 核 + 每步数毫秒的大规模环境里结论可能相反，应由实测决定。** 本机数据是**负收益**，不能据本机打开。
2. **`OMP_PLACES=cores` 已撤**（`train/train.py:437`/L777-779）：服务器实测带上它 `M2_infer` 2.4 → 19-32 ms/tok（涨 8-13×）。**这是本项目「线程绑核反而更慢」的实测教训，不要再试。**
---

## 6. 优化建议（按「预期收益 × 风险」排序）

> ⚠⚠ **全部收益数字都是本机 x86 实测或代码结构推算。昇腾必须复测（本项目已有四次方向相反）。**
> 「是否改变数值语义」是本报告最重要的一栏 —— **任何非「逐位」的项都要走探针 PPL A/B（P146 纪律）。**
> **现状列**已按当前源码复核。

### 建议 1：让 `_csr_clip` 走 prange（或让生产路径重新走融合核）—— ✅ **已落地（P202）**

- **影响段**：`PC_learn`。
- **已落地内容**：`_csr_clip` 加 `prange`+`nogil`（`phdnet/sparse_pc.py:100`），**逐位不变**（逐元素 clamp 无归约顺序问题，与融合核内已有的 prange clip 同构）；新增 `_pc_learn_fused_homeo`（L545）把 homeostasis 路径的 12 步（add_outer×2 + oja×2 + clip×4 + scale_rows×4）融为 **1 次核调用** → **12→1 次核调用 + 消掉 4 次串行 O(nnz)**，本机 1b 档实测 **1.34×**。
- **数值语义**：**逐位不变**（`gate_parallel_consistency.py` 的判据就是这条；四组合对拍全部逐位一致）。
- **风险**：**低**。注意 `_pc_learn_fused_homeo` 覆盖 12 个操作（原 `_pc_learn_fused` 只覆盖 8 个），`homeostasis=False` 的库默认路径也需逐位对拍（已补）。
- **前置**：`eta_oja=0`（>0 会正反馈发散）→ `_csr_oja_up` 是死代码路径，融合它无风险。

### 建议 2：`idx` 从 int64 压到 int32 —— ⚠ **部分落地（P211）**

- **影响段**：`M2_infer`（主要）、`PC_learn`。
- **已落地部分**：`phdnet/sparse_pc.py:306` 的 `_compress_csr_idx` 已实现（放不下时**原样返回**，不静默截断），并在**稀疏读出**路径调用（`phdnet/readout.py:632`）：1B 档读出 CSR 内存 **79.8 → 53.2 MB**、前向 **1.29×**、更新 **1.30×**、`learn_softmax` **1.18×**，且 matvec/add_outer 与 int64 **逐位相同**（门禁 `gate_cpu_opt_p211.py` A 例）。
- **未落地部分**：**M2/PC 栈自己的 4 个 CSR（`up0/up1/dn0/dn1`）仍是 int64** → §3.4 的「`idx` 占 67%、改 int32 省 4.19 MiB（≈33% 总流量）」对 M2 侧**依然成立**，收益尚未兑现。
- **数值语义**：**逐位不变**（int32 足够表示 n ≤ 2³¹，本项目 n ≤ 4096；索引值不变，只改容器宽度）。
- **两个坑**：① **值不能降到 bf16**（P110）；② **昇腾 gather 对 uint16 支持需实测**（可能触发类型转换反而更慢）。
- **风险**：**中** —— M2/PC 侧要动 `_random_csr`/`_transpose_csr`/`_from_dense_csr` 三个构造函数 + 5 个 numba 核 + **Cython 签名硬编码 int64**（`phdnet/_cykernels.pyx:148/173/202`）+ ckpt CSR 形状校验（`train/ckpt_1b.py:615`）。**P186 的 u16 压缩实现已随 `phdnet_rs/` 于 P188 整体删除**（当前唯一存在的压缩实现就是上面那个 int32 的 `_compress_csr_idx`）。
- **前置**：必须重跑 `tests/gates/gate_parallel_consistency.py` + `gate_csr_vectorization_perf.py`；ckpt 兼容性要单独设计。

### 建议 3：服务器 A/B `--numba-threads` 8 / 16 / 32 —— ❌ **仍未做**

- **影响段**：`M2_infer`、`PC_learn`、`M1_encode`。
- **预期改善**：**未知**。本机 x86 数据 8 是平台点、16/32 负收益；但**本机 8 核 ≠ 昇腾 191 核**，带宽饱和点可能后移（§1.4）。
- **怎么跑**：脚本已就绪 → `python tools/bench/bench_server_threads.py --threads 8,16,32`（服务器上执行；输出对齐判据表 + `outputs/experiments/server_threads_<ts>.json`）。**只测量、不改默认。**
- **数值语义**：**逐位不变**（纯线程数，prange 行间并行不改行内归约顺序 —— `gate_m2_infer_kernels.py` B2 例专门验这条）。
- **风险**：**极低**（一个 CLI 参数，三个 run）。
- **判据**：端到端 ms/tok（主）+ `segments(win)` 的 `M2_infer` + **CS/s**（若线程数升高但 CS/s 也升 → 线程空转等锁）。

### 建议 4：`segments(win)` 取代 `segments(cum)` 作为诊断口径 —— ✅ **已落地（P115）**

**已落地**：`train/train.py:1700` 同时打印 `segments(cum)` 与 `segments(win)`（500 步滑窗），诊断一律读 `segments(win)`。零风险、纯日志。

### 建议 5（NPU，需服务器）：稀疏路径重评 `--torch-compile` —— ❌ 仍未做

- **影响段**：读出（M6，占端到端 64.7%）。
- **预期改善**：`phdnet/config.py:560-566` 记「已消掉其中 4 个 kernel（softmax/log/sub/addmm_ 融成一个）」+「本机 A/B（400-token smoke）快 15%」。但**这是 2026-09-30 的 dense 口径数字，稀疏路径从未验证**。
- **数值语义**：稀疏路径已放开且「本机实测融合与 eager **逐位相同**」→ **预期逐位**，但需门禁复验。
- **风险**：**中** —— P58 是 fhz 的**显式指令**「图优化关了吧」，`phdnet/config.py` 明确写「我不单方面推翻」。**必须先取得 fhz 授权。** 且 **P195 记录主权重模式下融合核必须旁路**（`accel_readout.py:527`）。
- **前置**：取得授权 + 服务器 `--torch-compile` A/B，判据是 `--probe-every` 的 probe_PPL。（同目标的替代路径：P205a 的 `--beta p0,p1` NPU Graph Capture。）

### 建议 6（NPU，需服务器）：`--nll-sync-every 1` vs `8` A/B —— **诊断优先于优化** —— ❌ 仍未做

- **影响段**：读出。**预期改善 0** —— 这是**诊断实验**，不是优化。
- **目的**：确认「NPU 假占用是等 CPU」这个假设（§4.4）。若 `sync_every=1` 与 `8` 读出耗时相同 → 无隐藏同步；若 1 明显更慢 → 还有同步点（判据来自 `docs/PHD-Net_CPU-NPU重叠执行计划.md` §4）。
- **风险**：**零**（一个 CLI 参数）。⚠ A/B 时须记录 `--beta` 是否开 `p2`（P205a 双流会改变 nll 链的等待对象）。
---

## 7. 可验证根因汇总（每题一行答案）

| # | 问题 | 根因 | 证据强度 |
|---|---|---|---|
| 1 | numba 线程封顶影响多大？ | **不是瓶颈**（本机）。2~8 线程即平台（1.8~2.4×），16/32 负收益。**但 x86 结论不构成昇腾证据** | 本机 x86 实测 + 源码（8 个 prange 区） |
| 2 | 负载均衡 / learn 串行段 | 行级负载完全均衡。真问题：**生产路径 `homeostasis=True` 曾使融合核被旁路**（12 次独立核调用 + 4 次串行 O(nnz) clip）→ **已修** | 源码逐行 + 本机线程缩放实测 |
| 3 | `_pc_infer_fused` 结构 | `n_steps` 循环在**核内**；每次 infer = **8 次 SpMV**；**无可合并冗余**。真浪费：**`idx` int64 占访存 67%** | 源码逐段数 + 本机访存核算 |
| 4 | NPU 假占用原因 | **最可能是等 CPU**（量级唯一对得上：35%）+ **Vector 单元低效**（差 4.7×）。**launch 开销量级不够**。⚠ 本机无 NPU | 代码分析 + 项目已有服务器记录（非本机实测） |
| 5 | numba 与预取争用 | **会争，但争带宽不争锁**。≈272 个潜在线程竞争 152 核；「剩下的核」在跑预取/分词，**不是等锁也不是全空闲** | 源码核数账 + 项目已有服务器遥测 |

---

## 8. ⚠ 本机 x86 实测 vs 需服务器昇腾复测（纪律清单）

1. **线程数结论**（建议 3）：本机 8 线程是平台 ≠ 昇腾 8 线程是平台。**本项目已有四次方向相反**，`docs/PHD-Net_并行与加速架构分析.md` 明记「同一条 CSR 行级 prange，在 x86 上无可测差别、在昇腾上赢 5.41×」。
2. **`--m2-kernel` 选择**：现为 `auto` 按平台择优。本机 fused 比 plain 快 1.36×，但昇腾实测 fused 慢 3-4× → **昇腾上仍应走 plain**。
3. **`idx` int32 压缩**（建议 2）：带宽饱和的**线程数饱和点**平台相关；且昇腾 gather 对窄索引的支持需实测。
4. **`_csr_clip` 加 prange / homeostasis 融合核**（建议 1，已落地）：本机 clip 1T/8T 仅 1.31×（纯访存），昇腾上可能同样低 —— **收益可能没有本机推算的那么确定**，但「逐位不变 + 消掉 4 次串行扫描」的结构性理由成立。
5. **NPU 全部判断**（问题 4）：本机无 NPU。
6. **`--omp-wait passive`**：本机墙钟 +90%（负收益），文档明记「191 核环境结论可能相反」。
7. **`OMP_PLACES` 不要再试**：服务器实测 `M2_infer` 2.4 → 19-32 ms/tok（已撤）。

---

## 9. 附录：为什么「token 500 的 19.59 / 15.28」不可信（**本题最重要的诊断**）

**源码事实（三步）**：
1. `phdnet/model.py:316` `self._prof[name] = self._prof.get(name, 0.0) + dt` —— **从 token 0 起累加、从不清零**。
2. `train/train.py:1694` `f"{k} {v * 1000 / max(1, i - done):.2f}"` —— 除以**总步数**（本题日志 = 500）。
3. ⇒ 第 500 个 token 打出的 `segments(cum)` = 稳态 + 启动期开销/500。

**启动期一次性成本（本机 x86 实测冷编译）**：`_pc_infer_fused` 9.62 s（+19.2 ms/tok）、`_pc_learn_fused` 3.94 s（+7.9）、`_csr_scale_rows`×4 1.11 s（+2.2）、`_csr_add_outer`/`_csr_oja_up` 各 0.96 s（+3.8）、`_csr_clip`×4 0.21 s（+0.4）→ **小计 ~16.8 s = +33.6 ms/tok**。

> ⚠ **诚实边界**：16.8 s **不含 `_csr_matvec` 自身的编译**（脚本在计时序列前已先调它 4 次算 `r1/r2/e0/e1`）；`_csr_matvec` 单独实测冷编译 ≈ **19.09 ms** —— 量级相对 16.8 s 可忽略，但严格说总冷编译 = 16.8 s + 19 ms + M1/M3/M4b 各自的核。同签名只编译一次（`add_outer(dn1)`/`oja_up(up1)`/`clip(up1..dn1)`/`scale_rows(up1..dn1)` 的「首次」都在 0.06-0.4 ms，与已编译的同签名核共享缓存）。

**⇒ 与服务器日志的 `M2_infer 19.59 + PC_learn 15.28 = 34.87 ms/tok` 量级吻合。** 这不是证明（服务器是 aarch64，编译时间不同），但说明**「cum 口径 + token 500」完全可能产生这个量级的数字**。项目内已有独立记录佐证（`docs/PHD-Net_CPU-NPU重叠执行计划.md` §5.1）：「`_prof` 是**从 token 0 起全程累加**……所以「当前瞬时成本低于打印值」。这解释了为什么 `readout` 从 28.77 一路"衰减"到 7.68 —— **不是模型变快了，是早期昂贵样本（numba 首次编译、设备 kernel 首次编译）被摊薄**。」P115 因此补了 `segments(win)`（滑窗真实值）。

**⇒ 结论：服务器日志里 `M2_infer 19.59 / PC_learn 15.28` 应判为「cum 口径污染 + LTM 生长期」的混合值，不是稳态并行瓶颈。诊断必须改用 `segments(win)`。**

⚠ 另有一个**非 JIT**的 cum 污染源：**LTM 生长期**。本机 1b 档 300→600 token，`M4b_ltm` 从 9.86 降到 5.23（win）—— `big_ltm` 的事件驱动表随经验生长，前 500 步的表稀疏度/哈希冲突与稳态不同。题目日志 `M4b_ltm 7.07 + M4b_imprint 7.07` 正好也处在这个衰减区间。

---

## 10. 给下一步的明确建议（按优先级）

1. **诊断口径已就绪**（零风险，已落地）：读服务器日志一律用 `segments(win)`。**在没有 win 口径数据之前，不要对 M2_infer/PC_learn 做任何优化决策** —— 19.59/15.28 很可能是噪声。
2. **跑两个零风险 A/B**：**建议 3**（线程数）→ `python tools/bench/bench_server_threads.py --threads 8,16,32`；**建议 6**（NPU 等 CPU 假设的证实/否证）→ `--nll-sync-every 1` vs `8`（记得记录 `--beta`）。一次上线拿到两个结论。
3. **建议 1 已落地**：剩下的是**在服务器复测收益**，不是再写代码。
4. **建议 2 的 M2/PC 侧留到最后**：收益可能最大（−33% 访存）但触及 ckpt 格式、三个构造函数、5 个核与 Cython 签名硬编码 int64，且昇腾 gather 对窄索引的支持未实测。读出侧已先行落地，可作实现范式。
5. **NPU 侧（建议 5）必须先取得 fhz 授权**再动 `--torch-compile`（P58 是显式指令）；同目标的替代路径是 P205a 的 `--beta p0,p1`。
6. ⚠ **一切本机结论到昇腾都要复测**：本项目已有**四次方向相反**（M1 fp32、M2 融合核、serial vs plain、`OMP_PLACES`），x86 的「够用 / 更快」在昇腾上不自动成立。

---

## 11. 建议 2（idx int32）的预研结论与落地现状

**P202-2 本机预研**（单核 median-of-7，三次独立运行，1b 尺寸 131,072 边）：idx int64 **160.8 / 151.3 / 180.7 µs** → idx int32 **128.1 / 121.9 / 141.7 µs**，加速 **1.255× / 1.241× / 1.276×**。

**数值**：max|d| = 4.77e-07，allclose(1e-6) 通过，**逐位不保证**（两个核名不同 → numba 各自编译，fastmath 重结合产物可能不同 —— 与 Cython 开关的「fp32 级漂移」同一口径）。

**当时的决定：暂缓**，理由：① 收益 1.25× < 诊断时的理论估计（带宽模型 1.15× 指的是 SpMV 本身，实际端到端还要除以「SpMV 占 M2_infer 的比例」）；② 改动面：三个构造函数 + 5 个 numba 核 + **Cython 签名硬编码 int64** + ckpt CSR 形状校验；③ **非逐位** → 违反「逐位不变」约束，需要 rebaseline（P146 探针 A/B）；④ 服务器线程 / nll-sync A/B 还没跑。

**后续落地（P211）**：改用「**同一份核按 dtype 特化**」的写法（不再另写 int32 版核）→ int32 与 int64 的 matvec/add_outer **逐位相同**（`tests/gates/gate_cpu_opt_p211.py` A 例 max|Δ|=0），并**已接入稀疏读出路径**（1B 档 CSR 内存 79.8→53.2 MB、前向 1.29×/更新 1.30×）。

**重启条件（对 M2/PC 侧仍然有效）**：服务器 `segments(win)` 显示 M2_infer 稳态仍 >5 ms/tok **且** msprof 显示带宽受限（而非 barrier/等 CPU），此时 §3.4 的收益估算成立、值得把 `_compress_csr_idx` 接到 `SparsePCStack` 的 4 个 CSR 上。

---

## 12. 复现方式

本报告的性能数字来自 P201 期间的 14 个**只读临时脚本**（`_p201_bench*.py`），它们未入库、已随 P206「清理调试临时文件」删除；**结论与关键数字已固化在本报告 §1.2 / §1.3 / §2.3 / §3.4 / §9**。当前可用的常驻基准脚本：`tools/bench/bench_cpu_local.py`（本机回归基准 + 线程缩放）、`tools/bench/bench_server_threads.py`（**服务器端到端线程 A/B**，建议 3 用）、`tools/bench/bench_readout_segments.py` / `bench_readout_sparse.py`（读出分段与稀疏读出基准）。

---

⚠ **收尾警告**：本报告所有**性能**数字均为 **x86 本机实测**；服务器是昇腾 aarch64，**本项目已有四次方向相反**的实测 —— x86 的「够用」「更快」都不构成昇腾证据。**任何优化在昇腾生效前必须复测。**
