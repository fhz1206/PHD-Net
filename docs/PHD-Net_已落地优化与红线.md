# 已落地优化与红线（现状清单）

> **适用范围**：当前生效的优化、必须知道的禁忌、防止重踩的设计教训。
> **数据截止**：2026-11-04（P214）。
> **只保留现状**：演进过程 / 逐项实测明细见 git 历史；
> 全部性能数字的唯一出处是 [性能评估与迭代方案](PHD-Net_性能评估与迭代方案.md)；
> 当前默认值见 [现状速查](PHD-Net_现状速查.md) §3；完整红线见其 §7。
> 与代码冲突时**一律以代码为准**。

---

## 1. 当前生效的优化

| 优化 | 状态 | 判据 / 门禁 |
|---|---|---|
| **int32 CSR 索引**（`_compress_csr_idx`） | 默认开 | `gate_cpu_opt_p211` |
| **forward-only 融合核**（6 次核调用 → 1 个 numba 核） | 默认开 | `gate_cpu_opt_p211` B1/B1b/B1c 逐位对拍 |
| **forward-only 默认开启** | 默认开 | `--no-forward-only` 关闭；`--forward-only-eta 0` ≡ 旧路径 |
| **LTM 内联 / CSR 在线表** | 默认开 | `gate_ltm_csr_equivalence` / `gate_ltm_learn_batch` |
| **fp8→int8 查表**（`_E4M3_POW2`） | 已接线 | `gate_fp8_int8_convert` 17 项；nogil 路径 **3.06×**，数值逐位相同 |
| **fp8 转化效率报告**（`report=`） | 已接线 | 同上；字段 `ms` / `MiB_in` / `MiB_out` / `MiB_per_s` / `ratio_vs_fp32_copy` / `conv` |
| **M2 推理核按平台择优**（`default_m2_kernel()`） | `--m2-kernel auto` 默认 | x86→`fused`（实测 1.34–2.05×）；昇腾 aarch64→`plain`（P113 5.41×）。可显式覆盖 |
| **`--replay-every` 接线** | 已修 | 只需判 `> 0`；`argparse default=0` 已表达「沿库默认」 |
| **`homeostasis` 参数** | 已修（融合核条件化） | `gate_cpu_opt_p211` B1b/B1c：`False` 与 `True` 结果必须**不同** |
| **forward-only 接受 `learn_scale` 调制** | 已修 | `gate_cpu_opt_p211` B1d/B1e；普通训练 `learn_scale=1.0` 逐位不变 |
| **LTM 哈希槽位查找 + 持久化缓冲**（P214） | 默认开 | `gate_ltm_hash_lookup` **13 项**；`imprint` **1.83×**，逐位一致 |
| **统一门禁 runner** | 已接线 | `tools/run_gates.py`（自动选带 numpy 的解释器 + 性能敏感门禁串行） |
| **numba 线程 A/B 脚本** | 已接线，**服务器未跑** | `tools/bench/bench_server_threads.py --threads 8,16,32` |
| **NPU 假占用判读工具** | 已接线 | `tools/probe/diagnose_npu_utilization.py` |

⚠ M2 三态（`plain`/`fused`/`serial`）**不是逐位相同**，是**容差级**
（`_RTOL=1e-6`）：fused vs plain `max|Δ|` ≈ 1e-7（fp32 舍入），
端到端 `final_ppl` 13017.891 → 13017.892。与「逐位不变」的优化性质**不同**。

---

## 2. 已删除的死代码（勿再重新实现）

| 对象 | 为什么删 |
|---|---|
| 稠密 `forward_target_delta`（`np.outer`） | 生产零调用，已被稀疏融合核取代，且**与铁律②结构性稀疏冲突** |
| `learn_forward_target` 中 `return` 后 17 行 | 不可达死代码 |
| `_fwd_cache` 字段 + 误导注释 | 从未被读写；真缓存在 `forward_only.py` |
| `phdnet/backends/fp8_capability.py`（244 行） | 零 import；能力探测已被 `accel_readout._probe` + 降级链取代。其「转化效率」需求已由实际在用的 `fp8_to_int8_codes(report=)` 兑现 |
| `ltm_kernel._encode_hash_uniq` | P68 实测 0.84× 负收益已回滚的残留死核 |
| `torch_backend._resolve_device` | 零引用；其 docstring 声称的「A1 修复」从未接线 |
| `ms_stream.local_name_of` / `precision_policy.DEFAULT_ITER_DTYPE` | 零引用的孤立符号 |
| config 里 `readout_master_weight` / `readout_shadow_refresh` 重复定义 | 后者覆盖前者，值相同但语义含糊 |

---

## 3. 保留不动的实现（有实测依据，勿误伤）

| 实现 | 依据 |
|---|---|
| `learn_predictive` 多核调用 | 融合版实测 0.91–1.36× **负收益** |
| `SparseLTM.encode` 用 `dict.fromkeys` | P95b/P98 逐项实测；`np.unique` 94.3 µs 更慢 |
| 读出前向走 BLAS | numba 版慢 0.68× 且非逐位等价 |
| fp8 转换默认走 torch | 实测快 5.6× |
| `plain` 作为昇腾默认 M2 核 | 融合核在昇腾慢 3–4×（prange 屏障成本随核数放大） |

---

## 4. 必须知道的设计教训（防重踩）

| # | 教训 | 出处 |
|---|---|---|
| 1 | **不要拿库默认值当哨兵**（`args.x != 4` 那种写法）：库默认一改就失效 → `--replay-every 4` 曾被静默丢弃 | P212 bug ① |
| 2 | **融合核改造最容易漏「条件化语义」**：把多核调用合并成 1 个核时，每个核原本的条件分支都要显式搬进核内，否则参数被静默忽略，且**生产恒为 True 时完全看不出来** | P212 bug ② |
| 3 | **门禁只测 True 分支 = 语义分叉不可见** → 要补「False 与 True 必须不同」的断言 | `gate_cpu_opt_p211` B1c |
| 4 | **新默认路径接入既有子系统要检查「信号是否对齐」**：RL 的策略梯度只调制读出，forward-only 固定强度注入与奖励无关的随机码目标 → 主干被扰动、策略学不动。修法：同口径接受 `learn_scale` | P212 |
| 5 | **RL 的 `learn_scale` 可以是负的**（advantage<0 应反向更新）→ 判据用 `!= 0.0` 而非 `> 0.0`，否则丢掉一半策略梯度 | 同上 |
| 6 | **docstring 声称的门禁/工具要先核实存在**：`fp8_int8_convert.py` 声称由 `verify_fp8_int8_convert.py` 把关，该文件**从来就不存在** → 保护链是断的，代码里恰好藏着未落地的查表优化 | P212 |
| 7 | **RL 玩具判据脆弱**（seed 偏移 1/2/3 → PASS/FAIL/FAIL）→ **不要靠改 seed 让它过**，要修它暴露的真缺陷 | P212 |
| 8 | **`_decay_read_arr` 必须逐个调 `_decay_read`** —— `gate_ltm_trace_semantics` 的突变测试注入点；内联会让负向对照变空操作 | 契约 |
| 9 | **`_learn_batch` 的 K/V 必须预留到 `m_out`** —— 核内会生长到 m_out，收紧会广播失败 | 契约 |
| 10 | **x86 的性能结论不构成昇腾证据**，连「无差别」也不构成（项目已有四次方向相反） | 铁律 |
| 11 | **判性能一律看 `segments(win)`**；`segments(cum)` 是含 JIT 冷编译的历史平均 | 纪律 |
| 12 | **A/B 判据只能用 `probe_PPL`** —— sliding PPL 信噪比仅 0.40，<20% 测不出 | 纪律 |

---

## 5. 门禁现状

| 门禁 | 项数 | 覆盖 |
|---|---|---|
| `gate_cpu_opt_p211` | 19 | int32 索引 / FO 融合核逐位等价（含 `homeostasis=False`）/ **`learn_scale` 调制契约** / LTM 内联 |
| `gate_forward_only` | 30 | 固定随机投影 / 层目标学习 / 「eta=0 ≡ 旧路径」逐位判据 |
| `gate_fp8_int8_convert` | 17 | 查表与 powf 逐位等价 / 边缘位模式 / NaN / 转化效率报告 |
| `gate_ltm_hash_lookup` | **13** | 两核逐位一致 / int8 / 饱和期 / **显式重复键首个槽** / **真实脏缓冲** / 哈希表容量 / **开关分派负向对照** |

**全量 54/54 通过**（`python tools/run_gates.py --jobs 4`）。

### 5.0 P217 新增：`index_select` gather 开关（默认未改）

AtomGit NPU Notebook（**Ascend910B4**）实测：
`torch.index_select(ht, 0, Wi.ravel()).view(Wi.shape)` 与生产 `ht[Wi]`
**逐位相同**（`max|Δ|=0`），但快 **52.80×（fp16）/ 29.63×（fp32）**
（6.483→0.123 / 6.743→0.225 ms/step）。已接成
`--readout-gather-impl index_select`（第三选项，与 `index`/`take` 并列），
门禁 `gate_gather_impl.py` **17/17** 锁住逐位等价 + 负向对照。

⚠ **默认仍是 `index`** —— 那是孤立算子 benchmark，生产 `_sp_gather` 还有
缓存 / fp8·int8 量化臂 / `_gather_dtype` 降精度分支；改默认前必须
服务器 A/B（`segments(win)` readout + 端到端）+ `probe_PPL`。
⚠ benchmark 里「稠密 SpMM」一项**不等价**（`scatter_` 对重复列覆盖，
`max|Δ|≈1.05`），别当成可用实现。

### 5.1 门禁有效性契约（改动门禁前必读）

**门禁必须把业务失败传播为进程退出码** —— 只 `print("FAIL")` 不算把关。
`tools/run_gates.py` 与 CI 都按 `returncode` 判定，一个「打印 FAIL 但 `rc=0`」
的门禁等于**没有门禁**。

| 规则 | 说明 |
|---|---|
| ① 结果收集 + 退出码 | 用 `check()` / `_RESULTS` 收集，汇总处 `raise SystemExit(0 if all_ok else 1)` |
| ② 禁止裸 `assert` 作唯一断言 | `python -O` 会整段跳过；改显式收集 |
| ③ 禁止零断言的 `gate_*.py` | 只打印结果的探针请放 `tools/probe/`，别占用 `gate_` 前缀 |
| ④ 参考实现必须与生产**同精度同口径** | 否则会把「精度不同」误判成「并行不一致」；确实不能逐位时按 `gate_cython_kernels` 惯例用容差并**如实打出 `max|Δ|`** |
| ⑤ 开关类契约要有**负向对照** | 验证开关被忽略时门禁会 FAIL，而不只是「True 分支跑通」 |
| ⑥ 脏数据测试必须写在**会被复用**的对象上 | 写进新建的孤立对象 = 空操作 |

**防回归**：`python tools/audit_gates.py --strict`（静态初筛无退出机制的门禁）。
⚠ 静态只是初筛：新增/修改门禁后，还要**人工动态复核** —— 把业务断言强制改成
`False`，确认 `rc != 0`。临时脚本必须放在 `tests/gates/` 内，否则
`Path(__file__).parents[2]` 定位仓库根会失效（会制造「全部能检出」的假阳性）。

**性能敏感门禁**：含 `perf_counter` / 墙钟时限的门禁必须**串行**跑
（`tools/run_gates.py` 自动检测 + 显式名单），并发抢 CPU 会造成假失败。
