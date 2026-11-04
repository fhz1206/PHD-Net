# CPU / NPU 重叠：决策记录（当前结论）

> **本文不是待执行计划**，是「为什么不做跨 token 流水」的**决策记录**。
> **结论**：`--readout-pipeline` 是 **fail-fast 死开关**（传了直接报错退出），
> 跨 token 流水**当前不做**。证据门禁：
> `tests/gates/gate_cpu_npu_overlap_contract.py`（19/19）。
> **数据截止**：2026-11-04（P214）。
> 性能数字唯一出处：[性能评估与迭代方案](PHD-Net_性能评估与迭代方案.md)；
> NPU 假占用的诊断见 [NPU 利用率诊断与优化 P214](PHD-Net_NPU利用率诊断与优化_P214.md)；
> `--beta` 四档现状见 [NPU 加速特性现状](P205a_NPU流水线设计.md)。

---

## 1. 当前结论（一句话）

训练步的设备路径**本来就没有阻塞式同步** —— `forward_dev` 返回**设备张量、
零同步**，训练步把 `y` 置 `None`；`step()` 内唯一的显式 D2H 是 `nll_dev.item()`，
被 `nll_sync_every=8` 门控。**代码里每步的显式同步点已压到 1/8 步一次。**

「12.09 = CPU 4.26 + NPU 7.83 完全串行」若真存在，成因是
**运行时下发 / 队列背压**（与 msprof 的 `aclnnIndex` 89.3% /
`aicore_time=0` / `Task Wait` 6.8× 同源），**不是**代码里的显式同步。

→ 再叠一层跨 token 流水 = **在已有异步之上叠第二层异步**，
收益未测、语义风险高 → **不做**。

---

## 2. 为什么跨 token 流水不能做（三条红线）

### 2.1 依赖链：只有 `M1(t+1)` 能安全重叠

`phdnet/model.py::step` 的实际编排：

```
step(t):
  M1_encode → M2_infer → M3_pred → M5_mod → M4a_wm → M4b_*     ← CPU
  → M6 forward_dev + learn_softmax                                ← 设备
  → PC_learn   (CPU，用 cache = **本步** M2 的输出)
  → STDP_learn (CPU，用 self._prev_rate 与**本步** rate)
```

- `M2(t+1)` 依赖 PC 权重，而它在 `PC_learn(t)` 里更新；
- `PC_learn(t)` 在 `M6(t)` **之后** →
  **`M6(t) → PC_learn(t) → M2(t+1)` 是不可打断的串行链**；
- `STDP_learn` 依赖 `_prev_rate`（来自读出输出）→ 同样在读出之后。

| 段 | 能否与 M6(t) 重叠 |
|---|---|
| `M1_encode(t+1)`（只依赖 x） | ✅ 可 |
| `M2_infer(t+1)`（等 `PC_learn(t)`） | ❌ |
| M3 / M4a / M4b（用本步 rate / LTM 状态） | ❌ |
| `PC_learn` / `STDP`（在读出之后） | ❌ |

→ 可重叠量只有 `M1`，修正后收益约 **1.09×**（不是早期估的 1.53×）。
**禁止把理论上限 `max(4.26, 7.83)` 当承诺** —— 那是理想重叠，不是实测。

### 2.2 前移 `learn_softmax` = 换一套时序规则（铁律③）

把 t 步的 `learn_softmax` 延迟到 t+1 步 `forward_dev` 之后、
在同一个 NPU 批里执行 → **会改变 `PC_learn` 读到的 nll 时序**
（`task_modulation` 用 nll 门控本步主干学习率）→ 语义变更。
同理，把 `PC_learn` / `STDP_learn` 移到 M6 **之前**会改变 `_prev_pc` 的赋值时机。
**逐 token 语义（铁律③）在这里不是约束，是必须严格保住的东西。**

### 2.3 `readout_recurrence` 打开会造成静默语义错误

唯一真跨 token 的 `y` 依赖是 `readout_recurrence`（默认 **False**）。
一旦打开它，流水会让 `recur_cue` 拿到**上一步**的 y → **静默的语义错误**
（不报错、不崩溃，只是学错）。

---

## 3. 已实现 / 未实现的相邻手段

| 手段 | 状态 | 说明 |
|---|---|---|
| `--nll-sync-every 8`（默认） | ✅ 已实现 | `.item()` 硬同步摊到 1/8 步 |
| `forward_dev` 零同步 + pinned H2D | ✅ 已实现 | `learn_softmax` 复用缓存的 `ht`/`y`，**不重复 H2D** |
| `--beta p2` 双流（H2D 独立流 + nll 独立流） | ✅ 已实现，默认关 | CPU 只等 nll 链，不等主流排空 |
| `--beta p0/p1`（Graph Capture / 融合图） | ✅ 已实现，默认关 | 消除 kernel launch 开销与图间下发间隙 |
| `--beta p3`（双缓冲流水线） | ❌ **未实现** | `--help` 里列了，**代码里没有 p3 分支** → 传了静默无效 |
| `--readout-pipeline` | ❌ **fail-fast 死开关** | 传了直接报错退出（P192 定案） |
| `torch_compile`（稀疏路径） | 默认 False（P58 指令） | 本机已实测融合与 eager 逐位相同，服务器 A/B 有空间 |

---

## 4. 诚实边界（仍成立的判读纪律）

- **所有重叠收益数字都是估算**，基于「实测分段耗时 + 理想重叠」，
  **没有任何一条来自实测的重叠实验** → 禁止当承诺。
- **x86 结论不构成昇腾证据**（本项目已有四次方向相反）。
  稀疏前向改 SpMM 是否真快，**只能在服务器上测**。
- CS/s 稳态与 NPU 99% 并存：热路径已无每步同步点（静态复查），
  但**静态扫描 ≠ 运行时结论**。真正的判据是
  **`--nll-sync-every 1` vs `8` A/B**：
  若两者读出耗时相同 → 确无隐藏同步；若 `1` 明显更慢 → 还有同步点。
- **CSR 索引压缩的两个坑**（外部建议核对结论，仍然有效）：
  ① **值不能降到 bf16**（P110 实测非目标行更新保留率仅 5.79%，会退化成纯 Hebbian），
  只能是 fp32 → 只有索引能压（8 B → 2 B，省 ~22%，不是 96×）；
  ② **昇腾 gather 对 uint16 索引的支持需实测**（可能触发类型转换反而更慢）。
  P186 的 u16 Rust 实现已随 `phdnet_rs/` 于 P188 整体删除，**当前不存在**。
  ⚠ P211 已落地 **int32 CSR 索引**（`_compress_csr_idx`），是这一方向的现行状态。

---

## 5. 当前该做什么

**不要动训练主循环。** 有效路径按性价比：

1. **CPU 侧加速**（已落地）：P214 的 LTM 哈希槽位查找 + 持久化缓冲 →
   `imprint` **1.83×**，直接缩短 NPU 等 CPU 的空窗。
2. **服务器 A/B**：`tools/bench/bench_server_threads.py --threads 8,16,32`
   （补 P204 遗留的线程数空白）。
3. **NPU 侧融合 A/B**：`--beta p0,p1,p2` + `--nll-sync-every 1 vs 8`，
   判据用 `segments(win)` 的 readout + 端到端 ms/tok + **probe_PPL**。
4. **单点最大热点**：`Index` 算子占 89.3% 且跑在 Vector 单元 →
   改成 Cube 友好的 SpMM / 合批，是唯一能大幅挪动 `aicore_time` 的方向
   （风险高，须先有 msprof 基线）。

详见 [NPU 利用率诊断与优化 P214](PHD-Net_NPU利用率诊断与优化_P214.md) §6 的
服务器待执行清单。
