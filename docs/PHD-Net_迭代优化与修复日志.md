# PHD-Net 迭代优化与修复日志（2026-09-29 ~ 09-30，P53–P81）

> 本文记录两天集中迭代的**完整过程**：动机 → 做法 → 实测 → 教训。
> 配套文档：当前结论见 `README.md` 与 `docs/PHD-Net_性能评估与迭代方案.md`；
> 缺陷台账见 `BUGS.md`；架构级约束分析见 `docs/PHD-Net_并行与加速架构分析.md`。

## 0. 总览

| 维度 | 起点 | 终点 |
|---|---|---|
| 1B 服务器步时 | 26–104 ms/tok（且超线性恶化） | 预期 ~20 ms/tok（P70–P78 全部落地后待复测） |
| NPU 利用率观测 | 恒 `--`（遥测缺失） | NPU% 98% 可见、HBM 可见 |
| M4b_ltm | 63.5 ms/tok（占 76%）且超线性 | 2.7–22 → 预期个位数（P70 predict 44.6× + P78 learn 多核） |
| 读出损失计算 | softmax+log+cast+scatter 4 kernel | cross_entropy 单 kernel（P80） |
| 线程治理 | numba 191 线程 / BLAS 191 线程 | 各限 8 + PROC_BIND 绑核 |
| 检查点 | bf16 保存崩溃 | uint16 位模式闭环（P81） |
| 缺陷台账 | 无 | `BUGS.md` 30 条（含被否决的优化） |

---

## 1. 稳定性修复（先让它能跑）

| P | 问题 | 根因 | 修复 |
|---|---|---|---|
| P54 | 训练启动 `IndentationError`；`_data_provenance` 被误删 | 改结构不动引用 | 恢复；`verify_ms_stream` 加语法门禁 |
| P54 | `--help` 长期崩溃 | argparse help 里裸 `%` | `%%` 转义；**所有 help 文本新增后必须跑 `--help`** |
| P55 | NPU 首跑崩 `FakeTensor - None` | P45 融合核漏改 `target_idx` 分支 | 融合核逐行镜像 eager；2×2 矩阵验证 9/9 |
| P57 | NPU/HBM 遥测恒 `--` | `Telemetry()` 无参构造 → 分支永不触发 | device 自动探测；失败打印原因 |
| P58 | `torch.compile` 编译失败崩生产 | inductor **首次调用**才编译 | 失败永久回落 eager + 告警；`--torch-compile` 默认关 |
| P81 | bf16 检查点保存崩 `unsupported ScalarType BFloat16` | numpy 无原生 bf16 | 存 uint16 位模式，与加载侧 P46 解码闭环（无损） |

## 2. 性能优化（每条都有实测）

### 2.1 消除同步与传输（P58/P62，端到端 −20~30%）
- `nll_sync_every` 默认 1 → **8**：每步 `.item()` 把 NPU 延迟全额暴露给 CPU，
  流水线无法重叠——这就是「CPU 忙 NPU 空闲」的真相（CPU 其实在等）。
- h 的 H2D 走 **pinned 暂存 + `non_blocking`**（pageable 会阻塞）。
- 遥测禁用 `torch.npu.utilization()`（**它同步设备流**）。

### 2.2 CPU 机制层 numba 化（P59/P61/P68/P70/P77/P78）
| P | 对象 | 实测 |
|---|---|---|
| P59 | M2 学习侧融合（8 核 → 1） | 1.15–1.72×，逐位 |
| P61 | M1 编码器 fp32 | **1.80×**（x86；⚠ 昇腾反例见 §4） |
| P68 | recall 反投影 numba（rev 静态 → 预 CSR） | **8–35×**（nogil 串行，prange 会竞态） |
| P70 | `predict_arr`（跳过 dict 合并） | **44.6×**（8.78 → 0.197 ms，服务器形态） |
| P77 | M1 GEMV 平台自适应（aarch64 → numba 核） | 服务器 58 ms → ~1 ms（待复测） |
| P78 | `learn` 批量多核（gather + prange） | 服务器 20 s/次 → 毫秒级（待复测） |

### 2.3 线程与绑核治理（P62/P71/P73/P74）
- numba prange 限 **8 线程**（P22：核内 1→6 线程仅 1.16×，访存饱和）。
- **BLAS 线程也曾是 191** → 在 `import numpy` **之前**限 8（晚设无效）。
- `OMP_PROC_BIND=close` 绑核保留；⚠ `OMP_PLACES=cores` 在 191 核上反而让
  M2 慢 8–13×（place 表随核数膨胀，每次同步遍历）→ **已回滚**。

## 3. 被否决的方案（负收益同样是结论）

| 方案 | 实测 | 死因 |
|---|---|---|
| `learn_predictive` 融合核 | 大范围 0.91–0.96× | 12 个 prange 段的调度成本 > 省下的核启动 |
| LTM 表向量化（predict/learn Python 循环） | 0.72–0.76× | 原版靠 `ltp<=0` 短路极便宜；向量化无条件付出 |
| gather+prange 首版（P67） | 0.96× 且多步不一致 | gather/scatter 拷贝 + 生长分支被 cap 卡住 + 误绑 growth_guidance |
| `encode` numba 去重 | 0.84× | 核内 O(n²) 线性扫描输给 Python `set` 的 O(1) 哈希 |

**共性教训**：本机 x86 上「看起来是热点」的 Python 循环，在规模、平台、
短路行为三者的真实组合下未必是热点——**每一次都要实测，负收益立即回滚**。

## 4. 平台差异专题（x86 ≠ aarch64，三次反噬）

1. **fp32 权重 @ fp64 输入**脱离 BLAS，慢 5.9×（x86 就有）→ 输入必须同 dtype。
2. **P61 fp32 编码器**：x86 快 1.80×，**昇腾慢 70×**（0.85 → 58 ms）——
   aarch64 numpy 对 1024×2048 单行 GEMV 病态慢，fp32/fp64 都慢 → 平台自适应
   （aarch64 走自写 numba GEMV，x86 走 BLAS）。
3. **P52 融合核**：x86 快 1.15–2.16×，**昇腾慢 8–13×**（prange+fastmath 退化）
   → `--m2-kernel {fused,plain}` A/B 开关，默认 plain。
4. **`OMP_PLACES=cores`**：191 核 place 表让每次线程池同步遍历 → M2 再慢 8×，
   CS/s 升到 650 万 → 回滚，只留 `OMP_PROC_BIND=close`。

**规则**：跨平台项目里，**任何优化必须在目标机器复测**；x86 的结论只在 x86 成立。

## 5. 观测能力建设（「修一半」的另一半）

- `--step-profiling`：九段分解 + **主循环三段**（`loop: tokenize/encode_onehot/step`，
  P72——此前分段之和 < 总耗时的缺口无法归因）。
- `[ltm-diag]`（每 10 次）：prev/cur 规模、组合数、**rate_dims**（稀疏度演化）、
  表行数、recall 绑定遍历量。⚠ 门槛要考虑**触发频率**——设 1000 次时条件触发
  的 imprint 一行都不出。
- `[sample lang=…] CJK=…%`：`--lang` 是否生效**一眼可见**（实测那次 argv 根本
  没带 `--lang`）。
- `GC <对象数>M/gen2 <次数>`：GC 假设**被数据证伪**（恒定）→ 调优保留但非解药。
- `[numba] cache dir / size`：持久缓存是否命中可见（P79）。
- 遥测：NPU%（npu-smi 三级定位 + 表头定位解析）、HBM、CS/s、CPU 核数。

## 6. 数据问题（代码之外）

实测远程分片（ModelScope `fhzfhz/Mixture-General-Mini`）：
- 52 片中 0–45 全是 `infinity_m7core`，lang 分布 **en 50% / unk 47% / zh 仅 2%**；
- **`unk` 是代码**（CJK 占比 0%）；
- 只有末片是 `ultrafineweb_l3_zh`。
- 与「M7_Core 中文 3,697 万块 ≈ 95%」的记录**严重不符**（原记录在删除清单的
  元数据中，已内联进 README）→ 需核对数据制备/上传环节。

## 7. 当前状态与开放项

- **已上线待服务器复测**：P70（predict 44.6×）、P76（M1 dtype 修复）、
  P77（平台自适应 GEMV）、P78（learn 多核）、P80（nll 单 kernel）。
- **开放项**：
  1. `M4b` 剩余成本归因（`M4b_imprint` 内层段已加，等日志）；
  2. bf16 读出学习退化的决策（`--readout-dtype fp32` 对照）；
  3. 数据集语种构成核对（#18）；
  4. 检查点 `compact_csr` 的 0.8–37 s 硬停顿（向量化候选）；
  5. M6 读出幂律稀疏化（脑同构，方案已给，待性能地基确认后开工）。
- **预期**：全部落地后 1B 服务器步时 ≈ **20 ms/tok 以内**，其中 readout ~5.5 ms
  （访存受限）为最大单段——再往下需要 CANN 自定义算子或 msprof 内核级 profiling。
