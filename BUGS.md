# BUGS.md — 缺陷与教训台账

> 记录**真实发生过的缺陷**：症状 → 根因 → 修复 → 门禁缺口 → 提交。
> 目的不是记流水账，而是让同类问题**下次被门禁提前拦住**。
> 维护约定：每条必须有 commit；被回滚的优化也记（负收益同样是结论）。

---

## 0. 索引

| # | 症状 | 类型 | 提交 |
|---|---|---|---|
| 1 | 训练启动 `IndentationError` | 我引入 | `0b90c80` |
| 2 | `train.py --help` 长期崩溃 | 既有遗留 | `0b90c80` |
| 3 | fast 门禁 9/9 但训练入口是坏的 | **流程漏洞** | `0b90c80` |
| 4 | NPU/HBM 遥测恒为 `--` | 既有遗留 | `8d52662` |
| 5 | `npu-smi` 找不到 → AI Core% 恒 `--` | 环境 | `8f63ea9` |
| 6 | NPU 首次真跑训练即崩（`FakeTensor - None`） | 既有遗留 | `d67d0e5` |
| 7 | inductor 编译失败会崩生产（无回落） | 既有遗留 | `d67d0e5` |
| 8 | 融合核漏乘系数（写测试时抓出） | 我引入 | `4ce0a48` |
| 9 | 逐位对拍「假阳性」：基线用错实现 | **方法论** | `4ce0a48` |
| 10 | `learn_predictive` 融合：大范围负收益 | 优化否决 | `4ce0a48` |
| 11 | LTM 表向量化：负收益 0.73× | 优化否决 | 回滚（无提交） |
| 12 | gather+prange 方案：拷贝吃掉收益 | 优化否决 | 回滚（无提交） |
| 13 | `encode` numba 化：O(n²) 输给 `set` | 优化否决 | `720faf2` |
| 14 | `[ltm-diag]` 一行都不输出 | 我引入 | `720faf2` |
| 15 | fp32 权重 @ fp64 输入 → 脱离 BLAS 慢 5.9× | 性能陷阱 | `0f203d5` |
| 16 | 遥测自己同步设备流、破坏流水 | 我引入 | `b229d91` |
| 17 | 「`--lang zh` 后还是英文」误判 | **观测错误** | `6b9a29e` |
| 18 | 数据集语种与设计不符（中文仅 2%） | **数据问题** | 待查 |
| 19 | 计时误导：命中首项 / 规模猜错 54 倍 | **方法论** | — |
| 20 | `_append` 满行时 IndexError | 潜在（生产有守卫） | 待修 |
| 21 | 验证脚本里塞 `or True` 假断言 | 我引入 | 当场修 |
| 22 | 用 `grep` 过滤验证输出，把 Traceback 一起滤掉 | **流程漏洞** | `0b90c80` |
| 23 | M4b 占 76% ms/tok：`predict` 的 1.8 万次 dict 更新 | 性能（已修） | `391fa06` |

---

## 1. 训练入口类

### #1 训练启动 `IndentationError`（`0b90c80`）
- **症状**：服务器 `File "train_1b/train.py", line 385, IndentationError: unexpected indent`。
- **根因**：把 `else:` 分支改写成 `if not remote_active` 时，**原 else 里那行 `print` 变成孤立缩进块**；同一次编辑还把 `_data_provenance` 整段删掉（`save_model` 仍引用 → 即使缩进修好也会 NameError）。两处都是「改结构不动引用」。
- **修复**：恢复正确缩进 + 补回 `_data_provenance` 定义。
- **门禁缺口**：见 #3。

### #2 `train.py --help` 长期崩溃（`0b90c80`）
- **症状**：`--help` 抛 `TypeError: must be real number, not dict`。
- **根因**：`--torch-compile` 的 help 文本里有**裸 `%`**（`measured 15% faster`）。argparse 用 `help % params` 格式化 → `% f` 要求 real number。**自 P45 起 `--help` 就一直是坏的**（训练不调 `format_help` 所以不炸）。
- **修复**：改 `15%%`。
- **教训**：argparse help 里所有字面 `%` 必须转义；新增 help 文本后要跑一次 `--help`。

### #3 fast 门禁 9/9 但训练入口是坏的（`0b90c80`）
- **症状**：#1/#2 都没被 `python tests/run_tests.py fast`（9/9 全绿）拦住。
- **根因**：fast 门禁只跑 `tests/` 下的检查，**从不 import `train_1b/train.py`**。
- **修复**：`tests/verifiers/verify_ms_stream.py` 增加两道真门禁 ——
  ① 对全部改动文件 `py_compile`；② subprocess 真跑 `train.py --help` 并断言
  `returncode == 0` 且新参数出现在 stdout。
- **门禁缺口（仍存在）**：fast 门禁本身仍不 import 训练入口，靠 verifier 兜。
  **改 `train_1b/*.py` 后必须单独编译/--help**。

### #22 用 `grep` 过滤验证输出，把 Traceback 一起滤掉（`0b90c80`）
- **症状**：`train.py --help | grep remote` 无输出，我据此以为「参数已加上」，
  实际那次调用正在崩溃。
- **修复**：不再用 `grep` 判断成功，改为断言 exit code / 捕获 stderr。
- **教训**：**过滤输出 = 隐藏错误**。验证要断言，不要过滤。

---

## 2. 读出 / 加速器类

### #6 NPU 首次真跑训练即崩（`d67d0e5`）
- **症状**：`TorchRuntimeError: ... unsupported operand type(s) for -: 'FakeTensor' and 'NoneType'`，
  崩在 `accel_readout.py:159` 的 `dp = (p - t32)`。
- **根因**：P45 的「`target_idx` 路径就地改 p」只实现在 **eager 分支**，
  `torch.compile` 融合核 `_train_step_core` 漏改，而该路径 `t32 = None`。
  既有 `verify_accel_readout.py` 的 learn_softmax 用例**只走 target 数组路径**，
  从不传 `target_idx` → 恰好漏掉崩的那条。
- **修复**：融合核逐行镜像 eager（`p − onehot` 与 `p[c] −= 1` 数值等价）；
  顺带把 softmax/nll 移入 eager-only（compiled 路径原白算一次 51962 元素 softmax）。
- **门禁缺口**：新增 `tests/verifiers/verify_accel_readout_p55.py`（2×2 矩阵：
  target 数组 / target_idx × eager / compiled，9/9）。

### #7 inductor 编译失败会崩生产（`d67d0e5`）
- **症状**：`AccelReadout.__init__` 有 try/except 回落，但**首次调用 `_fused` 才真正
  触发 inductor 编译** → Windows 无 C++ 编译器时直接抛 `InductorError`。
- **根因**：构造期的 `torch.compile(...)` 只是包装，不会失败；回落写在了错误的层。
- **修复**：首次执行失败 → **永久回落 eager + `warnings.warn` 记原因**（P19「回落 + 记原因」纪律）。

### #4 NPU/HBM 遥测恒为 `--`（`8d52662`）
- **症状**：日志 `NPU/GPU --%  HBM --/--GB`，但服务器确有 NPU。
- **根因**：`Telemetry()` 一直**无参构造** → `device=""` → `sample()` 的加速器分支
  **永不触发**（P41 遗留；P19 把读出搬上 NPU 后没人回补遥测）。
- **修复**：自动探测 device（`torch.npu` → `torch.cuda`，显式传入仍优先）；
  首次采样失败打印原因（原来 `except: pass` 全吞）；`TeeLogger` 接管 `sys.stderr`
  （回落告警/inductor 日志此前不落盘）。
- **门禁缺口**：`verify_ms_stream.py` 里加了「无 NPU 机器探测为空且行为不变」的断言。

### #5 `npu-smi` 找不到 → AI Core% 恒 `--`（`8f63ea9`）
- **症状**：HBM 有值（走 `torch.npu.memory_allocated()`，host 侧计数器）但 AI Core% 没有。
- **根因**：AI Core% 走 `npu-smi`，而训练是**直接 `python train.py` 启动**的，
  没 source Ascend 的 `set_env.sh` → `shutil.which("npu-smi")` 落空。
- **修复**：三级定位 `$NPU_SMI_PATH` → PATH → 常见安装路径；首次成功/失败各打一行。

### #16 遥测自己同步设备流、破坏流水（`b229d91`）
- **症状**：P58 目标之一是「CPU/NPU 重叠」，但 `torch.npu.utilization()` 内部会
  **同步设备流** —— 每次采样都在训练热路径上砍一刀。
- **修复**：AI Core% 改走 `npu-smi`（子进程外部查询，零同步）；
  `memory_allocated` 是 host 侧计数器，保留。
- **教训**：**观测手段本身不能破坏被观测的流水线**。

---

## 3. 优化与否决类（负收益同样是结论）

### #8 融合核漏乘系数（`4ce0a48`）
- **症状**：自写 `learn_predictive` 融合核的逐位对拍 `max|Δ| = 0.4`。
- **根因**：两次外积都用了 `eta_pc`，**漏乘 `(1−mix)` 与 `mix`**。
- **修复**：补回系数（该核后来因 #10 整体删除）。
- **教训**：逐位对拍是有效的 —— 它在提交前抓到了这个 bug。

### #9 逐位对拍「假阳性」：基线用错实现（`4ce0a48`）
- **症状**：参考实现里用**纯 Python** matvec 当基线，报告 `dn0/dn1` 不一致。
- **根因**：fastmath 下重结合/FMA 决策与 numba 不同，**纯 Python 基线天然差 1 ulp**。
- **修复**：基线换成 numba 串行版 → `max|Δ| = 0`。
- **教训（P15 再次生效）**：**逐位对拍的基线必须是同一库、同一 fastmath 口径的
  串行实现**，否则满屏假阳性。

### #10 `learn_predictive` 融合：大范围负收益（`4ce0a48`）
- **数据**：26 万边 1.36× / 105 万边 **0.96×** / 419 万边 **0.91×**。
- **根因**：12 个 prange 段的线程调度成本 > 省下的 11 次核启动；191 核服务器只会更差。
- **决定**：**删除**（不留「备用」死代码），验证脚本改为回归护栏。

### #11 LTM 表向量化：负收益 0.73×（回滚，无提交）
- **数据**：predict 48/512/2048 行 = 1.00× / 0.92× / 1.36×（无规律）；learn 0.74×。
- **根因**：原版靠 `ltp <= 0` 短路让循环极便宜，预取/向量化**无条件付出**。
- **决定**：全部回滚。**分支密集的代码不要预先向量化。**

### #12 gather + prange 方案：拷贝吃掉收益（回滚，无提交）
- **数据**：800×800 组合，0.36 → 0.38 ms（0.96×）。
- **根因**：gather/scatter 的 O(R·m_out) 拷贝吃掉了并行收益；且多步序列逐位不一致
  （核内生长被预留容量卡住；误把结构可塑性绑到 `growth_guidance` —— 原路径生长
  只受 `size < m_out` 限制）。
- **决定**：回滚。要多核必须**在线维护连续 CSR 镜像**（零 gather），独立立项。

### #13 `encode` numba 化：O(n²) 输给 `set`（`720faf2`）
- **数据**：0.84×（多次跑 0.84–1.75× 波动）。
- **根因**：核内去重是线性扫描，Python `set` 是 O(1) 哈希。
- **决定**：调用方回退 Python；核函数留在 `ltm_kernel.py` 仅作记录。
  要重启需改成排序+相邻去重或小型开放寻址哈希表（参照 P18 词表合并）。

### #23 M4b 占 76%：`predict` 的 Python dict 遍历（`391fa06`）
- **症状**（18:00 服务器日志，token 7500）：`M4b_ltm 63.51 / 总 83.55 ms/tok`
  （**76%**）且随步数增长；`M2_infer` 已降到 1.70，其余段均 <1。
  `[ltm-diag] active=256，scores 72 → 308+`。
- **根因**：`OnlineCSRTable.predict` 遍历 256 个活跃行 × 约 72 槽 = **1.8 万次
  dict 更新 + numpy 标量取值/step** ≈ 8.8 ms（本机实测）× 每次 recall 触发
  → 与 63 ms 吻合。P68 只 numba 化了 recall 的**反投影**，dict 构造本身仍在。
- **修复**：`predict_arr`（numpy gather，返回 (keys, vals) **不合并重复键**）
  + 核内按「行序 → 槽位序」逐次累加（浮点顺序与原 `p[k] += v` 完全一致 →
  逐位相同）。**实测 predict 8.780 → 0.197 ms（44.6×）**，recall 投影 21.9×。
- **关键判断**：这一段**故意不上 numba gather 核** —— 1.8 万元素比 P67 的
  64 万组合小两个数量级，核的固定开销会吃掉收益（P11/P12/P13 三次负收益）。
- **门禁**：`verify_ltm_kernels.py` 扩到 10/10，覆盖 recall **全链路**逐位
  （int8/fp64 × 行满+键随机）。

### #15 fp32 权重 @ fp64 输入 → 脱离 BLAS 慢 5.9×（`0f203d5`）
- **症状**：M1 编码器改 fp32 后反而从 857 µs 变 5047 µs。
- **根因**：混合 dtype 让 numpy 离开 BLAS 的 sgemv，走逐元素慢路径。
- **修复**：输入一并 cast 到 fp32（n_in 个元素，代价可忽略）→ 606.7 → 337.7 µs（1.80×）。
- **推广**：numpy 路径上「bf16/fp16 存储 + fp32 迭代」必然更慢（每次付上采样
  转换）→ M1 默认 fp32，bf16 语义由读出侧（NPU 原生）承担。

### #19 计时误导：命中首项 / 规模猜错 54 倍（方法论）
- **症状一**：按 append 顺序填键 → `_find_slot` 总是命中首项（0.1 µs），
  把 numpy 优化测成负收益；打乱后才是平均情形。
- **症状二**：本机 800×800 组合 0.36 ms，服务器 65k 组合却 19.5 ms/tok
  —— **差 600 倍**，因为服务器 active 只有 256、每组合 300 ns（aarch64 的
  Python 解释器 + dict 访问更贵），不是组合数问题。
- **教训**：**计时必须复现「命中位置随机 + 行已满 + 真实规模」**；
  **本机复现与服务器差一个数量级时，所有本机优化结论作废**。

---

## 4. 观测与诊断类

### #14 `[ltm-diag]` 一行都不输出（`720faf2`）
- **症状**：加了 LTM 规模诊断，日志里完全没有。
- **根因**：门槛设成**每 1000 次 imprint**，而 imprint 是**条件触发**
  （`mode=="encode"` 且 gate 达标）→ 3500 token 攒不到 1000 次。
- **修复**：改每 10 次（fhz 指示）。
- **教训**：**诊断门槛要考虑触发频率**；宁可先打前 3 次。

### #17 「`--lang zh` 后还是英文」误判（`6b9a29e`）
- **症状**：fhz 报告加了 `--lang zh` 但日志仍是英文。
- **根因**：**那份日志的 argv 里根本没有 `--lang`**（只有 `--step-profiling`）。
  本机实测过滤本身有效（`lang='zh'` 出中文、`'en'` 出英文/代码）。
- **修复**：训练循环保留最近 8 个真实 target token，每个日志点打印
  `[sample lang=zh] CJK=98% | '…'` → 语种是否生效**一眼可见**。
- **教训**：**「行为没生效」先查命令行，再查代码**。

### #21 验证脚本里塞 `or True` 假断言（当场修）
- **症状**：写 `verify_pc_learn_fused.py` 时为了少写几行，加了 `... or True`。
- **修复**：自查时全部换成真断言。
- **教训**：**测试不能自己骗自己**；`or True` / `assert True` 等价于没测。

---

## 5. 数据与设计缺口

### #18 数据集语种与设计不符（中文仅 2%）— **待处理**
- **实测**（ModelScope `fhzfhz/Mixture-General-Mini` 的 52 个 pretrain 分片）：
  - 第 0–45 片全是 `infinity_m7core`，lang 分布 **en 50% / unk 47% / zh 仅 2%**；
  - **`unk` 其实是代码**（`assigned_fires = []`，CJK 占比 0%）；
  - 只有末尾第 51 片是 `ultrafineweb_l3_zh`（中文 99.7%）。
- **与设计冲突**：文档记录「M7_Core 中文 3,697 万块 ≈ 95%」。
- **后果**：`--lang zh` 过滤后只剩 ~2% 的行；`--remote-fraction 0.3` 取的前 16 片
  又全是 m7core → 「中文训练」实际只用到极小部分数据。
- **待办**：核对数据制备/上传环节（中文块是否没进分片，或 lang 推断把中文判成
  en/unk）。**这是数据问题，不是代码问题。**

### #20 `_append` 满行时 IndexError（潜在，生产有守卫）— **待修**
- **症状**：`_grow_row` 在 `len(keys[i]) == m_out` 时 `new_cap = min(len*2, m_out)`
  不增长 → `_append` 抛 `IndexError: index 72 out of bounds`。
- **触发条件**：`m_out` 不是 `grow_chunk` 的倍数（1B 档正是 72 / 64）。
- **为何没炸生产**：`learn` 的 `size < m_out` 守卫挡住了所有生产路径，仅内部误用触发。
- **待办**：给 `_append` 加清晰防御（满行时报错而非索引异常）。

---

## 6. 现存诊断点（有意保留，便于线上排障）

| 位置 | 输出 | 用途 |
|---|---|---|
| `phdnet/bigltm.py` | `[ltm-diag] imprints=… prev/cur/combos/rate_dims/rows` | LTM 规模与稀疏度演化（每 10 次） |
| `phdnet/bigltm.py` | `[ltm-diag] recalls=… active/scores/bindings` | 召回侧遍历量（每 10 次） |
| `phdnet/telemetry.py` | `GC <对象数>M/gen2 <次数>` | 验证/排除 GC 假设 |
| `phdnet/telemetry.py` | `[telemetry] npu-smi = <path>` / `⚠ 未找到` | 设备遥测可用性 |
| `train_1b/train.py` | `[gc] freeze() + threshold(...)` | GC 调优生效确认 |
| `train_1b/train.py` | `[sample lang=…] CJK=…% \| '…'` | 语种过滤是否生效 |
| `train_1b/train.py`（`--step-profiling`） | `segments: M1_encode … M4b_ltm …` | 九段耗时分解 |

诊断代码**不要删**——它们是下一次线上排障的唯一入口。若某项已确认无用
（如 GC 假设被证伪），保留成本极低（一次 `get_objects()` + 一个整数）。
