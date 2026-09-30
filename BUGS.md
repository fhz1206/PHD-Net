# BUGS.md — 缺陷与教训台账

> **数据截止**：2026-09-30。
> **每条结构**：症状 → 根因 → 修复 → **门禁缺口** → commit。
> **怎么用这份文档**：目的**不是**记流水账，而是让同类问题**下次被门禁提前拦住**。
> 因此每条都要看「门禁缺口」那一栏 —— 那里写的是**为什么 CI 没拦住**。
> 只看「症状/根因」学不到东西，只看「修复」会重犯。
>
> **范围约定**：只写「真实发生过的缺陷」。机制原理见 `docs/PHD-Net_架构设计.md`；
> **性能数字一律去 `docs/PHD-Net_性能评估与迭代方案.md`**（本文只在解释根因时引用，
> 不复制数字）；怎么加东西见 `docs/PHD-Net_扩展指南.md`。
> **没修的条目写明「未修 + 原因」，不留「应该会修」的空头。**

---

## 0. 索引（34 条，按主题分组）

### A. 训练入口与可选依赖边界（#1–#3、#22、#32）

| # | 症状 | 类型 | 提交 |
|---|---|---|---|
| 1 | 训练启动 `IndentationError` | 我引入 | `afdd5a2` |
| 2 | `train.py --help` 长期崩溃 | 既有遗留 | `afdd5a2` |
| 3 | fast 门禁 9/9 但训练入口是坏的 | **流程漏洞** | `afdd5a2` |
| 22 | 用 `grep` 过滤验证输出，把 Traceback 一起滤掉 | **流程漏洞** | `afdd5a2` |
| 32 | `to_numpy` 用了 `torch` 但 `model.py` 不 import torch | 我引入（当场修） | `4666870` |

### B. 读出与加速器（#4–#7、#16、#24、#30、#31、**#34**）

| # | 症状 | 类型 | 提交 |
|---|---|---|---|
| 34 | **fp8 能力表遗漏 → 默认 fp8 被静默回落，改造等于没生效** | **🔴 最严重** | `52499ff` |
| 4 | NPU/HBM 遥测恒为 `--` | 既有遗留 | `8b36f23` |
| 5 | `npu-smi` 找不到 → AI Core% 恒 `--` | 环境 | `0a0dfd8` |
| 6 | NPU 首次真跑训练即崩（`FakeTensor - None`） | 既有遗留 | `49aea05` |
| 7 | inductor 编译失败会崩生产（无回落） | 既有遗留 | `49aea05` |
| 16 | 遥测自己同步设备流、破坏流水 | 我引入 | `b00bb90` |
| 24 | NPU% 恒 `--`：解析器依赖该机没有的 Bus-Id 列 | 环境/解析 | `10f0396` |
| 30 | bf16 读出的检查点保存崩（numpy 无 bf16） | 我引入（bf16 默认） | `3e53fec` |
| 31 | P80 补丁静默 no-op → `_correct_pinned` 未定义 | 我引入（当场修） | `90b7b63` |

### C. M4b / LTM（#11–#14、#23、#26、#27）

| # | 症状 | 类型 | 提交 |
|---|---|---|---|
| 23 | M4b 占 76% ms/tok：`predict` 的 1.8 万次 dict 更新 | 性能（已修） | `f874db0` |
| 26 | 诊断行 `np.diff(rev_indptr)` 每次物化 134 MB | **我引入** | `5c4c303` |
| 27 | `_rev_to_csr` 构造期 1677 万次 Python 循环 | 我引入 | `5c4c303` |
| 11 | LTM 表向量化：负收益 0.73× | 优化否决 | 回滚（无提交） |
| 12 | gather+prange 方案：拷贝吃掉收益 | 优化否决 | 回滚（无提交） |
| 13 | `encode` numba 化：O(n²) 输给 `set` | 优化否决 | `7c5f21f` |
| 14 | `[ltm-diag]` 一行都不输出 | 我引入 | `7c5f21f` |

### D. 精度与 dtype（#8、#15、#20）

| # | 症状 | 类型 | 提交 |
|---|---|---|---|
| 8 | 融合核漏乘系数（写测试时抓出） | 我引入 | `ca8d39f` |
| 15 | fp32 权重 @ fp64 输入 → 脱离 BLAS 慢 5.9× | 性能陷阱 | `c8148b4` |
| 20 | `_append` 满行时 IndexError | 潜在（生产有守卫） | **未修** |

### E. 优化与否决（#9、#10）

| # | 症状 | 类型 | 提交 |
|---|---|---|---|
| 9 | 逐位对拍「假阳性」：基线用错实现 | **方法论** | `ca8d39f` |
| 10 | `learn_predictive` 融合：大范围负收益 | 优化否决 | `ca8d39f` |

### F. CI 与工程（#25、#28、#29、#33）

| # | 症状 | 类型 | 提交 |
|---|---|---|---|
| 33 | GitHub Actions 首跑即红（outputs / 数据集 / 依赖 import） | **CI 设计缺口** | `0a5ab0b`（仍红，待 traceback） |
| 25 | 编辑时删掉 `try` 的 `except` → SyntaxError | 我引入（当场修） | `87cdcbf` |
| 28 | BLAS 线程未限（191 线程跑小 sgemv） | 既有遗留 | `5c4c303` |
| 29 | 每步 `argmax` 扫全词表 + 重复 `stdp.predict` | 既有遗留 | `5c4c303` |

### G. 观测与方法论（#17、#18、#19、#21）

| # | 症状 | 类型 | 提交 |
|---|---|---|---|
| 18 | 数据集语种与设计不符（中文仅 2%） | **数据问题** | **未修**（待查） |
| 19 | 计时误导：命中首项 / 规模猜错 54 倍 | **方法论** | — |
| 17 | 「`--lang zh` 后还是英文」误判 | **观测错误** | `f0732e0` |
| 21 | 验证脚本里塞 `or True` 假断言 | 我引入 | 当场修 |

---

## 1. 训练入口与可选依赖边界

### #1 训练启动 `IndentationError`（`afdd5a2`）
- **症状**：服务器 `File "train_1b/train.py", line 385, IndentationError: unexpected indent`。
- **根因**：把 `else:` 分支改写成 `if not remote_active` 时，**原 else 里那行 `print` 变成孤立缩进块**；同一次编辑还把 `_data_provenance` 整段删掉（`save_model` 仍引用 → 即使缩进修好也会 NameError）。两处都是「**改结构不动引用**」。
- **修复**：恢复正确缩进 + 补回 `_data_provenance` 定义。
- **门禁缺口**：见 #3。

### #2 `train.py --help` 长期崩溃（`afdd5a2`）
- **症状**：`--help` 抛 `TypeError: must be real number, not dict`。
- **根因**：`--torch-compile` 的 help 文本里有**裸 `%`**（`measured 15% faster`）。argparse 用 `help % params` 格式化 → `% f` 要求 real number。**自 P45 起 `--help` 就一直是坏的** —— 训练不调 `format_help` 所以不炸。
- **修复**：改 `15%%`。
- **门禁缺口**：见 #3（新增的 `--help` 子进程断言当場抓住的同类问题）。

### #3 fast 门禁 9/9 但训练入口是坏的（`afdd5a2`）
- **症状**：#1/#2 都没被 `python tests/run_tests.py fast`（**9/9 全绿**）拦住。
- **根因**：fast 门禁只跑 `tests/` 下的检查，**从不 import `train_1b/train.py`**。
- **修复**：`tests/verifiers/verify_ms_stream.py` 增加两道真门禁 ——
  ① 对全部改动文件 `py_compile`；② subprocess 真跑 `train.py --help`，断言
  `returncode == 0` 且新参数出现在 stdout。
- **门禁缺口（仍存在）**：⚠ fast 门禁**本身**仍不 import 训练入口，靠 verifier 兜。
  **改 `train_1b/*.py` 后必须单独 `py_compile` + 跑一次 `--help`**，否则等于没测。

### #22 用 `grep` 过滤验证输出，把 Traceback 一起滤掉（`afdd5a2`）
- **症状**：`train.py --help | grep remote` 无输出，据此以为「参数已加上」，实际那次调用正在崩溃。
- **根因**：把「过滤输出」当成「断言成功」。退出码和 stderr 被管道吃掉了。
- **修复**：不再用 `grep` 判断成功，改为断言 exit code / 捕获 stderr。
- **门禁缺口**：**流程缺口，非代码缺口**。验证要断言，不要过滤 —— 这条和 #2 一起
  说明「`--help` 这条命令必须真的跑出来给人看」。

### #32 `to_numpy` 引用 torch 但模块不 import（`4666870`）
- **症状**：服务器检查点保存崩 `NameError: name 'torch' is not defined` @ `phdnet/model.py:519`。
- **根因**：P81 的 bf16 修复写了 `x.dtype == torch.bfloat16`，但 `phdnet/model.py`
  **从不 import torch**（torch 是**可选依赖**，只有读出 accel 路径才有张量）→ 纯 numpy
  训练路径上必然 NameError。
- **修复**：惰性 `import torch as _t`（有 bf16 张量 ⇒ torch 必然已装）。中途试过
  `x.view("uint16")` 字符串 dtype —— 本机 torch 版本不接受字符串。
- **门禁缺口**：⚠ **bf16 round-trip 用例有 torch，但 fast 门禁（9/9）没有** →
  「bf16 round-trip」应并入 fast 集合（未做）。

---

## 2. 读出与加速器

### 🔴 #34 fp8 能力表遗漏 → 默认 fp8 被静默回落，改造在生产等于没生效（`52499ff`）
> **本台账里严重度最高的一条**：不是崩溃、不是变慢，而是**一整轮精��改造在生产上完全没生效**，
> 且**不报任何错**。

- **症状**：P84 把 `--readout-dtype` 默认改成 fp8（forward fp8 副本 + fp16 更新）后，
  生产训练日志里读出后端一直是 `numba-cpu(回落)`，而 `AccelReadout` 的 fp8 代码路径
  从未被执行。无异常、无告警，fp8 一直「配置正确」。
- **根因**：`phdnet/backends/accel_readout.py::_unsupported_reason()` 是一张
  **「加速读出未实现项」的黑名单**，其中仍列着 `fp8`（P19 写下、P84 实现后**没同步删**）。
  链路是：`--readout-dtype fp8`（默认）→ `_unsupported_reason()` 返回非 None →
  `pick_readout_backend()` 走 `_fallback(...)` → 返回 numba CPU 原路径 +
  `_accel_fallback_reason`。
  为什么难以发现：回落**本来就是设计好的安全行为**（不静默丢功能），原因只写在一个
  属性里、启动日志打一行、不 raise、不改退出码。
- **修复**（`52499ff`）：从能力表移除 fp8，并在 docstring 里写明教训与检查清单 ——
  「**加新 dtype 时务必同步这张能力表**」。
- **门禁缺口（关键）**：⚠ **没有任何测试断言「配置声明支持的后端 = 实际选中的后端」**。
  现有 verifier 都是「跑通即可」，而回落后 numba CPU 路径**也是跑通的**。
  这类缺陷对测试不可见，只能靠「能力表与实现逐条对账」人工抓。
  **应有的门禁**：一张表驱动测试 —— 枚举 `cfg.readout_dtype` 的全部合法取值 ×
  各后端能力开关，断言「声明支持 ⇒ `_unsupported_reason()` 返回 None」；
  再加一条生产侧断言：启动日志若出现 `fallback reason`，**非显式 `--accel cpu` 时应当 fail-fast 或至少醒目告警**。

### #4 NPU/HBM 遥测恒为 `--`（`8b36f23`）
- **症状**：日志 `NPU/GPU --%  HBM --/--GB`，但服务器确有 NPU。
- **根因**：`Telemetry()` 一直**无参构造** → `device=""` → `sample()` 的加速器分支**永不触发**（P41 遗留；P19 把读出搬上 NPU 后没人回补遥测）。
- **修复**：自动探测 device（`torch.npu` → `torch.cuda`，显式传入仍优先）；首次采样失败打印原因（原来 `except: pass` 全吞）；`TeeLogger` 接管 `sys.stderr`（回落告警 / inductor 日志此前不落盘）。
- **门禁缺口**：`verify_ms_stream.py` 加了「无 NPU 机器探测为空且行为不变」的断言。
  ⚠ 仍未覆盖「有 NPU 的机器上遥测非空」——本机无法验证，只能在服务器侧人工确认。

### #5 `npu-smi` 找不到 → AI Core% 恒 `--`（`0a0dfd8`）
- **症状**：HBM 有值（走 `torch.npu.memory_allocated()`，host 侧计数器）但 AI Core% 没有。
- **根因**：AI Core% 走 `npu-smi`，而训练是**直接 `python train.py` 启动**的，没 source Ascend 的 `set_env.sh` → `shutil.which("npu-smi")` 落空。
- **修复**：三级定位 `$NPU_SMI_PATH` → PATH → 常见安装路径；首次成功/失败各打一行。
- **门禁缺口**：路径解析是纯函数、可测，但**「这台机器上真实存在哪个路径」不可测**。
  已在 `_parse_npu_smi` 失败时打印原始输出前 14 行，便于对齐真实格式。

### #6 NPU 首次真跑训练即崩（`FakeTensor - None`）（`49aea05`）
- **症状**：`TorchRuntimeError: ... unsupported operand type(s) for -: 'FakeTensor' and 'NoneType'`，崩在 `accel_readout.py` 的 `dp = (p - t32)`。
- **根因**：P45 的「`target_idx` 路径就地改 p」只实现在 **eager 分支**，`torch.compile`融合核 `_train_step_core` 漏改，而该路径 `t32 = None`。既有 `verify_accel_readout.py` 的 learn_softmax 用例**只走 target 数组路径**，从不传 `target_idx` → 恰好漏掉崩的那条。
- **修复**：融合核逐行镜像 eager（`p − onehot` 与 `p[c] −= 1` 数值等价）；顺带把 softmax/nll 移入 eager-only（compiled 路径原白算一次全词表 softmax）。
- **门禁**：新增 `tests/verifiers/verify_accel_readout_p55.py`（2×2 矩阵：target 数组 / target_idx × eager / compiled，**9/9**）。

### #7 inductor 编译失败会崩生产（无回落）（`49aea05`）
- **症状**：`AccelReadout.__init__` 有 try/except 回落，但**首次调用 `_fused` 才真正触发 inductor 编译** → Windows 无 C++ 编译器时直接抛 `InductorError`。
- **根因**：构造期的 `torch.compile(...)` 只是包装，不会失败；**回落写在了错误的层**。
- **修复**：首次执行失败 → **永久回落 eager + `warnings.warn` 记原因**（P19「回落 + 记原因」纪律）。
- **门禁缺口**：⚠ 无法在有编译器的机器上自然复现这条；现有门禁只覆盖 eager/compiled
  都**成功**的情况。

### #16 遥测自己同步设备流、破坏流水（`b00bb90`）
- **症状**：P58 目标之一是「CPU/NPU 重叠」，但 `torch.npu.utilization()` 内部会**同步设备流** —— 每次采样都在训练热路径上砍一刀。
- **修复**：AI Core% 改走 `npu-smi`（子进程外部查询，零同步）；`memory_allocated` 是 host 侧计数器，保留。
- **教训**：**观测手段本身不能破坏被观测的流水线**。

### #24 NPU% 恒 `--`：解析器依赖该机没有的 Bus-Id 列（`10f0396`）
- **症状**：`npu-smi` 路径已定位到 `/usr/local/bin/npu-smi`，但日志仍是 `NPU/GPU --%`。
- **根因**：`_parse_npu_smi` 找「含 `0x` 总线号的数据行」—— 该机输出没有这一列 → 所有行被跳过。
- **修复**：改为**按表头定位列**（找含 `AICore` 的表头行 → 推出 AICore / HBM-Usage 列序 → 按列号取值），与型号无关；失败时打印原始输出前 14 行一次。
- **门禁**：`verify_ms_stream.py` 加了两种布局（有/无 Bus-Id、小写表头）的解析断言。

### #30 bf16 读出的检查点保存崩溃（`3e53fec`）
- **症状**：服务器 `save_model` → `to_numpy(ro_W)` → `TypeError: Got unsupported ScalarType BFloat16`。
- **根因**：`readout_dtype=bf16` 使设备张量是 bf16，而 **numpy 无原生 bf16** → `.numpy()` 直接抛。bf16 检查点路径此前从未走到「保存」这一步。
- **修复**：`to_numpy` 对 bf16 张量存 **uint16 位模式**（`view(torch.uint16)`），与加载侧 P46 的解码（`ckpt_dtype=bf16` + `dtype.kind in "ui"` → `view(bfloat16)`）正好闭环 —— 无损，且检查点保持 bf16 体积（不膨胀成 fp32）。
- **门禁**：`verify_ms_stream.py` 增加 bf16 round-trip 用例。

### #31 P80 补丁静默 no-op → `_correct_pinned` 未定义（`90b7b63`）
- **症状**：`AttributeError: 'AccelReadout' object has no attribute '_correct_pinned'`。
- **根因**：P80 用字符串 replace 加 `self._correct_pinned = None`，匹配串带了一段**注释后缀**，实际文件里没有 → **replace 无匹配、静默成功**，`print('patched')` 照常打印 → 我以为生效了。
- **修复**：初始化放进 `__init__`（精确匹配实际文本）+ 调用侧 `getattr` 防御兜底。
- **门禁/流程缺口**：⚠ **replace 类补丁必须 grep 断言改后文本存在**，否则静默失败。
  与 #25（Edit 截断 except）、#27（第一次改法被对拍抓出段序错位）同属
  **「补丁未真正落盘 / 落盘了一半」家族**，是本台账里最高频的一类自我缺陷。

---

## 3. M4b / LTM

### #23 M4b 占 76%：`predict` 的 Python dict 遍历（`f874db0`）
- **症状**（服务器日志）：`M4b_ltm 63.51 / 总 83.55 ms/tok`（**76%**）且随步数增长；`M2_infer` 已降到 1.70，其余段均 <1。`[ltm-diag] active=256，scores 72 → 308+`。
- **根因**：`OnlineCSRTable.predict` 遍历 256 个活跃行 × 约 72 槽 = **1.8 万次dict 更新 + numpy 标量取值/step**，每次 recall 触发一次 → 与 63 ms 吻合。P68 只 numba 化了 recall 的**反投影**，dict 构造本身仍在。
- **修复**：`predict_arr`（numpy gather，返回 `(keys, vals)` **不合并重复键**）+ 核内按「行序 → 槽位序」逐次累加（浮点顺序与原 `p[k] += v` 完全一致 → 逐位相同）。**实测 predict 8.780 → 0.197 ms（44.6×）**，recall 投影 21.9×。
- **关键判断**：这一段**故意不上 numba gather 核** —— 1.8 万元素比 P67 的 64 万组合小两个数量级，核的固定开销会吃掉收益（P11/P12/P13 三次负收益的教训）。
- **门禁**：`verify_ltm_kernels.py` 扩到 **10/10**，覆盖 recall **全链路**逐位（int8/fp64 × 行满+键随机）。

### #26 诊断行 `np.diff(rev_indptr)` 每次物化 134 MB（我引入，`5c4c303`）
- **症状**：每 8 步一次的 recall 诊断，在 1B 档白烧约 6 ms/tok。
- **根因**：P68 给 recall 加诊断时写的 `np.diff(self._rev_indptr)` —— `rev_indptr` 按**大空间最大 key** 铺开，1B 档长 16,774,731（**134 MB**），每次 recall 完整物化（本机 33 ms）**只为打印一行 `bindings` 数**。
- **修复**：只在打印分支内计算 + 改 `(indptr[v+1]-indptr[v]).sum()`（O(|valid|)，数值完全相同）。
- **教训**：**诊断代码也在热路径上**。打印一行数字不等于零成本；诊断必须和它产出的数字一样便宜。
- **门禁缺口**：⚠ 无。这条本质是「写诊断时没看数据结构的规模」，靠 review 抓。
  与 #14（诊断门槛没考虑触发频率）同属「诊断本身的纪律」类。

### #27 `_rev_to_csr` 构造期 1677 万次 Python 循环（我引入，`5c4c303`）
- **症状**：1B 档启动多花约 10 s。
- **根因**：`_rev_to_csr()` 构造期 `for i in range(n_keys)` 逐个查 dict → 1B 档 **1677 万次 Python 循环**。
- **修复**：只遍历实际存在的 key（≤ 4096 条）+ 向量化前缀和，**保持 key 升序**（与原实现段序一致 → 逐位相同）。
- **⚠ 修复过程中自己制造又抓到一次 bug**：第一次改成「dict 插入序 + 生成器」时`indices` 与 `indptr` 段序错位，被 `verify_ltm_kernels.py` 的「rev CSR 逐维一致」用例当场抓出。
- **教训**：这就是逐位对拍的价值 —— **优化「形状/顺序」的代码时，对拍是唯一防线**（纯计时看不出段序错位，结果仍「跑得动」）。
- **门禁**：靠 `verify_ltm_kernels.py` 既有对拍用例，无需新增。

### #11 LTM 表向量化：负收益 0.73×（回滚，无提交）
- **数据**：predict 48/512/2048 行 = 1.00× / 0.92× / 1.36×（**无规律**）；learn 0.74×。
- **根因**：原版靠 `ltp <= 0` 短路让循环极便宜，预取/向量化**无条件付出**。
- **决定**：全部回滚。**分支密集的代码不要预先向量化。**

### #12 gather + prange 方案：拷贝吃掉收益（回滚，无提交）
- **数据**：800×800 组合，0.36 → 0.38 ms（0.96×）。
- **根因**：gather/scatter 的 O(R·m_out) 拷贝吃掉了并行收益；且多步序列**逐位不一致**（核内生长被预留容量卡住；误把结构可塑性绑到 `growth_guidance` —— 原路径生长只受 `size < m_out` 限制）。
- **决定**：回滚。要多核必须**在线维护连续 CSR 镜像**（零 gather），独立立项。
  ⚠ **该立项至今未做** —— M4b 的 imprint（learn）侧仍是 Python 循环，是当前 LTM 侧
  最大的未修项（性能文档里的「已知残余」即此）。

### #13 `encode` numba 化：O(n²) 输给 `set`（`7c5f21f`）
- **数据**：0.84×（多次跑0.84–1.75× 波动）。
- **根因**：核内去重是线性扫描，Python `set` 是 O(1) 哈希。
- **决定**：调用方回退 Python；核函数留在 `ltm_kernel.py` 仅作记录。要重启需改成排序+相邻去重或小型开放寻址哈希表（参照 P18 词表合并）。
- **门禁缺口**：⚠ 无计时护栏 —— 「回退了所以没人会再误用」，但核函数还在文件里，
  下次有人看到它会以为它可用。

### #14 `[ltm-diag]` 一行都不输出（我引入，`7c5f21f`）
- **症状**：加了 LTM 规模诊断，日志里完全没有。
- **根因**：门槛设成**每 1000 次 imprint**，而 imprint 是**条件触发** → 3500 token 攒不到 1000 次。
- **修复**：改每 10 次。
- **教训**：**诊断门槛要考虑触发频率**；宁可先打前 3 次。（配套纪律：`[ltm-diag]` / `[sample lang=]` / `[numba] cache` 现在都是「低频 + 常开」，见 §4。）

---

## 4. 精度与 dtype

### #8 融合核漏乘系数（写测试时抓出）（`ca8d39f`）
- **症状**：自写`learn_predictive` 融合核的逐位对拍 `max|Δ| = 0.4`。
- **根因**：两次外积都用了 `eta_pc`，**漏乘 `(1−mix)` 与 `mix`**。
- **修复**：补回系数（该融合核后来因 #10 整体弃用）。
- **教训**：**逐位对拍是有效的** —— 它在提交前抓到了这个 bug。这是本台账里
  「投资回报率最高」的一类门禁。

### #15 fp32 权重 @ fp64 输入 → 脱离 BLAS 慢 5.9×（`c8148b4`）
- **症状**：M1 编码器改 fp32 后反而从 857 µs 变 5047 µs。
- **根因**：混合 dtype 让 numpy **离开 BLAS 的 sgemv**，走逐元素慢路径。
- **修复**：输入一并 cast 到 fp32（n_in 个元素，代价可忽略）→ 606.7 → 337.7 µs（1.80×）。
- **推广**：numpy 路径上「bf16/fp16 存储 + fp32 迭代」必然更慢（每次付上采样转换）→ 该平台默认 fp32，半精度语义由读出侧承担。
- **⚠ 后续反转**：本条结论在 **aarch64/昇腾 上被推翻** —— 那里 fp32 sgemv 反而**慢约 70×**，
  故 `--encoder-dtype` 默认值改为 fp64。**教训：dtype 优化的方向依赖平台，
  x86 的结论不构成昇腾的证据**（详见 `docs/PHD-Net_性能评估与迭代方案.md` 的平台差异表）。

### #20 `_append` 满行时 IndexError — **未修**（潜在，生产有守卫）
- **症状**：`_grow_row` 在 `len(keys[i]) == m_out` 时 `new_cap = min(len*2, m_out)` 不增长 → `_append` 抛 `IndexError: index 72 out of bounds`。
- **触发条件**：`m_out` 不是 `grow_chunk` 的倍数（1B 档正是 72 / 64）。
- **为何没炸生产**：`learn` 的 `size < m_out` 守卫挡住了所有生产路径，仅内部误用触发。
- **未修原因**：当前**无法从生产路径触发**，改它需要动 LTM 的生长边界（#12 已判定该区域
  近期要整体重做），单独加防御收益低于风险。**待办**：随在线 CSR 镜像立项时，
  给 `_append` 加「满行时报错而非索引异常」的清晰防御。

---

## 5. 优化与否决（负收益同样是结论）

### #9 逐位对拍「假阳性」：基线用错实现（`ca8d39f`）
- **症状**：参考实现里用**纯 Python** matvec 当基线，报告 `dn0/dn1` 不一致。
- **根因**：fastmath 下重结合 / FMA 决策与 numba 不同，**纯 Python 基线天然差 1 ulp**。
- **修复**：基线换成numba 串行版 → `max|Δ| = 0`。
- **教训（多次生效）**：**逐位对拍的基线必须是同一库、同一 fastmath 口径的串行实现**，
  否则满屏假阳性。假阳性比没测试更危险 —— 它会让人去「修」正确代码。

### #10 `learn_predictive` 融合：大范围负收益（`ca8d39f`）
- **数据**：26 万边 1.36× / 105 万边 **0.96×** / 419 万边 **0.91×**。
- **根因**：12 个 prange 段的线程调度成本 > 省下的 11 次核启动；191 核服务器只会更差。
- **决定**：**不调用**融合版（保留原多核调用路径），`sparse_pc.py` 里留注释记录实测数据
  与判定；验证脚本改为回归护栏。
- **⚠ 与旧记述的差异**：早期记为「整体删除」，实际是**保留函数、停止调用**
  （`verify_pc_learn_fused.py` 的回归护栏仍覆盖它）。删除会让护栏失去被测对象。
- **教训**：#11/#12/#13/#10 四次连续负收益，共同结论是——
  **在访存/调度受限的规模上，加并行度不是免费午餐**。

---

## 6. CI 与工程

### #33 GitHub Actions首跑即红 — **仍红**（`0a5ab0b`）
- **症状**：首次推送即三个 step 失败：`outputs` 目录不存在 / 数据集缺失 / 依赖 import 失败。
- **根因**：**CI 设计缺口** —— workflow 直接复用了「本地已存在」的假设（已生成的
  `outputs/`、已上传的 datasets、只在本机装好的可选依赖）。
- **修复**（分三步）：`82c7390` 修首次失败 → `3a6afc8` 加失败诊断（env + 完整 traceback 重跑）
  → `0a5ab0b` 钉 `numpy<2.3`（numba import 失败）并让诊断步骤容错。
- **未修原因 / 现状**：⚠ **两套 CI 中 GitHub Actions 仍红**（`.github/workflows/ci.yml`），
  **无日志权限，拿不到 traceback**，只能看到匿名可读的 check-run annotations。
  GitCode 侧（`.gitcode/workflows/ci.yml`）正常。**不要声称 CI 全绿。**
- **门禁缺口**：诊断步骤容错 = 失败时也能拿到日志，但拿不到日志时**没有任何自动升级路径**。
  这是**工具链权限问题**，不是代码问题。

### #25 编辑时删掉 `try` 的 `except` → SyntaxError（当场修，`87cdcbf`）
- **症状**：加主循环计时时报 `SyntaxError: expected 'except' or 'finally'`。
- **根因**：那次 Edit 的 `new_string` 只写到 `print(...)`，**把配对的 `except Exception` 块截掉了** —— 与 #1（孤立缩进块）同源的「改结构不动配对」。
- **修复**：补回 `except` 块。
- **门禁**：本次 `py_compile` 立刻抓到 → **印证 #3 新增的语法门禁有效**。
  这是 #3 那道门禁的第一个真实战果。

### #28 BLAS 线程未限（191 线程跑小 sgemv）（既有遗留，`5c4c303`）
- **症状**：191 核机器上主循环只用 1.1–3.2 核、CS/s 250 万+。
- **根因**：`OPENBLAS/MKL/OMP_NUM_THREADS` 从未限上限。P62 只限了 numba prange 线程 → OpenBLAS 拉 **191 线程**去跑 1024×2048 的 sgemv（M1 编码器）。仓库里早就有 `multi_device.configure_host_threads()`，但**无生产调用点** —— 典型的「**修了一半**」。
- **修复**：在 **`import numpy` 之前**设四个环境变量（OpenBLAS 初始化后再设无效）。
- **门禁缺口**：⚠ 无法在 CI 上验证（CI 机器核数不同）。「限线程」这类修复的正确性
  **依赖部署机器的核数**，只能人工确认。相关默认值见
  `train_1b/README.md` §2.2（`--numba-threads` / `--omp-proc-bind`）。

### #29 每步 `argmax` 扫全词表 + 重复 `stdp.predict`（既有遗留，`5c4c303`）
- **症状**：两个每步可省的 CPU 开销。
- **根因**：①每步 `int(np.argmax(target))` 扫全词表（可达 73,958 元素），而调用方
  `stoi[t0]` 就在手边 → 改为 `step(..., target_idx=...)` 形参（零风险，且兑现 P40 注释里
  「设备侧 onehot 路径」的技术债）；②`stdp.predict` 每步被调**两次且输入是同一对象**
  → 训练路径复用，⚠ 但 `readonly=True` 时两者确实不同 → **保留原调用**。
- **门禁缺口**：无专项门禁；靠 `verify_ms_stream.py` 覆盖 `target_idx` 路径不被改坏
  （与 #6 的 2×2 矩阵同一套）。

**#26–#29 的来源**：两个子代理分别审计 `phdnet/` 机制层与 `train_1b/` 主循环，
交叉验证出 4 条，**其中最严重的两条（#26/#27）是我自己前一版引入的**。
审计的价值不只在于抓到别人的 bug，也在于抓到自己的。

**未采纳（记录理由）**：`learn` 侧 numba 化需先建逐位门禁（当前无覆盖）；
`_ensure` 的 O(n²) 字符串切片（收益小、需改 tokenize 状态机）。

---

## 7. 观测与方法论

### #18 数据集语种与设计不符（中文仅 2%）— **未修**（待查）
- **实测**（ModelScope `fhzfhz/Mixture-General-Mini` 的 52 个 pretrain 分片）：
  - 第 0–45 片全是 `infinity_m7core`，lang 分布 **en 50% / unk 47% / zh 仅 2%**；
  - **`unk` 其实是代码**（`assigned_fires = []`，CJK 占比 0%）；
  - 只有末尾第 51 片是 `ultrafineweb_l3_zh`（中文 99.7%）。
- **与设计冲突**：设计记录「M7_Core 中文 3,697 万块 ≈ 95%」。
- **后果**：`--lang zh` 过滤后只剩 ~2% 的行；`--remote-fraction 0.3` 取的前 16 片又全是
  m7core → 「中文训练」实际只用到极小部分数据。
- **未修原因**：**这是数据制备/上传环节的问题，不是代码问题** —— 代码侧过滤逻辑本机实测
  有效（`lang='zh'` 出中文）。必须先核对数据制备与上传环节（中文块是否没进分片，
  或 lang 推断把中文判成 en/unk）。**在查清之前不要重新采样覆盖现有分片**（会丢证据）。
- **门禁缺口**：⚠ **无任何门禁会发现「语料分布与设计不符」**。`[sample lang=]` 能验证
  「过滤生效了」，但验证不了「过滤后的语料足够中文」。

### #19 计时误导：命中首项 / 规模猜错 54 倍（方法论，无提交）
- **症状一**：按 append 顺序填键 → `_find_slot` 总是命中首项（0.1 µs），把 numpy 优化
  测成负收益；打乱后才是平均情形。
- **症状二**：本机 800×800 组合 0.36 ms，服务器 65k 组合却 19.5 ms/tok —— **差 600 倍**，
  因为服务器 active 只有 256、每组合 300 ns（aarch64 的 Python 解释器 + dict 访问更贵），
  **不是组合数问题**。
- **教训**：**计时必须复现「命中位置随机 + 行已满 + 真实规模」**；
  **本机复现与服务器差一个数量级时，所有本机优化结论作废**。
  配套：`--step-profiling` 的九段 + 主循环三段就是为「真实规模下测量」建的。

### #17 「`--lang zh` 后还是英文」误判（`f0732e0`）
- **症状**：报告加了 `--lang zh` 但日志仍是英文。
- **根因**：**那份日志的 argv 里根本没有 `--lang`**（只有 `--step-profiling`）。本机实测过滤本身有效。
- **修复**：训练循环保留最近 8 个真实 target token，每个日志点打印
  `[sample lang=zh] CJK=98% | '…'` → 语种是否生效**一眼可见**；并在日志头打印完整
  `argv`（`[log] start: … argv: …`）。
- **教训**：**「行为没生效」先查命令行，再查代码**。新增观测手段要能回答
  「我以为传了的那个参数，到底传了没」。

### #21 验证脚本里塞 `or True` 假断言（我引入，当场修）
- **症状**：写 `verify_pc_learn_fused.py` 时为了少写几行，加了 `... or True`。
- **修复**：自查时全部换成真断言。
- **教训**：**测试不能自己骗自己**；`or True` / `assert True` 等价于没测。

---

## 8. 现存诊断点（有意保留，便于线上排障）

**不要删这些** —— 它们是下一次线上排障的唯一入口。若某项已确认无用，保留成本极低
（一次 `get_objects()` + 一个整数）。

| 位置 | 输出 | 用途 |
|---|---|---|
| `phdnet/bigltm.py` | `[ltm-diag] imprints=… prev/cur/combos/rate_dims/rows/k_hash` | LTM 规模与稀疏度演化（每 10 次） |
| `phdnet/bigltm.py` | `[ltm-diag] recalls=… active/scores/bindings` | 召回侧遍历量（每 10 次） |
| `phdnet/telemetry.py` | `GC <对象数>M/gen2 <次数>`（遥测行内） | 验证/排除 GC 假设 |
| `phdnet/telemetry.py` | `[telemetry] npu-smi = <path>` / `[telemetry] ⚠ 未找到` / `[telemetry] npu-smi parse FAILED; raw head://…` | 设备遥测可用性与解析失败取证 |
| `train_1b/train.py` | `[gc] freeze() + threshold(50000, 200, 200); tracked objects = …` | GC 调优生效确认 |
| `train_1b/train.py` | `[sample lang=…] CJK=…% \| '…'` | 语种过滤是否真生效（#17/#18 的观测入口） |
| `train_1b/train.py` | `[numba] cache dir = … \| size = … MB \| 预期启动 ~2.6 s` | numba 持久缓存是否真被命中（`cache=True` 有没有被漏掉） |
| `train_1b/train.py`（`--step-profiling`） | `segments: M1_encode … M4b_ltm …` | 九段耗时分解（按耗时降序取前 6） |
| `train_1b/train.py`（`--step-profiling`） | `loop: tokenize … encode_onehot … step …` | **主循环三段** —— `net.step` 之外的开销 |

日志字段的完整速查表见 `train_1b/README.md` §7.1。

---

## 9. 结构性教训汇总

1. **「改结构不动配对」是本台账最高频的自我缺陷族**：#1（孤立缩进块 + 删定义）、
   #25（截断 `except`）、#27（第一次优化改法段序错位）、#31（replace 无匹配静默成功）。
   → 对策已落地：`py_compile` 门禁 + 「replace 类补丁必须 grep 断言改后文本存在」。
2. **诊断代码也在热路径上**：#26（134 MB/次只为打一行）、#14（门槛没考虑触发频率）、
   #16（观测手段本身破坏流水线）。
3. **配置声明与实际实现必须对账**：#34（能力表遗漏 → 一整轮改造静默失效）是本台账里
   最严重的一条，因为**它没有任何报错**。
4. **负收益同样是结论**：#10/#11/#12/#13 四次连续否决，都因为「访存/调度受限规模上
   加并行度要先算固定开销」。
5. **x86 的性能结论不构成昇腾的证据**：#15 的结论在两个平台上**方向相反**。
6. **门禁只能拦住「门禁想得到的东西」**：#18（语料分布错）与 #33（CI 仍红）都超出现有
   9 项 fast 门禁的视野。
