# BUGS.md — 缺陷与教训台账

> **数据截止**：2026-09-30（对应 P94）。**每条结构**：症状 → 根因 → 修复 → **门禁缺口**（CI 为什么没拦住） → commit。
> **怎么用**：目的**不是**记流水账，而是让同类问题**下次被门禁提前拦住**。只看症状/根因学不到东西，
> 只看修复会重犯 —— 价值在「门禁缺口」那一栏。
> **范围**：只写真实发生过的缺陷。机制原理见 `docs/PHD-Net_架构设计.md`；**性能数字一律去
> `docs/PHD-Net_性能评估与迭代方案.md`**（本文只在解释根因时引用，不复制）；加东西见
> `docs/PHD-Net_扩展指南.md`；后端与昇腾踩坑见 `docs/PHD-Net_硬件后端适配报告.md`。
> **未修的条目写明「未修 + 原因」**，不留「应该会修」的空头。
> **commit 纪律**：历史经过 force push 改写，旧 hash 已失效；此类条目写「commit 见 <P 号的提交信息>」，不编造 hash。

**索引（52 条）**：A 训练入口与可选依赖边界 A1–A9（9）｜B 读出与加速器 B1–B17（17）｜
C M4b/LTM C1–C7（7）｜D 精度与 dtype D1–D7（7）｜E 否决与观测方法论 E1–E6（6）｜
F CI 与工程 F1–F4（4）｜G 环境与数据 G1–G2（2）。
**最严重的两条是 B3 与 B12**：都不是崩溃、不是变慢，而是**一整轮改造在生产上完全没生效且不报任何错**
（B3 = fp8 能力表遗漏；B12 = 稀疏读出默认值让 NPU 读出加速全程回落 CPU）。
**最隐蔽的一条是 B13**：半 ULP 的数量级估算代替实测，默认值改了**两次都是错的**。
**最容易反复误判的一条是 B14**：numba.cuda 确实存在，但硬绑 NVIDIA 驱动。

---

## A. 训练入口与可选依赖边界

### A1 训练启动 `IndentationError`，同一次编辑还删掉了被引用的定义（`afdd5a2`）
- **症状**：服务器 `File "train/train.py", line 385, IndentationError: unexpected indent`。
- **根因**：把 `else:` 分支改写成 `if not remote_active` 时，原 else 里那行 `print` 变成孤立缩进块；**同一次编辑**把 `_data_provenance` 整段删掉（`save_model` 仍引用 → 缩进修好也会 NameError）。两处同源：**改结构不动引用**。
- **修复**：恢复正确缩进 + 补回 `_data_provenance` 定义。**门禁缺口**：fast 门禁 9/9 全绿仍未拦住 → 见 A3。

### A2 `train.py --help` 长期崩溃（`afdd5a2`）
- **症状**：`--help` 抛 `TypeError: must be real number, not dict`。
- **根因**：`--torch-compile` 的 help 文本里有**裸 `%`**（`measured 15% faster`）。argparse 用 `help % params` 格式化 → `% f` 要求 real number。**自 P45 起 `--help` 一直坏着** —— 训练不调 `format_help` 所以不炸。
- **修复**：改 `15%%`。**门禁缺口**：A3 新增的 `--help` 子进程断言当场抓住的同类问题。

### A3 fast 门禁 9/9 但训练入口是坏的（**流程漏洞**，`afdd5a2`）
- **症状**：A1/A2 都没被 `python tests/run_tests.py fast` 拦住。
- **根因**：fast 门禁只跑 `tests/` 下的检查，**从不 import `train/train.py`**。
- **修复**：`tests/verifiers/verify_ms_stream.py` 加两道真门禁 —— ① 对全部改动文件 `py_compile`；② subprocess 真跑 `train.py --help`，断言 `returncode == 0` 且新参数出现在 stdout。
- **门禁缺口（仍存在）**：⚠ fast 门禁**本身**仍不 import 训练入口，靠 verifier 兜。**改 `train/*.py` 后必须单独 `py_compile` + 跑一次 `--help`**，否则等于没测。

### A4 编辑时截断 `try` 的配对 `except` → SyntaxError（`87cdcbf`）
- **症状**：加主循环计时时报 `SyntaxError: expected 'except' or 'finally'`。
- **根因**：那次 `Edit` 的 `new_string` 只写到 `print(...)`，**把配对的 `except Exception` 块截掉了**。
- **修复**：补回 `except` 块。**门禁**：本次 `py_compile` 立刻抓到 → **A3 那道门禁的第一个真实战果**，与 A1 同源。

### A5 字符串 replace 无匹配是**静默成功**（连续两次；本组高频项）
- **症状一**（`aa51bab`）：P77 加的 M4b 计时拆分**没有生效**，M4b_ltm 整段仍在计时里。
- **症状二**（`90b7b63`）：P80 用字符串 replace 加的 `self._correct_pinned = None` 没落盘，`print('patched')` 照常打印。
- **根因**：`str.replace` 的 needle 带了**实际文件中不存在的注释后缀** → 无匹配、零副作用、退出码 0。与 A1/A4 同属「**补丁未真正落盘 / 落盘了一半**」家族。
- **修复**：A5 一补上缺失的 `M4b_imprint` 内层分段；A5 二见 A6。**门禁缺口（流程）**：⚠ **replace 类补丁必须 `grep` 断言改后文本存在**。本组已连续踩 4 次（A1/A4/A5×2），是本台账最高频的自我缺陷族。

### A6 `_correct_pinned` 未定义 → `AttributeError`（`90b7b63`）
- **症状**：服务器 `AttributeError: 'AccelReadout' object has no attribute '_correct_pinned'` in `_eager_step`。
- **根因**：A5 二的补丁静默 no-op，构造期从未定义该属性。
- **修复**：初始化放进 `__init__`（**精确匹配实际文本**）+ 调用侧改 `getattr(...)` 惰性初始化兜底 —— 任何未来漏掉初始化的构造路径都降级为惰性 pinned buffer，而不是 AttributeError。
- **门禁缺口**：⚠ 无。属性缺失只在**服务器真跑**时暴露；fast 门禁不 import 训练入口，也没有「构造后所有 `_eager_step` 依赖属性都存在」这类反射断言。

### A7 `to_numpy` 引用 torch，但模块从不 import torch（P83b，`4666870`）
- **症状**：服务器检查点保存崩 `NameError: name 'torch' is not defined` @ `phdnet/model.py:519`。
- **根因**：P81 的 bf16 修复写了 `x.dtype == torch.bfloat16`，而 `phdnet/model.py` **从不 import torch**（torch 是**可选依赖**，只有读出 accel 路径才有张量）→ 纯 numpy 训练路径必然 NameError。
- **修复**：惰性 `import torch as _t`（有 bf16 张量 ⇒ torch 必然已装）。中途试过 `x.view("uint16")` 字符串 dtype —— 本机 torch 版本不接受字符串。
- **门禁缺口**：⚠ **bf16 round-trip 用例有 torch，但 fast 门禁（9/9）没有** → 本地全绿、服务器崩。该用例应并入 fast 集合（未做）。

### A8 P80 的 `F.cross_entropy` 悄悄破坏了 NLL 容差判据（`fd05fb3` 引入 / `d780502` 修）
- **症状**：`verify_accel_readout` 长期报 **1 FAIL**，改动前后都一样。
- **根因**：P80 把 nll 从 `log(softmax)` 换成单核 `F.cross_entropy`（内含 logsumexp）—— **数学等价、浮点归约顺序不同**。加速侧与 numba 参考实现相对差 **3.46e-5**，判据是 **1e-6**。`git stash` 双向确认：**不是当次改动引入的回归**，是 P80 悄悄把容差判据作废了。
- **修复**（`d780502` 的 P93 部分）：容差放宽到 1e-4，**理由写在断言旁边**；并注明 nll 是**上报标量、从不喂给学习** —— 真正影响学习的是 `dp → W`，由下一条 1e-5 的断言守着（实测 1.2e-7）。
- **门禁缺口（关键）**：⚠ **容差判据本身没有对账机制**。一条长期 FAIL 的断言会被当成「已知噪声」忽略，而没人问「它什么时候开始红的」。**门禁红了必须归因，不能等它变绿。**

### A9 用 `grep` 过滤验证输出，把 Traceback 一起滤掉（`afdd5a2`）
- **症状**：`train.py --help | grep remote` 无输出，据此以为「参数已加上」，实际那次调用正在崩溃。
- **根因**：把「过滤输出」当成「断言成功」—— 退出码和 stderr 被管道吃掉。
- **修复**：改为断言 exit code / 捕获 stderr。**门禁缺口**：**流程缺口，非代码缺口**。与 A2 一起说明「`--help` 必须真的跑出来给人看」。

---

## B. 读出与加速器

### B1 NPU 首次真跑训练即崩 `FakeTensor - None`（P55，`49aea05`）
- **症状**：`TorchRuntimeError: unsupported operand type(s) for -: 'FakeTensor' and 'NoneType'`，崩在 `accel_readout.py` 的 `dp = (p - t32)`。
- **根因**：P45 的「`target_idx` 路径就地改 p」只实现在 **eager 分支**，`torch.compile` 融合核 `_train_step_core` 漏改，而该路径 `t32 = None`。既有 `verify_accel_readout.py` 的 learn_softmax 用例**只走 target 数组路径**，从不传 `target_idx` → 恰好漏掉崩的那条。
- **修复**：融合核逐行镜像 eager（`p − onehot` 与 `p[c] −= 1` 数值等价）；顺带把 softmax/nll 移入 eager-only（compiled 路径原白算一次全词表 softmax）。
- **门禁**：新增 `tests/verifiers/verify_accel_readout_p55.py`（2×2 矩阵：target 数组 / target_idx × eager / compiled，**9/9**，容差 1e-5 fp32）。

### B2 inductor **惰性编译** → 编译失败会崩生产（`49aea05`）
- **症状**：`__init__` 有 try/except 回落，但**首次调用 `_fused`** 才真正触发编译 → 无 C++ 编译器时直接抛 `InductorError`。
- **根因**：`torch.compile(...)` 在**构造期只是包装、不会失败**；**回落写在了错误的层**。
- **修复**：首次执行失败 → **永久回落 eager + `warnings.warn` 记原因**（P19「回落 + 记原因」纪律）。
- **门禁缺口**：⚠ 无法在有编译器的机器上自然复现；现有门禁只覆盖 eager/compiled **都成功**的情况。

### B3 🔴 fp8 能力表遗漏 → 默认 fp8 被静默回落，**整轮改造在生产等于没生效**（`52499ff`）
> 本台账严重度最高的一条：不是崩溃、不是变慢，而是**一整轮精度改造完全没生效且不报任何错**。
- **症状**：P84 把 `--readout-dtype` 默认改成 fp8 后，生产日志里读出后端一直是 `numba-cpu(回落)`，`AccelReadout` 的 fp8 代码路径**从未被执行**。无异常、无告警。
- **根因**：`phdnet/backends/accel_readout.py::_unsupported_reason()` 是一张**「加速读出未实现项」黑名单**，其中仍列着 `fp8`（P19 写下、P84 实现后**没同步删**）。链路：`--readout-dtype fp8`（默认）→ 返回非 None → `pick_readout_backend()` 走 `_fallback(...)` → numba CPU 原路径 + `_accel_fallback_reason`。**为什么难发现**：回落**本来就是设计好的安全行为**（不静默丢功能），原因只写在一个属性里、启动日志打一行、不 raise、不改退出码。
- **修复**：从能力表移除 fp8（fp4 仍拒绝），并在 docstring 写明教训 ——「**加新 dtype 时务必同步这张能力表**」。
- **门禁缺口（关键）**：⚠ **没有任何测试断言「配置声明支持的后端 = 实际选中的后端」**。现有 verifier 都是「跑通即可」，而回落后 numba CPU 路径**也是跑通的** → 这类缺陷对测试不可见。**应有的门禁**：表驱动测试 —— 枚举 `cfg.readout_dtype` 全部合法取值 × 各后端能力开关，断言「声明支持 ⇒ `_unsupported_reason()` 返回 None」；再加快照：`--accel` 非显式 `cpu` 时出现 `fallback reason` 应醒目告警。

### B4 加速器全面禁用 fp8/fp4（P92，`d780502`；前序 P85 `646db01` / P86 `f033b9c`）
- **症状**：昇腾 Ascend910B4 + CANN 8.5 + torch_npu 2.9 上，`float8_e4m3fn` / `float8_e5m2` 的 **create / cast / matmul 全部 ERR01007**；`torch.ops` 里 **零个 fp8 算子**；CPU torch 能分配 fp8 但**无 fp8 addmv**。
- **根因**：P85 探针**只测张量创建**（CPU 能分配 → 探针通过，随后炸在 addmm）→ 改测真实 matmul 后才判死；P86 在能力表显式拒绝；P92 最终**从加速后端整体删除 fp8 机器**（不再运行时探测），`_DT` 收缩为 `{fp32, fp16, bf16}`，fp8/fp4 连构造都不允许（立即报 unsupported dtype）。
- **修复**：读出默认 **fp16**（P85）；bf16 半 ULP ≈ 2e-4 ≫ 非目标行更新 |dp| ≈ 1e-6 → 更新被舍、退化为纯 Hebbian，是当日 PPL 震荡的根因，故**不回 bf16**。**numba CPU 路径保留** P9/P12 的 fp8/fp4 位算法量化核 → `--accel cpu --readout-dtype fp8` 仍可用（「除 CPU 外」的那一半）。
- **门禁缺口**：⚠ 探针只在**开发机（x86、无 NPU）**跑过，正确路径是「报告 torch_npu 不可用」。真正的证据来自服务器人工反馈 —— **能力表里的一行「不支持」只能靠目标机器实测确认**。

### B5 fp8 失败必须在**使用点**兜底（P90，`927a4ba`）
- **症状**：服务器 19:17 `RuntimeError: 'Float8_e4m3fn has not been supported'` from `forward_dev` —— 该机器跑的是**早于 P85 探针的代码**（直接进了 `W.to(float8_e4m3fn)`）。
- **根因**：把「探测 + 拒绝」当成唯一防线，**没考虑使用者拿不到新代码**。且两个平台失败点**不同**：昇腾炸在 cast（ERR01007），这台 CPU 能建 fp8 张量但炸在 `@`（addmv 无实现）—— **只护 cast 不够**。
- **修复**：量化和 `forward_dev` 的 matmul **各自** try/except → 失败永久回落 fp16 + 告警、**不让异常逃逸**；仅当已回落后再出错才重抛（真错误仍暴露）。已用「模拟旧构建」验证：forward 返回 fp16、回退标志翻转、训练继续、5319 个非目标行仍获得更新。
- **门禁缺口**：⚠ 无法自然复现（旧代码已被覆盖）。**教训比门禁重要**：能力探测是**乐观**的，使用点兜底是**悲观**的，两者都要有 —— 不能靠「大家都 pull 了最新代码」。

### B6 NPU/HBM 遥测恒为 `--`（P57，`8b36f23`）
- **症状**：日志 `NPU/GPU --%  HBM --/--GB`，但服务器确有 NPU。
- **根因**：`Telemetry()` 一直**无参构造** → `device=""` → `sample()` 的加速器分支**永不触发**（P41 遗留；P19 把读出搬上 NPU 后没人回补遥测）。
- **修复**：自动探测 device（`torch.npu` → `torch.cuda`，显式传入仍优先）；首次采样失败打印原因（原 `except: pass` 全吞）；`TeeLogger` 接管 `sys.stderr`（回落告警 / inductor 日志此前不落盘 —— 这正是「读出为什么没融合」在服务器上不可见的原因）。
- **门禁缺口**：`verify_ms_stream.py` 只加了「无 NPU 机器探测为空且行为不变」的断言。⚠ **「有 NPU 的机器上遥测非空」本机无法验证**，只能在服务器侧人工确认。

### B7 `npu-smi` 不在 PATH → AI Core% 恒 `--`（P63，`0a0dfd8`）
- **症状**：HBM 有值（走 `torch.npu.memory_allocated()`，host 侧计数器）但 AI Core% 没有。
- **根因**：AI Core% 走 `npu-smi`，而训练是**直接 `python train.py` 启动**的，没 source Ascend 的 `set_env.sh` → `shutil.which("npu-smi")` 落空。
- **修复**：三级定位 `$NPU_SMI_PATH` → PATH → 常见安装路径（`/usr/local/Ascend/driver/tools/npu-smi` 等）；首次成功/失败各打一行。
- **门禁缺口**：路径解析是纯函数、**可测**，但「这台机器上真实存在哪个路径」**不可测**。

### B8 `npu-smi` 解析依赖该机没有的 Bus-Id 列（`10f0396`）
- **症状**：`npu-smi` 已定位到 `/usr/local/bin/npu-smi`，日志仍是 `NPU% --`。
- **根因**：解析器找「含 `0x` 总线号的数据行」—— 该机输出**没有这一列** → 所有行被跳过。
- **修复**：改为**按表头定位列**（找含 `AICore` 的表头行 → 推出 AICore / HBM-Usage 列序 → 按列号取值），**与型号无关**；失败时打印原始输出前 14 行一次。
- **门禁**：加了两种布局（有/无 Bus-Id、小写表头）的解析断言 —— **合成布局能测，真实格式仍靠人读那 14 行**。

### B9 遥测自己同步设备流、破坏被观测的流水（P58，`b00bb90`）
- **症状**：P58 目标之一是 CPU/NPU 重叠，但 `torch.npu.utilization()` 内部会**同步设备流** —— 每次采样都在训练热路径上砍一刀。
- **修复**：AI Core% 改走 `npu-smi`（子进程外部查询，零同步）；`memory_allocated` 是 host 侧计数器，保留。**教训**：**观测手段本身不能破坏被观测的流水线**。

### B10 绑核与线程：`OMP_PLACES` 引来 8–13× 退化并被回滚（P71 `10f0396` / P73 `5c4c303` / P74 `753d157`）
- **症状**（服务器 12:50 日志）：`CS/s 250 万–600 万`、只用 1.1–3.2 核（191 核）；引入 `OMP_PLACES=cores` 后 **M1_encode 0.85 → 58–60 ms/tok（70×）、M2_infer 2.4 → 19–32（8–13×）**。
- **根因（三条独立缺陷）**：① P71 numba/OpenMP 线程**无核心亲和性**，到处漂移（cache miss 与上下文切换的直接来源）；② P73 `*_NUM_THREADS` **从未限上限**，OpenBLAS 拉 **191 线程**跑 1024×2048 的 sgemv —— 仓库早有 `multi_device.configure_host_threads()` 但**无生产调用点**，典型的「**修了一半**」；③ P74 `OMP_PLACES=cores` 在 191 核上要建 191 项 place 表，OpenMP/numba 每次线程池同步都要遍历它 → 假设**被实测否掉**。
- **修复**：`--omp-proc-bind`（默认开）只设 `OMP_PROC_BIND=close`（**不设 `OMP_PLACES`**）；`--numba-threads` 默认 8，**四个 BLAS 变量必须在 `import numpy` 之前**设（OpenBLAS 初始化后再设无效）。
- **门禁缺口**：⚠ 三条都**无法在 CI 上验证**（CI 机器核数不同）。「限线程」的正确性**依赖部署机器的核数**，只能人工确认。⚠ P74 是**假设被实测否掉**的记录 —— 保留它是为了避免有人再把 `OMP_PLACES` 加回来。

### B11 fp8 能力探针本身测错了东西（P85 `646db01` → P87 `0263cc8`）
- **症状**：第一版探针**通过**，随后在 `addmm` 上炸。
- **根因**：探针只测**张量创建** —— CPU torch **能分配** fp8 张量，只是**不能乘**。
- **修复**：探针改测真实 matmul；`tools/probe_fp8.py` 进一步枚举 `torch.ops` 下的 fp8 算子、torch_npu 的 fp8 属性、设备名与 CANN 版本（区分 910B/910C/A5），输出三档结论：标准 dtype 可用 → 解除拒绝；标准 dtype 阻塞但有专用算子 → 改走 `torch.ops`；什么都没有 → 保持拒绝，并记成「**芯片有 fp8，框架没暴露**」，等 CANN 跟进后一处改动即可翻回。
- **门禁缺口**：⚠ `probe_fp8.py` 本机跑只验证了 x86 分支（正确报 torch_npu 不可用）。

### B15 主机侧 epoch 判据在「缓存命中」路径上失效 → 静默用错 gather（P122）
- **症状**：`forward(hA) → learn_softmax(hB) → learn_softmax(hA)` 序列下，
  第三次的前向与更新**都用了 hB 的 gather 结果**去算 hA 的输出。
  **不崩、不报错的静默数值错误**（形状/dtype/量纲全对）。
- **根因**：P112 把 `torch.equal` 换成主机侧 epoch 判据
  （`_cache_g_epoch == _ht_epoch`）以消除每步设备同步。但 `_ht_epoch` 只在
  **上传新 h** 时递增，而 `_lookup_ht` 的**缓存命中**路径会返回**上一次上传的
  ht**却**不把 epoch 回退** → 判据 `2==2` 成立→ 复用了 hB 的 gather。
- **修复**：新增 `_ht_epoch_at_cache` 记录「当前 `_cache_ht` 上传时的 epoch」，
  `_lookup_ht` 命中时把 `_ht_epoch` 置回它。
- **门禁**：`verify_accel_sparse.py` C4c 三例（缓存归属 / 污染后前向与干净实例
  一致 / 同 h 连续调用仍复用）。**已验证门禁真能抓**——临时撤掉修复后 C4c1 FAIL。
- **教训**：把「每次都做」的重活换成「按计数器判据」时，**必须枚举计数器
  不递增的所有路径**，别只检查「值递增的那条」。

### B16 自己写的「import 前设环境变量」纪律被import 链打破（P120→P122）
- **症状**：P120 加的 CANN 环境治理（TASK_QUEUE_ENABLE=2 等）在真实运行时
  **可能完全无效**。
- **根因**：`from phdnet.backends.cann_env import apply_cann_env`会先执行
  `phdnet/__init__.py`（→ `model.py` → `import numpy`）与
  `phdnet/backends/__init__.py`（→ `accel_readout.py` → `import torch`）。
  实测「import 后 `numpy in sys.modules == True`」→ **numpy/torch 在设置
  环境变量之前就已被导入**，而 CANN/torch_npu 在库初始化时读这些变量。
- **修复**：改用 `importlib.util.spec_from_file_location` **按文件路径加载**，
  绕开包 `__init__`；并把调用点移到 `_ROOT` 定义之后（首次插入时`_ROOT`
  尚未定义，直接报 NameError）。
- **门禁**：`verify_cann_env.py` 只测 `apply_cann_env()` 函数本身，
  ⚠ **测不到「调用时机对不对」** ——那是本条漏过的原因。补法：断言
  `train.py` 里用的是 `spec_from_file_location` 而非 `from ... import`。
- **教训**：**「在某处之前设置」这类纪律无法靠测函数验证**，只能靠
  静态检查调用形式（或在函数里加一条「我必须在库导入前被调用」的断言）。

### B17 摊销优化「收益 0 + 污染权重」（P116 设计 → P122 审计推翻）
- **症状**：`--ltm-imprint-amortize N`（P116 加的方案 B）声称能省 imprint 开销，
  实测**收益恰为 0**，且 N≥2 时**污染权重**。
- **根因 1（收益 0）**：`cur = self.encode(rate)` 在摊销分支**之前无条件执行**；
  而摊销只把 N−1 次 `learn` **攒起来一次性提交**——调用次数一次没少
  （实测 N=1/2/3/4 → 23/22/21/20 次），「提交数 + 缓冲残留 == 基线」。
  且 `buf[-1:]` **永不flush** → 尾部 N−1 对**永久丢失**（持续性数据丢失）。
  成本侧：动的正是大头（`learn` 6.53 ms/call vs `encode` 0.054 ms/step）。
- **根因 2（污染）**：提交循环在**同一个 `step_count`** 下连续调 N−1 次 `learn`，
  共享神经元 `dt = t - stamp.get(i,0) == 0` → `trace*lam**0+1.0`
  = **该衰减的没衰减**。生产尺寸（n_dim=1024/m_out=72）实测 `max|Δw|`：
  N=2 → **7.10**、N=3 → **11.98**。⚠ **小尺寸测会漏掉**（n_dim=64 时为 0）。
- **修复**：整体移除；`amortize>1` 改 **fail-fast `ValueError`**（保留签名不破坏
  调用面），CLI help 写明移除原因。
- **门禁**：verifier B 组重写为「fail-fast + 生产尺寸逐键比对」。
  ⚠ **旧 verifier 给了假通过**：它用 `m_out=8` 且比「行内权重和」，
  恰好把逐键差抵消成 0 → **在最危险的真实配置下静默通过**。
- **教训**：① 「攒起来批处理」不等于「省下次数」，若每次的**前置计算**在分支外，
  批处理只改时机；② **验证必须用生产尺寸**——小尺寸会掩盖污染；
  ③ **门禁的比对口径也会造假**（求和抵消），要逐键比。

### B14「numba 能上GPU」是**平台绑定**的，换硬件就失效（P121）
- **症状**：fhz 提出「numba 有 API 能把运算放到 CUDA 上」——**这个 API 确实存在**
  （`numba.cuda`，`@cuda.jit` 直接编译 CUDA kernel，可访问 NumPy 数组、
  自动 host↔device 传输）。但它**硬绑 NVIDIA CUDA 驱动**。
- **为什么在本项目用不了**（逐条核实，非印象）：
  | 依赖 | 昇腾 910B |
  |---|---|
  | `CUDA_HOME` / `/usr/local/cuda`（Toolkit ≥ 11.2） | ✗ 不存在 |
  | `libcuda`（`NUMBA_CUDA_DRIVER` 指定的驱动库路径） | ✗ 不存在 |
  | Compute Capability ≥ 5.0（Maxwell+） | ✗ NPU 无此概念 |
  | 后端包：`numba-cuda` / ROCm(AMD) | ✗ **无昇腾后端** |
- **修复**：无（这是生态边界，不是可修的缺陷）。P18 已实测确认同一条边界：
  numba 只编译到 CPU 机器码 → 加速器必须走 torch 栈，而 torch 栈缺 7 项机制
  （含 `big_ltm`）→ **整体迁移会丢机制**且权重不与生产 ckpt 通用。
- **门禁缺口**：⚠ **这类「库支持某硬件」的断言无法用 CI 验证**（CI 机上没有昇腾，
  也没有 NVIDIA 卡）。只能靠**查库的依赖清单**（`CUDA_HOME` / 驱动库 / 后端包）
  来判断，不能凭「API 名字里有个 GPU」就以为可用。
- **顺带查出的真实缺口**：`parallel=True` 的 njit 有 **11 个缺 `nogil=True`**，
  含 `--m2-kernel plain` 走的 5 个 CSR matvec 核（`sparse_pc.py:43/55/66/86/97`）。
  ⚠ **`nogil` 是有成本的选项**，不是「免费的并行」：本机实测单次调用**慢 7.8%**
  （0.090→0.097 ms，数值逐位相同）→ 单线程路径下是纯亏，只在「与其它 Python
  线程并发」时回本。**不要为了看起来更并行到处加**（同 P22 的先量再开）。

### B12 稀疏读出把整个 NPU 读出加速静默关掉（P110 日志发现→ P111 修）
- **症状**：服务器日志 `[readout] backend=numba-cpu(回落) | fallback reason: readout_conn_k>0（稀疏读出未在加速后端实现）`。读出占端到端 **~89%**（P19），回落等于**默认配置下设备加速全部失效**，而日志只在启动时打一行，训练中无人察觉。
- **根因**：`--readout-conn-k` 默认 128（P108）撞上 `AccelReadout._unsupported_reason` 的拒绝表 → `pick_readout_backend` 回落到 numba CSR 路径。**P19 纪律的副作用**：拒绝表本是「防止能跑但语义不同」，但一个**默认值**命中它，就等于默认关掉加速。
- **修复（P111）**：加速后端实现均匀k 的 gather-GEMV（前向 `y[i]=ΣⱼW[i,j]·h[Wi[i,j]]`，更新按行rank-1），拒绝表移除该项；非均匀行宽（幂律变长 CSR）与两级读出仍拒绝 → fail-fast，不静默走错算法。1B 档实测 val 37.9 MB + idx 75.7 MB = **113.6 MB vs 稠密 fp32 908.8 MB（8.0×）**。
- **门禁**：新增 `tests/verifiers/verify_accel_sparse.py` 21 例（数值等价/调用面/拒绝路径/零回归）。⚠ 实现过程中该验证器抓出**两个真 bug**：① 前向写成 `gather(...).sum(1)`——**把权重整个丢掉**，形状 dtype 全对只有数值错；② `_csr` 每次调用返回**新的 D2H 副本**，ckpt 的 `val[:] = ...` 写进无人引用的数组 → **检查点「恢复成功」但权重是旧值**。两者都靠数值对拍抓出，形状断言抓不到。

### B13 低精度读出把 p − t 退化成纯 Hebbian（P105 判断被 P110 实测推翻）
- **症状**：日志警告「half precision rounds away the perceptron's non-target-row updates (|dp| ~1e-6 vs bf16 half-ULP ~2e-4), so learning degenerates toward pure Hebbian」—— 但当时的默认是 **fp16**，即警告文案在陈述自己默认值的危害。
- **根因**：P105 认定「fp16 半 ULP 比 bf16 小 8 倍，能保住非目标行更新」。P110 用 `tools/probe_readout_precision.py` 实测（判据：**同精度舍入后元素是否真的变了** = 更新是否被保留）：`|W|~1e-2, |dp|~1e-6` 时非目标行保留率 **fp32 99.95% / fp16 26.67% / bf16 5.79%**；目标行三者都 100%（目标更新 ~1e-2 ≫ 半ULP）→ fp16 只是「比 bf16 好」，**离正确还差一个数量级**，73% 的非目标行更新仍被丢弃。
- **修复（P110）**：默认改回 **fp32**（fhz授权），低精度降级为显式 opt-in；启动日志按精度分叉（fp32 打印「精确 p − t 规则」，低精度才警告）。
- **教训**：① **半 ULP 的数量级估算不能替代实测**——「8 倍差距」在跨越两个数量级的问题面前不成立；② 精度判据必须用**同精度舍入后的基线**，第一版拿 fp32 值与 fp16 值直接比，得到「fp16 保留 100%」的**反向假阳性**；③ 修正 P105 时也是先改 bf16→fp16（同样无效），隔一天才被实测推翻——**改动前先问「这个修复的机制是什么」，不要先改再说**。

---

## C. M4b / LTM

### C1 P67 首版 gather+prange 多步序列逐位不一致（三个坑；**commit 见 P78 的提交信息**）
> P67 本身的提交在历史改写后已失效，hash 不可引用。首版失败的事实与三个坑由 P78（`12c184f`）的提交信息完整记录。
- **症状**：6 步 bit-exact 序列**逐位不一致**；800×800 组合 0.36 → 0.38 ms（**0.96×**，负收益）。
- **根因（三个独立坑）**：① 核内**生长分支被预留容量 cap 卡住**（原路径只受 `size < m_out` 限制，首版照抄了 `growth_guidance` 守卫）；② **误把结构可塑性绑到 `growth_guidance`**（它只该喂 `in_deg`）；③ **import 块漏了 `prange`** —— 又一个「补丁未落盘」家族实例。
- **修复**：P78（`12c184f`）重做：生长分支改为**无条件**、scatter 执行原路径在行填满预留后的 `_grow_row` 扩容、`_touch` 留在 Python（核前运行，语义不变）→ 通过 6 步序列对拍。
- **门禁**：新增 `verify_ltm_learn_batch.py` 5/5（单步 fp64/int8 × guidance 逐位 + 6 步生长/量化/衰减序列）。⚠ 阈值 4096 组合：低于它 gather 开销主导。

### C2 `encode` numba 化：O(n²) 线性去重输给 Python `set`（`7c5f21f`）
- **数据**：**0.84×**（多次跑 0.84–1.75× 波动）。**根因**：核内去重是线性扫描，Python `set` 是 O(1) 哈希。
- **决定**：调用方回退 Python；核函数留在 `ltm_kernel.py` **仅作记录**。要重启需改成排序+相邻去重，或小型开放寻址哈希表（参照 P18 词表合并）。
- **门禁缺口**：⚠ **无计时护栏** —— 「回退了所以没人会再误用」，但核函数还在文件里，下次有人看到它会以为它可用。

### C3 M4b 占 76% ms/tok：`predict` 的 1.8 万次 Python dict 更新（P70，`f874db0`）
- **症状**（服务器日志）：`M4b_ltm 63.51 / 总 83.55 ms/tok`（**76%**）且随步数增长；`M2_infer` 已降到 1.70，其余段 <1。`[ltm-diag] active=256，scores 72 → 308+`。
- **根因**：`OnlineCSRTable.predict` 遍历 256 活跃行 × 约 72 槽 = **1.8 万次 dict 更新 + numpy 标量取值/step**。C2 之后 recall 投影已 numba 化，但**跑在已物化的 dict 上** → dict 工作量原样保留。
- **修复**：`predict_arr`（numpy gather，返回 `(keys, vals)` **不合并重复键**）+ 核内按「行序 → 槽位序」逐次累加（浮点顺序与原 `p[k] += v` 完全一致 → **逐位相同**）。**实测 predict 8.780 → 0.197 ms（44.6×）**，recall 投影 21.9×。
- **关键判断**：这段**故意不上 numba gather 核** —— 1.8 万元素比 C1 的 64 万组合小两个数量级，核的固定开销会吃掉收益。
- **门禁**：`verify_ltm_kernels.py` 扩到 **10/10**，覆盖 recall **全链路**逐位（int8/fp64 × 行满 + 键随机）。

### C4 `[ltm-diag]` 一行都不输出（我引入，`7c5f21f`）
- **症状**：加了 LTM 规模诊断，日志里完全没有。**根因**：门槛设成**每 1000 次 imprint**，而 imprint 是**条件触发**（`mode=='encode'` 且超阈值）→ 3500 token 攒不到 1000 次。
- **修复**：改每 10 次。**门禁缺口**：⚠ 无。**诊断门槛要考虑触发频率**；宁可先打前 3 次。

### C5 诊断行 `np.diff(rev_indptr)` 每次物化 134 MB（我引入，`5c4c303`）
- **症状**：每 8 步一次的 recall 诊断，在 1B 档白烧约 **6 ms/tok**。
- **根因**：`rev_indptr` 按**大空间最大 key** 铺开，1B 档长 16,774,731（**134 MB**），`np.diff` 每次 recall 完整物化（本机 33 ms）**只为打印一行 `bindings` 数**。
- **修复**：只在打印分支内计算 + 改 `(indptr[v+1]-indptr[v]).sum()`（O(|valid|)，数值完全相同）。
- **教训**：**诊断代码也在热路径上**。打印一行数字不等于零成本；诊断必须和它产出的数字一样便宜。**门禁缺口**：⚠ 无 —— 本质是「写诊断时没看数据结构的规模」，靠 review 抓。

### C6 `_rev_to_csr` 构造期 1677 万次 Python 循环（我引入，`5c4c303`）
- **症状**：1B 档启动多花约 10 s。**根因**：构造期 `for i in range(n_keys)` 逐个查 dict → 1B 档 **1677 万次 Python 循环**。
- **修复**：只遍历实际存在的 key（≤ 4096 条）+ 向量化前缀和，**保持 key 升序**（与原实现段序一致 → 逐位相同）。
- **⚠ 修复过程中自己制造又抓到一次 bug**：第一次改成「dict 插入序 + 生成器」时 `indices` 与 `indptr` 段序错位，被 `verify_ltm_kernels.py` 的「rev CSR 逐维一致」用例当场抓出。
- **教训**：优化「形状/顺序」的代码时**逐位对拍是唯一防线** —— 纯计时看不出段序错位，结果仍「跑得动」。**门禁**：靠既有对拍用例，无需新增。

### C7 LTM imprint 侧每次调用约 20 s（Python `O(prev×cur)` 循环；部分修复，P78 `12c184f`）
- **症状**：`[ltm-diag]` 显示 5k 步内 recalls=80 但 imprints < 10，而 `M4b_ltm` 累积 41 ms/tok × 5000 ≈ **205 s**。也回答了 fhz「为什么一直只用一个核」—— 整条路径是 GIL 下的纯 Python。
- **根因**：单次 imprint 是 `O(prev×cur)` ≈ 6.5 万组合的 `_find_slot` 线性扫描 + dict 访问；在 aarch64 上（Python 解释器与 dict 访问比 x86 慢 10–20×）**单次约 20 秒**。
- **修复**（`12c184f`）：批量多核 learn，见 C1。阈值 4096 组合。
- **未修 / 诚实边界**：⚠ **本机计时只有 1.0–1.1×**（x86 Python 在这个规模已经够快），**收益完全押在服务器侧**（预期 20 s/call → 毫秒级）—— **该结论尚未经服务器复测**。在线 CSR 镜像（零 gather 多核）至今未做。

---

## D. 精度与 dtype

### D1 fp32 权重 @ fp64 输入 → 脱离 BLAS 慢 5.9×（`c8148b4`）
- **症状**：M1 编码器改 fp32 后反而从 857 µs 变 5047 µs。**根因**：混合 dtype 让 numpy **离开 BLAS 的 sgemv**，走逐元素慢路径。
- **修复**：输入一并 cast 到 fp32（n_in 个元素，代价可忽略）→ 606.7 → 337.7 µs（1.80×）。
- **⚠ 后续反转**：本条结论在 **aarch64/昇腾上被推翻** —— 那里 fp32 sgemv 反而**慢约 70×**，故 `--encoder-dtype` 默认值改为 fp64。**教训：dtype 优化的方向依赖平台，x86 的结论不构成昇腾的证据。**

### D2 混合 dtype 第二次误判：输入 dtype 没跟随权重（P76，`77f9b78`）
- **症状**：13:22 日志 M1_encode 仍是 62–67 ms/tok —— P61 把输入硬 cast 成 fp32，P75 又把权重默认翻回 fp64，**两者相遇成为混合 dtype**（fp64 W @ fp32 x），numpy 离开 BLAS。
- **根因**：D1 的规则在 P61 与 P75 分别「修对」了各自那一侧，没有任何一处要求**两端同时**看。两次修改各自都通过了自己的门禁。
- **修复**：把输入 cast 到 `W.dtype`。本机核对：top-k 跨 dtype 一致，x86 上 fp64 953 µs / fp32 487 µs。
- **门禁缺口（关键）**：⚠ **没有任何断言检查 `A.dtype == b.dtype`**。混合 dtype 不报错、结果正确，只是慢 —— **它在所有正确性门禁下都是隐形的**。**应有的门禁**：所有 `A @ b` 前的 dtype 断言。

### D3 平台自适应 GEMV：aarch64 numpy GEMV 病态（P77，`30ba887`；补漏 `aa51bab`）
- **症状**：修了 D2 的混合 dtype 后 M1_encode **仍是 57–58 ms/tok**，而同形状在 x86 只要 0.5–1 ms。
- **根因**：**是平台、不是 dtype** —— aarch64 上 numpy 对 1024×2048 单行乘积的 GEMV 在 fp32/fp64 **都**病态。
- **修复**：`SparseEncoder.encode` 平台自适应 —— aarch64 走自写 numba GEMV（`_gemv_rows`，行间 prange、行内顺序累加，容差 1–2 ulp），x86 保留 BLAS（实测 953 µs vs 1532 µs，**BLAS 本地赢，所以不做全局翻转**）。顺带把 M4b 计时拆出 `M4b_imprint` 内层段（因为上一次编辑静默 no-op 了，`aa51bab`）。
- **⚠ 关键判断**：**故意没有**因为一次服务器日志就翻转全局默认 —— x86 与昇腾**三个案例方向相反**（见 E4）。
- **门禁缺口**：⚠ 平台分支意味着**两条路径的门禁覆盖不同**：`_gemv_rows` 只在 aarch64 生效，本机（x86）**永远跑不到它**。

### D4 bf16 检查点保存崩溃（numpy 无 bf16）（P81 `3e53fec`；P84 补 fp8 `473c58b`）
- **症状**：`save_model` → `to_numpy(ro_W)` → `TypeError: Got unsupported ScalarType BFloat16`。
- **根因**：`readout_dtype=bf16` 使设备张量是 bf16，而 **numpy 无原生 bf16** → `.numpy()` 直接抛。bf16 检查点路径**此前从未走到「保存」这一步**。P84 换 fp8 后**同一处会再崩一次**（numpy 也无原生 fp8）。
- **修复**：`to_numpy` 对 bf16 存 **uint16 位模式**、fp8 存 **uint8**（`view(torch.uint16)`），与加载侧 P46 的解码（`ckpt_dtype=bf16` + `dtype.kind in "ui"` → `view(bfloat16)`）闭环 —— 无损，且检查点保持原体积。
- **门禁**：bf16 round-trip 用例加进 `verify_ms_stream.py`（41/41）。⚠ 但 A7 已指出：**fast 门禁仍没有 torch**，这条在 fast 集合里依然覆盖不到。

### D5 M2 融合核在昇腾退化 3–4×，默认改回 plain（P75 `57c5be7` / P76 `77f9b78`）
- **症状**：`--m2-kernel A/B`：plain 6.7–11.9 ms/tok vs fused 20–27（**3–4× 慢**）。
- **根因**：P52 的融合核在 x86 快 1.15–2.16×，在昇腾 aarch64 退化。P75 拒绝凭**一次**服务器日志翻转默认（20–27 ms 也可能是 prange barrier 成本、fastmath codegen，或只是与 P61 落在同一窗口）。
- **修复**：新增 `--m2-kernel {fused,plain}`，默认 plain；两条路径对拍到 1–2 ulp（融合核用 fastmath，**不逐位**）。`plain` 的实现是**从 git 历史逐字取回**的 —— fhz 明确记录「我先凭记忆写了一份，是错的」，真实版本 `tanh`/`clip(0.15)` 用法不同。
- **门禁**：`verify_pc_learn_fused.py` 12/12、fused/plain 一致到 1–2 ulp。**教训**：⚠ **凭记忆重写历史实现是错的**；**「不确定就回历史里取」应该是硬规则**。

### D6 bf16 读出让学习退化（P85，`646db01`）
- **症状**：PPL 长期震荡不降。**根因**：bf16 半 ULP ≈ 2e-4 是非目标行更新 `|dp| ≈ 1e-6` 的**约 200 倍** → 更新被舍入 → 训练**退化为纯 Hebbian**。
- **修复**：默认改 **fp16**（昇腾原生支持、成熟 fp16 GEMV、同样带宽、**保住微小更新**）。本机验证：fp16 主副本下 **5319 个非目标行格点获得更新**（bf16 下为 0）。
- **门禁缺口**：⚠ 「更新被舍」**没有任何门禁会报警** —— 训练不崩、PPL 只是不好。这类「数值语义退化」比崩溃难查得多。

### D7 `_append` 满行时 IndexError — **未修**（潜在，生产有守卫）
- **症状**：`_grow_row` 在 `len(keys[i]) == m_out` 时 `new_cap = min(len*2, m_out)` 不增长 → `_append` 抛 `IndexError: index 72 out of bounds`。**触发条件**：`m_out` 不是 `grow_chunk` 的倍数（1B 档正是 72 / 64）。
- **为何没炸生产**：`learn` 的 `size < m_out` 守卫挡住了所有生产路径，仅内部误用触发。
- **未修原因**：当前**无法从生产路径触发**，改它要动 LTM 生长边界（C1 已判定该区域近期整体重做），单独加防御收益低于风险。**待办**：随在线 CSR 镜像立项时，给 `_append` 加「满行时报错而非索引异常」的清晰防御。

---

## E. 优化否决与观测方法论

### E1 逐位对拍「假阳性」：基线用错实现（`ca8d39f`）
- **症状**：参考实现用**纯 Python** matvec 当基线，报 `dn0/dn1` 不一致。**根因**：fastmath 下的重结合 / FMA 决策与 numba 不同，**纯 Python 基线天然差 1 ulp**。
- **修复**：基线换成 numba 串行版 → `max|Δ| = 0`。
- **教训（反复生效）**：**逐位对拍的基线必须是同一库、同一 fastmath 口径的串行实现**，否则满屏假阳性。假阳性比没测试更危险 —— 它会让人去「修」正确代码。D5 又用了一次同一条规则。

### E2 融合核漏乘系数（写测试时抓出，`ca8d39f`）
- **症状**：自写 `learn_predictive` 融合核的逐位对拍 `max|Δ| = 0.4`。**根因**：两次外积都用了 `eta_pc`，**漏乘 `(1−mix)` 与 `mix`**。
- **修复**：补回系数（该融合核后来因 E3 停用）。**教训**：**逐位对拍是本台账投资回报率最高的一类门禁** —— 它在**提交前**抓到了这个 bug。

### E3 四次连续负收益否决（访存/调度受限规模上，加并行度不是免费午餐）
| 方案 | 数据 | 根因 | 决定 |
|---|---|---|---|
| `learn_predictive` 融合（12→1） | 26 万边 1.36× / 105 万 **0.96×** / 419 万 **0.91×**（`ca8d39f`） | 12 个 prange 段的线程调度成本 > 省下的 11 次核启动；191 核服务器只会更差 | **保留函数、停止调用**，留注释记录实测数据；验证脚本改为回归护栏 |
| LTM 表向量化 | predict 48/512/2048 行 = 1.00× / 0.92× / 1.36×（**无规律**）；learn 0.74× | 原版靠 `ltp <= 0` 短路让循环极便宜，预取/向量化**无条件付出** | 全部回滚（无提交） |
| gather + prange | 800×800 组合 0.36 → 0.38 ms（**0.96×**） | gather/scatter 的 `O(R·m_out)` 拷贝吃掉并行收益；且多步逐位不一致（见 C1） | 回滚（无提交）。正确做法是**在线维护连续 CSR 镜像**，独立立项 |
| `encode` numba 化 | **0.84×**（见 C2） | 核内 O(n²) 线性去重输给 O(1) 哈希 | 回滚，核函数留档 |
- **共同结论**：**分支密集的代码不要预先向量化**；**核的固定开销要先算**。**⚠ 与旧记述的差异**：`learn_predictive` 早期记为「整体删除」，实际是**保留函数、停止调用** —— 删除会让回归护栏失去被测对象。
- **门禁缺口**：⚠ 四次都**没有计时护栏**。判据只存在于人的记忆与本文档里。

### E4 x86 与昇腾方向相反的三个案例（方法论铁律）
| 改动 | x86 | 昇腾 aarch64 |
|---|---|---|
| M1 编码器 fp32 | 快 **1.80×** | 慢 **约 70×** |
| M2 融合核 | 快 1.15–2.16× | 慢 **3–8×** |
| `OMP_PLACES=cores` | 无感 | 慢 **8×**，CS/s 升到 600 万 |
- **结论**：**x86 的性能结论不构成昇腾的证据**；跨平台项目里任何优化必须在目标机器复测。`--m2-kernel`、`--encoder-dtype` 这两个 A/B 开关就是这条铁律的产物。`tools/bench_local.py`（P89，`b5217a3`）**只回答「在这台机器上变快还是变慢」**，**不产出文档数字** —— 今天已测出三个方向相反的案例。

### E5 计时误导：命中首项 / 规模猜错两个数量级（方法论）
- **症状一**：按 append 顺序填键 → `_find_slot` 总是命中首项（0.1 µs），把 numpy 优化**测成负收益**；打乱后才是平均情形。
- **症状二**：本机 800×800 组合 0.36 ms，服务器 6.5 万组合却 19.5 ms/tok（**约 54 倍**）—— 因为服务器 active 只有 256、每组合 300 ns（aarch64 的 Python 解释器与 dict 访问更贵），**不是组合数问题**。本机复现（0.53 ms）比服务器快 37× → **所有本机优化结论作废**。
- **教训**：**计时必须复现「命中位置随机 + 行已满 + 真实规模」**；**本机复现与服务器差一个数量级时，本机结论作废**。配套：`--step-profiling` 的九段 + 主循环三段就是为「真实规模下测量」建的（P72，`87cdcbf`）。

### E6 观测手段自欺：`--lang` 误判 与 `or True` 假断言
- **症状一**（`f0732e0`）：报告「加了 `--lang zh` 但日志还是英文」。**那份日志的 argv 里根本没有 `--lang`**（只有 `--step-profiling`）。本机实测过滤本身有效。
- **修复**：训练循环保留最近 8 个真实 target token，每个日志点打印 `[sample lang=zh] CJK=98% | '…'` → 语种是否生效**一眼可见**；日志头打印完整 `argv`。
- **症状二**：写 `verify_pc_learn_fused.py` 时为少写几行加了 `... or True`。
- **教训**：**「行为没生效」先查命令行，再查代码**；**测试不能自己骗自己** —— `or True` / `assert True` 等价于没测。

---

## F. CI 与工程

### F1 GitHub Actions 首跑即红（**CI 设计缺口**，`82c7390`）
- **症状**：首次推送（sha `90b7b63`）即失败。① `backend-torch / bench_accel`：`open('outputs/bench_accel.json','w')` 但**新 checkout 上 `outputs/` 不存在** → FileNotFoundError；② `tests / verify_parallel_consistency` 硬依赖 `datasets/pretrain/pretrain_*.parquet`，而它**被 gitignore 且本地 29 GB 副本已删**。
- **根因**：workflow 直接复用了「本地已存在」的假设（已生成的 `outputs/`、已上传的 datasets、只在本机装好的可选依赖）。
- **修复**：`makedirs(outputs/experiments, exist_ok=True)` 且打印路径与实际一致；`verify_parallel_consistency` 在 glob 匹配为空时**合成一份小而确定的中文 parquet**（pyarrow，text/lang/src，8 样本 × 约 9k 字符）跑同一条流水线 —— 该脚本的目标是多核等价性，不是语料本身。本机带 fallback：serial=8 / parallel=8 块，逐位一致。`seg_equiv` / `vocab_parallel` 不受影响（`eval_corpus/` 43 KB 是 git 跟踪的）。
- **门禁缺口**：这是**门禁自己**的缺口，只能靠真跑一次 CI 才能发现 —— **推上去之前无法本地复现**。

### F2 依赖 import 损坏 → 诊断步骤自己先崩（`0a5ab0b`）
- **症状**：F1 修完后诊断步骤**也失败**，且死在第一行 `import numpy, numba, pyarrow, torch` —— `bash -e` 下这直接中止整个 step，**这本身就是诊断**：CI 镜像上依赖 import 是坏的。
- **根因**：最新 numpy（2.3+）配较旧 numba wheel 无法 import。
- **修复**：钉 `numpy<2.3` + `numba>=0.61`（backend-torch job 把 torch CPU 轮子与 `numpy<2.3` 放在同一次 resolve）；诊断改为每个 import 各自 `|| true`，**一个坏 import 不再截断整份报告**。
- **门禁缺口**：⚠ 首次诊断**失败**时反而**什么都看不到** —— 诊断工具本身被同样的故障打倒，是个盲区（`3a6afc8` 先加了 `if:failure()` 伴随步骤，仍被这条打倒）。

### F3 三轮修复后 GitHub Actions **仍红**，且无日志权限 — **不要声称 CI 全绿**（`0a5ab0b`）
- **症状**：`0a5ab0b` 之后 `.github/workflows/ci.yml` **仍失败**。
- **根因**：**工具链权限问题，不是代码问题** —— 无 admin token，**拿不到日志**，只能看到匿名可读的 check-run annotations。
- **修复（已做）**：`3a6afc8` 给两个失败 step 加 `if:failure()` 伴随步骤，打印解释器/依赖版本并**重跑失败命令输出完整 traceback**，可直接在 GitHub UI 阅读。
- **未修 / 现状**：⚠ **两套 CI 中 GitHub Actions 仍红**；GitCode 侧（`.gitcode/workflows/ci.yml`）正常。**本文任何地方都不宣称 CI 全绿。** 门禁缺口：拿不到日志时**没有任何自动升级路径**。

### F4 仓库**没有 Jenkinsfile**（文档事实纠正，`52499ff`）
- **症状**：旧文档记「CI 三套（含 Jenkins）」。
- **根因**：`Jenkinsfile` 已在 `2b4bd3d`（P9 时代「删 Jenkinsfile」）中删除，但文档口径没跟上。
- **修复**：全文档统一为 **CI 两套**（GitHub Actions + GitCode），并把「一处事实一个权威出处」定为规范。
- **⚠ 同一轮重写暴露的真 bug 就是 B3** —— 文档重写顺带抓出了 fp8 能力表遗漏。
- **门禁缺口**：⚠ **无任何机制校验文档里的「仓库里有什么」**。这类事实靠人核对，容易随删除而腐烂。

---

## G. 环境与数据

### G1 NPU 设备启动失败 `507033` → 读出回落 CPU — **未修**（环境问题，`c74b3fb`）
- **症状**（服务器 2026-09-30 18:59）：`LazySetDevice ... error code is 507033`、`ERR00100 PTA call acl api failed`、`Failed to start the device`、`open device 0 failed`。
- **判断**：**设备层问题，不是代码**。回落机制正常工作（日志里 `backend=numba-cpu(回落) | fallback reason: ...`，**没崩**，只是退回 CPU，主循环机制完整）。507033 = `device retain` 失败，最常见是**上一次训练进程未退出、仍占着设备**。
- **改进**（P88）：训练启动期做一次**真实的设备张量探测**（建 net 之前），把 CANN 的天书翻译成三条可操作检查（残留进程 / 容器设备映射 / 设备占满），外加 `--accel cpu` 逃生口。
- **未修原因**：取决于服务器环境（需在机器上确认 `npu-smi info` 的进程列表）。**门禁缺口**：⚠ 无法在 CI 复现。**但回落纪律把它变成了非事故** —— 这是「回落 + 记原因」唯一一次在真实故障上收到回报。

### G2 数据集语种与设计不符（中文仅 2%）— **未修**（数据问题）
- **实测**（ModelScope `fhzfhz/Mixture-General-Mini` 的 52 个 pretrain 分片，`f0732e0`/`9180855`）：第 0–45 片全是 `infinity_m7core`，lang 分布 **en 50% / unk 47% / zh 仅 2%**；**`unk` 其实是代码**（`assigned_fires = []`，CJK 占比 0%）；只有末尾第 51 片是 `ultrafineweb_l3_zh`（中文 99.7%）。
- **与设计冲突**：设计记录「M7_Core 中文 3,697 万块 ≈ 95%」。`--remote-fraction 0.3` 取的前 16 片又全是 m7core → 「中文训练」实际只用到极小部分数据。
- **未修原因**：**这是数据制备/上传环节的问题，不是代码问题** —— 代码侧过滤逻辑本机实测有效（`lang='zh'` 出中文）。**在查清之前不要重新采样覆盖现有分片**（会丢证据）。
- **门禁缺口**：⚠ **无任何门禁会发现「语料分布与设计不符」**。`[sample lang=]` 能验证「过滤生效了」，但**验证不了「过滤后的语料足够中文」**。

---

### #34 子代理交叉审查：P99 整条链净效果为零 + 三个阻断缺陷（已修）
**背景**：fhz 要求「用子代理多审查几遍」。一个审查代理审另一个代理（和我）当天写的代码，
结论是**不能合并**，实测证据充分：

1. **【阻断·总开关】`self.fused = bool(fused)` 抹掉三态** —— `"serial"` 被转成 `True`，
   于是 `--m2-kernel serial` 实际走 parallel 核，**P99 整条链净效果为零**。
   spy 计数确认三种取值全部落到 parallel 核。
2. **【阻断】serial 核里 `np.clip(标量)`** —— numba nopython 不支持标量重载，
   首次调用 `TypingError`；**旁证：`__pycache__` 里唯独没有
   `_pc_infer_fused_serial-*.nbi`**——该核从未成功编译过（作者本地也没跑通）。
3. **【阻断】`e0` 语义不等价** —— serial 用 `+=` 累加、parallel 每步重算，
   且 parallel 在步外还有一次重算。实测 serial 与 parallel 差 **40 倍**
   （|S−L|=3.58 vs |P−L|=0.089），并污染 n_steps≥2 时的 r1 轨迹。
4. **【阻断】`infer.py` 读 `args.lang` 但从未定义 `--lang`** → 推理入口 100% 崩
   （i18n 接线只做了一半）。
5. **【高】`BATCH_MIN_COMBOS` 全仓库无人读取**（阈值硬编码 4096）→
   `verify_ltm_learn_batch.py` 的「禁用/强制批量」切换**完全无效**，对拍脚本
   在拿批量路径和自己比 → P78 的门禁是空转的。
6. **【高·当前不可达】`prev` 含重复 id 时批量路径静默算错**（gather 两次、
   scatter 覆盖，Python 路径则累加多次）。当前 prev 来自 `dict.fromkeys`（已去重）
   故不可达，但属于无声陷阱 → 已加显式去重。
7. **【中】i18n 单字条目误伤**：`re` alternation 左最先匹配，短条目抢在长条目前 →
   `'检查点已保存完成'` → `'checkpoint saveddone'`（残留「完成」）。
8. **【低】`translate()` 每个匹配点重建 142 项 dict**（81 µs/call）；
   `T()` 是零调用的死代码；`__getattr__` 在 `_w` 缺失时无限递归。

**已修**：1–6（serial 核现与 parallel **逐位一致**，n_steps=1/2/3 全部 max|Δ|=0；
`BATCH_MIN_COMBOS` 真正生效后 `verify_ltm_learn_batch` 5/5 才是真对拍）。
7–8 待处理。**门禁全绿**：fast 9/9、ms_stream 41/41、pc_learn_fused 12/12、
ltm_learn_batch 5/5、ltm_kernels 10/10、`infer.py --help` 正常。

**同一个审查员对 `_ltm_learn_rows` 的评价是正面的**：量化/裁剪/乘法结合顺序/prange 写集
全部实测逐位无误（`eta*a*b` 与 `eta*(a*b)` 在 35% 的随机三元组上不同，核内左结合**正确**）。

## 9. 结构性教训汇总

1. **补丁未落盘家族（本台账最高频）**：A1（孤立缩进块 + 删定义）、A4（截断 `except`）、A5×2（`str.replace` 无匹配静默成功）、A6（`_correct_pinned`）、C1③（import 块漏 `prange`）、C6（第一次改法段序错位）、F4（文档口径没跟上删除）。→ 已落地：`py_compile` 门禁 + **`grep` 断言改后文本存在** + 「不确定就从 git 历史逐字取回」（D5）。
2. **诊断代码也在热路径上**：C5（134 MB/次只为打一行）、C4（门槛没考虑触发频率）、B9（观测手段本身破坏流水线）、A2（`--help` 从来没真跑过）。
3. **配置与实现必须对账**：**B3（fp8 能力表遗漏 → 整轮改造在生产等于没生效）**与 **B12（`readout_conn_k` 默认值撞拒绝表 → NPU 读出加速全程失效）**是本台账最严重的两条，因为它们**都没有任何报错**。已有的对账是「文档重写」顺带抓出来的，不是门禁抓的。→ **B12 已落地修法**（P111 真正实现稀疏读出而非只改文档口径）。
4. **负收益同样是结论**：E3 四次连续否决 + C1 负收益回滚，共同结论是**访存/调度受限规模上加并行度要先算固定开销**。但四次都**没留计时护栏** —— 结论只活在文档里。
5. **x86 的结论不构成昇腾的证据**：D1 在两个平台上**方向相反**；E4 三个案例方向相反。→ 已落地：`tools/bench_local.py` 明确「不产出文档数字」+ 三个 A/B 开关。
6. **门禁只能拦住门禁想得到的东西**：G2（语料分布错）、F3（CI 仍红）、F1（CI 自己的设计缺口）、D2（混合 dtype —— **在所有正确性门禁下都是隐形的**）、D6（更新被舍 → 训练退化但不崩）、**B13（半 ULP 数量级估算代替实测 → 默认值改了两次都是错的）**都超出现有 9 项 fast 门禁的视野。
7. **容差判据本身需要归因**：A8 的一条断言长期红着没人问「它什么时候开始红的」—— **门禁红了必须归因，不能等它变绿。** B12 的 A3 断言也是同理：它断言「稀疏必须回落」，P111 修好后**必须同步改断言**，否则会把正确的修复当成回归（反向风险与 B3 同源）。
8. **数值 bug 需要数值门禁，形状/dtype 断言抓不到**：B12 的前向 bug（`gather.sum(1)` 丢掉全部权重）形状、dtype、量纲全对，只有**数值对拍**能抓住；`_csr` 返回副本导致检查点静默不生效，同样只有数值对拍能抓住。→ 已落地：`verify_accel_sparse.py` 的A1–A4 四条数值断言（容差判据，跨库归约顺序不同故不要求逐位）。

---

## 10. 现存诊断点（有意保留，便于线上排障）

**不要删这些** —— 它们是下一次线上排障的唯一入口。若某项已确认无用，保留成本极低。

| 位置 | 输出 | 用途 |
|---|---|---|
| `phdnet/bigltm.py` | `[ltm-diag] imprints=… prev/cur/combos/rate_dims/rows/k_hash` | LTM 规模与稀疏度演化（每 10 次） |
| `phdnet/bigltm.py` | `[ltm-diag] recalls=… active/scores/bindings` | 召回侧遍历量（每 10 次） |
| `phdnet/telemetry.py` | `GC <对象数>M/gen2 <次数>`（遥测行内） | 验证 / 排除 GC 假设 |
| `phdnet/telemetry.py` | `[telemetry] npu-smi = <path>` / `⚠ 未找到` / `npu-smi parse FAILED; raw head://…` | 设备遥测可用性与解析失败取证（B7/B8） |
| `train/train.py` | `[gc] freeze() + threshold(50000, 200, 200); tracked objects = …` | GC 调优生效确认 |
| `train/train.py` | `[sample lang=…] CJK=…% \| '…'` | 语种过滤是否真生效（G2 的观测入口） |
| `train/train.py` | `[numba] cache dir = … \| size = … MB \| 预期启动 ~2.6 s` | numba 持久缓存是否真被命中（`cache=True` 有没有被漏掉） |
| `train/train.py` | `[log] start: … argv: …` | 完整命令行 —— 「行为没生效」先查这里（E6） |
| `train/train.py`（`--step-profiling`） | `segments: M1_encode … M4b_ltm …` | 九段耗时分解（按耗时降序取前 6） |
| `train/train.py` | `[readout] backend=… \| fallback reason: …` | **读出是否真在设备上**（B12：曾长期回落 numba 而无人察觉）—— 判断加速生效看这行，不看「检测到设备」 |
| `train/train.py` | `[readout] compute precision = … — 精确 p − t 规则` / `WARNING: 半精度会舍入丢弃…` | 读出精度与语义警告（B13：低精度不退化纯 Hebbian） |
| `train/train.py`（`--step-profiling`） | `loop: tokenize … encode_onehot … step …` | **主循环三段** —— `net.step` 之外的开销（P72） |
| `train/train.py` | `[读出] …` / `fallback reason: …` | **以后台行为准，不以「探测到设备」为准**（B3/B4/G1） |
| `phdnet/i18n.py` | `install_stream_filter()` 覆盖 11 个文件 47 处中文输出 | `--lang en` 全英文；未登记的短语保持中文并在日志里露出来 |

日志字段的完整速查表见 `train/README.md` §7.1。
