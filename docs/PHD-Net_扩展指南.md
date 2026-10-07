# PHD-Net 扩展指南

> **适用范围**：要**扩展 PHD-Net** 的开发者——加机制、加后端、加精度、加数据通路、加训练阶段、加验证器。
> **数据截止**：2026-10-01。与代码冲突时**一律以代码为准**，并回头修本文。
> **配套**：[`BUGS.md`](../BUGS.md)（缺陷台账，本文每条 ⚠ 都能在那里查到完整症状与根因）、
> [`PHD-Net_架构设计.md`](PHD-Net_架构设计.md)（机制与容量账）、
> [`PHD-Net_性能评估与迭代方案.md`](PHD-Net_性能评估与迭代方案.md)（**所有**性能数字的唯一出处）、
> [`PHD-Net_硬件后端适配报告.md`](PHD-Net_硬件后端适配报告.md)（后端矩阵与昇腾真机踩坑）。

本文是**操作手册**：每节给出「改哪里 → 怎么验证 → 门禁在哪 → 门禁缺口」。
所有 ⚠ 标记都是本项目**真实发生过**的事，不是假想风险。

---

## 0. 五分钟速查表

| 你要做的事 | 改哪些文件 | 必过门禁 |
|---|---|---|
| 加一个新机制（如 M7） | 新模块 + `phdnet/model.py::step` 接入 + 计时段 | fast 9/9 + **新写数值对拍**（§2.3） |
| 加一个读出后端 / 换设备 | `phdnet/backends/accel_readout.py` + `resolve_accel_device()` | `verify_accel_readout.py`（全PASS）+ `verify_accel_sparse.py`（22/22，若涉及稀疏） |
| 加一种 dtype | **三处**（见 §4.1） | `verify_accel_readout.py` + `verify_ms_stream.py`（检查点往返） |
| 改一个 numba 核 | `phdnet/ltm_kernel.py` 等 | 对拍（**逐位优先**）+ `verify_ltm_kernels.py` |
| 加一个数据源 | `phdnet/corpus.py`（统一接口）+ `train/corpus_stream.py` | `verify_stream_tokenize.py`、`verify_ms_stream.py` |
| 加一个训练阶段（SFT/RL） | `tools/train_rl.py` 模式 + `phdnet/rl.py` | 阶段语义对拍 + fast 9/9 |
| 加一个 CLI 参数 | `train/train.py` argparse | **`--help` 实际能跑**（§5，含 6 次踩坑） |
| 加一个 verifier | `tests/verifiers/verify_*.py` | 自己先跑；别把自己扫进去（§6.4） |
| 加一个工具脚本 | `tools/`（标注生产轨/非生产轨） | 注明平台 + commit（§7） |
| 调一个已有开关 | 见 §9 速查表 | 对应 verifier |

### 架构铁律（不可违反）

① 禁自注意力 / 位置编码 / 堆叠层；
② 每个机制必须有神经认知对应物；
③ **逐 token 语义**——状态跨 token/shard/epoch 连续演化，`W` 每步原地更新 → **不允许批处理**；
④ 行为变更以 config 开关承载且**默认关闭**。

> ⚠ ④的**例外**（fhz 明确指令，default-on）：`sparse_conn=True`、`k_sparse=16`、`csr_online=True`。
> 库默认值与 CLI 默认值**可以不一致**（例：`config.encoder_dtype=fp64` 而生产 CLI 覆盖为 `fp32`），
> 但**必须是有意的**，且文档要写明是哪一个在生效。

---

## 1. 代码地图与 M1–M6 接入点

```
phdnet/
  model.py                PHDNet.step —— 唯一的机制编排入口；to_numpy/_nelem 在文件末尾
  sparse_encoder.py       M1 稀疏分布式编码（k-WTA）；resolve_model_dtype / _gemv_rows（P77 平台自适应 GEMV）
  sparse_pc.py            M2 预测编码主干（CSR）；_csr_matvec（行级 prange）/ _pc_infer_fused_serial
  plasticity.py           M3 STDP（只走 numba CPU）
  wm.py                   M4a 工作记忆（PFC 漏整合 + 摘要槽）
  bigltm.py + sparse_table.py   M4b 大容量事件驱动稀疏表
  modulator.py            M5 神经调制（ACh/NE/DA/5-HT 四通道）
  readout.py              M6 读出（numba 路径：稠密W 或 CSR）
  config.py               全部配置项 —— 注释即设计意图的权威来源
  sparse_alloc.py         幂律连接数分配器（**P124 起已接入 readout**，默认 alpha=0 关闭）
  backends/
    accel_readout.py      加速器读出（M6 上设备）
    multi_device.py       多设备（模型并行）
  telemetry.py            CPU/NPU/HBM/GC遥测（npu-smi 零同步查询）
  i18n.py                 终端输出语言（zh/en）
train/
  train.py                生产训练入口（单轨）
  infer.py                生产推理入口
  ckpt_1b.py              检查点保存/恢复
  config_1b.py            四档预设 + 容量账
  tokenizer_core.py       分词热路径（numba nogil）
tests/verifiers/          26 个专项验证器（P110–P124 新增：verify_m2_kernels /
                           verify_cann_env / verify_lang_semantics /
                           verify_powlaw_readout / verify_p116_sched）
tools/                    工具脚本
```

### M1–M6 在 `step` 里的接入顺序

`phdnet/model.py::PHDNet.step` 是**唯一**的机制编排入口。接入新机制时在既有机制之间插入，
并用 `_prof_t` / `_prof_end` 加计时段（`--step-profiling` 会按耗时降序输出前 6 段）。

⚠ **不要重排既有机制的顺序**：机制之间有数据依赖（s0 → M2 推理 → e0/e1 → M3/M4 → 读出）。
顺序即语义，`verify_*` 的数值对拍会立刻抓到。

---

## 2. 检查单：加一个新机制

| # | 步骤 | 落点 | 验证方式 |
|---|---|---|---|
| 1 | 加配置字段 + 注释（写动机与**数值影响**） | `phdnet/config.py` | 编译 |
| 2 | 实现模块 | `phdnet/<新模块>.py` | 单模块自测 |
| 3 | 接入编排 + 计时段 | `phdnet/model.py::step` | `run_tests.py fast` |
| 4 | 铁律声明（这个机制的神经认知对应物是什么） | `docs/PHD-Net_架构设计.md` 机制表 | 人工评审 |
| 5 | fast 测试断言 | `tests/checks_*.py` | fast 9/9 |
| 6 | 专项 verifier | `tests/verifiers/verify_<新机制>.py` | 新写的对拍 |

### 2.1 命名与默认值

- 机制编号 `M<N>`，配置字段 snake_case，CLI 参数 `--<kebab-case>`。
- 配置默认值与 CLI 默认值**尽量一致**；不一致必须是有意的（例：`encoder_dtype` 库默认 fp64 因为它是形状参数，生产 CLI 覆盖为 fp32）。

### 2.2 逐 token 语义的红线

新机制**不得**引入 batch 维或梯度。原因不是保守，是本项目的架构约束：
`step` 每步原地更新 `W`、M3 的迹、M4a 的槽位、M4b 的 CSR，状态跨 token 连续演化。
一旦能批处理，整个训练语义与所有基线锚点全部作废。

### 2.3 ⚠ 门禁：必须写**数值对拍**，形状断言抓不到 bug

真实事故：稀疏读出前向写成 `gather(...).sum(1)` —— **把权重整个丢掉**，
只把 gather 到的 h 分量相加。**形状对、dtype 对、量纲对**，只有逐元素对拍能抓到。

对拍写法：与**独立的参考实现**逐元素比，打印实测 `max|Δ|`，不要只打 PASS/FAIL。

---

## 3. 检查单：加一个读出后端 / 换设备

这是本项目**最贵的教训区**。读出占端到端 **~89%**，后端错了整轮改造等于没做。

### 3.1 完整调用面（少一个就崩）

| 成员 | 类型 | 缺失后果 |
|---|---|---|
| `__call__` / `forward` | 方法 | 前向崩 |
| `forward_dev` | 方法 | 无设备内路径 → 退回同步，热路径被打断 |
| `learn` / `learn_softmax` | 方法 | 训练崩 |
| `W` | 属性 | 权重访问崩 |
| `W_cpu` / `load_W` | 方法 | **保存/恢复检查点时崩** |
| `n_synapses` | 方法 | 参数统计崩 |
| `conn_k` / `hidden` | **只读 property** | 检查点与统计崩 |
| `stats()` | 方法 | 诊断脚本崩 |
| `dtype_name` | property | 日志/ckpt 元信息崩 |
| `_csr` | property | 稀疏检查点崩 |
| `_to_dev` / `_staged_to_dev` | 方法 | H2D 路径缺失 |

> ⚠ **`conn_k` 必须是只读 property**。若写成可写属性，两侧后端语义不对称，
> `ckpt_1b` 的 `net.readout.conn_k` 读法会分叉。
> ⚠ 若不实现稀疏，`_csr` 应**显式抛 `NotImplementedError`**，而不是返回 `None` ——
> 显式报错能立刻定位，而不是在别处变成更费解的 `AttributeError`。

### 3.2 ⚠⚠ 三处能力表必须一致（本项目最大的静默失效来源）

| 位置 | 内容 | 不一致的后果 |
|---|---|---|
| CLI `choices` | `train/train.py::--readout-dtype` | 用户能选到后端不支持的值 |
| `accel_readout.py::_DT` | 支持的 dtype 名集合 | 构造期 `ValueError` → 兜底回落 |
| `accel_readout.py::_unsupported_reason` | 拒绝原因（**带说明**） | 回落但**原因丢失** |

真实事故链（三次，一次比一次隐蔽）：

1. **fp8 能力表遗漏** → `--readout-dtype fp8`（当时的默认值）被静默回落到 numba CPU，
   整个 P84 精度改造在生产上**完全没生效且不报任何错**。
2. **`readout_conn_k>0` 撞拒绝表** → NPU 读出加速全程回落 CPU 也不报错
   （读出占端到端 89%）。**默认值撞拒绝表 = 静默关掉全部加速。**
3. **int16/int32 在 `choices` 放行但 `_DT` 没有** → 构造期 `ValueError` →
   `pick_readout_backend` 的 `except Exception` 兜底回落，且
   **`_accel_fallback_reason=None`（原因丢失）**。已于 P112 修：在能力表显式拒绝并说明原因。

**改dtype 时三处一起改，并跑 `verify_accel_readout.py` 的能力表用例。**

### 3.3 未实现的配置：回落 + 记原因

不允许「能跑但语义不同」。回落必须：

- 静默安全（不崩、不改数值语义）；
- **把原因写进 `readout._accel_fallback_reason`**，并在训练日志里打印：
  `[readout] backend=numba-cpu(回落) | fallback reason: ...`。

### 3.4 跨设备等价：用容差，不用逐位

torch CPU 与 numpy BLAS 的**归约顺序不同**，fp32 下 `max|Δ| ≈ 4e-06` 属正常。
跨库逐位相等**不是**正确性要求。同设备（torch CPU vs torch CPU）才逐位。

### 3.5 ⚠ 设备张量三条禁令

| 禁止 | 原因 | 正确做法 |
|---|---|---|
| 裸 `np.asarray(设备张量)` | 昇腾 `npu:0` 直接抛错 | 走 `phdnet.model.to_numpy()` / `_nelem()` |
| `torch.as_tensor(numpy数组)` 当权重 | **共享内存** + W 原地更新 → 调用方的参考权重被静默改掉 | `torch.tensor(...)`（复制） |
| 热路径上 `.item()` / `bool(tensor)` / `torch.equal` / `.cpu()` | **强制设备同步**，把P34 用 `nll_sync_every` 消除掉的同步又加回来 | 主机侧标记（如 epoch 整数）；`.item()` 改累积到设备、每 N 步同步 |

> ⚠ 真实事故：稀疏 gather 的缓存判据用 `torch.equal(cache_src, ht)` 判「是不是同一个 h」——
> 它逐元素比较后返回 Python bool，在 NPU 上是**每步一次硬同步**，发生在最热路径。
> 改为主机侧 epoch 整数后，**判据必须覆盖所有上传路径**：
> `forward()` 走 `_to_dev`、`forward_dev()` 走 `_staged_to_dev`，
> 漏掉一处就会把上一步的 gather 结果静默复用给新 h（实测：`max|Δ|=1.45`）。

### 3.6 ⚠ 门禁缺口

**「三处能力表一致」在 CI 上是验不全的**：CI 机器通常没有昇腾，`accel_readout="auto"`
会走硬件探测回落 numba，拿不到真实后端。可靠做法是直查 `_unsupported_reason`（纯函数、可测），
本项目已如此。**真正的确认只有一条**：服务器日志里 `[readout] backend=` 显示真在设备上。

---

## 4. 检查单：加一种精度 / dtype

### 4.1 三处 + 一致性

| # | 落点 | 说明 |
|---|---|---|
| 1 | `phdnet/readout.py::RO_DTYPES` | CPU 路径支持的 dtype 名 |
| 2 | `phdnet/backends/accel_readout.py::_DT` + `_unsupported_reason` | 加速路径（见 §3.2） |
| 3 | `train/train.py::--readout-dtype` 的 `choices` + `phdnet/sparse_encoder.py::resolve_model_dtype` | CLI 与编码器 |
| 4 | 检查点存/取（位模式） | `train/ckpt_1b.py`，低精度存 uint8/uint16，加载侧按 `ckpt_dtype` 解码闭环 |

### 4.2 ⚠ 低精度不是免费的速度收益：它会破坏学习规则

这是本项目最重要的精度结论（实测 `tools/probe_readout_precision.py`）。
判据：**同精度舍入后，元素是否真的变了**（变了 = 更新被保留）。

| 精度 | 非目标行更新保留率（`\|W\|~1e-2, \|dp\|~1e-6`） | 目标行 |
|---|---|---|
| **fp32** | **99.95%** | 100% |
| fp16 | 26.67% | 100% |
| bf16 | 5.79% | 100% |

目标行三者都 100%（目标更新 ~1e-2 ≫ 半 ULP），意味着低精度下
「**只有目标行的提升被保留**」= 学习规则退化为**纯 Hebbian**。
这是**语义损失，不是速度收益**。

> ⚠ 曾有判断「fp16 半 ULP 只有 bf16 的 1/8，能保住非目标行更新」——**实测推翻**：
> fp16 只是比 bf16 好，离正确还差一个数量级（26.67% vs 99.95%）。
> **半 ULP 的数量级估算不能替代实测。**
>
> ⚠ 判据本身的坑：基线必须经过**同一精度**的归约。
> 拿 fp32 值与 fp16 值直接比，会得到「fp16 保留 100%」的**反向假阳性**。

### 4.3 ⚠ 平台 × 版本相关的 dtype 支持

`numba` 的 dtype 支持**不是统一的**。真实事故：aarch64 numba **没有 float16 ArrayModel**
→ 在数据模型管理器里就拒绝 float16 数组 → numba GEMV 内部**必须用 fp32**。
（曾把 GEMV 内部改成 fp16，服务器直接崩，回滚。）

**加速器上也不统一**：fp8/fp4 在昇腾 910B + CANN 8.5 全 ERR01007（能建张量不能乘）；
int4 无矩阵乘单元。P189 起 fp8 走**位模式（uint8 承载）存储绕行**（+ fp16 迭代 +
可选 CPU 转换 int8 计算），显式 int 族请求仍禁（P163）。

---

## 5. 检查单：加一个 CLI 参数

argparse 在 `train/train.py`（生产）、`train/infer.py`、`tools/*.py`。

### 5.1 ⚠⚠ help 文本里的裸 `%` 会让 `--help` 崩 —— 已发生 6 次

`argparse._expand_help` 会对 help 文本做 `help % params`。任何未转义的 `%` 都会抛
`ValueError: unsupported format character`。必须写 `%%`。

**门禁已升级**：`tests/verifiers/verify_ms_stream.py` 现在会**跑遍全仓库每个 argparse 入口的
`--help`**（36 个文件），任何入口崩了都会被抓到。
（另有静态扫描辅助，但**只作提示不作判据**——AST 片段提取在跨行格式化表达式上会误报。）

改完参数**自己先跑一遍**：

```
python train/train.py --help ; echo "exit=$?"
```

### 5.2 ⚠ `default=` 与 help 文本必须一致

**已修的两个例子**：`--encoder-dtype` 实际 `fp32` 而 help 曾写 `fp64`；
`--m2-kernel` 实际 `serial` 而 help 曾写「默认 plain」。

⚠ **`--m2-kernel` 是修复范例，P113 已把它改对**：不只是把 `default=` 改成 `plain`，
**同时把 help 换成了实测数字**（6.92 → 1.28 ms/tok、5.41×、四个采样点 PPL 逐位相同）。
**这比删掉 help 里的数字更好** —— help 是用户唯一不用翻文档就能看到口径的地方。

**help 里不要重复陈述默认值**，或确保陈述与 `default=` 一致。

### 5.3 其它

- 百分比、倍数写在 help 里一律 `%%`（如「占 39%%」）。
- 新参数若影响热路径，加一个开关默认关闭，让用户显式开启。

---

## 6. 检查单：加一个验证器

放 `tests/verifiers/verify_*.py`。`python tests/verifiers/verify_<名>.py` 即可运行。

### 6.1 分层结构（照这个写）

| 层 | 验什么 | 失败意味着 |
|---|---|---|
| A 数值等价 | 与独立参考实现逐元素对拍 | **实现算错了** |
| B 调用面完整 | 属性/方法存在且语义对 | 生产某条路径会崩 |
| C 拒绝路径 | 不支持的配置 fail-fast | 会「能跑但语义不同」 |
| D 零回归 | 未涉及的路径行为不变 | 你改坏了别的地方 |

### 6.2 ⚠ 必须打印实测数字

每条 check 打印 `max|Δ|`，例如 `PASS A1 前向一致 [max|Δ|=8.103e-08]`。
**不要只打 PASS/FAIL** —— 数字是下一个人判断「这是否在容差内」的唯一依据。

### 6.3 ⚠ 容差判据的基线

基线必须经过**同一精度/同一路径**的归约。见 §4.2 的反向假阳性事故。
尺度也要用**当前**值的量级（更新后就更新后的量级，不是更新前的副本）。

### 6.4 ⚠ 写门禁时别把自己扫进去

真实事故：给 `verify_ms_stream.py` 加「跑遍所有 argparse 入口」的检查，
而它自己含 `add_argument` 文本且有 `__main__` → **自我递归调用把自己跑超时**。
排除 `os.path.abspath(__file__)` 与 `tools/archive/`（归档脚本不是活代码入口）。
子进程调用要加 `try/except TimeoutExpired`。

### 6.5 ⚠ 门禁红了先归因

`门禁红了必须归因，不能等它变绿`。先问两个问题：
① 它什么时候开始红的？② **它断言的是不是已废弃的行为？**

真实案例：`verify_accel_readout.py` 的 A3 长期断言「稀疏必须回落 numba」——
P111 恰恰是要取消这个回落。改代码前若不看清断言，**会把正确的修复当成回归**，
或者反过来——为了让门禁绿而把修复改回去。

---

## 7. 检查单：加一个工具脚本

放 `tools/`，argparse，文件头注明**生产轨 / 非生产轨**。

性能测量脚本的**诚实话术**（缺一条结论就不可信）：

| 要求 | 原因 |
|---|---|
| 只测量**不推荐** | 建议会变成无实测支撑的论断 |
| 读出精度**锁 fp32** 并说明理由 | 否则构成「bf16 稠密 vs fp32 稀疏」的混淆对比 |
| 数字一律 **best-of-N** | 本机单次噪声可达 3× |
| 注明**平台 + 档位 + commit** | 脱离口径的数字无意义 |
| 消融计时**不可信** | 会被 JIT 编译时间污染（需先预热） |
| 检测发散并标 `diverged` | 不能把爆掉的数字当正常结果打出去 |
| **欠训练告警必须打** | 小 token 预算下稀疏臂「PPL 更优」是**预算不足的伪影**，不是稀疏的结构优势 |

`tools/bench_local.py` 已明确「**不产出文档数字**」并提供三个 A/B 开关——
这是「x86 结论不构成昇腾证据」纪律的落地。

---

## 8. 检查单：文档与记忆同步

| 改动类型 | 要更新的文档 |
|---|---|
| 加机制 / 改容量 | `docs/PHD-Net_架构设计.md` 机制表 + 容量账 |
| 性能数字 | **只更新** `docs/PHD-Net_性能评估与迭代方案.md`（唯一出处），别处链接引用 |
| 后端 / dtype / 昇腾坑 | `docs/PHD-Net_硬件后端适配报告.md` |
| 并行度 / 线程 / 加速器上限 | `docs/PHD-Net_并行与加速架构分析.md` |
| CLI 参数 | `train/README.md` |
| 任何跨会话硬约束 | `.workbuddy/memory/MEMORY.md` |
| 当天做了什么 | `.workbuddy/memory/YYYY-MM-DD.md`（append-only） |

写文档的硬规则见 [`文档写作规范.md`](文档写作规范.md)：**一个事实只在一处写**、
数字必带口径、历史值必须显式标注、参数以源码为准（写前先 `grep add_argument` 核）。

---

## 9. 常用开关速查（调参时先看这里）

| 开关 | 作用 | 动它要注意 |
|---|---|---|
| `--readout-conn-k` | M6 稀疏化入边数（默认 128） | >0 时加速后端走 gather-GEMV；**只支持均匀 k**，非均匀行宽 fail-fast |
| `--readout-powlaw-alpha` | **幂律异质连接**指数（默认 **0.0 = 关闭**） | k_i ∝ 词频^alpha，行宽不等、总 nnz 仍受预算控制。**alpha=0 逐位等价于均匀 k**。⚠ 开 >0 会**回落 numba 读出**（加速器不支持非均匀行宽）；⚠ 词频是**代理值**（`1/rank`，词表是字典序非频次序）→ 开启前须 A/B |
| `--readout-powlaw-kmin/-kmax` | 幂律行宽夹紧（默认 1 / 0=不设上限） | 长尾不归零、高频不铺满 n_h |
| `--sparse-fwd-kernel` | 稀疏前向算子（默认 `mulsum`） | `mulsum`=物化 (n_out,k) 临时张量；`einsum` 不物化但**非逐位**且昇腾未实测 |
| `--ltm-imprint-amortize` | ⚠ **已于 P122 移除**，只接受 1 | 传 >1 直接 `ValueError`（实测收益为 0 且N≥2 污染权重）|
| `--lang` | 终端输出语言（默认 **en**） | ⚠ **对训练结果零影响**。语料过滤是 `--data-lang`（会改变结果）|
| `--readout-dtype` | 读出计算精度（默认 **fp32**） | 低精度破坏 p − t 规则（§4.2） |
| `--encoder-dtype` | M1 权重存储精度（默认 fp32） | 昇腾走平台自适应 GEMV，与 dtype 无关 |
| `--m2-kernel` | M2 推理核（默认 **`plain`**） | 三态见下方 §9.1。**`plain` 在昇腾赢 5.41×**（P113 实测）；`fused` 在昇腾退化 3–4×（已否）。门禁 `verify_m2_kernels.py` |
| `--nll-sync-every` | nll 同步周期（默认 8） | N>1 时消除每步硬同步 |
| `--numba-threads` | prange 线程上限（默认 8） | 191 核上 P22 实测 1→6 线程仅 1.16×（带宽饱和） |
| `--accel` | 读出设备（默认 auto） | `cpu/off/numba` = 强制 numba 路径 |
| `--omp-proc-bind` | 绑核（默认**开**，只设 `OMP_PROC_BIND=close`） | **不要**加 `OMP_PLACES`（191 核实测慢 8×，已回滚） |
| `--ckpt-dtype` | 检查点存储精度（默认 fp16） | 位模式存取，加载侧解码闭环 |
| `--torch-compile` | 图优化（默认**关**） | 只在首次调用编译 → 首次执行时捕获失败并永久回落 eager |
| `--step-profiling` | 九段耗时分解 | 判断优化是否生效看这个，不看「检测到设备」 |

### 9.1 `--m2-kernel` 三态：各自走什么，为什么 `plain` 在昇腾赢

`phdnet/sparse_pc.py::infer()` 是一个**三态分派**。三态不是「同一实现的三档并行度」，
而是**两条不同的代码路径**（融合 / 非融合），加一个并行度档：

| 取值 | 走什么 | 并行性 | 昇腾实测 | 状态 |
|---|---|---|---|---|
| **`plain`**<br>（**默认**） | `_csr_matvec` × 5 次<br>（`sparse_pc.py:43-53`） | ✅ **`prange(n)` + `parallel=True`**，行内按 `indptr` 顺序累加、**行间并行** | **1.28 ms/tok**（端到端 12.09） | ✅ **P113 实测采纳为默认** |
| `serial` | `_pc_infer_fused_serial`<br>（P99 加） | ❌ **单核**。融合的「一次调用 + 核内复用中间数组」保留，去掉 `parallel=True` | **6.92 ms/tok**（端到端 17.63） | **历史默认**（P113 前），现为显式 opt-out |
| `fused` | `_pc_infer_fused`<br>（P52） | ⚠ `parallel=True` 但有 **10 个 prange 屏障区** | fused 20–27 vs plain 6.7–11.9 → **慢 3–4×** | ❌ **已否决**，保留仅供对照 |

⚠ **两条容易被写错的纪律**：

1. **P76 只否了 `fused` 的「融合 + 10 区prange 屏障」，没有否 CSR 行级并行本身。**
   `--m2-kernel plain` 走的就是 `_csr_matvec`，它从 P43 起就带行级 prange，
   与融合核是两个独立维度。**把「融合核被否」写成「CSR 行级并行被否」是错的**——
   这条已由 P113 的实测背书（5.41×，不是推论）。
2. **P76 那组 plain 值 6.7–11.9 ms/tok 与现在的 1.28 不是同一口径**
   （P76 A/B 期间、线程配置未标、不同批次 run），**不可并列比较**。

**为什么 plain 在昇腾赢、在 x86 测不出来**（P113 的实测补充）：
同一份CSR 行级 prange，x86 上**从未 A/B**（性能文档 §五），
昇腾上赢 5.41×。这印证了性能文档的铁律③：
**x86 的性能结论不构成昇腾的证据，连「无差别」也不构成。**

**门禁**：新增机制改到 M2 的核时必须跑 `tests/verifiers/verify_m2_kernels.py`（**11 例**）：
A1 fused vs serial、A2 plain vs serial（容差 1e-7 量级——plain 是非融合路径，
内部 `np.tanh`/`np.clip` 的归约顺序与融合核的 `math.tanh` 不同）、
A3 `n_steps=1/2/3`、B 类断言行级 prange **不改变求和顺序**。
P113 的零回归证据见性能文档 §2.1（四个采样点 sliding PPL 与 serial 运行**逐位相同**）。

⚠ **库默认 ≠ CLI 默认**：`phdnet/config.py:198` 的 `pc_fused_kernel` 默认 `True`（= `fused`），
被 `train.py:495-496` 在 CLI 层覆盖为三态。**引用「M2 默认用哪个核」必须指明是哪一层**
（**CLI = `plain`；库 config = `fused`**）。

---

## 10. 收尾四步（fhz 固定要求）

1. **对拍 / 回归 / 冒烟验证**——fast 9/9 + 相关 verifier + `py_compile` + `--help` 实跑。
2. **写记忆日志**——当天做的事追加到 `.workbuddy/memory/YYYY-MM-DD.md`；
   跨会话硬约束才写进 `MEMORY.md`。
3. **commit + push**。⚠ **push 必须绕过凭据助手**，否则永久挂起：
   ```
   git -c credential.helper= -c credential.helper="!C:/Users/ASUS/.workbuddy/binaries/PortableGit/versions/1.2.0/mingw64/bin/git-credential-wincred.exe" push origin main
   ```
   （原因：便携 git 的首个 helper `git-credential-helper-selector.exe` 在非交互 shell 下永久挂起。）
   push 后 `git fetch origin main` 同步跟踪引用。
4. **报告工作树干净**（`git status -sb` 显示 `## main...origin/main`）。

### 收尾清缓存（fhz 明确要求，不等提醒）

清：`__pycache__`、`*.pyc`、`*.nbc`/`*.nbi`、`*_*.log`、`.tmp_*`、临时克隆目录。
**保留**：`outputs/numba_cache`（numba 持久化编译缓存，冷启动 3.96s → 2.60s）、
被文档引用的证据日志、数据集、生产模型产物（删前先确认）。

---

## 11. 三条最容易重犯的错误

| 错误 | 后果 | 根治 |
|---|---|---|
| **补丁未落盘**（`str.replace` 无匹配静默成功） | 以为改了其实没改，测的是旧代码 | 每次 replace 后 `grep` 断言新文本存在 |
| **改多行源码用 heredoc** | `\n` 变成真换行 → 语法错 | **只用 Edit/Write 工具** |
| **形状断言代替数值对拍** | 权重被丢掉这类 bug 全程绿灯 | §2.3 / §6.1 的 A 层 |

（第一条在 BUGS.md 里是**最高频家族**，5 次以上；第二条 3 次；第三条见 §3.5 的 `torch.equal` 事故。）