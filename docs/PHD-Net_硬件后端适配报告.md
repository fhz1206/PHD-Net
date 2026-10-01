# PHD-Net 硬件后端适配报告

> **适用范围**：加速后端的**能力矩阵、精度能力、机制性代价、昇腾真机踩坑、迁移路径、诊断入口**。
> **不写**：训练怎么跑（→ `../train/README.md`）、机制设计（→ `PHD-Net_架构设计.md`）、
> 任何性能数字（**唯一出处 `PHD-Net_性能评估与迭代方案.md`**，本文只链接不复制）。
> **数据截止：2026-10-01（P113）。**
>
> **相关文档**：
> 子包文件索引与每层设备归属 → `../phdnet/backends/README.md`；
> 缺陷台账（症状 → 根因 → 门禁缺口）→ `../BUGS.md`；
> 性能数字唯一出处 → `PHD-Net_性能评估与迭代方案.md`；
> 为什么吃不满 191 核 → `PHD-Net_并行与加速架构分析.md`；
> 写作规则与事实基线 → `文档写作规范.md`。

---

## 一、后端矩阵

### 1.1 三条轨的分工

| 轨 | 入口 | 覆盖 | 权重 | 状态 |
|---|---|---|---|---|
| **生产训练轨** | `train/train.py`、`train/infer.py` | numba CPU 跑 M1–M5 全机制 + **torch 加速读出**（M6）混合栈 | 生产 npz ckpt | **唯一生产轨** |
| **加速器读出** | `phdnet/backends/accel_readout.py::AccelReadout` | 只有 M6 读出，W 常驻设备 | 与生产 ckpt 同权重 | 生产默认 `accel_readout=auto` |
| **torch 全栈轨** | **已删除**（P30，commit `4120a5b`） | 曾覆盖 M1–M6 全 torch 化 | 与生产 ckpt **不通用** | ❌ **不存在了，不要试图复活** |

**torch 全栈轨已删除**：原 `phdnet/backends/torch_lm.py` 里的 `TorchPHDNet` /
`TorchWordLM` / `TorchSparsePC` / `TorchLTM` 等约 800 行，加上
`tools/train_torch_lm.py`、`tests/verifiers/verify_torch_lm.py`，在 P30 一次性删除
（合计约 1,400 行）。删除理由三条，均写在 `torch_lm.py` 现存 docstring 里：

1. **权重与生产 npz 检查点不通用**（不同实现、不同初始化顺序）；
2. **缺 20 项机制**，构造时显式 `NotImplementedError`（清单见 §六）；
3. 生产加速路径已由 `AccelReadout` 单独实现（只迁读出，不动 numba 主循环）。

`torch_lm.py` 现在**只剩 `resolve_device()`**，因为它是生产依赖
（`accel_readout.resolve_accel_device` 经它把 `auto` 解析成具体设备）。
`tools/train_torch_lm.py` 已不存在，但 `train/train.py:389` 与 `train/infer.py:222,248`
的提示文本仍在引用它 —— 见 §八「源码与文档不一致」。

### 1.2 加速器读出的设备覆盖

`resolve_device("auto")` 的择优顺序（`torch_lm.py:41-48`，与
`phdnet/device.py::probe()` 声明顺序一致）：

**昇腾 NPU → ROCm → CUDA → DirectML → CPU**

`probe_multi()` 判定「auto 有没有加速器」用的条件是
`v.get("ok") and v.get("count")`（`accel_readout.py:833`）—— 即**必须至少 1 张卡**。
`auto` 下无加速器时直接构造 numba `Readout`（连回落原因都不记，因为不是回落）。

| 设备 | 键 | 探测方式（`torch_backend.probe_devices()`） | 验证状态 |
|---|---|---|---|
| numba CPU | `cpu` | 恒可用 | ✅ 生产主力 |
| CANN·昇腾 NPU | `npu` | `import torch_npu` 即注册，`torch.npu.is_available()` | ✅ **真机实测** |
| CUDA（NVIDIA） | `cuda` | `torch.cuda.is_available()` | ⚠ 结构验证（无硬件） |
| ROCm（AMD） | `rocm` | `torch.cuda.is_available()` 且 `torch.version.hip` 非空 | ⚠ 结构验证（无硬件） |
| DirectML | `dml` | `torch_directml.device()` | ⚠ 结构验证（无硬件） |

⚠ **CUDA / ROCm / DirectML 三行只有结构验证**。引用这三行的任何陈述都必须标注
「未在真机验证」。ROCm 走 HIP 化的 `cuda` 接口，算子代码与 CUDA 完全相同。

### 1.3 多设备：模型并行，不是 DDP

`phdnet/backends/multi_device.py`。**为什么不是 DDP / DataParallel**：PHD-Net 是
逐 token 事件驱动的稀疏网络 —— 无 batch 维、无注意力、无反向图，全局状态
（WM 情景缓冲 / STDP 印迹 / LTM）跨样本连续携带。DDP 需要可分梯度 + batch 归约，
DataParallel 需要 batch 轴可切，本网络两者都没有。

**唯一实现的并行 = 读出列并行**（`MultiDeviceReadout`）：读出 W ∈ R^(V×H) 按
**词表行（V 维）**切分到各卡。每行 dot 独立、更新逐行外积互不依赖 → 无跨卡算子调用。

- 每步通信仅两次小向量：全局 logits y（V×4B 下发）+ dp 分片（上行）；
- softmax 主回路在主卡 fp32（与单卡 P9 协议一致）；
- `plan_parallel()` 如实报告 `not_parallel`：**PC 栈（稀疏主循环）/ STDP 印迹 /
  LTM 长时记忆 / 分词编码**；
- `configure_host_threads()` 在多卡时把 OMP/MKL/NUMEXPR/OPENBLAS 收敛到 1
  （必须在 numpy/torch 初始化前调用才完全生效）；单卡不动。

**等价性判据**：跨**不同**卡只做结构验证（容差一致）；**同设备分片 vs 单设备逐位一致**
（每行 dot 与分片顺序无关，softmax 在 cat 后的同一向量上计算），
由 `tests/verifiers/verify_multi_device.py` 给逐位断言。

LTM / 印迹分片只有**计划层**（`shard_ranges` 给均衡神经元区间），跨神经元 STDP
共激活对是唯一跨卡通信需求 —— **不提供假实现**。

---

## 二、实测环境

### 2.1 昇腾服务器（唯一的硬件结论来源）

| 项 | 值 |
|---|---|
| 芯片 | **Ascend NPU 910B4**（Atlas A2） |
| 平台架构 | **aarch64** |
| CPU 核数 | **191** |
| HBM | ~1.6 TB/s |
| **CANN** | **8.5** |
| **torch_npu** | **2.9.0** |
| torch | 2.9.0+cpu |
| Python | 3.11 |

⚠ **本文所有硬件能力结论都只在这个组合上成立**。换芯片型号、CANN 版本、
torch_npu 版本或架构（x86 / aarch64）中的任何一项，结论都必须重新实测。

**已知无害告警**：torch_npu 启动时会刷
`owner does not match` / `Permission mismatch` 一类告警，根因是
**CANN 安装目录属主与当前运行用户不一致**（镜像拷贝/多用户共用导致）。
判定为无害的依据：① 告警出现在 import 阶段而非算子执行阶段；
② 后续 `torch.npu.is_available()` 返回 True 且设备名可读；
③ `tools/accel_doctor.py` 的**真实规模张量试分配 + 前向 matvec 成功**；
④ 无 traceback、无 `ERR` 码。**判据是「算子能不能跑」，不是「有没有告警」** ——
这条与 §五 踩坑 4（能力探针必须试真实算子而非只看张量能否创建）是同一条纪律。
若 (3) 失败，则不是无害告警，按 §七 诊断入口排查。

### 2.2 x86 本机（仅用于 CPU 路径与对照）

x86 本机**只用于验证 CPU 侧结论**（numba 主循环、CPU torch 的 fp8 能力）。
`docs/文档写作规范.md` §2.6 记录了三次「x86 与昇腾方向相反」的反噬：
M1 fp32（x86 快 1.80× / 昇腾慢约 70×）、M2 融合核、`OMP_PLACES=cores`（昇腾慢 8×）。

→ **x86 的性能结论不构成昇腾的证据。** 本文引用的 x86 数据只有一类：
精度机制的**位级判据**（§四），它与芯片无关。

---

## 三、精度能力矩阵

### 3.1 逐档结论

`--readout-dtype` 的**实际能力**由两处共同决定：`accel_readout.py::_DT`（构造期
白名单）与 `_unsupported_reason()`（配置期拒绝表）。两处**必须同步** —— 见 §五 踩坑 15。

| 精度 | 加速后端 | 结论与依据 |
|---|---|---|
| **fp32** | ✅ 可用 | **默认可用**；P110 实测后 `--readout-dtype` 默认为 `fp32` |
| **fp16** | ✅ 可用 | 位保留率 26.67%（§四）→ 显式 opt-in |
| **bf16** | ✅ 可用 | 位保留率 5.79%（§四）→ 显式 opt-in |
| **int8 码本** | ✅ 已实现 | 存储 int8 + 计算 fp16 + 更新走 fp32 域再重量化（P104/P105） |
| **fp8** | ❌ **所有加速器已禁用** | 昇腾 910B + CANN 8.5 + torch_npu 2.9 全 ERR01007 |
| **fp4** | ❌ **所有加速器已禁用** | fp4 的 MX 块缩放同样无算子 |
| **int4** | ❌ 拒绝 | 910B **无 INT4 矩阵乘单元** |

### 3.2 fp8 / fp4：实测证据

`tools/probe_fp8.py` 在 Ascend910B4 / CANN 8.5 / torch_npu 2.9 上的结果
（结论同时写在 `probe_npu_quant.py` 的 docstring 里）：

- `float8_e4m3fn` 与 `float8_e5m2` 在 **npu 上 create / cast / matmul 全部 ERR01007**
  （`Float8_e4m3fn has not been supported`）；
- `torch.ops` 里**零个 fp8 专用算子**；
- **CPU torch**：能创建 fp8 张量、能 cast 往返，但 **matmul 抛
  `NotImplementedError: "addmv_impl_cpu" not implemented for 'Float8_e4m3fn'`** ——
  本机（x86，torch 2.14.0+cpu）实测复现，与 910B 的失败点**同源**：
  都没有 fp8 **矩阵乘**单元，只有张量与 cast。

⚠ **910B4 本身没有 FP8 单元**（910C 部分支持、950 原生 MXFP8/MXFP4）。
所以 ERR01007 是**硬件限制**，不是软件 bug —— 这也意味着 CANN 跟进不会翻回，
要翻回必须换芯片。`probe_fp8.py` 的三档判定逻辑保留在源码里，换硬件时重跑即可。

### 3.3 int8 码本路径（P104/P105）

「int8 码本」在加速后端**已实现**，三层分开：

- **存储**：`W` 是 `torch.int8` 码本，per-tensor `scale = 2·max|W|/127`
  （2× 余量纪律：训练中权重增长 ≤2× 不需要重标定；越界 clamp 饱和，不回绕）；
- **计算**：每步反量化 `codes.float()*scale` → cast 到 fp16 再 matmul。
  **不缓存 fp16 副本** —— W 每 step 都在更新，缓存必然过期；
- **更新**：`_int8_update()` 在 **fp32 域**算 `dp⊗ht`，再 `round`(RNE)/`clamp(±127)`
  重量化写回。scale 固定不重标定。

两条回落保护：① 构造期做一次探测（int8 反量化 + fp16 matmul 真跑），
失败则**永久回落 fp16** + `RuntimeWarning`；② 运行期 `forward_dev` 的
`try/except` 兜底 —— 因为**探测通过不代表所有 shape 都不炸**。

**精度取舍（知情）**：int8 步长 ≈ `max|W|/127` ≈ 8e-3 ≫ 非目标行 `|dp|` ≈ 1e-6
→ 非目标行更新被量化吃掉。目标行 `|dp|` ≈ 1 远大于步长，**必须完整保留**
（verifier 用 fp16 参考对拍把关）。这是「int8 存储降 4× 访存」的代价，不是 bug。

⚠ **`fp8` 现在是 `int8` 的别名**（P100 正名：`fp8` 本来就是 1 字节码本）。
`_DTYPE_ALIASES = {"fp8": "int8"}`，与 `phdnet/readout.py` 同口径。
所以配置里写 `dtype="fp8"` **不会**被 `ValueError` 拒绝、也**不会**静默回落 ——
它在加速后端走 int8 码本，在 numba CPU 走 P9/P12 位算法量化核。

### 3.4 稀疏读出锁定 fp32

`AccelReadout` 的稀疏模式（`conn_k>0`）**强制 fp32**（`accel_readout.py:170-175`）：
低精度会丢弃非目标行更新（§四实测），而稀疏读出的存在意义正是省流量，
不能再叠加语义损失。稀疏模式下 `_compiled` 永久为 `False`（融合核走稠密 `addmm_`，
稀疏不适用）。

### 3.5 ⚠ aarch64 numba 没有 float16 ArrayModel

**实测踩中并已回滚**（commit `b39a82c`，P108）：曾试图把 M1 numba GEMV 的
W/x/b 转 fp16 进核以省访存，aarch64 上直接抛
`NotImplementedError: float16，数据模型缺失`（拒绝发生在**数据模型管理器**里，
不是编译期）。**已回滚为 fp32 进核**；fp16 收益改由**存储 / 检查点侧**拿（P84）。

→ **aarch64 numba GEMV 内部必须 fp32**。任何「numba 核内部换低精度」的优化
在昇腾上都不成立，要换必须先在目标架构验证数据模型是否存在。

---

## 四、关键机制代价：低精度破坏学习规则

**这是本项目最重要的硬件相关结论**：读出低精度不是「访存换速度」的取舍，
而是**语义损失**。

### 4.1 测量方法

`tools/probe_readout_precision.py`（纯测量，不改任何生产代码）。
判据 = **同精度舍入后元素是否真的变了**（变了 = 更新被保留）：

```
baseline = _round(W0,     dtype)     ← 必须同精度
updated  = _round(W0 + dp, dtype)
保留率   = (updated != baseline).mean()
```

### 4.2 实测结果（`|W| ~ 1e-2`，4096 行，非目标行）

| 精度 | 非目标行保留率 | 目标行保留率 |
|---|---|---|
| **fp32** | **99.95%** | 100% |
| **fp16** | **26.67%** | 100% |
| **bf16** | **5.79%** | 100% |

目标行三者都 100%：目标更新 ~1e-2 ≫ 半 ULP。

→ 低精度下「只保留目标行的提升」= 学习规则**退化为纯 Hebbian**
（`p − t` 变成只有 `p[c] − t[c]` 那一项生效，非目标行完全不更新、不抑制）。
**这是语义损失，不是速度收益。**

### 4.3 ⚠ 判据陷阱：反向假阳性

第一版判据拿 **fp32 值**与 **fp16 值**直接比 → 得到「**fp16 保留 100%**」的
**反向假阳性**。原因：fp16 与 fp32 的舍入结果几乎总不相等，判据恒为真。

**基线必须经过同一精度的舍入**，否则整个测量无效。这条已写进
`probe_readout_precision.probe()` 的 docstring。

### 4.4 后果与现状

`train/train.py` 启动日志按精度分叉（`:701-711`）：fp32 打印「精确 `p − t` 规则」，
低精度打印警告并**点名实测保留率**。CLI 默认 `fp32`（P110，fhz 授权）；
低精度降级为**显式 opt-in**。

**教训**：**半 ULP 的数量级估算不能替代实测**。P105 曾认定「fp16 半 ULP 比 bf16
小 8 倍，能保住非目标行更新」→ 实测发现 fp16 只是「比 bf16 好」，
**离正确还差一个数量级**，73% 的非目标行更新仍被丢弃。修正过程还走了弯路：
先改 bf16→fp16（同样无效），隔一天才被实测推翻。

---

## 五、昇腾踩坑速查（16 条）

每条格式：**症状 → 根因 → 修复/绕过 → 是否会复发**。
「是否会复发」一栏是本表最有用的部分 —— 它决定这条是「已修」还是「只是被绕过」。

| # | 症状 | 根因 | 修复 / 绕过 | 会复发？ |
|---|---|---|---|---|
| 1 | NPU 首次真跑生产训练即崩 `FakeTensor - None` @ `accel_readout.py` | P45 把「`target_idx` 路径就地改 p」只实现在 **eager 分支**，`torch.compile` 融合核仍是 `dp = p - t32`，`t32=None` 时崩 dynamo。既有验证只走 target 数组路径 → 恰好漏掉崩的那条 | 两条路径**逐行等价**（`p − onehot` ≡ `p[c] −= 1`，数值也等价）；新增 `verify_accel_readout_p55.py` **2×2 矩阵**（target 数组/`target_idx` × eager/compiled） | **会**。任何新增的 eager 分支都可能漏改融合核。门禁 A4 接口扫描只查属性存在性，查不到分支等价性 → 必须靠 2×2 矩阵 |
| 2 | `torch.compile()` **只在首次调用才编译**，构造期 try/except 是假护栏 → 编译失败崩生产 | inductor **惰性编译**：构造期 `torch.compile()` 只是包装、不会失败，**首次调用**才真正编译 | 捕获点放在**首次执行**处；首次失败即 `self._compiled = False` **永久回落 eager** + `RuntimeWarning`。`--torch-compile` **默认关** | **会**。Windows 无 `cl`、NPU 上 inductor 异常都可能发生 |
| 3 | `torch.compile` + `reduce-overhead` 与「W 原地 mutate」本质冲突 | cudagraphs 捕获的张量不允许被后续 kernel mutate → 每次调用打印 `skipping cudagraphs due to mutated inputs` 并**静默退回**无 graph 模式 | **锁 `torch_compile_mode="default"`**（只融合 kernel，不启用 cudagraphs）。改成非原地重新分配会让流量翻倍，更不可接受 | **会**（只要有人传 `--torch-compile-mode reduce-overhead`）。功能正确，只是拿不到 graph 收益 |
| 4 | 设备探针只测**张量创建**→ fp8 能建不能乘，假通过后炸在 matmul | 第一版探针只测 `torch.zeros(..., dtype=fp8)`。CPU 能分配 fp8 → 判「可用」 | 探针必须试**真实算子**：`tools/accel_doctor.py` 试分配**真实规模**张量（默认 73,958×3,072 fp32）+ 一次前向 matvec。`probe_fp8.py` 三档（create / roundtrip / **matmul**）分开报 | **会**。新增 dtype 或新平台时必须重跑三档 |
| 5 | `torch.npu.utilization()` 同步设备流，破坏被观测的流水 | 它内部会同步设备流 —— 每 `log_every` 采样一次就在训练热路径上砍一刀全局同步，与「CPU 预计算提前 / NPU 流水」直接冲突 | AI Core% 改走 **`npu-smi` 子进程**（外部查询，**零同步**）；`memory_allocated` 是 host 侧计数器，保留 | **会**。任何新增的 torch 侧设备指标都要先问「它同步吗」 |
| 6 | `npu-smi` 不在 PATH → AI Core% 恒 `--` | 训练是**直接 `python train.py` 启动**、没 source Ascend 的 `set_env.sh`（能跑 `watch npu-smi` 的那个 shell 是交互式） | **三级路径解析**：环境变量 `$NPU_SMI_PATH` → `shutil.which("npu-smi")` → 常见安装路径候选表。只提示一次；首次成功打印实际路径 | **会**（换机器/换安装方式）。三级解析已覆盖主要情况 |
| 7 | `npu-smi` 输出**没有 Bus-Id 列** → 路径已定位但仍恒 `--` | 解析器找「含 `0x` 总线号的数据行」启发式，**该机输出没有这一列** → 所有数据行被跳过 | 改为**按表头定位列**：找含 `AICore` 的表头行（回退小写 `aicore`）→ 切列得 AICore / HBM-Usage 列序 → 按列号取值。**与型号无关** | **不会**（表头定位与型号解耦）。但门禁只能测**合成布局** —— 真实格式仍靠人读那 14 行原始输出 |
| 8 | 裸 `np.asarray(设备张量)` 在 `npu:0` 抛 `can't convert npu:0 device type tensor to numpy` | torch 设备张量不能直接进 numpy，检查点保存与参数统计都会踩 | 统一走 `phdnet/model.py::to_numpy()`（`.detach().to('cpu')`）与 `_nelem()` | **不会**（两个入口已收口）。⚠ 新增代码不得裸 `np.asarray` 设备张量 |
| 9 | `torch.as_tensor` / `from_numpy` 与传入数组**共享内存** | 这两个 API 不复制。W 是原位更新的 → 调用方（例如比较用的参考权重）被**静默改掉** | 权重类张量必须用 **`torch.tensor`**（复制）。`accel_readout.py:240` 有显式注释 | **会**（新人常写 `as_tensor`）。这是 API 语义，不是 bug |
| 10 | 热路径设备同步：`.item()` / `bool(tensor)` / `torch.equal` / `.cpu()` 都**强制同步** | 事故：稀疏 gather 缓存判据用 `torch.equal(self._cache_g_src, ht)` → 返回 Python bool → **每步一次硬同步**（P112 实例；当时服务器日志实测读出 7.68 ms/tok、占 43.5%，P113 口径为 7.83 ms / 64.7%），把 P34 用 `nll_sync_every` 消除掉的同步又加了回来 | 改为**主机侧 epoch 整数判据**：每次上传新 h 递增 `_ht_epoch`，`_sp_gather` 只比整数。语义等价（同一 epoch = 同一个 h），**零设备交互**。**系统性排查手段见踩坑 16** | **会**。⚠ 不能简化成「盲目复用缓存」——那会在 h 变化时静默用错结果。commit `7701138` |
| 11 | `W.add_(torch.outer(dp,ht))` **物化与 W 同尺寸的临时张量** | `torch.outer` 先生成完整 (n_out, n_h) 矩阵再被 `add_` 读回。1B 档 = **867 MiB** 临时张量，单步多 **1.73 GiB** 带宽（≈ 总流量的 **40%**） | 改 `addmm_` 做 **rank-1 AXPY**：`W.addmm_(dp.reshape(-1,1), ht.reshape(1,-1), alpha=-eta)`，不产生临时张量。**数值逐位相同** | **不会**（已换成 AXPY 口径）。⚠ 稀疏模式下 `addmm_` 是稠密的，等价写法是 gather 后逐行 `add_`（流量 `(n_out,k)` 而非 `(n_out,n_h)`） |
| 12 | 混合 dtype（fp32 W @ fp64 x）**掉出 BLAS** 走逐元素慢路径 | numpy 不做类型提升。两次误判：P61/P75 踩中 | 输入 dtype **必须跟随权重 dtype**（输入只有 `n_in` 个元素，转换代价可忽略）。慢 5.9×（x86）→ 约 70×（昇腾 aarch64） | **会**。新增 dtype 维度时每个入口都要检查 |
| 13 | 「探测到设备」≠「能用」 | `torch_npu` 与 `torch` **版本严格配对**（如 2.5.1 ↔ 2.5.1）。不匹配会出现探测成功但算子不可用 | `tools/accel_doctor.py`：环境矩阵（含 CANN 版本）+ **试分配真实规模张量 + 前向 matvec**。分配失败时明确打印「该设备被探测到但实际不可用（常见：算子缺失 / 显存不足 / 版本不配对）」 | **会**（升级 torch 或 torch_npu 时）。这是环境问题，不是代码问题 |
| 14 | 换后端只实现主要方法 → **生产首个 step 崩** | `AccelReadout` 必须对齐 `Readout` 的**全部**访问面：`__call__` / `forward` / `learn` / `learn_softmax` / `W` / `W_cpu` / `load_W` / `n_synapses` / `conn_k` / `hidden` / `stats()` / `dtype_name` / `_csr`。早期版本缺 `__call__` → `TypeError: not callable`；缺 `conn_k`/`stats()` → **保存检查点时**才崩 | **接口完整性自动扫描**：`verify_accel_readout.py::A4` 用正则扫全仓库 `readout\.([a-zA-Z_]\w*)` 访问面，断言加速后端**全部具备**（这比人工列举更强：新增访问点自动进扫描范围） | **会**（新增 `readout.X` 访问点时）。⚠ `conn_k` / `hidden` / `W_cpu` 必须与 numba `Readout` 同为**只读 property**，否则两侧语义不对称、ckpt 的读法会分叉 |
| 15 | **能力表与实现不同步 → 静默关掉全部加速，且零报错** | 两次实战：① fp8 能力表遗漏 → P84 实现了 fp8 并设为默认，**忘了从 `_unsupported_reason()` 删掉** → 生产日志一直 `numba-cpu(回落)`，整轮改造在生产**等于没生效且不报任何错**（本项目最严重的一条）。② `readout_conn_k>0` 撞拒绝表 → 稀疏读出把整个 NPU 读出加速**全程回落 CPU 也不报错**，而读出占端到端 **89%** | 拒绝表与 `_DT` 必须**同步**；加新 dtype 时两处一起改。`verify_accel_readout.py::A3` 对拒绝路径给断言 | **会**。**核心教训：默认值撞拒绝表 = 静默关掉全部加速。** 「回落 + 记原因」是好纪律，但**一个默认值命中它就等于默认关掉加速** —— 所以默认值必须与能力表一起审计 |
| **16** | **P112 实例：热路径设备同步点，只能靠静态扫描系统性找出** | 热路径上的**显式同步原语**（`torch.equal` / `.item()` / `.cpu()` / `bool(tensor)`）在NPU 上**每处都是一次硬同步**：CPU 被设备等住的时间**不计入 NPU 利用率，但全额计入端到端**。P112 的实例是稀疏 gather 缓存判据用 `torch.equal(cache_src, ht)` 判「是不是同一个 h」→ **每步一次硬同步**，等于把 P34 用 `nll_sync_every` 消除掉的同步又还回去。⚠ **它不报错、不影响数值、不触发任何告警** —— 日志里所有字段都正常，只是慢 | ① 修法：改**主机侧标记**（每次上传新 h 递增 `_ht_epoch`，`_sp_gather` 只比整数，**零设备交互**）；② **发现手段**：`tools/diag_readout_npu.py`（P113 新增）的**同步点静态扫描**——人工找这类点不可靠，改一处漏一处；③ 判据要覆盖**所有上传路径**（`forward()` 走 `_to_dev`、`forward_dev()` 走 `_staged_to_dev`，漏一处会静默复用旧 gather 结果，实测 `max\|Δ\|=1.45`） | **会**。⚠ **静态扫描只能证明「没有显式同步原语」，不能证明设备端没有隐式同步** —— 后者需 CANN 级profiling（msprof）。P113 扫描确认读出热路径**已无同步点**，这是「26.1× 差距归因于算子效率而非同步」的前提 |

### 5.1 两条补充纪律

**标识纪律**：用类属性 `AccelReadout._is_accel = True` 标识加速后端，
**不要用 `hasattr(readout, 'forward_dev')` 猜** —— numba `Readout` 也有同名旧接口，
曾导致误走设备路径崩溃。

**可选依赖纪律**：惰性 `import` + `getattr` 防御。未定义的补丁函数曾让修改
静默 no-op。`phdnet/model.py` **从不 import torch**（torch 是可选依赖），
所以 bf16 分支必须惰性 `import torch as _t` —— 直接写 `torch.bfloat16` 会让
纯 numpy 训练路径 `NameError`（BUGS A7）。

---

## 六、迁移路径与未迁移机制

### 6.1 已完成的迁移

| 节点 | 内容 |
|---|---|
| P10 / P14 | `backends/` 目录化与统一探针 / 多卡读出列并行 |
| P19 / P23 | **只迁移读出**（`AccelReadout`），确立「回落 + 记原因」/「完整调用面对齐」 |
| P28 | NPU 带宽根因：`torch.outer` 物化同尺寸临时张量 → `addmm_` rank-1 AXPY；设备侧 (h, y) 缓存；硬同步 3 → 1 |
| P29 / P30 | 基准工具三 bug 修正（**修复前的加速器数字一律不可信**）/ **旧 torch 栈整体删除**（约 1,400 行） |
| P36–P52 | 训练步设备直通（`forward_dev` + `target_idx`）、融合步核、`_is_accel` 显式标识 |
| P84 / P92 | fp8 前进副本 → 因昇腾不支持，**fp8/fp4 从加速后端整体禁用** |
| P104 / P105 | int8 存储 + fp16 计算码本；fp8 正名为 int8 别名 |
| P110 / P111 | 精度实测推翻半 ULP 估算 → 默认回 fp32 / 稀疏读出上加速后端（gather-GEMV） |

### 6.2 torch 全栈轨未迁移的机制（源码依据）

P30 删除前的 `TorchPHDNet.__init__` 有一段显式拒绝表。**这些机制在 torch 栈
构造时逐个检查 config，命中任一即 `raise NotImplementedError`**
（原文：`torch LM 栈暂不支持机制：{_unsupported}（请关闭或使用 numpy 后端）`）：

**第一组（机制本身未实现）**：`adaptive_lr`（自适应 LR）、`stdp_homeostasis`
（STDP 稳态）、`metaplasticity`（元可塑性）、`ei_synapses`（E-I 突触）、
`big_ltm`（大容量事件驱动稀疏表）、`retrieval_topk`、`readout_hidden`（两级读出）、
`readout_conn_k`（稀疏读出）、`lognormal_init`、`wm_summary_every`（摘要槽）、
`readout_recurrence`、`segment_check`、`multi_modulation`；
另有 `plateau_sleep`（T4.4）在 `train_stream` 里单独 `NotImplementedError`。

**第二组（7 项，numpy 端真实生效但 `step()` 从未实现 → 此前被静默忽略）**：
`dual_trace`、`learnable_encoder`、`critical_period`、`task_modulation`、
`auto_development`、`pc_predictive_target`、`neuron_target_rate`。
这一组最危险 —— 构造成功、训练照跑，但**与 numpy 逐张量不同**。
按模块 docstring「显式拒绝，不做静默近似」的承诺才改为构造时拒绝。

**第三组（结构约束）**：`sparse_conn=False`（要求稠密主干）显式拒绝 ——
torch 栈恒用稀疏 CSR 语义（`TorchSparsePC` 只实现 CSR 边表示）；
`readout_dtype ∈ {fp8, fp4}` 拒绝。

⚠ 这份清单共 **20 项 config 机制 + 1 项 `plateau_sleep`**，源码注释里
「缺 7 项机制」指的是**第二组**那 7 项静默忽略的，不是总数。

### 6.3 为什么不整体迁移

| 理由 | 说明 |
|---|---|
| **numba 事件驱动稀疏 + 在线 CSR 生长是 CPU 专属结构** | `numba njit` 只能编译到 CPU 机器码 —— **物理限制**，不是探测失败。M2 主干的结构稀疏 + 在线 CSR 生长、M3 逐突触时序更新、M4b 的 dict/CSR 邻接表生长，都不是「把数组搬到设备上」能解决的 |
| **torch 栈缺机制 → 整体迁移会丢机制** | 迁移 = 用一个缺 20 项机制的实现替换机制完整的实现，与项目铁律冲突。P30 的教训是：**不要维护一套永远赶不上主循环机制数的平行实现** |
| **权重不通用** | 不同实现、不同初始化顺序 → 权重与生产 npz 检查点**不通用**，切换等于放弃全部已有训练成果 |
| **只有读出值得迁** | 读出是唯一的大矩阵部件（V×H），且**逐 token 只有一个 h 向量**参与计算 → 每步只需传 12 KB 上行 + 少量下行，通信可忽略 |
| **变长字符串留 CPU** | 分词 / 词表构建是变长字符串处理，NPU 不擅长（要整段上传 + 变长输出） |

→ **混合执行架构**：数据侧（分词 / 编码）+ 学习侧（M1–M5）在 CPU，
只有 M6 读出在设备上。这是**有意的设计边界**，不是未完成项。

### 6.4 尚未实施的项

| 项 | 说明 |
|---|---|
| **M6 幂律稀疏化** | 架构级欠账（脑同构维度），既省访存又补同构，方案已设计未实施 |
| 设备端稀疏张量 | 100B 容量栈上设备的前置条件，当前缺 |
| 段级流水 | M1/M2 与 M3/M4/M5 重叠，收益上限 = max(CPU 侧, 设备侧) |
| `accumulate` 参数 | `AccelReadout.learn_softmax` 收了 minibatch 梯度累积参数但**未使用**（静默忽略）。当前默认 1 → 无行为差异，**启用前必须实现或显式拒绝** |
| 在线 CSR 镜像 | M4b 多核化的正确路径（零 gather），至今未做 |
| 设备启动 `507033` | NPU 设备启动失败（最常见是上次训练进程未退出仍占着设备）。**回落机制正常工作**（不崩，打印 fallback reason）→ **未修**，取决于服务器环境 |

---

## 七、诊断入口

### 7.1 工具

| 工具 | 回答的问题 | 备注 |
|---|---|---|
| `tools/backend_probe.py` | 各平台 ok / count / name | 探测层 |
| `tools/accel_doctor.py` | 设备**真的能算**吗 | ① 环境矩阵（torch / torch_npu / **CANN 版本** + 配对警告）② `resolve_devices` 结果 ③ **试分配真实规模张量 + 前向 matvec**（默认 73,958×3,072 fp32 ≈ 908 MB）④ 性能对照（设备 torch / CPU torch / numba 基线）⑤ 结论。`--no-bench` 只诊断 |
| `tools/probe_fp8.py` | fp8 卡在**芯片**还是**框架** | 三档分开测：`create` / `roundtrip` / **`matmul`**。`--json` 机器可读 |
| `tools/probe_npu_quant.py` | 昇腾**量化矩阵乘 API** 能不能用 | 定向探测 `npu_weight_quant_batchmatmul` / `npu_quant_matmul` / `npu_dynamic_quant` 等候选，各用 fp32/fp16/bf16 真调一次。同时记录 `torch.__config__`（排查构建不匹配） |
| `tools/probe_readout_precision.py` | 低精度**是否丢弃学习更新** | 纯测量，不改生产代码（§四） |
| **`tools/diag_readout_npu.py`**<br>（P113 新增） | **读出热路径还有没有设备同步点？时间花在带宽还是算子？** | 两个子问题：① **同步点静态扫描**（扫 `torch.equal` / `.item()` / `.cpu()` / `bool(tensor)` 等显式同步原语）；② **字节流量拆解**（分idx / val / 临时张量 `g` / `dp` 四项）。产出与性能文档 §3.5 同口径的字节数与「实测 ÷ 估算下界」倍数。⚠ **带宽下界是估算值不是实测**（依赖假设带宽），工具 docstring 已显式警告——**把下界当实测是本项目反复踩过的坑**（`BUGS.md` A8/B12） |
| `tools/bench_accel.py` | 读出热路径基准 + **等效带宽 GB/s** | 默认 V=73,958 / H=3,072（1B 真实词表规模），fp32/fp16/bf16 **三档都跑**。读出是 GEMV，**受带宽限制而非算力** → 等效带宽（3×|W| / 耗时）是**唯一可跨平台比较的指标** |
| `multi_device.capability_report()` | 后端 × 设备能力矩阵 + 本机探测 + 行动建议 | `verbose=True` 打印 |

### 7.2 ⚠ 判断加速是否生效：看日志，不看「检测到设备」

`train/train.py` 启动时打印：

- **`[capability]`** 行：检测到的加速器、`resolve_accel_device('auto')` 的解析结果、
  诊断命令建议；
- **`[readout] compute precision = ...`** 行：精度 + 是否「精确 `p − t` 规则」；
- **`[readout] backend=<...>`** 行：**实际选中了哪个后端**。
  `accel:<设备>` 时附 `(device ...)`；回落时附 `fallback reason: <原因>`。

**唯一可信的判据是 `backend=` 这一行。** 「检测到设备」只说明
`probe_devices()` 返回 ok，不代表能算（踩坑 4 / 13）。

训练循环中（P20，`train.py:874-878`）：

```
token 1,234  sliding PPL 394.4687  62.31 ms/tok  elapsed 12.3 min
  | readout 7.830 ms/tok (accel:npu@npu:0, 64.7% of total)
```

→ **加速生效的判据 = 这行的 `readout X ms/tok（后端@设备，占 Y% of total）`**。
⚠ **占比的判读方向在 P113 之后变了**：读出占 **64.7%**（NPU 侧是大头），
所以现在「占比 < 50%」这个旧判据**不再适用**——
当前状态下要判断瓶颈在哪一侧，应该看**哪一侧的墙钟更大**：
CPU 侧 4.26 ms vs NPU 侧 7.83 ms（性能文档 §4.1）→ **瓶颈在 NPU 侧的读出算子效率**。
（P111 段读出是 43.5%、CPU 侧占一半，那时「< 50% ⇒ 瓶颈在 CPU」是对的；
**那是历史判据，口径已变。**）

⚠ 注意与 `capability_report()` 打印的 `[能力矩阵]` 行区分：后者是静态能力表，
前者是本次运行的实测。**两者不一致时以训练日志为准。**

### 7.3 验证入口

| 脚本 | 覆盖 |
|---|---|
| `verify_accel_readout.py` | `AccelReadout` 对拍 + **接口完整性自动扫描**（A4：正则扫全仓库 `readout.X` 访问面）+ 拒绝路径断言（A3） |
| `verify_accel_readout_p55.py` | **2×2 矩阵**（target 数组 / `target_idx` × eager / compiled），判据容差 1e-5 |
| `verify_accel_sparse.py` | 稀疏读出 27 例：数值等价 / 调用面 / 拒绝路径 / 零回归 |
| `verify_multi_device.py` | 设备解析 / 分片均衡与余数 / 单设备 ≡ `AccelReadout` / 分片 ≡ 单设备（**逐位**）/ 计划报告 / host 线程收敛 |
| `bench_accel_path.py` | 复现读出路径的设备流量与墙钟对照 |
| **`verify_m2_kernels.py`**（P113 新增，11 例） | M2 推理核三态（fused/serial/plain）对拍。它属 CPU 侧而非设备侧，列在这里是因为**默认值切换的零回归证据**（A1/A2/A3 + 行级 prange 不改变求和顺序） |
| `python tests/run_tests.py fast` | 零回归门槛 **9/9** |

`verify_accel_sparse.py` 的价值在于它是**实现的准入门槛，不是事后补的测试** ——
它在开发中抓出两个真 bug：① 前向写成 `gather(...).sum(1)`，**把权重整个丢掉**
（形状 dtype 全对，只有数值错）；② `_csr` 每次返回**新的 D2H 副本**，
ckpt 的 `val[:] = ...` 写进无人引用的数组 → **检查点「恢复成功」但权重是旧值**。
两者都靠**数值对拍**抓出，形状断言抓不到。

⚠ **fast 门禁只跑 `tests/` 下的检查，从不 import `train/train.py`** ——
改训练入口后必须单独 `py_compile` + 跑一次 `--help`。

---

## 八、源码与文档不一致（本次重写发现）

以下都是**源码为准**、文档需修的项。前 3 项是本报告写作时实测发现的**活跃缺陷**。

| # | 不一致 | 证据 | 影响 |
|---|---|---|---|
| 1 | **能力表与 `_DT` 错位：`int16` / `int32` 被能力表放行但构造期 `ValueError`** | `_unsupported_reason()` 对 `int16`/`int32` 返回 `None`（放行），但 `_DT` 只有 `{fp32, fp16, bf16, int8}` → 构造抛 `ValueError: 不支持 dtype='int16'`。实测：`pick_readout_backend` 静默回落 `numba-cpu`，**`fallback_reason=None`（原因丢失！）** | 🔴 **活跃缺陷**。CLI `--readout-dtype` 的 `choices` 含 `int16`/`int32`，用户可选 → 选了就走 numba，且**日志不打印回落原因**（因为不是被 `_unsupported_reason` 拒的，是构造异常）。这正好是踩坑 15 的变种 |
| 2 | `train/train.py:389`、`train/infer.py:222,248` 仍引用**已删除**的 `tools/train_torch_lm.py` | commit `4120a5b`（P30）删除了该文件；三个引用点仍在打印「torch 栈 tools/train_torch_lm.py --device auto」 | 🟠 用户按提示去跑会 `file not found` |
| 3 | `phdnet/device.py::_verify` 永远返回 `False` | 函数体第一行就 `raise NotImplementedError`（`selftest_torch` 已随 P30 删除），`return bool(selftest_torch(...))` 是**不可达代码** | 🟡 `BackendInfo.verified` 恒 False。语义上是「等价自检已随旧栈移除」，但写法是死代码 + 会误导 |
| 4 | CLI `--readout-dtype` **实际默认 `fp32`**，但部分文档写 `fp16` | `train/train.py:206` `default="fp32"`；help 文本写「默认 fp32（P110 实测）」。旧版文档与 `文档写作规范.md` §2.4 仍写 fp16 | 🟡 P110 已改默认，文档未同步 |
| 5 | ~~CLI `--m2-kernel` 实际默认 `serial`，但 help 文本写「默认 plain」~~ | ✅ **P113 已解决**：`train/train.py:250` `default="plain"`，help 同步改写为实测数字（M2 6.92 → 1.28 ms/tok、5.41×、端到端 17.63 → 12.09、四个采样点 PPL 逐位相同）。**`serial` 现为历史默认** | 🟢 已修（本文是修复范例：help 不必删数字，改成**实测数字**更有用） |
| 6 | CLI `--encoder-dtype` **实际默认 `fp32`**，但 help 文本写「**默认 fp64**」 | `train/train.py:256` `default="fp32"`，help 写「**默认 fp64**：昇腾 aarch64 上 fp32 sgemv 实测慢约 70 倍」 | 🟡 help 自相矛盾（P107 已把默认改回 fp32） |
| 7 | 稀疏读出**已在加速后端实现**，但 `phdnet/backends/README.md` §四回落表仍列 `readout_conn_k > 0 → 回落` | `_unsupported_reason()` 对 `readout_conn_k=128` 返回 `None`（实测）；README 表格仍写「加速后端未实现」 | 🟡 子包 README 滞后于 P111 |
| 8 | `phdnet/backends/README.md` 标题写「精度：读出默认 int8」，正文表格写「默认 `--readout-dtype fp16`」 | 实际默认 `fp32` | 🟡 同一文件内自相矛盾 |
| 9 | `phdnet/backends/README.md` 称 verifier 有 `verify_accel_sparse` 外的引用、以及 `bench_accel.py` 三档 | `ls tests/verifiers/` = **22 个 `.py`**（21 个 `verify_*` + 1 个 `bench_accel_path`；P113 新增 `verify_m2_kernels.py`）。`文档写作规范.md` §2.7 已同步为 22 | 🟢 计数已同步（`verify_accel_sparse` / `verify_readout_intdtypes` / `verify_readout_sparse_gate` / `verify_ltm_learn_batch` / `verify_m2_kernels` 是后加的） |
| 10 | `phdnet/backends/torch_backend.py` 模块 docstring 首行仍写「STDP 关联核的 torch 实现」，但文件里**只有探针 + 基准** | `TorchSTDPCore` 已随 P30 删除；docstring 未同步 | 🟢 首行描述误导 |

---

## 九、引用纪律

本文出现的所有性能数字（ms/tok、GB/s、带宽、占比）一律**链接到
`PHD-Net_性能评估与迭代方案.md`**，不复制具体数值到本文表格里 ——
唯一例外是 §四的**精度保留率**（fp32 99.95% / fp16 26.67% / bf16 5.79%），
因为它是**机制性判据**（数值精度性质，与平台和档位无关），
不是性能数字，且由 `tools/probe_readout_precision.py` 一条命令可复现。

引用别处数字时必须带**平台 + 档位 + 版本**三件套，例如
「昇腾 191 核 + NPU，1B 档」。**不带平台的加速倍数在本项目没有意义** ——
x86 与昇腾常常方向相反（§二 2.2）。
