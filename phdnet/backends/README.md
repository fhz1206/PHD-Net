# `phdnet.backends` —— 硬件后端子包

> **本目录只放硬件后端。** 子包用法与索引；全局结论见
> `docs/PHD-Net_硬件后端适配报告.md`（后端矩阵、昇腾踩坑、迁移路径）。
> 性能数字见 `docs/PHD-Net_性能评估与迭代方案.md`（唯一出处，本文不复制）。
> **数据截止**：2026-09-30（P84：读出默认 fp8 forward + fp16 更新）。

支持的平台：**numba CPU（生产主力）/ CANN·昇腾 NPU / CUDA / ROCm / DirectML**。

---

## 一、文件索引

| 文件 | 公共 API | 用途 |
|---|---|---|
| `accel_readout.py` | `AccelReadout`、`pick_readout_backend(cfg, n_h, n_out, rng)`、`resolve_accel_device(spec)` | **生产加速读出**。`pick_readout_backend` 是模型侧唯一入口：按 `cfg.accel_readout` 选路，不可用时回落 numba 并记原因 |
| `multi_device.py` | `resolve_devices`、`probe_multi`、`shard_ranges`、`plan_parallel`、`capability_report`、`configure_host_threads`、`MultiDeviceReadout` | 多卡（读出**列并行**）、分片计划、能力矩阵 |
| `torch_backend.py` | `probe_devices()`、`bench_readout(device, V, H, dtype, …)` | 设备探针（诚实降级 + 告警）；读出基准（**含设备同步与等效带宽 GB/s**，P29 修正版，默认 V=73,958 = 1B 真实词表） |
| `torch_lm.py` | `resolve_device(device, allow_fallback)` | 仅存设备解析。auto 择优顺序：**昇腾 → ROCm → CUDA → DirectML → CPU**。其余（整套 torch 版 PHD-Net）已随 P30 删除 |

`__init__.py` 导出：`AccelReadout`、`pick_readout_backend`、`resolve_accel_device`、
`bench_readout`、`probe_devices`、`resolve_device`。

---

## 二、怎么用

### 训练 / 推理入口

```bash
python train_1b/train.py --accel auto     # auto=有加速器就用，否则回落 numba CPU
python train_1b/infer.py  --accel auto
python train_1b/train.py --accel npu       # 显式指定设备（不可用则回落并记原因）
python train_1b/train.py --devices auto    # 多卡：读出列并行（与 --accel 正交）
```

**确认加速是否真的生效**：看启动日志的 `[读出] 后端=accel:<设备>` / `numba-cpu`；
回落时同行打印 `fallback reason: <原因>`。**不要**用"探测到设备"判断——见 §四。

### 诊断与基准

```bash
python tools/backend_probe.py     # 探测层：各平台 ok / count / name
python tools/accel_doctor.py      # 试分配 + 一次前向 matvec，验证设备真的能算；环境矩阵
python tools/accel_doctor.py --V 20000 --H 3072    # 显存不足时缩小规模
python tools/bench_accel.py       # 读出基准，报等效带宽 GB/s（跨平台唯一可比指标）
```

---

## 三、关键口径

| 口径 | 内容 |
|---|---|
| **只迁移读出** | 加速的正确姿势是只迁移 M6 读出，**不维护平行实现**。P30 已删除整套 torch 版 PHD-Net（约 1,400 行）：权重不通用、缺 7 项机制、从未进生产 |
| **回落 + 记原因**（P19） | 无加速器 / torch 缺失 / 配置不兼容 / 构造异常 → 回落原 `Readout`（**默认路径逐位不变**），原因写在 `_accel_fallback_reason`，启动日志打印。**不静默** |
| **未实现配置 fail-fast**（P19） | `readout_hidden>0`（两级读出）、`readout_conn_k>0`（稀疏读出）等路径加速后端**未实现** → **显式回落并记原因**，绝不"能跑但语义不同" |
| **完整调用面对齐**（P23 教训） | `AccelReadout` 必须对齐 `Readout` 的全部访问面：`W` / `learn_softmax` / `learn` / `__call__` / `n_synapses` / `conn_k` / `hidden` / `stats()` / `W_cpu` / `load_W`。缺一项会在**生产保存检查点**时才崩 |
| **等价性判据** | 跨实现 / 跨设备只宣称**容差一致**（`atol/rtol`，归约顺序不同）；**同设备**分片路径可逐位。`bench_accel.py` 报**等效带宽 GB/s**（GEMV 的唯一跨平台可比指标） |
| **numba 只能编译到 CPU** | 物理限制。加速器机器上生产主循环仍跑 CPU，日志里的"numba nogil 线程 ×N"指 CPU 线程。上加速器的唯一路径 = `--accel auto` 让读出走 `AccelReadout` |

### 精度（现行默认 fp8，P84）

| 项 | 口径 |
|---|---|
| 默认 | `--readout-dtype fp8`：**forward 用 fp8_e4m3fn 副本**（1B 档读 320 → 80 MB，`--fp8-refresh` 默认 8 步重建）+ **更新用 fp16 主副本** |
| 动机 | **不是省访存**（更新侧 fp16 使总访存 +42%），而是**学习精度**：bf16 半 ULP ≈ 2e-4 ≫ 非目标行更新 \|dp\| ≈ 1e-6 → 更新被舍 → 退化为纯 Hebbian |
| 设备边界 | fp8 matmul **仅昇腾 / CUDA**；CPU torch **自动回落 fp16 主副本**（`tdtype` 同步回落，否则 `addmv` dtype 不匹配），只告警一次 |
| 检查点 | `to_numpy` 对 bf16/fp8 存**位模式**（uint16 / uint8），加载侧按 `ckpt_dtype` 解码闭环（numpy 无原生 bf16，不这样存会崩） |
| 其它 | M1 编码器默认 **fp64**（昇腾 aarch64 numpy GEMV 病态 → 平台自适应走自写 numba 核，x86 走 BLAS）；分词 / onehot 缓冲 fp16（值域 0/1，无损） |

各精度的 PPL / 带宽数字见《性能评估与迭代方案》。

---

## 四、接口纪律（踩过的坑，改代码前先读）

1. **用类属性 `AccelReadout._is_accel = True` 标识加速后端**——
   **不要用 `hasattr(readout, 'forward_dev')` 猜**：numba `Readout` 也有同名旧接口，
   曾导致误走设备路径崩溃（P40 教训）。
2. **`torch.compile` 是惰性编译**（P58）：`torch.compile()` 构造期不编译、**首次调用**才
   编译 → 首次失败**永久回落 eager + 告警**；`--torch-compile` **默认关**。
   `reduce-overhead`/cudagraphs 与本实现**本质冲突**（W 每步原地 mutate 输入）→ 只锁
   `torch_compile_mode="default"`，不修冲突。
3. **融合步核必须逐行镜像 eager 路径**（P55）：`target_idx` 路径的"就地改 p"曾只写在 eager
   分支，`torch.compile` 融合核漏改 → NPU 首跑即崩 `FakeTensor - None`。
4. **可选依赖边界要惰性 import + `getattr` 防御**（P82/P83b）：`_correct_pinned` 未定义曾
   让补丁静默 no-op；`to_numpy` 引用未 import 的 torch 曾直接崩。
5. **`learn_softmax` 的 `accumulate` 参数当前未被使用**：加速后端会静默忽略 minibatch 梯度
   累积。当前默认 1 → 无行为差异，**启用前必须实现或显式拒绝**。
6. **x86 的性能结论不构成证据**：M1 fp32、M2 融合核、`OMP_PLACES=cores` 三项在昇腾上分别
   慢 70× / 8–13× / 8×。跨平台改动必须在目标机器复测。

---

## 五、昇腾实测坑位速查（P55–P83）

详细根因与处置见 `docs/PHD-Net_硬件后端适配报告.md` §八。

| # | 现象 | 根因一句话 |
|---|---|---|
| P55 | NPU 首跑崩 `FakeTensor - None` | 融合核漏改 `target_idx` 分支 |
| P57/P63 | NPU / HBM / AI Core% 恒 `--` | `Telemetry()` 无参构造 + `npu-smi` 不在非交互 shell 的 PATH + 解析依赖该机没有的 Bus-Id 列（改走**按表头定位列**解析；**不能用 `torch.npu.utilization()`**，它同步设备流） |
| P58 | CPU 忙、NPU 空闲 | 每步 `.item()` 把设备延迟全额暴露给 CPU；**pageable H2D 阻塞** → `--nll-sync-every` 默认 8 + pinned 暂存 `non_blocking` |
| P71/P73 | CS/s 250 万–600 万；只用 1.1–3.2 核 | 线程无核心亲和性（→ `OMP_PROC_BIND=close`）；`*_NUM_THREADS` 从未限，OpenBLAS 拉 191 线程跑小 sgemv → **必须在 `import numpy` 之前**限 8 |
| P74 | `OMP_PLACES=cores` 反噬 | 191 核 place 表每次线程池同步都遍历 → M2 再慢 8× → **已回滚，只留 `PROC_BIND`** |
| P76/P77 | M1 慢 70×（两次误判） | aarch64 numpy GEMV 病态 + 混合 dtype 脱离 BLAS → 平台自适应 GEMV |
| P81/P84 | bf16 检查点保存崩 | numpy 无原生 bf16 → 存位模式，加载侧解码闭环 |

---

## 六、验证入口

| 脚本 | 覆盖 |
|---|---|
| `tests/verifiers/verify_accel_readout.py` | `AccelReadout` 对拍 + **A4 接口完整性扫描**（正则扫全仓库 `readout.X` 访问面，断言加速后端具备） |
| `tests/verifiers/verify_accel_readout_p55.py` | P55 的 2×2 矩阵（eager/compiled × target 数组/`target_idx`） |
| `tests/verifiers/verify_multi_device.py` | 设备解析 / 分片均衡与余数 / 单设备 ≡ `AccelReadout` / 分片 ≡ 单设备（逐位）/ 计划报告 / host 线程收敛 |
| `tests/verifiers/verify_rl.py` | REINFORCE 训练回路（零新增算子，与后端正交） |
| `tests/verifiers/bench_accel_path.py` | 复现读出路径流量与墙钟对照 |
| `python tests/run_tests.py fast` | 零回归门槛，**9/9** |

**周边配套**（子包外，但属同一批变更）：`phdnet/telemetry.py::Telemetry`（CPU/RAM/NPU 利用率/
HBM/CS 率，非阻塞；IPC 需 perf，Python 拿不到就诚实返回 `None`）；`tools/fetch_ms.py`
（三源采样器，`--plan web=3,code=2,math=1`，零原始落盘，断点续跑）；
`NUMBA_CACHE_DIR=outputs/numba_cache` 持久缓存（train/infer/train_rl 顶部设置）。
