# phdnet.backends —— 硬件后端子包（P10，2026-09-27 目录化）

CUDA / ROCm / CANN·昇腾 NPU / CPU 统一后端层。算子代码设备无关（仅
device/dtype 不同），跨设备判据为**容差一致**（不宣称逐位——归约顺序不同）。

## 文件

| 文件 | 内容 |
|---|---|
| `torch_backend.py` | 基础层：设备探针、STDP 核、读出热路径、自检、基准 |
| `torch_lm.py` | 词级 LM 全栈（M1–M6）torch 化：`TorchWordLM` / `TorchPHDNet` |
| `__init__.py` | 公共 API re-export |

旧路径 `phdnet.torch_backend` 保留兼容 shim（`from .backends.torch_backend import *`），
7 处既有引用零改动。

## torch_backend.py 公共 API

- `probe_devices() -> dict`：统一设备探针（npu / rocm / cuda / cpu），诚实降级 + 告警；
- `TorchSTDPCore`：STDP 时序关联核（numpy↔torch 每步小向量互转）；
- `TorchReadout`：读出热路径（fp32 / fp16 / bf16；fp8 需 CUDA≥8.9 真机）；
- `selftest_torch(device, dtype)`：等价性自检（容差判据，对照 `_numpy_reference`）；
- `bench_readout(device, V, H, ...)`：读出前向/更新基准。

## torch_lm.py —— 词级 LM 全栈（M1–M6 同一份算子覆盖多硬件）

### 设计要点（诚实声明）

1. **设备无关**：`resolve_device()` 解析 auto/cpu/cuda/rocm/npu；显式请求
   不可用设备时**诚实报错**（`allow_fallback=True` 才告警回退 CPU）。
2. **同 seed 等价初始化**：按 `PHDNet.__init__` 同一顺序消费同一
   `np.random.default_rng(cfg.seed)`；M1/M2 直接实例化 numpy 参考实现
   （`SparseEncoder` / `SparsePCStack`）抽取初始化数组后搬运设备——
   第 0 步权重与 numpy 版逐位一致，等价性对比从零点起步。
3. **机制 ↔ 神经认知对应**（铁律②；无自注意力 / 无位置编码 / 无堆叠层）：
   M1 V1 稀疏编码（k-WTA）、M2 皮层预测编码（CSR matvec / 误差 ΔW / Oja）、
   M3 STDP（`_ProdSTDPCore` 复刻 numba 热路径语义，见类内诚实标注）、
   M4 门控 WM + 快慢 LTM、M5 调制器（Welford z，设备无关直接复用）、
   M6 softmax 感知器读出（fp32 主回路）。
4. **数据侧留 CPU**：分词器（确定性整数哈希 SDR）留 CPU numpy，每步仅传
   n_input 维向量。
5. **数值协议**：网络状态 fp32（`cfg.torch_dtype`）；softmax/NLL fp32 主回路
   （numpy 版为 fp64 主回路——跨实现数值差异主来源，容差判据见下）。

### 已知限制（显式 NotImplementedError，不做静默近似）

- `readout_dtype` 仅 fp32/fp16/bf16（fp8 需 CUDA≥8.9；fp4 无 torch 原生类型且已禁用）；
- 未迁移机制：adaptive_lr / stdp_homeostasis / metaplasticity / ei_synapses /
  big_ltm / retrieval_topk / readout_hidden / readout_conn_k / lognormal_init /
  wm_summary_every / readout_recurrence / segment_check / plateau_sleep /
  multi_modulation（构造时逐项检查并拒绝）；
- M3 复用 TorchSTDPCore 的 numpy↔torch 每步互转：CPU 上开销可忽略，
  CUDA 真机上是已知优化点。

## 验证与使用

```bash
# 等价性验证（torch vs numpy 主实现，同 seed、冻结语料；逐设备对比）
python tools/verify_torch_lm.py --device cpu          # 或 --device all / cuda,npu

# 独立训练入口（数据侧 CPU 多进程留 numpy，网络计算在指定设备）
python tools/train_torch_lm.py --device cpu --preset smoke --data eval --tokens 2000
python tools/train_torch_lm.py --device auto --preset base --data eval \
    --tokens 50000 --ckpt outputs/torch_lm_base.npz --resume
```

验收结果（2026-09-28，CPU）：

- `verify_torch_lm.py`：PPL 相对差 **0.00003%**（容差 1%），权重轨迹
  readout / stdp / pc Pearson 相关全部 **1.000000**（容差 0.99）→ PASS；
- fast 11/11 零回归 PASS；
- 冒烟训练 2,000 token 跑通（15 ms/token，OOV 0）；smoke preset 的 PPL
  曲线中段上冲经 numpy 同口径对拍确认为小栈固有动态（两版曲线四位小数一致），
  非 torch 实现差异。
