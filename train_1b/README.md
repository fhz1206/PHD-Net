# train_1b —— PHD-Net 1B 参数模型训练

PHD-Net 的 1B 档训练子项目：在**事件驱动稀疏类脑语义**下，训练总突触参数
容量 ≥ 1×10^9 的词级语言模型。不依赖 GPU 也能在小预算内真实训练与续训
（每步成本只正比于活跃神经元数，与 1B 总容量无关——这是架构使然，见下）。

## 容量口径（1B 在哪里）

| 组成 | 机制 | 规模（`--preset 1b` 默认） |
|---|---|---|
| **大空间事件驱动表（1B 主体）** | M4b 容量栈 `big_ltm`：结构可塑性，共激活神经元间生长新突触，每神经元 ≤60 条出边为硬容量约束 | 2^24 × 60 = **1,006,632,960** |
| 主干稀疏连接 | M2 预测编码栈，CSR 结构性稀疏（12.5% 连接率） | 4 × 1024×128 = 524,288 |
| STDP 侧向核 | M3 时序关联，每神经元 16 出边 | 1,024×16 = 16,384 |
| 编码器（稠密） | M1 稀疏编码，组合输入 2048→1024 | ≈ 2,098,176 |
| 读出头 | M6 softmax 感知器，h=3×1024 三拼 | 词表 × 3,072 |
| **总突触参数容量** | | **≈ 1.01–1.07×10^9 ≥ 1B** ✓ |

**诚实边界**：大空间表的 1B 是**容量**上限（`count_params` 计已生长突触），
训练初期利用率低、随经验增长；启动与日志均打印实时利用率，不做任何虚标。
"每步成本与总容量无关"成立的前提是 SDR 稀疏编码（6.25% 激活率）+ 事件驱动
局部计算——密集 1B 模型（权重+梯度+优化器 ≈16 GB、每步 ≥10^9 FLOPs）在
无独显机器上物理不可行，本档通过稀疏结构绕开，而非否认该物理约束。

## 文件

| 文件 | 职责 |
|---|---|
| `train.py` | 主入口：CLI、容量验算打印、训练循环、日志、检查点 |
| `config_1b.py` | 档位预设（smoke / 1b / 1b_max）、构建配置、容量验算 |
| `ckpt_1b.py` | 完整检查点：大空间表（CSR 快照 + 迹/时间戳）+ 词表 + 权重 |

## 用法

```bash
# 冒烟：分钟级验证全管线（容量 <1B，仅功能验证）
python train_1b/train.py --preset smoke --data sft --tokens 2000

# 标准 1B 档（本机 CPU ≈0.1–0.5 s/token，适合小预算/验证）
python train_1b/train.py --preset 1b --data sft --tokens 100000

# 生产长跑（大内存机器；1e9 token 级；断点续训）
python train_1b/train.py --preset 1b --data pretrain_zh --resume

# 查看大空间表实时利用率
python train_1b/train.py --preset 1b --data sft --report
```

产物位置（fhz 2026-09-25 指令：**模型统一存 `models/`**）：

- `models/phdnet1b_{preset}_{data}.npz` —— 滚动检查点（`--ckpt-every` 触发 + 收尾）
- `models/phdnet1b_{preset}_{data}_final.npz` —— 收尾另存的最终模型
- `outputs/train_logs/train_1b_*.log` —— 训练日志（含 `[METRIC]` 尾行）

## 检查点内容（相对生产版的增强）

`ckpt_1b.py` 保存/恢复**完整可续训状态**，生产版 `tools/train_production.py`
不包含前三项：

1. 大空间表邻接结构（`compact_csr` 快照，dict 版与在线 CSR 版均支持）
2. 突触迹与时间戳（t_pre/t_post/stamp_pre/stamp_post）+ 步数 + 入度
3. 词表与分词器（seg.vocab / max_len / tokens；SDR 哈希确定性重建）
4. 主干 CSR 四权重、编码器、STDP、WM、读出（稠密 W 或稀疏 CSR 三元组）

续训要求 `--data` 与 `--max-chars` 与原训练一致（词表大小 fail-fast 校验）。

## 硬件需求（`--preset 1b`，fp64）

| 项 | 需求 |
|---|---|
| 内存 | 静态 ≈0.5–0.6 GB（读出占大头）+ 大空间表已生长突触（dict 版 ≈100 B/条；`--csr-online` 切在线 CSR ≈10 B/条，逐位等价已由 `tools/verify_csr_equiv.py` 验证） |
| 磁盘 | 检查点 ≈0.3–0.6 GB/份（npz 未压缩，速度优先） |
| 吞吐 | 本机纯 CPU 实测见日志 `ms/token`；10^9 token 生产训练需 GPU/集群（参考 `tools/estimate_scale.py` 外推：256M 档本机 157 ms/token ⇒ 10^9 token ≈ 5 年，1B 档生产训练必须换硬件） |

## 与既有 1B 工作的关系

- `tools/train_1b.py` —— 1B 突触容量的**演示核**（ABC 序列，无词表/语料）；
  本子项目是其**全管线生产化**（词级 LM + parquet 语料 + 检查点/续训/日志）。
- `tools/bench_1b_migrate.py` —— dict 版 vs 在线 CSR 版的性能对拍（容量口径同源）。
- `--preset 1b_max`：主干 4096（连接率不变）+ 读出稀疏化建议
  `--readout-conn-k 512`，供大内存机器放大固定突触部分。
