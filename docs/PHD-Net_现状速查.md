# PHD-Net 现状速查（一页式，2026-10-31 P210）

> 本文是**当前事实基线**：数字、命令、开关默认值、已知问题、文档索引。
> 开关默认值全部从源码核实（`phdnet/config.py` / `train/train.py` / `train/config_1b.py`）；
> 与代码冲突时以代码为准。历史叙事与已推翻结论见 `BUGS.md` 与 git 历史，本文不搬运。

---

## 1. 当前测试 / 门禁数字

| 项 | 值 | 来源 |
|---|---|---|
| `python tests/run_tests.py fast` | **9/9**（<1 分钟，零回归门槛） | `[历史锚点]`，持续门禁 |
| `tests/gates/gate_*.py` 专项门禁 | **48 个**（`ls` 现查，勿写死）；P210 时点 **48/48 全绿** | `[本会话实测]` `outputs/test/reorg_gates/summary.json`；`gate_csr_vectorization_perf` 曾 26/27 属 golden 基准缺失，后续已绿 |
| P194–P199 新增门禁 | `gate_insession_learning` / `gate_readout_master_weight` / `gate_generate_truncation` / `gate_cc_traces` / `gate_cli_i18n`（30 项显式契约） | `[源码]` tests/gates/ |
| P210 新增门禁 | `gate_forward_only`（26 项：固定随机投影 / 层目标学习 / 接线与零回归） | `[源码]` tests/gates/gate_forward_only.py |
| P192 发布验证 | fast 9/9 + 40/41 verifier（唯一失败即 `gate_ltm_trace_semantics` (a) 组，**P193 已修 → 32/32**） | `[历史锚点]` `outputs/test/p192_publication/summary.json` |
| CI | 两套 workflow（`.gitcode/` + `.github/`），无 Jenkinsfile；远端状态未核验，**不宣称全绿** | `[源码]` |

## 2. 快速命令

```bash
# 冒烟（分钟级，只验管线；日志「⚠ < 1B」是预期）
python train/train.py --preset smoke --data sft --tokens 2000

# 标准 1B 档（远程 ModelScope 流式，零落盘）
python train/train.py --preset 1b --data pretrain --remote-data

# 推理 / 对话（P198：top-k 截断在 infer 侧默认 topk=8）
python train/infer.py --model outputs/models/phdnet1b_1b_pretrain_final.npz --chat

# 只看检查点的大空间表统计
python train/train.py --preset 1b --data pretrain --report

# 门禁（改动后必跑；fast 不 import train/*.py，改训练侧后另跑 py_compile + --help）
python tests/run_tests.py fast

# 生成质量评估（bigram 合法率 + distinct；P198）
py -3.14 tools/eval/eval_generation.py --ckpt <npz> --vocab <txt> --tau 0.8 --topk 40 --topp 0.9

# 词表预算诊断（训练前估 exposure = tokens/vocab；P198）
py -3.14 tools/diag_vocab_budget.py --vocab-sample-chars 4000000 --tokens 100000

# PPL 复测（只对比不更新 ANCHOR）
py -3.14 tools/eval/rebaseline.py
```

## 3. 全部开关默认值表（源码核实，2026-10-17）

### 3.1 库配置 `phdnet/config.py`（关键项）

| 项 | 默认 | 备注 |
|---|---|---|
| `k_sparse` | 16 | 1b 档 `max(8, w//8)`=128 |
| `eta_pc` | **0.02** | P194 起：发育窗口退火到 0（非恒 0） |
| `eta_oja` | **0.0** | P198：>0 生产规模正反馈发散（3000 步 probe_PPL 9.1e11）；>0 触发 RuntimeWarning |
| `pc_dev_steps` | 0（库）/ **20000（1b 档）** | 表征发育余弦窗口 |
| `eta_stdp` | 0.03（1b 档 0.02） | |
| `pc_w_max` / `w_max` | 2.0 / 1.0 | PC 主干 / STDP 核 |
| `m_lateral` / `lambda_trace` | 16 / 0.35 | |
| `n_wm_slots` / `gamma_wm` / `gate_thresh` | 4 / 0.85 / 0.35 | |
| `conn_k` | 0（自动 n_in//8 ≈12.5%） | M2 主干 CSR 唯一实现（稠密栈已删） |
| `readout_dtype` | **fp32**（P194 Q4，P110 反证） | fp8/bf16/fp16 可显式指定；fp4/int4 已删除（P191e） |
| `readout_conn_k` | 0（库）/ **128（生产 CLI）** | 128/3072=4.2% 连接率 |
| `readout_powlaw_alpha` | 0.0 | 幂律已接入但默认关（词频是代理值） |
| `readout_master_weight` | **False** | P195：fp32 master + 低精度影子；默认 off |
| `readout_shadow_refresh` | 8 | 影子重量化间隔（步） |
| `micro_batch` | 1（CLI `--micro-batch`） | 读出梯度 B 步累积；主干仍逐 token |
| `sdr_novelty` / `sdr_novelty_cap` / `novelty_gate_floor` | **True** / 65536 / 0.85 | P197 |
| `ltm_imprint_gate` | **0.3**（旧 0.8） | P197 |
| `task_modulation` / `error_gated_memory` | **True / True** | P197 误差广播 |
| `plateau_sleep` / `readout_replay` | **True / True** | P197（`plateau_patience=2`、`replay_buffer=256`、`replay_every=4`、`replay_eta=0.5`） |
| `big_ltm` / `big_ltm_N` / `big_ltm_m` / `big_ltm_k` | False / 2²⁴ / 60 / 4（1b 档 True/2²⁴/72/4） | |
| `csr_online`（库） | False | **CLI 默认开** |
| `homeostasis` | False（**1b 档 True**） | P194 防表征发散 |
| `mod_gain` / `multi_modulation` | 1.5 / False | 四通道可选 |
| `encoder_dtype` | fp64（库）/ **fp32（CLI）** | 迭代恒 fp32 |
| `nll_sync_every` | 8 | P62 |
| `step_profiling` | **True**（P198 起） | `--no-step-profiling` 关 |
| `torch_compile` | False | P58 指令；P195 master 模式下强制禁用 |
| `cython_kernels` | off（库）/ **auto（CLI）** | off/auto/force；fp32 级漂移，非逐位等价 |
| `cann_dispatch` | False | 只核对，不新增行为 |
| `readout_pipeline` | False | **已停用：传了直接报错退出** |
| `readout_gather_dtype` / `sparse_fwd_kernel` / `readout_gather_impl` | fp16 / einsum / index | P147/P131/P134 |
| `fp8_refresh` / `fp8_conv` | 8 / torch | fp8 位模式→int8 转换核 |
| `ltm_imprint_amortize` | 1 | P122 移除，>1 直接报错 |

### 3.2 训练 CLI `train/train.py`（与库不同的项 + 独有项）

| 开关 | 默认 | 备注 |
|---|---|---|
| `--preset` | 1b | smoke/1b/1b_max/30b |
| `--data` | sft | sft/pretrain/pretrain_zh/eval/**cc_traces**（P196） |
| `--lang` | en | P199：显式消息目录（`_CATALOG` 整句 + `_WORDMAP` 多字词扫描）；只管终端文案 |
| `--data-lang` | all | 会改变训练结果 |
| `--readout-dtype` | fp32 | choices: fp32/fp16/bf16/fp8 |
| `--readout-conn-k` | **128** | 库默认 0（稠密），两者不一致是事实 |
| `--encoder-dtype` | fp32 | x86 上比 fp64 快 1.80× |
| `--ckpt-dtype` | fp16 | 位模式存储；master 模式 ro_W 存 fp32 逐位 |
| `--m2-kernel` | plain | P113 服务器实测（M2_infer 5.41×）；fused/serial 留作对照 |
| `--forward-only` | **关** | P210：前向only 层目标学习（FP+FTP 内核，给 M2 主干补任务信号）。关闭时轨迹逐位不变；**收益未达显著**（smoke probe_PPL −0.32%/−2.52%，<20% 判据）→ 候选而非定案。详见 `docs/PHD-Net_前向only训练路径.md` |
| `--csr-online` | **开** | 在线可写 CSR（5.74× 省）；`--no-csr-online` 回退 dict |
| `--micro-batch` | 1 | P194 Q1 |
| `--readout-master-weight` / `--readout-shadow-refresh` | off / 8 | P195 |
| `--readout-powlaw-alpha/kmin/kmax` | 0 / 1 / 0 | alpha>0 回落 numba（加速器对非均匀行宽 fail-fast，正确行为） |
| `--numba-threads` / `--nll-sync-every` / `--ckpt-every` | 8 / 8 / 50000 | |
| `--step-profiling` | **开** | P198 起 |
| `--vocab-scan` / `--vocab-sample-chars` | head / 4,000,000 | |
| `--remote-fraction` | 0.3 | 前缀子集 |
| `--context-milestone` | 1,000,000 | |

## 4. 已知问题清单（诚实边界）

1. **1B 档 PPL 高的根因 = 词表 × 曝光**：51,962 词表下 exposure 不足（<30 接近随机、
   ≥100 才充足）。`tools/diag_vocab_budget.py` 训练前估算。[本会话实测：同 100K token，
   词表 20,913→PPL 3322；词表 3,392→PPL 1310]
2. **「正常对话」未达到**：bigram 合法率 24–35%（1B 档）；接近正常需 ≥10⁸ token 曝光——
   数据量问题，不是算法问题。
3. **`eta_oja>0` 发散**（P198）：3000 步 probe_PPL 9.1e11（=0 时 916）；默认 0 + 运行时警告。
4. **低精度默认 fp32**：P195 master 解除 P110 舍入丢失（fp8 无 master PPL 2.84 → 有 master
   2.15，fp32 基线 2.14），代价 +866.7 MiB（1b 档）+ ≤N 步 staleness。[本会话实测]
5. **历史 PPL 锚点不可比**：P192 η 缩放修正 + P193 LTD 迹补偿 = 学习规则语义变更；
   旧锚点（394.4687/359.2603）只存 git 历史。P198 的发育窗口验证（3403→389，
   3,392 词表 40K token）是另一组口径。
6. **幂律读出默认关**（词频代理 = 1/rank，词表字典序非频次序，方向可能反）；
   开启回落 numba。
7. **SM 会话学习已达**（印迹触发 0.67% → ~100%、cc_traces 3K token PPL 2474→1308、
   LTM 生长 +81%），**但生成质量需 ≥10⁸ token 曝光**。[本会话实测]
8. **fast 门禁不 import `train/train.py`** → 改训练侧后必须单独 `py_compile` + `--help`（A3）。
9. 本机 x86 无昇腾：Cython 核、einsum、take 等「昇腾收益」均**无实测**；x86 结论不构成
   昇腾证据（已多次方向反转）。
10. `capacity_report` 静态内存注释仍写「CSR = int32+fp64」，实际 val 已 fp32（P173）——
    内存数字为上界估计，以启动打印与 `[big-ltm]` 行为准。

## 5. 文档索引

| 文档 | 职责 |
|---|---|
| `README.md` | 项目一句话、怎么跑、任务导向文档地图 |
| `docs/PHD-Net_架构设计.md` | 最权威设计文档（M1–M6 + 学习能力层 P194–P198） |
| `docs/PHD-Net_现状速查.md` | 本文（一页式现状） |
| `docs/PHD-Net_性能评估与迭代方案.md` | **所有性能数字**（§6.1/6.2/6.4 精度与发散实测） |
| `docs/PHD-Net_竞争力与脑同构性评估.md` | 与 Transformer/大脑对比结论、能力边界 |
| `docs/PHD-Net_并行与加速架构分析.md` | 为什么吃不满多核/NPU（论证） |
| `docs/PHD-Net_硬件后端适配报告.md` | 后端矩阵与昇腾踩坑 |
| `docs/PHD-Net_扩展指南.md` | 怎么加东西、检查单 |
| `train/README.md` | 训练侧全部开关与日志字段 |
| `BUGS.md` | 缺陷台账（53 条 + P192/P193 补充） |

## 6. 工程 / 目录现状

- 训练入口 **`train/train.py`**（唯一生产入口）；推理 `train/infer.py`；容量验证
  `train/config_1b.py::capacity_report`；torch 训练轨已删（P30）。
- 产物 `outputs/{test,experiments,smoke,models}`；对话 `chat/`；验证 `tests/gates/`。
- 版本 v0.1.0-alpha（已并入 main）；历史 force push 过，**旧 hash 全失效，服务器 clone 须 reset --hard**。
- `requirements.txt`：numpy>=2.0 / numba>=0.60 / psutil / pyarrow（+Cython 构建）；torch 可选。
- 词表快照唯一权威 = JSON（words+seg_vocab+max_len+sha1）。
- `tools/bench/bench_cpu_local.py` 只测本机 CPU 相对变化，**不产出文档数字**。
- 已删除且勿再引用：rust 栈（`phdnet_rs/`，P188）、稠密 PC 栈（pc.py）、
  torch 栈全套、`sparse_conn` 开关（P193）、fp4/int4（P191e）、`readout_pipeline`（P192 停用）。
