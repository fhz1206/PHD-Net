# PHD-Net 现状速查（一页式，2026-11-04 P214）

> 本文是**当前事实基线**：数字、命令、开关默认值、已知问题、文档索引。
> 开关默认值全部从源码核实（`phdnet/config.py` / `train/train.py` / `train/config_1b.py`）；
> 与代码冲突时以代码为准。历史叙事与已推翻结论见 `BUGS.md` 与 git 历史，本文不搬运。

---

## 1. 当前测试 / 门禁数字

| 项 | 值 | 来源 |
|---|---|---|
| `python tests/run_tests.py fast` | **9/9**（<1 分钟，零回归门槛） | 持续门禁 |
| 专项门禁 | **54 个**（`ls tests/gates/gate_*.py` 现查，勿写死）；**55/55 全绿** | `outputs/experiments/gates_*.json` |
| **统一入口** | `python tools/run_gates.py [--jobs 4]` —— 自动实测选能 `import numpy` 的解释器 + 性能敏感门禁串行 + 输出 JSON | `tools/run_gates.py` |
| 关键门禁 | `gate_forward_only` 30 项 / `gate_cpu_opt_p211` 19 项 / `gate_fp8_int8_convert` 17 项 / **`gate_ltm_hash_lookup` 13 项**（P214 哈希槽位查找逐位契约） | `tests/gates/` |
| CI | 两套 workflow（`.gitcode/` + `.github/`），无 Jenkinsfile；远端状态未核验，**不宣称全绿** | 源码 |

⚠ **本机解释器陷阱**：`python` 指向 WorkBuddy 内置 3.13（**无 numpy**），
`py -c` 与 `py <脚本>` 按不同规则选解释器 → 直接 `py tests/gates/xxx.py`
可能报 `No module named 'numpy'`。**一律用 `tools/run_gates.py`**。

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

## 3. 全部开关默认值表（源码核实，2026-11-04）

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
| `sdr_novelty` / `sdr_novelty_cap` / `novelty_gate_floor` | **True** / 65536 / 0.85 | 会话中学习四开关默认 True |
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
| `--readout-gather-impl` | **`auto`**（按平台择优） | P218：NPU→`index_select`（生产 A/B：readout(win) **3.14×**、端到端 **1.81×**、probe_PPL 9 点逐位相同）；CPU→`index`（本机实测 index_select 慢 11%）。显式 `index`/`take`/`index_select` 仍可覆盖。详见 `Yan架构启发与并行化评估.md` §4/§5.2 |
| `--encoder-dtype` | fp32 | x86 上比 fp64 快 1.80× |
| `--ckpt-dtype` | fp16 | 位模式存储；master 模式 ro_W 存 fp32 逐位 |
| `--m2-kernel` | **`auto`** | P212 按平台择优：x86→`fused`（实测 1.34–2.05×）、昇腾 aarch64→`plain`（P113 5.41×）；可显式 `serial`/`fused`/`plain` 覆盖 |
| `--no-forward-only` | 关（即前向only **开**） | P211：前向only 层目标学习（FP+FTP 内核，给 M2 主干补任务信号）。**P211 起默认开启**；`--forward-only-eta 0` = 退回旧行为（逐位等价，门禁 C3b）。smoke 档 PPL 13017.891 vs 13355.328（**改善 2.53%**，方向为正但仍**低于 <20% 判据**）→ 生产档判据仍需服务器 A/B。详见 `docs/PHD-Net_前向only训练路径.md` |
| `--csr-online` | **开** | 在线可写 CSR（5.74× 省）；`--no-csr-online` 回退 dict |
| `--micro-batch` | 1 | P194 Q1 |
| `--readout-master-weight` / `--readout-shadow-refresh` | off / 8 | P195 |
| `--readout-powlaw-alpha/kmin/kmax` | 0 / 1 / 0 | alpha>0 回落 numba（加速器对非均匀行宽 fail-fast，正确行为） |
| `--numba-threads` / `--nll-sync-every` / `--ckpt-every` | **16** / 8 / 50000 | 线程数的服务器 A/B **尚未做** → 用 `tools/bench/bench_server_threads.py --threads 8,16,32` |
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

---

## 7. 红线速查（禁令 / 契约，改动前必读）

| # | 红线 | 为什么 |
|---|---|---|
| 1 | **x86 的性能结论不构成昇腾证据**，连「无差别」也不构成 | 项目已有四次方向相反 |
| 2 | **判性能一律看 `segments(win)`**，不看 `segments(cum)` | cum 是含 JIT 冷编译的历史平均 |
| 3 | **A/B 判据只能用 `probe_PPL`**（`--probe-every 1000 --probe-tokens 2048`） | sliding PPL 信噪比仅 0.40，<20% 测不出 |
| 4 | **`eta_oja > 0` 会发散** | 3000 步 probe_PPL 9.1e11（=0 时 916） |
| 5 | **值不能降到 bf16**（非目标行更新保留率仅 5.79%） | 会退化成纯 Hebbian；只有索引能压 |
| 6 | `--readout-pipeline` 是 **fail-fast 死开关** | P192 定案：叠第二层异步 + 语义风险 |
| 7 | **不要做跨 token 流水** | `M6(t)→PC_learn(t)→M2(t+1)` 串行链不可断；前移换时序规则（铁律③）；`readout_recurrence` 会静默拿错 y |
| 8 | **`--beta p3` 未实现**（help 有、代码无） | 传了静默无效 |
| 9 | **三条线程治理时序约束**：BLAS cap 在 `import numpy` 前 / `numba.set_num_threads` 在任何核首次执行前 / `OMP_PROC_BIND` 在 numba 初始化前 | 违反后**无法事后补救** |
| 10 | `_decay_read_arr` **必须逐个调 `_decay_read`** | `gate_ltm_trace_semantics` 的突变测试注入点；内联会让负向对照变空操作 |
| 11 | `_learn_batch` 的 K/V **必须预留到 `m_out`** | 核内会生长到 m_out，收紧会广播失败 |
| 12 | **混合 dtype**（`fp32 W @ fp64 x`）会脱离 BLAS 走慢路径 | 且**没有任何断言检查** `A.dtype == b.dtype` |
| 13 | `learn_predictive` 保持**多核调用** | P59 实测融合版 0.91–1.36× 负收益 |
| 14 | 读出前向走 **BLAS**；fp8 转换默认走 **torch** | 实测 numba 慢 0.68× / torch 快 5.6× |
| 15 | **不要为提速改读出精度**（fp16/bf16 舍掉非目标行更新） | 保留率 fp32 99.95% / fp16 26.67% / bf16 5.79% / int8-fp8 0.04% |
| 16 | **优化 NPU 的目标是把 `aicore_time` 提上去 / 把 `wait` 降下来** | 设备占用率高 ≠ 算力高；占用下降但 ms/tok 下降是好事 |
| 17 | **门禁必须把业务失败传播为退出码**（只 `print("FAIL")` 不算） | `run_gates.py`/CI 按 returncode 判定；新增门禁后用 `python tools/audit_gates.py --strict` 初筛 + 强制失败动态复核 |

