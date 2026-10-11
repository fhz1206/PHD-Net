# PHD-Net 现状速查（工作记忆归档，2026-10-06）

> 本文由原 `.workbuddy/memory/`（MEMORY.md + 15 份日期日志，2026-09-16 ~ 2026-10-06）合并提炼而成，只留存**仍然成立的现状**；已被推翻的中间结论仅保留主题清单（见文末附录），过程叙事不再搬运。

## 1. 架构铁律（不可协商）

- 禁 self-attention / 位置编码 / 堆叠层；每项机制须有神经认知对应物（M1–M13，M7–M12 认知层 + M13 上下文漂移情景记忆）。
- 逐 token 流式语义：状态跨 token/shard/epoch 连续，**禁止批处理**（预训练不可批；仅 RL rollout 可批）。
- 新增行为一律 config 开关默认关闭、默认路径逐位不变；例外（fhz 拍板）：`sparse_conn=True`、`k_sparse=16`、读出 `conn_k=128`。
- 表征冻结 `eta_pc=0`（可塑性改进多轮全负）；PPL 增益只来自检索/记忆通路。
- 🔴 **分层定案（P170，曾）**：Python 顶层（编排/门禁/回落/日志）+ Rust 编译 cdylib 作无状态算子库。⚠ **`phdnet_rs/` 已整体删除（P188「删除 rust 版本，默认全部改回 python」）** —— 现为纯 Python + numba（+ 可选 Cython nogil 核，P192）。
- 稠密 PC 栈已删：`SparsePCStack` 唯一实现，`sparse_conn=False` config 期 fail-fast。

## 2. 工程 / 目录现状

- 训练入口 **`train/train.py`**（train_1b/ 已改名 train/）；推理 `train/infer.py`；容量验证 `tools/train_1b_capacity.py`；torch 轨已删（P30）。
- 产物 `outputs/{test,experiments,smoke,models}`；对话 `chat/`；验证 `tests/verifiers/`（**别写死数量**，`ls tests/verifiers/*.py` 现查；fast 门禁 9/9）。
- 版本 v0.1.0-alpha（已并入 main）；历史 force push 过，**旧 hash 全失效，服务器 clone 须 reset --hard**。
- CI **两套**：`.gitcode/workflows/ci.yml`（GitCode）+ `.github/workflows/ci.yml`（GitHub 镜像）。⚠ **仓库没有 Jenkinsfile** —— 旧版本这里写「三套并存」，是错的。GitHub 侧当前仍红（无日志权限，待 traceback）
- `requirements.txt`：numpy>=2.0 / numba>=0.60 / psutil / pyarrow 必装，torch 可选。
- 词表快照唯一权威 = **JSON**（words+seg_vocab+max_len+sha1）；`tok_vocab≠tok_tokens`，推理校验比 tok_tokens。
- `tools/bench_local.py` 只测本机 CPU 相对变化，**不产出文档数字**。

## 3. 现行默认值

| 项 | 默认 | 备注 |
|---|---|---|
| sparse_conn / k_sparse | True / 16 | fhz 拍板例外 |
| eta_pc / eta_oja | 0 / 0 | 表征冻结 |
| eta_readout | 0.15 | fhz 拍板 |
| 读出 conn_k | 128（连接率 4.2%）| 已上加速器 gather-GEMV（P111）|
| **读出精度（P189，fhz 2026-10-07）** | **fp8** | 见下表 |
| 检查点存储 | fp16 | P107b |
| fp8/fp4 显式请求 | fp8 已放行；fp4/int4 仍禁 | P189 覆盖 P167 的 PHD_FP8 开关 |
| int 族显式请求 | **禁用**（raise）| P163，P189 未动 |
| `--m2-kernel` | plain | 昇腾实测比 fused 快 5.41×；fused/serial 留作对照 |
| `--nll-sync-every` | 8 | |
| `--numba-threads` | 8 | |
| `--ckpt-every` | 50000 | |
| `--csr-online` | 开 | |
| torch_compile | 关 | P58 指令；P128 放开稀疏融合核，待服务器 A/B |
| `--lang` | en | 只管终端文案；语料过滤是 `--data-lang {all,zh,en}` |
| `--fp8-conv` | torch | P189：fp8→int8 转换核（**torch/nogil**）。🔴 **`rust` 选项已随 P188「删除 rust 版本」整体删除**，传旧值**静默按 torch 处理**（不报错，两条 Python 路径逐位一致）；~~x86 实测 rust LUT 核 8.44× 快（3.73 vs 31.5ms @1b 档）~~ **该数字已随实现删除而作废**，不可作为选型依据。⚠ torch/nogil 的 x86 口径昇腾待 A/B |

### P189 默认精度表（读出 M6，请求 fp8 时）

| 层 | 精度 | 说明 |
|---|---|---|
| W 存储（设备常驻）| **fp8 e4m3fn 位模式（uint8 承载）** | 1 B/元素；910B4 连 fp8 张量都建不出（ERR01007），uint8 存位模式 |
| 前向计算 | fp16（查表反量化 → fp16 GEMV/gather） | 910B 无 fp8 算子；fp16 matmul 真实可用 |
| 迭代（learn）| **fp16 域**：反量化 → addcmul_/addmm_ rank-1 → RNE cast 回 fp8 位模式 | fhz「迭代用 fp16」；P110 代价：非目标行更新大多被舍入丢 |
| int8 计算（前向 W8A8）| fp8 请求降级时启用 | 每步 CPU 转换 fp8→int8 码本（P154/P155；`--fp8-conv` 选核）|
| 降级链 | fp8 → fp16 → bf16 → fp32 | int 族已从链剔除（P163）；显式请求 fp8 不再被静态剔除（P189）|
| 编码器 M1 | fp32（权重 fp64×fp32 输入 → fp32 输出）| P107；P188 修了 BLAS 路径 b 漏 cast |
| M2 CSR val / M3 STDP / M4a WM | fp32 | P173 |
| 幂律读出 `--readout-powlaw-alpha` | 0 | 加速器对非均匀行宽 fail-fast |

- CANN 环境治理（`cann_env.py`，须在 import numpy/torch 前 importlib 加载）：TASK_QUEUE_ENABLE=2 / COMBINED_ENABLE=1 / expandable_segments / MULTI_STREAM_MEMORY_REUSE；不覆盖用户显式值。ROCm 默认 `TORCH_BLAS_PREFER_HIPBLASLT=1`（零实测，诚实标注）。
- `OMP_PROC_BIND=close`、无 OMP_PLACES（P74 已撤）；BLAS 线程 cap 8（import numpy 前）。
- 20 个 parallel=True numba 核已加 nogil=True（x86 实测负收益 −7.8%~−14%，**待服务器 A/B 定去留**）。
- **P192（2026-10-08）三开关，全部默认关闭**：
  | 开关 | 默认 | 实际是什么 |
  |---|---|---|
  | `--cython-kernels {off,auto,force}` | `off` | **真换实现**：M2/M3 走 `phdnet/_cykernels.pyx` 的 nogil 核。⚠ 有 fp32 级漂移（~1 ulp，**非逐位等价**） |
  | `--readout-pipeline` | 关 | **已停用，传了即报错退出**（fail-fast）。P117 已把读出做成异步（`forward_dev` 零同步 + `nll_sync_every=8`），额外流水没有保持语义的实现或 NPU 收益实测；贸然前移学习会改变逐 token 时序 |
  | `--cann-dispatch` | 关 | **不新增行为**，只核对 P120 已默认开的 `TASK_QUEUE_ENABLE=2`/`COMBINED_ENABLE=1` 是否**真的生效**（官方列了两个静默失效条件，都不打日志） |

  ⚠ **P192 推翻了一个旧结论**：本表早期版本写「方案 A（CPU/NPU 重叠）决定不做（真实收益仅 ~1.09×）」。
  那个 1.09× 的依据是「NPU 在等 CPU」，但**代码核对（P192）发现训练步设备路径本就没有阻塞式
  `.item()`/`.cpu()`** —— 唯一显式同步被 `nll_sync_every` 门控。故「12.09 = 4.26 + 7.83 完全串行」
  若真存在，成因是**运行时下发/队列背压**，**不是**代码里的显式同步。
  要证实这一点需要服务器 `msprof`，**本机（x86、无昇腾）无法验证**，故不给结论。
  门禁把这一事实钉死：`tests/verifiers/verify_overlap_contract.py`（23/23）。

  ⚠ **P192 同时更正了「推理侧也要重叠」这个提法**（当前自回归推理有跨 token 数据依赖）：
  | | 训练步 | 推理步 `sample_next` |
  |---|---|---|
  | 读出 | `forward_dev` → 设备张量、**零同步** | `readout(h)` → `y.float().cpu().numpy()` **硬 D2H** |
  | 下一步输入依赖上一步输出？ | 否 | **是**（自回归） |
  → 推理的 CPU 与 NPU **天然串行**，且 NPU 必须等 CPU 回读 y 才能采样
  （`infer.py:172`）——不能直接并行处理相邻生成 token。
  但全量 logits D2H 与 host 采样是当前实现选择，不是数学上必须如此；
  设备端采样等仍是候选，须独立验证采样语义与端到端收益。
  实测 host 采样段（x86、V=51,962、top-k=8）约 **0.77 ms/token**，仅占训练 CPU 侧
  4.26 ms 的 ~18%，**不是主要矛盾**（⚠ x86 口径、未含 D2H 与 NPU GEMV）。

## 4. 评测口径与锚点

- 字符归一 PPL = exp(总 NLL / 评估段字符数)；冻结语料 `eval_corpus/`（internal_corpus **27,034** 字符 + ood_wiki **20,187**，不随文档编辑漂移）。
- **历史 fp32 锚点在干净 HEAD `da8f294` 上仍可复现**，不能写成「HEAD 已回归」或「锚点作废」。
  当前工作区包含尚未提交的 STDP 学习语义修正，实测 PPL 与该历史口径不同；
  默认 fp8 与历史显式 fp32 也不是同一比较口径。
- **归因边界**：η 缩放修正让 STDP 核的 LTP/LTD 两项同乘学习率，纠正旧调用仅缩放
  LTP 的错误。这是学习行为修正，不能冒充 Cython `off` 下的无数值变化优化。
  有效隔离对照中仅反转 η 即精确恢复锚点；全工作区的微小残差未进一步分解。
  时序反转在 BASE 中无影响是因为 dual_trace=False，不可外推到 dual=True。
- **完整实测值、平台、配置及归因记录**集中在
  [性能评估与迭代方案 §10.1](PHD-Net_性能评估与迭代方案.md#101-评测锚点)。
  这些是 x86 本机结果，**不是昇腾/NPU 测量**；修正后须另建版本化基线，历史锚点不覆盖。
- 复测命令 `tools/rebaseline.py`。⚠ 该脚本**只对比不更新** `ANCHOR`；更新锚点前必须先
  定性「是回归还是口径变更」，并写明是两者中的哪一个。
- 容量口径（带 readout_conn_k 与词表）：生产 conn_k=128 → 1.217e9；稠密读出同词表 1.370e9；30b preset big_n 2^29 × big_m 56。
- 1B 档读出规模：vocab 73,958 × n_h 3,072；稀疏 k=128 = 113.6 MB/步（稠密 fp32 866.7 MiB，8.0× 省）；稀疏后每步字节流量 229.15 MiB。
- 判定口径：256 维「相当（达标）」非「更强」；256 维 / 128 维消融不可比；`evaluate()` OOV 步跳 NLL 但字符照计 → 高 OOV 文本 ppl_char 被压低（探针用 ppl_pen）。

## 5. 性能定性结论（P179–P183 修正后现行口径）

⚠⚠ **本节有一半是 Rust 臂的历史记录，而 `phdnet_rs/` 已整体删除**（P182 修 bug、
P188 fhz「删除 rust 版本，默认全部改回 python」）。下面标 🔴 的条目**代码里已不存在**，
保留仅为解释当年结论的由来，**不可作为现行选型依据**。

- 「CPU numba 核已饱和、无优化空间」**已推翻**。三个真因已修：①M1 一直走 BLAS 而非已存在的 `simd::gemv_row_avx2`（AVX2 比 BLAS 快 1.09–1.75×，🔴 该 Rust 核已删，现为 `_prefer_blas_gemv` 运行时探测）；②M2 瓶颈是 Python 校验层 `idx.min()/max()` 全扫 9.47M 元素 = 6.05ms = 65.8%；③Rust M2 融合核长期损坏（idx 漏 int64→i32，P182 已修，门禁 5FAIL→0FAIL，🔴 整个 Rust 臂随后按 P188 删除）。
- 🔴 修后（Rust 臂口径）：M2 SpMV 9.18→**3.8ms（2.4×）**、带宽 9.6→**18.8 GB/s**。
- 🔴 换核按规模门限：M2 `rs_fused_min_nnz` 默认 393216；M1 `PHDNET_M1_MIN_CELLS` 默认 1<<23。⚠ **这两个配置项已随 P188 删除**，`config.py` 不再有 `rs_*` 字段。
- 🔴 i32 idx 是 Rust SpMV 关键（P178）：i64 时 idx 流量翻倍 + SIMD gather 前须转换，压在 8.5 GB/s。
- P180 负结果：动态 work-stealing 池无收益，已回退（负载不均不是瓶颈）。
- **读出瓶颈的现行定性**：msprof（P144）显示 `aclnnIndex` 占 89.3% 但 `aicore_time=0`、`wait` 是耗时的 **6.8×** → NPU 空转等 CPU；有效带宽仅 8.1 GB/s；实测读出 7.83 ms = 带宽下界 0.300 ms 的 **26.1×** → **不是访存受限，是算子下发/算子效率**。诊断 `tools/diag_readout_npu.py`。
  ⚠ **P192 的修正**：这条「wait = 6.8×」是**运行时下发/队列背压**层面的证据，
  **不是**代码里某行显式同步造成的（训练步设备路径已零同步，见 §3 P192 条）。
  二者不矛盾，但**不可混用**：前者说「NPU 在等」，后者说「代码没写同步」——
  等的是**下发队列**，不是 `.item()`。
- 读出是 GEMV、带宽受限：NPU 50% 利用率是结构性上限；看**字节流量**不看利用率。msprof（P144）：aclnnIndex 占 89.3% 但 aicore_time=0、wait 是耗时的 6.8× → **NPU 空转等 CPU**，方向是流水线而非换算子；有效带宽仅 8.1 GB/s；实测读出 7.83ms = 带宽下界的 26.1×，瓶颈是算子效率/launch 非带宽；诊断 `tools/diag_readout_npu.py`。
- CPU 侧收口：M4b_ltm <1ms/tok（P70/P78）；分词/词表扫描 ~0% 不要优化。
- 本机测量环境限制：`process_time()`/`GetProcessTimes` 均 0；跨进程绝对值与缩放率不可比（cargo 2.38× vs Python 1.2-1.3×），只能同进程交错比 + best-of-3；x86 结论不是昇腾证据（已四次方向反转：M1 fp32、M2 融合核、OMP_PLACES、fused/plain）。
- 服务器 1b 档实测：P113 **12.09 ms/tok**（M2 plain 5.41× 后；CPU 4.26 vs NPU 7.83，读出占 64.7%）；P127 日志 15.84 ms/tok（读出 10.99 回升，**原因待查**）。
- 默认值撞加速器拒绝表 = 静默关掉全部加速（B12 族）；改默认须核对 `_unsupported_reason` + 看训练日志 `[readout] backend=` 行确认真在设备上。

## 6. 数据与训练现状

- 预训练 = 中英全量 3906 万块（52 parquet 分片 4.8GB）；SFT 用 assistant-marker 双标记状态机（只对回复计损失）；两阶段 `--init-from`；RL = REINFORCE（`tools/train_rl.py`，奖励稀疏→优势恒 0 有诊断）。
- 远程数据 `ms://owner/dataset/path`（ModelScope 直连 API，HTTP Range 流式零落盘，勿 lfs pull）；生产仓 `ms://fhzfhz/Mixture-General-Mini/`（pretrain 52 分片 / sft 178MB）；`--data pretrain|pretrain_zh|sft` 自动走远程。
- 本机 `datasets/` 已删（32.28 GiB，git LFS 历史可恢复）；`tools/fetch_l3.py` / `fetch_ms.py` 三源采样（994 GB 源，统一 schema text/lang/src）。
- 语料角色：UltraInteract 英文 SFT、deepctrl 中文 SFT、Magpie-R1 与 Ultra-FineWeb-L3 = 预训练。
- 🔴 `phdnet_rs/src/acl.rs` 曾是纯传输层（无计算核）；NPU 计算实际走 Python torch_npu（`accel_readout.py`）。CANN Rust 库（cann/cann-sys、rust-ascend、ascend-rs）调研完成，均需 aarch64 + CANN SDK 在服务器做 —— ⚠ **但 P188 已删掉 `phdnet_rs/`，故 CANN 若要接只能走 Cython（`libacl_rt.so`）或 ctypes，不是 Rust**。

## 7. 未解问题 / 仍有效待办

1. 服务器 pull 最新 + `--step-profiling` 复测：读 `segments(win)`（勿读 cum）；确认读出 10.99ms 回升原因、nogil 去留、torch_compile A/B、方案 C einsum A/B、M1 AVX2 / M2 融合核昇腾复测。
2. **远程分片中文占比仅 ~2%（文档口径 95%）、`unk` 47% 实为代码** → 数据构成排查（若属实是单位收益最高项）。
3. 幂律变长 CSR 的稀疏读出（加速器只支持均匀 k，非均匀行宽 fail-fast）。
4. fp32 化（P173 + P176 融合核 fp32 统一）后须重新 rebaseline。
5. stepfun 报告两方向：列稀疏 update（次级杠杆，forward 319MB 不可省）、CSR idx 压缩 uint16（1.43×，val 不能降 bf16）。
6. `bench_local.py` 测 fp64 而生产 fp32 的门禁盲点待修；verifier 门禁会腐烂 → 定期全量跑（`ls tests/verifiers/*.py` 现查数量，⚠ 别写死）。
7. `phdnet/generate.py` 字符级老接口清理待 fhz 确认。
8. GitHub Actions 状态待确认绿（根因已修）。

## 8. 固化工程纪律

- Edit/Write 工具改源码，**禁 heredoc**；任何 replace/Edit 后必须 grep 断言新文本存在（「补丁未落盘」犯过 5+ 次）。
- argparse help 裸 `%` → `%%`；np.lexsort 最后一个 key 是主键；gitignore 不支持行尾注释。
- 子代理后台进程回合结束即清理 → 长实验由主进程 Bash 启动；numba 缓存 `NUMBA_CACHE_DIR=outputs/numba_cache`；计时/对拍脚本必须落盘 .py（`python -c` 下 numba cache 崩）。
- 精度判据：低精度保留率基线必须经同一精度舍入，否则反向假阳性；对拍基线必须是 numba 串行版（同 fastmath）；跨库/跨设备用容差 ~4e-06，勿逐位。
- 混合 dtype = BLAS 慢路径（5.9×–70×）→ 输入 cast 跟随 W.dtype。
- numba：字面量常量 < 2³²；哈希槽位乘法混合 `(key*2654435761)&mask`；对拍集必含极小词表/低扇出用例；先量规模再优化。
- 换后端必须对齐完整调用面（__call__/learn/learn_softmax/W/conn_k/hidden/stats/n_synapses，A4 扫全仓库）；未实现配置回落+记原因；用显式标识 `_is_accel` 不用 hasattr 猜。
- fast 门禁不 import train 入口 → 改 `train/*.py` 后单独 py_compile / `--help`；verifier 比对两端须同一时刻、门禁用生产尺寸。
- gitcode 推送：`git -c credential.helper= -c credential.helper="!<wincred.exe 绝对路径>" push origin main`（PAT 存 wincred）；>100MiB 拒、LFS 配额满、仓库限额 1.0 GiB。
- `tools/rebaseline.py` 打印的「连接率」是 M2 主干（非 M6 读出 4.2%）。

## 附录：已过时 / 已被推翻主题（明细见 BUGS.md 与 git 历史）

- 版本命名沿革（v0.0.0、v1/v2 称呼、force push 前 hash）；旧锚点体系（99.36/89.15/89.48/78.16/97.2596/96.72/90.2480/73.1166 及语料漂移史）。
- 精度路线中间结论（均被推翻）：fp16 读出保非目标行（P105→P110 推翻）→ fp32 默认（P110）→ **fp16 默认（P163/P173 定案）**；bf16 默认期；fp8 forward+fp16 更新（P84，能力表遗漏致静默回落）；fp8 全线（P85/P90/P92）；int8 默认（P104/P104b→P106b 回退）；GEMV 内部 fp16（P107b，aarch64 numba 无 float16，P108 回滚）。
- 性能方向中间误判：「numba 饱和无空间」（P179 推翻）；「Rust 多线程更快/慢 2.5×」系列（校验层假象）；i32 TEMP 核（未提交即被 P179 口径覆盖）；动态 work-stealing 池（P180 回退）；LTM 向量化/predict 负收益（P60）；M4b gather 多核（P67 失败，P78 另法成功）；imprint 摊销（P122 移除）；prange 词涌现（P22 回退）；局部表复用（P21 破坏正确性）。
- 已删除代码：稠密 PC 栈（pc.py/pc_topk）、旧 torch 栈全套（TorchPHDNet/TorchWordLM/TorchReadout/TorchSTDPCore、train_torch_lm、verify_torch_lm）、train_production.py、旧 tools/train_1b.py、13 个一次性脚本。
- 已删除数据管线：MIMO、R1-Distill-110k、CodeX-2M/Qwen38 mix、hermes toolcall、UltraInteract flat、`mix` 训练流、websearch/chat_openai 演示线。
- 早期消融负结果：minibatch、课程学习、事件驱动小网络、生成式回放（无害无增益）、T3.2 检索融合（移出推荐）、top-k 检索（已证伪）。
- 已合并/删除文档：代码审计报告、对标 Transformer 优化路线图、迭代优化与修复日志（并入 BUGS.md）、v2 架构设计（并入架构设计）。
- 纯过程叙事：各日文档重写、子代理分工、push 超时、CI 反复修复、git filter-branch、ModelScope/atomgit 探测、TELEMETRY 修复细节。
