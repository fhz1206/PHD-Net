# PHD-Net 性能评估与迭代方案（现状版）

> 评估对象：`phdnet/` 包（v0.0.0）+ `train_1b/`（1B 档流式训练管线）
> 本文为**现状快照**（2026-09-29 更新）：历史迭代过程（P1–P7、O1–O7、历史基线漂移、
> 已删除语料的记录）不再保留，只保留当前成立的结论、当前基线与当前开放项。
> **1B 档服务器实测（191 核昇腾 + ModelScope 30% 分片）已并入**，架构级分析见
> `docs/PHD-Net_并行与加速架构分析.md`，缺陷台账见 `BUGS.md`。

## 〇、2026-09-29 服务器实测快照（1B 档，token 7500）

| 项 | 实测 | 处置 |
|---|---|---|
| 总耗时 | 26 ms/tok（最优段；区间 20–83） | — |
| NPU 读出 | 9.5–13.5 ms/tok（11–53%） | 320 MB bf16 ×2 ≈ **64 GB/s** → 访存受限而非算力受限；已做 nll 累积 / 关图优化 / pinned H2D |
| `M4b_ltm` | 曾 **63.5 ms/tok（76%）** | 根因是 `predict` 的 1.8 万次 dict 更新 → P70 `predict_arr` **44.6×**（本机 8.78→0.197 ms） |
| 进程用核 | 1.1–3.2 / 191；CS/s 250 万–600 万 | BLAS 线程未限（**191 线程跑 1024×2048 sgemv**）→ P73 已在 `import numpy` 前限 8 线程 |
| 语种 | `--lang zh` 过滤有效（中文仅占分片 **2%**，`unk` 47% 是代码） | 语种自检 `[sample lang=zh] CJK=…%`；数据集构成问题待核 |
| PPL | 1.4–2.1 万不降 | bf16 读出使非目标行更新被舍入（\|dp\|≈1e-6 vs 半 ULP≈2e-4）→ 退化纯 Hebbian；对照 `--readout-dtype fp32` |
| 未归属开销 | 分段之和 < 总耗时（主循环从未被测量） | P72 已加主循环三段计时（`loop:`）；实测 0.06/0.13/step 102 → 黑洞在 step 内 |
| **M1_encode** | 0.85 → **58–60**（昇腾） | P77 平台自适应：aarch64 走自写 numba GEMV，x86 走 BLAS |
| **M2_infer** | 2.4 → **20–27**（昇腾） | P76 `--m2-kernel plain`（A/B 实测快 3–4×），默认 plain |
| **M4b_ltm** | 元凶 = imprint（aarch64 单次 ~20 s） | P70 predict 44.6× + P78 learn 批量多核；P67 首版已回滚 |
| readout | 9.5–13.5 → **5.5–7** | P80 cross_entropy 单 kernel + pinned correct；建议 A/B fp16 |
| 检查点 | 每 5 万步硬停 0.8–37 s | P83 `compact_csr` 向量化 + **异步写盘**（主线程快照 + 后台 worker） |
| 线程 | BLAS 191 线程 / OMP_PLACES | P73 BLAS 限 8（`import numpy` 前）；P74 撤 OMP_PLACES；P71 保留 PROC_BIND |
| 语种 | `--lang zh` 有效但中文仅 2% | 语种自检 `[sample lang=zh] CJK=…%` |
> 历史细节见 git 历史；回归门槛：`tests/run_tests.py` fast **9/9**。
> 配套：《架构设计》《对标 Transformer 优化路线图》《竞争力与脑同构性评估》《硬件后端适配报告》

---

## 一、环境与口径

| 项 | 值 |
|---|---|
| 本机 | Windows 11 / 8 线程 CPU / 12.6 GB RAM / **无 GPU·NPU** |
| 解释器 | Python 3.13（托管 venv：numpy 2.5.3 / numba 0.67 / pyarrow 25 / torch 2.14+cpu） |
| 评测语料 | `eval_corpus/internal_corpus.txt`（23,504 字符冻结基准）+ `eval_corpus/ood_wiki.txt`（远域探针）|
| ⚠ 治理约定 | **架构介绍文档（docs/*.md）不是数据集**；`eval_corpus/` 下的冻结文本是评测基准（非训练数据）|
| 训练数据 | sft（中文对话 331.9 万条）/ pretrain 分片（M7_Core+Magpie+Ultra-FineWeb）→ ModelScope `fhzfhz/Mixture-General-Mini` |
| PPL 口径 | 字符归一 PPL = exp(Σ token NLL / 评估段字符数)；跨粒度比较统一用该口径 |

**当前默认配置**（fhz 指令固化）：读出精度 **fp32**（fp64 已停止支持，可选 fp16/bf16/fp8/fp4）；
`sparse_conn=True` / `k_sparse=16`（库默认）；`eta_readout=0.15`（**fhz 2026-09-27 拍板落地**：0.05→0.15；
实测最优区间 0.15–0.20，0.25+ 退化，0.5 发散；新锚点 4K 90.2480 / 全语料 73.1166；⚠ 2026-09-28 eval_corpus 更换为维基语料后，现行锚点为 4K 394.4687 / 全语料 359.2603，本节数字为旧语料口径历史账）。

## 二、当前基线（全部可由 `tools/rebaseline.py` / `tools/audit_precision.py` 复现）

| 口径 | 配置 | ppl_char | 备注 |
|---|---|---|---|
| 4,000 字符（80/20） | BASE（256 维栈 / k_sparse=32 / 读出 fp32 / **eta=0.15**） | **90.2480**（−6.7% vs 旧 eta=0.05 锚点 96.7241） | 新默认锚点 |
| 全语料（18,803 字符） | 同上 | **73.1166**（−5.7%） | 同上 |
| 4,000 字符 | BASE + 1B 容量栈（big_ltm） | 95.4945 | −1.3%（来自替换稠密 LTM，非容量印迹） |
| 对照：Transformer 0.52M / 2.38M | 同语料 | 85.27 / 85.66 | PHD-Net 低 8.3%（达标） |

## 三、读出精度体系（P9，2026-09-26）

**指令**：停止 fp64 支持；新增 fp16/bf16/fp8/fp4；默认 fp32。实现于 `phdnet/readout.py`：
低精度 = 原生位型码本存储（fp16/bf16 → uint16，fp8 → uint8，fp4 → 半字节打包）+
查表反量化计算 + 位算法重量化融合核（bf16 RNE 位截断 / fp16 IEEE RNE 位算法 /
fp8 e4m3fn / fp4 e2m1 + 逐张量缩放）；softmax/NLL 保持 fp64 主回路。

| 精度 | 存储（1B 读出） | 4K ppl | Δ | 全语料 ppl |
|---|---|---|---|---|
| **fp32（默认，eta=0.05 时）** | 113.3 MB | 96.7241 | — | 77.5261（=历史锚点） |
| fp32（默认，**eta=0.15**） | 113.3 MB | **90.2480** | −6.70% | **73.1166** |
| fp16 | 56.6 MB | 96.5633 | −0.17% | — |
| bf16 | 56.6 MB | 95.6040 | **−1.16%** | — |
| fp8 | 28.3 MB | 95.0823 | **−1.70%** | — |
| fp4 | 14.2 MB | 102.4156 | +5.88% | — |

- 逐位验证：四种量化融合核 vs numpy 参考码本**逐元素一致**（`tools/audit_precision.py` L1）；
- 低精度 PPL 反而略优（fp8 −1.7%）＝量化噪声的正则化效应（本项目口径下实测）；
- **CPU 速度边界（诚实）**：量化更新核为计算受限（重量化位运算 > 带宽节省），
  fp32 更新仍最快（13.3 ms vs fp16 226 ms）；低精度的真实收益在**存储压缩（2–8×）**
  与 **GPU/加速器原生张量核**（`AccelReadout`，见《硬件后端适配报告》；旧 `TorchReadout`
  已随旧 torch 栈删除，P30）；
- fp4 劣化 +5.9%：e2m1 分辨率粗，当前逐张量缩放不足，块缩放（MX）立项候选。

## 四、1B 训练管线性能（train_1b）

**扫描引擎选型（P13，fhz 2026-09-28 指令「线程 + GIL 解锁吃满核心」）**：原锚点链热路径
是纯 Python 对象操作（dict/set 查找、str 切片），**全程持有 GIL**——单纯线程化只能单核；
纯 Python 也无法靠线程并行。故顺序是：先把热循环编译为释放 GIL 的机器码
（`train_1b/tokenizer_core.py`：`@njit(nogil=True)` 的 CSR-trie 贪心轨道 + δ 前缀链），
再用 `ThreadPoolExecutor` 并行——共享内存、零 pickle，主进程只做 µs 级 ndarray 接链。
numba 不可用时自动回退进程池（逐位一致）。P7 融合读出核同此思路（prange）。

| 项 | 实测 | 说明 |
|---|---|---|
| 读出占比（1B 预设，V=9,219） | fwd+update ≈ **89%** | 227 MB fp64 权重 → fp32 后 113 MB |
| 融合核带宽墙 | 1→8 线程 4.1→9.3 GB/s | 本机写混合带宽上限，fp32 逐位等价空间已耗尽 |
| 多核词表 | 词涌现按 L 并行 + 锚点链扫描 | `verify_vocab_parallel.py` 18 例逐位等价全 PASS |
| 锚点链扫描引擎（P13） | **numba `nogil` 线程池**（`tokenizer_core.py`） | 本机 6 核：8.30 → **21.41 Mchar/s（2.58×）**；w=6 与 w=4 持平（主进程 GIL 侧到顶） |
| 查找结构自适应（P15） | 小词表 CSR 二分 / 大词表**边哈希**（阈值 2 万边） | 5 万词表 **0.66 → 11.59 Mchar/s（17×，best-of-3）**；1,270 词表 10.85 Mchar/s（边哈希同口径） |
| 锚点链合并 | 词表内 bincount 位图 OR + OOV unique | 消除逐 token Python `set.update`（原主进程瓶颈） |
| 词表结构构建 | 一次遍历邻接表 + BFS 展平 | **修复 O(节点×词表) 的卡死**（5 万词建 trie 265s → 0.48s） |
| 组大小 | 缺省 32M → **1M 字符** | 32M 组时并行度塌缩（组数 ≪ worker 数），且每组数百万 token 的字符串 pickle 成单点 |
| 数据加载（P16 收敛） | **预取默认 1 进程**（满核解码），`--prefetch-workers` 可调 | 解码在 pyarrow 内多线程且释放 GIL，单进程即吃满核；多进程只增内存（每进程 ~100–200 MB）与调度。fhz 2026-09-28 定调「默认一个进程」 |
| 在途数据量（P25） | 预取队列深度 64 → **8192**，另加 `--prefetch-batch` | ⚠ 深度受**按文件序归并**约束：单生产者时囤积≈0，depth 不是主杠杆；真正有效的是 `--prefetch-workers`（多生产者）与 `--prefetch-batch`（大 IPC 批次） |
| SFT（P26） | `--assistant-marker` 回复掩码 + `--init-from` 两阶段 | 此前 SFT 语料被当普通语料训（全 token 计损失）；现只对助手回复计损失（多轮实测 ~50% 步，此前 ~0.2%），prompt 段仅推进状态；双标记状态机零滞后 |
| RL（P27） | REINFORCE，零新增算子（η ← rl_lr × advantage） | 玩具任务奖励 0.4 → 1.6 验证学习有效；无 value 网络（诚实边界） |
| 词表来源（P17） | ① `--resume` 用检查点**自包含**词表 ② `--vocab-file` 外部词表 ③ head/full 扫描 | ① ② **完全跳过扫描**（旧实现 resume 也会白扫一遍 full 词表，几十分钟） |
| 词表快照（P17/P18） | 词表一确定即落盘 `outputs/models/vocab_*.json`（**唯一权威**） | 崩溃不丢；含跨行 token（`words` + `seg_vocab` + `max_len` + sha1）；`.txt` 镜像改为可选（`txt_mirror=True`，**不可回读**）；推理侧自动交叉校验 |
| 词涌现（P18→P22） | numba **nogil**（hash 去重 + 边回比 + 解析式熵）+ 层间线程池 | 4.7M 字符生产规模：63.1s → **9.0s（7.00×）**，逐位一致（verify A 例 24/24）。**核内 prange 已回退**（P22）：实测 1→6 线程仅 1.16×，瓶颈是内存流量（两张 ~64 MB 哈希表远超 L3）；局部词表复用尝试破坏正确性已放弃（教训：性能优化先过对拍） |
| 加速器可用性（P18） | 能力矩阵显式声明 | **numba 只能编译到 CPU**（物理限制），故 NPU/CUDA 机器上生产入口仍走 CPU；加速器需走 torch 栈（权重不通用） |
| 读出计时（P20） | 训练日志 `[计时] 读出 X ms/tok（后端@设备，占 Y%）` | 直接看出加速是否生效、耗时是否转移到 PC 栈；配 `tools/accel_doctor.py` 做设备侧一次性诊断 |
| 设备张量兼容（P17） | `to_numpy()` / `_nelem()` 统一转换 | 昇腾机器实测：`np.asarray(npu:0 tensor)` 抛 "can't convert npu:0 device type tensor" → 参数统计与检查点保存双崩，现已兼容 torch 设备张量 |
| 顺序归并保序 | 按全局文件序号 reorder | 与串行产出逐位一致（`verify_vocab_parallel` D 例） |
| 零等待代码 | 主循环无 sleep/轮询/忙等 | 仅 OS 级阻塞 |

## 五、泛化能力（`tools/audit_gen_eval.py`，2026-09-26 实测）

**协议（历史记录，P42 已删除 mix 训练流）**：sft 单域 100K vs mix（sft+pretrain
50/50 轮转）——结论保留（混合域泛化更好：近域 −19% / 远域 −23%），但**现行训练
改为两阶段全量**（`--data pretrain_zh` 预训练 → `--data sft` 微调），不再用 mix。
100K tokens（smoke 256 维，词表 ≈31K）；域内 held-out / 近域（技术文档）/ 远域（维基）三域 readonly
评测；2-gram 无泄漏基线。

| 配置 | 域内 ppl_pen | 近域 inflation | 远域 inflation |
|---|---|---|---|
| sft 单域 | 64.09 | 3.96× | 19.53× |
| **mix 混合** | 81.31 | **3.21×（−19%）** | **15.03×（−23%）** |

**判读**：
1. **跨域泛化不足是当前主短板**，且**根因是数据 ≫ 架构**——总预算不变下，混合域训练
   即改善跨域 inflation 19–23%，代价是域内 +27%（预算分摊）→ 域多样性与域内深度的权衡实证；
2. **优化路线**：P0 扩大总预算 + 多域混合（M7_Core 中文子集已就位）；P1 训练量 8×
   （数据轴 β=0.085 → 域内预期 −20~30%）；P2 `eta_readout` 0.05→0.2（待拍板）；
   P3 生产训练 `--vocab-scan full`（多核后数小时级）消除训练域 OOV；
3. big_ltm 印迹门控开关 `ltm_imprint_gate`（默认 0.8）：调低印迹 2.7–6.7× 但 PPL ±0.1% 无收益；
   表生长饱和于触达神经元×60 出边（利用率 0.098%）——容量栈的真实价值场景（1M 长程检索）
   待长程训练验证。

## 六、硬件后端适配（P10，详见《硬件后端适配报告》）

- `probe_devices()`：CUDA / ROCm / CANN·NPU / CPU 统一探针（诚实降级 + 告警）；
- ~~`TorchReadout` / `TorchSTDPCore` + `selftest_torch`~~：已随旧 torch 栈删除（P30）；
- **NPU 读出优化（P28）**：`torch.outer` 物化 867 MiB 临时张量是带宽杀手 → `addmm_` AXPY
  原地融合；流量 4,334 → 2,600 MiB/步（−40%，触底）；CPU 墙钟 2.07×；数值逐位相同；
- **基准工具三 bug（P29）**：异步设备无 synchronize、dtype 从未传入、测旧对象 →
  修复前所有加速器数字不可信；现报等效带宽 GB/s（`tools/bench_accel.py`）；
- **旧 torch 栈删除（P30）**：~1,400 行；torch 栈 10 个 fast 用例与 `verify_torch_lm` 随栈删除；
  保留 `resolve_device` / `probe_devices` / `bench_readout` / `multi_device` / `AccelReadout`。
- 本机实测：CPU 参考路径 fp32 40.9 ms/token（读出规模 9219×3072）；三平台真机回归待硬件。
- **多卡自动适配（P14）**：`phdnet/backends/multi_device.py` —— `resolve_devices("auto")`
  取全部同型号设备、读出按词表行**列并行**（逐位等价已验，通信 ~74 KB/步）、
  `shard_ranges` 给出 LTM 神经元分片计划；DDP/DP 不适用（无 batch 维、无梯度），
  详见《硬件后端适配报告》§六。
- CI：`.gitcode/workflows/ci.yml`（GitCode Action）+ `.github/workflows/ci.yml`（GitHub 镜像）；验证脚本统一在 `tests/verifiers/`。

## 七、开放项（按优先级）

| 项 | 状态 | 说明 |
|---|---|---|
| P0 泛化：多域 + 大预算训练 | 数据就位，待长跑 | M7_Core 中文子集混合流式 |
| ~~P2 `eta_readout` 切换~~ | **✅ 已拍板落地（0.15）** | 新锚点 90.2480 / 73.1166（`tools/rebaseline.py`） |
| fp4 块缩放（MX） | 立项候选 | +5.9% 劣化 → 逐块 scale 可解 |
| 1M context 长程验证 | 待长跑 | big_ltm 检索通路价值验证 |
| ~~LM 全栈 torch 化~~ | **✅ 已随旧 torch 栈删除（P30）** | 加速读出走 `AccelReadout`；CSR/大空间表映射为独立工程 |
| CI 真机 runner | 待硬件 | GitHub GPU runner / 自建 NPU 节点 |

## 八、复现入口

```bash
python tests/run_tests.py fast                 # 零回归门槛（9 项）
python tools/rebaseline.py                     # 当前基线复测（90.2480 / 73.1166，eta=0.15）
python tools/audit_precision.py                # 精度体系验证（L1 逐位 / L2 带宽 / L3 PPL）
python tools/audit_prof_1b.py                  # 1B 生产配置模块级剖析
python tools/audit_gen_eval.py --ckpt outputs/smoke/phdnet1b_smoke_sft.npz \
    --skip-tokens 100000 --indomain-tokens 30000   # 域内泛化评测（P42：--data mix 已删）
python tools/audit_imprint_gate.py [--full]    # big_ltm 印迹门控实验
python tests/verifiers/verify_vocab_parallel.py  # 多核词表/加载逐位对拍
python tools/bench_accel.py                    # 加速器探针 + 读出基准（CUDA/ROCm/NPU/CPU）
python tests/run_tests.py fast                # CI 回归（三平台 CI 同源，见 .gitcode/workflows/ci.yml）
```
