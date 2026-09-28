# PHD-Net 性能评估与迭代方案（现状版）

> 评估对象：`phdnet/` 包（v0.0.0）+ `train_1b/`（1B 档流式训练管线）
> 本文为**现状快照**（2026-09-27 整理）：历史迭代过程（P1–P7、O1–O7、历史基线漂移、
> 已删除语料的记录）不再保留，只保留当前成立的结论、当前基线与当前开放项。
> 历史细节见 git 历史；回归门槛：`tests/run_tests.py` fast **11/11**。
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
  与 **GPU 原生张量核**（`TorchReadout`，见《硬件后端适配报告》）；
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
| 多进程数据加载 | W = 核数×0.8（≤文件数）+ 顺序归并 | 与串行产出逐位一致；mix 模式为样本级轮转（串行源） |
| 零等待代码 | 主循环无 sleep/轮询/忙等 | 仅 OS 级阻塞 |

## 五、泛化能力（`tools/audit_gen_eval.py`，2026-09-26 实测）

**协议**：sft（中文对话）单域 100K tokens vs **mix**（sft + pretrain 中文子集样本级 50/50 轮转）
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
- `TorchReadout`：读出热路径 torch 化（fp32/fp16/bf16 原生；fp8 需 CUDA ≥ 8.9；fp4 走 CPU 码本）；
- `TorchSTDPCore` + `selftest_torch` 等价自检（沿用）；
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
| LM 全栈 torch 化 | 立项 | 读出已 torch 化；CSR/大空间表映射为独立工程 |
| CI 真机 runner | 待硬件 | GitHub GPU runner / 自建 NPU 节点 |

## 八、复现入口

```bash
python tests/run_tests.py fast                 # 零回归门槛（11 项）
python tools/rebaseline.py                     # 当前基线复测（90.2480 / 73.1166，eta=0.15）
python tools/audit_precision.py                # 精度体系验证（L1 逐位 / L2 带宽 / L3 PPL）
python tools/audit_prof_1b.py                  # 1B 生产配置模块级剖析
python tools/audit_gen_eval.py --ckpt outputs/smoke/phdnet1b_smoke_mix.npz \
    --skip-tokens 100000 --indomain-tokens 30000 --data mix   # 三域泛化评测
python tools/audit_imprint_gate.py [--full]    # big_ltm 印迹门控实验
python tests/verifiers/verify_vocab_parallel.py  # 多核词表/加载逐位对拍
python tools/bench_accel.py                    # 加速器探针 + 读出基准（CUDA/ROCm/NPU/CPU）
python tests/run_tests.py fast                # CI 回归（三平台 CI 同源，见 .gitcode/workflows/ci.yml）
```
