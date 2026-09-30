# Mixture-General-Mini · 上传清单与流程

适用范围：`datasets/` 交付物的分片约束、`.gitattributes` 与 LFS 的关系、以及发布动作的归属。
数据截止 2026-09-30。相关文档：`datasets/README.md`（目录结构与远程读取）。

> **上传动作由 fhz 本人执行。** `tools/upload_modelscope.py` 只负责**清单校验**与
> 分片推送这两件事，不负责决策（推哪个仓、要不要拆仓、限额怎么绕）。

## 上传内容（54 个 parquet 分片 + 3 份元文件）

```
sft/                          2 片，178.0 MB      ← SFT
pretrain/                      52 片，4.46 GiB     ← 预训练
  ├── pretrain_000.*          M7_Core 前段
  ├── pretrain_001.*
  ├── pretrain_002.*
  ├── pretrain_003.*
  └── pretrain_004.*          M7_Core 尾段 + Magpie-R1 CoT + Ultra-FineWeb-L3 中文采样
.gitattributes / README.md / UPLOAD_README.md
```

合计 **54 片 / 4.64 GiB**（按 git 对象声明的原始字节数统计；2026-09-24 清单口径）。
最大单片 **96,947,138 B ≈ 92.4 MiB**（`pretrain/pretrain_001.007.parquet`），
最小 30,506,789 B ≈ 29.1 MiB。schema 统一 `text (string) / lang (string) / src (string)`。

## 分片大小约束

远端 pre-receive 钩子的硬约束（2026-09-24 实测）：

1. **单文件 ≤ 100 MiB** —— 本清单已满足（最大 92.4 MiB）。
2. **仓库总 size ≤ 1.0 GiB** —— 本清单合计 **4.64 GiB，超出约 4.6×**。
   远端仓库曾在累计 1.8 GiB 时被拒推。
3. **Git LFS 命名空间配额已满**（实测 166 GB / 限额 100 GB）→ LFS 路线不可用。

分片由 `tools/split_parquet.py`（切成合法 parquet，无需还原步骤）或
`tools/pack_corpus.py`（gzip 分卷 + `SHA256SUMS.txt`）产出。若源文件超过 100 MiB，
必须先过这两者之一。

## `.gitattributes` 与 LFS 的关系

`datasets/.gitattributes` 的意图是**让 parquet 绕开 LFS**：

```
*.bin.* filter=lfs diff=lfs merge=lfs -text
*.bz2   filter=lfs diff=lfs merge=lfs -text

# 本机覆盖：parquet 分片均 <=100 MiB，直接以普通 git 对象入库（不走 LFS）
sft/*.parquet      -filter -diff -merge
pretrain/*.parquet -filter -diff -merge
```

最后两行的 `-filter` 表示对 parquet 禁用 LFS 过滤器。

**⚠ 实测与意图不符**：`datasets` 仓库 HEAD（`e7d4b1a`）里 **54 个 parquet 全部是
LFS 指针**（每个 133 B 的 `version https://git-lfs.github.com/spec/v1` 文本），
尽管 `.gitattributes` 声明了 `-filter`。即：这批分片**实际走了 LFS**，与注释里
「直接以普通 git 对象入库」的说法相反；LFS 配额问题也并未因这批入库而消失
（见上节第 3 条）。上面那两条 `*.bin.*` / `*.bz2` 规则只对这两类后缀生效，与 parquet 无关。

结论：**不要把 `.gitattributes` 当作「parquet 已绕过 LFS」的凭据**。要确认某个分片的
实际存储方式，看它的对象是不是 133 B 的 LFS 指针：

```bash
git -C datasets cat-file -p :pretrain/pretrain_000.000.parquet | head -1
# 期望看到 version https://git-lfs.github.com/spec/v1 → 说明是 LFS 指针
```

## atomgit 限额 1 GiB < 交付 4.64 GiB 的现实

限额 1.0 GiB 对上 4.64 GiB 的交付 —— **这条路推不全**，不要反复重试。已排除的选项：
单文件分片（已做，最大 92.4 MiB < 100 MiB，不是瓶颈）、Git LFS（配额已满）、
一次性推全量（1.0 GiB 限额直接拒绝）。剩下的可行方向（**由 fhz 决策**）：

- 在 atomgit 网页端删除并重建空仓库后**分批推送**（每批 ≤900 MB），
  或先确认该账号的 size 限额是否 >5 GiB 再一次推全量；
- 拆成多个仓库（如 `Mixture-General-Mini-pretrain-00x`，每仓 1 GB 以内）；
- 按需缩减分片范围。

**权威副本走 ModelScope**，不依赖 atomgit：`fhzfhz/Mixture-General-Mini`，
训练侧 `--remote-data` 直读（HTTP Range 流式）。所以 atomgit 推不全**不阻塞训练**。

## ModelScope 上传由谁执行

- **执行人**：fhz。本目录的工具不代替人做发布决策。
- **工具**：`tools/upload_modelscope.py`，只有两个动作：无参数 = 推送 `datasets/` 到
  `fhzfhz/Mixture-General-Mini`（排除 `.git` / `eval/` / `*/raw/` / `*.txt`）；
  `--verify` = 只核对远端文件清单与本地应有清单的差集，不写入。
- **前置**：需要 `modelscope` SDK 与有效 token（仓库 id 已在脚本内配置）。

```bash
python tools/upload_modelscope.py            # 推送
python tools/upload_modelscope.py --verify   # 只校验远端清单
```

## 本地目录约定

- `eval/` 内置语料**不外发**（`.gitignore` 与上传脚本的 `IGNORE` 都排除）。
  评测冻结基准的权威位置是主仓库根 `eval_corpus/`，不在本目录。
- 原始 txt / jsonl（`infinity_m7core.txt` 等）在 `*/raw/`，**不入库**，
  仅供制备脚本直读；散装中间 parquet（各单语料版）保留为 merge 源，同样不入库。
