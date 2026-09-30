# Mixture-General-Mini · 上传清单与流程

适用范围：`datasets/` 交付物的分片约束、`.gitattributes` 与 LFS 的关系、
发布动作的归属与限额现实。
**数据截止：2026-09-30。** 相关文档：`README.md`（目录结构与远程读取）、
`../tools/README.md`（工具索引）。

> **上传动作由 fhz 本人执行。** `../tools/upload_modelscope.py` 只负责**清单校验**与
> **分片推送**这两件事，不负责决策（推哪个仓、要不要拆仓、限额怎么绕）。

---

## 一、上传内容（54 个 parquet 分片 + 3 份元文件）

工具口径（`tools/upload_modelscope.py` docstring + `local_expect()`）：

| 路径 | 片数 | 说明 |
|---|---|---|
| `sft/*.parquet` | 2 | SFT |
| `pretrain/*.parquet` | 52 | 预训练（含 M7_Core 段与中文 L3 采样段） |
| `.gitattributes` / `README.md` / `UPLOAD_README.md` | 3 | 元文件 |

排除清单（脚本内 `IGNORE`）：`.git/**`、`raw/**`、`sft/raw/**`、`pretrain/raw/**`、
`eval/**`、`*.txt`。

**已记录的实测值（2026-09-24 清单口径）**：合计 **54 片 / 约 4.6 GiB**，
最大单片约 92.4 MiB。分片明细随每次制备变化，**引用前请以
`ls -l datasets/sft/*.parquet datasets/pretrain/*.parquet` 的实况为准**；
若与本文数字不符，**以实况为准并更新本文**（需确认项）。

---

## 二、分片大小约束

| 约束 | 值 | 状态 |
|---|---|---|
| 单文件 ≤ 100 MiB | 最大片约 92.4 MiB | 已满足 |
| 仓库总 size ≤ 1.0 GiB | 交付约 4.6 GiB | **超出约 4.6×，此路推不全** |
| Git LFS 命名空间配额 | 曾实测 166 GB / 限额 100 GB | **已满，LFS 路线不可用** |

分片产出工具（二选一，都不需要还原步骤）：

- `../tools/split_parquet.py` —— 切成**合法 parquet**，`pyarrow` 可直读；
- `../tools/pack_corpus.py` —— gzip 分卷 + `SHA256SUMS.txt`（绕开单文件限额时用）。

源文件超过 100 MiB 时**必须**先过这两者之一。

---

## 三、`.gitattributes` 与 LFS 的关系

`datasets/.gitattributes` 现状：

```
*.bin.* filter=lfs diff=lfs merge=lfs -text
*.bz2 filter=lfs diff=lfs merge=lfs -text

# 本机覆盖：parquet 分片均 <=100 MiB，直接以普通 git 对象入库（不走 LFS）
sft/*.parquet      -filter -diff -merge
pretrain/*.parquet -filter -diff -merge
```

设计意图是**让 parquet 绕开 LFS**：最后两行的 `-filter` 表示对 parquet 禁用 LFS 过滤器；
前两条只对 `.bin.*` / `.bz2` 后缀生效，**与 parquet 无关**。

⚠ **实测与意图不符（记录，需复核）**：此前一次提交（记录为 `e7d4b1a`）里
54 个 parquet **全部是 LFS 指针**（每个 133 B 的
`version https://git-lfs.github.com/spec/v1` 文本），尽管 `.gitattributes` 声明了 `-filter`
—— 即这批分片**实际走了 LFS**，与注释里「直接以普通 git 对象入库」相反，
LFS 配额问题也未因这批入库而消失。

**结论：不要把 `.gitattributes` 当作「parquet 已绕过 LFS」的凭据。** 要确认某个分片的
实际存储方式，看它的对象是不是 133 B 的 LFS 指针：

```bash
git -C datasets cat-file -p :pretrain/pretrain_000.000.parquet | head -1
# 出现 version https://git-lfs.github.com/spec/v1 → 是 LFS 指针
```

**需确认**：上述观察是在**加 `-filter` 之前**的提交上做的；加规则之后新提交的分片
是否仍走 LFS，需按上面命令复核一次。

---

## 四、atomgit 限额 1 GiB < 交付约 4.6 GiB 的现实

限额 1.0 GiB 对上约 4.6 GiB 的交付 —— **这条路推不全，不要反复重试**。
已排除的选项：单文件分片（已做，最大约 92.4 MiB < 100 MiB，不是瓶颈）、
Git LFS（配额已满）、一次性推全量（1.0 GiB 限额直接拒绝）。

剩下的可行方向（**由 fhz 决策**）：

- 在 atomgit 网页端删除并重建空仓库后**分批推送**（每批 ≤ 900 MB），
  或先确认该账号的 size 限额是否 > 5 GiB 再一次推全量；
- 拆成多个仓库（如 `Mixture-General-Mini-pretrain-00x`，每仓 1 GB 以内）；
- 按需缩减分片范围。

**需确认**：远端限额数值（单文件 1.0 GiB / 总 size、LFS 配额）来自 2026-09-24 的实测，
平台策略可能已变；**推送前先在网页端确认当前限额**。

**权威副本走 ModelScope**，不依赖 atomgit：`fhzfhz/Mixture-General-Mini`，
训练侧 `--remote-data` 直读（HTTP Range 流式）。所以 **atomgit 推不全不阻塞训练**。

---

## 五、ModelScope 上传由谁执行

- **执行人**：fhz。本目录的工具不代替人做发布决策。
- **工具**：`../tools/upload_modelscope.py`，只有两个动作：
  - 无参数 = 推送 `datasets/` 到 `fhzfhz/Mixture-General-Mini`
    （排除 `.git` / `eval/` / `*/raw/` / `*.txt`）；
  - `--verify` = 只核对远端文件清单与本地应有清单的差集，**不写入**。
- **前置**：需要 `modelscope` SDK 与有效 token。仓库 id 已在脚本内配置；
  **token 目前也硬编码在脚本里**（`TOKEN = "ms-..."`）—— 能读到本仓库的人都能读到它。
  收敛为环境变量 / 凭据库属**需确认**的待办。

```bash
python tools/upload_modelscope.py            # 推送
python tools/upload_modelscope.py --verify   # 只校验远端清单
```

`--verify` 的判据是「远端文件清单 ⊇ 本地应有清单」，输出差集；
它**不校验内容哈希**，因此不能证明远端内容与本地一致。

---

## 六、本地目录约定

- `eval/` 内置语料**不外发**（`.gitignore` 与上传脚本 `IGNORE` 双重排除）。
  评测冻结基准的权威位置是**主仓库根 `eval_corpus/`**，不在本目录。
- 原始 txt / jsonl 在 `*/raw/`，**不入库**，仅供制备脚本直读；
  散装中间 parquet（各单语料版）保留为 merge 源，同样不入库。
- 上传缓存 `datasets/.ms_upload_cache` 已在主仓库 `.gitignore` 中排除。
