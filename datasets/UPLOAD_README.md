# Mixture-General-Mini · 上传清单与说明

> 生成：2026-09-25（Claw）。上传由 fhz 执行。

## 上传内容（54 个 parquet 分片，均 ≤100 MiB，可直接 pyarrow 读取）

```
sft/                          2 片，178 MB，3,319,142 条   ← SFT（UltraInteract 英文 + deepctrl 中文）
pretrain/                    52 片，4.60 GB，39,059,702 条 ← 预训练
  ├── pretrain_000.*        11 片（Infinity-Instruct M7_Core 前段）
  ├── pretrain_001.*        11 片（M7_Core 中段）
  ├── pretrain_002.*        11 片（M7_Core 后段）
  ├── pretrain_003.*        11 片（M7_Core 尾段）
  └── pretrain_004.*         8 片（M7_Core 尾部 + Magpie-R1 CoT + Ultra-FineWeb-L3 中文采样）
```

schema 统一为 `text (string) / lang (string: zh|en) / src (string: 来源语料名)`。

## ⚠ 仓库限额警告（2026-09-24 实测）

gitcode/atomgit 同一后端，pre-receive 钩子强制：
1. 单文件 ≤ 100 MiB（本清单已满足，最大 96.9 MB）；
2. **仓库总 size ≤ 1.0 GiB** —— 本清单合计 **4.64 GiB，超出限额 ~4.6×**。
   远端仓库在 2026-09-24 已因累计 1.8 GiB 被拒推。
3. Git LFS：命名空间配额 166 GB / 100 GB **已满**，LFS 路线不可用。

**可行选项**（按优先级）：
- 在 atomgit 网页端**删除并重建空仓库**后分批推送（每批 ≤900 MB），
  或确认 atomgit 对该账号的 size 限额 >5 GiB 后一次推全量；
- 或拆成多个仓库（如 Mixture-General-Mini-pretrain-00x，每仓 1 GB）；
- 或按需缩减（例如 pretrain 只保留 000/001 两档 ≈ 1.9 GiB，仍超，需再减）。

## 本地读取示例

```python
from phdnet.corpus import load_text, iter_texts
text = load_text("datasets/pretrain/pretrain_000.000.parquet", limit_chars=4096)
for row in iter_texts("datasets/sft/sft_000.000.parquet"):
    ...
```

## 本地目录约定

- `eval/` 内置语料**不外发**；原始 txt 与 merged 原件在 */raw/（不入库）（评测基线锚点，见 PHD-Net 仓库文档）；
- 原始 txt（infinity_m7core.txt 等）保留于本地供训练脚本直读，未入库；
- 散装中间 parquet（deepctrl/ultrainteract/magpie/ultrafineweb 单语料版）保留为 merge 源。
