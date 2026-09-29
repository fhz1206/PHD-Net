"""ModelScope 数据集直读（``ms://`` 协议）——HTTP Range 流式，零原始落盘。

fhz 2026-09-29（服务器 ``--data pretrain`` 报「glob 无匹配」后拍板走此路）：
**默认仍从本地 datasets/ 加载**（逐位不变）；用户通过 ``--remote-data`` 切换
为 ModelScope 直读——**HTTP Range 流式**（206 分块随机读，pyarrow 直接消费
可 seek 文件流，与 tools/fetch_l3.py 同款、项目已实测），**只支持
ModelScope**（数据集托管于 modelscope.cn，atomgit 限额 1 GiB 本就推不全）。

实现（fhz 指示「用 modelscope 库」）：
- 列目录：``HubApi.get_dataset_files``（官方 SDK，分页支持，MS_TOKEN 可选）；
- 读文件：fsspec HTTP Range 可 seek 流 → ``pq.ParquetFile(filelike)``，
  **零原始落盘**（依赖 fsspec + aiohttp）；
- **分片级抽样**：``--remote-fraction 0.3`` = 取排序后**前 30%** 分片
  （前缀子集——保持数据顺序语义，词表扫描与训练流开头逐字符一致）；
  worker 进程经环境变量继承 fraction。

路径格式：``ms://<owner>/<dataset>/<数据集内路径或 glob>``
例：``ms://fhzfhz/Mixture-General-Mini/pretrain/pretrain_*.parquet``
"""
from __future__ import annotations

import fnmatch
import math
import os
import urllib.parse
from pathlib import Path as _Path

_PREFIX = "ms://"
# Range 随机读块大小（fsspec http block_size；顺序读时退化为预取窗口）
_BLOCK_MB = 8
# 列目录分页大小：取 modelscope SDK 默认值（服务端同量级，已实测 /pretrain
# root_path 带前导斜杠 + page_size=100 一次取全 52 分片）
_PAGE_SIZE = 100
# Range 读超时/重试（审计 C1：长跑中一次网络抖动不该崩全局训练）
_HTTP_TIMEOUT_S = 60
_READ_RETRIES = 3
_RETRY_BACKOFF_S = 2.0
# 远程模式的预取进程上限（审计 B2：8 进程 × 独立连接池可能打爆服务端）
REMOTE_MAX_PROCS = 4


class MsFile:
    """远程 ModelScope 文件的轻量 Path 鸭子（供 corpus 层统一消费）。

    只实现下游真正用到的接口：``suffix`` / ``name`` / ``__str__`` / 排序比较。
    ``__str__`` 返回可再次解析的完整 ``ms://`` spec——主进程展开一次，
    分派给 worker 的是**具体文件** spec（worker 内零列目录请求）。
    """

    __slots__ = ("ns", "fp", "size")

    def __init__(self, ns: str, fp: str, size: int | None = None):
        self.ns = ns
        self.fp = fp                                  # 数据集内相对路径
        self.size = size                              # 字节数（列表 API 提供）

    @property
    def name(self) -> str:
        return self.fp.rsplit("/", 1)[-1]

    @property
    def suffix(self) -> str:
        n = self.name
        return ("." + n.rsplit(".", 1)[-1]) if "." in n else ""

    def __str__(self) -> str:
        return f"{_PREFIX}{self.ns}/{self.fp}"

    def __repr__(self) -> str:
        return f"MsFile({str(self)!r})"

    def __eq__(self, other):
        return isinstance(other, MsFile) and (self.ns, self.fp) == (other.ns, other.fp)

    def __hash__(self):
        return hash((self.ns, self.fp))

    def __lt__(self, other):
        return str(self) < str(other)


def _parse(spec: str) -> tuple[str, str]:
    """``ms://<owner>/<dataset>/<path>`` → (repo_id, 数据集内路径)。

    ns 是 ModelScope repo_id（owner/dataset **两段**）——⚠ 不能用单段
    partition 解析（会把数据集名切进路径，服务器 400 的根因）。
    路径按 URL 规范解码一次（防「已编码的 %20」被二次编码成 %2520）。
    """
    parts = str(spec)[len(_PREFIX):].split("/", 2)
    if len(parts) < 3 or not all(p.strip() for p in parts):
        raise ValueError(
            f"ms:// 路径格式应为 ms://<owner>/<dataset>/<path>: {spec}")
    tail = urllib.parse.unquote(parts[2])
    if parts[2].startswith("/") or "//" in tail:
        raise ValueError(f"ms:// 路径不得含空段 '//': {spec}")
    if ".." in tail.split("/"):
        raise ValueError(f"ms:// 路径不得含 '..'（路径穿越）: {spec}")
    return f"{parts[0]}/{parts[1]}", tail.lstrip("/")


def _fraction() -> float:
    """分片抽样比例（环境变量传递——spawn worker 不继承模块级变量）。"""
    raw = os.environ.get("PHDNET_REMOTE_FRACTION", "1")
    try:
        f = float(raw)
    except ValueError:
        raise ValueError(f"PHDNET_REMOTE_FRACTION 不是数字: {raw!r}")
    if f <= 0 or f > 1:
        raise ValueError(f"--remote-fraction 必须在 (0, 1]（收到 {f}）")
    return f


def _api():
    from modelscope.hub.api import HubApi
    return HubApi()


def _auth_headers() -> dict:
    """私有数据集需 MS_TOKEN（公开数据集如 fhzfhz/Mixture-General-Mini 免认证）。

    ⚠ 必须带浏览器 UA：ModelScope WAF 拦截默认 `Python-urllib` UA
    （实测 HTTP 400，curl 同 URL 200）。
    """
    h = {"User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36"}
    tok = os.environ.get("MS_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _list_tree(repo_id: str, root: str) -> list[dict]:
    """列数据集目录（官方 SDK 分页循环；root 允许为空 = 仓库根）。

    分页判据用**空页终止**而非 `len(files) < page_size`：服务端可能把
    page_size 钳到更小值（SDK 默认 100 暗示服务端同量级），用请求值比较会
    在首页就误判「已到末页」→ **静默丢数据**（审计 B3，必修）。
    page_size 取 SDK 默认 100（与端点已验证组合一致）。
    """
    api = _api()
    out, page = [], 1
    while True:
        files = api.get_dataset_files(repo_id=repo_id,
                                      root_path=("/" + root) if root else "/",
                                      recursive=False, page_number=page,
                                      page_size=_PAGE_SIZE)
        if not files:
            return out
        out.extend(files)
        page += 1


def _take_prefix(hits: list[str]) -> list[str]:
    """分片级抽样：取排序后**前** fraction 比例（前缀子集，顺序语义不变）。"""
    f = _fraction()
    if f >= 1.0:
        return hits
    n = max(1, math.ceil(len(hits) * f))
    return hits[:n]


def expand_ms(spec: str) -> list[MsFile]:
    """解析 ``ms://`` spec → 远程文件列表（确定性排序 + 分片级抽样）。

    - 无通配符 → 返回单文件，**不列目录**（worker 内零额外请求）；
      分片抽样只发生在主进程的 glob 展开处。
    - 带通配符 → 列最后一层目录 + fnmatch 过滤文件名。
      **只支持单层**（pattern 不含 `/`、不支持 `**` 跨目录）——与
      ``tools/fetch_ms.py::walk_parquets`` 的手工递归不同，这里显式拒绝而非
      静默失效（审计 A4）。
    """
    repo_id, tail = _parse(spec)
    if not any(ch in tail for ch in "*?["):
        return [MsFile(repo_id, tail)]
    root, _, pat = tail.rpartition("/")
    if any(ch in root for ch in "*?[]"):
        raise NotImplementedError(
            f"ms:// 仅支持单层 glob（目录部分不能含通配符）: {spec}")
    files = _list_tree(repo_id, root)
    hits = sorted(f["Path"] for f in files
                  if f.get("Type") == "blob"
                  and fnmatch.fnmatch(f["Path"].rsplit("/", 1)[-1], pat))
    if not hits:
        raise FileNotFoundError(
            f"ms:// glob 无匹配: {spec}（root={root or '<仓库根>'}, "
            f"目录内共 {len(files)} 个对象）。常见原因：① 路径/模式写错；"
            f"② 仓库非公开——设置环境变量 MS_TOKEN；③ ModelScope WAF 拦截"
            f"（HTTP 400/403，非认证问题）")
    hits = _take_prefix(hits)
    sizes = {f["Path"]: f.get("Size") for f in files if f.get("Type") == "blob"}
    return [MsFile(repo_id, h, sizes.get(h)) for h in hits]


def open_ms_file(spec: str, block_mb: int = _BLOCK_MB):
    """``ms://`` spec → 可 seek 的 HTTP 文件流（Range 随机读，零落盘）。

    依赖 fsspec + aiohttp（fetched from tools/fetch_l3.py 同款，项目已实测）。
    私有数据集经 MS_TOKEN 认证。**不重试**：重试需重建文件流（fsspec 句柄
    不可复用），由调用方 ``phdnet.corpus.open_parquet_source`` 统一重试。
    """
    try:
        import fsspec
    except ImportError as e:                          # pragma: no cover
        raise SystemExit("Range 流式需要 fsspec：pip install fsspec aiohttp") from e
    repo_id, fp = _parse(spec)
    url = (f"https://www.modelscope.cn/api/v1/datasets/{repo_id}/repo"
           f"?Revision=master&FilePath={urllib.parse.quote(fp)}")
    h = _auth_headers()
    kw = {"trust_env": True, "headers": h}
    try:                                              # 显式超时（审计 C1：
        import aiohttp                                # aiohttp 默认 5min 总超时
        kw["timeout"] = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_S,
                                              connect=15, sock_read=_HTTP_TIMEOUT_S)
    except ImportError:                               # pragma: no cover
        pass
    fs = fsspec.filesystem("http", client_kwargs=kw)
    return fs.open(url, "rb", block_size=int(block_mb) << 20)


def is_remote(path) -> bool:
    """是否远程 spec（字符串以 ms:// 开头；MsFile 实例恒真）。"""
    if isinstance(path, MsFile):
        return True
    return isinstance(path, str) and path.startswith(_PREFIX)


def ms_total_size(files: list[MsFile]) -> int:
    """远程文件总字节数（已下载的按缓存实际大小；未下载的按 API Size）。"""
    total = 0
    for f in files:
        if f.size is not None:
            total += f.size
    return total


def local_name_of(p) -> str:
    """远程/本地对象的纯文件名（日志与分派用）。"""
    return p.name if isinstance(p, MsFile) else _Path(str(p)).name
