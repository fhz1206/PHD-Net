"""ms:// 远程数据源纯函数验证（**零网络**）——P54 审计补测。

fast 门禁 9/9 不覆盖 ``phdnet/ms_stream.py``，这里补齐无需联网的部分：
spec 解析 / 分片抽样 / MsFile 鸭子语义 / 本地路径不误判。
联网部分（列目录 + Range 读）靠服务器首次 ``--remote-data`` 冒烟。

运行：``python tests/verifiers/verify_ms_stream.py``（退出码 0 = 全 PASS）
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "train"))

import phdnet.ms_stream as ms          # noqa: E402
from phdnet.corpus import expand_paths, is_remote_path   # noqa: E402

CASES: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    CASES.append((name, bool(cond)))
    print(f"  {'✓' if cond else '✗'} {name}")


def _main() -> int:
    print("ms:// 纯函数验证（零网络）")

    # ── spec 解析（repo_id 两段；防路径穿越 / 空段 / URL 双重编码）─────────
    check("解析 owner/dataset/path 三段",
          ms._parse("ms://fhzfhz/Mixture-General-Mini/pretrain/a.parquet")
          == ("fhzfhz/Mixture-General-Mini", "pretrain/a.parquet"))
    check("路径遍历 '..' 被拒",
          _raises(ValueError, ms._parse, "ms://o/d/../etc/passwd"))
    check("空段 '//' 被拒",
          _raises(ValueError, ms._parse, "ms://o/d//a.parquet"))
    check("缺段被拒", _raises(ValueError, ms._parse, "ms://o/d"))
    check("URL 编码只解一次（a%20b 不被二次编码）",
          ms._parse("ms://o/d/a%20b.parquet")[1] == "a b.parquet")

    # ── 本地路径绝不误判为远程（零回归前提）──────────────────────────────
    for local in ("datasets/pretrain/pretrain_000.000.parquet",
                  "eval_corpus/internal_corpus.txt",
                  "ms:/single/slash.parquet", "C:/data/x.parquet",
                  "a/ms://b/c", ""):
        check(f"本地路径不误判: {local!r}",
              not is_remote_path(local) and not ms.is_remote(local))
    check("ms:// 字符串判为远程",
          is_remote_path("ms://o/d/p.parquet") and ms.is_remote("ms://o/d/p.parquet"))
    check("MsFile 实例判为远程", ms.is_remote(ms.MsFile("o/d", "p.parquet")))

    # ── expand_paths：本地 glob 走原逻辑；ms:// 单文件分支零网络 ────────────
    local_dir = _ROOT / "eval_corpus"
    if (local_dir / "internal_corpus.txt").exists():
        hits = expand_paths(local_dir / "internal_corpus.txt")
        check("本地单文件 glob 正常展开", len(hits) == 1
              and hits[0].suffix == ".txt")
    one = ms.expand_ms("ms://o/d/pretrain/a.parquet")
    check("ms:// 单文件分支：只返回 1 个 MsFile（不列目录）", len(one) == 1)
    check("单文件分支：str 可再解析（worker 分派幂等）",
          ms._parse(str(one[0])) == ("o/d", "pretrain/a.parquet"))
    check("单文件分支：suffix/name 鸭子语义",
          one[0].suffix == ".parquet" and one[0].name == "a.parquet")

    # ── 分片抽样：前缀子集（顺序语义不变）────────────────────────────────
    hits = [f"p/{i:03d}.parquet" for i in range(52)]
    os.environ["PHDNET_REMOTE_FRACTION"] = "0.3"
    check("fraction=0.3 → 前 30%（ceil(52×0.3)=16）",
          len(ms._take_prefix(hits)) == 16)
    check("抽样结果是前缀子集（顺序不变）",
          ms._take_prefix(hits) == hits[:16])
    os.environ["PHDNET_REMOTE_FRACTION"] = "1"
    check("fraction=1 → 全量", ms._take_prefix(hits) == hits)
    os.environ["PHDNET_REMOTE_FRACTION"] = "0"
    check("fraction 越界被拒（0）", _raises(ValueError, ms._fraction))
    os.environ["PHDNET_REMOTE_FRACTION"] = "1.5"
    check("fraction 越界被拒（>1）", _raises(ValueError, ms._fraction))
    os.environ["PHDNET_REMOTE_FRACTION"] = "abc"
    check("fraction 非数字被拒", _raises(ValueError, ms._fraction))
    os.environ.pop("PHDNET_REMOTE_FRACTION", None)
    check("fraction 缺省 = 1.0（全量）", ms._fraction() == 1.0)

    # ── MsFile 不能被 Path 化（这是 corpus 侧必须先判 is_remote 的原因）──
    from pathlib import Path as _P
    check("Path(ms://…) 会丢语义 → 调用方须先判 is_remote",
          not ms.is_remote(_P(str(one[0]))))

    # ── corpus 侧：远程 spec 不被 Path 化（B4 回归测试，源码级静态断言）────
    import inspect
    from phdnet import corpus as _c
    check("corpus_stats 走远程分支（不 Path() 化 ms://）",
          "is_remote_path(path)" in inspect.getsource(_c.corpus_stats))
    check("corpus_stats 远程体积来自 API Size（无 getsize）",
          "ms_total_size" in inspect.getsource(_c.corpus_stats))
    check("_iter_one 对非 parquet 远程源显式 NotImplementedError",
          "NotImplementedError" in inspect.getsource(_c._iter_one))
    check("load_text 对非 parquet 远程源显式 NotImplementedError",
          "NotImplementedError" in inspect.getsource(_c.load_text))
    check("load_text 只展开一次路径（远程不再二次列目录）",
          inspect.getsource(_c.load_text).count("expand_paths(path)") == 1)
    check("open_parquet_source 本地分支逐位等价（ParquetFile(str(p))）",
          "pq.ParquetFile(str(p))" in inspect.getsource(_c.open_parquet_source))
    check("open_parquet_source 远程有重试",
          "_READ_RETRIES" in inspect.getsource(_c.open_parquet_source))
    check("_list_tree 空页终止（不按 len<page_size 判末页）",
          "if not files:" in inspect.getsource(ms._list_tree))
    check("train.py 已删 stream_factory 死代码（不再绕过 data_path）",
          "stream_factory" not in (_ROOT / "train" / "train.py").read_text(
              encoding="utf-8"))

    # ── 语法门禁：改动过的文件必须能编译（审计 P54 教训：fast 门禁不 import
    #    train/train.py，缩进错误曾直接漏到服务器才炸）──────────────────
    import py_compile
    import tempfile
    for rel in ("train/train.py", "train/corpus_stream.py",
                "phdnet/ms_stream.py", "phdnet/corpus.py", "phdnet/readout.py",
                "tests/verifiers/verify_ms_stream.py"):
        ok = True
        with tempfile.TemporaryDirectory() as td:
            try:
                py_compile.compile(str(_ROOT / rel), doraise=True,
                                   cfile=str(Path(td) / "out.pyc"))
            except py_compile.PyCompileError as e:
                ok = False
                print(f"    ! {rel}: {e}")
        check(f"编译通过: {rel}", ok)
    # train.py 必须真的能被 import（argparse 层不出错）——不跑训练
    import subprocess
    r = subprocess.run([sys.executable, str(_ROOT / "train" / "train.py"),
                        "--help"], capture_output=True, text=True,
                       cwd=str(_ROOT), timeout=180)
    check("train.py --help 可执行（导入 + argparse 正常）",
          r.returncode == 0 and "--remote-data" in r.stdout
          and "--remote-fraction" in r.stdout)

    n_ok = sum(1 for _, ok in CASES if ok)
    print(f"\n通过 {n_ok}/{len(CASES)}")
    return 0 if n_ok == len(CASES) else 1


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception:                    # noqa: BLE001
        return False
    return False


if __name__ == "__main__":
    raise SystemExit(_main())


def test_bf16_tonumpy_roundtrip() -> None:
    """P81：bf16 张量 → to_numpy(uint16 位模式) → view 回 bf16，无损。

    服务器实测（2026-09-30）：readout_dtype=bf16 时检查点保存崩在
    `TypeError: Got unsupported ScalarType BFloat16`（numpy 无原生 bf16）。
    加载侧 P46 已支持 uint16 位模式解码，保存侧必须配套。
    """
    import torch
    from phdnet.model import to_numpy

    t = torch.randn(256, dtype=torch.bfloat16)
    arr = to_numpy(t)
    assert arr.dtype == np.uint16, f"应为 uint16 位模式，得到 {arr.dtype}"
    back = torch.from_numpy(arr).view(torch.bfloat16)
    assert torch.equal(back, t), "bf16 round-trip 必须无损"
    # 非 bf16 路径不受影响
    a32 = to_numpy(torch.randn(8))
    assert a32.dtype == np.float32
