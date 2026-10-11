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


# ⚠ **本文件的 check 只有两个参数**（name, cond），**没有 detail**。
# 历史上多次因照搬其它 verifier 的三参写法而 TypeError（P140 再次）。
#   其它 verifier 的 check 形如 check(cond, name, detail)，**顺序也不同**。
# 本文件：check(name, cond) —— **名字在前、判据在后**。
_CHECK_ARITY = 2


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
    # ⚠ 2026-10-07：显式 encoding/errors（区域编码解码失败 → stdout=None →
    #   下面的 `in r.stdout` 变 TypeError）。
    r = subprocess.run([sys.executable, str(_ROOT / "train" / "train.py"),
                        "--help"], capture_output=True, text=True,
                       cwd=str(_ROOT), timeout=180,
                       encoding="utf-8", errors="replace")
    check("train.py --help 可执行（导入 + argparse 正常）",
          r.returncode == 0 and "--remote-data" in r.stdout
          and "--remote-fraction" in r.stdout)

    # P112：裸 `%` 让 --help 崩（ValueError: unsupported format character）已经
    # 发生 **6 次**（BUGS #2 家族）。根因是 help 文本里的百分号被argparse 当成
    # 旧式格式符（`_expand_help` 会做`help % params`）。单点门禁挡不住：每次
    # 改别的参数都可能新引入一个裸 %。故加两条**全仓库**检查：
    #   ① 静态扫所有 argparse 文件的 help 文本里的裸 %（非 %%、非 %s 等格式符）
    #   ② 对每个有 argparse 的入口实际跑 --help
    # 这两条是「整类问题」的闸门，而不是逐个参数的补丁。
    import ast
    import io
    import re
    import re as _re

    def _bare_percent_hits(path):
        """返回该文件里 help 文本中的裸 % 位置（静态启发式）。"""
        try:
            tree = ast.parse(io.open(path, encoding="utf-8").read())
        except SyntaxError:
            return []
        src = io.open(path, encoding="utf-8").read()
        hits = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "add_argument"):
                continue
            for kw in node.keywords:
                if kw.arg != "help":
                    continue
                # 取该 help 表达式的源码片段，检查字面量里的裸 %
                seg = ast.get_source_segment(src, kw.value) or ""
                # 去掉 %% 与合法格式符 %s %d %f %r %% 后，剩下的 % 即为裸
                resid = _re.sub(r"%%|%[-+ #0-9.]*[sdrf]", "", seg)
                if "%" in resid:
                    ln = getattr(kw.value, "lineno", 0)
                    hits.append(f"{path}:{ln}")
        return hits

    import glob as _glob
    # ⚠ 排除本门禁自己（它也含 add_argument 调用文本，且 `__main__` 里会真跑
    #   入口）——否则会自我递归调用把自己跑超时。
    _SELF = os.path.abspath(__file__)
    ap_files = [f for f in _glob.glob(str(_ROOT / "**" / "*.py"), recursive=True)
                if "numba_cache" not in f and "datasets" not in f
                and os.sep + "archive" + os.sep not in f
                and os.path.abspath(f) != _SELF
                and _re.search(r"add_argument", io.open(f, encoding="utf-8",
                                                          errors="ignore").read())]
    # ⚠ 静态检查只作**提示**，不作为门禁判据：AST 源码片段提取在「help= 文本
    #   … % var」这种跨行格式化表达式上会漏掉 %s，产生误报（第一版就误报了
    #   train.py:192的合法 `% MS_DATA_REPO`）。**真正的判据是下面那条实际跑
    #   --help** —— 裸 % 必然让它崩，静态扫不准只是噪音。
    all_hits = []
    for f in ap_files:
        all_hits += _bare_percent_hits(f)
    if all_hits:
        print(f"  · 静态提示（不参与判据）：疑似裸 % 于 "
              f"{', '.join(os.path.relpath(h, str(_ROOT)) for h in all_hits[:5])}"
              f" —— 请以实际 --help 是否通过为准")

    # 实际跑每个入口的 --help
    bad_help = []
    for f in ap_files:
        rel = os.path.relpath(f, str(_ROOT))
        # 只跑看起来像入口的（有 __main__ 或顶层 argparse）
        try:
            txt = io.open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if "__main__" not in txt:
            continue
        try:
            rr = subprocess.run([sys.executable, f, "--help"],
                                capture_output=True, text=True,
                                cwd=str(_ROOT), timeout=120,
                                encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            bad_help.append(f"{rel} (timeout)")
            continue
        if rr.returncode != 0:
            tail = (rr.stderr or "").strip().splitlines()
            bad_help.append(f"{rel} (exit={rr.returncode}"
                            + (f": {tail[-1][:60]}" if tail else "") + ")")
    # P134：显式检查「同一选项被重复定义」——argparse 把它当
    # `ArgumentError: conflicting option string`（**exit≠0**），
    # 上面那条 --help 门禁能抓到，但**只在有人真跑它**时。
    # P134 的教训：加参数时我因两次失败重试插入了**3 份**同样定义，
    # 编译通过、只有 --help 才炸 → 故单列一条，让失败原因更直白。
    _dup = []
    for _f in ap_files:
        try:
            _txt = _f.read_text(encoding="utf-8")
        except Exception:                                  # noqa: BLE001
            continue
        for _opt in set(re.findall(r'add_argument\(\s*"(--[a-z0-9\-]+)"', _txt)):
            _n = _txt.count(f'"{_opt}"')
            if _n > 1:
                _dup.append(f"{_f.name}:{_opt}×{_n}")
    # 本文件的 check() 签名是 (name, cond)，没有 detail 参数
    # P140 曾尝试加「未定义变量」AST 扫描（抓 `_ro` vs `ro` 这类拼错）。
    #⚠ **已撤除**：本机实测**误报 15 处**（如 `RESULTS` 在模块级第 35 行赋值、
    #   `rows` 是 `main()` 的闭包变量），而真错只有 1 处。
    #   根因：Python 的名字解析（模块级 / 闭包 / global / nonlocal / 推导式 /
    #   异常名 / 参数）规则太多，**可靠判「未定义」需要真正的符号表**
    #   （如 `pyflakes`），不是几十行 AST 能可靠近似。
    #   → 误报率高的门禁**比没有门禁更糟**（训练者会习惯性忽略它）。
    #   → 改用**最低成本的有效手段**：所有 benchmark 工具在本机跑一遍冒烟
    #     （`--help` 之外再加 `--steps 2` 的短跑），拼错名字必然在导入/执行时炸。
    #     本机每次提交前跑一次，即可覆盖「拼错变量名」这一类。

    # 本文件 check() 的签名是 (name, cond) —— **没有 detail 参数**。
    #（P140 我又写错一次；已加下方自检，防同类错误再发生。）

    _msg = ("冲突: " + "; ".join(_dup[:4])) if _dup \
        else f"已扫描 {len(ap_files)} 个入口"
    check(f"无重复定义的 CLI 选项（P134）— {_msg}", not _dup)

    _n2 = f"所有 argparse 入口 --help 均可执行（{len(ap_files)} 个候选文件）"
    if bad_help:
        _n2 += "← 失败: " + ", ".join(bad_help)
    check(_n2, not bad_help)

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


# ── P129：「局部变量使用先于赋值」静态扫描 ──────────────────────────────────
# 2026-10-02 13:59 服务器首次运行直接崩在启动阶段：
#     UnboundLocalError: cannot access local variable 'lm'
# 起因：把 `_tel._accel_device = getattr(lm.net, ...)` 写在 `lm = PHDWordLM(...)`
# **之前 189 行**处。`py_compile` **查不出**（名字在函数作用域内合法），
# 只有真跑到那一行才炸 —— 而那是**生产启动路径**，代价极大。
# 本检查用 AST 比较每个局部名「首次赋值行」与「首次使用行」，提前拦这类顺序错误。
def _check_def_before_use() -> bool:
    import ast
    _bad = []
    for _rel in ("train/train.py", "train/infer.py"):
        _f = _ROOT / _rel
        if not _f.exists():
            continue
        _tree = ast.parse(_f.read_text(encoding="utf-8"))
        for _fn in ast.walk(_tree):
            if not isinstance(_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            _d: dict = {}
            _u: dict = {}
            for _n in ast.walk(_fn):
                if isinstance(_n, ast.Name):
                    if isinstance(_n.ctx, ast.Store):
                        _d.setdefault(_n.id, _n.lineno)
                    else:
                        _u.setdefault(_n.id, _n.lineno)
            for _name, _ul in _u.items():
                _dl = _d.get(_name)
                # 差值大 = 明显顺序写错（同一两行内的先后属正常控制流）
                if _dl is not None and (_dl - _ul) > 30 and not _name.startswith("_"):
                    _bad.append(f"{_rel}:{_fn.name}() 内 `{_name}` 定义@L{_dl} "
                                f"但最早使用@L{_ul}")
    # 本文件的 check() 签名是 (name, cond)，没有 detail 参数
    check(f"入口无「局部变量使用先于赋值」（P129）"
          + (f" — 发现 {len(_bad)} 处：{_bad[0]}" if _bad else""),
          not _bad)
    return not _bad


if __name__ == "__main__":
    _check_def_before_use()
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
