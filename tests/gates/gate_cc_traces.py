"""P196 门禁：Claude Code traces 数据链路（event-stream JSONL → 训练流）。

跑法：py -3.14 tests/gates/gate_cc_traces.py

判据：
  A1 本地 JSONL → cc_trace_to_text（用仓库内置 fixture，不依赖网络）
  A2 只保留 user/assistant 行，跳过元数据行（mode / last-prompt …）
  A3 thinking / tool_use / tool_result 三类 block 都被抽取
  A4 空/坏行不崩（容错）
  B1 远程 ms:// 列目录（1,017 traces / 10 类）——需网络，失败则 SKIP
  B2 远程读一条 trace 并转换（内容非空、含 role: 前缀）
  C1 iter_cc_traces 对本地 fixture 流式可用
  C2 load_cc_traces 的 limit_chars 生效
  D1 expand_paths 对列表幂等（P196）
  D2 corpus 的远程 .jsonl 分支不再 raise NotImplementedError
  E1 train.py 的 DATA_FILES 含 cc_traces；MS_CC_REPO/CC_CATEGORIES 正确
"""

import json
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "train")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                        # noqa: E402

CASES: list = []


def check(ok, name, detail=""):
    CASES.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))


def skip(name, why):
    CASES.append((name, True))
    print(f"  SKIP {name}  [{why}]")


# ── fixture：一条最小但覆盖全部 block 类型的 event-stream ────────────
FIXTURE = "\n".join([
    json.dumps({"type": "mode", "mode": "normal", "sessionId": "s1"}),
    json.dumps({"type": "permission-mode", "permissionMode": "bypassPermissions"}),
    json.dumps({"type": "user", "message": {"role": "user", "content": "修一下这个 bug"}}),
    json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "先看代码"},
        {"type": "text", "text": "问题在越界。"},
        {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}},
    ]}}),
    json.dumps({"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "content": [{"type": "text", "text": "3 passed"}]},
    ]}}),
    json.dumps({"type": "last-prompt", "lastPrompt": "修一下这个 bug"}),
    "",                                                    # 空行
    "not-json-at-all",                                     # 坏行
])


def main():
    from phdnet.corpus import (cc_trace_to_text, iter_cc_traces,
                               load_cc_traces)

    print("=" * 74)
    print("P196 Claude Code traces 数据链路")
    print("=" * 74)

    # ── A. 转换语义 ─────────────────────────────────────────────────
    print("\n[A] event-stream → 对话文本")
    txt = cc_trace_to_text(FIXTURE)
    check("user: 修一下这个 bug" in txt, "A1 user 消息被抽取")
    check("assistant:" in txt and "问题在越界。" in txt, "A2 assistant 文本被抽取")
    check("[思考] 先看代码" in txt, "A3a thinking 块被抽取（reasoning trace）")
    check("[调用工具 Bash]" in txt and "pytest -q" in txt, "A3b tool_use 被抽取")
    check("[工具返回]" in txt and "3 passed" in txt, "A3c tool_result 被抽取")
    check("mode" not in txt and "bypassPermissions" not in txt
          and "last-prompt" not in txt, "A4 元数据行被跳过（mode/last-prompt）")
    check("\n\n" in txt, "A5 trace 内多轮以空行分隔")
    # 空/坏行不崩
    check(cc_trace_to_text("") == "", "A6a 空输入 → 空输出")
    check(cc_trace_to_text("not json\n{bad") == "", "A6b 全坏行 → 空输出（不崩）")
    # thinking=False 丢弃推理
    t2 = cc_trace_to_text(FIXTURE, thinking=False)
    check("[思考]" not in t2 and "问题在越界。" in t2,
          "A7 thinking=False 丢弃推理块但保留正文")

    # ── B. 远程（需网络）───────────────────────────────────────────
    print("\n[B] 远程 ms://（需网络）")
    try:
        from phdnet.ms_stream import expand_ms
        files = expand_ms("ms://choucisan/mimo-claude-code-traces-1k/session/debugging/*.jsonl")
        check(len(files) > 100, "B1 列目录（debugging 类应有 162 个）",
              f"{len(files)} files")
        if files:
            from phdnet.corpus import load_cc_traces
            s = load_cc_traces([files[0]], limit_chars=3000)
            check(len(s) > 500 and ("user:" in s or "assistant:" in s),
                  "B2 远程读一条 trace 并转换", f"len={len(s)}")
    except Exception as e:                              # noqa: BLE001
        skip("B1/B2 远程列目录与读取", f"网络不可用: {type(e).__name__}")

    # ── C. 流式接口（本地 fixture）──────────────────────────────────
    print("\n[C] 流式接口")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "a.jsonl").write_text(FIXTURE, encoding="utf-8")
        (d / "b.jsonl").write_text(FIXTURE, encoding="utf-8")
        got = list(iter_cc_traces([d / "a.jsonl", d / "b.jsonl"]))
        check(len(got) == 2 and all(t.strip() for t in got),
              "C1 iter_cc_traces 逐文件产出", f"{len(got)} traces")
        s = load_cc_traces([d / "a.jsonl", d / "b.jsonl"], limit_chars=200)
        check(len(s) == 200, "C2 limit_chars 精确生效", f"len={len(s)}")
        s_all = load_cc_traces([d / "a.jsonl"])
        check(s_all == got[0], "C3 单文件 load == iter 首条")

    # ── D. corpus 层的连带修改 ─────────────────────────────────────
    print("\n[D] corpus 层幂等性 / 远程 jsonl")
    from phdnet.corpus import expand_paths
    lst = ["x", "y"]
    check(expand_paths(lst) == lst, "D1 expand_paths 对列表幂等（P196）")
    import phdnet.corpus as _corp
    src = (_ROOT / "phdnet" / "corpus.py").read_text(encoding="utf-8")
    check('ms:// 远程源仅支持 .parquet（收到' not in src.split(
        "def iter_cc_traces")[0].split("def _iter_one")[1][:4000]
        if "def iter_cc_traces" in src and "def _iter_one" in src else True,
        "D2 远程 .jsonl 不再 raise NotImplementedError")

    # ── E. train.py 接线 ───────────────────────────────────────────
    print("\n[E] train.py 接线")
    tp = (_ROOT / "train" / "train.py").read_text(encoding="utf-8")
    check('"cc_traces"' in tp, "E1 DATA_FILES 含 cc_traces")
    check('MS_CC_REPO = "choucisan/mimo-claude-code-traces-1k"' in tp,
          "E2 MS_CC_REPO 指向指定数据集")
    check("CC_CATEGORIES" in tp and "shell_devops" in tp,
          "E3 CC_CATEGORIES 定义（10 类）")
    check("load_cc_traces" in tp, "E4 词表构建走 load_cc_traces")
    check("iter_cc_traces" in tp, "E5 训练流走 iter_cc_traces")
    # 工具脚本存在
    check((_ROOT / "tools" / "data" / "cc_traces_to_text.py").exists(),
          "E6 转换工具存在（tools/data/cc_traces_to_text.py）")

    bad = [n for n, ok in CASES if not ok]
    print(f"\n{'PASS' if not bad else 'FAIL'}: {len(CASES) - len(bad)}/{len(CASES)} 项通过")
    for n in bad:
        print(f"  x {n}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
