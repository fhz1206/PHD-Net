"""把 ModelScope 的 mimo-claude-code-traces-1k 会话轨迹转换为 PHD-Net 文本语料。

数据源：https://www.modelscope.cn/datasets/choucisan/mimo-claude-code-traces-1k
许可证：MIT（可自由用作训练数据）
格式：session/<类目>/<hash>.jsonl，每行一条事件（Claude Code 会话轨迹）；
      `type ∈ {user, assistant}` 的行携带 `message: {role, content}`，
      content 为字符串（用户提示）或分块列表（assistant：text / thinking / tool_use…）。

用法：
    python tools/prepare_mimo.py --pull    # 首次：git lfs pull 拉取真实内容（约 40 MB）
    python tools/prepare_mimo.py           # 仅做转换（要求仓库已克隆到 data/ 下）

输出（固定外部语料，解决"语料=文档、文档一改基线就漂移"的问题）：
    data/mimo_train.txt   训练语料（按会话 90% 划分，确定性：按路径排序取模）
    data/mimo_eval.txt    评估语料（按会话 10% 划分）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://www.modelscope.cn/datasets/choucisan/mimo-claude-code-traces-1k.git"
DATA = Path(__file__).resolve().parents[1] / "datasets"
REPO = DATA / "mimo-claude-code-traces-1k"
EVAL_EVERY = 10          # 每 10 个会话取 1 个进评估集（按排序位置取模，确定性）


def ensure_repo(pull: bool) -> None:
    if not REPO.exists():
        env = {**__import__("os").environ, "GIT_LFS_SKIP_SMUDGE": "1",
               "GIT_TERMINAL_PROMPT": "0"}
        print("克隆仓库（跳过 LFS 内容）…")
        r = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)],
                           capture_output=True, text=True, env=env, timeout=1800)
        if r.returncode != 0:
            sys.exit(f"克隆失败：{(r.stderr or '')[-300:]}")
    if pull:
        print("git lfs pull（拉取真实数据内容，约 40 MB）…")
        r = subprocess.run(["git", "lfs", "pull"], cwd=str(REPO),
                           capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            sys.exit(f"lfs pull 失败：{(r.stderr or '')[-300:]}")


def message_text(o: dict) -> str:
    """提取一条 user/assistant 消息的纯文本（只要 text 块，跳过 thinking/tool_use）。"""
    msg = o.get("message")
    if not isinstance(msg, dict):
        return ""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    parts = []
    if isinstance(c, list):
        for part in c:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
    return "\n".join(p for p in parts if p.strip())


def convert() -> None:
    files = sorted(REPO.glob("session/*/*.jsonl"))
    if not files:
        sys.exit("未找到 session/**/*.jsonl —— 请先用 --pull 拉取数据")
    train_parts: list[str] = []
    eval_parts: list[str] = []
    n_sessions = n_msgs = n_chars = 0
    for idx, f in enumerate(files):
        blocks = []
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except Exception:
                    continue                              # 坏行跳过（容错）
                if o.get("type") not in ("user", "assistant"):
                    continue
                t = message_text(o)
                if t.strip():
                    blocks.append(t.strip())
                    n_msgs += 1
        if not blocks:
            continue
        session_text = "\n\n".join(blocks)
        n_chars += len(session_text)
        (eval_parts if idx % EVAL_EVERY == 0 else train_parts).append(session_text)
        n_sessions += 1

    train_out = DATA / "mimo_train.txt"
    eval_out = DATA / "mimo_eval.txt"
    train_out.write_text("\n\n".join(train_parts), encoding="utf-8")
    eval_out.write_text("\n\n".join(eval_parts), encoding="utf-8")
    print(f"会话 {n_sessions} 个（训练 {len(train_parts)} / 评估 {len(eval_parts)}）"
          f"  消息 {n_msgs:,} 条  正文 {n_chars:,} 字符")
    print(f"输出: {train_out.name} {train_out.stat().st_size / 1024 / 1024:.1f} MB / "
          f"{eval_out.name} {eval_out.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="mimo 轨迹 → PHD-Net 文本语料")
    ap.add_argument("--pull", action="store_true", help="先执行 git lfs pull")
    args = ap.parse_args()
    ensure_repo(args.pull)
    convert()
