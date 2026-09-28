"""chat/tui.py —— PHD-Net 聊天 TUI 入口（仅 Python 标准库 + numpy）。

界面（ANSI 转义序列实现，Windows Terminal / 新版 conhost 可用）：
  - 顶部状态栏：模型名 / 词表规模 / 设备 / 对话轮次（反显固定在首行）
  - 对话历史区：用户消息青色、助手消息绿色、系统提示暗色
  - 底部输入框：`>` 提示符；行尾反斜杠续行、单独一行 `.` 结束多行输入
  - 助手回复逐字符打印（打字机效果；PHDWordLM.generate 一次性返回，纯渲染模拟）

后端：
  - word（默认）：`phdnet.word_lm.PHDWordLM`，pickle 模型（chat_r1sft.py 同款加载）
  - 1b：`train_1b/infer.py` 的自包含 npz 检查点（词表内嵌，无需语料）

降级：终端不支持 ANSI（无 tty / 环境探测失败 / TERM=dumb）时自动退化为
普通逐行打印；`--force-tui` 可强制走 TUI 渲染路径（管道/重定向调试用）。

用法：
  python chat/tui.py                                  # word 后端（默认模型 outputs/r1sft_model.pkl）
  python chat/tui.py --model outputs/r1sft_model.pkl  # 指定 pickle 模型
  python chat/tui.py --ckpt outputs/models/phdnet1b_xxx.npz   # 1B 检查点
  python chat/tui.py --selftest                       # 无模型：随机小模型走通 UI 循环
  python chat/tui.py --selftest --prompt "你好"       # 单次模式验证渲染路径

斜杠命令：/help /quit /clear /save <path>

诚实边界：模型是词级联想续写器（见 train_1b/infer.py 的说明），生成的是
格式上连贯的文本，不保证事实正确或真正的语义问答。
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_ROOT / "train_1b")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phdnet.word_lm import PHDWordLM  # noqa: E402

CHAT_TEMPLATE = "用户：{q}\n助手："
SELFTEST_CHARS = 500  # selftest 语料截断长度

# ────────────────────────────── ANSI ──────────────────────────────
ESC = "\x1b"
RESET = f"{ESC}[0m"
BOLD = f"{ESC}[1m"
DIM = f"{ESC}[2m"
CYAN = f"{ESC}[36m"     # 用户
GREEN = f"{ESC}[32m"    # 助手
MAGENTA = f"{ESC}[35m"  # 系统/错误
INVERSE = f"{ESC}[7m"   # 状态栏


def detect_ansi(force: bool) -> bool:
    """探测终端 ANSI 支持；不支持则上层退化为逐行打印。"""
    os.system("")  # Windows：激活 conhost 的 ANSI 转义支持（WT/新 conhost 必备）
    if force:
        return True
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        return bool(os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM")
                    or os.environ.get("ANSICON") or os.environ.get("ConEmuANSI")
                    == "on")
    return os.environ.get("TERM", "") != "dumb"


class Screen:
    """TUI 渲染器。rich=True 走光标定位 + 滚动区；否则普通逐行打印。

    布局（rich 模式）：
      第 1 行        状态栏（滚动区之外，滚动不破坏）
      第 2..h-1 行   对话历史滚动区（追加式输出，自然上滚）
      第 h 行        输入框
    """

    def __init__(self, rich: bool):
        self.rich = rich
        self.out = sys.stdout
        self.transcript: list[tuple[str, str]] = []   # (role, text)
        self.rows, self.cols = shutil.get_terminal_size()
        self._status = ""
        if rich and self.rows < 5:
            self.rich = False                          # 终端太矮，降级

    # ── 生命周期 ──
    def enter(self) -> None:
        if not self.rich:
            return
        self.out.write(f"{ESC}[2J{ESC}[H")             # 清屏 + 归位
        self.out.write(f"{ESC}[2;{self.rows - 1}r")    # 滚动区 = 历史区
        self.out.flush()

    def exit(self) -> None:
        if not self.rich:
            return
        self.out.write(f"{ESC}[r")                     # 复位滚动区
        self.out.write(f"{ESC}[{self.rows};1H{ESC}[2K")
        self.out.write(f"{ESC}[{self.rows - 1};1H")    # 光标停到历史区底部
        self.out.flush()

    # ── 状态栏 ──
    def set_status(self, text: str) -> None:
        self._status = text
        if not self.rich:
            return
        bar = f" {text} ".ljust(self.cols)[:self.cols]
        self.out.write(f"{ESC}[1;1H{ESC}[2K{INVERSE}{BOLD}{bar}{RESET}")
        self.out.flush()

    # ── 历史区 ──
    def _at_region_bottom(self) -> None:
        self.out.write(f"{ESC}[{self.rows - 1};1H")

    def emit(self, label: str, text: str, color: str) -> None:
        self.transcript.append((label, text))
        body = f"{color}{BOLD}{label}{RESET}{color}{text}{RESET}"
        if not self.rich:
            self.out.write(body.replace("\n", "\r\n") + "\r\n")
        else:
            self._at_region_bottom()
            self.out.write(body.replace("\n", "\r\n") + "\r\n")
        self.out.flush()

    def note(self, text: str) -> None:
        """系统提示（暗色），不进 transcript。"""
        if not self.rich:
            self.out.write(f"{DIM}{text}{RESET}\r\n")
        else:
            self._at_region_bottom()
            self.out.write(f"{DIM}{text}{RESET}\r\n")
        self.out.flush()

    def typewriter(self, label: str, text: str, color: str,
                   delay: float = 0.008) -> None:
        """打字机效果：逐字符输出（PHDWordLM 一次性返回，纯渲染模拟流式）。"""
        self.transcript.append((label, text))
        if not self.rich:
            self.out.write(f"{color}{BOLD}{label}{RESET}{color}")
            self.out.flush()
        else:
            self._at_region_bottom()
            self.out.write(f"{color}{BOLD}{label}{RESET}{color}")
        for ch in text:
            self.out.write("\r\n" if ch == "\n" else ch)
            self.out.flush()
            if delay > 0:
                time.sleep(delay)
        self.out.write(f"{RESET}\r\n")
        self.out.flush()

    def clear(self) -> None:
        self.transcript.clear()
        if self.rich:
            self.enter()
            self.set_status(self._status)
        else:
            self.out.write("─" * 40 + "\r\n")
            self.out.flush()

    # ── 输入框 ──
    def read_line(self, prompt: str) -> str | None:
        """读一行；EOF/Ctrl-C 返回 None。rich 模式定位到第 h 行输入框。"""
        try:
            if self.rich:
                self.out.write(f"{ESC}[{self.rows};1H{ESC}[2K{DIM}>{RESET} ")
                self.out.flush()
                return input("")
            return input(f"{DIM}{prompt}{RESET}" if sys.stdout.isatty() else prompt)
        except (EOFError, KeyboardInterrupt):
            self.out.write("\r\n" if self.rich else "\n")
            return None


# ────────────────────────────── 后端加载 ──────────────────────────────
def device_label() -> str:
    try:
        from phdnet.device import probe
        return probe().name
    except Exception:
        return "cpu"


def build_selftest_model() -> PHDWordLM:
    """随机初始化小模型（eval_common.py 的 BASE/SEG 缩小版），走通 UI 循环用。"""
    from phdnet.config import PHDNetConfig
    corpus = _ROOT / "eval_corpus" / "internal_corpus.txt"
    text = (corpus.read_text(encoding="utf-8")[:SELFTEST_CHARS]
            if corpus.exists() else
            "用户：你好。助手：你好，很高兴见到你。用户：今天天气如何？助手：今天天气晴朗。")
    cfg = PHDNetConfig(n_sdr=64, k_sparse=8, n_mid=64, n_top=64,
                       eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
                       pred_in_readout=True)
    lm = PHDWordLM(text, cfg, seg_kwargs=dict(max_len=6, min_count=1, min_entropy=0.0))
    lm.train_stream(text, log_every=10 ** 9)   # 快速过一遍语料（500 字符级）
    return lm


def load_backend(args) -> tuple[PHDWordLM, str]:
    """返回 (lm, 模型显示名)。word=pickle，1b=train_1b 自包含 npz。"""
    if args.selftest:
        return build_selftest_model(), "selftest(小模型)"

    if args.ckpt:                                   # --ckpt → 强制 1b
        path, backend = Path(args.ckpt), "1b"
    elif args.model:
        path = Path(args.model)
        backend = ("1b" if path.suffix == ".npz" else "word") \
            if args.backend == "word" and path.suffix == ".npz" else args.backend
    else:
        path, backend = Path("outputs/r1sft_model.pkl"), "word"

    if not path.exists():
        sys.exit(f"未找到模型 {path}\n"
                 f"  word 后端：--model outputs/r1sft_model.pkl（outputs/train_r1sft.py 产物）\n"
                 f"  1b 后端：  --ckpt outputs/models/phdnet1b_*.npz\n"
                 f"  或 --selftest 用随机小模型走通 UI。")

    if backend == "1b":
        import infer as infer_1b                    # train_1b/infer.py
        lm = infer_1b.load_from_ckpt(path)
    else:
        with open(path, "rb") as f:
            lm = pickle.load(f)
    return lm, path.name


# ────────────────────────────── 生成 ──────────────────────────────
def warmup(lm: PHDWordLM, tokens, oov: list) -> tuple[str | None, str]:
    """prompt 预热（learn=False）。返回尾部两个已消化 token (prev, cur)。"""
    prev = cur = None
    tail: list[str] = []
    stoi = lm.tok.stoi
    for tk in tokens:
        if tk in stoi:
            lm.net.step(lm.tok.encode_composite(tk, prev), learn=False)
            prev = tk
            tail.append(tk)
            if len(tail) > 2:
                tail.pop(0)
        else:
            oov.append(tk)
    if len(tail) == 2:
        return tail[0], tail[1]
    if len(tail) == 1:
        return None, tail[0]
    return None, lm.tok.tokens[0]


def sample_next(lm: PHDWordLM, prev: str | None, cur: str, tau: float,
                topk: int, rng: np.random.Generator) -> str:
    """自回归一步：读出分布 → 温度 τ → top-k 截断 → 采样（train_1b/infer.py 同式）。"""
    y = lm.net.step(lm.tok.encode_composite(cur, prev),
                    learn=False)["y"].astype(np.float64) / max(tau, 1e-6)
    y -= y.max()
    p = np.exp(y)
    p /= p.sum()
    if 0 < topk < len(p):
        idx = np.argpartition(-p, topk - 1)[:topk]
        pp = p[idx] / p[idx].sum()
        j = int(rng.choice(idx, p=pp))
    else:
        j = int(rng.choice(len(p), p=p))
    return lm.tok.tokens[j]


def generate_reply(lm: PHDWordLM, user_msg: str, n_tokens: int, tau: float,
                   topk: int, seed: int, streaming: bool) -> str:
    """预热 + 自回归续写。word 后端走 lm.tokenize，1b 后端走流式分词（同 infer.py）。"""
    prompt = CHAT_TEMPLATE.format(q=user_msg)
    if streaming:                                   # 1b：与训练侧逐位一致的流式分词
        from corpus_stream import SEP, StreamingTokenizer
        tokens = StreamingTokenizer(lm.tok.seg, iter([prompt + SEP]))
    else:
        tokens = lm.tokenize(prompt)
    rng = np.random.default_rng(seed)
    prev, cur = warmup(lm, tokens, [])
    out = []
    for _ in range(n_tokens):
        nxt = sample_next(lm, prev, cur, tau, topk, rng)
        out.append(nxt)
        prev, cur = cur, nxt
    return "".join(out)


# ────────────────────────────── 斜杠命令 ──────────────────────────────
HELP_TEXT = ("命令：/help 帮助 | /quit 退出 | /clear 清屏 | /save <path> 保存对话\n"
             "多行输入：行尾 \\ 续行，单独一行 . 结束多行输入并提交")


def handle_command(cmd: str, scr: Screen) -> bool:
    """返回 False 表示退出。"""
    parts = cmd.split(maxsplit=1)
    name = parts[0].lower()
    if name in ("/quit", "/q", "/exit"):
        return False
    if name == "/clear":
        scr.clear()
        scr.note("已清屏。")
    elif name == "/save":
        if len(parts) < 2:
            scr.note("用法：/save <path>")
        else:
            path = Path(parts[1].strip())
            path.write_text("\n\n".join(f"{r}{t}" for r, t in scr.transcript),
                            encoding="utf-8")
            scr.note(f"对话已保存至 {path}（{len(scr.transcript)} 条）")
    elif name == "/help":
        scr.note(HELP_TEXT)
    else:
        scr.note(f"未知命令 {name}，/help 查看帮助。")
    return True


# ────────────────────────────── 主循环 ──────────────────────────────
def run(args) -> None:
    print("加载模型中...", flush=True)
    lm, model_name = load_backend(args)
    vocab = len(lm.tok)
    scr = Screen(rich=detect_ansi(args.force_tui))
    scr.enter()
    status = (f"PHD-Net TUI │ {model_name} │ 词表 {vocab:,} │ "
              f"{device_label()} │ 轮次 0")
    scr.set_status(status)
    if not args.prompt:
        scr.note("输入消息后回车发送；/help 查看命令，/quit 退出。")

    def one_turn(q: str, turn: int) -> None:
        scr.set_status(f"PHD-Net TUI │ {model_name} │ 词表 {vocab:,} │ "
                       f"{device_label()} │ 轮次 {turn}")
        scr.emit("你 > ", q, CYAN)
        t0 = time.perf_counter()
        reply = generate_reply(lm, q, args.max_tokens, args.tau, args.topk,
                               args.seed + turn * 97, streaming=args.backend == "1b")
        scr.typewriter("助手 > ", reply, GREEN)
        scr.note(f"（{len(lm.tokenize(reply))} tokens，"
                 f"{time.perf_counter() - t0:.1f}s）")

    if args.prompt:                                  # 单次模式：验证渲染路径
        one_turn(args.prompt, 1)
        scr.exit()
        return

    turn = 0
    while True:
        buf: list[str] = []
        while True:                                  # 多行：行尾 \ 续行，. 结束
            line = scr.read_line("> " if not buf else "… ")
            if line is None:
                line = "/quit"
            if buf and line.rstrip("\r\n") == ".":
                break
            if line.endswith("\\") and len(line) >= 1:
                buf.append(line[:-1])
                continue
            buf.append(line)
            break
        msg = "\n".join(buf).strip()
        if not msg:
            continue
        if msg.startswith("/"):
            if not handle_command(msg, scr):
                break
            continue
        turn += 1
        one_turn(msg, turn)

    scr.exit()
    print("再见。")


def main() -> None:
    ap = argparse.ArgumentParser(description="PHD-Net 聊天 TUI（纯标准库）")
    ap.add_argument("--backend", choices=["word", "1b"], default="word",
                    help="word=pickle 模型 / 1b=train_1b 自包含 npz（默认 word）")
    ap.add_argument("--model", default=None, help="word 后端 pickle 模型路径")
    ap.add_argument("--ckpt", default=None, help="1b 后端 npz 检查点（隐含 --backend 1b）")
    ap.add_argument("--tau", type=float, default=0.8, help="采样温度")
    ap.add_argument("--max-tokens", type=int, default=200, help="单轮生成 token 数")
    ap.add_argument("--topk", type=int, default=8, help="top-k 截断（0=全分布）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=None, help="单次模式：发送一条消息后退出")
    ap.add_argument("--selftest", action="store_true",
                    help="无模型：随机小模型走通 UI 循环")
    ap.add_argument("--force-tui", action="store_true",
                    help="强制 TUI 渲染路径（管道/重定向调试用）")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
