"""复现性验证：同一份配置在两个独立 Python 进程中是否给出逐位相同的结果。

背景：demo_m4_capacity 两次运行分别报出 69.37 与 68.23（差 1.7%），怀疑存在跨进程不可复现。
怀疑对象：Python 默认开启 hash 随机化（PYTHONHASHSEED 未固定），
         任何依赖 str hash / set 迭代顺序的路径都会漂移。

⚠ P215 审计修复：原实现**只跑一个进程、只 print 一行 RESULT、没有任何断言**，
   却被 `gate_*.py` 前缀纳入 runner → 永远 PASS（零断言假绿），
   且完全没有兑现 docstring 宣称的「两个独立进程逐位比对」。
   现改为：父进程以**两个不同 PYTHONHASHSEED** 各起一个子进程，
   比对两次完整结果是否**逐位相同**；不同即判失败。
"""

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# eval_common 随功能分类移到了 tests/evals/；平铺导入契约见 tests/checks_common.py
for _sub in ("demos", "evals", "gates"):
    _p = str(ROOT / "tests" / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_common import BASE, SEG                                # noqa: E402
from phdnet.config import PHDNetConfig                       # noqa: E402
from phdnet.word_lm import PHDWordLM                         # noqa: E402

# P126 修复：语料早已从 `datasets/eval/` 迁到**仓库根 `eval_corpus/`**
# （`datasets/README.md` 有记录），但本验证器一直指旧路径 → FileNotFoundError。
# 与 gate_readout_fused 同源：**门禁自身腐烂**，只是没人跑它。
# 加 fallback：两个位置都试，给出清晰错误而不是裸 FileNotFoundError。
_DOC_CANDIDATES = (ROOT / "eval_corpus" / "internal_corpus.txt",
                   ROOT / "datasets" / "eval" / "internal_corpus.txt")
DOC = next((p for p in _DOC_CANDIDATES if p.exists()), _DOC_CANDIDATES[0])


def _run_once() -> dict:
    """跑一次完整训练 + 评测，返回可逐位比对的结果字典。"""
    text = DOC.read_text(encoding="utf-8")[:6000]
    tr, ev = text[:4800], text[4800:]
    lm = PHDWordLM(text, PHDNetConfig(**BASE), seg_kwargs=SEG)
    lm.train_stream(tr)
    m = lm.evaluate(ev)
    return {
        "ppl_char": float(m["ppl_char"]),
        "bpc": float(m["bpc"]),
        "n_tok": int(m.get("n_tok") or 0),
        "hash_rand": int(sys.flags.hash_randomization),
    }


def main() -> int:
    # ── 子进程模式：只输出机器可读的结果，不再递归 spawn ──
    if "--child" in sys.argv:
        print("RESULT " + json.dumps(_run_once(), sort_keys=True))
        return 0

    # ── 父进程：两个不同 PYTHONHASHSEED 的独立进程 ──
    seeds = ("0", "12345")
    outs: list[str] = []
    for seed in seeds:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        r = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()),
                            "--child"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=300)
        if r.returncode != 0:
            print(f"FAIL 子进程 PYTHONHASHSEED={seed} 退出码 {r.returncode}")
            print((r.stderr or "")[-500:])
            return 1
        line = next((l for l in (r.stdout or "").splitlines()
                     if l.startswith("RESULT ")), "")
        if not line:
            print(f"FAIL 子进程 PYTHONHASHSEED={seed} 未产出 RESULT")
            return 1
        outs.append(line[len("RESULT "):])
        print(f"  PYTHONHASHSEED={seed}: {outs[-1]}")

    # ⚠ `hash_rand`（sys.flags.hash_randomization）只是**诊断字段**：
    #   父进程刻意设了不同的 PYTHONHASHSEED，它必然不同（0 vs 1），
    #   不是模型结果的差异 → 比对时排除，只比数值结果。
    def _numeric(line: str) -> dict:
        d = json.loads(line)
        return {k: d[k] for k in ("ppl_char", "bpc", "n_tok")}

    same = (_numeric(outs[0]) == _numeric(outs[1]))
    print("-" * 70)
    print(f"跨进程逐位一致: {same}   → {'PASS' if same else 'FAIL'}")
    if not same:
        print("  ⚠ 两个 hash seed 下结果不同 → 存在依赖 str hash / set 迭代顺序的路径，"
              "这是 docstring 所怀疑的**真复现性缺陷**，不要靠固定 seed 掩盖。")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
