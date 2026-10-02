"""复现性验证：同一份配置在两个独立 Python 进程中是否给出逐位相同的结果。

背景：demo_m4 两次运行分别报出 69.37 与 68.23（差 1.7%），怀疑存在跨进程不可复现。
怀疑对象：Python 默认开启 hash 随机化（PYTHONHASHSEED 未固定），
         任何依赖 str hash / set 迭代顺序的路径都会漂移。
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.eval_common import BASE, SEG                      # noqa: E402
from phdnet.config import PHDNetConfig                       # noqa: E402
from phdnet.word_lm import PHDWordLM                         # noqa: E402

# P126 修复：语料早已从 `datasets/eval/` 迁到**仓库根 `eval_corpus/`**
# （`datasets/README.md` 有记录），但本验证器一直指旧路径 → FileNotFoundError。
# 与 verify_readout_fused 同源：**门禁自身腐烂**，只是没人跑它。
# 加 fallback：两个位置都试，给出清晰错误而不是裸 FileNotFoundError。
_DOC_CANDIDATES = (ROOT / "eval_corpus" / "internal_corpus.txt",
                   ROOT / "datasets" / "eval" / "internal_corpus.txt")
DOC = next((p for p in _DOC_CANDIDATES if p.exists()), _DOC_CANDIDATES[0])


def main() -> None:
    text = DOC.read_text(encoding="utf-8")[:6000]
    tr, ev = text[:4800], text[4800:]
    lm = PHDWordLM(text, PHDNetConfig(**BASE), seg_kwargs=SEG)
    lm.train_stream(tr)
    m = lm.evaluate(ev)
    print(f"RESULT ppl_char={m['ppl_char']:.6f} bpc={m['bpc']:.6f} "
          f"n_tok={m.get('n_tok')} hash_rand={sys.flags.hash_randomization}")


if __name__ == "__main__":
    main()
