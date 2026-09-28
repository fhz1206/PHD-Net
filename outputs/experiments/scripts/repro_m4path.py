"""定位 demo_m4 的跨进程不可复现来源。

现象：同一份 demo_m4.py 两次独立运行分别报出
    "接入后字符归一 PPL 68.23 vs 稠密 74.00"（本次）  与
    "接入后字符归一 PPL 69.37"（早前全量运行），差 ~1.7%。
对照：默认 BASE 配置已被验证为跨进程逐位一致（见 outputs/repro_check.py），
     因此嫌疑集中在 big_ltm=True 的 SparseSynapseTable 路径。

本脚本用**更小的预算**分别跑 base / big 两套配置，打印精确 PPL，
连续跑两次即可判断哪一条路径不稳定。
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phdnet.config import PHDNetConfig          # noqa: E402
from phdnet.word_lm import PHDWordLM            # noqa: E402

DOC = ROOT / "datasets" / "eval" / "internal_corpus.txt"
SEG = dict(max_len=6, min_count=5, min_entropy=1.0)
BASE = dict(n_sdr=256, k_sparse=32, n_mid=256, n_top=256,
            eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
            pred_in_readout=True)
BIG = dict(n_sdr=256, k_sparse=32, n_mid=512, n_top=1024,
           eta_pc=0.0, eta_oja=0.0, eta_stdp=0.02, seed=11,
           pred_in_readout=True, big_ltm=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000


def one(name: str, cfg: PHDNetConfig, text: str, tr: str, ev: str) -> None:
    lm = PHDWordLM(text, cfg, seg_kwargs=SEG)
    t0 = time.perf_counter()
    lm.train_stream(tr)
    dt = time.perf_counter() - t0
    m = lm.evaluate(ev)
    extra = ""
    if cfg.big_ltm:
        extra = " 表生长 %d" % lm.net.ltm.table.stats()["grown"]
    print(f"  {name:<26} ppl_char={m['ppl_char']:.6f} bpc={m['bpc']:.6f} "
          f"n={m.get('n_tok')} {dt:.1f}s{extra}")


def main() -> None:
    full = DOC.read_text(encoding="utf-8")
    print(f"预算 {N} 字符 / 全文 {len(full)} 字符  hash_rand={sys.flags.hash_randomization}")
    # 保持 80/20 划分，但只取前 N 字符，使脚本能在 1 分钟内跑完
    text = full[:N]
    split = int(len(text) * 0.8)
    tr, ev = text[:split], text[split:]
    one("稠密 LTM base", PHDNetConfig(**BASE), text, tr, ev)
    one("大容量 LTM big", PHDNetConfig(**BIG), text, tr, ev)


if __name__ == "__main__":
    main()
