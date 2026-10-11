"""Report resolved paths for a gate and whether each candidate root works."""
from pathlib import Path

GATE = Path("D:/AiModel/train/tests/gates/gate_stream_tokenize.py").resolve()
print("gate file:", GATE)
for i in range(4):
    print(f"  parents[{i}] = {GATE.parents[i]}")

print()
print("repo root  =", GATE.parents[1])
print("  phdnet      :", (GATE.parents[1] / "phdnet" / "__init__.py").exists())
print("  train pkg   :", (GATE.parents[1] / "train" / "config_1b.py").exists())
print("  tools/bench :", (GATE.parents[1] / "tools" / "bench" / "bench_readout_sparse.py").exists())
print()
print("So `parents[1]` IS the repo root here (repo = D:\\AiModel\\train).")
print("`parents[2]` = D:\\AiModel  -> one level too high.")
