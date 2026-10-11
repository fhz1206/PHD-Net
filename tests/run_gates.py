"""Run every gate under tests/gates/ serially and report a summary.

Exit code is non-zero if any gate fails; the known pre-existing LTM trace
failure is reported as a failure, never suppressed.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "tests" / "gates"
OUT = ROOT / "outputs" / "test" / "reorg_gates"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
               OMP_NUM_THREADS="8", NUMBA_NUM_THREADS="8")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(GATES), str(ROOT / "tests" / "demos"),
         str(ROOT / "tests" / "evals")])
    records = []
    for path in sorted(GATES.glob("gate_*.py")):
        start = time.perf_counter()
        print(f"RUN {path.name}", flush=True)
        try:
            cp = subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env,
                                capture_output=True, timeout=900)
            out = (cp.stdout.decode("utf-8", "replace")
                   + "\n[stderr]\n" + cp.stderr.decode("utf-8", "replace"))
            code = cp.returncode
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode("utf-8", "replace") + "\nTIMEOUT"
            code = "timeout"
        (OUT / (path.stem + ".log")).write_text(out, encoding="utf-8")
        rec = {"gate": path.name, "exit_code": code,
               "seconds": round(time.perf_counter() - start, 1)}
        records.append(rec)
        print(json.dumps(rec), flush=True)
    failed = [r for r in records if r["exit_code"] != 0]
    (OUT / "summary.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"\nSUMMARY {len(records)-len(failed)}/{len(records)} passed")
    for f in failed:
        print(f"  FAIL {f['gate']} (exit {f['exit_code']})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())