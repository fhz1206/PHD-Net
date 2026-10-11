"""Fix off-by-one root resolution in tests/gates/*.py left by the reorg.

All gates live at tests/gates/<name>.py, so the repository root is
parents[1], not parents[2]. parents[2] points one directory ABOVE the repo
(D:\\AiModel when the repo is D:\\AiModel\\train), which silently breaks every
`_ROOT / "train"` style lookup.

The rewrite is textual and idempotent.
"""

from __future__ import annotations

import re
from pathlib import Path

GATES = Path(__file__).resolve().parent / "tests" / "gates"

# `parents[2]` -> `parents[1]` on lines that resolve a root from __file__.
# Guard: only rewrite when the surrounding line mentions __file__/Path, so a
# legitimate unrelated parents[2] in a data path is not touched by accident.
PATTERNS = [
    re.compile(r"(?P<pre>Path\(__file__\)\s*(?:\.resolve\(\)\s*)*\.parents\[)2(?P<post>\])"),
    re.compile(r"(?P<pre>_P\(__file__\)\s*(?:\.resolve\(\)\s*)*\.parents\[)2(?P<post>\])"),
    re.compile(r"(?P<pre>_P2\(__file__\)\s*(?:\.resolve\(\)\s*)*\.parents\[)2(?P<post>\])"),
    re.compile(r"(?P<pre>pathlib\.Path\(__file__\)\s*(?:\.resolve\(\)\s*)*\.parents\[)2(?P<post>\])"),
    re.compile(r"(?P<pre>_HERE\.parents\[)2(?P<post>\])"),
]

changed: list[str] = []
for path in sorted(GATES.glob("*.py")):
    text = path.read_text(encoding="utf-8")
    new = text
    for pattern in PATTERNS:
        new = pattern.sub(lambda m: f"{m.group('pre')}1{m.group('post')}", new)
    if new != text:
        path.write_text(new, encoding="utf-8", newline="")
        changed.append(path.name)

print(f"rewrote {len(changed)} gates")
for name in changed:
    print(f"  {name}")

# Report what remains for manual review.
leftover = []
for path in sorted(GATES.glob("*.py")):
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "parents[2]" in line:
            leftover.append(f"{path.name}:{i}: {line.strip()}")
if leftover:
    print("\nstill parents[2] (each checked below):")
    for item in leftover:
        print("  " + item)
