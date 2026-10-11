import re, sys
from pathlib import Path

GATES = Path(r"D:\AiModel\train\tests\gates")
files = sorted(GATES.glob("*.py"))
print(f"found {len(files)} files")

BASES = [
    ("Path(__file__)", re.compile(r"Path\(__file__\)(\s*\.resolve\(\)\s*)*\.parents\[(\d)\]")),
    ("_P(__file__)",   re.compile(r"_P\(__file__\)(\s*\.resolve\(\)\s*)*\.parents\[(\d)\]")),
    ("_P2(__file__)",  re.compile(r"_P2\(__file__\)(\s*\.resolve\(\)\s*)*\.parents\[(\d)\]")),
    ("pathlib.Path(__file__)", re.compile(r"pathlib\.Path\(__file__\)(\s*\.resolve\(\)\s*)*\.parents\[(\d)\]")),
]

reverted = 0
dirbased = 0
for path in files:
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    dirty = False
    for line in lines:
        new = line
        if re.search(r"_HERE\s*\.\s*parents\[", new):
            dirbased += 1
        for name, pat in BASES:
            m = pat.search(new)
            if m and m.group(2) == "1":
                new = pat.sub(lambda mm: mm.group(0).replace("[1]", "[2]"), new, count=1)
                dirty = True
                reverted += 1
                break
        out.append(new)
    if dirty:
        path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="")

print(f"reverted {reverted} __file__-based lines to parents[2]")
print(f"preserved {dirbased} dir-based (_HERE.parents[...]) lines")
