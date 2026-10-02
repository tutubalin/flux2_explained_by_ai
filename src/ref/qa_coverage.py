#!/usr/bin/env python3
"""QA: the article claims to explain every line. Which lines does it actually show?

Collects every line range quoted by a <!--CODE a-b ...--> listing or a
<!--T a-b|...--> annotation across src/parts/*.html, and reports the non-blank
lines of model.py that no listing and no annotation ever covers.

Blank lines and lines that are only closing brackets are counted separately: a
listing that spans them shows them to the reader even though nobody annotates a
`)` on its own.

Usage:  python3 src/ref/qa_coverage.py [parts_dir]
"""
import pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "model.py"
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
PARTS = pathlib.Path(ARGS[0]) if ARGS else HERE.parent / "parts"

LINES = SRC.read_text().split("\n")
N = len(LINES) - 1 if LINES and LINES[-1] == "" else len(LINES)

CODE_RE = re.compile(r"<!--CODE (\d+)-(\d+)")
T_RE = re.compile(r"<!--T (\d+)(?:-(\d+))?\|")

shown, annotated = set(), set()
for part in sorted(PARTS.glob("[0-9][0-9]_*.html")):
    text = part.read_text()
    for a, b in CODE_RE.findall(text):
        shown |= set(range(int(a), int(b) + 1))
    for m in T_RE.finditer(text):
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        shown |= set(range(lo, hi + 1))
        annotated |= set(range(lo, hi + 1))

nonblank = [i for i in range(1, N + 1) if LINES[i - 1].strip()]
trivial = [i for i in nonblank
           if re.fullmatch(r"[\s\)\]\},:]+", LINES[i - 1]) or LINES[i - 1].strip() in ("else:", "):")]
missing = [i for i in nonblank if i not in shown]

print(f"qa_coverage: {N} lines in model.py, {len(nonblank)} non-blank")
print(f"   shown in a listing or annotation : {len([i for i in nonblank if i in shown])}/{len(nonblank)}"
      f"  ({100*len([i for i in nonblank if i in shown])/len(nonblank):.1f} %)")
print(f"   carrying their own annotation    : {len([i for i in nonblank if i in annotated])}/{len(nonblank)}")
print(f"   non-blank lines never shown      : {len(missing)}")
for i in missing:
    print(f"      L{i}: {LINES[i-1][:100]}")
out_of_range = sorted(x for x in shown if x < 1 or x > N)
if out_of_range:
    print(f"   !! quoted line numbers outside the file: {out_of_range}")
sys.exit(1 if (missing or out_of_range) else 0)
