#!/usr/bin/env python3
"""QA: every "L455" / "lines 496, 518" in the prose, checked against model.py.

The article quotes source line numbers in running text as well as in its
walkthroughs. Those went stale more than once, so this script lists every one of
them next to the line it points at, and flags what it can prove wrong:

  * a reference to a line that does not exist;
  * a row of the §16 symbol table whose line number does not hold the symbol;
  * a line of HTML that quotes one or more line numbers and also names a
    <code>identifier</code> from model.py — within NEAR characters of the
    reference, so unrelated clauses of a long table row are ignored — which
    appears nowhere near any of the quoted lines.
    "Near" means the quoted range +-1 line, or the def/class that encloses it —
    so `<td><code>causal_attn_fn</code><br>L810–811</td>` passes even though the
    function's name is 50 lines above the slice being cited.

Context is one line of the part file at a time: the parts are written with one
HTML element per line, so a line is a sentence, a table row or a caption, which is
exactly the granularity at which a line reference makes a claim.

Skipped: <!--T--> / <!--CODE--> directives (qa_trace.py and the build check those),
SVG figure text, and any line talking about a different upstream file
(sampling.py, autoencoder.py, …) whose line numbers are not model.py's.
ALLOWED below lists the handful of rows that claim a name is *absent*.

Everything else can be skimmed by a human with -v: a line number can be right
while the sentence around it is about something else, and no script can tell.

Usage:  python3 src/ref/qa_linerefs.py [-v] [parts_dir]
"""
import ast, io, pathlib, re, sys, tokenize

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "model.py"
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
PARTS = pathlib.Path(ARGS[0]) if ARGS else HERE.parent / "parts"

TEXT = SRC.read_text()
LINES = TEXT.split("\n")
N = len(LINES) - 1 if LINES and LINES[-1] == "" else len(LINES)

NAMES = set()
for tok in tokenize.generate_tokens(io.StringIO(TEXT).readline):
    if tok.type == tokenize.NAME:
        NAMES.add(tok.string)
    elif tok.type == tokenize.STRING:
        NAMES |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", tok.string))

# name -> every line it is defined on (a def/class name may repeat)
SCOPE_OF_LINE = [set() for _ in range(N + 2)]
tree = ast.parse(TEXT)
for node in ast.walk(tree):
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
            if 1 <= ln <= N:
                SCOPE_OF_LINE[ln].add(node.name)

REF_RE = re.compile(r"\bL(\d{2,3})(?:\s*[–-]\s*(?:L)?(\d{2,3}))?"
                    r"|\blines?\s+(\d{2,3})(?:\s*[–-]\s*(\d{2,3}))?")
CODE_RE = re.compile(r"<code>(.*?)</code>", re.S)
# the cheat-sheet table of §16: <td class="m">710</td><td><code>timestep_embedding</code></td>
ROW_RE = re.compile(r'<td class="m">(\d+)</td>\s*<td><code>(?:·\s*)?([A-Za-z_][\w.]*)</code>')
NEAR = 90          # characters of context on either side of a reference that count as "about" it
OTHER_FILE = ("sampling.py", "autoencoder.py", "text_encoder.py", "util.py", "docs/", "README")
SKIP = {"self", "None", "True", "False"}
# names too generic to be evidence about a particular line: the article uses them as
# layout shorthand (<code>[txt, ref, img]</code>) and as module qualifiers.
GENERIC = {"txt", "img", "ref", "vec", "pe", "cache", "torch", "nn", "split", "cat", "F"}
# (part file, first quoted line) -> why the identifier check does not apply
ALLOWED = {
    ("02_map_bigpicture.html", 375): "the row's whole point is that SelfAttention has NO forward()",
}


def symbol_rows(fatal):
    """Check the line-numbered symbol tables (the §16 cheat sheet)."""
    n = 0
    for part in sorted(PARTS.glob("[0-9][0-9]_*.html")):
        for lineno, line in enumerate(part.read_text().split("\n"), 1):
            for m in ROW_RE.finditer(line):
                ln, name = int(m.group(1)), m.group(2)
                n += 1
                leaf = name.rsplit(".", 1)[-1]
                if ln > N:
                    fatal.append(f"{part.name}:{lineno}: table says {name} is at L{ln}, "
                                 f"but model.py ends at L{N}")
                elif not re.search(rf"\b{re.escape(leaf)}\b", LINES[ln - 1]):
                    fatal.append(f"{part.name}:{lineno}: table says {name} is at L{ln}, but that "
                                 f"line is\n      L{ln}: {LINES[ln-1].strip()[:88]}")
    return n


def main():
    fatal, all_refs = [], []
    for part in sorted(PARTS.glob("[0-9][0-9]_*.html")):
        in_svg = False
        for lineno, line in enumerate(part.read_text().split("\n"), 1):
            if "<svg" in line:
                in_svg = True
            if in_svg:
                if "</svg>" in line:
                    in_svg = False
                continue
            if line.lstrip().startswith(("<!--T ", "<!--CODE ", "<!--RAW")):
                continue
            if any(f in line for f in OTHER_FILE):
                continue

            refs, near = [], []
            for m in REF_RE.finditer(line):
                lo = int(m.group(1) or m.group(3))
                hi = int(m.group(2) or m.group(4) or 0) or lo
                refs.append((lo, hi))
                # only <code> spans close to the reference make a claim about it
                near.append(line[max(0, m.start() - NEAR):m.end() + NEAR])
            if not refs:
                continue
            plain = " ".join(re.sub(r"<[^>]+>", "", line).split())

            bad = [f"L{lo}–{hi}" if lo != hi else f"L{lo}" for lo, hi in refs if max(lo, hi) > N]
            if bad:
                fatal.append(f"{part.name}:{lineno}: {', '.join(bad)} beyond the end of model.py "
                             f"({N} lines)\n      …{plain[:150]}…")
                continue

            window, scopes = [], set()
            for lo, hi in refs:
                window.append("\n".join(LINES[max(0, lo - 2):min(N, hi + 1)]))
                for ln in range(lo, min(hi, N) + 1):
                    scopes |= SCOPE_OF_LINE[ln]
            window = "\n".join(window)

            codes = set()
            for span in CODE_RE.findall(" ".join(near)):
                span = re.sub(r"<[^>]+>", "", span).replace("&gt;", ">").replace("&lt;", "<")
                codes |= {w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", span)} - SKIP - GENERIC
            named = sorted(w for w in codes if w in NAMES)
            missing = [w for w in named
                       if not re.search(rf"\b{re.escape(w)}\b", window)
                       and w not in scopes
                       and not any(w in s for s in scopes)]
            if missing and (part.name, refs[0][0]) not in ALLOWED:
                rng = ", ".join(f"L{lo}" if lo == hi else f"L{lo}–{hi}" for lo, hi in refs)
                fatal.append(f"{part.name}:{lineno}: {rng} quoted near {missing}, but those names "
                             f"occur neither on the quoted lines nor in their enclosing scope\n"
                             f"      L{refs[0][0]}: {LINES[refs[0][0]-1].strip()[:88]}\n"
                             f"      …{plain[:150]}…")
            for lo, hi in refs:
                all_refs.append((part.name, lineno, lo, hi, LINES[lo - 1].strip()[:66], plain[:96]))

    n_rows = symbol_rows(fatal)
    print(f"qa_linerefs: {len(all_refs)} prose line references, {n_rows} symbol-table rows "
          f"| {len(fatal)} suspicious")
    for f in fatal:
        print("   !!", f)
    if "-v" in sys.argv:
        for p, ln, lo, hi, src, ctx in all_refs:
            rng = f"L{lo}" if lo == hi else f"L{lo}-{hi}"
            print(f"   {p}:{ln:<5} {rng:<10} {src:<68} …{ctx}")
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
