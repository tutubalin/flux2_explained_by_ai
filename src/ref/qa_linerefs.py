#!/usr/bin/env python3
"""QA: every "L455" / "lines 496, 518" in the prose, checked against model.py.

The article quotes source line numbers in running text as well as in its
walkthroughs. Those went stale more than once, so this script lists every one of
them next to the line it points at, and flags what it can prove wrong:

  * a reference to a line that does not exist;
  * a row of the §16 symbol table whose line number does not hold the symbol;
  * a line of HTML that quotes line numbers and also names a <code>identifier</code>
    from model.py which appears on none of them. "On" means the quoted lines +-1,
    widened to the innermost function body around them (a row that cites L115 means
    `forward`, and usually names something `forward` calls), or the name of an
    enclosing def/class. With one reference on the line every identifier is checked
    against it; with several — a design-table row — against their union, because
    attributing identifiers to individual numbers in one long row is guesswork.
    ALLOWED lists the rows that claim a name is deliberately *abs*ent.
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
import os, ast, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path(os.environ.get("FLUX2_SRC", str(HERE / "model.py")))
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
PARTS = pathlib.Path(ARGS[0]) if ARGS else HERE.parent / "parts"

TEXT = SRC.read_text()
LINES = TEXT.split("\n")
N = len(LINES) - 1 if LINES and LINES[-1] == "" else len(LINES)

tree = ast.parse(TEXT)

# Identifiers the file really uses. Deliberately NOT every word in every string:
# docstrings would contribute "and", "the", "gate", and then a sentence like
# "L132 and L206" would be accused of quoting a name that is not on those lines.
NAMES = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Name):
        NAMES.add(node.id)
    elif isinstance(node, ast.Attribute):
        NAMES.add(node.attr)
    elif isinstance(node, ast.arg):
        NAMES.add(node.arg)
    elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        NAMES.add(node.name)
    elif isinstance(node, ast.keyword) and node.arg:
        NAMES.add(node.arg)
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        if "->" in node.value:            # an einops rearrange pattern: its letters are axes
            NAMES |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", node.value))

# every line a def/class body covers, so that quoting a slice inside a function
# counts as naming that function; and the innermost function body around each line,
# because "L115" in a table row means "this function", not "these three lines"
SCOPE_OF_LINE = [set() for _ in range(N + 2)]
BODY_OF_LINE = [None] * (N + 2)
tree = ast.parse(TEXT)
for node in ast.walk(tree):
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        lo, hi = node.lineno, (node.end_lineno or node.lineno)
        for ln in range(lo, hi + 1):
            if 1 <= ln <= N:
                SCOPE_OF_LINE[ln].add(node.name)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cur = BODY_OF_LINE[ln]
                    if cur is None or (hi - lo) < (cur[1] - cur[0]):
                        BODY_OF_LINE[ln] = (lo, hi)

REF_RE = re.compile(r"\bL(\d{2,3})(?:\s*[–-]\s*(?:L)?(\d{2,3}))?"
                    r"|\blines?\s+(\d{2,3})(?:\s*[–-]\s*(\d{2,3}))?")
CODE_RE = re.compile(r"<code>(.*?)</code>", re.S)
# the cheat-sheet table of §16: <td class="m">710</td><td><code>timestep_embedding</code></td>
ROW_RE = re.compile(r'<td class="m">(\d+)</td>\s*<td><code>(?:·\s*)?([A-Za-z_][\w.]*)</code>')
# sibling files the article also quotes, whose line numbers are NOT the checked file's
OTHER_FILE = tuple(f for f in ("model.py", "sampling.py", "autoencoder.py", "text_encoder.py",
                                "util.py", "cli.py", "docs/", "README") if f != SRC.name)
SKIP = {"self", "None", "True", "False"}
# names too generic to be evidence about a particular line: the article uses them as
# layout shorthand (<code>[txt, ref, img]</code>) and as module qualifiers.
GENERIC = {"txt", "img", "ref", "vec", "pe", "cache", "torch", "nn", "split", "cat", "F"}
# (part file, first quoted line) -> why the identifier check does not apply
# Keyed by the first line number quoted, not by file name: the part files get renamed
# whenever the article is restructured, and these rows are about model.py, not about them.
ALLOWED = {} if SRC.name != "model.py" else {
    375: "the row's whole point is that SelfAttention has NO forward()",
    722: "the row's point is that LastLayer is NOT in the fp32 list",
    446: "the row's point is that scaled_dot_product_attention replaced the code "
         "these two lines no longer use",
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
                                 f"but {SRC.name} ends at L{N}")
                elif not re.search(rf"\b{re.escape(leaf)}\b", LINES[ln - 1]):
                    fatal.append(f"{part.name}:{lineno}: table says {name} is at L{ln}, but that "
                                 f"line is\n      L{ln}: {LINES[ln-1].strip()[:88]}")
    return n


def main():
    fatal, all_refs = [], []
    for part in sorted(PARTS.glob("[0-9][0-9]_*.html")):
        in_svg = False
        units, cur, cur_start = [], [], 0
        for lineno, raw_line in enumerate(part.read_text().split("\n"), 1):
            line = raw_line
            if "<svg" in raw_line:
                in_svg = True
            if in_svg:
                if "</svg>" in raw_line:
                    in_svg = False
                continue
            if line.lstrip().startswith(("<!--T ", "<!--CODE ", "<!--RAW")):
                continue
            if any(f in line for f in OTHER_FILE):
                continue
            # prose paragraphs wrap across source lines, and a sentence that cites two
            # line numbers often has one on each — so analyse whole paragraphs: a new
            # unit starts at a blank line or a line that opens with a tag
            stripped = line.strip()
            if not stripped or stripped.startswith("<"):
                units.append((cur_start, " ".join(cur)))
                cur, cur_start = [], lineno
            cur.append(stripped)

        units.append((cur_start, " ".join(cur)))
        for lineno, line in units:
            if not line.strip():
                continue
            marks = list(REF_RE.finditer(line))
            if not marks:
                continue
            plain = " ".join(re.sub(r"<[^>]+>", "", line).split())
            refs = []
            for m in marks:
                lo = int(m.group(1) or m.group(3))
                hi = int(m.group(2) or m.group(4) or 0) or lo
                refs.append((lo, hi))
                if hi > N:
                    fatal.append(f"{part.name}:{lineno}: L{lo}–{hi} is beyond the end of {SRC.name} "
                                 f"({N} lines)\n      …{plain[:150]}…")
                elif lo == hi and not re.search(r"[A-Za-z_]{2,}", LINES[lo - 1]):
                    fatal.append(f"{part.name}:{lineno}: cites L{lo} alone, but that line carries no "
                                 f"name to point at\n      L{lo}: {LINES[lo-1].strip()[:88]!r}")

            def window_of(lo, hi):
                """The quoted lines +-1, widened to the innermost function body around
                them: a table row that says 'L115' means 'forward', not 'these three
                lines', and the identifiers it names are usually called from inside it."""
                wlo, whi = max(0, lo - 2), min(N, hi + 1)
                scopes = set()
                for ln in range(lo, min(hi, N) + 1):
                    scopes |= SCOPE_OF_LINE[ln]
                    body = BODY_OF_LINE[ln]
                    if body:
                        wlo, whi = min(wlo, body[0] - 1), max(whi, min(N, body[1] + 1))
                return "\n".join(LINES[wlo:whi]), scopes

            def idents(fragment):
                out = set()
                for span in CODE_RE.findall(fragment):
                    span = re.sub(r"<[^>]+>", "", span).replace("&gt;", ">").replace("&lt;", "<")
                    out |= {w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", span)} - SKIP - GENERIC
                return sorted(w for w in out if w in NAMES)

            def accuse(rng, lo, missing):
                if lo in ALLOWED:
                    return
                fatal.append(f"{part.name}:{lineno}: {rng} quoted near {missing}, but those names "
                             f"occur neither on the quoted lines nor in their enclosing scope\n"
                             f"      L{lo}: {LINES[lo-1].strip()[:88]}\n"
                             f"      …{plain[:150]}…")

            if len(refs) == 1:
                # One reference on the line: every identifier on it is a claim about that
                # reference, so check each against that reference's own window.
                lo, hi = refs[0]
                if hi <= N:
                    window, scopes = window_of(lo, hi)
                    missing = [w for w in idents(line)
                               if not re.search(rf"\b{re.escape(w)}\b", window)
                               and not any(w in s for s in scopes)]
                    if missing:
                        accuse(f"L{lo}" if lo == hi else f"L{lo}–{hi}", lo, missing)
            else:
                # Several references on one line — a design-table row, typically. Which
                # identifier belongs to which number is guesswork (the row's trailing
                # clause lands in the last reference's segment no matter what it is about),
                # so only require that each identifier appears among the lines quoted.
                windows, scopes = [], set()
                for lo, hi in refs:
                    if hi <= N:
                        w, s = window_of(lo, hi)
                        windows.append(w)
                        scopes |= s
                blob = "\n".join(windows)
                missing = [w for w in idents(line)
                           if not re.search(rf"\b{re.escape(w)}\b", blob)
                           and not any(w in s for s in scopes)]
                if missing:
                    rng = ", ".join(f"L{lo}" if lo == hi else f"L{lo}–{hi}" for lo, hi in refs)
                    accuse(rng, refs[0][0], missing)

            for lo, hi in refs:
                if hi <= N:
                    all_refs.append((part.name, lineno, lo, hi,
                                     LINES[lo - 1].strip()[:66], plain[:96]))

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
