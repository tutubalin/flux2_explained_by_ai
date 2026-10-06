#!/usr/bin/env python3
"""QA: every cross-reference in the article resolves to something that exists.

The article is full of "explained in §7.9", "see FIG 5", "the router of §8.2" — and
its sections have been renumbered more than once. Nothing else catches a reference
that survived a renumbering, so this checks four kinds of them:

  1. `§N` and `§N.M` in prose, table cells and figure text must name a section or
     subsection that the page actually has;
  2. `FIG N` must name a figure the page actually has, and the figure numbers must
     run 1..n with no gaps or duplicates (they are also the reading order);
  3. every contents-entry number must equal the number of the section it links to;
  4. section numbers themselves must run 0..n with no gaps or duplicates.

Usage:  python3 src/ref/qa_xrefs.py [-v] [parts_dir]
"""
import pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
PARTS = pathlib.Path(ARGS[0]) if ARGS else HERE.parent / "parts"

SEC_DEF = re.compile(r'<span class="secno">§\s*(\d+)</span>')
SUB_DEF = re.compile(r"<h3[^>]*>\s*(\d+\.\d+)\s*·")
FIG_DEF = re.compile(r'<span class="figno">FIG\s*(\d+)</span>')
TOC_ENTRY = re.compile(r'<a href="#([\w-]+)">\s*(\d+)\s*·')
SEC_ID = re.compile(r'<section[^>]*id="([\w-]+)"[^>]*>\s*<h2[^>]*>\s*<span class="secno">§\s*(\d+)')
SEC_REF = re.compile(r"§\s*(\d+(?:\.\d+)?)")
# a subsection number written without its §: "8.1 · timestep_embedding" in the contents,
# "8.3 MLPEmbedder" in a dependency table, "§8.1–8.10" in a range. Deliberately not
# every decimal in the page — the lookahead keeps 7.9 M, 1.1× and 21.85 GB out.
BARE_SUB = re.compile(r"(?<![\d.])(\d{1,2}\.\d{1,2})(?=\s*(?:·|<code>|</td>|</a>|'s))")
RANGE_REF = re.compile(r"§\s*(\d{1,2}\.\d{1,2})\s*[–-]\s*(?:§\s*)?(\d{1,2}\.\d{1,2})")
NUMCELL = re.compile(r'<t[dh] class="(?:m|num)"[^>]*>.*?</t[dh]>')
FIG_REF = re.compile(r"\bFIG\s*(\d+)")


def main():
    fatal, notes = [], []
    text = {}
    for part in sorted(PARTS.glob("[0-9][0-9]_*.html")):
        text[part.name] = part.read_text()
    whole = "\n".join(text.values())

    # ---- what exists
    secs = {}                                   # number -> file
    for name, t in text.items():
        for m in SEC_DEF.finditer(t):
            secs.setdefault(int(m.group(1)), []).append(name)
    subs = set()
    for name, t in text.items():
        subs |= {m.group(1) for m in SUB_DEF.finditer(t)}
    figs = {}
    for name, t in text.items():
        for m in FIG_DEF.finditer(t):
            figs.setdefault(int(m.group(1)), []).append(name)

    # ---- 4. section numbering must be dense
    if secs:
        expect = set(range(min(secs), max(secs) + 1))
        for gap in sorted(expect - set(secs)):
            fatal.append(f"no section is numbered §{gap} (numbering jumps {max(n for n in secs if n < gap)} → "
                         f"{min(n for n in secs if n > gap)})")
    for n, where in sorted(secs.items()):
        if len(where) > 1:
            fatal.append(f"§{n} is defined twice: {', '.join(where)}")

    # ---- 2. figure numbering must be dense
    if figs:
        expect = set(range(1, max(figs) + 1))
        for gap in sorted(expect - set(figs)):
            fatal.append(f"no figure is numbered FIG {gap}")
    for n, where in sorted(figs.items()):
        if len(where) > 1:
            fatal.append(f"FIG {n} is defined twice: {', '.join(where)}")

    # ---- 1. section / subsection references
    n_refs = 0
    for name, t in text.items():
        in_svg = False
        for lineno, raw in enumerate(t.split("\n"), 1):
            # inside a figure, only the visible text can carry a reference
            if "<svg" in raw:
                in_svg = True
            line = " ".join(re.findall(r">([^<>]+)<", raw)) if in_svg else raw
            if "</svg>" in raw:
                in_svg = False
            for m in SEC_REF.finditer(line):
                ref = m.group(1)
                n_refs += 1
                if "." in ref:
                    if ref not in subs:
                        fatal.append(f"{name}:{lineno}: refers to §{ref}, but no subsection "
                                     f"{ref} exists")
                elif int(ref) not in secs:
                    fatal.append(f"{name}:{lineno}: refers to §{ref}, but no section {ref} exists")
            for m in FIG_REF.finditer(line):
                if int(m.group(1)) not in figs:
                    fatal.append(f"{name}:{lineno}: refers to FIG {m.group(1)}, but no such "
                                 f"figure exists (the page has FIG 1–{max(figs) if figs else 0})")

    # ---- 2b. bare subsection numbers, and subsections under the wrong parent
    cur = None
    for name, t in text.items():
        in_svg = False
        for lineno, line in enumerate(t.split("\n"), 1):
            if "<svg" in line:
                in_svg = True
            if line.lstrip().startswith(("<!--T ", "<!--CODE ", "<!--RAW")):
                continue
            for m in SEC_DEF.finditer(line):
                cur = int(m.group(1))
            for m in re.finditer(r"<h3[^>]*>\s*(\d+)\.(\d+)", line):
                if cur is not None and int(m.group(1)) != cur:
                    fatal.append(f"{name}:{lineno}: subsection {m.group(1)}.{m.group(2)} sits inside "
                                 f"the section numbered §{cur}")
            if in_svg:
                if "</svg>" in line:
                    in_svg = False
                continue
            clean = NUMCELL.sub(" ", line)
            plainctx = re.sub(r"<[^>]+>", "", line).strip()[:120]
            for m in BARE_SUB.finditer(clean):
                if m.group(1) not in subs:
                    fatal.append(f"{name}:{lineno}: '{m.group(1)}' reads as a subsection number, but "
                                 f"no subsection {m.group(1)} exists\n      …{plainctx}…")
            for m in RANGE_REF.finditer(clean):        # "§8.1–§8.10": both ends must exist
                for ref in m.groups():
                    if ref not in subs:
                        fatal.append(f"{name}:{lineno}: the range §{m.group(1)}–{m.group(2)} cites "
                                     f"§{ref}, which does not exist\n      …{plainctx}…")

    # ---- 3. contents entries must agree with the sections they link to
    id2sec = {m.group(1): int(m.group(2)) for m in SEC_ID.finditer(whole)}
    n_toc = 0
    for name, t in text.items():
        for m in TOC_ENTRY.finditer(t):
            anchor, num = m.group(1), int(m.group(2))
            n_toc += 1
            if anchor not in id2sec:
                fatal.append(f"{name}: contents entry '{num} · …' links to #{anchor}, "
                             f"which is not a section id")
            elif id2sec[anchor] != num:
                fatal.append(f"{name}: contents entry says §{num} but #{anchor} is §{id2sec[anchor]}")

    print(f"qa_xrefs: {len(secs)} sections (§{min(secs) if secs else 0}–§{max(secs) if secs else 0}), "
          f"{len(subs)} subsections, {len(figs)} figures, "
          f"{n_refs} §-references, {n_toc} contents entries | {len(fatal)} broken")
    for f in fatal:
        print("   !!", f)
    if "-v" in sys.argv:
        print("   sections:", ", ".join(str(n) for n in sorted(secs)))
        print("   figures :", ", ".join(str(n) for n in sorted(figs)))
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
