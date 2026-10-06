#!/usr/bin/env python3
"""Derive every number in the article from the reading order of parts/.

The build concatenates `parts/[0-9][0-9]_*.html` in filename order, so the file
names *are* the structure. This script reads that order and rewrites the numbers to
match it, which is what makes reordering sections (or inserting a new one) a
one-command operation instead of a hundred hand edits:

  * `<span class="secno">§ N</span>`      → 1..n in document order
  * `<h3>N.M ·`                           → parent section, index within it
  * `<span class="figno">FIG N</span>`    → 1..n in document order
  * every reference to those: `§N`, `§N.M`, `FIG N`, `section N`, a bare `N.M` in a
    contents entry or a dependency table, and `§N.M–N.M` ranges.

References are numbers, so all the old→new maps are built first and then applied in
a single pass per file — nothing cascades. Run `qa_xrefs.py` afterwards; it is the
check that this script did its job.

Usage:  python3 src/renumber.py [--dry]
"""
import pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
PARTS = HERE / "parts"
DRY = "--dry" in sys.argv

SEC_DEF = re.compile(r'(<span class="secno">§\s*)(\d+)(</span>)')
SUB_DEF = re.compile(r'(<h3[^>]*>\s*)(\d+)\.(\d+)(\s*·)')
FIG_DEF = re.compile(r'(<span class="figno">FIG\s*)(\d+)(</span>)')
SEC_REF = re.compile(r'(§\s*)(\d+)(\.\d+)?')
FIG_REF = re.compile(r'\bFIG\s*(\d+)')
TOC_REF = re.compile(r'(<a href="#[\w-]+">\s*)(\d+)(\s*·)')
BARE_SUB = re.compile(r'(?<![\d.])(\d{1,2}\.\d{1,2})(?=\s*(?:·|<code>|</td>|</a>|\'s))')
WORD_SEC = re.compile(r'\bsection (\d+)\b')


def main():
    files = sorted(PARTS.glob("[0-9][0-9]_*.html"))
    texts = {f.name: f.read_text() for f in files}

    # ---- pass 1: definitions, in reading order.
    # Each occurrence gets its new label from its POSITION, because old labels are not
    # unique: inserting a subsection numbered like its neighbour (two "4.2"s) would
    # otherwise make both look up the same map entry and keep the same number.
    sec_map, sub_map, fig_map = {}, {}, {}     # old label -> new label, for references
    seq_of = {}                                # file -> list of (kind, new label) in order
    cur_sec_new = 0
    sub_seen = 0
    n_fig = 0
    for name in sorted(texts):
        t = texts[name]
        events = []
        for m in SEC_DEF.finditer(t):
            events.append((m.start(), "sec", m.group(2)))
        for m in SUB_DEF.finditer(t):
            events.append((m.start(), "sub", f"{m.group(2)}.{m.group(3)}"))
        for m in FIG_DEF.finditer(t):
            events.append((m.start(), "fig", m.group(2)))
        seq_of[name] = []
        for pos, kind, old in sorted(events):
            if kind == "sec":
                cur_sec_new += 1
                sub_seen = 0
                sec_map.setdefault(int(old), str(cur_sec_new))
                seq_of[name].append(("sec", str(cur_sec_new)))
            elif kind == "sub":
                sub_seen += 1
                new = f"{cur_sec_new}.{sub_seen}"
                sub_map.setdefault(old, new)
                seq_of[name].append(("sub", new))
            else:
                n_fig += 1
                fig_map.setdefault(int(old), str(n_fig))
                seq_of[name].append(("fig", str(n_fig)))

    if len(sec_map) != cur_sec_new:
        print(f"!! {len(sec_map)} distinct old section numbers but {cur_sec_new} sections found — "
              f"a duplicate or a gap; fix the source before renumbering")
        return 1

    def new_sec(n):
        return str(sec_map.get(int(n), int(n)))

    def new_sub(nm):
        return sub_map.get(nm, nm)

    # One pass, one match per position: definitions and references are alternatives in
    # a single master regex, so a number that has just been rewritten is never rescanned
    # and re-mapped (which is what happens if you substitute definitions first and then
    # run a "§N" reference pass over the result).
    MASTER = re.compile(
        r'(?P<secdef><span class="secno">§\s*)(?P<sd>\d+)(?P<sd2></span>)'
        r'|(?P<figdef><span class="figno">FIG\s*)(?P<fd>\d+)(?P<fd2></span>)'
        r'|(?P<h3><h3[^>]*>\s*)(?P<h3a>\d+)\.(?P<h3b>\d+)(?P<h3c>\s*·)'
        r'|(?P<toc><a href="#[\w-]+">\s*)(?P<tn>\d+)(?P<tc>\s*·)'
        r'|(?P<secref>§\s*)(?P<sr>\d+)(?P<srf>\.\d+)?'
        r'|(?P<figref>\bFIG\s*)(?P<fr>\d+)'
        r"|(?P<bare>(?<![\d.])\d{1,2}\.\d{1,2})(?=\s*(?:·|<code>|</td>|</a>|'s))"
        r'|(?P<word>\bsection )(?!\d+\s*(?:lines|line|of them|steps))(?P<wn>\d+)\b'
    )

    def rewrite(name, t):
        pending = {"sec": [], "sub": [], "fig": []}
        for kind, new in seq_of.get(name, []):
            pending[kind].append(new)

        def take(kind):
            return pending[kind].pop(0) if pending[kind] else None

        def sub(m):
            g = m.groupdict()
            if g["sd"] is not None:
                return f"{g['secdef']}{take('sec') or new_sec(g['sd'])}{g['sd2']}"
            if g["fd"] is not None:
                return f"{g['figdef']}{take('fig') or fig_map.get(int(g['fd']), g['fd'])}{g['fd2']}"
            if g["h3a"] is not None:
                return f"{g['h3']}{take('sub') or new_sub(g['h3a'] + '.' + g['h3b'])}{g['h3c']}"
            if g["tn"] is not None:
                return f"{g['toc']}{new_sec(g['tn'])}{g['tc']}"
            if g["sr"] is not None:
                if g["srf"]:
                    return f"{g['secref']}{new_sub(g['sr'] + g['srf'])}"
                return f"{g['secref']}{new_sec(g['sr'])}"
            if g["fr"] is not None:
                return f"{g['figref']}{fig_map.get(int(g['fr']), int(g['fr']))}"
            if g["bare"] is not None:
                return new_sub(g["bare"])
            return f"{g['word']}{new_sec(g['wn'])}"
        return MASTER.sub(sub, t)

    changed = 0
    for name in sorted(texts):
        new = rewrite(name, texts[name])
        if new != texts[name]:
            changed += 1
            if not DRY:
                (PARTS / name).write_text(new)

    # generators that emit a part carry numbers too, and must follow the same maps
    for gen in sorted(HERE.glob("gen_*.py")):
        src = gen.read_text()
        new = rewrite(gen.name, src)
        if new != src:
            changed += 1
            if not DRY:
                gen.write_text(new)
    print(f"renumber: {len(sec_map)} sections → §1–§{cur_sec_new}, "
          f"{len(sub_map)} subsections, {len(fig_map)} figures → FIG 1–{len(fig_map)}; "
          f"{changed} file(s) {'would change' if DRY else 'updated'}")
    if "-v" in sys.argv or DRY:
        print("   sections:", ", ".join(f"{k}→{v}" for k, v in sorted(sec_map.items())))
        print("   figures :", ", ".join(f"{k}→{v}" for k, v in sorted(fig_map.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
