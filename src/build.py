#!/usr/bin/env python3
"""Build index.html from parts/, fixing an SVG/CSS specificity gotcha on the way.

Gotcha: a CSS rule like `svg .lbl-b { fill: var(--text) }` overrides an element's
`fill="#f6a93b"` presentation attribute (author CSS beats presentation attributes).
So when a <text> element carries BOTH a fill-setting class and a fill attribute,
we promote the attribute to an inline style, which wins over the stylesheet.
"""
import os, re, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import codex

PARTS = pathlib.Path(os.environ.get("FLUX2_PARTS", str(HERE / "parts")))
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "index.html"

FILL_CLASSES = {"lbl", "lbl-b", "lbl-s", "lbl-m", "lbl-mb", "ttl", "mono", "ax", "axb", "big", "eq"}


def promote_fill(tag: str) -> str:
    """Rewrite one <text ...> start tag so an explicit fill wins over the CSS class default."""
    cm = re.search(r'class="([^"]*)"', tag)
    if not cm or not (set(cm.group(1).split()) & FILL_CLASSES):
        return tag

    fm = re.search(r'(?<![\w:-])fill="([^"]*)"', tag)
    if not fm:
        return tag
    val = fm.group(1)

    selfclose = tag.rstrip().endswith("/>")
    body = tag[: fm.start()] + tag[fm.end():]          # drop the fill attribute
    body = body.rstrip()
    body = body[:-2].rstrip() if selfclose else body[:-1].rstrip()

    sm = re.search(r'style="([^"]*)"', body)
    if sm:
        newstyle = f"fill:{val};" + sm.group(1)
        body = body[: sm.start(1)] + newstyle + body[sm.end(1):]
    else:
        body += f' style="fill:{val}"'
    return body + (" />" if selfclose else ">")


def fix_svg_fills(html: str) -> str:
    def fix_block(m):
        block = m.group(0)
        return re.sub(r"<text\b[^>]*>", lambda t: promote_fill(t.group(0)), block)
    return re.sub(r"<svg\b.*?</svg>", fix_block, html, flags=re.S)


parts = sorted(PARTS.glob("[0-9][0-9]_*.html"))
print(f"assembling {len(parts)} parts:")
buf = []
for p in parts:
    txt = p.read_text()
    print(f"   {p.name:<34} {len(txt):>8,} chars")
    buf.append(txt)
html = "\n".join(buf)
n_code = len(re.findall(r"<!--CODE ", html))
n_tr = len(re.findall(r"<!--T ", html))
html = codex.expand(html)
html = fix_svg_fills(html)

# sanity checks
n_svg = len(re.findall(r"<svg\b", html))
n_sec = len(re.findall(r"<section\b", html))
ids = re.findall(r'<(?:section|h2|h3)[^>]*id="([^"]+)"', html)
hrefs = set(re.findall(r'href="#([^"]+)"', html))
missing = sorted(h for h in hrefs if h not in set(ids))
print(f"\n   {len(html):,} chars total | {n_svg} SVG figures | {n_sec} sections")
print(f"   {n_code} exact code listings | {n_tr} traced source lines")
print(f"   anchors referenced but missing: {missing if missing else 'none'}")
dupes = [i for i in set(ids) if ids.count(i) > 1]
print(f"   duplicate ids: {dupes if dupes else 'none'}")
for t in ("section", "div", "figure", "table", "svg", "main", "aside"):
    o = len(re.findall(rf"<{t}\b", html)); c = len(re.findall(rf"</{t}>", html))
    if o != c:
        print(f"   !! <{t}> open={o} close={c} MISMATCH")

OUT.write_text(html)
print(f"\nwrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")

# ---------------------------------------------------------------- content QA
# Three static checks over the *parts* (not the built page), because the article's
# whole claim is that its quoted line numbers and annotations match the real file:
#   qa_trace     every annotation describes lines that contain what it talks about
#   qa_coverage  every non-blank line of model.py is shown somewhere
#   qa_linerefs  every "L455" in the prose points at a plausible line
#   verify4      re-derives every quoted number from model.py and greps the page for it
qa_fail = 0
sys.stdout.flush()          # keep the QA output in order when stdout is a pipe
QA = [("qa_trace.py", PARTS), ("qa_coverage.py", PARTS), ("qa_linerefs.py", PARTS),
      ("verify4.py", OUT)]        # verify4 also greps the page it just wrote
for script, arg in QA:
    print()
    qa_fail |= subprocess.run([sys.executable, str(HERE / "ref" / script), str(arg)]).returncode
if qa_fail:
    print("\n!! content QA reported problems — see above")
sys.exit(qa_fail)
