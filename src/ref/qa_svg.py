#!/usr/bin/env python3
"""QA helper: extract every <svg> from an HTML file and render it to PNG for eyeballing."""
import re, sys, pathlib
import resvg_py

_CAND = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"]
_FONTS = [f for f in _CAND if pathlib.Path(f).exists()]

CSS = """

.lbl{font-size:12.5px;fill:#b3c1d1}
.lbl-b{font-size:13px;fill:#e8eef6;font-weight:700}
.lbl-s{font-size:10.5px;fill:#7d8da1}
.lbl-m{font-size:10.5px;fill:#7d8da1}
.lbl-mb{font-size:11px;fill:#35d6f0;font-weight:600}
.ttl{font-size:14px;fill:#e8eef6;font-weight:800}
.mono{}
.ax{font-size:11px;fill:#8fa3b8}
.axb{font-size:11.5px;fill:#e8eef6;font-weight:700}
.big{font-size:17px;fill:#e8eef6;font-weight:800}
.eq{font-size:12px;fill:#ffd479}
"""

src = pathlib.Path(sys.argv[1]).read_text()
outdir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "qa"); outdir.mkdir(parents=True, exist_ok=True)
svgs = re.findall(r"<svg\b.*?</svg>", src, re.S)
print(f"found {len(svgs)} svg elements")
want = sys.argv[2] if len(sys.argv) > 2 else None
for i, s in enumerate(svgs):
    if want and want not in s:
        continue
    vb = re.search(r'viewBox="([\d.\-\s]+)"', s)
    zoom = 1.5
    kw = {}
    if vb:
        p = [float(x) for x in vb.group(1).split()]
        kw = dict(width=int(p[2]*zoom), height=int(p[3]*zoom))
    out = outdir / f"svg_{i:02d}.png"
    # resvg's CSS parser mangles unquoted multi-word families, so set them as attributes
    def _fam(m):
        tag = m.group(0)
        if "font-family=" in tag:
            return tag
        mono = bool(re.search(r'class="[^"]*(?:lbl-m|lbl-mb|eq|mono)[^"]*"', tag))
        fam = "DejaVu Sans Mono" if mono else "DejaVu Sans"
        return tag[:-1].rstrip() + f' font-family="{fam}">'
    s = re.sub(r"<text\b[^>]*>", _fam, s)
    idx = s.index(">") + 1
    s = s[:idx] + f"<style>{CSS}</style>" + s[idx:]
    try:
        png = resvg_py.svg_to_bytes(svg_string=s, background="#0c1119",
                            font_files=_FONTS or None,
                            sans_serif_family="DejaVu Sans", monospace_family="DejaVu Sans Mono", **kw)
        out.write_bytes(bytes(png))
        print(f"  [{i:02d}] -> {out}  {kw} ({len(png)//1024} KB)")
    except Exception as e:
        print(f"  [{i:02d}] FAILED: {e}")
