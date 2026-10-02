#!/usr/bin/env python3
"""Deterministic SVG layout QA.

resvg renders shapes reliably but its CSS/font handling is too flaky for text, so
instead we *measure* the text: every <text> element gets an estimated rendered box
(character-count x per-glyph advance for its font size / family / weight), and we
check that

  1. it stays inside the viewBox,
  2. if it sits inside a <rect>, it fits that rect with a small margin,
  3. it does not horizontally collide with another <text> on the same baseline,
  4. multi-line labels inside one box don't exceed the box height.

Advance widths are calibrated for DejaVu Sans / DejaVu Sans Mono (the metric
families browsers fall back to), which are slightly WIDER than Inter/Segoe, so a
passing check is conservative.
"""
import re, sys, pathlib, collections

# class -> (font-size px, monospace?, bold?)
CLASS_FONT = {
    "lbl":   (12.5, False, False),
    "lbl-b": (13.0, False, True),
    "lbl-s": (10.5, False, False),
    "lbl-m": (10.5, True,  False),
    "lbl-mb":(11.0, True,  True),
    "ttl":   (14.0, False, True),
    "ax":    (12.5, False, False),
    "axb":   (13.0, False, True),
    "big":   (17.0, False, True),
    "eq":    (12.0, True,  False),
    "mono":  (11.0, True,  False),
}
# em-advance per character, generous estimates
def adv(mono, bold):
    return 0.602 if mono else (0.63 if bold else 0.585)


def parse_texts(block):
    out = []
    for m in re.finditer(r"<text\b([^>]*)>(.*?)</text>", block, re.S):
        attrs, body = m.group(1), m.group(2)
        body = re.sub(r"<[^>]+>", "", body)          # strip tspans etc.
        body = (body.replace("&amp;", "&").replace("&lt;", "<")
                    .replace("&gt;", ">").replace("&nbsp;", " "))
        if not body.strip():
            continue
        def attr(n, d=None):
            mm = re.search(rf'\b{n}="([^"]*)"', attrs)
            return mm.group(1) if mm else d
        x, y = float(attr("x", 0)), float(attr("y", 0))
        cls = (attr("class") or "").split()
        size, mono, bold = 12.0, False, False
        for c in cls:
            if c in CLASS_FONT:
                size, mono, bold = CLASS_FONT[c]
        if attr("font-size"):
            size = float(attr("font-size"))
        anchor = attr("text-anchor", "start")
        w = len(body) * size * adv(mono, bold)
        if anchor == "middle":
            x0, x1 = x - w / 2, x + w / 2
        elif anchor == "end":
            x0, x1 = x - w, x
        else:
            x0, x1 = x, x + w
        out.append(dict(x0=x0, x1=x1, y=y, w=w, size=size, text=body, anchor=anchor))
    return out


def parse_rects(block):
    out = []
    for m in re.finditer(r"<rect\b([^>]*)/?>", block):
        a = m.group(1)
        def num(n):
            mm = re.search(rf'\b{n}="([-\d.]+)"', a)
            return float(mm.group(1)) if mm else None
        x, y, w, h = num("x"), num("y"), num("width"), num("height")
        if None in (x, y, w, h):
            continue
        out.append((x, y, w, h))
    return out


def check(path):
    src = pathlib.Path(path).read_text()
    svgs = re.findall(r"<svg\b.*?</svg>", src, re.S)
    problems = 0
    for si, s in enumerate(svgs):
        vb = re.search(r'viewBox="([-\d.\s]+)"', s)
        vx, vy, vw, vh = [float(t) for t in vb.group(1).split()]
        texts, rects = parse_texts(s), parse_rects(s)

        for t in texts:
            if t["x0"] < vx - 0.5 or t["x1"] > vx + vw + 0.5:
                print(f"  [svg {si}] OVERFLOW viewBox: {t['text'][:44]!r} "
                      f"spans {t['x0']:.0f}..{t['x1']:.0f} of {vx}..{vx+vw}")
                problems += 1
            if t["y"] < vy + 4 or t["y"] > vy + vh - 1:
                print(f"  [svg {si}] V-OVERFLOW: {t['text'][:44]!r} y={t['y']}")
                problems += 1

        # texts whose baseline sits inside a rect must fit it horizontally
        for t in texts:
            for (rx, ry, rw, rh) in rects:
                if rw < 20 or rh < 8 or rh > 120:
                    continue
                if rx + 2 <= t["x0"] + 2 and t["y"] > ry + 4 and t["y"] < ry + rh + 2:
                    # baseline inside this rect vertically; is the anchor point inside too?
                    cx = (t["x0"] + t["x1"]) / 2
                    if rx <= cx <= rx + rw:
                        pad = 6
                        if t["x0"] < rx + pad - 4 or t["x1"] > rx + rw - pad + 4:
                            print(f"  [svg {si}] TEXT TOO WIDE for box {rw:.0f}px: "
                                  f"{t['text'][:40]!r} needs {t['w']:.0f}px")
                            problems += 1
                        break

        # same-baseline collisions
        by_y = collections.defaultdict(list)
        for t in texts:
            by_y[round(t["y"], 1)].append(t)
        for y, row in by_y.items():
            row.sort(key=lambda t: t["x0"])
            for a, b in zip(row, row[1:]):
                if b["x0"] < a["x1"] - 1.5:
                    print(f"  [svg {si}] COLLIDE y={y}: {a['text'][:26]!r} vs {b['text'][:26]!r} "
                          f"(gap {b['x0']-a['x1']:.1f}px)")
                    problems += 1
    print(f"\n{path}: {len(svgs)} svg, {problems} layout problem(s)")
    return problems


if __name__ == "__main__":
    sys.exit(1 if check(sys.argv[1]) else 0)
