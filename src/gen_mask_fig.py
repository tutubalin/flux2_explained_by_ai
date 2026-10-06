#!/usr/bin/env python3
"""Generate parts/52_mask_fig.html — the attention-mask figure (verified mask, drawn as SVG)."""
import os, pathlib

HERE = pathlib.Path(__file__).resolve().parent

N_TXT, N_REF, N_IMG = 2, 3, 4
L = N_TXT + N_REF + N_IMG
CELL, GAP = 30, 2
COL = {"txt": "#35d6f0", "ref": "#b18cff", "img": "#f6a93b"}


def kind(i):
    return "txt" if i < N_TXT else ("ref" if i < N_TXT + N_REF else "img")


def tag(i):
    if i < N_TXT:
        return f"t{i}"
    if i < N_TXT + N_REF:
        return f"r{i - N_TXT}"
    return f"i{i - N_TXT - N_REF}"


def labels(x0, y0, rows):
    out = []
    for j in range(L):
        out.append(
            f'<text x="{x0 + j * (CELL + GAP) + CELL / 2:.0f}" y="{y0 - 8}" class="axb" '
            f'text-anchor="middle" fill="{COL[kind(j)]}">{tag(j)}</text>'
        )
    for i, rk in enumerate(rows):
        n = i if rk == "txt" else (i - N_TXT if rk == "ref" else sum(1 for r in rows[:i] if r == "img"))
        out.append(
            f'<text x="{x0 - 10}" y="{y0 + i * (CELL + GAP) + CELL / 2 + 4:.0f}" class="axb" '
            f'text-anchor="end" fill="{COL[rk]}">{rk[0]}{n}</text>'
        )
    return "".join(out)


def grid(x0, y0, rows, on_fn):
    out = []
    for i, rk in enumerate(rows):
        for j in range(L):
            on = on_fn(i, j)
            fill = COL[rk] if on else "#141c28"
            op = "0.9" if on else "1"
            out.append(
                f'<rect x="{x0 + j * (CELL + GAP)}" y="{y0 + i * (CELL + GAP)}" width="{CELL}" '
                f'height="{CELL}" rx="4" fill="{fill}" fill-opacity="{op}" stroke="#22303f" stroke-width="1"/>'
            )
    return "".join(out)


rows_full = [kind(i) for i in range(L)]
left = grid(70, 110, rows_full,
            lambda i, j: True if rows_full[i] != "ref" else (N_TXT <= j < N_TXT + N_REF))
rows_c = [r for r in rows_full if r != "ref"]
right = grid(560, 110, rows_c, lambda i, j: True)

svg = f'''<svg viewBox="0 0 1040 430" role="img" aria-label="Attention isolation pattern: text and image tokens attend to everything, reference tokens attend only to themselves; with the KV cache the reference rows simply vanish from the query side while their keys stay available.">
<text x="40" y="30" class="ttl">Who is allowed to look at whom</text>
<text x="70" y="66" class="axb">without cache · forward_kv_extract()</text>
<text x="70" y="84" class="ax">queries ↓ · keys →</text>
{labels(70, 110, rows_full)}{left}
<text x="560" y="66" class="axb">with cache · forward_kv_cached()</text>
<text x="560" y="84" class="ax">queries ↓ · keys → (violet keys come from the cache)</text>
{labels(560, 110, rows_c)}{right}
<rect x="70" y="402" width="14" height="14" rx="3" fill="#35d6f0"/><text x="92" y="414" class="ax">text query row</text>
<rect x="200" y="402" width="14" height="14" rx="3" fill="#b18cff"/><text x="222" y="414" class="ax">reference query row</text>
<rect x="360" y="402" width="14" height="14" rx="3" fill="#f6a93b"/><text x="382" y="414" class="ax">image query row</text>
<rect x="500" y="402" width="14" height="14" rx="3" fill="#141c28" stroke="#22303f"/><text x="522" y="414" class="ax">never attended</text>
<text x="660" y="414" class="ax">coloured cells = attended</text>
</svg>'''

fig = f'''<figure>
<div class="fig">
{svg}
</div>
<figcaption><span class="figno">FIG 25</span><b>The isolation rule, measured.</b> I fed one-hot values
through <code>causal_attn_fn</code> and read back which key each query actually mixed: on the left is what
came out. (No mask tensor exists anywhere in the code — this pattern is what the <em>slicing</em> of §16.2
produces, and it is drawn as a grid only because a grid is the easiest way to see it.) Text and image rows are completely dense — every token sees every token, in both directions. The
reference block is an isolated island: references read each other and nothing else. On the right, the
cached variant: the reference <em>rows</em> are gone (we no longer compute their outputs) but their
<em>columns</em> remain, supplied from the cache, so text and image still see them exactly as before.</figcaption>
</figure>'''

OUT = pathlib.Path(os.environ.get("FLUX2_MASKFIG", str(HERE / "parts" / "52_mask_fig.html")))
OUT.write_text(fig + "\n")
print("wrote", OUT, len(fig), "chars")
