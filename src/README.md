# Article sources & verification harness

Everything needed to rebuild `../index.html` and to re-derive every number quoted in it.

| file | role |
|------|------|
| `build.py` | concatenates `parts/[0-9][0-9]_*.html` in order, expands the `<!--CODE …-->` / `<!--TRACE …-->` / `<!--RAW …-->` directives via `codex.py`, fixes an SVG/CSS fill-specificity gotcha, prints a sanity report, writes `../index.html` |
| `codex.py` | byte-exact, syntax-highlighted extraction of line ranges from `ref/model.py`; also renders the line-by-line trace tables |
| `gen_mask_fig.py` | regenerates `parts/07_mask_fig.html` — the attention-mask figure, drawn from the same mask the code builds |
| `parts/` | the article, authored as ordered HTML chunks (the build is a pure concatenation + directive expansion) |
| `ref/model.py` | verbatim copy of `src/flux2/model.py` — the line-number ground truth for every listing |
| `ref/qa_layout.py` | dependency-free SVG text-fit / overflow / collision checker; the build is not done until it reports `0 layout problem(s)` |
| `ref/qa_svg.py` | renders each `<svg>` to PNG (`resvg-py`, optional) so the figures can be eyeballed |
| `ref/verify1.py` | exact parameter census per config, tensor-shape trace, KV-cache bit-exactness, RoPE orthogonality & mask semantics (needs `torch`, `einops`) |
| `ref/verify2.py`, `ref/verify3.py` | further numerical checks quoted in the article (relative-position invariance, per-block parameter counts) |
| `ref/sampling_ref.py` | the schedule helpers, reproduced to draw the timestep curves of FIG 3 |

## Rebuild

```bash
python3 build.py                     # -> ../index.html
python3 ref/qa_layout.py ../index.html
python3 ref/qa_svg.py ../index.html out/   # optional, needs: pip install resvg-py
```

The verification scripts construct the real model on the `meta` device, so they need
`pip install torch einops` but download no weights.
