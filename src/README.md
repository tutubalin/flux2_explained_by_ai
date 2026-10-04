# Article sources & verification harness

Everything needed to rebuild `../index.html` and to re-derive every number quoted in it.

| file | role |
|------|------|
| `build.py` | concatenates `parts/[0-9][0-9]_*.html` in order, expands the `<!--CODE …-->` / `<!--TRACE …-->` / `<!--RAW …-->` directives via `codex.py`, fixes an SVG/CSS fill-specificity gotcha, prints a sanity report, refuses to write a page containing an unterminated `<!--T …` directive (which would otherwise be swallowed silently and hide part of the page in the browser), writes `../index.html`, then runs the content checks below and fails if any of them reports a problem |
| `codex.py` | byte-exact, syntax-highlighted extraction of line ranges from `ref/model.py`; also renders the line-by-line trace rows. The *code* beside an annotation can therefore never drift from the source — the *prose* is what `qa_trace.py` polices |
| `gen_mask_fig.py` | regenerates `parts/09_mask_fig.html` — the attention-isolation figure, drawn from the same rule the code implements |
| `parts/` | the article, authored as ordered HTML chunks, one file per band of sections (`00_head` … `12_loop_design_refs`); the build is a pure concatenation + directive expansion, so the file names decide the reading order |
| `ref/model.py` | verbatim copy of `src/flux2/model.py` (git blob `9b1f5b50f8a966d6f4b9ed33b6f125695dc707f0`) — the line-number ground truth for every listing |
| `ref/qa_layout.py` | dependency-free SVG text-fit / overflow / collision checker; the build is not done until it reports `0 layout problem(s)` |
| `ref/qa_trace.py` | annotation alignment: fails on an annotation whose quoted lines carry no code, or that names an identifier which occurs nowhere in `model.py`; lists legitimate cross-references with `-v` |
| `ref/qa_coverage.py` | fails if any non-blank line of `model.py` is never shown in a listing or an annotation |
| `ref/qa_linerefs.py` | fails if a place where the prose cites a source line points at lines containing none of the identifiers that same sentence names, or if a row of the §17 cheat-sheet table gives a line that does not hold the symbol it names (`-v` lists every reference for skimming) |
| `ref/qa_xrefs.py` | fails if a `§8.9`, a bare `8.1`, a `§8.1–§8.10` range, a `FIG 5` or a contents entry points at something that does not exist; if the section or figure numbering has a gap; or if a subsection sits under the wrong parent section (this is the gate that makes a renumbering sweep safe) |
| `ref/qa_svg.py` | renders each `<svg>` to PNG (`resvg-py`, optional) so the figures can be eyeballed |
| `ref/verify1.py` | exact parameter census per config, per-block costs, tensor-shape trace through a miniature model, KV-cache bit-exactness, RoPE orthogonality, mask semantics (needs `torch`, `einops`) |
| `ref/verify2.py` | reference-token K/V invariance across timesteps, `forward` ≡ `forward_kv_extract` with zero references, `LastLayer` shapes |
| `ref/verify3.py` | the analytic numbers, cross-checked against the real meta-device model, including the measured module census (171 `Linear`, 128 `RMSNorm`, 81 parameter-free `LayerNorm`, 0 biases): parameter formulas, token counts, real `get_schedule` curves, FLOPs per step, weight and KV-cache memory, shared-vs-per-block modulation, steps above `t = 0.7` (needs `torch`) |
| `ref/verify4.py` | the same arithmetic again in pure standard library — structure census, parameter counts, schedule, FLOPs, memory — and a claim census that greps `index.html` for every number it re-derived, so a figure cannot survive in the page once it stops being true. Runs on every build |
| `ref/sampling_ref.py` | the schedule helpers, reproduced to draw the timestep curves of FIG 3 |

## Rebuild

```bash
python3 build.py                          # -> ../index.html, then the three content checks
python3 ref/qa_layout.py ../index.html
python3 ref/qa_svg.py ../index.html out/  # optional, needs: pip install resvg-py
```

The verification scripts construct the real model on the `meta` device, so they need
`pip install torch einops` but download no weights. `qa_layout.py`, `qa_trace.py`,
`qa_coverage.py`, `qa_linerefs.py` and `qa_xrefs.py` are pure standard library.
