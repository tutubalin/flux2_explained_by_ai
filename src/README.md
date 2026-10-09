# Article sources & verification harness

Everything needed to rebuild `../index.html` and `../sampling.html` and to re-derive every number
quoted in them. Two articles, one harness: `FLUX2_SRC` says which vendored source is the line-number
ground truth and `FLUX2_PARTS` says which parts directory is the article; both default to article 1.

| file | role |
|------|------|
| `build.py` | concatenates `$FLUX2_PARTS/[0-9][0-9]_*.html` (default `parts/`) in order, expands the `<!--CODE …-->` / `<!--TRACE …-->` / `<!--RAW …-->` directives via `codex.py`, fixes an SVG/CSS fill-specificity gotcha, prints a sanity report, refuses to write a page containing an unterminated `<!--T …` directive (which would otherwise be swallowed silently and hide part of the page in the browser), writes the output page (`../index.html`, or the name given on the command line), then runs the content checks below — including the numeric gate registered for the source file in `NUMERIC_GATE` — and fails if any of them reports a problem |
| `codex.py` | byte-exact, syntax-highlighted extraction of line ranges from `$FLUX2_SRC` (default `ref/model.py`); also renders the line-by-line trace rows. The *code* beside an annotation can therefore never drift from the source — the *prose* is what `qa_trace.py` polices |
| `gen_mask_fig.py` | regenerates `parts/52_mask_fig.html` — the attention-isolation figure, drawn from the same rule the code implements |
| `shared/` | `style.html` — every design token and CSS rule, included by both heads — and `foot.html`, the closing `</main>` plus the progress-bar / scroll-spy / drawer JavaScript; article 2 reuses both verbatim |
| `parts/` | article 1, one file per chapter or band of chapters (`00_head` … `64_refs`). The file names decide the reading order, and the numbering follows from it — see `renumber.py` |
| `parts_sampling/` | article 2, same convention (`00_head` … `99_foot`), numbered by hand against its own §1–§20 space |
| `renumber.py` | derives every section, subsection and figure number from the order of `parts/*.html` (article 1 only), and rewrites all references to match, in a single pass so nothing cascades. Adding or moving a chapter is: put the file where it belongs, run this, run the build |
| `ref/sampling.py` | verbatim copy of `src/flux2/sampling.py` (git blob `1b581083d2bd603ef3eb79605ee5321fc36bff01`) — the line-number ground truth of article 2 |
| `ref/model.py` | verbatim copy of `src/flux2/model.py` (git blob `9b1f5b50f8a966d6f4b9ed33b6f125695dc707f0`) — the line-number ground truth for every listing |
| `ref/qa_layout.py` | dependency-free SVG text-fit / overflow / collision checker; the build is not done until it reports `0 layout problem(s)` |
| `ref/qa_trace.py` | annotation alignment: fails on an annotation whose quoted lines carry no code, or that names an identifier which occurs nowhere in the vendored sources; lists legitimate cross-references with `-v` |
| `ref/qa_coverage.py` | fails if any non-blank line of `model.py` is never shown in a listing or an annotation |
| `ref/qa_linerefs.py` | fails if a place where the prose cites a source line points at lines containing none of the identifiers that same sentence names, or if a row of the §17 cheat-sheet table gives a line that does not hold the symbol it names (`-v` lists every reference for skimming) |
| `ref/qa_order.py` | fails if a reference is phrased as prior knowledge ("you met in §7", "recall §22", "as §24 explained", "§9 above") but points at a *later* chapter — a forward pointer phrased as a pointer is fine |
| `ref/qa_xrefs.py` | fails if a `§8.9`, a bare `8.1`, a `§8.1–§8.10` range, a `FIG 5` or a contents entry points at something that does not exist; if the section or figure numbering has a gap; or if a subsection sits under the wrong parent section (this is the gate that makes a renumbering sweep safe) |
| `ref/qa_svg.py` | renders each `<svg>` to PNG (`resvg-py`, optional) so the figures can be eyeballed |
| `ref/verify1.py` | exact parameter census per config, per-block costs, tensor-shape trace through a miniature model, KV-cache bit-exactness, RoPE orthogonality, mask semantics (needs `torch`, `einops`) |
| `ref/verify2.py` | reference-token K/V invariance across timesteps, `forward` ≡ `forward_kv_extract` with zero references, `LastLayer` shapes |
| `ref/verify3.py` | the analytic numbers, cross-checked against the real meta-device model, including the measured module census (171 `Linear`, 128 `RMSNorm`, 81 parameter-free `LayerNorm`, 0 biases): parameter formulas, token counts, real `get_schedule` curves, FLOPs per step, weight and KV-cache memory, shared-vs-per-block modulation, steps above `t = 0.7` (needs `torch`) |
| `ref/verify_sampling.py` | article 2's numbers in pure standard library — the logit identity of the shift, every quoted μ and ladder, the bfloat16 collision counts, the pixel and crop arithmetic, the token and cache totals — then a claim census over `sampling.html`. Runs on every build of article 2 |
| `ref/verify4.py` | the same arithmetic again in pure standard library — structure census, parameter counts, schedule, FLOPs, memory — and a claim census that greps `index.html` for every number it re-derived, so a figure cannot survive in the page once it stops being true. Runs on every build |
| `ref/sampling_ref.py` | the schedule helpers, reproduced to draw the timestep curves of FIG 3 |

## Rebuild

```bash
python3 build.py                          # -> ../index.html, then the six content checks
FLUX2_SRC=ref/sampling.py FLUX2_PARTS=parts_sampling python3 build.py ../sampling.html
python3 ref/qa_layout.py parts_sampling/30_part2_map.html   # per figure file, 0 problems expected
python3 ref/qa_svg.py ../sampling.html out/  # optional, needs: pip install resvg-py
```

The verification scripts construct the real model on the `meta` device, so they need
`pip install torch einops` but download no weights. `qa_layout.py`, `qa_trace.py`,
`qa_coverage.py`, `qa_linerefs.py`, `qa_xrefs.py` and `qa_order.py` are pure standard library.
