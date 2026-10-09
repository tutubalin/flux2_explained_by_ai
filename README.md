# FLUX.2, explained line by line — `model.py` *and* `sampling.py`

Two single-page, self-contained HTML tutorials that walk through every class, every function and
every line of the two files that make [black-forest-labs/flux2](https://github.com/black-forest-labs/flux2)
generate images:

| article | the file it explains | size of the file | what it is |
|---------|---------------------|------------------|------------|
| **[`index.html`](./index.html)** | [`src/flux2/model.py`](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py) | 833 lines · 14 classes · 36 functions | the transformer at the heart of FLUX.2 (32 B / 9 B / 4 B parameters): the thing that, at every step, predicts *which direction to push the image* |
| **[`sampling.html`](./sampling.html)** | [`src/flux2/sampling.py`](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/sampling.py) | 442 lines · 21 functions · 0 classes | the sampler that drives it: the timestep shift, the empirically fitted schedule, the reference-image preparation, and the three denoising loops (including the KV-cache path) |

Open either file in any browser — no internet connection, build step or dependency required.
Everything (styles, diagrams, interactivity) is inlined in each one file, and the two articles
link to each other from their top bars.

## What is inside

Both articles run in **five parts**, and the first part of each contains no source code at all.
Every idea the file depends on is built from nothing before a single listing appears, so the later
parts read as recognition rather than discovery.

### `index.html` — model.py

| Part | § | Contents |
|------|---|----------|
| **1 · Pure theory** | 1 | Rectified flow in five minutes — the one equation, why the paths are straight, the walk drawn as a filmstrip, and why the steps are not evenly spaced |
| | 2 | FLUX.2's structure — the three models (text encoder, transformer, VAE) and which of them `model.py` is |
| | 3 | From prompt to tokens — a frozen 24 B language model, three depths tapped and glued, and where 15 360 comes from |
| | 4 | From pixels to tokens — the VAE, 8× + 2× compression, and why one token is 128 numbers covering 16 × 16 pixels |
| | 5 | One language — how 128 and 15 360 both become 6 144, and why the two doors are plain matrices |
| | 6 | Attention, from zero — queries, keys, values, √d, softmax, heads |
| | 7 | The frequency trick — how one float becomes a 256-number vector, and why a cosine *and* a sine |
| | 8 | Position — the four-axis RoPE, and how 64 rotations fill one head's 128 numbers (interactive playground) |
| | 9 | Time and guidance — what the two settings are, what classifier-free guidance is, and how they become one vector |
| | 10 | SiLUActivation — the soft switch, and its two jobs (activation, SwiGLU valve) |
| | 11 | Modulation — LayerNorm from its formula, the three dials, why a gate, and what sharing them costs |
| | 12 | The KV cache — what is worth caching, what exactly is stored, and the three conditions that make it *exact* |
| **2 · Code overview** | 13–15 | A map of all 833 lines (every symbol with its line range), where the 32 223 281 152 parameters go, and the three configurations with their two `__init__` guards |
| **3 · Small parts** | 16 | The eleven primitives in dependency order: `timestep_embedding`, `SiLUActivation`, `MLPEmbedder`, `rope`, `EmbedND`, `apply_rope`, `RMSNorm`/`QKNorm`, `SelfAttention`, `Modulation`, `LastLayer` |
| **4 · Big pieces** | 17–22 | `causal_attn_fn` (the router, with the 9×9 isolation pattern drawn cell by cell), `SingleStreamBlock`, `DoubleStreamBlock`, `Flux2` and its three forwards, the modulation blends, the KV cache paths |
| **5 · Zooming out** | 23–26 | The denoising loop and what a step costs, twelve design decisions, a cheat sheet of every symbol and constant, and the references |

### `sampling.html` — sampling.py

| Part | § | Contents |
|------|---|----------|
| **1 · Pure theory** | 1 | What sits outside the file — the four callers, and the one job the sampler has |
| | 2 | The timestep shift — one expression, read as *add μ in logit space*, with the shift drawn as a curve family |
| | 3 | Euler integration — why `x + (t′−t)·v` is exact on a straight path, and what bfloat16 does to a ladder of times |
| | 4 | Addresses — the four-axis id grid (`t, h, w, l`), and why references live on planes 10, 20, 30 |
| | 5 | Reference preparation — the pixel budget chain, from a 2048² photo to 15 876 tokens, bar by bar |
| | 6–7 | Guidance distillation (why `denoise_cfg` exists at all) and the KV cache, from the sampler's side of the contract |
| **2 · Code overview** | 8 | A map of all 442 lines: the four families, who calls whom, and the one function nothing calls |
| **3 · The machinery** | 9 | The position machinery — `compress_time`, `scatter_ids` and the three addressers, drawn as a scatter |
| | 10–11 | The image-preparation chain line by line, and `vanilla_guidance`, proved unused with a grep |
| **4 · Schedule & loops** | 12–13 | `compute_empirical_mu` — eight fitted constants and a knee at 4 300 tokens — plus `get_schedule` and `encode_image_refs` |
| | 14–16 | The three loops: `denoise`, `denoise_cached` and `denoise_cfg`, each with its sequence layout drawn |
| **5 · Zooming out** | 17–20 | An end-to-end run of `scripts/cli.py` with real shapes, twelve design decisions, a cheat sheet of every symbol and constant, and the references |

Every code listing in both articles is byte-exact from the source file with real line numbers, and
every line is followed by a plain-English trace. **`index.html` draws 32 numbered figures** (26 of
them in Part 1, where there is no code to look at instead) across 27 listings and 169 traced lines;
**`sampling.html` draws 15** across 21 listings and 151 traced lines, covering all 348 non-blank
lines of the file.

Numbering in `index.html` is *derived*, not maintained: `python3 src/renumber.py` reads the order of
`src/parts/*.html` and rewrites every section, subsection, figure and cross-reference to match.
Inserting a chapter is writing one file and running that script. (`sampling.html` is numbered by
hand; `ref/qa_xrefs.py` polices its numbering instead.)

### Claims that were verified, not assumed

Parameter counts, token counts, FLOPs, cache sizes, rotation orthogonality (`max|RRᵀ−I| = 5.96e-8`),
relative-position invariance of RoPE, the attention isolation pattern, and the bit-exactness of the KV
cache (`forward_kv_cached` with a stale cache ≡ `forward_kv_extract`, max |Δ| = 0.0) were all computed
directly from the repository's code. On the sampling side: the logit-space identity behind the shift
(`logit(t′) = σ·logit(t) + μ`, exact to 1e-12), every quoted μ and schedule ladder, the bfloat16
collision counts of each ladder, the pixel budgets and crop arithmetic, and the token and cache-byte
totals of a real run. The scripts that did it are in [`src/ref/`](./src/ref).

The build also refuses to emit a page containing an unterminated `<!--T …` annotation — a missing
`-->` used to be swallowed silently, which cost the page seven explanations and hid everything after
them in the browser. Beyond that, six checks run on every build of *both* articles, because they
quote real line numbers throughout:

| check | what it proves |
|-------|----------------|
| `ref/qa_trace.py` | no annotation quotes a line that carries no code, and no annotation names an identifier that does not exist in the vendored sources (this is how a walkthrough that had drifted six lines from its own code was caught) |
| `ref/qa_coverage.py` | every non-blank line of the file is shown in a listing or an annotation — 480 of 692 for `model.py` carry an annotation of their own, 327 of 348 for `sampling.py` |
| `ref/qa_linerefs.py` | every place the prose cites a source line points at lines containing what the sentence claims, and every row of a cheat-sheet symbol table names a symbol that really lives on the line it gives |
| `ref/qa_xrefs.py` | every `§7.9`, every `FIG 5` and every contents entry resolves to something that exists, and the section and figure numbering have no gaps |
| `ref/qa_order.py` | no chapter tells the reader they have already met something a *later* chapter explains — the bottom-up promise, enforced mechanically ("which you met in §7" from §5 is a failure; "§7 builds it" is not) |
| `ref/verify4.py` / `ref/verify_sampling.py` | every quoted number is re-derived from the source with no dependencies at all, and then looked up in the built page — so a figure cannot stay in the page once it stops being true |

Statements about *why* the authors made a choice are a different kind of claim — neither file carries
many comments — so those are badged **`inference`** in the pages. Numbers and structure never are.

## Repository layout

```
index.html          article 1: model.py (the deliverable — open this)
sampling.html       article 2: sampling.py (the companion — open this)
src/                everything needed to rebuild and re-verify both
  shared/           style.html (all CSS, both heads) and foot.html (closing tags + JS)
  parts/            article 1 authored as ordered HTML chunks (00_head … 99_foot)
  parts_sampling/   article 2, same convention, its own §1–§20 numbering
  build.py          concatenates a parts dir, expands code directives, runs the QA checks
  codex.py          byte-exact, syntax-highlighted extraction of source line ranges
  renumber.py       re-derives article 1's numbering from the order of parts/
  gen_mask_fig.py   regenerates the attention-isolation figure (parts/52_mask_fig.html)
  ref/model.py      verbatim copy of the file being explained (line-number ground truth)
  ref/sampling.py   verbatim copy of the second file (line-number ground truth)
  ref/qa_layout.py  dependency-free SVG text-fit / collision checker
  ref/qa_trace.py   annotation-to-line alignment checker (runs on every build)
  ref/qa_coverage.py  "every line is shown" checker (runs on every build)
  ref/qa_linerefs.py  prose line-reference checker (runs on every build)
  ref/qa_xrefs.py   section / figure / contents cross-reference checker (runs on every build)
  ref/qa_order.py   bottom-up ordering checker (runs on every build)
  ref/qa_svg.py     renders every figure to PNG for eyeballing (needs `pip install resvg-py`)
  ref/verify1-3.py  the numerical verifications that run the real model (need `torch`, `einops`)
  ref/verify4.py    article 1's numbers in pure standard library, then greps index.html for them
  ref/verify_sampling.py  article 2's numbers the same way, then greps sampling.html
  ref/sampling_ref.py  schedule helpers used to draw the timestep curves
```

Rebuild with (the two environment variables select which article is built):

```bash
python3 src/build.py                                    # article 1 -> ./index.html
FLUX2_SRC=src/ref/sampling.py FLUX2_PARTS=src/parts_sampling \
    python3 src/build.py sampling.html                  # article 2 -> ./sampling.html
python3 src/ref/qa_layout.py src/parts/52_mask_fig.html # per-figure text-fit check, 0 problems
```

*Written by an AI agent, with every number checked against the code it describes.*
