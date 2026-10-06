# FLUX.2 `model.py`, explained line by line

A single-page, self-contained HTML tutorial that walks through every class, every function and
every line of [`src/flux2/model.py`](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py)
from [black-forest-labs/flux2](https://github.com/black-forest-labs/flux2) — the 833-line transformer
at the heart of the FLUX.2 image generators (32 B / 9 B / 4 B parameters).

**➡️ Read it here: [`index.html`](./index.html)** — open the file in any browser, no internet
connection, build step or dependency required. Everything (styles, diagrams, interactivity) is
inlined in that one file.

## What is inside

The article runs in **five parts**, and the first one contains no source code at all. Every idea the file
depends on is built from nothing before a single listing appears, so Parts 3 and 4 read as recognition
rather than discovery.

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

Every code listing is byte-exact from the source file with real line numbers, and every line is followed by
a plain-English trace. **Thirty-one hand-drawn SVG diagrams** show how the modules connect and what shape
the data has between them — twenty-three of them in Part 1, where there is no code to look at instead.

Numbering is *derived*, not maintained: `python3 src/renumber.py` reads the order of `src/parts/*.html` and
rewrites every section, subsection, figure and cross-reference to match. Inserting a chapter is writing one
file and running that script.

### Claims that were verified, not assumed

Parameter counts, token counts, FLOPs, cache sizes, rotation orthogonality (`max|RRᵀ−I| = 5.96e-8`),
relative-position invariance of RoPE, the attention isolation pattern, and the bit-exactness of the KV
cache (`forward_kv_cached` with a stale cache ≡ `forward_kv_extract`, max |Δ| = 0.0) were all computed
directly from the repository's code. The scripts that did it are in [`src/ref/`](./src/ref).

The build also refuses to emit a page containing an unterminated `<!--T …` annotation — a missing
`-->` used to be swallowed silently, which cost the page seven explanations and hid everything after them
in the browser. Beyond that, five checks run on every build, because the article quotes real line numbers
throughout:

| check | what it proves |
|-------|----------------|
| `ref/qa_trace.py` | no annotation quotes a line that carries no code, and no annotation names an identifier that does not exist in `model.py` (this is how a walkthrough that had drifted six lines from its own code was caught) |
| `ref/qa_coverage.py` | all 692 non-blank lines of `model.py` are shown in a listing or an annotation — 480 of them carry an annotation of their own |
| `ref/qa_linerefs.py` | every place the prose cites a source line points at lines containing what the sentence claims, and every row of the cheat-sheet symbol table names a symbol that really lives on the line it gives |
| `ref/qa_xrefs.py` | every `§7.9`, every `FIG 5` and every contents entry resolves to something that exists, and the section and figure numbering have no gaps |
| `ref/verify4.py` | every quoted number is re-derived from the source with no dependencies at all, and then looked up in `index.html` — so a figure cannot stay in the page once it stops being true |

Statements about *why* the authors made a choice are a different kind of claim — the file carries
almost no comments — so those are badged **`inference`** in the page. Numbers and structure
never are.

## Repository layout

```
index.html          the article (the deliverable — open this)
src/                everything needed to rebuild and re-verify it
  parts/            the article authored as ordered HTML chunks (00_head … 12_loop_design_refs)
  build.py          concatenates parts/, expands code directives, runs the QA checks -> index.html
  codex.py          byte-exact, syntax-highlighted extraction of model.py line ranges
  gen_mask_fig.py   regenerates the attention-isolation figure (parts/09_mask_fig.html)
  ref/model.py      verbatim copy of the file being explained (line-number ground truth)
  ref/qa_layout.py  dependency-free SVG text-fit / collision checker
  ref/qa_trace.py   annotation-to-line alignment checker (runs on every build)
  ref/qa_coverage.py  "every line is shown" checker (runs on every build)
  ref/qa_linerefs.py  prose line-reference checker (runs on every build)
  ref/qa_xrefs.py   section / figure / contents cross-reference checker (runs on every build)
  ref/qa_svg.py     renders every figure to PNG for eyeballing (needs `pip install resvg-py`)
  ref/verify1-3.py  the numerical verifications that run the real model (need `torch`, `einops`)
  ref/verify4.py    re-derives every quoted number with no dependencies, then greps index.html for it
  ref/sampling_ref.py  schedule helpers used to draw the timestep curves
```

Rebuild with:

```bash
python3 src/build.py                     # writes ./index.html, then runs the three QA checks
python3 src/ref/qa_layout.py index.html  # must report 0 layout problems
```

*Written by an AI agent, with every number checked against the code it describes.*
