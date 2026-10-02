# FLUX.2 `model.py`, explained line by line

A single-page, self-contained HTML tutorial that walks through every class, every function and
every line of [`src/flux2/model.py`](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py)
from [black-forest-labs/flux2](https://github.com/black-forest-labs/flux2) — the 833-line transformer
at the heart of the FLUX.2 image generators (32 B / 9 B / 4 B parameters).

**➡️ Read it here: [`index.html`](./index.html)** — open the file in any browser, no internet
connection, build step or dependency required. Everything (styles, diagrams, interactivity) is
inlined in that one file.

## What is inside

The article is ordered **bottom-up**: nothing is used before it has been explained.

| § | Contents |
|---|----------|
| 0–1 | A map of the file (every symbol with its line range, plus the whole import block) and what FLUX.2 actually is — including exactly where the 32 223 281 152 parameters go |
| 2–3 | Rectified flow in five minutes (with the real `denoise()` loop); how pixels become 128-channel tokens, where 15360 comes from, and what a reference image costs |
| 4 | Attention from zero: queries, keys, values, √d, softmax, heads — assumed knowledge nowhere else in the article |
| 5–6 | The four-axis rotary position encoding (interactive playground) and the three config dataclasses, with the two `__init__` guards |
| 7 | Every small primitive, in dependency order: `timestep_embedding`, `SiLUActivation`, `MLPEmbedder`, `rope`, `EmbedND`, `apply_rope`, `RMSNorm`/`QKNorm`, `SelfAttention`, `Modulation`, `LastLayer` |
| 8 | `causal_attn_fn` — the attention router, both branches, with the exact 9×9 isolation pattern drawn cell by cell |
| 9–10 | `SingleStreamBlock` and `DoubleStreamBlock`, each with a wiring diagram of every layer and tensor shape |
| 11–12 | `Flux2.__init__` and the three forward passes; the modulation-blending helpers |
| 13 | The KV cache: why it is *exact*, what it costs in GB, what it buys in speed |
| 14–15 | The denoising loop, what a step costs in FLOPs, and twelve design decisions explained |
| 16–17 | A complete cheat sheet of every symbol and constant, and references (SD3/MM-DiT, RoPE, SwiGLU, DiT, …) |

Every code listing is byte-exact from the source file with real line numbers, and every line is
followed by a plain-English trace. Fourteen hand-drawn SVG diagrams show how the modules connect and
what shape the data has between them.

### Claims that were verified, not assumed

Parameter counts, token counts, FLOPs, cache sizes, rotation orthogonality (`max|RRᵀ−I| = 5.96e-8`),
relative-position invariance of RoPE, the attention isolation pattern, and the bit-exactness of the KV
cache (`forward_kv_cached` with a stale cache ≡ `forward_kv_extract`, max |Δ| = 0.0) were all computed
directly from the repository's code. The scripts that did it are in [`src/ref/`](./src/ref).

Four further checks run on every build, because the article quotes real line numbers throughout:

| check | what it proves |
|-------|----------------|
| `ref/qa_trace.py` | no annotation quotes a line that carries no code, and no annotation names an identifier that does not exist in `model.py` (this is how a walkthrough that had drifted six lines from its own code was caught) |
| `ref/qa_coverage.py` | all 692 non-blank lines of `model.py` are shown in a listing or an annotation — 480 of them carry an annotation of their own |
| `ref/qa_linerefs.py` | every `L455`-style reference in the prose points at lines that contain what the sentence claims |
| `ref/verify4.py` | every quoted number is re-derived from the source with no dependencies at all, and then looked up in `index.html` — so a figure cannot stay in the page once it stops being true |

Statements about *why* the authors made a choice are a different kind of claim — the file carries
almost no comments — so those are badged **`inference`** in the page. Numbers and structure
never are.

## Repository layout

```
index.html          the article (the deliverable — open this)
src/                everything needed to rebuild and re-verify it
  parts/            the article authored as ordered HTML chunks
  build.py          concatenates parts/, expands code directives, runs the QA checks -> index.html
  codex.py          byte-exact, syntax-highlighted extraction of model.py line ranges
  gen_mask_fig.py   regenerates the attention-isolation figure (parts/08_mask_fig.html)
  ref/model.py      verbatim copy of the file being explained (line-number ground truth)
  ref/qa_layout.py  dependency-free SVG text-fit / collision checker
  ref/qa_trace.py   annotation-to-line alignment checker (runs on every build)
  ref/qa_coverage.py  "every line is shown" checker (runs on every build)
  ref/qa_linerefs.py  prose line-reference checker (runs on every build)
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
