# FLUX.2 `model.py`, explained line by line

A single-page, self-contained HTML tutorial that walks through every class, every function and
every line of [`src/flux2/model.py`](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py)
from [black-forest-labs/flux2](https://github.com/black-forest-labs/flux2) — the 834-line transformer
at the heart of the FLUX.2 image generators (32 B / 9 B / 4 B parameters).

**➡️ Read it here: [`index.html`](./index.html)** — open the file in any browser, no internet
connection, build step or dependency required. Everything (styles, diagrams, interactivity) is
inlined in that one file.

## What is inside

| § | Contents |
|---|----------|
| 0–1 | A map of the file (every symbol with its line range) and what FLUX.2 actually is |
| 2–3 | Rectified flow in five minutes; how pixels become 128-channel tokens (with real schedule curves) |
| 4–5 | The four-axis rotary position encoding (interactive playground) and the three config dataclasses |
| 6 | Every small primitive: `timestep_embedding`, `MLPEmbedder`, `EmbedND`, `rope`, `apply_rope`, `RMSNorm`, `QKNorm`, `SiLUActivation`, `SelfAttention`, `Modulation`, `LastLayer` |
| 7 | `causal_attn_fn` — the masked attention, with the exact 9×9 mask drawn cell by cell |
| 8–9 | `SingleStreamBlock` and `DoubleStreamBlock`, each with a wiring diagram of every layer and tensor shape |
| 10–11 | The three `Flux2` forward passes and the modulation-blending helpers |
| 12 | The KV cache: why it is *exact*, what it costs in GB, what it buys in speed |
| 13–14 | The denoising loop, cost tables, and twelve design decisions explained |
| 15–16 | A complete cheat sheet of every symbol, and references (SD3/MM-DiT, RoPE, SwiGLU, DiT, …) |

Every code listing is byte-exact from the source file with real line numbers, and every line is
followed by a plain-English trace. Eleven hand-drawn SVG diagrams show how the modules connect and
what shape the data has between them.

### Claims that were verified, not assumed

Parameter counts, token counts, FLOPs, cache sizes, rotation orthogonality (`max|RRᵀ−I| = 5.96e-8`),
relative-position invariance of RoPE, the attention mask, and the bit-exactness of the KV cache
(`forward_kv_cached` with a stale cache ≡ `forward_kv_extract`, max |Δ| = 0.0) were all computed
directly from the repository's code. The scripts that did it are in [`src/ref/`](./src/ref).

## Repository layout

```
index.html          the article (the deliverable — open this)
src/                everything needed to rebuild and re-verify it
  parts/            the article authored as ordered HTML chunks
  build.py          concatenates parts/, expands code directives  ->  index.html
  codex.py          byte-exact, syntax-highlighted extraction of model.py line ranges
  gen_mask_fig.py   regenerates the attention-mask figure (parts/07_mask_fig.html)
  ref/model.py      verbatim copy of the file being explained (line-number ground truth)
  ref/qa_layout.py  dependency-free SVG text-fit / collision checker
  ref/qa_svg.py     renders every figure to PNG for eyeballing (needs `pip install resvg-py`)
  ref/verify*.py    the numerical verifications quoted in the article
  ref/sampling_ref.py  schedule helpers used to draw the timestep curves
```

Rebuild with:

```bash
python3 src/build.py            # writes ./index.html
python3 src/ref/qa_layout.py index.html   # must report 0 layout problems
```

*Written by an AI agent, with every number checked against the code it describes.*
