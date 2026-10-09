#!/usr/bin/env python3
"""QA: do the line-by-line annotations actually describe the lines they quote?

The build pipeline guarantees that the *code* shown beside an annotation is
byte-exact for the quoted line range. It cannot guarantee that the *prose* belongs
to those lines — that is what drifted in earlier revisions of this article, when a
whole walkthrough ended up explaining lines 2–6 away from the ones it quoted.

For every <!--T lo-hi|note--> row in src/parts/*.html this script runs three checks.

FATAL (exit status 1):
  A. fabricated name   a <code>identifier</code> in the note that occurs nowhere in
                       model.py and is not part of the torch/einops API. This is how
                       the annotations that talked about a non-existent `mod_tuple`
                       and `out2` were caught.
  B. empty target      the quoted lines contain no code at all once comments,
                       docstrings and punctuation are removed — an annotation can
                       never belong to a blank line, a lone `)` or an argument list
                       continuation, and rows like that were always off-by-N.

ADVISORY (printed, not fatal):
  C. cross-reference   a real model.py name mentioned in the note that does not
                       appear on the quoted lines. Most of these are legitimate
                       ("exactly as in Modulation", "consumed by apply_rope"), so
                       they are listed for a human to skim; notes that carry an
                       explicit "§n" or "Lnnn" marker are skipped entirely.

Usage:  python3 src/ref/qa_trace.py [parts_dir]
"""
import os, io, pathlib, re, sys, tokenize

HERE = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path(os.environ.get("FLUX2_SRC", str(HERE / "model.py")))
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
PARTS = pathlib.Path(ARGS[0]) if ARGS else HERE.parent / "parts"

TEXT = SRC.read_text()
LINES = TEXT.split("\n")

ROW_RE = re.compile(r"<!--T (\d+)(?:-(\d+))?\|(.*?)-->", re.S)
CODE_RE = re.compile(r"<code>(.*?)</code>", re.S)
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Names that are part of the surrounding API rather than of model.py itself.
EXTERNAL = {
    "state_dict", "numel", "parameters", "named_parameters", "modules", "to", "float", "chunk",
    "expand", "cat", "stack", "einsum", "arange", "cartesian_prod", "linspace", "full_like",
    "rsqrt", "mean", "reshape", "unsqueeze", "clone", "contiguous", "type_as", "is_floating_point",
    "eval", "matmul", "softmax", "ModuleList", "Sequential", "Linear", "LayerNorm", "Parameter",
    "GroupNorm", "BatchNorm2d", "functional", "device", "dtype", "shape", "ndim", "Tensor",
    "rearrange", "dataclass", "field", "prod", "log", "exp", "cos", "sin", "zeros_like",
    "squeeze", "torch", "np", "cuda", "item", "tolist", "append", "zip", "range", "sorted",
    "min", "max", "sum", "abs", "round", "int", "str", "list", "dict", "set", "bool",
}
# Prose words and single-letter maths symbols that happen to sit inside <code>.
SKIP = {
    "code", "None", "True", "False", "self", "cls", "def", "return", "import", "from", "model",
    "py", "html", "svg", "https", "arXiv", "fraction", "fractions", "out", "in", "of", "the",
    "num", "num_txt", "num_ref", "num_img", "seq", "len", "id", "ids", "hl", "nd", "dim", "eps",
    "theta", "omega", "mu", "sigma", "silu", "swish", "adaLN", "bf16", "fp8", "fp32", "int4",
    "float32", "bfloat16", "int64", "kv", "cache", "mlp", "txt", "img", "ref", "pe", "vec",
    "batch", "batched", "token", "tokens", "step", "steps", "block", "blocks", "head", "heads",
    "GPU", "GPUs", "ODE", "MLP", "MLPs", "QK", "RoPE", "SwiGLU", "GLU", "VAE", "SiLU",
    "_d", "_i", "_j", "n", "d", "i", "j", "k", "q", "v", "x", "t", "e", "B", "L", "D", "H", "K", "N",
}


def source_names(text):
    names = set()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.NAME:
            names.add(tok.string)
        elif tok.type == tokenize.STRING:
            # names that live inside einops patterns, dict keys and f-strings
            names |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", tok.string))
        elif tok.type == tokenize.COMMENT:
            # `noqa`, `F841`, TODO tags: quoted from the file's own comments
            names |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", tok.string))
    return names


# The articles explain a repository, not one file in isolation: sampling.py is only
# readable next to model.py and vice versa. A name is therefore "fabricated" only if
# it occurs nowhere in the vendored sources -- the checkers themselves excluded.
CORPUS = sorted(f for f in HERE.glob("*.py")
                if not f.name.startswith(("qa_", "verify", "gen_")))
NAMES = set().union(*(source_names(f.read_text()) for f in CORPUS))

# lines that are inside a triple-quoted string, by line number
DOCSTRING_LINES = set()
for tok in tokenize.generate_tokens(io.StringIO(TEXT).readline):
    if tok.type == tokenize.STRING and tok.string[:3] in ('"""', "'''"):
        DOCSTRING_LINES |= set(range(tok.start[0], tok.end[0] + 1))


def code_content(lo, hi):
    """The identifiers that the quoted lines actually execute (no comments/docstrings)."""
    out = []
    for ln in range(lo, min(hi, len(LINES)) + 1):
        if ln in DOCSTRING_LINES:
            continue
        line = LINES[ln - 1]
        line = re.sub(r"#.*$", "", line)
        out += [w for w in IDENT_RE.findall(line) if w not in ("self",)]
    return out


def identifiers(note):
    res, seen = [], set()
    for span in CODE_RE.findall(note):
        span = re.sub(r"<[^>]+>", "", span)
        span = span.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
        for w in IDENT_RE.finditer(span):
            w = w.group(0)
            if len(w) < 2 or w in SKIP or w in EXTERNAL or w in seen:
                continue
            seen.add(w)
            res.append(w)
    return res


def main():
    fatal, advice, rows = [], [], 0
    for part in sorted(PARTS.glob("[0-9][0-9]_*.html")):
        text = part.read_text()
        marks = len(re.findall(r"<!--T ", text))
        seen = 0
        for m in ROW_RE.finditer(text):
            lo = int(m.group(1))
            hi = int(m.group(2)) if m.group(2) else lo
            note = m.group(3)
            plain = re.sub(r"<[^>]+>", "", note)
            rows += 1
            seen += 1
            if "<!--" in note or "-->" in note:
                fatal.append(f"{part.name} L{lo}-{hi}: the annotation text contains a comment "
                             f"marker — this row swallowed the next directive (a '-->' is missing)")
            if lo > len(LINES) or hi > len(LINES):
                fatal.append(f"{part.name} L{lo}-{hi}: line numbers beyond end of file ({len(LINES)})")
                continue

            # --- B: does the quoted range contain any code at all?
            # A range that is nothing but a docstring is quoting prose on purpose, so it
            # is exempt; otherwise a single line must carry one identifier and a
            # multi-line quote at least two, or the annotation is off by some lines.
            content = code_content(lo, hi)
            all_doc = all(ln in DOCSTRING_LINES for ln in range(lo, hi + 1))
            need = 0 if all_doc else (1 if lo == hi else 2)
            if len(content) < need:
                quoted = " ⏎ ".join(LINES[l - 1] for l in range(lo, hi + 1)).strip()
                fatal.append(
                    f"{part.name} L{lo}-{hi}: quoted lines carry no code "
                    f"(identifiers: {content}) — annotation is off by some lines?\n"
                    f"        quoted: {quoted[:90]!r}\n        note:   {plain[:120]}")

            # --- A / C
            quoted_text = "\n".join(LINES[l - 1] for l in range(lo, hi + 1))
            marked = ("§" in note) or re.search(r"\bL\d{2,3}\b", note)
            for w in identifiers(note):
                if w not in NAMES:
                    fatal.append(
                        f"{part.name} L{lo}-{hi}: <code>{w}</code> occurs nowhere in the vendored sources "
                        f"and is not a torch/einops API name — fabricated?\n        note: {plain[:140]}")
                elif not re.search(rf"\b{re.escape(w)}\b", quoted_text) and not marked:
                    advice.append(f"{part.name} L{lo}-{hi}: <code>{w}</code> — cross-reference to elsewhere in the file")

        # A row whose "-->" is missing still matches ROW_RE — it just swallows the
        # next marker, so the annotation is silently lost at build time. Counting the
        # markers and the parsed rows is the only way to see that from here.
        if seen != marks:
            fatal.append(f"{part.name}: {marks} <!--T markers but only {seen} parse as rows — "
                         f"one is missing its closing '-->'")

    print(f"qa_trace: {rows} annotation rows | {len(fatal)} fatal | {len(advice)} advisory cross-references")
    for f in fatal:
        print("   !!", f)
    if "-v" in sys.argv:
        for a in advice:
            print("   ·", a)
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
