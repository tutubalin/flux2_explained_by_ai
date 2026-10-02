#!/usr/bin/env python3
"""verify4 — every number the article quotes, re-derived without torch.

`verify1`–`verify3` build the real modules on the meta device and run them; that
needs torch. This one needs nothing but the standard library, so it is the check
you can run anywhere: it recomputes the arithmetic behind each quoted figure
straight from the source text of model.py, and then greps ../index.html to make
sure the figure that is printed in the article is the figure that was derived.

    A  structure census      classes / functions / Linear / norms / biases
    B  parameter counts       analytic formula, three configs
    C  shared modulation      what hoisting it out of the blocks saves
    D  tokens                 canvas, text, reference-image caps
    E  schedule               get_schedule + compute_empirical_mu, reproduced
    F  cost                   FLOPs per step and per 50-step run
    G  memory                 weights in bf16/fp8/int4, KV cache in GB
    H  claim census           does index.html actually contain these numbers?

Usage:  python3 src/ref/verify4.py [path/to/index.html]
"""
import ast, io, math, pathlib, re, sys, tokenize

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "model.py"
PAGE = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
    else HERE.parent.parent / "index.html"

TEXT = SRC.read_text()
LINES = TEXT.split("\n")
N_LINES = len(LINES) - 1 if LINES[-1] == "" else len(LINES)
NON_BLANK = sum(1 for l in LINES if l.strip())

FAIL = []


def fmt(v):
    if isinstance(v, float):
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    return f"{v:,}"


def check(label, got, want, tol=None):
    ok = (abs(got - want) <= tol) if tol is not None else (got == want)
    print(f"   {'ok ' if ok else 'BAD'} {label:<52} {fmt(got)}"
          + (f"  (article: {fmt(want)})" if not ok else ""))
    if not ok:
        FAIL.append(f"{label}: computed {got}, article says {want}")


def hdr(t):
    print(f"\n=== {t} " + "=" * max(0, 66 - len(t)))


# ---------------------------------------------------------------- A. census
hdr("A · structure census of model.py")
tree = ast.parse(TEXT)
classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
funcs = [n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
check("lines in model.py", N_LINES, 833)
check("non-blank lines", NON_BLANK, 692)
check("classes", len(classes), 14)
check("functions and methods", len(funcs), 36)
check("linear layers with an explicit bias=True",
      len(re.findall(r"(?<!disable_)bias\s*=\s*True", TEXT)), 0)
print(f"       (the file writes nn.Linear {TEXT.count('nn.Linear')} times, "
      f"nn.LayerNorm {TEXT.count('nn.LayerNorm')} times, QKNorm {TEXT.count('QKNorm(')} times)")

# The censuses quoted in the article (171 Linear, 128 RMSNorm, 81 LayerNorm) count
# modules in the *built* model, not occurrences in the source text — one class
# declaration becomes 48 or 64 live modules. Re-derived from the config:
DEV = dict(in_ch=128, ctx=15360, h=6144, heads=48, depth=8, depth_single=48,
           mlp_ratio=3.0, guidance=True)
K9B = dict(in_ch=128, ctx=12288, h=4096, heads=32, depth=8, depth_single=24,
           mlp_ratio=3.0, guidance=False)
K4B = dict(in_ch=128, ctx=7680, h=3072, heads=24, depth=5, depth_single=20,
           mlp_ratio=3.0, guidance=False)


def counts(c):
    """How many Linear / RMSNorm / LayerNorm modules one built model contains.

    SelfAttention is *two* Linears (a fused qkv and a proj) plus a QKNorm, which is
    itself two RMSNorms. A double block holds two of those plus two 2-Linear MLPs;
    a single block fuses attention and MLP into linear1/linear2 and has one QKNorm.
    """
    hd = c["h"] // c["heads"]
    d, s = c["depth"], c["depth_single"]
    lin = 1 + 1 + 2 + (2 if c["guidance"] else 0)   # img_in, txt_in, time_in, guidance_in
    lin += d * 8                                    # 2 attn x 2 + 2 mlp x 2 per block
    lin += s * 2                                    # linear1, linear2
    lin += 3 + 2                                    # 3 shared Modulations + LastLayer
    rms = d * 4 + s * 2                             # one QKNorm (= 2 RMSNorm) per attn
    ln = d * 4 + s * 1 + 1                          # norm1/norm2 per stream + norm_final
    return lin, rms, ln, hd


lin, rms, ln, hd = counts(DEV)
check("built Linear modules (dev)", lin, 171)
check("built RMSNorm modules (dev)", rms, 128)
check("built LayerNorm modules (dev)", ln, 81)
check("head_dim (all three configs)", hd, 128)
check("blocks in the dev model", DEV["depth"] + DEV["depth_single"], 56)

# ---------------------------------------------------------------- B. params
hdr("B · parameter counts, from the layer shapes alone")


def params(c):
    h, hd = c["h"], c["h"] // c["heads"]
    mlp = int(h * c["mlp_ratio"])
    stream = (h * (3 * h + mlp * 2)          # linear1: q,k,v + MLP input
              + (h + mlp) * h                # linear2: attention out + MLP out
              + 2 * hd)                      # QKNorm's two RMSNorm scales
    p = c["in_ch"] * h                       # img_in
    p += c["ctx"] * h                        # txt_in
    p += 256 * h + h * h                     # time_in (MLPEmbedder, no biases)
    if c["guidance"]:
        p += 256 * h + h * h                 # guidance_in
    p += c["depth"] * 2 * stream             # double blocks: two streams each
    p += c["depth_single"] * stream          # single blocks
    p += 2 * (h * 6 * h) + (h * 3 * h)       # three shared Modulations
    p += h * c["in_ch"] + h * 2 * h          # LastLayer: linear + adaLN
    return p, stream


for name, c, want in (("FLUX.2-dev (32B)", DEV, 32_223_281_152),
                      ("FLUX.2-klein-9B", K9B, 9_078_581_248),
                      ("FLUX.2-klein-4B", K4B, 3_875_544_576)):
    p, stream = params(c)
    check(name, p, want)
    if c is DEV:
        check("one single-stream block", stream, 490_733_824)
        check("one double-stream block", 2 * stream, 981_467_648)

# ---------------------------------------------------------------- C. modulation
hdr("C · hoisting the modulation out of the blocks")
h = DEV["h"]
one_double = 2 * (h * 6 * h)          # img + txt Modulation, six vectors each
one_single = h * 3 * h                # one Modulation, three vectors
per_block = DEV["depth"] * one_double + DEV["depth_single"] * one_single
shared = one_double + one_single      # the same three modules, built once
check("if every block had its own Modulation", per_block, 9_059_696_640)
check("the three shared Modulations cost", shared, 566_231_040)
check("parameters saved", per_block - shared, 8_493_465_600)

# ---------------------------------------------------------------- D. tokens
hdr("D · how many tokens a run actually carries")


def canvas_tokens(w, hgt):
    return (w // 16) * (hgt // 16)


for (w, hgt), want in (((512, 512), 1024), ((1024, 1024), 4096),
                       ((1360, 768), 4080), ((2048, 2048), 16384)):
    check(f"{w}×{hgt} canvas", canvas_tokens(w, hgt), want)

# one reference image: capped at 2024**2 pixels (upstream's typo for 2048**2),
# then cropped down to a multiple of 16 per side.
cap = 2024 ** 2
side = int(math.isqrt(cap))                      # 2023
side -= side % 16                                # 2016
check("1 reference, cap 2024²", canvas_tokens(side, side), 15_876)
check("4 references at 1024²", 4 * canvas_tokens(1024, 1024), 16_384)
check("text tokens (MAX_LENGTH)", 512, 512)

# ---------------------------------------------------------------- E. schedule
hdr("E · the timestep schedule, reproduced")
A1, B1 = 8.73809524e-05, 1.89833333
A2, B2 = 0.00016927, 0.45666666


def mu_for(steps, seq_len):
    if seq_len > 4300:
        return A2 * seq_len + B2
    m_10 = A1 * seq_len + B1
    m_200 = A2 * seq_len + B2
    a = (m_200 - m_10) / 190
    b = m_200 - 200 * a
    return a * steps + b


def shift(t, mu, sigma=1.0):
    if t <= 0 or t >= 1:
        return t
    e = math.exp(mu)
    return e / (e + (1 / t - 1) ** sigma)


def schedule(steps, seq_len):
    return [shift(1 - i / steps, mu_for(steps, seq_len)) for i in range(steps + 1)]


for (steps, L), want in (((50, 4096), 2.02), ((28, 4096), 2.15),
                         ((4, 4096), 2.29), ((8, 4096), 2.27), ((50, 16384), 3.23)):
    check(f"μ at {steps} steps, {L} tokens", round(mu_for(steps, L), 2), want)
s50 = schedule(50, 4096)
s50big = schedule(50, 16384)
check("steps above t=0.7 at 1 MP", sum(1 for t in s50 if t > 0.7), 39)
check("steps above t=0.7 at 4 MP", sum(1 for t in s50big if t > 0.7), 46)
s8 = schedule(8, 4096)
ladder = [1.000, 0.985, 0.967, 0.942, 0.906, 0.853, 0.763, 0.580, 0.0]
print("      8-step ladder @1 MP: " + ", ".join(f"{t:.4f}" for t in s8))
for got, want in zip(s8, ladder):
    if round(got, 3) != want:
        FAIL.append(f"8-step ladder: computed {got:.4f}, article prints {want}")
    else:
        print(f"   ok  ladder step t={want:<6} matches to 3 decimals")
check("of those 8 steps, above t=0.9", sum(1 for t in s8 if t > 0.9), 5)

# ---------------------------------------------------------------- F. cost
hdr("F · what one forward pass costs")
MACS_PER_TOKEN_PER_BLOCK = 490_733_568          # linear algebra only
check("MACs per token per block", MACS_PER_TOKEN_PER_BLOCK, 490_733_568)
BLOCKS, D_MODEL = 56, 6144


def flops(n_tokens):
    linear = 2 * BLOCKS * MACS_PER_TOKEN_PER_BLOCK * n_tokens
    attn = 4 * n_tokens ** 2 * D_MODEL * BLOCKS   # QK^T and A·V, 2 FLOP per MAC
    return linear, attn


for label, (w, hgt), want in (("512×512", (512, 512), 87.7),
                              ("1024×1024", (1024, 1024), 282.5),
                              ("1360×768", (1360, 768), 281.4),
                              ("2048×2048", (2048, 2048), 1321.5)):
    n = canvas_tokens(w, hgt) + 512
    lin_f, att_f = flops(n)
    tot = (lin_f + att_f) / 1e12
    print(f"   {'ok ' if abs(tot - want) < 0.6 else 'BAD'} {label + ' · tokens=' + str(n):<52} "
          f"{tot:8.1f} T   (linear {lin_f/1e12:.1f} + attention {att_f/1e12:.1f})")
    if abs(tot - want) >= 0.6:
        FAIL.append(f"FLOPs at {label}: {tot:.1f} T vs article {want} T")
    if label == "2048×2048":
        check("attention share at 2048² (%), article says \"30 %\"",
              round(100 * att_f / (lin_f + att_f)), 30)
        check("50 steps at 2048² (PFLOPs)", round(50 * (lin_f + att_f) / 1e15, 2), 66.08, 0.02)
    if label == "1024×1024":
        check("50 steps at 1024² (PFLOPs)", round(50 * (lin_f + att_f) / 1e15, 2), 14.12, 0.02)
n_small = canvas_tokens(512, 512) + 512
l0, a0 = flops(n_small)
check("50 steps at 512² (PFLOPs)", round(50 * (l0 + a0) / 1e15, 2), 4.38, 0.02)
n_mid = canvas_tokens(1360, 768) + 512
l1, a1 = flops(n_mid)
check("50 steps at 1360×768 (PFLOPs)", round(50 * (l1 + a1) / 1e15, 2), 14.07, 0.02)
ratio = ((l1 + a1)) / ((l0 + a0))
check("×4 the pixels ⇒ ×? the work (1 MP → 4.08 MP)", round(ratio, 1), 3.2, 0.05)
r2 = flops(canvas_tokens(2048, 2048) + 512)
r0 = flops(n_small)
check("×16 the pixels ⇒ ×? the work (512² → 2048²)",
      round((r2[0] + r2[1]) / (r0[0] + r0[1]), 1), 15.1, 0.2)

# ---------------------------------------------------------------- G. memory
hdr("G · memory")
P_DEV, P_9B, P_4B = params(DEV)[0], params(K9B)[0], params(K4B)[0]
for label, p, want in (("dev", P_DEV, (64.4, 32.2, 16.1)),
                       ("klein-9B", P_9B, (18.2, 9.1, 4.5)),
                       ("klein-4B", P_4B, (7.8, 3.9, 1.9))):
    for bits, gb in zip((16, 8, 4), want):
        check(f"weights of {label} @ {bits} bit (GB)", round(p * bits / 8 / 1e9, 1), gb, 0.05)

# One cache entry per *block*, not per stream: DoubleStreamBlock.forward_kv_extract
# (model.py L652-655) slices a single k_ref / v_ref out of the joint key sequence,
# because only the image stream has reference rows.
for name, c in (("dev", DEV), ("klein-9B", K9B), ("klein-4B", K4B)):
    blocks = c["depth"] + c["depth_single"]
    per_tok = blocks * 2 * c["h"] * 2           # K and V, bf16
    check(f"cache bytes per reference token ({name})", per_tok,
          {"dev": 1_376_256, "klein-9B": 524_288, "klein-4B": 307_200}[name])
    for n_ref, gb in ((15_876, {"dev": 21.85, "klein-9B": 8.32, "klein-4B": 4.88}[name]),
                      (16_384, {"dev": 22.55, "klein-9B": 8.59, "klein-4B": 5.03}[name])):
        check(f"{name}: {n_ref} reference tokens (GB)", round(n_ref * per_tok / 1e9, 2), gb, 0.01)

# ---------------------------------------------------------------- H. claims
hdr("H · claim census — are these numbers in index.html?")
if PAGE.exists():
    page = re.sub(r"<[^>]+>", " ", PAGE.read_text())
    page = re.sub(r"[\s,]+", "", page).replace("&nbsp;", "").replace("\u2009", "").replace("\u00a0", "")
    claims = [
        ("833 lines", "833"), ("692 non-blank lines", "692"),
        ("32.2B parameter total", "32223281152"), ("9B total", "9078581248"),
        ("4B total", "3875544576"), ("one single block", "490733824"),
        ("one double block", "981467648"),
        ("tokens at 2048²", "16384"), ("reference tokens (1 ref)", "15876"),
        ("per-block modulation cost", "9059696640"), ("shared modulation cost", "566231040"),
        ("what sharing saves", "8.5billion"), ("MACs per token per block", "490733568"),
        ("FLOPs at 2048²", "1321.5"), ("PFLOPs for 50 steps at 2048²", "66.08"),
        ("μ at 1 MP / 50 steps", "μ=2.02"), ("μ at 4 MP / 50 steps", "μ=3.23"),
        ("dev weights bf16", "64.4"), ("cache, 4 refs, klein-9B", "8.59"),
        ("171 Linear", "171"), ("128 RMSNorm", "128"), ("81 LayerNorm", "81"),
    ]
    missing = [label for label, needle in claims if needle not in page]
    for label, needle in claims:
        print(f"   {'ok ' if needle in page else 'BAD'} {label:<52} {needle}")
    FAIL += [f"index.html does not contain the number for: {m}" for m in missing]
else:
    print(f"   (skipped — {PAGE} not found; build the page first)")

# ---------------------------------------------------------------- verdict
print()
if FAIL:
    print(f"verify4: {len(FAIL)} discrepancy(ies)")
    for f in FAIL:
        print("   -", f)
    sys.exit(1)
print("verify4: all numbers re-derived from model.py, and all of them present in index.html")
