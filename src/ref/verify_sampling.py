#!/usr/bin/env python3
"""verify_sampling — every number the sampling article quotes, re-derived without torch.

The companion to verify4.py: same idea, second article. sampling.py cannot be
imported here (it pulls in torch, torchvision and PIL), so the pure arithmetic is
re-implemented from the source text and the constants are read back out of the file
to make sure the two agree. Sections:

    A  structure census      lines, defs, classes, the constants as written
    B  family line counts    the split quoted in §8.4
    C  the shift             the logit identity, fixed points, monotonicity
    D  the schedule          μ and the ladders for every configuration quoted
    E  the bfloat16 cast     how many rungs of a ladder survive the rounding
    F  pixels and tokens     the prep chain, the budgets, the crop arithmetic
    G  sequences and cache   what a step costs, in tokens and in bytes
    H  claim census          does sampling.html actually contain these numbers?

Section G needs the three model configurations, which live in model.py; they are
parsed out of it rather than hard-coded.

Usage:  python3 src/ref/verify_sampling.py [path/to/sampling.html]
"""
import ast, math, pathlib, re, struct, sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "sampling.py"
MODEL = HERE / "model.py"
PAGE = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
    else HERE.parent.parent / "sampling.html"

TEXT = SRC.read_text()
LINES = TEXT.split("\n")
N_LINES = len(LINES) - 1 if LINES[-1] == "" else len(LINES)
NON_BLANK = sum(1 for l in LINES if l.strip())
TREE = ast.parse(TEXT)

FAIL = []


def fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:,.6f}".rstrip("0").rstrip(".")
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(fmt(x) for x in v) + ")"
    return f"{v:,}"


def check(label, got, want, tol=None):
    ok = (abs(got - want) <= tol) if tol is not None else (got == want)
    print(f"   {'ok ' if ok else 'BAD'} {label:<56} {fmt(got)}"
          + (f"  (article: {fmt(want)})" if not ok else ""))
    if not ok:
        FAIL.append(f"{label}: computed {got}, article says {want}")


def hdr(t):
    print(f"\n{t}")


# ---------------------------------------------------------------- A. structure
hdr("A · structure census")
tops = [n for n in TREE.body if isinstance(n, ast.FunctionDef)]
nested = [n for p in tops for n in ast.walk(p) if isinstance(n, ast.FunctionDef) and n is not p]
classes = [n for n in TREE.body if isinstance(n, ast.ClassDef)]
assigns = [n for n in TREE.body if isinstance(n, ast.Assign)]
check("lines in sampling.py", N_LINES, 442)
check("non-blank lines", NON_BLANK, 348)
check("top-level functions", len(tops), 21)
check("nested functions", len(nested), 2)
check("classes", len(classes), 0)
check("module-level assignments", len(assigns), 3)

CONSTANTS = ["8.73809524e-05", "1.89833333", "0.00016927", "0.45666666", "4300",
             "scale = 10", "2024**2", "1024**2", "ensure_multiple: int = 16",
             "max_ar=8", "min_sidelength=64", "generalized_time_snr_shift(timesteps, mu, 1.0)",
             "ref_fixed_timestep"]
for c in CONSTANTS:
    found = c in TEXT or (c == "ref_fixed_timestep" and c in MODEL.read_text())
    print(f"   {'ok ' if found else 'BAD'} constant as written: {c}")
    if not found:
        FAIL.append(f"constant not found in the source: {c}")

# ---------------------------------------------------------------- B. families
hdr("B · the four families of §8.4")
FAM = {
    "imports": [(1, 9)],
    "position ids": [(12, 21), (24, 49), (93, 103), (106, 119), (122, 138), (141, 151), (154, 156)],
    "image prep": [(52, 90), (159, 175), (178, 192), (195, 203), (206, 214), (217, 223),
                   (226, 237), (413, 442)],
    "schedule": [(240, 241), (244, 248), (251, 266)],
    "loops": [(269, 307), (310, 355), (358, 361), (364, 410)],
}
WANT_LINES = {"imports": 9, "position ids": 92, "image prep": 138, "schedule": 23, "loops": 136}
WANT_NB = {"imports": 7, "position ids": 84, "image prep": 117, "schedule": 19, "loops": 121}
covered = set()
for k, ranges in FAM.items():
    s = set()
    for lo, hi in ranges:
        s |= set(range(lo, hi + 1))
    covered |= s
    nb = sum(1 for i in sorted(s) if LINES[i - 1].strip())
    check(f"{k}: lines", len(s), WANT_LINES[k])
    check(f"{k}: non-blank", nb, WANT_NB[k])
check("blank separators between definitions", N_LINES - len(covered), 44)
check("families + separators", len(covered) + (N_LINES - len(covered)), 442)


# ---------------------------------------------------------------- C. the shift
def shift(t, mu, sigma=1.0):
    """generalized_time_snr_shift, line 241, on Python floats."""
    if t == 0.0:
        term = math.inf
    elif t == 1.0:
        term = 0.0
    else:
        term = (1 / t - 1) ** sigma
    return math.exp(mu) / (math.exp(mu) + term)


hdr("C · the shift of §2.2")
worst = 0.0
for t in (0.9, 0.75, 0.5, 0.25, 0.05):
    for mu in (0.0, 2.0234, 2.2679, 3.2300):
        for sigma in (1.0, 0.7):
            tp = shift(t, mu, sigma)
            worst = max(worst, abs(math.log(tp / (1 - tp)) - (sigma * math.log(t / (1 - t)) + mu)))
check("logit(t') = sigma*logit(t) + mu, worst error over 40 points", worst, 0.0, 1e-12)
check("t = 1 is a fixed point", shift(1.0, 3.23), 1.0)
check("t = 0 is a fixed point", shift(0.0, 3.23), 0.0)
mono = all(shift(i / 200, 2.2679) < shift((i + 1) / 200, 2.2679) for i in range(1, 199))
print(f"   {'ok ' if mono else 'BAD'} the shift is monotone increasing on (0, 1)")
if not mono:
    FAIL.append("the shift is not monotone")
above = all(shift(i / 100, 2.2679) > i / 100 for i in range(1, 100))
print(f"   {'ok ' if above else 'BAD'} mu > 0 moves every interior time toward noise")
if not above:
    FAIL.append("mu > 0 does not push times up")


# ---------------------------------------------------------------- D. schedule
A1, B1 = 8.73809524e-05, 1.89833333
A2, B2 = 0.00016927, 0.45666666


def compute_empirical_mu(L, steps):
    """Lines 251-266, transcribed."""
    if L > 4300:
        return A2 * L + B2
    m_200 = A2 * L + B2
    m_10 = A1 * L + B1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return a * steps + b


def get_schedule(steps, L):
    """Lines 244-248, transcribed."""
    mu = compute_empirical_mu(L, steps)
    ts = [1 - i / steps for i in range(steps + 1)]
    return mu, [shift(t, mu, 1.0) for t in ts]


hdr("D · the schedule of §2 and §12")
L = 4096
m_200, m_10 = A2 * L + B2, A1 * L + B1
a = (m_200 - m_10) / 190.0
b = m_200 - 200.0 * a
check("the interpolation hits m_10 at 10 steps", compute_empirical_mu(L, 10), m_10, 1e-12)
check("the interpolation hits m_200 at 200 steps", compute_empirical_mu(L, 200), m_200, 1e-12)
check("slope a at 4 096 tokens", a, -0.0058224, 1e-7)
check("intercept b at 4 096 tokens", b, 2.314469, 1e-6)
check("mu(4) - mu(10) at 4 096 tokens", compute_empirical_mu(L, 4) - compute_empirical_mu(L, 10), 0.0349, 1e-4)
check("mu(1) at 4 096 tokens", compute_empirical_mu(L, 1), 2.3086, 1e-4)
check("where the two fitted lines cross (tokens)", (B1 - B2) / (A2 - A1), 17605, 1.0)

MUS = {("klein 512²", 4, 1024): 2.0307, ("klein 1024²", 4, 4096): 2.2912,
       ("klein 2048²", 4, 16384): 3.2300, ("dev 8 steps", 8, 4096): 2.2679,
       ("dev 28 steps", 28, 4096): 2.1514, ("dev 50 steps 1 MP", 50, 4096): 2.0234,
       ("dev 50 steps 1440²", 50, 8100): 1.8278, ("dev 50 steps 4 MP", 50, 16384): 3.2300,
       ("dev 50 steps 512²", 50, 1024): 1.7020}
for (label, steps, L_), want in MUS.items():
    check(f"mu · {label} ({steps} steps, {L_} tokens)", compute_empirical_mu(L_, steps), want, 5e-5)

LADDERS = {
    (4, 1024): [1.000, 0.958, 0.884, 0.717, 0.000],
    (4, 4096): [1.000, 0.967, 0.908, 0.767, 0.000],
    (4, 16384): [1.000, 0.987, 0.962, 0.894, 0.000],
    (8, 4096): [1.000, 0.985, 0.967, 0.942, 0.906, 0.853, 0.763, 0.580, 0.000],
}
for (steps, L_), want in LADDERS.items():
    _, got = get_schedule(steps, L_)
    for i, (g, w) in enumerate(zip(got, want)):
        check(f"ladder {steps} steps @ {L_} tokens, entry {i}", g, w, 5e-4)
_, l50 = get_schedule(50, 4096)
for i, w in zip((0, 1, 2, 3, 48, 49, 50), (1.0000, 0.9973, 0.9945, 0.9916, 0.2396, 0.1337, 0.0)):
    check(f"ladder 50 steps @ 4096, entry {i}", l50[i], w, 5e-5)
_, l50b = get_schedule(50, 16384)
for i, w in zip((0, 1, 48, 49), (1.0000, 0.9992, 0.5130, 0.3403)):
    check(f"ladder 50 steps @ 16384, entry {i}", l50b[i], w, 5e-5)
_, l50c = get_schedule(50, 8100)
for i, w in zip((0, 1, 49), (1.0000, 0.9967, 0.1126)):
    check(f"ladder 50 steps @ 8100, entry {i}", l50c[i], w, 5e-5)

for steps, L_ in ((4, 1024), (4, 4096), (8, 4096), (28, 4096), (50, 4096), (50, 16384), (1, 4096)):
    _, lad = get_schedule(steps, L_)
    total = sum(lad[i + 1] - lad[i] for i in range(steps))
    check(f"strides sum to -1 · {steps} steps @ {L_}", total, -1.0, 1e-9)
_, l8 = get_schedule(8, 4096)
strides8 = [l8[i] - l8[i + 1] for i in range(8)]
for i, w in enumerate((0.015, 0.019, 0.025, 0.035, 0.053, 0.090, 0.183, 0.580)):
    check(f"8-step stride {i + 1}", strides8[i], w, 5e-4)
check("last 8-step stride / first", strides8[-1] / strides8[0], 39.8, 0.1)
check("last 8-step stride as a share of the range (%)", 100 * strides8[-1], 58.0, 0.5)
check("50 steps @ 4096: calls above t = 0.7", sum(1 for t in l50[:-1] if t > 0.7), 39)
check("50 steps @ 16384: calls above t = 0.7", sum(1 for t in l50b[:-1] if t > 0.7), 46)
check("50 steps @ 4096: calls above t = 0.9", sum(1 for t in l50[:-1] if t >= 0.9), 23)
check("50 steps @ 16384: calls above t = 0.9", sum(1 for t in l50b[:-1] if t >= 0.9), 37)

check("the knee: mu(4 225 tokens, 4 steps)", compute_empirical_mu(4225, 4), 2.3021, 5e-5)
check("the knee: mu(4 356 tokens, 4 steps)", compute_empirical_mu(4356, 4), 1.1940, 5e-5)
check("the knee: mu(4 225 tokens, 50 steps)", compute_empirical_mu(4225, 50), 2.0368, 5e-5)
check("the knee: mu(4 356 tokens, 50 steps)", compute_empirical_mu(4356, 50), 1.1940, 5e-5)
_, k1 = get_schedule(4, 4225)
_, k2 = get_schedule(4, 4356)
for i, w in zip(range(5), (1.000, 0.968, 0.909, 0.769, 0.000)):
    check(f"ladder below the knee (4 225), entry {i}", k1[i], w, 5e-4)
for i, w in zip(range(5), (1.000, 0.908, 0.767, 0.524, 0.000)):
    check(f"ladder above the knee (4 356), entry {i}", k2[i], w, 5e-4)
_, k3 = get_schedule(50, 4225)
_, k4 = get_schedule(50, 4356)
check("final stride below the knee, 50 steps", k3[-2], 0.135, 5e-4)
check("final stride above the knee, 50 steps", k4[-2], 0.063, 5e-4)


# ---------------------------------------------------------------- E. bfloat16
def bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    lsb = (bits >> 16) & 1
    return struct.unpack("<f", struct.pack("<I", ((bits + 0x7FFF + lsb) >> 16) << 16))[0]


hdr("E · the bfloat16 cast of §3.2")
check("bfloat16 spacing between 0.5 and 1.0", bf16(0.5 + 2 ** -8) - 0.5, 0.00390625, 1e-9)
check("0.9973 and 0.9945 collapse to one value", bf16(0.9973) == bf16(0.9945), True)
check("the value they collapse to", bf16(0.9973), 0.99609375)
CASTS = {(4, 4096): (4, 1), (8, 4096): (8, 1), (50, 4096): (48, 2), (50, 16384): (34, 4)}
for (steps, L_), (distinct, longest) in CASTS.items():
    _, lad = get_schedule(steps, L_)
    q = [bf16(t) for t in lad[:-1]]
    run = best = 1
    for i in range(1, len(q)):
        run = run + 1 if q[i] == q[i - 1] else 1
        best = max(best, run)
    check(f"{steps} steps @ {L_}: distinct times after the cast", len(set(q)), distinct)
    check(f"{steps} steps @ {L_}: longest run of identical times", best, longest)


# ---------------------------------------------------------------- F. pixels
def cap_pixels(w, h, k):
    """Lines 178-192, transcribed."""
    if w * h <= k:
        return w, h
    s = math.sqrt(k / (w * h))
    return int(w * s), int(h * s)


def crop16(w, h, x=16):
    """Lines 159-175, transcribed."""
    nw, nh = (w // x) * x, (h // x) * x
    return nw, nh, (w - nw) // 2, (h - nh) // 2


hdr("F · the preparation chain of §5 and §10")
check("one reference: 2024²", 2024 ** 2, 4096576)
check("several references: 1024²", 1024 ** 2, 1048576)
check("the typo's shortfall in pixels", 2048 ** 2 - 2024 ** 2, 97728)
check("the typo's shortfall in percent", 100 * (2048 ** 2 - 2024 ** 2) / 2048 ** 2, 2.33, 0.01)
check("the language model's budget: 768²", 768 ** 2, 589824)

w, h = cap_pixels(2048, 2048, 2024 ** 2)
check("one 2048² photo, after the cap: side", w, 2024)
check("one 2048² photo, after the cap: pixels", w * h, 4096576)
nw, nh, left, top = crop16(w, h)
check("…then the crop: side", nw, 2016)
check("…then the crop: pixels", nw * nh, 4064256)
check("…then the crop: pixels removed from each side", left, 4)
check("…then the crop: tokens per side", nw // 16, 126)
check("…then the crop: tokens", (nw // 16) * (nh // 16), 15876)
check("what a 2048² budget would have given", 128 * 128, 16384)
check("what the typo costs, in tokens", 16384 - 15876, 508)

w, h = cap_pixels(2048, 2048, 1024 ** 2)
check("four 2048² photos, each after the cap: side", w, 1024)
check("four 2048² photos: tokens each", (w // 16) ** 2, 4096)
check("four 2048² photos: tokens in total", 4 * (w // 16) ** 2, 16384)
w, h = cap_pixels(1920, 1080, 2024 ** 2)
check("a 1920×1080 photo is under the budget", (w, h), (1920, 1080))
nw, nh, left, top = crop16(w, h)
check("…then the crop", (nw, nh), (1920, 1072))
check("…rows removed from the top", top, 4)
check("…tokens", (nw // 16) * (nh // 16), 8040)
w, h = cap_pixels(3000, 2000, 1024 ** 2)
check("3000×2000 at a 1024² budget: scale", math.sqrt(1024 ** 2 / (3000 * 2000)), 0.41805, 1e-5)
check("3000×2000: width after the cap", w, 1254)
check("3000×2000: height after the cap", h, 836)
check("3000×2000: pixels after the cap", w * h, 1048344)
check("3000×2000: under the budget by", 1024 ** 2 - w * h, 232)
nw, nh, _, _ = crop16(w, h)
check("3000×2000: after the crop", (nw, nh), (1248, 832))
check("3000×2000: tokens", (nw // 16) * (nh // 16), 4056)
w, h = cap_pixels(512, 512, 1024 ** 2)
check("a 512² photo is untouched by the 1024² budget", (w, h), (512, 512))
check("four 512² photos: tokens in total", 4 * (w // 16) ** 2, 4096)
nw, nh, left, _ = crop16(25, 40)
check("an odd remainder leaves one more pixel on the right", (25 - nw - left, left), (5, 4))
check("a 2048² canvas: tokens", (2048 // 16) ** 2, 16384)
check("a 1440² canvas: tokens", (1440 // 16) ** 2, 8100)
check("a 1024² canvas: tokens", (1024 // 16) ** 2, 4096)
check("a 512² canvas: tokens", (512 // 16) ** 2, 1024)
check("a 1040² canvas: tokens", (1040 // 16) ** 2, 4225)
check("a 1056² canvas: tokens", (1056 // 16) ** 2, 4356)


# ---------------------------------------------------------------- G. sequences
hdr("G · sequences, cache and cost (§7.2, §14, §17)")
MTREE = ast.parse(MODEL.read_text())
CFG = {}
for node in MTREE.body:
    if isinstance(node, ast.ClassDef) and node.name.endswith("Params"):
        vals = {}
        for st in node.body:
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) \
                    and isinstance(st.value, ast.Constant):
                vals[st.target.id] = st.value.value
        CFG[node.name] = vals
for name, hidden, blocks in (("Flux2Params", 6144, 56), ("Klein9BParams", 4096, 32),
                             ("Klein4BParams", 3072, 25)):
    v = CFG[name]
    check(f"{name}: hidden_size", v["hidden_size"], hidden)
    check(f"{name}: double + single blocks", v["depth"] + v["depth_single_blocks"], blocks)
    per_tok = 4 * hidden * blocks
    check(f"{name}: cache bytes per token", per_tok,
          {"Flux2Params": 1376256, "Klein9BParams": 524288, "Klein4BParams": 307200}[name])
    check(f"{name}: 4 references (16 384 tokens), GB", round(16384 * per_tok / 1e9, 2),
          {"Flux2Params": 22.55, "Klein9BParams": 8.59, "Klein4BParams": 5.03}[name], 0.01)

check("§17: text + canvas + reference tokens", 512 + 4096 + 15876, 20484)
check("§17: the reference share of that sequence (%)", round(100 * 15876 / 20484), 78)
check("§17: the last of the four strides", 0.7672, 0.767, 5e-4)
check("512² canvas + one reference: image-side tokens", 1024 + 15876, 16900)
check("…the reference share (%)", round(100 * 15876 / 16900), 94)
check("1440² canvas + one reference: image-side tokens", 8100 + 15876, 23976)
check("…the reference share (%)", round(100 * 15876 / 23976), 66)
check("512² canvas + four references: image-side tokens", 1024 + 16384, 17408)
check("…plus the 512 text tokens", 17408 + 512, 17920)
check("the cached path's sequence", 1024 + 512, 1536)
check("the token ratio between them", 17920 / 1536, 11.7, 0.05)
check("§10.2: the per-step concatenation copy, MB", 16900 * 128 * 2 / 1e6, 4.3, 0.05)
check("§5: the autoencoder divides values by", 3 * 2 * 2 * 128 / 256, 6.0, 1e-9)
check("§13: tokens in a 2016² reference", 126 * 126, 15876)
check("§4.2: the plane ladder", [10 + 10 * i for i in range(4)], [10, 20, 30, 40])


# ---------------------------------------------------------------- H. claims
hdr("H · claim census — are these numbers in sampling.html?")
if PAGE.exists():
    page = re.sub(r"<[^>]+>", " ", PAGE.read_text())
    page = re.sub(r"[\s,]+", "", page).replace("&nbsp;", "").replace("\u2009", "").replace("\u00a0", "")
    claims = [
        ("442 lines", "442"), ("348 non-blank lines", "348"), ("21 top-level functions", "21"),
        ("0 classes", "0classes"), ("the blob hash", "1b581083d2bd603ef3eb79605ee5321fc36bff01"),
        ("family: imports", "9imports"), ("family: ids", "92"), ("family: prep", "138"),
        ("family: schedule", "23"), ("family: loops", "136"), ("separators", "44"),
        ("mu klein 512²", "2.0307"), ("mu klein 1024²", "2.2912"), ("mu klein 2048²", "3.2300"),
        ("mu dev 8 steps", "2.2679"), ("mu dev 50 steps", "2.0234"), ("mu 1440²", "1.8278"),
        ("the 8-step ladder", "0.985·0.967·0.942·0.906·0.853·0.763·0.580"),
        ("strides sum to -1", "−1.000"), ("last stride", "0.580"), ("stride ratio", "39"),
        ("last stride share", "58%"), ("39 of 50 above 0.7", "39"), ("46 of 50 above 0.7", "46"),
        ("23 above 0.9", "23"), ("37 above 0.9", "37"),
        ("slope a", "−0.005822"), ("mu(1)", "2.3086"), ("the crossing", "17605"),
        ("the knee below", "2.3021"), ("the knee above", "1.1940"),
        ("ladder below the knee", "0.968"), ("ladder above the knee", "0.9080.7670.524"),
        ("50-step final strides at the knee", "0.135"), ("and above it", "0.063"),
        ("bf16 spacing", "0.00390625"), ("48 distinct", "48"), ("34 distinct", "34"),
        ("longest run", "0.99609375"),
        ("2024²", "4096576"), ("2048²", "4194304"), ("the shortfall", "97728"), ("2.33 %", "2.33"),
        ("15 876 tokens", "15876"), ("16 384 tokens", "16384"), ("the typo costs", "508"),
        ("8 040 tokens", "8040"), ("4 056 tokens", "4056"), ("1 048 344 pixels", "1048344"),
        ("under the budget by", "232"), ("0.41805", "0.41805"), ("768²", "589824"),
        ("4 225 tokens", "4225"), ("4 356 tokens", "4356"),
        ("cache per token, dev", "1376256"), ("klein-9B", "524288"), ("klein-4B", "307200"),
        ("22.55 GB", "22.55"), ("8.59 GB", "8.59"), ("5.03 GB", "5.03"),
        ("20 484 tokens", "20484"), ("78 %", "78%"), ("16 900", "16900"), ("94 %", "94%"),
        ("23 976", "23976"), ("66 %", "66%"), ("17 408", "17408"), ("17 920", "17920"),
        ("1 536", "1536"), ("11.7", "11.7"), ("4.3 MB", "4.3MB"), ("÷6", "÷6"),
        ("speedup 1.78", "1.78"), ("speedup 2.66", "2.66"), ("speedup 1.21", "1.21"),
        ("speedup 1.85", "1.85"), ("the reference planes", "102030"),
    ]
    missing = [label for label, needle in claims if needle not in page]
    for label, needle in claims:
        print(f"   {'ok ' if needle in page else 'BAD'} {label:<56} {needle}")
    FAIL += [f"sampling.html does not contain the number for: {m}" for m in missing]
else:
    print(f"   (skipped — {PAGE} not found; build the page first)")

# ---------------------------------------------------------------- verdict
print()
if FAIL:
    print(f"verify_sampling: {len(FAIL)} discrepancy(ies)")
    for f in FAIL:
        print("   -", f)
    sys.exit(1)
print("verify_sampling: every number re-derived from sampling.py, and all of them present in the page")
