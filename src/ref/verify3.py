"""Analytical parameter formulas + real-world cost numbers for the article."""
import sys, math
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import torch
import model as M
from sampling_ref import get_schedule, compute_empirical_mu

def h_int(x): return f"{x:,}"

for label, p in [("FLUX.2 [dev]", M.Flux2Params()), ("Klein 9B", M.Klein9BParams()), ("Klein 4B", M.Klein4BParams())]:
    h, nh, r = p.hidden_size, p.num_heads, p.mlp_ratio
    m = int(h * r)
    hd = h // nh
    single = 4*h*h + 3*h*m + 2*hd
    double = 2*(4*h*h + 3*h*m) + 4*hd
    emb = (p.in_channels*h) + (p.context_in_dim*h) + 2*((256*h)+(h*h)) \
          + 2*(6*h*h) + (3*h*h) + (h*p.in_channels + h*(2*h))
    if not p.use_guidance_embed:
        emb -= (256*h)+(h*h)
    total = p.depth*double + p.depth_single_blocks*single + emb
    with torch.device("meta"):
        actual = sum(math.prod(q.shape) for q in M.Flux2(p).parameters())
    print(f"{label}: formula={h_int(total)}  actual={h_int(actual)}  match={total==actual}")
    print(f"   double block = {h_int(double)} = 2x single ({h_int(single)}) + norms")
    print(f"   blocks total = {h_int(p.depth*double + p.depth_single_blocks*single)}")
    print(f"   modulation layers total = {h_int(2*(6*h*h) + 3*h*h)} "
          f"({100*(2*(6*h*h)+3*h*h)/total:.2f}% of model) -- SHARED by all {p.depth+p.depth_single_blocks} blocks")
    print()

print("=" * 78)
print("TOKEN COUNTS (VAE: 8x downsample, then 2x2 patchify => 16x total; 128 channels)")
print("=" * 78)
print(f"{'resolution':>14s} {'megapixels':>11s} {'latent grid':>13s} {'img tokens':>11s} "
      f"{'+512 txt':>10s} {'attn pairs (M)':>15s}")
for W, H in [(512,512),(768,768),(1024,1024),(1360,768),(1440,1440),(2048,2048),(2688,1536)]:
    gw, gh = W//16, H//16
    n = gw*gh
    tot = n + 512
    print(f"{f'{W}x{H}':>14s} {W*H/1e6:>11.2f} {f'{gh}x{gw}':>13s} {n:>11,d} {tot:>10,d} {tot*tot/1e6:>15.1f}")

print()
print("=" * 78)
print("TIMESTEP SCHEDULE  get_schedule(num_steps, image_seq_len)")
print("=" * 78)
for n_img in (1024, 4096, 16384):
    for ns in (4, 28, 50):
        mu = compute_empirical_mu(n_img, ns)
        s = get_schedule(ns, n_img)
        print(f"  seq_len={n_img:>6,d}  steps={ns:>3d}  mu={mu:.4f}  "
              f"t: {s[0]:.4f} -> {s[1]:.4f} -> {s[len(s)//2]:.4f} -> ... -> {s[-1]:.4f}")
print("\n  raw linspace(1,0,9) :", [round(float(z),4) for z in torch.linspace(1,0,9)])
print("  shifted  (4096, 8)  :", [round(z,4) for z in get_schedule(8, 4096)])
print("  shifted  (16384, 8) :", [round(z,4) for z in get_schedule(8, 16384)])
print("  => the shift pushes timesteps toward t=1 (the noisy end); stronger for more tokens")

print()
print("=" * 78)
print("FLOPs (matmul only, per denoising step, batch=1, FLUX.2 [dev])")
print("=" * 78)
h, nh, r, DEPTH, DSINGLE = 6144, 48, 3.0, 8, 48
m = int(h*r)
# MACs a *token* pays in one block. A double block has two streams, but each token only
# ever passes through ONE of them (image tokens the image weights, text tokens the text
# weights) -- and one stream costs exactly the same as a whole single block:
#   qkv 3h^2 + proj h^2 + mlp (h*2m + m*h) = 4h^2 + 3hm.
per_token = 4*h*h + 3*h*m
print(f"  MACs per token per block (single block, or ONE stream of a double block): {per_token:,}")
for W, H in [(512,512), (1024,1024), (1360,768), (2048,2048)]:
    n_img = (W//16)*(H//16); n_txt = 512; L = n_img + n_txt
    lin = 2 * (DEPTH + DSINGLE) * L * per_token      # 2 = MAC->FLOP
    att = 2 * (DEPTH + DSINGLE) * 2 * L * L * h      # q.k^T and weights.v per block
    tot = lin + att
    print(f"  {W}x{H}: {n_img:,} img + {n_txt} txt (L={L:,})")
    print(f"     linear layers  : {lin/1e12:8.1f} TFLOPs")
    print(f"     attention QK+AV: {att/1e12:8.1f} TFLOPs")
    print(f"     TOTAL / step   : {tot/1e12:8.1f} TFLOPs   ->  x50 steps = {50*tot/1e15:.2f} PFLOPs")
print("  NB multiplying a double block's FULL parameter count by every token double-counts:")
print(f"     that (wrong) method gives {2*DEPTH*L*2*per_token/1e12 + 2*DSINGLE*L*per_token/1e12:.1f}"
      f" TFLOPs of 'linear' work at 2048^2 instead of {2*DEPTH*L*per_token/1e12:.1f}.")

print()
print("=" * 78)
print("MEMORY")
print("=" * 78)
for name, n in [("FLUX.2 [dev]", 32_223_281_152), ("Klein 9B", 9_078_581_248), ("Klein 4B", 3_875_544_576)]:
    print(f"  {name:>14s}: bf16 {n*2/1e9:6.1f} GB | fp8 {n/1e9:5.1f} GB | int4 {n*0.5/1e9:5.1f} GB")
print("\n  KV cache for reference tokens: bytes = n_blocks * 2 (K and V) * n_ref * hidden * 2 (bf16)")
# one reference capped at 2024^2 px (sampling.py L58) then centre-cropped to a multiple of 16
one_ref  = (2016//16)**2          # = 15,876 tokens
four_ref = 4 * (1024//16)**2      # = 16,384 tokens
for label, p in [("FLUX.2 [dev]", M.Flux2Params()), ("Klein 9B", M.Klein9BParams()), ("Klein 4B", M.Klein4BParams())]:
    nb = p.depth + p.depth_single_blocks
    per_tok = nb * 2 * p.hidden_size * 2
    print(f"     {label:>14s}: {nb} blocks, hidden {p.hidden_size} -> {per_tok/1e6:.3f} MB per ref token;"
          f"  1 ref ({one_ref:,d} tok) = {one_ref*per_tok/1e9:.2f} GB;"
          f"  4 refs ({four_ref:,d} tok) = {four_ref*per_tok/1e9:.2f} GB")

print()
print("=" * 78)
print("SHARED vs PER-BLOCK MODULATION (section 1.1)")
print("=" * 78)
p = M.Flux2Params(); h = p.hidden_size
shared = 2*(6*h*h) + 3*h*h
per_block = p.depth*2*(6*h*h) + p.depth_single_blocks*(3*h*h)
print(f"  3 shared Modulation matrices : {shared:,} parameters")
print(f"  one per block (8 double x 2 triples of 6h^2, 48 single x 3h^2): {per_block:,}")
print(f"  => sharing saves {per_block-shared:,} parameters ({(per_block-shared)/1e9:.2f} B)")

print()
print("=" * 78)
print("STEPS ABOVE t=0.7 (sections 2.3 / 14.1)")
print("=" * 78)
for ns, L in [(50, 4096), (50, 16384), (8, 4096)]:
    s = get_schedule(ns, L)
    print(f"  steps={ns:>3d} seq_len={L:>6,d} mu={compute_empirical_mu(L, ns):.4f}: "
          f"{sum(1 for z in s[:-1] if z > 0.7)} of {ns} above 0.7, "
          f"{sum(1 for z in s[:-1] if z > 0.9)} above 0.9")

print()
print("=" * 78)
print("MODULE CENSUS, measured on the real meta-device models (sections 1.1 / 7.3)")
print("=" * 78)
import torch.nn as nn
for label, p in [("FLUX.2 [dev]", M.Flux2Params()),
                 ("Klein 9B", M.Klein9BParams()),
                 ("Klein 4B", M.Klein4BParams())]:
    with torch.device("meta"):
        net = M.Flux2(p)
    mods = list(net.modules())
    n_lin = sum(isinstance(x, nn.Linear) for x in mods)
    n_rms = sum(isinstance(x, M.RMSNorm) for x in mods)
    n_ln = sum(isinstance(x, nn.LayerNorm) for x in mods)
    n_ln_affine = sum(isinstance(x, nn.LayerNorm) and x.elementwise_affine for x in mods)
    n_qk = sum(isinstance(x, M.QKNorm) for x in mods)
    n_mod = sum(isinstance(x, M.Modulation) for x in mods)
    n_mlp = sum(isinstance(x, M.MLPEmbedder) for x in mods)
    n_bias = sum(1 for name, _ in net.named_parameters() if name.endswith(".bias"))
    n_drop = sum(isinstance(x, nn.Dropout) for x in mods)
    print(f"{label}:")
    print(f"   Linear {n_lin} | RMSNorm {n_rms} (in {n_qk} QKNorm) | LayerNorm {n_ln} "
          f"(with affine params: {n_ln_affine})")
    print(f"   Modulation {n_mod} | MLPEmbedder {n_mlp} | bias vectors {n_bias} | Dropout {n_drop}")
    print(f"   RMSNorm scale elements: {sum(math.prod(q.shape) for n, q in net.named_parameters() if 'scale' in n)}")
