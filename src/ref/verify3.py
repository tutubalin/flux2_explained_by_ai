"""Analytical parameter formulas + real-world cost numbers for the article."""
import sys, math
sys.path.insert(0, "/home/user/ref")
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
print("FLOPs (matmul only, per denoising step, batch=1)")
print("=" * 78)
h, nh, r = 6144, 48, 3.0
m = int(h*r)
dbl_lin = 2*(4*h*h + 3*h*m)      # weights touched per double block
sgl_lin = 4*h*h + 3*h*m
for n_txt, n_img in [(512, 4096), (512, 16384)]:
    L = n_txt + n_img
    f_dbl = 8  * (2*L*dbl_lin + 4*L*L*h)     # 2 = MAC->FLOP
    f_sgl = 48 * (2*L*sgl_lin + 4*L*L*h)
    tot = f_dbl + f_sgl
    print(f"  {n_img:,d} img + {n_txt} txt tokens (L={L:,d}):")
    print(f"     linear layers  : {(f_dbl+f_sgl - 8*4*L*L*h - 48*4*L*L*h)/1e12:8.1f} TFLOPs")
    print(f"     attention QK+AV: {(8*4*L*L*h + 48*4*L*L*h)/1e12:8.1f} TFLOPs")
    print(f"     TOTAL / step   : {tot/1e12:8.1f} TFLOPs   ->  x50 steps = {50*tot/1e15:.2f} PFLOPs")

print()
print("=" * 78)
print("MEMORY")
print("=" * 78)
for name, n in [("FLUX.2 [dev]", 32_223_281_152), ("Klein 9B", 9_078_581_248), ("Klein 4B", 3_875_544_576)]:
    print(f"  {name:>14s}: bf16 {n*2/1e9:6.1f} GB | fp8 {n/1e9:5.1f} GB | int4 {n*0.5/1e9:5.1f} GB")
print("\n  KV cache for ref tokens (Klein 9B, 4 refs @1024x1024, bf16):")
h9, nb9 = 4096, 8+24
nref = 4 * (1024//16)**2
print(f"     ref tokens = {nref:,d};  per block 2*K/V*nref*h*2 bytes = {2*2*nref*h9*2/1e6:.1f} MB")
print(f"     all {nb9} blocks: {nb9*2*2*nref*h9*2/1e9:.2f} GB")
