"""Deeper verification: RoPE semantics, modulation blending, and KV-cache exactness."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from einops import rearrange
import model as M

torch.manual_seed(0)

print("=" * 70)
print("1. RoPE internals")
print("=" * 70)
# rope() requires pos of shape (B, N)
pos = torch.arange(5).float().unsqueeze(0)          # (1,5)
r = M.rope(pos, dim=8, theta=2000)
print(f"   rope(pos=(1,5), dim=8, theta=2000) -> {tuple(r.shape)}")
print(f"   interpretation: (B, N, dim//2=4, 2, 2) -> {4} rotation planes of 2x2 matrices")
print(f"   pos=0 matrix for plane 0:\n{r[0,0,0]}\n   (identity, as expected)")
# check it's a rotation matrix
Rm = r[0, 3, 1]
print(f"   pos=3 plane 1 matrix:\n{Rm}")
print(f"   R^T R = I ? max|RR^T - I| = {float((Rm @ Rm.T - torch.eye(2)).abs().max()):.2e}")
print(f"   det(R) = {float(torch.det(Rm)):.6f}  (should be +1 => pure rotation)")

# frequency ladder
for theta in (2000,):
    for dim in (8, 32):
        scale = torch.arange(0, dim, 2).float() / dim
        omega = 1.0 / (theta ** scale)
        print(f"   theta={theta}, dim={dim}: {len(omega)} planes, "
              f"omega[0]={float(omega[0]):.4f} ... omega[-1]={float(omega[-1]):.6f}")

# EmbedND
emb = M.EmbedND(dim=128, theta=2000, axes_dim=[32, 32, 32, 32])
ids = torch.zeros(1, 7, 4)
out = emb(ids)
print(f"\n   EmbedND(dim=128, axes_dim=[32,32,32,32]) on ids (1,7,4) -> {tuple(out.shape)}")
print(f"   = (B, 1(heads), L, 64 planes, 2, 2);  4 axes x 16 planes = 64 planes -> 128 dims")

# apply_rope preserves norms and is a real rotation
q = torch.randn(1, 2, 7, 128)
k = torch.randn(1, 2, 7, 128)
q2, k2 = M.apply_rope(q, k, out)
print(f"   apply_rope: q {tuple(q.shape)} -> {tuple(q2.shape)}")
print(f"   per-token |q| before: {[round(float(z),5) for z in q.norm(dim=-1)[0,0]]}")
print(f"   per-token |q| after : {[round(float(z),5) for z in q2.norm(dim=-1)[0,0]]}  (norm preserved)")

# relative-position property: <RoPE(p)q, RoPE(p+d)k> depends only on d
def rope_at(p, dim=128, theta=2000):
    ids = torch.zeros(1, 1, 4); ids[0, 0, 2] = p   # vary the 'w' axis only
    return M.EmbedND(dim=dim, theta=theta, axes_dim=[32]*4)(ids)

qq = torch.randn(1, 1, 1, 128); kk = torch.randn(1, 1, 1, 128)
def dot_at(p1, p2):
    a, _ = M.apply_rope(qq, kk, rope_at(p1))
    b, _ = M.apply_rope(kk, qq, rope_at(p2))
    return float((a[0,0,0] * b[0,0,0]).sum())
print(f"\n   relative-position invariance (dot products should match by delta):")
print(f"     delta=2: (p=0,p=2)={dot_at(0,2):.6f}   (p=5,p=7)={dot_at(5,7):.6f}   (p=13,p=15)={dot_at(13,15):.6f}")
print(f"     delta=3: (p=0,p=3)={dot_at(0,3):.6f}   (p=5,p=8)={dot_at(5,8):.6f}")

print()
print("=" * 70)
print("2. Modulation blending")
print("=" * 70)
B, D = 2, 16
vec = torch.randn(B, D)
mod_d = M.Modulation(D, double=True)
md = mod_d(vec)                      # ((shift,scale,gate), (shift,scale,gate))
print(f"   Modulation(double=True) returns a 2-tuple: "
      f"(mod1={len(md[0])} chunks, mod2={len(md[1])} chunks), each {tuple(md[0][0].shape)} (B,1,D)")
mod_s = M.Modulation(D, double=False)
ms = mod_s(vec)
s1, s2 = ms                          # (triple, None)
print(f"   Modulation(double=False) returns (triple of {len(s1)}, {s2}), each {tuple(s1[0].shape)}")

n_ref, n_txt, L_img = 3, 4, 7
bd = M._blend_double_mods(md, md, n_ref, L_img)
print(f"   _blend_double_mods: mod1 triple each -> {tuple(bd[0][0].shape)}  (B, n_ref+L_img, D)")
bs = M._blend_single_mods(s1, s1, n_txt, n_ref, n_txt + n_ref + L_img)
print(f"   _blend_single_mods: each -> {tuple(bs[0].shape)}  (B, n_txt+n_ref+L_img, D)")
# verify the blend actually places different values in different slots
img_m = (torch.full((B,1,D), 1.0), torch.full((B,1,D), 2.0), torch.full((B,1,D), 3.0))
ref_m = (torch.full((B,1,D), 10.0), torch.full((B,1,D), 20.0), torch.full((B,1,D), 30.0))
bl = M._blend_mod_triple(img_m, ref_m, n_ref, L_img)
print(f"   _blend_mod_triple shift column per token: {[float(z) for z in bl[0][0,:,0]]}")
print(f"      -> first {n_ref} slots = ref (10.0), remaining {L_img-n_ref} = img (1.0)")
bls = M._blend_single_mods(img_m, ref_m, n_txt, n_ref, n_txt+n_ref+L_img)
print(f"   _blend_single_mods shift per token: {[float(z) for z in bls[0][0,:,0]]}")
print(f"      -> [txt x{n_txt}=1.0][ref x{n_ref}=10.0][img x{L_img}=1.0]")

print()
print("=" * 70)
print("3. causal_attn_fn mask semantics")
print("=" * 70)
n_txt, n_ref, n_img = 2, 3, 4
L = n_txt + n_ref + n_img
H = 2
Dh = L                                  # head_dim == L so one-hot V reveals the mask
q = torch.randn(1, H, L, Dh)
k = torch.randn(1, H, L, Dh)
v_onehot = torch.eye(L).unsqueeze(0).unsqueeze(0).expand(1, H, L, L).contiguous()
o2 = M.causal_attn_fn(q, k, v_onehot, n_txt, n_ref)
print(f"   output layout: rearrange('b h n d -> b n (h d)') -> {tuple(o2.shape)}")
mask = (o2.reshape(1, L, H * L).reshape(1, L, H, L).amax(dim=2)[0] > 1e-6).int()
labels = [f"txt{i}" for i in range(n_txt)] + [f"ref{i}" for i in range(n_ref)] + [f"img{i}" for i in range(n_img)]
print("\n   Empirical attention mask (rows=query, cols=key, 1=can attend):")
print("        " + "".join(f"{l:>6s}" for l in labels))
for i, l in enumerate(labels):
    print(f"   {l:>5s}" + "".join(f"{int(mask[i,j]):>6d}" for j in range(L)))
print("   => txt and img are fully bidirectional over everything;")
print("      ref tokens attend ONLY to ref tokens (never txt, never img).")

# numerical equivalence of cached vs full path for the txt/img rows
Dh2 = 8
q = torch.randn(1, H, L, Dh2); k = torch.randn(1, H, L, Dh2); v = torch.randn(1, H, L, Dh2)
o_full = M.causal_attn_fn(q, k, v, n_txt, n_ref)              # (1, L, H*Dh2)
cache = {"k_ref": k[:, :, n_txt:n_txt + n_ref].clone(),
         "v_ref": v[:, :, n_txt:n_txt + n_ref].clone()}
q_n = torch.cat([q[:, :, :n_txt], q[:, :, n_txt + n_ref:]], dim=2)
k_n = torch.cat([k[:, :, :n_txt], k[:, :, n_txt + n_ref:]], dim=2)
v_n = torch.cat([v[:, :, :n_txt], v[:, :, n_txt + n_ref:]], dim=2)
o_cached = M.causal_attn_fn(q_n, k_n, v_n, n_txt, n_ref, cache)          # (1, n_txt+n_img, H*Dh2)
o_full_n = torch.cat([o_full[:, :n_txt], o_full[:, n_txt + n_ref:]], dim=1)
print(f"\n   cached-path vs full-path on txt+img rows: shapes {tuple(o_cached.shape)} vs {tuple(o_full_n.shape)}")
print(f"   max|diff| = {float((o_cached - o_full_n).abs().max()):.3e}   <-- cache is EXACT")
o_ref = M.causal_attn_fn(q, k, v, n_txt, n_ref)[:, n_txt:n_txt + n_ref]
print(f"   (the {n_ref} ref rows are computed but discarded on cached steps)")

print()
print("=" * 70)
print("4. KV-CACHE EXACTNESS on a tiny end-to-end model")
print("=" * 70)
class TP:
    in_channels=8; context_in_dim=12; hidden_size=32; num_heads=4
    depth=2; depth_single_blocks=2; axes_dim=[2,2,2,2]; theta=2000
    mlp_ratio=2.0; use_guidance_embed=True
tiny = M.Flux2(TP()).eval()

B, n_txt, n_ref, n_img = 1, 3, 2, 5
ctx = torch.randn(B, n_txt, TP.context_in_dim)
ctx_ids = torch.zeros(B, n_txt, 4); ctx_ids[0,:,0]=0; ctx_ids[0,:,3]=torch.arange(n_txt).float()
x_img = torch.randn(B, n_img, TP.in_channels)
x_img_ids = torch.zeros(B, n_img, 4)
x_img_ids[0,:,1] = torch.arange(n_img).float() % 3
x_img_ids[0,:,2] = torch.arange(n_img).float() // 3
x_ref = torch.randn(B, n_ref, TP.in_channels)
x_ref_ids = torch.zeros(B, n_ref, 4)
x_ref_ids[0,:,0] = torch.tensor([10., 20.])   # separate 't' slots per reference
guid = torch.tensor([3.5])

caches = {}
preds = {}
for t in (0.9, 0.5, 0.1):
    with torch.no_grad():
        p, c = tiny.forward_kv_extract(x_img, x_img_ids, torch.tensor([t]), ctx, ctx_ids,
                                       guid, x_ref, x_ref_ids)
    caches[t] = c
    preds[t] = p

d = max(float((caches[0.9]["double_blocks"][i]["k_ref"] - caches[0.1]["double_blocks"][i]["k_ref"]).abs().max())
        for i in range(len(caches[0.9]["double_blocks"])))
d2 = max(float((caches[0.9]["single_blocks"][i]["v_ref"] - caches[0.1]["single_blocks"][i]["v_ref"]).abs().max())
         for i in range(len(caches[0.9]["single_blocks"])))
print(f"   ref K/V identical across timesteps t=0.9 vs t=0.1:")
print(f"     max|diff| double-block k_ref = {d:.3e}")
print(f"     max|diff| single-block v_ref = {d2:.3e}")
print("   => the reference tokens' K/V really are step-invariant, so caching is EXACT.")

with torch.no_grad():
    p_cached = tiny.forward_kv_cached(x_img, x_img_ids, torch.tensor([0.5]), ctx, ctx_ids,
                                      guid, caches[0.9])
print(f"\n   forward_kv_extract(t=0.5) vs forward_kv_cached(t=0.5, cache from t=0.9):")
print(f"     shapes {tuple(preds[0.5].shape)} vs {tuple(p_cached.shape)}")
print(f"     max|diff| = {float((preds[0.5]-p_cached).abs().max()):.3e}")
print(f"     rel err   = {float((preds[0.5]-p_cached).norm()/preds[0.5].norm()):.3e}")

print()
print("=" * 70)
print("5. forward() == forward_kv_extract() with ZERO ref tokens?")
print("=" * 70)
empty_ref = torch.zeros(B, 0, TP.in_channels)
empty_ref_ids = torch.zeros(B, 0, 4)
with torch.no_grad():
    p0, _ = tiny.forward_kv_extract(x_img, x_img_ids, torch.tensor([0.5]), ctx, ctx_ids,
                                    guid, empty_ref, empty_ref_ids)
    pf = tiny(x_img, x_img_ids, torch.tensor([0.5]), ctx, ctx_ids, guid)
print(f"   max|diff| = {float((p0-pf).abs().max()):.3e}  (0 ref tokens => identical to plain forward)")

print()
print("=" * 70)
print("6. LastLayer / adaLN")
print("=" * 70)
ll = tiny.final_layer
x = torch.randn(B, n_img, TP.hidden_size); v = torch.randn(B, TP.hidden_size)
with torch.no_grad():
    o = ll(x, v)
print(f"   in {tuple(x.shape)} + vec {tuple(v.shape)} -> {tuple(o.shape)} (out_channels={TP.in_channels})")
print(f"   norm_final has affine params? {ll.norm_final.elementwise_affine}")
print(f"   adaLN_modulation = {ll.adaLN_modulation}")
