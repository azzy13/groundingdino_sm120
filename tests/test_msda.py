"""Compare GroundingDINO's compiled MSDeformAttn kernel against the pure-PyTorch reference.
Usage: python test_msda.py  (run with groundingdino importable, i.e. the built _C on sys.path)"""
import sys, time, torch, torch.nn.functional as F
from groundingdino import _C

def ref(value, shapes, loc, attn):
    N, S, M, D = value.shape; _, Lq, M, L, P, _ = loc.shape
    vl = value.split([h * w for h, w in shapes.tolist()], dim=1)
    grids = 2 * loc - 1; out = []
    for l, (h, w) in enumerate(shapes.tolist()):
        v = vl[l].flatten(2).transpose(1, 2).reshape(N * M, D, h, w)
        g = grids[:, :, :, l].transpose(1, 2).flatten(0, 1)
        out.append(F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
    a = attn.transpose(1, 2).reshape(N * M, 1, Lq, L * P)
    return (torch.stack(out, -2).flatten(-2) * a).sum(-1).view(N, M * D, Lq).transpose(1, 2).contiguous()

torch.manual_seed(0)
dev = "cuda"; print("GPU:", torch.cuda.get_device_name(), "cc", torch.cuda.get_device_capability())
N, M, D, Lq, L, P = 2, 8, 32, 900, 4, 4
shapes = torch.tensor([[100, 100], [50, 50], [25, 25], [13, 13]], dtype=torch.long, device=dev)
lsi = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
S = int(shapes.prod(1).sum())
ok = True
for dt, tol in [(torch.float32, 1e-4), (torch.float64, 1e-10)]:
    value = torch.rand(N, S, M, D, device=dev, dtype=dt) * 0.01
    loc = torch.rand(N, Lq, M, L, P, 2, device=dev, dtype=dt)
    attn = torch.rand(N, Lq, M, L, P, device=dev, dtype=dt) + 1e-5
    attn /= attn.sum((-1, -2), keepdim=True)
    v, l_, a = [t.clone().requires_grad_() for t in (value, loc, attn)]
    out = _C.ms_deform_attn_forward(v, shapes, lsi, l_, a, 64)
    exp = ref(value, shapes, loc, attn)
    err = (out - exp).abs().max().item(); good = err < tol; ok &= good
    print(f"{dt} forward max abs err {err:.2e} {'OK' if good else 'FAIL'}")
    go = torch.rand_like(out)
    gv, gl, ga = _C.ms_deform_attn_backward(value, shapes, lsi, loc, attn, go.contiguous(), 64)
    rv, rl, ra = [t.clone().requires_grad_() for t in (value, loc, attn)]
    ref(rv, shapes, rl, ra).backward(go)
    for name, g1, g2 in [("grad_value", gv, rv.grad), ("grad_loc", gl, rl.grad), ("grad_attn", ga, ra.grad)]:
        e = ((g1 - g2).abs().max() / g2.abs().max().clamp_min(1e-12)).item(); good = e < (1e-3 if dt == torch.float32 else 1e-8); ok &= good
        print(f"{dt} {name} rel err {e:.2e} {'OK' if good else 'FAIL'}")

# speed: CUDA kernel vs PyTorch fallback
value = torch.rand(1, S, M, D, device=dev); loc = torch.rand(1, Lq, M, L, P, 2, device=dev); attn = torch.rand(1, Lq, M, L, P, device=dev)
for name, fn in [("cuda _C", lambda: _C.ms_deform_attn_forward(value, shapes, lsi, loc, attn, 64)), ("pytorch fallback", lambda: ref(value, shapes, loc, attn))]:
    for _ in range(10): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(200): fn()
    torch.cuda.synchronize(); print(f"{name}: {(time.perf_counter() - t) / 200 * 1e3:.3f} ms/call")
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
