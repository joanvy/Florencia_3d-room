"""Check the C++ rasterizer (forward + backward) against a dense PyTorch reference."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gs.render import _Rasterize  # noqa: E402

TILE = 16


def reference(means2d, conics, colors, opac, depths, radii, W, H, bg):
    order = torch.argsort(depths)
    means2d, conics, colors, opac, radii = means2d[order], conics[order], colors[order], opac[order], radii[order]
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    px = (xs.reshape(-1) + 0.5)[:, None]
    py = (ys.reshape(-1) + 0.5)[:, None]
    tx = (xs.reshape(-1) // TILE)[:, None]
    ty = (ys.reshape(-1) // TILE)[:, None]
    r = radii[None].float()
    in_tile = ((tx >= torch.floor((means2d[None, :, 0] - r) / TILE)) & (tx < torch.ceil((means2d[None, :, 0] + r) / TILE)) &
               (ty >= torch.floor((means2d[None, :, 1] - r) / TILE)) & (ty < torch.ceil((means2d[None, :, 1] + r) / TILE)) & (r > 0))
    dx = means2d[None, :, 0] - px
    dy = means2d[None, :, 1] - py
    power = -0.5 * (conics[None, :, 0] * dx * dx + conics[None, :, 2] * dy * dy) - conics[None, :, 1] * dx * dy
    alpha = opac[None] * torch.exp(power)
    keep = in_tile & (power <= 0) & (alpha >= 1 / 255)
    alpha = torch.where(keep, alpha.clamp(max=0.99), torch.zeros_like(alpha))
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], 1), 1)
    img = (alpha * T) @ colors
    Tf = T[:, -1] * (1 - alpha[:, -1])
    img = img + Tf[:, None] * bg[None]
    return img.reshape(H, W, 3)


torch.manual_seed(0)
N, W, H = 60, 50, 40
means2d = (torch.rand(N, 2) * torch.tensor([W, H])).requires_grad_()
L = torch.randn(N, 2, 2) * 1.5
cov = L @ L.transpose(1, 2) + torch.eye(2) * 2
det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2
conics = torch.stack([cov[:, 1, 1] / det, -cov[:, 0, 1] / det, cov[:, 0, 0] / det], -1).requires_grad_()
colors = torch.rand(N, 3).requires_grad_()
opac = (torch.rand(N) * 0.8 + 0.1).requires_grad_()
depths = torch.rand(N)
lam = 0.5 * (cov[:, 0, 0] + cov[:, 1, 1]) + torch.sqrt((0.25 * (cov[:, 0, 0] - cov[:, 1, 1]) ** 2 + cov[:, 0, 1] ** 2))
radii = torch.ceil(3 * torch.sqrt(lam)).int()
bg = torch.tensor([0.2, 0.3, 0.4])
target = torch.rand(H, W, 3)

img, _ = _Rasterize.apply(means2d, conics, colors, opac, depths, radii, W, H, bg, None)
ref = reference(means2d, conics, colors, opac, depths, radii, W, H, bg)
print("forward max abs diff:", (img - ref).abs().max().item())

ok = True
g1 = torch.autograd.grad(((img - target) ** 2).sum(), [means2d, conics, colors, opac])
g2 = torch.autograd.grad(((ref - target) ** 2).sum(), [means2d, conics, colors, opac])
for name, a, b in zip(["means2d", "conics", "colors", "opac"], g1, g2):
    rel = (a - b).norm() / (b.norm() + 1e-12)
    print(f"grad {name}: rel err {rel.item():.2e}")
    ok &= rel.item() < 1e-3
ok &= (img - ref).abs().max().item() < 1e-4
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
