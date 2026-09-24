"""Differentiable Gaussian splat rendering on CPU.

Projection / covariance / SH evaluation are plain PyTorch (autograd); the
tile rasterizer is a C++ extension (rasterize.cpp) with a hand-written
backward pass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.cpp_extension import load

_here = os.path.dirname(os.path.abspath(__file__))
_ext = load(
    name="gs_cpu_rasterize",
    sources=[os.path.join(_here, "rasterize.cpp")],
    extra_cflags=["-O3", "-fopenmp", "-march=native", "-ffast-math"],
    extra_ldflags=["-fopenmp"],
    verbose=False,
)

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199


@dataclass
class Camera:
    viewmat: torch.Tensor  # [4,4] world -> camera (OpenCV: x right, y down, z forward)
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def scaled(self, s: float) -> "Camera":
        return Camera(self.viewmat, self.fx * s, self.fy * s, self.cx * s, self.cy * s,
                      int(round(self.width * s)), int(round(self.height * s)))

    @property
    def center(self) -> torch.Tensor:
        R, t = self.viewmat[:3, :3], self.viewmat[:3, 3]
        return -R.T @ t


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = map(lambda v: np.asarray(v, float), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = -R @ eye
    return torch.tensor(m, dtype=torch.float32)



class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, colors, opacities, depths, radii, width, height, background, absgrad_out):
        image, final_T, n_contrib, ranges, ids = _ext.forward(
            means2d.contiguous(), conics.contiguous(), colors.contiguous(), opacities.contiguous(),
            depths.contiguous(), radii.contiguous(), width, height, background.contiguous())
        ctx.save_for_backward(means2d, conics, colors, opacities, background, final_T, n_contrib, ranges, ids)
        ctx.size = (width, height)
        ctx.absgrad_out = absgrad_out
        ctx.mark_non_differentiable(final_T)
        return image, final_T

    @staticmethod
    def backward(ctx, grad_image, _grad_T):
        means2d, conics, colors, opacities, background, final_T, n_contrib, ranges, ids = ctx.saved_tensors
        width, height = ctx.size
        d_m, d_con, d_col, d_op, absgrad = _ext.backward(
            means2d.contiguous(), conics.contiguous(), colors.contiguous(), opacities.contiguous(),
            background.contiguous(), final_T, n_contrib, ranges, ids, grad_image, width, height)
        if ctx.absgrad_out is not None:
            ctx.absgrad_out.copy_(absgrad)
        return d_m, d_con, d_col, d_op, None, None, None, None, None, None


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(q.shape[:-1] + (3, 3))


def sh_to_color(sh0: torch.Tensor, shN: torch.Tensor | None, dirs: torch.Tensor) -> torch.Tensor:
    """sh0 [N,3], shN [N,3,3] (degree 1: coeffs for y, z, x), dirs [N,3] unit view dirs."""
    c = SH_C0 * sh0
    if shN is not None and shN.shape[-1] > 0:
        x, y, z = dirs.unbind(-1)
        c = c - SH_C1 * y[:, None] * shN[..., 0] + SH_C1 * z[:, None] * shN[..., 1] - SH_C1 * x[:, None] * shN[..., 2]
    return torch.clamp(c + 0.5, min=0.0)


def render(
    means: torch.Tensor,  # [N,3]
    quats: torch.Tensor,  # [N,4] wxyz (unnormalized ok)
    log_scales: torch.Tensor,  # [N,3]
    opacity_logits: torch.Tensor,  # [N]
    sh0: torch.Tensor,  # [N,3]
    shN: torch.Tensor | None,  # [N,3,K]
    cam: Camera,
    background: torch.Tensor | None = None,
    near: float = 0.05,
    eps2d: float = 0.3,
    track_absgrad: bool = False,
):
    """Returns (image [H,W,3], info dict)."""
    if background is None:
        background = torch.zeros(3)
    R = cam.viewmat[:3, :3]
    t = cam.viewmat[:3, 3]
    p = means @ R.T + t
    x, y, z = p.unbind(-1)

    # Cull: behind the near plane or far outside the frustum.
    tan_x = 0.5 * cam.width / cam.fx
    tan_y = 0.5 * cam.height / cam.fy
    with torch.no_grad():
        visible = (z > near) & (x.abs() < 1.3 * tan_x * z + 1e-6) & (y.abs() < 1.3 * tan_y * z + 1e-6)
    idx = visible.nonzero().squeeze(1)

    pz = z[idx]
    px, py = x[idx], y[idx]
    means2d = torch.stack([cam.fx * px / pz + cam.cx, cam.fy * py / pz + cam.cy], -1)

    # EWA splatting: cov2d = J W Sigma W^T J^T
    rot = quat_to_rotmat(quats[idx])
    S = torch.exp(log_scales[idx])
    M = rot * S[:, None, :]
    cov3d = M @ M.transpose(1, 2)
    cov_c = R @ cov3d @ R.T
    lim_x, lim_y = 1.3 * tan_x, 1.3 * tan_y
    tx = (px / pz).clamp(-lim_x, lim_x) * pz
    ty = (py / pz).clamp(-lim_y, lim_y) * pz
    zeros = torch.zeros_like(pz)
    J = torch.stack([
        cam.fx / pz, zeros, -cam.fx * tx / (pz * pz),
        zeros, cam.fy / pz, -cam.fy * ty / (pz * pz),
    ], -1).reshape(-1, 2, 3)
    cov2d = J @ cov_c @ J.transpose(1, 2)
    a = cov2d[:, 0, 0] + eps2d
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + eps2d
    det = (a * c - b * b).clamp(min=1e-12)
    conics = torch.stack([c / det, -b / det, a / det], -1)

    with torch.no_grad():
        mid = 0.5 * (a + c)
        lam = mid + torch.sqrt((mid * mid - det).clamp(min=0.1))
        radii = torch.ceil(3.0 * torch.sqrt(lam)).int()
        on_screen = ((means2d[:, 0] + radii > 0) & (means2d[:, 0] - radii < cam.width) &
                     (means2d[:, 1] + radii > 0) & (means2d[:, 1] - radii < cam.height))
        radii = torch.where(on_screen & (det > 1e-12), radii, torch.zeros_like(radii))

    dirs = torch.nn.functional.normalize(means[idx] - cam.center, dim=-1)
    colors = sh_to_color(sh0[idx], shN[idx] if shN is not None else None, dirs)
    opac = torch.sigmoid(opacity_logits[idx])

    # Filled in by the backward pass: per-gaussian sum of |d loss / d mean2d| (densification signal).
    absgrad_vis = torch.zeros(len(idx), 2) if track_absgrad else None
    image, final_T = _Rasterize.apply(means2d, conics, colors, opac, pz.detach().contiguous(), radii.contiguous(),
                                      cam.width, cam.height, background, absgrad_vis)
    info = {"visible_idx": idx, "radii": radii, "absgrad_vis": absgrad_vis, "means2d": means2d,
            "alpha": 1 - final_T}
    return image, info
