"""Generate a synthetic 'room' splat so the viewer can be built and deployed
before the real capture is processed. Y-up, metres, floor at y=0.

    python pipeline/make_placeholder.py work/placeholder.ply
"""

import sys

import numpy as np

from splat_io import SH_C0, Splats, write_ply

rng = np.random.default_rng(7)
parts = []


def plane(origin, u, v, color, spacing=0.03, noise=0.04):
    """Flat, disc-like splats covering the parallelogram origin + a*u + b*v."""
    origin, u, v = map(np.asarray, (origin, u, v))
    nu = max(2, int(np.linalg.norm(u) / spacing))
    nv = max(2, int(np.linalg.norm(v) / spacing))
    a, b = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv))
    a, b = a.ravel(), b.ravel()
    xyz = origin + a[:, None] * u + b[:, None] * v
    rgb = np.clip(np.asarray(color) + rng.normal(0, noise, (len(xyz), 3)), 0, 1)
    # Orient the flat axis (z of the local frame) along the plane normal.
    n = np.cross(u, v)
    n /= np.linalg.norm(n)
    z = np.array([0, 0, 1.0])
    axis = np.cross(z, n)
    s = np.linalg.norm(axis)
    if s < 1e-8:
        q = np.array([1, 0, 0, 0.0]) if n[2] > 0 else np.array([0, 1, 0, 0.0])
    else:
        ang = np.arctan2(s, np.dot(z, n))
        axis /= s
        q = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])
    parts.append((xyz, rgb, np.log([spacing * 0.9, spacing * 0.9, 0.002]), q))


def box(lo, hi, color, **kw):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    d = hi - lo
    X, Y, Z = np.diag(d)
    plane(lo, X, Z, color, **kw)  # bottom
    plane(lo + Y, Z, X, color, **kw)  # top
    plane(lo, Y, X, color, **kw)
    plane(lo + Z, X, Y, color, **kw)
    plane(lo, Z, Y, color, **kw)
    plane(lo + X, Y, Z, color, **kw)


W, D, H = 4.0, 3.6, 2.7  # x: width, z: depth, y: height
plane([0, 0, 0], [0, 0, D], [W, 0, 0], [0.55, 0.40, 0.28])  # wooden floor
plane([0, H, 0], [W, 0, 0], [0, 0, D], [0.93, 0.92, 0.88])  # ceiling
plane([0, 0, 0], [W, 0, 0], [0, H, 0], [0.86, 0.80, 0.70])  # back wall
plane([0, 0, D], [0, H, 0], [W, 0, 0], [0.86, 0.80, 0.70])  # front wall
plane([0, 0, 0], [0, H, 0], [0, 0, D], [0.80, 0.74, 0.64])  # left wall
plane([W, 0, 0], [0, 0, D], [0, H, 0], [0.80, 0.74, 0.64])  # right wall
box([1.0, 0, 0.02], [3.0, 0.55, 2.1], [0.95, 0.95, 0.95], spacing=0.025)  # bed
box([1.0, 0, 0.02], [3.0, 1.1, 0.12], [0.35, 0.22, 0.15], spacing=0.025)  # headboard
box([1.3, 0.55, 0.2], [1.9, 0.68, 0.55], [0.98, 0.98, 1.0], spacing=0.02)  # pillow
box([2.1, 0.55, 0.2], [2.7, 0.68, 0.55], [0.98, 0.98, 1.0], spacing=0.02)  # pillow
plane([W - 0.01, 0.9, 1.2], [0, 0, 1.2], [0, 1.2, 0], [0.55, 0.62, 0.66], spacing=0.02)  # mirror
plane([0.01, 0.8, 1.0], [0, 1.4, 0], [0, 0, 1.6], [0.75, 0.86, 0.95], spacing=0.02)  # window

xyz = np.concatenate([p[0] for p in parts]).astype(np.float32)
rgb = np.concatenate([p[1] for p in parts])
scale = np.concatenate([np.tile(p[2], (len(p[0]), 1)) for p in parts]).astype(np.float32)
rot = np.concatenate([np.tile(p[3], (len(p[0]), 1)) for p in parts]).astype(np.float32)
n = len(xyz)
s = Splats(
    xyz=xyz,
    f_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
    f_rest=np.zeros((n, 3, 0), np.float32),
    opacity=np.full(n, 4.0, np.float32),
    scale=scale,
    rot=rot,
)
write_ply(sys.argv[1], s)
print(f"wrote {n} splats to {sys.argv[1]}")
