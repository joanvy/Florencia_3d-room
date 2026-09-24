"""Read/write 3D Gaussian Splatting PLY files (the INRIA 3DGS layout).

Fields per splat: x y z, f_dc_0..2, f_rest_* (optional), opacity (logit),
scale_0..2 (log), rot_0..3 (w x y z quaternion).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814


@dataclass
class Splats:
    xyz: np.ndarray  # (N,3) float32
    f_dc: np.ndarray  # (N,3) float32  SH band 0
    f_rest: np.ndarray  # (N,3,K) float32 higher SH bands (K may be 0)
    opacity: np.ndarray  # (N,) float32  logit
    scale: np.ndarray  # (N,3) float32  log scale
    rot: np.ndarray  # (N,4) float32  wxyz

    def __len__(self) -> int:
        return len(self.xyz)

    def subset(self, mask: np.ndarray) -> "Splats":
        return Splats(self.xyz[mask], self.f_dc[mask], self.f_rest[mask], self.opacity[mask], self.scale[mask], self.rot[mask])

    @property
    def rgb(self) -> np.ndarray:
        return np.clip(self.f_dc * SH_C0 + 0.5, 0, 1)

    @property
    def alpha(self) -> np.ndarray:
        return 1 / (1 + np.exp(-self.opacity))


def read_ply(path: str) -> Splats:
    v = PlyData.read(path)["vertex"].data
    names = v.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    f_dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)
    rest_names = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    if rest_names:
        f_rest = np.stack([v[n] for n in rest_names], 1).astype(np.float32).reshape(len(xyz), 3, -1)
    else:
        f_rest = np.zeros((len(xyz), 3, 0), np.float32)
    opacity = np.asarray(v["opacity"], np.float32)
    scale = np.stack([v[f"scale_{i}"] for i in range(3)], 1).astype(np.float32)
    rot = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    return Splats(xyz, f_dc, f_rest, opacity, scale, rot)


def write_ply(path: str, s: Splats) -> None:
    n = len(s)
    # Always write full degree-3 SH (15 coeffs/channel): Spark's SPZ transcoder
    # rejects PLYs with fewer f_rest fields. Missing bands are zero.
    k = 15
    f_rest = np.zeros((n, 3, k), np.float32)
    f_rest[:, :, : s.f_rest.shape[2]] = s.f_rest[:, :, :k]
    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    fields += [f"f_rest_{i}" for i in range(3 * k)]
    fields += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    cols = [s.xyz, np.zeros((n, 3), np.float32), s.f_dc, f_rest.reshape(n, -1), s.opacity[:, None], s.scale, s.rot]
    data = np.concatenate(cols, 1).astype(np.float32)
    arr = np.empty(n, dtype=[(f, "f4") for f in fields])
    for i, f in enumerate(fields):
        arr[f] = data[:, i]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(path)
