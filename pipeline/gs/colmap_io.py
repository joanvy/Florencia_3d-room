"""Minimal reader for COLMAP text models (cameras.txt, images.txt, points3D.txt)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class ColmapImage:
    name: str
    camera_id: int
    qvec: np.ndarray  # wxyz, world -> camera
    tvec: np.ndarray

    @property
    def R(self) -> np.ndarray:
        w, x, y, z = self.qvec / np.linalg.norm(self.qvec)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    @property
    def viewmat(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.tvec
        return m

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.tvec


@dataclass
class ColmapCamera:
    model: str
    width: int
    height: int
    params: np.ndarray

    def pinhole(self):
        """(fx, fy, cx, cy) — the model must be undistorted (PINHOLE/SIMPLE_PINHOLE)."""
        p = self.params
        if self.model == "PINHOLE":
            return p[0], p[1], p[2], p[3]
        if self.model == "SIMPLE_PINHOLE":
            return p[0], p[0], p[1], p[2]
        raise ValueError(f"camera model {self.model} is not undistorted; run colmap image_undistorter")


def read_model(path: str):
    cameras: dict[int, ColmapCamera] = {}
    with open(os.path.join(path, "cameras.txt")) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            el = line.split()
            cameras[int(el[0])] = ColmapCamera(el[1], int(el[2]), int(el[3]), np.array(el[4:], float))

    images: list[ColmapImage] = []
    with open(os.path.join(path, "images.txt")) as f:
        lines = [l for l in f if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        el = lines[i].split()
        if len(el) < 10:
            continue
        images.append(ColmapImage(el[9], int(el[8]), np.array(el[1:5], float), np.array(el[5:8], float)))
    images.sort(key=lambda im: im.name)

    xyz, rgb = [], []
    with open(os.path.join(path, "points3D.txt")) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            el = line.split()
            xyz.append([float(v) for v in el[1:4]])
            rgb.append([int(v) for v in el[4:7]])
    return cameras, images, np.array(xyz, np.float32), np.array(rgb, np.float32) / 255.0
