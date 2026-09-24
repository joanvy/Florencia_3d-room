"""Render fake phone 'videos' of the synthetic room to test the full pipeline.

    python pipeline/tests/synth_capture.py work/placeholder.ply work/synth/videos
"""

import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gs.render import Camera, look_at, render  # noqa: E402
from splat_io import SH_C0, read_ply  # noqa: E402


def texture(xyz, rng):
    """Multi-scale random sinusoid 'value noise' + a few posters: gives SIFT something to lock onto."""
    t = np.zeros(len(xyz))
    for freq in [1.5, 3, 6, 12, 24]:
        for _ in range(3):
            k = rng.normal(size=3) * freq
            t += np.sin(xyz @ k + rng.uniform(0, 6.3)) / freq**0.5
    t = (t - t.min()) / (t.max() - t.min())
    return t


def main():
    ply, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(1)
    s = read_ply(ply)
    rgb = s.rgb * (0.45 + 0.9 * texture(s.xyz, rng))[:, None]
    for _ in range(12):  # posters with random colours
        c = rng.uniform(0.05, 0.95, 3)
        centre = s.xyz[rng.integers(len(s))]
        m = np.abs(s.xyz - centre).max(1) < rng.uniform(0.1, 0.3)
        rgb[m] = c * (0.6 + 0.4 * texture(s.xyz[m] * 3, rng))[:, None]
    rgb = np.clip(rgb, 0, 1)

    P = {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in
         dict(means=s.xyz, quats=s.rot, scales=s.scale, opac=s.opacity, sh0=(rgb - 0.5) / SH_C0).items()}
    W, H = 960, 540
    paths = {
        "perimeter": [(np.array([2 + 1.5 * np.cos(a), 1.5, 1.8 + 1.3 * np.sin(a)]),
                       np.array([2 - 0.5 * np.cos(a), 1.1, 1.8 - 0.5 * np.sin(a)]))
                      for a in np.linspace(0, 2 * np.pi, 150)],
        "centre_spin": [(np.array([2.0, 1.1, 2.4]),
                         np.array([2 + np.cos(a), 1.1 + 0.3 * np.sin(3 * a), 2.4 + np.sin(a)]))
                        for a in np.linspace(0, 2 * np.pi, 150)],
    }
    for name, path in paths.items():
        d = os.path.join(out, name + "_frames")
        os.makedirs(d, exist_ok=True)
        for i, (eye, tgt) in enumerate(path):
            cam = Camera(look_at(eye, tgt), 0.6 * W, 0.6 * W, W / 2, H / 2, W, H)
            with torch.no_grad():
                img, _ = render(P["means"], P["quats"], P["scales"], P["opac"], P["sh0"], None, cam)
            Image.fromarray((img.clamp(0, 1).numpy() * 255).astype(np.uint8)).save(f"{d}/{i:04d}.png")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "10", "-i", f"{d}/%04d.png",
                        "-pix_fmt", "yuv420p", "-crf", "18", os.path.join(out, name + ".mp4")], check=True)
        print("wrote", name)


if __name__ == "__main__":
    main()
