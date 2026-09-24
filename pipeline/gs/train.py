"""Train a 3D Gaussian splat from an undistorted COLMAP reconstruction, on CPU.

    python -m gs.train --colmap work/colmap/undistorted --out work/train [--iters 7000]

Follows the original 3DGS recipe (L1 + D-SSIM, adaptive density control with
clone/split/prune and periodic opacity reset) at a reduced resolution and
iteration count so it finishes in hours on a few CPU cores. Output is an
INRIA-layout PLY with SH degree 1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .colmap_io import read_model
from .render import SH_C0, Camera, render

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from splat_io import Splats, write_ply  # noqa: E402


# ---------------------------------------------------------------------------
# Data


def load_dataset(colmap_dir: str, max_width: int):
    sparse = os.path.join(colmap_dir, "sparse_txt")
    cameras, images, xyz, rgb = read_model(sparse)
    cams, pixels, names = [], [], []
    for im in images:
        c = cameras[im.camera_id]
        fx, fy, cx, cy = c.pinhole()
        s = min(1.0, max_width / c.width)
        cam = Camera(torch.tensor(im.viewmat, dtype=torch.float32), fx, fy, cx, cy, c.width, c.height).scaled(s)
        img = Image.open(os.path.join(colmap_dir, "images", im.name)).convert("RGB")
        img = img.resize((cam.width, cam.height), Image.LANCZOS)
        cams.append(cam)
        pixels.append(torch.from_numpy(np.asarray(img).copy()))  # uint8 [H,W,3]
        names.append(im.name)
    return cams, pixels, names, xyz, rgb


# ---------------------------------------------------------------------------
# Loss


def _gauss_window(size=11, sigma=1.5):
    g = torch.exp(-((torch.arange(size) - size // 2) ** 2) / (2 * sigma**2))
    g = g / g.sum()
    w = g[:, None] * g[None, :]
    return w.expand(3, 1, size, size).contiguous()


_WIN = _gauss_window()


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b: [H,W,3] in [0,1]."""
    a = a.permute(2, 0, 1)[None]
    b = b.permute(2, 0, 1)[None]
    mu_a = F.conv2d(a, _WIN, groups=3)
    mu_b = F.conv2d(b, _WIN, groups=3)
    s_aa = F.conv2d(a * a, _WIN, groups=3) - mu_a**2
    s_bb = F.conv2d(b * b, _WIN, groups=3) - mu_b**2
    s_ab = F.conv2d(a * b, _WIN, groups=3) - mu_a * mu_b
    C1, C2 = 0.01**2, 0.03**2
    m = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a**2 + mu_b**2 + C1) * (s_aa + s_bb + C2))
    return m.mean()


# ---------------------------------------------------------------------------
# Model + optimizer (a small Adam we can resize when gaussians are added/removed)

PARAMS = ["means", "quats", "scales", "opacities", "sh0", "shN"]


class Model:
    def __init__(self, xyz: np.ndarray, rgb: np.ndarray, lrs: dict):
        n = len(xyz)
        xyz_t = torch.from_numpy(xyz).float()
        # Initial scale: mean distance to the 3 nearest neighbours.
        from scipy.spatial import cKDTree

        d, _ = cKDTree(xyz).query(xyz, k=4)
        dist = np.clip(d[:, 1:].mean(1), 1e-4, None)
        self.p = {
            "means": xyz_t,
            "quats": torch.cat([torch.ones(n, 1), torch.zeros(n, 3)], 1) + torch.randn(n, 4) * 0.01,
            "scales": torch.log(torch.from_numpy(dist).float())[:, None].repeat(1, 3),
            "opacities": torch.full((n,), math.log(0.1 / 0.9)),
            "sh0": (torch.from_numpy(rgb).float() - 0.5) / SH_C0,
            "shN": torch.zeros(n, 3, 3),
        }
        for v in self.p.values():
            v.requires_grad_(True)
        self.lrs = dict(lrs)
        self.m = {k: torch.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: torch.zeros_like(v) for k, v in self.p.items()}
        self.step_count = {k: 0 for k in self.p}

    def __len__(self):
        return len(self.p["means"])

    @torch.no_grad()
    def step(self, b1=0.9, b2=0.999, eps=1e-15):
        for k, p in self.p.items():
            if p.grad is None:
                continue
            g = p.grad
            self.step_count[k] += 1
            t = self.step_count[k]
            self.m[k].mul_(b1).add_(g, alpha=1 - b1)
            self.v[k].mul_(b2).addcmul_(g, g, value=1 - b2)
            mhat = self.m[k] / (1 - b1**t)
            vhat = self.v[k] / (1 - b2**t)
            p.sub_(self.lrs[k] * mhat / (vhat.sqrt() + eps))
            p.grad = None

    @torch.no_grad()
    def keep(self, mask: torch.Tensor):
        for k in PARAMS:
            self.p[k] = self.p[k][mask].detach().requires_grad_(True)
            self.m[k] = self.m[k][mask]
            self.v[k] = self.v[k][mask]

    @torch.no_grad()
    def append(self, new: dict):
        for k in PARAMS:
            self.p[k] = torch.cat([self.p[k].detach(), new[k]]).requires_grad_(True)
            self.m[k] = torch.cat([self.m[k], torch.zeros_like(new[k])])
            self.v[k] = torch.cat([self.v[k], torch.zeros_like(new[k])])

    def to_splats(self) -> Splats:
        p = {k: v.detach().numpy() for k, v in self.p.items()}
        q = p["quats"] / np.linalg.norm(p["quats"], axis=1, keepdims=True)
        return Splats(p["means"], p["sh0"], p["shN"], p["opacities"], p["scales"], q)


# ---------------------------------------------------------------------------
# Adaptive density control


class Densifier:
    def __init__(self, model: Model, scene_scale: float, args):
        self.model = model
        self.scale = scene_scale
        self.a = args
        self.reset_stats()

    def reset_stats(self):
        n = len(self.model)
        self.grad_accum = torch.zeros(n)
        self.count = torch.zeros(n)

    def accumulate(self, info, cam: Camera):
        idx = info["visible_idx"]
        g = info["absgrad_vis"].clone()
        g[:, 0] *= cam.width / 2.0
        g[:, 1] *= cam.height / 2.0
        on = info["radii"] > 0
        self.grad_accum.index_add_(0, idx[on], g[on].norm(dim=-1))
        self.count.index_add_(0, idx[on], torch.ones(int(on.sum())))

    @torch.no_grad()
    def refine(self, step: int):
        m, a = self.model, self.a
        n0 = len(m)
        avg = self.grad_accum / self.count.clamp(min=1)
        high = avg > a.grow_grad
        max_scale = torch.exp(m.p["scales"]).max(1).values
        small = max_scale <= a.grow_scale * self.scale

        budget = a.max_gaussians - n0
        if budget <= 0:
            high[:] = False
        elif int(high.sum()) > budget:
            # Keep only the strongest candidates.
            thr = torch.topk(avg[high], budget).values[-1]
            high &= avg >= thr

        clone = high & small
        split = high & ~small

        new = {}
        if clone.any():
            for k in PARAMS:
                new.setdefault(k, []).append(m.p[k][clone].detach().clone())
        if split.any():
            S = torch.exp(m.p["scales"][split])
            from .render import quat_to_rotmat

            R = quat_to_rotmat(m.p["quats"][split])
            for _ in range(2):
                offs = (R @ (S * torch.randn_like(S))[..., None])[..., 0]
                new.setdefault("means", []).append(m.p["means"][split] + offs)
                new.setdefault("scales", []).append(torch.log(S / 1.6))
                for k in ["quats", "opacities", "sh0", "shN"]:
                    new.setdefault(k, []).append(m.p[k][split].detach().clone())
        if new:
            m.append({k: torch.cat(v) for k, v in new.items()})
        n_added = len(m) - n0

        # Remove split originals, transparent and (after first reset) oversized gaussians.
        remove = torch.zeros(len(m), dtype=torch.bool)
        remove[:n0] |= split
        remove |= torch.sigmoid(m.p["opacities"]) < a.prune_opa
        if step > a.reset_every:
            remove |= torch.exp(m.p["scales"]).max(1).values > a.prune_scale * self.scale
        m.keep(~remove)
        self.reset_stats()
        return int(clone.sum()), int(split.sum()), int(remove.sum()), n_added

    @torch.no_grad()
    def reset_opacity(self):
        o = self.model.p["opacities"]
        o.clamp_(max=math.log(0.01 / 0.99))  # opacity <= 0.01
        self.model.m["opacities"].zero_()
        self.model.v["opacities"].zero_()


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap", required=True, help="undistorted COLMAP dir (images/, sparse_txt/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--width", type=int, default=640, help="training image width")
    ap.add_argument("--final-width", type=int, default=0, help="switch to this width for the last 30%% of steps")
    ap.add_argument("--max-gaussians", type=int, default=600_000)
    ap.add_argument("--grow-grad", type=float, default=0.0008)
    ap.add_argument("--grow-scale", type=float, default=0.01)
    ap.add_argument("--prune-opa", type=float, default=0.005)
    ap.add_argument("--prune-scale", type=float, default=0.1)
    ap.add_argument("--refine-every", type=int, default=100)
    ap.add_argument("--refine-start", type=int, default=300)
    ap.add_argument("--refine-stop", type=int, default=0, help="default: 60%% of iters")
    ap.add_argument("--reset-every", type=int, default=2000)
    ap.add_argument("--sh-start", type=int, default=1000, help="step at which SH degree 1 starts training")
    ap.add_argument("--holdout", type=int, default=8, help="every Nth image is held out for eval")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.refine_stop:
        args.refine_stop = int(args.iters * 0.6)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(args.out, exist_ok=True)

    cams, pixels, names, xyz, rgb = load_dataset(args.colmap, max(args.width, args.final_width))
    train_ids = [i for i in range(len(cams)) if args.holdout <= 0 or i % args.holdout != 0]
    test_ids = [i for i in range(len(cams)) if args.holdout > 0 and i % args.holdout == 0]
    centers = torch.stack([c.center for c in cams])
    scene_scale = float((centers - centers.mean(0)).norm(dim=1).max()) * 1.1
    print(f"{len(cams)} images ({len(train_ids)} train / {len(test_ids)} test), {len(xyz)} SfM points, "
          f"scene scale {scene_scale:.3f}", flush=True)

    base_lrs = {"means": 1.6e-4 * scene_scale, "quats": 1e-3, "scales": 5e-3, "opacities": 5e-2,
                "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    model = Model(xyz, rgb, base_lrs)
    dens = Densifier(model, scene_scale, args)

    def cam_at(i: int, width: int):
        c = cams[i]
        s = width / c.width
        return c.scaled(s) if abs(s - 1) > 1e-6 else c

    def target_at(i: int, cam: Camera):
        t = pixels[i]
        if t.shape[1] != cam.width:
            t = F.interpolate(t.permute(2, 0, 1)[None].float(), size=(cam.height, cam.width), mode="area")[0].permute(1, 2, 0)
            return t / 255.0
        return t.float() / 255.0

    log = []
    t0 = time.time()
    order: list[int] = []
    for step in range(1, args.iters + 1):
        width = args.final_width if args.final_width and step > 0.7 * args.iters else args.width
        # Exponential decay of the position learning rate (1.6e-4 -> 1.6e-6).
        model.lrs["means"] = base_lrs["means"] * (0.01 ** (step / args.iters))
        if not order:
            order = train_ids.copy()
            random.shuffle(order)
        i = order.pop()
        cam = cam_at(i, width)
        gt = target_at(i, cam)
        bg = torch.rand(3)
        shN = model.p["shN"] if step >= args.sh_start else model.p["shN"].detach() * 0
        img, info = render(model.p["means"], model.p["quats"], model.p["scales"], model.p["opacities"],
                           model.p["sh0"], shN, cam, background=bg, track_absgrad=True)
        gt_bg = gt  # real photos have no transparency; background only shows through holes
        l1 = (img - gt_bg).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - ssim(img, gt_bg))
        loss.backward()

        if step < args.refine_stop:
            dens.accumulate(info, cam)
        model.step()

        if args.refine_start <= step < args.refine_stop and step % args.refine_every == 0:
            c, s, r, a = dens.refine(step)
            print(f"  refine@{step}: clone {c}, split {s}, +{a}, pruned {r} -> {len(model)}", flush=True)
        if step % args.reset_every == 0 and step < args.refine_stop:
            dens.reset_opacity()

        if step % 50 == 0 or step == 1:
            el = time.time() - t0
            print(f"step {step}/{args.iters} loss {loss.item():.4f} l1 {l1.item():.4f} n={len(model)} "
                  f"{el / step:.2f}s/it eta {(args.iters - step) * el / step / 60:.0f}min", flush=True)
            log.append({"step": step, "loss": loss.item(), "n": len(model), "t": el})

        if step % 1000 == 0 or step == args.iters:
            write_ply(os.path.join(args.out, "point_cloud.ply"), model.to_splats())
            with torch.no_grad():
                if test_ids:
                    j = test_ids[(step // 1000) % len(test_ids)]
                    c = cam_at(j, args.width)
                    im, _ = render(model.p["means"], model.p["quats"], model.p["scales"], model.p["opacities"],
                                   model.p["sh0"], model.p["shN"], c)
                    both = torch.cat([im.clamp(0, 1), target_at(j, c)], 1)
                    Image.fromarray((both.numpy() * 255).astype(np.uint8)).save(os.path.join(args.out, f"eval_{step:05d}.jpg"))

    # Hold-out PSNR
    psnrs = []
    with torch.no_grad():
        for j in test_ids:
            c = cam_at(j, args.width)
            im, _ = render(model.p["means"], model.p["quats"], model.p["scales"], model.p["opacities"],
                           model.p["sh0"], model.p["shN"], c)
            mse = ((im.clamp(0, 1) - target_at(j, c)) ** 2).mean().item()
            psnrs.append(-10 * math.log10(max(mse, 1e-10)))
    summary = {"n_gaussians": len(model), "test_psnr": float(np.mean(psnrs)) if psnrs else None,
               "minutes": (time.time() - t0) / 60, "scene_scale": scene_scale, "log": log}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"done: {len(model)} gaussians, test PSNR {summary['test_psnr']}, {summary['minutes']:.0f} min")


if __name__ == "__main__":
    main()
