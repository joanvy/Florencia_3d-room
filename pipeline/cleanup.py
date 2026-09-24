"""Post-process a trained splat into the web-ready room.

  align   : level the floor (Y up, floor at y=0), square the walls to the X/Z
            axes and scale to metres (ceiling height or camera height).
  plan    : render a top-down floor plan + labelled perspective views with a
            1 m grid, used to decide which boxes to edit.
  apply   : apply edits.json (crop, floaters, delete boxes + floor fill,
            mirror), budget the splat count, write the final PLY.

    python pipeline/cleanup.py align --ply work/train/point_cloud.ply --colmap work/colmap/undistorted --out work/aligned.ply
    python pipeline/cleanup.py plan  --ply work/aligned.ply --out work/plan
    python pipeline/cleanup.py apply --ply work/aligned.ply --edits pipeline/edits.json --out work/final.ply

edits.json (all coordinates in aligned metres, Y up):
{
  "room": {"min": [x,y,z], "max": [x,y,z]},          # crop; everything outside is dropped
  "keep_outside": [{"min": [...], "max": [...]}],       # e.g. the view through a window
  "remove": [{"name": "suitcase", "min": [...], "max": [...], "fill_floor": true}],
  "mirror": {"axis": "x", "value": 3.98, "min": [y0, z0], "max": [y1, z1], "mode": "reflect"},
                                                        # reflect: viewer renders a live reflection
                                                        # glass: bake a flat grey pane instead
  "max_splats": 900000
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splat_io import SH_C0, Splats, read_ply, write_ply  # noqa: E402

AXES = {"x": 0, "y": 1, "z": 2}


# ---------------------------------------------------------------------------
# Geometry helpers


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], 1)


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def quat_to_rotmats(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], 1).reshape(-1, 3, 3)


def transform(s: Splats, R: np.ndarray, t: np.ndarray, scale: float) -> Splats:
    """x' = scale * R x + t, applied to positions, orientations and sizes."""
    q = np.tile(rotmat_to_quat(R), (len(s), 1))
    # Degree-1 SH are a vector field in (y, z, x) order; rotate them with the scene.
    # colour_1 = C1 * (-y*sh[0] + z*sh[1] - x*sh[2]) = C1 * dir . v  with  v = (-sh[2], -sh[0], sh[1]).
    rest = s.f_rest.copy()
    if rest.shape[2] >= 3:
        v = np.stack([-rest[..., 2], -rest[..., 0], rest[..., 1]], -1)  # [N,3ch,xyz]
        v = np.einsum("ij,ncj->nci", R, v)
        rest[..., 0], rest[..., 1], rest[..., 2] = -v[..., 1], v[..., 2], -v[..., 0]
    return Splats(
        xyz=(scale * s.xyz @ R.T + t).astype(np.float32),
        f_dc=s.f_dc,
        f_rest=rest.astype(np.float32),
        opacity=s.opacity,
        scale=(s.scale + np.log(scale)).astype(np.float32),
        rot=quat_mul(q, s.rot).astype(np.float32),
    )


def normals(s: Splats) -> np.ndarray:
    """Normal of each splat = its shortest axis (meaningful for flat splats)."""
    R = quat_to_rotmats(s.rot)
    k = np.argmin(s.scale, 1)
    return R[np.arange(len(s)), :, k]


def flatness(s: Splats) -> np.ndarray:
    sc = np.sort(s.scale, 1)
    return sc[:, 1] - sc[:, 0]  # log-ratio of middle to smallest axis


def ransac_plane(pts: np.ndarray, n_hint: np.ndarray, iters=2000, thr=0.02, max_angle_deg=25, rng=None):
    rng = rng or np.random.default_rng(0)
    best, best_count = None, -1
    cos_max = np.cos(np.radians(max_angle_deg))
    for _ in range(iters):
        p = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if abs(n @ n_hint) < cos_max:
            continue
        d = -n @ p[0]
        count = int((np.abs(pts @ n + d) < thr).sum())
        if count > best_count:
            best, best_count = (n, d), count
    n, d = best
    inl = np.abs(pts @ n + d) < thr
    # Refine with least squares on inliers.
    c = pts[inl].mean(0)
    _, _, vt = np.linalg.svd(pts[inl] - c, full_matrices=False)
    n = vt[2] * np.sign(vt[2] @ n)
    return n, -n @ c, inl


# ---------------------------------------------------------------------------
# align


def cmd_align(args):
    from gs.colmap_io import read_model

    s = read_ply(args.ply)
    _, images, _, _ = read_model(os.path.join(args.colmap, "sparse_txt"))
    centers = np.array([im.center for im in images])
    # Phones are held roughly upright: camera -y (OpenCV) averages to world up.
    up = -np.mean([im.R.T @ np.array([0, 1, 0]) for im in images], 0)
    up /= np.linalg.norm(up)

    solid = s.alpha > 0.5
    pts = s.xyz[solid]
    span = np.percentile(np.linalg.norm(pts - np.median(pts, 0), axis=1), 90)
    rng = np.random.default_rng(0)
    sample = pts[rng.choice(len(pts), min(len(pts), 60000), replace=False)]

    # Orientation: the dominant horizontal plane (floor, bed top and tables are all parallel).
    n_floor, _, _ = ransac_plane(sample, up, thr=0.01 * span, rng=rng)
    if n_floor @ up < 0:
        n_floor = -n_floor
    # Heights along that normal: the floor is the lowest strong peak, the ceiling the highest.
    h = pts @ n_floor
    bin_w = 0.005 * span
    hist, edges = np.histogram(h, bins=np.arange(h.min(), h.max() + bin_w, bin_w))
    hist = np.convolve(hist, np.ones(5) / 5, "same")
    peaks = [k for k in range(1, len(hist) - 1)
             if hist[k] >= hist[k - 1] and hist[k] >= hist[k + 1] and hist[k] >= 0.25 * hist.max()]
    centre = lambda k: (edges[k] + edges[k + 1]) / 2  # noqa: E731
    floor_h = centre(peaks[0])
    d_floor = -floor_h
    cam_h = np.median(centers @ n_floor)
    ceiling_h = None
    if centre(peaks[-1]) > cam_h + 0.05 * span:
        ceiling_h = centre(peaks[-1]) - floor_h
    print(f"height peaks (rel. floor): {[round(centre(k) - floor_h, 3) for k in peaks]}")

    # Rotation taking floor normal -> +Y.
    y = n_floor
    # Wall directions: histogram of horizontal normal angles of vertical flat splats (mod 90 deg).
    nrm = normals(s)[solid & (flatness(s) > 1.0)]
    horiz = nrm - np.outer(nrm @ y, y)
    hn = np.linalg.norm(horiz, axis=1)
    vertical = hn > 0.95
    a_ref = np.cross(y, [1, 0, 0] if abs(y[0]) < 0.9 else [0, 0, 1])
    a_ref /= np.linalg.norm(a_ref)
    b_ref = np.cross(y, a_ref)
    hv = horiz[vertical] / hn[vertical, None]
    ang = np.arctan2(hv @ b_ref, hv @ a_ref) % (np.pi / 2)
    hist, edges = np.histogram(ang, bins=180, range=(0, np.pi / 2))
    hist = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(7) / 7, "valid")
    theta = edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2
    x = np.cos(theta) * a_ref + np.sin(theta) * b_ref
    z = np.cross(x, y)
    R = np.stack([x, y, z])  # rows: new axes expressed in old coords

    # Metric scale.
    cam_height = float(np.median(centers @ n_floor + d_floor))
    if ceiling_h and args.ceiling > 0:
        scale = args.ceiling / ceiling_h
        how = f"ceiling {ceiling_h:.3f} units -> {args.ceiling} m"
    else:
        scale = args.camera_height / cam_height
        how = f"median camera height {cam_height:.3f} units -> {args.camera_height} m"

    t = np.zeros(3)
    out = transform(s, R, t, scale)
    out.xyz[:, 1] += scale * d_floor  # floor plane -> y = 0
    # Put the room's footprint corner near the origin.
    solid_xyz = out.xyz[(out.alpha > 0.5)]
    lo = np.percentile(solid_xyz, 2, axis=0)
    out.xyz[:, 0] -= lo[0]
    out.xyz[:, 2] -= lo[2]
    write_ply(args.out, out)

    cams = (scale * centers @ R.T)
    cams[:, 1] += scale * d_floor
    cams[:, 0] -= lo[0]
    cams[:, 2] -= lo[2]
    meta = {"scale": scale, "scale_from": how, "R": R.tolist(), "floor_offset": float(scale * d_floor),
            "shift": [float(-lo[0]), 0.0, float(-lo[2])], "cameras": cams.tolist()}
    with open(os.path.splitext(args.out)[0] + ".align.json", "w") as f:
        json.dump(meta, f)
    hi = np.percentile(solid_xyz, 98, axis=0) - [lo[0], 0, lo[2]]
    print(f"aligned: {how}; room approx {hi[0]:.2f} x {hi[2]:.2f} m, top {hi[1]:.2f} m; {len(out)} splats")


# ---------------------------------------------------------------------------
# plan (inspection renders)


def cmd_plan(args):
    import torch
    from PIL import Image, ImageDraw

    from gs.render import Camera, look_at, render

    s = read_ply(args.ply)
    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.splitext(args.ply)[0] + ".align.json"
    cams = np.array(json.load(open(meta_path))["cameras"]) if os.path.exists(meta_path) else None

    solid = s.alpha > 0.3
    lo = np.percentile(s.xyz[solid], 1, axis=0)
    hi = np.percentile(s.xyz[solid], 99, axis=0)

    def to_t(sub: Splats):
        return {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in
                dict(m=sub.xyz, q=sub.rot, s=sub.scale, o=sub.opacity, c=sub.f_dc).items()}

    def draw(sub, cam, path, grid=None):
        t = to_t(sub)
        with torch.no_grad():
            img, _ = render(t["m"], t["q"], t["s"], t["o"], t["c"], None, cam, background=torch.tensor([0.1, 0.1, 0.1]))
        im = Image.fromarray((img.clamp(0, 1).numpy() * 255).astype(np.uint8))
        if grid:
            grid(ImageDraw.Draw(im))
        im.save(path)

    # Top-down plan of everything below a height cut (removes the ceiling).
    for cut in args.cuts:
        sub = s.subset(s.xyz[:, 1] < cut)
        cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
        extent = max(hi[0] - lo[0], hi[2] - lo[2]) * 1.15
        W = H = 900
        height = 40.0  # far above => near-orthographic
        f = W / extent * height
        eye = np.array([cx, height, cz])
        cam = Camera(look_at(eye, [cx, 0, cz], up=(0, 0, -1)), f, f, W / 2, H / 2, W, H)

        def grid(d, cx=cx, cz=cz, f=f):
            def px(x, z):
                return (W / 2 + (x - cx) * f / height, H / 2 + (z - cz) * f / height)
            for gx in np.arange(np.floor(lo[0]) - 1, hi[0] + 1.01, 0.5):
                d.line([px(gx, lo[2] - 1), px(gx, hi[2] + 1)], fill=(255, 60, 60) if gx % 1 == 0 else (120, 40, 40))
                if gx % 1 == 0:
                    d.text(px(gx + 0.02, lo[2] - 0.3), f"x={gx:.0f}", fill=(255, 120, 120))
            for gz in np.arange(np.floor(lo[2]) - 1, hi[2] + 1.01, 0.5):
                d.line([px(lo[0] - 1, gz), px(hi[0] + 1, gz)], fill=(60, 160, 255) if gz % 1 == 0 else (30, 70, 120))
                if gz % 1 == 0:
                    d.text(px(lo[0] - 0.4, gz + 0.02), f"z={gz:.0f}", fill=(120, 180, 255))
            if cams is not None:
                for c in cams[::3]:
                    x, y = px(c[0], c[2])
                    d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 0))
        draw(sub, cam, os.path.join(args.out, f"plan_below_{cut:.1f}m.png"), grid)

    # Perspective views from the room centre towards each wall and corner, at eye height.
    c = np.array([(lo[0] + hi[0]) / 2, 1.5, (lo[2] + hi[2]) / 2])
    W, H = 960, 640
    for k, a in enumerate(np.arange(0, 360, 45)):
        tgt = c + [np.cos(np.radians(a)), -0.35, np.sin(np.radians(a))]
        cam = Camera(look_at(c, tgt), 0.45 * W, 0.45 * W, W / 2, H / 2, W, H)
        draw(s, cam, os.path.join(args.out, f"view_{k}_{int(a):03d}deg.png"))
    print(f"wrote plan renders to {args.out}; room bbox (1-99%): {np.round(lo, 2)} .. {np.round(hi, 2)}")


# ---------------------------------------------------------------------------
# apply


def in_box(xyz, b):
    return np.all((xyz >= np.asarray(b["min"])) & (xyz <= np.asarray(b["max"])), 1)


def fill_floor(s: Splats, box: dict, rng, spacing=0.012) -> Splats:
    """Cover the floor footprint of a removed object with flat splats whose colours
    are copied from the floor just outside the box (nearest-neighbour inpainting)."""
    from scipy.spatial import cKDTree

    lo, hi = np.asarray(box["min"]), np.asarray(box["max"])
    floor = (np.abs(s.xyz[:, 1]) < 0.03) & (s.alpha > 0.4)
    ring = floor & ~in_box(s.xyz, {"min": lo - [0.35, 1, 0.35], "max": hi + [0.35, 1, 0.35]}) & \
        in_box(s.xyz, {"min": lo - [0.8, 1, 0.8], "max": hi + [0.8, 1, 0.8]})
    src = s.subset(ring)
    if len(src) < 20:
        print(f"  warning: little floor around {box.get('name')}, skipping fill")
        return None
    xs = np.arange(lo[0], hi[0], spacing)
    zs = np.arange(lo[2], hi[2], spacing)
    gx, gz = np.meshgrid(xs, zs)
    pts = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], 1)
    pts[:, [0, 2]] += rng.uniform(-spacing / 2, spacing / 2, (len(pts), 2))
    # Mirror each point across the nearest box edge so texture continues naturally.
    tree = cKDTree(src.xyz[:, [0, 2]])
    reflect = pts.copy()
    dl, dh = pts[:, [0, 2]] - lo[[0, 2]], hi[[0, 2]] - pts[:, [0, 2]]
    for i, ax in enumerate([0, 2]):
        near_lo = dl[:, i] < dh[:, i]
        reflect[:, ax] = np.where(near_lo, lo[ax] - dl[:, i] - 0.35, hi[ax] + dh[:, i] + 0.35)
    use_x = np.minimum(dl[:, 0], dh[:, 0]) < np.minimum(dl[:, 1], dh[:, 1])
    probe = np.where(use_x[:, None], np.stack([reflect[:, 0], pts[:, 2]], 1), np.stack([pts[:, 0], reflect[:, 2]], 1))
    _, j = tree.query(probe, k=4)
    f_dc = src.f_dc[j].mean(1)
    n = len(pts)
    q = np.tile([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0], (n, 1))  # flat axis (z) -> world Y
    return Splats(
        xyz=pts.astype(np.float32),
        f_dc=f_dc.astype(np.float32),
        f_rest=np.zeros((n, 3, s.f_rest.shape[2]), np.float32),
        opacity=np.full(n, 4.0, np.float32),
        scale=np.log(np.tile([spacing * 0.9, spacing * 0.9, 0.001], (n, 1))).astype(np.float32),
        rot=q.astype(np.float32),
    )


def concat(a: Splats, b: Splats) -> Splats:
    return Splats(*(np.concatenate([getattr(a, k), getattr(b, k)]) for k in
                    ["xyz", "f_dc", "f_rest", "opacity", "scale", "rot"]))


def cmd_apply(args):
    from scipy.spatial import cKDTree

    s = read_ply(args.ply)
    e = json.load(open(args.edits))
    rng = np.random.default_rng(0)
    n0 = len(s)
    log = []

    # 1. Crop to the room (+ explicit keep-outside regions such as a window view).
    if "room" in e:
        keep = in_box(s.xyz, e["room"])
        for b in e.get("keep_outside", []):
            keep |= in_box(s.xyz, b)
        s = s.subset(keep)
        log.append(f"crop: {n0 - len(s)} removed")

    # 2. Floaters: nearly transparent, huge, or isolated splats.
    n1 = len(s)
    big = np.exp(s.scale).max(1) > e.get("max_splat_size", 0.5)
    faint = s.alpha < e.get("min_alpha", 0.02)
    d, _ = cKDTree(s.xyz).query(s.xyz, k=9)
    lonely = d[:, -1] > np.percentile(d[:, -1], 99.5) * 1.5
    s = s.subset(~(big | faint | lonely))
    log.append(f"floaters: {n1 - len(s)} removed")

    # 3. Objects to delete (clothes, suitcases...), optionally re-growing the floor under them.
    fills = []
    for b in e.get("remove", []):
        m = in_box(s.xyz, b)
        s = s.subset(~m)
        log.append(f"remove {b.get('name', '?')}: {int(m.sum())} splats")
        if b.get("fill_floor"):
            f = fill_floor(s, b, rng)
            if f is not None:
                fills.append(f)
    for f in fills:
        s = concat(s, f)
        log.append(f"floor fill: +{len(f)} splats")

    # 4. Mirror: the reconstruction puts a phantom 'reflected room' behind the glass.
    mir = e.get("mirror")
    if mir:
        ax = AXES[mir["axis"]]
        others = [i for i in range(3) if i != ax]
        v = mir["value"]
        inside_sign = np.sign(np.median(s.xyz[:, ax]) - v)  # which side the real room is on
        behind = (s.xyz[:, ax] - v) * inside_sign < -0.01
        lo2, hi2 = np.asarray(mir["min"]), np.asarray(mir["max"])
        # The phantom room is seen through the glass, so it spreads outward with depth:
        # remove everything behind the wall plane within a generous frustum of the mirror.
        depth = np.abs(s.xyz[:, ax] - v)
        grow = depth * mir.get("spread", 1.5)
        within = np.all((s.xyz[:, others] >= lo2 - grow[:, None]) & (s.xyz[:, others] <= hi2 + grow[:, None]), 1)
        phantom = behind & within
        s = s.subset(~phantom)
        log.append(f"mirror phantom room: {int(phantom.sum())} removed")
        # Whatever was reconstructed on the glass itself goes too: in "reflect" mode the
        # viewer draws a live reflection there, and nothing may sit between the
        # reflected camera and the glass.
        pane = (np.abs(s.xyz[:, ax] - v) < 0.03) & np.all((s.xyz[:, others] >= lo2) & (s.xyz[:, others] <= hi2), 1)
        s = s.subset(~pane)
        log.append(f"mirror pane: {int(pane.sum())} removed")
        if mir.get("mode", "reflect") == "glass":
            # Replace with a flat, slightly blue-grey reflective-looking pane.
            sp = 0.01
            a = np.arange(lo2[0], hi2[0], sp)
            b = np.arange(lo2[1], hi2[1], sp)
            ga, gb = np.meshgrid(a, b)
            n = ga.size
            pts = np.zeros((n, 3))
            pts[:, ax] = v
            pts[:, others[0]] = ga.ravel()
            pts[:, others[1]] = gb.ravel()
            # Soft vertical gradient + faint diagonal sheen reads as glass.
            u = (gb.ravel() - lo2[1]) / max(hi2[1] - lo2[1], 1e-6)
            w = (ga.ravel() - lo2[0]) / max(hi2[0] - lo2[0], 1e-6)
            base = np.array(mir.get("color", [0.55, 0.60, 0.64]))
            sheen = 0.08 * np.exp(-((u - w * 0.8 - 0.1) ** 2) / 0.01)
            rgb = np.clip(base[None] * (0.85 + 0.2 * u[:, None]) + sheen[:, None], 0, 1)
            normal_axis_quat = {0: [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0],
                                1: [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0],
                                2: [1, 0, 0, 0]}[ax]
            glass = Splats(
                xyz=pts.astype(np.float32),
                f_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
                f_rest=np.zeros((n, 3, s.f_rest.shape[2]), np.float32),
                opacity=np.full(n, 4.0, np.float32),
                scale=np.log(np.tile([sp * 0.9, sp * 0.9, 0.001], (n, 1))).astype(np.float32),
                rot=np.tile(normal_axis_quat, (n, 1)).astype(np.float32),
            )
            s = concat(s, glass)
            log.append(f"mirror glass: +{n} splats")

    # 5. Budget for phones: drop the least important splats (opacity x footprint).
    cap = e.get("max_splats", 900_000)
    if len(s) > cap:
        imp = s.alpha * np.exp(s.scale).prod(1) ** (1 / 3)
        keep = np.argsort(-imp)[:cap]
        s = s.subset(np.sort(keep))
        log.append(f"budget: trimmed to {cap}")

    write_ply(args.out, s)
    print("\n".join(log))
    print(f"{n0} -> {len(s)} splats -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("align")
    a.add_argument("--ply", required=True)
    a.add_argument("--colmap", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--ceiling", type=float, default=2.7, help="assumed ceiling height in m (0 = use camera height)")
    a.add_argument("--camera-height", type=float, default=1.45)
    p = sub.add_parser("plan")
    p.add_argument("--ply", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cuts", type=float, nargs="+", default=[1.2, 2.2])
    x = sub.add_parser("apply")
    x.add_argument("--ply", required=True)
    x.add_argument("--edits", required=True)
    x.add_argument("--out", required=True)
    args = ap.parse_args()
    {"align": cmd_align, "plan": cmd_plan, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
