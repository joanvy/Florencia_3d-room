"""Videos -> sharp frames -> COLMAP camera poses (CPU only).

    python pipeline/prepare.py --videos work/videos --work work [--frames 280]

Steps
 1. ffmpeg: decode every video at a few fps (auto-rotated, long side 1600 px).
 2. Keep the sharpest frame in each short window (variance of Laplacian),
    spread evenly so the total is about --frames.
 3. COLMAP SIFT features (one shared camera per video).
 4. Matching on an explicit pair list: neighbours within a video, plus every
    k-th frame against every k-th frame of all videos (ties recordings together
    without paying for full exhaustive matching on CPU).
 5. Incremental mapping, keep the largest model, undistort to PINHOLE and
    export as text for the trainer (work/colmap/undistorted/{images,sparse_txt}).
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp")


def run(cmd: list[str]):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def sharpness(path: str) -> float:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def extract(videos: list[str], raw_dir: str, fps: float, long_side: int):
    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0].replace(" ", "_")
        out = os.path.join(raw_dir, name)
        os.makedirs(out, exist_ok=True)
        if glob.glob(os.path.join(out, "*.jpg")):
            continue
        scale = f"scale='if(gt(iw,ih),{long_side},-2)':'if(gt(iw,ih),-2,{long_side})'"
        run(["ffmpeg", "-loglevel", "error", "-i", v, "-vf", f"fps={fps},{scale}", "-q:v", "2",
             os.path.join(out, "f_%05d.jpg")])


def select(raw_dir: str, img_dir: str, total: int):
    groups = sorted(d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d)))
    counts = {g: len(glob.glob(os.path.join(raw_dir, g, "*.jpg"))) for g in groups}
    n_all = sum(counts.values())
    for g in groups:
        frames = sorted(glob.glob(os.path.join(raw_dir, g, "*.jpg")))
        want = max(8, round(total * counts[g] / n_all))
        window = max(1, len(frames) // want)
        os.makedirs(os.path.join(img_dir, g), exist_ok=True)
        kept = 0
        for s in range(0, len(frames), window):
            chunk = frames[s:s + window]
            best = max(chunk, key=sharpness)
            shutil.copy(best, os.path.join(img_dir, g, os.path.basename(best)))
            kept += 1
        print(f"{g}: {len(frames)} decoded -> {kept} sharpest kept", flush=True)


def pairs_file(img_dir: str, path: str, neighbours: int, stride: int):
    groups = sorted(d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d)))
    seqs = [[f"{g}/{f}" for f in sorted(os.listdir(os.path.join(img_dir, g)))] for g in groups]
    pairs = set()
    for seq in seqs:
        for i in range(len(seq)):
            for j in range(i + 1, min(len(seq), i + 1 + neighbours)):
                pairs.add((seq[i], seq[j]))
    keys = [im for seq in seqs for im in seq[::stride]]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pairs.add((keys[i], keys[j]))
    with open(path, "w") as f:
        for a, b in sorted(pairs):
            f.write(f"{a} {b}\n")
    print(f"{len(pairs)} image pairs to match", flush=True)


def largest_model(sparse_dir: str) -> str:
    best, best_n = None, -1
    for d in sorted(os.listdir(sparse_dir)):
        p = os.path.join(sparse_dir, d)
        txt = p + "_txt"
        os.makedirs(txt, exist_ok=True)
        run(["colmap", "model_converter", "--input_path", p, "--output_path", txt, "--output_type", "TXT"])
        with open(os.path.join(txt, "images.txt")) as f:
            n = sum(1 for l in f if not l.startswith("#")) // 2
        print(f"model {d}: {n} images registered", flush=True)
        if n > best_n:
            best, best_n = p, n
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--frames", type=int, default=280, help="approximate total frames kept")
    ap.add_argument("--fps", type=float, default=6)
    ap.add_argument("--long-side", type=int, default=1600)
    ap.add_argument("--features", type=int, default=4096)
    ap.add_argument("--neighbours", type=int, default=12)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    videos = sorted(p for p in glob.glob(os.path.join(args.videos, "*")) if p.lower().endswith(VIDEO_EXT))
    if not videos:
        sys.exit(f"no videos found in {args.videos}")
    raw = os.path.join(args.work, "frames_raw")
    col = os.path.join(args.work, "colmap")
    img = os.path.join(col, "images")
    db = os.path.join(col, "database.db")
    os.makedirs(raw, exist_ok=True)

    extract(videos, raw, args.fps, args.long_side)
    if not os.path.isdir(img):
        select(raw, img, args.frames)

    if not os.path.exists(db):
        run(["colmap", "feature_extractor", "--database_path", db, "--image_path", img,
             "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera_per_folder", "1",
             "--SiftExtraction.use_gpu", "0", "--SiftExtraction.max_num_features", str(args.features),
             "--SiftExtraction.estimate_affine_shape", "0", "--SiftExtraction.domain_size_pooling", "1"])
        pairs = os.path.join(col, "pairs.txt")
        pairs_file(img, pairs, args.neighbours, args.stride)
        run(["colmap", "matches_importer", "--database_path", db, "--match_list_path", pairs,
             "--match_type", "pairs", "--SiftMatching.use_gpu", "0", "--SiftMatching.guided_matching", "1"])

    sparse = os.path.join(col, "sparse")
    if not os.path.isdir(sparse) or not os.listdir(sparse):
        os.makedirs(sparse, exist_ok=True)
        run(["colmap", "mapper", "--database_path", db, "--image_path", img, "--output_path", sparse,
             "--Mapper.ba_global_function_tolerance", "0.000001", "--Mapper.num_threads", str(os.cpu_count() or 4)])

    model = largest_model(sparse)
    und = os.path.join(col, "undistorted")
    if os.path.isdir(und):
        shutil.rmtree(und)
    run(["colmap", "image_undistorter", "--image_path", img, "--input_path", model, "--output_path", und,
         "--output_type", "COLMAP"])
    txt = os.path.join(und, "sparse_txt")
    os.makedirs(txt, exist_ok=True)
    run(["colmap", "model_converter", "--input_path", os.path.join(und, "sparse"), "--output_path", txt,
         "--output_type", "TXT"])
    n_total = sum(len(files) for _, _, files in os.walk(img))
    with open(os.path.join(txt, "images.txt")) as f:
        n_reg = sum(1 for l in f if not l.startswith("#")) // 2
    print(f"registered {n_reg}/{n_total} frames -> {und}", flush=True)


if __name__ == "__main__":
    main()
