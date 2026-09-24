#!/usr/bin/env bash
# Full pipeline: videos -> COLMAP -> CPU splat training -> align -> plan renders.
# After inspecting work/plan/*.png, write pipeline/edits.json and run finish.sh.
#
#   pipeline/run_all.sh <videos_dir> [iters] [train_width]
set -euo pipefail
cd "$(dirname "$0")/.."

VIDEOS=${1:?usage: pipeline/run_all.sh <videos_dir> [iters] [train_width]}
ITERS=${2:-7000}
WIDTH=${3:-640}

python3 pipeline/prepare.py --videos "$VIDEOS" --work work
(cd pipeline && python3 -m gs.train --colmap ../work/colmap/undistorted --out ../work/train \
  --iters "$ITERS" --width "$WIDTH")
python3 pipeline/cleanup.py align --ply work/train/point_cloud.ply --colmap work/colmap/undistorted --out work/aligned.ply
python3 pipeline/cleanup.py plan --ply work/aligned.ply --out work/plan
echo "Now inspect work/plan/*.png, write pipeline/edits.json, then run pipeline/finish.sh"
