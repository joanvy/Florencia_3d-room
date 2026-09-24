#!/usr/bin/env bash
# Apply edits (crop, remove clutter, mirror), compress to SPZ for the web viewer.
#
#   pipeline/finish.sh [edits.json] [sh_degree]
set -euo pipefail
cd "$(dirname "$0")/.."

EDITS=${1:-pipeline/edits.json}
SH=${2:-1}

python3 pipeline/cleanup.py apply --ply work/aligned.ply --edits "$EDITS" --out work/final.ply
node pipeline/to_spz.mjs work/final.ply public/splats/room.spz "$SH"
ls -lh public/splats/room.spz
