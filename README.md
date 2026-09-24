# Florencia · Hotel Room 3D

A walk-around 3D model of a hotel room in Florence, captured on a phone and
served as a Gaussian splat that runs on phones.

- **Viewer**: Vite + three.js + [Spark](https://sparkjs.dev) (`src/`). Viewpoint
  buttons, touch controls, camera kept inside the room, render-on-demand,
  adaptive resolution, and a real-time planar **mirror** reflection.
- **Pipeline** (`pipeline/`): phone videos → sharp frames → COLMAP poses →
  Gaussian splat training on **CPU** (custom C++ rasterizer) → cleanup
  (level/scale, crop, remove clutter, mirror) → `.spz` for the web.
- **Hosting**: Vercel, static build (`vercel.json`).

## Viewer

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # -> dist/
```

`public/scene.json` configures the splat file, viewpoints, camera bounds and
the mirror. Coordinates are metres, Y up, floor at y = 0.

## Pipeline

Needs `ffmpeg`, `colmap` (CPU build is fine), Python 3 with `torch numpy scipy
pillow opencv-python-headless plyfile ninja`, and Node.

```bash
# 1. videos -> poses -> trained splat -> aligned + plan renders
pipeline/run_all.sh path/to/videos 7000 640
# 2. look at work/plan/*.png, write pipeline/edits.json (see cleanup.py docstring)
# 3. apply edits and compress
pipeline/finish.sh pipeline/edits.json 1
```

Tests: `python pipeline/tests/test_rasterize.py` checks the rasterizer's
forward/backward against a dense PyTorch reference;
`pipeline/tests/synth_capture.py` renders fake videos of a synthetic room to
exercise the whole chain.

### Room-specific decisions

- **Clothes and suitcases on the floor**: removed with boxes in `edits.json`
  (`remove`, `fill_floor: true` re-grows the floor under them from the
  surrounding floor texture).
- **Person in the bed**: kept on purpose.
- **Big mirror**: splatting reconstructs a mirror as a fake "room behind the
  glass" (which would also show whoever was filming). The pipeline deletes that
  phantom room and the glass (`mirror.mode: "reflect"`), and the viewer draws a
  true reflection in its place (`scene.json` → `mirror`).

## Status / hand-off

- [x] Viewer, mirror, phone tuning — tested headless at Pixel 7 size.
- [x] CPU trainer — gradients verified; synthetic room reconstructs.
- [x] Video → COLMAP — 170/180 synthetic frames registered.
- [x] Cleanup tools — alignment, plan renders, edits.
- [ ] **Get the 4 videos** from Google Drive folder `Florencia_3d-room`
      (Drive connector needed re-auth, and Drive download hosts were blocked by
      the session's network policy). Put them in `work/videos/` and run step 1.
- [ ] Write `pipeline/edits.json` from the plan renders, run step 3, set
      viewpoints/bounds/mirror in `public/scene.json`.
- [ ] Vercel: import this repo at vercel.com/new (framework: Vite).
