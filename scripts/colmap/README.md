# COLMAP sparse reconstruction from video

## Result

`IMG_4425.mp4` (6.1 s, 1920x1080, 30 fps) reconstructed successfully.

| | |
|---|---|
| Frames used | 61 (10 fps, downscaled to 1600 px wide) |
| Registered | 61 / 61 |
| Sparse points | 18,075 |
| Wall time | 13m19s |
| Point cloud | `sfm_out/sparse_points.ply` (265 KB, binary PLY, XYZ + RGB) |
| COLMAP model | `sfm_out/sparse/0/` (cameras / images / points3D `.bin`) |

## Machine assessment

Yes, this machine handles sparse SfM on short clips fine.

- **CPU** 4 cores — the bottleneck. Bundle adjustment and mapping are CPU-bound and
  accounted for most of the 13 minutes.
- **RAM** 9 GB — plenty here; a few hundred frames would still be OK.
- **GPU** GTX 1050 Ti, 4 GB VRAM, driver 560.94, CUDA works in-container.
  SIFT extraction and matching ran on it. 4 GB is *not* enough for comfortable
  dense/MVS at 1080p — stick to sparse, or run dense at heavily reduced resolution.
- **Disk** 941 GB free.

Rough scaling: cost grows super-linearly with frame count under `--exhaustive`
(pairs = n²/2). At ~200+ frames switch to sequential matching (drop the
`--exhaustive` flag) or expect hours.

## Two gotchas that shaped the scripts

1. **GPU is invisible to the container under plain `--nv` on WSL2.** `nvidia-smi`
   inside reports *"GPU access blocked by the operating system"*. WSL puts its
   driver shims in `/usr/lib/wsl/lib`, which `--nv` does not know about. Fix,
   already in `run.sh`:

   ```
   apptainer exec --nv \
     --bind /usr/lib/wsl:/usr/lib/wsl \
     --env LD_LIBRARY_PATH=/usr/lib/wsl/lib \
     colmap_pycolmap_cuda124.sif ...
   ```

2. **The `.sif` has no ffmpeg.** So `run.sh` extracts frames on the host with the
   system ffmpeg, then calls `sfm.py --no-extract` inside the container against
   the resulting folder.

## Usage

```bash
cd colmap_tests
./run.sh IMG_4425.mp4 --fps 10 --exhaustive
```

Flags:

| Flag | Default | Notes |
|---|---|---|
| `--fps` | 10 | frames extracted per second of video |
| `--max-width` | 1600 | frames downscaled to this width |
| `--workdir` | `sfm_out` | wiped and rebuilt each run |
| `--exhaustive` | off | match all pairs; good under ~200 frames |
| `--overlap` | 15 | sequential match window when not exhaustive |

If reconstruction fails or registers few images, raise `--fps` (more overlap
between frames) before anything else.

## Reprojection overlay video

`overlay.sh` + `overlay.py` project the sparse cloud back through each recovered
camera and draw it over the frame that camera came from, then encode an mp4. It
is the quickest visual check that the poses and the geometry actually agree — if
the reconstruction were wrong, the dots would swim against the footage instead of
sticking to surfaces.

```bash
./overlay.sh                       # -> sfm_out/overlay.mp4
./overlay.sh --side-by-side --out cmp.mp4
```

Current output: `sfm_out/overlay.mp4`, 61 frames, 1600x900, 10 fps, 24 MB. The
file is large for 6 seconds because thousands of hard-edged dots are exactly the
high-frequency detail h.264 spends bits on; raise `--crf` in the script if you
want it smaller.

| Flag | Default | Notes |
|---|---|---|
| `--color` | `depth` | `depth` (turbo by inverse depth), `rgb` (each point's own colour), `green` |
| `--radius` | 2 | dot radius in pixels |
| `--alpha` | 0.85 | dot opacity |
| `--observed-only` | off | draw only the points actually tracked in that frame, not the whole cloud |
| `--side-by-side` | off | original and overlay stacked horizontally |
| `--fps` | 10 | must match the `--fps` used for `run.sh` |
| `--out` | `sfm_out/overlay.mp4` | |

Note there is no depth test — the cloud is sparse, so points on hidden surfaces
still draw over foreground objects. That is normal for this kind of check.

How it is wired: the container has numpy but no ffmpeg, cv2 or PIL, while the
host has ffmpeg but a bare Python. So host ffmpeg decodes to raw rgb24 on a pipe,
`overlay.py` in the container draws with plain numpy array writes and pipes raw
frames back, and host ffmpeg encodes. Nothing large is written to disk in between.

## Blender scene (camera + footage + points)

`to_blender.sh` builds a ready-to-open `.blend` containing the animated camera,
the footage as an image plane, and the point cloud. This is the thing to use for
tweaking the camera path — the bare PLY below is only if you want points alone.

```bash
./to_blender.sh                      # -> sfm_out/colmap_scene.blend
```

Then open it (Windows Blender 5.2.1, verified working from WSL over the share):

```
"C:\Program Files\Blender Foundation\Blender 5.2\blender-launcher.exe" \
  "colmap_scene.blend"
```

### What is in the file

Everything sits in a `COLMAP` collection:

| Object | What it is |
|---|---|
| `Camera` | one keyframe per solved frame, scene frames 1-61, LINEAR interpolation |
| `Footage` | image plane parented to the camera, showing the frame that camera saw |
| `SparsePoints` | 18,075 points with photo colours, rendered via a Mesh-to-Points modifier |
| `CameraPath` | polyline through the camera centres, for reference while editing |
| `COLMAP_Root` | empty parent; move/rotate/scale it to place the whole solve |

Camera intrinsics are converted properly: `lens` from the focal length against a
36 mm sensor, `shift_x`/`shift_y` from the principal point, `pixel_aspect_y` if
pixels are non-square, and clip planes sized to the scene (the default 100 m
`clip_end` would have cut off the backdrop).

### Three things it does that matter

1. **Undistortion.** Blender's camera has no radial-distortion term, so a
   SIMPLE_RADIAL model would leave the footage misaligned by ~20 px at the
   corners. The exporter runs COLMAP's undistorter first and works from the
   resulting PINHOLE model and rectified frames, written to
   `sfm_out/undistorted/`. Pass `--no-undistort` to skip it.
2. **The plane is a backdrop, not a mid-scene plane.** It sits just beyond the
   farthest point. Placed at mid-depth it would occlude most of the cloud, and
   since the alignment is a pure projection, distance costs nothing.
3. **The solve is stood upright.** COLMAP's world axes are arbitrary — there is
   no gravity in a reconstruction. The exporter estimates world up from the mean
   camera orientation (handheld footage is held roughly level) and puts that
   correction on `COLMAP_Root`, leaving the poses and points at their original
   coordinates. Verified: the trunk is vertical and the camera path is level to
   within 0.5 units. `--no-align` disables it.

### Tweaking the camera path

The camera carries 61 location + rotation keyframes. To change the move:

- **Small adjustments** — scrub to a frame, move the camera, and it re-keys.
- **A different move entirely** — select the camera, `Object > Animation > Clear
  Keyframes`, then animate freely. The points and footage plane stay put; the
  plane keeps riding the camera, so you always see the footage from wherever you
  point it.
- **Smoothing the solved path** — the keys are LINEAR on purpose (solved poses
  are measurements, and bezier handles overshoot between them). Select all keys
  in the Graph Editor and set Bezier if you want it eased.

Scale is arbitrary — SfM recovers geometry only up to an unknown factor. Scale
`COLMAP_Root` to match whatever else is in your scene.

### Files

| File | Runs where | Does what |
|---|---|---|
| `export_scene.py` | container | model → `sfm_out/blender/scene.json`, poses already in Blender convention |
| `blender_import.py` | Blender | scene.json → objects; also works pasted into the Scripting tab |
| `to_blender.sh` | WSL | runs both, translating paths with `wslpath` |

`--json-only` stops after the export if you want to drive Blender yourself.

## Batch runs

`batch.sh` runs the whole chain (SfM -> overlay mp4 -> .blend) over several
videos, one workdir each under `out/`, so runs never clobber each other. A
failure on one video is logged and the batch moves on to the next.

```bash
./batch.sh IMG_4380.mp4 IMG_4424.mp4 IMG_4426.mp4      # FPS=10 MATCH=--exhaustive
```

Per video you get `out/<stem>/` containing `images/`, `sparse/0/`,
`sparse_points.ply`, `overlay.mp4`, `undistorted/`, `blender/scene.json` and
`colmap_scene.blend`, plus a full log at `out/<stem>.log`.

### Results

All four clips solved with every frame registered. Point counts track how much
texture a scene has, not how well it solved -- the open park is full of distinct
bark and leaf-litter detail, the dim forest interiors much less so.

| Clip | Length | Frames @ 10 fps | Registered | Points | Wall time |
|---|---|---|---|---|---|
| `IMG_4425` | 6.1 s | 61 | 61 / 61 | 18,075 | 13 min |
| `IMG_4380` | 10.9 s | 109 | 109 / 109 | 56,207 | 38 min |
| `IMG_4424` | 12.5 s | 125 | 125 / 125 | 23,224 | 20 min |
| `IMG_4426` | 6.5 s | 65 | 65 / 65 | 10,216 | 34 min |

Each `overlay.mp4` was checked at frame 30: the dots sit on trunks and ground
litter rather than drifting, so poses and geometry agree in all three new solves.

Open any of them with:

```
"C:\Program Files\Blender Foundation\Blender 5.2\blender-launcher.exe" "colmap_scene.blend"
```

### One thing this batch exposed

`IMG_4426` split into **two** models: a 2-image fragment written to `sparse/0`
and the real 65-image solve in `sparse/1`. Incremental mapping does that when a
few frames cannot be connected to the rest. `sfm.py` already picked the largest
model for the PLY, but `overlay.sh` and `to_blender.sh` both read `sparse/0`, so
they would have visualised the 2-frame fragment. `sfm.py` now swaps the best
model into `sparse/0` and says so in its output, which keeps every downstream
script reading one well-known path.

## Conditioning sequences for video diffusion

`cond.sh` builds the 61-frame guidance clip: frames 1 and 61 are the real
footage, frames 2-60 are the sparse cloud rendered through the solved camera as
flat coloured squares on black. That is the test -- can the model invent the
in-between frames from two real anchors plus a moving 3D scaffold.

```bash
./cond.sh --workdir out/IMG_4380          # 832x480, 61 frames, 4 px squares
```

Outputs, per clip:

| Path | What |
|---|---|
| `<workdir>/cond/frames/frame_0001.png` .. `frame_0061.png` | the sequence, left on disk |
| `<workdir>/cond/cond.mp4` | the same frames encoded at the solve fps |
| `<workdir>/cond_scene.blend` | the scene it was rendered from, for tweaking the camera |

| Flag | Default | Notes |
|---|---|---|
| `--workdir` | `sfm_out` | must already contain `blender/scene.json` |
| `--width` / `--height` | 832 / 480 | |
| `--frames` | 61 | uses the first 61 solved frames |
| `--point-px` | 4 | on-screen square size, in pixels |
| `--span` | off | spread the 61 frames across the whole clip instead of taking the first 61 |
| `--fps` | from the solve | |

### Four details that had to be right

1. **Constant on-screen size.** A point cloud with a fixed world radius shrinks
   with distance, which encodes depth twice and makes far detail vanish. Each
   square is instead scaled by its own distance along the view axis,
   `size = point_px * depth / focal_px`, so every point lands on 4 px wherever
   it is. The depth is evaluated in geometry nodes per frame, so it keeps
   working when the camera moves.
2. **Actually square, not diamond.** The quads are instanced with the rotation
   of the camera itself, so they stay parallel to the image plane and project to
   axis-aligned squares rather than foreshortened diamonds.
3. **832x480 is not 16:9.** Rather than crop, the render is squeezed to the
   target shape and the two real frames are squeezed by exactly the same
   `scale=832:480`, so the whole field of view survives and the anchors still
   line up with the points pixel for pixel. Getting there needs one Blender
   quirk: both pixel aspect values are clamped to >= 1, so a ratio below one has
   to be applied by widening X instead of narrowing Y. Set `pixel_aspect_y` to
   0.97 and Blender silently stores 1.0.
4. **Colour passes through untouched.** View transform is `Standard`, not AgX,
   and the point colours are converted sRGB -> linear on the way in, so what
   comes out of the render is the original photo colour.

Unlike `overlay.sh`, this is a real 3D render, so near squares occlude far ones
through the depth buffer.

Verified by screen-blending a rendered frame over the real frame it came from:
the squares disappear into the bark and leaf litter they were sampled from.

### The second test: moving the camera

Open `<workdir>/cond_scene.blend`. The camera carries 61 LINEAR keyframes; edit
or clear them as described under *Tweaking the camera path* above, then render
with `Ctrl F12`. `Footage_Ref` is the original frames on a backdrop plane, kept
as a viewport reference and disabled for rendering.

One thing to keep in mind: once the camera leaves the solved path, frames 1 and
61 of the real footage no longer show what that camera sees, so for that test
render all 61 frames rather than substituting the anchors back in.

## Point cloud only

`sfm_out/sparse_points.ply` is the bare cloud if you want just that:
`File > Import > Stanford PLY`. Note it has no faces or edges, so it shows as
dots in the viewport but **renders as nothing** until you add a Geometry Nodes
*Mesh to Points* modifier. Colours arrive as a vertex colour attribute named
`Col`. The `.blend` above already handles all of this.
