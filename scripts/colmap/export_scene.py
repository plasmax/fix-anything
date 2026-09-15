#!/usr/bin/env python3
"""Export a COLMAP sparse model to scene.json for blender_import.py.

Runs inside the apptainer container (needs pycolmap). Everything that depends on
COLMAP conventions is resolved here -- poses arrive in scene.json already in
Blender's coordinate system -- so the Blender side stays dumb.

By default the model and frames are undistorted first, giving a PINHOLE camera
that Blender can reproduce exactly (Blender has no radial-distortion term on the
camera, so a SIMPLE_RADIAL model would leave the image plane misaligned by
~20 px at the corners).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap

# COLMAP cameras look down +Z with +Y down; Blender's look down -Z with +Y up.
CV_TO_GL = np.diag([1.0, -1.0, -1.0])


def undistort(work: Path, model: Path, images: Path) -> tuple:
    out = work / "undistorted"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    pycolmap.undistort_images(
        output_path=out, input_path=model, image_path=images,
        output_type="COLMAP",
    )
    return out / "sparse", out / "images"


def upright_matrix(rots: np.ndarray) -> np.ndarray:
    """Rotation standing the solve upright, from the average camera orientation.

    COLMAP's world axes are arbitrary -- there is no gravity in the solve -- but
    handheld footage is held roughly level, so the mean camera up vector is a
    good estimate of world up. Applied to the root empty only, so the exported
    poses and points keep their original coordinates.
    """
    up = rots[:, :, 1].mean(axis=0)          # Blender-convention camera +Y = up
    fwd = (-rots[:, :, 2]).mean(axis=0)      # camera -Z = forward
    n = np.linalg.norm(up)
    if n < 1e-9:
        return np.eye(3)
    up = up / n

    fwd = fwd - np.dot(fwd, up) * up         # level the mean viewing direction
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        return np.eye(3)
    fwd = fwd / n

    right = np.cross(fwd, up)
    m = np.stack([right, fwd, up])           # rows map world axes onto X, Y, Z
    if np.linalg.det(m) < 0:
        m[0] = -m[0]
    return m


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", type=Path, default=Path("sfm_out"))
    p.add_argument("--out", type=Path, default=None,
                   help="default <workdir>/blender/scene.json")
    p.add_argument("--no-undistort", action="store_true",
                   help="use the raw model/frames; the image plane will not line "
                        "up perfectly at the edges")
    p.add_argument("--plane-distance", type=float, default=0.0,
                   help="image plane distance in front of the camera; 0 = auto, "
                        "placed behind the cloud so it acts as a backdrop")
    p.add_argument("--sensor-width", type=float, default=36.0)
    p.add_argument("--point-px", type=float, default=2.5,
                   help="point radius expressed in pixels at median scene depth")
    p.add_argument("--no-align", action="store_true",
                   help="keep COLMAP's arbitrary world axes instead of standing "
                        "the solve upright on the root empty")
    p.add_argument("--fps", type=float, default=10.0,
                   help="playback fps; should match run.sh --fps")
    args = p.parse_args()

    work = args.workdir.resolve()
    model, images = work / "sparse" / "0", work / "images"
    if not args.no_undistort:
        print("undistorting model and frames ...", flush=True)
        model, images = undistort(work, model, images)

    rec = pycolmap.Reconstruction(str(model))
    posed = sorted((im for im in rec.images.values() if im.has_pose),
                   key=lambda im: im.name)
    if not posed:
        raise SystemExit("no posed images in " + str(model))

    cams = {im.camera_id for im in posed}
    if len(cams) != 1:
        raise SystemExit("expected a single camera, found {}".format(len(cams)))
    cam = posed[0].camera
    fx, fy = cam.focal_length_x, cam.focal_length_y
    cx, cy = cam.principal_point_x, cam.principal_point_y
    W, H = cam.width, cam.height

    xyz = np.array([pt.xyz for pt in rec.points3D.values()], dtype=np.float64)
    rgb = np.array([pt.color for pt in rec.points3D.values()], dtype=np.uint8)

    frames, depths, rots = [], [], []
    for i, im in enumerate(posed):
        T = im.cam_from_world().matrix()
        R_wc, t = T[:, :3], T[:, 3]
        R_cw = R_wc.T
        centre = -R_cw @ t
        rot = R_cw @ CV_TO_GL  # camera axes in world, Blender convention
        frames.append({
            "scene_frame": i + 1,
            "name": im.name,
            "location": centre.tolist(),
            "rotation_matrix": rot.tolist(),
        })
        rots.append(rot)
        d = (xyz @ R_wc.T + t)[:, 2]
        depths.append(d[d > 1e-6])

    alld = np.concatenate(depths)
    # Behind the cloud, not in it: a plane at mid-depth would occlude most of the
    # points. Alignment is a pure projection, so distance is free to choose.
    plane_d = args.plane_distance or float(np.percentile(alld, 99) * 1.05)

    # Size points in pixels-at-typical-depth, not as a fraction of the bounding
    # box: a handful of distant outliers can inflate the box by 100x.
    median_d = float(np.median(alld))
    point_radius = args.point_px * median_d / fx

    # The plane's centre is the ray through pixel (W/2, H/2), which is not the
    # principal point unless the camera is perfectly centred.
    plane = {
        "distance": plane_d,
        "width": plane_d * W / fx,
        "height": plane_d * H / fy,
        "offset_x": plane_d * (W / 2.0 - cx) / fx,
        "offset_y": -plane_d * (H / 2.0 - cy) / fy,
    }

    root = (np.eye(3) if args.no_align
            else upright_matrix(np.array(rots)))

    scene = {
        "source": str(model),
        "image_dir": str(images),
        "root_matrix": root.tolist(),
        "camera": {
            "model": cam.model_name,
            "width": W, "height": H,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "sensor_width": args.sensor_width,
            "lens_mm": fx * args.sensor_width / W,
            # Blender measures shift against the larger image dimension.
            "shift_x": (cx - W / 2.0) / max(W, H),
            "shift_y": (H / 2.0 - cy) / max(W, H),
            "pixel_aspect_y": fx / fy,
            # Default clip_end is 100, which would cut off the backdrop.
            "clip_start": max(float(np.percentile(alld, 1)) * 0.1, 1e-3),
            "clip_end": plane_d * 2.0,
        },
        "plane": plane,
        "fps": args.fps,
        "frames": frames,
        "points": {
            "count": int(len(xyz)),
            "radius": point_radius,
            "xyz": xyz.round(6).tolist(),
            "rgb": rgb.tolist(),
        },
    }

    out = args.out or (work / "blender" / "scene.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scene))
    print("wrote {} ({} frames, {} points, plane at {:.2f}, "
          "point radius {:.4f})".format(
              out, len(frames), len(xyz), plane_d, point_radius))
    print("image_dir: {}".format(images))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
