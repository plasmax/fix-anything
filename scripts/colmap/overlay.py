#!/usr/bin/env python3
"""Project a COLMAP sparse model through its own cameras onto the source frames.

Reads raw rgb24 frames on stdin, writes annotated raw rgb24 frames on stdout.
The container image has only numpy (no cv2/PIL/ffmpeg), so decoding, encoding and
all drawing are done as plain array writes; overlay.sh supplies the ffmpeg ends.

All logging goes to stderr because stdout is the video stream.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pycolmap

INVALID_POINT3D_ID = 18446744073709551615  # colmap's uint64 sentinel


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def disc_offsets(radius: int) -> np.ndarray:
    """(N,2) integer dy,dx offsets covering a filled disc."""
    r = int(radius)
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    keep = (dy * dy + dx * dx) <= r * r
    return np.stack([dy[keep], dx[keep]], axis=1)


def turbo(t: np.ndarray) -> np.ndarray:
    """Cheap turbo-like colormap for t in [0,1] -> (N,3) uint8.

    Piecewise-linear through blue/cyan/green/yellow/red; close enough to turbo
    for reading depth off a video, and needs no matplotlib.
    """
    stops = np.array([
        [48, 18, 59], [70, 134, 251], [27, 229, 181],
        [165, 254, 60], [251, 155, 6], [122, 4, 3],
    ], dtype=np.float64)
    t = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    i = np.floor(t).astype(int)
    i = np.minimum(i, len(stops) - 2)
    f = (t - i)[:, None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def project(rec, image, all_points_xyz, all_points_rgb, observed_only):
    """Return (uv, depth, rgb) for the points this camera sees."""
    cam = image.camera
    T = image.cam_from_world().matrix()

    if observed_only:
        ids = [p.point3D_id for p in image.points2D
               if p.point3D_id != INVALID_POINT3D_ID]
        if not ids:
            return None
        xyz = np.array([rec.points3D[i].xyz for i in ids])
        rgb = np.array([rec.points3D[i].color for i in ids], dtype=np.uint8)
    else:
        xyz, rgb = all_points_xyz, all_points_rgb

    cam_xyz = xyz @ T[:, :3].T + T[:, 3]
    front = cam_xyz[:, 2] > 1e-6
    if not front.any():
        return None
    cam_xyz, rgb = cam_xyz[front], rgb[front]

    uv = np.asarray(cam.img_from_cam(cam_xyz))
    inb = ((uv[:, 0] >= 0) & (uv[:, 0] < cam.width)
           & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height))
    if not inb.any():
        return None
    return uv[inb], cam_xyz[inb, 2], rgb[inb]


def draw(frame, uv, depth, rgb, offsets, mode, alpha, dmin, dmax):
    """Blend discs into frame in far-to-near order so near points win."""
    order = np.argsort(-depth)
    uv, depth, rgb = uv[order], depth[order], rgb[order]

    if mode == "depth":
        # Inverse depth spreads nearby structure out instead of bunching it.
        t = (1.0 / np.clip(depth, 1e-6, None) - 1.0 / dmax) / \
            (1.0 / dmin - 1.0 / dmax + 1e-12)
        colors = turbo(t)
    elif mode == "rgb":
        colors = rgb
    else:
        colors = np.broadcast_to(np.array([0, 255, 80], np.uint8), rgb.shape)

    h, w = frame.shape[:2]
    cy = np.rint(uv[:, 1]).astype(np.int64)
    cx = np.rint(uv[:, 0]).astype(np.int64)

    # Expand every centre into its disc, then blend all pixels in one pass.
    ys = (cy[:, None] + offsets[None, :, 0]).ravel()
    xs = (cx[:, None] + offsets[None, :, 1]).ravel()
    cols = np.repeat(colors, len(offsets), axis=0)

    ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    ys, xs, cols = ys[ok], xs[ok], cols[ok]

    cur = frame[ys, xs].astype(np.float32)
    frame[ys, xs] = (cur * (1 - alpha) + cols.astype(np.float32) * alpha
                     ).astype(np.uint8)
    return frame


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--frame-list", type=Path, required=True,
                   help="frame basenames, one per line, in the order ffmpeg "
                        "feeds them on stdin")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--color", choices=["depth", "rgb", "green"], default="depth")
    p.add_argument("--observed-only", action="store_true",
                   help="draw only points actually tracked in each frame "
                        "instead of the whole cloud")
    args = p.parse_args()

    rec = pycolmap.Reconstruction(str(args.model))
    by_name = {im.name: im for im in rec.images.values() if im.has_pose}
    log("model: {} points, {} posed images".format(
        rec.num_points3D(), len(by_name)))

    xyz = np.array([pt.xyz for pt in rec.points3D.values()])
    rgb = np.array([pt.color for pt in rec.points3D.values()], dtype=np.uint8)

    # Global depth range keeps the colormap stable across the whole clip.
    depths = []
    for im in by_name.values():
        T = im.cam_from_world().matrix()
        d = (xyz @ T[:, :3].T + T[:, 3])[:, 2]
        depths.append(d[d > 1e-6])
    alld = np.concatenate(depths)
    dmin, dmax = np.percentile(alld, [2, 98])
    log("depth range (2-98 pct): {:.2f} .. {:.2f}".format(dmin, dmax))

    names = [n.strip() for n in args.frame_list.read_text().splitlines()
             if n.strip()]
    offsets = disc_offsets(args.radius)
    nbytes = args.width * args.height * 3
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer

    drawn = skipped = 0
    for i, name in enumerate(names):
        buf = stdin.read(nbytes)
        if len(buf) < nbytes:
            log("stdin ended early at frame {} ({})".format(i, name))
            break
        frame = np.frombuffer(buf, np.uint8).reshape(
            args.height, args.width, 3).copy()

        image = by_name.get(name)
        if image is None:
            skipped += 1  # unregistered frame: pass through untouched
        else:
            got = project(rec, image, xyz, rgb, args.observed_only)
            if got is None:
                skipped += 1
            else:
                uv, depth, pcol = got
                frame = draw(frame, uv, depth, pcol, offsets,
                             args.color, args.alpha, dmin, dmax)
                drawn += 1
        stdout.write(frame.tobytes())

    stdout.flush()
    log("done: {} frames annotated, {} passed through".format(drawn, skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
