#!/usr/bin/env python3
"""Video -> sparse COLMAP point cloud (PLY) for Blender import.

Run inside the apptainer container; see run.sh.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pycolmap


def extract_frames(video: Path, out_dir: Path, fps: float, max_width: int) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    vf = "fps={},scale='min({},iw)':-2".format(fps, max_width)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", vf,
         "-q:v", "2", str(out_dir / "frame_%04d.jpg")],
        check=True,
    )
    return len(list(out_dir.glob("*.jpg")))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--workdir", type=Path, default=Path("sfm_out"))
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--max-width", type=int, default=1600)
    p.add_argument("--overlap", type=int, default=15,
                   help="sequential matching window")
    p.add_argument("--exhaustive", action="store_true",
                   help="match every pair; fine for a few hundred frames")
    p.add_argument("--no-extract", action="store_true",
                   help="reuse <workdir>/images; the container has no ffmpeg, "
                        "so run.sh extracts on the host and passes this")
    args = p.parse_args()

    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    images = work / "images"
    database = work / "database.db"
    sparse = work / "sparse"

    if args.no_extract:
        n = len(list(images.glob("*.jpg")))
        print("[1/4] reusing {} frames in {}".format(n, images), flush=True)
    else:
        n = extract_frames(args.video, images, args.fps, args.max_width)
        print("[1/4] extracted {} frames -> {}".format(n, images), flush=True)
    if n < 8:
        print("too few frames; raise --fps", file=sys.stderr)
        return 1

    if database.exists():
        database.unlink()

    # GTX 1050 Ti has 4 GB: cap SIFT work so the GPU extractor does not OOM.
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = args.max_width
    extraction.sift.max_num_features = 8192
    extraction.sift.estimate_affine_shape = False
    extraction.sift.domain_size_pooling = False

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"

    pycolmap.extract_features(
        database_path=database,
        image_path=images,
        camera_mode=pycolmap.CameraMode.SINGLE,  # one lens, fixed zoom
        reader_options=reader,
        extraction_options=extraction,
        device=pycolmap.Device.cuda,
    )
    print("[2/4] features done", flush=True)

    if args.exhaustive:
        pycolmap.match_exhaustive(database_path=database, device=pycolmap.Device.cuda)
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = args.overlap
        pairing.quadratic_overlap = True
        pycolmap.match_sequential(
            database_path=database,
            pairing_options=pairing,
            device=pycolmap.Device.cuda,
        )
    print("[3/4] matching done", flush=True)

    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)

    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = -1
    recs = pycolmap.incremental_mapping(
        database_path=database, image_path=images, output_path=sparse, options=opts
    )
    if not recs:
        print("reconstruction failed: no model. Try --fps 10 or --exhaustive.",
              file=sys.stderr)
        return 2

    best_id = max(recs, key=lambda k: recs[k].num_points3D())
    rec = recs[best_id]

    # Mapping can split into several models; everything downstream reads
    # sparse/0, so make sure that is the best one rather than merely the first.
    if best_id != 0:
        tmp = sparse / "_swap"
        (sparse / "0").rename(tmp)
        (sparse / str(best_id)).rename(sparse / "0")
        tmp.rename(sparse / str(best_id))
        print("      model {} was the best of {}; swapped into sparse/0".format(
            best_id, len(recs)), flush=True)

    ply = work / "sparse_points.ply"
    rec.export_PLY(str(ply))
    print("[4/4] model {}: {}/{} images registered, {} points".format(
        best_id, rec.num_reg_images(), n, rec.num_points3D()), flush=True)
    print("      PLY -> {}".format(ply), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
