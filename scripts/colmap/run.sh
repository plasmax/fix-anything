#!/usr/bin/env bash
# Video -> sparse point cloud.
#   ./run.sh IMG_4425.mp4 [--workdir sfm_out] [--fps 10] [--max-width 1600] [sfm.py flags...]
#
# Frames are extracted here on the host because the .sif has no ffmpeg.
# --bind /usr/lib/wsl is required on WSL2, otherwise the container cannot see the GPU.
set -euo pipefail

SIF=colmap_pycolmap_cuda124.sif
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

VIDEO=$1; shift
WORKDIR=sfm_out
FPS=10
MAXW=1600
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --workdir)   WORKDIR=$2; shift 2 ;;
    --fps)       FPS=$2; shift 2 ;;
    --max-width) MAXW=$2; shift 2 ;;
    *)           PASSTHRU+=("$1"); shift ;;
  esac
done

VIDEO=$(readlink -f "$VIDEO")
mkdir -p "$WORKDIR"
WORKDIR=$(readlink -f "$WORKDIR")
IMAGES="$WORKDIR/images"

rm -rf "$IMAGES"; mkdir -p "$IMAGES"
ffmpeg -v error -i "$VIDEO" \
  -vf "fps=$FPS,scale='min($MAXW,iw)':-2" -q:v 2 \
  "$IMAGES/frame_%04d.jpg"
echo "host ffmpeg: $(ls "$IMAGES" | wc -l) frames -> $IMAGES"

exec apptainer exec --nv \
  --bind /usr/lib/wsl:/usr/lib/wsl \
  --env LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  --pwd "$HERE" \
  "$SIF" python3 "$HERE/sfm.py" "$VIDEO" \
    --workdir "$WORKDIR" --max-width "$MAXW" --no-extract "${PASSTHRU[@]+"${PASSTHRU[@]}"}"
