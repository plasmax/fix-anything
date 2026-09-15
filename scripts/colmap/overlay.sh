#!/usr/bin/env bash
# Overlay the sparse cloud, projected through each recovered camera, onto the
# frames the reconstruction was built from, and encode an mp4.
#
#   ./overlay.sh [--workdir sfm_out] [--out overlay.mp4] [--fps 10]
#                [--color depth|rgb|green] [--radius 2] [--alpha 0.85]
#                [--observed-only] [--side-by-side]
#
# ffmpeg decodes on the host (the .sif has none) and pipes raw rgb24 through
# overlay.py inside the container, which pipes annotated raw frames back out.
set -euo pipefail

SIF=colmap_pycolmap_cuda124.sif
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

WORKDIR="$HERE/sfm_out"
OUT=""
FPS=10
SIDE_BY_SIDE=0
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --workdir)       WORKDIR=$2; shift 2 ;;
    --out)           OUT=$2; shift 2 ;;
    --fps)           FPS=$2; shift 2 ;;
    --side-by-side)  SIDE_BY_SIDE=1; shift ;;
    *)               PASSTHRU+=("$1"); shift ;;
  esac
done

WORKDIR=$(readlink -f "$WORKDIR")
IMAGES="$WORKDIR/images"
MODEL="$WORKDIR/sparse/0"
OUT=${OUT:-"$WORKDIR/overlay.mp4"}

[[ -d $IMAGES ]] || { echo "no frames at $IMAGES; run run.sh first" >&2; exit 1; }
[[ -d $MODEL  ]] || { echo "no model at $MODEL; run run.sh first"  >&2; exit 1; }

LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT
# LC_ALL=C so the container's frame order matches ffmpeg's glob order exactly.
(cd "$IMAGES" && LC_ALL=C ls *.jpg) > "$LIST"
N=$(wc -l < "$LIST")

read -r W H < <(ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height -of csv=p=0 \
  "$IMAGES/$(head -1 "$LIST")" | tr ',' ' ')
echo "overlaying $N frames at ${W}x${H} -> $OUT"

if [[ $SIDE_BY_SIDE -eq 1 ]]; then
  # Re-read the originals as a second input and stack them next to the overlay.
  ENCODE=(ffmpeg -y -v error
    -f rawvideo -pix_fmt rgb24 -s "${W}x${H}" -r "$FPS" -i -
    -framerate "$FPS" -pattern_type glob -i "$IMAGES/*.jpg"
    -filter_complex "[1:v][0:v]hstack=inputs=2,format=yuv420p"
    -c:v libx264 -preset slow -crf 18 -movflags +faststart "$OUT")
else
  ENCODE=(ffmpeg -y -v error
    -f rawvideo -pix_fmt rgb24 -s "${W}x${H}" -r "$FPS" -i -
    -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p
    -movflags +faststart "$OUT")
fi

ffmpeg -v error -framerate "$FPS" -pattern_type glob -i "$IMAGES/*.jpg" \
       -f rawvideo -pix_fmt rgb24 - \
  | apptainer exec --bind /usr/lib/wsl:/usr/lib/wsl \
      --env LD_LIBRARY_PATH=/usr/lib/wsl/lib --pwd "$HERE" \
      "$SIF" python3 "$HERE/overlay.py" \
        --model "$MODEL" --frame-list "$LIST" --width "$W" --height "$H" \
        "${PASSTHRU[@]+"${PASSTHRU[@]}"}" \
  | "${ENCODE[@]}"

ls -lh "$OUT"
