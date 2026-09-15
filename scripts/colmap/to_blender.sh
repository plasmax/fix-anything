#!/usr/bin/env bash
# Export the solve to a .blend with an animated camera, footage plane and points.
#
#   ./to_blender.sh [--workdir sfm_out] [--out sfm_out/colmap_scene.blend]
#                   [--fps 10] [--plane-distance 0] [--point-radius 0]
#                   [--no-undistort] [--json-only]
#
# Step 1 runs pycolmap in the container; step 2 runs Windows Blender (the WSL
# build did not work here) over the \\wsl.localhost share, so paths are
# translated with wslpath.
set -euo pipefail

SIF=/home/mlast/dev/apptainertests/colmap_pycolmap_cuda124.sif
BLENDER=${BLENDER:-"/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

WORKDIR="$HERE/sfm_out"
OUT=""
POINT_RADIUS=0
JSON_ONLY=0
EXPORT_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --workdir)       WORKDIR=$2; shift 2 ;;
    --out)           OUT=$2; shift 2 ;;
    --point-radius)  POINT_RADIUS=$2; shift 2 ;;
    --json-only)     JSON_ONLY=1; shift ;;
    *)               EXPORT_ARGS+=("$1"); shift ;;
  esac
done

WORKDIR=$(readlink -f "$WORKDIR")
OUT=${OUT:-"$WORKDIR/colmap_scene.blend"}
SCENE="$WORKDIR/blender/scene.json"

[[ -d "$WORKDIR/sparse/0" ]] || { echo "no model in $WORKDIR; run run.sh first" >&2; exit 1; }

echo "== 1/2 exporting scene.json"
apptainer exec --bind /usr/lib/wsl:/usr/lib/wsl \
  --env LD_LIBRARY_PATH=/usr/lib/wsl/lib --pwd "$HERE" \
  "$SIF" python3 "$HERE/export_scene.py" \
    --workdir "$WORKDIR" --out "$SCENE" "${EXPORT_ARGS[@]+"${EXPORT_ARGS[@]}"}"

[[ $JSON_ONLY -eq 1 ]] && { echo "$SCENE"; exit 0; }

[[ -x "$BLENDER" ]] || { echo "blender not found at $BLENDER (set BLENDER=)" >&2; exit 1; }

# The .blend stores the image-sequence path, so Blender must see the frames the
# same way it will after the file is reopened: as Windows UNC paths.
IMAGES=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['image_dir'])" "$SCENE")
WIN_SCENE=$(wslpath -w "$SCENE")
WIN_IMAGES=$(wslpath -w "$IMAGES")
WIN_OUT=$(wslpath -w "$OUT")
WIN_SCRIPT=$(wslpath -w "$HERE/blender_import.py")

echo "== 2/2 building $OUT"
"$BLENDER" --background --factory-startup --python "$WIN_SCRIPT" -- \
  --scene "$WIN_SCENE" --images "$WIN_IMAGES" --save "$WIN_OUT" \
  --point_radius "$POINT_RADIUS" 2>&1 | tr -d '\r' | grep -vE '^(Blender|Read prefs|found bundled|Warning: class)' || true

ls -lh "$OUT"
echo
echo "open with:"
echo "  \"C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender-launcher.exe\" \"$(wslpath -w "$OUT")\""
