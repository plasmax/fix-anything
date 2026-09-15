#!/usr/bin/env bash
# Point-cloud conditioning sequence for a video diffusion model.
#
#   ./cond.sh [--workdir sfm_out] [--width 832] [--height 480]
#             [--frames 61] [--point-px 4] [--span] [--fps 10]
#
# Produces, under <workdir>/cond/:
#   frames/frame_0001.png .. frame_0061.png   image sequence, kept on disk
#   cond.mp4                                  the same thing encoded
# and a reusable scene at <workdir>/cond_scene.blend.
#
# Frames 1 and N are the real footage; 2..N-1 are the point cloud rendered
# through the solved camera, so the model gets a start frame, an end frame and
# a moving 3D scaffold in between.
set -euo pipefail

BLENDER=${BLENDER:-"/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

WORKDIR="$HERE/sfm_out"
W=832
H=480
N=61
PX=4
SPAN=0
FPS=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --workdir)   WORKDIR=$2; shift 2 ;;
    --width)     W=$2; shift 2 ;;
    --height)    H=$2; shift 2 ;;
    --frames)    N=$2; shift 2 ;;
    --point-px)  PX=$2; shift 2 ;;
    --fps)       FPS=$2; shift 2 ;;
    --span)      SPAN=1; shift ;;
    *)           echo "unknown flag $1" >&2; exit 1 ;;
  esac
done

WORKDIR=$(readlink -f "$WORKDIR")
SCENE="$WORKDIR/blender/scene.json"
COND="$WORKDIR/cond"
FRAMES="$COND/frames"
BLEND="$WORKDIR/cond_scene.blend"

[[ -f "$SCENE" ]] || { echo "no $SCENE; run to_blender.sh first" >&2; exit 1; }
[[ -x "$BLENDER" ]] || { echo "blender not found at $BLENDER (set BLENDER=)" >&2; exit 1; }

rm -rf "$FRAMES"
mkdir -p "$FRAMES"

# Blender runs on the Windows side, so it needs the frames as a UNC path -- and
# that is also what gets baked into the saved .blend.
IMAGES=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['image_dir'])" "$SCENE")

echo "== 1/3 building scene and rendering frames 2..$((N-1))"
"$BLENDER" --background --factory-startup --python "$(wslpath -w "$HERE/cond_build.py")" -- \
  --scene "$(wslpath -w "$SCENE")" \
  --images "$(wslpath -w "$IMAGES")" \
  --save "$(wslpath -w "$BLEND")" \
  --render "$(wslpath -w "$FRAMES")" \
  --width "$W" --height "$H" --frames "$N" --point_px "$PX" --span "$SPAN" \
  2>&1 | tr -d '\r' | grep -vE "^(Blender|Read prefs|found bundled|Warning: class|Fra:)" || true

MANIFEST="$FRAMES/manifest.json"
[[ -f "$MANIFEST" ]] || { echo "blender produced no manifest; see output above" >&2; exit 1; }

# The real count can be lower than N if the solve had fewer frames.
COUNT=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['names']))" "$MANIFEST")
FIRST=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['names'][0])" "$MANIFEST")
LAST=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['names'][-1])" "$MANIFEST")

echo "== 2/3 substituting real footage into frames 1 and $COUNT"
# The same squeeze the render used, so the two line up pixel for pixel.
ffmpeg -y -v error -i "$IMAGES/$FIRST"  -vf "scale=$W:$H" "$FRAMES/$(printf 'frame_%04d.png' 1)"
ffmpeg -y -v error -i "$IMAGES/$LAST"   -vf "scale=$W:$H" "$FRAMES/$(printf 'frame_%04d.png' "$COUNT")"

MISSING=$(( COUNT - $(ls "$FRAMES"/frame_*.png | wc -l) ))
[[ $MISSING -eq 0 ]] || { echo "expected $COUNT frames, $MISSING missing" >&2; exit 1; }

echo "== 3/3 encoding"
[[ -n "$FPS" ]] || FPS=$(python3 -c "import json,sys;print(int(round(json.load(open(sys.argv[1]))['fps'])))" "$SCENE")
ffmpeg -y -v error -framerate "$FPS" -start_number 1 -i "$FRAMES/frame_%04d.png" \
  -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -movflags +faststart \
  "$COND/cond.mp4"

ls -lh "$COND/cond.mp4"
echo "frames: $FRAMES ($COUNT x ${W}x${H})"
echo "scene:  $(wslpath -w "$BLEND")"
