#!/usr/bin/env bash
# Run the whole chain (SfM -> overlay mp4 -> .blend) over several videos.
#
#   ./batch.sh IMG_4380.mp4 IMG_4424.mp4 ...
#
# Each video gets its own workdir, out/<stem>/, so runs never clobber each other.
# A failure on one video is logged and the batch moves on to the next.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FPS=${FPS:-10}
MATCH=${MATCH:---exhaustive}

for VIDEO in "$@"; do
  STEM=$(basename "$VIDEO"); STEM=${STEM%.*}
  WORK="$HERE/out/$STEM"
  LOG="$HERE/out/$STEM.log"
  mkdir -p "$WORK"
  echo "######## $STEM -> $WORK (log: $LOG)"
  {
    echo "==== $(date -Is) $STEM"
    set -x
    "$HERE/run.sh"        "$VIDEO" --workdir "$WORK" --fps "$FPS" $MATCH   || exit 1
    "$HERE/overlay.sh"    --workdir "$WORK" --fps "$FPS"                   || exit 2
    "$HERE/to_blender.sh" --workdir "$WORK" --fps "$FPS"                   || exit 3
  } >"$LOG" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "   ok: $WORK/overlay.mp4, $WORK/colmap_scene.blend"
  else
    echo "   FAILED at stage $rc -- see $LOG"
    tail -20 "$LOG"
  fi
done
