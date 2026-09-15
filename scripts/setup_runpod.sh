#!/usr/bin/env bash
# FixAnything RunPod bootstrap. Run from anywhere on a fresh pod:
#   FORK_URL=https://github.com/<you>/fix-anything.git bash setup_runpod.sh
# Env vars (all optional):
#   FORK_URL      git URL of your fork (default below)
#   WORKDIR       where to clone (default /workspace/fix-anything)
#   HF_TOKEN      Hugging Face token (only needed for gated/private repos)
#   DOWNLOAD=0    skip the model download (default: download via scripts/download.sh)
#   DIT           which DiT weights to download: fp8 (default, 16 GB) | bf16 (33 GB) | original (33 GB)
#   TEXT_ENCODER=1 also download the 11 GB umT5 encoder (only needed to encode new prompts)
#   MAPANYTHING=1 also install the optional MapAnything extra
#   GIT_NAME / GIT_EMAIL  set global git identity
set -euo pipefail

FORK_URL="${FORK_URL:-https://github.com/plasmax/fix-anything.git}"
WORKDIR="${WORKDIR:-/workspace/fix-anything}"
PY=/usr/bin/python3.10

[ -n "${GIT_NAME:-}" ]  && git config --global user.name  "$GIT_NAME"
[ -n "${GIT_EMAIL:-}" ] && git config --global user.email "$GIT_EMAIL"

# --- code -------------------------------------------------------------------
if [ ! -d "$WORKDIR/.git" ]; then git clone "$FORK_URL" "$WORKDIR"; fi
cd "$WORKDIR"

# --- uv + venv (Python 3.10; uv downloads it if the pod lacks it) -------------
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -x "$PY" ] || PY=3.10
[ -d .venv ] || uv venv --python "$PY" .venv --prompt fixanything
V=.venv/bin/python

# --- packages (README recipe; torch cu126 so the prebuilt flash-attn wheel matches) ---
uv pip install --python $V pip "setuptools<82" wheel hf_transfer
uv pip install --python $V torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
$V -m pip install --no-build-isolation -e .
$V -m pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
if [ "${MAPANYTHING:-0}" = "1" ]; then
  # torchaudio must be pinned or the extra drags torch to 2.11
  $V -m pip install --no-build-isolation -e ".[mapanything]" "torchaudio==2.6.0" --extra-index-url https://download.pytorch.org/whl/cu126
fi

command -v git-lfs >/dev/null || { apt-get update && apt-get install -y git-lfs; }
# --- weights (DIT / TEXT_ENCODER / HF_TOKEN are read by scripts/download.sh) ----
[ "${DOWNLOAD:-1}" = "1" ] && bash scripts/download.sh

# --- smoke test ----------------------------------------------------------------
$V - <<'PY'
import torch, flash_attn, diffsynth
from fixanything.pipelines import WanVideoPipeline
print("OK:", torch.__version__, torch.cuda.get_device_name(0), "flash-attn", flash_attn.__version__)
PY
echo "Done. Activate with: source $WORKDIR/.venv/bin/activate"
echo "Try:  python scripts/run_inference.py --input examples/dl3dv_tracks/input.mp4 --output_dir outputs/dl3dv_tracks"
