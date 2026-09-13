#!/usr/bin/env bash
# FixAnything RunPod bootstrap. Run from anywhere on a fresh pod:
#   FORK_URL=https://github.com/<you>/fix-anything.git bash setup_runpod.sh
# Env vars (all optional):
#   FORK_URL      git URL of your fork (default below)
#   WORKDIR       where to clone (default /workspace/fix-anything)
#   HF_TOKEN      Hugging Face token (only needed for gated/private repos)
#   PROMPT_EMBEDS path to a local prompt_embeds.pt to install into models/ (skips the 11 GB T5 encoder)
#   DOWNLOAD=1    also download VAE, CLIP, tokenizer, FixAnything LoRA and the Comfy-Org bf16 DiT
#   MAPANYTHING=1 also install the optional MapAnything extra
#   GIT_NAME / GIT_EMAIL  set global git identity
set -euo pipefail

FORK_URL="${FORK_URL:-https://github.com/kvuong2711/fix-anything.git}"
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

# --- model dir layout ---------------------------------------------------------
WAN=models/Wan-AI/Wan2.1-I2V-14B-480P
mkdir -p "$WAN"
[ -e checkpoints ] || ln -s models checkpoints
[ -n "${PROMPT_EMBEDS:-}" ] && cp "$PROMPT_EMBEDS" models/prompt_embeds.pt

# --- weights (optional). The venv's `hf` CLI is .venv/bin/hf -------------------
if [ "${DOWNLOAD:-0}" = "1" ]; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  HF=.venv/bin/hf
  [ -n "${HF_TOKEN:-}" ] && $HF auth login --token "$HF_TOKEN" --add-to-git-credential >/dev/null
  $HF download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN" \
      --include "Wan2.1_VAE.pth" "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" "google/*"
  # T5 encoder only if no precomputed prompt embeddings
  [ -f models/prompt_embeds.pt ] || $HF download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN" --include "models_t5_umt5-xxl-enc-bf16.pth"
  # single-file bf16 DiT (32.8 GB). Key layout is recognised by DiffSynth; adjust the glob in run_inference.py to
  # split_files/diffusion_models/*.safetensors (or symlink it to diffusion_pytorch_model.safetensors in $WAN).
  $HF download Comfy-Org/Wan_2.1_ComfyUI_repackaged --local-dir "$WAN" \
      --include "split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors"
  $HF download kvuong2711/fix-anything fixanything_lora.safetensors --local-dir models
fi

# --- smoke test ----------------------------------------------------------------
$V - <<'PY'
import torch, flash_attn, diffsynth
from fixanything.pipelines import WanVideoPipeline
print("OK:", torch.__version__, torch.cuda.get_device_name(0), "flash-attn", flash_attn.__version__)
PY
echo "Done. Activate with: source $WORKDIR/.venv/bin/activate   (hf CLI: $WORKDIR/.venv/bin/hf)"
