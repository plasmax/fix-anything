#!/usr/bin/env bash
# FixAnything model downloader. Run from the repo root with the venv already set up:
#   bash scripts/download.sh
# Env vars (all optional):
#   WORKDIR       repo root (default /workspace/fix-anything)
#   HF_TOKEN      Hugging Face token (only needed for gated/private repos)
#   PROMPT_EMBEDS path to a local prompt_embeds.pt to install into models/ (skips the 11 GB T5 encoder)
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/fix-anything}"
cd "$WORKDIR"

# --- model dir layout ---------------------------------------------------------
WAN=models/Wan-AI/Wan2.1-I2V-14B-480P
mkdir -p "$WAN"
[ -e checkpoints ] || ln -s models checkpoints
[ -n "${PROMPT_EMBEDS:-}" ] && cp "$PROMPT_EMBEDS" models/prompt_embeds.pt

# --- weights. The venv's `hf` CLI is .venv/bin/hf --------------------------------
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
