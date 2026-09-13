#!/usr/bin/env bash
# Download the FixAnything models into models/ using the repo venv (wrapper around scripts/download_models.py).
#   bash scripts/download.sh               # lightest setup (~22 GB): fp8 DiT + VAE + CLIP + LoRA
#   DIT=bf16 bash scripts/download.sh      # Comfy-Org bf16 DiT (33 GB) instead
#   DIT=original bash scripts/download.sh  # Wan-AI's sharded bf16 release (33 GB)
# Env vars (all optional):
#   DIT             fp8 (default) | bf16 | original
#   TEXT_ENCODER=1  also fetch the 11 GB umT5 encoder + tokenizer (only to encode new prompts; the defaults ship pre-encoded)
#   HF_TOKEN        Hugging Face token (only needed for gated/private repos)
#   WORKDIR         repo root (default: the parent of this script's folder)
set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$WORKDIR"
V=.venv/bin/python

$V -c "import hf_transfer" 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1
[ -n "${HF_TOKEN:-}" ] && .venv/bin/hf auth login --token "$HF_TOKEN" --add-to-git-credential >/dev/null

# models/prompt_embeds.pt is stored with git LFS; a clone made without LFS only has the pointer file
[ "$(stat -c %s models/prompt_embeds.pt)" -gt 1024 ] || { git lfs install --local >/dev/null; git lfs pull; }

extra=()
[ "${TEXT_ENCODER:-0}" = "1" ] && extra+=(--text_encoder)
$V scripts/download_models.py --model_dir models --dit "${DIT:-fp8}" ${extra[@]+"${extra[@]}"}
