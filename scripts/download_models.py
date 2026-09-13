"""Download the FixAnything checkpoints into models/.

The default is the lightest working setup (~22 GB): VAE, CLIP image encoder, FixAnything LoRA and the
Comfy-Org fp8_e4m3fn repack of the Wan2.1-I2V-14B-480P DiT (16 GB). The 11 GB umT5 text encoder and its
tokenizer are skipped because models/prompt_embeds.pt (in the repo, via git LFS) already holds the default
prompts encoded; pass --text_encoder to fetch them if you want to encode other prompts.

Example:
    python scripts/download_models.py                  # fp8 DiT (16 GB)
    python scripts/download_models.py --dit bf16       # Comfy-Org bf16 repack (33 GB)
    python scripts/download_models.py --dit original   # Wan-AI's sharded bf16 release (33 GB)
"""
import argparse
import os

from huggingface_hub import hf_hub_download
from diffsynth.utils import ModelConfig
from run_inference import (WAN_MODEL_ID, WAN_MODEL_FILES, WAN_TEXT_ENCODER_FILE, WAN_TOKENIZER_FILES,  # noqa: E402
                           WAN_DIT_FILES, PROMPT_EMBEDS_FILE)

LORA_REPO, LORA_FILE = "kvuong2711/fix-anything", "fixanything_lora.safetensors"
COMFY_REPO = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
COMFY_DIT = {
    "fp8": "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors",
    "bf16": "split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", type=str, default="models")
    p.add_argument("--dit", type=str, default="fp8", choices=["fp8", "bf16", "original"],
                   help="DiT weights: Comfy-Org fp8_e4m3fn repack (default, 16 GB), Comfy-Org bf16 repack (33 GB) "
                        "or the original Wan-AI shards (33 GB).")
    p.add_argument("--text_encoder", action="store_true",
                   help="Also download the umT5 text encoder and tokenizer (11 GB); only needed to encode new prompts.")
    p.add_argument("--source", type=str, default="huggingface", choices=["huggingface", "modelscope"],
                   help="Where to download the Wan-AI files from (the Comfy-Org repacks are Hugging Face only).")
    args = p.parse_args()

    patterns = [f for f in WAN_MODEL_FILES if f != WAN_TEXT_ENCODER_FILE]
    if args.text_encoder:
        patterns += [WAN_TEXT_ENCODER_FILE, WAN_TOKENIZER_FILES]
    elif not os.path.isfile(os.path.join(args.model_dir, PROMPT_EMBEDS_FILE)):
        raise SystemExit(f"{args.model_dir}/{PROMPT_EMBEDS_FILE} not found: run `git lfs pull`, or pass --text_encoder.")
    if args.dit == "original":
        patterns.append(WAN_DIT_FILES)

    for pattern in patterns:
        ModelConfig(model_id=WAN_MODEL_ID, origin_file_pattern=pattern,
                    local_model_path=args.model_dir, download_resource=args.source).download_if_necessary()
    wan_dir = os.path.join(args.model_dir, WAN_MODEL_ID)
    if args.dit in COMFY_DIT:
        hf_hub_download(COMFY_REPO, COMFY_DIT[args.dit], local_dir=wan_dir)
    hf_hub_download(LORA_REPO, LORA_FILE, local_dir=args.model_dir)
    print(f"Base model in {wan_dir}/, LoRA at {args.model_dir}/{LORA_FILE}")


if __name__ == "__main__":
    main()
