"""Run FixAnything on a rendered video (a folder of frames or a video file).

Example:
    python scripts/run_inference.py --input examples/dl3dv_3dgs/input.mp4 --output_dir outputs/dl3dv_3dgs

`load_pipeline()` and `fix_video()` below are the whole inference API.
"""
import argparse
import glob
import json
import math
import os
import struct

import torch
from diffsynth.utils import ModelConfig

from fixanything.pipelines import WanVideoPipeline
from fixanything.data import load_frames, save_video, crop_and_resize, side_by_side

# Base model (same layout on ModelScope and Hugging Face).
WAN_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAN_TEXT_ENCODER_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
WAN_MODEL_FILES = [
    WAN_TEXT_ENCODER_FILE,
    "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
]
WAN_TOKENIZER_FILES = "google/*"
# The DiT: the original multi-file release, or a Comfy-Org single-file repack
# (Comfy-Org/Wan_2.1_ComfyUI_repackaged, bf16 or fp8_e4m3fn) placed in the same model folder.
WAN_DIT_FILES = "diffusion_pytorch_model*.safetensors"
WAN_DIT_REPACK_FILES = "split_files/diffusion_models/wan2.1_i2v_480p_14B_*.safetensors"
SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
                      "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}
# Precomputed prompt embeddings (scripts/encode_prompts.py). If this file exists, the text encoder and
# tokenizer are not loaded (and need not be present) -- see `load_pipeline(prompt_embeds=...)`.
PROMPT_EMBEDS_FILE = "prompt_embeds.pt"

DEFAULT_PROMPT = "A clean, high-quality, photorealistic video with sharp details, smooth motion, and natural lighting."
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def _model_config(model_dir, pattern, download_source="huggingface", **kwargs):
    # Skip the remote download check when the files are already there (so this also works offline).
    present = len(glob.glob(os.path.join(model_dir, WAN_MODEL_ID, pattern))) > 0
    return ModelConfig(
        model_id=WAN_MODEL_ID, origin_file_pattern=pattern, local_model_path=model_dir,
        download_resource=download_source, skip_download=present, **kwargs,
    )


def safetensors_dtype(path):
    """Predominant tensor dtype (by element count) of a .safetensors file, read from the header only."""
    with open(path, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    header.pop("__metadata__", None)
    counts = {}
    for v in header.values():
        counts[v["dtype"]] = counts.get(v["dtype"], 0) + math.prod(v["shape"])
    return SAFETENSORS_DTYPES[max(counts, key=counts.get)]


def find_dit(model_dir):
    """Locate the DiT weights in `<model_dir>/Wan-AI/Wan2.1-I2V-14B-480P/`.

    Returns the original multi-file shards (a list) if present, else a single-file repack, preferring bf16.
    Returns None if nothing is there (the caller falls back to downloading the original).
    """
    wan_dir = os.path.join(model_dir, WAN_MODEL_ID)
    shards = sorted(glob.glob(os.path.join(wan_dir, WAN_DIT_FILES)))
    if shards:
        return shards
    repacks = sorted(glob.glob(os.path.join(wan_dir, WAN_DIT_REPACK_FILES)))
    if not repacks:
        return None
    bf16 = [p for p in repacks if p.endswith("_bf16.safetensors")]
    return (bf16 or repacks)[0]


def load_pipeline(lora_path, model_dir="checkpoints", device="cuda", torch_dtype=torch.bfloat16,
                  lora_alpha=1.0, download_source="huggingface", prompt_embeds="auto",
                  dit_path=None, dit_dtype="auto"):
    """Build the Wan2.1-I2V-14B pipeline, apply the FixAnything LoRA and enable VRAM offloading.

    Base-model files are looked up in `<model_dir>/Wan-AI/Wan2.1-I2V-14B-480P/` and downloaded
    from `download_source` ("huggingface" or "modelscope") if missing.

    prompt_embeds: precomputed prompt embeddings written by scripts/encode_prompts.py.
        "auto" (default): use `<model_dir>/prompt_embeds.pt` if it exists, else load the text encoder.
        A path: use that file (must exist). None: always load the text encoder.
        When embeddings are used, the umT5 text encoder and tokenizer are neither loaded nor downloaded.
    dit_path: explicit DiT weights (original shards or a Comfy-Org single-file repack). None: see `find_dit()`.
    dit_dtype: dtype the DiT weights are stored in while offloaded. "auto": the file's own dtype, so an
        fp8_e4m3fn repack stays fp8 (half the CPU RAM and PCIe traffic); compute is always `torch_dtype`.
        With fp8 storage the LoRA is kept unmerged so it is not rounded into fp8. The weights are first
        loaded in `torch_dtype` (lossless for fp8 files) and cast back to fp8 by the VRAM manager.
    """
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"FixAnything LoRA not found: {lora_path}")
    if prompt_embeds == "auto":
        candidate = os.path.join(model_dir, PROMPT_EMBEDS_FILE)
        prompt_embeds = candidate if os.path.isfile(candidate) else None
    elif prompt_embeds is not None and not os.path.isfile(prompt_embeds):
        raise FileNotFoundError(f"Prompt embeddings not found: {prompt_embeds} (create them with scripts/encode_prompts.py)")

    if dit_path is None:
        dit_path = find_dit(model_dir)
    if dit_path is None:
        dit_config = _model_config(model_dir, WAN_DIT_FILES, download_source, offload_device="cpu")
    else:
        paths = dit_path if isinstance(dit_path, list) else [dit_path]
        for p in paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"DiT weights not found: {p}")
        if dit_dtype == "auto":
            dit_dtype = safetensors_dtype(paths[0])
        print(f"[load_pipeline] DiT: {dit_path} stored as {dit_dtype}, computing in {torch_dtype}.")
        dit_config = ModelConfig(path=dit_path, offload_device="cpu")
    fp8_dit = dit_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    model_files = WAN_MODEL_FILES
    tokenizer_config = _model_config(model_dir, WAN_TOKENIZER_FILES, download_source)
    if prompt_embeds is not None:
        print(f"[load_pipeline] Using precomputed prompt embeddings from {prompt_embeds}; text encoder not loaded.")
        model_files = [f for f in WAN_MODEL_FILES if f != WAN_TEXT_ENCODER_FILE]
        tokenizer_config = None
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype, device=device,
        model_configs=[dit_config] + [_model_config(model_dir, p, download_source, offload_device="cpu") for p in model_files],
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
        prompt_embeds_path=prompt_embeds,
    )
    if pipe.dit is None:
        raise RuntimeError(f"DiffSynth did not recognise {dit_path} as the Wan2.1-I2V-14B DiT.")
    if not fp8_dit:
        pipe.load_lora(pipe.dit, lora_path, alpha=lora_alpha)
    pipe.enable_vram_management(dit_dtype=dit_dtype if fp8_dit else None)
    if fp8_dit:
        pipe.load_lora(pipe.dit, lora_path, alpha=lora_alpha, hotload=True)
    return pipe


def fix_video(
    pipe, frames, clean_frame_indices=None,
    num_frames=61, num_repeat_last=4, height=480, width=832,
    seed=1, num_inference_steps=10, cfg_scale=5.0,
    prompt=DEFAULT_PROMPT, negative_prompt=DEFAULT_NEGATIVE_PROMPT, tiled=True,
):
    """Refine a rendered video.

    Args:
        frames: list of PIL images (the degraded rendering); the first `num_frames` are used.
        clean_frame_indices: indices of input frames that are already clean (e.g. rendered at the input
            views); the model keeps them and refines the rest. List of ints or whitespace-separated
            string. Default: first and last frame. Pass [] to refine every frame.
    The model runs on `num_frames + num_repeat_last` frames (the last frame is repeated, as in training)
    and the output is cropped back to `num_frames`.
    Returns:
        (generated_frames, input_frames): two lists of `num_frames` PIL images of size (width, height).
    """
    if len(frames) < num_frames:
        raise ValueError(f"Need {num_frames} input frames, got {len(frames)}.")
    if len(frames) > num_frames:
        print(f"[fix_video] Using the first {num_frames} of {len(frames)} frames.")
    frames = [crop_and_resize(f, height, width) for f in frames[:num_frames]]
    ref_video = frames + [frames[-1]] * num_repeat_last

    if clean_frame_indices is None:
        clean_frame_indices = [0, num_frames - 1]
    elif isinstance(clean_frame_indices, str):
        clean_frame_indices = [int(x) for x in clean_frame_indices.split()]
    clean = set(clean_frame_indices)
    if num_frames - 1 in clean:   # the repeats of the last frame inherit its flag
        clean.update(range(num_frames, num_frames + num_repeat_last))

    gen_video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=ref_video[0],
        reference_video=ref_video,
        clean_frame_indices=sorted(clean),
        num_frames=num_frames + num_repeat_last, height=height, width=width,
        seed=seed, tiled=tiled,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
    )
    return gen_video[:num_frames], ref_video[:num_frames]


def parse_args():
    p = argparse.ArgumentParser(description="FixAnything inference")
    p.add_argument("--input", type=str, required=True,
                   help="Rendered video file, or a folder of frames.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--lora_path", type=str, default="checkpoints/fixanything_lora.safetensors")
    p.add_argument("--model_dir", type=str, default="checkpoints",
                   help="Folder containing Wan-AI/Wan2.1-I2V-14B-480P/ (downloaded there if missing).")
    p.add_argument("--prompt_embeds", type=str, default="auto",
                   help="Precomputed prompt embeddings from scripts/encode_prompts.py. 'auto' (default) uses "
                        "<model_dir>/prompt_embeds.pt if present; 'none' always loads the umT5 text encoder.")
    p.add_argument("--dit", type=str, default=None,
                   help="DiT weights file: the original shards' folder is searched by default, else a Comfy-Org "
                        "single-file repack (bf16 or fp8_e4m3fn) under <model_dir>/Wan-AI/Wan2.1-I2V-14B-480P/split_files/.")
    p.add_argument("--dit_dtype", type=str, default="auto", choices=["auto", "bf16", "fp8"],
                   help="Storage dtype of the DiT weights ('auto': the file's own dtype). Compute is always bf16.")
    p.add_argument("--clean_frame_indices", type=str, default=None,
                   help='Space-separated indices of input frames to keep as-is (default: first and last, "0 60"). '
                        'Use "" to refine every frame.')
    p.add_argument("--num_frames", type=int, default=61, help="Input frames used (the first ones).")
    p.add_argument("--num_repeat_last", type=int, default=4,
                   help="The last frame is repeated this many times for the model (as in training); the output is cropped back.")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--fps", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    frames, _ = load_frames(args.input, args.height, args.width, num_frames=args.num_frames)
    print(f"Loaded {len(frames)} frames from {args.input}")
    clean = None if args.clean_frame_indices is None else [int(x) for x in args.clean_frame_indices.split()]

    prompt_embeds = None if args.prompt_embeds.lower() == "none" else args.prompt_embeds
    dit_dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[args.dit_dtype]
    pipe = load_pipeline(args.lora_path, args.model_dir, prompt_embeds=prompt_embeds,
                         dit_path=args.dit, dit_dtype=dit_dtype)
    gen_video, ref_video = fix_video(
        pipe, frames, clean_frame_indices=clean,
        num_frames=args.num_frames, num_repeat_last=args.num_repeat_last,
        height=args.height, width=args.width,
        seed=args.seed, num_inference_steps=args.num_inference_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_video(gen_video, os.path.join(args.output_dir, "generated.mp4"), fps=args.fps)
    save_video(ref_video, os.path.join(args.output_dir, "input.mp4"), fps=args.fps)
    save_video(side_by_side(ref_video, gen_video), os.path.join(args.output_dir, "side_by_side.mp4"), fps=args.fps)
    print(f"Saved results to {args.output_dir}/ (generated.mp4, input.mp4, side_by_side.mp4 = input | generated)")


if __name__ == "__main__":
    main()
