"""Pre-encode prompts with the umT5-XXL text encoder so inference can run without it.

FixAnything uses fixed prompts, so the 11 GB text encoder only ever produces the same few
embeddings. This script computes them once and stores them in a small file; afterwards
`run_inference.py` loads the embeddings instead of the encoder and
`models_t5_umt5-xxl-enc-bf16.pth` can be deleted.

Example:
    python scripts/encode_prompts.py --model_dir checkpoints
    # then optionally:  rm checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth

Extra prompts can be added at any time (the output file is merged, not overwritten):
    python scripts/encode_prompts.py --prompt "..." --negative_prompt "..."

Output: a torch file (loadable with `weights_only=True`) with
    {"format": ..., "text_len": 512, "prompts": {prompt_text: bf16 tensor of shape (1, 512, 4096)}, "seq_lens": {...}}
The tensors are exactly what `WanPrompter.encode_prompt()` returns (padding positions zeroed).
"""
import argparse
import glob
import os
import sys

import torch
from diffsynth.models import ModelManager
from diffsynth.prompters import WanPrompter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_inference import WAN_MODEL_ID, DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT  # noqa: E402

FORMAT = "fixanything.prompt_embeds.v1"
T5_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
TOKENIZER_DIR = os.path.join("google", "umt5-xxl")


def default_prompt_embeds_path(model_dir):
    return os.path.join(model_dir, "prompt_embeds.pt")


def load_prompt_embeds(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"{path} is not a FixAnything prompt-embedding file (expected format {FORMAT!r}).")
    return data


def parse_args():
    p = argparse.ArgumentParser(description="Pre-encode FixAnything prompts with umT5-XXL")
    p.add_argument("--model_dir", type=str, default="checkpoints",
                   help="Folder containing Wan-AI/Wan2.1-I2V-14B-480P/ (text encoder + google/umt5-xxl tokenizer).")
    p.add_argument("--output", type=str, default=None,
                   help="Output file (default: <model_dir>/prompt_embeds.pt). Existing entries are kept and merged.")
    p.add_argument("--prompt", type=str, action="append", default=[],
                   help="Additional positive prompt to encode (repeatable).")
    p.add_argument("--negative_prompt", type=str, action="append", default=[],
                   help="Additional negative prompt to encode (repeatable).")
    p.add_argument("--no_defaults", action="store_true",
                   help="Do not encode the built-in default prompt / negative prompt / empty prompt.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true", help="Re-encode prompts that are already in the output file.")
    return p.parse_args()


def main():
    args = parse_args()
    output = args.output or default_prompt_embeds_path(args.model_dir)
    wan_dir = os.path.join(args.model_dir, WAN_MODEL_ID)
    t5_path = os.path.join(wan_dir, T5_FILE)
    tokenizer_path = os.path.join(wan_dir, TOKENIZER_DIR)
    if not os.path.isfile(t5_path):
        raise FileNotFoundError(f"Text encoder not found: {t5_path}")
    if not glob.glob(os.path.join(tokenizer_path, "*")):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    # (prompt, positive) pairs to encode. `positive` is passed to the prompter for parity with the pipeline
    # (it only matters if prompt refiners are registered, which FixAnything does not do).
    todo = []
    if not args.no_defaults:
        todo += [(DEFAULT_PROMPT, True), (DEFAULT_NEGATIVE_PROMPT, False), ("", False)]
    todo += [(p, True) for p in args.prompt] + [(p, False) for p in args.negative_prompt]

    if os.path.isfile(output):
        data = load_prompt_embeds(output)
        print(f"Merging into existing {output} ({len(data['prompts'])} prompts).")
    else:
        data = {"format": FORMAT, "text_encoder": T5_FILE, "text_len": 512, "prompts": {}, "seq_lens": {}}
    if not args.force:
        todo = [(p, pos) for p, pos in todo if p not in data["prompts"]]
    if not todo:
        print("Nothing to encode: all prompts already present. Use --force to re-encode.")
        return

    print(f"Loading text encoder from {t5_path} ...")
    model_manager = ModelManager()
    model_manager.load_model(t5_path, device="cpu", torch_dtype=torch.bfloat16)
    text_encoder = model_manager.fetch_model("wan_video_text_encoder")
    if text_encoder is None:
        raise RuntimeError(f"DiffSynth did not recognise {t5_path} as the Wan text encoder.")
    text_encoder = text_encoder.to(args.device).eval()
    prompter = WanPrompter(tokenizer_path=tokenizer_path)
    prompter.fetch_models(text_encoder)

    with torch.no_grad():
        for prompt, positive in todo:
            emb = prompter.encode_prompt(prompt, positive=positive, device=args.device)
            ids, mask = prompter.tokenizer(prompt, return_mask=True, add_special_tokens=True)
            seq_len = int(mask.gt(0).sum())
            data["prompts"][prompt] = emb.to("cpu", torch.bfloat16).contiguous()
            data["seq_lens"][prompt] = seq_len
            print(f"  encoded ({'positive' if positive else 'negative'}, {seq_len} tokens, shape {tuple(emb.shape)}): "
                  f"{prompt[:60]!r}{'...' if len(prompt) > 60 else ''}")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(data, output)
    print(f"Saved {len(data['prompts'])} prompt embeddings to {output} ({os.path.getsize(output) / 1e6:.1f} MB).")
    print(f"You can now delete the text encoder: {t5_path}")


if __name__ == "__main__":
    main()
