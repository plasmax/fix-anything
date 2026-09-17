"""FixAnything nodes for ComfyUI.

FixAnything replaces WanImageToVideo. The stock node encodes the first
frame plus grey filler; FixAnything needs the whole degraded render encoded, plus a
mask marking which frames to keep.
"""
import os

import torch

import comfy.model_management
import comfy.sd
import comfy.utils
import node_helpers
import folder_paths

DEFAULT_PROMPT = "A clean, high-quality, photorealistic video with sharp details, smooth motion, and natural lighting."
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def parse_indices(text, length):
    indices = []
    for part in text.split():
        i = int(part)
        if i < 0:
            i += length
        if not 0 <= i < length:
            raise ValueError(f"clean frame index {part} is outside 0..{length - 1}")
        indices.append(i)
    return indices


class FixAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 16384, "step": 16}),
                "clean_frame_indices": ("STRING", {"default": "0 60 61 62 63 64"}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, images, width, height,
               clean_frame_indices, clip_vision_output=None):
        length = images.shape[0]
        if length % 4 != 1:
            raise ValueError(f"need 4n+1 frames, got {length}")

        images = comfy.utils.common_upscale(
            images.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        concat_latent_image = vae.encode(images[:, :, :, :3])

        lat_h, lat_w = concat_latent_image.shape[-2], concat_latent_image.shape[-1]

        # ComfyUI flips this later, so 1 means refine and 0 means keep.
        mask = torch.ones((1, 1, length, lat_h, lat_w))
        for i in parse_indices(clean_frame_indices, length):
            mask[:, :, i] = 0.0

        # Frame 0 fills a whole latent frame, the rest go in groups of 4.
        mask = torch.cat([mask[:, :, :1].repeat(1, 1, 3, 1, 1), mask], dim=2)
        mask = mask.view(1, mask.shape[2] // 4, 4, lat_h, lat_w).transpose(1, 2)

        values = {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        if clip_vision_output is not None:
            values["clip_vision_output"] = clip_vision_output
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)

        latent = torch.zeros(
            [1, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=comfy.model_management.intermediate_device())
        return (positive, negative, {"samples": latent})


class FixAnythingPromptEmbeds:
    """Loads prompt_embeds.pt so umT5 is not needed.

    Takes a full path, or a bare filename that lives in models/text_encoders/.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embeds_file": ("STRING", {"default": "prompt_embeds.pt"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "load"
    CATEGORY = "conditioning/video_models"

    def load(self, embeds_file):
        path = embeds_file
        if not os.path.isfile(path):
            path = folder_paths.get_full_path_or_raise("text_encoders", embeds_file)
        data = torch.load(path, map_location="cpu", weights_only=True)
        prompts = data["prompts"]
        for text in (DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT):
            if text not in prompts:
                raise KeyError(f"no embedding for {text!r}. available: {list(prompts)}")
        return (
            [[prompts[DEFAULT_PROMPT].float(), {}]],
            [[prompts[DEFAULT_NEGATIVE_PROMPT].float(), {}]],
        )


class FixAnythingLoraLoader:
    """Same as LoraLoaderModelOnly, but adds the diffusion_model. prefix the keys are missing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, model, lora_name, strength_model):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(path, safe_load=True)

        prefixed = {}
        for k, v in lora.items():
            if not k.startswith("diffusion_model."):
                k = "diffusion_model." + k
            prefixed[k] = v

        model, _ = comfy.sd.load_lora_for_models(model, None, prefixed, strength_model, 0)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "FixAnything": FixAnything,
    "FixAnythingPromptEmbeds": FixAnythingPromptEmbeds,
    "FixAnythingLoraLoader": FixAnythingLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FixAnything": "FixAnything Conditioning",
    "FixAnythingPromptEmbeds": "FixAnything Prompt Embeds",
    "FixAnythingLoraLoader": "FixAnything Lora Loader",
}
