"""Debug node for ComfyUI.

Drop this file into ComfyUI/custom_nodes/.

Accepts any input and prints what it got. Being an output node forces
ComfyUI to execute the branch feeding it, which is the whole point.
"""
import torch


class AnyType(str):
    def __ne__(self, other):
        return False


any_type = AnyType("*")


class DebugPrint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (any_type,),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "debug"
    OUTPUT_NODE = True

    def run(self, value):
        text = describe(value)
        print(f"[DebugPrint] {text}")
        return {"ui": {"text": [text]}}


def describe(value):
    if torch.is_tensor(value):
        return f"tensor shape={tuple(value.shape)} dtype={value.dtype}"
    if isinstance(value, dict):
        for k, v in value.items():
            if torch.is_tensor(v):
                return f"dict key={k!r} shape={tuple(v.shape)} dtype={v.dtype}"
        return f"dict keys={list(value.keys())}"
    return f"type={type(value).__name__} value={value}"


NODE_CLASS_MAPPINGS = {
    "DebugPrint": DebugPrint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DebugPrint": "Debug Print",
}
