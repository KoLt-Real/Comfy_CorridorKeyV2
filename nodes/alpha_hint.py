from ..pipeline.birefnet import VARIANTS as _VARIANTS  # (repo, resolution) registry

# The dropdown is the registry's key set: a variant added there shows up here, and a
# variant offered here can never be one load_birefnet doesn't know. Import stays cheap —
# pipeline.birefnet pulls torch/transformers lazily, inside its functions.
VARIANTS = list(_VARIANTS)


class CorridorKeyAlphaHint:
    """Generate the coarse alpha hint with BiRefNet (auto-downloaded, cached)."""

    CATEGORY = "CorridorKeyerV2"
    FUNCTION = "hint"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha_hint",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "variant": (VARIANTS, {"default": "General (1024)"}),
                "precision": (["fp16", "fp32"], {"default": "fp16"}),
                "binarize": ("BOOLEAN", {"default": False,
                             "tooltip": "CorridorKey accepts a soft hint; threshold only if you need a hard mask"}),
                "dilate": ("INT", {"default": 0, "min": -64, "max": 64,
                           "tooltip": "grow (positive) or erode (negative) the hint, in pixels"}),
            },
        }

    def hint(self, image, variant, precision, binarize, dilate):
        import comfy.utils

        from ..pipeline.birefnet import hint_batch, load_birefnet

        handle = load_birefnet(variant, precision)
        pbar = comfy.utils.ProgressBar(image.shape[0])
        hint = hint_batch(handle, image, binarize, dilate, progress_cb=lambda _i: pbar.update(1))
        return (hint,)


NODE_CLASS_MAPPINGS = {"CorridorKeyAlphaHint": CorridorKeyAlphaHint}
NODE_DISPLAY_NAME_MAPPINGS = {"CorridorKeyAlphaHint": "CorridorKeyerV2 — Alpha Hint (BiRefNet)"}
