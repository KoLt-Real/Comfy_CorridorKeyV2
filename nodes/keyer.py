import logging

logger = logging.getLogger(__name__)

# Auto-hint defaults when no mask is wired (matches CorridorKey's "hint is generated
# for you" behaviour). A slight dilation keeps the subject fully covered.
_AUTO_HINT_VARIANT = "General (1024)"
_AUTO_HINT_DILATE = 8


class CorridorKeyKeyer:
    """Run the keyer: image (+ optional coarse alpha hint) -> raw alpha matte + straight sRGB FG.

    If no ``mask`` is wired, a coarse hint is generated internally with BiRefNet —
    the alpha hint is optional, exactly like the CorridorKey app. Wire the dedicated
    "Alpha Hint (BiRefNet)" node (or any MASK) when you want explicit control.

    Outputs are the raw model predictions resized back to the input resolution;
    chain Despeckle (alpha) / Despill (fg) / Premultiply & Comp downstream.
    """

    CATEGORY = "CorridorKeyerV2"
    FUNCTION = "key"
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("alpha", "fg_srgb")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CORRIDORKEY_MODEL",),
                "image": ("IMAGE",),
                "refiner_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1,
                                  "tooltip": "scales the refiner's alpha-edge correction"}),
                "input_is_linear": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "coarse alpha hint (rough is fine); batch of 1 broadcasts. "
                                  "Leave unconnected to auto-generate one with BiRefNet."}),
            },
        }

    def key(self, model, image, refiner_scale, input_is_linear, mask=None):
        import comfy.utils

        from ..pipeline.keyer import key_batch

        if mask is None:
            # No hint wired: generate one with BiRefNet (auto-downloads on first use).
            from ..pipeline.birefnet import hint_batch, load_birefnet

            logger.info("CorridorKey Keyer: no mask wired — auto-generating a hint with BiRefNet (%s)",
                        _AUTO_HINT_VARIANT)
            hint_pbar = comfy.utils.ProgressBar(image.shape[0])
            handle = load_birefnet(_AUTO_HINT_VARIANT, "fp16")
            mask = hint_batch(handle, image, binarize=False, dilate=_AUTO_HINT_DILATE,
                              progress_cb=lambda _i: hint_pbar.update(1))

        pbar = comfy.utils.ProgressBar(image.shape[0])
        alpha, fg = key_batch(
            model, image, mask,
            refiner_scale=refiner_scale,
            input_is_linear=input_is_linear,
            progress_cb=lambda _i: pbar.update(1),
        )
        return (alpha, fg)


NODE_CLASS_MAPPINGS = {"CorridorKeyKeyer": CorridorKeyKeyer}
NODE_DISPLAY_NAME_MAPPINGS = {"CorridorKeyKeyer": "CorridorKeyerV2 — Keyer"}
