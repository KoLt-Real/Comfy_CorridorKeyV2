# Deliberately local rather than imported from pipeline.vendor.color_utils: nodes must not
# pull torch/cv2 at registration time. The vendored table only knows green/blue anyway —
# despill is a plain channel subtraction, so it also works on a red screen, which no
# checkpoint covers.
_SCREEN_CHANNELS = {"red": 0, "green": 1, "blue": 2}


class CorridorKeyDespill:
    """Luminance-preserving screen-spill removal on a straight sRGB foreground."""

    CATEGORY = "CorridorKeyerV2"
    FUNCTION = "despill"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "screen_color": (["green", "blue", "red"], {"default": "green"}),
            },
            "optional": {
                "model": ("CORRIDORKEY_MODEL", {"tooltip": "if connected, its screen color overrides the widget"}),
            },
        }

    def despill(self, image, strength, screen_color, model=None):
        from ..pipeline.post import despill_batch

        channel = model.screen_channel if model is not None else _SCREEN_CHANNELS[screen_color]
        return (despill_batch(image, strength, channel),)


NODE_CLASS_MAPPINGS = {"CorridorKeyDespill": CorridorKeyDespill}
NODE_DISPLAY_NAME_MAPPINGS = {"CorridorKeyDespill": "CorridorKeyerV2 — Despill"}
