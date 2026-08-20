class CorridorKeyColorspace:
    """Exact piecewise sRGB <-> linear conversion (CorridorKey's own math)."""

    CATEGORY = "CorridorKeyerV2"
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["sRGB_to_linear", "linear_to_sRGB"],),
            },
        }

    def convert(self, image, direction):
        from ..pipeline.post import convert_colorspace

        return (convert_colorspace(image, direction),)


NODE_CLASS_MAPPINGS = {"CorridorKeyColorspace": CorridorKeyColorspace}
NODE_DISPLAY_NAME_MAPPINGS = {"CorridorKeyColorspace": "CorridorKeyerV2 — Colorspace Convert"}
