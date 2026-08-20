class CorridorKeyDespeckle:
    """Remove small disconnected islands (tracking markers, noise) from a matte."""

    CATEGORY = "CorridorKeyerV2"
    FUNCTION = "despeckle"
    RETURN_TYPES = ("MASK",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "min_island_size": ("INT", {"default": 400, "min": 0, "max": 4096,
                                    "tooltip": "islands below this pixel area are removed "
                                               "(0 = off; values under 20 are treated as 20)"}),
                "dilation": ("INT", {"default": 25, "min": 0, "max": 128}),
                "blur_size": ("INT", {"default": 5, "min": 0, "max": 51}),
            },
        }

    def despeckle(self, mask, min_island_size, dilation, blur_size):
        from ..pipeline.post import despeckle_batch

        return (despeckle_batch(mask, min_island_size, dilation, blur_size),)


NODE_CLASS_MAPPINGS = {"CorridorKeyDespeckle": CorridorKeyDespeckle}
NODE_DISPLAY_NAME_MAPPINGS = {"CorridorKeyDespeckle": "CorridorKeyerV2 — Despeckle Matte"}
