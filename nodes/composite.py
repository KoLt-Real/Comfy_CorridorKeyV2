class CorridorKeyPremultComp:
    """Premultiply the FG in linear space and composite a preview.

    ``premult_linear`` + ``alpha`` together are upstream's "Processed" RGBA pass
    (ComfyUI IMAGEs are 3-channel, so the alpha stays on its own MASK wire).
    """

    CATEGORY = "CorridorKeyerV2"
    FUNCTION = "composite"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("premult_linear", "comp_srgb")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fg_srgb": ("IMAGE", {"tooltip": "straight sRGB foreground (despilled)"}),
                "alpha": ("MASK",),
                "generate_comp": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "background": ("IMAGE", {"tooltip": "sRGB; defaults to a checkerboard"}),
            },
        }

    def composite(self, fg_srgb, alpha, generate_comp, background=None):
        from ..pipeline.post import premult_and_comp

        return premult_and_comp(fg_srgb, alpha, background, generate_comp)


NODE_CLASS_MAPPINGS = {"CorridorKeyPremultComp": CorridorKeyPremultComp}
NODE_DISPLAY_NAME_MAPPINGS = {"CorridorKeyPremultComp": "CorridorKeyerV2 — Premultiply & Comp"}
