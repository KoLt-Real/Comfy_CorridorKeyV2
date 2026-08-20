from ..pipeline.prepare import CKPTS as _CKPTS  # screen_color -> (repo, file, channel)

# Same rule as the BiRefNet variants: the offered screen colors are exactly those a
# checkpoint exists for, so the dropdown can't propose one ensure_corridorkey rejects.
SCREEN_COLORS = list(_CKPTS)


class CorridorKeyModelLoader:
    """Load (and cache) the GreenFormer keyer — auto-downloads the checkpoint."""

    CATEGORY = "CorridorKeyerV2"
    FUNCTION = "load"
    RETURN_TYPES = ("CORRIDORKEY_MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "screen_color": (SCREEN_COLORS,),
                "precision": (["fp16", "fp32"], {"default": "fp16"}),
                "refiner": (["full", "tiled", "off"], {"default": "full",
                            "tooltip": "tiled: EZ-CorridorKey low-VRAM refiner (identical output)"}),
                "torch_compile": ("BOOLEAN", {"default": False,
                                  "tooltip": "~1.9GB VRAM regime, but slow first-run compilation"}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "tile_overlap": ("INT", {"default": 128, "min": 32, "max": 512, "step": 32}),
            },
        }

    def load(self, screen_color, precision, refiner, torch_compile, tile_size=512, tile_overlap=128):
        from ..pipeline.loader import load_corridorkey

        model = load_corridorkey(
            screen_color=screen_color,
            precision=precision,
            refiner_mode=refiner,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            compile_model=torch_compile,
        )
        return (model,)


NODE_CLASS_MAPPINGS = {"CorridorKeyModelLoader": CorridorKeyModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"CorridorKeyModelLoader": "CorridorKeyerV2 — Load Model"}
