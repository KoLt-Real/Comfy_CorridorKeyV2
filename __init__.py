"""ComfyFactory CorridorKey — neural green/blue screen keyer nodes.

Wraps the CorridorKey engine (Corridor Digital / Niko Pueringer) with the
community VRAM optimizations (fp16 + fused-SDPA Hiera + optional torch.compile
and EZ-CorridorKey's tiled refiner). See NOTICE for attribution and license
(CC BY-NC-SA 4.0 + Corridor Digital terms — non-commercial).
"""

import os

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
    import folder_paths

    folder_paths.add_model_folder_path("corridorkey", os.path.join(folder_paths.models_dir, "corridorkey"))
    folder_paths.add_model_folder_path("birefnet", os.path.join(folder_paths.models_dir, "birefnet"))
except ImportError:
    pass  # imported outside ComfyUI (tests) — pipeline.prepare has its own fallback

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
