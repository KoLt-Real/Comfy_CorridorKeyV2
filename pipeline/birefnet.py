"""BiRefNet alpha-hint generation (the lightweight hint model from upstream).

Loads the ZhengPeng7 BiRefNet checkpoints through transformers'
``AutoModelForImageSegmentation`` (remote code — verified working with
transformers 5.x). The hint model has its own cache, independent from the
keyer's, so both can stay resident (~1.3 GB of weights combined).
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# UI variant -> (HF repo, processing resolution)
VARIANTS: dict[str, tuple[str, int]] = {
    "General (1024)": ("ZhengPeng7/BiRefNet", 1024),
    "General-HR (2048)": ("ZhengPeng7/BiRefNet_HR", 2048),
    "Matting (1024)": ("ZhengPeng7/BiRefNet-matting", 1024),
    "Matting-HR (2048)": ("ZhengPeng7/BiRefNet_HR-matting", 2048),
    "Lite (1024)": ("ZhengPeng7/BiRefNet_lite", 1024),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Upstream binarizes at 10/255.
_BINARIZE_THRESHOLD = 10.0 / 255.0


@dataclass
class BiRefNetHandle:
    model: object
    resolution: int
    dtype: object
    device: object


_CACHE: dict[tuple, BiRefNetHandle] = {}


def load_birefnet(variant: str, precision: str = "fp16") -> BiRefNetHandle:
    """Return a ready BiRefNet handle, from cache or freshly loaded (auto-download)."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown BiRefNet variant '{variant}'. Valid: {sorted(VARIANTS)}")
    key = (variant, precision)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    import torch

    import comfy.model_management as mm

    from .prepare import ensure_birefnet

    if _CACHE:
        logger.info("Evicting previous BiRefNet (variant/precision changed)")
        _CACHE.clear()
        gc.collect()
        mm.soft_empty_cache()

    device = mm.get_torch_device()
    if precision == "fp16" and device.type not in ("cuda", "mps"):
        logger.warning("fp16 requested on %s — using fp32 instead", device.type)
        precision = "fp32"
    dtype = torch.float16 if precision == "fp16" else torch.float32

    repo_id, resolution = VARIANTS[variant]
    local_dir = ensure_birefnet(repo_id)
    logger.info("Loading BiRefNet %s (%s) from %s", variant, precision, local_dir)

    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(local_dir, trust_remote_code=True)
    model.to(device=device, dtype=dtype)
    model.eval()

    handle = BiRefNetHandle(model=model, resolution=resolution, dtype=dtype, device=device)
    _CACHE[key] = handle
    return handle


def _erode(mask, radius: int):
    """Erosion for a [0,1] mask via the vendored (dilate-based) morphology."""
    from .vendor import color_utils as cu

    return 1.0 - cu.dilate_mask(1.0 - mask, radius)


def hint_batch(
    handle: BiRefNetHandle,
    images,  # [B,H,W,3] float32 0-1 (ComfyUI IMAGE, CPU)
    binarize: bool = False,
    dilate: int = 0,
    progress_cb: Callable[[int], None] | None = None,
):
    """Generate a [B,H,W] alpha hint (soft by default — CorridorKey takes linear hints)."""
    import torch
    import torchvision.transforms.v2 as T
    import torchvision.transforms.v2.functional as TF

    import comfy.model_management as mm

    from .vendor import color_utils as cu

    b, h, w = images.shape[0], images.shape[1], images.shape[2]
    res = handle.resolution
    mean = torch.tensor(IMAGENET_MEAN, dtype=handle.dtype, device=handle.device)
    std = torch.tensor(IMAGENET_STD, dtype=handle.dtype, device=handle.device)

    out = []
    with torch.inference_mode():
        for i in range(b):
            mm.throw_exception_if_processing_interrupted()

            x = images[i].permute(2, 0, 1)[None].to(handle.device, handle.dtype, non_blocking=True)
            x = TF.resize(x, [res, res], interpolation=T.InterpolationMode.BILINEAR)
            x = TF.normalize(x, mean, std)

            preds = handle.model(x)
            pred = (preds[-1] if isinstance(preds, (list, tuple)) else preds).sigmoid().float()
            del x, preds

            mask = TF.resize(pred, [h, w], interpolation=T.InterpolationMode.BILINEAR)[0, 0]
            del pred

            if dilate > 0:
                mask = cu.dilate_mask(mask, dilate)
            elif dilate < 0:
                mask = _erode(mask, -dilate)
            if binarize:
                mask = (mask > _BINARIZE_THRESHOLD).float()

            out.append(mask.clamp(0.0, 1.0).cpu())
            del mask

            if progress_cb is not None:
                progress_cb(i)

    mm.soft_empty_cache()
    return torch.stack(out)
