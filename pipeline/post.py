"""Post-processing wrappers over the vendored color_utils, in ComfyUI layouts.

All functions take/return CPU float32 tensors in ComfyUI conventions
(IMAGE [B,H,W,3], MASK [B,H,W]) and run the math on the GPU in chunks —
everything here is fp32 (the fp16 path ends at the model forward).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_CHUNK = 8  # frames per GPU round-trip for cheap per-pixel math

# clean_matte_torch derives its flood-fill iteration count from area_threshold // 20, so
# below 20 it runs *zero* merge passes: every pixel keeps a unique label, no component ever
# reaches the threshold, and the whole matte is wiped to black. Floor the value we hand it.
_MIN_AREA_THRESHOLD = 20


def _device():
    import comfy.model_management as mm

    return mm.get_torch_device()


def _interrupt():
    import comfy.model_management as mm

    mm.throw_exception_if_processing_interrupted()


def despeckle_batch(masks, area_threshold: int, dilation: int, blur_size: int):
    """clean_matte_torch per frame (connected components are the memory driver)."""
    import torch

    from .vendor import color_utils as cu

    if area_threshold <= 0:
        return masks
    area_threshold = max(area_threshold, _MIN_AREA_THRESHOLD)

    dev = _device()
    out = []
    with torch.inference_mode():
        for i in range(masks.shape[0]):
            _interrupt()
            m = masks[i][None, None].to(dev, torch.float32)  # [1,1,H,W]
            cleaned = cu.clean_matte_torch(m, area_threshold, dilation=dilation, blur_size=blur_size)
            out.append(cleaned[0, 0].clamp(0.0, 1.0).cpu())
            del m, cleaned
    return torch.stack(out)


def despill_batch(images, strength: float, screen_channel: int):
    """despill_torch on [B,3,H,W] chunks; luminance-preserving spill removal."""
    import torch

    from .vendor import color_utils as cu

    if strength <= 0.0:
        return images

    dev = _device()
    out = []
    with torch.inference_mode():
        for i in range(0, images.shape[0], _CHUNK):
            _interrupt()
            chunk = images[i:i + _CHUNK].permute(0, 3, 1, 2).to(dev, torch.float32)
            despilled = cu.despill_torch(chunk, strength, screen_channel=screen_channel)
            out.append(despilled.permute(0, 2, 3, 1).clamp(0.0, 1.0).cpu())
            del chunk, despilled
    return torch.cat(out)


def premult_and_comp(fg, alpha, background, generate_comp: bool):
    """sRGB straight FG + alpha -> (premultiplied linear FG, sRGB comp preview).

    Replicates the tail of upstream's ``_postprocess_torch``: linearize, premultiply,
    composite straight over a linear background (checkerboard by default).
    ``premult_linear`` + the alpha wire together are upstream's "Processed" RGBA.
    """
    import torch

    from .vendor import color_utils as cu

    b, h, w = fg.shape[0], fg.shape[1], fg.shape[2]
    if alpha.shape[0] not in (1, b):
        raise ValueError(f"alpha batch {alpha.shape[0]} incompatible with fg batch {b}")
    # Same up-front check for the background: without it a mismatched batch only blows up
    # mid-loop (broadcast error), after the keyer has already run.
    if background is not None and background.shape[0] not in (1, b):
        raise ValueError(
            f"background batch {background.shape[0]} incompatible with fg batch {b}")

    dev = _device()

    bg_lin = None
    if background is not None:
        bg = background
        if bg.shape[1:3] != (h, w):
            import torch.nn.functional as F

            logger.info("Resizing background %s -> %s", tuple(bg.shape[1:3]), (h, w))
            bg = F.interpolate(bg.permute(0, 3, 1, 2), size=(h, w), mode="bilinear",
                               align_corners=False).permute(0, 2, 3, 1)
        bg_lin = cu.srgb_to_linear(bg.to(dev, torch.float32))  # [Bbg,H,W,3]

    # The checkerboard depends only on (w, h, device) — building it per chunk would redo
    # the same full-frame tensor for every 8 frames.
    checker_bg = None
    if generate_comp and bg_lin is None:
        checker_bg = cu.get_checkerboard_linear_torch(w, h, dev).permute(1, 2, 0)[None]

    out_premult, out_comp = [], []
    with torch.inference_mode():
        for i in range(0, b, _CHUNK):
            _interrupt()
            f = fg[i:i + _CHUNK].to(dev, torch.float32)                       # [n,H,W,3]
            a = alpha[(slice(0, 1) if alpha.shape[0] == 1 else slice(i, i + _CHUNK))]
            a = a.to(dev, torch.float32)[..., None]                            # [n|1,H,W,1]

            f_lin = cu.srgb_to_linear(f)
            premult = cu.premultiply(f_lin, a)
            out_premult.append(premult.clamp(min=0.0).cpu())

            if generate_comp:
                if bg_lin is None:
                    bg_chunk = checker_bg                                      # [1,H,W,3]
                else:
                    bg_chunk = bg_lin if bg_lin.shape[0] == 1 else bg_lin[i:i + _CHUNK]
                comp = cu.linear_to_srgb(cu.composite_straight(f_lin, bg_chunk, a))
                out_comp.append(comp.clamp(0.0, 1.0).cpu())
            del f, a, f_lin, premult

    premult_out = torch.cat(out_premult)
    if generate_comp:
        return premult_out, torch.cat(out_comp)
    # comp disabled: hand back a 1px black IMAGE so the wire stays typed.
    return premult_out, torch.zeros(b, 1, 1, 3)


def convert_colorspace(images, direction: str):
    """Exact piecewise sRGB <-> linear (cheap: stays on CPU)."""
    from .vendor import color_utils as cu

    x = images.float()
    if direction == "sRGB_to_linear":
        return cu.srgb_to_linear(x)
    if direction == "linear_to_sRGB":
        return cu.linear_to_srgb(x)
    raise ValueError(f"Unknown direction '{direction}'")
