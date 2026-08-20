"""The keyer inference path — upstream's ``process_frame`` decomposed for ComfyUI.

ComfyUI hands us IMAGE batches [B,H,W,3] float32 0-1 (CPU) and MASK batches
[B,H,W]. We process one frame at a time at the model's native 2048x2048 window
(VRAM hygiene straight from upstream: free the 4-channel input right after the
forward, cast predictions to fp32 immediately, move each frame's outputs to CPU
before the next frame). Post-processing (despill/despeckle/premult) lives in
``post.py`` — this module returns the raw predictions resized back.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


def _prep_masks(masks, images):
    """Broadcast/validate a MASK batch against an IMAGE batch. Returns [B_m,H,W]."""
    if masks.dim() == 2:  # [H,W] -> [1,H,W]
        masks = masks[None]
    b_img, h, w = images.shape[0], images.shape[1], images.shape[2]
    b_mask = masks.shape[0]
    if b_mask not in (1, b_img):
        raise ValueError(
            f"mask batch {b_mask} incompatible with image batch {b_img} (need 1 or {b_img})"
        )
    if masks.shape[-2:] != (h, w):
        import torch.nn.functional as F

        logger.info("Resizing mask %s -> %s to match the image", tuple(masks.shape[-2:]), (h, w))
        masks = F.interpolate(masks[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
    return masks


def preprocess_frame(image_1chw, mask_11hw, img_size: int, input_is_linear: bool, mean, std):
    """[1,3,H,W]+[1,1,H,W] on device -> ImageNet-normalized [1,4,S,S] (upstream order)."""
    import torchvision.transforms.v2 as T
    import torchvision.transforms.v2.functional as TF

    from .vendor import color_utils as cu

    # Resize first; if the source is linear, resize in linear to preserve
    # energy/highlights, THEN convert to sRGB for the model (upstream).
    image_1chw = TF.resize(image_1chw, [img_size, img_size], interpolation=T.InterpolationMode.BILINEAR)
    if input_is_linear:
        image_1chw = cu.linear_to_srgb(image_1chw)
    mask_11hw = TF.resize(mask_11hw, [img_size, img_size], interpolation=T.InterpolationMode.BILINEAR)

    import torch

    image_1chw = TF.normalize(image_1chw, mean, std)
    return torch.concat((image_1chw, mask_11hw), -3)  # [1,4,S,S]


def run_greenformer(ck, inp, refiner_scale):
    """One forward pass. Returns (alpha [1,1,S,S], fg [1,3,S,S]) in fp32."""
    import torch

    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()

    # Always pass the scale as a device tensor: a Python float would be baked in
    # as a constant by torch.compile and each new value would trigger a recompile.
    scale_t = torch.as_tensor(refiner_scale, device=ck.device, dtype=torch.float32)
    with torch.autocast(device_type=ck.device.type, dtype=torch.float16, enabled=ck.mixed_precision):
        pred = ck.model(inp, refiner_scale=scale_t)

    # fp16 numerics end here — everything downstream is fp32.
    return pred["alpha"].float(), pred["fg"].float()


def resize_back(alpha_11ss, fg_13ss, h: int, w: int):
    """GPU bilinear resize back to the source resolution (upstream's GPU path)."""
    import torchvision.transforms.v2 as T
    import torchvision.transforms.v2.functional as TF

    alpha = TF.resize(alpha_11ss, [h, w], interpolation=T.InterpolationMode.BILINEAR)
    fg = TF.resize(fg_13ss, [h, w], interpolation=T.InterpolationMode.BILINEAR)
    return alpha, fg


def key_batch(
    ck,
    images,  # [B,H,W,3] float32 0-1 (ComfyUI IMAGE, CPU)
    masks,   # [B,H,W] or [1,H,W] or [H,W] float32 0-1 (alpha hint, linear)
    refiner_scale: float = 1.0,
    input_is_linear: bool = False,
    progress_cb: Callable[[int], None] | None = None,
):
    """Key a whole batch frame by frame. Returns (alpha [B,H,W], fg_srgb [B,H,W,3]) on CPU."""
    import torch

    import comfy.model_management as mm

    masks = _prep_masks(masks, images)
    broadcast = masks.shape[0] == 1 and images.shape[0] > 1
    b, h, w = images.shape[0], images.shape[1], images.shape[2]

    mean = torch.tensor((0.485, 0.456, 0.406), dtype=ck.dtype, device=ck.device)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=ck.dtype, device=ck.device)

    out_alpha, out_fg = [], []
    with torch.inference_mode():
        for i in range(b):
            mm.throw_exception_if_processing_interrupted()

            image = images[i].permute(2, 0, 1)[None].to(ck.device, ck.dtype, non_blocking=True)
            mask = masks[0 if broadcast else i][None, None].to(ck.device, ck.dtype, non_blocking=True)

            inp = preprocess_frame(image, mask, ck.img_size, input_is_linear, mean, std)
            del image, mask

            alpha, fg = run_greenformer(ck, inp, refiner_scale)
            del inp

            alpha, fg = resize_back(alpha, fg, h, w)
            out_alpha.append(alpha[0, 0].clamp(0.0, 1.0).cpu())
            out_fg.append(fg[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu())
            del alpha, fg

            if progress_cb is not None:
                progress_cb(i)

    mm.soft_empty_cache()
    return torch.stack(out_alpha).float(), torch.stack(out_fg).float()
