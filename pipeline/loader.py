"""Build, load and cache the GreenFormer keyer model.

Re-expresses upstream's ``CorridorKeyEngine._load_model`` / ``_compile`` on top of
ComfyUI's device management, with the community VRAM options:

    precision   fp16 (default) | fp32       — weights cast; autocast only for fp32
                                              (upstream: autocast on fp16 weights is slower)
    refiner     full (default) | tiled | off — "tiled" caps the refiner's full-res
                                              activations via EZ-CorridorKey's tent-blended tiles
    compile     off (default)               — torch.compile + dummy warmup, eager fallback

One model is resident at a time (module-level cache keyed by the full config);
the weights are ~300 MB so eviction is about compiled-graph memory, not weights.
"""

from __future__ import annotations

import gc
import logging
import math
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Persist torch.compile autotune cache across runs (upstream does the same;
# default is /tmp which gets wiped on reboot).
_INDUCTOR_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "corridorkey", "inductor")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _INDUCTOR_CACHE)

IMG_SIZE = 2048  # native processing resolution of the released checkpoints

# Recommended (not required) timm: below this the Hiera encoder still runs correctly
# but without the fused-SDPA path, so it uses more VRAM. See requirements.txt.
_TIMM_RECOMMENDED = (1, 0, 27)
_timm_warned = False


def _warn_if_old_timm() -> None:
    """One-shot info nudge when timm predates the low-VRAM Hiera attention path."""
    global _timm_warned
    if _timm_warned:
        return
    _timm_warned = True
    try:
        import timm

        parts = tuple(int(x) for x in timm.__version__.split(".")[:3])
        if parts < _TIMM_RECOMMENDED:
            logger.warning(
                "timm %s detected — CorridorKey works, but timm>=%s enables the fused-SDPA "
                "Hiera path for noticeably lower VRAM. Consider upgrading, or use the "
                "'tiled' refiner to cap memory.",
                timm.__version__, ".".join(map(str, _TIMM_RECOMMENDED)),
            )
    except Exception:  # noqa: BLE001 — a version-string nicety must never break loading
        pass


@dataclass
class CorridorKeyModel:
    """What flows through the CORRIDORKEY_MODEL link in the graph."""

    model: object  # GreenFormer (possibly torch.compile-wrapped)
    screen_color: str
    screen_channel: int  # 0=R, 1=G, 2=B — for downstream despill
    img_size: int
    dtype: object  # torch.dtype of the weights
    mixed_precision: bool  # autocast fp16 during forward (fp32 weights only)
    device: object  # torch.device
    refiner_mode: str  # "full" | "tiled" | "off"
    compiled: bool


_MODEL_CACHE: dict[tuple, CorridorKeyModel] = {}


def _interp_pos_embed(v, target_shape):
    """Bicubic-resize a [1, N, C] pos_embed to a new grid size (upstream logic)."""
    import torch.nn.functional as F

    n_src, n_dst, c = v.shape[1], target_shape[1], v.shape[2]
    grid_src = int(math.sqrt(n_src))
    grid_dst = int(math.sqrt(n_dst))
    v_img = v.permute(0, 2, 1).view(1, c, grid_src, grid_src)
    v_img = F.interpolate(v_img, size=(grid_dst, grid_dst), mode="bicubic", align_corners=False)
    return v_img.flatten(2).transpose(1, 2)


def _load_state(model, ckpt_path: str, device) -> None:
    """safetensors -> strip '_orig_mod.' -> pos_embed interp -> load (strict=False)."""
    from safetensors.torch import load_file

    state_dict = load_file(ckpt_path, device=str(device))
    model_state = model.state_dict()

    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[10:]
        if "pos_embed" in k and k in model_state and v.shape != model_state[k].shape:
            logger.info("Resizing %s from %s to %s", k, tuple(v.shape), tuple(model_state[k].shape))
            v = _interp_pos_embed(v, model_state[k].shape)
        new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    # With refiner="off" the module isn't built: its checkpoint keys are expected leftovers.
    unexpected = [k for k in unexpected if not (model.refiner is None and k.startswith("refiner."))]
    if missing:
        logger.warning("Missing keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)


def _compile(ck: CorridorKeyModel) -> None:
    """torch.compile + dummy-forward warmup; falls back to eager on any failure."""
    import torch

    try:
        logger.info("Compiling GreenFormer (mode=max-autotune, first run takes a while)...")
        compiled = torch.compile(ck.model, mode="max-autotune")
        dummy = torch.zeros(1, 4, ck.img_size, ck.img_size, dtype=ck.dtype, device=ck.device)
        # Same call signature as key_batch (tensor refiner_scale) so the warmup
        # compiles the graph that will actually run.
        scale = torch.ones((), device=ck.device, dtype=torch.float32)
        with torch.inference_mode():
            compiled(dummy, refiner_scale=scale)
        del dummy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ck.model = compiled
        ck.compiled = True
        logger.info("GreenFormer compiled")
    except Exception as exc:  # noqa: BLE001 — compile is an optimization, never fatal
        logger.warning("torch.compile failed (%s); falling back to eager mode.", exc)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_corridorkey(
    screen_color: str = "green",
    precision: str = "fp16",
    refiner_mode: str = "full",
    tile_size: int = 512,
    tile_overlap: int = 128,
    compile_model: bool = False,
) -> CorridorKeyModel:
    """Return a ready CorridorKeyModel, from cache or freshly built."""
    key = (screen_color, precision, refiner_mode, tile_size, tile_overlap, compile_model)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    import torch

    import comfy.model_management as mm

    from .prepare import CKPTS, ensure_corridorkey
    from .vendor.model_transformer import GreenFormer

    _warn_if_old_timm()

    # One resident model: evict any differently-configured instance first.
    if _MODEL_CACHE:
        logger.info("Evicting previous CorridorKey model (config changed)")
        _MODEL_CACHE.clear()
        gc.collect()
        mm.soft_empty_cache()

    device = mm.get_torch_device()
    if precision == "fp16" and device.type not in ("cuda", "mps"):
        # Upstream issue #264: forced fp16 hangs on CPU.
        logger.warning("fp16 requested on %s — using fp32 instead", device.type)
        precision = "fp32"
    dtype = torch.float16 if precision == "fp16" else torch.float32
    # Autocast only helps fp32 weights; on fp16 weights it is pure overhead (upstream).
    mixed_precision = dtype == torch.float32 and device.type == "cuda"

    torch.set_float32_matmul_precision("high")

    ckpt_path = ensure_corridorkey(screen_color)
    logger.info("Loading CorridorKey (%s, %s, refiner=%s) from %s",
                screen_color, precision, refiner_mode, ckpt_path)

    model = GreenFormer(img_size=IMG_SIZE, use_refiner=(refiner_mode != "off"))
    model.to(device)
    model.eval()
    _load_state(model, ckpt_path, device)
    model.to(dtype)

    if refiner_mode == "tiled" and model.refiner is not None:
        model.refiner._tile_size = tile_size
        model.refiner._tile_overlap = tile_overlap

    ck = CorridorKeyModel(
        model=model,
        screen_color=screen_color,
        screen_channel=CKPTS[screen_color][2],
        img_size=IMG_SIZE,
        dtype=dtype,
        mixed_precision=mixed_precision,
        device=device,
        refiner_mode=refiner_mode,
        compiled=False,
    )

    if compile_model:
        if refiner_mode == "tiled" and model.refiner is not None:
            # Tiled path: compile only the fixed-shape tile kernel; the tile loop
            # itself is @torch.compiler.disable so the full-graph compile skips it.
            model.refiner.compile_tile_kernel()
        _compile(ck)

    _MODEL_CACHE[key] = ck
    return ck
