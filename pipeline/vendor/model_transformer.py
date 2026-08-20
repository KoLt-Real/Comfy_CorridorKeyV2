# Vendored from:
#   https://github.com/nikopueringer/CorridorKey
#   file:   CorridorKeyModule/core/model_transformer.py
#   commit: 97e55a453060745bead1befd293f6e523c4b845c (main, 2026-07)
# with two community improvements merged from:
#   https://github.com/edenaion/EZ-CorridorKey
#   file:   CorridorKeyModule/core/model_transformer.py
#   commit: ded54c8235a1776c070aa92233d842572698608d (main, 2026-07)
#   1. Tiled CNN refiner (tent-blended 512px tiles — caps full-resolution
#      activations, the dominant VRAM term at 2048x2048). Credit: Marclie /
#      Ed Zisk (EZ-CorridorKey). Checkpoint keys are unchanged.
#   2. `refiner_scale` as a forward() kwarg (replaces upstream's forward hook);
#      the scale only affects the alpha delta — FG always gets the full
#      correction to avoid Hiera attention-window block artifacts.
# The merged blocks are fenced with "EZ-CorridorKey" comments below.
# License: CC BY-NC-SA 4.0 + Corridor Digital additional terms (see NOTICE).
# Note: the Hiera FlashAttention/SDPA patch that EZ-CorridorKey ships is NOT
# vendored — timm >= 1.0.27 contains the fused fix natively (hence the pin
# in requirements.txt).
from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Linear embedding: C_in -> C_out."""

    def __init__(self, input_dim: int = 2048, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DecoderHead(nn.Module):
    def __init__(
        self, feature_channels: list[int] | None = None, embedding_dim: int = 256, output_dim: int = 1
    ) -> None:
        super().__init__()
        if feature_channels is None:
            feature_channels = [112, 224, 448, 896]

        # MLP layers to unify channel dimensions
        self.linear_c4 = MLP(input_dim=feature_channels[3], embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=feature_channels[2], embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=feature_channels[1], embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=feature_channels[0], embed_dim=embedding_dim)

        # Fuse
        self.linear_fuse = nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.relu = nn.ReLU(inplace=True)

        # Predict
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv2d(embedding_dim, output_dim, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = features

        n, _, h, w = c4.shape

        # Resize to C1 size (which is H/4)
        _c4 = self.linear_c4(c4.flatten(2).transpose(1, 2)).transpose(1, 2).view(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.shape[2:], mode="bilinear", align_corners=False)

        _c3 = self.linear_c3(c3.flatten(2).transpose(1, 2)).transpose(1, 2).view(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.shape[2:], mode="bilinear", align_corners=False)

        _c2 = self.linear_c2(c2.flatten(2).transpose(1, 2)).transpose(1, 2).view(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.shape[2:], mode="bilinear", align_corners=False)

        _c1 = self.linear_c1(c1.flatten(2).transpose(1, 2)).transpose(1, 2).view(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = self.bn(_c)
        _c = self.relu(_c)

        x = self.dropout(_c)
        x = self.classifier(x)

        return x


class RefinerBlock(nn.Module):
    """
    Residual Block with Dilation and GroupNorm (Safe for Batch Size 2).
    """

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gn1 = nn.GroupNorm(8, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += residual
        out = self.relu(out)
        return out


class CNNRefinerModule(nn.Module):
    """
    Dilated Residual Refiner (Receptive Field ~65px).
    designed to solve Macroblocking artifacts from Hiera.
    Structure: Stem -> Res(d1) -> Res(d2) -> Res(d4) -> Res(d8) -> Projection.
    """

    def __init__(self, in_channels: int = 7, hidden_channels: int = 64, out_channels: int = 4) -> None:
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Dilated Residual Blocks (RF Expansion)
        self.res1 = RefinerBlock(hidden_channels, dilation=1)
        self.res2 = RefinerBlock(hidden_channels, dilation=2)
        self.res3 = RefinerBlock(hidden_channels, dilation=4)
        self.res4 = RefinerBlock(hidden_channels, dilation=8)

        # Final Projection (No Activation, purely additive logits)
        self.final = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        # Tiny Noise Init (Whisper) - Provides gradients without shock
        nn.init.normal_(self.final.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.final.bias, 0)

        # --- EZ-CorridorKey addition: tiled refiner state (no checkpoint keys) ---
        self._tile_size = 0
        self._tile_overlap = 128
        self._compiled_process_tile = None
        self._blend_weight_cache = {}
        # --- end EZ-CorridorKey addition ---

    # --- EZ-CorridorKey addition: tiled refiner ------------------------------
    def _process_tile_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Run refiner pipeline on a single tile. x: [B, 7, H, W]."""
        feat = self.stem(x)
        feat = self.res1(feat)
        feat = self.res2(feat)
        feat = self.res3(feat)
        feat = self.res4(feat)
        # Output Scaling (10x Boost)
        # Allows the Refiner to predict small stable values (e.g. 0.5) that become strong corrections (5.0).
        return self.final(feat) * 10.0

    def compile_tile_kernel(self) -> None:
        """Compile the fixed-shape tile CNN without changing checkpoint keys."""
        if self._compiled_process_tile is None:
            self._compiled_process_tile = torch.compile(
                self._process_tile_impl, dynamic=False, fullgraph=True,
            )

    def _process_tile(self, x: torch.Tensor) -> torch.Tensor:
        if self._compiled_process_tile is not None:
            return self._compiled_process_tile(x)
        return self._process_tile_impl(x)

    @torch.compiler.disable
    def _forward_tiled(self, full_input: torch.Tensor, out_channels: int) -> torch.Tensor:
        tile_size = self._tile_size
        tile_overlap = self._tile_overlap
        stride = tile_size - tile_overlap
        if stride <= 0:
            raise ValueError(
                f"Invalid refiner tiling parameters: tile_size={tile_size}, "
                f"tile_overlap={tile_overlap}"
            )

        B, _, H, W = full_input.shape
        delta_sum = full_input.new_zeros((B, out_channels, H, W))
        weight_sum = full_input.new_zeros((B, 1, H, W))

        for y0 in range(0, H, stride):
            for x0 in range(0, W, stride):
                y1 = min(y0 + tile_size, H)
                x1 = min(x0 + tile_size, W)
                # Adjust start to ensure full tile_size when possible.
                y0_adj = max(0, y1 - tile_size)
                x0_adj = max(0, x1 - tile_size)

                tile = full_input[:, :, y0_adj:y1, x0_adj:x1]
                tile_delta = self._process_tile(tile)

                tile_h, tile_w = tile_delta.shape[2], tile_delta.shape[3]
                # Only ramp edges that overlap with adjacent tiles, not image boundaries.
                at_top = (y0_adj == 0)
                at_bottom = (y1 == H)
                at_left = (x0_adj == 0)
                at_right = (x1 == W)
                blend_w = self._get_blend_weight(
                    tile_h, tile_w, tile_overlap,
                    at_top, at_bottom, at_left, at_right,
                    full_input.device, full_input.dtype,
                )

                delta_sum[:, :, y0_adj:y1, x0_adj:x1] += tile_delta * blend_w
                weight_sum[:, :, y0_adj:y1, x0_adj:x1] += blend_w

        return delta_sum / weight_sum.clamp(min=1e-6)

    def _get_blend_weight(self, h, w, overlap, at_top, at_bottom, at_left, at_right, device, dtype):
        key = (h, w, overlap, at_top, at_bottom, at_left, at_right, device, dtype)
        weight = self._blend_weight_cache.get(key)
        if weight is None:
            weight = self._blend_weight(h, w, overlap,
                                        at_top, at_bottom, at_left, at_right,
                                        device, dtype)
            self._blend_weight_cache[key] = weight
        return weight

    @staticmethod
    def _blend_weight(h, w, overlap, at_top, at_bottom, at_left, at_right, device, dtype):
        """Linear blend ramp for tile overlap regions.

        Edges at image boundaries (at_top/bottom/left/right=True) keep
        full weight — only internal tile-to-tile overlaps get ramped.
        """
        weight = torch.ones(1, 1, h, w, device=device, dtype=dtype)
        if overlap <= 0:
            return weight
        ramp = torch.linspace(0.0, 1.0, overlap, device=device, dtype=dtype)
        for i in range(min(overlap, h)):
            if not at_top:
                weight[:, :, i, :] *= ramp[i]
            if not at_bottom:
                weight[:, :, h - 1 - i, :] *= ramp[i]
        for i in range(min(overlap, w)):
            if not at_left:
                weight[:, :, :, i] *= ramp[i]
            if not at_right:
                weight[:, :, :, w - 1 - i] *= ramp[i]
        return weight

    def forward(self, img: torch.Tensor, coarse_pred: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional tiled processing.

        Tile size is set via the _tile_size attribute (0 = full resolution).
        """
        # img: [B, 3, H, W] (ImageNet-normalized RGB), coarse_pred: [B, 4, H, W]
        full_input = torch.cat([img, coarse_pred], dim=1)  # [B, 7, H, W]
        tile_size = self._tile_size

        if tile_size <= 0:
            return self._process_tile(full_input)

        _, _, H, W = full_input.shape

        # Skip tiling if image fits in a single tile
        if H <= tile_size and W <= tile_size:
            return self._process_tile(full_input)

        return self._forward_tiled(full_input, coarse_pred.shape[1])
    # --- end EZ-CorridorKey addition ------------------------------------------


class GreenFormer(nn.Module):
    def __init__(
        self,
        encoder_name: str = "hiera_base_plus_224.mae_in1k_ft_in1k",
        in_channels: int = 4,
        img_size: int = 512,
        use_refiner: bool = True,
    ) -> None:
        super().__init__()

        # --- Encoder ---
        # Load Pretrained Hiera
        # 1. Create Target Model (512x512, Random Weights)
        # We use features_only=True, which wraps it in FeatureGetterNet
        logger.info("Initializing %s (img_size=%d)", encoder_name, img_size)
        self.encoder = timm.create_model(encoder_name, pretrained=False, features_only=True, img_size=img_size)
        # We skip downloading/loading base weights because the user's checkpoint
        # (loaded immediately after this) contains all weights, including correctly
        # trained/sized PosEmbeds. This keeps the project offline-capable using only local assets.
        logger.info("Skipped downloading base weights (relying on custom checkpoint)")

        # Patch First Layer for 4 channels
        if in_channels != 3:
            self._patch_input_layer(in_channels)

        # Get feature info
        # Verified Hiera Base Plus channels: [112, 224, 448, 896]
        # We can try to fetch dynamically
        try:
            feature_channels = self.encoder.feature_info.channels()
        except (AttributeError, TypeError):
            feature_channels = [112, 224, 448, 896]
        logger.info("Feature channels: %s", feature_channels)

        # --- Decoders ---
        embedding_dim = 256

        # Alpha Decoder (Outputs 1 channel)
        self.alpha_decoder = DecoderHead(feature_channels, embedding_dim, output_dim=1)

        # Foreground Decoder (Outputs 3 channels)
        self.fg_decoder = DecoderHead(feature_channels, embedding_dim, output_dim=3)

        # --- Refiner ---
        # CNN Refiner
        # In Channels: 3 (RGB) + 4 (Coarse Pred) = 7
        self.use_refiner = use_refiner
        if self.use_refiner:
            self.refiner = CNNRefinerModule(in_channels=7, hidden_channels=64, out_channels=4)
        else:
            self.refiner = None
            logger.info("Refiner module DISABLED (backbone-only mode)")

    def _patch_input_layer(self, in_channels: int) -> None:
        """
        Modifies the first convolution layer to accept `in_channels`.
        Copies existing RGB weights and initializes extras to zero.
        """
        # Hiera: self.encoder.model.patch_embed.proj

        try:
            patch_embed = self.encoder.model.patch_embed.proj
        except AttributeError:
            # Fallback if timm changes structure or for other models
            patch_embed = self.encoder.patch_embed.proj
        weight = patch_embed.weight.data  # [Out, 3, K, K]
        bias = patch_embed.bias.data if patch_embed.bias is not None else None

        new_in_channels = in_channels
        out_channels, _, k, k = weight.shape

        # Create new conv
        new_conv = nn.Conv2d(
            new_in_channels,
            out_channels,
            kernel_size=k,
            stride=patch_embed.stride,
            padding=patch_embed.padding,
            bias=(bias is not None),
        )

        # Copy weights
        new_conv.weight.data[:, :3, :, :] = weight
        # Initialize new channels to 0 (Weight Patching)
        new_conv.weight.data[:, 3:, :, :] = 0.0

        if bias is not None:
            new_conv.bias.data = bias

        # Replace in module
        try:
            self.encoder.model.patch_embed.proj = new_conv
        except AttributeError:
            self.encoder.patch_embed.proj = new_conv

        logger.info("Patched input layer: 3 → %d channels (extra initialized to 0)", in_channels)

    # --- EZ-CorridorKey change: refiner_scale as a forward kwarg --------------
    # (upstream scales the whole refiner delta through a forward hook; EZ scales
    # only the alpha delta so FG always gets the full anti-macroblocking fix)
    def forward(self, x: torch.Tensor, refiner_scale: torch.Tensor | float | None = None) -> dict[str, torch.Tensor]:
        # x: [B, 4, H, W]
        input_size = x.shape[2:]

        # Encode
        features = self.encoder(x)  # Returns list of features

        # Decode Streams
        alpha_logits = self.alpha_decoder(features)  # [B, 1, H/4, W/4]
        fg_logits = self.fg_decoder(features)  # [B, 3, H/4, W/4]

        # Upsample to full resolution (Bilinear)
        # These are the "Coarse" LOGITS
        alpha_logits_up = F.interpolate(alpha_logits, size=input_size, mode="bilinear", align_corners=False)
        fg_logits_up = F.interpolate(fg_logits, size=input_size, mode="bilinear", align_corners=False)

        # Coarse Probs (for Loss and Refiner Input)
        alpha_coarse = torch.sigmoid(alpha_logits_up)
        fg_coarse = torch.sigmoid(fg_logits_up)

        # --- Refinement (CNN Hybrid) ---
        # Input to refiner: RGB Image (first 3 channels of x) + Coarse Predictions (Probs)
        # We give the refiner 'Probs' as input features because they are normalized [0,1]
        rgb = x[:, :3, :, :]

        # Feed the Refiner
        coarse_pred = torch.cat([alpha_coarse, fg_coarse], dim=1)  # [B, 4, H, W]

        # Refiner outputs DELTA LOGITS
        # The refiner predicts the correction in valid score space (-inf, inf)
        if self.use_refiner and self.refiner is not None:
            delta_logits = self.refiner(rgb, coarse_pred)
        else:
            # Zero Deltas
            delta_logits = torch.zeros_like(coarse_pred)

        delta_alpha = delta_logits[:, 0:1]
        delta_fg = delta_logits[:, 1:4]

        # Refiner scale only affects alpha edges — FG always gets full
        # correction to prevent Hiera attention-window block artifacts.
        if refiner_scale is not None:
            if torch.is_tensor(refiner_scale):
                refiner_scale = refiner_scale.to(device=delta_alpha.device, dtype=delta_alpha.dtype)
            delta_alpha = delta_alpha * refiner_scale

        # Residual Addition in Logit Space
        # This allows infinite correction capability and prevents saturation blocking
        alpha_final_logits = alpha_logits_up + delta_alpha
        fg_final_logits = fg_logits_up + delta_fg

        # Final Activation
        alpha_final = torch.sigmoid(alpha_final_logits)
        fg_final = torch.sigmoid(fg_final_logits)

        return {"alpha": alpha_final, "fg": fg_final}
    # --- end EZ-CorridorKey change --------------------------------------------
