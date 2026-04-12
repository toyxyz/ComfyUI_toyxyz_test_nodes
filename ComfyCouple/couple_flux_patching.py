"""
Flux-specific attention patching logic for regional prompting.
Based on Fluxtapoz attention masking and extended for TiledDiffusion.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from .couple_utils import (
    ARCH_CONFIGS,
    build_region_metadata,
    find_best_divisors,
    log_debug,
    log_warning,
    save_region_metadata,
)


class RegionalMask(torch.nn.Module):
    """
    Regional attention mask for Flux.

    This preserves the original Fluxtapoz masking rules and adds
    per-tile batch masks when TiledDiffusion injects tile metadata
    into transformer_options.
    """

    def __init__(
        self,
        region_conds: List[torch.Tensor],
        region_masks: List[torch.Tensor],
        start_percent: float,
        end_percent: float,
        apply_t5_background: bool = True,
    ) -> None:
        super().__init__()
        for i, (cond, mask) in enumerate(zip(region_conds, region_masks)):
            self.register_buffer(f"region_cond_{i}", cond)
            self.register_buffer(f"region_mask_{i}", mask)

        self.num_regions = len(region_masks)
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.apply_t5_background = apply_t5_background

        self._cached_mask = None
        self._cached_shape = None

    def __call__(self, q, transformer_options, txt_size):
        current_sigma = transformer_options["sigmas"][0]
        current_percent = 1.0 - (
            current_sigma.item() if isinstance(current_sigma, torch.Tensor) else float(current_sigma)
        )

        if not (self.start_percent <= current_percent < self.end_percent):
            return None

        batch_size, _, seq_len, _ = q.shape
        img_len = seq_len - txt_size
        tiled_context = self._get_tiled_context(transformer_options, batch_size)

        cache_key = self._build_cache_key(
            batch_size, seq_len, txt_size, img_len, q.device, q.dtype, tiled_context
        )
        if self._cached_mask is not None and self._cached_shape == cache_key:
            return self._cached_mask

        log_debug(
            f"[Flux Dynamic] Building mask for seq_len={seq_len}, txt_size={txt_size}, img_len={img_len}"
        )

        if tiled_context is None:
            h_attn, w_attn = self._infer_hw_from_img_len(img_len)
            log_debug(f"[Flux Dynamic] Inferred attention size: {h_attn}x{w_attn}")
            region_masks_spatial = self._build_full_spatial_masks(h_attn, w_attn, q.device, q.dtype)
            regional_mask = self._build_mask_from_spatial_masks(
                region_masks_spatial, img_len, txt_size, seq_len, q.device
            )
        else:
            regional_mask = self._build_tiled_mask(
                tiled_context, batch_size, img_len, txt_size, seq_len, q.device, q.dtype
            )

        self._cached_mask = regional_mask
        self._cached_shape = cache_key
        return regional_mask

    def _build_cache_key(
        self,
        batch_size: int,
        seq_len: int,
        txt_size: int,
        img_len: int,
        device: torch.device,
        dtype: torch.dtype,
        tiled_context: Optional[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        key = (batch_size, seq_len, txt_size, img_len, str(device), str(dtype))
        if tiled_context is None:
            return key

        bbox_key = tuple((bbox.x, bbox.y, bbox.w, bbox.h) for bbox in tiled_context["bboxes"])
        return key + (
            bbox_key,
            tuple(tiled_context["full_shape"]),
            tuple(tiled_context["shift"]),
            tiled_context["shift_condition"],
        )

    def _get_tiled_context(
        self, transformer_options: Dict[str, Any], batch_size: int
    ) -> Optional[Dict[str, Any]]:
        bboxes = transformer_options.get("tiled_diffusion_bboxes")
        full_shape = transformer_options.get("tiled_diffusion_full_shape")
        if not bboxes or full_shape is None:
            return None

        num_tiles = len(bboxes)
        if num_tiles <= 0:
            return None

        return {
            "bboxes": bboxes,
            "full_shape": full_shape,
            "shift": transformer_options.get("tiled_diffusion_shift", (0, 0)),
            "shift_condition": transformer_options.get("tiled_diffusion_shift_condition", True),
            "batch_per_tile": max(1, math.ceil(batch_size / num_tiles)),
        }

    def _normalize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 3:
            return mask[0]
        return mask

    def _apply_tiled_shift(
        self,
        mask: torch.Tensor,
        full_shape: Tuple[int, int],
        shift: Tuple[int, int],
        shift_condition: bool,
    ) -> torch.Tensor:
        if shift == (0, 0):
            return mask

        sh_h, sh_w = shift
        full_h, full_w = full_shape
        mask_h, mask_w = mask.shape[-2:]
        sh_h_mask = round(sh_h * mask_h / full_h)
        sh_w_mask = round(sh_w * mask_w / full_w)

        if sh_h_mask == 0 or sh_w_mask == 0:
            return mask.roll(shifts=(sh_h_mask, sh_w_mask), dims=(-2, -1))
        if shift_condition:
            return mask.roll(shifts=sh_h_mask, dims=-2)
        return mask.roll(shifts=sh_w_mask, dims=-1)

    def _infer_hw_from_img_len(self, img_len: int) -> Tuple[int, int]:
        first_mask = getattr(self, "region_mask_0")
        mask_h, mask_w = first_mask.shape[-2:]

        expected_h = mask_h // 16
        expected_w = mask_w // 16
        if expected_h * expected_w == img_len:
            return expected_h, expected_w

        aspect_ratio = mask_h / mask_w
        w = int(math.sqrt(img_len / aspect_ratio))
        h = img_len // w

        while h * w != img_len and w > 1:
            w -= 1
            h = img_len // w

        if h * w != img_len:
            h = w = int(math.sqrt(img_len))

        log_debug(f"[Flux Dynamic] Mask shape: {mask_h}x{mask_w}, inferred attention: {h}x{w}")
        return h, w

    def _infer_tiled_hw(self, img_len: int, bbox: Any) -> Tuple[int, int]:
        aspect_ratio = bbox.w / bbox.h if bbox.h > 0 else 1.0
        return find_best_divisors(img_len, aspect_ratio)

    def _build_full_spatial_masks(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> List[torch.Tensor]:
        region_masks_spatial = []
        for i in range(self.num_regions):
            mask = self._normalize_mask(getattr(self, f"region_mask_{i}"))
            mask_resized = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0).to(dtype=dtype, device=device),
                size=(h, w),
                mode="area",
            )
            region_masks_spatial.append(mask_resized[0, 0])
        return region_masks_spatial

    def _build_tiled_mask(
        self,
        tiled_context: Dict[str, Any],
        batch_size: int,
        img_len: int,
        txt_size: int,
        total_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        h_attn, w_attn = self._infer_tiled_hw(img_len, tiled_context["bboxes"][0])
        log_debug(
            f"[Flux Dynamic] Building tiled mask for {len(tiled_context['bboxes'])} tiles "
            f"at attention size {h_attn}x{w_attn}"
        )

        region_masks_spatial = self._build_tiled_spatial_masks(
            tiled_context, batch_size, h_attn, w_attn, device, dtype
        )
        return self._build_mask_from_spatial_masks(
            region_masks_spatial, img_len, txt_size, total_len, device
        )

    def _build_tiled_spatial_masks(
        self,
        tiled_context: Dict[str, Any],
        batch_size: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        bboxes = tiled_context["bboxes"]
        full_h, full_w = tiled_context["full_shape"]
        shift = tiled_context["shift"]
        shift_condition = tiled_context["shift_condition"]
        batch_per_tile = tiled_context["batch_per_tile"]

        region_masks_spatial = []
        for i in range(self.num_regions):
            working_mask = self._normalize_mask(getattr(self, f"region_mask_{i}"))
            working_mask = self._apply_tiled_shift(
                working_mask, tiled_context["full_shape"], shift, shift_condition
            )

            mask_h, mask_w = working_mask.shape[-2:]
            scale_h = mask_h / full_h
            scale_w = mask_w / full_w

            tile_masks = []
            for bbox in bboxes:
                y1 = int(bbox.y * scale_h)
                x1 = int(bbox.x * scale_w)
                y2 = min(int((bbox.y + bbox.h) * scale_h), mask_h)
                x2 = min(int((bbox.x + bbox.w) * scale_w), mask_w)

                cropped = working_mask[y1:y2, x1:x2]
                if cropped.numel() == 0:
                    cropped = working_mask.new_zeros((1, 1))

                resized = torch.nn.functional.interpolate(
                    cropped.unsqueeze(0).unsqueeze(0).to(dtype=dtype, device=device),
                    size=(h, w),
                    mode="area",
                )[0, 0]
                tile_masks.append(resized.unsqueeze(0).expand(batch_per_tile, -1, -1))

            region_mask = torch.cat(tile_masks, dim=0)[:batch_size]
            region_masks_spatial.append(region_mask)

        return region_masks_spatial

    def _build_mask_from_spatial_masks(
        self,
        region_masks_spatial: List[torch.Tensor],
        img_len: int,
        txt_size: int,
        total_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        region_conds = [getattr(self, f"region_cond_{i}") for i in range(self.num_regions)]

        batch_size = 1
        if region_masks_spatial and region_masks_spatial[0].dim() == 3:
            batch_size = region_masks_spatial[0].shape[0]

        t5_len = 256 if self.apply_t5_background else 0
        text_len = t5_len + sum(cond.shape[1] for cond in region_conds)

        if text_len != txt_size:
            log_warning(f"Text length mismatch: calculated={text_len}, actual={txt_size}")
            text_len = txt_size

        regional_mask = torch.zeros((batch_size, total_len, total_len), dtype=torch.bool, device=device)

        if self.apply_t5_background:
            regional_mask[:, 0:t5_len, 0:t5_len] = True

        self_attend_masks = torch.zeros((batch_size, img_len, img_len), dtype=torch.bool, device=device)
        union_masks = torch.zeros((batch_size, img_len, img_len), dtype=torch.bool, device=device)

        current_seq_pos = t5_len
        for region_cond, spatial_mask in zip(region_conds, region_masks_spatial):
            region_text_len = region_cond.shape[1]
            next_seq_pos = current_seq_pos + region_text_len

            if spatial_mask.dim() == 2:
                spatial_mask = spatial_mask.unsqueeze(0)

            flat_mask = spatial_mask.reshape(batch_size, img_len, 1)
            flat_mask_bool = flat_mask > 0.01

            regional_mask[:, current_seq_pos:next_seq_pos, current_seq_pos:next_seq_pos] = True
            regional_mask[:, current_seq_pos:next_seq_pos, text_len:] = flat_mask_bool.transpose(-1, -2)
            regional_mask[:, text_len:, current_seq_pos:next_seq_pos] = flat_mask_bool

            img_mask_full = flat_mask_bool.expand(-1, -1, img_len)
            img_mask_full_t = img_mask_full.transpose(-1, -2)
            self_attend_masks = torch.logical_or(
                self_attend_masks,
                torch.logical_and(img_mask_full, img_mask_full_t),
            )
            union_masks = torch.logical_or(
                union_masks,
                torch.logical_or(img_mask_full, img_mask_full_t),
            )

            current_seq_pos = next_seq_pos

        background_masks = torch.logical_not(union_masks)
        regional_mask[:, text_len:, text_len:] = torch.logical_or(background_masks, self_attend_masks)

        if self.apply_t5_background:
            uncovered_pixels = torch.stack(region_masks_spatial).sum(dim=0) < 0.01
            uncovered_pixels_flat = uncovered_pixels.reshape(batch_size, img_len)
            regional_mask[:, text_len:, 0:t5_len] = uncovered_pixels_flat.unsqueeze(-1).expand(-1, -1, t5_len)

        log_debug(
            f"[Flux Dynamic] Mask built: {regional_mask.sum().item()}/{regional_mask.numel()} "
            f"({regional_mask.sum().item() / regional_mask.numel() * 100:.1f}%)"
        )

        if batch_size == 1:
            return regional_mask[0]
        return regional_mask


class RegionalConditioning(torch.nn.Module):
    """Regional conditioning tensor for Flux."""

    def __init__(self, region_cond: torch.Tensor, start_percent: float, end_percent: float) -> None:
        super().__init__()
        self.register_buffer("region_cond", region_cond)
        self.start_percent = start_percent
        self.end_percent = end_percent

    def __call__(self, transformer_options, *args, **kwargs):
        current_sigma = transformer_options["sigmas"][0]
        current_percent = 1.0 - (
            current_sigma.item() if isinstance(current_sigma, torch.Tensor) else float(current_sigma)
        )

        if self.start_percent <= current_percent < self.end_percent:
            return self.region_cond
        return None


def find_regional_mask_module(transformer_options: Dict[str, Any]) -> Tuple[Optional[RegionalMask], Optional[str], Optional[Any]]:
    """Return the first registered Flux regional mask plus its block location."""
    patches_replace = transformer_options.get("patches_replace", {})

    for block_type in ["double", "single"]:
        block_patches = patches_replace.get(block_type, {})
        for block_id, patch_content in block_patches.items():
            if isinstance(patch_content, dict):
                for key, module in patch_content.items():
                    if key == "mask_fn" or (isinstance(key, tuple) and key[0] == "mask_fn"):
                        if isinstance(module, RegionalMask):
                            return module, block_type, block_id
            elif isinstance(patch_content, RegionalMask):
                return patch_content, block_type, block_id

    return None, None, None


class FluxAttentionPatcher:
    """
    Manages attention masking for Flux regional prompting.
    """

    def __init__(
        self,
        model: Any,
        regions: List[Dict],
        start_percent: float = 0.0,
        end_percent: float = 0.5,
        attn_override: Optional[Dict] = None,
        apply_t5_background: bool = True,
    ):
        self.model = model
        self.regions = regions
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.attn_override = attn_override if attn_override is not None else ARCH_CONFIGS["flux"]
        self.apply_t5_background = apply_t5_background

    def patch(self) -> Tuple[Any, Any]:
        """Apply Flux regional patching with dynamic calculation."""
        active_regions = [r for r in self.regions if r.get("strength", 1.0) > 0]

        if not active_regions:
            log_warning("No active regions for Flux patching")
            return self.model, self.regions[0]["positive"] if self.regions else None

        for idx, region in enumerate(active_regions):
            if region["mask"].dim() not in [2, 3]:
                raise ValueError(f"Region {idx} mask must be 2D or 3D, got {region['mask'].dim()}D")
            mask_sum = region["mask"].sum().item()
            if mask_sum < 0.001:
                log_warning(f"Region {idx} has empty mask (sum={mask_sum:.6f})")

        print(f"[ComfyCouple Flux] Processing {len(active_regions)} regions (dynamic calculation)")

        region_conds = []
        region_masks = []
        for region in active_regions:
            cond_tensor = region["positive"][0][0]
            region_conds.append(cond_tensor)
            region_masks.append(region["mask"])

        regional_conditioning = torch.cat(region_conds, dim=1)

        regional_mask_module = RegionalMask(
            region_conds,
            region_masks,
            self.start_percent,
            self.end_percent,
            self.apply_t5_background,
        )
        regional_cond_module = RegionalConditioning(
            regional_conditioning,
            self.start_percent,
            self.end_percent,
        )

        new_model = self.model.clone()
        new_model.set_model_patch(regional_cond_module, "regional_conditioning")

        for block_idx in self.attn_override["double"]:
            new_model.set_model_patch_replace(regional_mask_module, "double", "mask_fn", int(block_idx))

        for block_idx in self.attn_override["single"]:
            new_model.set_model_patch_replace(regional_mask_module, "single", "mask_fn", int(block_idx))

        print(
            f"[ComfyCouple Flux] Patched {len(self.attn_override['double'])} double + "
            f"{len(self.attn_override['single'])} single blocks"
        )
        print(f"[ComfyCouple Flux] Active timesteps: {self.start_percent * 100:.0f}% - {self.end_percent * 100:.0f}%")
        print("[ComfyCouple Flux] Mask computed dynamically at runtime")

        transformer_options = new_model.model_options.setdefault("transformer_options", {})
        save_region_metadata(
            transformer_options,
            build_region_metadata(
                "flux",
                region_masks=region_masks,
                region_strengths=[region.get("strength", 1.0) for region in active_regions],
                region_conditioning=region_conds,
                background_present=any(region.get("is_background", False) for region in active_regions),
                debug_handles={
                    "double_blocks": list(self.attn_override["double"]),
                    "single_blocks": list(self.attn_override["single"]),
                },
                start_percent=self.start_percent,
                end_percent=self.end_percent,
                apply_t5_background=self.apply_t5_background,
                attn_override={
                    "double": list(self.attn_override["double"]),
                    "single": list(self.attn_override["single"]),
                },
            ),
        )

        return new_model, active_regions[0]["positive"]
