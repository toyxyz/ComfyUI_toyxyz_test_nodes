"""
Anima-specific regional prompting patcher.

This implementation mirrors the sd-forge-couple Anima approach closely:
- patch only cross-attention forwards
- leave the surrounding block math untouched
- expand only the positive branch by region count
- merge attention outputs with normalized region masks
"""

import math
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import comfy.patcher_extension
import comfy.sampler_helpers
import comfy.samplers

from .couple_utils import build_region_metadata, log_warning, save_region_metadata


class AnimaAttentionPatcher:
    WRAPPER_KEY = "ComfyCouple_AnimaDiffusionModel"

    def __init__(self, model: Any, regions: List[Dict[str, Any]]) -> None:
        self.model = model
        self.regions = regions
        self._region_bundle_cache: Dict[Tuple[Any, ...], Optional[Dict[str, Any]]] = {}

    def patch(self) -> Any:
        patched_model = self.model.clone()
        transformer_options = patched_model.model_options.setdefault("transformer_options", {})
        save_region_metadata(
            transformer_options,
            build_region_metadata(
                "anima",
                region_masks=[r["mask"] for r in self.regions if isinstance(r.get("mask"), torch.Tensor)],
                region_strengths=[float(r.get("strength", 1.0)) for r in self.regions],
                region_conditioning=[
                    r["context_tensor"]
                    if isinstance(r.get("context_tensor"), torch.Tensor)
                    else r["conditioning"][0][0]
                    for r in self.regions
                    if isinstance(r.get("context_tensor"), torch.Tensor)
                    or (r.get("conditioning") and len(r["conditioning"]) > 0)
                ],
                background_present=any(bool(r.get("is_background")) for r in self.regions),
                wrapper="diffusion_model",
                region_conditioning_objects=[
                    r.get("conditioning")
                    for r in self.regions
                ],
                region_background_flags=[
                    bool(r.get("is_background"))
                    for r in self.regions
                ],
            ),
        )
        patched_model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            self.WRAPPER_KEY,
            self._build_diffusion_wrapper(patched_model),
        )
        return patched_model

    def _build_diffusion_wrapper(self, patched_model: Any):
        def anima_diffusion_wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
            transformer_options = kwargs.get("transformer_options", {})
            cond_or_uncond = transformer_options.get("cond_or_uncond", [])

            if not cond_or_uncond:
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)

            batch_chunks = len(cond_or_uncond)
            if x.shape[0] % batch_chunks != 0:
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)

            diffusion_model = executor.class_obj
            chunk_size = x.shape[0] // batch_chunks
            region_bundle = self._get_region_bundle(
                patched_model,
                chunk_size,
                x,
                context,
                int(diffusion_model.patch_spatial),
            )
            if not region_bundle:
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)

            original_forwards = []
            try:
                for block in diffusion_model.blocks:
                    attn = block.cross_attn
                    original_forwards.append((attn, attn.forward))
                    attn.forward = types.MethodType(
                        self._build_cross_attn_forward(attn.forward, region_bundle),
                        attn,
                    )
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)
            finally:
                for attn, original_forward in original_forwards:
                    attn.forward = original_forward

        return anima_diffusion_wrapper

    def _get_region_bundle(
        self,
        patched_model: Any,
        batch_size: int,
        x: torch.Tensor,
        context: torch.Tensor,
        patch_spatial: int,
    ) -> Optional[Dict[str, Any]]:
        cache_key = (
            batch_size,
            str(x.device),
            str(x.dtype),
            str(context.device),
            str(context.dtype),
            tuple(x.shape[1:]),
            patch_spatial,
        )
        cached = self._region_bundle_cache.get(cache_key)
        if cached is not None:
            return cached

        bundle = self._build_region_bundle(patched_model, batch_size, x, context, patch_spatial)
        self._region_bundle_cache[cache_key] = bundle
        return bundle

    def _build_region_bundle(
        self,
        patched_model: Any,
        batch_size: int,
        x: torch.Tensor,
        context: torch.Tensor,
        patch_spatial: int,
    ) -> Optional[Dict[str, Any]]:
        sample_x = x[:batch_size]
        region_contexts: List[torch.Tensor] = []
        weighted_masks: List[torch.Tensor] = []
        raw_masks: List[torch.Tensor] = []

        for region in self.regions:
            preencoded_context = region.get("context_tensor")
            if isinstance(preencoded_context, torch.Tensor):
                crossattn = preencoded_context
            else:
                conditioning = region.get("conditioning")
                if not conditioning:
                    continue

                converted = comfy.sampler_helpers.convert_cond(conditioning)
                encoded = comfy.samplers.encode_model_conds(
                    patched_model.model.extra_conds,
                    converted,
                    sample_x,
                    sample_x.device,
                    "positive",
                )
                if not encoded:
                    continue

                if len(encoded) > 1:
                    log_warning("Anima currently uses the first conditioning item per region")

                encoded_item = next(
                    (item for item in encoded if item.get("model_conds", {}).get("c_crossattn") is not None),
                    None,
                )
                if encoded_item is None:
                    continue

                crossattn_cond = encoded_item["model_conds"]["c_crossattn"]
                crossattn = crossattn_cond.process_cond(batch_size=batch_size, area=None).cond
            region_contexts.append(crossattn.to(device=context.device, dtype=context.dtype))

            mask = region["mask"]
            if mask.dim() == 3:
                mask = mask[0]
            mask = mask.detach().to(dtype=torch.float32).clamp_min(0.0)
            raw_masks.append(mask)
            weighted_masks.append(mask * float(max(region.get("strength", 1.0), 0.0)))

        if not region_contexts:
            return None

        normalized_masks = self._normalize_region_masks(weighted_masks, raw_masks)
        if normalized_masks is None:
            return None

        token_lengths = [ctx.shape[1] for ctx in region_contexts]
        target_tokens = self._lcm_for_list(token_lengths)

        expanded_contexts = []
        for ctx in region_contexts:
            repeats = max(target_tokens // ctx.shape[1], 1)
            expanded_contexts.append(ctx.repeat(1, repeats, 1))

        return {
            "contexts": expanded_contexts,
            "context_tokens": target_tokens,
            "masks": normalized_masks,
            "mask_cache": {},
            "num_regions": len(expanded_contexts),
            "patch_spatial": patch_spatial,
            "latent_height": int(x.shape[-2]),
            "latent_width": int(x.shape[-1]),
        }

    def _normalize_region_masks(
        self,
        weighted_masks: List[torch.Tensor],
        raw_masks: List[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not weighted_masks:
            return None

        weighted_stack = torch.stack(weighted_masks, dim=0)
        weighted_sum = weighted_stack.sum(dim=0, keepdim=True)
        if torch.all(weighted_sum > 1e-6):
            return weighted_stack / weighted_sum.clamp_min(1e-6)

        raw_stack = torch.stack(raw_masks, dim=0)
        raw_sum = raw_stack.sum(dim=0, keepdim=True)
        if torch.any(raw_sum <= 1e-6):
            log_warning("Anima masks must cover the full image; skipping Anima regional patch")
            return None

        log_warning("Anima region strengths left uncovered pixels; falling back to raw mask normalization")
        return raw_stack / raw_sum.clamp_min(1e-6)

    def _build_cross_attn_forward(self, original_forward, region_bundle: Dict[str, Any]):
        def patched_cross_attn_forward(self_attn, x, context=None, rope_emb=None, transformer_options=None):
            if transformer_options is None:
                transformer_options = {}

            cond_or_uncond = transformer_options.get("cond_or_uncond", [])
            if context is None or not cond_or_uncond:
                return original_forward(
                    x,
                    context=context,
                    rope_emb=rope_emb,
                    transformer_options=transformer_options,
                )

            num_chunks = len(cond_or_uncond)
            if x.shape[0] % num_chunks != 0:
                return original_forward(
                    x,
                    context=context,
                    rope_emb=rope_emb,
                    transformer_options=transformer_options,
                )

            batch_size = x.shape[0] // num_chunks
            x_chunks = x.chunk(num_chunks, dim=0)
            context_chunks = context.chunk(num_chunks, dim=0)

            target_tokens = self._lcm_for_list(
                [region_bundle["context_tokens"], int(context_chunks[0].shape[1])]
            )
            region_context_tensor = torch.cat(
                [self._expand_context_tokens(ctx, target_tokens) for ctx in region_bundle["contexts"]],
                dim=0,
            )
            new_x = []
            new_context = []

            for idx, cond_flag in enumerate(cond_or_uncond):
                if cond_flag == 1:
                    new_x.append(x_chunks[idx])
                    new_context.append(self._expand_context_tokens(context_chunks[idx], target_tokens))
                else:
                    new_x.append(x_chunks[idx].repeat(region_bundle["num_regions"], 1, 1))
                    new_context.append(region_context_tensor)

            x_in = torch.cat(new_x, dim=0)
            context_in = torch.cat(new_context, dim=0)
            out = original_forward(
                x_in,
                context=context_in,
                rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

            seq_len = out.shape[1]
            mask = self._get_sequence_mask(
                region_bundle,
                seq_len,
                out.device,
                out.dtype,
                batch_size,
                transformer_options=transformer_options,
            )

            outputs = []
            pos = 0
            for cond_flag in cond_or_uncond:
                if cond_flag == 1:
                    outputs.append(out[pos : pos + batch_size])
                    pos += batch_size
                else:
                    chunk = out[pos : pos + region_bundle["num_regions"] * batch_size]
                    chunk = chunk.view(region_bundle["num_regions"], batch_size, seq_len, -1)
                    outputs.append((chunk * mask).sum(dim=0))
                    pos += region_bundle["num_regions"] * batch_size

            return torch.cat(outputs, dim=0)

        return patched_cross_attn_forward

    def _get_sequence_mask(
        self,
        region_bundle: Dict[str, Any],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        transformer_options: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        tiled_context = self._get_tiled_context(transformer_options, batch_size)
        if tiled_context is not None:
            return self._get_tiled_sequence_mask(
                region_bundle,
                seq_len,
                device,
                dtype,
                batch_size,
                tiled_context,
            )

        cache_key = (seq_len, str(device), str(dtype), batch_size)
        cached = region_bundle["mask_cache"].get(cache_key)
        if cached is not None:
            return cached

        masks = region_bundle["masks"].to(device=device, dtype=dtype)
        num_regions = masks.shape[0]

        h_tokens = max(region_bundle["latent_height"] // region_bundle["patch_spatial"], 1)
        w_tokens = max(region_bundle["latent_width"] // region_bundle["patch_spatial"], 1)
        spatial_tokens = max(h_tokens * w_tokens, 1)
        temporal_tokens = max(int(math.ceil(seq_len / spatial_tokens)), 1)

        resized = F.interpolate(
            masks.unsqueeze(1),
            size=(h_tokens, w_tokens),
            mode="bilinear",
            align_corners=False,
        )
        flattened = resized.view(num_regions, 1, spatial_tokens).repeat(1, temporal_tokens, 1)
        flattened = flattened.view(num_regions, 1, -1, 1)[:, :, :seq_len, :]
        flattened = flattened.expand(num_regions, batch_size, seq_len, 1)

        region_bundle["mask_cache"][cache_key] = flattened
        return flattened

    def _get_tiled_context(
        self,
        transformer_options: Optional[Dict[str, Any]],
        batch_size: int,
    ) -> Optional[Dict[str, Any]]:
        if not transformer_options:
            return None

        bboxes = transformer_options.get("tiled_diffusion_bboxes")
        full_shape = transformer_options.get("tiled_diffusion_full_shape")
        if not bboxes or full_shape is None:
            return None

        num_tiles = len(bboxes)
        if num_tiles <= 0:
            return None

        return {
            "bboxes": bboxes,
            "full_shape": tuple(full_shape),
            "shift": transformer_options.get("tiled_diffusion_shift", (0, 0)),
            "shift_condition": transformer_options.get("tiled_diffusion_shift_condition", True),
            "batch_per_tile": max(1, math.ceil(batch_size / num_tiles)),
        }

    def _get_tiled_sequence_mask(
        self,
        region_bundle: Dict[str, Any],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        tiled_context: Dict[str, Any],
    ) -> torch.Tensor:
        bbox_key = tuple((bbox.x, bbox.y, bbox.w, bbox.h) for bbox in tiled_context["bboxes"])
        cache_key = (
            "tiled",
            seq_len,
            str(device),
            str(dtype),
            batch_size,
            bbox_key,
            tuple(tiled_context["full_shape"]),
            tuple(tiled_context["shift"]),
            bool(tiled_context["shift_condition"]),
            int(tiled_context["batch_per_tile"]),
        )
        cached = region_bundle["mask_cache"].get(cache_key)
        if cached is not None:
            return cached

        masks = region_bundle["masks"].to(device=device, dtype=dtype)
        num_regions = masks.shape[0]
        batch_per_tile = tiled_context["batch_per_tile"]
        full_h, full_w = tiled_context["full_shape"]
        patch_spatial = max(int(region_bundle["patch_spatial"]), 1)

        tiled_masks = []
        for i in range(num_regions):
            working_mask = masks[i]
            working_mask = self._apply_tiled_shift(
                working_mask,
                tiled_context["full_shape"],
                tiled_context["shift"],
                tiled_context["shift_condition"],
            )

            mask_h, mask_w = working_mask.shape[-2:]
            scale_h = mask_h / max(full_h, 1)
            scale_w = mask_w / max(full_w, 1)

            region_tile_masks = []
            for bbox in tiled_context["bboxes"]:
                y1 = int(bbox.y * scale_h)
                x1 = int(bbox.x * scale_w)
                y2 = min(int((bbox.y + bbox.h) * scale_h), mask_h)
                x2 = min(int((bbox.x + bbox.w) * scale_w), mask_w)

                cropped = working_mask[y1:y2, x1:x2]
                if cropped.numel() == 0:
                    cropped = working_mask.new_zeros((1, 1))

                h_tokens = max(int(bbox.h) // patch_spatial, 1)
                w_tokens = max(int(bbox.w) // patch_spatial, 1)
                spatial_tokens = max(h_tokens * w_tokens, 1)
                temporal_tokens = max(int(math.ceil(seq_len / spatial_tokens)), 1)

                resized = F.interpolate(
                    cropped.unsqueeze(0).unsqueeze(0),
                    size=(h_tokens, w_tokens),
                    mode="bilinear",
                    align_corners=False,
                )
                flattened = resized.view(1, spatial_tokens).repeat(temporal_tokens, 1).view(1, -1, 1)
                flattened = flattened[:, :seq_len, :].expand(batch_per_tile, -1, -1)
                region_tile_masks.append(flattened)

            region_mask = torch.cat(region_tile_masks, dim=0)[:batch_size]
            tiled_masks.append(region_mask)

        stacked = torch.stack(tiled_masks, dim=0)
        region_bundle["mask_cache"][cache_key] = stacked
        return stacked

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
        sh_h_mask = round(sh_h * mask_h / max(full_h, 1))
        sh_w_mask = round(sh_w * mask_w / max(full_w, 1))

        if sh_h_mask == 0 or sh_w_mask == 0:
            return mask.roll(shifts=(sh_h_mask, sh_w_mask), dims=(-2, -1))
        if shift_condition:
            return mask.roll(shifts=sh_h_mask, dims=-2)
        return mask.roll(shifts=sh_w_mask, dims=-1)

    def _expand_context_tokens(self, context: torch.Tensor, target_tokens: int) -> torch.Tensor:
        current_tokens = int(context.shape[1])
        if current_tokens == target_tokens:
            return context

        repeats = max(target_tokens // max(current_tokens, 1), 1)
        expanded = context.repeat(1, repeats, 1)
        return expanded[:, :target_tokens, :]

    def _lcm_for_list(self, numbers: List[int]) -> int:
        current = max(numbers[0], 1)
        for number in numbers[1:]:
            current = self._lcm(current, max(number, 1))
        return current

    def _lcm(self, a: int, b: int) -> int:
        return a * b // math.gcd(a, b)
