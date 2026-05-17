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

from .couple_utils import build_region_metadata, log_debug, log_warning, save_region_metadata


class AnimaAttentionPatcher:
    WRAPPER_KEY = "ComfyCouple_AnimaDiffusionModel"
    PREENCODED_CONTEXTS_KEY = "comfycouple_anima_preencoded_contexts"
    MODE_CROSS_ATTENTION = "cross_attention"
    MODE_FULL_BLOCK = "full_block"

    def __init__(
        self,
        model: Any,
        regions: List[Dict[str, Any]],
        region_mode: str = MODE_CROSS_ATTENTION,
    ) -> None:
        self.model = model
        self.regions = regions
        self.region_mode = (
            region_mode
            if region_mode in {self.MODE_CROSS_ATTENTION, self.MODE_FULL_BLOCK}
            else self.MODE_CROSS_ATTENTION
        )
        self._region_bundle_cache: Dict[Tuple[Any, ...], Optional[Dict[str, Any]]] = {}
        self._logged_tiled_layout = False
        self._logged_tiled_full_block_fallback = False

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
                anima_region_mode=self.region_mode,
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
            comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
            self.WRAPPER_KEY,
            self._build_sampler_wrapper(),
        )
        patched_model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            self.WRAPPER_KEY,
            self._build_diffusion_wrapper(patched_model),
        )
        return patched_model

    def _build_sampler_wrapper(self):
        converted_regions = []
        static_contexts: List[Optional[torch.Tensor]] = []

        for index, region in enumerate(self.regions):
            context_tensor = region.get("context_tensor")
            static_contexts.append(context_tensor if isinstance(context_tensor, torch.Tensor) else None)
            if isinstance(context_tensor, torch.Tensor):
                continue

            conditioning = region.get("conditioning")
            if not conditioning:
                continue

            converted = comfy.sampler_helpers.convert_cond(conditioning)
            if not converted:
                continue

            if len(converted) > 1:
                log_warning("Anima currently uses the first conditioning item per region")

            converted_regions.append((index, converted[0]))

        def anima_sampler_wrapper(executor, *args, **kwargs):
            if not converted_regions and not any(isinstance(ctx, torch.Tensor) for ctx in static_contexts):
                return executor(*args, **kwargs)

            try:
                extra_options = args[2] if len(args) > 2 and isinstance(args[2], dict) else kwargs.get("extra_options")
                if not isinstance(extra_options, dict):
                    return executor(*args, **kwargs)

                model_options = extra_options.get("model_options")
                if not isinstance(model_options, dict):
                    return executor(*args, **kwargs)

                transformer_options = dict(model_options.get("transformer_options", {}))
                preencoded_contexts: List[Optional[torch.Tensor]] = list(static_contexts)
                processed_count = 0

                if converted_regions:
                    guider = args[0] if len(args) > 0 else kwargs.get("guider")
                    noise = args[4] if len(args) > 4 else kwargs.get("noise")
                    latent_image = args[5] if len(args) > 5 else kwargs.get("latent_image")
                    denoise_mask = args[6] if len(args) > 6 else kwargs.get("denoise_mask")
                    seed = extra_options.get("seed")

                    if guider is None or noise is None or latent_image is None or seed is None:
                        raise ValueError("missing sampler inputs for Anima conditioning pre-encoding")

                    processed = comfy.samplers.process_conds(
                        guider.inner_model,
                        noise,
                        {"positive": [converted for _, converted in converted_regions]},
                        noise.device,
                        latent_image,
                        denoise_mask,
                        seed,
                        latent_shapes=[latent_image.shape],
                    ).get("positive", [])

                    if len(processed) != len(converted_regions):
                        log_warning(
                            "Anima pre-encoding returned a different number of conditioning items; "
                            "missing regions will use diffusion-time encoding"
                        )

                    for (region_index, _), processed_item in zip(converted_regions, processed):
                        model_conds = processed_item.get("model_conds", {})
                        crossattn_cond = model_conds.get("c_crossattn")
                        crossattn = getattr(crossattn_cond, "cond", None)
                        if isinstance(crossattn, torch.Tensor):
                            preencoded_contexts[region_index] = crossattn
                            processed_count += 1

                transformer_options[self.PREENCODED_CONTEXTS_KEY] = preencoded_contexts
                model_options["transformer_options"] = transformer_options
                ready_count = sum(1 for ctx in preencoded_contexts if isinstance(ctx, torch.Tensor))
                static_count = sum(1 for ctx in static_contexts if isinstance(ctx, torch.Tensor))
                if converted_regions:
                    print(
                        "[ComfyCouple] Anima sampler pre-encoded "
                        f"{processed_count}/{len(converted_regions)} region conditioning(s) "
                        f"({ready_count}/{len(self.regions)} contexts ready)"
                    )
                    log_debug(
                        "Anima sampler pre-encode: "
                        f"processed={processed_count}, static={static_count}, ready={ready_count}, "
                        f"regions={len(self.regions)}"
                    )
                elif static_count:
                    print(
                        "[ComfyCouple] Anima using "
                        f"{static_count}/{len(self.regions)} precomputed region context(s)"
                    )
                    log_debug(
                        "Anima static precomputed contexts: "
                        f"static={static_count}, ready={ready_count}, regions={len(self.regions)}"
                    )

            except Exception as e:
                try:
                    extra_options = args[2] if len(args) > 2 and isinstance(args[2], dict) else kwargs.get("extra_options")
                    model_options = extra_options.get("model_options") if isinstance(extra_options, dict) else None
                    transformer_options = (
                        model_options.get("transformer_options", {})
                        if isinstance(model_options, dict)
                        else {}
                    )
                    transformer_options.pop(self.PREENCODED_CONTEXTS_KEY, None)
                except Exception:
                    pass
                log_warning(f"Anima sampler pre-encoding failed; falling back to diffusion-time encoding: {e}")

            return executor(*args, **kwargs)

        return anima_sampler_wrapper

    def _build_diffusion_wrapper(self, patched_model: Any):
        def anima_diffusion_wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
            diffusion_model = executor.class_obj
            patch_spatial = int(max(getattr(diffusion_model, "patch_spatial", 1), 1))
            transformer_options = dict(kwargs.get("transformer_options", {}) or {})
            transformer_options["activations_shape"] = self._get_activations_shape(x, patch_spatial)
            kwargs["transformer_options"] = transformer_options

            cond_or_uncond = transformer_options.get("cond_or_uncond", [])
            if not cond_or_uncond:
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)

            batch_chunks = len(cond_or_uncond)
            if x.shape[0] % batch_chunks != 0:
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)

            chunk_size = x.shape[0] // batch_chunks
            region_bundle = self._get_region_bundle(
                patched_model,
                chunk_size,
                x,
                context,
                patch_spatial,
                transformer_options,
            )
            if not region_bundle:
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)

            original_forwards = []
            try:
                blocks = self._get_anima_blocks(diffusion_model)
                tiled_bboxes = transformer_options.get("tiled_diffusion_bboxes")
                use_full_block = self.region_mode == self.MODE_FULL_BLOCK and blocks and not tiled_bboxes
                if (
                    self.region_mode == self.MODE_FULL_BLOCK
                    and blocks
                    and tiled_bboxes
                    and not self._logged_tiled_full_block_fallback
                ):
                    print(
                        "[ComfyCouple] Anima full_block mode detected with Tiled Diffusion; "
                        "using cross_attention mode for tiled batch alignment"
                    )
                    self._logged_tiled_full_block_fallback = True

                if use_full_block:
                    for block_index, block in enumerate(blocks):
                        original_forwards.append((block, block.forward))
                        block.forward = types.MethodType(
                            self._build_block_forward(
                                block.forward,
                                region_bundle,
                                base_batch_size=chunk_size,
                                is_last_block=block_index == len(blocks) - 1,
                            ),
                            block,
                        )
                else:
                    attn_modules = self._get_cross_attention_modules(diffusion_model)
                    if not attn_modules:
                        return executor(x, timesteps, context, fps, padding_mask, **kwargs)
                    for attn in attn_modules:
                        original_forwards.append((attn, attn.forward))
                        attn.forward = types.MethodType(
                            self._build_cross_attn_forward(attn.forward, region_bundle),
                            attn,
                        )
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)
            finally:
                for module, original_forward in original_forwards:
                    module.forward = original_forward

        return anima_diffusion_wrapper

    def _get_activations_shape(self, x: torch.Tensor, patch_spatial: int) -> List[int]:
        activations_shape = list(x.shape)
        if len(activations_shape) >= 4:
            activations_shape[-2] = max(activations_shape[-2] // patch_spatial, 1)
            activations_shape[-1] = max(activations_shape[-1] // patch_spatial, 1)
        return activations_shape

    def _get_anima_blocks(self, diffusion_model: Any) -> List[Any]:
        blocks = getattr(diffusion_model, "blocks", None)
        return list(blocks) if blocks is not None else []

    def _get_cross_attention_modules(self, diffusion_model: Any) -> List[Any]:
        modules = []
        for block in self._get_anima_blocks(diffusion_model):
            attn = getattr(block, "cross_attn", None)
            if attn is not None and callable(getattr(attn, "forward", None)):
                modules.append(attn)

        if modules:
            return modules

        named_modules = getattr(diffusion_model, "named_modules", None)
        if not callable(named_modules):
            return []

        for name, module in named_modules():
            if "cross_attn" in name and callable(getattr(module, "forward", None)):
                modules.append(module)

        return modules

    def _get_region_bundle(
        self,
        patched_model: Any,
        batch_size: int,
        x: torch.Tensor,
        context: torch.Tensor,
        patch_spatial: int,
        transformer_options: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        cache_key = (
            batch_size,
            str(x.device),
            str(x.dtype),
            str(context.device),
            str(context.dtype),
            tuple(x.shape[1:]),
            patch_spatial,
            self._preencoded_contexts_signature(transformer_options),
        )
        cached = self._region_bundle_cache.get(cache_key)
        if cached is not None:
            return cached

        bundle = self._build_region_bundle(
            patched_model,
            batch_size,
            x,
            context,
            patch_spatial,
            transformer_options,
        )
        self._region_bundle_cache[cache_key] = bundle
        return bundle

    def _preencoded_contexts_signature(
        self,
        transformer_options: Optional[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        if not transformer_options:
            return ()

        contexts = transformer_options.get(self.PREENCODED_CONTEXTS_KEY)
        if not isinstance(contexts, list):
            return ()

        return tuple(
            (
                id(ctx),
                tuple(ctx.shape),
                str(ctx.device),
                str(ctx.dtype),
            )
            if isinstance(ctx, torch.Tensor)
            else None
            for ctx in contexts
        )

    def _build_region_bundle(
        self,
        patched_model: Any,
        batch_size: int,
        x: torch.Tensor,
        context: torch.Tensor,
        patch_spatial: int,
        transformer_options: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        sample_x = x[:batch_size]
        region_contexts: List[torch.Tensor] = []
        weighted_masks: List[torch.Tensor] = []
        raw_masks: List[torch.Tensor] = []
        preencoded_contexts = (
            transformer_options.get(self.PREENCODED_CONTEXTS_KEY)
            if isinstance(transformer_options, dict)
            else None
        )
        preencoded_used = 0

        for region_index, region in enumerate(self.regions):
            preencoded_context = region.get("context_tensor")
            if (
                not isinstance(preencoded_context, torch.Tensor)
                and isinstance(preencoded_contexts, list)
                and region_index < len(preencoded_contexts)
                and isinstance(preencoded_contexts[region_index], torch.Tensor)
            ):
                preencoded_context = preencoded_contexts[region_index]

            if isinstance(preencoded_context, torch.Tensor):
                crossattn = preencoded_context
                preencoded_used += 1
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
            crossattn = self._match_context_batch(crossattn, batch_size)
            region_contexts.append(crossattn.to(device=context.device, dtype=context.dtype))

            mask = region["mask"]
            if mask.dim() == 3:
                mask = mask[0]
            mask = mask.detach().to(dtype=torch.float32).clamp_min(0.0)
            raw_masks.append(mask)
            weighted_masks.append(mask * float(max(region.get("strength", 1.0), 0.0)))

        if not region_contexts:
            return None

        if preencoded_used > 0:
            log_debug(
                f"Anima region bundle using {preencoded_used}/{len(region_contexts)} pre-encoded context(s)"
            )

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
            "activations_shape": (
                list(transformer_options.get("activations_shape"))
                if isinstance(transformer_options, dict)
                and isinstance(transformer_options.get("activations_shape"), (list, tuple))
                else None
            ),
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

    def _match_context_batch(self, context: torch.Tensor, batch_size: int) -> torch.Tensor:
        if context.shape[0] == batch_size:
            return context

        if context.shape[0] == 1:
            return context.repeat(batch_size, 1, 1)

        if context.shape[0] > batch_size:
            return context[:batch_size]

        repeats = int(math.ceil(batch_size / max(int(context.shape[0]), 1)))
        return context.repeat(repeats, 1, 1)[:batch_size]

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
            if context.shape[0] != x.shape[0] or context.shape[0] % num_chunks != 0:
                return original_forward(
                    x,
                    context=context,
                    rope_emb=rope_emb,
                    transformer_options=transformer_options,
                )

            tiled_layout = self._get_tiled_batch_layout(transformer_options, x.shape[0], num_chunks)
            if tiled_layout is not None:
                if not self._logged_tiled_layout:
                    print(
                        "[ComfyCouple] Anima Tiled Diffusion batch layout detected: "
                        f"{tiled_layout['num_tiles']} tile(s) x {num_chunks} condition chunk(s), "
                        f"batch_per_tile={tiled_layout['batch_per_tile']}"
                    )
                    self._logged_tiled_layout = True
                x_chunks = self._split_tiled_cond_chunks(x, num_chunks, tiled_layout)
                context_chunks = self._split_tiled_cond_chunks(context, num_chunks, tiled_layout)
            else:
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
            expanded_cond_or_uncond = []

            for idx, cond_flag in enumerate(cond_or_uncond):
                if cond_flag == 1:
                    new_x.append(x_chunks[idx])
                    new_context.append(self._expand_context_tokens(context_chunks[idx], target_tokens))
                    expanded_cond_or_uncond.append(cond_flag)
                else:
                    new_x.append(x_chunks[idx].repeat(region_bundle["num_regions"], 1, 1))
                    new_context.append(region_context_tensor)
                    expanded_cond_or_uncond.extend([cond_flag] * region_bundle["num_regions"])

            x_in = torch.cat(new_x, dim=0)
            context_in = torch.cat(new_context, dim=0)
            expanded_transformer_options = dict(transformer_options)
            expanded_transformer_options["cond_or_uncond"] = expanded_cond_or_uncond
            out = original_forward(
                x_in,
                context=context_in,
                rope_emb=rope_emb,
                transformer_options=expanded_transformer_options,
            )

            outputs = []
            pos = 0
            for cond_flag in cond_or_uncond:
                if cond_flag == 1:
                    outputs.append(out[pos : pos + batch_size])
                    pos += batch_size
                else:
                    chunk = out[pos : pos + region_bundle["num_regions"] * batch_size]
                    outputs.append(
                        self._merge_region_outputs(
                            chunk,
                            region_bundle,
                            batch_size,
                            transformer_options,
                        )
                    )
                    pos += region_bundle["num_regions"] * batch_size

            if tiled_layout is not None:
                return self._combine_tiled_cond_chunks(outputs, tiled_layout)
            return torch.cat(outputs, dim=0)

        return patched_cross_attn_forward

    def _build_block_forward(
        self,
        original_forward,
        region_bundle: Dict[str, Any],
        base_batch_size: int,
        is_last_block: bool,
    ):
        def patched_block_forward(
            self_block,
            x_B_T_H_W_D,
            emb_B_T_D,
            crossattn_emb,
            *args,
            **kwargs,
        ):
            transformer_options = kwargs.get("transformer_options", {})
            if transformer_options is None:
                transformer_options = {}
                kwargs["transformer_options"] = transformer_options

            cond_or_uncond = transformer_options.get("cond_or_uncond", [])
            if not cond_or_uncond:
                return original_forward(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, *args, **kwargs)

            num_chunks = len(cond_or_uncond)
            original_total = base_batch_size * num_chunks
            expanded_sizes = [
                base_batch_size * (region_bundle["num_regions"] if cond_flag != 1 else 1)
                for cond_flag in cond_or_uncond
            ]
            expanded_total = sum(expanded_sizes)
            if x_B_T_H_W_D.shape[0] not in {original_total, expanded_total}:
                return original_forward(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, *args, **kwargs)

            if emb_B_T_D.shape[0] != original_total or crossattn_emb.shape[0] != original_total:
                return original_forward(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, *args, **kwargs)

            x_is_expanded = x_B_T_H_W_D.shape[0] == expanded_total
            x_chunks = (
                self._split_expanded_chunks(x_B_T_H_W_D, expanded_sizes)
                if x_is_expanded
                else list(x_B_T_H_W_D.chunk(num_chunks, dim=0))
            )
            emb_chunks = emb_B_T_D.chunk(num_chunks, dim=0)
            context_chunks = crossattn_emb.chunk(num_chunks, dim=0)

            target_tokens = self._lcm_for_list(
                [region_bundle["context_tokens"], int(context_chunks[0].shape[1])]
            )
            region_context_tensor = torch.cat(
                [self._expand_context_tokens(ctx, target_tokens) for ctx in region_bundle["contexts"]],
                dim=0,
            )

            expanded_x = []
            expanded_emb = []
            expanded_context = []
            expanded_cond_or_uncond = []
            for idx, cond_flag in enumerate(cond_or_uncond):
                if cond_flag == 1:
                    expanded_x.append(x_chunks[idx])
                    expanded_emb.append(emb_chunks[idx])
                    expanded_context.append(self._expand_context_tokens(context_chunks[idx], target_tokens))
                    expanded_cond_or_uncond.append(cond_flag)
                else:
                    if x_is_expanded:
                        expanded_x.append(x_chunks[idx])
                    else:
                        expanded_x.append(
                            x_chunks[idx].repeat(
                                region_bundle["num_regions"],
                                *([1] * (x_chunks[idx].dim() - 1)),
                            )
                        )
                    expanded_emb.append(
                        emb_chunks[idx].repeat(
                            region_bundle["num_regions"],
                            *([1] * (emb_chunks[idx].dim() - 1)),
                        )
                    )
                    expanded_context.append(region_context_tensor)
                    expanded_cond_or_uncond.extend([cond_flag] * region_bundle["num_regions"])

            patched_kwargs = dict(kwargs)
            expanded_transformer_options = dict(transformer_options)
            expanded_transformer_options["cond_or_uncond"] = expanded_cond_or_uncond
            patched_kwargs["transformer_options"] = expanded_transformer_options
            self._expand_block_kwarg_tensor(
                patched_kwargs,
                "adaln_lora_B_T_3D",
                cond_or_uncond,
                base_batch_size,
                region_bundle["num_regions"],
            )
            self._expand_block_kwarg_tensor(
                patched_kwargs,
                "extra_per_block_pos_emb",
                cond_or_uncond,
                base_batch_size,
                region_bundle["num_regions"],
            )

            out = original_forward(
                torch.cat(expanded_x, dim=0),
                torch.cat(expanded_emb, dim=0),
                torch.cat(expanded_context, dim=0),
                *args,
                **patched_kwargs,
            )

            if not is_last_block:
                return out

            outputs = []
            pos = 0
            for cond_flag, expanded_size in zip(cond_or_uncond, expanded_sizes):
                if cond_flag == 1:
                    outputs.append(out[pos : pos + base_batch_size])
                    pos += base_batch_size
                else:
                    chunk = out[pos : pos + expanded_size]
                    outputs.append(
                        self._merge_region_outputs(
                            chunk,
                            region_bundle,
                            base_batch_size,
                            transformer_options,
                        )
                    )
                    pos += expanded_size

            return torch.cat(outputs, dim=0)

        return patched_block_forward

    def _split_expanded_chunks(self, tensor: torch.Tensor, sizes: List[int]) -> List[torch.Tensor]:
        chunks = []
        pos = 0
        for size in sizes:
            chunks.append(tensor[pos : pos + size])
            pos += size
        return chunks

    def _get_tiled_batch_layout(
        self,
        transformer_options: Optional[Dict[str, Any]],
        total_batch: int,
        num_chunks: int,
    ) -> Optional[Dict[str, int]]:
        if not transformer_options or num_chunks <= 1:
            return None

        bboxes = transformer_options.get("tiled_diffusion_bboxes")
        if not bboxes:
            return None

        num_tiles = len(bboxes)
        if num_tiles <= 1:
            return None

        divisor = num_tiles * num_chunks
        if divisor <= 0 or total_batch % divisor != 0:
            return None

        return {
            "num_tiles": int(num_tiles),
            "batch_per_tile": int(total_batch // divisor),
        }

    def _split_tiled_cond_chunks(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        tiled_layout: Dict[str, int],
    ) -> List[torch.Tensor]:
        num_tiles = tiled_layout["num_tiles"]
        batch_per_tile = tiled_layout["batch_per_tile"]
        expected_batch = num_tiles * num_chunks * batch_per_tile
        if tensor.shape[0] != expected_batch:
            return list(tensor.chunk(num_chunks, dim=0))

        trailing_shape = tuple(tensor.shape[1:])
        tile_major = tensor.reshape(num_tiles, num_chunks, batch_per_tile, *trailing_shape)
        permute_order = (1, 0, 2, *range(3, tile_major.dim()))
        chunk_major = tile_major.permute(permute_order).reshape(
            num_chunks,
            num_tiles * batch_per_tile,
            *trailing_shape,
        )
        return list(chunk_major.unbind(dim=0))

    def _combine_tiled_cond_chunks(
        self,
        chunks: List[torch.Tensor],
        tiled_layout: Dict[str, int],
    ) -> torch.Tensor:
        if not chunks:
            raise ValueError("cannot combine empty tiled chunks")

        num_tiles = tiled_layout["num_tiles"]
        batch_per_tile = tiled_layout["batch_per_tile"]
        trailing_shape = tuple(chunks[0].shape[1:])

        reshaped = [
            chunk.reshape(num_tiles, batch_per_tile, *trailing_shape)
            for chunk in chunks
        ]
        tile_major = torch.stack(reshaped, dim=1)
        return tile_major.reshape(num_tiles * len(chunks) * batch_per_tile, *trailing_shape)

    def _expand_block_kwarg_tensor(
        self,
        kwargs: Dict[str, Any],
        key: str,
        cond_or_uncond: List[int],
        batch_size: int,
        num_regions: int,
    ) -> None:
        tensor = kwargs.get(key)
        if not isinstance(tensor, torch.Tensor):
            return

        total_batch = batch_size * len(cond_or_uncond)
        if tensor.shape[0] != total_batch:
            return

        expanded = []
        for chunk, cond_flag in zip(tensor.chunk(len(cond_or_uncond), dim=0), cond_or_uncond):
            if cond_flag == 1:
                expanded.append(chunk)
            else:
                expanded.append(chunk.repeat(num_regions, *([1] * (chunk.dim() - 1))))
        kwargs[key] = torch.cat(expanded, dim=0)

    def _merge_region_outputs(
        self,
        region_output: torch.Tensor,
        region_bundle: Dict[str, Any],
        batch_size: int,
        transformer_options: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        num_regions = region_bundle["num_regions"]
        feature_dim = int(region_output.shape[-1])
        leading_shape = tuple(region_output.shape[1:-1])
        seq_len = math.prod(leading_shape) if leading_shape else 1
        mask = self._get_sequence_mask(
            region_bundle,
            seq_len,
            region_output.device,
            region_output.dtype,
            batch_size,
            transformer_options=transformer_options,
        )
        chunk = region_output.reshape(num_regions, batch_size, seq_len, feature_dim)
        merged = (chunk * mask).sum(dim=0)
        return merged.reshape(batch_size, *leading_shape, feature_dim)

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

        activations_shape = region_bundle.get("activations_shape")
        if isinstance(activations_shape, list) and len(activations_shape) >= 4:
            h_tokens = max(int(activations_shape[-2]), 1)
            w_tokens = max(int(activations_shape[-1]), 1)
        else:
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
            full_mask = F.interpolate(
                working_mask.unsqueeze(0).unsqueeze(0),
                size=(full_h, full_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

            working_mask = self._apply_tiled_shift(
                full_mask,
                tiled_context["full_shape"],
                tiled_context["shift"],
                tiled_context["shift_condition"],
            )

            region_tile_masks = []
            for bbox in tiled_context["bboxes"]:
                y1 = max(int(bbox.y), 0)
                x1 = max(int(bbox.x), 0)
                y2 = min(int(bbox.y + bbox.h), full_h)
                x2 = min(int(bbox.x + bbox.w), full_w)

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

        if sh_h == 0 or sh_w == 0:
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
