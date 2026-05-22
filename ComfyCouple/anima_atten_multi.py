"""
Anima tag attention mixing node.

The node patches Anima cross-attention during sampling and mixes independently
encoded tag streams into the normal base prompt stream.
"""

import math
import types
from typing import Any, Dict, List, Optional, Tuple

import torch

import comfy.patcher_extension
import comfy.sampler_helpers
import comfy.samplers

from nodes import CLIPTextEncode

from .couple_utils import log_debug, log_warning


class AnimatagAttentionMixer:
    WRAPPER_KEY = "ComfyCouple_AnimaAttenMulti"
    PREENCODED_CONTEXTS_KEY = "comfycouple_anima_tag_contexts"

    def __init__(
        self,
        model: Any,
        tags: List[Dict[str, Any]],
        mix_strength: float,
        apply_to_uncond: bool,
        debug: bool = False,
    ) -> None:
        self.model = model
        self.tags = tags
        self.mix_strength = max(0.0, min(float(mix_strength), 1.0))
        self.apply_to_uncond = bool(apply_to_uncond)
        self.debug = bool(debug)
        self._logged_tiled_layout = False
        self._mixed_call_count = 0
        self._mixed_chunk_count = 0
        self._patched_module_count = 0
        self._active_tag_context_count = 0

    def patch(self) -> Any:
        if self.mix_strength <= 0.0 or not self.tags:
            return self.model

        patched_model = self.model.clone()
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

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[Anima Atten multi] {message}")
            log_debug(message)

    def _reset_debug_counters(self) -> None:
        self._mixed_call_count = 0
        self._mixed_chunk_count = 0
        self._patched_module_count = 0
        self._active_tag_context_count = 0

    def _build_sampler_wrapper(self):
        converted_tags = []
        for index, tag in enumerate(self.tags):
            conditioning = tag.get("conditioning")
            if not conditioning:
                continue

            converted = comfy.sampler_helpers.convert_cond(conditioning)
            if not converted:
                continue

            if len(converted) > 1:
                log_warning("Anima Atten multi uses the first conditioning item per tag")

            converted_tags.append((index, converted[0]))

        def tag_sampler_wrapper(executor, *args, **kwargs):
            if not converted_tags:
                return executor(*args, **kwargs)

            self._reset_debug_counters()
            try:
                extra_options = args[2] if len(args) > 2 and isinstance(args[2], dict) else kwargs.get("extra_options")
                if not isinstance(extra_options, dict):
                    return executor(*args, **kwargs)

                model_options = extra_options.get("model_options")
                if not isinstance(model_options, dict):
                    return executor(*args, **kwargs)

                guider = args[0] if len(args) > 0 else kwargs.get("guider")
                noise = args[4] if len(args) > 4 else kwargs.get("noise")
                latent_image = args[5] if len(args) > 5 else kwargs.get("latent_image")
                denoise_mask = args[6] if len(args) > 6 else kwargs.get("denoise_mask")
                seed = extra_options.get("seed")

                if guider is None or noise is None or latent_image is None or seed is None:
                    raise ValueError("missing sampler inputs for Anima tag pre-encoding")

                processed = comfy.samplers.process_conds(
                    guider.inner_model,
                    noise,
                    {"positive": [converted for _, converted in converted_tags]},
                    noise.device,
                    latent_image,
                    denoise_mask,
                    seed,
                    latent_shapes=[latent_image.shape],
                ).get("positive", [])

                preencoded_contexts: List[Optional[torch.Tensor]] = [None] * len(self.tags)
                if len(processed) != len(converted_tags):
                    log_warning(
                        "Anima Atten multi pre-encoding returned a different number of conditioning items; "
                        "missing tags will use diffusion-time encoding"
                    )

                processed_count = 0
                for (tag_index, _), processed_item in zip(converted_tags, processed):
                    model_conds = processed_item.get("model_conds", {})
                    crossattn_cond = model_conds.get("c_crossattn")
                    crossattn = getattr(crossattn_cond, "cond", None)
                    if isinstance(crossattn, torch.Tensor):
                        preencoded_contexts[tag_index] = crossattn
                        processed_count += 1

                transformer_options = dict(model_options.get("transformer_options", {}))
                transformer_options[self.PREENCODED_CONTEXTS_KEY] = preencoded_contexts
                model_options["transformer_options"] = transformer_options
                self._debug(
                    f"pre-encoded {processed_count}/{len(converted_tags)} tag conditioning(s)"
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
                log_warning(f"Anima Atten multi tag pre-encoding failed; falling back to diffusion-time encoding: {e}")

            try:
                return executor(*args, **kwargs)
            finally:
                self._debug(
                    f"patched {self._patched_module_count} cross-attention module(s); "
                    f"active tag context(s): {self._active_tag_context_count}; "
                    f"attention mix applied {self._mixed_call_count} time(s) "
                    f"across {self._mixed_chunk_count} condition chunk(s)"
                )

        return tag_sampler_wrapper

    def _build_diffusion_wrapper(self, patched_model: Any):
        def tag_diffusion_wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
            transformer_options = dict(kwargs.get("transformer_options", {}) or {})
            kwargs["transformer_options"] = transformer_options

            cond_or_uncond = transformer_options.get("cond_or_uncond", [])
            if not cond_or_uncond or not isinstance(context, torch.Tensor):
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)

            batch_chunks = len(cond_or_uncond)
            if x.shape[0] % batch_chunks != 0:
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)

            chunk_size = x.shape[0] // batch_chunks
            tag_contexts = self._get_tag_contexts(
                patched_model,
                chunk_size,
                x,
                context,
                transformer_options,
            )
            if not tag_contexts:
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)

            diffusion_model = executor.class_obj
            attn_modules = self._get_cross_attention_modules(diffusion_model)
            if not attn_modules:
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)

            original_forwards = []
            try:
                self._patched_module_count = max(self._patched_module_count, len(attn_modules))
                self._active_tag_context_count = max(self._active_tag_context_count, len(tag_contexts))
                for attn in attn_modules:
                    original_forwards.append((attn, attn.forward))
                    attn.forward = types.MethodType(
                        self._build_cross_attn_forward(attn.forward, tag_contexts),
                        attn,
                    )
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)
            finally:
                for module, original_forward in original_forwards:
                    module.forward = original_forward

        return tag_diffusion_wrapper

    def _get_cross_attention_modules(self, diffusion_model: Any) -> List[Any]:
        blocks = getattr(diffusion_model, "blocks", None)
        modules = []

        if blocks is not None:
            for block in list(blocks):
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

    def _get_tag_contexts(
        self,
        patched_model: Any,
        batch_size: int,
        x: torch.Tensor,
        context: torch.Tensor,
        transformer_options: Dict[str, Any],
    ) -> List[Tuple[torch.Tensor, float, int]]:
        preencoded_contexts = transformer_options.get(self.PREENCODED_CONTEXTS_KEY)
        contexts: List[Tuple[torch.Tensor, float, int]] = []
        sample_x = x[:batch_size]

        for tag_index, tag in enumerate(self.tags):
            weight = float(tag.get("weight", 1.0))
            if weight <= 0.0:
                continue
            sign = int(tag.get("sign", 1))
            if sign not in {-1, 1}:
                sign = 1

            crossattn = None
            if (
                isinstance(preencoded_contexts, list)
                and tag_index < len(preencoded_contexts)
                and isinstance(preencoded_contexts[tag_index], torch.Tensor)
            ):
                crossattn = preencoded_contexts[tag_index]

            if crossattn is None:
                conditioning = tag.get("conditioning")
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

                encoded_item = next(
                    (item for item in encoded if item.get("model_conds", {}).get("c_crossattn") is not None),
                    None,
                )
                if encoded_item is None:
                    continue

                crossattn_cond = encoded_item["model_conds"]["c_crossattn"]
                crossattn = crossattn_cond.process_cond(batch_size=batch_size, area=None).cond

            crossattn = self._match_context_batch(crossattn, batch_size)
            contexts.append((crossattn.to(device=context.device, dtype=context.dtype), weight, sign))

        return contexts

    def _build_cross_attn_forward(
        self,
        original_forward,
        tag_contexts: List[Tuple[torch.Tensor, float, int]],
    ):
        def patched_cross_attn_forward(self_attn, x, context=None, rope_emb=None, transformer_options=None):
            if transformer_options is None:
                transformer_options = {}

            base_out = original_forward(
                x,
                context=context,
                rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

            cond_or_uncond = transformer_options.get("cond_or_uncond", [])
            if context is None or not cond_or_uncond or not tag_contexts:
                return base_out

            num_chunks = len(cond_or_uncond)
            if x.shape[0] % num_chunks != 0 or context.shape[0] != x.shape[0]:
                return base_out

            tiled_layout = self._get_tiled_batch_layout(transformer_options, x.shape[0], num_chunks)
            if tiled_layout is not None:
                if not self._logged_tiled_layout:
                    print(
                        "[Anima Atten multi] Tiled Diffusion batch layout detected: "
                        f"{tiled_layout['num_tiles']} tile(s) x {num_chunks} condition chunk(s), "
                        f"batch_per_tile={tiled_layout['batch_per_tile']}"
                    )
                    self._logged_tiled_layout = True
                x_chunks = self._split_tiled_cond_chunks(x, num_chunks, tiled_layout)
                base_chunks = self._split_tiled_cond_chunks(base_out, num_chunks, tiled_layout)
            else:
                x_chunks = x.chunk(num_chunks, dim=0)
                base_chunks = base_out.chunk(num_chunks, dim=0)

            outputs = []
            mixed_this_call = 0
            for idx, cond_flag in enumerate(cond_or_uncond):
                base_chunk = base_chunks[idx]
                should_apply = self.apply_to_uncond or cond_flag != 1
                if not should_apply:
                    outputs.append(base_chunk)
                    continue

                x_chunk = x_chunks[idx]
                positive_accum = None
                positive_weight = 0.0
                negative_accum = None
                negative_weight = 0.0
                chunk_transformer_options = dict(transformer_options)
                chunk_transformer_options["cond_or_uncond"] = [cond_flag]

                for tag_context, weight, sign in tag_contexts:
                    tag_context = self._match_context_batch(tag_context, x_chunk.shape[0])
                    tag_out = original_forward(
                        x_chunk,
                        context=tag_context,
                        rope_emb=rope_emb,
                        transformer_options=chunk_transformer_options,
                    )
                    weighted = tag_out * float(weight)
                    if sign < 0:
                        negative_accum = weighted if negative_accum is None else negative_accum + weighted
                        negative_weight += float(weight)
                    else:
                        positive_accum = weighted if positive_accum is None else positive_accum + weighted
                        positive_weight += float(weight)

                if (
                    (positive_accum is None or positive_weight <= 0.0)
                    and (negative_accum is None or negative_weight <= 0.0)
                ):
                    outputs.append(base_chunk)
                    continue

                mixed = base_chunk
                if positive_accum is not None and positive_weight > 0.0:
                    positive_mix = positive_accum / positive_weight
                    mixed = torch.lerp(base_chunk, positive_mix, self.mix_strength)
                if negative_accum is not None and negative_weight > 0.0:
                    negative_mix = negative_accum / negative_weight
                    negative_scale = self._negative_scale(positive_weight, negative_weight)
                    mixed = mixed - self.mix_strength * negative_scale * (negative_mix - base_chunk)

                outputs.append(mixed.to(device=base_chunk.device, dtype=base_chunk.dtype))
                self._mixed_chunk_count += 1
                mixed_this_call += 1

            if mixed_this_call > 0:
                self._mixed_call_count += 1

            if tiled_layout is not None:
                return self._combine_tiled_cond_chunks(outputs, tiled_layout)
            return torch.cat(outputs, dim=0)

        return patched_cross_attn_forward

    def _negative_scale(self, positive_weight: float, negative_weight: float) -> float:
        if negative_weight <= 0.0:
            return 0.0

        reference_weight = positive_weight if positive_weight > 0.0 else 1.0
        return negative_weight / max(reference_weight + negative_weight, 1e-12)

    def _match_context_batch(self, context: torch.Tensor, batch_size: int) -> torch.Tensor:
        if context.shape[0] == batch_size:
            return context

        if context.shape[0] == 1:
            return context.repeat(batch_size, 1, 1)

        if context.shape[0] > batch_size:
            return context[:batch_size]

        repeats = int(math.ceil(batch_size / max(int(context.shape[0]), 1)))
        return context.repeat(repeats, 1, 1)[:batch_size]

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


class AnimaAttenMulti:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Anima model to patch for tag attention mixing"}),
                "clip": ("CLIP", {"tooltip": "Anima text encoder used to encode the tag prompts"}),
                "base_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Base prompt. The node encodes this and outputs positive conditioning for KSampler."
                }),
                "tag": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": (
                        "Comma-separated tags and weights. Positive weights add tag style; negative weights subtract it. "
                        "Examples: (3d:1.0), (photorelistic:1.3), (sketch:-0.5)."
                    )
                }),
                "mix_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Interpolates from the base attention output to the weighted tag attention mix."
                }),
                "weight_power": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.25,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": (
                        "Applies effective_weight = abs(weight) ^ weight_power before mixing positive and negative tags. "
                        "1.0 keeps raw weights, 1.5 gently sharpens, 2.0 strongly sharpens, "
                        "4.0+ makes high-weight tags dominate. Values below 1.0 flatten tag balance."
                    )
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print diagnostic messages while sampling."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "positive")
    FUNCTION = "patch"
    CATEGORY = "ToyxyzTestNodes"

    def patch(
        self,
        model,
        clip,
        base_prompt: str,
        tag: str,
        mix_strength: float,
        weight_power: float,
        debug: bool,
    ) -> Tuple[Any]:
        positive = CLIPTextEncode().encode(clip, str(base_prompt if base_prompt is not None else ""))[0]

        if mix_strength <= 0.0:
            return (model, positive)

        tags = []
        for index, tag_text in enumerate(self._split_tag_inputs(tag), start=1):
            parsed = self._parse_tag_input(tag_text)
            if parsed is None:
                continue

            tag_name, raw_weight = parsed
            effective_weight = self._effective_weight(raw_weight, weight_power)
            sign = -1 if raw_weight < 0.0 else 1
            prompt = self._build_tag_prompt(tag_name, base_prompt)
            conditioning = CLIPTextEncode().encode(clip, prompt)[0]
            tags.append({
                "tag": tag_name,
                "raw_weight": raw_weight,
                "weight": effective_weight,
                "sign": sign,
                "prompt": prompt,
                "conditioning": conditioning,
            })
            if debug:
                mode = "negative" if sign < 0 else "positive"
                print(
                    f"[Anima Atten multi] tag_{index}: tag='{tag_name}', "
                    f"weight={raw_weight:g}, effective={effective_weight:g}, mode={mode}"
                )

        if not tags:
            log_warning("Anima Atten multi has no active tag inputs; returning model unchanged")
            return (model, positive)

        if debug:
            positive_total = sum(float(item["weight"]) for item in tags if int(item.get("sign", 1)) > 0)
            negative_total = sum(float(item["weight"]) for item in tags if int(item.get("sign", 1)) < 0)
            reference_total = positive_total if positive_total > 0.0 else 1.0
            negative_scale = (
                negative_total / max(reference_total + negative_total, 1e-12)
                if negative_total > 0.0
                else 0.0
            )
            print(
                "[Anima Atten multi] effective totals: "
                f"positive={positive_total:g}, negative={negative_total:g}, "
                f"negative_scale={negative_scale:g}"
            )

        mixer = AnimatagAttentionMixer(
            model=model,
            tags=tags,
            mix_strength=mix_strength,
            apply_to_uncond=False,
            debug=debug,
        )
        return (mixer.patch(), positive)

    def _split_tag_inputs(self, text: Any) -> List[str]:
        raw = str(text if text is not None else "")
        parts = []
        current = []
        depth = 0

        for char in raw:
            if char == "(":
                depth += 1
            elif char == ")" and depth > 0:
                depth -= 1

            if char == "," and depth == 0:
                item = "".join(current).strip()
                if item:
                    parts.append(item)
                current = []
                continue

            current.append(char)

        item = "".join(current).strip()
        if item:
            parts.append(item)

        return parts

    def _parse_tag_input(self, text: Any) -> Optional[Tuple[str, float]]:
        raw = str(text if text is not None else "").strip()
        if not raw:
            return None

        inner = raw
        if inner.startswith("(") and inner.endswith(")") and len(inner) >= 2:
            inner = inner[1:-1].strip()

        if not inner:
            return None

        tag = inner
        weight = 1.0
        if ":" in inner:
            maybe_tag, maybe_weight = inner.rsplit(":", 1)
            maybe_tag = maybe_tag.strip()
            maybe_weight = maybe_weight.strip()
            if maybe_weight:
                try:
                    weight = float(maybe_weight)
                except ValueError as e:
                    raise ValueError(f"Invalid tag weight in '{raw}'") from e
                tag = maybe_tag

        tag = tag.strip()
        if not tag or weight == 0.0:
            return None

        return tag, weight

    def _effective_weight(self, raw_weight: float, weight_power: float) -> float:
        power = float(weight_power)
        if not math.isfinite(power) or power <= 0.0:
            raise ValueError(f"weight_power must be a positive finite number, got {weight_power}")

        effective = abs(float(raw_weight)) ** power
        if not math.isfinite(effective) or effective <= 0.0:
            raise ValueError(
                f"Effective tag weight must be positive and finite, got {effective} "
                f"from weight={raw_weight} and weight_power={weight_power}"
            )
        return effective

    def _build_tag_prompt(self, tag: str, base_prompt: str) -> str:
        parts = [tag.strip(), str(base_prompt if base_prompt is not None else "").strip()]
        return ", ".join(part for part in parts if part)
