"""
Core logic for Comfy Couple custom nodes - OPTIMIZED + FLUX SUPPORT + LORA HOOK
With built-in Flux injection capability and Regional LoRA Hook support
✅ Flux now uses dynamic calculation (no latent input needed)
✅ Regional LoRA Hook support for SD 1.5 / SDXL
✅ FIXED: Background now works with LoRA hooks
✅ FIXED: skip_positive now works correctly with/without LoRA hooks
"""
import torch
import copy
from typing import Dict, List, Tuple, Optional, Any

from nodes import CLIPTextEncode, ConditioningCombine, ConditioningSetMask, ConditioningAverage
from .couple_utils import (
    log_debug, log_warning, validate_strength, create_zero_conditioning,
    MaskProcessor, MASK_EPSILON, detect_model_architecture, ModelType,
    FLUX_BLOCK_PRESETS, get_flux_preset, detect_or_force_architecture,
    get_model_type_input_options,
)
from .couple_patching import AttentionPatcher
from .couple_flux_patching import FluxAttentionPatcher
from .couple_anima_patching import AnimaAttentionPatcher
from .couple_strategies import build_region_strategy

# Debug mode can be toggled
import os
DEBUG_ENABLED = os.environ.get('COMFYCOUPLE_DEBUG', '0') == '1'

if DEBUG_ENABLED:
    from . import couple_utils
    couple_utils.DEBUG_MODE = True
    print("[ComfyCouple] Debug mode enabled (set COMFYCOUPLE_DEBUG=0 to disable)")

CHAIN_META_KEY = "__comfycouple_chain_meta__"
CHAIN_BASE_PROMPT_TEXT_KEY = "base_prompt_text"
CHAIN_BACKGROUND_PROMPT_TEXT_KEY = "background_prompt_text"


class ComfyCoupleBasePrompt:
    """Attach shared base prompt text to the region chain."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Common base prompt text shared through the region chain. Appended to every region prompt_text that uses the chain text system."
                }),
            },
            "optional": {
                "region": ("region", {"tooltip": "Optional existing region chain"}),
            }
        }

    RETURN_TYPES = ("region",)
    FUNCTION = "process"
    CATEGORY = "ToyxyzTestNodes"

    def process(
        self,
        base_prompt_text: str,
        region: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]]]:
        region_list = copy.deepcopy(region) if region is not None else []
        region_list.append({
            CHAIN_META_KEY: {
                CHAIN_BASE_PROMPT_TEXT_KEY: base_prompt_text,
            }
        })
        return (region_list,)


class ComfyCoupleBackgroundPrompt:
    """Attach shared background prompt text to the region chain."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_prompt_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Background prompt text shared through the region chain. Appended to every region prompt_text and used for uncovered background areas."
                }),
            },
            "optional": {
                "region": ("region", {"tooltip": "Optional existing region chain"}),
            }
        }

    RETURN_TYPES = ("region",)
    FUNCTION = "process"
    CATEGORY = "ToyxyzTestNodes"

    def process(
        self,
        background_prompt_text: str,
        region: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]]]:
        region_list = copy.deepcopy(region) if region is not None else []
        region_list.append({
            CHAIN_META_KEY: {
                CHAIN_BACKGROUND_PROMPT_TEXT_KEY: background_prompt_text,
            }
        })
        return (region_list,)

class ComfyCoupleRegion:
    """Define a region with prompt and mask"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Region mask (white=100%, black=0%)"}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Prompt strength (<1: blend, 1: full, >1: emphasize) [Flux: 0 or 1 only]"
                }),
                "prompt_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Primary prompt text for this region. When prompt_text and clip are connected, the chain text system uses this for all supported model types."
                }),
                "clip": ("CLIP", {
                    "tooltip": "CLIP used to encode prompt_text for the chain text system. Use the same CLIP family as the connected model."
                }),
            },
            "optional": {
                "lora_hook": ("HOOKS", {
                    "tooltip": "LoRA hook from 'Create Hook LoRA' node (SD 1.5/SDXL only, ignored in Flux). Use 'Combine Hooks' node to merge multiple LoRAs before connecting."
                }),
                "region": ("region", {"tooltip": "Chain multiple regions"}),
            }
        }

    RETURN_TYPES = ("region",)
    FUNCTION = "process"
    CATEGORY = "ToyxyzTestNodes"

    def process(self, mask: torch.Tensor, strength: float, prompt_text: str, clip: Any,
                lora_hook: Optional[Any] = None,
                region: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[Dict[str, Any]]]:
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"Mask must be torch.Tensor, got {type(mask)}")

        current_region = {
            "mask": mask, 
            "strength": strength,
            "prompt_text": prompt_text,
            "clip": clip,
            "lora_hook": lora_hook  # ✅ Store LoRA hook
        }
        
        region_list = region if region is not None else []
        first_region = next((item for item in region_list if "mask" in item), None)
        
        # Auto-resize if needed
        if first_region is not None and first_region["mask"].shape != mask.shape:
            try:
                target_shape = first_region["mask"].shape
                current_region["mask"] = MaskProcessor.resize_to_match(mask, target_shape)
                log_debug(f"Auto-resized mask: {mask.shape} -> {target_shape}")
            except Exception as e:
                raise RuntimeError(f"Mask resize failed: {e}")

        region_list.append(current_region)
        return (region_list,)


class ComfyCoupleRegionMulti:
    """Define multiple regions at once from mask and prompt text lists."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK", {"tooltip": "Mask list input. Each mask becomes one region in order."}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Shared prompt strength for all generated regions (<1: blend, 1: full, >1: emphasize) [Flux: 0 or 1 only]"
                }),
                "prompt_texts": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "String list input. Each text is matched to the mask at the same index."
                }),
                "clip": ("CLIP", {
                    "tooltip": "CLIP used to encode every prompt_text in the chain text system. Use the same CLIP family as the connected model."
                }),
            },
            "optional": {
                "region": ("region", {"tooltip": "Optional existing region chain to append to"}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("region",)
    FUNCTION = "process"
    CATEGORY = "ToyxyzTestNodes"

    @staticmethod
    def _resolve_single_value(values: Any, name: str, allow_none: bool = False) -> Any:
        if isinstance(values, list):
            filtered = [value for value in values if value is not None]
            if not filtered:
                if allow_none:
                    return None
                raise ValueError(f"ComfyCouple Region multi requires '{name}' to be connected")
            if len(filtered) > 1:
                raise ValueError(f"ComfyCouple Region multi expects a single '{name}' value")
            return filtered[0]

        if values is None and not allow_none:
            raise ValueError(f"ComfyCouple Region multi requires '{name}' to be connected")
        return values

    @staticmethod
    def _resolve_mask_list(masks: Any) -> List[torch.Tensor]:
        if isinstance(masks, list):
            mask_list = masks
        else:
            mask_list = [masks]

        resolved_masks = []
        for index, mask in enumerate(mask_list):
            if mask is None:
                continue
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"Mask at index {index} must be torch.Tensor, got {type(mask)}")
            resolved_masks.append(mask)

        if not resolved_masks:
            raise ValueError("ComfyCouple Region multi requires at least one mask")

        return resolved_masks

    @staticmethod
    def _resolve_prompt_text_list(prompt_texts: Any) -> List[str]:
        if isinstance(prompt_texts, list):
            return [str(text if text is not None else "") for text in prompt_texts]
        return [str(prompt_texts if prompt_texts is not None else "")]

    @staticmethod
    def _find_reference_mask_shape(region_list: List[Dict[str, Any]]) -> Optional[torch.Size]:
        first_region = next((item for item in region_list if isinstance(item, dict) and "mask" in item), None)
        if first_region is None:
            return None
        return first_region["mask"].shape

    def process(
        self,
        masks,
        strength,
        prompt_texts,
        clip,
        region=None,
    ) -> Tuple[List[Dict[str, Any]]]:
        mask_list = self._resolve_mask_list(masks)
        prompt_text_list = self._resolve_prompt_text_list(prompt_texts)
        strength_value = float(self._resolve_single_value(strength, "strength"))
        clip_value = self._resolve_single_value(clip, "clip")
        region_chain = self._resolve_single_value(region, "region", allow_none=True)
        region_list = list(region_chain) if region_chain is not None else []

        if len(mask_list) != len(prompt_text_list):
            raise ValueError(
                f"ComfyCouple Region multi requires the same number of masks and prompt_texts "
                f"(got {len(mask_list)} masks and {len(prompt_text_list)} prompt_texts)"
            )

        target_shape = self._find_reference_mask_shape(region_list)

        for index, (mask, prompt_text) in enumerate(zip(mask_list, prompt_text_list)):
            current_mask = mask

            if target_shape is None:
                target_shape = current_mask.shape
            elif current_mask.shape != target_shape:
                try:
                    current_mask = MaskProcessor.resize_to_match(current_mask, target_shape)
                    log_debug(f"Auto-resized multi region mask {index}: {mask.shape} -> {target_shape}")
                except Exception as e:
                    raise RuntimeError(f"Mask resize failed for multi region index {index}: {e}")

            region_list.append({
                "mask": current_mask,
                "strength": strength_value,
                "prompt_text": prompt_text,
                "clip": clip_value,
                "lora_hook": None,
            })

        return (region_list,)


class ComfyCoupleMask:
    """Process multiple regions with individual prompts - Supports SD/SDXL/Flux"""

    @classmethod
    def INPUT_TYPES(cls):
        # Flux block preset descriptions for tooltip
        flux_preset_options = list(FLUX_BLOCK_PRESETS.keys())
        flux_preset_tooltip = "Flux attention block preset:\n" + "\n".join([
            f"• {name}: {FLUX_BLOCK_PRESETS[name]['description']}" 
            for name in flux_preset_options
        ])
        
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to apply regional prompting"}),
                "region": ("region", {"tooltip": "Connect ComfyCoupleRegion node"}),
                "negative": ("CONDITIONING", {"tooltip": "Global negative prompt"}),
                "cross_region_attention": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend boundaries (0: separate, 0.3: soft, 1: mixed) [SD/SDXL only]"
                }),
                "cross_region_mode": (["self_exclusion", "global_average"], {
                    "default": "self_exclusion",
                    "tooltip": "Cross-region blending mode [SD/SDXL only]"
                }),
                "feather_mask": ("INT", {
                    "default": 0, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Soften mask edges (0: sharp, 10-30: soft)"
                }),
                "model_type": (get_model_type_input_options(), {
                    "default": "auto",
                    "tooltip": "Model architecture (auto: detect, flux: force Flux mode)"
                }),
                "skip_positive_conditioning": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip positive conditioning output (use zero conditioning). True: Only attention patching is applied, no positive output. False: Output all regions with full conditioning. If you are using the Lora hook, set it to false.[SD/SDXL only, always True for Flux]"
                }),
            },
            "optional": {
                # Flux-specific options
                "flux_block_preset": (flux_preset_options, {
                    "default": "default",
                    "tooltip": flux_preset_tooltip
                }),
                "flux_start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Flux: Start applying regional mask at this % [Flux only]"
                }),
                "flux_end_percent": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Flux: Stop applying regional mask at this % [Flux only]"
                }),
                "auto_inject_flux": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-inject Flux modifications. This must be set to True when using the Flux model. Once executed, it will remain active until the model cache is freed."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "process"
    CATEGORY = "ToyxyzTestNodes"

    def process(self, model, region, negative, **kwargs) -> Tuple[Any, Any, Any]:
        if not isinstance(region, list) or not region:
            raise ValueError("At least one region required")

        region, chain_metadata = self._split_region_chain(region)
        if not region:
            raise ValueError("At least one real region required")

        arch_info = self._detect_architecture(model, kwargs.get("model_type", "auto"))
        kwargs = dict(kwargs)
        kwargs["_chain_metadata"] = chain_metadata
        kwargs = self._apply_capability_constraints(model, arch_info, region, kwargs)
        strategy = build_region_strategy(arch_info)
        display_name = arch_info.display_name or arch_info.type.value.upper()

        print(f"[ComfyCouple] Detected: {display_name}")

        model = strategy.prepare_model(self, model, kwargs)
        return strategy.process(self, model, region, negative, **kwargs)

    def _split_region_chain(self, region_chain: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        regions: List[Dict[str, Any]] = []
        chain_metadata: Dict[str, Any] = {}

        for item in region_chain:
            if isinstance(item, dict) and CHAIN_META_KEY in item:
                chain_metadata.update(item[CHAIN_META_KEY])
            else:
                regions.append(item)

        return regions, chain_metadata

    def _apply_capability_constraints(
        self,
        model: Any,
        arch_info: Any,
        region: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        display_name = arch_info.display_name or arch_info.type.value.upper()

        has_any_lora = any(r.get("lora_hook") is not None for r in region)
        if has_any_lora and not arch_info.supports("supports_lora_hooks"):
            print(f"[ComfyCouple] Warning: LoRA hooks are not supported for {display_name} and will be ignored")
            log_warning(f"LoRA hooks detected but {display_name} does not support them")

        cross_region_attention = kwargs.get("cross_region_attention", 0.0)
        if cross_region_attention > 0.0 and not arch_info.supports("supports_cross_region_attention"):
            print(f"[ComfyCouple] Warning: Cross-region attention is not supported for {display_name}; using 0.0")
            log_warning(f"cross_region_attention ignored for {display_name}")
            kwargs["cross_region_attention"] = 0.0

        if arch_info.supports("requires_injection") and not kwargs.get("auto_inject_flux", True):
            print(f"[ComfyCouple] Warning: {display_name} may require injection; auto_inject_flux is disabled")
            log_warning(f"{display_name} usually requires injection for regional prompting")

        transformer_options = model.model_options.get("transformer_options", {})
        if transformer_options.get("tiled_diffusion") and not arch_info.supports("supports_tiled_diffusion"):
            print(f"[ComfyCouple] Warning: TiledDiffusion is not supported for {display_name}")
            log_warning(f"TiledDiffusion metadata detected but {display_name} does not support it")

        return kwargs

    def _inject_flux_if_needed(self, model: Any) -> Any:
        """
        Inject Flux modifications if not already injected
        
        This allows ComfyCouple to work without Configure Modified Flux node
        """
        try:
            from .flux_injection import inject_flux_for_comfycouple
            
            # Check if already injected
            if hasattr(model.model.diffusion_model, '_comfycouple_flux_injected'):
                log_debug("Flux already injected")
                return model
            
            # Perform injection
            print("[ComfyCouple] Auto-injecting Flux modifications...")
            model = inject_flux_for_comfycouple(model)
            
            return model
            
        except ImportError as e:
            log_warning(f"Flux injection failed: {e}")
            log_warning("Please use 'Configure Modified Flux' node from Fluxtapoz")
            return model
        except Exception as e:
            log_warning(f"Flux injection error: {e}")
            log_warning("Continuing without injection - may not work correctly")
            return model

    def _detect_architecture(self, model: Any, model_type_str: str) -> Any:
        """Detect or force model architecture"""
        return detect_or_force_architecture(model, model_type_str)

    def _process_flux(self, model, region, negative, arch_info, **kwargs) -> Tuple[Any, Any, Any]:
        """Process Flux model using attention mask approach - WITH DYNAMIC CALCULATION"""
        chain_metadata = kwargs.get("_chain_metadata", {})

        # Remove duplicates
        region = self._remove_duplicate_regions(region)
        log_debug(f"Processing {len(region)} unique regions (Flux mode - dynamic calculation)")
        
        # Validate strengths for Flux (warn if not 0 or 1)
        for i, r in enumerate(region):
            orig_strength = r['strength']
            r['strength'] = validate_strength(orig_strength, f"Region {i}", is_flux=True)
            if abs(orig_strength - r['strength']) > 0.01:
                log_warning(f"Flux only supports strength 0 or 1, converted {orig_strength} → {r['strength']}")
        
        # Prepare regions (feathering, normalization)
        prepared_regions = self._prepare_regions(region, kwargs, is_flux=True)
        
        # Filter active regions (strength > 0)
        active_regions = [r for r in prepared_regions if r['strength'] > 0]
        
        if not active_regions:
            raise ValueError("All regions have zero strength")
        
        # Calculate background mask (same logic as SD/SDXL)
        combined_mask = torch.stack([r['mask'] for r in active_regions]).sum(dim=0)
        background_mask = (1.0 - combined_mask).clamp(min=0.0)
        has_background = background_mask.sum() > MASK_EPSILON

        final_active_regions = []
        for r in active_regions:
            final_active_regions.append({
                **r,
                "positive": self._resolve_region_conditioning(r, chain_metadata),
            })

        # Add background region if needed
        if has_background:
            bg_positive = self._resolve_background_conditioning(prepared_regions, chain_metadata)
            bg_region = {
                'positive': bg_positive,
                'mask': background_mask,
                'strength': 1.0,
                'is_background': True,
            }
            final_regions = [bg_region] + final_active_regions
            log_debug(f"Background added (coverage: {background_mask.sum().item():.1f} pixels)")
        else:
            final_regions = final_active_regions
            log_debug("No background area (regions cover entire image)")
        
        # Get Flux block preset
        preset_name = kwargs.get('flux_block_preset', 'default')
        attn_override = get_flux_preset(preset_name)
        
        print(f"[ComfyCouple Flux] Using block preset: '{preset_name}'")
        print(f"[ComfyCouple Flux] Double blocks ({len(attn_override['double'])}): {attn_override['double']}")
        print(f"[ComfyCouple Flux] Single blocks ({len(attn_override['single'])}): {attn_override['single']}")
        
        # ✅ Create Flux patcher WITHOUT latent
        flux_patcher = FluxAttentionPatcher(
            model=model,
            regions=final_regions,
            start_percent=kwargs.get('flux_start_percent', 0.0),
            end_percent=kwargs.get('flux_end_percent', 0.5),
            attn_override=attn_override
        )
        
        # Patch model (attention patching)
        patched_model, placeholder_positive = flux_patcher.patch()
        
        # Build positive output like SD/SDXL
        processed_conds = []
        
        # Process all final_regions (including background)
        for r in final_regions:
            processed_conds.append({
                'conditioning': r['positive'],
                'mask': r['mask'],
                'strength': r['strength']
            })
        
        # Combine all regions using ConditioningSetMask + ConditioningCombine
        combined_positive = self._convert_to_output_format(processed_conds)
        
        # Create zero conditioning
        zero_positive = create_zero_conditioning(negative)
        
        # Use ConditioningAverage to blend
        final_positive = ConditioningAverage().addWeighted(
            zero_positive,
            combined_positive,
            0.0
        )[0]
        
        log_debug("Flux positive output: combined all regions + background with ConditioningAverage (strength=0.0)")
        print(f"[ComfyCouple Flux] Positive output: {len(final_regions)} regions combined (including background)")
        print(f"[ComfyCouple Flux] Mask will be calculated dynamically based on query tensor")
        
        return (patched_model, final_positive, negative)

    def _process_anima(self, model, region, negative, arch_info, **kwargs) -> Tuple[Any, Any, Any]:
        """Process Anima using an apply_model wrapper on the cloned ModelPatcher."""
        chain_metadata = kwargs.get("_chain_metadata", {})
        region = self._remove_duplicate_regions(region)
        self._validate_mask_sizes_for_upscale(region)
        log_debug(f"Processing {len(region)} unique regions (Anima mode)")

        prepared_regions = self._prepare_regions(region, kwargs, is_flux=False)
        active_regions, background_mask = self._process_strength_zero_regions(prepared_regions)
        if not active_regions:
            raise ValueError("All regions have zero strength")

        processed_regions = []
        for r in active_regions:
            region_cond = self._resolve_region_conditioning(r, chain_metadata)
            processed_regions.append(
                {
                    "conditioning": region_cond,
                    "mask": r["mask"],
                    "strength": r["strength"],
                    "is_background": False,
                }
            )

        if background_mask.sum() > MASK_EPSILON:
            final_bg = self._resolve_background_conditioning(prepared_regions, chain_metadata)
            processed_regions.append(
                {
                    "conditioning": final_bg,
                    "mask": background_mask,
                    "strength": 1.0,
                    "is_background": True,
                }
            )

        patcher = AnimaAttentionPatcher(model=model, regions=processed_regions)
        patched_model = patcher.patch()

        if kwargs.get("skip_positive_conditioning", True):
            final_positive_output = create_zero_conditioning(negative)
        else:
            print("[ComfyCouple] Anima positive output passthrough is experimental")
            log_warning("Anima skip_positive_conditioning=False may change conditioning behavior depending on downstream nodes")
            final_positive_output = self._convert_to_output_format(processed_regions)
        return (patched_model, final_positive_output, negative)

    def _process_sd_sdxl(self, model, region, negative, arch_info, **kwargs) -> Tuple[Any, Any, Any]:
        """Process SD/SDXL model using original approach with LoRA Hook support"""
        
        chain_metadata = kwargs.get("_chain_metadata", {})

        # Remove duplicates
        region = self._remove_duplicate_regions(region)
        self._validate_mask_sizes_for_upscale(region)
        log_debug(f"Processing {len(region)} unique regions (SD/SDXL mode)")

        prepared_regions = self._prepare_regions(region, kwargs, is_flux=False)
        
        # Separate active and zero-strength regions
        active_regions, background_mask = self._process_strength_zero_regions(prepared_regions)
        
        if not active_regions:
            raise ValueError("All regions have zero strength")

        # ✅ Process conditioning with LoRA Hook support
        all_processed_conds = []
        
        # Process active regions
        for r in active_regions:
            # Step 1: Resolve chain-text conditioning or fallback conditioning
            region_cond = self._resolve_region_conditioning(r, chain_metadata)
            
            # Step 2: Apply mask metadata FIRST
            region_cond = ConditioningSetMask().append(
                region_cond, r['mask'], "default", r['strength']
            )[0]
            
            # Step 3: Attach LoRA hook AFTER ConditioningSetMask
            # Follows ComfyUI's Cond Set Props behavior: preserve existing hooks
            has_lora = False
            if r.get('lora_hook') is not None:
                for cond_item in region_cond:
                    # Get existing hooks (if any)
                    existing_hooks = cond_item[1].get('hooks', None)
                    new_hook = r['lora_hook']
                    
                    if existing_hooks is None:
                        # No existing hooks: just set the new one
                        cond_item[1]['hooks'] = new_hook
                    else:
                        # Merge hooks using clone() method
                        # HookGroup has a clone() method that creates a merged copy
                        try:
                            if hasattr(existing_hooks, 'clone'):
                                # Clone existing and merge with new
                                merged_hooks = existing_hooks.clone()
                                if hasattr(merged_hooks, 'hooks') and hasattr(new_hook, 'hooks'):
                                    # Both are HookGroups, merge their hooks lists
                                    merged_hooks.hooks.extend(new_hook.hooks)
                                    cond_item[1]['hooks'] = merged_hooks
                                else:
                                    # Can't merge properly, use new hook
                                    log_warning("Cannot merge hooks properly, using new hook only")
                                    cond_item[1]['hooks'] = new_hook
                            else:
                                # No clone method, overwrite with new hook
                                log_warning("Existing hooks don't support clone(), using new hook only")
                                cond_item[1]['hooks'] = new_hook
                        except Exception as e:
                            log_warning(f"Hook merge failed: {e}, using new hook only")
                            cond_item[1]['hooks'] = new_hook
                
                has_lora = True
                log_debug(f"Attached LoRA hook to region with mask shape {r['mask'].shape}")
            
            all_processed_conds.append({
                'conditioning': region_cond,
                'has_lora': has_lora,  # ✅ Track LoRA presence
                'mask': r['mask'],
                'strength': r['strength']
            })
        
        # Process background
        has_background = background_mask.sum() > MASK_EPSILON
        if has_background:
            final_bg = self._resolve_background_conditioning(prepared_regions, chain_metadata)
            final_bg = ConditioningSetMask().append(
                final_bg, background_mask, "default", 1.0
            )[0]
            
            all_processed_conds.append({
                'conditioning': final_bg,
                'has_lora': False,  # ✅ Background doesn't have LoRA hook
                'mask': background_mask,
                'strength': 1.0
            })
            log_debug(f"Background included (mask coverage: {background_mask.sum().item():.1f})")

        # ✅ AttentionPatcher always uses ALL regions (for spatial control)
        # Convert to patcher format: [[tensor, {mask, mask_strength}], ...]
        patcher_conds = []
        for c in all_processed_conds:
            # Extract tensor from conditioning
            cond_tensor = c['conditioning'][0][0]
            patcher_conds.append([
                cond_tensor,
                {
                    "mask": c['mask'],
                    "mask_strength": c['strength']
                }
            ])
        
        patcher = AttentionPatcher(
            model=model,
            positive=patcher_conds,
            negative=negative,
            cross_region_attention=kwargs.get("cross_region_attention", 0.0),
            cross_region_mode=kwargs.get("cross_region_mode", "self_exclusion"),
            model_type=arch_info.type.value
        )
        patched_model, _, _ = patcher.patch()

        # ✅ FIXED: Simplified skip logic
        skip_positive = kwargs.get("skip_positive_conditioning", True)

        if skip_positive:
            # ✅ 항상 zero conditioning 사용 (LoRA hook 유무와 무관)
            final_positive_output = create_zero_conditioning(negative)
            print(f"[ComfyCouple] Skip=True: Using zero conditioning (attention patching only)")
            log_debug("Skip=True: Zero conditioning output, all regions applied via attention patching")
        else:
            # ✅ 모든 리전 출력 (LoRA hook 있는 것과 없는 것 모두 포함)
            final_positive_output = self._convert_to_output_format_lora(all_processed_conds)
            
            # LoRA 카운트 (디버그용)
            lora_count = sum(1 for c in all_processed_conds if c['has_lora'])
            non_lora_count = len(all_processed_conds) - lora_count
            
            if lora_count > 0:
                print(f"[ComfyCouple] Skip=False: Output {lora_count} LoRA regions + {non_lora_count} non-LoRA regions (total: {len(all_processed_conds)})")
                log_debug(f"Skip=False: {lora_count} LoRA + {non_lora_count} non-LoRA regions")
            else:
                print(f"[ComfyCouple] Skip=False: Output all {len(all_processed_conds)} regions")
                log_debug(f"Skip=False: Output all {len(all_processed_conds)} regions")

        return (patched_model, final_positive_output, negative)

    def _build_chain_text_conditioning(
        self,
        region: Dict[str, Any],
        chain_metadata: Dict[str, Any],
    ) -> Optional[Any]:
        base_prompt_text = str(chain_metadata.get(CHAIN_BASE_PROMPT_TEXT_KEY, "") or "").strip()
        background_prompt_text = str(chain_metadata.get(CHAIN_BACKGROUND_PROMPT_TEXT_KEY, "") or "").strip()
        prompt_text = str(region.get("prompt_text", "") or "").strip()
        clip = region.get("clip")

        if not prompt_text or clip is None:
            return None

        merged_prompt = self._merge_prompt_text(prompt_text, base_prompt_text, background_prompt_text)
        return CLIPTextEncode().encode(clip, merged_prompt)[0]

    def _build_chain_background_conditioning(
        self,
        regions: List[Dict[str, Any]],
        chain_metadata: Dict[str, Any],
    ) -> Optional[Any]:
        background_prompt_text = str(chain_metadata.get(CHAIN_BACKGROUND_PROMPT_TEXT_KEY, "") or "").strip()
        base_prompt_text = str(chain_metadata.get(CHAIN_BASE_PROMPT_TEXT_KEY, "") or "").strip()
        if not background_prompt_text:
            return None

        clip = next((r.get("clip") for r in regions if r.get("clip") is not None), None)
        if clip is None:
            return None

        merged_prompt = self._merge_prompt_text(background_prompt_text, base_prompt_text)
        return CLIPTextEncode().encode(clip, merged_prompt)[0]

    def _merge_prompt_text(self, *prompt_parts: str) -> str:
        parts = [str(part).strip() for part in prompt_parts if str(part).strip()]
        return ", ".join(parts)

    def _resolve_region_conditioning(
        self,
        region: Dict[str, Any],
        chain_metadata: Dict[str, Any],
    ) -> Any:
        chain_cond = self._build_chain_text_conditioning(region, chain_metadata)
        if chain_cond is not None:
            return chain_cond

        missing = []
        if not str(region.get("prompt_text", "") or "").strip():
            missing.append("prompt_text")
        if region.get("clip") is None:
            missing.append("clip")

        if missing:
            raise ValueError(
                f"ComfyCouple Region requires {', '.join(missing)} for the chain text system"
            )

        raise ValueError("ComfyCouple Region could not build conditioning from the chain text system")

    def _resolve_background_conditioning(
        self,
        regions: List[Dict[str, Any]],
        chain_metadata: Dict[str, Any],
    ) -> Any:
        chain_cond = self._build_chain_background_conditioning(regions, chain_metadata)
        if chain_cond is not None:
            return chain_cond
        return create_zero_conditioning(self._resolve_region_conditioning(regions[0], chain_metadata))

    # ===== Helper methods =====

    def _validate_mask_sizes_for_upscale(self, region_list: List[Dict]) -> None:
        if not region_list:
            return
        
        first_shape = region_list[0]["mask"].shape
        all_same = all(r["mask"].shape == first_shape for r in region_list)
        
        if all_same:
            mask_h, mask_w = first_shape[-2], first_shape[-1]
            log_debug(f"All region masks: {mask_h}×{mask_w}")
            print(f"[ComfyCouple] Mask size: {mask_h}×{mask_w}")
            print(f"[ComfyCouple] Masks will auto-scale to match image resolution")
        else:
            log_warning("Region masks have different sizes - may cause unexpected results")

    def _remove_duplicate_regions(self, region_list: List[Dict]) -> List[Dict]:
        """Remove only truly identical regions to avoid dropping distinct masks."""
        unique_regions = []

        for region in region_list:
            mask = region["mask"]
            duplicate_idx = None

            for idx, existing in enumerate(unique_regions):
                existing_mask = existing["mask"]
                if mask.shape == existing_mask.shape and torch.equal(mask, existing_mask):
                    duplicate_idx = idx
                    break

            if duplicate_idx is not None:
                unique_regions[duplicate_idx] = region
            else:
                unique_regions.append(region)

        if len(unique_regions) != len(region_list):
            log_warning(f"Removed {len(region_list) - len(unique_regions)} duplicate regions")

        return unique_regions

    def _prepare_regions(self, region_list: List[Dict], kwargs: Dict, is_flux: bool = False) -> List[Dict[str, Any]]:
        """Copy, validate, feather, and normalize regions"""
        feather_pixels = kwargs.get("feather_mask", 0)
        
        prepared_regions = []
        for i, r in enumerate(region_list):
            new_region = {
                "mask": r["mask"].clone(),
                "strength": validate_strength(r["strength"], f"Region {i}", is_flux=is_flux),
                "prompt_text": r.get("prompt_text", ""),
                "clip": r.get("clip"),
                "lora_hook": r.get("lora_hook")  # ✅ Preserve LoRA hook
            }
            if feather_pixels > 0:
                try:
                    new_region["mask"] = MaskProcessor.apply_feather(new_region["mask"], feather_pixels)
                except Exception as e:
                    log_warning(f"Feathering failed for region {i}: {e}")
            prepared_regions.append(new_region)
        
        # ✅ Apply overlap normalization (original behavior)
        return MaskProcessor.normalize_overlaps(prepared_regions)

    def _process_strength_zero_regions(self, regions: List[Dict]) -> Tuple[List[Dict], torch.Tensor]:
        """Separate active regions and calculate background mask"""
        active_regions = []
        zero_strength_count = 0
        
        for r in regions:
            if r["strength"] > 0.0:
                active_regions.append(r)
            else:
                zero_strength_count += 1
                log_debug(f"Region with strength=0.0 will use background")
        
        if active_regions:
            combined_active_mask = torch.stack([r['mask'] for r in active_regions]).sum(dim=0)
            background_mask = (1.0 - combined_active_mask).clamp(min=0.0)
        else:
            background_mask = torch.ones_like(regions[0]["mask"])
        
        log_debug(f"Active regions: {len(active_regions)}, Zero-strength regions: {zero_strength_count}")
        return active_regions, background_mask

    def _convert_to_output_format(self, processed_conds: List[Dict[str, Any]]) -> Any:
        """Convert to ComfyUI output format (original method for Flux)"""
        if not processed_conds:
            raise ValueError("No processed conditionings available")
        
        combined_positive = None
        
        for item in processed_conds:
            masked_cond = ConditioningSetMask().append(
                item['conditioning'],
                item['mask'],
                "default",
                item['strength']
            )[0]
            
            if combined_positive is None:
                combined_positive = masked_cond
            else:
                combined_positive = ConditioningCombine().combine(
                    combined_positive,
                    masked_cond
                )[0]
        
        return combined_positive

    def _convert_to_output_format_lora(self, processed_conds: List[Dict[str, Any]]) -> Any:
        """
        ✅ Convert to ComfyUI output format (for LoRA Hook support)
        Note: conditioning already has mask metadata from ConditioningSetMask
        """
        if not processed_conds:
            raise ValueError("No processed conditionings available")
        
        combined_positive = None
        
        for item in processed_conds:
            # ✅ Use conditioning directly (already has mask + LoRA hook)
            region_cond = item['conditioning']
            
            if combined_positive is None:
                combined_positive = region_cond
            else:
                combined_positive = ConditioningCombine().combine(
                    combined_positive,
                    region_cond
                )[0]
        
        return combined_positive
