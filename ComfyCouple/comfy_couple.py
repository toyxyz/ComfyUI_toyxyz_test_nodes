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
from typing import Dict, List, Tuple, Optional, Any, Callable

from nodes import ConditioningCombine, ConditioningConcat, ConditioningSetMask, ConditioningAverage
from .couple_utils import (
    log_debug, log_warning, validate_strength, create_zero_conditioning,
    MaskProcessor, MASK_EPSILON, detect_model_architecture, ModelType,
    FLUX_BLOCK_PRESETS, get_flux_preset
)
from .couple_patching import AttentionPatcher
from .couple_flux_patching import FluxAttentionPatcher

# Debug mode can be toggled
import os
DEBUG_ENABLED = os.environ.get('COMFYCOUPLE_DEBUG', '0') == '1'

if DEBUG_ENABLED:
    from . import couple_utils
    couple_utils.DEBUG_MODE = True
    print("[ComfyCouple] Debug mode enabled (set COMFYCOUPLE_DEBUG=0 to disable)")

COND_METHODS: Dict[str, Callable] = {
    'concat': lambda a, b: ConditioningConcat().concat(a, b)[0],
    'combine': lambda a, b: ConditioningCombine().combine(a, b)[0]
}

class ComfyCoupleRegion:
    """Define a region with prompt and mask"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "Prompt for this region"}),
                "mask": ("MASK", {"tooltip": "Region mask (white=100%, black=0%)"}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Prompt strength (<1: blend, 1: full, >1: emphasize) [Flux: 0 or 1 only]"
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

    def process(self, positive: Any, mask: torch.Tensor, strength: float,
                lora_hook: Optional[Any] = None,
                region: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[Dict[str, Any]]]:
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"Mask must be torch.Tensor, got {type(mask)}")

        current_region = {
            "positive": positive, 
            "mask": mask, 
            "strength": strength,
            "lora_hook": lora_hook  # ✅ Store LoRA hook
        }
        
        region_list = region if region is not None else []
        
        # Auto-resize if needed
        if region_list and region_list[0]["mask"].shape != mask.shape:
            try:
                target_shape = region_list[0]["mask"].shape
                current_region["mask"] = MaskProcessor.resize_to_match(mask, target_shape)
                log_debug(f"Auto-resized mask: {mask.shape} -> {target_shape}")
            except Exception as e:
                raise RuntimeError(f"Mask resize failed: {e}")

        region_list.append(current_region)
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
                "model_type": (["auto", "sd15", "sdxl", "flux"], {
                    "default": "auto",
                    "tooltip": "Model architecture (auto: detect, flux: force Flux mode)"
                }),
                "skip_positive_conditioning": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip positive conditioning output (use zero conditioning). True: Only attention patching is applied, no positive output. False: Output all regions with full conditioning. If you are using the Lora hook, set it to false.[SD/SDXL only, always True for Flux]"
                }),
            },
            "optional": {
                "background": ("CONDITIONING", {"tooltip": "Prompt for unmasked areas"}),
                "background_to_region_mode": (["none", "concat", "combine"], {
                    "default": "concat",
                    "tooltip": "How to apply background to regions"
                }),
                "base_prompt": ("CONDITIONING", {"tooltip": "Common prompt for all regions"}),
                "base_prompt_method": (["concat", "combine"], {
                    "default": "concat",
                    "tooltip": "How to merge base prompt"
                }),
                "background_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Background prompt strength"
                }),
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

        # Detect model architecture
        arch_info = self._detect_architecture(model, kwargs.get("model_type", "auto"))
        is_flux = arch_info.is_flux
        
        print(f"[ComfyCouple] Detected: {arch_info.type.value.upper()}")
        
        # ✅ Flux Injection (if enabled and Flux detected)
        if is_flux and kwargs.get("auto_inject_flux", True):
            model = self._inject_flux_if_needed(model)
        
        # Route to appropriate implementation
        if is_flux:
            return self._process_flux(model, region, negative, arch_info, **kwargs)
        else:
            return self._process_sd_sdxl(model, region, negative, arch_info, **kwargs)

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
        if model_type_str == "auto":
            return detect_model_architecture(model)
        elif model_type_str == "flux":
            from .couple_utils import ArchitectureInfo
            return ArchitectureInfo(type=ModelType.FLUX, use_cfg=True, dim=4096, is_flux=True)
        elif model_type_str == "sdxl":
            from .couple_utils import ArchitectureInfo
            return ArchitectureInfo(type=ModelType.SDXL, use_cfg=True, dim=2048, is_flux=False)
        else:  # sd15
            from .couple_utils import ArchitectureInfo
            return ArchitectureInfo(type=ModelType.SD15, use_cfg=True, dim=768, is_flux=False)

    def _process_flux(self, model, region, negative, arch_info, **kwargs) -> Tuple[Any, Any, Any]:
        """Process Flux model using attention mask approach - WITH DYNAMIC CALCULATION"""
        
        # ✅ Check for LoRA hooks and warn
        has_any_lora = any(r.get('lora_hook') is not None for r in region)
        if has_any_lora:
            print("[ComfyCouple] ⚠️ Warning: LoRA hooks are not supported in Flux mode and will be ignored")
            log_warning("LoRA hooks detected but Flux mode does not support them")
        
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
        
        # Add background region if needed
        bg_positive = kwargs.get('background')
        if has_background and bg_positive is not None:
            # Apply base_prompt to background if provided
            if kwargs.get('base_prompt'):
                base_method = kwargs.get('base_prompt_method', 'concat')
                bg_positive = COND_METHODS[base_method](bg_positive, kwargs['base_prompt'])
            
            # Create background region
            bg_region = {
                'positive': bg_positive,
                'mask': background_mask,
                'strength': 1.0
            }
            # Add background as first region
            final_regions = [bg_region] + active_regions
            log_debug(f"Background added (coverage: {background_mask.sum().item():.1f} pixels)")
        else:
            final_regions = active_regions
            if has_background:
                log_debug("No background provided, unmask area will not be filled")
            else:
                log_debug("No background area (regions cover entire image)")
        
        # Apply background_to_region_mode to active regions (same as SD/SDXL)
        if kwargs.get('background') and kwargs.get('background_to_region_mode', 'concat') != 'none':
            bg_mode = kwargs.get('background_to_region_mode', 'concat')
            for r in active_regions:
                r['positive'] = COND_METHODS[bg_mode](r['positive'], kwargs['background'])
            log_debug(f"Applied background_to_region_mode={bg_mode} to {len(active_regions)} regions")
        
        # Apply base_prompt to active regions
        if kwargs.get('base_prompt'):
            base_method = kwargs.get('base_prompt_method', 'concat')
            for r in active_regions:
                r['positive'] = COND_METHODS[base_method](r['positive'], kwargs['base_prompt'])
        
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

    def _process_sd_sdxl(self, model, region, negative, arch_info, **kwargs) -> Tuple[Any, Any, Any]:
        """Process SD/SDXL model using original approach with LoRA Hook support"""
        
        # Remove duplicates
        region = self._remove_duplicate_regions(region)
        self._validate_mask_sizes_for_upscale(region)
        log_debug(f"Processing {len(region)} unique regions (SD/SDXL mode)")

        # Validate strengths
        bg_strength = validate_strength(kwargs.get("background_strength", 1.0), "Background", is_flux=False)
        prepared_regions = self._prepare_regions(region, kwargs, is_flux=False)
        
        # Get background
        bg_positive = kwargs.get("background")
        if bg_positive is None:
            log_debug("No background provided, creating zero conditioning")
            bg_positive = create_zero_conditioning(prepared_regions[0]["positive"])
        
        # Separate active and zero-strength regions
        active_regions, background_mask = self._process_strength_zero_regions(prepared_regions, bg_strength)
        
        if not active_regions:
            raise ValueError("All regions have zero strength")

        # ✅ Process conditioning with LoRA Hook support
        all_processed_conds = []
        
        # Process active regions
        for r in active_regions:
            # Step 1: Copy conditioning
            region_cond = copy.deepcopy(r['positive'])
            
            # Step 2: Apply background/base_prompt modifiers
            region_cond = self._apply_conditioning_modifiers(
                region_cond, kwargs, is_background=False
            )
            
            # Step 3: Apply mask metadata FIRST
            region_cond = ConditioningSetMask().append(
                region_cond, r['mask'], "default", r['strength']
            )[0]
            
            # Step 4: ✅ Attach LoRA hook AFTER ConditioningSetMask
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
            final_bg = self._apply_conditioning_modifiers(bg_positive, kwargs, is_background=True)
            final_bg = ConditioningSetMask().append(
                final_bg, background_mask, "default", bg_strength
            )[0]
            
            all_processed_conds.append({
                'conditioning': final_bg,
                'has_lora': False,  # ✅ Background doesn't have LoRA hook
                'mask': background_mask,
                'strength': bg_strength
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
        """Remove duplicate regions based on mask fingerprint"""
        unique_regions = []
        seen_masks = {}
        
        for r in region_list:
            mask = r["mask"]
            mask_flat = mask.flatten()
            mask_key = (
                tuple(mask.shape),
                float(mask.sum().item()),
                float(mask.var().item()),
                float(mask_flat[0].item()) if mask.numel() > 0 else 0.0,
                float(mask_flat[-1].item()) if mask.numel() > 0 else 0.0,
            )
            
            if mask_key in seen_masks:
                idx = seen_masks[mask_key]
                unique_regions[idx] = r
            else:
                seen_masks[mask_key] = len(unique_regions)
                unique_regions.append(r)
        
        if len(unique_regions) != len(region_list):
            log_warning(f"Removed {len(region_list) - len(unique_regions)} duplicate regions")
        
        return unique_regions

    def _prepare_regions(self, region_list: List[Dict], kwargs: Dict, is_flux: bool = False) -> List[Dict[str, Any]]:
        """Copy, validate, feather, and normalize regions"""
        feather_pixels = kwargs.get("feather_mask", 0)
        
        prepared_regions = []
        for i, r in enumerate(region_list):
            new_region = {
                "positive": r["positive"],
                "mask": r["mask"].clone(),
                "strength": validate_strength(r["strength"], f"Region {i}", is_flux=is_flux),
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

    def _process_strength_zero_regions(self, regions: List[Dict], bg_strength: float) -> Tuple[List[Dict], torch.Tensor]:
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

    def _apply_conditioning_modifiers(self, conditioning: Any, kwargs: Dict, 
                                     is_background: bool = False) -> Any:
        """Apply background and base prompt to conditioning"""
        modified_cond = copy.deepcopy(conditioning)
        
        if not is_background and kwargs.get("background") and kwargs.get("background_to_region_mode") != "none":
            bg_mode = kwargs.get("background_to_region_mode", "concat")
            modified_cond = COND_METHODS[bg_mode](modified_cond, kwargs["background"])

        if "base_prompt" in kwargs:
            base_method = kwargs.get("base_prompt_method", "concat")
            modified_cond = COND_METHODS[base_method](modified_cond, kwargs["base_prompt"])

        return modified_cond

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