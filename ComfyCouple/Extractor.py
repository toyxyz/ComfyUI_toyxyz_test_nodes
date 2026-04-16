"""
Region Extractor Node - Extract specific region(s) from coupled attention patch
✅ EXTENDED: Now supports conditioning extraction with LoRA hooks
"""

import torch
import copy
from typing import Any, Tuple, List, Dict, Optional
import comfy.patcher_extension

from .couple_utils import (
    log_debug, log_warning, MaskProcessor, MASK_EPSILON,
    ArchitectureInfo, ModelType,
    create_zero_conditioning, detect_or_force_architecture,
    get_model_type_input_options,
    get_region_metadata,
)

from .couple_patching import AttentionPatcher, find_attn2_patch
from .couple_flux_patching import FluxAttentionPatcher, find_regional_mask_module
from .couple_anima_patching import AnimaAttentionPatcher


class ComfyCoupleRegionExtractor:
    """
    Extract specific region(s) from an existing coupled model
    ✅ NEW: Can also extract conditioning with LoRA hooks preserved
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coupled_model": ("MODEL", {
                    "tooltip": "Model with regional attention already applied. For Anima, the extracted model is the main output that carries the region selection."
                }),
                "extraction_mask": ("MASK", {
                    "tooltip": "Mask defining region(s) to extract (white=include, black=exclude)"
                }),
                "extraction_mode": (["single_best", "all_matching"], {
                    "default": "single_best",
                    "tooltip": "single_best: Extract only the best matching region | all_matching: Extract all regions above threshold"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum overlap % to consider a match (0.5 = 50% coverage required)"
                }),
                "background_mode": (["zero", "replicate", "original", "remove"], {
                    "default": "replicate",
                    "tooltip": "zero: Empty prompt | replicate: Copy extracted region(s) | original: Keep other regions | remove: No background. For Anima, this affects the extracted model patching rather than extracted conditioning."
                }),
            },
            "optional": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "✅ NEW: Optional conditioning to extract from (must be from ComfyCoupleMask with skip_positive=False). Extracts matching region's conditioning with LoRA hooks preserved."
                }),
                "extracted_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Prompt strength for extracted region(s) (0.0=ignore, 1.0=normal, >1.0=emphasize). For Anima, the extracted model carries the usable region result."
                }),
                "background_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Prompt strength for background area (only used if background_mode ≠ remove)"
                }),
                "model_type": (get_model_type_input_options(), {
                    "default": "auto",
                    "tooltip": "auto: Detect automatically | sd15/sdxl/flux/anima: Force specific architecture. Anima supports model extraction, but extracted_conditioning is intentionally disabled."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("extracted_model", "extracted_conditioning", "matched_indices", "region_info")
    FUNCTION = "extract"
    CATEGORY = "ToyxyzTestNodes"

    def extract(self, coupled_model: Any, extraction_mask: torch.Tensor,
                extraction_mode: str = "single_best",
                threshold: float = 0.5, background_mode: str = "replicate",
                conditioning: Optional[Any] = None,
                extracted_strength: float = 1.0,
                background_strength: float = 1.0,
                model_type: str = "auto") -> Tuple[Any, Any, str, str]:
        """
        Extract specific region(s) from coupled model based on mask
        ✅ NEW: Also extracts conditioning if provided
        """
        # Detect architecture
        arch_info = self._detect_architecture(coupled_model, model_type)
        is_flux = arch_info.is_flux
        
        display_name = arch_info.display_name or arch_info.type.value.upper()
        print(f"[RegionExtractor] Architecture: {display_name}")
        print(f"[RegionExtractor] Extraction mode: {extraction_mode}")
        print(f"[RegionExtractor] Background: {background_mode}, "
              f"Extracted strength: {extracted_strength:.2f}, "
              f"Background strength: {background_strength:.2f}")

        if not arch_info.supports("supports_extractor"):
            raise ValueError(f"{display_name} does not support Region Extractor")

        if conditioning is not None:
            print(f"[RegionExtractor] ✅ Conditioning extraction enabled")
        
        # Extract region data from coupled model
        if arch_info.type == ModelType.ANIMA:
            extracted_model, indices, info = self._extract_anima(
                coupled_model, extraction_mask, extraction_mode, threshold,
                background_mode, extracted_strength, background_strength
            )
        elif is_flux:
            extracted_model, indices, info = self._extract_flux(
                coupled_model, extraction_mask, extraction_mode, threshold,
                background_mode, extracted_strength, background_strength
            )
        else:
            extracted_model, indices, info = self._extract_sd_sdxl(
                coupled_model, extraction_mask, extraction_mode, threshold,
                background_mode, extracted_strength, background_strength
            )
        
        # ✅ NEW: Extract conditioning if provided
        extracted_conditioning = None
        if arch_info.type == ModelType.ANIMA:
            if conditioning is not None:
                extracted_conditioning = create_zero_conditioning(conditioning)
            else:
                extracted_conditioning = create_zero_conditioning([(torch.zeros(1, 77, arch_info.dim), {})])
            print("[RegionExtractor] Anima extracted_conditioning is disabled; outputting zero conditioning")
        elif conditioning is not None:
            try:
                extracted_conditioning, cond_info = self._extract_conditioning(
                    conditioning, extraction_mask, extraction_mode, threshold,
                    background_mode, extracted_strength, background_strength
                )
                print(f"[RegionExtractor] {cond_info}")
            except Exception as e:
                log_warning(f"Conditioning extraction failed: {e}")
                # Create dummy zero conditioning as fallback
                extracted_conditioning = create_zero_conditioning(conditioning)
        else:
            # No conditioning provided - create dummy with correct dimension
            extracted_conditioning = create_zero_conditioning([(torch.zeros(1, 77, arch_info.dim), {})])
        
        return (extracted_model, extracted_conditioning, indices, info)

    def _detect_architecture(self, model: Any, model_type_str: str) -> ArchitectureInfo:
        """Detect or force model architecture"""
        return detect_or_force_architecture(model, model_type_str)

    def _extract_conditioning(self, conditioning: Any, extraction_mask: torch.Tensor,
                            extraction_mode: str, threshold: float, background_mode: str,
                            extracted_strength: float, background_strength: float) -> Tuple[Any, str]:
        """
        ✅ NEW: Extract conditioning based on mask matching
        Preserves LoRA hooks and all metadata
        """
        extraction_mask = extraction_mask.float()
        if extraction_mask.dim() == 3:
            extraction_mask = extraction_mask[0]
        
        # Analyze conditioning list
        matches = []
        unmatched = []
        
        for idx, (cond_tensor, metadata) in enumerate(conditioning):
            region_mask = metadata.get('mask', None)
            
            if region_mask is None:
                # No mask = global conditioning, treat as always matching
                log_debug(f"Conditioning {idx}: No mask (global), treating as background")
                unmatched.append(idx)
                continue
            
            # Calculate coverage
            coverage = self._calculate_coverage(extraction_mask, region_mask)
            
            if coverage >= threshold:
                matches.append({
                    'index': idx,
                    'coverage': coverage,
                    'tensor': cond_tensor,
                    'metadata': copy.deepcopy(metadata),
                })
                log_debug(f"Conditioning {idx}: Matched with {coverage:.1%} coverage ✓")
            else:
                unmatched.append(idx)
                log_debug(f"Conditioning {idx}: Below threshold ({coverage:.1%} < {threshold:.1%})")
        
        if not matches:
            log_warning(f"No conditioning matched extraction mask with threshold {threshold:.0%}")
            # Return empty conditioning
            return create_zero_conditioning(conditioning[:1]), "No matches found"
        
        # Sort by coverage (highest first)
        matches.sort(key=lambda x: x['coverage'], reverse=True)
        
        # Apply extraction mode
        if extraction_mode == "single_best":
            selected_matches = [matches[0]]
            print(f"[CondExtractor] Single best: Region {matches[0]['index']} ({matches[0]['coverage']:.1%})")
        else:  # all_matching
            selected_matches = matches
            print(f"[CondExtractor] All matching: {len(matches)} regions")
            for m in matches:
                print(f"  Region {m['index']}: {m['coverage']:.1%} coverage")
        
        # Build output conditioning
        output_conds = []
        
        # Add matched regions with extracted_strength
        lora_count = 0
        for match in selected_matches:
            # Deep copy metadata to preserve LoRA hooks
            new_metadata = match['metadata']
            
            # Update strength
            new_metadata['mask_strength'] = extracted_strength
            
            # Track LoRA
            if new_metadata.get('hooks') is not None:
                lora_count += 1
            
            output_conds.append((match['tensor'], new_metadata))
        
        # Add background if needed
        if background_mode != "remove":
            background_mask = self._calculate_background_mask(
                extraction_mask, 
                [conditioning[m['index']][1].get('mask') for m in selected_matches]
            )
            
            if background_mask.sum() > MASK_EPSILON:
                if background_mode == "zero":
                    bg_tensor = torch.zeros_like(selected_matches[0]['tensor'])
                    # ✅ Copy metadata structure to preserve pooled_output and other required fields
                    bg_metadata = copy.deepcopy(selected_matches[0]['metadata'])
                    bg_metadata['mask'] = background_mask
                    bg_metadata['mask_strength'] = background_strength
                    # Remove hooks from zero background
                    bg_metadata.pop('hooks', None)
                
                elif background_mode == "replicate":
                    # Average matched tensors
                    bg_tensor = torch.stack([m['tensor'] for m in selected_matches]).mean(dim=0)
                    bg_metadata = copy.deepcopy(selected_matches[0]['metadata'])
                    bg_metadata['mask'] = background_mask
                    bg_metadata['mask_strength'] = background_strength
                    # Keep LoRA hook from first region
                
                elif background_mode == "original":
                    # Average unmatched regions
                    if unmatched:
                        unmatched_tensors = [conditioning[i][0] for i in unmatched]
                        # Pad to same length
                        max_len = max(t.shape[1] for t in unmatched_tensors)
                        padded = []
                        for t in unmatched_tensors:
                            if t.shape[1] < max_len:
                                pad = torch.zeros(t.shape[0], max_len - t.shape[1], t.shape[2],
                                                device=t.device, dtype=t.dtype)
                                t = torch.cat([t, pad], dim=1)
                            padded.append(t)
                        bg_tensor = torch.stack(padded).mean(dim=0)
                        # ✅ Use first unmatched region's metadata to preserve pooled_output
                        bg_metadata = copy.deepcopy(conditioning[unmatched[0]][1])
                        bg_metadata['mask'] = background_mask
                        bg_metadata['mask_strength'] = background_strength
                        # No LoRA for background from unmatched regions
                    else:
                        bg_tensor = torch.zeros_like(selected_matches[0]['tensor'])
                        # ✅ Copy metadata structure to preserve pooled_output
                        bg_metadata = copy.deepcopy(selected_matches[0]['metadata'])
                        bg_metadata['mask'] = background_mask
                        bg_metadata['mask_strength'] = background_strength
                        bg_metadata.pop('hooks', None)
                
                output_conds.append((bg_tensor, bg_metadata))
        
        # Build info string
        matched_indices = [m['index'] for m in selected_matches]
        indices_str = ",".join(str(i) for i in matched_indices)
        info = (f"Extracted {len(selected_matches)} conditioning(s) [{indices_str}]/{len(conditioning)}, "
                f"mode={extraction_mode}")
        
        if lora_count > 0:
            info += f", {lora_count} with LoRA hooks ✓"
        
        if background_mode != "remove":
            info += f", background={background_mode}"
        
        return (output_conds, info)

    def _extract_anima_conditioning(
        self,
        coupled_model: Any,
        extraction_mask: torch.Tensor,
        extraction_mode: str,
        threshold: float,
        background_mode: str,
        extracted_strength: float,
        background_strength: float,
    ) -> Tuple[Any, str]:
        """Build extracted conditioning for Anima directly from shared region metadata."""
        transformer_options = coupled_model.model_options.get("transformer_options", {})
        region_masks, region_conds, _, _ = self._extract_regions_from_metadata(transformer_options)

        if not region_masks or not region_conds:
            raise ValueError("No Anima region metadata available for conditioning extraction")

        matched_indices, match_scores = self._find_matching_regions(
            extraction_mask, region_masks, threshold
        )

        if not matched_indices:
            raise ValueError(f"No Anima conditioning matched extraction mask with threshold {threshold:.0%}")

        if extraction_mode == "single_best":
            matched_indices = [matched_indices[0]]
            match_scores = [match_scores[0]]
            print(f"[CondExtractor] Anima single best: Region {matched_indices[0]} ({match_scores[0]:.1%})")
        else:
            print(f"[CondExtractor] Anima all matching: {len(matched_indices)} regions")
            for idx, score in zip(matched_indices, match_scores):
                print(f"  Region {idx}: {score:.1%} coverage")

        output_conds = []

        if background_mode != "remove":
            unmatched_indices = [i for i in range(len(region_masks)) if i not in matched_indices]
            background_mask = self._calculate_background_mask(
                extraction_mask,
                [region_masks[idx] for idx in matched_indices],
            )

            if background_mask.sum() > MASK_EPSILON:
                if background_mode == "zero":
                    bg_tensor = torch.zeros_like(region_conds[matched_indices[0]])
                elif background_mode == "replicate":
                    bg_tensor = self._average_compatible_tensors(
                        [region_conds[idx] for idx in matched_indices],
                        region_conds[matched_indices[0]],
                    )
                elif background_mode == "original" and unmatched_indices:
                    bg_tensor = self._average_compatible_tensors(
                        [region_conds[idx] for idx in unmatched_indices],
                        region_conds[unmatched_indices[0]],
                    )
                else:
                    bg_tensor = torch.zeros_like(region_conds[matched_indices[0]])

                output_conds.append((
                    bg_tensor,
                    {"mask": background_mask, "mask_strength": background_strength},
                ))

        for idx in matched_indices:
            output_conds.append((
                region_conds[idx],
                {"mask": region_masks[idx], "mask_strength": extracted_strength},
            ))

        indices_str = ",".join(str(i) for i in matched_indices)
        info = (f"Extracted {len(matched_indices)} Anima conditioning(s) "
                f"[{indices_str}]/{len(region_conds)}, mode={extraction_mode}")

        if background_mode != "remove":
            info += f", background={background_mode}"

        return (output_conds, info)

    def _calculate_coverage(self, extraction_mask: torch.Tensor, 
                          region_mask: torch.Tensor) -> float:
        """Calculate overlap coverage between extraction and region mask"""
        region_mask = region_mask.float()
        
        if region_mask.dim() == 3:
            region_mask = region_mask[0]
        
        # Resize if needed
        if region_mask.shape != extraction_mask.shape:
            region_mask = MaskProcessor.resize_to_match(region_mask, extraction_mask.shape)
            region_mask = region_mask.to(device=extraction_mask.device, dtype=extraction_mask.dtype)
        
        # Calculate coverage
        intersection = (extraction_mask * region_mask).sum()
        extraction_area = extraction_mask.sum()
        
        if extraction_area < MASK_EPSILON:
            return 0.0
        
        coverage = (intersection / extraction_area).item()
        return coverage

    def _calculate_background_mask(self, extraction_mask: torch.Tensor,
                                   matched_masks: List[torch.Tensor]) -> torch.Tensor:
        """Calculate background mask (areas not covered by matched regions)"""
        combined_mask = torch.zeros_like(extraction_mask)
        
        for mask in matched_masks:
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask[0]
                if mask.shape != extraction_mask.shape:
                    mask = MaskProcessor.resize_to_match(mask, extraction_mask.shape)
                    mask = mask.to(device=extraction_mask.device, dtype=extraction_mask.dtype)
                combined_mask = torch.maximum(combined_mask, mask)
        
        background_mask = (1.0 - combined_mask).clamp(min=0.0)
        return background_mask

    def _extract_flux(self, coupled_model: Any, extraction_mask: torch.Tensor,
                     extraction_mode: str, threshold: float, background_mode: str,
                     extracted_strength: float, background_strength: float) -> Tuple[Any, str, str]:
        """Extract region(s) from Flux model - FULLY FIXED VERSION"""
        
        try:
            transformer_options = coupled_model.model_options.get("transformer_options", {})
            region_masks, region_conds, _, metadata = self._extract_regions_from_metadata(transformer_options)
            if region_masks and region_conds:
                num_regions = len(region_masks)
                attn_override = metadata.get("attn_override")
                start_percent = metadata.get("start_percent", 0.0)
                end_percent = metadata.get("end_percent", 0.5)
                apply_t5_background = metadata.get("apply_t5_background", True)

                print("[RegionExtractor] Found Flux regional metadata in transformer_options")
            else:
                regional_mask_module, found_block_type, found_block_id = find_regional_mask_module(transformer_options)
                if regional_mask_module is None:
                    raise ValueError("Model doesn't have Flux regional patching applied")
                
                print(f"[RegionExtractor] Found regional mask in {found_block_type} block ('{found_block_id}')")
                
                num_regions = regional_mask_module.num_regions
                region_masks = []
                region_conds = []
                
                for i in range(num_regions):
                    mask = getattr(regional_mask_module, f'region_mask_{i}')
                    cond = getattr(regional_mask_module, f'region_cond_{i}')
                    region_masks.append(mask)
                    region_conds.append(cond)

                attn_override = None
                start_percent = regional_mask_module.start_percent
                end_percent = regional_mask_module.end_percent
                apply_t5_background = regional_mask_module.apply_t5_background

                log_debug(f"Found {num_regions} regions in Flux model")
            
            # Find matching region(s)
            matched_indices, match_scores = self._find_matching_regions(
                extraction_mask, region_masks, threshold
            )
            
            if not matched_indices:
                raise ValueError(f"No regions match extraction mask with threshold {threshold:.0%}")
            
            # Filter based on extraction_mode
            if extraction_mode == "single_best":
                matched_indices = [matched_indices[0]]
                match_scores = [match_scores[0]]
                print(f"[RegionExtractor] Single best: Region {matched_indices[0]} ({match_scores[0]:.1%})")
            else:
                print(f"[RegionExtractor] All matching: {len(matched_indices)} regions")
                for idx, score in zip(matched_indices, match_scores):
                    print(f"  Region {idx}: {score:.1%} coverage")
            
            # Create new region list for FluxAttentionPatcher
            new_regions = []
            
            if background_mode == "remove":
                # Only matched regions, no background
                for idx in matched_indices:
                    new_regions.append({
                        'positive': [(region_conds[idx], {})],
                        'mask': region_masks[idx],
                        'strength': extracted_strength
                    })
            
            else:
                # Calculate background mask
                unmatched_indices = [i for i in range(num_regions) if i not in matched_indices]
                
                # Create union of matched masks
                background_mask = torch.ones_like(region_masks[0])
                for idx in matched_indices:
                    background_mask = background_mask - region_masks[idx]
                background_mask = background_mask.clamp(min=0)
                
                if background_mask.sum() > MASK_EPSILON:
                    # Choose background conditioning
                    if background_mode == "zero":
                        background_cond = torch.zeros_like(region_conds[matched_indices[0]])
                        log_debug("Background: zero conditioning")
                    
                    elif background_mode == "replicate":
                        # Average all extracted regions (equal weight)
                        ref_shape = region_conds[matched_indices[0]].shape[1]
                        compatible = [region_conds[i] for i in matched_indices if region_conds[i].shape[1] == ref_shape]
                        background_cond = torch.stack(compatible).mean(dim=0)
                        log_debug(f"Background: replicated from {len(compatible)} region(s)")
                    
                    elif background_mode == "original":
                        # Average unmatched regions (equal weight)
                        if unmatched_indices:
                            ref_shape = region_conds[unmatched_indices[0]].shape[1]
                            compatible = [region_conds[i] for i in unmatched_indices if region_conds[i].shape[1] == ref_shape]
                            background_cond = torch.stack(compatible).mean(dim=0)
                            log_debug(f"Background: original ({len(compatible)} regions)")
                        else:
                            background_cond = torch.zeros_like(region_conds[matched_indices[0]])
                    
                    else:
                        background_cond = torch.zeros_like(region_conds[matched_indices[0]])
                    
                    # Add background region first
                    new_regions.append({
                        'positive': [(background_cond, {})],
                        'mask': background_mask,
                        'strength': background_strength
                    })
                
                # Add matched regions
                for idx in matched_indices:
                    new_regions.append({
                        'positive': [(region_conds[idx], {})],
                        'mask': region_masks[idx],
                        'strength': extracted_strength
                    })
            
            # Create new Flux patched model
            patcher = FluxAttentionPatcher(
                coupled_model, 
                new_regions,
                start_percent=start_percent,
                end_percent=end_percent,
                attn_override=attn_override,
                apply_t5_background=apply_t5_background,
            )
            
            new_model, _ = patcher.patch()
            
            # Build output strings
            indices_str = ",".join(str(i) for i in matched_indices)
            region_info = (f"Flux: Extracted {len(matched_indices)} region(s) "
                          f"[{indices_str}]/{num_regions}, "
                          f"mode={extraction_mode}, "
                          f"extracted_strength={extracted_strength:.2f}")
            
            if background_mode != "remove":
                region_info += f", background={background_mode} (strength={background_strength:.2f})"
            
            print(f"[RegionExtractor] {region_info}")
            
            return (new_model, indices_str, region_info)
        
        except Exception as e:
            raise RuntimeError(f"Failed to extract Flux region: {e}")

    def _extract_anima(self, coupled_model: Any, extraction_mask: torch.Tensor,
                      extraction_mode: str, threshold: float, background_mode: str,
                      extracted_strength: float, background_strength: float) -> Tuple[Any, str, str]:
        """Extract region(s) from Anima model using shared metadata."""

        try:
            transformer_options = coupled_model.model_options.get("transformer_options", {})
            region_masks, region_conds, region_strengths, metadata = self._extract_regions_from_metadata(transformer_options)
            region_conditioning_objects = list(metadata.get("region_conditioning_objects", []))

            if not region_masks or not region_conds:
                raise ValueError("Model doesn't have Anima regional metadata available")

            num_regions = len(region_masks)
            log_debug(f"Found {num_regions} regions in Anima model")

            matched_indices, match_scores = self._find_matching_regions(
                extraction_mask, region_masks, threshold
            )

            if not matched_indices:
                raise ValueError(f"No regions match extraction mask with threshold {threshold:.0%}")

            if extraction_mode == "single_best":
                matched_indices = [matched_indices[0]]
                match_scores = [match_scores[0]]
                print(f"[RegionExtractor] Single best: Region {matched_indices[0]} ({match_scores[0]:.1%})")
            else:
                print(f"[RegionExtractor] All matching: {len(matched_indices)} regions")
                for idx, score in zip(matched_indices, match_scores):
                    print(f"  Region {idx}: {score:.1%} coverage")

            extracted_regions = []

            if background_mode != "remove":
                unmatched_indices = [i for i in range(num_regions) if i not in matched_indices]

                background_mask = torch.ones_like(region_masks[0])
                for idx in matched_indices:
                    background_mask = background_mask - region_masks[idx]
                background_mask = background_mask.clamp(min=0)

                if background_mask.sum() > MASK_EPSILON:
                    if background_mode == "zero":
                        background_cond = torch.zeros_like(region_conds[matched_indices[0]])
                        background_region = {
                            "context_tensor": background_cond,
                            "mask": background_mask,
                            "strength": background_strength,
                            "is_background": True,
                        }
                        log_debug("Anima background: zero conditioning")
                    elif background_mode == "replicate":
                        if matched_indices[0] < len(region_conditioning_objects) and region_conditioning_objects[matched_indices[0]] is not None:
                            background_region = {
                                "conditioning": copy.deepcopy(region_conditioning_objects[matched_indices[0]]),
                                "mask": background_mask,
                                "strength": background_strength,
                                "is_background": True,
                            }
                        else:
                            background_cond = self._average_compatible_tensors(
                                [region_conds[i] for i in matched_indices],
                                region_conds[matched_indices[0]],
                            )
                            background_region = {
                                "context_tensor": background_cond,
                                "mask": background_mask,
                                "strength": background_strength,
                                "is_background": True,
                            }
                        log_debug(f"Anima background: replicated from {len(matched_indices)} region(s)")
                    elif background_mode == "original" and unmatched_indices:
                        if unmatched_indices[0] < len(region_conditioning_objects) and region_conditioning_objects[unmatched_indices[0]] is not None:
                            background_region = {
                                "conditioning": copy.deepcopy(region_conditioning_objects[unmatched_indices[0]]),
                                "mask": background_mask,
                                "strength": background_strength,
                                "is_background": True,
                            }
                        else:
                            background_cond = self._average_compatible_tensors(
                                [region_conds[i] for i in unmatched_indices],
                                region_conds[unmatched_indices[0]],
                            )
                            background_region = {
                                "context_tensor": background_cond,
                                "mask": background_mask,
                                "strength": background_strength,
                                "is_background": True,
                            }
                        log_debug(f"Anima background: original ({len(unmatched_indices)} region(s))")
                    else:
                        background_cond = torch.zeros_like(region_conds[matched_indices[0]])
                        background_region = {
                            "context_tensor": background_cond,
                            "mask": background_mask,
                            "strength": background_strength,
                            "is_background": True,
                        }

                    extracted_regions.append(background_region)

            for idx in matched_indices:
                strength = region_strengths[idx] if idx < len(region_strengths) else 1.0
                strength = extracted_strength if extracted_strength is not None else strength
                if idx < len(region_conditioning_objects) and region_conditioning_objects[idx] is not None:
                    extracted_regions.append({
                        "conditioning": copy.deepcopy(region_conditioning_objects[idx]),
                        "mask": region_masks[idx],
                        "strength": strength,
                        "is_background": False,
                    })
                else:
                    extracted_regions.append({
                        "context_tensor": region_conds[idx],
                        "mask": region_masks[idx],
                        "strength": strength,
                        "is_background": False,
                    })

            new_model = self._create_anima_patched_model(coupled_model, extracted_regions)

            indices_str = ",".join(str(i) for i in matched_indices)
            region_info = (f"Anima: Extracted {len(matched_indices)} region(s) "
                          f"[{indices_str}]/{num_regions}, "
                          f"mode={extraction_mode}, "
                          f"extracted_strength={extracted_strength:.2f}")

            if background_mode != "remove":
                region_info += f", background={background_mode} (strength={background_strength:.2f})"

            print(f"[RegionExtractor] {region_info}")

            return (new_model, indices_str, region_info)

        except Exception as e:
            raise RuntimeError(f"Failed to extract Anima region: {e}")

    def _extract_sd_sdxl(self, coupled_model: Any, extraction_mask: torch.Tensor,
                        extraction_mode: str, threshold: float, background_mode: str,
                        extracted_strength: float, background_strength: float) -> Tuple[Any, str, str]:
        """Extract region(s) from SD/SDXL model"""
        
        try:
            # Extract region data from attention patches
            transformer_options = coupled_model.model_options.get("transformer_options", {})
            region_masks, region_conds, region_strengths, metadata = self._extract_regions_from_metadata(transformer_options)

            if not region_masks:
                first_patch = find_attn2_patch(transformer_options)
                if first_patch is None:
                    raise ValueError("Model doesn't have SD/SDXL regional patching applied")

                region_masks, region_conds, region_strengths = self._extract_regions_from_patch(coupled_model, first_patch)
            
            if not region_masks:
                raise ValueError("No region data found in coupled model")
            
            num_regions = len(region_masks)
            log_debug(f"Found {num_regions} regions in SD/SDXL model")
            
            # Find matching region(s)
            matched_indices, match_scores = self._find_matching_regions(
                extraction_mask, region_masks, threshold
            )
            
            if not matched_indices:
                raise ValueError(f"No regions match extraction mask with threshold {threshold:.0%}")
            
            # Filter based on extraction_mode
            if extraction_mode == "single_best":
                matched_indices = [matched_indices[0]]
                match_scores = [match_scores[0]]
                print(f"[RegionExtractor] Single best: Region {matched_indices[0]} ({match_scores[0]:.1%})")
            else:
                print(f"[RegionExtractor] All matching: {len(matched_indices)} regions")
                for idx, score in zip(matched_indices, match_scores):
                    print(f"  Region {idx}: {score:.1%} coverage")
            
            # Create new conditioning list
            extracted_conds = []
            
            if background_mode != "remove":
                # Calculate background mask (areas not covered by matched regions)
                unmatched_indices = [i for i in range(num_regions) if i not in matched_indices]
                
                background_mask = torch.ones_like(region_masks[0])
                for idx in matched_indices:
                    background_mask = background_mask - region_masks[idx]
                background_mask = background_mask.clamp(min=0)
                
                if background_mask.sum() > MASK_EPSILON:
                    # Choose background conditioning
                    if background_mode == "zero":
                        background_cond = torch.zeros_like(region_conds[matched_indices[0]])
                        log_debug("Background: zero conditioning")
                    
                    elif background_mode == "replicate":
                        # Average all extracted regions (equal weight)
                        ref_shape = region_conds[matched_indices[0]].shape[1]
                        compatible = [region_conds[i] for i in matched_indices if region_conds[i].shape[1] == ref_shape]
                        background_cond = torch.stack(compatible).mean(dim=0)
                        log_debug(f"Background: replicated from {len(compatible)} region(s)")
                    
                    elif background_mode == "original":
                        # Average unmatched regions (equal weight)
                        if unmatched_indices:
                            ref_shape = region_conds[unmatched_indices[0]].shape[1]
                            compatible = [region_conds[i] for i in unmatched_indices if region_conds[i].shape[1] == ref_shape]
                            background_cond = torch.stack(compatible).mean(dim=0)
                            log_debug(f"Background: original ({len(compatible)} regions)")
                        else:
                            background_cond = torch.zeros_like(region_conds[matched_indices[0]])
                    
                    else:
                        background_cond = torch.zeros_like(region_conds[matched_indices[0]])
                    
                    extracted_conds.append([
                        background_cond,
                        {"mask": background_mask, "mask_strength": background_strength}
                    ])
            
            # Add all matched regions with custom strength
            for idx in matched_indices:
                extracted_conds.append([
                    region_conds[idx],
                    {"mask": region_masks[idx], "mask_strength": extracted_strength}
                ])
            
            # Create new patched model
            new_model = self._create_patched_model(
                coupled_model, extracted_conds, transformer_options
            )
            
            # Build output strings
            indices_str = ",".join(str(i) for i in matched_indices)
            region_info = (f"SD/SDXL: Extracted {len(matched_indices)} region(s) "
                          f"[{indices_str}]/{num_regions}, "
                          f"mode={extraction_mode}, "
                          f"extracted_strength={extracted_strength:.2f}")
            
            if background_mode != "remove":
                region_info += f", background={background_mode} (strength={background_strength:.2f})"
            
            print(f"[RegionExtractor] {region_info}")
            
            return (new_model, indices_str, region_info)
        
        except Exception as e:
            raise RuntimeError(f"Failed to extract SD/SDXL region: {e}")

    def _find_matching_regions(self, extraction_mask: torch.Tensor,
                              region_masks: List[torch.Tensor],
                              threshold: float) -> Tuple[List[int], List[float]]:
        """
        Find all regions that match the extraction mask above threshold
        
        Returns:
            matched_indices: List of region indices (sorted by score, highest first)
            match_scores: Corresponding coverage scores
        """
        extraction_mask = extraction_mask.float()
        
        if extraction_mask.dim() == 3:
            extraction_mask = extraction_mask[0]
        
        matches = []  # [(index, score), ...]
        
        for idx, region_mask in enumerate(region_masks):
            region_mask = region_mask.to(device=extraction_mask.device, dtype=extraction_mask.dtype)
            
            # Resize if needed
            if region_mask.shape != extraction_mask.shape:
                region_mask = MaskProcessor.resize_to_match(region_mask, extraction_mask.shape)
                region_mask = region_mask.to(device=extraction_mask.device, dtype=extraction_mask.dtype)
            
            # Calculate overlap metrics
            intersection = (extraction_mask * region_mask).sum()
            union = ((extraction_mask + region_mask) > 0).float().sum()
            
            if union > MASK_EPSILON and extraction_mask.sum() > 0:
                iou = (intersection / union).item()
                coverage = (intersection / extraction_mask.sum()).item()
                
                # Use coverage as primary metric
                score = coverage
                log_debug(f"Region {idx}: IoU={iou:.3f}, Coverage={coverage:.3f}")
                
                # Add to matches if above threshold
                if score >= threshold:
                    matches.append((idx, score))
                    print(f"[RegionExtractor] Region {idx} matched: {score:.1%} coverage ✓")
                else:
                    log_debug(f"Region {idx} below threshold: {score:.1%} < {threshold:.1%}")
        
        if not matches:
            log_warning(f"No regions matched with threshold {threshold:.0%}")
            return [], []
        
        # Sort by score (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        matched_indices = [m[0] for m in matches]
        match_scores = [m[1] for m in matches]
        
        if len(matches) > 1:
            print(f"[RegionExtractor] Found {len(matches)} matching regions "
                  f"(best: Region {matched_indices[0]} at {match_scores[0]:.1%})")
        
        return matched_indices, match_scores

    def _extract_regions_from_metadata(
        self, transformer_options: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], Dict[str, Any]]:
        metadata = get_region_metadata(transformer_options)
        if metadata is None:
            return [], [], [], {}

        region_masks = [
            mask for mask in metadata.get("region_masks", [])
            if isinstance(mask, torch.Tensor)
        ]
        region_conds = [
            cond for cond in metadata.get("region_conditioning", [])
            if isinstance(cond, torch.Tensor)
        ]
        region_strengths = list(metadata.get("region_strengths", []))

        if region_masks:
            log_debug("Extracted region data from shared metadata")

        return region_masks, region_conds, region_strengths, metadata

    def _extract_regions_from_patch(self, model: Any, patch_function: Any) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
        """Extract region data using robust transformer_options or fallback to closure"""
        try:
            pos_masks = []
            padded_pos_conds = []
            pos_strengths = []
            
            transformer_options = model.model_options.get("transformer_options", {})
            regions_data = get_region_metadata(transformer_options)
            
            if regions_data is not None:
                pos_masks = regions_data.get("region_masks", [])
                padded_pos_conds = regions_data.get("region_conditioning", [])
                pos_strengths = regions_data.get("region_strengths", [])
                log_debug("Extracted region data from metadata successfully")
            else:
                # Fallback to fragile closure extraction (for backward compatibility)
                log_warning("comfycouple_regions not found in transformer_options, falling back to closure.")
                closure = patch_function.__closure__
                if closure is not None:
                    closure_vars = {}
                    for i, cell in enumerate(closure):
                        try:
                            closure_vars[patch_function.__code__.co_freevars[i]] = cell.cell_contents
                        except Exception as e:
                            pass
                    
                    pos_masks = closure_vars.get('pos_masks', [])
                    padded_pos_conds = closure_vars.get('padded_pos_conds', [])
                    pos_strengths = closure_vars.get('pos_strengths', [])

            if not pos_masks or not padded_pos_conds:
                return [], [], []
            
            region_masks = []
            region_conds = []
            region_strengths = []
            
            for i, mask in enumerate(pos_masks):
                if isinstance(mask, torch.Tensor):
                    region_masks.append(mask)
                    if i < len(padded_pos_conds):
                        region_conds.append(padded_pos_conds[i])
                    if i < len(pos_strengths):
                        region_strengths.append(pos_strengths[i])
            
            log_debug(f"Extracted {len(region_masks)} regions from patch closure")
            
            return region_masks, region_conds, region_strengths
        
        except Exception as e:
            log_warning(f"Failed to extract regions from patch: {e}")
            return [], [], []

    def _average_compatible_tensors(self, tensors: List[torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
        """Average tensors that share the same token length, fallback to reference shape when needed."""
        if not tensors:
            return torch.zeros_like(fallback)

        ref_shape = tensors[0].shape[1]
        compatible = [tensor for tensor in tensors if tensor.shape[1] == ref_shape]
        if not compatible:
            return torch.zeros_like(fallback)

        return torch.stack(compatible).mean(dim=0)

    def _create_anima_patched_model(self, base_model: Any, extracted_regions: List[Dict[str, Any]]) -> Any:
        """Create a new Anima patched model from extracted region contexts."""
        if not extracted_regions:
            raise ValueError("No extracted Anima regions available")

        new_model = base_model.clone()
        if hasattr(new_model, "remove_wrappers_with_key"):
            new_model.remove_wrappers_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                AnimaAttentionPatcher.WRAPPER_KEY,
            )
        transformer_options = new_model.model_options.setdefault("transformer_options", {})
        wrappers = transformer_options.get("wrappers", {})
        diffusion_wrappers = wrappers.get("diffusion_model", {})
        if AnimaAttentionPatcher.WRAPPER_KEY in diffusion_wrappers:
            diffusion_wrappers.pop(AnimaAttentionPatcher.WRAPPER_KEY, None)
        if not diffusion_wrappers and "diffusion_model" in wrappers:
            wrappers.pop("diffusion_model", None)

        patcher = AnimaAttentionPatcher(
            model=new_model,
            regions=extracted_regions,
        )
        return patcher.patch()

    def _create_patched_model(self, base_model: Any, extracted_conds: List,
                            original_transformer_options: Dict) -> Any:
        """Create new model with only extracted region patches"""
        new_model = base_model.clone()
        metadata = get_region_metadata(original_transformer_options) or {}
        model_type = metadata.get("model_type") or "auto"
        debug_handles = metadata.get("debug_handles", {})
        arch_info = detect_or_force_architecture(new_model, model_type)
        
        # Create dummy negative conditioning
        neg_cond = create_zero_conditioning([(extracted_conds[0][0], {})])
        
        # Create new patcher
        patcher = AttentionPatcher(
            model=new_model,
            positive=extracted_conds,
            negative=neg_cond,
            cross_region_attention=debug_handles.get("cross_region_attention", 0.0),
            cross_region_mode=debug_handles.get("cross_region_mode", "self_exclusion"),
            model_type=arch_info.type.value
        )
        
        patched_model, _, _ = patcher.patch()
        
        return patched_model
