"""
Region Extractor Node - Extract specific region(s) from coupled attention patch
"""

import torch
import copy
from typing import Any, Tuple, List, Dict, Optional

from .couple_utils import (
    log_debug, log_warning, MaskProcessor, MASK_EPSILON,
    detect_model_architecture, ArchitectureInfo, ModelType,
    create_zero_conditioning, get_flux_preset
)

from .couple_patching import AttentionPatcher, set_model_patch
from .couple_flux_patching import FluxAttentionPatcher, RegionalMask, RegionalConditioning


class ComfyCoupleRegionExtractor:
    """
    Extract specific region(s) from an existing coupled model
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coupled_model": ("MODEL", {
                    "tooltip": "Model with regional attention already applied"
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
                    "tooltip": "zero: Empty prompt | replicate: Copy extracted region(s) | original: Keep other regions | remove: No background"
                }),
            },
            "optional": {
                "extracted_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Prompt strength for extracted region(s) (0.0=ignore, 1.0=normal, >1.0=emphasize)"
                }),
                "background_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Prompt strength for background area (only used if background_mode ≠ remove)"
                }),
                "model_type": (["auto", "sd15", "sdxl", "flux"], {
                    "default": "auto",
                    "tooltip": "auto: Detect automatically | sd15/sdxl/flux: Force specific architecture"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("extracted_model", "matched_indices", "region_info")
    FUNCTION = "extract"
    CATEGORY = "ToyxyzTestNodes"

    def extract(self, coupled_model: Any, extraction_mask: torch.Tensor,
                extraction_mode: str = "single_best",
                threshold: float = 0.5, background_mode: str = "replicate",
                extracted_strength: float = 1.0,
                background_strength: float = 1.0,
                model_type: str = "auto") -> Tuple[Any, str, str]:
        """
        Extract specific region(s) from coupled model based on mask
        """
        # Detect architecture
        arch_info = self._detect_architecture(coupled_model, model_type)
        is_flux = arch_info.is_flux
        
        print(f"[RegionExtractor] Architecture: {arch_info.type.value.upper()}")
        print(f"[RegionExtractor] Extraction mode: {extraction_mode}")
        print(f"[RegionExtractor] Background: {background_mode}, "
              f"Extracted strength: {extracted_strength:.2f}, "
              f"Background strength: {background_strength:.2f}")
        
        # Extract region data from coupled model
        if is_flux:
            return self._extract_flux(
                coupled_model, extraction_mask, extraction_mode, threshold,
                background_mode, extracted_strength, background_strength
            )
        else:
            return self._extract_sd_sdxl(
                coupled_model, extraction_mask, extraction_mode, threshold,
                background_mode, extracted_strength, background_strength
            )

    def _detect_architecture(self, model: Any, model_type_str: str) -> ArchitectureInfo:
        """Detect or force model architecture"""
        if model_type_str == "auto":
            return detect_model_architecture(model)
        elif model_type_str == "flux":
            return ArchitectureInfo(type=ModelType.FLUX, use_cfg=True, dim=4096, is_flux=True)
        elif model_type_str == "sdxl":
            return ArchitectureInfo(type=ModelType.SDXL, use_cfg=True, dim=2048, is_flux=False)
        else:  # sd15
            return ArchitectureInfo(type=ModelType.SD15, use_cfg=True, dim=768, is_flux=False)

    def _extract_flux(self, coupled_model: Any, extraction_mask: torch.Tensor,
                     extraction_mode: str, threshold: float, background_mode: str,
                     extracted_strength: float, background_strength: float) -> Tuple[Any, str, str]:
        """Extract region(s) from Flux model - FULLY FIXED VERSION"""
        
        try:
            # Find regional mask in patches
            transformer_options = coupled_model.model_options.get("transformer_options", {})
            patches_replace = transformer_options.get("patches_replace", {})
            
            regional_mask_module = None
            found_block_type = None
            found_block_id = None
            
            for block_type in ["double", "single"]:
                if block_type in patches_replace:
                    block_patches_dict = patches_replace[block_type]
                    for block_id, patch_content in block_patches_dict.items():
                        # Try to find mask_fn in nested dict
                        if isinstance(patch_content, dict):
                            for key, module in patch_content.items():
                                if key == "mask_fn" or (isinstance(key, tuple) and key[0] == "mask_fn"):
                                    regional_mask_module = module
                                    found_block_type = block_type
                                    found_block_id = block_id
                                    break
                        # Or direct RegionalMask instance
                        elif isinstance(patch_content, RegionalMask):
                            regional_mask_module = patch_content
                            found_block_type = block_type
                            found_block_id = block_id
                            break
                    
                    if regional_mask_module is not None:
                        break
            
            if regional_mask_module is None or not isinstance(regional_mask_module, RegionalMask):
                raise ValueError("Model doesn't have Flux regional patching applied")
            
            print(f"[RegionExtractor] Found regional mask in {found_block_type} block ('{found_block_id}')")
            
            # Extract stored regions
            num_regions = regional_mask_module.num_regions
            region_masks = []
            region_conds = []
            
            for i in range(num_regions):
                mask = getattr(regional_mask_module, f'region_mask_{i}')
                cond = getattr(regional_mask_module, f'region_cond_{i}')
                region_masks.append(mask)
                region_conds.append(cond)
            
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
            
            # ✅ 핵심 수정: 리전 순서 (background 먼저, matched regions 나중에)
            extracted_regions = []
            
            # 1. Add background region FIRST (if needed) - INDEX 0
            if background_mode != "remove":
                matched_masks = [region_masks[i].float() for i in matched_indices]
                combined_matched = torch.stack(matched_masks).sum(dim=0).clamp(0, 1)
                background_mask = (1.0 - combined_matched).clamp(min=0.0)
                
                if background_mask.sum() > MASK_EPSILON:
                    unmatched_indices = [i for i in range(num_regions) if i not in matched_indices]
                    
                    # Choose background conditioning
                    if background_mode == "zero":
                        background_cond = torch.zeros_like(region_conds[matched_indices[0]])
                        log_debug("Background: zero conditioning")
                    
                    elif background_mode == "replicate":
                        background_cond = torch.cat([region_conds[i] for i in matched_indices], dim=1)
                        log_debug(f"Background: replicated from {len(matched_indices)} extracted region(s)")
                    
                    elif background_mode == "original":
                        if unmatched_indices:
                            other_conds = [region_conds[i] for i in unmatched_indices]
                            background_cond = torch.cat(other_conds, dim=1)
                            log_debug(f"Background: original ({len(unmatched_indices)} regions)")
                        else:
                            background_cond = torch.zeros_like(region_conds[matched_indices[0]])
                    
                    else:
                        background_cond = torch.zeros_like(region_conds[matched_indices[0]])
                    
                    # Add background as first region (index 0)
                    extracted_regions.append({
                        'mask': background_mask,
                        'positive': [(background_cond, {})],
                        'strength': background_strength
                    })
                    log_debug(f"Background mask coverage: {background_mask.sum().item():.1f} pixels")
            
            # 2. Add matched regions AFTER background (index 1, 2, ...)
            for idx in matched_indices:
                extracted_regions.append({
                    'mask': region_masks[idx].float(),
                    'positive': [(region_conds[idx], {})],
                    'strength': extracted_strength
                })
            
            # Restore original attention override blocks
            original_attn_override = {
                'double': [],
                'single': []
            }
            
            for block_type in ["double", "single"]:
                if block_type in patches_replace:
                    for block_id in patches_replace[block_type].keys():
                        # Extract numeric block ID
                        if isinstance(block_id, tuple):
                            numeric_id = block_id[1] if len(block_id) > 1 else block_id[0]
                        else:
                            numeric_id = block_id
                        
                        if isinstance(numeric_id, int):
                            original_attn_override[block_type].append(numeric_id)
            
            log_debug(f"Restored attn_override: {original_attn_override}")
            
            # ✅ 핵심 수정: Clean base model and create new patches
            base_model = coupled_model.clone()
            
            # Remove old regional_conditioning and patches_replace
            if 'transformer_options' in base_model.model_options:
                transformer_opts = base_model.model_options['transformer_options']
                
                # Remove regional_conditioning from patches
                if 'patches' in transformer_opts:
                    if 'regional_conditioning' in transformer_opts['patches']:
                        log_debug("Removing old regional_conditioning from base model")
                        del transformer_opts['patches']['regional_conditioning']
                
                # Remove patches_replace
                if 'patches_replace' in transformer_opts:
                    log_debug("Removing old patches_replace from base model")
                    del transformer_opts['patches_replace']
            
            # Create new FluxAttentionPatcher with extracted regions
            flux_patcher = FluxAttentionPatcher(
                model=base_model,
                regions=extracted_regions,
                start_percent=regional_mask_module.start_percent,
                end_percent=regional_mask_module.end_percent,
                attn_override=original_attn_override,
                apply_t5_background=regional_mask_module.apply_t5_background
            )
            
            new_model, _ = flux_patcher.patch()
            
            # Build output strings
            indices_str = ",".join(str(i) for i in matched_indices)
            
            # ✅ 수정: 올바른 카운트
            num_background = 1 if background_mode != "remove" else 0
            num_extracted = len(matched_indices)
            
            region_info = (f"Flux: Extracted {num_extracted} region(s) "
                          f"[{indices_str}]/{num_regions}, "
                          f"mode={extraction_mode}, "
                          f"extracted_strength={extracted_strength:.2f}")
            
            if background_mode != "remove":
                region_info += f", background={background_mode} (strength={background_strength:.2f})"
            
            print(f"[RegionExtractor] {region_info}")
            print(f"[RegionExtractor] Final regions: {len(extracted_regions)} "
                  f"({num_background} background + {num_extracted} extracted)")
            
            return (new_model, indices_str, region_info)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to extract Flux region: {e}")

    def _extract_sd_sdxl(self, coupled_model: Any, extraction_mask: torch.Tensor,
                        extraction_mode: str, threshold: float, background_mode: str,
                        extracted_strength: float, background_strength: float) -> Tuple[Any, str, str]:
        """Extract region(s) from SD/SDXL model"""
        
        try:
            transformer_options = coupled_model.model_options.get("transformer_options", {})
            patches_replace = transformer_options.get("patches_replace", {})
            attn2_patches = patches_replace.get("attn2", {})
            
            if not attn2_patches:
                raise ValueError("Model doesn't have regional attention patches applied")
            
            # Get first patch to extract region data
            first_patch_key = list(attn2_patches.keys())[0]
            first_patch = attn2_patches[first_patch_key]
            
            # Extract region data
            region_masks, region_conds, region_strengths = self._extract_regions_from_patch(first_patch)
            
            if not region_masks:
                raise ValueError("Could not extract region data from attention patches")
            
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
                # Calculate background mask (unmatched regions)
                unmatched_indices = [i for i in range(num_regions) if i not in matched_indices]
                
                if unmatched_indices:
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
                            # Average all extracted regions
                            background_cond = region_conds[matched_indices[0]].clone()
                            for idx in matched_indices[1:]:
                                if region_conds[idx].shape[1] == background_cond.shape[1]:
                                    background_cond = (background_cond + region_conds[idx]) / 2
                            log_debug(f"Background: replicated from {len(matched_indices)} region(s)")
                        
                        elif background_mode == "original":
                            # Average unmatched regions
                            background_cond = region_conds[unmatched_indices[0]]
                            for idx in unmatched_indices[1:]:
                                if region_conds[idx].shape[1] == background_cond.shape[1]:
                                    background_cond = (background_cond + region_conds[idx]) / 2
                            log_debug(f"Background: original ({len(unmatched_indices)} regions)")
                        
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

    def _extract_regions_from_patch(self, patch_function) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
        """Extract region data from attention patch closure"""
        try:
            closure = patch_function.__closure__
            if closure is None:
                return [], [], []
            
            closure_vars = {}
            for i, cell in enumerate(closure):
                try:
                    closure_vars[patch_function.__code__.co_freevars[i]] = cell.cell_contents
                except:
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

    def _create_patched_model(self, base_model: Any, extracted_conds: List,
                            original_transformer_options: Dict) -> Any:
        """Create new model with only extracted region patches"""
        new_model = base_model.clone()
        arch_info = detect_model_architecture(new_model)
        
        # Create dummy negative conditioning
        neg_cond = create_zero_conditioning([(extracted_conds[0][0], {})])
        
        # Create new patcher
        patcher = AttentionPatcher(
            model=new_model,
            positive=extracted_conds,
            negative=neg_cond,
            cross_region_attention=0.0,
            cross_region_mode="self_exclusion",
            model_type=arch_info.type.value
        )
        
        patched_model, _, _ = patcher.patch()
        
        return patched_model

