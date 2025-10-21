"""
Attention patching logic for regional prompting - OPTIMIZED
"""
import torch
import copy
from typing import Dict, List, Tuple, Any

import comfy
from comfy.ldm.modules.attention import optimized_attention

from .couple_utils import (
    ModelType, ArchitectureInfo, ARCH_CONFIGS,
    log_debug, detect_model_architecture, MaskProcessor,
    pad_conditioning_tensors, ensure_dtype_device
)

def set_model_patch(model: Any, patch: Any, key: Any, attn_type: str = 'attn2') -> None:
    """Register custom attention patch to model"""
    transformer_options = model.model_options.setdefault("transformer_options", {})
    patches_replace = transformer_options.setdefault("patches_replace", {})
    attn_patches = patches_replace.setdefault(attn_type, {})
    attn_patches[key] = patch

class AttentionPatcher:
    """Manages attention coupling for regional prompting"""

    def __init__(self, model: Any, positive: Any, negative: Any, 
                 cross_region_attention: float, cross_region_mode: str, model_type: str):
        self.model = model
        self.positive_conds = copy.deepcopy(positive)
        self.negative_conds = copy.deepcopy(negative)
        self.cross_region_attention = cross_region_attention
        self.cross_region_mode = cross_region_mode
        self.manual_model_type = model_type

    def patch(self) -> Tuple[Any, Any, Any]:
        """Apply attention coupling based on model architecture"""
        self.arch_info = self._get_architecture_info()
        print(f"[ComfyCouple] Model: {self.arch_info.type.value.upper()}, Dim: {self.arch_info.dim}, Cross-Attention applied")
        return self._apply_patch()

    def _get_architecture_info(self) -> ArchitectureInfo:
        if self.manual_model_type == "auto":
            arch_info = detect_model_architecture(self.model)
            print(f"[ComfyCouple] Auto-detected: {arch_info.type.value.upper()}")
        else:
            model_type_enum = ModelType.SDXL if self.manual_model_type == "sdxl" else ModelType.SD15
            dim = 2048 if model_type_enum == ModelType.SDXL else 768
            arch_info = ArchitectureInfo(type=model_type_enum, use_cfg=True, dim=dim)
            print(f"[ComfyCouple] Manual: {self.manual_model_type.upper()}")
        return arch_info

    def _apply_patch(self) -> Tuple[Any, Any, Any]:
        new_model = self.model.clone()
        
        dtype = new_model.model.diffusion_model.dtype
        device = comfy.model_management.get_torch_device()
        
        self._prepare_conditioning(dtype, device)
        self._patch_model_blocks(new_model)
        
        return new_model, self.positive_conds, self.negative_conds

    def _prepare_conditioning(self, dtype: torch.dtype, device: torch.device) -> None:
        """Extract and preprocess conditioning data"""
        self.pos_masks, self.pos_cond_tensors, self.pos_strengths = self._extract_cond_parts(
            self.positive_conds, dtype, device)
        self.neg_masks, self.neg_cond_tensors, self.neg_strengths = self._extract_cond_parts(
            self.negative_conds, dtype, device)

        self.padded_pos_conds = pad_conditioning_tensors(self.pos_cond_tensors)
        self.padded_neg_conds = pad_conditioning_tensors(self.neg_cond_tensors) if self.arch_info.use_cfg else []

    def _extract_cond_parts(self, conditions: List[Any], dtype: torch.dtype, device: torch.device) -> Tuple[List[Any], List[torch.Tensor], List[float]]:
        """Separate masks, tensors, and strengths from conditioning list"""
        if not conditions:
            # 아키텍처에 맞는 차원 사용 (SD 1.5: 768, SDXL: 2048)
            return [False], [torch.zeros(1, 77, self.arch_info.dim, device=device, dtype=dtype)], [1.0]

        masks = [c[1].get("mask", False) for c in conditions]
        conds = [c[0].to(device, dtype=dtype) for c in conditions]
        strengths = [c[1].get("mask_strength", 1.0) for c in conditions]
        
        # NOTE: Mask normalization already done in comfy_couple.py
        # No need to normalize again here
            
        return masks, conds, strengths

    def _patch_model_blocks(self, model: Any) -> None:
        """Apply patches to attention blocks"""
        diff_model = model.model.diffusion_model
        config = ARCH_CONFIGS[self.arch_info.type.value]
        is_sdxl = self.arch_info.type == ModelType.SDXL
        
        def patch_block(block_type, block_ids, indices_config=None):
            for block_id in block_ids:
                indices = range(indices_config[block_id]) if is_sdxl and indices_config else [0]
                for idx in indices:
                    try:
                        attn_module = diff_model.__getattr__(f'{block_type}_blocks')[block_id][1].transformer_blocks[idx].attn2
                        key = (block_type, block_id, idx) if is_sdxl else (block_type, block_id)
                        set_model_patch(model, self._make_sd_patch(attn_module), key)
                        log_debug(f"Patched {block_type} block {block_id}, idx {idx}")
                    except (AttributeError, IndexError) as e:
                        log_debug(f"Skip {block_type} block {block_id}, idx {idx}: {e}")
                        pass  # Skip non-existent blocks silently

        patch_block('input', config['input_blocks'], config.get('input_indices'))
        
        for idx in range(config['middle_count']):
            try:
                attn = diff_model.middle_block[1].transformer_blocks[idx].attn2
                key = ("middle", 0, idx) if is_sdxl else ("middle", 0)
                set_model_patch(model, self._make_sd_patch(attn), key)
                log_debug(f"Patched middle block idx {idx}")
            except (AttributeError, IndexError) as e:
                log_debug(f"Skip middle block idx {idx}: {e}")
                pass

        patch_block('output', config['output_blocks'], config.get('output_indices'))

    def _make_sd_patch(self, original_attn_module: Any):
        """Create attention patch with captured parameters"""
        # Capture current values in immutable form to avoid reference issues
        pos_masks = self.pos_masks[:]  # Shallow copy of list
        neg_masks = self.neg_masks[:]
        padded_pos_conds = self.padded_pos_conds[:]
        padded_neg_conds = self.padded_neg_conds[:]
        pos_strengths = tuple(self.pos_strengths)  # Immutable tuple
        neg_strengths = tuple(self.neg_strengths)
        cross_region_attention = self.cross_region_attention
        cross_region_mode = self.cross_region_mode
        
        def attention_patch(q, k, v, extra_options):
            return self._process_attention(
                q, k, v, extra_options, original_attn_module,
                pos_masks, neg_masks,
                padded_pos_conds, padded_neg_conds,
                pos_strengths, neg_strengths,
                cross_region_attention, cross_region_mode
            )
        return attention_patch

    def _process_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          extra_options: Dict, module: Any,
                          pos_masks: List, neg_masks: List,
                          padded_pos_conds: List, padded_neg_conds: List,
                          pos_strengths: List[float], neg_strengths: List[float],
                          cross_region_attention: float, cross_region_mode: str) -> torch.Tensor:
        """Process attention with CFG support"""
        q_batches = q.chunk(len(extra_options["cond_or_uncond"]), dim=0)
        batch_size, seq_len, _ = q_batches[0].shape
        orig_shape = extra_options["original_shape"]
        
        # Calculate masks for current resolution (no caching for immediate strength updates)
        masks_uncond = MaskProcessor.from_query(neg_masks, q_batches[0], orig_shape)
        masks_cond = MaskProcessor.from_query(pos_masks, q_batches[0], orig_shape)
        
        k_uncond, v_uncond = self._compute_kv(padded_neg_conds, module, q)
        k_cond, v_cond = self._compute_kv(padded_pos_conds, module, q)
        
        outputs = []
        for i, cond_type in enumerate(extra_options["cond_or_uncond"]):
            masks, k_t, v_t, strengths = (
                (masks_uncond, k_uncond, v_uncond, neg_strengths) if cond_type == 1 
                else (masks_cond, k_cond, v_cond, pos_strengths)
            )
            num_regions = len(strengths)

            q_expanded = q_batches[i].repeat(num_regions, 1, 1)
            k_expanded = k_t.repeat_interleave(batch_size, dim=0)
            v_expanded = v_t.repeat_interleave(batch_size, dim=0)
            
            attn_output = optimized_attention(q_expanded, k_expanded, v_expanded, extra_options["n_heads"])
            
            if cross_region_attention > 0.0 and num_regions > 1:
                attn_output = self._apply_cross_region_blending(
                    attn_output, num_regions, batch_size, module,
                    cross_region_attention, cross_region_mode
                )
            
            attn_output.mul_(ensure_dtype_device(masks, q.dtype, q.device))
            attn_output = attn_output.view(num_regions, batch_size, seq_len, -1)
            
            # Apply strength (always, no conditional)
            strength_tensor = torch.tensor(strengths, device=q.device, dtype=q.dtype).view(num_regions, 1, 1, 1)
            attn_output.mul_(strength_tensor)
            
            outputs.append(attn_output.sum(dim=0))
        
        return torch.cat(outputs, dim=0)
        
    def _compute_kv(self, cond_tensors: List[torch.Tensor], module: Any, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute key and value tensors from conditioning"""
        if not cond_tensors: 
            return torch.empty(0, device=q.device, dtype=q.dtype), torch.empty(0, device=q.device, dtype=q.dtype)
        
        context = torch.cat(cond_tensors, dim=0)
        k = ensure_dtype_device(module.to_k(context), q.dtype, q.device)
        v = ensure_dtype_device(module.to_v(context), q.dtype, q.device)
        return k, v

    def _apply_cross_region_blending(self, qkv: torch.Tensor, num_regions: int, 
                                    batch_size: int, module: Any,
                                    cross_region_attention: float, 
                                    cross_region_mode: str) -> torch.Tensor:
        """Apply cross-region blending to soften boundaries"""
        head_dim = module.heads * module.dim_head
        qkv_reshaped = qkv.view(num_regions, batch_size, -1, head_dim)
        
        if cross_region_mode == "self_exclusion":
            other_regions_mean = (qkv_reshaped.sum(dim=0, keepdim=True) - qkv_reshaped) / max(num_regions - 1, 1)
        else:
            other_regions_mean = qkv_reshaped.mean(dim=0, keepdim=True)
        
        blended = torch.lerp(qkv_reshaped, other_regions_mean, cross_region_attention)
        return blended.view(num_regions * batch_size, -1, head_dim)