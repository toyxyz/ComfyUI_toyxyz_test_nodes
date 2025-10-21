"""
Flux-specific attention patching logic for regional prompting
Based on Fluxtapoz approach with attention mask blocking
✅ Dynamic calculation with EXACT original logic preserved
"""
import torch
import math
from typing import Dict, List, Tuple, Any, Optional

from .couple_utils import (
    log_debug, log_warning, ARCH_CONFIGS
)


class RegionalMask(torch.nn.Module):
    """
    Regional attention mask for Flux - DYNAMIC CALCULATION
    Preserves exact original logic, only computes sizes from query
    """
    def __init__(self, region_conds: List[torch.Tensor],
                 region_masks: List[torch.Tensor], 
                 start_percent: float, end_percent: float,
                 apply_t5_background: bool = True) -> None:
        super().__init__()
        # Store region data
        for i, (cond, mask) in enumerate(zip(region_conds, region_masks)):
            self.register_buffer(f'region_cond_{i}', cond)
            self.register_buffer(f'region_mask_{i}', mask)
        
        self.num_regions = len(region_masks)
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.apply_t5_background = apply_t5_background
        
        # Cache
        self._cached_mask = None
        self._cached_shape = None

    def __call__(self, q, transformer_options, txt_size):
        """
        Called during attention computation
        """
        current_sigma = transformer_options['sigmas'][0]
        current_percent = 1 - current_sigma
        
        if not (self.start_percent <= current_percent < self.end_percent):
            return None
        
        # Extract shape from query (Flux: [B, H, L, D])
        b, heads, seq_len, head_dim = q.shape
        img_len = seq_len - txt_size
        
        # Cache check
        cache_key = (seq_len, txt_size, img_len)
        if self._cached_mask is not None and self._cached_shape == cache_key:
            return self._cached_mask
        
        log_debug(f"[Flux Dynamic] Building mask for seq_len={seq_len}, txt_size={txt_size}, img_len={img_len}")
        
        # Infer h_attn, w_attn from img_len
        h_attn, w_attn = self._infer_hw_from_img_len(img_len)
        
        log_debug(f"[Flux Dynamic] Inferred attention size: {h_attn}x{w_attn}")
        
        # Build mask using EXACT original logic
        regional_mask = self._build_mask_original_logic(
            h_attn, w_attn, img_len, txt_size, seq_len, q.device, q.dtype
        )
        
        # Cache
        self._cached_mask = regional_mask
        self._cached_shape = cache_key
        
        return regional_mask
    
    def _infer_hw_from_img_len(self, img_len: int) -> Tuple[int, int]:
        """
        Infer h_attn, w_attn from img_len based on mask shape
        
        Returns: (h, w) where h * w = img_len
        """
        # Get first mask to determine original dimensions
        first_mask = getattr(self, 'region_mask_0')
        mask_h, mask_w = first_mask.shape[-2:]  # [height, width]
        
        # Calculate attention size (VAE /8, Flux /2 = total /16)
        # mask_h/16 * mask_w/16 should equal img_len
        expected_h = mask_h // 16
        expected_w = mask_w // 16
        
        # Verify
        if expected_h * expected_w == img_len:
            return expected_h, expected_w
        
        # Fallback: use aspect ratio preserving calculation
        aspect_ratio = mask_h / mask_w  # height/width ratio
        w = int(math.sqrt(img_len / aspect_ratio))
        h = img_len // w
        
        # Adjust to match img_len exactly
        while h * w != img_len and w > 1:
            w -= 1
            h = img_len // w
        
        if h * w != img_len:
            # Last resort: square
            h = w = int(math.sqrt(img_len))
        
        log_debug(f"[Flux Dynamic] Mask shape: {mask_h}x{mask_w}, inferred attention: {h}x{w}")
        
        return h, w
    
    def _build_mask_original_logic(self, h: int, w: int, img_len: int,
                                   txt_size: int, total_len: int,
                                   device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Build mask using EXACT original logic from Fluxtapoz
        This is copied from the original _build_regional_mask_and_cond
        """
        # Collect region conds and masks
        region_conds = []
        region_masks_spatial = []
        
        for i in range(self.num_regions):
            cond = getattr(self, f'region_cond_{i}')
            mask = getattr(self, f'region_mask_{i}')
            
            region_conds.append(cond)
            
            # Resize mask to attention resolution
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            
            # Original logic: resize to (h, w) which is attention resolution
            mask_resized = torch.nn.functional.interpolate(
                mask.unsqueeze(0).to(dtype=dtype, device=device),
                size=(h, w),
                mode='nearest-exact'
            )
            region_masks_spatial.append(mask_resized[0, 0])  # [h, w]
        
        # Calculate T5 length
        t5_len = 256 if self.apply_t5_background else 0
        
        # Calculate regional text length from actual conds
        regional_text_len = sum(cond.shape[1] for cond in region_conds)
        text_len = t5_len + regional_text_len
        
        # Sanity check
        if text_len != txt_size:
            log_warning(f"Text length mismatch: calculated={text_len}, actual={txt_size}")
            # Use actual txt_size to be safe
            text_len = txt_size
            regional_text_len = txt_size - t5_len
        
        # Initialize mask matrix (False = block attention)
        regional_mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)
        
        # 1. T5 tokens attend to themselves
        if self.apply_t5_background:
            regional_mask[0:t5_len, 0:t5_len] = True
        
        # 2. Process each region - EXACT ORIGINAL LOGIC
        self_attend_masks = torch.zeros((img_len, img_len), dtype=torch.bool, device=device)
        union_masks = torch.zeros((img_len, img_len), dtype=torch.bool, device=device)
        
        current_seq_pos = t5_len
        
        for idx, (region_cond, spatial_mask) in enumerate(zip(region_conds, region_masks_spatial)):
            region_text_len = region_cond.shape[1]
            next_seq_pos = current_seq_pos + region_text_len
            
            # Flatten spatial mask and expand to text length
            # spatial_mask: [h, w], 1=region, 0=background
            flat_mask = spatial_mask.flatten().unsqueeze(1).repeat(1, region_text_len)  # [img_len, text_len]
            
            # 2a. Region text attends to itself
            regional_mask[current_seq_pos:next_seq_pos, current_seq_pos:next_seq_pos] = True
            
            # 2b. Region text → corresponding image region only
            regional_mask[current_seq_pos:next_seq_pos, text_len:] = flat_mask.transpose(-1, -2)
            
            # 2c. Image region → corresponding region text only
            regional_mask[text_len:, current_seq_pos:next_seq_pos] = flat_mask
            
            # 2d. Image region pixels attend to each other (same region only)
            img_mask_full = flat_mask[:, :1].repeat(1, img_len)  # [img_len, img_len]
            img_mask_full_t = img_mask_full.transpose(-1, -2)
            self_attend_masks = torch.logical_or(
                self_attend_masks,
                torch.logical_and(img_mask_full, img_mask_full_t)
            )
            
            # Track union of all regions
            union_masks = torch.logical_or(
                union_masks,
                torch.logical_or(img_mask_full, img_mask_full_t)
            )
            
            current_seq_pos = next_seq_pos
        
        # 3. Background handling
        background_masks = torch.logical_not(union_masks)
        background_and_self_attend = torch.logical_or(background_masks, self_attend_masks)
        regional_mask[text_len:, text_len:] = background_and_self_attend
        
        # 4. Uncovered background pixels can attend to T5
        if self.apply_t5_background:
            uncovered_pixels = torch.stack(region_masks_spatial).sum(dim=0) < 0.01
            uncovered_pixels_flat = uncovered_pixels.flatten()
            regional_mask[text_len:, 0:t5_len] = uncovered_pixels_flat.unsqueeze(1).repeat(1, t5_len)
        
        log_debug(f"[Flux Dynamic] Mask built: {regional_mask.sum().item()}/{regional_mask.numel()} "
                  f"({regional_mask.sum().item()/regional_mask.numel()*100:.1f}%)")
        
        return regional_mask


class RegionalConditioning(torch.nn.Module):
    """Regional conditioning tensor for Flux"""
    def __init__(self, region_cond: torch.Tensor, start_percent: float, end_percent: float) -> None:
        super().__init__()
        self.register_buffer('region_cond', region_cond)
        self.start_percent = start_percent
        self.end_percent = end_percent

    def __call__(self, transformer_options, *args, **kwargs):
        current_sigma = transformer_options['sigmas'][0]
        current_percent = 1 - current_sigma
        
        if self.start_percent <= current_percent < self.end_percent:
            return self.region_cond
        return None


class FluxAttentionPatcher:
    """
    Manages attention masking for Flux regional prompting
    ✅ NOW WITH DYNAMIC CALCULATION - preserves original logic
    """

    def __init__(self, model: Any, regions: List[Dict],
                 start_percent: float = 0.0, end_percent: float = 0.5,
                 attn_override: Optional[Dict] = None,
                 apply_t5_background: bool = True):
        self.model = model
        self.regions = regions
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.attn_override = attn_override if attn_override is not None else ARCH_CONFIGS['flux']
        self.apply_t5_background = apply_t5_background

    def patch(self) -> Tuple[Any, Any]:
        """Apply Flux regional patching with dynamic calculation"""
        # Filter active regions
        active_regions = [r for r in self.regions if r.get('strength', 1.0) > 0]
        
        if not active_regions:
            log_warning("No active regions for Flux patching")
            return self.model, self.regions[0]['positive'] if self.regions else None
        
        # Validate regions
        for idx, r in enumerate(active_regions):
            if r['mask'].dim() not in [2, 3]:
                raise ValueError(f"Region {idx} mask must be 2D or 3D, got {r['mask'].dim()}D")
            mask_sum = r['mask'].sum().item()
            if mask_sum < 0.001:
                log_warning(f"Region {idx} has empty mask (sum={mask_sum:.6f})")
        
        print(f"[ComfyCouple Flux] Processing {len(active_regions)} regions (dynamic calculation)")
        
        # Extract region data
        region_conds = []
        region_masks = []
        
        for r in active_regions:
            cond_tensor = r['positive'][0][0]  # [1, seq_len, 4096]
            region_conds.append(cond_tensor)
            region_masks.append(r['mask'])
        
        # Build regional conditioning for model injection
        regional_conditioning = torch.cat(region_conds, dim=1)
        
        # Create wrapper modules
        regional_mask_module = RegionalMask(
            region_conds,  # Pass actual conds for text length
            region_masks,
            self.start_percent, 
            self.end_percent,
            self.apply_t5_background
        )
        regional_cond_module = RegionalConditioning(
            regional_conditioning, 
            self.start_percent, 
            self.end_percent
        )
        
        # Clone and patch model
        new_model = self.model.clone()
        
        # Inject regional conditioning
        new_model.set_model_patch(regional_cond_module, 'regional_conditioning')
        
        # Patch attention blocks
        for block_idx in self.attn_override['double']:
            new_model.set_model_patch_replace(regional_mask_module, "double", "mask_fn", int(block_idx))
        
        for block_idx in self.attn_override['single']:
            new_model.set_model_patch_replace(regional_mask_module, "single", "mask_fn", int(block_idx))
        
        print(f"[ComfyCouple Flux] Patched {len(self.attn_override['double'])} double + "
              f"{len(self.attn_override['single'])} single blocks")
        print(f"[ComfyCouple Flux] Active timesteps: {self.start_percent*100:.0f}% - {self.end_percent*100:.0f}%")
        print(f"[ComfyCouple Flux] Mask computed dynamically at runtime")
        
        return new_model, active_regions[0]['positive']