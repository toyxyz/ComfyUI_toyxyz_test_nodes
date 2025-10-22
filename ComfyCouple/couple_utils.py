"""
Utility functions and data classes for Comfy Couple - WITH FLUX SUPPORT
✅ Updated for Flux dynamic calculation
"""
import torch
import torch.nn.functional as F
import math
import copy
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Constants
DEBUG_MODE = False
MASK_EPSILON = 1e-6
OVERLAP_THRESHOLD = 1.0
MIN_STRENGTH = 0.0
MAX_STRENGTH = 10.0

# Flux Block Presets
FLUX_BLOCK_PRESETS = {
    'default': {
        'double': [i for i in range(1, 19, 2)],  # [1,3,5,7,9,11,13,15,17]
        'single': [i for i in range(1, 38, 2)],  # [1,3,5,...,35,37]
        'description': 'odd blocks only, balanced coverage'
    },
    'balanced': {
        'double': [0, 2, 4, 6, 8, 10, 12, 14],
        'single': [1, 3, 5, 7, 9, 11, 13, 15, 18, 20, 22],
        'description': 'Even double blocks with selective single blocks - smoother transitions'
    },
    'input_heavy': {
        'double': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'single': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14],
        'description': 'Focus on early layers - stronger layout control, may have sharper boundaries'
    },
    'output_heavy': {
        'double': [10, 11, 12, 13, 14, 15, 16, 17, 18],
        'single': [20, 22, 24, 26, 28, 30, 32, 34, 36],
        'description': 'Focus on late layers - softer boundaries, may reduce regional precision'
    },
    'middle_focus': {
        'double': [4, 5, 6, 7, 8, 9, 10, 11, 12],
        'single': [8, 10, 12, 14, 16, 18, 20, 22, 24, 26],
        'description': 'Middle layers emphasis - balanced control and coherence'
    },
    'light': {
        'double': [2, 4, 6, 8, 10, 12],
        'single': [5, 10, 15, 20, 25, 30],
        'description': 'Minimal blocks - fastest, softest boundaries, less precise control'
    },
    'aggressive': {
        'double': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'single': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        'description': 'Heavy coverage - strongest control, may create visible boundaries'
    },
    'all_blocks': {
        'double': list(range(19)),  # [0,1,2,...,18]
        'single': list(range(38)),  # [0,1,2,...,37]
        'description': 'All 56 blocks - maximum control but highest computational cost'
    }
}

ARCH_CONFIGS = {
    'sdxl': {
        'input_blocks': [4, 5, 7, 8], 
        'input_indices': {4: 2, 5: 2, 7: 10, 8: 10},
        'middle_count': 10,
        'output_blocks': list(range(6)), 
        'output_indices': {0: 10, 1: 10, 2: 10, 3: 2, 4: 2, 5: 2}
    },
    'sd15': {
        'input_blocks': [1, 2, 4, 5, 7, 8, 10, 11], 
        'middle_count': 1, 
        'output_blocks': list(range(3, 12))
    },
    'flux': FLUX_BLOCK_PRESETS['default']  # Default preset
}

class ModelType(Enum):
    SDXL = "sdxl"
    SD15 = "sd15"
    FLUX = "flux"

@dataclass
class ArchitectureInfo:
    type: ModelType
    use_cfg: bool
    dim: int = 768
    is_flux: bool = False

# Logging
def log_debug(message: str) -> None:
    if DEBUG_MODE: 
        print(f"[ComfyCouple DEBUG] {message}")

def log_warning(message: str) -> None:
    print(f"[ComfyCouple WARNING] {message}")

def get_flux_preset(preset_name: str) -> Dict[str, List[int]]:
    """Get Flux block preset by name"""
    if preset_name not in FLUX_BLOCK_PRESETS:
        log_warning(f"Unknown preset '{preset_name}', using 'default'")
        preset_name = 'default'
    
    preset = FLUX_BLOCK_PRESETS[preset_name]
    log_debug(f"Using Flux preset '{preset_name}': {preset['description']}")
    log_debug(f"  Double blocks ({len(preset['double'])}): {preset['double']}")
    log_debug(f"  Single blocks ({len(preset['single'])}): {preset['single']}")
    
    return {
        'double': preset['double'],
        'single': preset['single']
    }

# Validation
def validate_strength(strength: float, name: str = "strength", is_flux: bool = False) -> float:
    if is_flux:
        # Flux only supports 0 or 1 (will be treated as boolean)
        if strength > 0.5:
            return 1.0
        else:
            return 0.0
    
    clamped = max(MIN_STRENGTH, min(MAX_STRENGTH, strength))
    if clamped != strength:
        log_warning(f"{name} ({strength}) clamped to valid range: {clamped}")
    return clamped

# Tensor utilities
def ensure_dtype_device(tensor: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if tensor.dtype != dtype or tensor.device != device:
        return tensor.to(dtype=dtype, device=device)
    return tensor

def create_zero_conditioning(ref_cond: List[Tuple[torch.Tensor, Dict]]) -> List[Tuple[torch.Tensor, Dict]]:
    """Create zero conditioning with proper dimensions"""
    zero_cond = []
    for cond_tensor, metadata in ref_cond:
        new_metadata = copy.deepcopy(metadata)
        if "pooled_output" in new_metadata and isinstance(new_metadata["pooled_output"], torch.Tensor):
            new_metadata["pooled_output"] = torch.zeros_like(new_metadata["pooled_output"])
        zero_cond.append((torch.zeros_like(cond_tensor), new_metadata))
    return zero_cond

def find_best_divisors(seq_len: int, aspect_ratio: float) -> Tuple[int, int]:
    if seq_len <= 0: 
        raise ValueError("Sequence length must be positive")
    
    best_h, best_w, min_diff = 1, seq_len, float('inf')
    for h in range(1, int(math.sqrt(seq_len)) + 1):
        if seq_len % h == 0:
            w = seq_len // h
            for ch, cw in [(h, w), (w, h)]:
                diff = abs(cw / ch - aspect_ratio)
                if diff < min_diff:
                    min_diff, best_h, best_w = diff, ch, cw
    return best_h, best_w

def pad_conditioning_tensors(cond_list: List[torch.Tensor]) -> List[torch.Tensor]:
    if not cond_list: 
        return []
    max_len = max(c.shape[1] for c in cond_list if isinstance(c, torch.Tensor))
    return [
        F.pad(c, (0, 0, 0, max_len - c.shape[1])) if isinstance(c, torch.Tensor) and c.shape[1] < max_len else c
        for c in cond_list
    ]

# Model detection
def detect_model_architecture(model: Any) -> ArchitectureInfo:
    """
    Detect model architecture (SD1.5, SDXL, or Flux)
    """
    try:
        diff_model = model.model.diffusion_model
        
        # Check for Flux (DiT architecture with double/single blocks)
        if hasattr(diff_model, 'double_blocks') and hasattr(diff_model, 'single_blocks'):
            log_debug("Detected Flux DiT architecture")
            return ArchitectureInfo(type=ModelType.FLUX, use_cfg=True, dim=4096, is_flux=True)
        
        # Check for SDXL
        if hasattr(diff_model, 'label_emb'):
            return ArchitectureInfo(type=ModelType.SDXL, use_cfg=True, dim=2048, is_flux=False)
        
        # Default to SD1.5
        return ArchitectureInfo(type=ModelType.SD15, use_cfg=True, dim=768, is_flux=False)
        
    except Exception as e:
        log_warning(f"Model detection failed, assuming SD1.5: {e}")
        return ArchitectureInfo(type=ModelType.SD15, use_cfg=True, dim=768, is_flux=False)

# Mask processing
class MaskProcessor:
    @staticmethod
    def resize_to_sequence(mask: torch.Tensor, seq_len: int, aspect_ratio: float, 
                          dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        m = mask.unsqueeze(0).unsqueeze(0) if mask.dim() == 2 else mask.unsqueeze(1)
        th, tw = find_best_divisors(seq_len, aspect_ratio)
        
        resized = F.interpolate(
            m.to(dtype=dtype, device=device), 
            size=(th, tw), 
            mode="nearest"
        )
        flat = resized.view(1, -1, 1)
        
        if flat.shape[1] != seq_len:
            return F.pad(flat, (0, 0, 0, seq_len - flat.shape[1])) if flat.shape[1] < seq_len else flat[:, :seq_len, :]
        return flat

    @staticmethod
    def from_query(masks: List[Any], q: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
        """Convert masks to attention-compatible sequence format"""
        b, seq_len, dim = q.shape
        h, w = (original_shape[2], original_shape[3]) if len(original_shape) > 2 else (64, 64)
        aspect = w / h if h > 0 else 1.0
        
        processed = []
        for m in masks:
            if isinstance(m, torch.Tensor):
                resized = MaskProcessor.resize_to_sequence(m, seq_len, aspect, q.dtype, q.device)
                processed.append(resized.expand(b, -1, dim))
            else:
                processed.append(torch.ones((b, seq_len, dim), device=q.device, dtype=q.dtype))
        
        return torch.cat(processed, dim=0)
    
    @staticmethod
    def apply_feather(mask: torch.Tensor, feather_pixels: int) -> torch.Tensor:
        if feather_pixels <= 0: 
            return mask
        
        max_feather = int(min(mask.shape[-2:]) * 0.25)
        pixels = min(feather_pixels, max_feather)
        if pixels < 1: 
            return mask

        m = mask.unsqueeze(0).unsqueeze(0) if mask.dim() == 2 else mask.unsqueeze(1)
        
        try:
            import torchvision.transforms.functional as TF
            blurred = TF.gaussian_blur(m, kernel_size=pixels * 2 + 1)
        except ImportError:
            log_warning("torchvision not installed, skipping feathering")
            return mask
        
        return blurred.squeeze(0).squeeze(0) if mask.dim() == 2 else blurred.squeeze(1)

    @staticmethod
    def normalize_overlaps(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(regions) <= 1: 
            return regions
        
        masks = torch.stack([r["mask"] for r in regions])
        mask_sum = masks.sum(dim=0)
        
        overlap_area = mask_sum > OVERLAP_THRESHOLD
        if overlap_area.any():
            overlap_pct = overlap_area.sum().item() / mask_sum.numel() * 100
            log_debug(f"Normalizing overlapping masks ({overlap_pct:.2f}%)")
            clamped_sum = mask_sum.clamp(min=MASK_EPSILON)
            for i, region in enumerate(regions):
                region["mask"] = torch.where(overlap_area, masks[i] / clamped_sum, masks[i])
        
        return regions
    
    @staticmethod
    def resize_to_match(mask: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Resize mask to match target shape"""
        if mask.shape[-2:] == target_shape[-2:]: 
            return mask.float()  # ✅ float32로 변환
        
        # ✅ 수정: 입력을 float32로 변환
        m = mask.float()
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:
            m = m.unsqueeze(1)
        
        resized = F.interpolate(
            m, 
            size=target_shape[-2:], 
            mode="nearest"
        )
        
        return resized.squeeze(0).squeeze(0) if mask.dim() == 2 else resized.squeeze(1)
    
    @staticmethod
    def resize_for_flux_latent(mask: torch.Tensor, latent_h: int, latent_w: int, 
                               dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """
        [DEPRECATED] No longer used in dynamic calculation mode
        
        Resize mask for Flux latent space
        Flux uses h//2, w//2 for attention resolution
        
        This method is kept for backward compatibility only.
        New code should rely on dynamic calculation in RegionalMask.__call__()
        """
        target_h = latent_h // 2
        target_w = latent_w // 2
        
        m = mask.unsqueeze(0).unsqueeze(0) if mask.dim() == 2 else mask.unsqueeze(1)
        resized = F.interpolate(
            m.to(dtype=dtype), 
            size=(target_h, target_w), 
            mode="nearest-exact"
        )
        
        return resized.squeeze(0).squeeze(0) if mask.dim() == 2 else resized.squeeze(1)
