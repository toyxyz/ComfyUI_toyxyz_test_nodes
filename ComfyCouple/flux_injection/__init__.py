"""
Flux model injection module for ComfyCouple
Based on Fluxtapoz implementation
"""

from .model import inject_flux
from .layers import inject_blocks

__all__ = ['inject_flux', 'inject_blocks', 'inject_flux_for_comfycouple']


def inject_flux_for_comfycouple(model):
    """
    Inject Flux modifications for regional prompting support
    
    This function:
    1. Checks if injection already performed
    2. Injects custom Flux class (adds regional_conditioning support)
    3. Injects custom Block classes (adds mask_fn support)
    
    Args:
        model: ComfyUI model object
        
    Returns:
        model: Modified model (same instance, class replaced)
    """
    diffusion_model = model.model.diffusion_model
    
    # Check if already injected (prevent duplicate injection)
    if hasattr(diffusion_model, '_comfycouple_flux_injected'):
        print("[ComfyCouple] Flux already injected, skipping")
        return model
    
    # Perform injection
    print("[ComfyCouple] Injecting Flux modifications...")
    inject_flux(diffusion_model)
    inject_blocks(diffusion_model)
    
    # Mark as injected
    diffusion_model._comfycouple_flux_injected = True
    
    print("[ComfyCouple] Flux injection complete")
    return model