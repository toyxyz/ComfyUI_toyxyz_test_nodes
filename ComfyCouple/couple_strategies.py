"""
Strategy dispatch layer for ComfyCouple model support.

This keeps model-family branching in one place so new architectures can be
added later without rewriting the node entrypoints again.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type

from .couple_utils import ArchitectureInfo, get_model_support_profile


@dataclass
class BaseRegionStrategy:
    """Common strategy interface for region-processing backends."""

    arch_info: ArchitectureInfo

    def prepare_model(self, processor: Any, model: Any, kwargs: Dict[str, Any]) -> Any:
        return model

    def process(
        self,
        processor: Any,
        model: Any,
        region: List[Dict[str, Any]],
        negative: Any,
        **kwargs,
    ) -> Tuple[Any, Any, Any]:
        raise NotImplementedError


class UNetCrossAttentionStrategy(BaseRegionStrategy):
    """Strategy for SD1.5 / SDXL style U-Net attn2 patching."""

    def process(
        self,
        processor: Any,
        model: Any,
        region: List[Dict[str, Any]],
        negative: Any,
        **kwargs,
    ) -> Tuple[Any, Any, Any]:
        return processor._process_sd_sdxl(model, region, negative, self.arch_info, **kwargs)


class FluxMaskStrategy(BaseRegionStrategy):
    """Strategy for Flux regional masking."""

    def prepare_model(self, processor: Any, model: Any, kwargs: Dict[str, Any]) -> Any:
        if kwargs.get("auto_inject_flux", True):
            return processor._inject_flux_if_needed(model)
        return model

    def process(
        self,
        processor: Any,
        model: Any,
        region: List[Dict[str, Any]],
        negative: Any,
        **kwargs,
    ) -> Tuple[Any, Any, Any]:
        return processor._process_flux(model, region, negative, self.arch_info, **kwargs)


class AnimaApplyModelStrategy(BaseRegionStrategy):
    """Strategy for Anima regional prompting via apply_model wrapper patching."""

    def process(
        self,
        processor: Any,
        model: Any,
        region: List[Dict[str, Any]],
        negative: Any,
        **kwargs,
    ) -> Tuple[Any, Any, Any]:
        return processor._process_anima(model, region, negative, self.arch_info, **kwargs)


STRATEGY_REGISTRY: Dict[str, Type[BaseRegionStrategy]] = {
    "unet_attn2": UNetCrossAttentionStrategy,
    "anima_apply_model": AnimaApplyModelStrategy,
    "flux_mask": FluxMaskStrategy,
}


def build_region_strategy(arch_info: ArchitectureInfo) -> BaseRegionStrategy:
    strategy_cls = STRATEGY_REGISTRY.get(arch_info.strategy_key)
    if strategy_cls is None:
        profile = get_model_support_profile(arch_info.profile_key) if arch_info.profile_key else None
        raise KeyError(
            f"No region strategy registered for '{arch_info.strategy_key}'"
            + (f" (profile={profile.key})" if profile is not None else "")
        )
    return strategy_cls(arch_info)
