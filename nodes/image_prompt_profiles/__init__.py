"""Independent prompt-type profiles; unknown types never silently fall back."""
from dataclasses import dataclass, field
from typing import Mapping
from types import MappingProxyType

from . import normal


@dataclass(frozen=True)
class ImagePromptProfile:
    system_prompt: str
    image_analysis_prompt: str
    reference_policy: str
    enhancement_prompts: Mapping[str, str] = field(default_factory=dict)
    enhancement_style_guidance: str = ""
    enhanced_system_prompt: str = ""
    viewpoint_policy: str = ""
    camera_resolution_prompt: str = ""
    camera_system_prompt: str = ""
    camera_distance_prompt: str = ""  # Optional complete system policy for resolved wide/extreme wide.
    camera_intent_prompt: str = ""
    camera_enhancement_prompts: Mapping[str, str] = field(default_factory=dict)


# Each future type supplies its own module. Normal is general image prompting,
# not a promise of special support for any particular image model.
PROMPT_PROFILES = MappingProxyType({
    "normal": ImagePromptProfile(
        system_prompt=normal.SYSTEM_PROMPT,
        image_analysis_prompt=normal.IMAGE_ANALYSIS_PROMPT,
        reference_policy=normal.REFERENCE_POLICY,
        enhancement_prompts=MappingProxyType(normal.ENHANCEMENT_PROMPTS),
        enhancement_style_guidance=normal.ENHANCEMENT_STYLE_GUIDANCE,
        enhanced_system_prompt=normal.ENHANCED_SYSTEM_PROMPT,
        viewpoint_policy=normal.VIEWPOINT_POLICY,
        camera_resolution_prompt=normal.CAMERA_RESOLUTION_PROMPT,
        camera_system_prompt=normal.CAMERA_SYSTEM_PROMPT,
        camera_distance_prompt=normal.CAMERA_DISTANCE_PROMPT,
        camera_intent_prompt=normal.CAMERA_INTENT_PROMPT,
        camera_enhancement_prompts=MappingProxyType(normal.CAMERA_ENHANCEMENT_PROMPTS),
    ),
})
