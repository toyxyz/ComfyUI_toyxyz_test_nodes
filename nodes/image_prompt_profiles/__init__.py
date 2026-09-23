"""Independent prompt-type profiles; unknown types never silently fall back."""
from dataclasses import dataclass, field
from typing import Mapping
from types import MappingProxyType

from . import default, qwen_image_2_1


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
    output_mode: str = "scene"


# Scene and edit writers are independent. Legacy names resolve at the node boundary.
PROMPT_PROFILES = MappingProxyType({
    "default": ImagePromptProfile(
        system_prompt=default.SYSTEM_PROMPT,
        image_analysis_prompt=default.IMAGE_ANALYSIS_PROMPT,
        reference_policy=default.REFERENCE_POLICY,
        enhancement_prompts=MappingProxyType(default.ENHANCEMENT_PROMPTS),
        enhancement_style_guidance=default.ENHANCEMENT_STYLE_GUIDANCE,
        enhanced_system_prompt=default.ENHANCED_SYSTEM_PROMPT,
        viewpoint_policy=default.VIEWPOINT_POLICY,
        camera_resolution_prompt=default.CAMERA_RESOLUTION_PROMPT,
        camera_system_prompt=default.CAMERA_SYSTEM_PROMPT,
        camera_distance_prompt=default.CAMERA_DISTANCE_PROMPT,
        camera_intent_prompt=default.CAMERA_INTENT_PROMPT,
        camera_enhancement_prompts=MappingProxyType(default.CAMERA_ENHANCEMENT_PROMPTS),
    ),
    "qwen_image_2.1": ImagePromptProfile(
        system_prompt=qwen_image_2_1.SYSTEM_PROMPT,
        image_analysis_prompt=qwen_image_2_1.IMAGE_ANALYSIS_PROMPT,
        reference_policy="",
        enhancement_prompts=MappingProxyType(qwen_image_2_1.ENHANCEMENT_PROMPTS),
        output_mode="edit",
    ),
})
