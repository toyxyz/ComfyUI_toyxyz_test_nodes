"""Discrete image shot, angle and style presets. No 3D scene or model calls."""
from .image_camera import shot_framing
from .image_style_presets import STYLE_PRESETS, STYLE_CATEGORIES

SHOT_LABELS = {
    "Extreme wide shot": "extreme wide", "Wide shot": "wide", "Full body shot": "full",
    "Cowboy shot": "medium-long", "Medium shot": "medium", "Medium close-up": "medium close-up",
    "Close-up": "close-up", "Extreme close-up": "extreme close-up",
}
VIEWS = {
    "Front view": {"viewpoint": "a straight-on frontal view"},
    "Front three-quarter — camera left": {"viewpoint": "a front three-quarter view from the subject's own right side", "viewpoint_side_cue": "The subject's own right side is nearer the lens; its front projects toward image-right."},
    "Front three-quarter — camera right": {"viewpoint": "a front three-quarter view from the subject's own left side", "viewpoint_side_cue": "The subject's own left side is nearer the lens; its front projects toward image-left."},
    "Side profile — subject right": {"viewpoint": "a strict profile from the subject's own right side", "viewpoint_side_cue": "The subject's front projects toward image-right, across the image."},
    "Side profile — subject left": {"viewpoint": "a strict profile from the subject's own left side", "viewpoint_side_cue": "The subject's front projects toward image-left, across the image."},
    "Rear view": {"viewpoint": "a straight rear view, retaining rear surfaces without turning the subject"},
    "High angle": {"viewing_angle": "an elevated viewpoint looking downward"},
    "Eye-level view": {"viewing_angle": "a level viewing direction from the selected subject's eye height; preserve its existing pose"},
    "Low angle": {"viewing_angle": "a low viewpoint looking upward"},
    "Worm's-eye view": {"viewing_angle": "a very low viewpoint near the base of the subject, looking steeply upward; preserve the subject's pose and existing setting"},
    "Drone view": {"viewing_angle": "an airborne oblique viewpoint above the subject, looking diagonally downward; retain the selected shot size, do not automatically widen the frame or add a drone to the scene"},
    "Aerial view": {"viewing_angle": "a high-altitude aerial viewpoint looking steeply down; retain the selected target and shot size rather than inventing a landscape or changing subject scale"},
    "Overhead view": {"viewing_angle": "directly overhead, looking vertically down"},
    "Dutch angle — clockwise": {"image_roll": "The image plane is tilted 20 degrees clockwise; this is camera roll, not a leaning subject or a change of subject pose."},
    "Dutch angle — counterclockwise": {"image_roll": "The image plane is tilted 20 degrees counterclockwise; this is camera roll, not a leaning subject or a change of subject pose."},
    "Over-the-shoulder view": {"viewpoint": "a view from just behind an existing observer's shoulder toward the user-requested subject or scene; only a small shoulder/back-of-head edge frames the foreground. Use an observer supported by the request or reference, not an invented extra person. If no observer is established, describe a shoulder-height rear-oblique viewpoint without adding a foreground person."},
    "First-person POV": {"viewpoint": "a first-person optical viewpoint from the user-designated observer, looking into the requested scene rather than showing that observer externally. Do not invent hands, limbs, held objects or actions. Shot framing applies to the observed target, not the observer's body; if unspecified, preserve the described scene."},
    "Isometric view": {"viewpoint": "an elevated three-quarter view", "viewing_angle": "looking down approximately 35 degrees", "projection": "isometric parallel projection with parallel lines remaining parallel, not perspective convergence; retain the user's medium and scene rather than inventing a miniature or diorama"},
}
PRESETS = ("From user prompt", *SHOT_LABELS, *VIEWS)
ANGLES = ("None", *VIEWS)


def compile_preset(settings):
    settings = settings if isinstance(settings, dict) else {}
    preset = settings.get("preset", PRESETS[0])
    if not isinstance(preset, str) or preset not in PRESETS:
        preset = PRESETS[0]
    shot = settings.get("shot_size", "From preset / user prompt")
    if not isinstance(shot, str) or shot not in ("None", "From user prompt", "From preset / user prompt", *SHOT_LABELS):
        shot = "From preset / user prompt"
    # Old payloads combined shot and angle in `preset`. Canonical v2 separates them.
    angle = settings.get("angle", preset if preset in VIEWS else ANGLES[0])
    if not isinstance(angle, str) or angle not in ANGLES:
        angle = ANGLES[0]
    style = settings.get("style", "None")
    if not isinstance(style, str) or style not in STYLE_PRESETS:
        style = "None"
    components = dict(VIEWS.get(angle, {}))
    resolved_shot = SHOT_LABELS.get(shot) or (SHOT_LABELS.get(preset) if "angle" not in settings else None)
    if resolved_shot:
        components.update(shot_size=resolved_shot, framing=shot_framing(resolved_shot))
    phrases = [components[k] for k in ("viewpoint", "viewpoint_side_cue", "viewing_angle", "projection", "image_roll") if k in components]
    if resolved_shot:
        label = "cowboy" if resolved_shot == "medium-long" else resolved_shot
        phrases.insert(0, label + " shot. " + components["framing"])
    prompt = ("Camera preset: " + ". ".join(phrases) + ". Explicit user instructions override this preset. Preserve subject action, pose, gaze, clothing and setting.") if phrases else ""
    camera_prompt = prompt
    style_prompt = STYLE_PRESETS.get(style, "")
    if style_prompt:
        prompt += ("\n\n" if prompt else "") + "Style preset: " + style_prompt + " Explicit user style, color, lighting and preservation instructions take priority. Apply only compatible rendering treatment; do not change subjects, pose, framing or setting to create the style."
    canonical_shot = next((label for label, value in SHOT_LABELS.items() if value == resolved_shot), "None")
    return {"kind": "image_camera_preset", "version": 3,
            "settings": {"angle": angle, "shot_size": canonical_shot, "style": style},
            "components": components, "camera_prompt": camera_prompt, "style_prompt": style_prompt, "prompt": prompt}


def style_guidance(value):
    """Rebuild style data from the allowlisted selection, never trust supplied prose."""
    if not isinstance(value, dict) or value.get("kind") != "image_camera_preset":
        return ""
    return compile_preset(value.get("settings"))["style_prompt"]


class ImageCameraPresets:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shot_size": (["None", *SHOT_LABELS], {"default": "Full body shot", "tooltip": "None adds no shot-size guidance. Otherwise select subject coverage: wide, full body, cowboy, medium or close-up. Explicit user target and crop always win. POV coverage applies to the observed target."}),
            "angle": (list(ANGLES), {"default": "None", "tooltip": "None adds no angle guidance. Camera viewpoint is independent of shot size: high, low, worm's-eye, drone, aerial, overhead, POV and more. Drone is oblique; aerial is steeper from higher above. No aircraft, pose or new background is added. Explicit user instructions always win."}),
            "style_category": (["All", *STYLE_CATEGORIES], {"default": "All", "tooltip": "Filter the style list by category. This browsing filter adds no prompt instructions. Changing category clears an incompatible style to None."}),
            "style": (["None", *STYLE_PRESETS], {"default": "None", "style_categories": STYLE_CATEGORIES, "tooltip": "Select a style from the chosen category. None leaves style to your prompt. Explicit user medium, colors, light and preservation instructions override this selection. Does not select a camera, change pose or add a setting."}),
        }}
    RETURN_TYPES = ("TOYXYZ_IMAGE_CAMERA", "STRING")
    # Keep the serialized socket type/class ID so saved preset links stay valid.
    RETURN_NAMES = ("preset", "preset_prompt")
    FUNCTION = "compose"
    CATEGORY = "ToyxyzTestNodes/Prompt"
    DESCRIPTION = "Prompt presets for image prompter. Supplies independent shot, angle and style guidance through the preset connection; explicit user instructions always take priority."

    def compose(self, shot_size=None, angle=None, preset=None, style="None", style_category="All"):
        # UI-only filter. Accept every known style regardless of category so API
        # callers and linked style inputs preserve their exact semantics.
        if shot_size is None:
            shot_size = "From preset / user prompt" if preset is not None else "Full body shot"
        settings = {"shot_size": shot_size, "style": style}
        if preset is not None:
            settings["preset"] = preset
        if angle is not None or preset is None:
            settings["angle"] = angle or ANGLES[0]
        result = compile_preset(settings)
        return result, result["prompt"]
