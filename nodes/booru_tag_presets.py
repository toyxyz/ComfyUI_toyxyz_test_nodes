"""Camera tags and pose-neutral view descriptions; preserve authored text."""
import math
import re
import random


def strength_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return 1.0
    return max(0.0, min(10.0, value)) if math.isfinite(value) else 1.0

CAMERA_ANGLES = {
    "None": (),
    "Dutch angle": ("dutch_angle",),
    "Sideways": ("sideways",),
    "Upside-down": ("upside-down",),
}
VERTICAL_VIEWS = {
    "None": (),
    "Above": ("from_above",),
    "Below": ("from_below",),
}
HORIZONTAL_VIEWS = {
    "None": (),
    "Front": ("straight-on",),
    "Behind": ("from_behind",),
    "Side": ("from_side",),
    "Left side": ("from_side",),
    "Right side": ("from_side",),
    "Front-left 45°": (),
    "Front-right 45°": (),
    "Rear-left 45°": ("from_behind",),
    "Rear-right 45°": ("from_behind",),
}
LEGACY_VERTICAL_VIEWS = {"Above 45°": "Above", "Below 45°": "Below",
                         "Bird's-eye view": "Above", "Worm's-eye view": "Below"}
PERSPECTIVE_DEPTH = {
    "None": (), "Perspective": ("perspective",), "Fisheye": ("fisheye",),
    "Atmospheric perspective": ("atmospheric_perspective",),
    "Vanishing point": ("vanishing_point",), "Panorama": ("panorama",),
    "Foreshortening": ("foreshortening",), "Isometric": ("isometric",),
}
FOCUS_BLUR = {key: {"None": (), "Enabled": (key,)} for key in
              ("depth_of_field", "blurry_background", "blurry_foreground", "bokeh", "soft_focus",
               "chromatic_aberration", "lens_flare", "motion_blur")}
SUBJECT_FACING = {
    "None": "",
    "Screen left": "The person faces left. Their face and torso are oriented toward the left edge of the image.",
    "Screen right": "The person faces right. Their face and torso are oriented toward the right edge of the image.",
}

# Describe only the viewing relationship, never a subject action, pose, gaze,
# wardrobe, environment or anatomical target inferred from camera position.
VIEW_DESCRIPTIONS = {
    "vertical_view": {
        "Above": "High-angle view, looking downward at the subject.",
        "Below": "Low-angle view, looking upward at the subject.",
    },
    "horizontal_view": {
        "Front": "Frontal view of the subject.",
        "Behind": "Rear view of the subject.",
        "Side": "Side-on view of the subject.",
        "Left side": "The subject is seen from its own left side, with its left-side surfaces nearer.",
        "Right side": "The subject is seen from its own right side, with its right-side surfaces nearer.",
        # One orientation clause, not repeated face/body/detail views. Long
        # weighted multi-part descriptions produced duplicate portraits/insets.
        # Avoid the noun "screen": under strong weighting the model can draw
        # a physical screen/panel beside the subject, even without any prop.
        **{f"Front-{side} 45°": f"Front three-quarter view of the subject facing diagonally toward the {side}." for side in ("left", "right")},
        **{f"Rear-{side} 45°": f"Rear three-quarter view of the subject facing diagonally away toward the {side}." for side in ("left", "right")},
    },
    "zoom": {
        # Framing is tag-only: high-weight space/frame prose generated white
        # margins, insets or multiple panels. Never add subject-count rules.
        "Very wide shot": "",
        "Wide shot": "",
        "Full body": "",
        "Cowboy shot": "",
        "Upper body": "",
        "Portrait": "",
        "Close-up": "",
        "Lower body": "",
    },
    "camera_angle": {
        "Dutch angle": "The image plane is tilted diagonally, with the whole scene rotated together.",
        "Sideways": "The image plane is rotated by a quarter turn, with the whole scene rotated together.",
        "Upside-down": "The image plane is rotated by a half turn, with the whole scene rotated together.",
    },
}


def effective_settings(settings):
    """Translate old continuous panel values into discrete descriptive selections."""
    settings = dict(settings)
    vertical = settings.get("vertical_view")
    if isinstance(vertical, str):
        settings["vertical_view"] = LEGACY_VERTICAL_VIEWS.get(vertical, vertical)
    if not settings.get("panel_enabled"):
        return settings
    resolved = dict(settings)
    x, y, z, roll = (axis_value(settings.get(k, 0)) for k in ("pos_x", "pos_y", "pos_z", "roll"))
    resolved.update(panel_enabled=False,
        horizontal_view="Behind" if abs(x) > .75 else "Left side" if x > .25 else "Right side" if x < -.25 else "Front",
        vertical_view="Above" if y > .2 else "Below" if y < -.2 else "None",
        zoom="Close-up" if z < -.6 else "Portrait" if z < -.2 else "Upper body" if z <= .2 else "Cowboy shot" if z <= .4 else "Full body" if z <= .7 else "Wide shot",
        camera_angle="Dutch angle" if abs(roll) >= .15 else "None")
    return resolved
ZOOMS = {
    "None": (),
    "Very wide shot": ("very_wide_shot",),
    "Wide shot": ("wide_shot",),
    "Full body": ("full_body",),
    "Cowboy shot": ("cowboy_shot",),
    "Upper body": ("upper_body",),
    "Portrait": ("portrait",),
    "Close-up": ("close-up",),
    "Lower body": ("lower_body",),
}


def axis_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return max(-1.0, min(1.0, value)) if math.isfinite(value) else 0.0


def preset_tags(preset):
    """Rebuild from allowlisted options; supplied payload prose is not trusted."""
    if not isinstance(preset, dict) or preset.get("kind") != "booru_tag_preset":
        return ""
    settings = preset.get("settings")
    if not isinstance(settings, dict):
        return ""
    if not settings.get("camera_enabled", True):
        return ""
    settings = effective_settings(settings)
    tags = []
    for key, choices in (("camera_angle", CAMERA_ANGLES),
                         ("vertical_view", VERTICAL_VIEWS),
                         ("horizontal_view", HORIZONTAL_VIEWS), ("zoom", ZOOMS),
                         ("perspective_depth", PERSPECTIVE_DEPTH), *FOCUS_BLUR.items()):
        value = settings.get(key)
        if isinstance(value, str):
            strength = strength_value(settings.get(key + "_strength", 1.0))
            selected = choices.get(value, ())
            tags.extend(selected if strength == 1.0 else
                        (f"({tag}:{strength:.2f})" for tag in selected))
    return ", ".join(tags)


def preset_prompt(preset):
    """Allowlisted tags plus concise view prose, independent of scene content."""
    tags = preset_tags(preset)
    if not isinstance(preset, dict) or preset.get("kind") != "booru_tag_preset" or not isinstance(preset.get("settings"), dict):
        return ""
    if not preset["settings"].get("camera_enabled", True):
        return ""
    settings = effective_settings(preset["settings"])
    clauses = []
    for key in ("horizontal_view", "vertical_view", "zoom", "camera_angle"):
        strength = strength_value(settings.get(key + "_strength", 1))
        if strength <= 0:
            continue  # Zero weight must not reintroduce the instruction in prose.
        value = settings.get(key)
        # Anatomical left/right captions previously defeated screen-facing cues.
        # Keep the side camera class, but let explicitly selected screen facing
        # specify the visible orientation, not a competing anatomical cue.
        direction = ""
        if key == "horizontal_view" and value in ("Left side", "Right side"):
            facing = "Screen left" if value == "Left side" else "Screen right"
            direction = SUBJECT_FACING[facing] if not settings.get("suppress_subject_facing") else ""
            value = "Side"
        clause = VIEW_DESCRIPTIONS[key].get(value) if isinstance(value, str) else None
        if key == "horizontal_view" and isinstance(value,str) and "45°" in value and settings.get("suppress_subject_facing"):
            clause = ("Front" if value.startswith("Front") else "Rear") + " three-quarter view of the subject."
        if clause:
            clause += (" " + direction) if direction else ""
            clauses.append(clause if strength == 1 else f"({clause}:{strength:.2f})")
    if not clauses:
        return tags
    return (tags + "\n" if tags else "") + " ".join(clauses)


def authored_camera_fields(text):
    """Conservative explicit camera cues, never infer a view from a subject pose."""
    groups = (("camera_angle", CAMERA_ANGLES), ("vertical_view", VERTICAL_VIEWS),
              ("horizontal_view", HORIZONTAL_VIEWS), ("zoom", ZOOMS),
              ("perspective_depth", PERSPECTIVE_DEPTH), *FOCUS_BLUR.items())
    tokens = set()
    for token in re.split(r"[,;\n]", text):
        token = token.strip().lower()
        weighted = re.fullmatch(r"\((.*):[+-]?(?:\d+(?:\.\d*)?|\.\d+)\)", token)
        tokens.add((weighted.group(1) if weighted else token).replace("_", " "))
    fields = {key for key, choices in groups if any(
        tag.replace("_", " ") in tokens for tags in choices.values() for tag in tags)}
    if tokens & {"from left", "from right"}: fields.add("horizontal_view")
    if tokens & {"bird's eye view", "bird's-eye view", "worm's eye view", "worm's-eye view"}:
        fields.add("vertical_view")
    patterns = {
        "vertical_view": r"\b(?:high-angle (?:view|shot)|low-angle (?:view|shot)|overhead view|view from (?:above|below)|(?:camera|viewpoint|lens) [^.!?\n]{0,60}looking (?:straight |steeply )?(?:downward|upward))\b",
        "horizontal_view": r"\b(?:frontal view|rear view|side-on view|(?:front|rear) three-quarter view|view from (?:behind|the front)|seen from (?:its|her|his|their) own (?:left|right) side)\b",
        "zoom": r"\b(?:full-body shot|head-and-shoulders framing|tight close-up|wide framing|upper-body framing|lower-body framing)\b",
        "camera_angle": r"\b(?:image plane|image roll|camera roll)\b",
    }
    for key, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE): fields.add(key)
    return fields


def append_preset_tags(tags, preset):
    # Do not append conflicting defaults over recognized explicit authored camera
    # instructions. No source edits, scene inference or blocking validation.
    if isinstance(preset, dict) and isinstance(preset.get("settings"), dict):
        settings = dict(effective_settings(preset["settings"]))
        for field in authored_camera_fields(tags):
            settings[field] = "None"
        if re.search(r"\b(?:faces|facing)\s+(?:diagonally\s+)?(?:away\s+)?(?:(?:toward(?:s)?|to)\s+)?(?:the\s+|screen[ -]|image[ -])?(?:left|right)\b", tags.replace("_", " "), re.IGNORECASE):
            settings["suppress_subject_facing"] = True
        preset = {**preset, "settings": settings}
    suffix = preset_prompt(preset)
    if not suffix:
        return tags
    if not tags.strip():
        return tags + suffix
    # Preserve every source character, including weights and trailing whitespace.
    separator = "\n" if tags.endswith("\n") or tags.rstrip().endswith((".", "!", "?")) or not preset_tags(preset) else " " if tags.rstrip().endswith((",", ";")) else ", "
    return tags + separator + suffix


class BooruTagPresets:
    RANDOM_CHOICES = {"camera_angle": CAMERA_ANGLES, "vertical_view": VERTICAL_VIEWS,
                      "horizontal_view": HORIZONTAL_VIEWS, "zoom": ZOOMS,
                      "perspective_depth": PERSPECTIVE_DEPTH, **FOCUS_BLUR}
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "camera_angle": ([*CAMERA_ANGLES, "Random"], {"default": "None",
                "tooltip": "Image rotation, not subject leaning. None adds nothing."}),
            "vertical_view": ([*VERTICAL_VIEWS, "Random"], {"default": "None",
                "tooltip": "View from above/below. These are category tags, not physical lens-height or tilt controls."}),
            "horizontal_view": ([*HORIZONTAL_VIEWS, "Random"], {"default": "None",
                "tooltip": "Left/Right side combine a side camera view with the person's face and torso facing screen left/right. Side leaves orientation unspecified. This does not mean image placement. The preview is a camera proxy; generation accuracy depends on the model."}),
            "zoom": ([*ZOOMS, "Random"], {"default": "None",
                "tooltip": "Visible subject coverage, not a numeric lens zoom. Close-up does not specify a target body part."}),
            **{key + "_strength": ("FLOAT", {
                "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.01,
                "tooltip": "Weight from 0 to 10 for all tags and the complete natural-language instruction generated by this selection. This is emphasis, not a physical angle.",
            }) for key in ("camera_angle", "vertical_view", "horizontal_view", "zoom")},
            "panel_enabled": ("BOOLEAN", {"default": False}),
            "camera_enabled": ("BOOLEAN", {"default": True}),
            **{key: ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01})
               for key in ("pos_x", "pos_y", "pos_z", "roll")},
        }, "optional": {
            "subject_facing": (list(SUBJECT_FACING), {"default": "None",
                "tooltip": "Legacy compatibility input, hidden and ignored. Use Horizontal view Left side or Right side instead."}),
            # Append optional widgets to preserve older serialized widget order.
            "perspective_depth": ([*PERSPECTIVE_DEPTH, "Random"], {"default": "None",
                "tooltip": "Tag-only depth or projection effect. Foreshortening depicts perspective shortening. Isometric uses non-converging parallel lines, not a calibrated 45-degree camera angle. Does not set subject count, pose, scenery or output dimensions. Not simulated in the preview."}),
            "perspective_depth_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.01,
                "tooltip": "Weight for the selected perspective/depth tag only. High weights may distort the image. The preview geometry is unchanged."}),
            **{name: definition for key, choices in list(FOCUS_BLUR.items())[:5] for name, definition in (
                (key, ([*choices, "Random"], {"default": "None", "tooltip": "Independent tag-only focus/blur effect. Combine effects as needed. Does not set subject count, pose or scenery. Not simulated in the 3D preview."})),
                (key + "_strength", ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Weight for this focus/blur tag only. High weights can reduce image detail."})))},
            "random_camera": ("BOOLEAN", {"default": False,
                "tooltip": "Legacy compatibility input, hidden and ignored. Use Random at the end of each selection list."}),
            "focus_blur_random": ("BOOLEAN", {"default": False,
                "tooltip": "Randomize the Focus / Blur combination on every execution. Overrides manual effect selections, preserves weights, and may select no effects."}),
            "focus_blur_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.01,
                "tooltip": "Shared weight for every selected Focus / Blur tag, including random combinations."}),
            # New effects follow existing optional widgets to preserve saved values.
            **{key: ([*choices, "Random"], {"default": "None",
                "tooltip": "Tag-only optical or motion effect. Uses the shared Focus / Blur strength. Not simulated in the 3D preview."})
               for key, choices in list(FOCUS_BLUR.items())[5:]},
        }}

    RETURN_TYPES = ("TOYXYZ_BOORU_PRESET",)
    RETURN_NAMES = ("camera",)
    FUNCTION = "build_camera"
    CATEGORY = "ToyxyzTestNodes/Prompt"
    DESCRIPTION = "Append camera tags and view prose without sorting source text. Horizontal Left/Right include face/torso screen direction. Each selection weight applies to all its tags and complete prose. No wardrobe or background presets."

    def build_camera(self, **kwargs):
        camera = self.compose(**kwargs)[0]
        if kwargs.get("focus_blur_random") or any(kwargs.get(key) == "Random" for key in self.RANDOM_CHOICES):
            return {"ui": {"camera_settings": [camera["settings"]]}, "result": (camera,)}
        return (camera,)

    @classmethod
    def IS_CHANGED(cls, random_camera=False, camera_enabled=True, **kwargs):
        return float("nan") if camera_enabled and (kwargs.get("focus_blur_random") or any(kwargs.get(key) == "Random" for key in cls.RANDOM_CHOICES)) else False

    def compose(self, camera_angle="None", vertical_view="None",
                horizontal_view="None", zoom="None", camera_angle_strength=2.0,
                vertical_view_strength=2.0, horizontal_view_strength=2.0, zoom_strength=2.0,
                panel_enabled=False, camera_enabled=True, pos_x=0.0, pos_y=0.0, pos_z=0.0, roll=0.0,
                subject_facing="None", perspective_depth="None", perspective_depth_strength=2.0,
                random_camera=False, focus_blur_random=False, focus_blur_strength=None, **focus_options):
        if isinstance(vertical_view, str):
            vertical_view = LEGACY_VERTICAL_VIEWS.get(vertical_view, vertical_view)
        settings = {}
        for key, value, choices in (
                ("camera_angle", camera_angle, CAMERA_ANGLES),
                ("vertical_view", vertical_view, VERTICAL_VIEWS),
                ("horizontal_view", horizontal_view, HORIZONTAL_VIEWS),
                ("zoom", zoom, ZOOMS), ("perspective_depth", perspective_depth, PERSPECTIVE_DEPTH)):
            settings[key] = value if isinstance(value, str) and (value in choices or value == "Random") else "None"
        for key, value in (("camera_angle", camera_angle_strength),
                           ("vertical_view", vertical_view_strength),
                           ("horizontal_view", horizontal_view_strength), ("zoom", zoom_strength),
                           ("perspective_depth", perspective_depth_strength)):
            settings[key + "_strength"] = strength_value(value)
        settings.update(panel_enabled=bool(panel_enabled), camera_enabled=bool(camera_enabled),
                        **{key: axis_value(value) for key, value in
                           (("pos_x", pos_x), ("pos_y", pos_y), ("pos_z", pos_z), ("roll", roll))})
        settings["subject_facing"] = subject_facing if isinstance(subject_facing, str) and subject_facing in SUBJECT_FACING else "None"
        for key, choices in FOCUS_BLUR.items():
            value = focus_options.get(key, "None")
            settings[key] = value if isinstance(value, str) and (value in choices or value == "Random") else "None"
            settings[key + "_strength"] = strength_value(focus_blur_strength if focus_blur_strength is not None else focus_options.get(key + "_strength", 2.0))
        settings["focus_blur_strength"] = strength_value(focus_blur_strength if focus_blur_strength is not None else 2.0)
        if camera_enabled:
            chooser = random.SystemRandom()
            previous = getattr(self, "_last_camera_choices", {})
            for key, choices in self.RANDOM_CHOICES.items():
                if key in FOCUS_BLUR:
                    continue
                if settings[key] == "Random":
                    candidates = [value for value in choices if value not in ("None", "Random")]
                    different = [value for value in candidates if value != previous.get(key)]
                    settings[key] = chooser.choice(different or candidates)
                    settings["panel_enabled"] = False
                if settings[key] != "None":
                    previous[key] = settings[key]
            self._last_camera_choices = previous
            if focus_blur_random or any(settings[key] == "Random" for key in FOCUS_BLUR):
                last = getattr(self, "_last_focus_blur_mask", None)
                mask = chooser.choice([value for value in range(1 << len(FOCUS_BLUR)) if value != last])
                for index, key in enumerate(FOCUS_BLUR):
                    settings[key] = "Enabled" if mask & (1 << index) else "None"
                settings["panel_enabled"] = False
                self._last_focus_blur_mask = mask
        result = {"kind": "booru_tag_preset", "version": 10, "settings": settings}
        return result, preset_tags(result), preset_prompt(result)
