"""Static spherical camera guidance; no video, subject styling or model loading."""
import json
import logging
import math

LOG = logging.getLogger(__name__)
FOV = 40.0
RATIOS = {"1:1": 1., "3:4": .75, "4:3": 4/3, "2:3": 2/3, "3:2": 1.5, "9:16": 9/16, "16:9": 16/9}
# Illustrative mannequin-envelope coverage; not an anatomical crop or actual image geometry.
# Keep JS geometry in parity. The framing centre stays fixed across shot sizes.
PROXY_HALF_EXTENTS = (.42, .8, .22)
SHOTS = {
    "extreme wide": .22,
    "wide": .45,
    "full": .85,
    "medium-long": 1.15,
    "medium": 1.5,
    "medium close-up": 2.,
    "close-up": 4.,
    "extreme close-up": 12.,
}
# Qualitative framing, independent of the illustrative proxy's anatomy/bounds.
# Only the selected entry is sent to the writer. User crop/scale overrides it.
SHOT_FRAMING = {
    "extreme wide": "A very distant view: the subject or user-selected region appears tiny relative to the frame, much smaller than in a wide shot. Its complete outline is a minute distant mark within a vast expanse of the existing background; that surrounding area occupies almost the entire image. This is extreme distance, not merely complete inclusion with a narrow border. This specifies apparent size, not a physically smaller subject, new surroundings or a new layout. A plain background stays plain.",
    "wide": "A long shot with the camera set well back. The complete subject or user-selected region appears small but recognizable, noticeably smaller in the frame than in a full shot. Broad visible space separates its outer outline from the frame boundaries; the existing background occupies most of the image. The target remains identifiable but does not fill the image height. Preserve the requested layout and setting. A plain background stays plain.",
    "full": "A complete, uncropped view of the entire subject or user-selected region. The camera is far enough back to include its whole outline inside the frame without clipping, with clearly visible clearance between its outer extents and the frame edges. Preserve the requested pose and layout; clearance need not be symmetric and must not override an explicit user crop or edge placement. This specifies framing, not new background content.",
    "medium-long": "The subject or user-selected region is prominent within the frame, with moderate surrounding space.",
    "medium": "The subject or user-selected region is large enough to extend beyond the frame; the image shows a substantial section rather than fitting the complete outline inside it.",
    "medium close-up": "A contextual view of the subject or user-selected region: a recognizable focal section is prominent together with adjacent structure. Local context distinguishes it from an isolated close-up.",
    "close-up": "A tight view of the subject or user-selected region: one focal section fills most of the image, with only a sliver of adjacent structure. Surface features are large and readable; broader context falls outside the frame.",
    "extreme close-up": "An extreme magnification of the subject or user-selected region: only a small section containing a single tiny feature or surface fragment fills the entire image edge to edge. The rest extends outside the frame. This is much tighter than a close-up, not a broad recognizable section. If the user names a small feature, isolate that feature and its immediate surface only; do not expand to the larger structure containing it.",
}
DEFAULT = {"version": 8, "azimuth": 0., "elevation": 0., "shot": "medium"}

# Semantic shot defaults, never a pose inferred from the preview mannequin.
HUMAN_SHOT_FRAMING = {
    "extreme wide": "the complete person from head to feet is a tiny distant figure; the existing surroundings occupy almost the entire frame",
    "wide": "the complete person from head to feet is small but recognizable, with broad surrounding space",
    "full": "the complete person from the top of the head to both feet is visible, with clear space above and below",
    "medium-long": "a cowboy shot includes the top of the head down to about mid-thigh, with the lower frame edge around mid-thigh",
    "medium": "a waist-up shot includes the complete head and upper body down to the waist, with the lower frame edge around the waist",
    "medium close-up": "a chest-up shot includes the complete head, shoulders and upper chest, with the lower frame edge around the chest",
    "close-up": "the head and face dominate the frame, with the forehead and chin included and very little surrounding space; retain the head view appropriate to the selected camera angle",
    "extreme close-up": "one tiny visible head detail, such as one eye or the lips in a compatible view, fills the frame rather than the complete face; rear views use a visible rear head detail instead of hidden facial features",
}


def shot_framing(shot):
    return (SHOT_FRAMING[shot] + " Human shot preset (whole-person targets only): " + HUMAN_SHOT_FRAMING[shot]
            + ". Explicit user target, crop and scale override this preset. Non-human/selected parts use generic framing. "
            "Preserve pose, gaze, clothing and background; never turn or reposition the subject to reveal hidden surfaces.")


def snap(value, step):
    # Symmetric half-step ties; JS Math.round alone differs for negative ties.
    return math.copysign(math.floor(abs(value)/step+.5)*step, value) if value else 0.


def number(value, default, low, high):
    try:
        value = float(value)
        return min(high, max(low, value)) if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def normalize(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            LOG.warning("Image camera: unreadable settings; using defaults.")
            value = {}
    value = value if isinstance(value, dict) else {}
    result = dict(DEFAULT)
    for key, low, high in [("azimuth", -180, 180), ("elevation", -90, 90)]:
        result[key] = number(value.get(key), DEFAULT[key], low, high)
    result["azimuth"] = snap(result["azimuth"], 45)
    result["elevation"] = snap(result["elevation"], 30)
    # Removed custom-distance settings fall back to medium; never retain a hidden override.
    shot = value.get("shot")
    if shot == "cowboy":
        shot = "medium-long"
    result["shot"] = shot if isinstance(shot, str) and shot in SHOTS else DEFAULT["shot"]
    return result


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def geometry(settings):
    s = normalize(settings)
    az, el = (math.radians(s[k]) for k in ("azimuth", "elevation"))
    direction = [math.sin(az)*math.cos(el), math.sin(el), math.cos(az)*math.cos(el)]
    right = [math.cos(az), 0., -math.sin(az)]
    up = [-math.sin(az)*math.sin(el), math.cos(el), -math.cos(az)*math.sin(el)]
    coverage = SHOTS[s["shot"]]
    target = [0., .9, 0.]
    tan = math.tan(math.radians(FOV)/2)
    aspect = 1.  # Square diagnostic viewport, never an output aspect-ratio instruction.
    distance = .15
    for x in (-PROXY_HALF_EXTENTS[0], PROXY_HALF_EXTENTS[0]):
        for y in (-PROXY_HALF_EXTENTS[1], PROXY_HALF_EXTENTS[1]):
            for z in (-PROXY_HALF_EXTENTS[2], PROXY_HALF_EXTENTS[2]):
                delta = [x, y, z]
                fit = max(abs(dot(delta, right))/(tan*aspect), abs(dot(delta, up))/tan)/coverage
                distance = max(distance, dot(delta, direction)+fit)
    position = [target[i]+direction[i]*distance for i in range(3)]
    return {"position": position, "target": target, "up": up, "right": right, "direction": direction,
            "distance": distance, "fov": FOV, "aspect": aspect}


def compile_camera(settings):
    s = normalize(settings)
    g = geometry(s)
    yaw = abs(s["azimuth"])
    side = "left" if s["azimuth"] > 0 else "right"
    if yaw < 15:
        view = "a straight-on frontal viewpoint"
    elif yaw < 75:
        view = f"a front three-quarter viewpoint from the subject's {side} side"
    elif yaw <= 105:
        view = f"a side-profile viewpoint from the subject's {side} side"
    elif yaw < 165:
        view = f"a rear three-quarter viewpoint from the subject's {side} side"
    else:
        view = "a straight rear viewpoint"
    elevation = s["elevation"]
    if abs(elevation) >= 89.5:
        angle = "directly overhead, looking straight down" if elevation > 0 else "directly beneath the subject, looking straight up"
        view = "an overhead viewpoint" if elevation > 0 else "an underside viewpoint"
    elif abs(elevation) < .5:
        angle = "with a level viewing direction"
    else:
        angle = {30: "from an elevated viewpoint, looking slightly downward",
                 60: "at a high angle, looking steeply downward",
                 -30: "at a low angle, looking upward",
                 -60: "from a worm's-eye viewpoint, looking steeply upward"}[elevation]
    # Only viewpoint, viewing angle and shot scale. The user determines the
    # photographed subject/part and crop; never choose them from proxy geometry.
    shot_name = "cowboy" if s['shot'] == "medium-long" else s['shot']
    prompt = f"Static camera framing: {shot_name} shot, from {view}, {angle}. {shot_framing(s['shot'])}"
    # Add the visible result, not a second ambiguous camera-left/right label.
    # At exact front/rear or vertical poles no lateral cue is meaningful.
    side_cue = ""
    if 0 < yaw < 180 and abs(elevation) < 90:
        image_side = "left" if s["azimuth"] > 0 else "right"
        facing = "toward the camera" if yaw < 90 else "away from the camera" if yaw > 90 else "across the image"
        side_cue = (f"The subject's own {side} side is nearer the camera; its front points "
                    f"toward image-{image_side}, {facing}.")
        prompt += " " + side_cue
    return {"version": DEFAULT["version"], "settings": s, "geometry": g, "prompt": prompt,
            "components": {"viewpoint": view, "viewpoint_side_cue": side_cue,
                           "viewing_angle": angle, "shot_size": s["shot"],
                           "framing": shot_framing(s["shot"])}}


def camera_guidance(value):
    if not isinstance(value, dict):
        return ""
    if value.get("kind") == "image_camera_preset":
        from .image_camera_presets import compile_preset
        return compile_preset(value.get("settings"))["camera_prompt"]
    # Recompile settings: never treat an arbitrary incoming prompt field as system instructions.
    return compile_camera(value.get("settings", {}))["prompt"] if "settings" in value else ""


def camera_components(value):
    """Independent defaults, rebuilt from settings rather than trusting supplied prose."""
    if not isinstance(value, dict) or "settings" not in value:
        return None
    if value.get("kind") == "image_camera_preset":
        from .image_camera_presets import compile_preset
        return compile_preset(value["settings"])["components"]
    return compile_camera(value["settings"])["components"]


class ImageCamera:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scene_data": ("STRING", {"default": json.dumps(DEFAULT), "multiline": True})}}

    RETURN_TYPES = ("TOYXYZ_IMAGE_CAMERA", "STRING")
    RETURN_NAMES = ("camera", "camera_prompt")
    FUNCTION = "compose"
    CATEGORY = "ToyxyzTestNodes/Prompt"
    DESCRIPTION = "Static spherical viewpoint and shot framing for image prompter. No video or camera movement."

    def compose(self, scene_data):
        camera = compile_camera(scene_data)
        return (camera, camera["prompt"])
