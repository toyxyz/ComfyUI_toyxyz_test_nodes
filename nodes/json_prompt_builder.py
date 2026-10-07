"""Model-specific caption serialization and strictly text-only enhancement."""
from __future__ import annotations

import copy
import json
import math
import re

FORMAT_NAMES = ("Ideogram 4", "Ming Image")
SOCKET_TYPE = "TOYXYZ_JSON_PROMPT"


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _colors(value, limit=16):
    if not isinstance(value, list):
        raise ValueError("Colors must be an array of #RRGGBB strings.")
    out = []
    for color in value:
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("Colors must use six-digit #RRGGBB notation.")
        if color.upper() not in out:
            out.append(color.upper())
    if len(out) > limit:
        raise ValueError(f"This palette allows at most {limit} colors.")
    return out


def _string(value):
    if not isinstance(value, str):
        raise ValueError("Descriptions and visible text must be strings.")
    return value


def _get(document, path):
    for key in path:
        document = document[key]
    return document


def _set(document, path, value):
    owner = _get(document, path[:-1]) if path[:-1] else document
    owner[path[-1]] = value


def _fields(document, model):
    paths = []
    if model == "Ideogram 4":
        if "high_level_description" in document:
            paths.append(["high_level_description"])
        paths += [["style_description", key] for key, value in document.get("style_description", {}).items() if isinstance(value, str)]
        paths.append(["compositional_deconstruction", "background"])
        paths += [["compositional_deconstruction", "elements", i, "desc"] for i in range(len(document["compositional_deconstruction"]["elements"]))]
    elif model == "Ming Image":
        paths += [["canvas_settings", key] for key in ("ambient_lighting", "image_style")]
        for i in range(len(document["layers"])):
            paths += [["layers", i, key] for key in ("description", "hierarchy_and_relation")]
    else:
        raise ValueError("Unsupported JSON target model.")
    return paths


def validate_document(document, model):
    """Validate native schema; this does not validate scene collisions or aesthetics."""
    if not isinstance(document, dict):
        raise ValueError("The caption must be a JSON object.")
    if model == "Ming Image":
        if list(document) != ["canvas_settings", "layers"]:
            raise ValueError("Ming requires canvas_settings followed by layers.")
        canvas = document["canvas_settings"]
        if not isinstance(canvas, dict) or list(canvas) != ["aspect_ratio", "ambient_lighting", "image_style"]:
            raise ValueError("Invalid Ming canvas_settings fields.")
        if not re.fullmatch(r"[1-9]\d*:[1-9]\d*", _string(canvas["aspect_ratio"])):
            raise ValueError("Invalid Ming aspect ratio.")
        _string(canvas["ambient_lighting"]); _string(canvas["image_style"])
        if not isinstance(document["layers"], list):
            raise ValueError("Ming layers must be an array.")
        for layer in document["layers"]:
            if not isinstance(layer, dict) or list(layer) != ["description", "coordinates", "hierarchy_and_relation", "color_specs"]:
                raise ValueError("Invalid Ming layer fields.")
            _string(layer["description"]); _string(layer["hierarchy_and_relation"])
            match = re.fullmatch(r"cx: (\d\.\d{3}), cy: (\d\.\d{3}), w: (\d\.\d{3}), h: (\d\.\d{3})", _string(layer["coordinates"]))
            if not match:
                raise ValueError("Ming coordinates must be a formatted string.")
            cx, cy, w, h = map(float, match.groups())
            if not all(0 <= v <= 1 for v in (cx, cy, w, h)) or w <= 0 or h <= 0 or cx-w/2 < -.001 or cy-h/2 < -.001 or cx+w/2 > 1.001 or cy+h/2 > 1.001:
                raise ValueError("Invalid Ming coordinates.")
            _colors(layer["color_specs"])
    elif model == "Ideogram 4":
        if set(document) - {"high_level_description", "style_description", "compositional_deconstruction"}:
            raise ValueError("Unexpected Ideogram top-level fields.")
        if "high_level_description" in document:
            _string(document["high_level_description"])
        if "style_description" in document:
            style = document["style_description"]
            if not isinstance(style, dict):
                raise ValueError("Invalid Ideogram style.")
            keys = ["aesthetics", "lighting", "photo", "medium"] if "photo" in style else ["aesthetics", "lighting", "medium", "art_style"]
            if list(style) != keys + (["color_palette"] if "color_palette" in style else []):
                raise ValueError("Invalid Ideogram style field order.")
            for key in keys:
                _string(style[key])
            _colors(style.get("color_palette", []))
        cd = document.get("compositional_deconstruction")
        if not isinstance(cd, dict) or list(cd) != ["background", "elements"] or not isinstance(cd["elements"], list):
            raise ValueError("Invalid Ideogram composition.")
        _string(cd["background"])
        for element in cd["elements"]:
            if not isinstance(element, dict) or element.get("type") not in ("obj", "text"):
                raise ValueError("Invalid Ideogram element type.")
            expected = ["type", "bbox"] + (["text"] if element["type"] == "text" else []) + ["desc"]
            if list(element) != expected + (["color_palette"] if "color_palette" in element else []):
                raise ValueError("Invalid Ideogram element fields.")
            bbox = element["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4 or any(type(v) is not int or not 0 <= v <= 1000 for v in bbox) or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                raise ValueError("Invalid Ideogram bounding box.")
            _string(element["desc"])
            if element["type"] == "text":
                _string(element["text"])
            _colors(element.get("color_palette", []), 5)
    else:
        raise ValueError("Unsupported JSON target model.")
    dumps(document)


def compile_builder(target_model, width, height, scene="", background="", style_mode="photo", style="", aesthetics="", lighting="", medium="", regions_data="[]", style_colors="[]"):
    if target_model not in FORMAT_NAMES or type(width) is not int or type(height) is not int or not 64 <= width <= 16384 or not 64 <= height <= 16384:
        raise ValueError("Invalid model or canvas size.")
    regions = json.loads(regions_data)
    colors = _colors(json.loads(style_colors))
    if not isinstance(regions, list) or len(regions) > 128:
        raise ValueError("Regions must be an array with at most 128 entries.")
    normalized = []
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("Each region must be an object.")
        coordinates = [region.get(key) for key in ("x", "y", "w", "h")]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in coordinates):
            raise ValueError("Region coordinates must be finite numbers.")
        x, y, w, h = coordinates
        if w <= 0 or h <= 0 or min(x, y) < 0 or x+w > 1.000001 or y+h > 1.000001:
            raise ValueError("Each region must have a positive size inside the canvas.")
        kind = region.get("type", "obj")
        if kind not in ("obj", "text"):
            raise ValueError("Region type must be obj or text.")
        normalized.append(dict(x=x, y=y, w=w, h=h, type=kind, desc=_string(region.get("desc", "")),
                               text=_string(region.get("text", "")), relation=_string(region.get("relation", "")), palette=_colors(region.get("palette", []), 5 if target_model == "Ideogram 4" else 16)))
    bindings = []
    if target_model == "Ideogram 4":
        document = {}
        if scene:
            document["high_level_description"] = _string(scene)
        if style_mode not in ("none", "photo", "art_style"):
            raise ValueError("Unknown style mode.")
        if style_mode != "none":
            sd = {"aesthetics": aesthetics, "lighting": lighting}
            if style_mode == "photo":
                sd.update(photo=style, medium=medium)
            else:
                sd.update(medium=medium, art_style=style)
            if colors:
                sd["color_palette"] = colors
            document["style_description"] = sd
        elements = []
        for r in normalized:
            x, y, w, h = (r[k] for k in ("x", "y", "w", "h"))
            element = {"type": r["type"], "bbox": [round(y*1000), round(x*1000), round((y+h)*1000), round((x+w)*1000)]}
            if r["type"] == "text":
                element["text"] = r["text"]
            element["desc"] = " ".join(part for part in (r["desc"], r["relation"]) if part)
            if r["palette"]:
                element["color_palette"] = r["palette"]
            elements.append(element)
        document["compositional_deconstruction"] = {"background": background, "elements": elements}
    else:
        divisor = math.gcd(width, height)
        document = {"canvas_settings": {"aspect_ratio": f"{width//divisor}:{height//divisor}", "ambient_lighting": lighting,
                                        "image_style": " ".join(part for part in (style, aesthetics, medium) if part)},
                    "layers": [{"description": background, "coordinates": "cx: 0.500, cy: 0.500, w: 1.000, h: 1.000",
                                "hierarchy_and_relation": "Full-canvas background behind all other layers.", "color_specs": colors}]}
        for r in normalized:  # UI order is back to front, matching Ming text-to-image.
            desc = r["desc"]
            index = len(document["layers"])
            if r["type"] == "text" and r["text"]:
                # Quotes/backslashes are escaped once by the outer JSON serializer,
                # never as extra characters inside the rendered-copy description.
                quoted = '"' + r["text"] + '"'
                desc = (desc + " Text reads exactly " + quoted + ".").strip()
                bindings.append({"path": ["layers", index, "description"], "literal": quoted, "token": f"__VISIBLE_TEXT_{index}__"})
            document["layers"].append({"description": desc, "coordinates": f"cx: {r['x']+r['w']/2:.3f}, cy: {r['y']+r['h']/2:.3f}, w: {r['w']:.3f}, h: {r['h']:.3f}",
                                      "hierarchy_and_relation": r["relation"], "color_specs": r["palette"]})
    validate_document(document, target_model)
    return {"version": 1, "target_model": target_model, "document": document, "bindings": bindings,
            "scene_context": _string(scene), "width": width, "height": height}


def checked_payload(payload):
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Connect a json prompter builder output.")
    payload = copy.deepcopy(payload)
    validate_document(payload.get("document"), payload.get("target_model"))
    paths = _fields(payload["document"], payload["target_model"])
    for binding in payload.get("bindings", []):
        if binding["path"] not in paths or not binding["literal"] or not re.fullmatch(r"__VISIBLE_TEXT_\d+__", binding["token"]) or _get(payload["document"], binding["path"]).count(binding["literal"]) != 1:
            raise ValueError("Invalid visible-text binding.")
    return payload


def text_fields(payload):
    fields = []
    for i, path in enumerate(_fields(payload["document"], payload["target_model"])):
        value = _get(payload["document"], path)
        for binding in payload.get("bindings", []):
            if binding["path"] == path:
                value = value.replace(binding["literal"], binding["token"])
        fields.append({"id": f"f{i}", "path": path, "text": value})
    return fields


def build_json_messages(payload, instruction, enhance="none", evidence=None, editing=False, defaults=None):
    levels = {"none": "Translate to English and organize without adding details.",
              "normal": "Develop useful compatible visual details only where unspecified.",
              "strong": "Develop rich concrete visual details where unspecified; avoid repetition and invented entities."}
    if enhance not in levels:
        raise ValueError("Unknown enhancement level.")
    system = ("You edit text fields inside an image caption. Return ONLY JSON with exactly one key fields: "
              "an array of objects with exactly id and text. Return every supplied id exactly once. "
              "Never output a full caption, coordinates, extra fields, or commentary. "
              "Write editable descriptions in fluent English at EVERY enhancement level. Translate source descriptions even when the user instruction is already English. "
              "Only protected visible-copy tokens and proper names may retain another language. "
              "Preserve explicit facts and their owners, counts, pose, clothing, negations, framing and relations. "
              "Do not add people, objects, layers, visible lettering, or a new composition. "
              "Keep every __VISIBLE_TEXT_N__ token character-for-character exactly once in its original field. "
              "Never copy these tokens to another field. Do not translate or duplicate visible copy. "
              "Use relation fields only for ownership, alignment, containment, stacking and occlusion. "
              "Geometry and palettes are immutable; keep descriptions compatible with them. "
              "Incorporate explicit actions and scene facts from scene_context and user_instruction into the appropriate existing description fields. "
              "Do not discard these instructions just because their facts are absent from a field's initial text. "
              "Never assign one region's appearance or action to another region. "
              "Optional camera/style defaults apply only when compatible with explicit user text and locked layout. They never override the user or change a bounding box. "
              + ("The edit request overrides conflicting source descriptions, but never locked geometry, palettes or visible copy. Replace obsolete text consistently; preserve unrelated wording and facts. " if editing else levels[enhance]))
    return [{"role": "system", "content": system}, {"role": "user", "content": dumps({
        "target_model": payload["target_model"], "scene_context": "" if editing else payload.get("scene_context", ""),
        "user_instruction": instruction, "locked_caption": payload["document"],
        "fields": text_fields(payload), "reference_evidence": evidence, "optional_defaults": defaults})}]


def protected_shape(document, model):
    result = copy.deepcopy(document)
    for path in _fields(document, model):
        _set(result, path, "__EDITABLE__")
    return dumps(result)  # key order is protected too


def validate_override(payload, candidate):
    validate_document(candidate, payload["target_model"])
    if protected_shape(candidate, payload["target_model"]) != protected_shape(payload["document"], payload["target_model"]):
        raise ValueError("JSON edit changed protected structure, geometry, palette or visible text.")
    for binding in payload.get("bindings", []):
        for path in _fields(candidate, payload["target_model"]):
            count = _get(candidate, path).count(binding["literal"])
            expected = _get(payload["document"], path).count(binding["literal"])
            if count != expected:
                raise ValueError("JSON edit changed or duplicated visible text.")
    return candidate


def apply_text_patch(payload, response):
    patch = json.loads(response)
    if not isinstance(patch, dict) or list(patch) != ["fields"] or not isinstance(patch["fields"], list):
        raise ValueError("Expected a fields-only JSON patch.")
    fields = {field["id"]: field for field in text_fields(payload)}
    seen = set()
    document = copy.deepcopy(payload["document"])
    for item in patch["fields"]:
        if not isinstance(item, dict) or set(item) != {"id", "text"} or not isinstance(item["id"], str) or item["id"] not in fields or item["id"] in seen:
            raise ValueError("Unknown, duplicate or malformed text field.")
        seen.add(item["id"])
        value = _string(item["text"])
        field = fields[item["id"]]
        for binding in payload.get("bindings", []):
            if binding["path"] == field["path"]:
                if value.count(binding["token"]) != 1:
                    raise ValueError("Visible-text placeholder was changed or duplicated.")
                value = value.replace(binding["token"], binding["literal"])
        if re.search(r"__VISIBLE_TEXT_\d+__", value):
            raise ValueError("Visible-text placeholder moved to another field.")
        _set(document, field["path"], value)
    if seen != set(fields):
        raise ValueError("The writer omitted text fields.")
    return validate_override(payload, document)


class JsonPrompterBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "target_model": (list(FORMAT_NAMES),),
            "width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16}),
            "scene": ("STRING", {"default": "", "multiline": True, "tooltip": "Overall scene context. Region positions and visible text stay locked."}),
            "background": ("STRING", {"default": "", "multiline": True}),
            "style_mode": (["photo", "art_style", "none"],),
            "style": ("STRING", {"default": ""}),
            "aesthetics": ("STRING", {"default": ""}),
            "lighting": ("STRING", {"default": ""}),
            "medium": ("STRING", {"default": ""}),
            "regions_data": ("STRING", {"default": "[]", "multiline": True}),
            "style_colors": ("STRING", {"default": "[]"}),
        }}

    RETURN_TYPES = (SOCKET_TYPE, "STRING", "INT", "INT")
    RETURN_NAMES = ("builder", "json", "width", "height")
    FUNCTION = "build"
    CATEGORY = "ToyxyzTestNodes/Prompt"
    DESCRIPTION = "Place regions and write descriptions for Ideogram 4 or Ming Image. Connect builder to image prompter. Regions are layout guidance, not generation masks."

    def build(self, **settings):
        payload = compile_builder(**settings)
        return {"ui": {"text": [dumps(payload["document"])]}, "result": (payload, dumps(payload["document"]), settings["width"], settings["height"])}
