"""Pinned Danbooru vocabulary and tag-forward Anima prompt formatting."""

from __future__ import annotations

import csv
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

try:
    from rapidfuzz import fuzz, process
except ImportError:  # Exact names, aliases, and component lookup still work.
    fuzz = process = None

LOG = logging.getLogger(__name__)
DATA_PATH = Path(__file__).with_name("data") / "danbooru-2026-09-24.csv"
DATA_REVISION = "79e7d75fcef571b7c9049db4659d1cc9e3970ce9"
DATA_SHA256 = "1f64a73ac7e11b12d78d89eb5b9fc73525ee4347a182733f1db01b9cc5c85dcd"

SEMANTIC_ALIASES = {
    "1woman": "1girl",
    "1man": "1boy",
    "adult_woman": "1girl",
    "young_woman": "1girl",
    "woman": "1girl",
    "adult_man": "1boy",
    "young_man": "1boy",
    "man": "1boy",
    "slim": "skinny",
    "glossy_skin": "shiny_skin",
    "oiled_skin": "shiny_skin",
    "train_platform": "train_station_platform",
    "railway_station": "train_station",
    "backlight": "backlighting",
    "rim_light": "backlighting",
    "rim_lighting": "backlighting",
    "shoreline": "shore",
    "water_droplets": "water_drop",
    "signage": "sign",
    "bob": "bob_cut",
    "half_open_eyes": "half-closed_eyes",
    "looking_at_camera": "looking_at_viewer",
    "facing_camera": "facing_viewer",
}
QUALITY = {"masterpiece", "best quality", "good quality", "normal quality",
           "low quality", "worst quality"}
RATING = {"safe", "sensitive", "nsfw", "explicit"}
DEFAULT_STYLE = {"depth of field", "bokeh", "photorealistic", "realistic",
                 "cinematic", "high contrast", "soft focus",
                 "shallow depth of field", "blurry background"}
SCORE = re.compile(r"score_[1-9]$")
YEAR = re.compile(r"year[ _]\d{4}$")
SCENE_FORBIDDEN = re.compile(
    r"\b(?:beautiful|gorgeous|stunning|photorealistic|realistic|cinematic|"
    r"masterpiece|quality|aesthetic|vibrant|highly detailed|illustration|"
    r"anime style|blonde|brunette|hair|skin|wearing|dressed|"
    r"shirt|skirt|dress|jacket|coat|bikini|jeans|boots|"
    r"illuminating|glistening|glowing|gleaming|"
    r"rainy|sunny|overcast|sandy)\b", re.I)
SCENE_EYE_COLOR = re.compile(
    r"\b(?:blue|green|brown|black|red|purple|golden|grey|gray)"
    r"(?:\s+eyes|-eyed)\b", re.I)
CAMERA_PROP_TAG = re.compile(r"(?:^|_)camera(?:_|$)")
PHYSICAL_CAMERA_SOURCE = re.compile(
    r"\b(?:holding|holds?|carrying|carries|using|uses|placing|places|"
    r"wearing|wears|visible|physical|film|digital|video|security|"
    r"on (?:a|the) (?:table|desk|shelf|tripod))\s+(?:a |the )?camera\b|"
    r"\b(?:a|the) camera\s+(?:lies|rests|sits|stands|hangs|is on|is held)\b|"
    r"\uce74\uba54\ub77c\ub97c\s*[^.!?]{0,24}?(?:\ub4e4|\uc7a1|\uc0ac\uc6a9|\ub193|\uc124\uce58)|"
    r"\uce74\uba54\ub77c\uac00\s*(?:\ub193|\uc788|\ubcf4|\uc7a5\ucc29)", re.I)
SOURCE_BOUND_TAGS = {
    "day": re.compile(r"\b(?:day|daytime|morning|afternoon|noon|sunny)\b|\ub0ae|\uc544\uce68|\uc624\ud6c4|\uc815\uc624", re.I),
    "sunlight": re.compile(r"\b(?:sun|sunlight|sunshine|sunny|sunbeam|sunrise)\b|\ud587\ube5b|\ud587\uc0b4|\ud0dc\uc591", re.I),
    "sun": re.compile(r"\b(?:sun|sunlight|sunshine|sunny|sunbeam|sunrise)\b|\ud587\ube5b|\ud587\uc0b4|\ud0dc\uc591", re.I),
    "sunbeam": re.compile(r"\b(?:sun|sunlight|sunshine|sunny|sunbeam|sunrise)\b|\ud587\ube5b|\ud587\uc0b4|\ud0dc\uc591", re.I),
    "sunset": re.compile(r"\b(?:sunset|dusk)\b|\uc77c\ubab0|\ub099\uc591|\uc800\ub141", re.I),
    "night": re.compile(r"\b(?:night|midnight)\b|\ubc24|\uc57c\uac04", re.I),
    "steam": re.compile(r"\b(?:steam|steaming)\b|\uc218\uc99d\uae30|\uc99d\uae30", re.I),
}
SOLAR_TAG = re.compile(r"\b(?:sun|sunlight|sunshine|sunbeam|sunlit|sunrise|sunset|sunny)\b", re.I)
OPEN_AIR_SOURCE = re.compile(
    r"\b(?:outdoors?|outside|open air|beach|shore|coast|ocean|sea|"
    r"park|field|forest|street|sidewalk|road|rooftop)\b|"
    r"\uc57c\uc678|\uc2e4\uc678|\ud574\ubcc0|\ubc14\ub2f7\uac00|\uacf5\uc6d0|\uc232|\uac70\ub9ac", re.I)
SKY_SOURCE = re.compile(r"\bsky\b|\ud558\ub298", re.I)
BLUE_SKY_SOURCE = re.compile(r"\bblue sky\b|\ud478\ub978 \ud558\ub298|\ud30c\ub780 \ud558\ub298", re.I)
CLEAR_SKY_SOURCE = re.compile(r"\bclear sky\b|\ub9d1\uc740 \ud558\ub298|\uad6c\ub984 \uc5c6\ub294 \ud558\ub298", re.I)
GLARE_SOURCE = re.compile(r"\b(?:glaring|glared|glares)\s+at\b|\ub178\ub824\ubcf4|\uc9f8\ub824\ubcf4", re.I)
COASTAL_SOURCE = re.compile(
    r"\b(?:beach|shore|coast|ocean|sea|sand|waves?|seaside|seashore)\b|"
    r"\ud574\ubcc0|\ubc14\ub2f7\uac00|\ubc14\ub2e4|\ubaa8\ub798|\ud30c\ub3c4|\ud574\uc548", re.I)
COASTAL_TAGS = {"beach", "ocean", "sea", "shore", "sand", "waves", "wave",
                "sea spray", "seaweed", "seashore", "coast"}
HORIZON_SOURCE = re.compile(r"\bhorizon\b|\uc218\ud3c9\uc120|\uc9c0\ud3c9\uc120", re.I)
FOAM_SOURCE = re.compile(r"\bfoam\b|\uac70\ud488", re.I)
CLOUD_SOURCE = re.compile(r"\bclouds?\b|\uad6c\ub984", re.I)
SUN_OBJECT_SOURCE = re.compile(r"\b(?:the\s+sun|sun)\b(?!light)|\ud0dc\uc591", re.I)
SETTING_SOURCE = re.compile(
    r"\b(?:indoors?|outdoors?|outside|beach|shore|coast|ocean|sea|"
    r"park|field|forest|street|sidewalk|road|rooftop|room|bedroom|"
    r"kitchen|library|station|platform|cafe|café|restaurant|garden|"
    r"office|classroom|school|hotel|train|bus|ship|airplane|city|"
    r"village|mountain|desert|studio)\b|"
    r"\ud574\ubcc0|\ubc14\ub2e4|\uacf5\uc6d0|\uc232|\uac70\ub9ac|\ub3c4\uc11c\uad00|"
    r"\uae30\ucc28\uc5ed|\uc2b9\uac15\uc7a5|\ubc29\uc548|\uce74\ud398|\uc2e4\ub0b4|\uc57c\uc678", re.I)


def has_explicit_setting(source: str) -> bool:
    """Only an authored place opens the strong environment-expansion pass."""
    return bool(SETTING_SOURCE.search(source))
WATER_SOURCE = re.compile(r"\b(?:water|rain|river|lake|pool|ocean|sea|beach)\b|"
                          r"\ubb3c|\ube44\uac00|\ube44 \uc624|\ube57\ubb3c|\ube57\ubc29\uc6b8|"
                          r"\uac15\ubb3c|\uac15\uac00|\uac15\ubcc0|\ud638\uc218|\uc218\uc601\uc7a5|\ubc14\ub2e4|\ud574\ubcc0", re.I)
WIND_SOURCE = re.compile(r"\b(?:wind|breeze|gust|storm)\b|\ubc14\ub78c|\ub3cc\ud48d", re.I)
LIGHT_SMILE_SOURCE = re.compile(
    r"\b(?:slight|subtle|faint|small|gentle)\s+smile\b|"
    r"\uc0b4\uc9dd\s*\uc6c3|\uc870\uae08\s*\uc6c3|\uc5f7\uc740\s*\ubbf8\uc18c|"
    r"\ud76c\ubbf8\ud55c\s*\ubbf8\uc18c", re.I)
EXTRA_SUBJECT_DETAIL = re.compile(
    r"\b(?:skin|hair|eyes?|face|bob|breasts?|chest|shirt|skirt|dress|pants|"
    r"jeans|shoes|boots|sleeves?|collar|sweater|pantyhose|coat|jacket|"
    r"bikini|swimsuit|socks?|gloves?|hat|eyewear)\b", re.I)
GAZE_DETAIL = {"looking at viewer", "facing viewer", "looking away",
               "looking back", "looking up", "looking down"}
COLOR_OR_NEGATION = {"black", "white", "red", "blue", "green", "yellow",
                     "brown", "blonde", "pink", "purple", "orange", "grey",
                     "gray", "no", "without", "not"}
SUBJECT_COMPONENTS = {"hair", "skin", "eye", "eyes", "face", "breast",
                      "breasts", "shirt", "skirt", "dress", "pants", "jeans",
                      "shoe", "shoes", "boot", "boots"}


def filter_enrichment_tags(tags: list[str]) -> list[str]:
    """Keep the second creative pass from changing established subject traits."""
    return [tag for tag in tags if not EXTRA_SUBJECT_DETAIL.search(tag.replace("_", " "))]


UNLOCATED_LIGHT_TAGS = {"shadow", "shadows", "light rays", "lens flare",
                        "rim lighting", "backlighting", "silhouette", "sunbeam"}


def filter_unlocated_lighting_tags(tags: list[str]) -> list[str]:
    """Without an authored place, permit only compatible lighting additions."""
    return [tag for tag in tags if tag.strip().lower().replace("_", " ")
            in UNLOCATED_LIGHT_TAGS]


def normalize(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip().lower().replace("\\(", "(").replace("\\)", ")"))


class TagDB:
    def __init__(self, path: Path = DATA_PATH):
        self.canonical: dict[str, int] = {}
        self.counts: dict[str, int] = {}
        self.aliases: dict[str, str] = {}
        pending = []
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) != 4 or not row[0] or not row[1].isdigit():
                    raise ValueError(f"Invalid Danbooru dictionary row: {row[:2]}")
                key = normalize(row[0])
                self.canonical[key] = int(row[1])
                self.counts[key] = int(row[2])
                pending.extend((normalize(alias), key) for alias in row[3].split(",") if alias.strip())
        for alias, key in pending:
            if alias not in self.canonical:
                self.aliases.setdefault(alias, key)
        punctuation_candidates: dict[str, str | None] = {}
        for key in self.canonical:
            flattened = key.replace("-", "_")
            if flattened != key:
                if flattened in punctuation_candidates and punctuation_candidates[flattened] != key:
                    punctuation_candidates[flattened] = None
                else:
                    punctuation_candidates[flattened] = key
        self.punctuation_aliases = {flat: key for flat, key in punctuation_candidates.items()
                                    if key and flat not in self.canonical and flat not in self.aliases}
        self.fuzzy_names = tuple(key.replace("_", " ") for key, category in self.canonical.items()
                                 if category == 0 and self.counts[key] >= 20)

    def resolve(self, value: str) -> tuple[str, int] | None:
        key = normalize(value)
        key = SEMANTIC_ALIASES.get(key, key)
        canonical = (key if key in self.canonical else
                     self.aliases.get(key) or self.punctuation_aliases.get(key))
        return (canonical, self.canonical[canonical]) if canonical else None

    @lru_cache(maxsize=4096)
    def search_generated(self, value: str) -> tuple[str, int] | None:
        """Resolve an LLM candidate without guessing across unrelated concepts."""
        exact = self.resolve(value)
        if exact:
            return exact
        key = normalize(value)
        if key.startswith("hold_"):
            held = self.resolve("holding_" + key[5:])
            if held:
                return held
        if process is not None and len(key) >= 5:
            matches = process.extract(key.replace("_", " "), self.fuzzy_names,
                                      scorer=fuzz.ratio, limit=2, score_cutoff=90)
            if matches:
                best_name, best_score, _ = matches[0]
                second_score = matches[1][1] if len(matches) > 1 else 0
                candidate = normalize(best_name)
                shared = set(key.split("_")) & set(candidate.split("_"))
                if (best_score >= 92 and best_score - second_score >= 4
                        and (shared or ("_" not in key and "_" not in candidate
                                        and key[0] == candidate[0]))):
                    return candidate, self.canonical[candidate]
        parts = key.split("_")
        if (2 <= len(parts) <= 3 and parts[0] not in COLOR_OR_NEGATION
                and not {"to", "on", "in", "at", "of", "with", "from",
                         "under", "above", "behind", "between"}.intersection(parts)
                and not SUBJECT_COMPONENTS.intersection(parts)):
            head = parts[-1]
            if self.canonical.get(head) == 0 and self.counts[head] >= 1000:
                return head, 0
        return None


_DB: TagDB | None = None


def get_db() -> TagDB:
    global _DB
    if _DB is None:
        _DB = TagDB()
    return _DB


def read_response_data(response: str) -> dict:
    text = response.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.I | re.S)
    if fence:
        text = fence[1]
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Image prompter: Anima writer returned invalid JSON.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("tags"), list) or not isinstance(data.get("scene"), str):
        raise RuntimeError("Image prompter: Anima writer must return tags and scene.")
    if any(not isinstance(tag, str) for tag in data["tags"]):
        raise RuntimeError("Image prompter: Anima writer returned invalid tag candidates.")
    evidence = data.get("source_tag_evidence", [])
    data["source_tag_evidence"] = [item for item in evidence
                                   if isinstance(item, dict)
                                   and isinstance(item.get("tag"), str)
                                   and isinstance(item.get("evidence"), str)] if isinstance(evidence, list) else []
    data["scene"] = " ".join(data["scene"].split())
    return data


def read_response(response: str) -> tuple[list[str], str]:
    data = read_response_data(response)
    return data["tags"], data["scene"]


def _supported_subject_detail(tag: str, evidence: dict[str, str], source: str) -> bool:
    key = normalize(tag)
    if key in {"bob", "bob_cut"}:
        return bool(re.search(r"\bbob\s+(?:cut|haircut)\b|\ub2e8\ubc1c", source, re.I))
    if key == "short_hair" and re.search(
            r"\bshort\s+hair\b|\uc9e7\uc740.{0,12}(?:\uba38\ub9ac|\ubaa8\ubc1c)|\ub2e8\ubc1c", source, re.I):
        return True
    proof = evidence.get(normalize(tag), "").strip()
    if (not proof or (len(proof) < 2 and proof != "\uae34")
            or proof.casefold() not in source.casefold()):
        return False
    start = source.casefold().find(proof.casefold())
    local = source[max(0, start - 10):start + len(proof) + 10]
    # The cited words must name a relevant trait, garment, or gaze, not just
    # a generic person noun from the same sentence.
    if re.search(r"\bhair\b", tag, re.I):
        if not re.search(r"hair|\uba38\ub9ac|\ubaa8\ubc1c|\ud5e4\uc5b4|\uae08\ubc1c|\ud751\ubc1c", local, re.I):
            return False
        key = normalize(tag)
        if key.startswith("long_") and not re.search(r"\blong\b|\uae34|\uae38", local, re.I):
            return False
        if key.startswith("short_") and not re.search(r"\bshort\b|\uc9e7", local, re.I):
            return False
        if key.startswith("blonde_") and not re.search(r"blond|\uae08\ubc1c|\uae08\uc0c9", local, re.I):
            return False
        if key.startswith("black_") and not re.search(r"\bblack\b|\ud751\ubc1c|\uac80\uc740|\uac80\uc815", local, re.I):
            return False
        return True
    if normalize(tag).replace("_", " ") in GAZE_DETAIL:
        return bool(re.search(r"look|gaze|view|fac|\ubc14\ub77c|\uc751\uc2dc|\uc2dc\uc120|\uc815\uba74|\uce74\uba54\ub77c", proof, re.I))
    if re.search(r"\beyes?\b", tag, re.I):
        if not re.search(r"eye|\ub208", local, re.I):
            return False
        key = normalize(tag)
        if "closed" in key and not re.search(r"clos|shut|\uac10|\ubc18\ucbe4", local, re.I):
            return False
        if "open" in key and not re.search(r"open|\ub728|\ub728\uace0|\ubc18\ucbe4", local, re.I):
            return False
        return True
    return bool(re.search(
        r"skin|eye|face|breast|chest|shirt|skirt|dress|pants|jeans|shoe|boot|"
        r"sleeve|collar|sweater|pantyhose|coat|jacket|bikini|swimsuit|sock|glove|hat|"
        r"\ud53c\ubd80|\ub208|\uc5bc\uad74|\uac00\uc2b4|\uc154\uce20|\ud2f0\uc154\uce20|\ube14\ub77c\uc6b0\uc2a4|\uce58\ub9c8|"
        r"\uc2a4\ucee4\ud2b8|\ub4dc\ub808\uc2a4|\uc6d0\ud53c\uc2a4|\ubc14\uc9c0|\uccad\ubc14\uc9c0|\uc2e0\ubc1c|"
        r"\ubd80\uce20|\uc7a5\ud654|\uc2a4\uc6e8\ud130|\ub2c8\ud2b8|\uc2a4\ud0c0\ud0b9|\ucf54\ud2b8|\uc790\ucf13|"
        r"\uc7ac\ud0b7|\ube44\ud0a4\ub2c8|\uc218\uc601\ubcf5|\uc591\ub9d0|\uc7a5\uac11|\ubaa8\uc790", proof, re.I))


def _tag_entry(tag: str, db: TagDB, explicit: bool) -> str | None:
    tag = tag.strip()
    if not tag:
        return None
    # Escaped literal parentheses are already valid ComfyUI prompt syntax.
    # Dictionary lookup may normalize them for identity, but must not remove
    # the user's escaping from the emitted tag.
    if explicit and (r"\(" in tag or r"\)" in tag):
        return tag
    if tag.startswith("@"):
        resolved = db.resolve(tag[1:])
        if resolved and resolved[1] == 1:
            return "@" + resolved[0].replace("_", " ")
        return tag if explicit else None
    resolved = db.resolve(tag) if explicit else db.search_generated(tag)
    if resolved:
        if not explicit and (resolved[1] != 0 or
                             resolved[0].replace("_", " ") in QUALITY | RATING | DEFAULT_STYLE):
            return None
        if resolved[1] == 1:
            return "@" + resolved[0].replace("_", " ")
        return resolved[0].replace("_", " ")
    lowered = tag.lower().strip()
    if lowered in QUALITY | RATING | DEFAULT_STYLE or SCORE.fullmatch(lowered) or YEAR.fullmatch(lowered):
        if not explicit:
            return None
        return lowered.replace("_", " ") if YEAR.fullmatch(lowered) else lowered
    return tag if explicit else None


def _recover_explicit_detail(tag: str, evidence: dict[str, str], source: str) -> str | None:
    """Retain a short, directly cited source detail when vocabulary lookup fails."""
    proof = evidence.get(normalize(tag), "").strip()
    if not proof or len(proof) < 2 or proof.casefold() not in source.casefold():
        return None
    phrase = tag.replace("_", " ").strip()
    if (not 1 <= len(phrase.split()) <= 5 or len(phrase) > 72
            or not re.fullmatch(r"[\w\s'-]+", phrase, re.UNICODE)
            or normalize(phrase).replace("_", " ") in QUALITY | RATING | DEFAULT_STYLE
            or SCORE.fullmatch(normalize(phrase)) or YEAR.fullmatch(normalize(phrase))):
        return None
    # For English evidence, require meaningful lexical agreement as well as
    # an exact source citation. A cited generic noun must not license a new prop.
    if re.search(r"[a-z]", proof, re.I):
        words = set(re.findall(r"[a-z]{3,}", phrase.casefold()))
        cited = set(re.findall(r"[a-z]{3,}", proof.casefold()))
        if not words or not words.intersection(cited):
            return None
    return phrase


def scene_needs_repair(scene: str) -> bool:
    """Flag appearance/style drift or runaway prose without dropping source facts."""
    if not scene:
        return False
    return bool(SCENE_FORBIDDEN.search(scene) or SCENE_EYE_COLOR.search(scene)
                or len(scene.split()) > 180
                or len(re.findall(r"[.!?](?:\s|$)", scene)) > 12)


def sanitize_scene(scene: str) -> str:
    """Remove simple appearance leakage while preserving pose sentences."""
    scene = re.sub(
        r"\b(?:blue|green|brown|black|red|purple|golden|grey|gray)\s+(?=eyes\b)",
        "", scene, flags=re.I)
    scene = re.sub(
        r"\b(?:blue|green|brown|black|red|purple|golden|grey|gray)-eyed\b",
        "", scene, flags=re.I)
    scene = re.sub(
        r",?\s*(?:illuminating|highlighting|lighting)\s+(?:her|his|their)\s+"
        r"(?:\w+\s+)?(?:shirt|skirt|dress|jacket|coat|bikini|jeans|boots)\b",
        "", scene, flags=re.I)
    return " ".join(scene.split())


def normalize_viewpoint_scene(scene: str, source_text: str) -> str:
    """Keep physical cameras, but name the observer in gaze/viewpoint prose."""
    if PHYSICAL_CAMERA_SOURCE.search(source_text):
        return scene
    scene = re.sub(r"\b(look(?:s|ing|ed)?(?:\s+directly)?\s+(?:at|into|toward|towards|away from)\s+)"
                   r"(?:the\s+)?camera\b", r"\1the viewer", scene, flags=re.I)
    scene = re.sub(r"\b(back\s+to\s+)(?:the\s+)?camera\b",
                   r"\1the viewer", scene, flags=re.I)
    scene = re.sub(r"\b((?:facing|faces|face)\s+)(?:the\s+)?camera\b",
                   r"\1the viewer", scene, flags=re.I)
    scene = re.sub(r"\b(?:the\s+)?camera\s+(?:is\s+)?(?:positioned\s+|placed\s+)?"
                   r"(?=(?:behind|above|below|in front of|to the (?:left|right) of)\b)",
                   "the viewpoint is ", scene, flags=re.I)
    return scene


def format_response(input_tags: str, response: str, *, edited: bool = False,
                    source_text: str = "", enhance: str = "normal",
                    natural_language: str = "", trace: list[dict] | None = None) -> str:
    """Keep explicit tag meanings, validate generated tags, and retain scene facts."""
    db = get_db()
    data = read_response_data(response)
    candidates = data["tags"]
    scene = normalize_viewpoint_scene(sanitize_scene(data["scene"]), source_text)
    source_evidence = {}
    for item in data.get("source_tag_evidence", []):
        if isinstance(item, dict) and isinstance(item.get("tag"), str) and isinstance(item.get("evidence"), str):
            source_evidence[normalize(item["tag"])] = item["evidence"]
    explicit = [] if edited else [tag.strip() for tag in input_tags.split(",") if tag.strip()]
    old_custom = ({normalize(tag) for tag in input_tags.split(",")
                   if tag.strip() and not db.resolve(tag)} if edited else set())
    ordered = []
    seen = set()
    dropped = []
    recovered = []
    tags_only_none = not edited and enhance == "none" and not natural_language
    generated = [] if tags_only_none else candidates

    # An LLM may describe a spatial relation as one candidate although the
    # vocabulary stores the action and object separately. Split only patterns
    # with unambiguous atomic meanings; leave the relation in scene prose.
    expanded_generated = []
    for candidate in generated:
        key = normalize(candidate)
        if key.startswith("sitting_on_"):
            expanded_generated.extend(["sitting", key[len("sitting_on_"):]])
        elif key.endswith("_on_ground"):
            expanded_generated.append(key[:-len("_on_ground")])
        elif re.fullmatch(r"(?:sitting|standing|kneeling|lying|walking|running)_"
                          r"(?:girl|boy|woman|man|person)", key):
            expanded_generated.append(key.split("_", 1)[0])
        else:
            expanded_generated.append(candidate)
    generated = expanded_generated
    specific_smile = any((db.resolve(tag) or (None,))[0] == "light_smile"
                         for tag in generated)
    for tag, authored in ([(tag, True) for tag in explicit]
                          + [(tag, edited and normalize(tag) in old_custom) for tag in generated]):
        original_tag = tag
        def record(status: str, reason: str, output: str = "") -> None:
            if trace is not None:
                trace.append({"source": "input" if authored else "writer",
                              "candidate": original_tag, "output": output,
                              "status": status, "reason": reason})
        if not authored and normalize(tag) == "smile" and LIGHT_SMILE_SOURCE.search(natural_language):
            tag = "light smile"
        if not authored and normalize(tag) == "smile" and specific_smile:
            record("rejected", "more specific smile candidate")
            continue
        if (not authored and not edited
                and (EXTRA_SUBJECT_DETAIL.search(tag.replace("_", " "))
                     or normalize(tag).replace("_", " ") in GAZE_DETAIL)
                and not _supported_subject_detail(tag, source_evidence, natural_language)):
            record("rejected", "unsupported person detail")
            continue
        if not authored and normalize(tag) == "platform" and re.search(r"\btrain\b", scene, re.I):
            tag = "train station platform"
        if not authored and normalize(tag) in {"tracks", "train_tracks"} and re.search(
                r"\btrain\b|\uAE30\uCC28", source_text + " " + scene, re.I):
            tag = "railroad tracks"
        entry = _tag_entry(tag, db, authored)
        if entry is None:
            detail = (None if authored or edited else
                      _recover_explicit_detail(tag, source_evidence, natural_language))
            if detail and detail.casefold() not in scene.casefold() and normalize(detail) not in seen:
                recovered.append(detail)
                record("recovered", "dictionary miss with exact source evidence", detail)
            else:
                if tag.strip():
                    dropped.append(tag.strip())
                record("rejected", "dictionary miss or unsupported generated tag")
            continue
        identity = entry.casefold()
        dedupe_key = normalize(entry)
        if not authored:
            if (CAMERA_PROP_TAG.search(normalize(entry))
                    and not PHYSICAL_CAMERA_SOURCE.search(source_text)):
                record("rejected", "camera object not in source")
                continue
            if identity == "glaring" and not GLARE_SOURCE.search(source_text):
                record("rejected", "expression not in source")
                continue
            if identity == "sky" and not (
                    SKY_SOURCE.search(source_text) or OPEN_AIR_SOURCE.search(source_text)):
                record("rejected", "sky not supported by source")
                continue
            if identity == "outdoors" and not OPEN_AIR_SOURCE.search(source_text):
                record("rejected", "outdoor setting not in source")
                continue
            if identity == "blue sky" and not BLUE_SKY_SOURCE.search(source_text):
                record("rejected", "sky color not in source")
                continue
            if identity == "clear sky" and not CLEAR_SKY_SOURCE.search(source_text):
                record("rejected", "clear sky not in source")
                continue
            if identity in COASTAL_TAGS and not COASTAL_SOURCE.search(source_text):
                record("rejected", "coastal setting not in source")
                continue
            if identity in {"cloud", "clouds", "cloudy sky"} and not CLOUD_SOURCE.search(source_text):
                record("rejected", "cloud not in source")
                continue
            if identity == "sun" and not SUN_OBJECT_SOURCE.search(source_text):
                record("rejected", "visible sun not in source")
                continue
            if identity == "horizon" and not (
                    HORIZON_SOURCE.search(source_text) or COASTAL_SOURCE.search(source_text)
                    or OPEN_AIR_SOURCE.search(source_text)):
                record("rejected", "horizon not supported by source")
                continue
            if identity == "foam" and not (
                    FOAM_SOURCE.search(source_text) or COASTAL_SOURCE.search(source_text)):
                record("rejected", "foam not supported by source")
                continue
            if identity == "water" and not WATER_SOURCE.search(source_text):
                record("rejected", "water not in source")
                continue
            if identity == "wind" and not (
                    WIND_SOURCE.search(source_text) or OPEN_AIR_SOURCE.search(source_text)):
                record("rejected", "wind not supported by source")
                continue
        cue = SOURCE_BOUND_TAGS.get(identity)
        if not authored and cue and not cue.search(source_text):
            record("rejected", "source-bound detail not in source")
            continue
        if not authored and cue is None and SOLAR_TAG.search(identity) and not SOURCE_BOUND_TAGS["sun"].search(source_text):
            record("rejected", "solar detail not in source")
            continue
        if dedupe_key not in seen:
            ordered.append(entry)
            seen.add(dedupe_key)
            record("accepted", "authored" if authored else "dictionary match", entry)
        else:
            record("duplicate", "already emitted", entry)
    if dropped:
        LOG.warning("Image prompter Anima: %d candidates absent from dictionary or not auto-added: %s",
                    len(dropped), ", ".join(dropped[:8]))
    if tags_only_none:
        scene = ""
    elif scene_needs_repair(scene):
        clean_sentences = []
        word_count = 0
        for part in re.split(r"(?<=[.!?])\s+", scene):
            part = part.strip()
            if (not part or scene_needs_repair(part)
                    or word_count + len(part.split()) > 180
                    or len(clean_sentences) >= 12):
                continue
            clean_sentences.append(part)
            word_count += len(part.split())
        scene = " ".join(clean_sentences)
        LOG.warning("Image prompter Anima: invalid scene clauses omitted; remaining pose text retained.")
    if recovered:
        unique_details = list(dict.fromkeys(recovered))[:5]
        scene = " ".join(part for part in
                         (scene, "Specified details: " + ", ".join(unique_details) + ".") if part)
    if not ordered and not scene:
        raise RuntimeError("Image prompter: Anima writer produced no usable tags or scene.")
    return "\n".join(part for part in (", ".join(ordered), scene) if part)
