import base64
import copy
import glob
import json
import math
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from typing import Any


MODEL_FPS = 24
MIN_SHOT_DURATION = 0.25
CURRENT_PROJECT_VERSION = 22
SUPPORTED_MODES = ("AUTO", "T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA")
SUPPORTED_DIALOGUE_MODES = ("spoken", "voiceover", "singing")
SUPPORTED_TRANSITIONS = ("cut", "cross-dissolve", "fade", "wipe")
SUPPORTED_LANGUAGES = (
    "Arabic", "Chinese", "English", "French", "German", "Italian",
    "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
)
REFERENCE_ROLES = {
    "picture": ("first_frame", "last_frame", "frame", "subject_identity"),
    "video": ("none", "video_editing", "video_continuation", "motion", "camera", "cuts_rhythm"),
    "audio": (
        "none", "full_signal_copy", "partial_signal_copy", "voice_delivery",
        "dialogue_lyrics", "sound_ambience", "music_rhythm",
    ),
}
SUBJECT_STRENGTHS = ("weak", "normal", "strong")
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
MAX_REF_FILES = 12
REF_VIDEO_MIN_SECONDS = 10 / MODEL_FPS
REF_VIDEO_MAX_SECONDS = 15.0
REF_VIDEO_TOTAL_SECONDS = 15.0
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
VIDEO_UPLOAD_MAX_BYTES = 2 * 1024 * 1024 * 1024
VIDEO_ANALYSIS_MAX_FRAMES = 10
DEFAULT_ENHANCE_MODEL_ID = "hf:JonathanColetti/Qwen3.8-27B-Uncensored-GGUF/Qwen3.8-27B-Uncensored-Q4_K_M.gguf"
DEFAULT_ENHANCE_MODEL_REPO = "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF"
DEFAULT_ENHANCE_MODEL_FILE = "Qwen3.8-27B-Uncensored-Q4_K_M.gguf"
DEFAULT_ENHANCE_MODEL_SIZE = 16810714528
QWEN_IMAGE_MODEL_ID = "hf:JonathanColetti/Qwen3.8-27B-Uncensored-GGUF/Q4_K_M+vision-f16"
DEFAULT_IMAGE_MODEL_ID = QWEN_IMAGE_MODEL_ID
QWEN_IMAGE_MODEL_REPO = DEFAULT_ENHANCE_MODEL_REPO
QWEN_IMAGE_MODEL_FILE = DEFAULT_ENHANCE_MODEL_FILE
QWEN_IMAGE_MMPROJ_FILE = "Qwen3.8-27B-Uncensored-vision-f16.gguf"
QWEN_IMAGE_MODEL_SIZE = DEFAULT_ENHANCE_MODEL_SIZE
QWEN_IMAGE_MMPROJ_SIZE = 927606912
QWEN_MODEL_DISPLAY_NAME = "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF · Q4_K_M + Vision F16"
QWEN_MODEL_VRAM_LABEL = "VRAM ≈ 20–22 GB"
LEGACY_LIGHTX2V_MODEL_ID = "hf:lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B"
LIGHTX2V_MODEL_ID = "hf:indhic-ai/MiniMax_H3-Prompt_Rewriter-8B-LORA-Merged-GGUF/Q8_0+vision-f16"
LIGHTX2V_MODEL_REPO = "indhic-ai/MiniMax_H3-Prompt_Rewriter-8B-LORA-Merged-GGUF"
LIGHTX2V_MODEL_FILE = "minimax-h3-prompt-rewriter-8b-Q8_0.gguf"
LIGHTX2V_MMPROJ_FILE = "mmproj-minimax-h3-prompt-rewriter-8b-f16.gguf"
LIGHTX2V_MODEL_SIZE = 8710000000
LIGHTX2V_MMPROJ_SIZE = 1160000000
LIGHTX2V_TOTAL_SIZE = LIGHTX2V_MODEL_SIZE + LIGHTX2V_MMPROJ_SIZE
LIGHTX2V_MODEL_DISPLAY_NAME = "indhic-ai/MiniMax_H3-Prompt_Rewriter-8B-LORA-Merged-GGUF · Q8_0 + Vision F16"
LIGHTX2V_MODEL_VRAM_LABEL = "VRAM ≈ 10–12 GB"
LIGHTX2V_SUPPORTED_MODES = ("T2VA", "I2VA", "FL2VA", "L2VA")
LIGHTX2V_SYSTEM_PROMPTS_PATH = os.path.join(
    os.path.dirname(__file__), "minimax_h3_lightx2v_system_prompts.json"
)
BASE_ENHANCE_MAX_NEW_TOKENS = 1800
RICH_ENHANCE_MAX_NEW_TOKENS = 3072
REF_ENHANCE_MAX_NEW_TOKENS = 3072
ENHANCE_CONTEXT_SIZE = 16384
_ENHANCE_LOCK = threading.Lock()
_ENHANCE_JOBS: dict[str, dict[str, Any]] = {}
_ENHANCE_JOBS_LOCK = threading.Lock()
_ENHANCE_CANCEL_EVENTS: dict[str, threading.Event] = {}
_ENHANCE_STOPPERS: dict[str, Any] = {}


class EnhancementCancelled(RuntimeError):
    pass


def _set_enhance_job(job_id: str, **values: Any) -> None:
    if not job_id:
        return
    with _ENHANCE_JOBS_LOCK:
        job = _ENHANCE_JOBS.setdefault(job_id, {})
        job.update(values)
        job["updated_at"] = time.time()


def _get_enhance_job(job_id: str) -> dict[str, Any]:
    with _ENHANCE_JOBS_LOCK:
        return dict(_ENHANCE_JOBS.get(job_id, {}))


def _begin_enhance_job(job_id: str) -> threading.Event:
    event = threading.Event()
    with _ENHANCE_JOBS_LOCK:
        _ENHANCE_CANCEL_EVENTS[job_id] = event
        _ENHANCE_STOPPERS.pop(job_id, None)
    return event


def _set_enhance_stopper(job_id: str, stopper: Any | None) -> None:
    with _ENHANCE_JOBS_LOCK:
        if stopper is None:
            _ENHANCE_STOPPERS.pop(job_id, None)
        else:
            _ENHANCE_STOPPERS[job_id] = stopper


def _cancel_enhance_job(job_id: str) -> bool:
    with _ENHANCE_JOBS_LOCK:
        event = _ENHANCE_CANCEL_EVENTS.get(job_id)
        stopper = _ENHANCE_STOPPERS.get(job_id)
        if event is not None:
            event.set()
    if stopper is not None:
        try:
            stopper()
        except Exception:
            pass
    if event is not None:
        _set_enhance_job(job_id, stage="cancelled", message="Prompt generation stopped by the user.")
    return event is not None


def _finish_enhance_job(job_id: str) -> None:
    with _ENHANCE_JOBS_LOCK:
        _ENHANCE_CANCEL_EVENTS.pop(job_id, None)
        _ENHANCE_STOPPERS.pop(job_id, None)


DEFAULT_PROJECT = {
    "version": CURRENT_PROJECT_VERSION,
    "mode": "AUTO",
    "requested_duration": 5.0,
    "user_request": "",
    "shots": [
        {
            "id": "shot-1",
            "duration": 5.0,
            "visual_action": "",
        }
    ],
    "references": [],
    "constraints": "",
    "verbatim_content": "",
    "enhance_model": DEFAULT_ENHANCE_MODEL_ID,
    "image_model": DEFAULT_IMAGE_MODEL_ID,
    "auto_run": False,
    "enhance": False,
    "enhanced_prompt": "",
}


CAMERA_SENTENCES = {
    "Static Shot": "The camera holds a static shot.",
    "Push In": "The camera pushes in toward the subject.",
    "Pull Out": "The camera pulls out from the subject.",
    "Zoom In": "The camera zooms in while remaining stationary.",
    "Zoom Out": "The camera zooms out while remaining stationary.",
    "Pan Left": "The camera pans left.",
    "Pan Right": "The camera pans right.",
    "Truck Left": "The camera trucks left.",
    "Truck Right": "The camera trucks right.",
    "Tilt Up": "The camera tilts up.",
    "Tilt Down": "The camera tilts down.",
    "Pedestal Up": "The camera pedestals up.",
    "Pedestal Down": "The camera pedestals down.",
    "Arc Shot": "The camera moves in an arc around the subject.",
    "Tracking Shot": "The camera tracks the moving subject.",
    "POV": "The shot uses the subject's point of view.",
    "Shake Slightly": "The camera shakes slightly.",
    "Shake Strongly": "The camera shakes strongly.",
    "Roll Clockwise": "The camera rolls clockwise.",
    "Roll Counterclockwise": "The camera rolls counterclockwise.",
}

CAMERA_ANGLE_SENTENCES = {
    "Eye Level Shot": "The camera uses an eye-level angle.",
    "Low Angle Shot": "The camera uses a low angle, looking upward at the subject.",
    "High Angle Shot": "The camera uses a high angle, looking downward at the subject.",
    "Bird's-Eye View": "The camera uses an extreme bird's-eye view from directly above.",
    "Worm's-Eye View": "The camera uses an extreme worm's-eye view from below.",
    "Overhead Shot": "The camera is positioned overhead and looks down on the scene.",
    "Top-Down Shot": "The camera points straight down in a top-down view.",
    "Ground-Level Shot": "The camera is positioned at ground level.",
    "Dutch Angle Shot": "The camera uses a Dutch angle with a visibly tilted horizon.",
    "Over-the-Shoulder Shot": "The camera uses an over-the-shoulder angle.",
    "Point-of-View Shot": "The camera shows the scene from the subject's point of view.",
    "Aerial Shot": "The camera uses a high aerial angle over the scene.",
    "Three-Quarter Angle": "The camera views the subject from a three-quarter angle.",
    "Profile Angle": "The camera views the subject in profile from the side.",
    "Rear Angle": "The camera views the subject from behind.",
}

CAMERA_FRAMING_SENTENCES = {
    "Extreme Close-Up Shot": "The composition uses an extreme close-up, isolating a very small facial or subject detail.",
    "Close-Up Shot": "The composition uses a close-up, filling the frame with the subject's face or primary detail.",
    "Medium Close-Up Shot": "The composition uses a medium close-up, framing the subject approximately from the chest upward.",
    "Medium Shot": "The composition uses a medium shot, framing the subject approximately from the waist upward.",
    "Cowboy Shot": "The composition uses a cowboy shot, framing the subject approximately from mid-thigh upward.",
    "Medium Long Shot": "The composition uses a medium long shot, framing most of the subject while retaining environmental context.",
    "Long Shot": "The composition uses a long shot, showing the full subject with substantial surrounding environment.",
    "Full Shot": "The composition uses a full shot, keeping the subject's entire body visible in frame.",
    "Wide Shot": "The composition uses a wide shot, emphasizing the environment and the subject's spatial context.",
}

TRANSITION_SENTENCES = {
    "cut": "cut to a new shot.",
    "cross-dissolve": "cross-dissolve into a new shot.",
    "fade": "fade into a new shot.",
    "wipe": "wipe into a new shot.",
}


SYSTEM_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "minimax_h3_system_prompts.json")


def _load_system_prompt_config(path: str = SYSTEM_PROMPTS_PATH) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to load MiniMax H3 system prompts from {path}: {error}") from error

    if not isinstance(config, dict):
        raise RuntimeError(f"MiniMax H3 system prompt config must be a JSON object: {path}")
    common = config.get("common")
    common_enhanced = config.get("common_enhanced")
    enhance_addendum = config.get("enhance_addendum", "")
    action_semantics = config.get("action_semantics")
    static_asset_rules = config.get("static_asset_rules")
    common_addendum = config.get("common_addendum", "")
    video_reference_common = config.get("video_reference_common")
    video_reference_roles = config.get("video_reference_roles")
    audio_reference_common = config.get("audio_reference_common")
    audio_reference_roles = config.get("audio_reference_roles")
    base = config.get("base")
    mode_addenda = config.get("mode_addenda", {})
    modes = config.get("modes")
    if (
        not isinstance(common, str)
        or not isinstance(common_enhanced, str)
        or not isinstance(enhance_addendum, str)
        or not isinstance(action_semantics, str)
        or not isinstance(static_asset_rules, dict)
        or not isinstance(common_addendum, str)
        or not isinstance(video_reference_common, str)
        or not isinstance(video_reference_roles, dict)
        or not isinstance(audio_reference_common, str)
        or not isinstance(audio_reference_roles, dict)
        or not isinstance(base, str)
        or not isinstance(mode_addenda, dict)
        or not isinstance(modes, dict)
    ):
        raise RuntimeError(
            "MiniMax H3 system prompt config requires strings 'common', 'common_enhanced', "
            "'common_addendum', 'enhance_addendum', 'action_semantics', 'video_reference_common', "
            "'audio_reference_common', and 'base', objects 'video_reference_roles', "
            "'audio_reference_roles', 'static_asset_rules', 'mode_addenda', "
            f"and 'modes': {path}"
        )

    expected_video_roles = set(REFERENCE_ROLES["video"])
    if set(video_reference_roles) != expected_video_roles or any(
        not isinstance(video_reference_roles[key], str) for key in expected_video_roles
    ):
        raise RuntimeError(
            "MiniMax H3 video_reference_roles must contain one string for every supported video role: "
            f"{path}"
        )

    expected_audio_roles = set(REFERENCE_ROLES["audio"])
    if set(audio_reference_roles) != expected_audio_roles or any(
        not isinstance(audio_reference_roles[key], str) for key in expected_audio_roles
    ):
        raise RuntimeError(
            "MiniMax H3 audio_reference_roles must contain one string for every supported audio role: "
            f"{path}"
        )

    required_static_rules = {"common", "enhanced", "FL2VA", "FL2VA_enhanced"}
    if set(static_asset_rules) != required_static_rules or any(
        not isinstance(static_asset_rules[key], str) for key in required_static_rules
    ):
        raise RuntimeError(
            "MiniMax H3 static_asset_rules requires string keys common, enhanced, FL2VA, "
            f"and FL2VA_enhanced: {path}"
        )

    expected_modes = set(SUPPORTED_MODES) - {"AUTO"}
    if set(modes) != expected_modes:
        missing = sorted(expected_modes - set(modes))
        extra = sorted(set(modes) - expected_modes)
        raise RuntimeError(
            f"MiniMax H3 system prompt modes do not match supported modes; missing={missing}, extra={extra}: {path}"
        )
    if any(not isinstance(modes[mode], str) for mode in expected_modes):
        raise RuntimeError(f"Every MiniMax H3 mode prompt must be a string: {path}")
    unknown_addenda = set(mode_addenda) - expected_modes
    if unknown_addenda or any(not isinstance(value, str) for value in mode_addenda.values()):
        raise RuntimeError(
            f"MiniMax H3 mode addenda must be strings for supported modes only: {path}"
        )
    return {
        "common": common,
        "common_enhanced": common_enhanced,
        "enhance_addendum": enhance_addendum,
        "action_semantics": action_semantics,
        "static_asset_rules": static_asset_rules,
        "common_addendum": common_addendum,
        "video_reference_common": video_reference_common,
        "video_reference_roles": video_reference_roles,
        "audio_reference_common": audio_reference_common,
        "audio_reference_roles": audio_reference_roles,
        "base": base,
        "mode_addenda": mode_addenda,
        "modes": modes,
    }


SYSTEM_PROMPT_CONFIG = _load_system_prompt_config()
COMMON_LLM_SYSTEM_RULES = SYSTEM_PROMPT_CONFIG["common"]
COMMON_LLM_SYSTEM_RULES += SYSTEM_PROMPT_CONFIG["action_semantics"]
COMMON_LLM_SYSTEM_RULES += SYSTEM_PROMPT_CONFIG["static_asset_rules"]["common"]
COMMON_LLM_SYSTEM_RULES += SYSTEM_PROMPT_CONFIG["common_addendum"]
ENHANCED_COMMON_LLM_SYSTEM_RULES = SYSTEM_PROMPT_CONFIG["common_enhanced"]
ENHANCED_COMMON_LLM_SYSTEM_RULES += SYSTEM_PROMPT_CONFIG["action_semantics"]
ENHANCED_COMMON_LLM_SYSTEM_RULES += SYSTEM_PROMPT_CONFIG["static_asset_rules"]["common"]
ENHANCED_COMMON_LLM_SYSTEM_RULES += SYSTEM_PROMPT_CONFIG["static_asset_rules"]["enhanced"]
ENHANCED_COMMON_LLM_SYSTEM_RULES += SYSTEM_PROMPT_CONFIG["common_addendum"]
BASE_LLM_SYSTEM_RULES = SYSTEM_PROMPT_CONFIG["base"]
MODE_LLM_SYSTEM_PROMPTS = {
    mode: COMMON_LLM_SYSTEM_RULES
    + ("" if mode == "REF2VA" else BASE_LLM_SYSTEM_RULES)
    + SYSTEM_PROMPT_CONFIG["modes"][mode]
    + SYSTEM_PROMPT_CONFIG["mode_addenda"].get(mode, "")
    + (SYSTEM_PROMPT_CONFIG["static_asset_rules"]["FL2VA"] if mode == "FL2VA" else "")
    for mode in SUPPORTED_MODES
    if mode != "AUTO"
}
ENHANCED_MODE_LLM_SYSTEM_PROMPTS = {
    mode: ENHANCED_COMMON_LLM_SYSTEM_RULES
    + ("" if mode == "REF2VA" else BASE_LLM_SYSTEM_RULES)
    + SYSTEM_PROMPT_CONFIG["modes"][mode]
    + SYSTEM_PROMPT_CONFIG["mode_addenda"].get(mode, "")
    + (SYSTEM_PROMPT_CONFIG["static_asset_rules"]["FL2VA"] if mode == "FL2VA" else "")
    + (SYSTEM_PROMPT_CONFIG["static_asset_rules"]["FL2VA_enhanced"] if mode == "FL2VA" else "")
    + SYSTEM_PROMPT_CONFIG["enhance_addendum"]
    for mode in SUPPORTED_MODES
    if mode != "AUTO"
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_alias(value: Any) -> str:
    alias = _clean_text(value).lstrip("@").strip()
    alias = re.sub(r"\s+", "_", alias)
    alias = re.sub(r"[^\w-]", "", alias, flags=re.UNICODE)
    return f"@{alias}" if alias else ""


def _number(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _contains_hangul(text: str) -> bool:
    return any("\uac00" <= character <= "\ud7a3" for character in text)


_INLINE_QUOTE_RE = re.compile(
    r'"([^"\r\n]+)"|“([^”\r\n]+)”|‘([^’\r\n]+)’|「([^」\r\n]+)」'
)
_VOCAL_CUE_RE = re.compile(
    r"말(?:한다|하며|하고|했다|하다|해)|외치|소리치|속삭|노래|부르|대사|보이스오버|"
    r"\b(?:say|says|said|speak|speaks|shout|shouts|whisper|whispers|sing|sings|voiceover)\b",
    flags=re.IGNORECASE,
)
_VISIBLE_TEXT_CUE_RE = re.compile(
    r"간판|화면|자막|문구|표지|라벨|텍스트|쓰여|적혀|표시|"
    r"\b(?:sign|screen|caption|subtitle|label|banner|visible text|reads|written|displayed)\b",
    flags=re.IGNORECASE,
)


def _inferred_dialogue_language(text: str) -> str:
    if re.search(r"[\uac00-\ud7a3\u3131-\u318e]", text):
        return "Korean"
    if re.search(r"[\u3040-\u30ff]", text):
        return "Japanese"
    if re.search(r"[\u0600-\u06ff]", text):
        return "Arabic"
    if re.search(r"[\u0400-\u04ff]", text):
        return "Russian"
    if re.search(r"[\u3400-\u9fff]", text):
        return "Chinese"
    return "English"


def _nearest_cue_distance(text: str, start: int, end: int, pattern: re.Pattern[str]) -> int | None:
    distances = []
    for match in pattern.finditer(text):
        if match.end() < start:
            distances.append(start - match.end())
        elif match.start() > end:
            distances.append(match.start() - end)
        else:
            distances.append(0)
    return min(distances) if distances else None


def _speaker_phrase_from_context(text: str, position: int) -> str:
    before = text[:position]
    candidates: list[tuple[int, str]] = []
    patterns = (
        (r"여성|여자", "the woman"),
        (r"남성|남자", "the man"),
        (r"소녀", "the girl"),
        (r"소년", "the boy"),
        (r"아이", "the child"),
        (r"\bwoman\b", "the woman"),
        (r"\bman\b", "the man"),
        (r"\bgirl\b", "the girl"),
        (r"\bboy\b", "the boy"),
        (r"\bchild\b", "the child"),
    )
    for pattern, phrase in patterns:
        matches = list(re.finditer(pattern, before, flags=re.IGNORECASE))
        if matches:
            candidates.append((matches[-1].start(), phrase))
    return max(candidates, default=(-1, "the on-screen speaker"))[1]


def _input_content_locks(project: dict[str, Any]) -> list[str]:
    """Extract lightweight pre-generation locks without adding another LLM pass."""
    detected: list[dict[str, Any]] = []
    generic_vocal_shots: list[int] = []
    for shot_number, shot in enumerate(project.get("shots", []), 1):
        action = _clean_text(shot.get("visual_action"))
        if not action:
            continue
        occupied: list[tuple[int, int]] = []
        for match in re.finditer(
            r"<d>\s*\[([^\]\r\n]+)\]\s*(.*?)\s*</d>",
            action,
            flags=re.DOTALL | re.IGNORECASE,
        ):
            prefix = action[max(0, match.start() - 120):match.start()]
            speaker_match = re.search(r"\((S[1-6](?:,S[1-6])*)\)[^()]*$", prefix, flags=re.IGNORECASE)
            identity_match = re.search(
                r"(?:^|[.!?\n])\s*([^.!?\n]{1,100}?)\s*\(S[1-6](?:,S[1-6])*\)[^()]*$",
                prefix,
                flags=re.IGNORECASE,
            )
            lowered_prefix = prefix.lower()
            if "says in an off-screen voiceover:" in lowered_prefix:
                vocal_form = "says in an off-screen voiceover:"
            elif "sings:" in lowered_prefix:
                vocal_form = "sings:"
            else:
                vocal_form = "says:"
            detected.append({
                "shot": shot_number,
                "kind": "vocal",
                "words": match.group(2).strip(),
                "language": match.group(1).strip(),
                "speaker": speaker_match.group(1).upper() if speaker_match else "",
                "speaker_phrase": identity_match.group(1).strip() if identity_match else _speaker_phrase_from_context(
                    action, match.start()
                ),
                "vocal_form": vocal_form,
            })
            occupied.append(match.span())

        for match in _INLINE_QUOTE_RE.finditer(action):
            if any(start <= match.start() and match.end() <= end for start, end in occupied):
                continue
            words = next(group for group in match.groups() if group is not None).strip()
            if not words:
                continue
            vocal_distance = _nearest_cue_distance(action, match.start(), match.end(), _VOCAL_CUE_RE)
            visible_distance = _nearest_cue_distance(action, match.start(), match.end(), _VISIBLE_TEXT_CUE_RE)
            if vocal_distance is None and visible_distance is None:
                continue
            kind = "vocal" if visible_distance is None or (
                vocal_distance is not None and vocal_distance <= visible_distance
            ) else "visible"
            local_context = action[max(0, match.start() - 80):min(len(action), match.end() + 80)].lower()
            if re.search(r"노래|부르|\b(?:sing|sings|sang)\b", local_context):
                vocal_form = "sings:"
            elif re.search(r"보이스오버|voiceover", local_context):
                vocal_form = "says in an off-screen voiceover:"
            else:
                vocal_form = "says:"
            detected.append({
                "shot": shot_number,
                "kind": kind,
                "words": words,
                "language": _inferred_dialogue_language(words) if kind == "vocal" else "",
                "speaker": "",
                "speaker_phrase": _speaker_phrase_from_context(action, match.start()),
                "vocal_form": vocal_form,
            })
        if _VOCAL_CUE_RE.search(action) and not any(
            item["shot"] == shot_number and item["kind"] == "vocal" for item in detected
        ):
            generic_vocal_shots.append(shot_number)

    vocal_items = [item for item in detected if item["kind"] == "vocal"]
    if len(vocal_items) == 1 and not vocal_items[0]["speaker"]:
        vocal_items[0]["speaker"] = "S1"

    locks: list[str] = []
    for item in detected:
        exact_words = json.dumps(item["words"], ensure_ascii=False)
        if item["kind"] == "visible":
            locks.append(
                f"[Shot {item['shot']}] visible text: preserve {exact_words} verbatim in English double quotes; "
                "never put it in <d> or assign a speaker ID."
            )
            continue
        speaker = f"({item['speaker']})" if item["speaker"] else "a stable parenthesized (Sx) ID"
        tag = f"<d>[{item['language']}] {item['words']}</d>"
        if item["speaker"]:
            source_rule = (
                "In REF2VA, identify a referenced visible speaker with the applicable <Subject N> label; "
                "do not replace it with a generic noun. "
                if str(project.get("mode", "")).upper() == "REF2VA"
                else ""
            )
            locks.append(
                f"[Shot {item['shot']}] vocal lock: copy this block character-for-character, including every "
                f"space and punctuation mark: `{tag}`. Use speaker ID {speaker} once before the block and the "
                f"vocal form `{item['vocal_form']}` once. Precede {speaker} with a visible speaker identity; "
                f"never begin the clause with bare {speaker}. {source_rule}For on-screen speech or singing, add only one "
                "short scene-specific lip-synchronization sentence. Never copy this instruction, a checklist, "
                "or hypothetical event examples into the answer."
            )
        else:
            locks.append(
                f"[Shot {item['shot']}] vocal content: the identifying phrase must include {speaker} before "
                f"the vocal block; reproduce exactly `{tag}` without changing spacing or punctuation. Unless "
                "the input explicitly says voiceover, make this diegetic speech physically produced by the "
                "visible speaker: the mouth must articulate every syllable in precise synchronization with the "
                "voice and the lips may close only after the line ends."
            )
    for shot_number in generic_vocal_shots:
        locks.append(
            f"[Shot {shot_number}] contains explicit vocal content in visual_action: the first vocal source "
            "must use (S1), and every vocal line must retain its inferred language and exact supplied words in "
            "<d>. Unless explicitly requested as voiceover, the visible speaker must physically produce the "
            "diegetic voice with natural mouth movement synchronized to every syllable; the lips close only "
            "after the line ends."
        )
    return locks


def align_frame_count(seconds: float) -> int:
    frames = max(5, int(round(seconds * MODEL_FPS)))
    while frames % 17 != 5:
        frames += 1
    return frames


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:06.3f}"


def _normalize_shot(raw: Any, index: int) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    visual_action = _clean_text(raw.get("visual_action"))

    # Version 12 and earlier stored dialogue and visible text in dedicated UI
    # fields. Preserve those projects by moving their content into the unified
    # natural-language Visual / action input exactly once.
    legacy_dialogue = _clean_text(raw.get("dialogue"))
    if legacy_dialogue:
        language = _clean_text(raw.get("dialogue_language"))
        language = language if language in SUPPORTED_LANGUAGES else "English"
        speaker = _clean_text(raw.get("dialogue_speaker")).upper().strip("()")
        speaker = speaker if re.fullmatch(r"S[1-6]", speaker) else "S1"
        delivery = _clean_text(raw.get("dialogue_delivery")) or "The on-screen speaker"
        mode = _clean_text(raw.get("dialogue_mode")).lower()
        mode = mode if mode in SUPPORTED_DIALOGUE_MODES else "spoken"
        wrapped = re.fullmatch(
            r"<d>\s*(?:\[([^\]\r\n]+)\]\s*)?(.*?)\s*</d>",
            legacy_dialogue,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if wrapped:
            supplied_language, legacy_dialogue = wrapped.groups()
            if supplied_language in SUPPORTED_LANGUAGES:
                language = supplied_language
        else:
            legacy_dialogue = re.sub(r"</?d>", "", legacy_dialogue, flags=re.IGNORECASE).strip()
        if mode == "voiceover":
            migrated = (
                f"{delivery} ({speaker}) says in an off-screen voiceover: "
                f"<d>[{language}] {legacy_dialogue}</d> while the corresponding "
                "on-screen character's lips remain completely closed."
            )
        elif mode == "singing":
            migrated = f"{delivery} ({speaker}) sings: <d>[{language}] {legacy_dialogue}</d>"
        else:
            migrated = f"{delivery} ({speaker}) says: <d>[{language}] {legacy_dialogue}</d>"
        visual_action = "\n".join(part for part in (visual_action, migrated) if part)

    legacy_visible_text = _clean_text(raw.get("visible_text"))
    if legacy_visible_text:
        visible_instruction = (
            f'A visible on-screen text element reads "{_quoted_prompt_text(legacy_visible_text)}".'
        )
        visual_action = "\n".join(part for part in (visual_action, visible_instruction) if part)

    legacy_diegetic_sound = _clean_text(raw.get("diegetic_sound"))
    if legacy_diegetic_sound:
        sound_instruction = f"Synchronized physical sound: {legacy_diegetic_sound}"
        visual_action = "\n".join(part for part in (visual_action, sound_instruction) if part)

    # Version 14 and earlier stored camera choices in separate selectors.
    # Preserve meaningful legacy choices once in the unified natural-language
    # input, then drop the obsolete fields from the normalized schema.
    legacy_camera: list[str] = []
    framing = _clean_text(raw.get("camera_framing"))
    angle = _clean_text(raw.get("camera_angle"))
    motion = _clean_text(raw.get("camera_motion"))
    if framing:
        legacy_camera.append(
            CAMERA_FRAMING_SENTENCES.get(framing, f"Camera framing: {framing}.")
        )
    if angle:
        legacy_camera.append(
            CAMERA_ANGLE_SENTENCES.get(angle, f"Camera angle: {angle}.")
        )
    if motion:
        legacy_camera.append(
            CAMERA_SENTENCES.get(motion, f"Camera motion: {motion}.")
        )
    transition = _clean_text(raw.get("transition")).lower()
    if index > 0 and transition in SUPPORTED_TRANSITIONS and transition != "cut":
        legacy_camera.append(f"Transition into this shot with a {transition}.")
    if legacy_camera:
        visual_action = "\n".join((visual_action, *legacy_camera)).strip()

    return {
        "id": _clean_text(raw.get("id")) or f"shot-{index + 1}",
        "duration": max(MIN_SHOT_DURATION, _number(raw.get("duration"), 1.0)),
        "visual_action": visual_action,
    }


def _normalize_reference(raw: Any, index: int) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    ref_type = _clean_text(raw.get("type")).lower()
    if ref_type not in ("picture", "video", "audio"):
        ref_type = "picture"
    role = _clean_text(raw.get("role")).lower()
    strength = _clean_text(raw.get("strength")).lower()
    if ref_type == "picture":
        if role in {"reference", "environment", "style", "storyboard"}:
            role = "subject_identity"
            strength = "weak"
        elif role == "subject_identity":
            # Version 9 and earlier used Subject as an implicit strong role.
            strength = strength if strength in SUBJECT_STRENGTHS else "strong"
        elif role not in {"first_frame", "last_frame", "frame"}:
            role = "subject_identity"
            strength = strength if strength in SUBJECT_STRENGTHS else "normal"
        else:
            strength = "normal"
    elif ref_type == "video":
        legacy_video_roles = {
            "reference": "none",
            "continuation": "video_continuation",
            "pacing": "cuts_rhythm",
        }
        role = role if role in REFERENCE_ROLES["video"] else legacy_video_roles.get(role, "none")
    elif ref_type == "audio":
        legacy_audio_roles = {
            "reference": "none",
            "voice_timbre": "voice_delivery",
            "dialogue": "dialogue_lyrics",
            "music_style": "music_rhythm",
            "sound_effect": "sound_ambience",
            "signal_copy": "partial_signal_copy",
        }
        role = role if role in REFERENCE_ROLES["audio"] else legacy_audio_roles.get(role, "none")
    description = _clean_text(raw.get("description"))
    # Picture analysis is transient enhancement evidence, never persisted
    # project input. Older versions wrote it here and thereby changed Raw Prompt.
    if ref_type == "picture":
        description = ""
    return {
        "id": _clean_text(raw.get("id")) or f"ref-{index + 1}",
        "type": ref_type,
        "role": role if role in REFERENCE_ROLES[ref_type] else "reference",
        "strength": strength if strength in SUBJECT_STRENGTHS else "normal",
        "alias": _normalize_alias(raw.get("alias")),
        "description": description,
        "duration": max(0.0, _number(raw.get("duration"), 0.0)),
        "source_duration": max(0.0, _number(raw.get("source_duration"), _number(raw.get("duration"), 0.0))),
        "trim_start": max(0.0, _number(raw.get("trim_start"), 0.0)),
        "timeline_start": _number(raw.get("timeline_start"), 0.0),
        "frame_index": max(0, int(round(_number(raw.get("frame_index"), 0.0)))),
        "image_filename": os.path.basename(_clean_text(raw.get("image_filename"))),
        "image_subfolder": _clean_text(raw.get("image_subfolder")).replace("\\", "/").strip("/"),
        "image_type": "input" if _clean_text(raw.get("image_type")).lower() != "input" else "input",
        "video_filename": os.path.basename(_clean_text(raw.get("video_filename"))),
        "video_subfolder": _clean_text(raw.get("video_subfolder")).replace("\\", "/").strip("/"),
        "video_type": "input",
        "audio_filename": os.path.basename(_clean_text(raw.get("audio_filename"))),
        "audio_subfolder": _clean_text(raw.get("audio_subfolder")).replace("\\", "/").strip("/"),
        "audio_type": "input",
    }


def infer_auto_mode(references: list[dict[str, Any]]) -> str:
    """Resolve Auto from an exact anchor layout so no extra asset is silently ignored."""
    if not references:
        return "T2VA"
    signature = [(ref.get("type"), ref.get("role")) for ref in references]
    if signature == [("picture", "first_frame")]:
        return "I2VA"
    if signature == [("picture", "first_frame"), ("picture", "last_frame")]:
        return "FL2VA"
    if signature == [("picture", "last_frame")]:
        return "L2VA"
    return "REF2VA"


def normalize_project(project_data: Any) -> tuple[dict[str, Any], list[str]]:
    parse_warnings: list[str] = []
    if isinstance(project_data, str):
        try:
            raw = json.loads(project_data) if project_data.strip() else {}
        except json.JSONDecodeError as exc:
            raw = {}
            parse_warnings.append(f"Project JSON was invalid and defaults were used: {exc.msg}.")
    elif isinstance(project_data, dict):
        raw = project_data
    else:
        raw = {}

    project = copy.deepcopy(DEFAULT_PROJECT)
    raw_version = raw.get("version")
    # Version 8 is a lossless cleanup migration from version 7: cached picture
    # analysis text is discarded. Do not report that expected upgrade as a warning.
    if raw_version is not None and raw_version != CURRENT_PROJECT_VERSION and raw_version not in {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}:
        relation = "newer than" if isinstance(raw_version, (int, float)) and raw_version > CURRENT_PROJECT_VERSION else "different from"
        parse_warnings.append(
            f"Project version {raw_version!r} is {relation} supported version {CURRENT_PROJECT_VERSION}; known fields were normalized."
        )
    mode = _clean_text(raw.get("mode")).upper()
    selected_mode = mode if mode in SUPPORTED_MODES else "AUTO"
    project["user_request"] = _clean_text(raw.get("user_request"))
    project["constraints"] = _clean_text(raw.get("constraints"))
    project["verbatim_content"] = _clean_text(raw.get("verbatim_content"))
    selected_enhance_model = _clean_text(raw.get("enhance_model"))
    if selected_enhance_model == LEGACY_LIGHTX2V_MODEL_ID:
        selected_enhance_model = LIGHTX2V_MODEL_ID
    project["enhance_model"] = (
        selected_enhance_model
        if selected_enhance_model in {DEFAULT_ENHANCE_MODEL_ID, LIGHTX2V_MODEL_ID}
        else DEFAULT_ENHANCE_MODEL_ID
    )
    selected_image_model = _clean_text(raw.get("image_model"))
    if selected_image_model == LEGACY_LIGHTX2V_MODEL_ID:
        selected_image_model = LIGHTX2V_MODEL_ID
    project["image_model"] = (
        selected_image_model
        if selected_image_model in {QWEN_IMAGE_MODEL_ID, LIGHTX2V_MODEL_ID}
        else QWEN_IMAGE_MODEL_ID
    )
    project["auto_run"] = raw.get("auto_run") is True
    project["enhance"] = raw.get("enhance") is True
    project["enhanced_prompt"] = _clean_text(raw.get("enhanced_prompt"))

    raw_shots = raw.get("shots")
    if isinstance(raw_shots, list) and raw_shots:
        project["shots"] = [_normalize_shot(item, i) for i, item in enumerate(raw_shots)]
    else:
        requested = min(15.0, max(MIN_SHOT_DURATION, _number(raw.get("requested_duration"), 5.0)))
        project["shots"][0]["duration"] = requested

    # Version 13 and earlier stored three dedicated audio UI values. Preserve
    # them once as natural-language instructions in the unified first-shot
    # prompt, then drop the obsolete schema fields.
    legacy_soundscape = _clean_text(raw.get("overall_soundscape"))
    legacy_music = _clean_text(raw.get("non_diegetic_music"))
    legacy_audio = []
    if legacy_soundscape:
        legacy_audio.append(f"Overall soundscape: {legacy_soundscape}")
    if legacy_music:
        legacy_audio.append(f"Non-diegetic music: {legacy_music}")
    if legacy_audio:
        project["shots"][0]["visual_action"] = "\n".join(
            part for part in (project["shots"][0]["visual_action"], *legacy_audio) if part
        )

    shot_total = sum(float(shot["duration"]) for shot in project["shots"])
    requested_duration = max(
        len(project["shots"]) * MIN_SHOT_DURATION,
        _number(raw.get("requested_duration"), shot_total),
    )
    if shot_total > 0 and not math.isclose(shot_total, requested_duration, abs_tol=0.0005):
        distributable = requested_duration - len(project["shots"]) * MIN_SHOT_DURATION
        weights = [max(0.0, float(shot["duration"]) - MIN_SHOT_DURATION) for shot in project["shots"]]
        weight_total = sum(weights)
        for index, shot in enumerate(project["shots"]):
            share = weights[index] / weight_total if weight_total else 1.0 / len(project["shots"])
            shot["duration"] = MIN_SHOT_DURATION + distributable * share
    project["requested_duration"] = round(requested_duration, 3)
    effective_duration = align_frame_count(requested_duration) / MODEL_FPS
    effective_frames = align_frame_count(requested_duration)
    for ref in project["references"]:
        if ref["type"] == "picture" and ref["role"] == "frame":
            ref["frame_index"] = min(ref["frame_index"], effective_frames - 1)
    raw_refs = raw.get("references")
    project["references"] = (
        [_normalize_reference(item, i) for i, item in enumerate(raw_refs)]
        if isinstance(raw_refs, list)
        else []
    )
    for ref in project["references"]:
        if ref["type"] != "video":
            continue
        source_duration = max(ref["source_duration"], ref["duration"])
        ref["source_duration"] = source_duration
        ref["trim_start"] = min(ref["trim_start"], max(0.0, source_duration - MIN_SHOT_DURATION))
        available = max(0.0, source_duration - ref["trim_start"])
        ref["duration"] = min(ref["duration"], available, REF_VIDEO_MAX_SECONDS)
        minimum_visible = min(REF_VIDEO_MIN_SECONDS, ref["duration"])
        ref["timeline_start"] = min(
            max(-ref["duration"] + minimum_visible, ref["timeline_start"]),
            effective_duration - minimum_visible,
        )
    if isinstance(raw_refs, list):
        for index, (raw_ref, ref) in enumerate(zip(raw_refs, project["references"]), 1):
            supplied = _clean_text(raw_ref.get("role")) if isinstance(raw_ref, dict) else ""
            if supplied and supplied.lower() != ref["role"]:
                expected_picture_migration = (
                    ref["type"] == "picture"
                    and supplied.lower() in {"reference", "environment", "style", "storyboard"}
                )
                expected_video_migration = (
                    ref["type"] == "video"
                    and supplied.lower() in {"reference", "continuation", "pacing"}
                )
                expected_audio_migration = (
                    ref["type"] == "audio"
                    and supplied.lower() in {
                        "reference", "voice_timbre", "dialogue", "music_style",
                        "sound_effect", "signal_copy",
                    }
                )
                if not expected_picture_migration and not expected_video_migration and not expected_audio_migration:
                    parse_warnings.append(
                        f"Reference {index} role={supplied!r} is invalid for {ref['type']} and was normalized to {ref['role']!r}."
                    )
    project["mode"] = infer_auto_mode(project["references"]) if selected_mode == "AUTO" else selected_mode
    project["mode_selection"] = selected_mode
    return project, parse_warnings


def _reference_labels(references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = {"picture": 0, "video": 0, "audio": 0}
    labeled = []
    for ref in references:
        item = dict(ref)
        counts[item["type"]] += 1
        label_name = {"picture": "Picture", "video": "Video", "audio": "Audio"}[item["type"]]
        item["label"] = f"<{label_name} {counts[item['type']]}>"
        labeled.append(item)
    return labeled


def validate_project(project: dict[str, Any], parse_warnings: list[str] | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings = list(parse_warnings or [])
    duration = float(project["requested_duration"])
    shot_total = sum(float(shot["duration"]) for shot in project["shots"])
    refs = _reference_labels(project["references"])
    aliases = [ref["alias"].lower() for ref in refs if ref["alias"]]
    pictures = [ref for ref in refs if ref["type"] == "picture"]
    videos = [ref for ref in refs if ref["type"] == "video"]
    audios = [ref for ref in refs if ref["type"] == "audio"]

    if len(aliases) != len(set(aliases)):
        errors.append("Reference aliases must be unique so each @mention resolves to exactly one reference.")

    if duration < 4.0 or duration > 15.0:
        errors.append(f"H3 output duration must be between 4 and 15 seconds; received {duration:.2f}s.")
    if not math.isclose(shot_total, duration, abs_tol=0.001):
        errors.append(f"Shot durations total {shot_total:.3f}s and must equal the {duration:.3f}s timeline duration.")
    if not project["user_request"] and not any(shot["visual_action"] for shot in project["shots"]):
        warnings.append("No overall request or shot action has been entered.")
    descriptive_text = [
        project["user_request"], project["constraints"],
        *(shot["visual_action"] for shot in project["shots"]),
        *(ref["description"] for ref in project["references"]),
    ]
    if any(_contains_hangul(text) for text in descriptive_text):
        warnings.append(
            "Direct compilation does not translate descriptive Korean text; English is recommended outside dialogue and visible text."
        )
    if project["mode"] == "T2VA" and refs:
        warnings.append("T2VA does not use reference assets; manifest entries will be context only.")
    if project["mode"] == "I2VA" and (not pictures or pictures[0]["role"] != "first_frame"):
        errors.append("I2VA requires <Picture 1> to have role=first_frame.")
    if project["mode"] == "FL2VA":
        if len(pictures) < 2 or pictures[0]["role"] != "first_frame" or pictures[1]["role"] != "last_frame":
            errors.append("FL2VA requires <Picture 1> role=first_frame and <Picture 2> role=last_frame, in that order.")
        if len(project["shots"]) > 1:
            warnings.append("FL2VA usually works best as one continuous shot unless cuts are intentional.")
    if project["mode"] == "L2VA" and (not pictures or pictures[0]["role"] != "last_frame"):
        errors.append("L2VA requires <Picture 1> to have role=last_frame.")
    if project["mode"] == "REF2VA" and not refs:
        errors.append("REF2VA requires at least one reference asset.")
    if len(pictures) > MAX_REF_IMAGES:
        errors.append(f"REF2VA accepts at most {MAX_REF_IMAGES} reference images; received {len(pictures)}.")
    if len(videos) > MAX_REF_VIDEOS:
        errors.append(f"REF2VA accepts at most {MAX_REF_VIDEOS} reference videos; received {len(videos)}.")
    if len(audios) > MAX_REF_AUDIOS:
        errors.append(f"REF2VA accepts at most {MAX_REF_AUDIOS} reference audio clips; received {len(audios)}.")
    if len(refs) > MAX_REF_FILES:
        errors.append(f"REF2VA accepts at most {MAX_REF_FILES} reference files in total; received {len(refs)}.")
    video_total = sum(ref["duration"] for ref in videos)
    for ref in videos:
        if ref["duration"] and not REF_VIDEO_MIN_SECONDS <= ref["duration"] <= REF_VIDEO_MAX_SECONDS:
            errors.append(f"{ref['label']} duration must be 2-15 seconds; received {ref['duration']:.2f}s.")
        elif not ref["duration"] and ref.get("video_filename"):
            warnings.append(f"{ref['label']} has no duration metadata; the 2-15 second limit cannot be verified.")
    if video_total > REF_VIDEO_TOTAL_SECONDS:
        errors.append(f"Reference-video duration totals {video_total:.2f}s; the maximum is 15.00s.")
    for ref in audios:
        role = ref.get("role", "none")
        description = _clean_text(ref.get("description"))
        if role in {"full_signal_copy", "partial_signal_copy", "dialogue_lyrics"} and not ref.get("audio_filename"):
            errors.append(f"{ref['label']} preset={role} requires an uploaded audio file.")
        if role == "none" and not description:
            warnings.append(f"{ref['label']} has preset=None and no user-defined audio relationship.")
        if role == "dialogue_lyrics" and not description:
            warnings.append(
                f"{ref['label']} dialogue/lyrics reuse has no exact words or transcription instructions; "
                "the prompt must not invent them."
            )
        if role == "voice_delivery" and not description:
            warnings.append(
                f"{ref['label']} voice/delivery reference does not identify a target speaker or voice traits."
            )

    effective_frames = align_frame_count(duration)
    effective_seconds = effective_frames / MODEL_FPS
    if effective_seconds > 15.0:
        warnings.append(f"Aligned duration is {effective_seconds:.2f}s, slightly beyond H3's 15-second envelope.")
    return errors, warnings


def _sentence(text: str) -> str:
    text = _clean_text(text)
    if not text:
        return ""
    return text if text[-1] in ".?!:;" else text + "."


def _replace_aliases(text: str, aliases: dict[str, str]) -> str:
    if not text or not aliases:
        return text
    lookup = {alias.lower(): replacement for alias, replacement in aliases.items()}
    pattern = re.compile(
        "(" + "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True)) + ")"
        r"(?![\w-])",
        flags=re.IGNORECASE,
    )
    return pattern.sub(lambda match: lookup[match.group(0).lower()], text)


def _quoted_prompt_text(text: str) -> str:
    """Keep user-visible text unambiguous inside a quoted prompt sentence."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\r", " ").replace("\n", " ")


def _reference_model(project: dict[str, Any]) -> dict[str, Any]:
    references = _reference_labels(project["references"])
    subject_count = 0
    aliases: dict[str, str] = {}
    definitions: list[str] = []
    retention: list[str] = []
    applications: list[str] = []
    task_types: list[str] = []
    label_plan: dict[str, dict[str, str]] = {}
    summary_relations: list[str] = []
    final_shot = len(project["shots"])

    def add_task(task_type: str):
        if task_type not in task_types:
            task_types.append(task_type)

    for ref in references:
        source_label = ref["label"]
        role_text = ref["role"].replace("_", " ")
        if ref["type"] == "video":
            role_text = {
                "none": "user-defined video relationship",
                "video_editing": "source video editing",
                "video_continuation": "source video continuation",
                "motion": "motion and action timing",
                "camera": "camera movement and viewpoint behavior",
                "cuts_rhythm": "cuts, pacing, rhythm, and temporal structure",
            }.get(ref["role"], role_text)
        elif ref["type"] == "audio":
            role_text = {
                "none": "user-defined audio relationship",
                "full_signal_copy": "complete source-audio reuse",
                "partial_signal_copy": "partial source-audio reuse",
                "voice_delivery": "voice timbre and delivery",
                "dialogue_lyrics": "dialogue or lyrics reuse",
                "sound_ambience": "sound effects and ambience",
                "music_rhythm": "music style, tempo, and rhythm",
            }.get(ref["role"], role_text)
        generic_reference_text = {
            "picture": "general visual reference",
            "video": "general video reference",
            "audio": "general audio reference",
        }[ref["type"]]
        description = _sentence(ref["description"])
        if ref["type"] == "picture" and ref["role"] in {"reference", "subject_identity"}:
            subject_count += 1
            subject = f"<Subject {subject_count}>"
            if ref["alias"]:
                aliases[ref["alias"]] = subject
            strength = "weak" if ref["role"] == "reference" else ref.get("strength", "strong")
            definition = f"{subject} is the reusable visible subject derived from {source_label}"
            if description:
                definition += f", described as {description.rstrip('.')}"
            definitions.append(definition + ".")
            marker = {
                "weak": "weak_reference",
                "normal": "partially_preserved",
                "strong": "fully_preserved",
            }[strength]
            retention_detail = {
                "weak": "retain only broad similarity in a small set of target-relevant visible characteristics",
                "normal": "retain core identity and primary visible appearance while allowing secondary details to vary",
                "strong": "preserve the complete visible subject identity, appearance, and source visual medium/rendering style wherever it appears",
            }[strength]
            retention.append(f"{subject}: {marker} - {retention_detail}.")
            if not ref["alias"]:
                applications.append(f"Apply {subject} only as its defined Subject content at {strength} strength.")
            role_contract = {
                "weak": "broad subject appearance similarity only; exclude source setting, style, composition, camera, lighting, palette, pose, and action",
                "normal": "core subject identity and primary visible appearance; secondary details may vary; exclude source setting, style, composition, camera, lighting, palette, pose, and action",
                "strong": "complete visible subject identity and appearance plus that subject's source visual medium/rendering style; preserve the style independently per subject; exclude source setting, composition, camera, lighting setup, scene-wide palette, pose, and action",
            }[strength]
            label_plan[subject] = {
                "kind": "Subject", "source": source_label, "role": "subject_identity", "marker": marker,
                "strength": strength, "contract": role_contract,
            }
            summary_relations.append(f"{subject} as a {strength}-strength subject reference")
            add_task("reference generation")
            continue

        if ref["alias"]:
            aliases[ref["alias"]] = source_label
        if ref["type"] == "picture" and ref["role"] == "first_frame":
            definition = f"{source_label} is the first frame of [Shot 1]"
        elif ref["type"] == "picture" and ref["role"] == "last_frame":
            definition = f"{source_label} is the final frame of [Shot {final_shot}]"
        elif ref["type"] == "picture" and ref["role"] == "frame":
            definition = f"{source_label} is the exact target frame at output frame {ref.get('frame_index', 0)}"
        elif ref["type"] == "video" and ref["role"] == "video_editing":
            definition = f"{source_label} is the source video for the target video edit"
        elif ref["type"] == "video" and ref["role"] == "video_continuation":
            definition = f"{source_label} is the continuation source for the target video"
        elif ref["type"] == "video" and ref["role"] == "none":
            definition = f"{source_label} has a user-defined video relationship"
        elif ref["type"] == "audio" and ref["role"] == "none":
            definition = f"{source_label} has a user-defined audio relationship"
        elif ref["type"] == "audio" and ref["role"] == "full_signal_copy":
            definition = f"{source_label} is the complete source audio to reuse"
        elif ref["type"] == "audio" and ref["role"] == "partial_signal_copy":
            definition = f"{source_label} supplies selected source-audio signal or layers to reuse"
        elif ref["role"] == "reference":
            definition = f"{source_label} is a {generic_reference_text}"
        else:
            definition = f"{source_label} is the {role_text} reference"
        if description:
            definition += f", described as {description.rstrip('.')}"
        if ref["duration"]:
            if ref["type"] == "video":
                definition += (
                    f", using only the first {ref['duration']:.2f} seconds as the configured "
                    "analysis and reference segment"
                )
            else:
                definition += f", with a source duration of {ref['duration']:.2f} seconds"
        definitions.append(definition + ".")

        if ref["type"] == "picture":
            if ref["role"] in ("first_frame", "last_frame", "frame"):
                marker = "fully_preserved"
                add_task("keyframe completion")
            else:
                marker = "weak_reference"
                add_task("reference generation")
            retention.append(f"{source_label}: {marker} - apply only its defined {role_text} role.")
        elif ref["type"] == "video":
            if ref["role"] == "video_editing":
                marker = "partially_preserved"
                add_task("video editing")
            elif ref["role"] == "video_continuation":
                marker = "partially_preserved"
                add_task("video continuation")
            else:
                marker = "weak_reference"
                add_task("reference generation")
            retention.append(f"{source_label}: {marker} - use only its defined {role_text} relationship.")
        else:
            if ref["role"] == "full_signal_copy":
                marker = "fully_copy"
                add_task("audio reuse")
            elif ref["role"] in ("partial_signal_copy", "dialogue_lyrics"):
                marker = "partially_copy"
                add_task("audio reuse")
            else:
                marker = "reference"
                add_task("audio reference")
            retention.append(f"{source_label}: {marker} - use only its defined {role_text} relationship.")
        video_contracts = {
            "none": "follow only the user-written relationship; do not infer an editing, continuation, motion, camera, or timing role",
            "video_editing": "treat as the source video being directly edited; preserve source timeline elements except those the user explicitly changes",
            "video_continuation": "continue from the source video's ending state, preserving final composition, positions, movement direction and momentum, camera behavior, lighting, and continuity unless changed",
            "motion": "reference only subject motion, action sequence, movement timing, and physical rhythm; do not copy identity, setting, style, camera, cuts, or audio",
            "camera": "reference only camera movement, viewpoint, framing progression, and camera timing; do not copy identity, setting, action content, style, or audio",
            "cuts_rhythm": "reference only cut placement, pacing, rhythm, and temporal structure; do not copy identity, setting, action content, visual style, or audio",
        }
        audio_contracts = {
            "none": "follow only the user-written audio relationship; do not infer signal copying, voice, dialogue, lyrics, effects, ambience, music, or rhythm",
            "full_signal_copy": "reuse the complete source audio signal as the target audio; do not invent or replace layers",
            "partial_signal_copy": "reuse only the user-specified source interval or audio layers; leave every other layer unspecified unless requested",
            "voice_delivery": "reference only voice timbre, delivery, accent, emotion, pace, and vocal texture; do not copy source words, music, ambience, or effects",
            "dialogue_lyrics": "reuse only exact user-supplied or reliably transcribed dialogue or lyrics; never invent, translate, correct, or complete unavailable words",
            "sound_ambience": "reference only user-described sound effects, ambience, room tone, and acoustic character; do not copy dialogue, lyrics, or music",
            "music_rhythm": "reference only user-described instrumentation, tempo, meter, rhythm, dynamics, structure, and musical mood; do not claim source-signal reuse",
        }
        label_plan[source_label] = {
            "kind": ref["type"].title(), "source": source_label,
            "role": ref["role"], "marker": marker,
            "contract": (
                video_contracts.get(ref["role"], f"use only as the defined {role_text} relationship")
                if ref["type"] == "video"
                else audio_contracts.get(ref["role"], f"use only as the defined {role_text} relationship")
            ),
        }
        summary_relations.append(f"{source_label} for {role_text}")
        if not ref["alias"]:
            if ref["role"] == "reference":
                applications.append(f"Apply {source_label} only as a {generic_reference_text}.")
            else:
                applications.append(f"Apply {source_label} only as the {role_text} reference.")

    return {
        "definitions": definitions,
        "retention": retention,
        "applications": applications,
        "aliases": aliases,
        "task_types": task_types or ["reference generation"],
        "label_plan": label_plan,
        "summary_relations": summary_relations,
    }


def _shot_description(project: dict[str, Any], effective_seconds: float, aliases: dict[str, str],
                      reference_applications: list[str] | None = None) -> str:
    requested_seconds = sum(float(shot["duration"]) for shot in project["shots"])
    scale = effective_seconds / requested_seconds if requested_seconds > 0 else 1.0
    cursor = 0.0
    blocks: list[str] = []
    for index, shot in enumerate(project["shots"], 1):
        fragments: list[str] = []
        if index == 1 and project["user_request"]:
            fragments.append(_sentence(_replace_aliases(project["user_request"], aliases)))
        if index == 1 and reference_applications:
            fragments.extend(reference_applications)
        if shot["visual_action"]:
            fragments.append(_sentence(_replace_aliases(shot["visual_action"], aliases)))
        if index == 1:
            prefix = "[Shot 1] "
            if not fragments:
                fragments.append("The scene begins with no additional shot-specific action specified.")
        else:
            prefix = f"[Shot {index}] At {format_timestamp(cursor)}, cut to a new shot. "
            if not fragments:
                fragments.append("The scene continues with no additional shot-specific action specified.")
        blocks.append(prefix + " ".join(fragment for fragment in fragments if fragment))
        cursor += float(shot["duration"]) * scale

    if project["constraints"]:
        blocks.append(_sentence(f"Throughout the video, {_replace_aliases(project['constraints'], aliases)}"))
    if project["verbatim_content"]:
        blocks.append(_sentence(f"Preserve this verbatim content exactly: {project['verbatim_content']}"))
    return " ".join(blocks)


def _fl2va_alignment_instruction(effective_seconds: float, final_shot: int) -> str:
    return (
        "How the reference pictures align with the target video — "
        "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
        f"Picture 2 (from Shot {final_shot}) aligns with the {effective_seconds:.2f}-second mark of the target video."
    )


I2VA_ALIGNMENT_INSTRUCTION = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)


def _l2va_alignment_instruction(effective_seconds: float, final_shot: int) -> str:
    return (
        "How the reference pictures align with the target video — "
        f"<Picture 1> (from [Shot {final_shot}]) aligns with the "
        f"{effective_seconds:.2f}-second mark of the target video."
    )


def _mode_prompt_preamble(mode: str) -> str:
    if mode == "REF2VA":
        return (
            "ACTIVE MODE: REF2VA FULL-REFERENCE.\n"
            "OUTPUT FAMILY: exactly six REF2VA sections.\n"
            "FORBIDDEN OUTPUT FIELD: integrated_multimodal_description.\n"
            "Do not answer in the three-field Base/T2VA format."
        )
    return (
        f"ACTIVE MODE: {mode}.\n"
        "OUTPUT FAMILY: MiniMax H3 Base format with exactly three fields.\n"
        "FORBIDDEN OUTPUT SECTIONS: subject_definitions, summary, retention_analysis, "
        "and detailed_description."
    )


def _single_pass_output_lock(mode: str, effective_seconds: float, final_shot: int,
                             expected_shots: list[int],
                             reference_model: dict[str, Any] | None = None,
                             content_locks: list[str] | None = None) -> str:
    shots = ", ".join(f"[Shot {number}]" for number in expected_shots)
    content_lock = ""
    if content_locks:
        content_lock = (
            "\nINPUT-DERIVED CONTENT LOCKS — these are binding, not output headings:\n- "
            + "\n- ".join(content_locks)
        )
    if mode == "REF2VA":
        label_plan = (reference_model or {}).get("label_plan", {})
        labels = ", ".join(label_plan) or "the locked labels above"
        retention_markers = "; ".join(
            f"{label} uses {plan.get('marker', 'the locked marker')}"
            for label, plan in label_plan.items()
        ) or "use the locked marker for each label"
        return f"""FINAL MODE LOCK — REF2VA
Highest-priority format lock. Return one <H3_PROMPT> block with no JSON, Markdown, or commentary.
Start exactly with `subject_definitions:` and never use `integrated_multimodal_description:`.
Use these headers once in this order:
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:
Define exactly these output labels in order: {labels}.
Use exactly these labels in order: {labels}. Keep them literal where their roles apply; create no others.
RETENTION OUTPUT MARKERS: {retention_markers}.
In retention_analysis, print only each label, applicable shots, its marker above, and a concise description. Never print input strength names or `=`.
detailed_description must contain exactly {shots}, once each in order; [Shot 1] has no timestamp.{content_lock}
Speaker IDs, exact <d> content, lip synchronization, and event order must be correct in the final output.
Do not invent people, dialogue, vocal reactions, or music. Use N/A for unrequested non-diegetic music.
End after non_diegetic_music and close </H3_PROMPT>."""

    if mode == "I2VA":
        opening = I2VA_ALIGNMENT_INSTRUCTION
        endpoint_lock = ""
    elif mode == "FL2VA":
        opening = _fl2va_alignment_instruction(effective_seconds, final_shot)
        endpoint_lock = (
            "\nPicture 2 is reached only at the effective end time. Never reveal the completed Picture 2 "
            "at the start of the final shot; make that shot continue the transition and stabilize exactly on "
            "Picture 2 only in its final frames. Never morph one character into another unless the input "
            "explicitly requests morphing or transformation. Bind Picture 2 traits only to the entity that "
            "matches or enters as the final-frame entity, never to a different affected character."
        )
    elif mode == "L2VA":
        opening = _l2va_alignment_instruction(effective_seconds, final_shot)
        endpoint_lock = ""
    else:
        opening = "integrated_multimodal_description:"
        endpoint_lock = ""
    return f"""FINAL MODE LOCK — {mode}
Highest-priority format lock. Return one <H3_PROMPT> block with no JSON, Markdown, or commentary.
Begin exactly with: {opening}
Use exactly these fields once in order: integrated_multimodal_description, overall_soundscape, non_diegetic_music.
For I2VA, FL2VA, and L2VA, keep the alignment line before the main field, never inside it.
Never use REF2VA sections. The timeline must contain exactly {shots}, once each in order; [Shot 1] has no timestamp.{endpoint_lock}{content_lock}
Preserve every explicit SHOT_PLAN action in order; omit none.
Speaker IDs, exact <d> content, lip synchronization, and event order must be correct in the final output.
Do not invent dialogue, vocal reactions, or music. overall_soundscape must not repeat or summarize speech. Use N/A for unrequested non-diegetic music.
End after non_diegetic_music and close </H3_PROMPT>."""


_BASE_FIELD_PATTERN = re.compile(
    r"^[ \t]*(integrated_multimodal_description|overall_soundscape|non_diegetic_music)[ \t]*:[ \t]*",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _base_prompt_sections(prompt: str) -> dict[str, str]:
    matches = list(_BASE_FIELD_PATTERN.finditer(prompt))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).lower()
        if name in sections:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        sections[name] = prompt[match.end():end].strip()
    return sections


def _remove_embedded_alignment(text: str) -> str:
    text = re.sub(
        r"For the target video,\s*at 0\.00 seconds into the target video,\s*"
        r"<Picture 1>\s*\(from \[Shot 1\]\)\s*is fully referenced\.\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"How the reference pictures align with the target video\s*[—-]\s*.*?"
        r"(?:mark of the target video|target video)\.\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return text.strip()


def _move_preamble_after_shot_one(text: str) -> str:
    shot = re.search(r"\[Shot\s+1\]", text, flags=re.IGNORECASE)
    if not shot:
        return text.strip()
    remainder = text[shot.end():].strip()
    # Alignment paraphrases and conversational lead-ins before the first shot
    # are transport noise, not scene content. The Base schema starts at Shot 1.
    return "[Shot 1]" + (" " + remainder if remainder else "")


def _normalize_base_enhanced_prompt(prompt: str, mode: str, effective_seconds: float,
                                    final_shot: int) -> str:
    """Canonicalize Base-mode field order and keyframe alignment without rewriting content."""
    sections = _base_prompt_sections(prompt)
    required = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
    if any(name not in sections for name in required):
        return prompt.strip()

    main = _move_preamble_after_shot_one(_remove_embedded_alignment(sections[required[0]]))
    body = "\n\n".join((
        "integrated_multimodal_description: " + main,
        "overall_soundscape: " + sections[required[1]],
        "non_diegetic_music: " + sections[required[2]],
    ))
    if mode == "I2VA":
        return I2VA_ALIGNMENT_INSTRUCTION + "\n\n" + body
    if mode == "FL2VA":
        return _fl2va_alignment_instruction(effective_seconds, final_shot) + "\n\n" + body
    if mode == "L2VA":
        return _l2va_alignment_instruction(effective_seconds, final_shot) + "\n\n" + body
    return body


def _base_prompt_structure_issues(prompt: str, mode: str, effective_seconds: float,
                                  final_shot: int) -> list[str]:
    matches = list(_BASE_FIELD_PATTERN.finditer(prompt))
    sections = _base_prompt_sections(prompt)
    required = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
    issues: list[str] = []
    if len(matches) != len(required) or list(sections) != list(required):
        issues.append("Use exactly the three Base-mode fields once and in the required order.")
        return issues
    main = sections[required[0]].lstrip()
    if not re.match(r"\[Shot\s+1\](?=\s|$)", main, flags=re.IGNORECASE):
        issues.append("integrated_multimodal_description must begin with [Shot 1].")
    expected_prefix = ""
    if mode == "I2VA":
        expected_prefix = I2VA_ALIGNMENT_INSTRUCTION
    elif mode == "FL2VA":
        expected_prefix = _fl2va_alignment_instruction(effective_seconds, final_shot)
    elif mode == "L2VA":
        expected_prefix = _l2va_alignment_instruction(effective_seconds, final_shot)
    if expected_prefix and not prompt.startswith(expected_prefix + "\n\n"):
        issues.append("Place the exact mode alignment instruction on the first line.")
    if mode == "T2VA" and not prompt.startswith("integrated_multimodal_description:"):
        issues.append("T2VA must begin directly with integrated_multimodal_description.")
    if re.search(r"For the target video,\s*at 0\.00 seconds", main, flags=re.IGNORECASE):
        issues.append("Remove the alignment instruction from the main description.")
    return issues


REF_PROMPT_FIELDS = (
    "subject_definitions", "summary", "retention_analysis",
    "detailed_description", "overall_soundscape", "non_diegetic_music",
)
_REF_FIELD_PATTERN = re.compile(
    r"^[ \t]*(subject_definitions|summary|retention_analysis|detailed_description|"
    r"overall_soundscape|non_diegetic_music)[ \t]*:[ \t]*",
    flags=re.IGNORECASE | re.MULTILINE,
)
_REF_LABEL_PATTERN = re.compile(r"<(Subject|Picture|Video|Audio)\s+(\d+)>", re.IGNORECASE)
_SHOT_HEADER_PATTERN = re.compile(
    r"(?:\A|(?<=\n))\s*\[Shot\s+(\d+)\](?=\s|$)"
    r"|\[Shot\s+(\d+)\]\s+At\s+\d{2}:\d{2}\.\d{3}",
    flags=re.IGNORECASE,
)


def _ref_prompt_sections(prompt: str) -> dict[str, str]:
    matches = list(_REF_FIELD_PATTERN.finditer(prompt))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).lower()
        if name in sections:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        sections[name] = prompt[match.end():end].strip()
    return sections


def _insert_ref_first_shot_header(detail: str, boundary: int) -> str:
    """Insert a real Shot 1 header into prose that precedes the first later-shot header."""
    opening = detail[:boundary].strip()
    remainder = detail[boundary:].strip()
    if not opening:
        return detail
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", opening) if part.strip()]
    if len(paragraphs) >= 2:
        paragraphs[1] = "[Shot 1] " + re.sub(
            r"^\[Shot\s+1\]\s*", "", paragraphs[1], flags=re.IGNORECASE,
        )
        opening = "\n\n".join(paragraphs)
    else:
        style_sentence = re.match(
            r"^(.+?\b(?:style|aesthetic|presentation|rendering)\b[^.!?]*[.!?])\s*(.+)$",
            opening,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if style_sentence:
            action_body = re.sub(
                r"^\[Shot\s+1\]\s*", "", style_sentence.group(2).strip(), flags=re.IGNORECASE,
            )
            opening = style_sentence.group(1).strip() + "\n[Shot 1] " + action_body
        else:
            opening = "[Shot 1] " + re.sub(
                r"^\[Shot\s+1\]\s*", "", opening, flags=re.IGNORECASE,
            )
    return opening + ("\n" + remainder if remainder else "")


def _normalize_ref_enhanced_prompt(prompt: str, task_types: list[str] | None = None,
                                   expected_shots: list[int] | None = None) -> str:
    sections = _ref_prompt_sections(prompt)
    if any(field not in sections for field in REF_PROMPT_FIELDS):
        return prompt.strip()
    if task_types:
        summary_body = re.sub(r"^\s*\[[^\]\n]*\]\s*", "", sections["summary"], count=1)
        sections["summary"] = f"[{' + '.join(task_types)}] {summary_body}".rstrip()
    detail = sections["detailed_description"].strip()
    header_matches = list(_SHOT_HEADER_PATTERN.finditer(detail))
    actual_headers = [int(match.group(1) or match.group(2)) for match in header_matches]
    if detail and expected_shots:
        if not actual_headers and expected_shots == [1]:
            detail = _insert_ref_first_shot_header(detail, len(detail))
        elif (expected_shots[0] == 1 and actual_headers == expected_shots[1:]
              and header_matches):
            detail = _insert_ref_first_shot_header(detail, header_matches[0].start())
        sections["detailed_description"] = detail
    return "\n\n".join(f"{field}:\n{sections[field]}" for field in REF_PROMPT_FIELDS)


def _canonical_ref_label(match: re.Match[str]) -> str:
    return f"<{match.group(1).title()} {int(match.group(2))}>"


def _ref_prompt_structure_issues(prompt: str, label_plan: dict[str, dict[str, str]]) -> list[str]:
    matches = list(_REF_FIELD_PATTERN.finditer(prompt))
    sections = _ref_prompt_sections(prompt)
    issues: list[str] = []
    if len(matches) != len(REF_PROMPT_FIELDS) or list(sections) != list(REF_PROMPT_FIELDS):
        issues.append("Use exactly the six REF2VA sections once and in the required order.")
        return issues
    empty_sections = [field for field in REF_PROMPT_FIELDS if not sections[field].strip()]
    if empty_sections:
        issues.append("Do not leave REF2VA sections empty: " + ", ".join(empty_sections) + ".")

    expected_labels = list(label_plan)
    definition_labels: list[str] = []
    for line in sections["subject_definitions"].splitlines():
        match = re.match(r"\s*<(Subject|Picture|Video|Audio)\s+(\d+)>", line, flags=re.IGNORECASE)
        if match:
            definition_labels.append(_canonical_ref_label(match))
    if definition_labels != expected_labels:
        issues.append(
            f"Define exactly these reference labels once and in order: {', '.join(expected_labels) or 'none'}."
        )
    definition_lines = {
        _canonical_ref_label(match): line
        for line in sections["subject_definitions"].splitlines()
        if (match := re.match(r"\s*<(Subject|Picture|Video|Audio)\s+(\d+)>", line, flags=re.IGNORECASE))
    }
    for label, plan in label_plan.items():
        if plan["kind"] != "Subject" or label not in definition_lines:
            continue
        source = plan["source"]
        if source.casefold() not in definition_lines[label].casefold():
            issues.append(f"{label} must cite its source asset {source} in subject_definitions.")

    retention_labels: list[str] = []
    retention_markers: dict[str, str] = {}
    for line in sections["retention_analysis"].splitlines():
        match = re.match(
            r"\s*<(Subject|Picture|Video|Audio)\s+(\d+)>\s*(?:\([^)]*\))?\s*:\s*"
            r"(fully_preserved|partially_preserved|attribute_transfer|weak_reference|"
            r"fully_copy|partially_copy|reference)\b",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            label = _canonical_ref_label(match)
            retention_labels.append(label)
            retention_markers[label] = match.group(3).lower()
    if retention_labels != expected_labels:
        issues.append("retention_analysis must contain exactly one ordered entry for every defined label.")
    for label, plan in label_plan.items():
        if label in retention_markers and retention_markers[label] != plan["marker"]:
            issues.append(f"{label} must use retention marker {plan['marker']} for its defined role.")
    if re.search(r"\(S\d+\)", sections["retention_analysis"], flags=re.IGNORECASE):
        issues.append("Do not place speaker IDs in retention_analysis.")

    summary_labels = {
        _canonical_ref_label(match) for match in _REF_LABEL_PATTERN.finditer(sections["summary"])
    }
    missing_summary = [label for label in expected_labels if label not in summary_labels]
    if missing_summary:
        issues.append("summary must mention every defined reference relationship: " + ", ".join(missing_summary) + ".")
    prefix = re.match(r"\s*\[([^\]]+)\]", sections["summary"])
    allowed_tasks = {
        "keyframe completion", "reference generation", "video editing",
        "video continuation", "audio reuse", "audio reference",
    }
    if not prefix or any(task.strip().lower() not in allowed_tasks for task in prefix.group(1).split("+")):
        issues.append("summary must begin with only valid REF2VA task types joined by ' + '.")

    downstream_text = "\n".join(sections[field] for field in REF_PROMPT_FIELDS[1:])
    downstream_labels = {
        _canonical_ref_label(match) for match in _REF_LABEL_PATTERN.finditer(downstream_text)
    }
    unexpected = sorted(downstream_labels.difference(expected_labels))
    if unexpected:
        issues.append("Remove undefined or source-only labels outside subject_definitions: " + ", ".join(unexpected) + ".")
    visual_labels = [
        label for label, plan in label_plan.items()
        if plan["kind"] in {"Subject", "Picture", "Video"}
    ]
    detailed_labels = {
        _canonical_ref_label(match)
        for match in _REF_LABEL_PATTERN.finditer(sections["detailed_description"])
    }
    missing_visual = [label for label in visual_labels if label not in detailed_labels]
    if missing_visual:
        issues.append("detailed_description must apply every defined visual relationship: " + ", ".join(missing_visual) + ".")
    audio_labels = [
        label for label, plan in label_plan.items() if plan["kind"] == "Audio"
    ]
    audio_application_text = "\n".join((
        sections["detailed_description"], sections["overall_soundscape"],
        sections["non_diegetic_music"],
    ))
    applied_audio_labels = {
        _canonical_ref_label(match) for match in _REF_LABEL_PATTERN.finditer(audio_application_text)
    }
    missing_audio = [label for label in audio_labels if label not in applied_audio_labels]
    if missing_audio:
        issues.append(
            "Apply every audio relationship in detailed_description, overall_soundscape, or "
            "non_diegetic_music as appropriate: " + ", ".join(missing_audio) + "."
        )
    if re.search(r"^\s*integrated_multimodal_description\s*:", prompt, flags=re.IGNORECASE | re.MULTILINE):
        issues.append("REF2VA must use detailed_description, not integrated_multimodal_description.")
    return issues


def _explicit_project_context(project: dict[str, Any]) -> str:
    values = [project.get("user_request", ""), project.get("constraints", ""), project.get("verbatim_content", "")]
    for shot in project.get("shots", []):
        values.append(shot.get("visual_action", ""))
    return "\n".join(_clean_text(value) for value in values if _clean_text(value))


def _ref_prompt_semantic_issues(prompt: str, project: dict[str, Any], explicit_context: str,
                                visual_evidence: dict[str, str] | None = None) -> list[str]:
    sections = _ref_prompt_sections(prompt)
    if any(field not in sections for field in REF_PROMPT_FIELDS):
        return []
    explicit = explicit_context.casefold()
    descriptive = "\n".join((
        sections["subject_definitions"], sections["summary"], sections["detailed_description"],
    )).casefold()
    issues: list[str] = []
    inferred_terms = {
        "young": ("젊은", "어린"),
        "teenage": ("청소년", "십대"),
        "middle-aged": ("중년",),
        "elderly": ("노인", "고령"),
        "east asian": ("동아시아",),
        "asian": ("아시아",),
        "japanese": ("일본인",),
        "korean": ("한국인",),
        "chinese": ("중국인",),
        "ukrainian": ("우크라이나인",),
    }
    for term, aliases in inferred_terms.items():
        if term in descriptive and term not in explicit and not any(alias in explicit_context for alias in aliases):
            message = "Remove demographic age, ethnicity, or nationality claims not explicitly supplied by the user."
            if message not in issues:
                issues.append(message)

    reference_model = _reference_model(project)
    label_plan = reference_model["label_plan"]
    retention_lines: dict[str, str] = {}
    for line in sections["retention_analysis"].splitlines():
        match = re.match(r"\s*<(Subject|Picture|Video|Audio)\s+(\d+)>", line, flags=re.IGNORECASE)
        if match:
            retention_lines[_canonical_ref_label(match)] = line
    for label, plan in label_plan.items():
        if plan["marker"] != "weak_reference":
            continue
        detail = retention_lines.get(label, "").split("-", 1)[-1].strip().casefold()
        exhaustive_list = detail.count(",") >= 2 and re.search(r"\b(?:retain|preserve|copy|match)\b", detail)
        strong_claim = re.search(
            r"\b(?:exact(?:ly)?|identical(?:ly)?|fully|strictly)\b|"
            r"\b(?:preserve|retain|copy|match)\s+(?:the\s+)?(?:complete|entire|all|identity)\b",
            detail,
        )
        if exhaustive_list or strong_claim:
            issues.append(
                f"{label} is weak_reference: describe only broad similarity in a small set of "
                "target-relevant characteristics, not an exhaustive or identity-preserving inventory."
            )

    summary = sections["summary"].casefold()
    for label, plan in label_plan.items():
        if plan["kind"] != "Subject" or label.casefold() not in summary:
            continue
        if plan["marker"] == "weak_reference" and re.search(
            r"\b(?:identity|styling|environment|composition|pose|camera|palette)\b", summary,
        ):
            issues.append(
                f"{label} is a weak general reference; summary must describe weak appearance guidance only, "
                "not identity, style, environment, composition, pose, camera, or palette transfer."
            )

    source_refs = {ref["label"]: ref for ref in _reference_labels(project.get("references", []))}
    visual_evidence = visual_evidence or {}
    detail_lower = sections["detailed_description"].casefold()
    style_opening = re.split(r"\[shot\s+1\]", detail_lower, maxsplit=1, flags=re.IGNORECASE)[0]
    environment_terms = (
        "beach", "ocean", "sea", "shore", "sand", "coast", "mountain", "forest", "street",
        "corridor", "hallway", "room", "studio", "sky", "cloud", "horizon",
    )
    style_terms = (
        "anime", "photorealistic", "cinematic", "illustration", "watercolor", "oil painting",
        "pixel art", "3d render", "color palette", "high-key", "low-key", "natural lighting",
    )
    def contains_term(text: str, term: str) -> bool:
        return bool(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text))

    def explicitly_transfers_source_style(text: str, label: str) -> bool:
        """Detect source attribution rather than harmless target-style overlap."""
        escaped_label = re.escape(label.casefold())
        attribution_patterns = (
            rf"{escaped_label}(?:'s)?[^.!?\n]{{0,80}}\b(?:style|aesthetic|rendering|lighting|palette)\b",
            rf"\b(?:style|aesthetic|rendering|lighting|palette)\b[^.!?\n]{{0,80}}{escaped_label}",
            r"\b(?:source|reference(?:d)?)(?:\s+(?:image|picture|asset))?(?:'s)?"
            r"[^.!?\n]{0,50}\b(?:style|aesthetic|rendering|lighting|palette)\b",
            r"\b(?:style|aesthetic|rendering|lighting|palette)\b[^.!?\n]{0,50}"
            r"\b(?:from|of|defined by|derived from|matching)\s+(?:the\s+)?"
            r"(?:source|reference(?:d)?(?:\s+(?:image|picture|asset))?)\b",
        )
        return any(re.search(pattern, text) for pattern in attribution_patterns)

    for label, plan in label_plan.items():
        if plan["kind"] != "Subject" or plan["role"] != "subject_identity":
            continue
        source_description = (
            visual_evidence.get(plan["source"])
            or source_refs.get(plan["source"], {}).get("description", "")
        ).casefold()
        if any(
            contains_term(source_description, term) and contains_term(detail_lower, term)
            and not contains_term(explicit, term)
            for term in environment_terms
        ):
            issues.append(
                f"{label} role={plan['role']} does not transfer source environment details; remove unrequested "
                "source setting content unless the target text explicitly requests it."
            )
        shares_unrequested_source_style = any(
            contains_term(source_description, term) and contains_term(style_opening, term)
            and not contains_term(explicit, term)
            for term in style_terms
        )
        style_transfer_violation = plan.get("strength") != "strong" and shares_unrequested_source_style and (
            plan["marker"] == "weak_reference"
            or explicitly_transfers_source_style(style_opening, label)
        )
        if style_transfer_violation:
            issues.append(
                f"{label} role={plan['role']} does not transfer source style; remove unrequested rendering, "
                "lighting, and palette claims unless the target text explicitly requests them."
            )
    silent_tokens = ("silence", "silent", "no sound", "mute", "무음", "소리 없")
    has_visible_action = bool(project.get("user_request") or any(
        shot.get("visual_action") for shot in project.get("shots", [])
    ))
    if (sections["overall_soundscape"].strip().upper() == "N/A" and has_visible_action
            and not any(token in explicit for token in silent_tokens)):
        issues.append("Replace overall_soundscape N/A with concise plausible ambience and physical action sounds unless silence was explicitly requested.")
    return issues


def _i2va_semantic_issues(prompt: str, explicit_context: str,
                          visual_context: str = "") -> list[str]:
    """Detect high-confidence I2VA fidelity failures suitable for one focused retry."""
    lower = prompt.casefold()
    explicit = explicit_context.casefold()
    visual = visual_context.casefold()
    issues: list[str] = []

    unsupported_terms = {
        "young": ("Remove inferred age descriptions.", ("젊은", "어린")),
        "teenage": ("Remove inferred age descriptions.", ("청소년", "십대")),
        "middle-aged": ("Remove inferred age descriptions.", ("중년",)),
        "elderly": ("Remove inferred age descriptions.", ("노인", "고령")),
        "east asian": ("Remove inferred ethnicity descriptions.", ("동아시아",)),
        "asian": ("Remove inferred ethnicity descriptions.", ("아시아",)),
        "japanese": ("Remove inferred nationality descriptions.", ("일본인",)),
        "korean": ("Remove inferred nationality descriptions.", ("한국인",)),
        "chinese": ("Remove inferred nationality descriptions.", ("중국인",)),
        "ukrainian": ("Remove inferred nationality descriptions.", ("우크라이나인",)),
        "blood": ("Remove graphic effects not explicitly supplied by the user.", ("피가", "피를", "피로", "피범벅", "혈액", "출혈")),
        "bone fragments": ("Remove graphic effects not explicitly supplied by the user.", ("뼈 조각", "뾏조각")),
        "gore": ("Remove graphic effects not explicitly supplied by the user.", ("고어",)),
    }
    for term, (message, aliases) in unsupported_terms.items():
        supported = term in explicit or any(alias in explicit_context for alias in aliases)
        if term in lower and not supported and message not in issues:
            issues.append(message)

    for term in ("revolver", "semi-automatic"):
        if term in lower and term not in explicit and term not in visual:
            message = "Do not specialize the weapon type beyond the supplied evidence."
            if message not in issues:
                issues.append(message)

    hidden_source_patterns = (
        "unseen object", "just below the frame", "below the frame's bottom edge",
        "below the frame’s bottom edge", "off-frame object", "out of frame to grasp",
    )
    if any(pattern in lower and pattern not in explicit for pattern in hidden_source_patterns):
        issues.append("Remove the invented hidden or off-frame source; describe only the first visible entry.")

    speculative_alternatives = ("oil or water", "photograph or rendering", "smiling or neutral")
    if any(pattern in lower and pattern not in explicit for pattern in speculative_alternatives):
        issues.append("Remove speculative alternatives joined by 'or'.")
    if any(term in lower and term not in explicit for term in ("likely", "suggesting", "suggests")):
        issues.append("Replace speculative interpretation with directly observable facts.")

    if re.search(r"\bmedium shot\b[^.]{0,180}\bmid[- ]thigh", lower):
        issues.append("Use one framing term consistent with a mid-thigh crop.")
    if re.search(r"\bslightly low\b[^.]{0,40}\beye[- ]level\b", lower):
        issues.append("Use one non-contradictory camera angle.")
    if ("hiss of smoke" in lower or "hiss of dissolving smoke" in lower) and "hiss" not in explicit:
        issues.append("Remove unsupported smoke sound effects.")

    unsupported_consequences = (
        "head snaps", "head jerks", "body begins to slump", "begins to slump",
        "head to jerk", "impact causes her head", "impact causes his head",
        "sways slightly from the force", "from the force of the movement",
        "due to the impact", "from the impact", "soft thud of the woman's head",
        "soft thud of the woman’s head",
    )
    if any(phrase in lower and phrase not in explicit for phrase in unsupported_consequences):
        issues.append("Remove physical consequences and reaction sounds not explicitly requested by the user.")
    return issues


def _sanitize_i2va_semantics(prompt: str, explicit_context: str) -> str:
    """Apply narrow deterministic repairs after a failed semantic LLM correction."""
    text = re.sub(
        r"\bmedium shot\b(?=[^.]{0,180}\bmid[- ]thigh)",
        "medium-full shot",
        prompt,
        flags=re.IGNORECASE,
    )
    explicit = explicit_context.casefold()

    speculative_patterns: list[re.Pattern[str]] = []
    if "suggesting" not in explicit and "suggests" not in explicit:
        speculative_patterns.extend((
            re.compile(r"\b(?:suggesting|which\s+suggests)\b", flags=re.IGNORECASE),
            re.compile(r"\bsuggests\b", flags=re.IGNORECASE),
        ))
    if "likely" not in explicit:
        speculative_patterns.append(re.compile(r"\blikely\b", flags=re.IGNORECASE))

    if speculative_patterns:
        parts = re.split(r"(?<=[.!?])(\s+)", text)
        for index in range(0, len(parts), 2):
            sentence = parts[index]
            matches = [pattern.search(sentence) for pattern in speculative_patterns]
            matches = [match for match in matches if match]
            if not matches:
                continue
            first = min(matches, key=lambda match: match.start())
            prefix = sentence[:first.start()].rstrip(" ,;:-")
            words = re.findall(r"[A-Za-z]+", prefix)
            if len(words) < 3 or (words and words[-1].casefold() in {
                "is", "are", "was", "were", "seems", "appears",
            }):
                prefix = ""
            parts[index] = prefix + ("." if prefix and not prefix.endswith((".", "!", "?")) else "")
            if (not parts[index] and index + 1 < len(parts)
                    and "\n" not in parts[index + 1]):
                parts[index + 1] = ""
        text = "".join(parts)

    if "oil or water" not in explicit:
        text = re.sub(
            r"(?:the\s+application\s+of\s+)?(?:body\s+)?oil\s+or\s+water",
            "a visible sheen",
            text,
            flags=re.IGNORECASE,
        )
    if "photograph or rendering" not in explicit:
        text = re.sub(r"photograph\s+or\s+rendering", "image", text, flags=re.IGNORECASE)
    if "smiling or neutral" not in explicit:
        text = re.sub(r"smiling\s+or\s+neutral", "restrained", text, flags=re.IGNORECASE)

    consequence_patterns: list[tuple[str, tuple[str, ...]]] = [
        (
            r"\b(?:(?:her|his|their|the woman's|the woman’s|the subject's|the subject’s)\s+)?"
            r"head\s+(?:snaps|jerks)(?:\s+sharply)?(?:\s+to\s+(?:her|his|their)\s+\w+\s+side)?",
            ("head snaps", "head jerks", "머리가 획", "고개가 획", "머리가 꺽", "고개가 꺽"),
        ),
        (
            r"\b(?:the\s+)?impact\s+causes\s+(?:her|his|their|the\s+subject's|the\s+subject’s)\s+"
            r"head\s+to\s+jerk(?:\s+sharply)?(?:\s+to\s+(?:her|his|their)\s+\w+\s+side)?",
            ("impact causes her head", "impact causes his head", "head to jerk", "충격으로 머리", "충격으로 고개"),
        ),
        (
            r"\b(?:(?:her|his|their|the woman's|the woman’s|the subject's|the subject’s)\s+)?"
            r"body\s+begins\s+to\s+slump(?:\s+slightly\s+forward)?",
            ("begins to slump", "body slumps", "쓰러", "주저앉", "고꾸라"),
        ),
        (
            r"\b(?:(?:her|his|their|the woman's|the woman’s|the subject's|the subject’s)\s+)?"
            r"body\s+(?:remains\s+upright\s+but\s+)?sways(?:\s+slightly)?\s+from\s+the\s+force"
            r"(?:\s+of\s+the\s+movement)?",
            ("sways slightly from the force", "sways from the force", "몸이 휘청", "몸이 흔들"),
        ),
        (
            r"\bdue\s+to\s+(?:the\s+)?impact\b|\bfrom\s+the\s+impact\b",
            ("due to the impact", "from the impact", "충격으로", "충격 때문"),
        ),
        (
            r"\b(?:the\s+)?soft\s+thud\s+of\b[^.!?\n]*",
            ("soft thud", "둔탁한", "쿠 소리", "쿠하는"),
        ),
    ]
    active = [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern, support_terms in consequence_patterns
        if not any(term in explicit for term in support_terms)
    ]
    if not active:
        return text.strip()

    parts = re.split(r"(?<=[.!?])(\s+)", text)
    for index in range(0, len(parts), 2):
        sentence = parts[index]
        matches = [pattern.search(sentence) for pattern in active]
        matches = [match for match in matches if match]
        if not matches:
            continue
        first = min(matches, key=lambda match: match.start())
        prefix = sentence[:first.start()]
        prefix = re.sub(
            r"(?:[,;:]\s*)?(?:(?:and|followed\s+(?:immediately\s+)?by|with)\s+)?"
            r"(?:the\s+)?$",
            "",
            prefix,
            flags=re.IGNORECASE,
        ).rstrip(" ,;:-")
        # A stranded possessive or pronoun contains no useful event.
        if re.fullmatch(r"(?:her|his|their|the woman'?s|the subject'?s)?", prefix, flags=re.IGNORECASE):
            prefix = ""
        parts[index] = prefix + ("." if prefix and not prefix.endswith((".", "!", "?")) else "")
        if (not parts[index] and index + 1 < len(parts)
                and "\n" not in parts[index + 1]):
            parts[index + 1] = ""
    return "".join(parts).strip()


def _qwen_reference_plan(project: dict[str, Any], effective_seconds: float,
                         visual_evidence: dict[str, str]) -> str:
    references = _reference_labels(project["references"])
    if not references:
        return "REFERENCE_PLAN:\nnone"
    refs_by_label = {ref["label"]: ref for ref in references}
    blocks: list[str] = []

    def evidence_for(source_label: str) -> str:
        if visual_evidence.get(source_label):
            return visual_evidence[source_label]
        ref = refs_by_label.get(source_label, {})
        if ref.get("type") == "picture" and ref.get("image_filename"):
            return "pending role-aware image analysis during enhancement"
        if ref.get("type") == "video" and ref.get("video_filename"):
            return "pending duration-limited ordered-frame analysis during enhancement"
        return "not supplied"

    if project["mode"] == "REF2VA":
        model = _reference_model(project)
        for label, plan in model["label_plan"].items():
            source = plan["source"]
            ref = refs_by_label.get(source, {})
            lines = [
                label,
                f"source: {source}",
                f"media_type: {str(ref.get('type') or plan['kind']).lower()}",
                f"role: {plan['role']}",
            ]
            if plan["kind"] == "Subject":
                lines.append(f"input_strength_for_definition_scope_only: {plan.get('strength', 'normal')}")
            lines.extend((
                f"retention_output_marker: {plan['marker']}",
                f"contract: {plan['contract']}",
            ))
            if ref.get("description"):
                lines.append(f"user_metadata: {ref['description']}")
            if ref.get("type") == "picture":
                lines.append(f"visual_evidence: {evidence_for(source)}")
                if ref.get("role") == "frame":
                    frame_index = min(
                        max(0, int(ref.get("frame_index", 0))),
                        max(0, align_frame_count(project["requested_duration"]) - 1),
                    )
                    lines.extend((
                        "anchor: exact whole frame at the assigned timeline position",
                        f"anchor_frame_index: {frame_index}",
                        f"anchor_time_seconds: {frame_index / MODEL_FPS:.3f}",
                        "anchor_contract: reach this complete image state exactly at this frame through continuous in-shot motion, then continue chronologically; this anchor never creates a cut or transition",
                    ))
            elif ref.get("type") == "video":
                lines.append(f"temporal_visual_evidence: {evidence_for(source)}")
            if ref.get("duration"):
                duration_key = (
                    "selected_source_duration_seconds"
                    if ref.get("type") == "video"
                    else "source_duration_seconds"
                )
                lines.append(f"{duration_key}: {ref['duration']:.2f}")
                if ref.get("type") == "video":
                    lines.append(f"source_trim_start_seconds: {ref.get('trim_start', 0.0):.2f}")
                    lines.append(f"target_timeline_start_seconds: {ref.get('timeline_start', 0.0):.2f}")
            blocks.append("\n".join(lines))
        return (
            "REFERENCE_PLAN:\n"
            f"task_types: {' + '.join(model['task_types'])}\n\n"
            + "\n\n".join(blocks)
        )

    for ref in references:
        lines = [ref["label"], f"media_type: {ref['type']}", f"role: {ref['role']}"]
        if ref["role"] == "first_frame":
            lines.extend(("anchor: exact opening frame", "anchor_time_seconds: 0.00"))
        elif ref["role"] == "last_frame":
            lines.extend(("anchor: exact final frame", f"anchor_time_seconds: {effective_seconds:.2f}"))
        elif ref["role"] == "frame":
            frame_index = min(
                max(0, int(ref.get("frame_index", 0))),
                max(0, align_frame_count(project["requested_duration"]) - 1),
            )
            lines.extend((
                "anchor: exact whole frame at the assigned timeline position",
                f"anchor_frame_index: {frame_index}",
                f"anchor_time_seconds: {frame_index / MODEL_FPS:.3f}",
                "anchor_contract: reach this complete image state exactly at this frame through continuous in-shot motion, then continue chronologically; this anchor never creates a cut or transition",
            ))
        if ref["description"]:
            lines.append(f"user_metadata: {ref['description']}")
        if ref["type"] == "picture":
            lines.append(f"visual_evidence: {evidence_for(ref['label'])}")
        elif ref["type"] == "video":
            lines.append(f"temporal_visual_evidence: {evidence_for(ref['label'])}")
        if ref["duration"]:
            duration_key = (
                "selected_source_duration_seconds"
                if ref["type"] == "video"
                else "source_duration_seconds"
            )
            lines.append(f"{duration_key}: {ref['duration']:.2f}")
            if ref["type"] == "video":
                lines.append(f"source_trim_start_seconds: {ref.get('trim_start', 0.0):.2f}")
                lines.append(f"target_timeline_start_seconds: {ref.get('timeline_start', 0.0):.2f}")
        blocks.append("\n".join(lines))
    plan = "REFERENCE_PLAN:\n" + "\n\n".join(blocks)
    if project["mode"] == "FL2VA":
        plan += (
            "\n\nENDPOINT_CONTRACT:\n"
            "Picture 1 is the exact whole frame only at 0.00 seconds.\n"
            f"Picture 2 is the exact whole frame only at {effective_seconds:.2f} seconds.\n"
            "Picture 2 is not the opening state of the final shot. The final shot continues from the preceding "
            "transition and reaches Picture 2 only in its final frames.\n"
            "Do not merge identities or transfer clothing, anatomy, materials, or features between distinct "
            "visible entities unless the user explicitly uses morph or transform language. Punch, hit, fall, "
            "enter, exit, and cut are not transformation requests. Content absent from Picture 2 must leave "
            "the frame before the endpoint.\n"
            "entity_binding: the entity visible in Picture 1 owns only Picture 1 traits; the entity visible in "
            "Picture 2 owns only Picture 2 traits. If the final shot introduces an entity matching Picture 2, "
            "that introduced entity is the final-frame entity. Never apply Picture 2 traits to the Picture 1 "
            "entity without an explicit morph or transform request."
        )
    return plan


def _qwen_video_timeline_plan(project: dict[str, Any], effective_seconds: float) -> str:
    labeled = _reference_labels(project.get("references", []))
    videos = [ref for ref in labeled if ref.get("type") == "video" and ref.get("duration")]
    if not videos:
        return "VIDEO_TIMELINE_PLAN:\nnone"

    placements: list[tuple[float, float, str]] = []
    lines = [
        "VIDEO_TIMELINE_PLAN:",
        f"target_range_seconds: 0.000-{effective_seconds:.3f}",
    ]
    for ref in videos:
        source_start, visible_duration, target_start = _visible_video_selection(ref, effective_seconds)
        if visible_duration <= 0:
            continue
        target_end = target_start + visible_duration
        source_end = source_start + visible_duration
        placements.append((target_start, target_end, ref["label"]))
        lines.append(
            f"{ref['label']}: target {target_start:.3f}-{target_end:.3f}; "
            f"selected source {source_start:.3f}-{source_end:.3f}; preset {ref.get('role', 'none')}"
        )

    merged: list[list[float]] = []
    for start, end, _label in sorted(placements):
        if not merged or start > merged[-1][1] + 1e-6:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor + 1e-6:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < effective_seconds - 1e-6:
        gaps.append((cursor, effective_seconds))
    lines.append(
        "uncovered_target_intervals: "
        + (", ".join(f"{start:.3f}-{end:.3f}" for start, end in gaps) if gaps else "none")
    )
    lines.extend((
        "timeline_contract:",
        "- Apply each <Video N> only inside its target interval; never stretch, freeze, loop, or hold it across an uncovered interval.",
        "- In every uncovered interval, execute the applicable SHOT_PLAN action and bridge only the adjacent boundary states needed for continuity.",
        "- A later video interval begins from that video's selected-source opening state, not its ending state.",
        "- Do not infer that people in different videos are the same person unless TARGET_REQUEST explicitly links them.",
        "- Do not invent a cut at a video boundary unless requested or required by an actual discontinuity; otherwise use a coherent continuous bridge.",
    ))
    return "\n".join(lines)


def _qwen_shot_plan(project: dict[str, Any], effective_seconds: float,
                    aliases: dict[str, str]) -> str:
    requested_seconds = sum(float(shot["duration"]) for shot in project["shots"])
    scale = effective_seconds / requested_seconds if requested_seconds > 0 else 1.0
    cursor = 0.0
    blocks: list[str] = []
    for index, shot in enumerate(project["shots"], 1):
        shot_seconds = float(shot["duration"]) * scale
        end = cursor + shot_seconds
        lines = [f"[Shot {index}]", f"time_range_seconds: {cursor:.3f}-{end:.3f}"]
        if index > 1:
            lines.append(f"required_output_header: [Shot {index}] At {format_timestamp(cursor)},")
        action = _replace_aliases(shot["visual_action"], aliases)
        if action:
            lines.append(f"visual_action: {action}")
            lines.append("action_contract: preserve every explicit action above in the same order; omit none")
            lines.append(
                "semantic_lock: translate faithfully; preserve every explicitly named actor, body part, "
                "object, quantity, direction, simultaneity, and physical action verb; never replace one with a broader state or euphemism"
            )
            lines.append(
                "motion_semantics_contract: preserve the source verb at its original specificity. If it denotes "
                "continuous or repeated motion, show distinct ongoing motion phases appropriate to that verb rather "
                "than replacing it with a single contact, broad pose, or static hold. Preserve stated actors, limb or "
                "hand count, contact target, direction, and simultaneity without inventing an exact cycle or step count"
            )
        if project["mode"] == "FL2VA" and index == len(project["shots"]):
            lines.extend((
                "opening_state: continue the incomplete transition from the preceding shot; do not reveal the completed Picture 2",
                "entity_continuity: an entering entity that matches Picture 2 is the same final-frame entity; do not rename or duplicate it",
                f"required_end_state: exact whole-frame match to Picture 2 at {effective_seconds:.2f} seconds",
            ))
        blocks.append("\n".join(lines))
        cursor = end
    return "SHOT_PLAN:\n" + "\n\n".join(blocks)


_TARGET_STYLE_PATTERNS = (
    (re.compile(r"(?i)(?:\b3d\s*(?:cg|cgi|animation)\b|3d\s*애니메이션)"), "3D CG animation"),
    (re.compile(r"(?i)(?:\b2d\s*animation\b|2d\s*애니메이션)"), "2D animation"),
    (re.compile(r"(?i)(?:\blive[\s-]*action\b|실사)"), "live-action"),
    (re.compile(r"(?i)(?:\bphotorealistic\b|포토리얼|사실적인\s*사진)"), "photorealistic live-action"),
    (re.compile(r"(?i)(?:\bstop[\s-]*motion\b|스톱\s*모션)"), "stop-motion"),
    (re.compile(r"(?i)(?:\bclaymation\b|클레이메이션)"), "claymation"),
    (re.compile(r"(?i)(?:\banime[\s-]*style(?:d)?\b|애니메\s*스타일|애니메이션풍)"), "anime-style animation"),
)
_LARGE_MOTION_RE = re.compile(
    r"(?i)(?:걷|걸|달리|뛰|기어|점프|춤|회전|\bwalk|\brun|\bcrawl|\bjump|\bdance|\bspin|\bturn)"
)


def _project_instruction_text(project: dict[str, Any]) -> str:
    return "\n".join((
        _clean_text(project.get("user_request")),
        *(_clean_text(shot.get("visual_action")) for shot in project.get("shots", [])),
        _clean_text(project.get("constraints")),
    ))


def _extract_target_style_lock(project: dict[str, Any]) -> str:
    text = _project_instruction_text(project)
    for pattern, canonical in _TARGET_STYLE_PATTERNS:
        if pattern.search(text):
            return canonical
    return ""


def _requires_action_visibility_lock(project: dict[str, Any]) -> bool:
    return bool(_LARGE_MOTION_RE.search(_project_instruction_text(project)))


def _enhanced_output_budget(effective_seconds: float, shot_count: int) -> str:
    if effective_seconds <= 6.5:
        words = "140-220" if shot_count == 1 else "180-300 total"
    elif effective_seconds <= 12.5:
        words = "220-360" if shot_count == 1 else "300-480 total"
    else:
        words = "380-620 total"
    return (
        "OUTPUT_BUDGET:\n"
        f"recommended_english_words: {words}\n"
        "priority: concise action clarity over exhaustive appearance inventory or decorative prose"
    )


def build_video_prompt(project: dict[str, Any], effective_seconds: float,
                       visual_evidence: dict[str, str] | None = None) -> str:
    """Build compact mode data for the single-pass Qwen H3 rewriter."""
    visual_evidence = visual_evidence or {}
    mode = project["mode"]
    model = _reference_model(project) if mode == "REF2VA" else None
    aliases = model["aliases"] if model else {}
    user_request = _replace_aliases(project["user_request"], aliases)
    target_style = _extract_target_style_lock(project)
    has_strong_subject = bool(model and any(
        plan.get("kind") == "Subject" and plan.get("strength") == "strong"
        for plan in model["label_plan"].values()
    ))
    if mode == "I2VA":
        reference_style_policy = (
            "Picture 1's observable visual medium/rendering style is part of the exact opening-frame anchor; "
            "preserve it unless the user explicitly requests a style change"
        )
    elif mode == "FL2VA":
        reference_style_policy = (
            "the observable media/rendering styles of Picture 1 and Picture 2 are endpoint evidence; preserve a "
            "shared style or describe only the requested transition needed to reach Picture 2"
        )
    elif mode == "L2VA":
        reference_style_policy = (
            "Picture 1's observable visual medium/rendering style is part of the exact final-frame anchor"
        )
    elif has_strong_subject:
        reference_style_policy = (
            "for each strong Subject, preserve that Subject's source visual medium/rendering style as part of its "
            "identity; keep different Subjects' styles independent and do not turn them into a target-wide style, "
            "source setting, composition, camera, lighting setup, or scene-wide palette"
        )
    else:
        reference_style_policy = "analysis evidence only; do not transfer or name it unless explicitly requested"
    sections = [
        "INPUT DATA ONLY - DO NOT COPY THESE KEYS INTO THE FINAL H3 PROMPT",
        "MODE_DATA:\n"
        f"mode: {mode}\n"
        f"requested_duration_seconds: {project['requested_duration']:.2f}\n"
        f"effective_duration_seconds: {effective_seconds:.2f}\n"
        f"shot_count: {len(project['shots'])}",
        "STYLE_POLICY:\n"
        "target_video_style: use only when explicitly requested in TARGET_REQUEST, SHOT_PLAN visual_action, or CONSTRAINTS\n"
        "when_unspecified: omit any target-wide style invented beyond a concrete keyframe or Strong Subject contract\n"
        f"reference_visual_style: {reference_style_policy}",
        "CAMERA_POLICY:\n"
        "source: infer composition, viewpoint, camera behavior, and explicit transition intent from TARGET_REQUEST and SHOT_PLAN visual_action\n"
        "per_shot: choose framing that contains the largest required visible action and final state; use a static shot only when all required events remain inside the opening crop, otherwise use one motivated reframe\n"
        "expression: write camera behavior as natural English; add amplitude and speed only when meaningful\n"
        "shot_boundaries: each configured shot after Shot 1 is an ordinary cut at its time-range start; use cross-dissolve, fade, or wipe only when explicitly requested\n"
        "frame_anchor_editing: Picture anchor times never create cuts or transitions; interpolate continuously between anchors inside each configured shot\n"
        "restraint: do not invent decorative motion or a new cut when a static camera or a small continuous camera move presents the action clearly",
    ]
    if user_request:
        sections.append("TARGET_REQUEST:\n" + user_request)
    if target_style:
        if mode == "FL2VA":
            style_application = (
                "use this target style except where an explicitly requested style transition is needed to reach an exact endpoint; "
                "the assigned Picture 1 and Picture 2 states remain exact at their timestamps"
            )
        elif mode == "L2VA":
            style_application = (
                "use this target style along the approach when compatible, but converge to Picture 1's exact final-frame medium and rendering state"
            )
        elif mode == "REF2VA":
            style_application = (
                "apply this as the target-video style; it overrides incompatible source presentation, while Strong Subjects retain identity, "
                "design, and compatible material traits; preserve a conflicting source medium only when mixed-media treatment is explicitly requested"
            )
        else:
            style_application = "begin Shot 1 with this style and maintain it throughout"
        sections.append(
            "TARGET_STYLE_LOCK:\n"
            f"canonical_style: {target_style}\n"
            "source: explicit user request\n"
            "priority: overrides an incompatible source-image medium or photographed-object presentation\n"
            f"application: {style_application}"
        )
        if project.get("references"):
            sections.append(
                "REFERENCE_MEDIUM_CONTRACT:\n"
                "use_reference_for: identity, design, assigned anchor pose and composition, action-relevant "
                "objects, and environment\n"
                f"target_medium: {target_style}\n"
                "exclude_incompatible_source_presentation: photographed collectible, rigid display object, product photography, "
                "and display-only joint sounds\n"
                "material_policy: preserve visible plastic, resin, vinyl, jelly, glass, or translucent material only when it is part "
                "of the character or object design and compatible with the requested target style"
            )
    if _requires_action_visibility_lock(project):
        sections.append(
            "ACTION_VISIBILITY_LOCK:\n"
            "A frame anchor is exact at its assigned time. Outside that instant, use one small motivated reframe "
            "only when the opening crop cannot show the requested body movement or interaction. Do not describe "
            "off-frame foot placement, steps, or contact sounds as visible action. For forward locomotion from a "
            "close crop, reveal enough of the stride to make locomotion observable."
        )
    if project.get("enhance") is True:
        sections.append(_enhanced_output_budget(effective_seconds, len(project["shots"])))
    sections.extend((
        _qwen_reference_plan(project, effective_seconds, visual_evidence),
        _qwen_video_timeline_plan(project, effective_seconds),
        _qwen_shot_plan(project, effective_seconds, aliases),
        "AUDIO_POLICY:\n"
        "source: infer audio intent only from TARGET_REQUEST, SHOT_PLAN visual_action, and locked audio relationships in REFERENCE_PLAN\n"
        "diegetic_and_ambience: derive concise synchronized action sounds and plausible environmental ambience from requested visible events and setting\n"
        "material_sound_fidelity: do not invent fabric rustle, clothing creak, plastic joint noise, smoke hiss, or another material sound unless that material and audible cause are established by user text or reference evidence\n"
        "music_routing: music with an in-scene source is diegetic and belongs in the shot timeline; requested BGM, background music, soundtrack, score, or source-free music belongs in non_diegetic_music\n"
        "when_music_unspecified: output non_diegetic_music: N/A",
    ))
    if project["constraints"]:
        sections.append("CONSTRAINTS:\n" + _replace_aliases(project["constraints"], aliases))
    if project["verbatim_content"]:
        sections.append("VERBATIM_CONTENT:\n" + project["verbatim_content"])
    return "\n\n".join(sections)


def build_llm_prompt(project: dict[str, Any], video_prompt: str) -> str:
    mode = project["mode"]
    expected_shots = list(range(1, len(project["shots"]) + 1))
    final_shot = len(expected_shots)
    effective_seconds = align_frame_count(project["requested_duration"]) / MODEL_FPS
    reference_model = _reference_model(project) if mode == "REF2VA" else None
    content_locks = _input_content_locks(project)
    system_prompt = "\n\n".join((
        _mode_prompt_preamble(mode),
        MODE_LLM_SYSTEM_PROMPTS[mode],
        _reference_system_modules(project) if mode == "REF2VA" else "",
        _single_pass_output_lock(
            mode, effective_seconds, final_shot, expected_shots, reference_model, content_locks,
        ),
    ))
    return (
        "SYSTEM PROMPT:\n"
        f"{system_prompt}\n\n"
        f"USER DATA FOR ACTIVE MODE {mode}:\n"
        f"{video_prompt}"
    )


def _llm_roots() -> list[str]:
    roots: list[str] = []
    try:
        import folder_paths

        try:
            roots.extend(folder_paths.get_folder_paths("LLM"))
        except KeyError:
            pass
        fallback = os.path.join(folder_paths.models_dir, "LLM")
        roots.append(fallback)
    except ImportError:
        pass
    return list(dict.fromkeys(os.path.abspath(path) for path in roots if path))


def _is_writer_gguf(path: str) -> bool:
    name = os.path.basename(path).lower()
    excluded = ("mmproj", "lora", "vision", "draft")
    return name.endswith(".gguf") and not any(token in name for token in excluded)


def list_enhance_models(roots: list[str] | None = None) -> list[dict[str, Any]]:
    roots = roots if roots is not None else _llm_roots()
    default_installed = any(
        os.path.isfile(path)
        for root in roots
        for path in glob.iglob(os.path.join(root, "**", DEFAULT_ENHANCE_MODEL_FILE), recursive=True)
    )
    lightx2v = _lightx2v_install_state(roots)
    return [{
        "id": DEFAULT_ENHANCE_MODEL_ID,
        "label": "Qwen3.8-27B Uncensored Q4_K_M",
        "installed": default_installed,
        "size": 0 if default_installed else DEFAULT_ENHANCE_MODEL_SIZE,
    }, {
        "id": LIGHTX2V_MODEL_ID,
        "label": LIGHTX2V_MODEL_DISPLAY_NAME,
        "installed": lightx2v["installed"],
        "size": lightx2v["missing_size"],
    }]


def _lightx2v_model_dir(roots: list[str]) -> str:
    return os.path.join(
        roots[0], "indhic-ai", "MiniMax_H3-Prompt_Rewriter-8B-LORA-Merged-GGUF",
    )


def _lightx2v_install_state(roots: list[str]) -> dict[str, Any]:
    if not roots:
        return {
            "installed": False, "model_installed": False, "mmproj_installed": False,
            "missing_size": LIGHTX2V_TOTAL_SIZE, "model_path": "", "mmproj_path": "",
        }
    model_dir = _lightx2v_model_dir(roots)
    model_path = os.path.join(model_dir, LIGHTX2V_MODEL_FILE)
    mmproj_path = os.path.join(model_dir, LIGHTX2V_MMPROJ_FILE)
    model_installed = os.path.isfile(model_path)
    mmproj_installed = os.path.isfile(mmproj_path)
    return {
        "installed": model_installed and mmproj_installed,
        "model_installed": model_installed,
        "mmproj_installed": mmproj_installed,
        "missing_size": (0 if model_installed else LIGHTX2V_MODEL_SIZE)
                        + (0 if mmproj_installed else LIGHTX2V_MMPROJ_SIZE),
        "model_path": model_path,
        "mmproj_path": mmproj_path,
    }


def list_image_models(roots: list[str] | None = None) -> list[dict[str, Any]]:
    roots = roots if roots is not None else _llm_roots()
    writer_model = next((path for root in roots for path in glob.iglob(os.path.join(root, "**", DEFAULT_ENHANCE_MODEL_FILE), recursive=True) if os.path.isfile(path)), None)
    qwen_mmproj = next((path for root in roots for path in glob.iglob(os.path.join(root, "**", QWEN_IMAGE_MMPROJ_FILE), recursive=True) if os.path.isfile(path)), None)
    qwen_installed = bool(writer_model and qwen_mmproj)
    qwen_missing_size = (0 if writer_model else QWEN_IMAGE_MODEL_SIZE) + (0 if qwen_mmproj else QWEN_IMAGE_MMPROJ_SIZE)
    lightx2v = _lightx2v_install_state(roots)
    return [{
        "id": QWEN_IMAGE_MODEL_ID,
        "label": (
            QWEN_MODEL_DISPLAY_NAME
            + (" · installed" if qwen_installed else f" · {qwen_missing_size / 1e9:.2f} GB missing")
            + f" · {QWEN_MODEL_VRAM_LABEL}"
        ),
        "installed": qwen_installed,
        "size": qwen_missing_size,
        "text_installed": bool(writer_model),
        "vision_installed": bool(qwen_mmproj),
        "text_size": 0 if writer_model else QWEN_IMAGE_MODEL_SIZE,
        "vision_size": 0 if qwen_mmproj else QWEN_IMAGE_MMPROJ_SIZE,
        "enhance_model": DEFAULT_ENHANCE_MODEL_ID,
        "image_model": QWEN_IMAGE_MODEL_ID,
        "supported_modes": list(SUPPORTED_MODES[1:]),
        "runtime": "llama.cpp",
    }, {
        "id": LIGHTX2V_MODEL_ID,
        "label": (
            LIGHTX2V_MODEL_DISPLAY_NAME
            + (" · installed" if lightx2v["installed"] else f" · {lightx2v['missing_size'] / 1e9:.2f} GB missing")
            + f" · {LIGHTX2V_MODEL_VRAM_LABEL} · R2V unsupported"
        ),
        "installed": lightx2v["installed"],
        "size": lightx2v["missing_size"],
        "text_installed": lightx2v["installed"],
        "vision_installed": lightx2v["installed"],
        "text_size": lightx2v["missing_size"],
        "vision_size": 0,
        "enhance_model": LIGHTX2V_MODEL_ID,
        "image_model": LIGHTX2V_MODEL_ID,
        "supported_modes": list(LIGHTX2V_SUPPORTED_MODES),
        "runtime": "llama.cpp-gguf",
    }]


def _download_image_component(repo_id: str, filename: str, local_dir: str, component_size: int,
                              completed_size: int, bundle_size: int, progress=None) -> str:
    from huggingface_hub import hf_hub_download

    stop_monitor = threading.Event()

    def monitor_download() -> None:
        last_size = -1
        while not stop_monitor.wait(0.35):
            candidates = glob.glob(os.path.join(local_dir, "**", "*.incomplete"), recursive=True)
            candidates.append(os.path.join(local_dir, filename))
            size = min(component_size, max((os.path.getsize(path) for path in candidates if os.path.isfile(path)), default=0))
            if size != last_size and progress:
                progress(stage="downloading", message=f"Downloading image model component: {filename}",
                         downloaded=completed_size + size, total=bundle_size)
                last_size = size

    monitor = threading.Thread(target=monitor_download, name="toyxyz-h3-image-download", daemon=True)
    monitor.start()
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    finally:
        stop_monitor.set()
        monitor.join(timeout=1)


def _resolve_lightx2v_model(progress=None) -> tuple[str, str]:
    roots = _llm_roots()
    if not roots:
        raise RuntimeError("ComfyUI has no registered models/LLM directory.")
    state = _lightx2v_install_state(roots)
    if state["installed"]:
        if progress:
            progress(stage="model_ready", message="LightX2V merged GGUF Q8_0 + Vision F16 bundle is installed.")
        return os.path.abspath(state["model_path"]), os.path.abspath(state["mmproj_path"])
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download the LightX2V GGUF bundle.") from exc

    model_dir = _lightx2v_model_dir(roots)
    os.makedirs(model_dir, exist_ok=True)
    total = int(state["missing_size"])
    completed = 0
    try:
        if progress:
            progress(
                stage="downloading", message="Starting LightX2V merged GGUF bundle download.",
                downloaded=0, total=total,
            )
        if not state["model_installed"]:
            _download_image_component(
                LIGHTX2V_MODEL_REPO, LIGHTX2V_MODEL_FILE, model_dir,
                LIGHTX2V_MODEL_SIZE, completed, total, progress,
            )
            completed += LIGHTX2V_MODEL_SIZE
        if not state["mmproj_installed"]:
            _download_image_component(
                LIGHTX2V_MODEL_REPO, LIGHTX2V_MMPROJ_FILE, model_dir,
                LIGHTX2V_MMPROJ_SIZE, completed, total, progress,
            )
        installed = _lightx2v_install_state(roots)
        if not installed["installed"]:
            raise RuntimeError("The LightX2V merged GGUF download completed but required files are missing.")
        if progress:
            progress(
                stage="downloading", message="LightX2V merged GGUF bundle download completed.",
                downloaded=total, total=total,
            )
            progress(
                stage="model_ready", message="LightX2V merged GGUF Q8_0 + Vision F16 bundle is ready.",
                downloaded=total, total=total,
            )
        return os.path.abspath(installed["model_path"]), os.path.abspath(installed["mmproj_path"])
    finally:
        pass


def _resolve_image_model(model_id: str, progress=None) -> tuple[str, str]:
    roots = _llm_roots()
    if not roots:
        raise RuntimeError("ComfyUI has no registered models/LLM directory.")
    if model_id != QWEN_IMAGE_MODEL_ID:
        raise ValueError("Only the Qwen3.8 Vision F16 image analysis bundle is supported.")
    repo_id = QWEN_IMAGE_MODEL_REPO
    model_file = QWEN_IMAGE_MODEL_FILE
    mmproj_file = QWEN_IMAGE_MMPROJ_FILE
    model_size = QWEN_IMAGE_MODEL_SIZE
    mmproj_size = QWEN_IMAGE_MMPROJ_SIZE
    bundle_name = "Qwen3.8 Vision F16"
    model_path = next((path for root in roots for path in glob.iglob(os.path.join(root, "**", model_file), recursive=True) if os.path.isfile(path)), None)
    mmproj_path = next((path for root in roots for path in glob.iglob(os.path.join(root, "**", mmproj_file), recursive=True) if os.path.isfile(path)), None)
    if model_path and mmproj_path:
        if progress:
            progress(stage="image_model_ready", message=f"{bundle_name} image analysis bundle is installed.")
        return os.path.abspath(model_path), os.path.abspath(mmproj_path)
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"huggingface_hub is required to download the {bundle_name} image model bundle.") from exc
    local_dir = roots[0]
    os.makedirs(local_dir, exist_ok=True)
    download_total = (0 if model_path else model_size) + (0 if mmproj_path else mmproj_size)
    completed_size = 0
    if not model_path:
        model_path = _download_image_component(
            repo_id, model_file, local_dir, model_size, 0, download_total, progress,
        )
        completed_size += model_size
    if not mmproj_path:
        mmproj_path = _download_image_component(
            repo_id, mmproj_file, local_dir, mmproj_size, completed_size, download_total, progress,
        )
    if progress:
        progress(stage="image_model_ready", message=f"{bundle_name} image analysis bundle is ready.",
                 downloaded=download_total, total=download_total)
    return os.path.abspath(model_path), os.path.abspath(mmproj_path)


def _resolve_local_model(model_id: str, roots: list[str]) -> str | None:
    if model_id == DEFAULT_ENHANCE_MODEL_ID:
        for root in roots:
            for path in glob.iglob(os.path.join(root, "**", DEFAULT_ENHANCE_MODEL_FILE), recursive=True):
                if os.path.isfile(path):
                    return os.path.abspath(path)
        return None
    if not model_id.startswith("local:"):
        raise ValueError("Unknown enhancement model selection.")
    relative = model_id[len("local:"):].replace("/", os.sep)
    for root in roots:
        root_path = os.path.abspath(root)
        candidate = os.path.abspath(os.path.join(root_path, relative))
        try:
            inside = os.path.commonpath((root_path, candidate)) == root_path
        except ValueError:
            inside = False
        if inside and os.path.isfile(candidate) and _is_writer_gguf(candidate):
            return candidate
    raise FileNotFoundError("The selected local GGUF model is no longer available.")


def _resolve_enhance_model(model_id: str, progress=None) -> str:
    roots = _llm_roots()
    if not roots:
        raise RuntimeError("ComfyUI has no registered models/LLM directory.")
    local = _resolve_local_model(model_id, roots)
    if local:
        if progress:
            progress(stage="model_ready", message="Selected GGUF model is installed.")
        return local
    if model_id != DEFAULT_ENHANCE_MODEL_ID:
        raise FileNotFoundError("The selected enhancement model was not found.")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download the default enhancement model.") from exc
    os.makedirs(roots[0], exist_ok=True)
    stop_monitor = threading.Event()

    def monitor_download() -> None:
        last_size = -1
        while not stop_monitor.wait(0.35):
            candidates = glob.glob(os.path.join(roots[0], "**", "*.incomplete"), recursive=True)
            candidates.append(os.path.join(roots[0], DEFAULT_ENHANCE_MODEL_FILE))
            size = max((os.path.getsize(path) for path in candidates if os.path.isfile(path)), default=0)
            if size != last_size and progress:
                progress(
                    stage="downloading",
                    message="Downloading the default GGUF model.",
                    downloaded=size,
                    total=DEFAULT_ENHANCE_MODEL_SIZE,
                )
                last_size = size

    monitor = threading.Thread(target=monitor_download, name="toyxyz-h3-download-progress", daemon=True)
    monitor.start()
    try:
        if progress:
            progress(stage="downloading", message="Starting model download.", downloaded=0, total=DEFAULT_ENHANCE_MODEL_SIZE)
        downloaded_path = hf_hub_download(
            repo_id=DEFAULT_ENHANCE_MODEL_REPO,
            filename=DEFAULT_ENHANCE_MODEL_FILE,
            local_dir=roots[0],
        )
        if progress:
            progress(
                stage="downloading",
                message="Model download completed.",
                downloaded=DEFAULT_ENHANCE_MODEL_SIZE,
                total=DEFAULT_ENHANCE_MODEL_SIZE,
            )
        return downloaded_path
    finally:
        stop_monitor.set()
        monitor.join(timeout=1)


def _find_llama_cli() -> str:
    configured = os.environ.get("TOYXYZ_LLAMA_CLI", "").strip()
    candidates = [configured, shutil.which("llama-cli") or "", shutil.which("llama-cli.exe") or ""]
    try:
        import folder_paths

        user_root = os.path.join(folder_paths.base_path, "user")
        candidates.extend(glob.glob(os.path.join(user_root, "**", "llama-cli.exe"), recursive=True))
        candidates.extend(glob.glob(os.path.join(user_root, "**", "llama-cli"), recursive=True))
    except ImportError:
        pass
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError(
        "llama-cli was not found. Install llama.cpp, install MiniMax-H3-Prompt-Rewriter-ComfyUI, "
        "or set TOYXYZ_LLAMA_CLI to the executable path."
    )


def _find_llama_server() -> str:
    configured = os.environ.get("TOYXYZ_LLAMA_SERVER", "").strip()
    candidates = [configured, shutil.which("llama-server") or "", shutil.which("llama-server.exe") or ""]
    try:
        cli_path = _find_llama_cli()
        sibling_name = "llama-server.exe" if os.name == "nt" else "llama-server"
        candidates.append(os.path.join(os.path.dirname(cli_path), sibling_name))
    except RuntimeError:
        pass
    try:
        import folder_paths

        user_root = os.path.join(folder_paths.base_path, "user")
        candidates.extend(glob.glob(os.path.join(user_root, "**", "llama-server.exe"), recursive=True))
        candidates.extend(glob.glob(os.path.join(user_root, "**", "llama-server"), recursive=True))
    except ImportError:
        pass
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError("llama-server was not found; image analysis will use llama-cli instead.")


def _available_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class _LlamaServerSession:
    def __init__(self, executable: str, model_path: str, mmproj_path: str,
                 image_model_id: str = DEFAULT_IMAGE_MODEL_ID, context_size: int = 8192,
                 extra_args: list[str] | None = None):
        self.executable = executable
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.image_model_id = image_model_id
        self.context_size = context_size
        self.extra_args = list(extra_args or [])
        self.port = _available_local_port()
        self.process: subprocess.Popen[str] | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 600.0) -> None:
        command = [
            self.executable, "-m", self.model_path, "--mmproj", self.mmproj_path,
            "--host", "127.0.0.1", "--port", str(self.port), "-c", str(self.context_size),
            "-ngl", "all", "-np", "1", "--no-webui", "--log-disable",
            "--jinja", "--timeout", "1800",
        ]
        if self.image_model_id == QWEN_IMAGE_MODEL_ID:
            command.extend([
                "--chat-template-kwargs", '{"enable_thinking":false}',
                "--reasoning", "off",
            ])
        command.extend(self.extra_args)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            text=True, creationflags=creationflags,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited during startup with code {self.process.returncode}.")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.2)
        raise RuntimeError("llama-server did not become ready within 10 minutes.")

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _chat(self, messages: list[dict[str, Any]], max_tokens: int, temperature: float) -> str:
        payload = json.dumps({
            "model": "local-model", "messages": messages, "stream": False,
            "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.05,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                result = json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"llama-server request failed with HTTP {exc.code}: {detail}") from exc
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("llama-server returned an unexpected chat-completion response.") from exc
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            )
        return str(content or "")

    def analyze_image(self, image_path: str, prompt: str) -> str:
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        with open(image_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
            ],
        }]
        return self._chat(messages, max_tokens=700, temperature=0.2)

    def analyze_images(self, image_paths: list[str], captions: list[str], prompt: str,
                       max_tokens: int = 1600) -> str:
        if len(image_paths) != len(captions) or not image_paths:
            raise ValueError("Ordered image paths and captions are required for multimodal analysis.")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path, caption in zip(image_paths, captions):
            mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            with open(image_path, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("ascii")
            content.extend((
                {"type": "text", "text": caption},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
            ))
        return self._chat([{"role": "user", "content": content}], max_tokens=max_tokens, temperature=0.2)

    def chat(self, messages: list[dict[str, Any]], max_tokens: int = 4096,
             temperature: float = 0.0) -> str:
        return self._chat(messages, max_tokens=max_tokens, temperature=temperature)


def _start_persistent_image_server(image_model_id: str, progress=None) -> _LlamaServerSession:
    executable = _find_llama_server()
    model_path, mmproj_path = _resolve_image_model(image_model_id or DEFAULT_IMAGE_MODEL_ID, progress)
    session = _LlamaServerSession(executable, model_path, mmproj_path, image_model_id, context_size=16384)
    try:
        session.start()
    except Exception:
        session.close()
        raise
    return session


def _clean_llm_output(text: str) -> str:
    # Older llama-cli builds can enter conversation mode when the prompt is
    # supplied with -f and mix their banner, echoed input and shutdown message
    # into stdout. Prefer explicit response markers, but keep a defensive
    # fallback for output produced by those builds.
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).replace("\r\n", "\n")
    marked = re.search(r"<H3_PROMPT>\s*(.*?)\s*</H3_PROMPT>", text, flags=re.DOTALL | re.IGNORECASE)
    if marked:
        text = marked.group(1)
    elif "\n> " in text:
        text = text.rsplit("\n> ", 1)[1]
        # The first line is llama-cli's echoed (often truncated) user prompt.
        text = text.partition("\n")[2]
    text = re.sub(r"(?:^|\n)Exiting\.\.\.\s*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    fence = re.fullmatch(r"```(?:text)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    return text


def _resolve_uploaded_image(image: dict[str, Any]) -> str:
    filename = os.path.basename(_clean_text(image.get("filename")))
    subfolder = _clean_text(image.get("subfolder")).replace("\\", "/").strip("/")
    if not filename or os.path.splitext(filename)[1].lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        raise ValueError("Select a supported uploaded image before analysis.")
    import folder_paths

    input_root = os.path.abspath(folder_paths.get_input_directory())
    candidate = os.path.abspath(os.path.join(input_root, subfolder.replace("/", os.sep), filename))
    try:
        inside = os.path.commonpath((input_root, candidate)) == input_root
    except ValueError:
        inside = False
    if not inside or not os.path.isfile(candidate):
        raise FileNotFoundError("The uploaded reference image is unavailable in ComfyUI's input directory.")
    return candidate


def _resolve_uploaded_video(video: dict[str, Any]) -> str:
    filename = os.path.basename(_clean_text(video.get("filename")))
    subfolder = _clean_text(video.get("subfolder")).replace("\\", "/").strip("/")
    if not filename or os.path.splitext(filename)[1].lower() not in VIDEO_EXTENSIONS:
        raise ValueError("Select a supported uploaded video before analysis.")
    import folder_paths

    input_root = os.path.abspath(folder_paths.get_input_directory())
    candidate = os.path.abspath(os.path.join(input_root, subfolder.replace("/", os.sep), filename))
    try:
        inside = os.path.commonpath((input_root, candidate)) == input_root
    except ValueError:
        inside = False
    if not inside or not os.path.isfile(candidate):
        raise FileNotFoundError("The uploaded reference video is unavailable in ComfyUI's input directory.")
    return candidate


def _resolve_uploaded_audio(audio: dict[str, Any]) -> str:
    filename = os.path.basename(_clean_text(audio.get("filename")))
    subfolder = _clean_text(audio.get("subfolder")).replace("\\", "/").strip("/")
    if not filename or os.path.splitext(filename)[1].lower() not in AUDIO_EXTENSIONS:
        raise ValueError("Select a supported uploaded audio file.")
    import folder_paths

    input_root = os.path.abspath(folder_paths.get_input_directory())
    candidate = os.path.abspath(os.path.join(input_root, subfolder.replace("/", os.sep), filename))
    try:
        inside = os.path.commonpath((input_root, candidate)) == input_root
    except ValueError:
        inside = False
    if not inside or not os.path.isfile(candidate):
        raise FileNotFoundError("The uploaded reference audio is unavailable in ComfyUI's input directory.")
    return candidate


def _find_ffmpeg() -> str:
    configured = os.environ.get("TOYXYZ_FFMPEG", "").strip()
    candidates = [configured, shutil.which("ffmpeg") or "", shutil.which("ffmpeg.exe") or ""]
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError):
        pass
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError("FFmpeg was not found. Install FFmpeg or set TOYXYZ_FFMPEG before analyzing videos.")


def _find_ffprobe() -> str | None:
    configured = os.environ.get("TOYXYZ_FFPROBE", "").strip()
    candidates = [configured, shutil.which("ffprobe") or "", shutil.which("ffprobe.exe") or ""]
    ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if ffmpeg:
        candidates.append(os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe" if os.name == "nt" else "ffprobe"))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def _probe_video_duration(video_path: str) -> float | None:
    ffprobe = _find_ffprobe()
    if not ffprobe:
        return None
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    completed = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", video_path],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", timeout=60,
        creationflags=creationflags, check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        duration = float(completed.stdout.strip())
    except ValueError:
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def _extract_video_analysis_frames(video_path: str, duration: float, output_dir: str,
                                   start_time: float = 0.0) -> tuple[list[str], list[float]]:
    duration = min(REF_VIDEO_MAX_SECONDS, max(REF_VIDEO_MIN_SECONDS, float(duration)))
    start_time = max(0.0, float(start_time))
    frame_count = min(VIDEO_ANALYSIS_MAX_FRAMES, max(4, int(math.ceil(duration * 1.5)) + 1))
    sampled_span = max(0.001, duration - min(0.05, duration / 100.0))
    sample_fps = (frame_count - 1) / sampled_span
    output_pattern = os.path.join(output_dir, "frame-%03d.jpg")
    video_filter = f"fps={sample_fps:.8f},scale='min(768,iw)':-2"
    command = [
        _find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss", f"{start_time:.3f}", "-t", f"{duration:.3f}",
        "-i", video_path, "-an", "-vf", video_filter, "-frames:v", str(frame_count),
        "-q:v", "3", "-y", output_pattern,
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    completed = subprocess.run(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", timeout=300,
        creationflags=creationflags, check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.splitlines()[-12:])
        raise RuntimeError(f"Video frame extraction failed with code {completed.returncode}.\n{tail}")
    frame_paths = sorted(glob.glob(os.path.join(output_dir, "frame-*.jpg")))
    if not frame_paths:
        raise RuntimeError("FFmpeg extracted no frames from the selected video segment.")
    timestamps = [min(duration, index / sample_fps) for index in range(len(frame_paths))]
    endpoint_time = max(0.0, duration - 0.05)
    if timestamps[-1] < endpoint_time - 0.05:
        endpoint_temp = os.path.join(output_dir, "endpoint-final.jpg")
        endpoint_command = [
            _find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss", f"{start_time + endpoint_time:.3f}",
            "-i", video_path, "-an", "-vf", "scale='min(768,iw)':-2", "-frames:v", "1",
            "-q:v", "3", "-y", endpoint_temp,
        ]
        endpoint = subprocess.run(
            endpoint_command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=120,
            creationflags=creationflags, check=False,
        )
        if endpoint.returncode == 0 and os.path.isfile(endpoint_temp):
            if len(frame_paths) < frame_count:
                endpoint_path = os.path.join(output_dir, f"frame-{len(frame_paths) + 1:03d}.jpg")
                os.replace(endpoint_temp, endpoint_path)
                frame_paths.append(endpoint_path)
                timestamps.append(endpoint_time)
            else:
                os.replace(endpoint_temp, frame_paths[-1])
                timestamps[-1] = endpoint_time
    return frame_paths, timestamps


def _video_analysis_prompt(role: str, duration: float, timestamps: list[float],
                           start_time: float = 0.0) -> str:
    role = role if role in REFERENCE_ROLES["video"] else "none"
    role_focus = {
        "none": "Describe the observable video content neutrally so the user-written relationship can be applied without guessing.",
        "video_editing": "Prioritize every source element needed for a scoped edit: visible subjects, performances, objects, environment, camera, cuts, timing, and continuity.",
        "video_continuation": "Prioritize the ending state, final composition, positions, motion direction and momentum, camera behavior, lighting, and unresolved actions.",
        "motion": "Prioritize subject actions, pose progression, direction, speed, contacts, interaction timing, and physical rhythm.",
        "camera": "Prioritize shot size, viewpoint, framing changes, camera motion type, direction, amplitude, speed, and stabilization.",
        "cuts_rhythm": "Prioritize shot boundaries, cut times, viewpoint changes, pacing, event rhythm, and temporal structure.",
    }[role]
    timestamp_text = ", ".join(f"{value:.3f}s" for value in timestamps)
    return f"""Analyze the supplied images as chronologically ordered samples from the selected source interval {start_time:.3f}-{start_time + duration:.3f} seconds of one reference video.
Sample timestamps are relative to the selected interval, in image order: {timestamp_text}.
{role_focus}
Infer change only when supported by adjacent samples. Never treat the samples as unrelated images, invent events between them, infer audio, or claim that an unseen detail exists.
Return exactly the eight labeled sections below as compact English evidence. Use explicit time ranges where supported.

VIDEO_OVERVIEW: source duration analyzed, visual medium, probable shot count supported by samples, and overall composition.
SUBJECTS: stable observable identities, clothing, props, initial positions, and which visible entity performs each action.
ACTION_TIMELINE: chronological actions, pose changes, movement paths, contacts, interactions, object states, and final state with timestamps.
CAMERA_EDITING: framing, viewpoint, camera movement, supported cut boundaries, and pacing; write unknown when samples cannot distinguish camera motion from subject motion.
ENVIRONMENT_OBJECTS: location, layout, surfaces, furniture, background elements, and action-relevant object relationships.
STYLE_LIGHTING: observable rendering medium, lighting direction and continuity, palette, materials, reflections, and shadows.
VISIBLE_TEXT: exact readable text with its timestamp; otherwise none visible.
EDIT_CONTINUITY: temporal and structural elements normally preserved by editing: performance, action order, timing, paths, contacts, object interaction, environment, camera, cuts, lighting continuity, and final state. Keep source appearance only in SUBJECTS; do not mark identity, body appearance, hair, clothing, or accessories as mandatory preservation.

Enclose the result exactly once in <VIDEO_ANALYSIS> and </VIDEO_ANALYSIS>."""


def _clean_video_analysis(text: str) -> str:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).replace("\r\n", "\n")
    marked = re.search(r"<VIDEO_ANALYSIS>\s*(.*?)\s*</VIDEO_ANALYSIS>", text, re.DOTALL | re.IGNORECASE)
    if marked:
        text = marked.group(1)
    elif re.search(r"<VIDEO_ANALYSIS>\s*", text, re.IGNORECASE):
        text = re.split(r"<VIDEO_ANALYSIS>\s*", text, maxsplit=1, flags=re.IGNORECASE)[1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"(?:^|\n)Exiting\.\.\.\s*$", "", text, flags=re.IGNORECASE).strip()


def _video_reference_system_modules(project: dict[str, Any]) -> str:
    videos = [ref for ref in project.get("references", []) if ref.get("type") == "video"]
    if not videos:
        return ""
    roles = [
        ref.get("role") if ref.get("role") in REFERENCE_ROLES["video"] else "none"
        for ref in videos
    ]
    mapping = ", ".join(f"<Video {index}>={role}" for index, role in enumerate(roles, 1))
    modules = [SYSTEM_PROMPT_CONFIG["video_reference_common"], f"\nVIDEO PRESET MAP: {mapping}."]
    for role in REFERENCE_ROLES["video"]:
        if role in roles:
            modules.append(SYSTEM_PROMPT_CONFIG["video_reference_roles"][role])
    return "".join(modules)


def _audio_reference_system_modules(project: dict[str, Any]) -> str:
    audios = [ref for ref in project.get("references", []) if ref.get("type") == "audio"]
    if not audios:
        return ""
    roles = [
        ref.get("role") if ref.get("role") in REFERENCE_ROLES["audio"] else "none"
        for ref in audios
    ]
    mapping = ", ".join(f"<Audio {index}>={role}" for index, role in enumerate(roles, 1))
    modules = [SYSTEM_PROMPT_CONFIG["audio_reference_common"], f"\nAUDIO PRESET MAP: {mapping}."]
    for role in REFERENCE_ROLES["audio"]:
        if role in roles:
            modules.append(SYSTEM_PROMPT_CONFIG["audio_reference_roles"][role])
    return "".join(modules)


def _reference_system_modules(project: dict[str, Any]) -> str:
    return _video_reference_system_modules(project) + _audio_reference_system_modules(project)


def analyze_reference_video(video: dict[str, Any], role: str, duration: float,
                            image_model_id: str = DEFAULT_IMAGE_MODEL_ID,
                            session: _LlamaServerSession | None = None, progress=None,
                            start_time: float = 0.0) -> dict[str, str]:
    video_path = _resolve_uploaded_video(video)
    actual_duration = _probe_video_duration(video_path)
    start_time = max(0.0, float(start_time))
    available_duration = max(0.0, actual_duration - start_time) if actual_duration else float(duration)
    analysis_duration = min(float(duration), available_duration)
    if analysis_duration <= 0:
        raise ValueError("Set a positive video duration before analysis.")
    if progress:
        progress(
            stage="reference_analysis",
            message=(f"Sampling source interval {start_time:.2f}-{start_time + analysis_duration:.2f}s of reference video "
                     f"for role '{role}': {os.path.basename(video_path)}"),
        )
    with tempfile.TemporaryDirectory(prefix="toyxyz-h3-video-") as frame_dir:
        frame_paths, timestamps = _extract_video_analysis_frames(
            video_path, analysis_duration, frame_dir, start_time=start_time,
        )
        captions = [f"Frame {index + 1} at {timestamp:.3f} seconds." for index, timestamp in enumerate(timestamps)]
        prompt = _video_analysis_prompt(role, analysis_duration, timestamps, start_time=start_time)
        if progress:
            progress(
                stage="reference_analysis",
                message=f"Analyzing {len(frame_paths)} ordered frames from {os.path.basename(video_path)}.",
            )
        if session is not None:
            output = session.analyze_images(frame_paths, captions, prompt)
            model_path, mmproj_path = session.model_path, session.mmproj_path
        else:
            model_path, mmproj_path = _resolve_image_model(image_model_id or DEFAULT_IMAGE_MODEL_ID, progress)
            command = [_find_llama_cli(), "-m", model_path, "--mmproj", mmproj_path]
            for frame_path in frame_paths:
                command.extend(("--image", frame_path))
            command.extend((
                "-p", prompt, "--single-turn", "--no-display-prompt", "--no-show-timings", "--simple-io",
                "--no-context-shift", "--log-disable", "--color", "off", "-c", "16384", "-n", "1600",
                "-ngl", "all", "--temp", "0.2", "--top-p", "0.9", "--top-k", "40", "--jinja",
            ))
            if image_model_id == QWEN_IMAGE_MODEL_ID:
                command.extend(("--chat-template-kwargs", '{"enable_thinking":false}', "--reasoning", "off"))
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            completed = subprocess.run(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", timeout=1800,
                creationflags=creationflags, check=False,
            )
            if completed.returncode != 0:
                tail = "\n".join(completed.stderr.splitlines()[-12:])
                raise RuntimeError(f"Video analysis failed with code {completed.returncode}.\n{tail}")
            output = completed.stdout
    analysis = _clean_video_analysis(output)
    if not analysis:
        raise RuntimeError("The vision model returned an empty video analysis.")
    return {
        "analysis": analysis, "model_path": model_path, "mmproj_path": mmproj_path,
        "analyzed_duration": f"{analysis_duration:.3f}",
        "analyzed_start": f"{start_time:.3f}",
        "frame_count": str(len(frame_paths)),
    }


@contextmanager
def _vision_compatible_image(image_path: str):
    """Provide a decoder-safe image path for llama.cpp vision backends.

    The bundled llama.cpp build can report a WebP as loaded while producing
    unrelated visual embeddings. Converting only WebP inputs to PNG avoids
    that silent failure without modifying the user's uploaded file.
    """
    if os.path.splitext(image_path)[1].lower() != ".webp":
        yield image_path
        return
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required to prepare WebP reference images for vision analysis.") from exc
    file_descriptor, converted_path = tempfile.mkstemp(prefix="toyxyz-h3-vision-", suffix=".png")
    os.close(file_descriptor)
    try:
        with Image.open(image_path) as source:
            normalized = ImageOps.exif_transpose(source)
            if normalized.mode not in {"RGB", "RGBA"}:
                normalized = normalized.convert("RGB")
            normalized.save(converted_path, format="PNG")
        yield converted_path
    finally:
        try:
            os.unlink(converted_path)
        except FileNotFoundError:
            pass


def _clean_reference_analysis(text: str) -> str:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).replace("\r\n", "\n")
    marked = re.search(r"<REFERENCE_ANALYSIS>\s*(.*?)\s*</REFERENCE_ANALYSIS>", text, re.DOTALL | re.IGNORECASE)
    if marked:
        text = marked.group(1)
    elif re.search(r"<REFERENCE_ANALYSIS>\s*", text, re.IGNORECASE):
        # Some vision models emit the opening transport marker but consume the
        # generation budget before the closing marker. Everything before the
        # opening marker is llama-cli startup output and echoed input.
        text = re.split(r"<REFERENCE_ANALYSIS>\s*", text, maxsplit=1, flags=re.IGNORECASE)[1]
    elif "\n> " in text:
        text = text.rsplit("\n> ", 1)[1].partition("\n")[2]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"(?:^|\n)Exiting\.\.\.\s*$", "", text, flags=re.IGNORECASE).strip()
    fence = re.fullmatch(r"```(?:text)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return (fence.group(1) if fence else text).strip()


def _reference_analysis_prompt(role: str) -> str:
    role = role if role in REFERENCE_ROLES["picture"] else "subject_identity"
    role_focus = {
        "first_frame": "Treat it as an opening-frame anchor. Prioritize the exact style, composition, pose, support, contact, scene layout, and action-relevant objects that must continue forward.",
        "last_frame": "Treat it as a final-frame anchor. Prioritize the exact style, pose, object state, support, contact, composition, viewpoint, and lighting on which motion must land.",
        "frame": "Treat it as an exact intermediate-frame anchor. Prioritize the complete scene state, composition, subjects, pose, objects, contacts, viewpoint, lighting, and continuity that must occur at its assigned output frame.",
        "subject_identity": "Prioritize stable identity features, hair, face, body silhouette, clothing, accessories, colors, distinctive objects, and the source medium or rendering style required by the assigned strength.",
    }[role]
    return f"""Analyze the supplied image as visual reference metadata for a MiniMax H3 video prompt.
{role_focus}
Return compact structured evidence using exactly the eight labeled lines below. Write only directly observable facts. Use "none visible" when a category has no evidence; never omit a line.

VISUAL_MEDIUM: Classify the image as specifically as visible evidence permits, such as live-action photograph, 2D anime illustration, cel-shaded animation frame, hand-drawn illustration, 3D CGI render, physical collectible figurine photograph, or three-dimensional anime-figurine render. Include only brief supporting cues. If uncertain, write "indeterminate image" and list the visible rendering cues; never guess a production method.
COMPOSITION: State shot size, viewpoint, crop boundaries, visible body range, and important frame positions.
SUBJECTS: State each important subject's stable observable identity features, hairstyle, clothing, accessories, expression, and position. Keep style separate from identity facts.
POSE_SUPPORT_CONTACT: State pose, seated or standing support, visible furniture or ground contact, hand placement, held objects, and occlusion.
ACTION_RELEVANT_OBJECTS: State visible fasteners, seams, layers, openings, containers, surfaces, foreground obstacles, tools, props, and their exact spatial relationships when they could constrain a later action.
ENVIRONMENT: State the evidenced location type, background layout, furniture, and important scene objects. Never replace a specific environment with a generic room, office, studio, or gradient.
LIGHTING_MATERIALS: State observable light direction and quality, colors, materials, reflections, depth, and shadows.
VISIBLE_TEXT: Quote only clearly readable text exactly; otherwise write "none visible".

An object or body part counts as visible only when its pixels are discernible inside the frame. Never infer an off-frame floor object, hidden pocket content, unseen hand-held item, cropped body part, nearby prop, or probable continuation beyond an image boundary. Never infer or label nationality, ethnicity, race, age, celebrity identity, occupation, personality, attractiveness, backstory, motion, sound, intent, or future action. Do not use speculative alternatives joined by or. Choose only what is visibly supported or omit it. Do not use quality praise, bullets, prose before the labels, or extra headings. Enclose the eight lines exactly once in <REFERENCE_ANALYSIS> and </REFERENCE_ANALYSIS>."""


def _report_reference_analysis(image: dict[str, Any], role: str, image_path: str, progress=None) -> None:
    if not progress:
        return
    analysis_index = int(_number(image.get("_analysis_index"), 0))
    analysis_total = int(_number(image.get("_analysis_total"), 0))
    ordinal = f" {analysis_index}/{analysis_total}" if analysis_index and analysis_total else ""
    progress(
        stage="reference_analysis",
        message=f"Analyzing reference image{ordinal} for role '{role}': {os.path.basename(image_path)}",
    )


def _analyze_reference_image_with_server(image: dict[str, Any], role: str,
                                         session: _LlamaServerSession, progress=None) -> dict[str, str]:
    role = role if role in REFERENCE_ROLES["picture"] else "subject_identity"
    image_path = _resolve_uploaded_image(image)
    _report_reference_analysis(image, role, image_path, progress)
    with _vision_compatible_image(image_path) as vision_path:
        analysis = _clean_reference_analysis(session.analyze_image(vision_path, _reference_analysis_prompt(role)))
    if not analysis:
        raise RuntimeError("The vision model returned an empty image analysis.")
    return {"analysis": analysis, "model_path": session.model_path, "mmproj_path": session.mmproj_path}


def analyze_reference_image(image: dict[str, Any], role: str = "subject_identity",
                            image_model_id: str = DEFAULT_IMAGE_MODEL_ID, progress=None) -> dict[str, str]:
    role = role if role in REFERENCE_ROLES["picture"] else "subject_identity"
    image_path = _resolve_uploaded_image(image)
    model_path, mmproj_path = _resolve_image_model(image_model_id or DEFAULT_IMAGE_MODEL_ID, progress)
    _report_reference_analysis(image, role, image_path, progress)
    prompt = _reference_analysis_prompt(role)
    with _vision_compatible_image(image_path) as vision_path:
        command = [
            _find_llama_cli(), "-m", model_path, "--mmproj", mmproj_path, "--image", vision_path,
            "-p", prompt, "--single-turn", "--no-display-prompt", "--no-show-timings", "--simple-io",
            "--no-context-shift", "--log-disable", "--color", "off", "-c", "8192", "-n", "700",
            "-ngl", "all", "--temp", "0.2", "--top-p", "0.9", "--top-k", "40", "--jinja",
        ]
        if image_model_id == QWEN_IMAGE_MODEL_ID:
            command.extend([
                "--chat-template-kwargs", '{"enable_thinking":false}',
                "--reasoning", "off",
            ])
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        completed = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=900,
            creationflags=creationflags, check=False,
        )
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.splitlines()[-12:])
        raise RuntimeError(f"Image analysis failed with code {completed.returncode}.\n{tail}")
    analysis = _clean_reference_analysis(completed.stdout)
    if not analysis:
        raise RuntimeError("The vision model returned an empty image analysis.")
    return {"analysis": analysis, "model_path": model_path, "mmproj_path": mmproj_path}


def _prompt_shot_numbers(prompt: str) -> list[int]:
    main_field = re.search(
        r"(?:integrated_multimodal_description|detailed_description)\s*:\s*",
        prompt,
        flags=re.IGNORECASE,
    )
    if not main_field:
        return []
    body = prompt[main_field.end():]
    body = re.split(r"\n\s*overall_soundscape\s*:", body, maxsplit=1, flags=re.IGNORECASE)[0]
    # A reference such as "Picture 1 comes from [Shot 1]" is not a shot
    # header. Count headers only when they start a body line, or when a later
    # inline header carries the guide-required timestamp.
    numbers: list[int] = []
    seen_spans: set[tuple[int, int]] = set()
    for match in _SHOT_HEADER_PATTERN.finditer(body):
        if match.span() in seen_spans:
            continue
        seen_spans.add(match.span())
        numbers.append(int(match.group(1) or match.group(2)))
    return numbers


def _build_lightx2v_original_prompt(project: dict[str, Any], effective_seconds: float) -> str:
    lines = [
        f"Create exactly {len(project['shots'])} shot(s) in the supplied order.",
        f"The target video ends at {effective_seconds:.2f} seconds.",
    ]
    elapsed = 0.0
    for index, shot in enumerate(project["shots"], 1):
        start = elapsed
        elapsed += float(shot["duration"])
        action = _clean_text(shot.get("visual_action")) or "Maintain a coherent visible state."
        if index == 1:
            lines.append(f"[Shot 1] 0.000-{elapsed:.3f} seconds: {action}")
        else:
            lines.append(f"[Shot {index}] begins at {start:.3f} seconds with a cut: {action}")
    if project.get("user_request"):
        lines.append(f"Overall request: {project['user_request']}")
    if project.get("constraints"):
        lines.append(f"Constraints: {project['constraints']}")
    if project.get("verbatim_content"):
        lines.append(f"Verbatim dialogue or visible text to preserve exactly: {project['verbatim_content']}")
    return "\n".join(lines)


def _lightx2v_reference_paths(project: dict[str, Any], mode: str) -> list[str]:
    pictures = [ref for ref in project["references"] if ref["type"] == "picture"]
    expected_roles = {
        "T2VA": (), "I2VA": ("first_frame",), "L2VA": ("last_frame",),
        "FL2VA": ("first_frame", "last_frame"),
    }[mode]
    paths: list[str] = []
    for role in expected_roles:
        reference = next((ref for ref in pictures if ref["role"] == role), None)
        if not reference or not reference.get("image_filename"):
            raise ValueError(f"LightX2V {mode} requires an uploaded {role.replace('_', ' ')} image.")
        paths.append(_resolve_uploaded_image({
            "filename": reference["image_filename"], "subfolder": reference["image_subfolder"],
        }))
    return paths


def _lightx2v_messages(prompt: str, task: str, resolution: str, duration: int,
                       image_paths: list[str]) -> list[dict[str, Any]]:
    with open(LIGHTX2V_SYSTEM_PROMPTS_PATH, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    content: list[dict[str, Any]] = []
    image_index = 0
    for item in config["task_messages"][task]:
        if item == "image":
            image_path = image_paths[image_index]
            image_index += 1
            mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
            with open(image_path, "rb") as image_handle:
                encoded = base64.b64encode(image_handle.read()).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            })
        else:
            content.append({"type": "text", "text": item})
    request = (
        f"task: {task}\nresolution: {resolution}\nduration: {duration}s\n"
        f"original_prompt: {prompt.strip()}"
    )
    content.append({"type": "text", "text": ("\n" if content else "") + request})
    timeline_editing_lock = _clean_text(config.get("timeline_editing_lock"))
    system_prompt = config["system"] + (f"\n\n{timeline_editing_lock}" if timeline_editing_lock else "")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _enhance_project_lightx2v(result: dict[str, Any], model_id: str, progress=None,
                              cancel_event: threading.Event | None = None,
                              job_id: str = "") -> dict[str, Any]:
    project = result["project"]
    mode = project["mode"]
    if mode == "REF2VA":
        raise ValueError(
            "lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B does not support R2V/REF2VA. "
            "Select Qwen3.8 + Vision F16 or use T2VA, I2VA, L2VA, or FL2VA."
        )
    if mode not in LIGHTX2V_SUPPORTED_MODES:
        raise ValueError(f"The LightX2V 8B rewriter does not support mode {mode}.")
    image_paths = _lightx2v_reference_paths(project, mode)
    if progress and image_paths:
        progress(
            stage="reference_analysis",
            message=(f"Passing {len(image_paths)} reference frame{'s' if len(image_paths) != 1 else ''} "
                     "directly to the LightX2V multimodal rewriter in role order."),
        )
    with _ENHANCE_LOCK:
        if cancel_event is not None and cancel_event.is_set():
            raise EnhancementCancelled("Prompt generation was stopped by the user.")
        model_path, mmproj_path = _resolve_lightx2v_model(progress)
        original_prompt = _build_lightx2v_original_prompt(project, result["effective_duration"])
        task = {"T2VA": "t2va", "I2VA": "i2va", "L2VA": "l2va", "FL2VA": "fl2va"}[mode]
        adapter_task = {"t2va": "t2av", "i2va": "i2av", "l2va": "l2av", "fl2va": "fl2av"}[task]
        duration = min(15, max(4, int(round(float(project["requested_duration"])))))
        resolution = "16:9" if mode == "T2VA" else "adaptive"
        messages = _lightx2v_messages(original_prompt, adapter_task, resolution, duration, image_paths)
        try:
            import comfy.model_management as model_management
            model_management.unload_all_models()
            model_management.soft_empty_cache(force=True)
        except (ImportError, AttributeError):
            pass
        if progress:
            progress(stage="generating", message="Loading LightX2V merged Qwen3-VL-8B GGUF Q8_0 + Vision F16.")
        session = _LlamaServerSession(
            _find_llama_server(), model_path, mmproj_path, LIGHTX2V_MODEL_ID,
            context_size=16384,
            extra_args=[
                "--chat-template-kwargs", '{"enable_thinking":false}',
                "--reasoning", "off", "--image-min-tokens", "1024",
            ],
        )
        try:
            if job_id:
                _set_enhance_stopper(job_id, session.close)
            session.start()
            if cancel_event is not None and cancel_event.is_set():
                raise EnhancementCancelled("Prompt generation was stopped by the user.")
            enhanced = _clean_llm_output(session.chat(messages, max_tokens=4096, temperature=0.0))
            if cancel_event is not None and cancel_event.is_set():
                raise EnhancementCancelled("Prompt generation was stopped by the user.")
        finally:
            session.close()
            if job_id:
                _set_enhance_stopper(job_id, None)
        if not enhanced:
            raise RuntimeError("The LightX2V rewriter returned an empty prompt.")
        if progress:
            progress(stage="complete", message="LightX2V prompt generation completed.")
        return {
            "enhanced_prompt": enhanced, "model": model_id, "model_path": model_path,
            "reference_analyses": [],
        }


def enhance_project(project_data: Any, model_id: str, image_model_id: str = DEFAULT_IMAGE_MODEL_ID,
                    progress=None, cancel_event: threading.Event | None = None,
                    job_id: str = "") -> dict[str, Any]:
    def check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise EnhancementCancelled("Prompt generation was stopped by the user.")

    check_cancelled()
    if progress:
        progress(stage="compiling", message="Compiling the current node inputs into a raw prompt.")
    result = compile_project(project_data, use_enhanced=False)
    if result["errors"]:
        raise ValueError("Fix project validation errors before generating the prompt.")
    project = result["project"]
    selected_model = model_id or DEFAULT_ENHANCE_MODEL_ID
    if selected_model == LIGHTX2V_MODEL_ID:
        return _enhance_project_lightx2v(
            result, selected_model, progress, cancel_event=cancel_event, job_id=job_id,
        )
    rich_enhance = project.get("enhance") is True
    reference_analyses: list[dict[str, str]] = []
    analysis_lines: list[str] = []
    picture_labels = {
        ref["id"]: ref["label"]
        for ref in _reference_labels(project["references"])
        if ref["type"] == "picture"
    }
    video_labels = {
        ref["id"]: ref["label"]
        for ref in _reference_labels(project["references"])
        if ref["type"] == "video"
    }
    # Uploaded visual references are intentionally re-analyzed on every enhancement.
    # A changed role or video-duration crop changes the evidence needed by the rewriter.
    pictures_to_analyze = [
        ref for ref in project["references"]
        if ref["type"] == "picture" and ref["image_filename"]
    ]
    videos_to_analyze = [
        ref for ref in project["references"]
        if ref["type"] == "video" and ref["video_filename"] and ref["duration"] > 0
    ]
    if pictures_to_analyze or videos_to_analyze:
        try:
            import comfy.model_management as model_management

            model_management.unload_all_models()
            model_management.soft_empty_cache(force=True)
        except (ImportError, AttributeError):
            pass
        with _ENHANCE_LOCK:
            total_assets = len(pictures_to_analyze) + len(videos_to_analyze)
            server_session: _LlamaServerSession | None = None
            try:
                if progress:
                    progress(
                        stage="image_model_check",
                        message="Starting the persistent visual-analysis server and loading the model once.",
                    )
                try:
                    server_session = _start_persistent_image_server(image_model_id, progress)
                    if job_id:
                        _set_enhance_stopper(job_id, server_session.close)
                    check_cancelled()
                    if progress:
                        progress(
                            stage="reference_analysis",
                            message=(
                                f"Persistent visual-analysis server is ready; analyzing {total_assets} "
                                f"reference asset{'s' if total_assets != 1 else ''} without reloading the model."
                            ),
                        )
                except Exception as exc:
                    if progress:
                        progress(
                            stage="reference_analysis",
                            message=f"Persistent server unavailable; using llama-cli fallback: {exc}",
                        )

                def run_analysis_batch(session: _LlamaServerSession | None) -> None:
                    for analysis_index, ref in enumerate(pictures_to_analyze, 1):
                        check_cancelled()
                        image_payload = {
                            "filename": ref["image_filename"],
                            "subfolder": ref["image_subfolder"],
                            "type": "input",
                            "_analysis_index": analysis_index,
                            "_analysis_total": total_assets,
                        }
                        if session is not None:
                            analyzed = _analyze_reference_image_with_server(
                                image_payload, ref["role"], session, progress,
                            )
                        else:
                            analyzed = analyze_reference_image(
                                image_payload, ref["role"], image_model_id, progress,
                            )
                        label = picture_labels[ref["id"]]
                        analysis = analyzed["analysis"]
                        analysis_lines.append(f"{label} [role={ref['role']}]: {analysis}")
                        reference_analyses.append({
                            "id": ref["id"], "label": label,
                            "role": ref["role"],
                            "filename": ref["image_filename"],
                            "analysis": analysis,
                        })
                    for video_index, ref in enumerate(videos_to_analyze, len(pictures_to_analyze) + 1):
                        check_cancelled()
                        source_start, selected_duration, target_start = _visible_video_selection(
                            ref,
                            align_frame_count(float(project.get("requested_duration") or ref["duration"])) / MODEL_FPS,
                        )
                        video_payload = {
                            "filename": ref["video_filename"],
                            "subfolder": ref["video_subfolder"],
                            "type": "input",
                            "_analysis_index": video_index,
                            "_analysis_total": total_assets,
                        }
                        analyzed = analyze_reference_video(
                            video_payload, ref["role"], selected_duration, image_model_id,
                            session=session, progress=progress, start_time=source_start,
                        )
                        label = video_labels[ref["id"]]
                        analysis = analyzed["analysis"]
                        analysis_lines.append(
                            f"{label} [role={ref['role']}, source_start_seconds={analyzed['analyzed_start']}, "
                            f"selected_duration_seconds={analyzed['analyzed_duration']}, "
                            f"target_start_seconds={target_start:.3f}]: {analysis}"
                        )
                        reference_analyses.append({
                            "id": ref["id"], "label": label, "type": "video",
                            "role": ref["role"], "filename": ref["video_filename"],
                            "analysis": analysis, "analyzed_duration": analyzed["analyzed_duration"],
                            "analyzed_start": analyzed["analyzed_start"],
                            "timeline_start": f"{target_start:.3f}",
                            "frame_count": analyzed["frame_count"],
                        })

                if server_session is not None:
                    try:
                        run_analysis_batch(server_session)
                    except EnhancementCancelled:
                        raise
                    except Exception as exc:
                        server_session.close()
                        server_session = None
                        analysis_lines.clear()
                        reference_analyses.clear()
                        if progress:
                            progress(
                                stage="reference_analysis",
                                message=f"Persistent image analysis failed; retrying with llama-cli: {exc}",
                            )
                        run_analysis_batch(None)
                else:
                    run_analysis_batch(None)
            finally:
                if server_session is not None:
                    server_session.close()
                    if progress:
                        progress(
                            stage="reference_analysis",
                            message="Persistent visual analysis completed; the vision model was released.",
                        )
                if job_id:
                    _set_enhance_stopper(job_id, None)

    # Raw Prompt remains the deterministic result of user-controlled fields.
    # Automatic image analyses are supplied only in the private LLM context.
    expected_shots = list(range(1, len(result["project"]["shots"]) + 1))
    shot_headers = ", ".join(f"[Shot {number}]" for number in expected_shots)
    mode = result["project"]["mode"]
    active_mode_prompts = ENHANCED_MODE_LLM_SYSTEM_PROMPTS if rich_enhance else MODE_LLM_SYSTEM_PROMPTS
    system_prompt = _mode_prompt_preamble(mode) + "\n\n" + active_mode_prompts[mode]
    reference_model = _reference_model(result["project"]) if mode == "REF2VA" else None
    system_prompt += (
        f"\n\nEXACT SHOTS: Use only {shot_headers}, once each in that order. "
        "Do not add, remove, split, merge, duplicate, or renumber shots or invent another cut."
    )
    if analysis_lines:
        if result["project"]["mode"] == "FL2VA":
            system_prompt += (
                "\n\nFL2VA IMAGE EVIDENCE: The analyses are authoritative endpoint evidence. "
                "Use them to plan interpolation, but do not copy their appearance inventories into the output."
            )
        elif result["project"]["mode"] == "I2VA":
            system_prompt += (
                "\n\nI2VA IMAGE EVIDENCE: The <Picture 1> analysis is the sole evidence for 0.00 seconds. "
                "Discard demographic guesses, hidden details, and speculation even if present in the analysis. "
                "The raw action starts after the anchor; never backfill it or an invented source into Picture 1."
            )
        elif result["project"]["mode"] == "REF2VA":
            system_prompt += (
                "\n\nREF2VA VISUAL EVIDENCE: Image analyses describe still sources; video analyses describe "
                "chronologically ordered samples from only the configured leading duration. "
                "Treat the supplied video timeline, subjects, actions, camera, cuts, environment, and final state as "
                "authoritative only where the evidence states them. Analyses describe source assets, not output labels. "
                "Follow the locked label plan below; do not promote a source-only Picture label into summary, "
                "retention_analysis, or detailed_description. Use only role-relevant facts and never transfer a "
                "source background into an incompatible target setting."
            )
        else:
            system_prompt += (
                "\n\nREFERENCE IMAGE EVIDENCE: Each analysis is authoritative observable evidence for its "
                "matching picture and role. Use relevant facts in context without contradiction or a detached appendix."
            )
    if reference_model:
        plan_lines = []
        for label, plan in reference_model["label_plan"].items():
            strength = f", input_definition_scope={plan['strength']}" if plan["kind"] == "Subject" else ""
            plan_lines.append(
                f"- {label}: kind={plan['kind']}, source={plan['source']}, role={plan['role']}"
                f"{strength}, retention={plan['marker']}; contract={plan['contract']}"
            )
        system_prompt += (
            "\n\nLOCKED REF2VA LABEL PLAN:\n" + "\n".join(plan_lines)
            + "\nDefine exactly these output labels in this order. Source labels that are not output labels may "
              "appear only as provenance inside subject_definitions. Mention every output label in summary and "
              "retention_analysis, and apply every visual output label in detailed_description."
        )
        system_prompt += _reference_system_modules(result["project"])
    system_prompt += (
        "\n\nOUTPUT: Return only the finished English H3 prompt, enclosed exactly once between "
        "<H3_PROMPT> and </H3_PROMPT>, with no commentary or Markdown fence."
    )
    system_prompt += "\n\n" + _single_pass_output_lock(
        mode,
        result["effective_duration"],
        len(expected_shots),
        expected_shots,
        reference_model,
        _input_content_locks(result["project"]),
    )
    evidence_by_label = {
        item["label"]: item["analysis"] for item in reference_analyses
    }
    user_prompt = build_video_prompt(
        result["project"], result["effective_duration"], evidence_by_label,
    )
    with _ENHANCE_LOCK, tempfile.TemporaryDirectory(prefix="toyxyz_h3_") as temp_dir:
        check_cancelled()
        if progress:
            progress(stage="model_check", message="Checking the selected GGUF model.")
        model_path = _resolve_enhance_model(model_id or DEFAULT_ENHANCE_MODEL_ID, progress)
        llama_cli = _find_llama_cli()
        system_file = os.path.join(temp_dir, "system.txt")
        user_file = os.path.join(temp_dir, "user.txt")
        with open(system_file, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(system_prompt)
        with open(user_file, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(user_prompt)
        try:
            import comfy.model_management as model_management

            model_management.unload_all_models()
            model_management.soft_empty_cache(force=True)
        except (ImportError, AttributeError):
            pass
        if progress:
            progress(
                stage="generating",
                message=("Loading Qwen3.8 and generating a richly enhanced prompt."
                         if rich_enhance else "Loading the model and generating the prompt."),
            )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        def run_generation(prompt_text: str, temperature: float = 0.22) -> str:
            max_new_tokens = (
                REF_ENHANCE_MAX_NEW_TOKENS if mode == "REF2VA"
                else RICH_ENHANCE_MAX_NEW_TOKENS if rich_enhance
                else BASE_ENHANCE_MAX_NEW_TOKENS
            )
            top_p = 0.93 if rich_enhance else 0.88
            top_k = 40 if rich_enhance else 20
            repeat_penalty = 1.03 if rich_enhance else 1.05
            command = [
                llama_cli, "-m", model_path, "-sysf", system_file, "-p", prompt_text,
                "--jinja", "--chat-template-kwargs", '{"enable_thinking":false}',
                "--single-turn", "--no-display-prompt", "--no-show-timings", "--simple-io", "--no-context-shift",
                "--log-disable", "--color", "off",
                "-c", str(ENHANCE_CONTEXT_SIZE), "-n", str(max_new_tokens),
                "-ngl", "all", "--temp", str(temperature), "--top-p", str(top_p), "--top-k", str(top_k),
                "--repeat-penalty", str(repeat_penalty),
            ]
            if cancel_event is None:
                try:
                    completed = subprocess.run(
                        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, encoding="utf-8", errors="replace", timeout=1800,
                        creationflags=creationflags, check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError("Prompt enhancement exceeded the 30-minute timeout.") from exc
            else:
                process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8", errors="replace", creationflags=creationflags,
                )
                if job_id:
                    _set_enhance_stopper(job_id, process.terminate)
                deadline = time.monotonic() + 1800
                try:
                    while True:
                        check_cancelled()
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            process.terminate()
                            raise RuntimeError("Prompt enhancement exceeded the 30-minute timeout.")
                        try:
                            stdout, stderr = process.communicate(timeout=min(0.25, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            continue
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                    if job_id:
                        _set_enhance_stopper(job_id, None)
                completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            if completed.returncode != 0:
                tail = "\n".join(completed.stderr.splitlines()[-12:])
                raise RuntimeError(f"llama-cli exited with code {completed.returncode}.\n{tail}")
            return _clean_llm_output(completed.stdout)

        # Enhancement is intentionally single-pass. The system prompt carries
        # the H3 structure and fidelity rules, while the generated text is
        # returned as-is after transport-noise removal. Do not normalize,
        # validate, sanitize, or send the output through a correction call.
        enhanced = run_generation(user_prompt, temperature=0.38 if rich_enhance else 0.22)
        if not enhanced:
            raise RuntimeError("The selected model returned an empty prompt.")
        if progress:
            progress(stage="complete", message="Prompt generation completed.")
        return {
            "enhanced_prompt": enhanced,
            "model": model_id,
            "model_path": model_path,
            "reference_analyses": reference_analyses,
        }


def compile_project(project_data: Any, use_enhanced: bool = True) -> dict[str, Any]:
    project, parse_warnings = normalize_project(project_data)
    effective_frames = align_frame_count(project["requested_duration"])
    effective_seconds = effective_frames / MODEL_FPS
    errors, warnings = validate_project(project, parse_warnings)
    report_lines = [*(f"ERROR: {item}" for item in errors), *(f"WARNING: {item}" for item in warnings)]
    if not report_lines:
        report_lines.append("OK: Project metadata passes Minimax-H3 prompt validation.")
    draft_video_prompt = build_video_prompt(project, effective_seconds)
    video_prompt = project["enhanced_prompt"] if use_enhanced and project["enhanced_prompt"] else draft_video_prompt
    return {
        "project": project,
        "draft_video_prompt": draft_video_prompt,
        "video_prompt": video_prompt,
        "enhanced_prompt": project["enhanced_prompt"],
        "llm_prompt": build_llm_prompt(project, draft_video_prompt),
        "validation_report": "\n".join(report_lines),
        "errors": errors,
        "warnings": warnings,
        "effective_frames": effective_frames,
        "effective_duration": round(effective_seconds, 6),
        "resolved_mode": project["mode"],
        "mode_selection": project["mode_selection"],
    }


def _blank_reference_image():
    import torch

    return torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")


def _load_reference_image_tensor(reference: dict[str, Any]):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    image_path = _resolve_uploaded_image({
        "filename": reference.get("image_filename"),
        "subfolder": reference.get("image_subfolder"),
    })
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy())[None,]


def _reference_image_outputs(project: dict[str, Any]) -> tuple[Any, ...]:
    pictures = [ref for ref in project.get("references", []) if ref.get("type") == "picture"]
    blank = _blank_reference_image()
    outputs: list[Any] = []
    for index in range(MAX_REF_IMAGES):
        if index >= len(pictures) or not pictures[index].get("image_filename"):
            outputs.append(blank.clone())
            continue
        try:
            outputs.append(_load_reference_image_tensor(pictures[index]))
        except (FileNotFoundError, OSError, ValueError):
            outputs.append(blank.clone())
    return tuple(outputs)


def _trim_audio_value(audio: Any, target_duration: float):
    if not isinstance(audio, dict):
        return audio
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if waveform is None or not sample_rate:
        return audio
    end_sample = min(
        waveform.shape[-1],
        max(1, int(round(float(target_duration) * int(sample_rate)))),
    )
    return {**audio, "waveform": waveform[..., :end_sample]}


def _visible_video_selection(reference: dict[str, Any], target_duration: float) -> tuple[float, float, float]:
    """Return source start, visible duration, and target start for the lane intersection."""
    clip_duration = max(0.0, float(reference.get("duration") or target_duration))
    timeline_start = float(reference.get("timeline_start") or 0.0)
    source_start = max(0.0, float(reference.get("trim_start") or 0.0))
    clipped_leading = max(0.0, -timeline_start)
    target_start = max(0.0, timeline_start)
    source_start += clipped_leading
    visible_duration = min(
        max(0.0, clip_duration - clipped_leading),
        max(0.0, target_duration - target_start),
    )
    return source_start, visible_duration, target_start


def _load_reference_video(reference: dict[str, Any], target_frame_count: int):
    from fractions import Fraction
    import torch
    from comfy_api.latest import InputImpl, Types

    video_path = _resolve_uploaded_video({
        "filename": reference.get("video_filename"),
        "subfolder": reference.get("video_subfolder"),
    })
    target_frame_count = max(1, int(target_frame_count))
    target_duration = target_frame_count / MODEL_FPS
    trim_start, visible_duration, _target_start = _visible_video_selection(reference, target_duration)
    selected_duration = min(target_duration, max(1.0 / MODEL_FPS, visible_duration))
    selected_frame_count = max(1, min(target_frame_count, int(round(selected_duration * MODEL_FPS))))
    video = InputImpl.VideoFromFile(video_path)
    trimmed = video.as_trimmed(trim_start, trim_start + selected_duration, strict_duration=False)
    if trimmed is None:
        raise ValueError("The reference video could not be trimmed to the target duration.")
    components = trimmed.get_components()
    source_count = int(components.images.shape[0])
    source_fps = float(components.frame_rate)
    if source_count <= 0 or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("The reference video contains no decodable frames or valid frame rate.")
    available_target_count = max(1, int(round(source_count * MODEL_FPS / source_fps)))
    # Source decoders commonly exclude the exact trim-end frame. When that
    # creates only a one-frame rounding deficit, preserve the requested 24fps
    # interval count and let the clamped index repeat the final decoded frame.
    # Larger deficits still mean the source is genuinely shorter and are not
    # padded.
    output_count = (
        selected_frame_count
        if selected_frame_count <= available_target_count + 1
        else available_target_count
    )
    source_indices = torch.round(
        torch.arange(output_count, dtype=torch.float64) * source_fps / MODEL_FPS
    ).to(dtype=torch.long).clamp_(0, source_count - 1)
    images = components.images.index_select(0, source_indices.to(components.images.device))
    output_duration = output_count / MODEL_FPS
    audio = _trim_audio_value(components.audio, output_duration)
    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=images,
            audio=audio,
            frame_rate=Fraction(MODEL_FPS),
        ),
        bit_depth=trimmed.get_bit_depth(),
    )


def _blank_reference_video():
    from fractions import Fraction
    from comfy_api.latest import InputImpl, Types

    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=_blank_reference_image(),
            audio=None,
            frame_rate=Fraction(MODEL_FPS),
        )
    )


def _blank_reference_audio():
    import torch

    return {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": 44100}


def _load_reference_audio(reference: dict[str, Any], target_duration: float):
    from comfy_extras.nodes_audio import load

    audio_path = _resolve_uploaded_audio({
        "filename": reference.get("audio_filename"),
        "subfolder": reference.get("audio_subfolder"),
    })
    waveform, sample_rate = load(audio_path)
    audio = _trim_audio_value({"waveform": waveform, "sample_rate": sample_rate}, target_duration)
    return {"waveform": audio["waveform"].unsqueeze(0), "sample_rate": sample_rate}


def _reference_media_outputs(project: dict[str, Any], target_frame_count: int) -> tuple[Any, ...]:
    pictures = [ref for ref in project.get("references", []) if ref.get("type") == "picture"]
    videos = [ref for ref in project.get("references", []) if ref.get("type") == "video"]
    audios = [ref for ref in project.get("references", []) if ref.get("type") == "audio"]
    blank = _blank_reference_image()
    target_duration = target_frame_count / MODEL_FPS
    outputs: list[Any] = []
    for reference in pictures[:MAX_REF_IMAGES]:
        if not reference.get("image_filename"):
            outputs.append(blank.clone())
        else:
            try:
                outputs.append(_load_reference_image_tensor(reference))
            except (FileNotFoundError, OSError, ValueError):
                outputs.append(blank.clone())
        if reference.get("role") == "frame":
            outputs.append(min(
                max(0, int(reference.get("frame_index", 0))),
                max(0, target_frame_count - 1),
            ))
    for reference in videos[:MAX_REF_VIDEOS]:
        if not reference.get("video_filename"):
            outputs.append(_blank_reference_video())
            continue
        try:
            outputs.append(_load_reference_video(reference, target_frame_count))
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            outputs.append(_blank_reference_video())
    for reference in audios[:MAX_REF_AUDIOS]:
        if not reference.get("audio_filename"):
            outputs.append(_blank_reference_audio())
            continue
        try:
            outputs.append(_load_reference_audio(reference, target_duration))
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            outputs.append(_blank_reference_audio())
    total_media_outputs = MAX_REF_IMAGES * 2 + MAX_REF_VIDEOS + MAX_REF_AUDIOS
    outputs.extend(blank.clone() for _ in range(total_media_outputs - len(outputs)))
    return tuple(outputs)


class _FlexibleOutputType(str):
    def __ne__(self, _value):
        return False


FLEXIBLE_MEDIA_TYPE = _FlexibleOutputType("*")


class MinimaxH3Prompter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_data": (
                    "STRING",
                    {"default": json.dumps(DEFAULT_PROJECT, ensure_ascii=False), "multiline": True},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT") + (FLEXIBLE_MEDIA_TYPE,) * (
        MAX_REF_IMAGES * 2 + MAX_REF_VIDEOS + MAX_REF_AUDIOS
    )
    RETURN_NAMES = (
        "generated_prompt",
        "length",
    ) + tuple(
        name
        for index in range(1, MAX_REF_IMAGES + 1)
        for name in (f"image_{index}", f"frame_{index}")
    ) + tuple(
        f"video_{index}" for index in range(1, MAX_REF_VIDEOS + 1)
    ) + tuple(f"audio_{index}" for index in range(1, MAX_REF_AUDIOS + 1))
    FUNCTION = "compile"
    CATEGORY = "ToyxyzTestNodes/Prompt"
    DESCRIPTION = "Director-style editor that directly compiles a production-ready MiniMax-H3 video prompt."

    def compile(self, project_data: str):
        result = compile_project(project_data)
        enhanced_prompt = result["enhanced_prompt"]
        auto_run = bool(result["project"].get("auto_run"))
        if auto_run:
            if result["errors"]:
                raise ValueError("Fix project validation errors before Auto Run can generate the prompt.")
            enhanced = enhance_project(
                project_data,
                result["project"].get("enhance_model") or DEFAULT_ENHANCE_MODEL_ID,
                result["project"].get("image_model") or DEFAULT_IMAGE_MODEL_ID,
            )
            enhanced_prompt = enhanced["enhanced_prompt"]
        outputs = (
            enhanced_prompt,
            result["effective_frames"],
            *_reference_media_outputs(result["project"], result["effective_frames"]),
        )
        if auto_run:
            return {
                "ui": {"auto_run_prompt": [enhanced_prompt]},
                "result": outputs,
            }
        return outputs


try:
    import asyncio
    from aiohttp import web
    from server import PromptServer

    if getattr(PromptServer, "instance", None) is not None:
        @PromptServer.instance.routes.post("/toyxyz/minimax_h3_prompter/upload-video")
        async def minimax_h3_prompter_upload_video(request):
            partial_path = ""
            try:
                if request.content_type == "application/octet-stream":
                    upload_id = request.query.get("upload_id", "")
                    if not re.fullmatch(r"[A-Za-z0-9]{12,80}", upload_id):
                        raise ValueError("Invalid video upload identifier.")
                    original_name = os.path.basename(request.query.get("filename", ""))
                    extension = os.path.splitext(original_name)[1].lower()
                    if extension not in VIDEO_EXTENSIONS:
                        raise ValueError("Supported video formats are MP4, WebM, MOV, MKV, AVI, and M4V.")
                    try:
                        chunk_index = int(request.query.get("chunk_index", "-1"))
                    except ValueError as exc:
                        raise ValueError("Invalid video chunk index.") from exc
                    if chunk_index < 0:
                        raise ValueError("Invalid video chunk index.")
                    safe_stem = re.sub(
                        r"[^\w.-]+", "_", os.path.splitext(original_name)[0], flags=re.UNICODE
                    ).strip("._")
                    safe_stem = safe_stem[:100] or "reference-video"
                    stored_name = f"{safe_stem}-{upload_id[:20]}{extension}"
                    import folder_paths

                    input_root = os.path.abspath(folder_paths.get_input_directory())
                    subfolder = "toyxyz_h3_references"
                    target_dir = os.path.abspath(os.path.join(input_root, subfolder))
                    if os.path.commonpath((input_root, target_dir)) != input_root:
                        raise ValueError("Invalid reference-video destination.")
                    os.makedirs(target_dir, exist_ok=True)
                    target_path = os.path.join(target_dir, stored_name)
                    partial_path = target_path + ".part"
                    if chunk_index == 0:
                        mode = "wb"
                    elif not os.path.isfile(partial_path):
                        raise ValueError("Video upload chunks arrived out of order.")
                    else:
                        mode = "ab"
                    existing_size = os.path.getsize(partial_path) if mode == "ab" else 0
                    received = 0
                    with open(partial_path, mode) as handle:
                        async for chunk in request.content.iter_chunked(1024 * 1024):
                            received += len(chunk)
                            if existing_size + received > VIDEO_UPLOAD_MAX_BYTES:
                                raise ValueError("Reference video exceeds the 2 GiB upload limit.")
                            handle.write(chunk)
                    if received == 0:
                        raise ValueError("An empty video upload chunk was received.")
                    if request.query.get("final") != "1":
                        return web.json_response({"status": "partial", "received": existing_size + received})
                    os.replace(partial_path, target_path)
                    partial_path = ""
                    duration = await asyncio.to_thread(_probe_video_duration, target_path)
                    if _find_ffprobe() and duration is None:
                        try:
                            os.unlink(target_path)
                        except FileNotFoundError:
                            pass
                        raise ValueError("The uploaded file is not a readable video.")
                    return web.json_response({
                        "status": "success", "name": stored_name, "subfolder": subfolder,
                        "type": "input", "duration": round(duration, 3) if duration else None,
                    })

                reader = await request.multipart()
                field = await reader.next()
                if field is None or field.name != "video" or not field.filename:
                    raise ValueError("A video file is required.")
                original_name = os.path.basename(field.filename)
                extension = os.path.splitext(original_name)[1].lower()
                if extension not in VIDEO_EXTENSIONS:
                    raise ValueError("Supported video formats are MP4, WebM, MOV, MKV, AVI, and M4V.")
                safe_stem = re.sub(r"[^\w.-]+", "_", os.path.splitext(original_name)[0], flags=re.UNICODE).strip("._")
                safe_stem = safe_stem[:100] or "reference-video"
                stored_name = f"{safe_stem}-{uuid.uuid4().hex[:10]}{extension}"
                import folder_paths

                input_root = os.path.abspath(folder_paths.get_input_directory())
                subfolder = "toyxyz_h3_references"
                target_dir = os.path.abspath(os.path.join(input_root, subfolder))
                if os.path.commonpath((input_root, target_dir)) != input_root:
                    raise ValueError("Invalid reference-video destination.")
                os.makedirs(target_dir, exist_ok=True)
                target_path = os.path.join(target_dir, stored_name)
                partial_path = target_path + ".part"
                total = 0
                with open(partial_path, "wb") as handle:
                    while True:
                        chunk = await field.read_chunk(size=1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > VIDEO_UPLOAD_MAX_BYTES:
                            raise ValueError("Reference video exceeds the 2 GiB upload limit.")
                        handle.write(chunk)
                if total == 0:
                    raise ValueError("The uploaded video is empty.")
                os.replace(partial_path, target_path)
                partial_path = ""
                duration = await asyncio.to_thread(_probe_video_duration, target_path)
                if _find_ffprobe() and duration is None:
                    try:
                        os.unlink(target_path)
                    except FileNotFoundError:
                        pass
                    raise ValueError("The uploaded file is not a readable video.")
                return web.json_response({
                    "status": "success", "name": stored_name, "subfolder": subfolder,
                    "type": "input", "duration": round(duration, 3) if duration else None,
                })
            except ValueError as exc:
                if partial_path:
                    try:
                        os.unlink(partial_path)
                    except FileNotFoundError:
                        pass
                return web.json_response({"status": "error", "message": str(exc)}, status=400)
            except Exception as exc:
                if partial_path:
                    try:
                        os.unlink(partial_path)
                    except FileNotFoundError:
                        pass
                return web.json_response({"status": "error", "message": str(exc)}, status=500)

        @PromptServer.instance.routes.get("/toyxyz/minimax_h3_prompter/video")
        async def minimax_h3_prompter_video(request):
            try:
                video_path = _resolve_uploaded_video({
                    "filename": request.query.get("filename", ""),
                    "subfolder": request.query.get("subfolder", ""),
                })
                return web.FileResponse(video_path)
            except (ValueError, FileNotFoundError) as exc:
                return web.json_response({"status": "error", "message": str(exc)}, status=404)

        @PromptServer.instance.routes.post("/toyxyz/minimax_h3_prompter/compile")
        async def minimax_h3_prompter_compile(request):
            try:
                payload = await request.json()
                result = compile_project(payload.get("project_data", ""))
                return web.json_response({
                    "status": "success",
                    "raw_prompt": result["draft_video_prompt"],
                    "video_prompt": result["video_prompt"],
                    "enhanced_prompt": result["enhanced_prompt"],
                    "validation_report": result["validation_report"],
                    "errors": result["errors"],
                    "warnings": result["warnings"],
                    "effective_frames": result["effective_frames"],
                    "effective_duration": result["effective_duration"],
                    "resolved_mode": result["resolved_mode"],
                    "mode_selection": result["mode_selection"],
                    "project": result["project"],
                })
            except Exception as exc:
                return web.json_response({"status": "error", "message": str(exc)}, status=500)

        @PromptServer.instance.routes.get("/toyxyz/minimax_h3_prompter/models")
        async def minimax_h3_prompter_models(_request):
            try:
                return web.json_response({
                    "status": "success",
                    "default": DEFAULT_ENHANCE_MODEL_ID,
                    "models": list_enhance_models(),
                    "image_default": DEFAULT_IMAGE_MODEL_ID,
                    "image_models": list_image_models(),
                })
            except Exception as exc:
                return web.json_response({"status": "error", "message": str(exc)}, status=500)

        @PromptServer.instance.routes.get("/toyxyz/minimax_h3_prompter/enhance/status")
        async def minimax_h3_prompter_enhance_status(request):
            job_id = _clean_text(request.query.get("job_id"))
            job = _get_enhance_job(job_id)
            return web.json_response({"status": "success", "job": job})

        @PromptServer.instance.routes.post("/toyxyz/minimax_h3_prompter/enhance/cancel")
        async def minimax_h3_prompter_enhance_cancel(request):
            try:
                payload = await request.json()
                job_id = _clean_text(payload.get("job_id"))
                if not job_id:
                    return web.json_response(
                        {"status": "error", "message": "A prompt generation job id is required."},
                        status=400,
                    )
                found = await asyncio.to_thread(_cancel_enhance_job, job_id)
                return web.json_response({
                    "status": "success",
                    "cancelled": found,
                    "message": "Prompt generation stop requested." if found else "The job is no longer active.",
                })
            except Exception as exc:
                return web.json_response({"status": "error", "message": str(exc)}, status=500)

        @PromptServer.instance.routes.post("/toyxyz/minimax_h3_prompter/analyze-reference")
        async def minimax_h3_prompter_analyze_reference(request):
            try:
                payload = await request.json()
                job_id = _clean_text(payload.get("job_id"))
                _set_enhance_job(job_id, stage="image_model_check", message="Checking the selected image analysis model.")
                try:
                    import comfy.model_management as model_management

                    model_management.unload_all_models()
                    model_management.soft_empty_cache(force=True)
                except (ImportError, AttributeError):
                    pass
                result = await asyncio.to_thread(
                    analyze_reference_image,
                    payload.get("image") if isinstance(payload.get("image"), dict) else {},
                    _clean_text(payload.get("role")).lower() or "subject_identity",
                    _clean_text(payload.get("image_model")) or DEFAULT_IMAGE_MODEL_ID,
                    lambda **values: _set_enhance_job(job_id, **values),
                )
                return web.json_response({"status": "success", **result})
            except (ValueError, FileNotFoundError) as exc:
                _set_enhance_job(locals().get("job_id", ""), stage="error", message=str(exc))
                return web.json_response({"status": "error", "message": str(exc)}, status=400)
            except Exception as exc:
                _set_enhance_job(locals().get("job_id", ""), stage="error", message=str(exc))
                return web.json_response({"status": "error", "message": str(exc)}, status=500)

        @PromptServer.instance.routes.post("/toyxyz/minimax_h3_prompter/enhance")
        async def minimax_h3_prompter_enhance(request):
            try:
                payload = await request.json()
                job_id = _clean_text(payload.get("job_id"))
                cancel_event = _begin_enhance_job(job_id)
                _set_enhance_job(job_id, stage="queued", message="Prompt generation request queued.")

                def report_progress(**values):
                    if cancel_event.is_set():
                        raise EnhancementCancelled("Prompt generation was stopped by the user.")
                    _set_enhance_job(job_id, **values)

                result = await asyncio.to_thread(
                    enhance_project,
                    payload.get("project_data", ""),
                    _clean_text(payload.get("model")) or DEFAULT_ENHANCE_MODEL_ID,
                    _clean_text(payload.get("image_model")) or DEFAULT_IMAGE_MODEL_ID,
                    report_progress,
                    cancel_event,
                    job_id,
                )
                return web.json_response({"status": "success", **result})
            except EnhancementCancelled as exc:
                _set_enhance_job(locals().get("job_id", ""), stage="cancelled", message=str(exc))
                return web.json_response({"status": "cancelled", "message": str(exc)}, status=409)
            except ValueError as exc:
                _set_enhance_job(locals().get("job_id", ""), stage="error", message=str(exc))
                return web.json_response({"status": "error", "message": str(exc)}, status=400)
            except Exception as exc:
                _set_enhance_job(locals().get("job_id", ""), stage="error", message=str(exc))
                return web.json_response({"status": "error", "message": str(exc)}, status=500)
            finally:
                _finish_enhance_job(locals().get("job_id", ""))
except ImportError:
    # Allows the pure compiler to be imported by lightweight tests outside ComfyUI.
    pass
