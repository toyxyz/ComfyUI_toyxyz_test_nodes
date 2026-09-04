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
import tarfile
import urllib.error
import urllib.request
import uuid
import zipfile
from contextlib import contextmanager
from typing import Any


MODEL_FPS = 24
MIN_TIMELINE_ITEM_FRAMES = 2
MIN_SHOT_DURATION = MIN_TIMELINE_ITEM_FRAMES / MODEL_FPS
CURRENT_PROJECT_VERSION = 28
SUPPORTED_MODES = ("AUTO", "T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA")
SUPPORTED_DIALOGUE_MODES = ("spoken", "voiceover", "singing")
SUPPORTED_TRANSITIONS = ("cut", "cross-dissolve", "fade", "wipe")
SUPPORTED_LANGUAGES = (
    "Arabic", "Chinese", "English", "French", "German", "Italian",
    "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
)
REFERENCE_ROLES = {
    "picture": ("first_frame", "last_frame", "frame", "storyboard", "subject_identity"),
    "video": (
        "none", "video_editing", "video_continuation", "subject_visual", "visual_style",
        "motion", "motion_camera", "camera", "cuts_rhythm",
    ),
    "audio": (
        "none", "full_signal_copy", "partial_signal_copy", "voice_delivery",
        "dialogue_lyrics", "sound_ambience", "music_rhythm",
    ),
}
SUBJECT_STRENGTHS = ("weak", "normal", "strong", "attribute_transfer", "style_transfer")
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
REMOVED_LIGHTX2V_MODEL_IDS = {
    "hf:lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B",
    "hf:indhic-ai/MiniMax_H3-Prompt_Rewriter-8B-LORA-Merged-GGUF/Q8_0+vision-f16",
}
OMNI_MODEL_ID = "hf:pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-Omni-GGUF/Q8_0+Qwen2.5-Omni-7B-Q4_K_M"
OMNI_ADAPTER_REPO = "pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-Omni-GGUF"
OMNI_ADAPTER_FILE = "MiniMax-H3-Prompt-Rewriter-LoRA-Omni-Q8_0.gguf"
OMNI_BASE_REPO = "ggml-org/Qwen2.5-Omni-7B-GGUF"
OMNI_BASE_FILE = "Qwen2.5-Omni-7B-Q4_K_M.gguf"
OMNI_MMPROJ_FILE = "mmproj-Qwen2.5-Omni-7B-Q8_0.gguf"
OMNI_ADAPTER_SIZE = 322961408
OMNI_BASE_SIZE = 4680000000
OMNI_MMPROJ_SIZE = 1550000000
OMNI_TOTAL_SIZE = OMNI_ADAPTER_SIZE + OMNI_BASE_SIZE + OMNI_MMPROJ_SIZE
OMNI_MODEL_DISPLAY_NAME = "pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-Omni-GGUF · Q8_0 + Qwen2.5-Omni-7B Q4_K_M"
OMNI_MODEL_VRAM_LABEL = "VRAM ≈ 9 GB"
OMNI_SYSTEM_PROMPTS_PATH = os.path.join(
    os.path.dirname(__file__), "minimax_h3_omni_system_prompts.json"
)
BASE_ENHANCE_MAX_NEW_TOKENS = 1800
RICH_ENHANCE_MAX_NEW_TOKENS = 3072
STRONG_ENHANCE_MAX_NEW_TOKENS = 4096
REF_ENHANCE_MAX_NEW_TOKENS = 3072
ENHANCE_CONTEXT_SIZE = 16384
_ENHANCE_LOCK = threading.Lock()
LLAMA_RUNTIME_RELEASE = "b10310"
LLAMA_RUNTIME_REPO = "ggml-org/llama.cpp"
LLAMA_RUNTIME_URL = (
    f"https://github.com/{LLAMA_RUNTIME_REPO}/releases/download/{LLAMA_RUNTIME_RELEASE}"
)
_LLAMA_RUNTIME_LOCK = threading.Lock()
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
            "kind": "shot",
            "duration": 5.0,
            "visual_action": "",
            "presets": {
                "camera_angle": "none", "camera_motion": "none", "camera_shot": "none",
                "camera_amplitude": "none", "camera_speed": "none", "style": "none",
            },
        }
    ],
    "references": [],
    "constraints": "",
    "verbatim_content": "",
    "enhance_model": DEFAULT_ENHANCE_MODEL_ID,
    "image_model": DEFAULT_IMAGE_MODEL_ID,
    "auto_run": False,
    "enhance": False,
    "enhance_level": "none",
    "enhanced_prompt": "",
}


CAMERA_PRESET_PROMPTS = {
    "camera_angle": {
        "none": "", "eye_level": "eye-level angle", "low_angle": "low angle looking upward",
        "high_angle": "high angle looking downward", "overhead": "overhead angle",
        "top_down": "top-down angle looking straight downward",
        "birds_eye": "extreme bird's-eye view", "worms_eye": "extreme worm's-eye view",
        "ground_level": "ground-level angle", "aerial": "high aerial angle over the scene",
        "dutch_angle": "Dutch angle with a tilted horizon",
        "over_shoulder": "over-the-shoulder angle", "pov": "subjective point-of-view angle",
        "three_quarter": "three-quarter angle", "profile": "profile angle from the side",
        "rear": "rear angle viewing the subject from behind",
    },
    "camera_motion": {
        "none": "", "static": "static shot with camera position and lens remaining still",
        "zoom_in": "zoom in by changing focal length while the camera remains stationary",
        "zoom_out": "zoom out by changing focal length while the camera remains stationary",
        "push_in": "camera push in by moving forward", "pull_out": "camera pull out by moving backward",
        "pan_left": "camera pan left from a fixed position", "pan_right": "camera pan right from a fixed position",
        "truck_left": "camera truck left with horizontal translation",
        "truck_right": "camera truck right with horizontal translation",
        "tilt_up": "camera tilt up from a fixed position", "tilt_down": "camera tilt down from a fixed position",
        "pedestal_up": "camera pedestal up with the entire camera moving upward",
        "pedestal_down": "camera pedestal down with the entire camera moving downward",
        # Keep legacy keys loadable, but describe them with the less ambiguous
        # H3 vocabulary used by the current UI.
        "dolly_left": "camera truck left with smooth horizontal translation",
        "dolly_right": "camera truck right with smooth horizontal translation",
        "dolly_zoom_in": (
            "experimental dolly zoom: physically move the camera forward while zooming out at a matched rate, "
            "keeping the subject exactly the same size in frame while the background perspective visibly expands"
        ),
        "dolly_zoom_out": (
            "experimental dolly zoom: physically move the camera backward while zooming in at a matched rate, "
            "keeping the subject exactly the same size in frame while the background perspective visibly compresses"
        ),
        "crane_up": "large-scale crane movement lifting the camera upward",
        "crane_down": "large-scale crane movement lowering the camera downward",
        "orbit_left": "orbit smoothly to the camera's left around the subject while keeping the subject centered and revealing changing background parallax",
        "orbit_right": "orbit smoothly to the camera's right around the subject while keeping the subject centered and revealing changing background parallax",
        "arc": "arc smoothly around the subject along one continuous curved camera path",
        "tracking": "track the moving subject while maintaining a stable relative framing",
        "follow": "track the moving subject continuously while maintaining a stable relative framing",
        "handheld": "natural handheld camera movement with organic operator motion",
        "shake_slightly": "slight camera shake", "shake_strongly": "strong camera shake",
        "pov": "POV camera movement from the subject's point of view",
        "roll_clockwise": "camera roll clockwise around the lens axis",
        "roll_counterclockwise": "camera roll counterclockwise around the lens axis",
    },
    "camera_shot": {
        "none": "",
        "extreme_close_up": "extreme close-up isolating a small detail such as the eyes or mouth",
        "close_up": "close-up centered on the face",
        "medium_close_up": "medium close-up framing the subject from the chest or shoulders upward",
        "medium_shot": "medium shot framing the subject from the waist upward",
        "medium_wide_shot": "medium wide shot framing the subject from the thighs or knees upward with some environmental context",
        "cowboy_shot": "medium-long framing from mid-thigh to slightly above the head, with both hands and the waist fully visible",
        "medium_full_shot": "medium-long framing from around the knees to slightly above the head",
        "full_shot": "full shot keeping the subject's entire body visible",
        "wide_shot": "wide shot showing the full subject with substantial surrounding environment",
        "extreme_wide_shot": "extreme wide shot dominated by the environment with the subject appearing very small",
        "establishing_shot": "establishing shot introducing the full location and spatial layout",
        "insert_shot": "insert shot isolating a specific object such as a phone screen, gun, or key",
        "detail_shot": "detail shot emphasizing a fine object or body detail",
        "two_shot": "two shot composing exactly two people together in one frame",
        "three_shot": "three shot composing exactly three people together in one frame",
        "group_shot": "group shot composing several people together in one frame",
    },
    "camera_amplitude": {
        "none": "", "small": "with small amplitude", "large": "with large amplitude",
    },
    "camera_speed": {
        "none": "", "slow": "at slow speed", "fast": "at fast speed",
    },
}

STYLE_PRESET_PROMPTS = {
    "none": "",
    "animation_2d": "polished hand-drawn 2D animation with coherent linework, layered flat color, readable silhouettes, expressive character acting, and continuously interpolated configured camera travel",
    "animation_3d": "polished feature-quality 3D animation with appealing sculpted geometry, consistent PBR materials, global illumination, expressive facial acting, natural articulated motion, and physically coherent secondary motion",
    "rough_hand_drawn_2d": "loose hand-drawn 2D animation with visible pencil strokes, rough construction lines, uneven organic contours, expressive smears, hand-drawn in-betweens, and energetic frame-by-frame character movement",
    "watercolor_2d": "hand-painted watercolor 2D animation with soft bleeding pigments, textured watercolor paper, delicate ink outlines, translucent color layering, painterly backgrounds, and gentle frame-by-frame character motion",
    "ink_wash_2d": "traditional ink-wash 2D animation with expressive black brush strokes, diluted grey ink gradients, handmade rice-paper texture, minimal color accents, and flowing brush-like character motion",
    "modern_flat_cartoon": "modern flat 2D cartoon animation with bold clean outlines, simplified geometric character shapes, flat colors, minimal shading, highly readable facial expressions, and snappy pose-to-pose character animation",
    "vintage_western_cartoon": "vintage hand-painted Western 2D cartoon animation with inked outlines, painted cel colors, watercolor backgrounds, slightly imperfect registration, subtle film grain, and lively frame-by-frame character acting",
    "comic_book_2d": "2D comic-book animation with bold black ink outlines, halftone-dot shading, flat spot colors, dramatic panel composition, speed lines, impact frames, and animated graphic transitions",
    "manga_monochrome_2d": "black-and-white manga animation with crisp pen-and-ink linework, screentone shading, cross-hatching, pure black shadows, white negative space, speed lines, and restrained panel-like motion",
    "paper_cutout_2d": "handcrafted paper-cutout 2D animation with layered paper shapes, visible fibers and cut edges, flat articulated pieces, practical-looking shadows, and deliberately stepped frame-by-frame motion",
    "anime_1980s_ova": "late-1980s Japanese OVA animation with detailed hand-drawn linework, dramatic painted shadows, muted analog colors, hand-painted backgrounds, subtle film grain, elaborate mechanical detail, and cinematic cel character animation",
    "anime_early_2000s_tv": "early-2000s Japanese TV anime with clean digital line art, simple cel shading, bright flat colors, restrained gradients, painted 2D backgrounds, and economical television character-animation timing",
    "theatrical_anime_2d": "high-budget theatrical Japanese 2D animation with finely drawn characters, detailed hand-painted environments, sophisticated cel shading, atmospheric painted depth, expressive facial acting, and fluid character motion",
    "stylized_feature_3d": "stylized feature-quality 3D animation with appealing sculpted character geometry, soft PBR materials, global illumination, cinematic rim lighting, physically believable cloth and hair response, and smooth expressive character animation",
    "photorealistic_3d_cg": "photorealistic cinematic 3D CG with physically based materials, detailed surface imperfections, global illumination, accurate reflections, volumetric atmospheric lighting, natural depth cues, and physically believable animation",
    "semi_realistic_3d": "semi-realistic stylized 3D character animation with anatomically grounded sculpted forms, softened proportions, detailed PBR materials, natural skin response, cinematic lighting, and expressive but physically coherent motion",
    "cel_shaded_3d": "anime-inspired cel-shaded 3D animation with clean toon outlines, hard-edged two-tone shading, simplified PBR materials, controlled specular highlights, expressive anime facial acting, and smooth dynamic character motion",
    "game_cinematic_3d": "high-end real-time 3D game cinematic with detailed character models, physically based materials, cinematic volumetric lighting, polished environment rendering, realistic simulation, and weighty motion-captured character animation",
    "low_poly_3d": "stylized low-poly 3D animation with faceted geometry, simplified silhouettes, restrained polygon detail, clean color blocking, lightweight materials, graphic lighting, and readable character motion",
    "ps1_retro_3d": "PlayStation-era retro 3D animation with low-poly geometry, affine texture warping, low-resolution hand-painted textures, vertex lighting, limited draw distance, subtle pixel jitter, and period-authentic game animation",
    "ps2_retro_3d": "early-2000s console-style 3D animation with moderately low-poly models, baked lighting, compressed textures, simple specular materials, restrained effects, and period-authentic real-time character motion",
    "voxel_3d": "voxel-based 3D animation with block-built geometry, crisp cubic silhouettes, grid-aligned materials, simple directional lighting, readable volumetric environments, and clean stepped character motion",
    "product_visualization_3d": "premium CGI product visualization with precise modeled geometry, physically based materials, controlled studio lighting, accurate reflections, macro surface detail, clean presentation, and smooth restrained object animation",
    "architectural_visualization_3d": "photorealistic architectural 3D visualization with accurate spatial scale, physically based building materials, global illumination, natural daylight, realistic reflections, atmospheric depth, and coherent environmental motion",
    "fantasy_stylized_3d": "stylized fantasy 3D animation with sculpted organic forms, richly detailed costumes and environments, tactile PBR materials, luminous atmospheric lighting, restrained magical effects, and expressive feature-quality motion",
    "chibi_3d": "cute chibi 3D animation with compact proportions, oversized expressive eyes, rounded sculpted forms, soft clean materials, bright gentle lighting, highly readable poses, and playful character motion",
    "dark_fantasy_cgi_3d": "dark fantasy cinematic 3D CG with weathered sculpted forms, detailed PBR materials, moody volumetric lighting, dense atmospheric effects, grounded physical simulation, and weighty character animation",
    "scifi_cgi_3d": "cinematic science-fiction 3D CG with precise hard-surface modeling, advanced PBR materials, emissive interface accents, volumetric lighting, coherent reflections, detailed environments, and physically believable animation",
    "figurine_animation": "a crafted figurine character coming fully alive with fluid expressive animation, natural body mechanics, responsive facial acting, and material-aware secondary motion while preserving its recognizable sculpted identity and surface appearance",
    "cinematic_live_action": "cinematic live-action film with realistic skin and materials, natural physical motion, controlled depth of field, practical lighting, and restrained filmic contrast",
    "smartphone_video": "natural smartphone-recorded video with realistic mobile exposure, compact-sensor detail, casual composition, and authentic available light",
    "photoreal_live_action": "photorealistic live-action footage with natural skin texture, physically plausible motion, coherent materials, realistic lighting, and grounded production detail",
    "documentary": "observational documentary footage with available-light realism, natural color response, unembellished environments, and authentic human behavior",
    "stop_motion": "stop-motion animation with handcrafted materials, intentional frame-by-frame movement, tactile surfaces, and consistent miniature-scale lighting",
    "anime_1990s": "authentic 1990s Japanese hand-drawn anime with traditional 2D cel animation, painted background art, visible ink linework, two-tone cel shading, restrained held-frame timing for character acting, subtle analog film texture, continuously interpolated configured camera travel, and strictly no 3D, CGI, or game-engine rendering",
    "retro_anime_motion_graphics": "polished retro-anime motion graphics with a limited palette, clean manga linework, halftone shading, sequential graphic reveals, UI-style wipes, pixel accents, poster-like composition, and stable protected typography",
    "retro_anime_noir_jazz": "retro Japanese anime opening artwork with a graphic noir-jazz aesthetic, bold silhouettes, moody contrast, vintage analog texture, and poster-like visual sensibility",
    "contemporary_anime": "contemporary Japanese 2D anime with clean line art, expressive character acting, saturated color grading, strong readable key poses, crisp highlights, selective impact-frame emphasis, emotionally cinematic presentation, and polished anime-PV finish",
    "contemporary_action_anime": "contemporary Japanese 2D action anime with sharp clean line art, dynamic perspective drawing, exaggerated but readable key poses, speed-line accents, selective impact frames, hard-edged highlights, and clear high-energy action staging",
    "western_cartoon": "western 2D cartoon animation with simplified shapes, clean outlines, flat painted colors, expressive squash-and-stretch, readable silhouettes, and hand-drawn broadcast-cartoon timing",
    "vhs_analog": "early-1990s VHS live-action with a soft analog image, mild tape softness, nostalgic color response, consumer-camcorder character, subtle analog imperfections, and era-authentic lighting",
    "vhs_rental_movie": "1980s VHS rental-movie live action with low-resolution analog character, soft optical detail, era-authentic lighting and color response, restrained tape imperfections, and a worn home-video transfer mood",
    "cyberpunk_live_action": "live-action cyberpunk with selective neon illumination, reflective and weathered materials, industrial surface detail, atmospheric haze, a green-magenta palette, and grounded cinematic realism",
    "epic_dark_fantasy": "epic dark-fantasy trailer with mythic atmosphere, weathered fantasy materials, mystical haze, solemn cinematic lighting, dramatic scale, and grounded physical detail",
    "dark_medieval_fantasy": "grounded dark medieval fantasy film with ancient ruins, weathered armor and cloth, candlelight and firelight, dense atmospheric fog, realistic practical materials, solemn dramatic scale, and a restrained mythic mood",
    "high_saturation_commercial": "high-saturation commercial with photoreal subjects, a bold controlled palette, clean subject presentation, glossy textures, crisp visual hierarchy, and premium lighting",
    "photoreal_graphic_hybrid": "photorealistic characters against flat graphic-animation design with controlled color palettes, clean commercial composition, and a cohesive hybrid live-action and graphic treatment",
    "phone_ugc_ad": "phone-shot UGC advertisement with casual creator-led realism, natural available lighting, conversational performance, realistic micro-expressions, and short-form social-media character",
    "authentic_smartphone_vlog": "authentic smartphone vlog with handheld selfie-camera character, natural room or available lighting, casual creator performance, realistic micro-expressions, conversational pacing, and an unpolished short-form social-media aesthetic",
    "sprite_16bit": "16-bit retro 2D game-sprite animation with readable pixel silhouettes, low-frame game timing, short loop-friendly movement, and simple retro-console animation logic",
    "sketch_anime": "hand-drawn sketch animation with rough textured outlines, minimalist flat coloring, loose line movement, white highlight accents, and an intentionally unfinished rough-animation finish",
    "lineart_anime": "lineart anime with clean ink contours, contour-focused shading, minimal fill rendering, crisp outline priority, and strong graphic readability",
    "anamorphic_cinema": "anamorphic cinematic live-action with widescreen composition, shallow depth of field, subtle anamorphic lens characteristics, practical lighting, atmospheric highlights, and restrained film grading",
    "cinematic_35mm": "cinematic live-action photographed with a 35mm film aesthetic, natural skin texture, subtle organic film grain, soft highlight roll-off, shallow depth of field, realistic optical behavior, practical lighting, and restrained film color grading",
    "film_noir": "classic film noir with high-contrast black-and-white cinematography, hard directional lighting, deep shadows, venetian-blind patterns, smoky atmosphere, and dramatic silhouettes",
    "neo_noir": "modern neo-noir cinema with deep shadows, selective neon lighting, a dark desaturated palette, reflective surfaces, restrained acting, and moody cinematic composition",
    "horror_cinema": "cinematic horror with oppressive low-key lighting, practical light sources, deep shadows, unsettling negative space, realistic textures, and grounded atmospheric tension",
    "analog_horror_1990s": "1990s analog horror with consumer VHS recording character, dim practical lighting, empty institutional atmosphere, soft analog detail, subtle tape noise, restrained distortion, and unsettling grounded realism",
    "scifi_mystery": "cinematic science-fiction mystery teaser with cold futuristic lighting, restrained production design, atmospheric haze, enigmatic technological accents, and realistic cinematic materials",
    "retro_futuristic_scifi": "retro-futuristic science fiction with analog interface design, practical miniature-inspired forms, industrial surface language, tungsten instrument lights, and vintage cinematic production design",
    "premium_product_film": "premium product film with controlled studio lighting, clean reflective surfaces, macro-level material detail, precise highlights, uncluttered composition, and luxury commercial presentation",
    "japanese_commercial": "stylized Japanese product commercial with precise subject presentation, clean visual hierarchy, bold graphic accents, premium lighting, controlled color, and polished advertising finish",
    "food_commercial": "premium food commercial with appetizing macro detail, glossy food textures, vivid controlled colors, crisp ingredient visibility, precise highlights, and polished advertising lighting",
    "music_video": "stylized cinematic music video with expressive performance, bold lighting design, atmospheric color grading, graphic visual accents, and strong beat-responsive energy",
    "anime_music_video": "high-energy anime music video with contemporary Japanese anime rendering, strong character poses, expressive facial detail, dramatic lighting changes, impact-frame styling, and rhythmic visual energy",
    "graphic_poster_animation": "graphic poster animation with protected original layout and typography, bold poster-like composition, sequential line-art treatment, geometric graphic elements, restrained parallax, and subtle looping accents",
    "minimalist_motion_design": "minimalist 2D motion design with clean geometric forms, a limited color palette, precise easing, simple graphic transitions, strong visual hierarchy, and smooth restrained animation",
    "game_cinematic": "high-end game cinematic with realistic character rendering, dramatic environmental lighting, detailed production design, coherent materials, and polished real-time-render aesthetics",
    "dark_retro_fantasy": "1970s-to-1990s dark-fantasy live-action film with a practical-effects atmosphere, weathered material detail, soft analog imagery, muted vintage color response, and mysterious mythic tone",
    "modern_cinematic_live_action": "photorealistic live-action cinema with natural skin texture, realistic materials, physically believable movement, cinematic lighting, shallow depth of field, subtle filmic contrast, natural lens behavior, and restrained color grading",
    "prestige_drama": "prestige live-action drama with naturalistic performances, subtle facial expressions, realistic skin texture, restrained cinematic lighting, soft contrast, shallow depth of field, carefully composed frames, slow controlled camera movement, and grounded production design",
    "intimate_relationship_drama": "intimate live-action relationship drama with quiet restrained acting, subtle micro-expressions, natural breathing and posture shifts, warm practical interior lighting, shallow depth of field, a realistic lived-in environment, carefully matched eyelines, and understated cinematic camera work",
    "short_form_microdrama": "grounded live-action microdrama with tight emotional pacing, medium-close framing, natural performances, shallow depth of field, warm realistic interior lighting, frequent shot-reverse-shot editing, strong but believable emotion, and a realistic everyday environment",
    "golden_hour_road_movie": "cinematic road movie with warm golden-hour sunlight, natural backlighting, soft atmospheric haze, gentle lens flare, a nostalgic color palette, relaxed natural performances, subtle wind movement, smooth restrained camera motion, and photorealistic live action",
    "natural_light_indie_film": "naturalistic indie film with soft available light, muted organic colors, imperfect realistic skin, understated performances, gentle handheld camera, shallow depth of field, subtle environmental movement, and intimate observational framing",
    "mountain_adventure_cinema": "photorealistic outdoor adventure film with a vast natural landscape, natural mountain light, atmospheric depth, realistic wind, physically grounded performance, a stable horizon, detailed environmental parallax, and cinematic scale",
    "survival_expedition_film": "realistic expedition film with a harsh natural environment, weathered clothing and equipment, cold natural daylight, wind-driven atmosphere, documentary-influenced cinematic framing, physically believable movement, and restrained color grading",
    "blue_hour_urban_cinema": "photorealistic urban cinema at blue hour with rain-soaked streets, wet pavement reflections, glowing storefront lights, cool ambient sky light mixed with warm practical lights, shallow depth of field, realistic rain physics, and cinematic city atmosphere",
    "rainy_city_one_take": "continuous cinematic one-take in a photorealistic rain-soaked city with low-angle tracking, realistic wet reflections, physically believable body movement, a smooth camera orbit, slow cinematic push-in, shallow depth of field, and natural rain and cloth physics",
    "urban_editorial": "polished urban editorial film with modern city architecture, natural city light, wide-angle movement, dynamic tracking shots, clean hard cuts, restrained fashion poses, realistic motion, and crisp cinematic pacing",
    "night_city_timelapse": "photorealistic midnight city timelapse with an elevated urban highway, persistent vehicle light trails, deep night exposure, luminous city lights, smooth temporal motion, and a realistic long-exposure photography aesthetic",
    "modern_neo_noir": "modern neo-noir cinema with deep blacks, selective practical lighting, wet reflective streets, restrained neon accents, moody shadows, shallow depth of field, slow deliberate camera movement, and realistic live-action texture",
    "classic_film_noir": "classic film noir with monochrome live-action cinematography, hard directional key light, deep black shadows, smoky interiors, dramatic silhouettes, venetian-blind light patterns, restrained dolly movement, and vintage film contrast",
    "crime_thriller": "realistic cinematic crime thriller with tense low-key lighting, practical fluorescent and tungsten sources, a muted color palette, controlled handheld camera, shallow focus, restrained performances, realistic urban locations, and slow-building tension",
    "thriller_1990s": "1990s live-action thriller with moody colored practical lighting, slightly heightened contrast, dramatic close-ups, fast purposeful cuts, subtle analog character, an unsettling atmosphere, and photorealistic cinematic texture",
    "cinematic_horror_live_action": "photorealistic cinematic horror with oppressive low-key lighting, practical light sources, deep shadow detail, unsettling negative space, restrained camera movement, realistic environmental texture, subtle atmospheric haze, and grounded horror realism",
    "found_footage_horror": "realistic handheld found footage with imperfect autofocus, slight exposure hunting, subtle handheld shake, fluorescent light flicker, natural sensor noise, delayed focus response, accidental framing, realistic reflections, and unpolished documentary camera behavior",
    "consumer_camcorder_horror": "consumer camcorder horror footage with imperfect handheld framing, automatic exposure shifts, soft digital detail, autofocus breathing, practical fluorescent lighting, minor sensor noise, and realistic accidental camera movement",
    "observational_documentary": "observational documentary live action with natural available lighting, an unobtrusive handheld camera, imperfect framing, realistic focus adjustments, natural body language, minimal cinematic polish, authentic environmental sound, and a candid unstaged atmosphere",
    "workplace_mockumentary": "photorealistic workplace mockumentary with subtle handheld camera shake, natural office fluorescent lighting, medium documentary framing, awkward pauses, restrained reaction shots, realistic office room tone, and deadpan timing",
    "reality_tv_documentary": "realistic reality-TV documentary with a handheld shoulder camera, reactive reframing, quick natural focus corrections, available interior lighting, spontaneous body language, imperfect composition, and realistic room ambience",
    "grounded_martial_arts_cinema": "hyper-realistic cinematic martial-arts action with grounded human physics, realistic weight transfer, fast physical choreography, a wet reflective environment, practical industrial lighting, cinematic dolly movement, speed ramping, controlled camera shake, and realistic motion blur",
    "gritty_close_quarters_action": "gritty close-quarters action film with physically believable combat, realistic inertia and recovery, practical lighting, a handheld cinematic camera, environmental debris and collisions, realistic impact reactions, and restrained motion blur",
    "dark_fantasy_live_action": "hyper-realistic dark fantasy cinema with a grounded medieval environment, gritty practical lighting, candlelight and firelight, smoky atmosphere, weathered materials, realistic body momentum, a practical-effects aesthetic, and handheld cinematic camera",
    "neon_cyberpunk_cinema": "photorealistic cyberpunk cinema with heavy neon rain, reflective wet surfaces, volumetric neon fog, glowing practical lights, subtle lens flare, dark futuristic production design, cinematic slow motion, and a dynamic orbiting camera",
    "dark_dystopian_scifi": "dark dystopian live-action science fiction with industrial futuristic architecture, cold practical lighting, dense atmospheric haze, restrained neon accents, weathered technology, realistic materials, and grounded cinematic realism",
    "prestige_scifi_drama": "prestige science-fiction cinema with restrained futuristic production design, natural human performances, soft volumetric atmosphere, clean practical lighting, subtle visual effects, realistic materials, and slow controlled camera movement",
    "high_fashion_editorial": "high-fashion editorial film with a premium fashion-photography aesthetic, realistic skin texture with visible pores, dramatic model posing, controlled editorial camera movement, sculptural lighting, a minimalist luxury mood, and magazine-grade composition",
    "korean_fashion_campaign": "premium Korean fashion campaign with international magazine editorial photography, controlled model posing, clean high-contrast composition, restrained camera motion, premium skin texture, a fashion-lookbook atmosphere, and polished editorial finish",
    "streetwear_fashion_film": "urban streetwear fashion film with natural city light, dynamic tracking shots, wide-angle movement, architectural backgrounds, hard editorial cuts, confident restrained poses, and a polished urban fashion aesthetic",
    "minimalist_premium_product": "minimalist premium product film with a pristine studio environment, controlled softbox lighting, precise product highlights, clean reflective surfaces, elegant macro details, restrained camera movement, and premium commercial finish",
    "luxury_automotive_commercial": "premium luxury automotive commercial with elegant restrained cinematography, a dark architectural environment, glossy controlled body reflections, slow precision camera movement, low-angle hero shots, premium practical lighting, and realistic automotive materials",
    "performance_car_commercial": "high-performance automotive commercial with a low tracking camera, physically realistic vehicle motion, aggressive but controlled camera movement, tire spray, realistic suspension load, detailed paint reflections, cinematic landscape, and restrained motion blur",
    "food_macro_commercial": "premium cinematic food commercial with ultra-realistic macro photography, glossy food textures, shallow depth of field, dramatic practical lighting, controlled slow motion, detailed steam and condensation, appetizing highlights, and polished advertising finish",
    "dark_surreal_commercial": "surreal cinematic commercial with photorealistic subjects, moody colored neon lighting, dark humor, rapid editorial cutting, exaggerated macro detail, unsettling character expressions, and a retro-thriller atmosphere",
    "ultra_realistic_pov": "ultra-realistic first-person POV footage with realistic hands, energetic handheld camera, physically believable recoil and body movement, lens droplets, environmental reflections, subtle motion blur, and immersive spatial audio",
    "smartphone_ugc": "photorealistic smartphone UGC video with arm's-length selfie framing, natural handheld movement, available daylight, realistic smartphone depth of field, casual creator performance, natural skin texture, conversational pacing, and authentic social-media realism",
    "film_1970s": "1970s live-action film aesthetic with warm analog color response, soft optical contrast, practical lighting, and restrained film grain",
    "cinema_1980s": "1980s cinematic live action with era-authentic production design, analog color response, practical lighting, and subtle film grain",
    "cinema_1990s": "1990s live-action cinema with slightly soft optical rendering, subtle analog texture, and era-authentic lighting and color response",
    "early_2000s_digital_cinema": "early-2000s digital cinema aesthetic with slightly harsh highlights, restrained saturation, realistic digital sensor response, and a period-accurate production look",
    "modern_digital_cinema": "modern digital cinema with clean high dynamic range, natural skin texture, controlled highlight roll-off, and restrained cinematic grading",
}


def _normalize_shot_presets(value: Any) -> dict[str, str]:
    raw = value if isinstance(value, dict) else {}
    normalized = {}
    for preset_name, choices in CAMERA_PRESET_PROMPTS.items():
        selected = _clean_text(raw.get(preset_name)).lower()
        normalized[preset_name] = selected if selected in choices else "none"
    selected_style = _clean_text(raw.get("style")).lower()
    normalized["style"] = selected_style if selected_style in STYLE_PRESET_PROMPTS else "none"
    return normalized


def _figurine_animation_system_module(
    project: dict[str, Any], mode: str, enhance_level: str,
) -> str:
    """Return opt-in figurine motion rules for only the shots using that style preset."""
    selected_shots: list[int] = []
    shot_number = 0
    for item in project.get("shots", []):
        if not _is_move(item):
            shot_number += 1
        if (_normalize_shot_presets(item.get("presets"))["style"] == "figurine_animation"
                and shot_number not in selected_shots):
            selected_shots.append(shot_number)
    if not selected_shots:
        return ""

    shot_scope = ", ".join(f"[Shot {index}]" for index in selected_shots)
    rules = SYSTEM_PROMPT_CONFIG["static_asset_rules"]
    modules = [
        "FIGURINE PRESET SCOPE: This preset was explicitly selected for "
        f"{shot_scope}. Apply the following rules only to those shots and never carry them into another shot.",
        rules["common"],
    ]
    if enhance_level in {"normal", "strong"}:
        modules.append(rules["enhanced"])
    if mode == "FL2VA":
        modules.append(rules["FL2VA"])
        if enhance_level in {"normal", "strong"}:
            modules.append(rules["FL2VA_enhanced"])
    return "\n\n".join(module.strip() for module in modules if module.strip())


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
    "Cowboy Shot": "The composition frames the subject from mid-thigh to slightly above the head, with both hands and the waist fully visible.",
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
    strong_enhance_addendum = config.get("strong_enhance_addendum", "")
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
        or not isinstance(strong_enhance_addendum, str)
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
            "'common_addendum', 'enhance_addendum', 'strong_enhance_addendum', 'action_semantics', 'video_reference_common', "
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
    legacy_camera_rule = (
        "Otherwise select framing that contains the whole action and use one motivated camera "
        "behavior per shot, described naturally with motion type and, when useful, amplitude and speed."
    )
    move_camera_rule = (
        "Otherwise select framing that contains the whole action and use one coherent physical camera "
        "path per Shot; configured Moves are consecutive phases of that path and must be described naturally."
    )
    common = common.replace(legacy_camera_rule, move_camera_rule)
    common_enhanced = common_enhanced.replace(legacy_camera_rule, move_camera_rule)
    modes = dict(modes)
    modes["REF2VA"] = modes["REF2VA"].replace(
        "Do not define a standalone Picture unless it is a configured frame anchor.",
        "Do not define a standalone Picture unless it is a configured frame anchor or a storyboard/shot-planning reference mapped to configured Shots.",
    )
    return {
        "common": common,
        "common_enhanced": common_enhanced,
        "enhance_addendum": enhance_addendum,
        "strong_enhance_addendum": strong_enhance_addendum,
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
COMMON_LLM_SYSTEM_RULES += SYSTEM_PROMPT_CONFIG["common_addendum"]
ENHANCED_COMMON_LLM_SYSTEM_RULES = SYSTEM_PROMPT_CONFIG["common_enhanced"]
ENHANCED_COMMON_LLM_SYSTEM_RULES += SYSTEM_PROMPT_CONFIG["action_semantics"]
ENHANCED_COMMON_LLM_SYSTEM_RULES += SYSTEM_PROMPT_CONFIG["common_addendum"]
BASE_LLM_SYSTEM_RULES = SYSTEM_PROMPT_CONFIG["base"]
MODE_LLM_SYSTEM_PROMPTS = {
    mode: COMMON_LLM_SYSTEM_RULES
    + ("" if mode == "REF2VA" else BASE_LLM_SYSTEM_RULES)
    + SYSTEM_PROMPT_CONFIG["modes"][mode]
    + SYSTEM_PROMPT_CONFIG["mode_addenda"].get(mode, "")
    for mode in SUPPORTED_MODES
    if mode != "AUTO"
}
ENHANCED_MODE_LLM_SYSTEM_PROMPTS = {
    mode: ENHANCED_COMMON_LLM_SYSTEM_RULES
    + ("" if mode == "REF2VA" else BASE_LLM_SYSTEM_RULES)
    + SYSTEM_PROMPT_CONFIG["modes"][mode]
    + SYSTEM_PROMPT_CONFIG["mode_addenda"].get(mode, "")
    + SYSTEM_PROMPT_CONFIG["enhance_addendum"]
    for mode in SUPPORTED_MODES
    if mode != "AUTO"
}

STRONG_ENHANCE_ADDENDUM = SYSTEM_PROMPT_CONFIG["strong_enhance_addendum"]
STRONG_MODE_LLM_SYSTEM_PROMPTS = {
    mode: ENHANCED_COMMON_LLM_SYSTEM_RULES
    + ("" if mode == "REF2VA" else BASE_LLM_SYSTEM_RULES)
    + SYSTEM_PROMPT_CONFIG["modes"][mode]
    + SYSTEM_PROMPT_CONFIG["mode_addenda"].get(mode, "")
    + STRONG_ENHANCE_ADDENDUM
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
    shot_number = 0
    for shot in project.get("shots", []):
        if not _is_move(shot):
            shot_number += 1
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


def _move_output_cues(project: dict[str, Any], effective_seconds: float) -> list[str]:
    """Return range-based inline timing phrases for Moves without creating headers."""
    cues: list[str] = []
    for take in _compile_timeline_takes(project, effective_seconds):
        for beat in take["beats"]:
            connector = (
                "without a cut"
                if beat["move_number"] == 1
                else "continuing the same uninterrupted camera path"
            )
            cues.append(
                f"From {format_timestamp(beat['start'])} to {format_timestamp(beat['end'])}, "
                f"{connector},"
            )
    return cues


def _compile_timeline_takes(project: dict[str, Any], effective_seconds: float) -> list[dict[str, Any]]:
    """Compile flat UI Shot/Move items into Shot-scoped takes and continuous beats.

    This is the shared timing representation used by prompt serialization and
    validation. A Shot starts a take; following Moves are beats inside that take.
    """
    items = project.get("shots", [])
    requested_seconds = sum(float(item.get("duration", 0.0)) for item in items)
    scale = effective_seconds / requested_seconds if requested_seconds > 0 else 1.0
    cursor = 0.0
    takes: list[dict[str, Any]] = []
    shot_number = 0
    for item_index, item in enumerate(items):
        start = cursor
        end = start + float(item.get("duration", 0.0)) * scale
        if not _is_move(item):
            shot_number += 1
            takes.append({
                "shot_number": shot_number,
                "start": start,
                "opening_end": end,
                "end": end,
                "opening": item,
                "beats": [],
            })
        elif takes:
            beats = takes[-1]["beats"]
            beats.append({
                "move_number": len(beats) + 1,
                "item_index": item_index,
                "start": start,
                "end": end,
                "duration": end - start,
                "item": item,
            })
            takes[-1]["end"] = end
        cursor = end
    return takes


def _camera_take_plan(project: dict[str, Any], effective_seconds: float) -> str:
    """Describe Shot-scoped camera takes and their owned Move intervals compactly."""
    items = project.get("shots", [])
    if not any(_is_move(item) for item in items):
        return ""
    requested_seconds = sum(float(item["duration"]) for item in items)
    scale = effective_seconds / requested_seconds if requested_seconds > 0 else 1.0
    cursor = 0.0
    shot_number = 0
    takes: list[dict[str, Any]] = []
    for item in items:
        start = cursor
        cursor += float(item["duration"]) * scale
        if _is_move(item):
            if takes:
                takes[-1]["moves"].append(start)
                takes[-1]["end"] = cursor
            continue
        shot_number += 1
        takes.append({"shot": shot_number, "start": start, "end": cursor, "moves": []})
    lines = ["CAMERA_TAKE_PLAN:"]
    for take in takes:
        start = format_timestamp(take["start"])
        end = format_timestamp(take["end"])
        opening = "opens the first camera take" if take["shot"] == 1 else "starts a new camera take with an intentional cut"
        if take["moves"]:
            move_times = ", ".join(format_timestamp(value) for value in take["moves"])
            ownership = f"owns continuous Moves beginning at {move_times}; no cut is allowed inside this take"
        else:
            ownership = "contains no Moves"
        lines.append(f"- [Shot {take['shot']}] {opening} at {start}; {ownership}; take ends at {end}.")
    return "\n".join(lines)


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
        "kind": "move" if index > 0 and _clean_text(raw.get("kind")).lower() == "move" else "shot",
        "duration": max(MIN_SHOT_DURATION, _number(raw.get("duration"), 1.0)),
        "visual_action": visual_action,
        "presets": _normalize_shot_presets(raw.get("presets")),
    }


def _normalize_reference(raw: Any, index: int) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    ref_type = _clean_text(raw.get("type")).lower()
    if ref_type not in ("picture", "video", "audio"):
        ref_type = "picture"
    role = _clean_text(raw.get("role")).lower()
    strength = _clean_text(raw.get("strength")).lower()
    if ref_type == "picture":
        if role in {"reference", "environment", "style"}:
            role = "subject_identity"
            strength = "weak"
        elif role == "subject_identity":
            # Version 9 and earlier used Subject as an implicit strong role.
            strength = strength if strength in SUBJECT_STRENGTHS else "strong"
        elif role not in {"first_frame", "last_frame", "frame", "storyboard"}:
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
        if role == "subject_visual":
            strength = strength if strength in SUBJECT_STRENGTHS else "normal"
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
    if ref_type == "picture" and role != "storyboard":
        description = ""
    storyboard_shot_ids = [
        _clean_text(value) for value in raw.get("storyboard_shot_ids", [])
        if _clean_text(value)
    ] if isinstance(raw.get("storyboard_shot_ids"), list) else []
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
        "storyboard_shot_ids": storyboard_shot_ids,
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
    if raw_version is not None and raw_version != CURRENT_PROJECT_VERSION and raw_version not in {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}:
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
    if selected_enhance_model in REMOVED_LIGHTX2V_MODEL_IDS:
        selected_enhance_model = DEFAULT_ENHANCE_MODEL_ID
    project["enhance_model"] = (
        selected_enhance_model
        if selected_enhance_model in {DEFAULT_ENHANCE_MODEL_ID, OMNI_MODEL_ID}
        else DEFAULT_ENHANCE_MODEL_ID
    )
    selected_image_model = _clean_text(raw.get("image_model"))
    if selected_image_model in REMOVED_LIGHTX2V_MODEL_IDS:
        selected_image_model = QWEN_IMAGE_MODEL_ID
    project["image_model"] = (
        selected_image_model
        if selected_image_model in {QWEN_IMAGE_MODEL_ID, OMNI_MODEL_ID}
        else QWEN_IMAGE_MODEL_ID
    )
    project["auto_run"] = raw.get("auto_run") is True
    raw_enhance_level = _clean_text(raw.get("enhance_level")).lower()
    project["enhance_level"] = (
        raw_enhance_level if raw_enhance_level in {"none", "normal", "strong"}
        else "normal" if raw.get("enhance") is True else "none"
    )
    # Retain the legacy boolean for old workflows and callers. Any expansion
    # level other than None uses the enhanced prompt family.
    project["enhance"] = project["enhance_level"] != "none"
    project["enhanced_prompt"] = _clean_text(raw.get("enhanced_prompt"))

    raw_shots = raw.get("shots")
    if isinstance(raw_shots, list) and raw_shots:
        project["shots"] = [_normalize_shot(item, i) for i, item in enumerate(raw_shots)]
    else:
        requested = min(15.0, max(MIN_SHOT_DURATION, _number(raw.get("requested_duration"), 5.0)))
        project["shots"][0]["duration"] = requested
    # Version 24 stored one preset bundle globally. Migrate it only to Shot 1
    # so later shots remain independently configurable.
    if isinstance(raw.get("presets"), dict) and project["shots"]:
        project["shots"][0]["presets"] = _normalize_shot_presets(raw["presets"])

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
    # Preserve frame-derived timeline minima such as 2 / 24 seconds. Display
    # timestamps remain millisecond-formatted, but internal fitting should not
    # shorten a two-frame item through three-decimal rounding.
    project["requested_duration"] = round(requested_duration, 6)
    effective_duration = align_frame_count(requested_duration) / MODEL_FPS
    effective_frames = align_frame_count(requested_duration)
    raw_refs = raw.get("references")
    project["references"] = (
        [_normalize_reference(item, i) for i, item in enumerate(raw_refs)]
        if isinstance(raw_refs, list)
        else []
    )
    for ref in project["references"]:
        if ref["type"] == "picture" and ref["role"] == "frame":
            ref["frame_index"] = min(ref["frame_index"], effective_frames - 1)
    for ref in project["references"]:
        if ref["type"] != "video":
            continue
        source_duration = max(ref["source_duration"], ref["duration"])
        ref["source_duration"] = source_duration
        ref["trim_start"] = min(ref["trim_start"], max(0.0, source_duration - MIN_SHOT_DURATION))
        available = max(0.0, source_duration - ref["trim_start"])
        # Preserve the complete source clip. Only its intersection with the
        # target timeline is decoded and sent downstream.
        ref["duration"] = min(ref["duration"], available)
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
                    and supplied.lower() in {"reference", "environment", "style"}
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


def _is_move(item: dict[str, Any]) -> bool:
    return item.get("kind") == "move"


def _shot_items(project: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in project.get("shots", []) if not _is_move(item)]


def _shot_groups(project: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Group each Shot with the following continuous Move intervals."""
    groups: list[list[dict[str, Any]]] = []
    for item in project.get("shots", []):
        if not _is_move(item) or not groups:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def validate_project(project: dict[str, Any], parse_warnings: list[str] | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings = list(parse_warnings or [])
    duration = float(project["requested_duration"])
    shot_total = sum(float(shot["duration"]) for shot in project["shots"])
    semantic_shots = _shot_items(project)
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
    if not semantic_shots or _is_move(project["shots"][0]):
        errors.append("The timeline must begin with a Shot; a Move cannot exist without a preceding Shot.")
    if not project["user_request"] and not any(shot["visual_action"] for shot in project["shots"]):
        warnings.append("No overall request or shot action has been entered.")
    descriptive_text = [
        project["user_request"], project["constraints"],
        *(shot["visual_action"] for shot in project["shots"]),
        *(ref["description"] for ref in project["references"]),
    ]
    if any("\ufffd" in text for text in descriptive_text):
        errors.append(
            "Prompt text contains the Unicode replacement character (U+FFFD), indicating that text was "
            "damaged before compilation. Re-enter the affected text as UTF-8."
        )
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
        if len(semantic_shots) > 1:
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
    effective_duration = align_frame_count(duration) / MODEL_FPS
    visible_video_durations = [
        _visible_video_selection(ref, effective_duration)[1] for ref in videos
    ]
    video_total = sum(visible_video_durations)
    for ref, visible_duration in zip(videos, visible_video_durations):
        if visible_duration and not REF_VIDEO_MIN_SECONDS <= visible_duration <= effective_duration:
            errors.append(
                f"{ref['label']} visible timeline segment must be 2-{effective_duration:.2f} seconds; "
                f"received {visible_duration:.2f}s."
            )
        elif not visible_duration and ref.get("video_filename"):
            warnings.append(f"{ref['label']} has no duration metadata; the 2-15 second limit cannot be verified.")
    reference_total_limit = max(REF_VIDEO_TOTAL_SECONDS, effective_duration)
    if video_total > reference_total_limit:
        errors.append(
            f"Reference-video duration totals {video_total:.2f}s; "
            f"the maximum is {reference_total_limit:.2f}s."
        )
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


def _reference_alias_is_environment(ref: dict[str, Any]) -> bool:
    """Recognize aliases intentionally naming a reusable place or setting."""
    alias = str(ref.get("alias") or "").lstrip("@").replace("-", "_").casefold()
    parts = {part for part in alias.split("_") if part}
    return bool(parts.intersection({
        "place", "location", "environment", "setting", "background", "scene",
        "room", "interior", "exterior", "restaurant", "street", "beach",
        "장소", "환경", "배경", "공간", "방", "식당", "거리", "해변",
    }))


def _reference_applicable_shots(project: dict[str, Any], ref: dict[str, Any]) -> list[int]:
    """Return the target shots in which a reference is expected to apply."""
    groups = _shot_groups(project)
    if not groups:
        return [1]

    role = ref.get("role")
    if ref.get("type") == "picture":
        if role == "first_frame":
            return [1]
        if role == "last_frame":
            return [len(groups)]
        if role == "frame":
            frame_index = max(0, int(ref.get("frame_index", 0)))
            anchor_time = frame_index / MODEL_FPS
            cursor = 0.0
            for index, group in enumerate(groups, 1):
                cursor += sum(max(0.0, float(item.get("duration", 0.0))) for item in group)
                if anchor_time < cursor or index == len(groups):
                    return [index]
        if role == "storyboard":
            selected = set(ref.get("storyboard_shot_ids") or [])
            if selected:
                mapped = [
                    index for index, group in enumerate(groups, 1)
                    if str(group[0].get("id") or "") in selected
                ]
                if mapped:
                    return mapped
            return list(range(1, len(groups) + 1))

    if ref.get("type") == "video" and ref.get("duration"):
        start = max(0.0, float(ref.get("timeline_start", 0.0)))
        end = start + max(0.0, float(ref.get("duration", 0.0)))
        cursor = 0.0
        applicable: list[int] = []
        for index, group in enumerate(groups, 1):
            shot_start = cursor
            cursor += sum(max(0.0, float(item.get("duration", 0.0))) for item in group)
            if start < cursor and end > shot_start:
                applicable.append(index)
        if applicable:
            return applicable

    alias = str(ref.get("alias") or "").lower()
    if alias:
        applicable = []
        for index, group in enumerate(groups, 1):
            if any(alias in str(item.get("visual_action") or "").lower() for item in group):
                applicable.append(index)
        if applicable:
            # A named environment persists as the scene context after its
            # first mention. Character Subjects use authored visible shots;
            # physical continuity must not falsely claim an off-camera actor
            # appears in an intervening shot.
            if (ref.get("type") == "picture" and role == "subject_identity"
                    and _reference_alias_is_environment(ref)):
                return list(range(applicable[0], len(groups) + 1))
            return applicable
        if alias in str(project.get("user_request") or "").lower():
            return list(range(1, len(groups) + 1))
        # An explicitly aliased Subject that is never requested is unused.
        # Exclude it instead of inventing an all-shot retention relationship.
        if ((ref.get("type") == "picture" and role == "subject_identity")
                or (ref.get("type") == "video" and role in {"subject_visual", "visual_style"})):
            return []

    return list(range(1, len(groups) + 1))


def _retention_prefix(label: str, plan: dict[str, Any], ref: dict[str, Any],
                      applicable_shots: list[int]) -> str:
    """Build the guide-compliant portion Qwen must copy verbatim."""
    marker = plan["marker"]
    shot_list = ", ".join(f"[Shot {number}]" for number in applicable_shots)
    if plan["kind"] == "Subject":
        return f"{label} (appears in {shot_list}): {marker} -"
    if plan["kind"] == "Picture":
        role = ref.get("role")
        if role == "first_frame":
            scope = "[Shot 1] first frame"
        elif role == "last_frame":
            scope = f"[Shot {applicable_shots[-1]}] final frame"
        elif role == "frame":
            scope = f"[Shot {applicable_shots[0]}] frame {max(0, int(ref.get('frame_index', 0)))}"
        elif role == "storyboard":
            scope = f"storyboard planning for {shot_list}"
        else:
            scope = f"applies to {shot_list}"
        return f"{label} ({scope}): {marker} -"
    if plan["kind"] == "Video":
        return f"{label} (applies to {shot_list}): {marker} -"
    return f"{label}: {marker} -"


def _reference_model(project: dict[str, Any]) -> dict[str, Any]:
    references = _reference_labels(project["references"])
    target_duration = align_frame_count(float(project.get("requested_duration") or 5.0)) / MODEL_FPS
    subject_count = 0
    aliases: dict[str, str] = {}
    definitions: list[str] = []
    retention: list[str] = []
    applications: list[str] = []
    task_types: list[str] = []
    label_plan: dict[str, dict[str, str]] = {}
    summary_relations: list[str] = []
    final_shot = len(_shot_items(project))

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
                "subject_visual": "subject and visible-content reference",
                "visual_style": "visual-style reference",
                "motion": "motion and action timing",
                "motion_camera": "motion, action timing, and camera behavior",
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
        is_picture_subject = ref["type"] == "picture" and ref["role"] in {"reference", "subject_identity"}
        is_video_subject = ref["type"] == "video" and ref["role"] in {"subject_visual", "visual_style"}
        if is_picture_subject or is_video_subject:
            applicable_shots = _reference_applicable_shots(project, ref)
            if not applicable_shots:
                continue
            subject_count += 1
            subject = f"<Subject {subject_count}>"
            if ref["alias"]:
                aliases[ref["alias"]] = subject
            if ref["role"] == "visual_style":
                strength = "weak"
            else:
                strength = "weak" if ref["role"] == "reference" else ref.get("strength", "normal")
            if ref["role"] == "visual_style" or strength == "style_transfer":
                definition = f"{subject} is the reusable visual style derived from {source_label}"
            else:
                definition = f"{subject} is the reusable visible subject derived from {source_label}"
            if description:
                definition += f", described as {description.rstrip('.')}"
            if ref["type"] == "video" and ref["duration"]:
                source_start, selected_duration, _target_start = _visible_video_selection(
                    ref, target_duration,
                )
                definition += f", sampled only from the selected {selected_duration:.2f}-second source interval"
                if source_start > 0.0005:
                    definition += f" beginning at {source_start:.2f} seconds"
            definitions.append(definition + ".")
            marker = {
                "weak": "weak_reference",
                "normal": "partially_preserved",
                "attribute_transfer": "attribute_transfer",
                "style_transfer": "attribute_transfer",
                "strong": "fully_preserved",
            }[strength]
            retention_detail = {
                "weak": "retain only broad similarity in a small set of target-relevant visible characteristics",
                "normal": "retain core identity and primary visible appearance while allowing secondary details to vary",
                "attribute_transfer": "transfer only the explicitly requested visible attributes to a different identifiable target subject without copying the source identity",
                "style_transfer": "transfer only the explicitly requested visual medium and rendering treatment without copying source identity, appearance, clothing, or scene content",
                "strong": "preserve the complete visible subject identity, appearance, and source visual medium/rendering style wherever it appears",
            }[strength]
            retention.append(f"{subject}: {marker} - {retention_detail}.")
            if not ref["alias"]:
                applications.append(f"Apply {subject} only as its defined Subject content at {strength} strength.")
            role_contract = {
                "weak": "broad subject appearance similarity only; exclude source setting, style, composition, camera, lighting, palette, pose, and action",
                "normal": "core subject identity and primary visible appearance; secondary details may vary; exclude source setting, style, composition, camera, lighting, palette, pose, and action",
                "attribute_transfer": "transfer only explicitly requested visible attributes to a different identifiable target subject; preserve the target subject's identity and exclude source identity, setting, style, composition, camera, lighting, palette, pose, and action",
                "style_transfer": "transfer only the explicitly requested source visual medium and rendering treatment to the identifiable target subject or target video; preserve the target identity, face, body, hairstyle, clothing, accessories, objects, and action; exclude source identity, appearance, wardrobe, props, environment, composition, camera, pose, action, and audio; do not infer physical attributes from the style source",
                "strong": "complete visible subject identity and appearance plus that subject's source visual medium/rendering style; preserve the style independently per subject; exclude source setting, composition, camera, lighting setup, scene-wide palette, pose, and action",
            }[strength]
            if ref["role"] == "visual_style":
                retention_detail = "reference only the requested visual medium, palette, lighting treatment, materials, and texture"
                role_contract = (
                    "visual style only: rendering medium, palette, lighting treatment, materials, and texture; "
                    "exclude source identity, face, body, hair, clothing, action, environment layout, composition, camera, cuts, and audio"
                )
            elif ref["type"] == "video":
                role_contract += "; exclude source motion, action timing, camera, cuts, and audio"
            label_plan[subject] = {
                "kind": "Subject", "source": source_label, "role": ref["role"], "marker": marker,
                "strength": strength, "contract": role_contract,
            }
            label_plan[subject]["applicable_shots"] = applicable_shots
            label_plan[subject]["retention_prefix"] = _retention_prefix(
                subject, label_plan[subject], ref, applicable_shots
            )
            summary_relations.append(
                f"{subject} as a visual-style reference"
                if ref["role"] == "visual_style"
                else f"{subject} as a {strength}-strength subject reference"
            )
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
        elif ref["type"] == "picture" and ref["role"] == "storyboard":
            applicable = _reference_applicable_shots(project, ref)
            shot_text = " and ".join(f"[Shot {number}]" for number in applicable)
            definition = (
                f"{source_label} is a storyboard reference for {shot_text}, defining their viewpoint, "
                "subject placement, approximate framing, explicitly depicted action beats, and shot order"
            )
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
                source_start, selected_duration, _target_start = _visible_video_selection(
                    ref, target_duration,
                )
                definition += (
                    f", using the selected {selected_duration:.2f}-second source interval"
                )
                if source_start > 0.0005:
                    definition += f" beginning at {source_start:.2f} seconds"
                definition += " as the configured analysis and reference segment"
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
            "subject_visual": "reference only the specified reusable visible subject content; do not copy source motion, action timing, camera, cuts, or audio",
            "visual_style": "reference only rendering medium, palette, lighting treatment, materials, and visual texture; do not copy source identity, action, environment layout, composition, camera, cuts, or audio",
            "motion": "transfer only actor-neutral pose progression, movement paths, direction, speed, contacts, interaction timing, and physical rhythm to the target subject; never copy or describe the source performer's identity, face, age, gender, body shape or proportions, skin, hair, clothing, accessories, materials, texture, rendering style, environment, camera, cuts, or audio",
            "motion_camera": "transfer only actor-neutral pose progression, movement paths, direction, speed, contacts, interaction timing, weight transfer, physical rhythm, camera path, viewpoint and framing progression, camera timing, and the synchronization between performance and camera; never copy or describe source identity, face, age, gender, body shape or proportions, skin, hair, clothing, accessories, props or visible content, materials, texture, rendering style, environment, lighting, cuts, visible text, or audio",
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
        picture_contracts = {
            "storyboard": (
                "use only storyboard panel order, shot order, viewpoint, approximate framing, subject placement, "
                "and explicitly depicted action beats for the applicable Shots; preserve every distinct panel "
                "viewpoint or shot-size change in order instead of collapsing them into generic tracking; do not treat it as an exact frame "
                "or transfer identity, clothing, visual style, lighting, palette, exact timing, or pose; inside each "
                "configured Shot, panel boundaries are chronological action beats within that one take and never "
                "create cuts or camera resets; cuts are allowed only at configured later Shot boundaries"
            ),
        }
        label_plan[source_label] = {
            "kind": ref["type"].title(), "source": source_label,
            "role": ref["role"], "marker": marker,
            "contract": (
                video_contracts.get(ref["role"], f"use only as the defined {role_text} relationship")
                if ref["type"] == "video"
                else picture_contracts.get(ref["role"], audio_contracts.get(
                    ref["role"], f"use only as the defined {role_text} relationship"
                ))
            ),
        }
        applicable_shots = _reference_applicable_shots(project, ref)
        label_plan[source_label]["applicable_shots"] = applicable_shots
        label_plan[source_label]["retention_prefix"] = _retention_prefix(
            source_label, label_plan[source_label], ref, applicable_shots
        )
        if ref["type"] == "picture" and ref["role"] == "frame":
            frame_index = max(0, int(ref.get("frame_index", 0)))
            label_plan[source_label]["frame_index"] = frame_index
            label_plan[source_label]["anchor_time_seconds"] = frame_index / MODEL_FPS
        summary_relations.append(f"{source_label} for {role_text}")
        if not ref["alias"]:
            if ref["role"] == "reference":
                applications.append(f"Apply {source_label} only as a {generic_reference_text}.")
            else:
                applications.append(f"Apply {source_label} only as the {role_text} reference.")

    if any(plan.get("kind") == "Picture" and plan.get("role") == "frame" for plan in label_plan.values()):
        effective_seconds = align_frame_count(project.get("requested_duration", 5.0)) / MODEL_FPS
        for anchor in _frame_anchor_schedule(project, effective_seconds):
            if anchor["label"] in label_plan:
                label_plan[anchor["label"]]["anchor_kind"] = anchor["anchor_kind"]

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
    shot_number = 0
    move_number = 0
    for item_index, shot in enumerate(project["shots"]):
        is_move = _is_move(shot)
        if is_move:
            move_number += 1
        else:
            shot_number += 1
            move_number = 0
        fragments: list[str] = []
        if item_index == 0 and project["user_request"]:
            fragments.append(_sentence(_replace_aliases(project["user_request"], aliases)))
        if item_index == 0 and reference_applications:
            fragments.extend(reference_applications)
        if shot["visual_action"]:
            fragments.append(_sentence(_replace_aliases(shot["visual_action"], aliases)))
        if is_move:
            end = cursor + float(shot["duration"]) * scale
            prefix = (
                f"From {format_timestamp(cursor)} to {format_timestamp(end)}, without a cut, the same physical "
                "camera continues through the uninterrupted take while "
            )
            if not fragments:
                fragments.append("the camera continues smoothly from its preceding state.")
        elif shot_number == 1:
            prefix = "[Shot 1] "
            if not fragments:
                fragments.append("The scene begins with no additional shot-specific action specified.")
        else:
            prefix = f"[Shot {shot_number}] At {format_timestamp(cursor)}, cut to a new shot. "
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
                             content_locks: list[str] | None = None,
                             move_cues: list[str] | None = None) -> str:
    shots = ", ".join(f"[Shot {number}]" for number in expected_shots)
    content_lock = ""
    if content_locks:
        content_lock = (
            "\nINPUT-DERIVED CONTENT LOCKS — these are binding, not output headings:\n- "
            + "\n- ".join(content_locks)
        )
    move_lock = ""
    if move_cues:
        move_lock = (
            "\nMandatory inline Move cues, in order: " + " | ".join(move_cues)
            + "\nCopy each range cue literally once at its chronological position inside its owning Shot, but "
              "embed it within the ongoing action paragraph or applicable frame-to-frame bridge instead of opening a new scene paragraph. "
              "A Move changes action or camera behavior inside the existing take; it does not require a new "
              "composition, completed state, pause, camera restatement, or cut. Only a later Shot header may cut."
        )
    if mode == "REF2VA":
        label_plan = (reference_model or {}).get("label_plan", {})
        labels = ", ".join(label_plan) or "the locked labels above"
        frame_definition_lock = ""
        role_definition_lock = ""
        label_lock = f"Define and use exactly these output labels in order: {labels}. Keep them literal; create no others."
        storyboard_plans = [
            (label, plan) for label, plan in label_plan.items()
            if plan.get("kind") == "Picture" and plan.get("role") == "storyboard"
        ]
        if storyboard_plans:
            storyboard_definitions = " | ".join(
                f"{label} is a storyboard reference for "
                + " and ".join(f"[Shot {number}]" for number in (plan.get("applicable_shots") or [1]))
                + ", defining their viewpoint, subject placement, approximate framing, explicitly depicted action beats, and shot order."
                for label, plan in storyboard_plans
            )
            role_definition_lock = (
                "\nMandatory storyboard definitions: " + storyboard_definitions
                + " Copy each definition literally once in subject_definitions. Storyboard Pictures plan only their "
                  "listed Shots and are never exact frames, Subjects, identity/style sources, or reasons to add cuts. "
                  "Within each listed Shot, interpret consecutive storyboard panels as chronological action beats "
                  "inside one uninterrupted take; panel boundaries never create cuts, transitions, viewpoint jumps, "
                  "or camera resets. Preserve each distinct panel viewpoint, shot size, screen direction, and subject "
                  "placement in order by translating it into continuous physical camera travel. Adjacent panels may "
                  "be merged only when their framing and action are materially the same. Never replace the ordered "
                  "framing progression with only `coherent framing`, `the camera tracks`, `the camera follows`, or an "
                  "equivalent generic summary. Only a configured later Shot header may cut."
            )
        frame_plans = [
            (label, plan) for label, plan in label_plan.items()
            if plan.get("kind") == "Picture" and plan.get("role") == "frame"
        ]
        if frame_plans:
            allow_dynamic_subjects = _allows_frame_continuity_subjects(label_plan)
            if allow_dynamic_subjects:
                label_lock = (
                    f"Define and use these locked Picture labels in order: {labels}. Keep them literal. "
                    "Only the recurring frame-continuity Subjects permitted below may be added before them."
                )
            frame_plans.sort(key=lambda item: (item[1].get("frame_index", 0), item[0]))
            anchor_schedule = " | ".join(
                f"{label}@{plan.get('anchor_time_seconds', 0.0):.3f}s/frame {plan.get('frame_index', 0)}"
                for label, plan in frame_plans
            )
            max_frame = max(0, int(round(effective_seconds * MODEL_FPS)) - 1)
            scheduled_anchors = []
            for label, plan in frame_plans:
                frame_index = int(plan.get("frame_index", 0))
                scheduled_anchors.append({
                    "label": label,
                    "frame_index": frame_index,
                    "time": float(plan.get("anchor_time_seconds", 0.0)),
                    "anchor_kind": (
                        plan.get("anchor_kind") or (
                            "opening" if frame_index == 0 else
                            "final" if frame_index == max_frame else
                            "intermediate"
                        )
                    ),
                })
            anchor_sentences = " | ".join(_frame_anchor_sentence(anchor) for anchor in scheduled_anchors)
            bridge_sentences_list = []
            by_shot: dict[int, list[tuple[str, dict[str, Any]]]] = {}
            for label, plan in frame_plans:
                shot_number = (plan.get("applicable_shots") or [1])[0]
                by_shot.setdefault(shot_number, []).append((label, plan))
            for shot_anchors in by_shot.values():
                shot_anchors.sort(key=lambda item: item[1].get("frame_index", 0))
                for (start_label, start_plan), (end_label, end_plan) in zip(shot_anchors, shot_anchors[1:]):
                    bridge_sentences_list.append(
                        f"From {format_timestamp(start_plan.get('anchor_time_seconds', 0.0))} to "
                        f"{format_timestamp(end_plan.get('anchor_time_seconds', 0.0))}, the same uninterrupted "
                        f"take develops continuously from {start_label} toward {end_label}."
                    )
            bridge_sentences = " | ".join(bridge_sentences_list)
            frame_definition_lock = (
                "\nFor every Picture frame anchor, subject_definitions must use its required_definition "
                "from REFERENCE_PLAN; never write that a Picture is derived from itself."
                + (
                    " Before those Picture definitions, create the smallest sequential set of <Subject N> definitions "
                    "needed to bind people, persistent objects, and environments demonstrably recurring across two or "
                    "more Picture anchors. Derive each such Subject from all supporting Pictures, never from speculation; "
                    "do not create a Subject for a one-frame-only element. Use the same Subjects throughout summary, "
                    "retention_analysis, and detailed_description."
                    if allow_dynamic_subjects else ""
                )
                + f"\nExact in-shot anchor schedule: {anchor_schedule}."
                " In detailed_description, preserve every Picture at its exact scheduled frame, even when that "
                "frame falls inside a Move. Write observable From-to action bridges between consecutive anchors. "
                "The opening anchor begins the shot exactly, intermediate anchors are states the ongoing motion "
                "passes through, and only an anchor assigned to the Shot's actual end is its final state. Pictures in one Shot remain "
                "states of one camera take and never open, reset, replace, or cut the scene."
                f"\nMandatory exact anchor sentences, in order: {anchor_sentences}"
                " Copy each sentence literally once in detailed_description at its chronological position."
                + (
                    f"\nMandatory frame-bridge sentences, in order: {bridge_sentences} "
                    "Copy each sentence literally once between its two anchor sentences, then describe the "
                    "overlapping Move actions inside that same bridge."
                    if bridge_sentences else ""
                )
            )
            if move_cues:
                move_schedule = " | ".join(
                    re.sub(
                        r"^From\s+([^,]+),.*$",
                        r"\1",
                        cue,
                        flags=re.IGNORECASE,
                    )
                    for cue in move_cues
                )
                move_lock = (
                    "\nInternal Move schedule, in order: " + move_schedule
                    + "\nPreserve every Move's action and timing, but do not copy its full range as a new From-to "
                      "paragraph. Mention a Move onset inline only where needed inside the already active Picture "
                      "bridge. A Move boundary never completes the scene, settles the image, restates the camera "
                      "contract, or interrupts the ongoing take."
                )
        retention_lines = "\n".join(
            plan.get("retention_prefix", f"{label}: {plan.get('marker', 'weak_reference')} -")
            for label, plan in label_plan.items()
        ) or "use the locked prefix for each label"
        return f"""FINAL MODE LOCK — REF2VA
Highest-priority format lock. Return plain text with no wrapper, JSON, Markdown, or commentary.
Start exactly with `subject_definitions:` and never use `integrated_multimodal_description:`.
Use these headers once in this order:
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:
{label_lock}{role_definition_lock}{frame_definition_lock}
RETENTION_LINE_PLAN:
{retention_lines}
Copy each RETENTION_LINE_PLAN prefix verbatim and in order; append one concise preservation description. Never alter its scope or marker or print strength names or `=`.
Every Subject listed as appearing in a shot must be visibly present and named in that shot's detailed_description.
detailed_description must contain exactly {shots}, once each in order; [Shot 1] has no header timestamp.{move_lock}{content_lock}
Complete every SHOT_PLAN verb and result visibly; do not stop at setup. Preserve physical state across shots; show a transition before a conflicting later action.
Speaker IDs, exact <d> content, lip synchronization, and event order must be correct in the final output.
Do not invent people, dialogue, vocal reactions, or music. Use N/A for unrequested non-diegetic music.
End after the non_diegetic_music value."""

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
Highest-priority format lock. Return plain text with no wrapper, JSON, Markdown, or commentary.
Begin exactly with: {opening}
Use exactly these fields once in order: integrated_multimodal_description, overall_soundscape, non_diegetic_music.
For I2VA, FL2VA, and L2VA, keep the alignment line before the main field, never inside it.
Never use REF2VA sections. The timeline must contain exactly {shots}, once each in order; [Shot 1] has no header timestamp.{move_lock}{endpoint_lock}{content_lock}
Preserve every explicit SHOT_PLAN action in order; omit none.
Complete every action verb and result visibly; do not stop at setup. Preserve physical state across shots; show a transition before a conflicting later action.
Speaker IDs, exact <d> content, lip synchronization, and event order must be correct in the final output.
Do not invent dialogue, vocal reactions, or music. overall_soundscape must not repeat or summarize speech. Use N/A for unrequested non-diegetic music.
End after the non_diegetic_music value."""


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


_CAMERA_SHOT_SCALE = {
    "extreme_close_up": 0, "detail_shot": 0, "insert_shot": 0,
    "close_up": 1, "medium_close_up": 2, "medium_shot": 3,
    "cowboy_shot": 4, "medium_wide_shot": 4, "medium_full_shot": 5,
    "full_shot": 6, "two_shot": 6, "three_shot": 6, "group_shot": 6,
    "wide_shot": 7, "establishing_shot": 7, "extreme_wide_shot": 8,
}
_ZOOM_CAMERA_MOTIONS = {"zoom_in", "zoom_out", "dolly_zoom_in", "dolly_zoom_out"}

# Decompose angle presets into spatial dimensions so camera continuity does not
# depend on special cases for particular preset pairs.
_CAMERA_ANGLE_STATE = {
    "none": ("unspecified", 0, "level", "external"),
    "eye_level": ("front", 0, "level", "external"),
    "low_angle": ("front", -1, "level", "external"),
    "high_angle": ("front", 1, "level", "external"),
    "overhead": ("front", 2, "level", "external"),
    "top_down": ("front", 3, "level", "external"),
    "birds_eye": ("front", 4, "level", "external"),
    "worms_eye": ("front", -3, "level", "external"),
    "ground_level": ("front", -2, "level", "external"),
    "aerial": ("front", 4, "level", "external"),
    "dutch_angle": ("front", 0, "rolled", "external"),
    "over_shoulder": ("rear_quarter", 0, "level", "over_shoulder"),
    "pov": ("front", 0, "level", "subjective"),
    "three_quarter": ("three_quarter", 0, "level", "external"),
    "profile": ("profile", 0, "level", "external"),
    "rear": ("rear", 0, "level", "external"),
}


def _camera_angle_motion_components(source_angle: str, target_angle: str) -> list[str]:
    """Return generic physical motion components between camera-angle states."""
    if source_angle == target_angle or target_angle == "none":
        return []
    source = _CAMERA_ANGLE_STATE.get(source_angle, _CAMERA_ANGLE_STATE["none"])
    target = _CAMERA_ANGLE_STATE.get(target_angle, _CAMERA_ANGLE_STATE["none"])
    components: list[str] = []
    if source[0] != target[0] and "unspecified" not in {source[0], target[0]}:
        components.append("arcing continuously around the subject toward the configured viewing side")
    if source[1] != target[1]:
        vertical = "raising" if target[1] > source[1] else "lowering"
        components.append(
            f"{vertical} its position and translating continuously as needed while coordinating its tilt"
        )
    if source[2] != target[2]:
        components.append("rolling smoothly around its optical axis toward the configured horizon")
    if source[3] != target[3]:
        components.append("translating continuously into the configured viewing relationship")
    if not components:
        components.append("travelling continuously into the configured camera position")
    return components


def _append_simultaneous_camera_components(base: str, components: list[str]) -> str:
    if not components:
        return base
    addition = (
        components[0] if len(components) == 1
        else ", ".join(components[:-1]) + ", and " + components[-1]
    )
    return f"{base}, while simultaneously {addition}"


def _shot_move_camera_specs(project: dict[str, Any], effective_seconds: float) -> list[dict[str, Any]]:
    """Compile deterministic camera endpoints and physical paths for every Move."""
    items = project.get("shots", [])
    requested_seconds = sum(float(item["duration"]) for item in items)
    scale = effective_seconds / requested_seconds if requested_seconds > 0 else 1.0
    cursor = 0.0
    shot_number = 0
    move_number = 0
    current_shot_size = "none"
    current_angle = "none"
    travel_direction: str | None = None
    specs: list[dict[str, Any]] = []
    for item_index, item in enumerate(items):
        start = cursor
        end = start + float(item["duration"]) * scale
        presets = _normalize_shot_presets(item.get("presets"))
        if not _is_move(item):
            shot_number += 1
            move_number = 0
            current_shot_size = presets["camera_shot"]
            current_angle = presets["camera_angle"]
            travel_direction = None
            cursor = end
            continue
        move_number += 1
        target_shot_size = presets["camera_shot"] if presets["camera_shot"] != "none" else current_shot_size
        target_angle = presets["camera_angle"] if presets["camera_angle"] != "none" else current_angle
        explicit_motion = presets["camera_motion"]
        shot_size_changed = target_shot_size != current_shot_size
        angle_changed = target_angle != current_angle
        camera_state_changed = shot_size_changed or angle_changed or explicit_motion != "none"
        previous_direction = travel_direction
        is_last_move = item_index == len(items) - 1 or not _is_move(items[item_index + 1])
        angle_motion = _camera_angle_motion_components(current_angle, target_angle)
        if explicit_motion != "none":
            motion_text = CAMERA_PRESET_PROMPTS["camera_motion"].get(explicit_motion, "continuous camera movement")
            physical = f"the same camera performs a {motion_text}"
            if explicit_motion in {"push_in", "zoom_in", "dolly_zoom_in"}:
                travel_direction = "forward"
            elif explicit_motion in {"pull_out", "zoom_out", "dolly_zoom_out"}:
                travel_direction = "backward"
        else:
            source_scale = _CAMERA_SHOT_SCALE.get(current_shot_size)
            target_scale = _CAMERA_SHOT_SCALE.get(target_shot_size)
            if source_scale is not None and target_scale is not None and target_scale > source_scale:
                travel_direction = "backward"
                physical = (
                    "the same camera continues dollying backward along the same physical path"
                    if previous_direction == "backward" else
                    "the same camera begins a smooth physical dolly backward along a continuous path"
                )
            elif source_scale is not None and target_scale is not None and target_scale < source_scale:
                travel_direction = "forward"
                physical = (
                    "the same camera continues dollying forward along the same physical path"
                    if previous_direction == "forward" else
                    "the same camera smoothly decelerates, reverses direction without a pause, and begins a controlled dolly forward along the same physical path"
                    if previous_direction == "backward" else
                    "the same camera begins a smooth physical dolly forward along a continuous path"
                )
            else:
                physical = (
                    "the same camera preserves its current framing distance"
                    if angle_motion else
                    "the uninterrupted image continues directly from the preceding frame; the camera position, "
                    "lens, orientation, framing, subject scale, and background perspective remain unchanged"
                )
        physical = _append_simultaneous_camera_components(physical, angle_motion)
        target_description = CAMERA_PRESET_PROMPTS["camera_shot"].get(target_shot_size, "")
        angle_description = CAMERA_PRESET_PROMPTS["camera_angle"].get(target_angle, "")
        endpoint_parts = []
        if target_description and (shot_size_changed or angle_changed):
            endpoint_verb = "reaching and settling into a stable" if is_last_move else "naturally reaching a"
            endpoint_parts.append(f"{endpoint_verb} {target_description} by {format_timestamp(end)}")
        if angle_description and target_angle != current_angle:
            article = "an" if angle_description[:1].lower() in "aeiou" else "a"
            endpoint_parts.append(f"progressively settling into {article} {angle_description}")
        endpoint = " and ".join(endpoint_parts)
        if endpoint:
            physical += ", " + endpoint
        elif not camera_state_changed:
            physical += (
                f" through {format_timestamp(end)}; only the requested subject action changes, unfolding "
                f"progressively across this interval and reaching its completed visible state by {format_timestamp(end)}"
            )
        elif is_last_move:
            physical += f", settling into a stable framing by {format_timestamp(end)}"
        else:
            physical += f", maintaining the established framing through {format_timestamp(end)}"
        if not camera_state_changed:
            physical += (
                ". The camera does not reframe to follow the action; the inherited composition and background "
                "geometry stay visually continuous."
            )
        elif explicit_motion in _ZOOM_CAMERA_MOTIONS:
            physical += (
                ", preserving the inherited compositional anchor, spatial axis, and continuous screen position."
            )
        elif is_last_move and previous_direction and travel_direction != previous_direction:
            physical += (
                ", coordinating camera height and tilt in one fluid motion only as needed to continuously "
                "track the subject and preserve natural background parallax."
            )
        else:
            physical += (
                ", keeping the inherited compositional anchor on a continuous screen path with continuous "
                "perspective and visible background parallax."
            )
        if camera_state_changed:
            boundary_bridge = (
                "the camera motion begins visibly from the exact preceding frame with no pose, framing, or "
                "viewpoint discontinuity, and "
                if move_number == 1 else
                "beginning from the fully reached camera and subject state of the preceding interval, "
            )
            physical = boundary_bridge + physical
        cue = (
            f"From {format_timestamp(start)} to {format_timestamp(end)}, without a cut,"
            if move_number == 1 else
            f"From {format_timestamp(start)} to {format_timestamp(end)}, continuing the same uninterrupted camera path,"
        )
        specs.append({
            "shot": shot_number, "move": move_number, "start": start, "end": end,
            "cue": cue, "camera_sentence": physical,
            "allows_zoom": explicit_motion in _ZOOM_CAMERA_MOTIONS,
            "camera_state_changed": camera_state_changed,
            "is_first_move": move_number == 1,
        })
        current_shot_size = target_shot_size
        current_angle = target_angle
        cursor = end
    return specs


def _shot_numbers_owning_moves(project: dict[str, Any]) -> set[int]:
    owners: set[int] = set()
    shot_number = 0
    for item in project.get("shots", []):
        if _is_move(item):
            if shot_number:
                owners.add(shot_number)
        else:
            shot_number += 1
    return owners


def _repair_move_transition_language(text: str) -> str:
    """Remove edit-like camera wording only inside a known Move interval."""
    text = re.sub(
        r"^\s*(?:During|Within)\s+this\s+(?:movement|interval),\s*"
        r"(?:maintaining|remaining|holding|executing|reframing)\b[^;]*;\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    if text and text[:1].islower():
        text = text[:1].upper() + text[1:]
    text = re.sub(
        r"\bthe\s+(?:same\s+)?camera\s+(?:cuts?|switches)\s+to\b",
        "the same camera continues physically toward",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:the\s+)?(?:camera|framing|view|composition)\s+transitions?\s+(?:into|to)\b",
        "the camera progressively moves into",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bto\s+transition\s+into\s+(?=an?\s+(?:extreme\s+)?(?:close-up|medium|cowboy|full|wide|establishing)\b)",
        "to progressively reach ",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _remove_leading_generated_camera_clause(text: str) -> str:
    """Drop a model-written Move camera clause while retaining its attached subject action."""
    stripped = text.lstrip()
    if not re.match(
        r"(?i)(?:(?:the|a)\s+)?(?:same\s+)?(?:camera|lens|dolly|framing|view|composition)\b",
        stripped,
    ):
        return stripped
    sentence_end = re.search(r"[.!?](?=\s|$)", stripped)
    if sentence_end:
        sentence = stripped[:sentence_end.start()]
        remainder = stripped[sentence_end.end():].lstrip()
    else:
        sentence = stripped
        remainder = ""
    connector = re.search(r"(?i)\b(as|while)\s+(.+)$", sentence)
    if connector:
        action = connector.group(2).strip()
        if connector.group(1).lower() == "while":
            action = "During this movement, " + action
        elif action:
            action = action[0].upper() + action[1:]
        remainder = action + "." + (" " + remainder if remainder else "")
    return remainder


def _move_take_contract(allows_zoom: bool, start: float, end: float,
                        camera_moves: bool = True) -> str:
    scope = f"From {format_timestamp(start)} to {format_timestamp(end)}, "
    if not camera_moves:
        return scope + (
            "one locked camera holds a continuous take; only the requested subject action changes."
        )
    if not allows_zoom:
        return scope + (
            "one stabilized camera and one consistent lens maintain a single uninterrupted physical path, "
            "continuous perspective, and natural background parallax."
        )
    return scope + (
        "one stabilized camera maintains a single uninterrupted physical path; focal length changes only in an "
        "explicit optical-zoom or dolly-zoom beat."
    )


def _enforce_move_camera_continuity(prompt: str, project: dict[str, Any],
                                    effective_seconds: float) -> str:
    """Deterministically preserve Shot-scoped continuous camera takes in final H3 prose."""
    owners = _shot_numbers_owning_moves(project)
    if not owners:
        return prompt
    mode = project.get("mode", "T2VA")
    field_name = "detailed_description" if mode == "REF2VA" else "integrated_multimodal_description"
    field_pattern = re.compile(
        rf"(^[ \t]*{field_name}[ \t]*:[ \t]*)(.*?)(?=^[ \t]*(?:overall_soundscape|non_diegetic_music)\s*:)",
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    field_match = field_pattern.search(prompt)
    if not field_match:
        return prompt
    main = field_match.group(2).strip()

    # Replace only each Move's leading camera clause with a deterministic path;
    # retain the model-written subject action and all intentional Shot cuts.
    specs = _shot_move_camera_specs(project, effective_seconds)
    anchor_counts: dict[int, int] = {}
    for anchor in _frame_anchor_schedule(project, effective_seconds):
        number = int(anchor.get("shot", 1))
        anchor_counts[number] = anchor_counts.get(number, 0) + 1
    frame_driven_shots = {number for number, count in anchor_counts.items() if count >= 2}
    cue_positions = []
    search_from = 0
    for spec in specs:
        # Multiple Picture anchors define the chronology of this Shot. Its Moves are
        # internal action cues, not independent endpoint paragraphs; rewriting every
        # Move as a completed camera state encourages H3 to synthesize a hidden cut.
        if spec["shot"] in frame_driven_shots:
            continue
        position = main.find(spec["cue"], search_from)
        if position >= 0:
            cue_positions.append((position, spec))
            search_from = position + len(spec["cue"])
    for index in range(len(cue_positions) - 1, -1, -1):
        position, spec = cue_positions[index]
        cue = spec["cue"]
        boundary = cue_positions[index + 1][0] if index + 1 < len(cue_positions) else len(main)
        next_shot = re.search(r"\[Shot\s+\d+\]", main[position + len(cue):boundary], re.IGNORECASE)
        if next_shot:
            boundary = position + len(cue) + next_shot.start()
        model_tail = main[position + len(cue):boundary]
        preserved_action = _repair_move_transition_language(
            _remove_leading_generated_camera_clause(model_tail)
        )
        replacement = cue + " " + spec["camera_sentence"]
        if preserved_action:
            replacement += " " + preserved_action
        suffix = main[boundary:]
        separator = " " if suffix else ""
        main = main[:position] + replacement.rstrip() + separator + suffix.lstrip()

    # Scope the take contract and opening hold to each Shot that owns Moves. A later Shot header
    # still represents an intentional cut and starts an independent contract.
    specs_by_shot = {
        number: [spec for spec in specs if spec["shot"] == number]
        for number in owners
    }
    shot_starts: dict[int, float] = {}
    shot_opening_presets: dict[int, dict[str, str]] = {}
    cursor = 0.0
    requested_seconds = sum(float(item["duration"]) for item in project.get("shots", []))
    scale = effective_seconds / requested_seconds if requested_seconds > 0 else 1.0
    shot_counter = 0
    for item in project.get("shots", []):
        if not _is_move(item):
            shot_counter += 1
            shot_starts[shot_counter] = cursor
            shot_opening_presets[shot_counter] = _normalize_shot_presets(item.get("presets"))
        cursor += float(item["duration"]) * scale
    for number in sorted(owners, reverse=True):
        header = re.compile(
            rf"(\[Shot\s+{number}\](?:\s+At\s+\d{{2}}:\d{{2}}\.\d{{3}},)?)",
            flags=re.IGNORECASE,
        )
        match = header.search(main)
        if not match:
            continue
        shot_specs = specs_by_shot[number]
        opening_presets = shot_opening_presets.get(number, _normalize_shot_presets(None))
        contract = _move_take_contract(
            opening_presets["camera_motion"] in _ZOOM_CAMERA_MOTIONS
            or any(spec["allows_zoom"] for spec in shot_specs),
            shot_starts[number], shot_specs[-1]["end"],
            camera_moves=(
                opening_presets["camera_motion"] not in {"none", "static"}
                or any(spec["camera_state_changed"] for spec in shot_specs)
            ),
        )
        opening_description = CAMERA_PRESET_PROMPTS["camera_shot"].get(
            opening_presets["camera_shot"], "opening framing",
        ) or "opening framing"
        opening_motion = opening_presets["camera_motion"]
        if opening_motion not in {"none", "static"}:
            motion_description = CAMERA_PRESET_PROMPTS["camera_motion"].get(
                opening_motion, "continuous camera movement",
            )
            opening = (
                f"From {format_timestamp(shot_starts[number])} to {format_timestamp(shot_specs[0]['start'])}, "
                f"the same camera executes the configured motion: {motion_description}, as one continuous physical "
                f"movement while maintaining the {opening_description}, the inherited compositional anchor, and "
                "continuous background geometry."
            )
        else:
            opening = (
                f"From {format_timestamp(shot_starts[number])} to {format_timestamp(shot_specs[0]['start'])}, "
                f"the same camera holds the {opening_description} with subtle stabilized drift."
            )
        following = main[match.end():match.end() + len(contract) + len(opening) + 500]
        additions = []
        if contract.lower() not in following.lower():
            additions.append(contract)
        if opening.lower() not in following.lower():
            additions.append(opening)
        if additions:
            # Keep a configured later-Shot cut sentence intact. Insert after
            # the opening sentence, before the first Move begins.
            sentence_end = re.search(r"[.!?](?=\s|$)", main[match.end():])
            insert_at = match.end() + sentence_end.end() if sentence_end else match.end()
            main = main[:insert_at] + " " + " ".join(additions) + main[insert_at:]

    rebuilt = field_match.group(1) + main + "\n\n"
    return prompt[:field_match.start()] + rebuilt + prompt[field_match.end():]


def _enforce_ref_frame_anchor_timing(prompt: str, project: dict[str, Any],
                                     effective_seconds: float) -> str:
    """Ensure every REF2VA frame label carries its exact in-shot time in final prose."""
    if project.get("mode") != "REF2VA":
        return prompt
    anchors = _frame_anchor_schedule(project, effective_seconds)
    if not anchors:
        return prompt
    field_pattern = re.compile(
        r"(^[ \t]*detailed_description[ \t]*:[ \t]*)(.*?)"
        r"(?=^[ \t]*(?:overall_soundscape|non_diegetic_music)\s*:)",
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    field_match = field_pattern.search(prompt)
    if not field_match:
        return prompt
    main = field_match.group(2).strip()
    search_from = 0
    for anchor in anchors:
        label = anchor["label"]
        exact = _frame_anchor_sentence(anchor)
        legacy_pattern = re.compile(
            rf"At\s+{re.escape(format_timestamp(anchor['time']))},\s*the\s+same\s+uninterrupted\s+take\s+"
            rf"exactly\s+reaches\s+{re.escape(label)}\.",
            flags=re.IGNORECASE,
        )
        legacy_match = legacy_pattern.search(main, search_from)
        if legacy_match:
            main = main[:legacy_match.start()] + exact + main[legacy_match.end():]
        existing = main.find(exact, search_from)
        if existing >= 0:
            search_from = existing + len(exact)
            continue
        state_pattern = re.compile(
            rf"At\s+(?:the\s+)?exact\s+state\s+of\s+{re.escape(label)}\s*,?",
            flags=re.IGNORECASE,
        )
        state_match = state_pattern.search(main, search_from)
        if state_match:
            replacement = exact + " Continuing from that exact anchored state,"
            main = main[:state_match.start()] + replacement + main[state_match.end():]
            search_from = state_match.start() + len(replacement)
            continue
        label_position = main.find(label, search_from)
        if label_position < 0:
            continue
        sentence_start = max(
            main.rfind(". ", search_from, label_position),
            main.rfind("! ", search_from, label_position),
            main.rfind("? ", search_from, label_position),
            main.rfind("\n", search_from, label_position),
        )
        sentence_start = search_from if sentence_start < 0 else sentence_start + 2
        main = main[:sentence_start] + exact + " " + main[sentence_start:]
        search_from = sentence_start + len(exact)

    # Keep consecutive anchors in one Shot explicitly joined by one active bridge.
    # Insert it immediately before its destination anchor so it cannot look like a
    # new Move paragraph or a camera reset.
    anchors_by_shot: dict[int, list[dict[str, Any]]] = {}
    for anchor in anchors:
        anchors_by_shot.setdefault(int(anchor.get("shot", 1)), []).append(anchor)
    for shot_anchors in anchors_by_shot.values():
        shot_anchors.sort(key=lambda item: (item["frame_index"], item["time"]))
        for start_anchor, end_anchor in zip(shot_anchors, shot_anchors[1:]):
            bridge = (
                f"From {format_timestamp(start_anchor['time'])} to {format_timestamp(end_anchor['time'])}, "
                f"the same uninterrupted take develops continuously from {start_anchor['label']} toward "
                f"{end_anchor['label']}."
            )
            if bridge.casefold() in main.casefold():
                continue
            destination = _frame_anchor_sentence(end_anchor)
            destination_at = main.find(destination)
            if destination_at >= 0:
                main = main[:destination_at] + bridge + " " + main[destination_at:]
    rebuilt = field_match.group(1) + main + "\n\n"
    return prompt[:field_match.start()] + rebuilt + prompt[field_match.end():]


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


def _enforce_retention_line_plan(prompt: str, label_plan: dict[str, dict[str, Any]]) -> str:
    """Repair only REF2VA retention prefixes; preserve model-written descriptions."""
    section_match = re.search(
        r"(^[ \t]*retention_analysis[ \t]*:[ \t]*\n?)(.*?)(?=^[ \t]*detailed_description[ \t]*:)",
        prompt,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if not section_match or not label_plan:
        return prompt

    body = section_match.group(2)
    source_lines = body.splitlines()
    rebuilt: list[str] = []
    fixed_markers = (
        "fully_preserved|partially_preserved|attribute_transfer|weak_reference|"
        "fully_copy|partially_copy|reference"
    )
    if _allows_frame_continuity_subjects(label_plan):
        inferred_subject_lines = []
        for line in source_lines:
            if not re.match(r"\s*<Subject\s+\d+>", line, flags=re.IGNORECASE):
                continue
            if re.search(rf"\b(?:{fixed_markers})\b", line, flags=re.IGNORECASE):
                inferred_subject_lines.append(line.strip())
        rebuilt.extend(inferred_subject_lines)
    for label, plan in label_plan.items():
        matching_line = next(
            (line for line in source_lines if re.match(
                rf"\s*{re.escape(label)}(?=\s|\(|:)", line, re.IGNORECASE,
            )),
            "",
        )
        description = ""
        if matching_line:
            marker_match = re.search(
                rf"\b(?:{fixed_markers})\b\s*(?:[-;,:]\s*)?(.*)$",
                matching_line,
                flags=re.IGNORECASE,
            )
            if marker_match:
                description = marker_match.group(1).strip()
        if not description:
            description = str(plan.get("contract") or "preserve only the defined reference relationship").strip()
        rebuilt.append(f"{plan['retention_prefix']} {description}")

    replacement = section_match.group(1) + "\n".join(rebuilt) + "\n\n"
    return prompt[:section_match.start()] + replacement + prompt[section_match.end():]


def _enforce_reference_definition_provenance(
    prompt: str, reference_model: dict[str, Any] | None,
) -> str:
    """Replace impossible Picture/Video/Audio self-derived definitions with locked definitions."""
    if not reference_model:
        return prompt
    expected_by_label: dict[str, str] = {}
    for definition in reference_model.get("definitions", []):
        match = re.match(r"\s*(<(?:Picture|Video|Audio)\s+\d+>)", definition, flags=re.IGNORECASE)
        if match:
            expected_by_label[match.group(1).casefold()] = definition.strip()
    section_match = re.search(
        r"(^[ \t]*subject_definitions[ \t]*:[ \t]*\n?)(.*?)(?=^[ \t]*summary[ \t]*:)",
        prompt,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if not section_match or not expected_by_label:
        return prompt
    lines = section_match.group(2).splitlines()
    for index, line in enumerate(lines):
        label_match = re.match(r"\s*(<(?:Picture|Video|Audio)\s+\d+>)", line, flags=re.IGNORECASE)
        if not label_match:
            continue
        label = label_match.group(1)
        if re.search(rf"\bderived\s+from\s+{re.escape(label)}", line, flags=re.IGNORECASE):
            expected = expected_by_label.get(label.casefold())
            if expected:
                lines[index] = expected
    replacement = section_match.group(1) + "\n".join(lines).rstrip() + "\n\n"
    return prompt[:section_match.start()] + replacement + prompt[section_match.end():]


def _enforce_framing_body_range(prompt: str) -> str:
    """Repair directly contradictory shot-size labels within one sentence."""
    sentence_pattern = re.compile(r"[^.!?\n]+(?:[.!?]|$)")

    def repair(match: re.Match[str]) -> str:
        sentence = match.group(0)
        lower = sentence.casefold()
        if (
            any(phrase in lower for phrase in (
                "entire body visible", "full body visible", "head to toe", "head-to-toe",
            ))
            and re.search(r"\bmedium shot\b", sentence, flags=re.IGNORECASE)
        ):
            return re.sub(r"\bmedium shot\b", "full shot", sentence, flags=re.IGNORECASE)
        return sentence

    return sentence_pattern.sub(repair, prompt)


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


def _allows_frame_continuity_subjects(label_plan: dict[str, dict[str, Any]]) -> bool:
    """Allow Qwen to bind recurring entities when every locked visual asset is a frame anchor."""
    plans = list(label_plan.values())
    return len(plans) >= 2 and all(
        plan.get("kind") == "Picture" and plan.get("role") == "frame"
        for plan in plans
    )


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
    inferred_subject_labels: list[str] = []
    if _allows_frame_continuity_subjects(label_plan):
        inferred_subject_labels = [label for label in definition_labels if label.startswith("<Subject ")]
        expected_subject_labels = [f"<Subject {index}>" for index in range(1, len(inferred_subject_labels) + 1)]
        if inferred_subject_labels != expected_subject_labels:
            issues.append("Recurring frame-continuity Subjects must be numbered sequentially from <Subject 1>.")
        if definition_labels != inferred_subject_labels + expected_labels:
            issues.append(
                "Define inferred recurring Subjects first, followed by exactly these locked Picture labels: "
                + ", ".join(expected_labels) + "."
            )
    elif definition_labels != expected_labels:
        issues.append(
            f"Define exactly these reference labels once and in order: {', '.join(expected_labels) or 'none'}."
        )
    output_labels = inferred_subject_labels + expected_labels
    definition_lines = {
        _canonical_ref_label(match): line
        for line in sections["subject_definitions"].splitlines()
        if (match := re.match(r"\s*<(Subject|Picture|Video|Audio)\s+(\d+)>", line, flags=re.IGNORECASE))
    }
    for label in inferred_subject_labels:
        sources = {
            _canonical_ref_label(match)
            for match in _REF_LABEL_PATTERN.finditer(definition_lines.get(label, ""))
            if match.group(1).casefold() == "picture"
        }
        if len(sources) < 2:
            issues.append(
                f"{label} is a recurring frame-continuity Subject and must cite at least two supporting Pictures."
            )
    for label, plan in label_plan.items():
        if plan["kind"] != "Subject" or label not in definition_lines:
            continue
        source = plan["source"]
        if source.casefold() not in definition_lines[label].casefold():
            issues.append(f"{label} must cite its source asset {source} in subject_definitions.")
    for label, plan in label_plan.items():
        if plan.get("kind") != "Picture" or plan.get("role") != "frame" or label not in definition_lines:
            continue
        definition = definition_lines[label]
        self_derived = re.search(
            rf"{re.escape(label)}\s+(?:is\s+)?derived\s+from\s+{re.escape(label)}",
            definition,
            flags=re.IGNORECASE,
        )
        if self_derived or not re.search(r"\b(?:exact\s+)?target\s+frame\b", definition, flags=re.IGNORECASE):
            issues.append(
                f"{label} must define its exact target-frame role and must not be described as derived from itself."
            )
    for label, plan in label_plan.items():
        if plan.get("kind") != "Picture" or plan.get("role") != "storyboard" or label not in definition_lines:
            continue
        definition = definition_lines[label]
        expected_shots = plan.get("applicable_shots") or [1]
        if not re.search(r"\bstoryboard\s+reference\b", definition, flags=re.IGNORECASE):
            issues.append(f"{label} must be defined as a storyboard reference, not as a frame or Subject source.")
        for shot_number in expected_shots:
            if f"[shot {shot_number}]" not in definition.casefold():
                issues.append(f"{label} storyboard definition must name [Shot {shot_number}].")

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
    if retention_labels != output_labels:
        issues.append("retention_analysis must contain exactly one ordered entry for every defined label.")
    for label, plan in label_plan.items():
        if label in retention_markers and retention_markers[label] != plan["marker"]:
            issues.append(f"{label} must use retention marker {plan['marker']} for its defined role.")
    for label in inferred_subject_labels:
        if retention_markers.get(label) != "fully_preserved":
            issues.append(f"{label} must use fully_preserved as a recurring exact-frame continuity Subject.")
    if re.search(r"\(S\d+\)", sections["retention_analysis"], flags=re.IGNORECASE):
        issues.append("Do not place speaker IDs in retention_analysis.")

    summary_labels = {
        _canonical_ref_label(match) for match in _REF_LABEL_PATTERN.finditer(sections["summary"])
    }
    missing_summary = [label for label in output_labels if label not in summary_labels]
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
    unexpected = sorted(downstream_labels.difference(output_labels))
    if unexpected:
        issues.append("Remove undefined or source-only labels outside subject_definitions: " + ", ".join(unexpected) + ".")
    visual_labels = inferred_subject_labels + [
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
    detail = sections["detailed_description"]
    shot_headers = list(_SHOT_HEADER_PATTERN.finditer(detail))
    shot_blocks: dict[int, str] = {}
    for index, header in enumerate(shot_headers):
        shot_number = int(header.group(1) or header.group(2))
        block_end = shot_headers[index + 1].start() if index + 1 < len(shot_headers) else len(detail)
        shot_blocks[shot_number] = detail[header.start():block_end]
    for label, plan in label_plan.items():
        if plan["kind"] != "Subject":
            continue
        missing_shots = [
            number for number in plan.get("applicable_shots", [])
            if label.casefold() not in shot_blocks.get(number, "").casefold()
        ]
        if missing_shots:
            shots = ", ".join(f"[Shot {number}]" for number in missing_shots)
            issues.append(f"{label} is declared visible in {shots}; name it visibly in each corresponding shot.")
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
        if plan["marker"] != "weak_reference" or plan.get("role") == "storyboard":
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
    scheduled_anchors = _frame_anchor_schedule(
        project, align_frame_count(project.get("requested_duration", 5.0)) / MODEL_FPS,
    )
    for anchor in scheduled_anchors:
        required = _frame_anchor_sentence(anchor)
        if required.casefold() not in detail_lower:
            issues.append(
                f"Place {anchor['label']} at its exact assigned time with this literal in-shot sentence: {required}"
            )
    scheduled_by_shot: dict[int, list[dict[str, Any]]] = {}
    for anchor in scheduled_anchors:
        scheduled_by_shot.setdefault(int(anchor.get("shot", 1)), []).append(anchor)
    for shot_anchors in scheduled_by_shot.values():
        shot_anchors.sort(key=lambda item: (item["frame_index"], item["time"]))
        for start_anchor, end_anchor in zip(shot_anchors, shot_anchors[1:]):
            bridge = (
                f"From {format_timestamp(start_anchor['time'])} to {format_timestamp(end_anchor['time'])}, "
                f"the same uninterrupted take develops continuously from {start_anchor['label']} toward "
                f"{end_anchor['label']}."
            )
            if bridge.casefold() not in detail_lower:
                issues.append(f"Join consecutive same-Shot frame anchors with this literal bridge: {bridge}")
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
        frame_sequences: dict[int, list[tuple[int, str]]] = {}
        max_frame = max(0, align_frame_count(project["requested_duration"]) - 1)
        for ref in references:
            if ref.get("type") != "picture" or ref.get("role") not in {
                "first_frame", "last_frame", "frame",
            }:
                continue
            if ref["role"] == "first_frame":
                frame_index = 0
            elif ref["role"] == "last_frame":
                frame_index = max_frame
            else:
                frame_index = min(max(0, int(ref.get("frame_index", 0))), max_frame)
            shot_number = _reference_applicable_shots(project, ref)[0]
            frame_sequences.setdefault(shot_number, []).append((frame_index, ref["label"]))
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
                        max_frame,
                    )
                    shot_number = plan.get("applicable_shots", [1])[0]
                    lines.extend((
                        "anchor: exact whole frame at the assigned timeline position",
                        f"anchor_role: {plan.get('anchor_kind', 'intermediate')} state within its owning Shot",
                        f"anchor_frame_index: {frame_index}",
                        f"anchor_time_seconds: {frame_index / MODEL_FPS:.3f}",
                        f"required_definition: {label} is the exact target frame at output frame {frame_index} in [Shot {shot_number}]",
                        "anchor_contract: reach this complete image state at this frame through continuous in-shot motion; this anchor never creates a cut or transition; never dissolve, morph, reset, or stop merely to reach it",
                    ))
                elif ref.get("role") == "storyboard":
                    applicable = plan.get("applicable_shots", [1])
                    shot_text = ", ".join(f"[Shot {number}]" for number in applicable)
                    definition_shots = " and ".join(f"[Shot {number}]" for number in applicable)
                    lines.extend((
                        f"applies_to: {shot_text}",
                        "planning_scope: panel order, shot order, viewpoint, approximate framing, subject placement, and explicitly depicted action beats only",
                        "excluded_scope: exact frame matching, exact timing, subject identity, clothing, visual style, lighting, palette, and pose locking",
                        "panel_boundary_contract: within each applicable configured Shot, consecutive panels are chronological action beats inside one uninterrupted take and never create cuts, transitions, viewpoint jumps, or camera resets; only a configured later Shot boundary may cut",
                        "framing_serialization_contract: detailed_description must preserve every distinct panel viewpoint, approximate shot size, screen direction, subject placement, and action beat in order, converting changes into physically continuous camera travel; merge only adjacent materially identical panels and never collapse the sequence into generic tracking language",
                        f"required_definition: {label} is a storyboard reference for {definition_shots}, defining their viewpoint, subject placement, approximate framing, explicitly depicted action beats, and shot order.",
                        "priority: explicit target Shot text, camera presets, and concrete frame anchors override storyboard planning",
                    ))
            elif ref.get("type") == "video":
                lines.append(f"temporal_visual_evidence: {evidence_for(source)}")
            if ref.get("duration"):
                displayed_duration = float(ref["duration"])
                displayed_trim_start = float(ref.get("trim_start", 0.0))
                displayed_timeline_start = float(ref.get("timeline_start", 0.0))
                if ref.get("type") == "video":
                    displayed_trim_start, displayed_duration, displayed_timeline_start = (
                        _visible_video_selection(ref, effective_seconds)
                    )
                duration_key = (
                    "selected_source_duration_seconds"
                    if ref.get("type") == "video"
                    else "source_duration_seconds"
                )
                lines.append(f"{duration_key}: {displayed_duration:.2f}")
                if ref.get("type") == "video":
                    lines.append(f"source_trim_start_seconds: {displayed_trim_start:.2f}")
                    lines.append(f"target_timeline_start_seconds: {displayed_timeline_start:.2f}")
            blocks.append("\n".join(lines))
        sequence_blocks = []
        for shot_number, anchors in sorted(frame_sequences.items()):
            if len(anchors) < 2:
                continue
            sequence = " -> ".join(
                f"{label}@frame {frame_index}" for frame_index, label in sorted(anchors)
            )
            sequence_blocks.append(f"[Shot {shot_number}]: {sequence}")
        sequence_plan = ""
        if sequence_blocks:
            sequence_plan = (
                "\n\nFRAME_ANCHOR_SEQUENCES:\n" + "\n".join(sequence_blocks)
                + "\ncontract: Each line is one uninterrupted take. Write chronological From-to intervals "
                  "between consecutive anchors and show the shortest observable subject, object, environment, "
                  "and camera development. Begin exactly from an anchor assigned to the Shot opening; describe "
                  "each intermediate Picture as a precise state that the ongoing motion naturally passes through, "
                  "not as a destination, transition, new composition, or scene replacement; reach an anchor "
                  "assigned to the Shot ending as the exact final state. Preserve one camera, lens, spatial axis, "
                  "perspective, and evolving background. Continue from each anchored state without pausing or "
                  "reintroducing its visual inventory. Only a configured later Shot may cut."
            )
        return (
            "REFERENCE_PLAN:\n"
            f"task_types: {' + '.join(model['task_types'])}\n\n"
            + "\n\n".join(blocks)
            + sequence_plan
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
            displayed_duration = float(ref["duration"])
            displayed_trim_start = float(ref.get("trim_start", 0.0))
            displayed_timeline_start = float(ref.get("timeline_start", 0.0))
            if ref["type"] == "video":
                displayed_trim_start, displayed_duration, displayed_timeline_start = (
                    _visible_video_selection(ref, effective_seconds)
                )
            duration_key = (
                "selected_source_duration_seconds"
                if ref["type"] == "video"
                else "source_duration_seconds"
            )
            lines.append(f"{duration_key}: {displayed_duration:.2f}")
            if ref["type"] == "video":
                lines.append(f"source_trim_start_seconds: {displayed_trim_start:.2f}")
                lines.append(f"target_timeline_start_seconds: {displayed_timeline_start:.2f}")
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


def _frame_anchor_schedule(project: dict[str, Any], effective_seconds: float) -> list[dict[str, Any]]:
    """Return exact intermediate-frame anchors with their owning Shot in chronological order."""
    max_frame = max(0, int(round(effective_seconds * MODEL_FPS)) - 1)
    schedule: list[dict[str, Any]] = []
    for ref in _reference_labels(project.get("references", [])):
        if ref.get("type") != "picture" or ref.get("role") != "frame":
            continue
        frame_index = min(max(0, int(ref.get("frame_index", 0))), max_frame)
        applicable = _reference_applicable_shots(project, ref)
        schedule.append({
            "label": ref["label"],
            "frame_index": frame_index,
            "time": frame_index / MODEL_FPS,
            "shot": applicable[0] if applicable else 1,
        })
    schedule.sort(key=lambda anchor: (anchor["time"], anchor["label"]))
    take_frames = {
        take["shot_number"]: (
            max(0, int(round(take["start"] * MODEL_FPS))),
            max(0, int(round(take["end"] * MODEL_FPS)) - 1),
        )
        for take in _compile_timeline_takes(project, effective_seconds)
    }
    by_shot: dict[int, list[dict[str, Any]]] = {}
    for anchor in schedule:
        by_shot.setdefault(anchor["shot"], []).append(anchor)
    for shot_anchors in by_shot.values():
        for index, anchor in enumerate(shot_anchors):
            shot_start_frame, shot_end_frame = take_frames.get(anchor["shot"], (0, max_frame))
            if anchor["frame_index"] == shot_start_frame:
                anchor["anchor_kind"] = "opening"
            elif anchor["frame_index"] == shot_end_frame:
                anchor["anchor_kind"] = "final"
            else:
                anchor["anchor_kind"] = "intermediate"
            anchor["sequence_index"] = index
            anchor["sequence_count"] = len(shot_anchors)
    return schedule


def _frame_anchor_sentence(anchor: dict[str, Any]) -> str:
    """Describe an exact guide frame without making an intermediate anchor sound like a cut."""
    timestamp = format_timestamp(anchor["time"])
    label = anchor["label"]
    kind = anchor.get("anchor_kind", "intermediate")
    if kind == "opening":
        return f"At {timestamp}, the shot begins exactly from {label}."
    if kind == "final":
        return f"At {timestamp}, the same uninterrupted take reaches the exact final-frame state of {label}."
    return (
        f"At {timestamp}, the ongoing uninterrupted motion passes precisely through {label} "
        "without a cut or camera reset."
    )


def _anchor_phase_lines(anchors: list[dict[str, Any]], start: float, end: float) -> list[str]:
    """Compile exact anchors inside one UI item into deterministic non-cut subphases."""
    if not anchors:
        return []
    lines = [
        "internal_anchor_phase_plan: these are subdivisions of this same timeline item, never Shots or cuts",
    ]
    phase_start = start
    inherited_label = "the preceding continuous scene state"
    phase_number = 1
    for anchor in anchors:
        anchor_time = min(max(anchor["time"], start), end)
        if anchor_time > phase_start + 1e-6:
            lines.append(
                f"phase_{phase_number}: From {phase_start:.3f} to {anchor_time:.3f} seconds, inherit "
                f"{inherited_label} and show the shortest physically continuous action and camera development "
                f"that naturally passes through {anchor['label']} at its exact assigned frame"
            )
            phase_number += 1
        lines.append(f"required_anchor_sentence: {_frame_anchor_sentence(anchor)}")
        inherited_label = f"the exact {anchor['label']} camera, subject, object, contact, and background state"
        phase_start = anchor_time
    if end > phase_start + 1e-6:
        lines.append(
            f"phase_{phase_number}: From {phase_start:.3f} to {end:.3f} seconds, inherit {inherited_label} "
            "and continue the remaining requested action without resetting or replacing the image"
        )
    return lines


def _frame_continuity_plan(project: dict[str, Any], effective_seconds: float,
                           aliases: dict[str, str]) -> str:
    """Compile frame-to-frame prose intervals while keeping Moves as in-shot events."""
    anchors = _frame_anchor_schedule(project, effective_seconds)
    if not anchors:
        return ""
    lines = [
        "FRAME_CONTINUITY_PLAN:",
        "role_separation: Shot starts a new camera take; Move is a timed event inside that take; Picture is an exact visual state the take passes through at one output frame",
        "prose_structure: organize detailed_description primarily as chronological frame-to-frame From-to bridges; weave overlapping Move actions and camera instructions into those bridges without turning a Move boundary into a paragraph reset, completed scene, or cut",
        "camera_contract: declare the Shot's single camera/lens/path once, then mention camera behavior again only when an overlapping Move physically changes it",
        "identity_contract: infer the smallest useful set of recurring Subjects visible across two or more Picture anchors and use those Subject labels to bind identity, persistent objects, and the shared environment across the full take",
    ]
    for take in _compile_timeline_takes(project, effective_seconds):
        shot_number = take["shot_number"]
        shot_anchors = [anchor for anchor in anchors if anchor["shot"] == shot_number]
        if not shot_anchors:
            continue
        events = [{
            "name": f"Shot {shot_number} opening",
            "start": take["start"],
            "end": take["opening_end"],
            "action": _replace_aliases(take["opening"].get("visual_action", ""), aliases),
        }]
        for beat in take["beats"]:
            events.append({
                "name": f"Move {beat['move_number']}",
                "start": beat["start"],
                "end": beat["end"],
                "action": _replace_aliases(beat["item"].get("visual_action", ""), aliases),
            })
        points: list[tuple[float, str]] = [(take["start"], "the inherited Shot opening state")]
        for anchor in shot_anchors:
            points.append((anchor["time"], anchor["label"]))
        if not any(anchor.get("anchor_kind") == "final" for anchor in shot_anchors):
            points.append((take["end"], "the requested Shot end state"))
        deduped: list[tuple[float, str]] = []
        for point in sorted(points, key=lambda value: value[0]):
            if deduped and abs(deduped[-1][0] - point[0]) < 1e-6:
                if point[1].startswith("<Picture"):
                    deduped[-1] = point
                continue
            deduped.append(point)
        lines.append(f"[Shot {shot_number}] one uninterrupted take")
        bridge_number = 1
        for (start, start_state), (end, end_state) in zip(deduped, deduped[1:]):
            if end <= start + 1e-6:
                continue
            active = []
            for event in events:
                if event["end"] <= start + 1e-6 or event["start"] >= end - 1e-6:
                    continue
                action = f": {event['action']}" if event["action"] else ""
                active.append(
                    f"{event['name']}@{event['start']:.3f}-{event['end']:.3f}{action}"
                )
            lines.append(
                f"bridge_{bridge_number}: From {start:.3f} to {end:.3f} seconds; continue from "
                f"{start_state}; show the shortest observable continuous state change toward {end_state}; "
                f"overlapping_events: {' | '.join(active) or 'none'}"
            )
            bridge_number += 1
        lines.append(
            "serialization: write these bridges as a flowing chronological take; an intermediate Picture is "
            "passed through without settling, while only an anchor actually assigned to the Shot end is the final state"
        )
    return "\n".join(lines)


def _qwen_shot_plan(project: dict[str, Any], effective_seconds: float,
                    aliases: dict[str, str]) -> str:
    requested_seconds = sum(float(shot["duration"]) for shot in project["shots"])
    scale = effective_seconds / requested_seconds if requested_seconds > 0 else 1.0
    cursor = 0.0
    blocks: list[str] = []
    shot_number = 0
    move_number = 0
    has_moves = any(_is_move(item) for item in project["shots"])
    move_cues = iter(_move_output_cues(project, effective_seconds))
    frame_counts: dict[int, int] = {}
    for anchor in _frame_anchor_schedule(project, effective_seconds):
        number = int(anchor.get("shot", 1))
        frame_counts[number] = frame_counts.get(number, 0) + 1
    frame_driven_shots = {number for number, count in frame_counts.items() if count >= 2}
    for item_index, shot in enumerate(project["shots"]):
        is_move = _is_move(shot)
        if is_move:
            move_number += 1
        else:
            shot_number += 1
            move_number = 0
        shot_seconds = float(shot["duration"]) * scale
        end = cursor + shot_seconds
        label = f"[Move {move_number} within Shot {shot_number}]" if is_move else f"[Shot {shot_number}]"
        lines = [label, f"time_range_seconds: {cursor:.3f}-{end:.3f}"]
        if is_move:
            move_cue = next(move_cues)
            lines.append("type: continuous in-shot beat; never a new shot or cut")
            if shot_number in frame_driven_shots:
                lines.extend((
                    f"internal_timing: {cursor:.3f}-{end:.3f} seconds",
                    "serialization: weave this action into the active FRAME_CONTINUITY bridge; do not open a "
                    "new paragraph, restate the camera contract, settle the image, or declare a completed state "
                    "at this Move boundary",
                    "continuity: inherit the exact preceding physical state; this Move is only an internal action "
                    "phase on the ongoing path toward the next Picture anchor",
                ))
            else:
                lines.append(
                    "continuity: inherit the exact preceding physical state and progressively reach this Move's "
                    "requested action/camera state by the range end; do not restart the scene"
                )
                lines.append(f"required_output_cue: {move_cue}")
        elif shot_number > 1:
            lines.append(f"required_output_header: [Shot {shot_number}] At {format_timestamp(cursor)},")
        action = _replace_aliases(shot["visual_action"], aliases)
        if action:
            lines.append(f"visual_action: {action}")
            if not is_move:
                lines.append("action_contract: preserve every explicit action above in the same order; omit none")
                lines.append(
                    "semantic_lock: translate faithfully; preserve every explicitly named actor, body part, "
                    "object, quantity, direction, simultaneity, and physical action verb"
                )
                lines.append(
                    "motion_semantics_contract: preserve the source verb at its original specificity; preserve continuous or repeated motion as ongoing phases, "
                    "not a single contact, broad pose, or static hold. Preserve stated actors, limb or hand count, contact target, direction, and simultaneity"
                )
        if project["mode"] == "FL2VA" and item_index == len(project["shots"]) - 1:
            lines.extend((
                "opening_state: continue the incomplete transition from the preceding shot; do not reveal the completed Picture 2",
                "entity_continuity: an entering entity that matches Picture 2 is the same final-frame entity; do not rename or duplicate it",
                f"required_end_state: exact whole-frame match to Picture 2 at {effective_seconds:.2f} seconds",
            ))
        blocks.append("\n".join(lines))
        cursor = end
    rules = (
        "TIMELINE_RULES:\n"
        "- Preserve every explicit action, actor, body part, object, direction, simultaneity, and verb in order.\n"
        "- A Shot creates a numbered header; a Move creates no header or cut.\n"
        "- Moves are consecutive beats of the owning Shot's single take. When FRAME_CONTINUITY_PLAN exists, "
        "Picture-to-Picture bridges control the prose and Move ranges stay internal; otherwise copy each range cue. "
        "Inherit the preceding state, name only needed physical camera travel, and keep an unchanged camera locked.\n\n"
        if has_moves else
        "TIMELINE_RULES:\n"
        "- Preserve every explicit action, actor, body part, object, direction, simultaneity, and verb in order.\n"
        "- Each configured Shot creates one numbered output header; later Shots begin with cuts.\n\n"
    )
    take_plan = _camera_take_plan(project, effective_seconds)
    frame_plan = _frame_continuity_plan(project, effective_seconds, aliases)
    return (
        rules
        + (take_plan + "\n\n" if take_plan else "")
        + "SHOT_PLAN:\n" + "\n\n".join(blocks)
        + ("\n\n" + frame_plan if frame_plan else "")
    )


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


def _enhanced_output_budget(effective_seconds: float, shot_count: int,
                            level: str = "normal") -> str:
    if level == "strong":
        if effective_seconds <= 6.5:
            words = "320-520" if shot_count == 1 else "420-680 total"
        elif effective_seconds <= 12.5:
            words = "480-720" if shot_count == 1 else "600-900 total"
        else:
            words = "800-1200 total"
        priority = "substantial creative scene development with dense new observable detail and no repetitive padding"
    elif effective_seconds <= 6.5:
        words = "180-280" if shot_count == 1 else "240-380 total"
        priority = "explicit action development and complete continuity resolution without creative plot expansion"
    elif effective_seconds <= 12.5:
        words = "280-440" if shot_count == 1 else "380-560 total"
        priority = "explicit action development and complete continuity resolution without creative plot expansion"
    else:
        words = "500-760 total"
        priority = "explicit action development and complete continuity resolution without creative plot expansion"
    return (
        "OUTPUT_BUDGET:\n"
        f"recommended_english_words: {words}\n"
        f"priority: {priority}"
    )


def _enhance_max_new_tokens(mode: str, enhance_level: str) -> int:
    if enhance_level == "strong":
        return STRONG_ENHANCE_MAX_NEW_TOKENS
    if mode == "REF2VA":
        return REF_ENHANCE_MAX_NEW_TOKENS
    if enhance_level == "normal":
        return RICH_ENHANCE_MAX_NEW_TOKENS
    return BASE_ENHANCE_MAX_NEW_TOKENS


def _estimated_mixed_prompt_tokens(text: str) -> int:
    """Conservative preflight estimate for mixed Korean/English llama.cpp input."""
    return int(math.ceil(len(text) / 3.2))


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
    has_storyboard = bool(model and any(
        plan.get("kind") == "Picture" and plan.get("role") == "storyboard"
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
    style_transfer_labels = [
        label for label, plan in (model or {}).get("label_plan", {}).items()
        if plan.get("kind") == "Subject" and plan.get("strength") == "style_transfer"
    ]
    if style_transfer_labels:
        reference_style_policy += (
            "; style-transfer Subjects " + ", ".join(style_transfer_labels)
            + " provide only their explicitly requested visual medium/rendering treatment; apply that treatment "
            "to the requested target while preserving the target's identity, face, body, hairstyle, clothing, "
            "accessories, objects, and action, and never copy the style source's character or scene content"
        )
    sections = [
        "INPUT DATA ONLY - DO NOT COPY THESE KEYS INTO THE FINAL H3 PROMPT",
        "MODE_DATA:\n"
        f"mode: {mode}\n"
        f"requested_duration_seconds: {project['requested_duration']:.2f}\n"
        f"effective_duration_seconds: {effective_seconds:.2f}\n"
        f"shot_count: {len(_shot_items(project))}\n"
        f"timeline_item_count: {len(project['shots'])}",
        "STYLE_POLICY:\n"
        "target_video_style: use only when explicitly requested in PROMPT_PRESETS, TARGET_REQUEST, SHOT_PLAN visual_action, or CONSTRAINTS\n"
        "when_unspecified: omit any target-wide style invented beyond a concrete keyframe or Strong Subject contract\n"
        f"reference_visual_style: {reference_style_policy}",
        "CAMERA_POLICY:\n"
        "source: obey explicit PROMPT_PRESETS first; otherwise infer composition, viewpoint, camera behavior, and explicit transition intent from TARGET_REQUEST and SHOT_PLAN visual_action\n"
        "per_shot: choose one coherent physical camera path that contains the required actions and final states; configured Moves are consecutive phases of that uninterrupted path\n"
        "expression: write camera behavior as natural English; add amplitude and speed only when meaningful\n"
        "shot_boundaries: each configured shot after Shot 1 is an ordinary cut at its time-range start; use cross-dissolve, fade, or wipe only when explicitly requested\n"
        "frame_anchor_editing: Picture anchor times never create cuts or transitions; interpolate continuously between anchors inside each configured shot\n"
        + (
            "storyboard_framing: preserve every distinct ordered panel viewpoint, approximate shot size, screen direction, and subject placement as recognizable camera states; connect them through physically continuous travel inside each owning Shot\n"
            "storyboard_compression: merge only adjacent panels whose framing and action are materially identical; never reduce a multi-framing storyboard to generic coherent framing, tracking, or following language\n"
            if has_storyboard else ""
        )
        + "restraint: do not invent decorative motion or a new cut when a static camera or a small continuous camera move presents the action clearly",
    ]
    shot_preset_blocks = []
    shot_number = 0
    move_number = 0
    for shot in project["shots"]:
        if _is_move(shot):
            move_number += 1
            preset_label = f"[Move {move_number} within Shot {shot_number}]"
        else:
            shot_number += 1
            move_number = 0
            preset_label = f"[Shot {shot_number}]"
        shot_presets = _normalize_shot_presets(shot.get("presets"))
        shot_preset_lines = []
        preset_style = STYLE_PRESET_PROMPTS.get(shot_presets["style"], "")
        if preset_style:
            shot_preset_lines.append(f"style: {preset_style}")
        for preset_name, choices in CAMERA_PRESET_PROMPTS.items():
            preset_prompt = choices.get(shot_presets[preset_name], "")
            if preset_prompt:
                shot_preset_lines.append(f"{preset_name}: {preset_prompt}")
        if shot_preset_lines:
            shot_preset_blocks.append(
                preset_label + "\n" + "\n".join(shot_preset_lines)
            )
    if shot_preset_blocks:
        sections.append(
            "PROMPT_PRESETS:\n"
            "scope: each block applies only to its named shot or Move timeline item; a Move preset is a continuous target state inside its owning Shot\n"
            "status: mandatory explicit user selections; preserve every non-none value in its named timeline item\n"
            "style_expression: when a Shot has style, state it naturally in that Shot's opening sentence; when a Move has style, develop it continuously inside the owning Shot without implying a cut\n"
            "conflicts: only an explicit instruction inside the same timeline item may refine a preset; otherwise the preset controls\n"
            "camera_expression: for a Shot, establish the selected camera state; for a Move, treat the selected shot size and angle as the state reached progressively by the end of its interval. "
            "Name physical travel such as dolly, track, crane, pan, or tilt, including a smooth reversal when necessary, and preserve the same camera, lens, axis, and continuous parallax; "
            "never present a Move preset as a fresh composition, use digital zoom unless explicitly selected, or output stacked labels\n"
            + "\n\n".join(shot_preset_blocks)
        )
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
        sections.append(_enhanced_output_budget(
            effective_seconds, len(_shot_items(project)), project.get("enhance_level", "normal")
        ))
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
    expected_shots = list(range(1, len(_shot_items(project)) + 1))
    final_shot = len(expected_shots)
    effective_seconds = align_frame_count(project["requested_duration"]) / MODEL_FPS
    reference_model = _reference_model(project) if mode == "REF2VA" else None
    content_locks = _input_content_locks(project)
    enhance_level = project.get("enhance_level", "normal" if project.get("enhance") else "none")
    active_prompts = (
        STRONG_MODE_LLM_SYSTEM_PROMPTS if enhance_level == "strong"
        else ENHANCED_MODE_LLM_SYSTEM_PROMPTS if enhance_level == "normal"
        else MODE_LLM_SYSTEM_PROMPTS
    )
    move_rules = ""
    if any(_is_move(item) for item in project["shots"]):
        move_rules = (
            "TIMED MOVE EVENTS\n"
            "A Shot starts a camera take. A Move is only a timed action or camera event inside the current take and "
            "never starts a shot, cut, reset, or new composition. Embed each Move cue once in the ongoing action "
            "or applicable frame-to-frame bridge; inherit the preceding physical state without restating the continuity contract."
        )
    system_prompt = "\n\n".join((
        _mode_prompt_preamble(mode),
        active_prompts[mode],
        move_rules,
        _figurine_animation_system_module(project, mode, enhance_level),
        _reference_system_modules(project) if mode == "REF2VA" else "",
        _single_pass_output_lock(
            mode, effective_seconds, final_shot, expected_shots, reference_model, content_locks,
            _move_output_cues(project, effective_seconds),
        ),
    ))
    return (
        "SYSTEM PROMPT:\n"
        f"{system_prompt}\n\n"
        f"USER DATA FOR ACTIVE MODE {mode}:\n"
        f"{video_prompt}"
    )


def _format_raw_model_prompt(system_prompt: str, user_prompt: str) -> str:
    """Format the exact text channels supplied to the prompt-generation model for UI debugging."""
    return (
        "===== SYSTEM PROMPT =====\n"
        f"{system_prompt}\n\n"
        "===== USER PROMPT =====\n"
        f"{user_prompt}"
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
    omni = _omni_install_state(roots)
    return [{
        "id": DEFAULT_ENHANCE_MODEL_ID,
        "label": "Qwen3.8-27B Uncensored Q4_K_M",
        "installed": default_installed,
        "size": 0 if default_installed else DEFAULT_ENHANCE_MODEL_SIZE,
    }, {
        "id": OMNI_MODEL_ID,
        "label": OMNI_MODEL_DISPLAY_NAME,
        "installed": omni["installed"],
        "size": omni["missing_size"],
    }]


def _omni_model_dir(roots: list[str]) -> str:
    return os.path.join(roots[0], "pytraveler", "MiniMax-H3-Prompt-Rewriter-LoRA-Omni-GGUF")


def _omni_install_state(roots: list[str]) -> dict[str, Any]:
    if not roots:
        return {
            "installed": False, "base_installed": False, "mmproj_installed": False,
            "adapter_installed": False, "missing_size": OMNI_TOTAL_SIZE,
            "base_path": "", "mmproj_path": "", "adapter_path": "",
        }
    model_dir = _omni_model_dir(roots)
    base_path = os.path.join(model_dir, OMNI_BASE_FILE)
    mmproj_path = os.path.join(model_dir, OMNI_MMPROJ_FILE)
    adapter_path = os.path.join(model_dir, OMNI_ADAPTER_FILE)
    base_installed = os.path.isfile(base_path)
    mmproj_installed = os.path.isfile(mmproj_path)
    adapter_installed = os.path.isfile(adapter_path)
    return {
        "installed": base_installed and mmproj_installed and adapter_installed,
        "base_installed": base_installed,
        "mmproj_installed": mmproj_installed,
        "adapter_installed": adapter_installed,
        "missing_size": (0 if base_installed else OMNI_BASE_SIZE)
                        + (0 if mmproj_installed else OMNI_MMPROJ_SIZE)
                        + (0 if adapter_installed else OMNI_ADAPTER_SIZE),
        "base_path": base_path, "mmproj_path": mmproj_path, "adapter_path": adapter_path,
    }


def list_image_models(roots: list[str] | None = None) -> list[dict[str, Any]]:
    roots = roots if roots is not None else _llm_roots()
    writer_model = next((path for root in roots for path in glob.iglob(os.path.join(root, "**", DEFAULT_ENHANCE_MODEL_FILE), recursive=True) if os.path.isfile(path)), None)
    qwen_mmproj = next((path for root in roots for path in glob.iglob(os.path.join(root, "**", QWEN_IMAGE_MMPROJ_FILE), recursive=True) if os.path.isfile(path)), None)
    qwen_installed = bool(writer_model and qwen_mmproj)
    qwen_missing_size = (0 if writer_model else QWEN_IMAGE_MODEL_SIZE) + (0 if qwen_mmproj else QWEN_IMAGE_MMPROJ_SIZE)
    omni = _omni_install_state(roots)
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
        "id": OMNI_MODEL_ID,
        "label": (
            OMNI_MODEL_DISPLAY_NAME
            + (" · installed" if omni["installed"] else f" · {omni['missing_size'] / 1e9:.2f} GB missing")
            + f" · {OMNI_MODEL_VRAM_LABEL}"
        ),
        "installed": omni["installed"],
        "size": omni["missing_size"],
        "text_installed": omni["base_installed"] and omni["adapter_installed"],
        "vision_installed": omni["mmproj_installed"],
        "text_size": (0 if omni["base_installed"] else OMNI_BASE_SIZE)
                     + (0 if omni["adapter_installed"] else OMNI_ADAPTER_SIZE),
        "vision_size": 0 if omni["mmproj_installed"] else OMNI_MMPROJ_SIZE,
        "enhance_model": OMNI_MODEL_ID,
        "image_model": OMNI_MODEL_ID,
        "supported_modes": list(SUPPORTED_MODES[1:]),
        "runtime": "llama.cpp-mtmd-gguf-lora",
    }]


def _download_image_component(repo_id: str, filename: str, local_dir: str, component_size: int,
                              completed_size: int, bundle_size: int, progress=None) -> str:
    # hf_xet writes through an opaque chunk cache, so the destination file can
    # remain at zero bytes until it is atomically completed. The regular Hub
    # HTTP path grows a visible .incomplete file that we can report live.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import hf_hub_download
    try:
        import huggingface_hub.constants as hf_constants
        hf_constants.HF_HUB_DISABLE_XET = True
    except (ImportError, AttributeError):
        pass

    stop_monitor = threading.Event()

    def monitor_download() -> None:
        last_size = -1
        while not stop_monitor.wait(0.35):
            candidates = glob.glob(os.path.join(local_dir, "**", "*.incomplete"), recursive=True)
            candidates.extend(glob.glob(os.path.join(local_dir, "**", "*.part"), recursive=True))
            matching = [path for path in candidates if filename.lower() in os.path.basename(path).lower()]
            if matching:
                candidates = matching
            candidates.append(os.path.join(local_dir, filename))
            size = min(component_size, max((os.path.getsize(path) for path in candidates if os.path.isfile(path)), default=0))
            if size != last_size and progress:
                progress(stage="downloading", message=f"Downloading image model component: {filename}",
                         downloaded=completed_size + size, total=bundle_size)
                last_size = size

    monitor = threading.Thread(target=monitor_download, name="toyxyz-h3-image-download", daemon=True)
    monitor.start()
    try:
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        if progress:
            progress(
                stage="downloading", message=f"Downloaded model component: {filename}",
                downloaded=completed_size + component_size, total=bundle_size,
            )
        return downloaded
    finally:
        stop_monitor.set()
        monitor.join(timeout=1)


def _resolve_omni_model(progress=None) -> tuple[str, str, str]:
    roots = _llm_roots()
    if not roots:
        raise RuntimeError("ComfyUI has no registered models/LLM directory.")
    state = _omni_install_state(roots)
    if state["installed"]:
        if progress:
            progress(stage="model_ready", message="Qwen2.5-Omni-7B + MiniMax-H3 Omni GGUF LoRA bundle is installed.")
        return tuple(os.path.abspath(state[key]) for key in ("base_path", "mmproj_path", "adapter_path"))
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download the Omni GGUF bundle.") from exc
    model_dir = _omni_model_dir(roots)
    os.makedirs(model_dir, exist_ok=True)
    total = int(state["missing_size"])
    completed = 0
    if progress:
        progress(stage="downloading", message="Starting Qwen2.5-Omni + rewriter LoRA download.", downloaded=0, total=total)
    components = (
        ("base_installed", OMNI_BASE_REPO, OMNI_BASE_FILE, OMNI_BASE_SIZE),
        ("mmproj_installed", OMNI_BASE_REPO, OMNI_MMPROJ_FILE, OMNI_MMPROJ_SIZE),
        ("adapter_installed", OMNI_ADAPTER_REPO, OMNI_ADAPTER_FILE, OMNI_ADAPTER_SIZE),
    )
    for installed_key, repo, filename, size in components:
        if not state[installed_key]:
            _download_image_component(repo, filename, model_dir, size, completed, total, progress)
            completed += size
    installed = _omni_install_state(roots)
    if not installed["installed"]:
        raise RuntimeError("The Omni GGUF download completed but one or more required files are missing.")
    if progress:
        progress(stage="downloading", message="Omni GGUF bundle download completed.", downloaded=total, total=total)
        progress(stage="model_ready", message="Qwen2.5-Omni-7B + MiniMax-H3 Omni GGUF LoRA bundle is ready.")
    return tuple(os.path.abspath(installed[key]) for key in ("base_path", "mmproj_path", "adapter_path"))


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
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download the default enhancement model.") from exc
    try:
        import huggingface_hub.constants as hf_constants
        hf_constants.HF_HUB_DISABLE_XET = True
    except (ImportError, AttributeError):
        pass
    os.makedirs(roots[0], exist_ok=True)
    stop_monitor = threading.Event()

    def monitor_download() -> None:
        last_size = -1
        while not stop_monitor.wait(0.35):
            candidates = glob.glob(os.path.join(roots[0], "**", "*.incomplete"), recursive=True)
            candidates.extend(glob.glob(os.path.join(roots[0], "**", "*.part"), recursive=True))
            matching = [
                path for path in candidates
                if DEFAULT_ENHANCE_MODEL_FILE.lower() in os.path.basename(path).lower()
            ]
            if matching:
                candidates = matching
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


def _llama_runtime_backend() -> str:
    requested = os.environ.get("TOYXYZ_LLAMA_BACKEND", "auto").strip().lower()
    if requested in {"cuda", "vulkan", "cpu"}:
        return requested
    if sys.platform == "win32":
        try:
            import torch
            if torch.cuda.is_available() and tuple(torch.cuda.get_device_capability(0)) in {
                (8, 6), (8, 9), (12, 0), (12, 1),
            }:
                return "cuda"
        except (ImportError, RuntimeError, AttributeError):
            pass
        return "vulkan"
    if sys.platform == "darwin":
        return "cpu"
    return "vulkan"


def _llama_runtime_assets(backend: str) -> tuple[str, ...]:
    assets = {
        ("win32", "cuda"): (
            f"llama-{LLAMA_RUNTIME_RELEASE}-bin-win-cuda-13.3-x64.zip",
            "cudart-llama-bin-win-cuda-13.3-x64.zip",
        ),
        ("win32", "vulkan"): (f"llama-{LLAMA_RUNTIME_RELEASE}-bin-win-vulkan-x64.zip",),
        ("win32", "cpu"): (f"llama-{LLAMA_RUNTIME_RELEASE}-bin-win-cpu-x64.zip",),
        ("linux", "vulkan"): (f"llama-{LLAMA_RUNTIME_RELEASE}-bin-ubuntu-vulkan-x64.tar.gz",),
        ("linux", "cpu"): (f"llama-{LLAMA_RUNTIME_RELEASE}-bin-ubuntu-x64.tar.gz",),
        ("darwin", "cpu"): (f"llama-{LLAMA_RUNTIME_RELEASE}-bin-macos-arm64.tar.gz",),
    }
    selected = assets.get((sys.platform, backend))
    if not selected:
        raise RuntimeError(f"No managed llama.cpp {backend} runtime is available for {sys.platform}.")
    return selected


def _llama_runtime_root() -> str:
    try:
        import folder_paths
        user_dir = folder_paths.get_user_directory()
    except (ImportError, AttributeError):
        user_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user")
    return os.path.join(user_dir, "toyxyz_minimax_h3", "runtime")


def _managed_llama_dir(backend: str) -> str:
    return os.path.join(_llama_runtime_root(), f"{LLAMA_RUNTIME_RELEASE}-{backend}")


def _runtime_binary(directory: str, names: tuple[str, ...]) -> str:
    if not os.path.isdir(directory):
        return ""
    for current, _dirs, files in os.walk(directory):
        for name in names:
            if name in files:
                return os.path.abspath(os.path.join(current, name))
    return ""


def _safe_extract_runtime(archive: str, destination: str) -> None:
    root = os.path.abspath(destination)

    def safe(name: str) -> bool:
        target = os.path.abspath(os.path.join(root, name))
        return target == root or target.startswith(root + os.sep)

    if archive.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            if any(not safe(item.filename) for item in bundle.infolist()):
                raise RuntimeError("The llama.cpp runtime archive contains an unsafe path.")
            bundle.extractall(root)
        return
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if any(not safe(item.name) for item in members):
            raise RuntimeError("The llama.cpp runtime archive contains an unsafe path.")
        bundle.extractall(root)


def _download_runtime_asset(url: str, destination: str, completed: int, total: int,
                            progress=None) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-toyxyz-Minimax-H3"})
    with urllib.request.urlopen(request, timeout=60) as response, open(destination, "wb") as handle:
        expected = int(response.headers.get("Content-Length") or 0)
        received = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
            received += len(block)
            if progress:
                progress(
                    stage="downloading", message=f"Downloading llama.cpp runtime: {os.path.basename(destination)}",
                    downloaded=completed + received, total=total or completed + expected,
                )
    return received


def _ensure_managed_llama_runtime(progress=None) -> str:
    backend = _llama_runtime_backend()
    directory = _managed_llama_dir(backend)
    required = (("llama-completion.exe", "llama-completion"),
                ("llama-mtmd-cli.exe", "llama-mtmd-cli"))
    if all(_runtime_binary(directory, names) for names in required):
        return directory
    with _LLAMA_RUNTIME_LOCK:
        if all(_runtime_binary(directory, names) for names in required):
            return directory
        assets = _llama_runtime_assets(backend)
        staging = directory + ".part"
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        os.makedirs(staging, exist_ok=True)
        sizes: list[int] = []
        for name in assets:
            try:
                request = urllib.request.Request(f"{LLAMA_RUNTIME_URL}/{name}", method="HEAD")
                with urllib.request.urlopen(request, timeout=30) as response:
                    sizes.append(int(response.headers.get("Content-Length") or 0))
            except (OSError, urllib.error.URLError, ValueError):
                sizes.append(0)
        total = sum(sizes) if all(sizes) else 0
        completed = 0
        try:
            if progress:
                progress(stage="downloading", message=f"Installing llama.cpp {LLAMA_RUNTIME_RELEASE} {backend} runtime.", downloaded=0, total=total)
            for name in assets:
                archive = os.path.join(staging, name)
                received = _download_runtime_asset(
                    f"{LLAMA_RUNTIME_URL}/{name}", archive, completed, total, progress,
                )
                completed += received
                _safe_extract_runtime(archive, staging)
                os.remove(archive)
            if not all(_runtime_binary(staging, names) for names in required):
                raise RuntimeError("The downloaded llama.cpp archive is missing required executables.")
            if os.path.isdir(directory):
                shutil.rmtree(directory)
            os.makedirs(os.path.dirname(directory), exist_ok=True)
            os.replace(staging, directory)
            if progress:
                progress(stage="model_ready", message=f"llama.cpp {LLAMA_RUNTIME_RELEASE} {backend} runtime is ready.", downloaded=total or completed, total=total or completed)
            return directory
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def _find_llama_cli(progress=None) -> str:
    configured = os.environ.get("TOYXYZ_LLAMA_CLI", "").strip()
    candidates = [configured]
    if not configured:
        directory = _ensure_managed_llama_runtime(progress)
        candidates.extend((_runtime_binary(directory, ("llama-cli.exe", "llama-cli")),
                           shutil.which("llama-cli") or "", shutil.which("llama-cli.exe") or ""))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError(
        "The managed llama.cpp runtime was installed without llama-cli. Delete its runtime folder "
        "to reinstall it, or set TOYXYZ_LLAMA_CLI to an executable path."
    )


def _find_llama_completion(progress=None) -> str:
    """Find llama.cpp's one-shot completion frontend.

    Current llama.cpp releases use ``llama-cli`` as an interactive terminal UI;
    that program can write the rendered input prompt to stdout.  The upstream
    MiniMax-H3 rewriter integration therefore uses ``llama-completion`` and
    keeps ``llama-cli`` only as a compatibility fallback for older bundles.
    """
    configured = os.environ.get("TOYXYZ_LLAMA_COMPLETION", "").strip()
    names = ("llama-completion.exe", "llama-completion")
    candidates = [configured]
    if not configured:
        directory = _ensure_managed_llama_runtime(progress)
        candidates.append(_runtime_binary(directory, names))
        candidates.extend(shutil.which(name) or "" for name in names)
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    # Old llama.cpp packages shipped only llama-cli, when it still behaved as
    # the non-interactive completion frontend.
    return _find_llama_cli(progress)


def _find_llama_server(progress=None) -> str:
    configured = os.environ.get("TOYXYZ_LLAMA_SERVER", "").strip()
    names = ("llama-server.exe", "llama-server")
    candidates = [configured]
    if not configured:
        directory = _ensure_managed_llama_runtime(progress)
        candidates.append(_runtime_binary(directory, names))
        candidates.extend(shutil.which(name) or "" for name in names)
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError("llama-server was not found; image analysis will use llama-cli instead.")


def _find_llama_mtmd_cli(progress=None) -> str:
    configured = os.environ.get("TOYXYZ_LLAMA_MTMD_CLI", "").strip()
    names = ("llama-mtmd-cli.exe", "llama-mtmd-cli")
    candidates = [configured]
    if not configured:
        directory = _ensure_managed_llama_runtime(progress)
        candidates.append(_runtime_binary(directory, names))
        candidates.extend(shutil.which(name) or "" for name in names)
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError(
        "The managed llama.cpp runtime was installed without llama-mtmd-cli. Delete its runtime folder "
        "to reinstall it, or set TOYXYZ_LLAMA_MTMD_CLI to an executable path."
    )


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
    executable = _find_llama_server(progress)
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
    elif re.search(r"\btask:\s*(?:T2AV|I2AV|FL2AV|L2AV|REF2AV)\b", text, re.IGNORECASE) \
            and re.search(r"\braw_prompt:\s*", text, re.IGNORECASE):
        # llama.cpp can also echo the complete short prompt inline, without a
        # truncation marker or response delimiter. Select the task-specific
        # final schema start. rfind avoids mistaking schema-shaped text inside
        # a user-supplied raw prompt for the actual assistant response.
        task_match = re.search(r"\btask:\s*(\w+)", text, re.IGNORECASE)
        task = task_match.group(1).upper() if task_match else ""
        starts = {
            "T2AV": "integrated_multimodal_description:",
            "I2AV": "For the target video,",
            "FL2AV": "How the reference pictures align with the target video",
            "L2AV": "How the reference pictures align with the target video",
            "REF2AV": "subject_definitions:",
        }
        marker = starts.get(task, "")
        start = text.rfind(marker) if marker else -1
        if start >= 0:
            text = text[start:]
    elif re.search(r"\(truncated\)", text, flags=re.IGNORECASE):
        # Some llama.cpp/mtmd builds echo a long --prompt without the legacy
        # "> " delimiter, terminate that echo with "(truncated)", and then
        # print the actual assistant response. Keep the first valid H3 schema
        # start after the final truncation marker. Include alignment lines so
        # I2AV/FL2AV/L2AV do not lose their required opening instruction.
        tail = re.split(r"\(truncated\)", text, flags=re.IGNORECASE)[-1]
        schema_start = re.search(
            r"(?:^|\s)(?:For the target video,|How the reference pictures align with "
            r"the target video|integrated_multimodal_description:|subject_definitions:)",
            tail, flags=re.MULTILINE,
        )
        if schema_start:
            text = tail[schema_start.start():].lstrip()
    text = re.sub(r"(?:^|\n)Exiting\.\.\.\s*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    # Never expose an unfinished reasoning trace when generation reaches its
    # token limit before </think>. Recover a valid schema if one follows the
    # trace; otherwise return empty so the caller reports generation failure.
    if re.search(r"<think\b[^>]*>", text, flags=re.IGNORECASE):
        schema = re.search(
            r"(?:For the target video,|How the reference pictures align with "
            r"the target video|integrated_multimodal_description:|subject_definitions:)",
            text, flags=re.IGNORECASE,
        )
        text = text[schema.start():].strip() if schema else ""
    # Some llama-completion builds render the model's textual end sentinel
    # instead of consuming it as a stop token. It is runtime metadata, not part
    # of a MiniMax-H3 prompt.
    text = re.sub(
        r"(?:\s*\[(?:end of text|end_of_text)\]\s*)+$", "", text,
        flags=re.IGNORECASE,
    ).strip()
    # llama-completion may print a generated chat/EOG token literally when
    # --special is enabled for pre-rendered ChatML input. These are transport
    # delimiters and must never become part of the H3 prompt.
    text = re.sub(
        r"(?:\s*<\|(?:im_end|endoftext|eot_id|end_of_text)\|>\s*)+$", "", text,
        flags=re.IGNORECASE,
    ).strip()
    fence = re.fullmatch(r"```(?:text)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    return text


def _render_omni_text_prompt(system_prompt: str, user_prompt: str) -> str:
    """Render one Qwen2.5-Omni text turn without enabling conversation mode.

    This is the text-only equivalent of the reference rewriter's GGUF chat
    template path.  Passing ``-sysf`` and ``-p`` separately lets recent
    llama-cli builds treat the request as an interactive turn and echo the
    ``task``/``raw_prompt`` block as generated text.
    """
    def safe(value: str) -> str:
        return value.replace("<|im_end|>", "").replace("<|im_start|>", "")

    return (
        "<|im_start|>system\n" + safe(system_prompt).strip() + "<|im_end|>\n"
        "<|im_start|>user\n" + safe(user_prompt).strip() + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _render_qwen3_text_prompt(system_prompt: str, user_prompt: str) -> str:
    """Render a non-interactive Qwen3 chat turn for llama-completion.

    Supplying ``-sysf``/``-p`` together with llama.cpp conversation mode has
    proven fragile on Windows CUDA builds, especially with long contexts.  A
    fully rendered ChatML turn avoids that frontend state machine. Qwen3's
    native non-thinking template ends an empty thinking block before generation.
    The caller must pass ``--special`` so these ChatML markers are tokenized as
    control tokens rather than ordinary text.
    """
    def safe(value: str) -> str:
        return value.replace("<|im_end|>", "").replace("<|im_start|>", "")

    return (
        "<|im_start|>system\n" + safe(system_prompt).strip() + "<|im_end|>\n"
        "<|im_start|>user\n" + safe(user_prompt).strip() + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _qwen_thinking_args(executable: str) -> list[str]:
    """Reasoning-off flags for the selected llama.cpp frontend generation."""
    if os.path.basename(executable).lower().startswith("llama-completion"):
        return ["--reasoning", "off"]
    return ["--chat-template-kwargs", '{"enable_thinking":false}']


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
        "subject_visual": "Prioritize only stable visible traits of the user-specified person, object, or environment; keep motion, camera, cuts, and audio separate.",
        "visual_style": "Prioritize only rendering medium, palette, lighting treatment, materials, shading, and visual texture; do not bind source identity or action to the style.",
        "motion": "Extract an actor-neutral motion plan only: pose progression, limb trajectories, direction, speed, contacts, interaction timing, weight transfer, and physical rhythm. Refer to performers only as Actor A, Actor B, and so on. Omit face, identity, age, gender, body shape and proportions, skin, hair, clothing, accessories, materials, texture, rendering medium, visual style, environment appearance, camera, cuts, and audio.",
        "motion_camera": "Extract only an actor-neutral kinematic plan and synchronized camera plan: pose progression, body-part and limb trajectories, locomotion, direction, speed, contacts, interaction timing, weight transfer, physical rhythm, shot size, viewpoint, framing changes, camera path, movement direction, amplitude, speed, stabilization, and subject-tracking relationship. Treat the video as a motion template rather than scene-content evidence. Refer to the principal performer only as Actor A; use Actor B or later only for action choreography that directly interacts with Actor A, never merely because a background person is visible. Do not name, count, locate, or describe source people, objects, props, architecture, scenery, or background events. If a source action uses an object, retain only the actor's body/limb trajectory and timing; do not identify or introduce the object. Omit face, identity, age, gender, body shape and proportions, skin, hair, clothing, accessories, all visible content, materials, texture, rendering medium, visual style, environment appearance, lighting, cuts, visible text, and audio.",
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
CAMERA_EDITING: framing, viewpoint, camera movement, supported cut boundaries, and pacing; write unknown when samples cannot distinguish camera motion from subject motion. For each framing state, use exactly one shot-size term consistent with its visible body range: close-up=head and shoulders, medium close-up=chest or shoulders upward, medium shot=waist upward, medium wide or medium full=thighs or knees upward, full shot=the entire body from head to toe.
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


def _scope_video_analysis(analysis: str, role: str) -> str:
    """Remove evidence that a narrowly scoped video preset must never transfer."""
    if role not in {"motion", "motion_camera"}:
        return analysis
    def section(name: str, following: str) -> str:
        match = re.search(
            rf"(?:^|\n){name}:\s*(.*?)(?=\n(?:{following}):|\Z)",
            analysis,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return match.group(1).strip() if match else "unavailable from the sampled frames"

    action_timeline = section(
        "ACTION_TIMELINE", "CAMERA_EDITING|ENVIRONMENT_OBJECTS|STYLE_LIGHTING|VISIBLE_TEXT|EDIT_CONTINUITY"
    )
    if role == "motion_camera":
        camera_editing = section(
            "CAMERA_EDITING", "ENVIRONMENT_OBJECTS|STYLE_LIGHTING|VISIBLE_TEXT|EDIT_CONTINUITY"
        )
        return (
            "MOTION_CAMERA_SCOPE: Transfer only actor-neutral motion, action timing, camera behavior, and "
            "their synchronization. This is a kinematic template, not source scene-content evidence. Never add "
            "a person, actor, creature, object, prop, architecture, environment feature, or background event from "
            "the source. Map motion only onto target entities already established by the target request or another "
            "authorized reference; discard unmatched actors and object identities. Source identity, body traits, "
            "skin, hair, clothing, accessories, visible content, materials, texture, style, environment, lighting, "
            "cuts, visible text, and audio are intentionally excluded.\n"
            f"ACTION_TIMELINE: {action_timeline}\n"
            f"CAMERA_EDITING: {camera_editing}"
        )
    return (
        "MOTION_ONLY_SCOPE: Transfer only actor-neutral motion and timing. Source performer appearance, "
        "identity, body traits, skin, hair, clothing, materials, texture, style, environment, camera, cuts, "
        "and audio are intentionally excluded.\n"
        f"ACTION_TIMELINE: {action_timeline}"
    )


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
            command = [_find_llama_cli(progress), "-m", model_path, "--mmproj", mmproj_path]
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
    analysis = _scope_video_analysis(_clean_video_analysis(output), role)
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
    if role == "storyboard":
        return """Analyze the supplied image only as storyboard planning evidence for a MiniMax H3 video prompt.
Read panels in their visible order. Keep each panel's framing, placement, and action bound together so a later writer cannot reduce the storyboard to a plot summary. Do not transfer performer identity, face, body, clothing, rendering style, palette, lighting treatment, exact output timing, or exact keyframe matching.
Return exactly the six compact labeled lines below. Use "none visible" when unsupported and do not add prose. PANEL_SEQUENCE must contain one semicolon-separated record for every visible panel in reading order; never merge panels there.

STORYBOARD_LAYOUT: panel count, grid/layout, and reading order.
PANEL_SEQUENCE: P1={viewpoint and approximate shot size | relative subject placement and screen direction | explicitly depicted action}; P2={...}; continue through every panel.
FRAMING_PROGRESSION: ordered distinct viewpoint and shot-size changes, retaining over-the-shoulder, profile, rear, high/low angle, close/wide, and insert/detail cues when visibly present.
SPATIAL_CONTINUITY: persistent environment layout, entrances/exits, travel direction, and subject-to-object relationships supported across panels.
PANEL_BOUNDARIES: visible cut/transition cues in the source storyboard; these are planning evidence and do not independently authorize target-video cuts.
VISIBLE_TEXT: quote only clearly readable planning text; otherwise write "none visible".

Enclose the six lines exactly once in <REFERENCE_ANALYSIS> and </REFERENCE_ANALYSIS>."""
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
            _find_llama_cli(progress), "-m", model_path, "--mmproj", mmproj_path, "--image", vision_path,
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


def _omni_system_prompt(mode: str) -> str:
    with open(OMNI_SYSTEM_PROMPTS_PATH, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return str(config["ref2av" if mode == "REF2VA" else "base"])


def _build_omni_raw_prompt(project: dict[str, Any], effective_seconds: float) -> str:
    reference_model = _reference_model(project) if project["mode"] == "REF2VA" else None
    aliases = reference_model["aliases"] if reference_model else {}
    applications = reference_model["applications"] if reference_model else []
    lines = [_shot_description(project, effective_seconds, aliases, applications)]
    shot_number = 0
    move_number = 0
    for shot in project["shots"]:
        if _is_move(shot):
            move_number += 1
            label = f"[Move {move_number} within Shot {shot_number}]"
        else:
            shot_number += 1
            move_number = 0
            label = f"[Shot {shot_number}]"
        presets = _normalize_shot_presets(shot.get("presets"))
        selected: list[str] = []
        style = STYLE_PRESET_PROMPTS.get(presets["style"], "")
        if style:
            selected.append(style)
        for name, choices in CAMERA_PRESET_PROMPTS.items():
            value = choices.get(presets[name], "")
            if value:
                selected.append(value)
        if selected:
            continuity = " Continue without a cut from the preceding camera state." if _is_move(shot) else ""
            lines.append(f"{label} Required presets: " + "; ".join(selected) + "." + continuity)
    return "\n".join(lines)


def _omni_reference_inputs(project: dict[str, Any], mode: str, temp_dir: str
                           ) -> tuple[list[tuple[str, str]], str]:
    labeled = _reference_labels(project["references"])
    if mode != "REF2VA":
        wanted = {
            "T2VA": set(), "I2VA": {"first_frame"}, "L2VA": {"last_frame"},
            "FL2VA": {"first_frame", "last_frame"},
        }[mode]
        labeled = [ref for ref in labeled if ref["type"] == "picture" and ref["role"] in wanted]
    attachments: list[tuple[str, str]] = []
    lines = ["Ordered MiniMax-H3 references:"] if mode == "REF2VA" else []
    effective = align_frame_count(float(project["requested_duration"])) / MODEL_FPS
    for index, ref in enumerate(labeled, 1):
        label = ref["label"]
        if mode == "I2VA":
            heading = f"{label} — exact first frame at 0.00 seconds:"
        elif mode == "L2VA":
            heading = f"{label} — exact final frame at {effective:.2f}s:"
        elif mode == "FL2VA" and ref["role"] == "first_frame":
            heading = f"{label} — exact first frame at 0.00 seconds:"
        elif mode == "FL2VA":
            heading = f"{label} — exact final frame at {effective:.2f}s:"
        else:
            heading = f"{label}:"
        lines.append(heading)
        if ref["type"] == "picture":
            path = _resolve_uploaded_image({
                "filename": ref.get("image_filename"), "subfolder": ref.get("image_subfolder"),
            })
            attachments.append(("image", path))
            lines.append("<__media__>")
        elif ref["type"] == "video":
            path = _resolve_uploaded_video({
                "filename": ref.get("video_filename"), "subfolder": ref.get("video_subfolder"),
            })
            clip_dir = os.path.join(temp_dir, f"video-{index}")
            os.makedirs(clip_dir, exist_ok=True)
            duration = max(REF_VIDEO_MIN_SECONDS, float(ref.get("duration") or effective))
            start = max(0.0, float(ref.get("trim_start") or 0.0))
            frames, timestamps = _extract_video_analysis_frames(path, duration, clip_dir, start)
            lines.append(
                f"The following {len(frames)} images are chronological samples of the selected "
                f"source interval; relative timestamps: "
                + ", ".join(f"{value:.3f}s" for value in timestamps) + "."
            )
            for frame in frames:
                attachments.append(("image", frame))
                lines.append("<__media__>")
        elif ref["type"] == "audio":
            path = _resolve_uploaded_audio({
                "filename": ref.get("audio_filename"), "subfolder": ref.get("audio_subfolder"),
            })
            attachments.append(("audio", path))
            lines.append("<__media__>")
    return attachments, "\n".join(lines)


def _run_omni_process(command: list[str], cancel_event: threading.Event | None,
                      job_id: str) -> str:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", creationflags=creationflags,
    )
    if job_id:
        _set_enhance_stopper(job_id, process.terminate)
    deadline = time.monotonic() + 1800
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                process.terminate()
                raise EnhancementCancelled("Prompt generation was stopped by the user.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.terminate()
                raise RuntimeError("Omni prompt generation exceeded the 30-minute timeout.")
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
    if process.returncode != 0:
        tail = "\n".join(stderr.splitlines()[-16:])
        raise RuntimeError(f"Omni llama.cpp runtime exited with code {process.returncode}.\n{tail}")
    return _clean_llm_output(stdout)


def _enhance_project_omni(result: dict[str, Any], model_id: str, progress=None,
                          cancel_event: threading.Event | None = None,
                          job_id: str = "") -> dict[str, Any]:
    project = result["project"]
    mode = project["mode"]
    if mode not in SUPPORTED_MODES[1:]:
        raise ValueError(f"The Omni rewriter does not support mode {mode}.")
    with _ENHANCE_LOCK, tempfile.TemporaryDirectory(prefix="toyxyz_h3_omni_") as temp_dir:
        if cancel_event is not None and cancel_event.is_set():
            raise EnhancementCancelled("Prompt generation was stopped by the user.")
        base_path, mmproj_path, adapter_path = _resolve_omni_model(progress)
        system_prompt = _omni_system_prompt(mode)
        if any(_is_move(item) for item in project["shots"]):
            system_prompt += (
                "\n\nOnly a configured Shot starts a new camera take. Each following Move is a range-based "
                "beat inside that take: inherit the preceding camera and subject state, continue one physical path, "
                "and reach the requested endpoint without a header or cut."
            )
        raw_prompt = _build_omni_raw_prompt(project, result["effective_duration"])
        task = {"T2VA": "T2AV", "I2VA": "I2AV", "L2VA": "L2AV", "FL2VA": "FL2AV", "REF2VA": "REF2AV"}[mode]
        resolution = "16:9" if mode == "T2VA" else "adaptive"
        attachments, reference_block = _omni_reference_inputs(project, mode, temp_dir)
        user_prompt = (
            (reference_block + "\n\n" if reference_block else "")
            + "Rewrite request:\n"
            + f"task: {task}\nresolution: {resolution}\n"
            + f"effective_duration: {result['effective_duration']:.2f}s\nraw_prompt: {raw_prompt}"
        )
        try:
            import comfy.model_management as model_management
            model_management.unload_all_models()
            model_management.soft_empty_cache(force=True)
        except (ImportError, AttributeError):
            pass
        if progress:
            progress(
                stage="generating",
                message=(f"Loading Qwen2.5-Omni-7B with the Omni rewriter LoRA and "
                         f"{len(attachments)} ordered media input(s)."),
            )
        if attachments:
            command = [
                _find_llama_mtmd_cli(progress), "--model", base_path, "--mmproj", mmproj_path,
                "--lora", adapter_path, "--n-gpu-layers", "999", "--ctx-size", "16384",
                "--predict", "4096", "--temp", "0", "--system-prompt", system_prompt,
            ]
            for kind, path in attachments:
                command.extend(["--audio" if kind == "audio" else "--image", path])
            command.extend(["--prompt", user_prompt])
        else:
            # Match the reference GGUF backend: render the complete chat turn
            # first, pass it through a file, and disable conversation mode.
            # Modern llama-cli is an interactive UI and may otherwise echo the
            # task/resolution/raw_prompt request block into stdout.
            prompt_file = os.path.join(temp_dir, "omni_prompt.txt")
            with open(prompt_file, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(_render_omni_text_prompt(system_prompt, user_prompt))
            command = [
                _find_llama_completion(progress), "--model", base_path, "--lora", adapter_path,
                "--file", prompt_file, "--n-gpu-layers", "999", "--ctx-size", "16384",
                "--predict", "4096", "--temp", "0", "-no-cnv", "-st",
                "--no-display-prompt", "--no-warmup", "--simple-io",
            ]
        enhanced = _run_omni_process(command, cancel_event, job_id)
        if not enhanced:
            raise RuntimeError("The Omni rewriter returned an empty prompt.")
        enhanced = _enforce_move_camera_continuity(
            enhanced, project, result["effective_duration"],
        )
        enhanced = _enforce_ref_frame_anchor_timing(
            enhanced, project, result["effective_duration"],
        )
        enhanced = _enforce_framing_body_range(enhanced)
        if progress:
            progress(stage="complete", message="Omni prompt generation completed.")
        return {
            "enhanced_prompt": enhanced, "model": model_id, "model_path": base_path,
            "reference_analyses": [],
            "raw_model_prompt": _format_raw_model_prompt(system_prompt, user_prompt),
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
    if selected_model == OMNI_MODEL_ID:
        return _enhance_project_omni(
            result, selected_model, progress, cancel_event=cancel_event, job_id=job_id,
        )
    enhance_level = project.get("enhance_level", "normal" if project.get("enhance") is True else "none")
    rich_enhance = enhance_level in {"normal", "strong"}
    strong_enhance = enhance_level == "strong"
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
    expected_shots = list(range(1, len(_shot_items(result["project"])) + 1))
    shot_headers = ", ".join(f"[Shot {number}]" for number in expected_shots)
    mode = result["project"]["mode"]
    active_mode_prompts = (
        STRONG_MODE_LLM_SYSTEM_PROMPTS if strong_enhance
        else ENHANCED_MODE_LLM_SYSTEM_PROMPTS if rich_enhance
        else MODE_LLM_SYSTEM_PROMPTS
    )
    system_prompt = _mode_prompt_preamble(mode) + "\n\n" + active_mode_prompts[mode]
    if any(_is_move(item) for item in result["project"]["shots"]):
        system_prompt += (
            "\n\nTIMED MOVE EVENTS: A Shot starts a camera take. A Move is only a timed action or camera "
            "event inside the current take, never a shot, cut, reset, or new composition. Embed each cue once in the "
            "ongoing action or applicable frame-to-frame bridge and inherit the preceding state without repeating the camera lock."
        )
    figurine_module = _figurine_animation_system_module(
        result["project"], mode, enhance_level,
    )
    if figurine_module:
        system_prompt += "\n\n" + figurine_module
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
            if reference_model and any(
                plan.get("kind") == "Picture" and plan.get("role") == "storyboard"
                for plan in reference_model["label_plan"].values()
            ):
                system_prompt += (
                    " For storyboard evidence, PANEL_SEQUENCE is the mandatory ordered camera-and-action plan: "
                    "carry every panel record into detailed_description, preserving distinct framing states and "
                    "converting their boundaries into continuous physical camera travel inside each configured Shot."
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
        if _allows_frame_continuity_subjects(reference_model["label_plan"]):
            label_instruction = (
                "\nThese Picture labels are locked. Before them, add only a minimal sequential set of Subject "
                "labels for people, persistent objects, and environments visibly recurring across at least two "
                "Pictures; cite every supporting Picture in each definition. Mention all resulting labels in "
                "summary and retention_analysis and apply them throughout detailed_description."
            )
        else:
            label_instruction = (
                "\nDefine exactly these output labels in this order. Source labels that are not output labels may "
                "appear only as provenance inside subject_definitions. Mention every output label in summary and "
                "retention_analysis, and apply every visual output label in detailed_description."
            )
        system_prompt += "\n\nLOCKED REF2VA LABEL PLAN:\n" + "\n".join(plan_lines) + label_instruction
        system_prompt += _reference_system_modules(result["project"])
    system_prompt += (
        "\n\nOUTPUT: Return only the finished English H3 prompt as plain text with no wrapper, "
        "commentary, or Markdown fence."
    )
    system_prompt += "\n\n" + _single_pass_output_lock(
        mode,
        result["effective_duration"],
        len(expected_shots),
        expected_shots,
        reference_model,
        _input_content_locks(result["project"]),
        _move_output_cues(result["project"], result["effective_duration"]),
    )
    evidence_by_label = {
        item["label"]: item["analysis"] for item in reference_analyses
    }
    user_prompt = build_video_prompt(
        result["project"], result["effective_duration"], evidence_by_label,
    )
    max_new_tokens = _enhance_max_new_tokens(mode, enhance_level)
    estimated_input_tokens = _estimated_mixed_prompt_tokens(system_prompt + "\n" + user_prompt)
    estimated_total_tokens = estimated_input_tokens + max_new_tokens
    if estimated_total_tokens > ENHANCE_CONTEXT_SIZE:
        raise ValueError(
            f"Estimated prompt context ({estimated_input_tokens} input + {max_new_tokens} output tokens) "
            f"exceeds the {ENHANCE_CONTEXT_SIZE}-token Qwen runtime limit. Shorten references, Shot/Move text, "
            "or use a lower Enhance level."
        )
    if progress and estimated_total_tokens > int(ENHANCE_CONTEXT_SIZE * 0.85):
        progress(
            stage="context_warning",
            message=(
                f"Estimated context use is {estimated_total_tokens}/{ENHANCE_CONTEXT_SIZE} tokens; "
                "generation is close to the runtime limit."
            ),
        )
    with _ENHANCE_LOCK, tempfile.TemporaryDirectory(prefix="toyxyz_h3_") as temp_dir:
        check_cancelled()
        if progress:
            progress(stage="model_check", message="Checking the selected GGUF model.")
        model_path = _resolve_enhance_model(model_id or DEFAULT_ENHANCE_MODEL_ID, progress)
        # llama-completion is the stable one-shot frontend in current
        # llama.cpp releases.  The complete Qwen3 turn is rendered below and
        # passed as a file so llama.cpp never enters conversation mode.
        llama_completion = _find_llama_completion(progress)
        user_file = os.path.join(temp_dir, "user.txt")
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
            top_p = 0.95 if strong_enhance else 0.93 if rich_enhance else 0.88
            top_k = 50 if strong_enhance else 40 if rich_enhance else 20
            repeat_penalty = 1.03 if rich_enhance else 1.05
            with open(user_file, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(_render_qwen3_text_prompt(system_prompt, prompt_text))
            command = [
                llama_completion, "-m", model_path, "--file", user_file,
                "--special", "-no-cnv", "-st",
                "--no-display-prompt", "--simple-io", "--no-context-shift",
                "--no-warmup", "--color", "off",
                "-c", str(ENHANCE_CONTEXT_SIZE), "-n", str(max_new_tokens),
                "-ngl", "999", "--temp", str(temperature), "--top-p", str(top_p), "--top-k", str(top_k),
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
                frontend = os.path.basename(command[0]) or "llama.cpp frontend"
                raise RuntimeError(f"{frontend} exited with code {completed.returncode}.\n{tail}")
            return _clean_llm_output(completed.stdout)

        # Enhancement is intentionally single-pass. Keep generated prose as-is;
        # REF2VA receives only deterministic retention-prefix repair.
        enhanced = run_generation(
            user_prompt,
            temperature=0.48 if strong_enhance else 0.38 if rich_enhance else 0.22,
        )
        if not enhanced:
            raise RuntimeError("The selected model returned an empty prompt.")
        if mode == "REF2VA" and reference_model:
            enhanced = _enforce_retention_line_plan(enhanced, reference_model["label_plan"])
            enhanced = _enforce_reference_definition_provenance(enhanced, reference_model)
        enhanced = _enforce_framing_body_range(enhanced)
        enhanced = _enforce_move_camera_continuity(
            enhanced, result["project"], result["effective_duration"],
        )
        enhanced = _enforce_ref_frame_anchor_timing(
            enhanced, result["project"], result["effective_duration"],
        )
        if progress:
            progress(stage="complete", message="Prompt generation completed.")
        return {
            "enhanced_prompt": enhanced,
            "model": model_id,
            "model_path": model_path,
            "reference_analyses": reference_analyses,
            "raw_model_prompt": _format_raw_model_prompt(system_prompt, user_prompt),
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
    frame_entries: list[dict[str, Any]] = []
    has_frame_references = any(
        reference.get("role") == "frame" for reference in pictures[:MAX_REF_IMAGES]
    )
    for reference in pictures[:MAX_REF_IMAGES]:
        loaded_image = None
        if not reference.get("image_filename"):
            outputs.append(blank.clone())
        else:
            try:
                loaded_image = _load_reference_image_tensor(reference)
                outputs.append(loaded_image)
            except (FileNotFoundError, OSError, ValueError):
                outputs.append(blank.clone())
        if reference.get("role") == "frame":
            frame_index = min(
                max(0, int(reference.get("frame_index", 0))),
                max(0, target_frame_count - 1),
            )
            if loaded_image is not None:
                frame_entries.append({"image": loaded_image, "frame_idx": frame_index})
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
    if has_frame_references:
        outputs.append({"type": "minimax_h3_frames", "frames": frame_entries})
    total_media_outputs = MAX_REF_IMAGES + MAX_REF_VIDEOS + MAX_REF_AUDIOS + 1
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
        MAX_REF_IMAGES + MAX_REF_VIDEOS + MAX_REF_AUDIOS + 1
    )
    RETURN_NAMES = (
        "generated_prompt",
        "length",
    ) + tuple(f"image_{index}" for index in range(MAX_REF_IMAGES)) + tuple(
        f"video_{index}" for index in range(1, MAX_REF_VIDEOS + 1)
    ) + tuple(f"audio_{index}" for index in range(1, MAX_REF_AUDIOS + 1)) + ("frames",)
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
                    "llm_prompt": result["llm_prompt"],
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
