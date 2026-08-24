import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "MinimaxH3Prompter";
const ENDPOINT = "/toyxyz/minimax_h3_prompter/compile";
const MODELS_ENDPOINT = "/toyxyz/minimax_h3_prompter/models";
const ENHANCE_ENDPOINT = "/toyxyz/minimax_h3_prompter/enhance";
const ENHANCE_STATUS_ENDPOINT = "/toyxyz/minimax_h3_prompter/enhance/status";
const ENHANCE_CANCEL_ENDPOINT = "/toyxyz/minimax_h3_prompter/enhance/cancel";
const VIDEO_UPLOAD_ENDPOINT = "/toyxyz/minimax_h3_prompter/upload-video";
const VIDEO_VIEW_ENDPOINT = "/toyxyz/minimax_h3_prompter/video";
const VIDEO_UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024;
const DEFAULT_ENHANCE_MODEL = "hf:JonathanColetti/Qwen3.8-27B-Uncensored-GGUF/Qwen3.8-27B-Uncensored-Q4_K_M.gguf";
const DEFAULT_MODEL_BUNDLE = "hf:JonathanColetti/Qwen3.8-27B-Uncensored-GGUF/Q4_K_M+vision-f16";
const LEGACY_LIGHTX2V_MODEL = "hf:lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B";
const LIGHTX2V_MODEL = "hf:indhic-ai/MiniMax_H3-Prompt_Rewriter-8B-LORA-Merged-GGUF/Q8_0+vision-f16";
// Includes the model selector/enhancement controls without forcing the whole
// editor into a nested scrollbar. Long reference lists and prompt previews
// retain their own scoped scroll areas.
const UI_HEIGHT = 900;
const UI_WIDTH = 1380;
const NODE_HEIGHT = UI_HEIGHT + 95;
const MIN_SHOT_DURATION = 0.25;
const VIDEO_OUTPUT_FPS = 24;
const MIN_VIDEO_CLIP_FRAMES = 10;
const MIN_VIDEO_CLIP_DURATION = MIN_VIDEO_CLIP_FRAMES / VIDEO_OUTPUT_FPS;
const CURRENT_PROJECT_VERSION = 22;
const MAX_REFERENCES = { picture: 9, video: 3, audio: 3, total: 12 };
const SHOT_SNAP_SECONDS = 0.05;
const MODES = ["AUTO", "T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"];
const LEGACY_DIALOGUE_LANGUAGES = [
  "Arabic", "Chinese", "English", "French", "German", "Italian",
  "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
];
const REFERENCE_ROLES = {
  picture: ["first_frame", "last_frame", "frame", "subject_identity"],
  video: ["none", "video_editing", "video_continuation", "motion", "camera", "cuts_rhythm"],
  audio: [
    "none", "full_signal_copy", "partial_signal_copy", "voice_delivery",
    "dialogue_lyrics", "sound_ambience", "music_rhythm",
  ],
};
const REFERENCE_TYPE_LABELS = { picture: "Image", video: "Video", audio: "Audio" };
const SUBJECT_STRENGTHS = ["weak", "normal", "strong"];
const SUBJECT_STRENGTH_LABELS = { weak: "Weak", normal: "Normal", strong: "Strong" };
const REFERENCE_ROLE_LABELS = {
  first_frame: "First frame",
  last_frame: "Last frame",
  frame: "Frame",
  subject_identity: "Subject",
  none: "None",
  video_editing: "Video editing",
  video_continuation: "Video continuation",
  motion: "Motion / action timing",
  camera: "Camera movement",
  cuts_rhythm: "Cuts / rhythm / temporal structure",
  full_signal_copy: "Full audio reuse",
  partial_signal_copy: "Partial audio reuse",
  voice_delivery: "Voice / delivery",
  dialogue_lyrics: "Dialogue / lyrics reuse",
  sound_ambience: "Sound / ambience",
  music_rhythm: "Music / rhythm",
};
const REFERENCE_ROLE_HELP = {
  first_frame: "The picture strictly anchors the opening frame.",
  last_frame: "The picture strictly anchors the final frame.",
  frame: "The picture anchors an exact movable frame on the target timeline.",
  subject_identity: "Use the image as a Subject; choose Weak, Normal, or Strong preservation separately.",
  none: "No preset relationship. Describe the intended use freely below.",
  video_editing: "Directly edit the source video while preserving the source elements specified below.",
  video_continuation: "Continue naturally from the ending state of the source video.",
  motion: "Reference the source video's subject motion, action timing, and movement rhythm only.",
  camera: "Reference the source video's camera movement and viewpoint behavior only.",
  cuts_rhythm: "Reference the source video's cuts, pacing, rhythm, and temporal structure only.",
  full_signal_copy: "Reuse the complete source audio signal for the available target duration.",
  partial_signal_copy: "Reuse only the source interval or audio layers specified below.",
  voice_delivery: "Reference voice timbre, accent, emotion, pace, and delivery without copying words.",
  dialogue_lyrics: "Reuse exact supplied or transcribed dialogue or lyrics without inventing unavailable words.",
  sound_ambience: "Reference described sound effects, ambience, room tone, and acoustic character.",
  music_rhythm: "Reference described instrumentation, tempo, beat, rhythm, dynamics, or structure.",
};

const AUDIO_DESCRIPTION_PLACEHOLDERS = {
  none: "Describe how this audio should guide the target in English",
  full_signal_copy: "Describe where the complete source audio should be used",
  partial_signal_copy: "Specify the interval or audio layers to reuse",
  voice_delivery: "Specify the target speaker and voice or delivery traits to reference",
  dialogue_lyrics: "Provide the exact dialogue or lyrics and where they should occur",
  sound_ambience: "Describe the sound effects, ambience, timing, or acoustic character to reference",
  music_rhythm: "Describe the instrumentation, tempo, rhythm, dynamics, or structure to reference",
};

const DEFAULT_PROJECT = () => ({
  version: CURRENT_PROJECT_VERSION,
  mode: "AUTO",
  requested_duration: 5,
  user_request: "",
  shots: [{
    id: crypto.randomUUID?.() || `shot-${Date.now()}`,
    duration: 5,
    visual_action: "",
  }],
  references: [],
  constraints: "",
  verbatim_content: "",
  enhance_model: DEFAULT_ENHANCE_MODEL,
  image_model: DEFAULT_MODEL_BUNDLE,
  auto_run: false,
  enhance: false,
  enhanced_prompt: "",
});

function hideWidget(widget) {
  if (!widget) return;
  widget.hidden = true;
  widget.options ||= {};
  widget.options.hidden = true;
  if (!window.LiteGraph || !window.LiteGraph.vueNodesMode) {
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
  }
  if (widget.element) widget.element.style.display = "none";
}

function uid(prefix) {
  return crypto.randomUUID?.() || `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function clampNumber(value, fallback, min = -Infinity, max = Infinity) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.min(max, Math.max(min, number)) : fallback;
}

function normalizeAlias(value) {
  const alias = String(value || "").trim().replace(/^@+/, "")
    .replace(/\s+/g, "_").replace(/[^\p{L}\p{N}_-]/gu, "");
  return alias ? `@${alias}` : "";
}

function inferAutoMode(references) {
  if (!references.length) return "T2VA";
  const signature = references.map(ref => `${ref.type}:${ref.role}`);
  if (signature.length === 1 && signature[0] === "picture:first_frame") return "I2VA";
  if (signature.length === 2
      && signature[0] === "picture:first_frame"
      && signature[1] === "picture:last_frame") return "FL2VA";
  if (signature.length === 1 && signature[0] === "picture:last_frame") return "L2VA";
  return "REF2VA";
}

function fitShotDurations(shots, requestedDuration) {
  if (!shots.length) return;
  const target = Math.max(requestedDuration, shots.length * MIN_SHOT_DURATION);
  const distributable = target - shots.length * MIN_SHOT_DURATION;
  const weights = shots.map(shot => Math.max(0, Number(shot.duration) - MIN_SHOT_DURATION));
  const weightTotal = weights.reduce((sum, value) => sum + value, 0);
  shots.forEach((shot, index) => {
    const share = weightTotal > 0 ? weights[index] / weightTotal : 1 / shots.length;
    shot.duration = MIN_SHOT_DURATION + distributable * share;
  });
}

function alignedFrameCount(seconds) {
  let frames = Math.max(5, Math.round(Number(seconds || 0) * VIDEO_OUTPUT_FPS));
  while (frames % 17 !== 5) frames += 1;
  return frames;
}

function migrateLegacyShotContent(shot, index = 0) {
  const parts = [String(shot?.visual_action || "").trim()].filter(Boolean);
  let dialogue = String(shot?.dialogue || "").trim();
  if (dialogue) {
    let language = LEGACY_DIALOGUE_LANGUAGES.includes(String(shot?.dialogue_language || ""))
      ? String(shot.dialogue_language) : "English";
    const wrapped = dialogue.match(/^<d>\s*(?:\[([^\]\r\n]+)\]\s*)?([\s\S]*?)\s*<\/d>$/i);
    if (wrapped) {
      if (LEGACY_DIALOGUE_LANGUAGES.includes(wrapped[1])) language = wrapped[1];
      dialogue = wrapped[2];
    } else {
      dialogue = dialogue.replace(/<\/?d>/gi, "").trim();
    }
    const speaker = /^S[1-6]$/i.test(String(shot?.dialogue_speaker || ""))
      ? String(shot.dialogue_speaker).toUpperCase() : "S1";
    const delivery = String(shot?.dialogue_delivery || "").trim() || "The on-screen speaker";
    const mode = ["spoken", "voiceover", "singing"].includes(String(shot?.dialogue_mode || "").toLowerCase())
      ? String(shot.dialogue_mode).toLowerCase() : "spoken";
    if (mode === "voiceover") {
      parts.push(`${delivery} (${speaker}) says in an off-screen voiceover: <d>[${language}] ${dialogue}</d> while the corresponding on-screen character's lips remain completely closed.`);
    } else if (mode === "singing") {
      parts.push(`${delivery} (${speaker}) sings: <d>[${language}] ${dialogue}</d>`);
    } else {
      parts.push(`${delivery} (${speaker}) says: <d>[${language}] ${dialogue}</d>`);
    }
  }
  const visibleText = String(shot?.visible_text || "").trim();
  if (visibleText) {
    const escaped = visibleText.replaceAll("\\", "\\\\").replaceAll('"', '\\"')
      .replaceAll("\r", " ").replaceAll("\n", " ");
    parts.push(`A visible on-screen text element reads "${escaped}".`);
  }
  const legacySound = String(shot?.diegetic_sound || "").trim();
  if (legacySound) parts.push(`Synchronized physical sound: ${legacySound}`);
  const framing = String(shot?.camera_framing || "").trim();
  const angle = String(shot?.camera_angle || "").trim();
  const motion = String(shot?.camera_motion || "").trim();
  if (framing) parts.push(`Camera framing: ${framing}.`);
  if (angle) parts.push(`Camera angle: ${angle}.`);
  if (motion) parts.push(`Camera motion: ${motion}.`);
  const transition = String(shot?.transition || "").trim().toLowerCase();
  if (index > 0 && ["cross-dissolve", "fade", "wipe"].includes(transition)) {
    parts.push(`Transition into this shot with a ${transition}.`);
  }
  return parts.join("\n");
}

function normalizeProject(value) {
  let raw = value;
  if (typeof raw === "string") {
    try { raw = JSON.parse(raw); } catch { raw = {}; }
  }
  if (!raw || typeof raw !== "object") raw = {};
  const project = DEFAULT_PROJECT();
  project.mode = MODES.includes(String(raw.mode || "").toUpperCase()) ? String(raw.mode).toUpperCase() : "AUTO";
  project.user_request = String(raw.user_request || "");
  project.constraints = String(raw.constraints || "");
  project.verbatim_content = String(raw.verbatim_content || "");
  const savedEnhanceModel = String(raw.enhance_model || "") === LEGACY_LIGHTX2V_MODEL
    ? LIGHTX2V_MODEL : String(raw.enhance_model || "");
  const savedImageModel = String(raw.image_model || "") === LEGACY_LIGHTX2V_MODEL
    ? LIGHTX2V_MODEL : String(raw.image_model || "");
  project.enhance_model = [DEFAULT_ENHANCE_MODEL, LIGHTX2V_MODEL].includes(savedEnhanceModel)
    ? savedEnhanceModel : DEFAULT_ENHANCE_MODEL;
  project.image_model = [DEFAULT_MODEL_BUNDLE, LIGHTX2V_MODEL].includes(savedImageModel)
    ? savedImageModel : DEFAULT_MODEL_BUNDLE;
  project.auto_run = raw.auto_run === true;
  project.enhance = raw.enhance === true;
  project.enhanced_prompt = String(raw.enhanced_prompt || "");
  if (Array.isArray(raw.shots) && raw.shots.length) {
    project.shots = raw.shots.map((shot, index) => ({
      id: String(shot?.id || uid(`shot-${index + 1}`)),
      duration: clampNumber(shot?.duration, 1, MIN_SHOT_DURATION, 60),
      visual_action: migrateLegacyShotContent(shot, index),
    }));
  } else {
    project.shots[0].duration = clampNumber(raw.requested_duration, 5, 0.1, 60);
  }
  const legacyAudio = [];
  if (String(raw.overall_soundscape || "").trim()) {
    legacyAudio.push(`Overall soundscape: ${String(raw.overall_soundscape).trim()}`);
  }
  if (String(raw.non_diegetic_music || "").trim()) {
    legacyAudio.push(`Non-diegetic music: ${String(raw.non_diegetic_music).trim()}`);
  }
  if (legacyAudio.length) {
    project.shots[0].visual_action = [project.shots[0].visual_action, ...legacyAudio].filter(Boolean).join("\n");
  }
  project.references = Array.isArray(raw.references) ? raw.references.map((ref, index) => {
    const type = ["picture", "video", "audio"].includes(ref?.type) ? ref.type : "picture";
    const suppliedRole = String(ref?.role || "").toLowerCase();
    let role = REFERENCE_ROLES[type].includes(suppliedRole) ? suppliedRole : "reference";
    let strength = String(ref?.strength || "").toLowerCase();
    if (type === "picture") {
      if (["reference", "environment", "style", "storyboard"].includes(suppliedRole)) {
        role = "subject_identity"; strength = "weak";
      } else if (suppliedRole === "subject_identity") {
        role = "subject_identity";
        strength = SUBJECT_STRENGTHS.includes(strength) ? strength : "strong";
      } else if (!["first_frame", "last_frame", "frame"].includes(suppliedRole)) {
        role = "subject_identity";
        strength = SUBJECT_STRENGTHS.includes(strength) ? strength : "normal";
      } else {
        role = suppliedRole; strength = "normal";
      }
    } else if (type === "video") {
      const legacyVideoRoles = {
        reference: "none", continuation: "video_continuation", pacing: "cuts_rhythm",
      };
      role = REFERENCE_ROLES.video.includes(suppliedRole)
        ? suppliedRole : (legacyVideoRoles[suppliedRole] || "none");
    } else if (type === "audio") {
      const legacyAudioRoles = {
        reference: "none", voice_timbre: "voice_delivery", dialogue: "dialogue_lyrics",
        music_style: "music_rhythm", sound_effect: "sound_ambience",
        signal_copy: "partial_signal_copy",
      };
      role = REFERENCE_ROLES.audio.includes(suppliedRole)
        ? suppliedRole : (legacyAudioRoles[suppliedRole] || "none");
    }
    return {
      id: String(ref?.id || uid(`ref-${index + 1}`)),
      type,
      role,
      strength: SUBJECT_STRENGTHS.includes(strength) ? strength : "normal",
      alias: normalizeAlias(ref?.alias),
      // Image analysis belongs only to the active Enhance request. Version 7
      // persisted it here, which made Raw Prompt change after enhancement.
      description: type === "picture" ? "" : String(ref?.description || ""),
      duration: clampNumber(ref?.duration, 0, 0, 60),
      source_duration: clampNumber(ref?.source_duration, clampNumber(ref?.duration, 0, 0, 60), 0, 36000),
      trim_start: clampNumber(ref?.trim_start, 0, 0, 36000),
      timeline_start: clampNumber(ref?.timeline_start, 0, -36000, 36000),
      frame_index: Math.round(clampNumber(ref?.frame_index, 0, 0, 1000000)),
      image_filename: String(ref?.image_filename || ""),
      image_subfolder: String(ref?.image_subfolder || "").replaceAll("\\", "/").replace(/^\/+|\/+$/g, ""),
      image_type: "input",
      video_filename: String(ref?.video_filename || ""),
      video_subfolder: String(ref?.video_subfolder || "").replaceAll("\\", "/").replace(/^\/+|\/+$/g, ""),
      video_type: "input",
      audio_filename: String(ref?.audio_filename || ""),
      audio_subfolder: String(ref?.audio_subfolder || "").replaceAll("\\", "/").replace(/^\/+|\/+$/g, ""),
      audio_type: "input",
    };
  }) : [];
  const shotTotal = project.shots.reduce((sum, shot) => sum + shot.duration, 0);
  const requestedDuration = clampNumber(
    raw.requested_duration,
    shotTotal || 5,
    project.shots.length * MIN_SHOT_DURATION,
    60,
  );
  if (shotTotal > 0 && Math.abs(shotTotal - requestedDuration) > 0.0005) {
    fitShotDurations(project.shots, requestedDuration);
  }
  project.requested_duration = requestedDuration;
  project.references.filter(ref => ref.type === "video").forEach(ref => {
    const sourceDuration = Math.max(ref.source_duration || ref.duration || 0, ref.duration || 0);
    ref.source_duration = sourceDuration;
    ref.trim_start = Math.min(Math.max(0, ref.trim_start || 0), Math.max(0, sourceDuration - MIN_SHOT_DURATION));
    const available = Math.max(0, sourceDuration - ref.trim_start);
    ref.duration = Math.min(Math.max(0, ref.duration || Math.min(15, available)), available, 15);
    const minimumVisible = Math.min(MIN_VIDEO_CLIP_DURATION, ref.duration);
    ref.timeline_start = Math.min(
      Math.max(-ref.duration + minimumVisible, ref.timeline_start || 0),
      requestedDuration - minimumVisible,
    );
  });
  return project;
}

function installStyles() {
  if (document.getElementById("mmh3-prompter-styles")) return;
  const style = document.createElement("style");
  style.id = "mmh3-prompter-styles";
  style.textContent = `
    .mmh3p { --bg:#181a1d; --panel:#202329; --line:#383d46; --soft:#2b3038; --text:#e4e7eb;
      --muted:#949ba7; --accent:#65b9ff; --warn:#e7b35c; --error:#ff7272; color:var(--text);
      width:100%; height:100%; min-height:0; box-sizing:border-box; display:grid;
      grid-template-rows:minmax(0,1fr) 92px; gap:8px;
      padding:8px; background:var(--bg); font:12px Inter,system-ui,sans-serif; overflow:hidden;
      position:relative; }
    .mmh3p * { box-sizing:border-box; }
    .mmh3p-workspace { width:100%; height:100%; min-height:0; display:grid;
      grid-template-columns:minmax(760px,1fr) minmax(380px,.44fr); gap:8px; }
    .mmh3p-main { min-width:0; min-height:0; display:flex; flex-direction:column; gap:8px; overflow:hidden; }
    .mmh3p-main > .mmh3p-panel { flex:0 0 auto; }
    .mmh3p-main > .mmh3p-grid { flex:1 1 0; min-height:0; }
    .mmh3p-row { display:flex; gap:7px; align-items:center; min-width:0; }
    .mmh3p-grow { flex:1; min-width:0; }
    .mmh3p-panel { background:var(--panel); border:1px solid var(--line); border-radius:7px; padding:8px; }
    .mmh3p-label { color:var(--muted); font-size:10px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
    .mmh3p input,.mmh3p select,.mmh3p textarea,.mmh3p button { color:var(--text); background:var(--soft);
      border:1px solid #464c57; border-radius:4px; font:inherit; outline:none; }
    .mmh3p input,.mmh3p select { height:27px; padding:3px 7px; }
    .mmh3p textarea { width:100%; padding:7px; resize:vertical; min-height:55px; line-height:1.35; }
    .mmh3p button { min-height:27px; padding:4px 9px; cursor:pointer; }
    .mmh3p button:hover { border-color:var(--accent); }
    .mmh3p button.active { color:#06131d; background:var(--accent); border-color:var(--accent); }
    .mmh3p-top { flex-wrap:wrap; }
    .mmh3p-badge { padding:4px 7px; background:#15171a; border:1px solid var(--line); border-radius:12px;
      color:var(--muted); font:11px ui-monospace,Consolas,monospace; white-space:nowrap; }
    .mmh3p-badge.error { color:var(--error); border-color:#713b43; background:#211518; }
    .mmh3p-toolbar { justify-content:space-between; }
    .mmh3p-timeline { height:112px; display:flex; align-items:stretch; gap:3px; padding-top:18px;
      position:relative; overflow-x:auto; }
    .mmh3p-ruler { position:absolute; left:0; right:0; top:0; color:#707784; font-size:9px;
      display:flex; justify-content:space-between; pointer-events:none; }
    .mmh3p-shot { min-width:100px; position:relative; padding:7px; border:1px solid #46505c;
      border-radius:5px; background:#182630; overflow:hidden; cursor:pointer; display:flex; flex-direction:column; gap:5px; }
    .mmh3p-shot.selected { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent) inset; }
    .mmh3p-shot.dragging { opacity:.45; }
    .mmh3p-shot-title { font-weight:700; color:#d9efff; white-space:nowrap; }
    .mmh3p-shot-summary { color:#aeb8c4; font-size:10px; overflow:hidden; line-height:1.25; }
    .mmh3p-shot-duration { margin-left:auto; padding:2px 6px; border-radius:10px; color:#a9d8ff;
      background:#111a21; border:1px solid #364653; font:9px ui-monospace,Consolas,monospace;
      white-space:nowrap; flex:0 0 auto; }
    .mmh3p-resize-handle { position:absolute; z-index:5; top:0; right:0; width:11px; height:100%;
      cursor:col-resize; touch-action:none; }
    .mmh3p-resize-handle::after { content:""; position:absolute; top:8px; bottom:8px; left:5px;
      width:2px; border-radius:2px; background:#65b9ff; opacity:.45; }
    .mmh3p-resize-handle:hover::after,.mmh3p-resize-handle.active::after { opacity:1; width:3px; }
    .mmh3p.resizing,.mmh3p.resizing * { cursor:col-resize !important; user-select:none !important; }
    .mmh3p-video-timeline { display:flex; flex-direction:column; gap:4px; margin-top:6px; }
    .mmh3p-video-timeline[hidden] { display:none; }
    .mmh3p-image-track { height:96px; position:relative; width:100%; }
    .mmh3p-image-lane { height:92px; position:relative; width:100%; overflow:hidden; border:1px solid #3d4651;
      border-radius:4px; background:repeating-linear-gradient(90deg,#171b20 0,#171b20 calc(10% - 1px),#2b323a 10%); }
    .mmh3p-image-anchor { position:absolute; top:1px; bottom:1px; min-width:2px; display:flex;
      align-items:stretch; overflow:visible; border:1px solid #d8b5ff; background:#9a58d1; }
    .mmh3p-image-anchor.first { left:1px; } .mmh3p-image-anchor.last { right:1px; }
    .mmh3p-image-anchor.frame { transform:translateX(-50%); cursor:grab; touch-action:none; }
    .mmh3p-image-anchor.frame:active { cursor:grabbing; }
    .mmh3p-image-anchor.frame::after { content:""; position:absolute; z-index:3; top:0; bottom:0; left:-5px; right:-5px; }
    .mmh3p-image-anchor-label { position:absolute; z-index:2; left:calc(100% + 5px); top:50%; display:flex;
      flex-direction:column; align-items:flex-start; gap:3px; padding:3px 5px; white-space:nowrap;
      border-radius:3px; transform:translateY(-50%);
      color:#f0ddff; background:rgba(12,10,18,.78); font:9px ui-monospace,Consolas,monospace; pointer-events:none; }
    .mmh3p-image-anchor-label.before { left:auto; right:calc(100% + 5px); }
    .mmh3p-image-anchor-preview { width:64px !important; height:44px !important; object-fit:contain !important; object-position:center;
      border:1px solid rgba(197,140,255,.65); border-radius:2px; background:#0b0f14; }
    .mmh3p-video-track { height:56px; position:relative; width:100%; }
    .mmh3p-video-track-label { position:absolute; z-index:6; left:6px; top:6px; max-width:68px; padding:2px 5px;
      color:#d9efff; background:rgba(10,18,24,.78); border-radius:3px; font:10px ui-monospace,Consolas,monospace;
      white-space:nowrap; overflow:hidden; text-overflow:ellipsis; pointer-events:none; }
    .mmh3p-video-lane { height:52px; position:relative; width:100%; overflow:hidden; border:1px solid #3d4651;
      border-radius:4px; background:repeating-linear-gradient(90deg,#171b20 0,#171b20 calc(10% - 1px),#2b323a 10%); }
    .mmh3p-video-canvas { position:relative; height:100%; min-width:100%; }
    .mmh3p-video-filmstrip { position:absolute; inset:0; z-index:0; display:flex; overflow:hidden; pointer-events:none;
      border-radius:2px; background:#111820; opacity:.9; }
    .mmh3p-video-thumb { height:100%; min-width:0; flex:1 1 0; object-fit:contain; object-position:center;
      background:#0b0f14; border-right:1px solid rgba(255,255,255,.16); }
    .mmh3p-video-filmstrip-loading { width:100%; display:flex; align-items:center; justify-content:center;
      color:#8e9aa7; font:9px ui-monospace,Consolas,monospace; }
    .mmh3p-video-clip { position:absolute; z-index:3; top:1px; bottom:1px; min-width:8px; border:2px solid #65b9ff;
      border-radius:3px; background:#214765; color:#fff; cursor:grab; user-select:none;
      display:flex; align-items:center; justify-content:center; padding:0 10px; font:9px ui-monospace,Consolas,monospace; white-space:nowrap; }
    .mmh3p-video-clip > span { position:absolute; top:50%; z-index:4; padding:2px 5px; border-radius:3px;
      background:rgba(8,14,19,.68); pointer-events:none; transform:translate(-50%,-50%); }
    .mmh3p-video-clip:active { cursor:grabbing; }
    .mmh3p-video-trim { position:absolute; z-index:5; top:0; bottom:0; width:8px; cursor:col-resize; background:#78c5ff; opacity:.75; }
    .mmh3p-video-trim.left { left:0; } .mmh3p-video-trim.right { right:0; }
    .mmh3p-video-trim:hover { opacity:1; }
    .mmh3p-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; min-height:0;
      align-items:stretch; overflow:hidden; }
    .mmh3p-grid > .mmh3p-panel { min-height:0; height:100%; overflow:hidden; }
    .mmh3p-editor { display:flex; flex-direction:column; gap:6px; min-height:0; overflow:hidden; }
    .mmh3p-visual-action-field { flex:1 1 auto; min-height:0; display:flex; flex-direction:column; }
    .mmh3p-visual-action-field .mmh3p-mention-wrap { flex:1 1 auto; height:auto; min-height:0; }
    .mmh3p-visual-action-field .mmh3p-mention-wrap textarea { height:100%; min-height:0; resize:none; overflow:auto; }
    .mmh3p-field { display:flex; flex-direction:column; gap:3px; }
    .mmh3p-field span { color:var(--muted); font-size:10px; }
    .mmh3p-editor textarea { min-height:44px; }
    .mmh3p-mention-wrap { position:relative; width:100%; min-height:58px; background:var(--soft);
      border:1px solid #464c57; border-radius:4px; overflow:visible; }
    .mmh3p-mention-wrap:focus-within { border-color:var(--accent); }
    .mmh3p-mention-wrap textarea { position:relative; z-index:2; display:block; width:100%; min-height:56px;
      margin:0; border:0; border-radius:0; background:transparent; resize:vertical; caret-color:var(--text); }
    .mmh3p-mention-backdrop { position:absolute; z-index:1; inset:0; padding:7px; overflow:hidden;
      color:transparent; white-space:pre-wrap; overflow-wrap:anywhere; pointer-events:none; line-height:1.35; }
    .mmh3p-mention-backdrop mark { color:transparent; background:rgba(101,185,255,.28);
      border:1px solid rgba(101,185,255,.75); border-radius:4px; box-shadow:0 0 0 1px rgba(9,31,47,.45); }
    .mmh3p-caret-mirror { position:absolute; z-index:-1; left:0; top:0; visibility:hidden;
      pointer-events:none; white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word; }
    .mmh3p-caret-marker { display:inline; }
    .mmh3p-mention-menu { position:absolute; z-index:50; display:none; left:0; top:100%;
      width:min(430px,100%); min-width:260px;
      max-height:190px; overflow:auto; padding:4px; background:#15191e; border:1px solid #53606e;
      border-radius:6px; box-shadow:0 8px 24px rgba(0,0,0,.5); }
    .mmh3p-mention-menu.open { display:block; }
    .mmh3p-mention-item { display:flex; align-items:center; gap:8px; width:100%; border:0 !important;
      background:transparent !important; text-align:left; padding:7px 8px !important; }
    .mmh3p-mention-item:hover,
    .mmh3p-mention-item.active { background:#263746 !important; }
    .mmh3p-mention-token { color:#9bd5ff; font:11px ui-monospace,Consolas,monospace; font-weight:700; }
    .mmh3p-mention-meta { color:var(--muted); font-size:10px; overflow:hidden; text-overflow:ellipsis;
      white-space:nowrap; }
    .mmh3p-mention-empty { color:var(--muted); padding:8px; }
    .mmh3p-reference-head { margin-bottom:8px; display:flex; flex-wrap:wrap; align-items:flex-start; justify-content:space-between; gap:7px 10px; }
    .mmh3p-reference-title { min-width:0; }
    .mmh3p-enhance-actions { margin-left:auto; display:flex; align-items:center; gap:7px; }
    .mmh3p-enhance-button {
      box-sizing:border-box; width:112px; min-width:112px; max-width:112px; flex:0 0 112px;
      white-space:nowrap; text-align:center;
    }
    .mmh3p-auto-run { display:flex; align-items:center; gap:5px; color:var(--muted); font-size:10px;
      white-space:nowrap; cursor:pointer; user-select:none; }
    .mmh3p-auto-run input { width:14px; height:14px; margin:0; accent-color:var(--accent); cursor:pointer; }
    .mmh3p-auto-run:has(input:checked) { color:#a9d8ff; }
    .mmh3p-reference-actions { flex:0 1 auto; display:flex; flex-wrap:wrap; justify-content:flex-end; align-items:center; gap:5px; }
    .mmh3p-reference-actions button { white-space:nowrap; }
    .mmh3p-reference-help { margin-top:2px; color:var(--muted); font-size:9px; line-height:1.3; }
    .mmh3p-references-panel { display:flex; flex-direction:column; min-height:0; overflow:hidden; }
    .mmh3p-reference-list { flex:1 1 auto; height:auto; min-height:0; overflow-y:auto; overflow-x:hidden;
      display:flex; flex-direction:column; gap:6px; }
    .mmh3p-ref { box-sizing:border-box; width:100%; min-width:0; display:grid; grid-template-columns:96px minmax(0,1fr); gap:9px; align-items:stretch;
      padding:7px; background:#191c20; border:1px solid #343944; border-radius:6px; }
    .mmh3p-ref-label { grid-column:1/-1; display:block; min-width:0; padding:0 2px 5px;
      border-bottom:1px solid #303640; color:#a9d8ff;
      font:600 10px ui-monospace,Consolas,monospace; white-space:nowrap; }
    .mmh3p-ref input,.mmh3p-ref select { box-sizing:border-box; width:100%; min-width:0; max-width:100%; font-size:10px; height:25px; }
    .mmh3p-ref textarea { box-sizing:border-box; width:100%; max-width:100%; min-height:48px; resize:vertical; font-size:10px; }
    .mmh3p-ref-preview { min-width:0; height:96px; border:1px solid #3e4651; border-radius:5px;
      background:#111419; overflow:hidden; display:flex; align-items:center; justify-content:center; color:var(--muted);
      font-size:9px; text-align:center; cursor:pointer; }
    .mmh3p-ref-preview:hover { border-color:var(--accent); color:#cce9ff; }
    .mmh3p-ref-preview img,.mmh3p-ref-preview video { width:100%; height:100%; object-fit:contain; object-position:center; display:block; }
    .mmh3p-ref-preview audio { width:92%; max-width:100%; }
    .mmh3p-ref-body { min-width:0; display:flex; flex-direction:column; gap:5px; }
    .mmh3p-ref.picture .mmh3p-ref-body { justify-content:center; }
    .mmh3p-ref-controls { min-width:0; display:grid; grid-template-columns:minmax(130px,1fr) 30px;
      gap:5px; align-items:center; }
    .mmh3p-ref-controls.metadata-alias { grid-template-columns:minmax(130px,1fr) minmax(80px,.65fr) 30px; }
    .mmh3p-ref-controls.video-metadata { grid-template-columns:minmax(0,1.4fr) minmax(0,.7fr) 30px; }
    .mmh3p-subject-strength-row { display:grid; grid-template-columns:auto minmax(72px,.7fr) minmax(90px,1fr); gap:7px;
      align-items:center; padding-top:1px; }
    .mmh3p-subject-strength-row span { color:var(--muted); font-size:9px; font-weight:700;
      letter-spacing:.04em; text-transform:uppercase; }
    .mmh3p-ref-controls button { white-space:nowrap; font-size:10px; }
    .mmh3p-ref .delete { color:#ff9797; padding:2px; }
    .mmh3p-preview-column { min-width:0; min-height:0; height:100%; display:grid;
      grid-template-rows:minmax(0,1fr); }
    .mmh3p-preview-panel { min-width:0; min-height:0; display:flex; flex-direction:column; }
    .mmh3p-preview { flex:1 1 auto; height:auto; min-height:0; max-height:none; overflow:auto; margin:0; padding:8px;
      white-space:pre-wrap; word-break:break-word; background:#121416; border:1px solid #30353d;
      border-radius:5px; color:#ccd2da; font:10px/1.42 ui-monospace,Consolas,monospace; }
    .mmh3p-preview-head { justify-content:space-between; margin-bottom:6px; }
    .mmh3p-log-panel { min-width:0; min-height:0; display:flex; flex-direction:column; }
    .mmh3p-log { flex:1; min-height:0; overflow:auto; margin-top:6px; padding:6px 8px;
      white-space:pre-wrap; word-break:break-word; color:#a9d1ad; background:#121416;
      border:1px solid #30353d; border-radius:5px; font:10px/1.4 ui-monospace,Consolas,monospace; }
    .mmh3p-log-line.error { color:var(--error); }
    .mmh3p-log-line.warning { color:var(--warn); }
    .mmh3p-log-line.download { color:#8dceff; }
    .mmh3p-log-line.analysis { color:#b9dcff; padding:4px 6px; border-left:2px solid #5798ce; }
    .mmh3p-empty { color:#777f8b; padding:14px; text-align:center; border:1px dashed #3c424c; border-radius:5px; }
    @media (max-width:850px) { .mmh3p-grid { grid-template-columns:1fr; }
      .mmh3p-ref { grid-template-columns:76px minmax(0,1fr); }
      .mmh3p-ref-controls { grid-template-columns:minmax(110px,1fr) 28px; }
      .mmh3p-ref-controls.metadata-alias { grid-template-columns:minmax(110px,1fr) minmax(72px,.6fr) 28px; }
      .mmh3p-ref-controls.video-metadata { grid-template-columns:minmax(0,1.25fr) minmax(0,.65fr) 28px; }
      .mmh3p-subject-strength-row { grid-template-columns:auto minmax(66px,.7fr) minmax(76px,1fr); } }
  `;
  document.head.appendChild(style);
}

class PrompterUI {
  constructor(node, root, stateWidget) {
    this.node = node;
    this.root = root;
    this.stateWidget = stateWidget;
    this.project = normalizeProject(stateWidget?.value);
    this.selectedShotId = this.project.shots[0]?.id;
    this.previewData = null;
    this.autoRunPreview = null;
    this.compileTimer = null;
    this.compileController = null;
    this.enhanceController = null;
    this.enhanceJobId = "";
    this.videoThumbnailCache = new Map();
    this.modelBundles = [];
    this.compileSequence = 0;
    this.mentionSelectionIndex = 0;
    this.mentionMenuSignature = "";
    this.visibleMentionEntries = [];
    this.build();
    this.render();
    this.commit(false);
    this.loadModels();
  }

  build() {
    this.root.className = "mmh3p";
    this.root.innerHTML = `
      <div class="mmh3p-workspace">
      <div class="mmh3p-main">
      <div class="mmh3p-panel mmh3p-row mmh3p-top">
        <span class="mmh3p-label">Mode</span><select data-el="mode"></select>
        <span class="mmh3p-label">Duration</span><input data-el="duration" type="number" min="0.1" max="60" step="0.1" style="width:76px">
        <span class="mmh3p-badge" data-el="effective">calculating</span>
        <span class="mmh3p-badge" data-el="path">fl2va</span>
      </div>
      <div class="mmh3p-panel mmh3p-row">
        <span class="mmh3p-label">Model</span>
        <select class="mmh3p-grow" data-el="model-bundle"><option value="">Loading Qwen3.8 bundle…</option></select>
        <span class="mmh3p-badge" data-el="model-status"></span>
      </div>
      <div class="mmh3p-panel">
        <div class="mmh3p-row mmh3p-toolbar">
          <div class="mmh3p-row"><button data-action="add-shot">+ Shot</button><button data-action="delete-shot">Delete Shot</button></div>
        </div>
        <div class="mmh3p-label" style="margin-top:6px">Drag a shot boundary to resize adjacent shots · double-click to split them evenly</div>
        <div class="mmh3p-timeline" data-el="timeline"><div class="mmh3p-ruler"><span>0.00s · 0f</span><span data-el="ruler-end">5.00s</span></div></div>
        <div class="mmh3p-video-timeline" data-el="video-timeline" hidden></div>
      </div>
      <div class="mmh3p-grid">
        <div class="mmh3p-panel mmh3p-editor">
          <div class="mmh3p-label">Prompt</div>
          <div class="mmh3p-field mmh3p-visual-action-field"><span>Visual / action / camera / dialogue / text / sound / music — type @ to insert a reference alias</span>
            <div class="mmh3p-mention-wrap">
              <div class="mmh3p-mention-backdrop" data-el="visual-action-highlight"></div>
              <textarea data-el="visual-action" spellcheck="true" placeholder="Describe visuals, actions, camera framing or movement, transitions, dialogue, visible text, sound, or music naturally in context."></textarea>
              <div class="mmh3p-caret-mirror" data-el="caret-mirror"></div>
              <div class="mmh3p-mention-menu" data-el="mention-menu" role="listbox"></div>
            </div>
          </div>
        </div>
        <div class="mmh3p-panel mmh3p-preview-panel">
          <div class="mmh3p-row mmh3p-preview-head">
            <span class="mmh3p-label">Generated Prompt</span>
            <div class="mmh3p-enhance-actions">
              <label class="mmh3p-auto-run" title="Use richer Qwen3.8 interpretation and expansion while preserving the requested events">
                <input data-el="enhance" type="checkbox"><span>Enhance</span>
              </label>
              <button class="mmh3p-enhance-button" data-action="enhance" type="button">Generate Prompt</button>
              <label class="mmh3p-auto-run" title="Generate the prompt as part of the ComfyUI queue when this node executes">
                <input data-el="auto-run" type="checkbox"><span>Auto Run</span>
              </label>
            </div>
          </div>
          <pre class="mmh3p-preview" data-el="preview">Compiling…</pre>
        </div>
      </div>
      </div>
      <div class="mmh3p-preview-column">
        <div class="mmh3p-panel mmh3p-references-panel">
          <div class="mmh3p-reference-head">
            <div class="mmh3p-reference-title">
              <div class="mmh3p-label">References</div>
              <div class="mmh3p-reference-help">Drag to reorder. Image, video, and audio numbering follows each type's downstream slot order.</div>
            </div>
            <div class="mmh3p-reference-actions">
              <button data-action="add-ref" data-ref-type="picture" type="button">+ Image</button>
              <button data-action="add-ref" data-ref-type="audio" type="button">+ Audio</button>
              <button data-action="add-ref" data-ref-type="video" type="button">+ Video</button>
            </div>
          </div>
          <div class="mmh3p-reference-list" data-el="references"></div>
        </div>
      </div>
      </div>
      <div class="mmh3p-panel mmh3p-log-panel">
        <span class="mmh3p-label">Execution log</span>
        <div class="mmh3p-log" data-el="log" aria-live="polite"></div>
      </div>`;

    this.els = Object.fromEntries([...this.root.querySelectorAll("[data-el]")].map(el => [el.dataset.el, el]));
    MODES.forEach(mode => this.els.mode.add(new Option(mode === "AUTO" ? "Auto" : mode, mode)));
    this.bind();
  }

  bind() {
    this.els.mode.addEventListener("change", () => { this.project.mode = this.els.mode.value; this.commit(); this.renderHeader(); });
    this.els.duration.addEventListener("change", () => {
      const minimumTimeline = this.project.shots.length * MIN_SHOT_DURATION;
      const target = clampNumber(this.els.duration.value, 5, minimumTimeline, 60);
      fitShotDurations(this.project.shots, target);
      this.commit(); this.render();
    });
    this.root.querySelector('[data-action="add-shot"]').addEventListener("click", () => this.addShot());
    this.root.querySelector('[data-action="delete-shot"]').addEventListener("click", () => this.deleteShot());
    this.root.querySelectorAll('[data-action="add-ref"]').forEach(button => {
      button.addEventListener("click", () => this.addReference(button.dataset.refType));
    });
    this.root.querySelector('[data-action="enhance"]').addEventListener("click", () => this.enhancePrompt());
    this.els.enhance.addEventListener("change", () => {
      this.project.enhance = this.els.enhance.checked;
      this.commit(false);
      this.appendLog(
        `Enhance ${this.project.enhance ? "enabled" : "disabled"}; Qwen3.8 will ${this.project.enhance ? "expand the prompt with richer contextual detail" : "use the standard concise generation mode"}.`,
        "enhance-mode",
      );
    });
    this.els["auto-run"].addEventListener("change", () => {
      this.project.auto_run = this.els["auto-run"].checked;
      this.commit(false);
      this.appendLog(
        `Auto Run ${this.project.auto_run ? "enabled" : "disabled"}; prompt generation ${this.project.auto_run ? "will run" : "will not run"} during ComfyUI queue execution.`,
        "auto-run",
      );
    });
    this.els["model-bundle"].addEventListener("change", () => {
      const bundleId = this.els["model-bundle"].value || DEFAULT_MODEL_BUNDLE;
      const profile = this.modelBundles.find(model => model.id === bundleId);
      this.project.enhance_model = profile?.enhance_model || DEFAULT_ENHANCE_MODEL;
      this.project.image_model = profile?.image_model || bundleId;
      this.commit(false);
      this.updateModelCompatibility(true);
    });
    this.root.querySelectorAll("[data-project]").forEach(input => input.addEventListener("input", () => {
      this.project[input.dataset.project] = input.value;
      this.commit();
    }));
    this.bindMentionEditor();
  }

  bindMentionEditor() {
    const editor = this.els["visual-action"];
    let composing = false;
    const updateShot = (commitChange = true) => {
      const shot = this.selectedShot();
      if (!shot) return;
      shot.visual_action = editor.value;
      this.syncMentionHighlight();
      this.renderMentionMenu();
      if (commitChange) { this.commit(); this.renderTimeline(); }
    };
    editor.addEventListener("compositionstart", () => { composing = true; });
    editor.addEventListener("compositionend", () => { composing = false; updateShot(true); });
    editor.addEventListener("input", () => updateShot(!composing));
    editor.addEventListener("click", () => this.renderMentionMenu());
    editor.addEventListener("keyup", event => {
      if (!["Escape", "ArrowDown", "ArrowUp", "Enter", "Tab"].includes(event.key)) this.renderMentionMenu();
    });
    editor.addEventListener("keydown", event => {
      if (event.key === "Escape") {
        this.hideMentionMenu();
        return;
      }
      const menuOpen = this.els["mention-menu"].classList.contains("open");
      if (!menuOpen || !this.visibleMentionEntries.length) return;
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const direction = event.key === "ArrowDown" ? 1 : -1;
        this.mentionSelectionIndex = (
          this.mentionSelectionIndex + direction + this.visibleMentionEntries.length
        ) % this.visibleMentionEntries.length;
        this.updateMentionSelection();
        return;
      }
      if (event.key === "Enter" || event.key === "Tab") {
        const mention = this.currentMention();
        const entry = this.visibleMentionEntries[this.mentionSelectionIndex];
        if (!mention || !entry) return;
        event.preventDefault();
        this.insertMention(entry.alias, mention);
      }
    });
    editor.addEventListener("scroll", () => {
      this.syncMentionHighlight();
      if (this.els["mention-menu"].classList.contains("open")) this.renderMentionMenu();
    });
    editor.addEventListener("blur", () => setTimeout(() => this.hideMentionMenu(), 120));
  }

  currentMention() {
    const editor = this.els["visual-action"];
    const end = editor.selectionStart ?? 0;
    const before = editor.value.slice(0, end);
    const match = before.match(/@[\p{L}\p{N}_-]*$/u);
    if (!match) return null;
    const start = end - match[0].length;
    if (start > 0 && /[\p{L}\p{N}_@-]/u.test(before[start - 1])) return null;
    return { start, end, query: match[0].slice(1).toLowerCase() };
  }

  mentionEntries() {
    const counts = { picture: 0, video: 0, audio: 0 };
    const seen = new Set();
    const entries = [];
    this.project.references.forEach(ref => {
      counts[ref.type] += 1;
      const alias = normalizeAlias(ref.alias);
      if (alias.length < 2 || seen.has(alias.toLowerCase())) return;
      seen.add(alias.toLowerCase());
      const labelName = REFERENCE_TYPE_LABELS[ref.type];
      entries.push({
        alias,
        label: `<${labelName} ${counts[ref.type]}>`,
        role: String(ref.role || "reference").replaceAll("_", " "),
        strength: ref.type === "picture" && ref.role === "subject_identity" ? ref.strength : "",
        description: String(ref.description || ""),
      });
    });
    return entries;
  }

  renderMentionMenu() {
    const mention = this.currentMention();
    const menu = this.els["mention-menu"];
    if (!mention || document.activeElement !== this.els["visual-action"]) {
      this.hideMentionMenu(); return;
    }
    const entries = this.mentionEntries().filter(entry =>
      entry.alias.slice(1).toLowerCase().includes(mention.query)
    );
    const signature = `${mention.start}:${mention.query}:${entries.map(entry => entry.alias).join("|")}`;
    if (signature !== this.mentionMenuSignature) {
      this.mentionSelectionIndex = 0;
      this.mentionMenuSignature = signature;
    }
    this.visibleMentionEntries = entries;
    this.mentionSelectionIndex = Math.min(this.mentionSelectionIndex, Math.max(0, entries.length - 1));
    menu.replaceChildren();
    if (!entries.length) {
      const empty = document.createElement("div");
      empty.className = "mmh3p-mention-empty";
      empty.textContent = this.project.references.some(ref => ref.alias)
        ? "No matching aliases" : "No aliases yet — enter a name in References";
      menu.appendChild(empty);
    } else {
      entries.forEach((entry, index) => {
        const button = document.createElement("button"); button.type = "button";
        button.className = `mmh3p-mention-item${index === this.mentionSelectionIndex ? " active" : ""}`;
        button.setAttribute("role", "option");
        button.setAttribute("aria-selected", index === this.mentionSelectionIndex ? "true" : "false");
        const token = document.createElement("span"); token.className = "mmh3p-mention-token"; token.textContent = entry.alias;
        const meta = document.createElement("span"); meta.className = "mmh3p-mention-meta";
        meta.textContent = `${entry.label} · ${entry.role}${entry.strength ? ` (${entry.strength})` : ""}${entry.description ? ` · ${entry.description}` : ""}`;
        button.append(token, meta);
        button.addEventListener("mousedown", event => event.preventDefault());
        button.addEventListener("mouseenter", () => {
          this.mentionSelectionIndex = index;
          this.updateMentionSelection();
        });
        button.addEventListener("click", () => this.insertMention(entry.alias, mention));
        menu.appendChild(button);
      });
    }
    menu.classList.add("open");
    this.positionMentionMenu(mention);
  }

  updateMentionSelection() {
    const buttons = [...this.els["mention-menu"].querySelectorAll(".mmh3p-mention-item")];
    buttons.forEach((button, index) => {
      const active = index === this.mentionSelectionIndex;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
    buttons[this.mentionSelectionIndex]?.scrollIntoView({ block: "nearest" });
  }

  positionMentionMenu(mention) {
    const editor = this.els["visual-action"];
    const mirror = this.els["caret-mirror"];
    const menu = this.els["mention-menu"];
    const style = getComputedStyle(editor);
    mirror.style.width = `${editor.clientWidth}px`;
    mirror.style.minHeight = `${editor.offsetHeight}px`;
    mirror.style.boxSizing = style.boxSizing;
    mirror.style.padding = style.padding;
    mirror.style.border = style.border;
    mirror.style.font = style.font;
    mirror.style.letterSpacing = style.letterSpacing;
    mirror.style.lineHeight = style.lineHeight;
    mirror.style.tabSize = style.tabSize;
    mirror.style.textAlign = style.textAlign;
    mirror.style.textIndent = style.textIndent;
    mirror.style.textTransform = style.textTransform;
    mirror.style.wordSpacing = style.wordSpacing;
    mirror.style.whiteSpace = style.whiteSpace;
    mirror.style.overflowWrap = style.overflowWrap;
    mirror.style.wordBreak = style.wordBreak;
    mirror.replaceChildren(document.createTextNode(editor.value.slice(0, mention.start)));
    const marker = document.createElement("span");
    marker.className = "mmh3p-caret-marker";
    marker.textContent = editor.value.slice(mention.start, mention.end) || "@";
    mirror.appendChild(marker);
    const lineHeight = Number.parseFloat(style.lineHeight) || Number.parseFloat(style.fontSize) * 1.35 || 18;
    const desiredLeft = marker.offsetLeft - editor.scrollLeft;
    const desiredTop = marker.offsetTop - editor.scrollTop + lineHeight + 3;
    const maxLeft = Math.max(0, editor.clientWidth - menu.offsetWidth);
    menu.style.left = `${Math.max(0, Math.min(maxLeft, desiredLeft))}px`;
    menu.style.top = `${Math.max(lineHeight, desiredTop)}px`;
  }

  insertMention(alias, mention) {
    const editor = this.els["visual-action"];
    editor.setRangeText(`${alias} `, mention.start, mention.end, "end");
    const shot = this.selectedShot();
    if (shot) shot.visual_action = editor.value;
    this.syncMentionHighlight();
    this.hideMentionMenu();
    editor.focus();
    this.commit(); this.renderTimeline();
  }

  hideMentionMenu() {
    this.els["mention-menu"]?.classList.remove("open");
    this.visibleMentionEntries = [];
    this.mentionMenuSignature = "";
    this.mentionSelectionIndex = 0;
  }

  syncMentionHighlight() {
    const editor = this.els["visual-action"];
    const backdrop = this.els["visual-action-highlight"];
    if (!editor || !backdrop) return;
    const text = editor.value;
    const regex = /@[\p{L}\p{N}_-]+/gu;
    let cursor = 0;
    const nodes = [];
    for (const match of text.matchAll(regex)) {
      if (match.index > cursor) nodes.push(document.createTextNode(text.slice(cursor, match.index)));
      const mark = document.createElement("mark"); mark.textContent = match[0]; nodes.push(mark);
      cursor = match.index + match[0].length;
    }
    if (cursor < text.length) nodes.push(document.createTextNode(text.slice(cursor)));
    if (text.endsWith("\n")) nodes.push(document.createTextNode(" "));
    backdrop.replaceChildren(...nodes);
    backdrop.scrollTop = editor.scrollTop;
    backdrop.scrollLeft = editor.scrollLeft;
  }

  totalDuration() { return this.project.shots.reduce((sum, shot) => sum + Number(shot.duration || 0), 0); }
  timelineFrameCount() { return alignedFrameCount(this.totalDuration()); }
  timelineDuration() { return this.timelineFrameCount() / VIDEO_OUTPUT_FPS; }
  shotTimelineDuration(shot) {
    return Number(shot.duration || 0) * this.timelineDuration() / Math.max(0.1, this.totalDuration());
  }
  shotTimelineRange(index) {
    const requestedTotal = Math.max(0.1, this.totalDuration());
    const requestedStart = this.project.shots.slice(0, index)
      .reduce((sum, shot) => sum + Number(shot.duration || 0), 0);
    const requestedEnd = requestedStart + Number(this.project.shots[index]?.duration || 0);
    const startSeconds = requestedStart / requestedTotal * this.timelineDuration();
    const endSeconds = requestedEnd / requestedTotal * this.timelineDuration();
    const startFrame = Math.max(0, Math.min(this.timelineFrameCount() - 1, Math.round(startSeconds * VIDEO_OUTPUT_FPS)));
    const nextStartFrame = index === this.project.shots.length - 1
      ? this.timelineFrameCount()
      : Math.max(startFrame + 1, Math.round(endSeconds * VIDEO_OUTPUT_FPS));
    const endFrame = Math.max(startFrame, Math.min(this.timelineFrameCount() - 1, nextStartFrame - 1));
    return { startSeconds, endSeconds, startFrame, endFrame };
  }
  shotTimelineLabel(index) {
    const range = this.shotTimelineRange(index);
    return `${range.startSeconds.toFixed(3)}–${range.endSeconds.toFixed(3)}s · F${range.startFrame}–${range.endFrame}`;
  }
  selectedShot() { return this.project.shots.find(shot => shot.id === this.selectedShotId) || this.project.shots[0]; }

  addShot() {
    const selected = this.selectedShot();
    if (!selected || selected.duration < MIN_SHOT_DURATION * 2) return;
    const selectedIndex = this.project.shots.findIndex(item => item.id === selected.id);
    const firstHalf = selected.duration / 2;
    selected.duration = firstHalf;
    const shot = {
      id: uid("shot"), duration: firstHalf, visual_action: "",
    };
    this.project.shots.splice(selectedIndex + 1, 0, shot);
    this.selectedShotId = shot.id; this.commit(); this.render();
  }

  deleteShot() {
    if (this.project.shots.length <= 1) return;
    const index = this.project.shots.findIndex(shot => shot.id === this.selectedShotId);
    const removed = this.project.shots.splice(Math.max(0, index), 1)[0];
    const recipientIndex = index > 0 ? index - 1 : 0;
    this.project.shots[recipientIndex].duration += removed.duration;
    this.selectedShotId = this.project.shots[recipientIndex].id;
    this.commit(); this.render();
  }

  addReference(type) {
    if (!REFERENCE_ROLES[type]) return;
    const typeCount = this.project.references.filter(ref => ref.type === type).length;
    if (typeCount >= MAX_REFERENCES[type] || this.project.references.length >= MAX_REFERENCES.total) return;
    this.project.references.push({
      id: uid("ref"), type, role: REFERENCE_ROLES[type][0], strength: "normal", alias: "", description: "", duration: 0,
      source_duration: 0, trim_start: 0, timeline_start: 0,
      frame_index: 0,
      image_filename: "", image_subfolder: "", image_type: "input",
      video_filename: "", video_subfolder: "", video_type: "input",
      audio_filename: "", audio_subfolder: "", audio_type: "input",
    });
    this.commit(); this.renderReferences(); this.renderHeader(); this.renderTimeline();
  }

  commit(refresh = true) {
    this.project.requested_duration = Number(this.totalDuration().toFixed(3));
    if (this.stateWidget) {
      this.stateWidget.value = JSON.stringify(this.project);
      try { this.stateWidget.callback?.(this.stateWidget.value); } catch {}
    }
    this.node.properties ||= {};
    this.node.properties.minimax_h3_project_version = CURRENT_PROJECT_VERSION;
    this.node.setDirtyCanvas?.(true, true);
    if (refresh) this.scheduleCompile();
  }

  appendLog(message, key = "", kind = "") {
    if (!this.els.log || !message) return;
    let line = key ? this.els.log.querySelector(`[data-log-key="${CSS.escape(key)}"]`) : null;
    if (!line) {
      line = document.createElement("div");
      line.className = "mmh3p-log-line";
      if (key) line.dataset.logKey = key;
      this.els.log.appendChild(line);
    }
    line.className = `mmh3p-log-line${kind ? ` ${kind}` : ""}`;
    const timestamp = new Date().toLocaleTimeString("en-GB", {
      hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
    });
    line.textContent = `[${timestamp}] ${message}`;
    // Moving an updated keyed entry to the end keeps the visible log in true
    // chronological order instead of retaining its original insertion slot.
    this.els.log.appendChild(line);
    while (this.els.log.children.length > 80) this.els.log.firstElementChild?.remove();
    this.els.log.scrollTop = this.els.log.scrollHeight;
  }

  async pollEnhanceProgress(jobId, controller) {
    let lastStage = "";
    while (!controller.signal.aborted) {
      try {
        const response = await api.fetchApi(`${ENHANCE_STATUS_ENDPOINT}?job_id=${encodeURIComponent(jobId)}`, {
          signal: controller.signal,
        });
        const data = await response.json();
        const job = data.job || {};
        if (job.stage === "downloading") {
          const downloaded = Number(job.downloaded || 0);
          const total = Number(job.total || 0);
          const percent = total > 0 ? Math.min(100, downloaded / total * 100) : 0;
          this.appendLog(
            `Downloading model: ${(downloaded / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB (${percent.toFixed(1)}%)`,
            "enhance-download",
            "download",
          );
        } else if (job.stage === "reference_analysis") {
          this.appendLog(job.message || "Analyzing registered visual references.", "enhance-reference-analysis");
        } else if (job.stage && job.stage !== lastStage) {
          this.appendLog(job.message || job.stage, `enhance-${job.stage}`, job.stage === "error" ? "error" : "");
        }
        lastStage = job.stage || lastStage;
        if (["complete", "error"].includes(job.stage)) return;
      } catch (error) {
        if (error?.name === "AbortError") return;
      }
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  }

  async loadModels() {
    const select = this.els["model-bundle"];
    const button = this.root.querySelector('[data-action="enhance"]');
    try {
      const response = await api.fetchApi(MODELS_ENDPOINT);
      const data = await response.json();
      if (!response.ok || data.status !== "success") throw new Error(data.message || "Could not list models");
      this.modelBundles = data.image_models || [];
      select.replaceChildren();
      this.modelBundles.forEach(model => select.add(new Option(model.label, model.id)));
      if (!this.modelBundles.length) throw new Error("No prompt generation model is available");
      const savedBundle = this.modelBundles.some(model => model.id === this.project.image_model)
        ? this.project.image_model : DEFAULT_MODEL_BUNDLE;
      select.value = savedBundle;
      const selected = this.modelBundles.find(model => model.id === savedBundle);
      this.project.enhance_model = selected?.enhance_model || DEFAULT_ENHANCE_MODEL;
      this.project.image_model = selected?.image_model || savedBundle;
      this.commit(false);
      this.updateModelCompatibility(false);
      this.appendLog(`${this.modelBundles.length} prompt generation model bundle(s) available.`, "models");
    } catch (error) {
      select.replaceChildren(new Option("Qwen3.8 + Vision F16 bundle unavailable", ""));
      button.disabled = true;
      button.title = error.message || String(error);
      this.appendLog(`Model list failed: ${error.message || error}`, "models", "error");
    }
  }

  updateModelCompatibility(writeLog = false) {
    const button = this.root.querySelector('[data-action="enhance"]');
    const bundleId = this.els["model-bundle"]?.value || this.project.image_model || DEFAULT_MODEL_BUNDLE;
    const profile = this.modelBundles.find(model => model.id === bundleId);
    const mode = this.project.mode === "AUTO" ? inferAutoMode(this.project.references) : this.project.mode;
    const supported = !profile?.supported_modes?.length || profile.supported_modes.includes(mode);
    const status = this.els["model-status"];
    const qwenEnhanceAvailable = (profile?.enhance_model || DEFAULT_ENHANCE_MODEL) === DEFAULT_ENHANCE_MODEL;
    if (this.els.enhance) {
      this.els.enhance.disabled = !qwenEnhanceAvailable;
      this.els.enhance.closest("label").title = qwenEnhanceAvailable
        ? "Use richer Qwen3.8 interpretation and expansion while preserving the requested events"
        : "Enhance mode is available only with Qwen3.8";
    }
    if (status) {
      status.textContent = supported ? "ready" : `${mode} unsupported`;
      status.classList.toggle("error", !supported);
    }
    if (button && !this.enhanceController) {
      button.disabled = !supported;
      button.title = supported ? "" : `${profile?.label || "Selected model"} does not support ${mode}.`;
    }
    if (!supported && writeLog) {
      this.appendLog(
        `The selected LightX2V 8B rewriter does not support ${mode}. Select Qwen3.8 for R2V/REF2VA.`,
        "model-compatibility", "error",
      );
    }
    return supported;
  }

  async enhancePrompt() {
    const button = this.root.querySelector('[data-action="enhance"]');
    if (this.enhanceController) {
      await this.cancelPromptGeneration();
      return;
    }
    const originalLabel = button.textContent;
    if (!this.updateModelCompatibility(true)) return;
    const selectedBundle = this.els["model-bundle"].value || DEFAULT_MODEL_BUNDLE;
    const modelInfo = this.modelBundles.find(model => model.id === selectedBundle);
    const selectedModel = modelInfo?.enhance_model || DEFAULT_ENHANCE_MODEL;
    const selectedImageModel = modelInfo?.image_model || selectedBundle;
    const needsImageAnalysis = this.project.references.some(ref =>
      ref.type === "picture" && ref.image_filename
    );
    const needsVideoAnalysis = this.project.references.some(ref =>
      ref.type === "video" && ref.video_filename && Number(ref.duration) > 0
    );
    const needsVisualAnalysis = needsImageAnalysis || needsVideoAnalysis;
    const missingBytes = Number(modelInfo?.text_size || 0)
      + (needsVisualAnalysis ? Number(modelInfo?.vision_size || 0) : 0);
    if (missingBytes > 0) {
      const sizeGb = (missingBytes / 1e9).toFixed(2);
      const purpose = needsVisualAnalysis ? "generate the prompt and analyze its visual references" : "generate the prompt";
      if (!window.confirm(`Download ${sizeGb} GB for ${modelInfo?.label || "Qwen3.8 + Vision F16"} and ${purpose}?`)) return;
    }
    clearTimeout(this.compileTimer);
    this.compileController?.abort();
    this.enhanceController?.abort();
    const controller = new AbortController();
    this.enhanceController = controller;
    const jobId = crypto.randomUUID?.() || `enhance-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    this.enhanceJobId = jobId;
    button.disabled = false;
    button.textContent = "Stop";
    button.title = "Stop the current prompt generation";
    const sourceProject = JSON.stringify({ ...this.project, enhanced_prompt: "" });
    this.appendLog("Prompt generation started.", `enhance-start-${jobId}`);
    const progressPromise = this.pollEnhanceProgress(jobId, controller);
    try {
      const response = await api.fetchApi(ENHANCE_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_data: sourceProject,
          model: selectedModel,
          image_model: selectedImageModel,
          job_id: jobId,
        }),
        signal: controller.signal,
      });
      const data = await response.json();
      if (!response.ok || data.status !== "success") throw new Error(data.message || "Prompt generation failed");
      if (JSON.stringify({ ...this.project, enhanced_prompt: "" }) !== sourceProject) {
        throw new Error("The project changed while prompt generation was running; the stale result was discarded.");
      }
      if (data.reference_analyses?.length) {
        const count = data.reference_analyses.length;
        this.appendLog(
          `Analyzed ${count} visual reference asset${count === 1 ? "" : "s"} using ${count === 1 ? "its" : "their"} assigned role${count === 1 ? "" : "s"}.`,
          `enhance-reference-result-${jobId}`,
        );
        data.reference_analyses.forEach((item, index) => {
          const label = item.label || `Picture ${index + 1}`;
          const role = item.role || "unknown";
          const filename = item.filename ? ` — ${item.filename}` : "";
          this.appendLog(
            `Vision analysis for ${label} [role=${role}]${filename}\n${item.analysis || "No analysis text returned."}`,
            `enhance-reference-analysis-result-${jobId}-${item.id || index}`,
            "analysis",
          );
        });
      }
      this.project.enhance_model = data.model;
      this.project.image_model = selectedBundle;
      this.project.enhanced_prompt = data.enhanced_prompt;
      this.autoRunPreview = null;
      if (modelInfo) modelInfo.installed = true;
      this.commit(false);
      this.renderReferences();
      await this.compile();
      this.appendLog("Generated prompt is ready.", `enhance-result-${jobId}`);
    } catch (error) {
      if (error?.name === "AbortError") return;
      this.appendLog(`Prompt generation failed: ${error.message || error}`, `enhance-error-${jobId}`, "error");
    } finally {
      controller.abort();
      await progressPromise;
      if (this.enhanceController === controller) this.enhanceController = null;
      if (this.enhanceJobId === jobId) this.enhanceJobId = "";
      button.textContent = originalLabel;
      this.updateModelCompatibility(false);
    }
  }

  async cancelPromptGeneration() {
    const controller = this.enhanceController;
    const jobId = this.enhanceJobId;
    if (!controller) return;
    const button = this.root.querySelector('[data-action="enhance"]');
    button.disabled = true;
    button.textContent = "Stopping…";
    this.appendLog("Stopping prompt generation.", `enhance-cancel-${jobId}`);
    try {
      if (jobId) {
        await api.fetchApi(ENHANCE_CANCEL_ENDPOINT, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ job_id: jobId }),
        });
      }
    } catch (error) {
      this.appendLog(`Stop request failed: ${error.message || error}`, `enhance-cancel-error-${jobId}`, "error");
    } finally {
      controller.abort();
      this.appendLog("Prompt generation stopped.", `enhance-cancelled-${jobId}`);
    }
  }

  scheduleCompile() {
    clearTimeout(this.compileTimer);
    this.compileTimer = setTimeout(() => this.compile(), 300);
  }

  async compile() {
    const sequence = ++this.compileSequence;
    this.compileController?.abort();
    const controller = new AbortController();
    this.compileController = controller;
    this.appendLog("Updating raw prompt from the current node inputs…", "compile");
    try {
      const response = await api.fetchApi(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_data: JSON.stringify(this.project) }),
        signal: controller.signal,
      });
      const data = await response.json();
      if (sequence !== this.compileSequence) return;
      if (!response.ok || data.status !== "success") throw new Error(data.message || "Compilation failed");
      this.previewData = data;
      this.resolvedMode = data.resolved_mode || inferAutoMode(this.project.references);
      this.els.effective.textContent = `${data.effective_frames}f / ${Number(data.effective_duration).toFixed(2)}s`;
      const validationLine = this.els.log?.querySelector('[data-log-key="validation"]');
      if (data.errors.length) {
        this.appendLog(`Validation failed: ${data.errors.join(" | ")}`, "validation", "error");
      } else if (data.warnings.length) {
        this.appendLog(`Validation warning: ${data.warnings.join(" | ")}`, "validation", "warning");
      } else {
        validationLine?.remove();
      }
      this.renderPreview();
      this.renderHeader();
      this.appendLog("Node inputs updated.", "compile");
    } catch (error) {
      if (error?.name === "AbortError" || sequence !== this.compileSequence) return;
      this.appendLog(`Input update failed: ${error.message || error}`, "compile", "error");
    } finally {
      if (sequence === this.compileSequence) this.compileController = null;
    }
  }

  render() {
    this.renderHeader(); this.renderTimeline(); this.renderShotEditor(); this.renderReferences();
    this.scheduleCompile();
  }

  renderHeader() {
    this.els.mode.value = this.project.mode;
    this.els["auto-run"].checked = this.project.auto_run === true;
    this.els.enhance.checked = this.project.enhance === true;
    this.els.duration.value = this.totalDuration().toFixed(2);
    const resolvedMode = this.project.mode === "AUTO"
      ? inferAutoMode(this.project.references)
      : this.project.mode;
    this.els.path.textContent = this.project.mode === "AUTO"
      ? `auto → ${resolvedMode.toLowerCase()}`
      : resolvedMode.toLowerCase();
    this.els["ruler-end"].textContent = `${this.timelineDuration().toFixed(2)}s · ${this.timelineFrameCount()}f`;
    this.root.querySelector('[data-action="add-shot"]').disabled =
      !this.selectedShot() || this.selectedShot().duration < MIN_SHOT_DURATION * 2;
    this.root.querySelectorAll('[data-action="add-ref"]').forEach(button => {
      const refType = button.dataset.refType;
      const refTypeCount = this.project.references.filter(ref => ref.type === refType).length;
      button.disabled =
        refTypeCount >= MAX_REFERENCES[refType] || this.project.references.length >= MAX_REFERENCES.total;
      button.title = button.disabled
        ? `Reference limit reached (${refTypeCount}/${MAX_REFERENCES[refType]}, total ${this.project.references.length}/${MAX_REFERENCES.total})`
        : `Add ${REFERENCE_TYPE_LABELS[refType].toLowerCase()} reference (${refTypeCount}/${MAX_REFERENCES[refType]})`;
    });
    if (this.modelBundles.length) this.updateModelCompatibility(false);
  }

  renderTimeline() {
    this.els.timeline.querySelectorAll(".mmh3p-shot").forEach(el => el.remove());
    const total = Math.max(0.1, this.timelineDuration());
    this.project.shots.forEach((shot, index) => {
      const card = document.createElement("div");
      card.className = `mmh3p-shot${shot.id === this.selectedShotId ? " selected" : ""}`;
      card.dataset.shotId = shot.id;
      card.draggable = true;
      card.style.flex = `${Math.max(0.1, shot.duration) / Math.max(0.1, this.totalDuration())} 1 0`;
      const head = document.createElement("div"); head.className = "mmh3p-row";
      const title = document.createElement("span"); title.className = "mmh3p-shot-title"; title.textContent = `Shot ${index + 1}`;
      const duration = document.createElement("span"); duration.className = "mmh3p-shot-duration";
      duration.textContent = this.shotTimelineLabel(index);
      head.append(title, duration);
      const summary = document.createElement("div"); summary.className = "mmh3p-shot-summary";
      summary.textContent = shot.visual_action || "Click to describe this shot.";
      card.append(head, summary);
      card.addEventListener("click", () => { this.selectedShotId = shot.id; this.renderTimeline(); this.renderShotEditor(); });
      card.addEventListener("dragstart", event => {
        if (this.root.classList.contains("resizing")) { event.preventDefault(); return; }
        event.dataTransfer.setData("text/plain", shot.id); card.classList.add("dragging");
      });
      card.addEventListener("dragend", () => card.classList.remove("dragging"));
      card.addEventListener("dragover", event => event.preventDefault());
      card.addEventListener("drop", event => {
        event.preventDefault();
        const sourceId = event.dataTransfer.getData("text/plain");
        const from = this.project.shots.findIndex(item => item.id === sourceId);
        const to = this.project.shots.findIndex(item => item.id === shot.id);
        if (from >= 0 && to >= 0 && from !== to) {
          const [moved] = this.project.shots.splice(from, 1); this.project.shots.splice(to, 0, moved);
          this.commit(); this.renderTimeline();
        }
      });
      if (index < this.project.shots.length - 1) {
        const handle = document.createElement("div");
        handle.className = "mmh3p-resize-handle";
        handle.title = "Drag to resize the two adjacent shots";
        handle.addEventListener("pointerdown", event => this.beginShotResize(event, index, handle));
        handle.addEventListener("dblclick", event => {
          event.preventDefault(); event.stopPropagation();
          const pairTotal = this.project.shots[index].duration + this.project.shots[index + 1].duration;
          this.project.shots[index].duration = pairTotal / 2;
          this.project.shots[index + 1].duration = pairTotal / 2;
          this.commit(); this.renderTimeline();
        });
        card.appendChild(handle);
      }
      this.els.timeline.appendChild(card);
    });
    this.renderVideoTimeline();
  }

  normalizeVideoClip(ref) {
    const targetDuration = Math.max(0.1, this.timelineDuration());
    const sourceDuration = Math.max(0, Number(ref.source_duration || ref.duration || 0));
    ref.source_duration = sourceDuration;
    ref.trim_start = Math.min(Math.max(0, Number(ref.trim_start || 0)), Math.max(0, sourceDuration - MIN_SHOT_DURATION));
    const available = Math.max(0, sourceDuration - ref.trim_start);
    ref.duration = Math.min(Math.max(0, Number(ref.duration || 0)), available, 15);
    const minimumVisible = Math.min(MIN_VIDEO_CLIP_DURATION, ref.duration);
    ref.timeline_start = Math.min(
      Math.max(-ref.duration + minimumVisible, Number(ref.timeline_start || 0)),
      targetDuration - minimumVisible,
    );
  }

  renderVideoTimeline() {
    const container = this.els["video-timeline"];
    if (!container) return;
    const videos = this.project.references.filter(ref => ref.type === "video");
    let pictureNumber = 0;
    const imageAnchors = this.project.references.flatMap(ref => {
      if (ref.type !== "picture") return [];
      pictureNumber += 1;
      return ["first_frame", "last_frame", "frame"].includes(ref.role)
        ? [{ ref, number: pictureNumber }]
        : [];
    });
    container.hidden = videos.length === 0 && imageAnchors.length === 0;
    container.replaceChildren();
    if (!videos.length && !imageAnchors.length) return;
    if (imageAnchors.length) {
      const imageHelp = document.createElement("div");
      imageHelp.className = "mmh3p-label";
      imageHelp.textContent = "Image anchors · first/last frames are fixed · drag Frame images to an exact output frame";
      const track = document.createElement("div"); track.className = "mmh3p-image-track";
      const lane = document.createElement("div"); lane.className = "mmh3p-image-lane";
      imageAnchors.forEach(({ ref, number }) => {
        const isFirst = ref.role === "first_frame";
        const isLast = ref.role === "last_frame";
        const isMovable = ref.role === "frame";
        const marker = document.createElement("div");
        marker.className = `mmh3p-image-anchor ${isFirst ? "first" : isLast ? "last" : "frame"}`;
        const label = document.createElement("div"); label.className = "mmh3p-image-anchor-label";
        const labelText = document.createElement("span"); label.appendChild(labelText);
        if (ref.image_filename) {
          const previewImage = document.createElement("img"); previewImage.className = "mmh3p-image-anchor-preview";
          previewImage.alt = ""; previewImage.draggable = false;
          previewImage.src = api.apiURL(`/view?${new URLSearchParams({
            filename: ref.image_filename, subfolder: ref.image_subfolder || "", type: "input",
          }).toString()}`);
          label.appendChild(previewImage);
        }
        const updateAnchor = () => {
          const frameCount = this.timelineFrameCount();
          const laneWidth = Math.max(1, lane.clientWidth || this.els.timeline.clientWidth || 1000);
          const frameWidth = Math.max(2, laneWidth / Math.max(1, frameCount));
          marker.style.width = `${frameWidth}px`;
          if (isFirst) {
            label.classList.remove("before");
            labelText.textContent = `Image ${number} · First · Frame 0`;
            marker.title = `<Image ${number}> exact first frame 0 at 0.00s`;
          } else if (isLast) {
            label.classList.add("before");
            labelText.textContent = `Image ${number} · Last · Frame ${frameCount - 1}`;
            marker.title = `<Image ${number}> exact last frame ${frameCount - 1} at ${this.timelineDuration().toFixed(2)}s`;
          } else {
            ref.frame_index = Math.max(0, Math.min(frameCount - 1, Math.round(Number(ref.frame_index || 0))));
            const rawCenter = frameCount > 1 ? ref.frame_index / (frameCount - 1) * laneWidth : 0;
            const center = Math.max(frameWidth / 2 + 1, Math.min(laneWidth - frameWidth / 2 - 1, rawCenter));
            marker.style.left = `${center}px`;
            label.classList.toggle("before", ref.frame_index >= frameCount / 2);
            labelText.textContent = `Image ${number} · Frame ${ref.frame_index} · ${(ref.frame_index / VIDEO_OUTPUT_FPS).toFixed(3)}s`;
            marker.title = `<Image ${number}> exact frame ${ref.frame_index}`;
          }
        };
        if (isMovable) {
          marker.addEventListener("pointerdown", event => {
            event.preventDefault(); event.stopPropagation(); marker.setPointerCapture?.(event.pointerId);
            const move = moveEvent => {
              const rect = lane.getBoundingClientRect();
              const ratio = Math.max(0, Math.min(1, (moveEvent.clientX - rect.left) / Math.max(1, rect.width)));
              ref.frame_index = Math.round(ratio * Math.max(0, this.timelineFrameCount() - 1));
              updateAnchor();
            };
            const end = upEvent => {
              marker.releasePointerCapture?.(upEvent.pointerId);
              marker.removeEventListener("pointermove", move);
              marker.removeEventListener("pointerup", end);
              marker.removeEventListener("pointercancel", end);
              this.commit(); this.renderTimeline();
            };
            marker.addEventListener("pointermove", move);
            marker.addEventListener("pointerup", end);
            marker.addEventListener("pointercancel", end);
          });
        }
        marker.appendChild(label); lane.appendChild(marker);
        updateAnchor();
      });
      track.appendChild(lane); container.append(imageHelp, track);
    }
    if (!videos.length) return;
    const help = document.createElement("div");
    help.className = "mmh3p-label";
    help.textContent = "Video clips · drag a clip to move · drag either edge to trim";
    container.appendChild(help);
    videos.forEach((ref, index) => {
      this.normalizeVideoClip(ref);
      const track = document.createElement("div"); track.className = "mmh3p-video-track";
      const label = document.createElement("div"); label.className = "mmh3p-video-track-label";
      label.textContent = `Video ${index + 1}`;
      const lane = document.createElement("div"); lane.className = "mmh3p-video-lane";
      if (ref.video_filename && ref.duration > 0) {
        const canvas = document.createElement("div"); canvas.className = "mmh3p-video-canvas";
        const clip = document.createElement("div"); clip.className = "mmh3p-video-clip";
        const filmstrip = document.createElement("div"); filmstrip.className = "mmh3p-video-filmstrip";
        const loading = document.createElement("div"); loading.className = "mmh3p-video-filmstrip-loading";
        loading.textContent = "Loading video frames…"; filmstrip.appendChild(loading);
        const left = document.createElement("div"); left.className = "mmh3p-video-trim left";
        const right = document.createElement("div"); right.className = "mmh3p-video-trim right";
        const text = document.createElement("span");
        const updateClip = () => {
          const total = Math.max(0.1, this.timelineDuration());
          clip.style.left = `${ref.timeline_start / total * 100}%`;
          clip.style.width = `${ref.duration / total * 100}%`;
          const visibleStart = Math.max(0, ref.timeline_start);
          const visibleEnd = Math.min(total, ref.timeline_start + ref.duration);
          const visibleDuration = Math.max(0, visibleEnd - visibleStart);
          const visibleFrames = Math.max(0, Math.round(visibleDuration * VIDEO_OUTPUT_FPS));
          const visibleCenter = (visibleStart + visibleEnd) / 2;
          const labelPosition = (visibleCenter - ref.timeline_start) / Math.max(0.001, ref.duration) * 100;
          text.style.left = `${labelPosition}%`;
          text.textContent = `${visibleDuration.toFixed(2)}s · ${visibleFrames} frames`;
          clip.title = `Target ${visibleStart.toFixed(2)}–${visibleEnd.toFixed(2)}s · Source ${ref.trim_start.toFixed(2)}–${(ref.trim_start + ref.duration).toFixed(2)}s`;
        };
        const beginEdit = (event, operation) => {
          event.preventDefault(); event.stopPropagation();
          const startX = event.clientX;
          const startTimeline = ref.timeline_start;
          const startTrim = ref.trim_start;
          const startDuration = ref.duration;
          const secondsPerPixel = this.timelineDuration() / Math.max(1, lane.getBoundingClientRect().width);
          const captureTarget = clip;
          captureTarget.setPointerCapture?.(event.pointerId);
          const onMove = moveEvent => {
            let delta = Math.round(((moveEvent.clientX - startX) * secondsPerPixel) / SHOT_SNAP_SECONDS) * SHOT_SNAP_SECONDS;
            if (operation === "move") {
              const minimumVisible = Math.min(MIN_VIDEO_CLIP_DURATION, startDuration);
              ref.timeline_start = Math.max(
                -startDuration + minimumVisible,
                Math.min(this.timelineDuration() - minimumVisible, startTimeline + delta),
              );
            } else if (operation === "left") {
              const minimumDuration = Math.min(MIN_VIDEO_CLIP_DURATION, startDuration);
              delta = Math.max(-startTrim, Math.min(startDuration - minimumDuration, delta));
              ref.timeline_start = startTimeline + delta;
              ref.trim_start = startTrim + delta;
              ref.duration = startDuration - delta;
            } else {
              const maxGrowth = Math.min(
                ref.source_duration - startTrim - startDuration,
                15 - startDuration,
              );
              const minimumDuration = Math.min(MIN_VIDEO_CLIP_DURATION, startDuration);
              delta = Math.max(minimumDuration - startDuration, Math.min(maxGrowth, delta));
              ref.duration = startDuration + delta;
            }
            updateClip();
          };
          const onUp = upEvent => {
            captureTarget.releasePointerCapture?.(upEvent.pointerId);
            captureTarget.removeEventListener("pointermove", onMove);
            captureTarget.removeEventListener("pointerup", onUp);
            captureTarget.removeEventListener("pointercancel", onUp);
            this.normalizeVideoClip(ref);
            this.commit(); this.renderReferences(); this.renderVideoTimeline();
          };
          captureTarget.addEventListener("pointermove", onMove);
          captureTarget.addEventListener("pointerup", onUp);
          captureTarget.addEventListener("pointercancel", onUp);
        };
        clip.addEventListener("pointerdown", event => beginEdit(event, "move"));
        left.addEventListener("pointerdown", event => beginEdit(event, "left"));
        right.addEventListener("pointerdown", event => beginEdit(event, "right"));
        clip.append(filmstrip, left, text, right);
        canvas.appendChild(clip); lane.appendChild(canvas);
        track.append(label, lane); container.appendChild(track);
        updateClip();
        this.populateVideoFilmstrip(ref, filmstrip);
        return;
      } else {
        const empty = document.createElement("div"); empty.className = "mmh3p-video-filmstrip-loading";
        empty.style.padding = "8px"; empty.textContent = "Upload a video to edit its clip";
        lane.appendChild(empty);
      }
      track.append(label, lane); container.appendChild(track);
    });
  }

  async populateVideoFilmstrip(ref, filmstrip) {
    const key = [
      `${ref.video_subfolder || ""}/${ref.video_filename}`,
      Number(ref.source_duration || 0).toFixed(3),
      Number(ref.trim_start || 0).toFixed(3),
      Number(ref.duration || 0).toFixed(3),
    ].join("|");
    let pending = this.videoThumbnailCache.get(key);
    if (!pending) {
      pending = (async () => {
        const video = document.createElement("video");
        video.muted = true; video.preload = "auto"; video.playsInline = true;
        video.src = api.apiURL(`${VIDEO_VIEW_ENDPOINT}?${new URLSearchParams({
          filename: ref.video_filename, subfolder: ref.video_subfolder || "",
        }).toString()}`);
        const waitFor = (eventName, timeout = 12000) => new Promise((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error(`Video ${eventName} timed out`)), timeout);
          video.addEventListener(eventName, () => { clearTimeout(timer); resolve(); }, { once: true });
          video.addEventListener("error", () => { clearTimeout(timer); reject(new Error("Video preview failed")); }, { once: true });
        });
        if (video.readyState < 1) { video.load(); await waitFor("loadedmetadata"); }
        if (video.readyState < 2) await waitFor("loadeddata");
        const sourceDuration = Math.max(0.01, Math.min(Number(ref.source_duration || video.duration || 0), video.duration || Infinity));
        const start = Math.min(Math.max(0, Number(ref.trim_start || 0)), Math.max(0, sourceDuration - 0.01));
        const duration = Math.max(0.01, Math.min(Number(ref.duration || 0), sourceDuration - start));
        const frameCount = Math.min(20, Math.max(4, Math.ceil(duration * 2.5)));
        const canvas = document.createElement("canvas");
        canvas.width = 160;
        canvas.height = Math.max(72, Math.round(160 * (video.videoHeight || 90) / Math.max(1, video.videoWidth || 160)));
        const context = canvas.getContext("2d", { alpha: false });
        const frames = [];
        for (let index = 0; index < frameCount; index += 1) {
          const selectedOffset = index / Math.max(1, frameCount - 1) * Math.max(0, duration - 0.03);
          const time = Math.min(sourceDuration - 0.01, start + selectedOffset);
          if (Math.abs(video.currentTime - time) > 0.01 || video.readyState < 2) {
            const seeked = waitFor("seeked");
            video.currentTime = time;
            await seeked;
          }
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
          frames.push(canvas.toDataURL("image/jpeg", 0.78));
        }
        video.removeAttribute("src"); video.load();
        return frames;
      })();
      this.videoThumbnailCache.set(key, pending);
    }
    try {
      const frames = await pending;
      if (!filmstrip.isConnected) return;
      filmstrip.replaceChildren(...frames.map(source => {
        const image = document.createElement("img");
        image.className = "mmh3p-video-thumb"; image.src = source; image.alt = ""; image.draggable = false;
        return image;
      }));
      filmstrip.title = "Frames sampled from the selected source interval";
    } catch (error) {
      this.videoThumbnailCache.delete(key);
      if (!filmstrip.isConnected) return;
      const unavailable = document.createElement("div"); unavailable.className = "mmh3p-video-filmstrip-loading";
      unavailable.textContent = "Frame preview unavailable"; filmstrip.replaceChildren(unavailable);
    }
  }

  beginShotResize(event, index, handle) {
    event.preventDefault(); event.stopPropagation();
    if (index < 0 || index >= this.project.shots.length - 1) return;
    const leftShot = this.project.shots[index];
    const rightShot = this.project.shots[index + 1];
    const pairTotal = leftShot.duration + rightShot.duration;
    if (pairTotal < MIN_SHOT_DURATION * 2) return;
    const startX = event.clientX;
    const startLeft = leftShot.duration;
    const timelineWidth = Math.max(1, this.els.timeline.getBoundingClientRect().width);
    const secondsPerPixel = this.totalDuration() / timelineWidth;
    this.root.classList.add("resizing"); handle.classList.add("active");
    handle.setPointerCapture?.(event.pointerId);

    const onMove = moveEvent => {
      let nextLeft = startLeft + (moveEvent.clientX - startX) * secondsPerPixel;
      nextLeft = Math.round(nextLeft / SHOT_SNAP_SECONDS) * SHOT_SNAP_SECONDS;
      nextLeft = Math.max(MIN_SHOT_DURATION, Math.min(pairTotal - MIN_SHOT_DURATION, nextLeft));
      leftShot.duration = nextLeft;
      rightShot.duration = pairTotal - nextLeft;
      const leftCard = this.els.timeline.querySelector(`[data-shot-id="${leftShot.id}"]`);
      const rightCard = this.els.timeline.querySelector(`[data-shot-id="${rightShot.id}"]`);
      if (leftCard && rightCard) {
        leftCard.style.flex = `${leftShot.duration / this.totalDuration()} 1 0`;
        rightCard.style.flex = `${rightShot.duration / this.totalDuration()} 1 0`;
        leftCard.querySelector(".mmh3p-shot-duration").textContent = this.shotTimelineLabel(index);
        rightCard.querySelector(".mmh3p-shot-duration").textContent = this.shotTimelineLabel(index + 1);
      }
    };
    const onUp = upEvent => {
      handle.releasePointerCapture?.(upEvent.pointerId);
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", onUp);
      handle.removeEventListener("pointercancel", onUp);
      this.root.classList.remove("resizing"); handle.classList.remove("active");
      this.commit(); this.renderHeader(); this.renderTimeline();
    };
    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onUp);
    handle.addEventListener("pointercancel", onUp);
  }

  renderShotEditor() {
    const shot = this.selectedShot();
    if (!shot) return;
    this.els["visual-action"].value = shot.visual_action || "";
    this.syncMentionHighlight();
    this.hideMentionMenu();
  }

  renderReferences() {
    this.syncReferenceOutputs();
    const container = this.els.references; container.replaceChildren();
    if (!this.project.references.length) {
      const empty = document.createElement("div"); empty.className = "mmh3p-empty";
      empty.textContent = "No references. Add metadata in the exact downstream order for each media type."; container.appendChild(empty); return;
    }
    const counts = { picture: 0, video: 0, audio: 0 };
    this.project.references.forEach(ref => {
      counts[ref.type] += 1;
      const label = `<${REFERENCE_TYPE_LABELS[ref.type]} ${counts[ref.type]}>`;
      const row = document.createElement("div"); row.className = `mmh3p-ref ${ref.type}`;
      row.draggable = true;
      row.title = "Drag to reorder references";
      const labelEl = document.createElement("span"); labelEl.className = "mmh3p-ref-label"; labelEl.textContent = label;
      const role = document.createElement("select");
      if (role) {
        REFERENCE_ROLES[ref.type].forEach(value => {
          const words = value.replaceAll("_", " ");
          role.add(new Option(REFERENCE_ROLE_LABELS[value] || words.charAt(0).toUpperCase() + words.slice(1), value));
        });
        role.value = ref.role;
        role.title = REFERENCE_ROLE_HELP[ref.role] || "How this reference should influence the generated video";
      }
      const strength = document.createElement("select");
      strength.className = "subject-strength";
      SUBJECT_STRENGTHS.forEach(value => strength.add(new Option(SUBJECT_STRENGTH_LABELS[value], value)));
      strength.value = SUBJECT_STRENGTHS.includes(ref.strength) ? ref.strength : "normal";
      strength.title = "Weak: broad similarity; Normal: core identity; Strong: complete visible appearance and the subject's source visual style";
      const alias = document.createElement("input"); alias.placeholder = "alias";
      alias.value = String(ref.alias || "").replace(/^@+/, "");
      const desc = ref.type === "picture" ? null : document.createElement("textarea");
      if (desc) {
        desc.className = "desc";
        desc.placeholder = ref.type === "audio"
          ? (AUDIO_DESCRIPTION_PLACEHOLDERS[ref.role] || AUDIO_DESCRIPTION_PLACEHOLDERS.none)
          : "Describe how this video should guide the target in English";
        desc.value = ref.description;
      }
      const del = document.createElement("button"); del.className = "delete"; del.textContent = "×"; del.title = "Delete reference";
      const preview = document.createElement("div"); preview.className = "mmh3p-ref-preview";
      const body = document.createElement("div"); body.className = "mmh3p-ref-body";
      const controls = document.createElement("div"); controls.className = "mmh3p-ref-controls";
      if (ref.type === "video" || ref.type === "audio") controls.classList.add("video-metadata");
      const strengthRow = document.createElement("div"); strengthRow.className = "mmh3p-subject-strength-row";
      const strengthLabel = document.createElement("span"); strengthLabel.textContent = "Strength";
      strengthRow.append(strengthLabel, strength, alias);
      const fileInput = document.createElement("input"); fileInput.type = "file";
      fileInput.accept = ref.type === "video"
        ? "video/mp4,video/webm,video/quicktime,video/x-matroska,video/x-msvideo,.m4v"
        : ref.type === "audio"
          ? "audio/wav,audio/mpeg,audio/flac,audio/ogg,audio/mp4,audio/aac,.m4a,.opus"
        : "image/png,image/jpeg,image/webp,image/bmp";
      fileInput.hidden = true;
      if (ref.type === "picture") {
        if (ref.image_filename) {
          const img = document.createElement("img"); img.alt = `${label} preview`;
          img.src = api.apiURL(`/view?${new URLSearchParams({
            filename: ref.image_filename, subfolder: ref.image_subfolder || "", type: "input",
          }).toString()}`);
          preview.appendChild(img);
          preview.title = "Click the preview to replace this image";
        } else {
          preview.textContent = "Click to add\nimage";
          preview.title = "Click to upload a reference image";
        }
        preview.addEventListener("click", event => { event.stopPropagation(); fileInput.click(); });
        fileInput.addEventListener("change", async () => {
          const file = fileInput.files?.[0];
          if (file) await this.uploadReferenceImage(ref, file);
        });
      } else if (ref.type === "video") {
        if (ref.video_filename) {
          const video = document.createElement("video");
          video.muted = true; video.controls = true; video.preload = "metadata";
          video.src = api.apiURL(`${VIDEO_VIEW_ENDPOINT}?${new URLSearchParams({
            filename: ref.video_filename, subfolder: ref.video_subfolder || "",
          }).toString()}`);
          video.addEventListener("pointerdown", event => event.stopPropagation());
          video.addEventListener("dblclick", event => {
            event.stopPropagation(); fileInput.click();
          });
          preview.appendChild(video);
          preview.title = "Double-click the video to replace it";
        } else {
          preview.textContent = "Click to add\nvideo";
          preview.title = "Click to upload a reference video";
        }
        preview.addEventListener("click", event => {
          if (event.target?.tagName === "VIDEO") return;
          event.stopPropagation(); fileInput.click();
        });
        fileInput.addEventListener("change", async () => {
          const file = fileInput.files?.[0];
          if (file) await this.uploadReferenceVideo(ref, file);
        });
      } else if (ref.type === "audio") {
        if (ref.audio_filename) {
          const audio = document.createElement("audio");
          audio.controls = true; audio.preload = "metadata";
          audio.src = api.apiURL(`/view?${new URLSearchParams({
            filename: ref.audio_filename, subfolder: ref.audio_subfolder || "", type: "input",
          }).toString()}`);
          audio.addEventListener("pointerdown", event => event.stopPropagation());
          preview.appendChild(audio);
          preview.title = "Double-click to replace this audio";
          preview.addEventListener("dblclick", event => { event.stopPropagation(); fileInput.click(); });
        } else {
          preview.textContent = "Click to add\naudio";
          preview.title = "Click to upload reference audio";
        }
        preview.addEventListener("click", event => {
          if (event.target?.tagName === "AUDIO") return;
          event.stopPropagation(); fileInput.click();
        });
        fileInput.addEventListener("change", async () => {
          const file = fileInput.files?.[0];
          if (file) await this.uploadReferenceAudio(ref, file);
        });
      }
      if (role) {
        role.addEventListener("change", () => {
          ref.role = role.value;
          if (ref.role === "subject_identity" && !SUBJECT_STRENGTHS.includes(ref.strength)) ref.strength = "normal";
          role.title = REFERENCE_ROLE_HELP[ref.role] || "How this reference should influence the generated video";
          if (ref.type === "picture") ref.description = "";
          this.commit(); this.renderReferences(); this.renderHeader(); this.renderTimeline();
        });
      }
      strength.addEventListener("change", () => {
        ref.strength = strength.value;
        this.commit();
      });
      if (desc) {
        desc.addEventListener("input", () => { ref.description = desc.value; this.commit(); });
      }
      del.addEventListener("click", () => {
        this.project.references = this.project.references.filter(item => item.id !== ref.id);
        this.commit(); this.renderReferences(); this.renderHeader(); this.renderTimeline();
      });
      row.addEventListener("dragstart", event => { event.dataTransfer.setData("text/plain", ref.id); row.classList.add("dragging"); });
      row.addEventListener("dragend", () => row.classList.remove("dragging"));
      row.addEventListener("dragover", event => event.preventDefault());
      row.addEventListener("drop", event => {
        event.preventDefault();
        const sourceId = event.dataTransfer.getData("text/plain");
        const from = this.project.references.findIndex(item => item.id === sourceId);
        const to = this.project.references.findIndex(item => item.id === ref.id);
        if (from >= 0 && to >= 0 && from !== to) {
          const [moved] = this.project.references.splice(from, 1);
          this.project.references.splice(to, 0, moved);
          this.commit(); this.renderReferences(); this.renderHeader(); this.renderTimeline();
        }
      });
      alias.addEventListener("input", () => {
        ref.alias = normalizeAlias(alias.value);
        this.syncMentionHighlight();
        this.commit();
      });
      alias.addEventListener("blur", () => {
        alias.value = String(ref.alias || "").replace(/^@+/, "");
        this.commit();
      });
      const showStrength = ref.type === "picture" && ref.role === "subject_identity";
      if (ref.type === "video") controls.append(role, alias, del);
      else if (ref.type === "audio") controls.append(role, alias, del);
      else controls.append(role, del);
      body.append(controls);
      if (showStrength) body.append(strengthRow);
      if (desc) body.append(desc);
      body.append(fileInput);
      row.append(labelEl, preview, body); container.appendChild(row);
    });
  }

  syncReferenceOutputs() {
    const fixedOutputCount = 2;
    const pictures = this.project.references.filter(ref => ref.type === "picture").slice(0, MAX_REFERENCES.picture);
    const pictureCount = pictures.length;
    const frameOutputCount = pictures.filter(ref => ref.role === "frame").length;
    const videoCount = Math.min(
      MAX_REFERENCES.video,
      this.project.references.filter(ref => ref.type === "video").length,
    );
    const audioCount = Math.min(
      MAX_REFERENCES.audio,
      this.project.references.filter(ref => ref.type === "audio").length,
    );
    const targetOutputCount = fixedOutputCount + pictureCount + frameOutputCount + videoCount + audioCount;
    this.node.outputs ||= [];
    while (this.node.outputs.length > targetOutputCount) {
      this.node.removeOutput(this.node.outputs.length - 1);
    }
    while (this.node.outputs.length < targetOutputCount) {
      const mediaIndex = this.node.outputs.length - fixedOutputCount;
      this.node.addOutput(`media_${mediaIndex + 1}`, "*");
    }
    if (this.node.outputs[0]) {
      this.node.outputs[0].name = "generated_prompt";
      this.node.outputs[0].type = "STRING";
    }
    if (this.node.outputs[1]) {
      this.node.outputs[1].name = "length";
      this.node.outputs[1].type = "INT";
    }
    let outputIndex = fixedOutputCount;
    pictures.forEach((ref, index) => {
      this.node.outputs[outputIndex].name = `image_${index + 1}`;
      this.node.outputs[outputIndex].type = "IMAGE";
      outputIndex += 1;
      if (ref.role === "frame") {
        this.node.outputs[outputIndex].name = `frame_${index + 1}`;
        this.node.outputs[outputIndex].type = "INT";
        outputIndex += 1;
      }
    });
    for (let index = 0; index < videoCount; index += 1) {
      const output = this.node.outputs[outputIndex + index];
      output.name = `video_${index + 1}`;
      output.type = "VIDEO";
    }
    outputIndex += videoCount;
    for (let index = 0; index < audioCount; index += 1) {
      const output = this.node.outputs[outputIndex + index];
      output.name = `audio_${index + 1}`;
      output.type = "AUDIO";
    }
    this.node._widgetSlotsDirty = true;
    this.node.setDirtyCanvas?.(true, true);
  }

  async uploadReferenceImage(ref, file) {
    this.appendLog(`Uploading reference image: ${file.name}`, `upload-${ref.id}`);
    const body = new FormData();
    body.append("image", file, file.name);
    body.append("subfolder", "toyxyz_h3_references");
    body.append("type", "input");
    body.append("overwrite", "false");
    try {
      const response = await api.fetchApi("/upload/image", { method: "POST", body });
      const data = await response.json();
      if (!response.ok || !data.name) throw new Error(data.error || response.statusText || "Upload failed");
      ref.image_filename = data.name;
      ref.image_subfolder = data.subfolder || "toyxyz_h3_references";
      ref.image_type = "input";
      ref.description = "";
      this.commit(); this.renderReferences(); this.renderTimeline();
      this.appendLog(`Reference image uploaded: ${data.name}`, `upload-${ref.id}`);
    } catch (error) {
      this.appendLog(`Image upload failed: ${error.message || error}`, `upload-${ref.id}`, "error");
    }
  }

  async uploadReferenceVideo(ref, file) {
    this.appendLog(`Uploading reference video: ${file.name}`, `upload-${ref.id}`);
    try {
      const uploadId = globalThis.crypto?.randomUUID?.().replaceAll("-", "")
        || `${Date.now()}${Math.random().toString(16).slice(2)}`;
      const chunkCount = Math.max(1, Math.ceil(file.size / VIDEO_UPLOAD_CHUNK_BYTES));
      let data = null;
      for (let index = 0; index < chunkCount; index += 1) {
        const start = index * VIDEO_UPLOAD_CHUNK_BYTES;
        const end = Math.min(file.size, start + VIDEO_UPLOAD_CHUNK_BYTES);
        const query = new URLSearchParams({
          upload_id: uploadId,
          filename: file.name,
          chunk_index: String(index),
          final: index === chunkCount - 1 ? "1" : "0",
        });
        const response = await api.fetchApi(`${VIDEO_UPLOAD_ENDPOINT}?${query}`, {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: file.slice(start, end),
        });
        const responseText = await response.text();
        try {
          data = responseText ? JSON.parse(responseText) : {};
        } catch (_error) {
          throw new Error(`HTTP ${response.status}: ${responseText.slice(0, 300) || response.statusText || "Upload failed"}`);
        }
        if (!response.ok || data.status === "error") {
          throw new Error(data.error || data.message || `HTTP ${response.status}: ${response.statusText || "Upload failed"}`);
        }
        this.appendLog(
          `Uploading reference video: ${((index + 1) / chunkCount * 100).toFixed(1)}%`,
          `upload-${ref.id}`,
        );
      }
      if (!data?.name) throw new Error("Upload completed without a stored video name.");
      ref.video_filename = data.name;
      ref.video_subfolder = data.subfolder || "toyxyz_h3_references";
      ref.video_type = "input";
      const actualDuration = Number(data.duration || 0);
      ref.source_duration = actualDuration > 0 ? actualDuration : 0;
      ref.trim_start = 0;
      ref.timeline_start = 0;
      ref.duration = actualDuration > 0 ? Math.min(15, actualDuration) : 0;
      this.commit(); this.renderReferences(); this.renderTimeline();
      const analysisSuffix = ref.duration > 0
        ? `; selected source interval 0.00–${Number(ref.duration).toFixed(2)} seconds`
        : "";
      this.appendLog(`Reference video uploaded: ${data.name}${analysisSuffix}`, `upload-${ref.id}`);
    } catch (error) {
      this.appendLog(`Video upload failed: ${error.message || error}`, `upload-${ref.id}`, "error");
    }
  }

  async uploadReferenceAudio(ref, file) {
    this.appendLog(`Uploading reference audio: ${file.name}`, `upload-${ref.id}`);
    const body = new FormData();
    body.append("image", file, file.name);
    body.append("subfolder", "toyxyz_h3_references");
    body.append("type", "input");
    body.append("overwrite", "false");
    try {
      const response = await api.fetchApi("/upload/image", { method: "POST", body });
      const responseText = await response.text();
      let data = {};
      try { data = responseText ? JSON.parse(responseText) : {}; }
      catch (_error) { throw new Error(`HTTP ${response.status}: ${responseText.slice(0, 300)}`); }
      if (!response.ok || !data.name) throw new Error(data.error || response.statusText || "Upload failed");
      ref.audio_filename = data.name;
      ref.audio_subfolder = data.subfolder || "toyxyz_h3_references";
      ref.audio_type = "input";
      this.commit(); this.renderReferences();
      this.appendLog(`Reference audio uploaded: ${data.name}`, `upload-${ref.id}`);
    } catch (error) {
      this.appendLog(`Audio upload failed: ${error.message || error}`, `upload-${ref.id}`, "error");
    }
  }

  renderPreview() {
    if (this.autoRunPreview) {
      this.els.preview.textContent = this.autoRunPreview;
      return;
    }
    if (!this.previewData) return;
    this.els.preview.textContent = this.project.enhanced_prompt
      ? this.previewData.video_prompt || this.project.enhanced_prompt
      : "No generated prompt yet. Click Generate Prompt to create one.";
  }

  showAutoRunPrompt(value) {
    const prompt = String(Array.isArray(value) ? value[0] || "" : value || "").trim();
    if (!prompt) return;
    this.autoRunPreview = prompt;
    this.els.preview.textContent = prompt;
    this.appendLog("Auto Run generated prompt is ready.", "auto-run-result");
  }

  load(value) {
    this.project = normalizeProject(value);
    this.autoRunPreview = null;
    if (!this.project.shots.some(shot => shot.id === this.selectedShotId)) this.selectedShotId = this.project.shots[0]?.id;
    this.render();
  }
}

app.registerExtension({
  name: "toyxyz.MinimaxH3Prompter",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalCreated?.apply(this, arguments);
      if (this._minimaxH3PrompterUI) return;
      installStyles();
      const stateWidget = this.widgets?.find(widget => widget.name === "project_data");
      hideWidget(stateWidget);
      const root = document.createElement("div");
      const domWidget = this.addDOMWidget("minimax_h3_prompter_ui", "minimax_h3_prompter_ui", root, {
        getValue: () => "", setValue: () => {},
        getMinHeight: () => UI_HEIGHT, getMaxHeight: () => UI_HEIGHT,
      });
      domWidget.serialize = false;
      this._minimaxH3PrompterUI = new PrompterUI(this, root, stateWidget);
      this.setSize([Math.max(this.size[0], UI_WIDTH), NODE_HEIGHT]);
      this._widgetSlotsDirty = true;
    };

    const originalConfigured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      originalConfigured?.apply(this, arguments);
      setTimeout(() => {
        const widget = this.widgets?.find(item => item.name === "project_data");
        hideWidget(widget);
        this._minimaxH3PrompterUI?.load(widget?.value);
        this.setSize([Math.max(this.size[0], UI_WIDTH), NODE_HEIGHT]);
        this._widgetSlotsDirty = true;
        this.setDirtyCanvas?.(true, true);
      }, 0);
    };

    const originalExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      originalExecuted?.apply(this, arguments);
      this._minimaxH3PrompterUI?.showAutoRunPrompt(message?.auto_run_prompt);
    };

    const originalRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      clearTimeout(this._minimaxH3PrompterUI?.compileTimer);
      this._minimaxH3PrompterUI?.compileController?.abort();
      this._minimaxH3PrompterUI?.enhanceController?.abort();
      originalRemoved?.apply(this, arguments);
    };
  },
});
