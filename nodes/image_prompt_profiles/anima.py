"""Tag-forward hybrid prompts for CircleStone Labs Anima."""

import re

from .default import (CAMERA_INTENT_PROMPT, CAMERA_RESOLUTION_PROMPT,
                      IMAGE_ANALYSIS_PROMPT)


def candidate_budget(input_tags: str, natural_language: str, enhance: str,
                     has_setting: bool) -> int:
    """Bound optional candidates by concrete source information, not a quota."""
    if enhance == "none":
        return 0
    tag_count = len([part for part in input_tags.split(",") if part.strip()])
    prose_units = len([part for part in re.split(r"[,;.!?\n]+", natural_language)
                       if part.strip()])
    density = min(12, tag_count + 2 * prose_units)
    if not has_setting:
        return min(3 if enhance == "strong" else 1, density)
    if enhance == "strong":
        return min(16, 3 + density)
    return min(8, 1 + density // 2)


def budget_instruction(budget: int) -> str:
    return (f"Optional new tag candidate ceiling: {budget}. This is not a target; "
            "use fewer or zero when no grounded visible detail remains. "
            "Explicit source facts do not count against this ceiling.")

SYSTEM_PROMPT = """Write one positive hybrid prompt for CircleStone Labs Anima.
Return ONLY JSON with "tags" (an array of individual Danbooru tag candidates)
and "scene" (concise English scene sentences, or ""), plus
"source_tag_evidence" (an array of {"tag": "...", "evidence": "..."}). The app
validates tags against a local Danbooru vocabulary, then prints the tags first
and the scene on the next line. Never include a tag list inside scene.

Read input_tags and natural_language as one user request. Preserve every explicit
visual fact, including count, subject ownership, degree, color, clothing, body
attributes, pose, gaze, camera crop, placement, object relationships, exclusions,
and exact visible lettering. Explicit user text outranks reference evidence and
presets. A comparison such as 'as if coated in oil' describes appearance, not
a confirmed material. Do not change a girl into a woman or vice versa. Do not
invent new subject traits, garments, gaze, crop, props, or people.
Keep each user-authored numeric prompt weight, including its exact number and
parenthesized syntax such as `(from front:4.92)`. Never round or reinterpret it.
For every tag derived from natural_language, provide the exact substring of
natural_language that explicitly supports it in source_tag_evidence. Quote the shortest relevant
phrase in its original language, not a translation or a generic person noun.
Do not list optional invented details in source_tag_evidence. The app drops
unsupported person-detail tags and may retain an explicit detail as brief
prose when no dictionary tag exists. Input_tags are preserved separately.
TAG ORDER: Keep input_tags in their original relative order, including weighted
tags at the end. Never move camera, quality, artist, or subject tags to the
front by category. Put newly translated or expanded tags AFTER the authored
tags. Expansion does not authorize rearranging the user's tags.

Tags carry the visual inventory. Include the user's valid tags and translate
natural-language visual facts into concrete tag candidates. Add compatible,
useful tags for distinct visible subject traits, clothing, action, framing,
objects, setting, and lighting according to enhancement strength. Prefer
established, specific Danbooru names; use lowercase and spaces for ordinary
tags, score_* for scores, and @ for artist tags. Avoid duplicates, contradictory
strength tiers, made-up compound phrases, and tags for off-frame details.
Represent actions and objects as separate established tags when their combined
relation is not a tag: `sitting`, `bench`, `bag`, not `sitting on bench` or
`bag on ground`. Put the exact relation in the scene.
Before writing the scene, check that each explicit subject count, pose or action,
held object, and setting in natural_language has a corresponding tag. For example,
"stands holding an umbrella" needs both `standing` and `holding umbrella`.
Treat camera words that describe gaze or viewing direction as the viewer's
viewpoint, not a camera device in the picture. Use `looking at viewer` only for
eye contact; use `facing viewer` for body orientation when eye contact is not
established. Use established perspective tags such as `from behind`, `from
above`, or `from below` for the view. In scene prose say "viewer" or "viewpoint"
instead of "camera" for these relations. Do not add `camera`, `holding camera`,
or other camera-prop tags from a viewing angle. If the user explicitly describes
a physical camera as an object, preserve that object and its relationships.
Use the established `1girl` tag for one female person even when described as an
adult woman, and `1boy` for one male person. For a railway station use
`train station`; for its platform use `train station platform`. Use
`backlighting` for light behind a subject. Do not infer daytime or sunlight
from a beach or from an unspecified bright light.
Sunlight on a person does not establish an outdoor location or visible sky.
Clothing such as a bikini does not establish a beach, water, sand, or horizon.
Do not add `sky`, `outdoors`, `blue sky`, or `clear sky` from sunlight or a
portrait crop alone. `glaring` describes a person's expression, not bright
sunlight. Enhance normal and strong add detail only in parts of the setting
that the user actually establishes; an unspecified setting stays unspecified.
Do not add default quality, score, safety, artist, or rendering-style tags.
Preserve any such tags only when the user explicitly supplies them.

The scene is subordinate to the tag list, but it must NOT omit the user's
spatial and bodily instructions for brevity. In concise connected English
sentences, state the subject's placement, orientation, posture,
limb and hand positions, gaze, eye openness, expression, action, and the
positions of other subjects, objects, and explicitly located light sources.
Translate every such fact from natural_language AND input_tags; do not merely
repeat "standing on a beach" when the user also described arms, gaze, or smile.
Use only as many words as the facts need. A complex multi-person scene may need
six or more sentences; preserve every explicit relation without padding.
Use fewer words when the input contains only one fact. Include no invented
pose or position. Do not narrate beauty, hair or eye color, clothing, body
attributes, visual style, quality, mood, weather, or lighting effects; those
belong in tags. You may locate a light behind someone, but do not describe
its glow on their clothes. If no spatial, pose, or object relation exists,
return an empty scene.
For cowboy shot, framing is approximately mid-thigh upward.
Return no Markdown, notes, negatives, or explanation."""

ENHANCEMENT = {
    "none": "Enhance none: tag only the explicit input facts and their direct Danbooru equivalents. Target zero optional detail tags. Keep scene minimal.",
    "normal": "Enhance normal: preserve all input facts, then propose additional candidate tags for compatible setting, object relationships, and lighting only when the input leaves room. Follow the source-dependent candidate budget below; it is a ceiling, not a target. Optional additions must NOT include new hairstyle, hair color, gaze, skin, body traits, garments, or footwear. Cover more than one aspect of the scene, but stop if additions would invent unrelated objects or alter the camera crop. Enrich tags, not scene prose.",
    "strong": (
        "Enhance strong: preserve all input facts, then build a richer tag "
        "inventory from the SAME established scene. Cover explicit subject, "
        "action, clothing, framing, setting, and light facts first. When the "
        "user names a setting, fill genuinely open visible parts with compatible "
        "surfaces, nearby environment, background layers, and light effects. "
        "Follow the source-dependent candidate budget below as a ceiling, "
        "not a target; dictionary lookup may reject some. Without a stated "
        "setting, do not invent one or pad the count. Never add a new hairstyle, "
        "hair color, gaze, skin or body trait, garment, footwear, person, prop, "
        "camera view, or time of day. Use distinct visible details instead of "
        "synonyms. Keep the scene prose spatial and concise."
    ),
}

TAG_ENRICHMENT_PROMPT = """Expand the tag inventory and complete the scene
sentence of an Anima prompt draft.
Return ONLY JSON with "tags" (additional Danbooru tag candidates, excluding
all tags already in draft_tags), "scene" (a revised English scene), and
"source_tag_evidence" (exact source substrings for any newly translated
explicit candidate; omit optional additions from this list).
Find concrete, compatible details that the first pass missed. Cover the same
subject and the visible foreground, surroundings, depth, and light where open.
Follow the source-dependent candidate budget as a ceiling, not a target;
local dictionary lookup may reject some. Never pad
with synonyms or details that contradict the user. User_input is authoritative.
Use only the location and objects established in user_input. Inspect related
dictionary-valid environment, object-relation, and lighting tags rather than
generic adjectives. Do not transfer scenery from a different setting.
Preserve its specified people, clothing, pose, gaze, framing, and objects.
If the user gave no setting, do not invent sky, outdoors, architecture, or
new background objects to meet the candidate target. Lighting may be expanded
with compatible light-effect tags without declaring a location.
This pass may add setting, object-relation, and lighting tags only; do not
propose new hair, skin, face, body, or clothing details. Do not invent another
person, garment, body trait, prop, camera view, time of
day, or light source. Do not add quality, safety, score, artist, or style tags.
Viewing direction refers to the viewer; never add a physical `camera` tag
unless user_input explicitly describes a camera device in the scene.
Use established lowercase Danbooru tags with spaces, not descriptive phrases.
Prefer atomic established tags when a descriptive compound may be absent from
the vocabulary: `railing` rather than `metal railing`, and `pavement` rather
than `concrete floor`. Avoid generic `background` and rendering tags such as
`depth of field`.
Rewrite draft_scene using user_input as the authority. Include EVERY explicit
person placement, orientation, pose, arm or hand position, gaze, eye openness,
smile or other expression, action, and object/light-source location that the
draft may have omitted. Use concise connected sentences, however many are
needed to preserve the explicit facts. Do not invent new pose or location details.
Do not use the scene for colors, clothing, beauty, quality, style, mood, or
lighting effects. All optional visual expansion belongs in tags."""

LIGHT_ENRICHMENT_PROMPT = """Enrich an Anima draft whose user input has NO
specified setting. Return ONLY JSON with "tags" (additional Danbooru candidates)
and "scene" (copy draft_scene exactly). Add only compatible visible lighting
effects from the stated light and silhouette, such as shadow, light rays,
rim lighting, or lens flare when appropriate. Aim for a few distinct useful
tags, never pad. Do not infer beach, sea, sky, cloud, sun disk, weather,
architecture, objects, clothing, facial features, style, or quality. Do not
alter the user's tags, pose, framing, or scene text."""

EDIT_PROMPT = """Revise the existing Anima prompt according to the user's edit.
Return ONLY JSON with "tags" (the complete revised array) and "scene" (a brief
spatial description, or ""). The edit overrides conflicting old tags and scene
facts. Preserve unrelated tags, including custom user tokens, and keep the
scene consistent with the revised tags. Do not add default quality, score,
safety, artist, or style tags. The scene must cover only placement, pose, action,
and object relationships, not style, beauty, appearance, or quality prose.
For gaze and viewing direction say viewer or viewpoint, not camera; preserve
an explicitly requested physical camera object."""

SCENE_REPAIR_PROMPT = """Rewrite ONLY the scene field of the supplied draft.
Preserve its tags array exactly. Compare with the original user input and
retain every explicitly given placement, orientation, posture, limb/hand
position, gaze, eye openness, expression, action, and object/light-source
location. Use concise English sentences; retain all explicit facts even when
the source needs a longer scene.
Remove appearance colors, clothing, beauty, quality, style, mood, weather,
lighting effects, atmosphere, and rendering. Do not invent new scene facts.
For gaze and viewing direction say viewer or viewpoint rather than camera.
Preserve an explicitly requested physical camera object.
Use an empty scene only if the source has no spatial or pose facts.
Return ONLY JSON with "tags" and "scene"."""
