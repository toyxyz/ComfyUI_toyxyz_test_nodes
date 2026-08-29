import importlib.util
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).parent / "nodes" / "minimax_h3_prompter.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_prompter_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def command_prompt(command):
    if "-p" in command:
        return command[command.index("-p") + 1]
    return Path(command[command.index("--file") + 1]).read_text(encoding="utf-8")


class MinimaxH3PrompterTests(unittest.TestCase):
    def compile(self, payload):
        return MODULE.compile_project(json.dumps(payload))

    def test_duration_uses_h3_frame_grid(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{"duration": 5, "visual_action": "A cyclist stops."}],
        })
        self.assertEqual(result["effective_frames"], 124)
        self.assertEqual(result["effective_frames"] % 17, 5)
        self.assertFalse(any("17k+5 grid" in warning for warning in result["warnings"]))
        self.assertTrue(result["video_prompt"].startswith("INPUT DATA ONLY"))
        self.assertIn("mode: T2VA", result["video_prompt"])
        self.assertIn("SHOT_PLAN:\n[Shot 1]", result["video_prompt"])
        self.assertIn("when_music_unspecified: output non_diegetic_music: N/A", result["video_prompt"])

    def test_korean_shot_text_is_preserved_in_qwen_input(self):
        action = "여자가 식탁에서 즐겁게 웃고 있다."
        result = self.compile({
            "mode": "T2VA",
            "shots": [{"duration": 5, "visual_action": action}],
        })
        self.assertIn(f"visual_action: {action}", result["video_prompt"])
        self.assertIn(f"visual_action: {action}", result["llm_prompt"])
        self.assertNotIn("\ufffd", result["llm_prompt"])

    def test_unicode_replacement_character_is_rejected_before_generation(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{"duration": 5, "visual_action": "damaged \ufffd text"}],
        })
        self.assertTrue(any("Unicode replacement character" in error for error in result["errors"]))

    def test_saved_enhancement_survives_subsequent_input_changes(self):
        saved = "integrated_multimodal_description: [Shot 1] Saved enhanced result."
        result = self.compile({
            "mode": "T2VA",
            "shots": [{"duration": 5, "visual_action": "A newly edited raw action."}],
            "enhanced_prompt": saved,
        })
        self.assertIn("A newly edited raw action.", result["draft_video_prompt"])
        self.assertEqual(result["enhanced_prompt"], saved)
        self.assertEqual(result["video_prompt"], saved)

    def test_qwen_enhance_level_is_normalized_and_migrates_legacy_toggle(self):
        default_project, _warnings = MODULE.normalize_project({})
        enabled_project, _warnings = MODULE.normalize_project({"enhance": True})
        strong_project, _warnings = MODULE.normalize_project({"enhance_level": "strong"})
        self.assertFalse(default_project["enhance"])
        self.assertEqual(default_project["enhance_level"], "none")
        self.assertTrue(enabled_project["enhance"])
        self.assertEqual(enabled_project["enhance_level"], "normal")
        self.assertTrue(strong_project["enhance"])
        self.assertEqual(strong_project["enhance_level"], "strong")

    def test_prompt_presets_default_to_none_and_normalize_known_values(self):
        default_project, _warnings = MODULE.normalize_project({})
        self.assertEqual(default_project["shots"][0]["presets"], {
            "camera_angle": "none", "camera_motion": "none", "camera_shot": "none",
            "camera_amplitude": "none", "camera_speed": "none", "style": "none",
        })
        project, _warnings = MODULE.normalize_project({
            "shots": [{"duration": 5, "presets": {
                "camera_angle": "low_angle", "camera_motion": "tracking",
                "camera_shot": "cowboy_shot", "camera_amplitude": "large",
                "camera_speed": "fast", "style": "animation_3d",
            }}]
        })
        presets = project["shots"][0]["presets"]
        self.assertEqual(presets["camera_angle"], "low_angle")
        self.assertEqual(presets["camera_motion"], "tracking")
        self.assertEqual(presets["camera_shot"], "cowboy_shot")
        self.assertEqual(presets["camera_amplitude"], "large")
        self.assertEqual(presets["camera_speed"], "fast")
        self.assertEqual(presets["style"], "animation_3d")

    def test_prompt_presets_are_compiled_as_explicit_camera_and_style_instructions(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{"duration": 5, "visual_action": "A woman walks forward.", "presets": {
                "camera_angle": "low_angle", "camera_motion": "tracking",
                "camera_shot": "cowboy_shot", "camera_amplitude": "large",
                "camera_speed": "fast", "style": "animation_3d",
            }}],
        })
        prompt = result["draft_video_prompt"]
        self.assertIn("PROMPT_PRESETS:", prompt)
        self.assertIn("style: polished 3D animation", prompt)
        self.assertIn("camera_angle: low angle looking upward", prompt)
        self.assertIn("camera_motion: tracking shot following the moving subject", prompt)
        self.assertIn("camera_shot: cowboy shot", prompt)
        self.assertIn("camera_amplitude: with large amplitude", prompt)
        self.assertIn("camera_speed: at fast speed", prompt)
        self.assertIn("combine selected motion, amplitude, and speed", prompt)
        self.assertIn("scope: each block applies only to its named shot", prompt)
        self.assertIn("style_expression: when a shot has style, state that style naturally in the opening sentence", prompt)
        self.assertIn("status: mandatory explicit user selections", prompt)
        self.assertIn("[Shot 1]\nstyle: polished 3D animation", prompt)

    def test_prompt_presets_are_isolated_per_shot(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [
                {"duration": 2.5, "visual_action": "A woman walks.", "presets": {
                    "camera_motion": "arc", "style": "animation_3d",
                }},
                {"duration": 2.5, "visual_action": "She stops.", "presets": {
                    "camera_motion": "static", "camera_shot": "close_up",
                }},
            ],
        })
        prompt = result["draft_video_prompt"]
        preset_section = prompt.split("PROMPT_PRESETS:\n", 1)[1].split("\n\nTARGET_REQUEST:", 1)[0]
        self.assertIn("[Shot 1]\nstyle: polished 3D animation with coherent modeled forms", preset_section)
        self.assertIn("camera_motion: arc shot moving around the subject", preset_section)
        self.assertIn("[Shot 2]\ncamera_motion: static shot with camera position", preset_section)
        self.assertIn("camera_shot: close-up centered on the face", preset_section)
        self.assertEqual(preset_section.count("style: polished 3D animation with coherent modeled forms"), 1)

    def test_legacy_global_presets_migrate_to_first_shot_only(self):
        project, _warnings = MODULE.normalize_project({
            "presets": {"camera_motion": "arc"},
            "shots": [{"duration": 2.5}, {"duration": 2.5}],
        })
        self.assertEqual(project["shots"][0]["presets"]["camera_motion"], "arc")
        self.assertEqual(project["shots"][1]["presets"]["camera_motion"], "none")

    def test_extended_camera_motion_presets_compile(self):
        expected = {
            "dolly_left": "camera dolly left", "dolly_right": "camera dolly right",
            "dolly_zoom_in": "move the camera forward while zooming out",
            "dolly_zoom_out": "move the camera backward while zooming in",
            "crane_up": "crane movement lifting", "crane_down": "crane movement lowering",
            "orbit_left": "orbit left around", "orbit_right": "orbit right around",
            "follow": "follow shot continuously following", "handheld": "natural handheld camera movement",
        }
        for motion, phrase in expected.items():
            with self.subTest(motion=motion):
                result = self.compile({
                    "mode": "T2VA",
                    "shots": [{"duration": 5, "visual_action": "A subject moves.",
                               "presets": {"camera_motion": motion}}],
                })
                self.assertIn("camera_motion:", result["draft_video_prompt"])
                self.assertIn(phrase, result["draft_video_prompt"])

    def test_document_style_presets_normalize_and_compile(self):
        expected_keys = {
            "anime_1990s", "retro_anime_motion_graphics", "retro_anime_noir_jazz",
            "contemporary_anime", "western_cartoon", "vhs_analog", "cyberpunk_live_action",
            "epic_dark_fantasy", "high_saturation_commercial", "photoreal_graphic_hybrid",
            "phone_ugc_ad", "sprite_16bit", "sketch_anime", "lineart_anime",
            "anamorphic_cinema", "film_noir", "neo_noir", "horror_cinema",
            "scifi_mystery", "retro_futuristic_scifi", "premium_product_film",
            "japanese_commercial", "food_commercial", "music_video", "anime_music_video",
            "graphic_poster_animation", "minimalist_motion_design", "game_cinematic",
            "dark_retro_fantasy", "contemporary_action_anime", "vhs_rental_movie",
            "dark_medieval_fantasy", "authentic_smartphone_vlog", "cinematic_35mm",
            "analog_horror_1990s", "figurine_animation",
        }
        self.assertTrue(expected_keys.issubset(MODULE.STYLE_PRESET_PROMPTS))
        for style in expected_keys:
            with self.subTest(style=style):
                result = self.compile({
                    "mode": "T2VA",
                    "shots": [{"duration": 5, "visual_action": "A subject moves.",
                               "presets": {"style": style}}],
                })
                self.assertIn(MODULE.STYLE_PRESET_PROMPTS[style], result["draft_video_prompt"])

    def test_general_style_presets_do_not_force_scene_content(self):
        generic_keys = (
            "animation_2d", "animation_3d", "cinematic_live_action",
            "smartphone_video", "photoreal_live_action", "documentary", "stop_motion",
        )
        combined = "\n".join(MODULE.STYLE_PRESET_PROMPTS[key] for key in generic_keys).lower()
        forbidden = (
            "locked straight-on", "dynamic framing", "handheld smartphone framing",
            "restrained camera movement", "slow suspenseful camera movement",
            "elegant slow camera movement", "rhythmic camera movement",
            "visually motivated cuts", "editing rhythm", "clean transitions",
            "rain-soaked streets", "monumental landscapes", "ancient armor and robes",
            "spacecraft interiors", "slow-motion splashes and particles",
            "dramatic natural landscapes",
        )
        for phrase in forbidden:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, combined)

    def test_live_action_cinematic_document_presets_are_complete(self):
        expected = {
            "modern_cinematic_live_action", "prestige_drama", "intimate_relationship_drama",
            "short_form_microdrama", "golden_hour_road_movie", "natural_light_indie_film",
            "mountain_adventure_cinema", "survival_expedition_film",
            "blue_hour_urban_cinema", "rainy_city_one_take", "urban_editorial",
            "night_city_timelapse", "modern_neo_noir", "classic_film_noir",
            "crime_thriller", "thriller_1990s", "cinematic_horror_live_action",
            "found_footage_horror", "consumer_camcorder_horror",
            "observational_documentary", "workplace_mockumentary", "reality_tv_documentary",
            "grounded_martial_arts_cinema", "gritty_close_quarters_action",
            "dark_fantasy_live_action", "neon_cyberpunk_cinema", "dark_dystopian_scifi",
            "prestige_scifi_drama", "high_fashion_editorial", "korean_fashion_campaign",
            "streetwear_fashion_film", "minimalist_premium_product",
            "luxury_automotive_commercial", "performance_car_commercial",
            "food_macro_commercial", "dark_surreal_commercial", "ultra_realistic_pov",
            "smartphone_ugc", "film_1970s", "cinema_1980s", "cinema_1990s",
            "early_2000s_digital_cinema", "modern_digital_cinema",
        }
        self.assertTrue(expected.issubset(MODULE.STYLE_PRESET_PROMPTS))
        for style in expected:
            with self.subTest(style=style):
                self.assertGreaterEqual(len(MODULE.STYLE_PRESET_PROMPTS[style].split()), 8)

    def test_style_ui_uses_labeled_categories(self):
        source = (Path(__file__).parent / "web" / "minimax_h3_prompter.js").read_text(encoding="utf-8")
        self.assertIn("const STYLE_PRESET_GROUPS = [", source)
        for category in (
            "General cinema / drama", "Natural light / outdoor cinema", "Urban cinema",
            "Noir / thriller", "Horror / found footage", "Documentary / reality",
            "Action / fantasy", "Science fiction", "Fashion / editorial",
            "Commercial / product", "POV / social video", "Film / era",
            "Physical character animation", "Animation / graphic", "Music / game / hybrid",
        ):
            self.assertIn(category, source)

    def test_general_style_presets_have_balanced_detail(self):
        general = (
            "animation_2d", "animation_3d", "cinematic_live_action", "smartphone_video",
            "photoreal_live_action", "documentary", "stop_motion",
        )
        for style in general:
            with self.subTest(style=style):
                self.assertGreaterEqual(len(MODULE.STYLE_PRESET_PROMPTS[style].split()), 12)

    def test_raw_model_prompt_formats_system_and_user_channels(self):
        raw = MODULE._format_raw_model_prompt("SYSTEM RULES", "CURRENT USER DATA")
        self.assertEqual(
            raw,
            "===== SYSTEM PROMPT =====\nSYSTEM RULES\n\n"
            "===== USER PROMPT =====\nCURRENT USER DATA",
        )

    def test_qwen_rich_enhance_contract_expands_without_changing_events(self):
        contract = MODULE.ENHANCED_COMMON_LLM_SYSTEM_RULES + MODULE.SYSTEM_PROMPT_CONFIG["enhance_addendum"]
        self.assertIn("creative cinematic rewriter", contract)
        self.assertIn("early-to-middle-to-late progression", contract)
        self.assertIn("Track which hand holds every object", contract)
        self.assertIn("NORMAL DEVELOPMENT REWRITER", contract)
        self.assertIn("materially fuller and more explicit", contract)
        self.assertIn("KEYFRAME DEVELOPMENT", contract)
        self.assertIn("every important change", contract)
        self.assertIn("camera preset governs the motion path", contract)
        self.assertIn("Make locomotion visible through displacement", contract)
        self.assertIn("object custody", contract)
        self.assertIn("non_diegetic_music", contract)

    def test_qwen_strong_enhance_adds_rewriter_depth_and_larger_budget(self):
        self.assertIn("STRONG CREATIVE REWRITER", MODULE.STRONG_ENHANCE_ADDENDUM)
        self.assertIn("opening composition", MODULE.STRONG_ENHANCE_ADDENDUM)
        self.assertIn("actively invent coherent production detail", MODULE.STRONG_ENHANCE_ADDENDUM)
        self.assertIn("minor anonymous background elements", MODULE.STRONG_ENHANCE_ADDENDUM)
        self.assertIn("musical treatment", MODULE.STRONG_ENHANCE_ADDENDUM)
        self.assertTrue(all(
            MODULE.STRONG_ENHANCE_ADDENDUM in MODULE.STRONG_MODE_LLM_SYSTEM_PROMPTS[mode]
            for mode in MODULE.STRONG_MODE_LLM_SYSTEM_PROMPTS
        ))
        self.assertTrue(all(
            "NORMAL DEVELOPMENT REWRITER" not in MODULE.STRONG_MODE_LLM_SYSTEM_PROMPTS[mode]
            for mode in MODULE.STRONG_MODE_LLM_SYSTEM_PROMPTS
        ))
        project, _warnings = MODULE.normalize_project({
            "mode": "T2VA", "enhance_level": "strong",
            "shots": [{"duration": 5, "visual_action": "A woman walks forward."}],
        })
        prompt = MODULE.build_video_prompt(project, 5.17)
        self.assertIn("recommended_english_words: 320-520", prompt)

    def test_korean_i2va_style_repetition_and_visibility_are_compiled_as_locks(self):
        result = self.compile({
            "mode": "I2VA",
            "enhance": True,
            "shots": [{
                "duration": 5,
                "visual_action": (
                    "여자 캐릭터가 카메라를 향해 걸어온다. "
                    "남자가 양손으로 여자 캐릭터의 가슴을 주무른다. 3D 애니메이션."
                ),
            }],
            "references": [{"type": "picture", "role": "first_frame", "filename": "frame.png"}],
        })
        prompt = result["draft_video_prompt"]
        self.assertIn("TARGET_STYLE_LOCK:\ncanonical_style: 3D CG animation", prompt)
        self.assertIn("REFERENCE_MEDIUM_CONTRACT:", prompt)
        self.assertIn("exclude_incompatible_source_presentation", prompt)
        self.assertIn("semantic_lock: translate faithfully", prompt)
        self.assertIn("motion_semantics_contract:", prompt)
        self.assertIn("preserve the source verb at its original specificity", prompt)
        self.assertIn("continuous or repeated motion", prompt)
        self.assertIn("Preserve stated actors, limb or hand count, contact target", prompt)
        self.assertIn("ACTION_VISIBILITY_LOCK:", prompt)
        self.assertIn("recommended_english_words: 180-280", prompt)

    def test_target_style_application_is_mode_aware(self):
        cases = {
            "I2VA": "begin Shot 1 with this style and maintain it throughout",
            "FL2VA": "explicitly requested style transition",
            "L2VA": "converge to Picture 1's exact final-frame medium",
            "REF2VA": "preserve a conflicting source medium only when mixed-media treatment is explicitly requested",
        }
        references = {
            "I2VA": [{"type": "picture", "role": "first_frame", "filename": "a.png"}],
            "FL2VA": [
                {"type": "picture", "role": "first_frame", "filename": "a.png"},
                {"type": "picture", "role": "last_frame", "filename": "b.png"},
            ],
            "L2VA": [{"type": "picture", "role": "last_frame", "filename": "a.png"}],
            "REF2VA": [{"type": "picture", "role": "subject_identity", "strength": "strong", "filename": "a.png"}],
        }
        for mode, expected in cases.items():
            with self.subTest(mode=mode):
                result = self.compile({
                    "mode": mode,
                    "shots": [{"duration": 5, "visual_action": "A character moves in 3D animation."}],
                    "references": references[mode],
                })
                self.assertIn(expected, result["draft_video_prompt"])
                self.assertIn("material_policy:", result["draft_video_prompt"])

    def test_optional_qwen_input_locks_are_omitted_when_not_applicable(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{"duration": 5, "visual_action": "A lamp remains on."}],
        })
        prompt = result["draft_video_prompt"]
        self.assertNotIn("TARGET_STYLE_LOCK:", prompt)
        self.assertNotIn("REFERENCE_MEDIUM_CONTRACT:", prompt)
        self.assertIn("motion_semantics_contract:", prompt)
        self.assertNotIn("ACTION_VISIBILITY_LOCK:", prompt)
        self.assertNotIn("OUTPUT_BUDGET:", prompt)

    def test_auto_mode_resolves_from_exact_reference_layout(self):
        cases = [
            ([], "T2VA"),
            ([{"type": "picture", "role": "first_frame"}], "I2VA"),
            ([
                {"type": "picture", "role": "first_frame"},
                {"type": "picture", "role": "last_frame"},
            ], "FL2VA"),
            ([{"type": "picture", "role": "last_frame"}], "L2VA"),
            ([{"type": "picture", "role": "reference"}], "REF2VA"),
            ([{"type": "video", "role": "motion", "duration": 3}], "REF2VA"),
            ([
                {"type": "picture", "role": "first_frame"},
                {"type": "picture", "role": "subject_identity"},
            ], "REF2VA"),
        ]
        for references, expected in cases:
            with self.subTest(expected=expected):
                result = self.compile({
                    "mode": "AUTO",
                    "shots": [{"duration": 5, "visual_action": "The subject moves."}],
                    "references": references,
                })
                self.assertEqual(result["resolved_mode"], expected)
                self.assertEqual(result["project"]["mode"], expected)
                self.assertEqual(result["mode_selection"], "AUTO")
                self.assertFalse(result["errors"])

    def test_auto_mode_does_not_silently_ignore_extra_or_reversed_anchors(self):
        for references in ([
            {"type": "picture", "role": "last_frame"},
            {"type": "picture", "role": "first_frame"},
        ], [
            {"type": "picture", "role": "first_frame"},
            {"type": "picture", "role": "last_frame"},
            {"type": "audio", "role": "sound_effect"},
        ]):
            with self.subTest(references=references):
                result = self.compile({
                    "mode": "AUTO",
                    "shots": [{"duration": 5, "visual_action": "The subject moves."}],
                    "references": references,
                })
                self.assertEqual(result["resolved_mode"], "REF2VA")

    def test_manual_mode_selection_remains_available(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{"duration": 5, "visual_action": "A cyclist stops."}],
            "references": [{"type": "picture", "role": "reference"}],
        })
        self.assertEqual(result["resolved_mode"], "T2VA")
        self.assertEqual(result["mode_selection"], "T2VA")

    def test_fl2va_requires_both_frame_roles(self):
        invalid = self.compile({
            "mode": "FL2VA",
            "shots": [{"duration": 8, "visual_action": "Open an umbrella."}],
            "references": [{"type": "picture", "role": "first_frame"}],
        })
        self.assertTrue(invalid["errors"])
        self.assertIn("last_frame", invalid["errors"][0])

        valid = self.compile({
            "mode": "FL2VA",
            "shots": [{"duration": 8, "visual_action": "Open an umbrella."}],
            "references": [
                {"type": "picture", "role": "first_frame"},
                {"type": "picture", "role": "last_frame"},
            ],
        })
        self.assertFalse(valid["errors"])
        self.assertIn("mode: FL2VA", valid["video_prompt"])
        self.assertIn("anchor: exact opening frame", valid["video_prompt"])
        self.assertIn("anchor_time_seconds: 8.00", valid["video_prompt"])

        wrong_order = self.compile({
            "mode": "FL2VA",
            "shots": [{"duration": 8, "visual_action": "Open an umbrella."}],
            "references": [
                {"type": "picture", "role": "last_frame"},
                {"type": "picture", "role": "first_frame"},
            ],
        })
        self.assertTrue(wrong_order["errors"])

    def test_reference_labels_follow_per_type_order(self):
        result = self.compile({
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "The subject walks."}],
            "references": [
                {"type": "picture", "role": "subject_identity"},
                {"type": "video", "role": "motion", "duration": 3},
                {"type": "picture", "role": "subject_identity"},
                {"type": "audio", "role": "voice_timbre"},
            ],
        })
        prompt = result["video_prompt"]
        self.assertLess(prompt.index("<Subject 1>"), prompt.index("<Video 1>"))
        self.assertLess(prompt.index("<Video 1>"), prompt.index("<Subject 2>"))
        self.assertIn("<Subject 1>\nsource: <Picture 1>", prompt)
        self.assertIn("<Subject 2>\nsource: <Picture 2>", prompt)
        self.assertIn("<Audio 1>\nsource: <Audio 1>", prompt)
        self.assertIn("role: voice_delivery", prompt)
        self.assertIn("REFERENCE_PLAN:", prompt)
        self.assertNotIn("subject_definitions:", prompt)

    def test_generic_picture_becomes_a_reusable_weak_subject(self):
        result = self.compile({
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "A woman sits on a chair."}],
            "references": [{
                "type": "picture",
                "role": "reference",
                "description": "A cute anime girl",
            }],
        })
        prompt = result["video_prompt"]
        self.assertNotIn("reference reference", prompt)
        self.assertIn("<Subject 1>\nsource: <Picture 1>", prompt)
        self.assertIn("retention_output_marker: weak_reference", prompt)
        self.assertIn("broad subject appearance similarity only", prompt)
        self.assertIn("exclude source setting, style, composition, camera, lighting, palette, pose, and action", prompt)
        self.assertNotIn("\n<Picture 1> is", prompt)

    def test_subject_strength_maps_to_h3_retention_marker_and_contract(self):
        cases = {
            "weak": ("weak_reference", "broad subject appearance similarity only"),
            "normal": ("partially_preserved", "core subject identity and primary visible appearance"),
            "strong": ("fully_preserved", "complete visible subject identity and appearance plus that subject's source visual medium/rendering style"),
        }
        for strength, (marker, contract) in cases.items():
            with self.subTest(strength=strength):
                result = self.compile({
                    "mode": "REF2VA",
                    "shots": [{"duration": 5, "visual_action": "@hero walks."}],
                    "references": [{
                        "type": "picture", "role": "subject_identity",
                        "strength": strength, "alias": "hero",
                    }],
                })
                prompt = result["draft_video_prompt"]
                self.assertIn("role: subject_identity", prompt)
                self.assertIn(f"input_strength_for_definition_scope_only: {strength}", prompt)
                self.assertIn(f"retention_output_marker: {marker}", prompt)
                self.assertIn(f"contract: {contract}", prompt)
                if strength == "strong":
                    self.assertIn("preserve the style independently per subject", prompt)
                    self.assertIn("exclude source setting, composition", prompt)
                else:
                    self.assertIn("exclude source setting, style, composition", prompt)

    def test_ref2va_retention_line_plan_uses_exact_guide_syntax_and_shot_scope(self):
        result = self.compile({
            "mode": "REF2VA",
            "shots": [
                {"duration": 2.5, "visual_action": "@hero enters."},
                {"duration": 2.5, "visual_action": "The camera holds on the room."},
            ],
            "references": [{
                "type": "picture", "role": "subject_identity",
                "strength": "strong", "alias": "hero",
            }],
        })
        model = MODULE._reference_model(result["project"])
        self.assertEqual(model["label_plan"]["<Subject 1>"]["applicable_shots"], [1])
        self.assertEqual(
            model["label_plan"]["<Subject 1>"]["retention_prefix"],
            "<Subject 1> (appears in [Shot 1]): fully_preserved -",
        )
        self.assertIn(
            "RETENTION_LINE_PLAN:\n<Subject 1> (appears in [Shot 1]): fully_preserved -",
            result["llm_prompt"],
        )
        self.assertNotIn("RETENTION OUTPUT MARKERS:", result["llm_prompt"])

    def test_ref2va_unused_aliased_subject_is_excluded_from_the_label_plan(self):
        result = self.compile({
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "@hero walks through the room."}],
            "references": [
                {"type": "picture", "role": "subject_identity", "alias": "unused"},
                {"type": "picture", "role": "subject_identity", "alias": "hero"},
            ],
        })
        model = MODULE._reference_model(result["project"])
        self.assertEqual(list(model["label_plan"]), ["<Subject 1>"])
        self.assertEqual(model["label_plan"]["<Subject 1>"]["source"], "<Picture 2>")
        self.assertNotIn("@unused", result["draft_video_prompt"])

    def test_ref2va_environment_subject_persists_after_its_last_alias_mention(self):
        result = self.compile({
            "mode": "REF2VA",
            "shots": [
                {"duration": 2, "visual_action": "@place contains @hero."},
                {"duration": 2, "visual_action": "@hero crosses the room."},
                {"duration": 2, "visual_action": "@hero sits down."},
            ],
            "references": [
                {"type": "picture", "role": "subject_identity", "alias": "place"},
                {"type": "picture", "role": "subject_identity", "alias": "hero"},
            ],
        })
        plan = MODULE._reference_model(result["project"])["label_plan"]
        self.assertEqual(plan["<Subject 1>"]["applicable_shots"], [1, 2, 3])

    def test_ref2va_character_scope_uses_only_authored_visible_shots(self):
        result = self.compile({
            "mode": "REF2VA",
            "shots": [
                {"duration": 2.5, "visual_action": "@man stands on a beach."},
                {"duration": 2.5, "visual_action": "@girl approaches @man."},
                {"duration": 2.5, "visual_action": "@girl kisses @man."},
                {"duration": 2.5, "visual_action": "@man faints."},
                {"duration": 2.5, "visual_action": "@girl lies on @man."},
                {"duration": 2.5, "visual_action": "@man struggles beneath @girl."},
            ],
            "references": [
                {"type": "picture", "role": "subject_identity", "alias": "girl", "strength": "normal"},
                {"type": "picture", "role": "subject_identity", "alias": "man", "strength": "normal"},
            ],
        })
        plan = MODULE._reference_model(result["project"])["label_plan"]
        self.assertEqual(plan["<Subject 1>"]["applicable_shots"], [2, 3, 5, 6])
        self.assertEqual(plan["<Subject 2>"]["applicable_shots"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(
            plan["<Subject 1>"]["retention_prefix"],
            "<Subject 1> (appears in [Shot 2], [Shot 3], [Shot 5], [Shot 6]): partially_preserved -",
        )

    def test_ref2va_retention_line_plan_formats_picture_video_and_audio_roles(self):
        result = self.compile({
            "mode": "REF2VA",
            "shots": [
                {"duration": 2.5, "visual_action": "Opening action."},
                {"duration": 2.5, "visual_action": "Closing action."},
            ],
            "references": [
                {"type": "picture", "role": "first_frame"},
                {"type": "picture", "role": "last_frame"},
                {"type": "video", "role": "motion", "duration": 1.0,
                 "timeline_start": 3.0},
                {"type": "audio", "role": "voice_delivery"},
            ],
        })
        plan = MODULE._reference_model(result["project"])["label_plan"]
        self.assertEqual(
            plan["<Picture 1>"]["retention_prefix"],
            "<Picture 1> ([Shot 1] first frame): fully_preserved -",
        )
        self.assertEqual(
            plan["<Picture 2>"]["retention_prefix"],
            "<Picture 2> ([Shot 2] final frame): fully_preserved -",
        )
        self.assertEqual(
            plan["<Video 1>"]["retention_prefix"],
            "<Video 1> (applies to [Shot 2]): weak_reference -",
        )
        self.assertEqual(
            plan["<Audio 1>"]["retention_prefix"],
            "<Audio 1>: reference -",
        )

    def test_ref2va_retention_prefix_repair_preserves_descriptions_and_order(self):
        result = self.compile({
            "mode": "REF2VA",
            "shots": [
                {"duration": 2.5, "visual_action": "@hero enters."},
                {"duration": 2.5, "visual_action": "@hero sits."},
            ],
            "references": [
                {"type": "picture", "role": "subject_identity", "alias": "hero",
                 "strength": "normal"},
                {"type": "audio", "role": "voice_delivery"},
            ],
        })
        plan = MODULE._reference_model(result["project"])["label_plan"]
        malformed = (
            "subject_definitions:\n...\n\nsummary:\n...\n\n"
            "retention_analysis:\n"
            "<Audio 1>: reference; preserve the measured delivery.\n"
            "<Subject 1>: Shots 1, 2; partially_preserved; preserve the core identity.\n\n"
            "detailed_description:\n[Shot 1] ...\n[Shot 2] ...\n\n"
            "overall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A"
        )
        repaired = MODULE._enforce_retention_line_plan(malformed, plan)
        retention = MODULE._ref_prompt_sections(repaired)["retention_analysis"]
        self.assertEqual(
            retention.splitlines(),
            [
                "<Subject 1> (appears in [Shot 1], [Shot 2]): partially_preserved - preserve the core identity.",
                "<Audio 1>: reference - preserve the measured delivery.",
            ],
        )

    def test_reference_alias_is_normalized_and_replaced(self):
        result = self.compile({
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "@hero dances on the floor."}],
            "references": [{
                "type": "picture",
                "role": "subject_identity",
                "alias": "hero",
                "description": "A red-haired anime heroine",
            }],
        })
        self.assertEqual(result["project"]["references"][0]["alias"], "@hero")
        self.assertNotIn("@hero", result["video_prompt"])
        self.assertIn("<Subject 1> dances on the floor.", result["video_prompt"])

    def test_reference_alias_spaces_are_normalized_and_duplicates_rejected(self):
        result = self.compile({
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "@hero_girl waves."}],
            "references": [
                {"type": "picture", "role": "subject_identity", "alias": "hero girl"},
                {"type": "audio", "role": "voice_timbre", "alias": "@hero_girl"},
            ],
        })
        self.assertEqual(result["project"]["references"][0]["alias"], "@hero_girl")
        self.assertTrue(any("aliases must be unique" in error for error in result["errors"]))

    def test_reference_alias_replacement_respects_token_boundary_and_case(self):
        result = self.compile({
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "@HERO waves beside @hero_extra."}],
            "references": [{"type": "picture", "role": "subject_identity", "alias": "hero"}],
        })
        self.assertIn("<Subject 1> waves beside @hero_extra.", result["video_prompt"])

    def test_video_alias_is_replaced_with_numbered_video_label(self):
        result = self.compile({
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "Follow @walkclip's camera movement."}],
            "references": [{
                "type": "video", "role": "camera", "alias": "walkclip", "duration": 5,
            }],
        })
        self.assertEqual(result["project"]["references"][0]["alias"], "@walkclip")
        self.assertNotIn("@walkclip", result["video_prompt"])
        self.assertIn("Follow <Video 1>'s camera movement.", result["video_prompt"])

    def test_reference_limits_are_errors(self):
        references = [
            {"type": "picture", "role": "reference", "description": str(index)}
            for index in range(10)
        ]
        result = self.compile({
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "Test."}],
            "references": references,
        })
        self.assertTrue(any("at most 9 reference images" in error for error in result["errors"]))

    def test_legacy_dialogue_is_migrated_into_visual_action(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{
                "duration": 5,
                "visual_action": "A woman turns toward the camera",
                "dialogue": "다시 시작하자.",
                "dialogue_speaker": "S2",
                "dialogue_language": "Korean",
                "dialogue_mode": "voiceover",
                "dialogue_delivery": "The woman with a calm low voice",
            }],
        })
        prompt = result["video_prompt"]
        self.assertIn(
            "visual_action: A woman turns toward the camera\n"
            "The woman with a calm low voice (S2) says in an "
            "off-screen voiceover: <d>[Korean] 다시 시작하자.</d> while the corresponding "
            "on-screen character's lips remain completely closed.",
            prompt,
        )
        self.assertNotIn("dialogue", result["project"]["shots"][0])
        self.assertNotIn("vocal_source:", prompt)
        self.assertNotIn("verbatim_dialogue_or_lyrics:", prompt)

    def test_dialogue_language_defaults_to_english(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{
                "duration": 5,
                "visual_action": "A speaker faces the camera.",
                "dialogue": "Hello there.",
            }],
        })
        self.assertIn(
            "The on-screen speaker (S1) says: "
            "<d>[English] Hello there.</d>",
            result["video_prompt"],
        )

    def test_prefabricated_dialogue_is_rebuilt_with_speaker_and_language_contract(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{
                "duration": 5,
                "dialogue": "<d>[Korean] 방구 발싸!</d>",
                "dialogue_speaker": "S1",
                "dialogue_language": "English",
            }],
        })
        self.assertIn(
            "visual_action: The on-screen speaker (S1) says: "
            "<d>[Korean] 방구 발싸!</d>",
            result["video_prompt"],
        )

    def test_malformed_prefabricated_dialogue_tag_is_rewrapped(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{"duration": 5, "dialogue": "<d>Hello"}],
        })
        self.assertIn(
            "visual_action: The on-screen speaker (S1) says: "
            "<d>[English] Hello</d>",
            result["video_prompt"],
        )

    def test_visible_text_is_safely_quoted(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{"duration": 5, "visible_text": 'Say "hello"\nnow'}],
        })
        self.assertIn(
            r'visual_action: A visible on-screen text element reads "Say \"hello\" now".',
            result["video_prompt"],
        )
        self.assertNotIn("visible_text", result["project"]["shots"][0])

    def test_visual_action_is_the_only_dialogue_and_visible_text_ui(self):
        source = (MODULE_PATH.parent.parent / "web" / "minimax_h3_prompter.js").read_text(encoding="utf-8")
        self.assertIn("Visual / action / camera / dialogue / text / sound / music", source)
        self.assertNotIn('data-el="raw-preview"', source)
        self.assertNotIn('<span class="mmh3p-label">Raw prompt</span>', source)
        self.assertIn(".mmh3p-visual-action-field { flex:1 1 auto; min-height:0", source)
        self.assertIn("height:100%; min-height:0; resize:none; overflow:auto", source)
        self.assertIn(".mmh3p-main > .mmh3p-grid { flex:1 1 0; min-height:0; }", source)
        self.assertIn(".mmh3p-references-panel { display:flex; flex-direction:column; min-height:0", source)
        self.assertIn(".mmh3p-reference-list { flex:1 1 auto; height:auto; min-height:0", source)
        self.assertNotIn('<div class="mmh3p-label">Selected shot</div>', source)
        prompt_index = source.index('<div class="mmh3p-label">Prompt</div>')
        enhanced_index = source.index('<span class="mmh3p-label">Generated Prompt</span>')
        references_index = source.index('<div class="mmh3p-label">References</div>')
        self.assertLess(prompt_index, enhanced_index)
        self.assertLess(enhanced_index, references_index)
        for field in (
            "dialogue_speaker", "dialogue_language", "dialogue_mode",
            "dialogue_delivery", "dialogue", "visible_text", "camera_framing",
            "camera_angle", "camera_motion", "transition",
        ):
            self.assertNotIn(f'data-shot="{field}"', source)

    def test_visual_action_dialogue_and_visible_text_receive_dynamic_content_locks(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{
                "duration": 5,
                "visual_action": '여성이 총을 쏘며 "총 발싸!"라고 말하고 간판에는 "OPEN"이라고 쓰여 있다.',
            }],
        })
        llm_prompt = result["llm_prompt"]
        self.assertIn("INPUT-DERIVED CONTENT LOCKS", llm_prompt)
        self.assertIn(
            "vocal lock: copy this block character-for-character, including every space and punctuation mark:",
            llm_prompt,
        )
        self.assertIn("Use speaker ID (S1) once before the block", llm_prompt)
        self.assertIn("Precede (S1) with a visible speaker identity", llm_prompt)
        self.assertIn("never begin the clause with bare (S1)", llm_prompt)
        self.assertIn("Never copy this instruction, a checklist", llm_prompt)
        self.assertNotIn("any gunshot, impact, abrupt recoil", llm_prompt)
        self.assertIn('visible text: preserve "OPEN" verbatim', llm_prompt)

    def test_unquoted_vocal_cue_still_locks_the_first_speaker_id(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{"duration": 5, "visual_action": "여성이 총발사라고 말한다."}],
        })
        self.assertIn(
            "[Shot 1] contains explicit vocal content in visual_action: the first vocal source must use (S1)",
            result["llm_prompt"],
        )

    def test_legacy_audio_fields_migrate_to_unified_visual_action(self):
        result = self.compile({
            "version": 13,
            "mode": "T2VA",
            "shots": [{
                "duration": 5,
                "visual_action": "A door closes.",
                "diegetic_sound": "A synchronized wooden door slam.",
            }],
            "overall_soundscape": "Quiet indoor room tone with distant ventilation.",
            "non_diegetic_music": "Slow sparse piano at a low volume.",
        })
        prompt = result["video_prompt"]
        action = result["project"]["shots"][0]["visual_action"]
        shot_sound = "Synchronized physical sound: A synchronized wooden door slam."
        soundscape = "Overall soundscape: Quiet indoor room tone with distant ventilation."
        music = "Non-diegetic music: Slow sparse piano at a low volume."
        self.assertIn(shot_sound, action)
        self.assertIn(soundscape, action)
        self.assertIn(music, action)
        self.assertIn(shot_sound, prompt)
        self.assertIn(soundscape, prompt)
        self.assertIn(music, prompt)
        self.assertIn("AUDIO_POLICY:", prompt)
        self.assertNotIn("AUDIO_INPUTS:", prompt)
        self.assertNotIn("diegetic_sound", result["project"]["shots"][0])
        self.assertNotIn("overall_soundscape", result["project"])
        self.assertNotIn("non_diegetic_music", result["project"])

        source = (MODULE_PATH.parent.parent / "web" / "minimax_h3_prompter.js").read_text(encoding="utf-8")
        self.assertNotIn('data-shot="diegetic_sound"', source)
        self.assertNotIn('data-project="overall_soundscape"', source)
        self.assertNotIn('data-project="non_diegetic_music"', source)
        self.assertNotIn("mmh3p-audio-grid", source)
        self.assertIn("dialogue / text / sound / music", source)

    def test_python_normalization_uses_ui_minimum_shot_duration(self):
        result = self.compile({
            "mode": "T2VA",
            "requested_duration": 0.1,
            "shots": [{"duration": 0.1}],
        })
        self.assertEqual(result["project"]["requested_duration"], 0.25)
        self.assertEqual(result["project"]["shots"][0]["duration"], 0.25)

    def test_duration_fitting_preserves_each_shot_minimum(self):
        result = self.compile({
            "mode": "T2VA",
            "requested_duration": 4,
            "shots": [{"duration": 0.25}, {"duration": 9.75}],
        })
        durations = [shot["duration"] for shot in result["project"]["shots"]]
        self.assertAlmostEqual(sum(durations), 4.0)
        self.assertGreaterEqual(min(durations), MODULE.MIN_SHOT_DURATION)

    def test_invalid_enums_are_normalized_with_warnings(self):
        result = self.compile({
            "mode": "REF2VA",
            "shots": [{
                "duration": 5,
                "dialogue": "Hello",
                "dialogue_speaker": "S99",
                "dialogue_language": "Klingon",
                "dialogue_mode": "whisper",
                "transition": "teleport",
            }],
            "references": [{"type": "audio", "role": "first_frame"}],
        })
        shot = result["project"]["shots"][0]
        self.assertNotIn("transition", shot)
        self.assertIn("The on-screen speaker (S1) says: <d>[English] Hello</d>", shot["visual_action"])
        self.assertNotIn("dialogue_speaker", shot)
        self.assertEqual(result["project"]["references"][0]["role"], "none")
        self.assertTrue(any("normalized" in warning for warning in result["warnings"]))

    def test_project_version_mismatch_warns_and_uses_current_schema(self):
        result = self.compile({
            "version": 999,
            "mode": "T2VA",
            "shots": [{"duration": 5, "visual_action": "Test."}],
        })
        self.assertEqual(result["project"]["version"], MODULE.CURRENT_PROJECT_VERSION)
        self.assertTrue(any("Project version" in warning for warning in result["warnings"]))

    def test_removed_picture_roles_migrate_to_weak_reference(self):
        for removed_role in ("environment", "style", "storyboard"):
            with self.subTest(role=removed_role):
                result = self.compile({
                    "version": 8,
                    "mode": "REF2VA",
                    "shots": [{"duration": 5, "visual_action": "A subject walks."}],
                    "references": [{"type": "picture", "role": removed_role}],
                })
                self.assertEqual(result["project"]["references"][0]["role"], "subject_identity")
                self.assertEqual(result["project"]["references"][0]["strength"], "weak")
                self.assertIn("retention_output_marker: weak_reference", result["draft_video_prompt"])
                self.assertFalse(any("role=" in warning for warning in result["warnings"]))

    def test_frontend_picture_role_list_contains_only_supported_choices(self):
        source = (MODULE_PATH.parent.parent / "web" / "minimax_h3_prompter.js").read_text(encoding="utf-8")
        self.assertIn('data-ref-type="picture" type="button">+ Image</button>', source)
        self.assertIn('data-ref-type="audio" type="button">+ Audio</button>', source)
        self.assertIn('data-ref-type="video" type="button">+ Video</button>', source)
        self.assertNotIn('data-el="new-ref-type"', source)
        self.assertNotIn('>+ Reference</button>', source)
        self.assertIn(
            'picture: ["first_frame", "last_frame", "frame", "subject_identity"]', source,
        )
        self.assertNotIn('reference: "Reference (weak)"', source)
        self.assertIn('const SUBJECT_STRENGTHS = ["weak", "normal", "strong"]', source)
        self.assertIn('first_frame: "First frame"', source)
        self.assertIn('last_frame: "Last frame"', source)
        self.assertNotIn('first_frame: "First frame anchor"', source)
        self.assertNotIn('last_frame: "Last frame anchor"', source)
        self.assertIn('strength.className = "subject-strength"', source)
        self.assertIn('strengthRow.className = "mmh3p-subject-strength-row"', source)
        self.assertIn('strengthRow.append(strengthLabel, strength, alias)', source)
        self.assertIn('body.append(strengthRow)', source)
        self.assertIn('if (ref.type === "video") controls.append(role, alias, del)', source)
        self.assertIn('else if (ref.type === "audio") controls.append(role, alias, del)', source)
        self.assertIn('full_signal_copy: "Full audio reuse"', source)
        self.assertIn('voice_delivery: "Voice / delivery"', source)
        self.assertIn('controls.classList.add("video-metadata")', source)
        self.assertNotIn('duration.placeholder = "seconds"', source)
        self.assertIn('ref.source_duration = actualDuration > 0 ? actualDuration : 0', source)
        self.assertIn('ref.duration = actualDuration > 0 ? Math.min(15, actualDuration) : 0', source)
        self.assertIn('const VIDEO_UPLOAD_ENDPOINT = "/toyxyz/minimax_h3_prompter/upload-video"', source)
        self.assertIn('const VIDEO_VIEW_ENDPOINT = "/toyxyz/minimax_h3_prompter/video"', source)
        self.assertIn("const VIDEO_UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024", source)
        self.assertIn('ref.type === "video" && ref.video_filename && Number(ref.duration) > 0', source)
        self.assertIn('if (file) await this.uploadReferenceVideo(ref, file)', source)
        self.assertIn('if (file) await this.uploadReferenceAudio(ref, file)', source)
        self.assertIn('async uploadReferenceAudio(ref, file)', source)
        self.assertIn('audio_filename: String(ref?.audio_filename || "")', source)
        self.assertIn('body: file.slice(start, end)', source)
        self.assertIn('"Content-Type": "application/octet-stream"', source)
        self.assertIn('data-el="video-timeline"', source)
        self.assertIn('renderVideoTimeline()', source)
        self.assertIn('populateVideoFilmstrip(ref, filmstrip)', source)
        self.assertIn('className = "mmh3p-video-filmstrip"', source)
        self.assertIn('mmh3p-video-lane { height:52px; position:relative; width:100%; overflow:hidden', source)
        self.assertIn('help.textContent = "Video clips · drag a clip to move · drag either edge to trim"', source)
        self.assertIn('const MIN_VIDEO_CLIP_FRAMES = 10', source)
        self.assertIn('const SHOT_SNAP_SECONDS = 1 / VIDEO_OUTPUT_FPS', source)
        self.assertIn('data-action="preset-tab" data-preset-tab="camera"', source)
        self.assertIn('data-action="preset-tab" data-preset-tab="style"', source)
        self.assertIn('data-preset="camera_angle"', source)
        self.assertIn('data-preset="camera_motion"', source)
        self.assertIn('data-preset="camera_shot"', source)
        self.assertNotIn('data-preset="camera_lens"', source)
        self.assertIn('extreme_close_up: "Extreme close-up (ECU)"', source)
        self.assertIn('medium_close_up: "Medium close-up (MCU)"', source)
        self.assertIn('medium_wide_shot: "Medium wide shot (MWS)"', source)
        self.assertIn('cowboy_shot: "Cowboy shot (CS)"', source)
        self.assertIn('medium_full_shot: "Medium full shot (MFS)"', source)
        self.assertIn('extreme_wide_shot: "Extreme wide shot (EWS)"', source)
        self.assertIn('establishing_shot: "Establishing shot (ES)"', source)
        self.assertIn('insert_shot: "Insert shot"', source)
        self.assertIn('detail_shot: "Detail shot"', source)
        self.assertIn('two_shot: "Two shot"', source)
        self.assertIn('three_shot: "Three shot"', source)
        self.assertIn('group_shot: "Group shot"', source)
        self.assertIn('data-preset="camera_amplitude"', source)
        self.assertIn('data-preset="camera_speed"', source)
        self.assertIn('data-preset="style"', source)
        self.assertIn('data-el="raw-prompt" type="checkbox"><span>Raw Prompt</span>', source)
        self.assertIn('this.rawPromptEnabled = this.els["raw-prompt"].checked', source)
        self.assertIn('this.previewData?.llm_prompt', source)
        self.assertIn('this.lastRawModelPrompt = String(data.raw_model_prompt || "")', source)
        self.assertIn('anime_1990s: "1990s Japanese hand-drawn anime"', source)
        self.assertIn('food_commercial: "Food commercial"', source)
        self.assertIn('dark_retro_fantasy: "Dark retro fantasy film"', source)
        self.assertIn('shot.presets[select.dataset.preset] = select.value', source)
        self.assertIn('this.selectedShot()?.presets?.[select.dataset.preset]', source)
        self.assertIn('id: uid("shot"), duration: firstHalf, visual_action: "",\n      presets: DEFAULT_SHOT_PRESETS(),', source)
        self.assertIn('zoom_in: "Zoom in"', source)
        self.assertIn('truck_left: "Truck left"', source)
        self.assertIn('pedestal_up: "Pedestal up"', source)
        self.assertIn('dolly_left: "Dolly left"', source)
        self.assertIn('dolly_right: "Dolly right"', source)
        self.assertIn('dolly_zoom_in: "Dolly zoom in"', source)
        self.assertIn('dolly_zoom_out: "Dolly zoom out"', source)
        self.assertIn('crane_up: "Crane up"', source)
        self.assertIn('crane_down: "Crane down"', source)
        self.assertIn('orbit_left: "Orbit left"', source)
        self.assertIn('orbit_right: "Orbit right"', source)
        self.assertIn('follow: "Follow shot"', source)
        self.assertIn('handheld: "Handheld"', source)
        self.assertIn('shake_strongly: "Shake strongly"', source)
        self.assertIn('roll_counterclockwise: "Roll counterclockwise"', source)
        self.assertIn('top_down: "Top-down"', source)
        self.assertIn('ground_level: "Ground-level"', source)
        self.assertIn('three_quarter: "Three-quarter"', source)
        self.assertIn('const shotFrameStep = this.totalDuration() / Math.max(1, this.timelineFrameCount())', source)
        self.assertIn('Math.round(nextLeft / shotFrameStep) * shotFrameStep', source)
        self.assertIn('`${visibleDuration.toFixed(2)}s · ${visibleFrames} frames`', source)
        self.assertIn('const labelPosition = (visibleCenter - ref.timeline_start)', source)
        self.assertIn('object-fit:contain; object-position:center', source)
        self.assertIn('imageHelp.textContent = "Image anchors · first/last frames are fixed · drag Frame images to an exact output frame"', source)
        self.assertIn('marker.className = `mmh3p-image-anchor ${isFirst ? "first" : isLast ? "last" : "frame"}`', source)
        self.assertIn('Image ${number} · Last · Frame ${frameCount - 1}', source)
        self.assertIn('ref.frame_index = Math.round(ratio * Math.max(0, this.timelineFrameCount() - 1))', source)
        self.assertIn('const frameWidth = Math.max(2, laneWidth / Math.max(1, frameCount))', source)
        self.assertIn('label.classList.toggle("before", ref.frame_index >= frameCount / 2)', source)
        self.assertIn('previewImage.className = "mmh3p-image-anchor-preview"', source)
        self.assertIn('timelineFrameCount() { return alignedFrameCount(this.totalDuration()); }', source)
        self.assertIn('this.timelineFrameCount()}f`', source)
        self.assertIn('shotTimelineRange(index)', source)
        self.assertIn('F${range.startFrame}–${range.endFrame}', source)
        self.assertIn('else controls.append(role, del)', source)
        self.assertNotIn('controls.append(role, strength, alias, del)', source)
        self.assertIn(
            'video: ["none", "video_editing", "video_continuation", "motion", "camera", "cuts_rhythm"]',
            source,
        )
        self.assertIn('"none", "full_signal_copy", "partial_signal_copy", "voice_delivery"', source)
        self.assertIn('const REFERENCE_TYPE_LABELS = { picture: "Image", video: "Video", audio: "Audio" }', source)
        self.assertIn('const label = `<${REFERENCE_TYPE_LABELS[ref.type]} ${counts[ref.type]}>`', source)
        self.assertNotIn('<option value="picture">Picture</option>', source)
        self.assertIn('const role = document.createElement("select")', source)
        self.assertIn('none: "None"', source)
        self.assertIn('video_editing: "Video editing"', source)
        self.assertIn('video_continuation: "Video continuation"', source)
        self.assertIn('motion: "Motion / action timing"', source)
        self.assertIn('camera: "Camera movement"', source)
        self.assertIn('cuts_rhythm: "Cuts / rhythm / temporal structure"', source)
        self.assertNotIn('voice_timbre: "Voice timbre"', source)
        self.assertNotIn('full_signal_copy: "Full signal copy"', source)
        self.assertIn('alias.placeholder = "alias"', source)
        self.assertNotIn('alias.placeholder = "@alias"', source)
        self.assertIn('event.key === "ArrowDown" || event.key === "ArrowUp"', source)
        self.assertIn('event.key === "Enter" || event.key === "Tab"', source)
        self.assertIn('button.classList.toggle("active", active)', source)
        self.assertIn('this.insertMention(entry.alias, mention)', source)
        self.assertIn('commit(refresh = true, preserveRawModelPrompt = false) {', source)
        self.assertNotIn('invalidateEnhancement', source)
        self.assertNotIn('this.project.enhanced_prompt = ""', source)
        self.assertNotIn('data-action="copy-raw"', source)
        self.assertNotIn("copyRawPrompt()", source)
        self.assertNotIn('environment: "Environment"', source)
        self.assertNotIn('style: "Visual style"', source)
        self.assertNotIn('storyboard: "Storyboard"', source)

    def test_legacy_audio_roles_migrate_to_audio_presets(self):
        expected = {
            "reference": "none",
            "voice_timbre": "voice_delivery",
            "dialogue": "dialogue_lyrics",
            "music_style": "music_rhythm",
            "sound_effect": "sound_ambience",
            "partial_signal_copy": "partial_signal_copy",
            "full_signal_copy": "full_signal_copy",
        }
        for legacy_role, normalized_role in expected.items():
            with self.subTest(role=legacy_role):
                result = self.compile({
                    "version": 10,
                    "mode": "REF2VA",
                    "shots": [{"duration": 5, "visual_action": "A person walks."}],
                    "references": [{
                        "type": "audio", "role": legacy_role,
                        "description": "Use the supplied audio according to this description.",
                    }],
                })
                self.assertEqual(result["project"]["references"][0]["role"], normalized_role)
                self.assertIn(f"role: {normalized_role}", result["draft_video_prompt"])

    def test_video_reference_presets_are_normalized_and_compiled(self):
        expected = {
            "reference": ("none", "reference generation"),
            "motion": ("motion", "subject motion, action sequence, movement timing"),
            "camera": ("camera", "camera movement, viewpoint, framing progression"),
            "pacing": ("cuts_rhythm", "cut placement, pacing, rhythm, and temporal structure"),
            "continuation": ("video_continuation", "video continuation"),
            "video_editing": ("video_editing", "video editing"),
        }
        for supplied_role, (normalized_role, expected_text) in expected.items():
            with self.subTest(role=supplied_role):
                result = self.compile({
                    "version": MODULE.CURRENT_PROJECT_VERSION,
                    "mode": "REF2VA",
                    "shots": [{"duration": 5, "visual_action": "A person walks."}],
                    "references": [{
                        "type": "video", "role": supplied_role, "duration": 5,
                        "description": "Use its temporal structure as described here.",
                    }],
                })
                self.assertEqual(result["project"]["references"][0]["role"], normalized_role)
                self.assertIn(f"role: {normalized_role}", result["draft_video_prompt"])
                self.assertIn(expected_text, result["draft_video_prompt"])

    def test_picture_analysis_metadata_never_enters_raw_prompt(self):
        stale_analysis = "A very long automatic image analysis that must remain private."
        result = self.compile({
            "version": 7,
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "@hero walks through a hotel corridor."}],
            "references": [{
                "id": "hero-ref", "type": "picture", "role": "reference", "alias": "hero",
                "description": stale_analysis, "image_filename": "hero.png",
            }],
        })
        self.assertEqual(result["project"]["references"][0]["description"], "")
        self.assertNotIn(stale_analysis, result["draft_video_prompt"])
        self.assertIn("<Subject 1>\nsource: <Picture 1>", result["draft_video_prompt"])
        self.assertFalse(any("Project version 7" in warning for warning in result["warnings"]))

    def test_raw_prompt_uses_qwen_data_contract_for_every_mode(self):
        cases = {
            "T2VA": [],
            "I2VA": [{"type": "picture", "role": "first_frame", "image_filename": "first.png"}],
            "FL2VA": [
                {"type": "picture", "role": "first_frame", "image_filename": "first.png"},
                {"type": "picture", "role": "last_frame", "image_filename": "last.png"},
            ],
            "L2VA": [{"type": "picture", "role": "last_frame", "image_filename": "last.png"}],
            "REF2VA": [{
                "type": "picture", "role": "subject_identity", "alias": "hero",
                "image_filename": "hero.png",
            }],
        }
        for mode, references in cases.items():
            with self.subTest(mode=mode):
                action = "@hero walks." if mode == "REF2VA" else "The subject walks."
                result = self.compile({
                    "mode": mode,
                    "shots": [{"duration": 5, "visual_action": action}],
                    "references": references,
                })
                prompt = result["draft_video_prompt"]
                self.assertTrue(prompt.startswith("INPUT DATA ONLY"))
                self.assertIn(f"mode: {mode}", prompt)
                self.assertIn("requested_duration_seconds: 5.00", prompt)
                self.assertIn("effective_duration_seconds: 5.17", prompt)
                self.assertIn("SHOT_PLAN:\n[Shot 1]", prompt)
                self.assertIn("STYLE_POLICY:", prompt)
                self.assertIn("when_unspecified: omit any target-wide style", prompt)
                self.assertIn("AUDIO_POLICY:", prompt)
                self.assertIn("source: infer audio intent only", prompt)
                self.assertNotIn("AUDIO_INPUTS:", prompt)
                self.assertNotIn("integrated_multimodal_description:", prompt)
                self.assertNotIn("subject_definitions:", prompt)
        ref_prompt = self.compile({
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "@hero walks."}],
            "references": [{
                "type": "picture", "role": "subject_identity", "alias": "hero",
                "image_filename": "hero.png",
            }],
        })["draft_video_prompt"]
        self.assertIn("<Subject 1>\nsource: <Picture 1>", ref_prompt)
        self.assertIn("retention_output_marker: fully_preserved", ref_prompt)
        self.assertIn("visual_action: <Subject 1> walks.", ref_prompt)

    def test_keyframe_raw_plans_preserve_reference_style_by_mode(self):
        cases = {
            "I2VA": ([{"type": "picture", "role": "first_frame"}], "exact opening-frame anchor"),
            "FL2VA": ([
                {"type": "picture", "role": "first_frame"},
                {"type": "picture", "role": "last_frame"},
            ], "endpoint evidence"),
            "L2VA": ([{"type": "picture", "role": "last_frame"}], "exact final-frame anchor"),
        }
        for mode, (references, phrase) in cases.items():
            with self.subTest(mode=mode):
                prompt = self.compile({
                    "mode": mode,
                    "shots": [{"duration": 5, "visual_action": "The subject moves."}],
                    "references": references,
                })["draft_video_prompt"]
                self.assertIn(phrase, prompt)

    def test_frontend_does_not_persist_reference_analysis_results(self):
        source = (MODULE_PATH.parent.parent / "web" / "minimax_h3_prompter.js").read_text(encoding="utf-8")
        self.assertNotIn("ref.description = item.analysis", source)
        self.assertIn('description: type === "picture" ? ""', source)

    def test_system_prompt_forbids_unrequested_video_style_in_every_mode(self):
        for mode in ("T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"):
            with self.subTest(mode=mode):
                prompt = MODULE.MODE_LLM_SYSTEM_PROMPTS[mode]
                self.assertIn("Use a target-wide style only when the user explicitly requests it", prompt)
                self.assertIn("When no target style or keyframe style applies", prompt)
        for mode in ("I2VA", "FL2VA", "L2VA"):
            with self.subTest(keyframe_style_mode=mode):
                self.assertIn("observable visual medium", MODULE.MODE_LLM_SYSTEM_PROMPTS[mode])
        self.assertIn("Weak and Normal do not transfer source style", MODULE.MODE_LLM_SYSTEM_PROMPTS["REF2VA"])
        self.assertIn(
            "begin detailed_description directly with [Shot 1]",
            MODULE.MODE_LLM_SYSTEM_PROMPTS["REF2VA"],
        )
        self.assertIn(
            "Picture 1 is the complete literal frame at 0.00 seconds",
            MODULE.MODE_LLM_SYSTEM_PROMPTS["I2VA"],
        )
        self.assertIn(
            "applies only after this exact opening instant when it conflicts with Picture 1",
            MODULE.MODE_LLM_SYSTEM_PROMPTS["I2VA"],
        )

    def test_system_prompts_are_loaded_from_json_config(self):
        config_path = MODULE_PATH.parent / "minimax_h3_system_prompts.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config, MODULE.SYSTEM_PROMPT_CONFIG)
        self.assertEqual(
            set(config["modes"]),
            {"T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"},
        )
        for mode, mode_rules in config["modes"].items():
            expected = config["common"]
            expected += config["action_semantics"]
            expected += config["common_addendum"]
            if mode != "REF2VA":
                expected += config["base"]
            expected += mode_rules
            expected += config["mode_addenda"].get(mode, "")
            self.assertEqual(MODULE.MODE_LLM_SYSTEM_PROMPTS[mode], expected)
            enhanced_expected = config["common_enhanced"]
            enhanced_expected += config["action_semantics"]
            enhanced_expected += config["common_addendum"]
            if mode != "REF2VA":
                enhanced_expected += config["base"]
            enhanced_expected += mode_rules
            enhanced_expected += config["mode_addenda"].get(mode, "")
            enhanced_expected += config["enhance_addendum"]
            self.assertEqual(MODULE.ENHANCED_MODE_LLM_SYSTEM_PROMPTS[mode], enhanced_expected)

    def test_figurine_motion_rules_are_opt_in_and_scoped_per_shot(self):
        project = {
            "shots": [
                {"presets": {"style": "none"}},
                {"presets": {"style": "figurine_animation"}},
            ],
        }
        standard = MODULE._figurine_animation_system_module(project, "FL2VA", "none")
        enhanced = MODULE._figurine_animation_system_module(project, "FL2VA", "normal")
        self.assertIn("explicitly selected for [Shot 2]", standard)
        self.assertNotIn("[Shot 1]", standard)
        self.assertIn("exact endpoint", standard)
        self.assertIn("character that comes fully alive, not as a rigid object", enhanced)
        self.assertIn("Do not expose or invent mechanical joints", enhanced)
        self.assertIn("natural expressive performance has priority over literal toy stiffness", enhanced)
        self.assertIn("alternating foot placement", enhanced)
        self.assertIn("physical release and first unsupported step", enhanced)
        self.assertIn("appearance interpolation", enhanced)
        self.assertEqual(
            MODULE._figurine_animation_system_module(
                {"shots": [{"presets": {"style": "none"}}]}, "FL2VA", "strong",
            ),
            "",
        )
        for prompt in MODULE.MODE_LLM_SYSTEM_PROMPTS.values():
            self.assertNotIn("FIGURINE ANIMATION PRESET", prompt)
        compiled = self.compile({
            "mode": "FL2VA",
            "enhance_level": "normal",
            "shots": [
                {"duration": 2.5, "visual_action": "A normal subject moves."},
                {"duration": 2.5, "visual_action": "The figurine moves its arms.",
                 "presets": {"style": "figurine_animation"}},
            ],
            "references": [
                {"type": "image", "role": "first_frame", "source": "first.png"},
                {"type": "image", "role": "last_frame", "source": "last.png"},
            ],
        })
        self.assertIn("explicitly selected for [Shot 2]", compiled["llm_prompt"])
        self.assertEqual(compiled["llm_prompt"].count("FIGURINE ANIMATION PRESET"), 1)

    def test_explicit_target_style_and_repeated_action_semantics_are_locked(self):
        for prompt in (
            MODULE.MODE_LLM_SYSTEM_PROMPTS["I2VA"],
            MODULE.ENHANCED_MODE_LLM_SYSTEM_PROMPTS["I2VA"],
        ):
            self.assertIn("Obey TARGET_STYLE_LOCK", prompt)
            self.assertIn("weaken a repeated action into one contact or static hold", prompt)
            self.assertIn("insert unsupported clothing over the named contact target", prompt)
            self.assertIn("Never replace explicitly requested subject motion with camera-only motion", prompt)

    def test_system_prompt_lengths_stay_qwen38_friendly(self):
        self.assertLessEqual(len(MODULE.COMMON_LLM_SYSTEM_RULES), 6000)
        for mode, prompt in MODULE.MODE_LLM_SYSTEM_PROMPTS.items():
            with self.subTest(mode=mode):
                self.assertLessEqual(len(prompt), 8800)
                lock = MODULE._single_pass_output_lock(
                    mode, 5.17, 2, [1, 2],
                    {"label_plan": {"<Subject 1>": {}}} if mode == "REF2VA" else None,
                    [],
                )
                combined = len(MODULE._mode_prompt_preamble(mode)) + len(prompt) + len(lock) + 4
                self.assertLessEqual(combined, 10200)
        for mode, prompt in MODULE.ENHANCED_MODE_LLM_SYSTEM_PROMPTS.items():
            with self.subTest(enhanced_mode=mode):
                self.assertLessEqual(len(prompt), 12500)

    def test_ref2va_prompt_rejects_literal_retention_placeholder(self):
        prompt = MODULE.MODE_LLM_SYSTEM_PROMPTS["REF2VA"]
        self.assertNotIn("fixed_marker", prompt)
        self.assertIn("with its applicable shots, then exactly its locked output marker", prompt)
        self.assertIn("Never print the UI strength words weak, normal, or strong", prompt)
        self.assertIn("never print an equals sign", prompt)

    def test_base_modes_forbid_ref2va_subject_labels(self):
        for mode in ("T2VA", "I2VA", "FL2VA", "L2VA"):
            with self.subTest(mode=mode):
                prompt = MODULE.MODE_LLM_SYSTEM_PROMPTS[mode]
                self.assertIn("Base modes never define or use <Subject N>", prompt)
                self.assertIn("the girl (S1)", prompt)

    def test_ref2va_system_prompt_has_compact_compliance_contract(self):
        prompt = MODULE.MODE_LLM_SYSTEM_PROMPTS["REF2VA"]
        self.assertIn("Define each image-derived Subject in one line", prompt)
        self.assertIn("derived from <Picture N>", prompt)
        self.assertIn("Begin summary with one bracketed list", prompt)
        self.assertIn("free of shot labels", prompt)
        self.assertIn("do not announce the same speech twice", prompt)

    def test_node_exposes_generated_prompt_length_and_ordered_picture_outputs(self):
        self.assertEqual(
            MODULE.MinimaxH3Prompter.RETURN_NAMES,
            ("generated_prompt", "length")
            + tuple(name for index in range(1, 10) for name in (f"image_{index}", f"frame_{index}"))
            + tuple(f"video_{index}" for index in range(1, 4))
            + tuple(f"audio_{index}" for index in range(1, 4)),
        )
        self.assertEqual(MODULE.MinimaxH3Prompter.RETURN_TYPES[:2], ("STRING", "INT"))
        self.assertEqual(len(MODULE.MinimaxH3Prompter.RETURN_TYPES), 26)

    def test_node_first_output_is_only_the_saved_enhanced_prompt(self):
        node = MODULE.MinimaxH3Prompter()
        project = {
            "mode": "T2VA",
            "shots": [{"duration": 5, "visual_action": "A woman opens a door."}],
            "enhanced_prompt": "enhanced result only",
        }
        outputs = node.compile(json.dumps(project))
        self.assertEqual(outputs[0], "enhanced result only")
        self.assertIsInstance(outputs[1], int)
        self.assertEqual(len(outputs), 26)
        self.assertEqual(tuple(outputs[2].shape), (1, 64, 64, 3))
        node = MODULE.MinimaxH3Prompter()
        output = node.compile(json.dumps({
            "mode": "T2VA",
            "shots": [{"duration": 5, "visual_action": "A static establishing shot"}],
        }))
        self.assertEqual(len(output), 26)
        self.assertEqual(output[0], "")
        self.assertIsInstance(output[1], int)

    def test_auto_run_is_disabled_by_default(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{"duration": 5, "visual_action": "A person walks."}],
        })
        self.assertFalse(result["project"]["auto_run"])

    def test_auto_run_executes_enhancement_inside_node_compile(self):
        project = {
            "mode": "T2VA",
            "auto_run": True,
            "shots": [{"duration": 5, "visual_action": "A person walks."}],
            "enhanced_prompt": "stale saved prompt",
        }
        node = MODULE.MinimaxH3Prompter()
        with (
            mock.patch.object(
                MODULE, "enhance_project",
                return_value={"enhanced_prompt": "queue enhanced prompt"},
            ) as enhance,
            mock.patch.object(MODULE, "_reference_media_outputs", return_value=()),
        ):
            outputs = node.compile(json.dumps(project))
        self.assertEqual(outputs["result"][0], "queue enhanced prompt")
        self.assertEqual(outputs["ui"]["auto_run_prompt"], ["queue enhanced prompt"])
        enhance.assert_called_once()

    def test_auto_run_ui_is_queue_integrated_not_browser_triggered(self):
        source = (MODULE_PATH.parent.parent / "web" / "minimax_h3_prompter.js").read_text(encoding="utf-8")
        self.assertIn('data-el="auto-run"', source)
        self.assertIn('this.project.auto_run = this.els["auto-run"].checked', source)
        self.assertIn('showAutoRunPrompt(message?.auto_run_prompt)', source)
        self.assertNotIn(
            'this.els["auto-run"].addEventListener("change", () => this.enhancePrompt()',
            source,
        )

    def test_enhance_list_is_below_generated_prompt_with_raw_prompt_beside_it(self):
        source = (MODULE_PATH.parent.parent / "web" / "minimax_h3_prompter.js").read_text(encoding="utf-8")
        preview_index = source.index('data-el="preview"')
        modebar_index = source.index('<div class="mmh3p-row mmh3p-preview-modebar">')
        level_index = source.index('data-el="enhance"')
        raw_index = source.index('data-el="raw-prompt"')
        generate_index = source.index('data-action="enhance"')
        self.assertLess(generate_index, level_index)
        self.assertLess(preview_index, modebar_index)
        self.assertLess(level_index, raw_index)
        self.assertIn('const ENHANCE_LEVELS = { none: "None", normal: "Normal", strong: "Strong" }', source)
        self.assertIn('this.project.enhance_level = this.els.enhance.value', source)
        self.assertIn('Strong: creates a much longer rewriter-style scene with compatible new staging', source)
        self.assertIn('Enhance levels are available only with Qwen3.8', source)

    def test_model_download_progress_updates_catalog_and_refreshes_after_generation(self):
        source = Path("web/minimax_h3_prompter.js").read_text(encoding="utf-8")
        self.assertIn("updateModelDownloadDisplay(bundleId, downloaded, total, complete = false)", source)
        self.assertIn("this.updateModelDownloadDisplay(bundleId, downloaded, total, percent >= 100)", source)
        self.assertIn("await this.loadModels(selectedBundle, false)", source)
        self.assertIn("modelInfo.text_size = 0", source)
        self.assertIn("modelInfo.vision_size = 0", source)

    def test_picture_outputs_follow_picture_reference_order(self):
        first = object()
        second = object()
        loaded = []

        def fake_load(reference):
            loaded.append(reference["image_filename"])
            return first if len(loaded) == 1 else second

        project = {
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "Two subjects walk."}],
            "references": [
                {"type": "picture", "role": "subject_identity", "image_filename": "first.png"},
                {"type": "audio", "role": "sound_effect", "description": "Footsteps"},
                {"type": "picture", "role": "subject_identity", "image_filename": "second.png"},
            ],
        }
        with mock.patch.object(MODULE, "_load_reference_image_tensor", side_effect=fake_load):
            outputs = MODULE.MinimaxH3Prompter().compile(json.dumps(project))
        self.assertEqual(loaded, ["first.png", "second.png"])
        self.assertIs(outputs[2], first)
        self.assertIs(outputs[3], second)

    def test_frontend_shows_one_image_slot_per_picture_reference(self):
        source = (MODULE_PATH.parent.parent / "web" / "minimax_h3_prompter.js").read_text(encoding="utf-8")
        self.assertIn("syncReferenceOutputs()", source)
        self.assertIn('const frameOutputCount = pictures.filter(ref => ref.role === "frame").length', source)
        self.assertIn('this.project.references.filter(ref => ref.type === "video").length', source)
        self.assertIn('this.project.references.filter(ref => ref.type === "audio").length', source)
        self.assertIn('this.node.outputs[outputIndex].name = `frame_${index + 1}`', source)
        self.assertIn("this.node.removeOutput(this.node.outputs.length - 1)", source)

    def test_frame_picture_emits_image_then_frame_index(self):
        project = {
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "Reach the reference frame."}],
            "references": [{
                "type": "picture", "role": "frame", "frame_index": 62,
                "image_filename": "frame.png",
            }],
        }
        marker = object()
        with mock.patch.object(MODULE, "_load_reference_image_tensor", return_value=marker):
            outputs = MODULE.MinimaxH3Prompter().compile(json.dumps(project))
        self.assertIs(outputs[2], marker)
        self.assertEqual(outputs[3], 62)
        compiled = MODULE.compile_project(json.dumps(project))["draft_video_prompt"]
        self.assertIn("anchor_frame_index: 62", compiled)
        self.assertIn("anchor_time_seconds: 2.583", compiled)
        self.assertIn("this anchor never creates a cut or transition", compiled)
        self.assertIn("Picture anchor times never create cuts or transitions", compiled)

    def test_llm_prompt_uses_mode_specific_english_system_prompt(self):
        expected_phrases = {
            "T2VA": "MODE: T2VA",
            "I2VA": "MODE: I2VA",
            "FL2VA": "MODE: FL2VA",
            "L2VA": "MODE: L2VA",
            "REF2VA": "MODE: REF2VA",
        }
        references = {
            "T2VA": [],
            "I2VA": [{"type": "picture", "role": "first_frame"}],
            "FL2VA": [
                {"type": "picture", "role": "first_frame"},
                {"type": "picture", "role": "last_frame"},
            ],
            "L2VA": [{"type": "picture", "role": "last_frame"}],
            "REF2VA": [{"type": "picture", "role": "subject_identity", "alias": "hero"}],
        }
        for mode, phrase in expected_phrases.items():
            with self.subTest(mode=mode):
                result = self.compile({
                    "mode": mode,
                    "shots": [{"duration": 5, "visual_action": "A person walks forward."}],
                    "references": references[mode],
                })
                self.assertIn(phrase, result["llm_prompt"])
                self.assertTrue(result["llm_prompt"].endswith(result["video_prompt"]))
                self.assertIn("PRIORITY\n1. Explicit user actions", result["llm_prompt"])
                self.assertIn("Minimal detail needed to make the request renderable", result["llm_prompt"])
                self.assertIn("When rules conflict, the higher priority wins", result["llm_prompt"])

    def test_llm_system_prompt_requires_sentence_level_enrichment(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{"duration": 5, "visual_action": "A woman opens a door."}],
        })
        llm_prompt = result["llm_prompt"]
        self.assertIn("concise chronological action", llm_prompt)
        self.assertIn("show only the intermediate motion needed to make it physically legible", llm_prompt)

    def test_fl2va_system_prompt_prioritizes_endpoints_over_appearance_reconstruction(self):
        result = self.compile({
            "mode": "FL2VA",
            "shots": [{"duration": 5, "visual_action": "The character transforms continuously."}],
            "references": [
                {"type": "picture", "role": "first_frame"},
                {"type": "picture", "role": "last_frame"},
            ],
        })
        llm_prompt = result["llm_prompt"]
        self.assertIn("Picture 1 is the complete opening frame", llm_prompt)
        self.assertIn("Treat both images as visual anchors rather than appearance inventories", llm_prompt)
        self.assertIn("never invent a full rotation, orbit, dramatic performance", llm_prompt)
        self.assertIn("usually 80-150 English words", llm_prompt)
        self.assertIn("Picture 2 is the complete frame reached only at the effective end", llm_prompt)
        self.assertIn("Do not keep a Picture 1-only entity visibly present", llm_prompt)
        self.assertIn("Use no <Subject N> labels", llm_prompt)
        self.assertIn("Picture 2 is reached only at the effective end time", llm_prompt)
        self.assertIn("Never reveal the completed Picture 2 at the start of the final shot", llm_prompt)
        self.assertIn("A hit, fall, entrance, exit, or cut never authorizes trait transfer", llm_prompt)
        self.assertIn("put a visible identity before every speaker ID", llm_prompt)
        self.assertIn("Begin each later shot naturally as [Shot N] At MM:SS.mmm,", llm_prompt)
        self.assertIn("Never morph one character into another", llm_prompt)
        self.assertIn("Bind Picture 2 traits only to the entity", llm_prompt)
        self.assertIn("Preserve every explicit SHOT_PLAN action in order; omit none", llm_prompt)
        self.assertIn("overall_soundscape must not repeat or summarize speech", llm_prompt)

    def test_fl2va_raw_plan_locks_picture_two_to_the_final_instant(self):
        result = self.compile({
            "mode": "FL2VA",
            "shots": [
                {"duration": 2.5, "visual_action": "The first character speaks."},
                {"duration": 2.5, "visual_action": "The second character enters."},
            ],
            "references": [
                {"type": "picture", "role": "first_frame"},
                {"type": "picture", "role": "last_frame"},
            ],
        })
        raw = result["draft_video_prompt"]
        self.assertIn("ENDPOINT_CONTRACT:", raw)
        self.assertIn("Picture 2 is not the opening state of the final shot", raw)
        self.assertIn("Do not merge identities or transfer clothing", raw)
        self.assertIn("Punch, hit, fall, enter, exit, and cut are not transformation requests", raw)
        self.assertIn("entity_binding: the entity visible in Picture 1 owns only Picture 1 traits", raw)
        self.assertIn("If the final shot introduces an entity matching Picture 2", raw)
        self.assertIn("opening_state: continue the incomplete transition", raw)
        self.assertIn("entity_continuity: an entering entity that matches Picture 2 is the same final-frame entity", raw)
        self.assertIn("required_end_state: exact whole-frame match to Picture 2 at 5.17 seconds", raw)
        self.assertEqual(raw.count("action_contract: preserve every explicit action above in the same order; omit none"), 2)
        self.assertIn("required_output_header: [Shot 2] At 00:02.583,", raw)

    def test_saved_enhancement_becomes_video_prompt_without_rewriting_llm_input(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{"duration": 5, "visual_action": "A woman opens a door."}],
            "enhanced_prompt": "integrated_multimodal_description: enhanced result",
        })
        self.assertEqual(result["video_prompt"], "integrated_multimodal_description: enhanced result")
        self.assertNotEqual(result["draft_video_prompt"], result["video_prompt"])
        self.assertTrue(result["llm_prompt"].endswith(result["draft_video_prompt"]))

    def test_enhance_model_list_exposes_qwen38_and_omni(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ("writer.gguf", "adapter-LoRA.gguf", "model-mmproj.gguf", "draft.gguf"):
                Path(root, name).write_bytes(b"x")
            models = MODULE.list_enhance_models([root])
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["id"], MODULE.DEFAULT_ENHANCE_MODEL_ID)
        self.assertEqual(models[1]["id"], MODULE.OMNI_MODEL_ID)

    def test_model_bundle_list_exposes_qwen_and_omni(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, MODULE.DEFAULT_ENHANCE_MODEL_FILE).write_bytes(b"qwen")
            Path(root, MODULE.QWEN_IMAGE_MMPROJ_FILE).write_bytes(b"qwen-projector")
            models = MODULE.list_image_models([root])
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["id"], MODULE.DEFAULT_IMAGE_MODEL_ID)
        self.assertTrue(models[0]["label"].startswith("JonathanColetti/Qwen3.8-27B-Uncensored-GGUF · Q4_K_M + Vision F16"))
        self.assertTrue(models[0]["label"].endswith("VRAM ≈ 20–22 GB"))
        self.assertTrue(models[0]["installed"])
        self.assertEqual(models[1]["id"], MODULE.OMNI_MODEL_ID)
        self.assertEqual(models[1]["supported_modes"], ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"])
        self.assertNotIn("image/video/audio", models[1]["label"])
        self.assertEqual(models[1]["runtime"], "llama.cpp-mtmd-gguf-lora")

    def test_normalize_project_preserves_supported_image_model_selection(self):
        project, _warnings = MODULE.normalize_project({"image_model": MODULE.QWEN_IMAGE_MODEL_ID})
        self.assertEqual(project["image_model"], MODULE.QWEN_IMAGE_MODEL_ID)

    def test_normalize_project_migrates_removed_lightx2v_selection_to_qwen(self):
        removed = next(iter(MODULE.REMOVED_LIGHTX2V_MODEL_IDS))
        project, _warnings = MODULE.normalize_project({
            "enhance_model": removed,
            "image_model": removed,
        })
        self.assertEqual(project["enhance_model"], MODULE.DEFAULT_ENHANCE_MODEL_ID)
        self.assertEqual(project["image_model"], MODULE.QWEN_IMAGE_MODEL_ID)

    def test_normalize_project_preserves_omni_selection(self):
        project, _warnings = MODULE.normalize_project({
            "enhance_model": MODULE.OMNI_MODEL_ID,
            "image_model": MODULE.OMNI_MODEL_ID,
        })
        self.assertEqual(project["enhance_model"], MODULE.OMNI_MODEL_ID)
        self.assertEqual(project["image_model"], MODULE.OMNI_MODEL_ID)

    def test_omni_system_prompts_are_bundled_for_base_and_ref_modes(self):
        base = MODULE._omni_system_prompt("I2VA")
        ref = MODULE._omni_system_prompt("REF2VA")
        self.assertIn("Exactly three core fields", base)
        self.assertIn("Return exactly these six sections", ref)
        self.assertIn("retention_analysis:", ref)
        # These hashes lock the UTF-8 text published in the official Omni
        # repository's system_prompt.py. In particular, this catches Windows
        # console encoding damage to em/en dashes and numeric ranges.
        self.assertEqual(
            hashlib.sha256(base.encode("utf-8")).hexdigest(),
            "6386c2cbcdbd865efcf4a2da9e18cac47d3654d544f1a95bd229f5748662eeb4",
        )
        self.assertEqual(
            hashlib.sha256(ref.encode("utf-8")).hexdigest(),
            "5970ae7039c4eab04990dc9ec893e56cbf67d22fe5c55946e41f3ba79ed9034d",
        )

    def test_omni_system_prompt_routes_only_ref2va_to_ref2av_contract(self):
        for mode in ("T2VA", "I2VA", "FL2VA", "L2VA"):
            with self.subTest(mode=mode):
                prompt = MODULE._omni_system_prompt(mode)
                self.assertIn("T2AV, I2AV, FL2AV, and L2AV modes", prompt)
                self.assertNotIn("subject_definitions:", prompt)
        self.assertIn("subject_definitions:", MODULE._omni_system_prompt("REF2VA"))

    def test_omni_raw_prompt_keeps_shots_aliases_and_per_shot_presets_concise(self):
        project, _warnings = MODULE.normalize_project({
            "mode": "REF2VA",
            "shots": [{
                "duration": 5, "visual_action": "@hero walks forward.",
                "presets": {"camera_motion": "orbit_left", "style": "cinematic_live_action"},
            }],
            "references": [{
                "type": "picture", "role": "subject_identity", "alias": "hero",
            }],
        })
        raw = MODULE._build_omni_raw_prompt(project, 5.167)
        self.assertIn("<Subject 1> walks forward", raw)
        self.assertIn("[Shot 1] Required presets:", raw)
        self.assertIn("orbit left", raw)
        self.assertNotIn("MODE_DATA:", raw)

    def test_qwen_vision_bundle_reuses_writer_model(self):
        with tempfile.TemporaryDirectory() as root:
            model = Path(root, MODULE.DEFAULT_ENHANCE_MODEL_FILE)
            projector = Path(root, MODULE.QWEN_IMAGE_MMPROJ_FILE)
            model.write_bytes(b"qwen")
            projector.write_bytes(b"vision")
            with mock.patch.object(MODULE, "_llm_roots", return_value=[root]):
                resolved = MODULE._resolve_image_model(MODULE.QWEN_IMAGE_MODEL_ID)
        self.assertEqual(resolved, (str(model.resolve()), str(projector.resolve())))

    def test_qwen_download_progress_counts_only_missing_projector(self):
        with tempfile.TemporaryDirectory() as root:
            model = Path(root, MODULE.DEFAULT_ENHANCE_MODEL_FILE)
            model.write_bytes(b"qwen")
            downloaded = Path(root, MODULE.QWEN_IMAGE_MMPROJ_FILE)
            calls = []

            def fake_download(repo, filename, local_dir, component_size, completed_size, bundle_size, progress):
                calls.append((filename, completed_size, bundle_size))
                downloaded.write_bytes(b"vision")
                return str(downloaded)

            with (
                mock.patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace()}),
                mock.patch.object(MODULE, "_llm_roots", return_value=[root]),
                mock.patch.object(MODULE, "_download_image_component", side_effect=fake_download),
            ):
                MODULE._resolve_image_model(MODULE.QWEN_IMAGE_MODEL_ID)
        self.assertEqual(calls, [(
            MODULE.QWEN_IMAGE_MMPROJ_FILE, 0, MODULE.QWEN_IMAGE_MMPROJ_SIZE,
        )])

    def test_llm_output_cleanup_removes_reasoning_and_fence(self):
        cleaned = MODULE._clean_llm_output("<think>hidden</think>\n```text\nfinal prompt\n```")
        self.assertEqual(cleaned, "final prompt")

    def test_llm_output_cleanup_rejects_unfinished_reasoning_trace(self):
        self.assertEqual(MODULE._clean_llm_output("<think>still reasoning"), "")

    def test_llm_output_cleanup_removes_qwen_chat_end_token(self):
        prompt = (
            "integrated_multimodal_description: [Shot 1] A woman walks.\n"
            "overall_soundscape: Footsteps.\nnon_diegetic_music: N/A<|im_end|>"
        )
        self.assertEqual(
            MODULE._clean_llm_output(prompt),
            "integrated_multimodal_description: [Shot 1] A woman walks.\n"
            "overall_soundscape: Footsteps.\nnon_diegetic_music: N/A",
        )

    def test_llm_output_cleanup_extracts_marked_prompt_from_cli_noise(self):
        noisy = "Loading model...\nllama.cpp banner\n<H3_PROMPT>\nfinal prompt\n</H3_PROMPT>\nExiting..."
        self.assertEqual(MODULE._clean_llm_output(noisy), "final prompt")

    def test_llm_output_cleanup_removes_legacy_interactive_cli_noise(self):
        noisy = "Loading model...\navailable commands:\n\n> echoed input ... (truncated)\nfinal prompt\n\nExiting..."
        self.assertEqual(MODULE._clean_llm_output(noisy), "final prompt")

    def test_llm_output_cleanup_removes_delimiterless_omni_prompt_echo(self):
        noisy = (
            "task: T2AV\nresolution: 16:9\neffective_duration: 5.17s\n"
            "raw_prompt: [Shot 1] long request ... (truncated)\n"
            "integrated_multimodal_description: [Shot 1] Final description.\n"
            "overall_soundscape: Final ambience.\nnon_diegetic_music: N/A"
        )
        self.assertEqual(
            MODULE._clean_llm_output(noisy),
            "integrated_multimodal_description: [Shot 1] Final description.\n"
            "overall_soundscape: Final ambience.\nnon_diegetic_music: N/A",
        )

    def test_llm_output_cleanup_removes_inline_omni_prompt_echo(self):
        noisy = (
            "task: T2AV resolution: 16:9 effective_duration: 5.17s "
            "raw_prompt: [Shot 1] Korean input ... (truncated) "
            "integrated_multimodal_description: [Shot 1] Final description. "
            "overall_soundscape: Final ambience. non_diegetic_music: N/A"
        )
        self.assertEqual(
            MODULE._clean_llm_output(noisy),
            "integrated_multimodal_description: [Shot 1] Final description. "
            "overall_soundscape: Final ambience. non_diegetic_music: N/A",
        )

    def test_llm_output_cleanup_removes_complete_inline_omni_prompt_echo(self):
        noisy = (
            "task: T2AV resolution: 16:9 effective_duration: 5.17s "
            "raw_prompt: [Shot 1] A woman walks. "
            "integrated_multimodal_description: [Shot 1] Final description. "
            "overall_soundscape: Final ambience. non_diegetic_music: N/A"
        )
        self.assertEqual(
            MODULE._clean_llm_output(noisy),
            "integrated_multimodal_description: [Shot 1] Final description. "
            "overall_soundscape: Final ambience. non_diegetic_music: N/A",
        )

    def test_llm_output_cleanup_removes_omni_echo_after_runtime_banner(self):
        noisy = (
            "llama.cpp runtime ready\n"
            "task: T2AV resolution: 16:9 effective_duration: 5.17s "
            "raw_prompt: [Shot 1] A woman walks. "
            "integrated_multimodal_description: [Shot 1] Final description. "
            "overall_soundscape: Final ambience. non_diegetic_music: N/A"
        )
        self.assertEqual(
            MODULE._clean_llm_output(noisy),
            "integrated_multimodal_description: [Shot 1] Final description. "
            "overall_soundscape: Final ambience. non_diegetic_music: N/A",
        )

    def test_omni_text_prompt_is_a_single_pre_rendered_assistant_turn(self):
        rendered = MODULE._render_omni_text_prompt("System rules", "Rewrite request")
        self.assertEqual(
            rendered,
            "<|im_start|>system\nSystem rules<|im_end|>\n"
            "<|im_start|>user\nRewrite request<|im_end|>\n"
            "<|im_start|>assistant\n",
        )
        self.assertNotIn("<|im_end|>\n<|im_start|>assistant\n<|im_end|>", rendered)

    def test_qwen3_text_prompt_prefills_completed_nonthinking_block(self):
        rendered = MODULE._render_qwen3_text_prompt("System rules", "Rewrite request")
        self.assertEqual(
            rendered,
            "<|im_start|>system\nSystem rules<|im_end|>\n"
            "<|im_start|>user\nRewrite request<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        self.assertTrue(rendered.endswith("<think>\n\n</think>\n\n"))

    def test_llama_completion_finder_prefers_configured_one_shot_frontend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = Path(temp_dir) / ("llama-completion.exe" if os.name == "nt" else "llama-completion")
            executable.write_bytes(b"")
            with mock.patch.dict(os.environ, {"TOYXYZ_LLAMA_COMPLETION": str(executable)}):
                self.assertEqual(MODULE._find_llama_completion(), str(executable.resolve()))

    def test_qwen_completion_uses_supported_reasoning_off_flag(self):
        self.assertEqual(
            MODULE._qwen_thinking_args("C:/runtime/llama-completion.exe"),
            ["--reasoning", "off"],
        )
        self.assertEqual(
            MODULE._qwen_thinking_args("C:/legacy/llama-cli.exe"),
            ["--chat-template-kwargs", '{"enable_thinking":false}'],
        )

    def test_managed_runtime_uses_its_own_user_namespace(self):
        fake_folder_paths = SimpleNamespace(get_user_directory=lambda: "C:/ComfyUI/user")
        with mock.patch.dict(sys.modules, {"folder_paths": fake_folder_paths}):
            root = MODULE._llama_runtime_root().replace("\\", "/")
        self.assertEqual(root, "C:/ComfyUI/user/toyxyz_minimax_h3/runtime")

    def test_managed_runtime_reuses_complete_local_install_without_network(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Path(temp_dir)
            (runtime / "llama-completion.exe").write_bytes(b"runtime")
            (runtime / "llama-mtmd-cli.exe").write_bytes(b"runtime")
            with (
                mock.patch.object(MODULE, "_llama_runtime_backend", return_value="cuda"),
                mock.patch.object(MODULE, "_managed_llama_dir", return_value=str(runtime)),
                mock.patch.object(MODULE.urllib.request, "urlopen") as urlopen,
            ):
                self.assertEqual(MODULE._ensure_managed_llama_runtime(), str(runtime))
            urlopen.assert_not_called()

    def test_llm_output_cleanup_removes_textual_end_sentinel(self):
        output = (
            "integrated_multimodal_description: [Shot 1] A woman walks.\n"
            "overall_soundscape: Footsteps.\n"
            "non_diegetic_music: N/A [end of text]"
        )
        self.assertEqual(
            MODULE._clean_llm_output(output),
            "integrated_multimodal_description: [Shot 1] A woman walks.\n"
            "overall_soundscape: Footsteps.\n"
            "non_diegetic_music: N/A",
        )

    def test_llm_output_cleanup_uses_task_schema_for_complete_i2av_echo(self):
        noisy = (
            "task: I2AV resolution: adaptive effective_duration: 5.17s "
            "raw_prompt: [Shot 1] A woman walks. "
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced. "
            "integrated_multimodal_description: [Shot 1] Final description. "
            "overall_soundscape: Final ambience. non_diegetic_music: N/A"
        )
        cleaned = MODULE._clean_llm_output(noisy)
        self.assertTrue(cleaned.startswith("For the target video,"))
        self.assertIn("integrated_multimodal_description:", cleaned)

    def test_llm_output_cleanup_preserves_alignment_after_omni_prompt_echo(self):
        noisy = (
            "task: I2AV\nraw_prompt: long request ... (truncated)\n"
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
            "integrated_multimodal_description: [Shot 1] Final description.\n"
            "overall_soundscape: Final ambience.\nnon_diegetic_music: N/A"
        )
        cleaned = MODULE._clean_llm_output(noisy)
        self.assertTrue(cleaned.startswith("For the target video,"))
        self.assertIn("integrated_multimodal_description:", cleaned)

    def test_llm_output_cleanup_keeps_ref2av_schema_after_omni_prompt_echo(self):
        noisy = (
            "task: REF2AV\nraw_prompt: long request ... (truncated)\n"
            "subject_definitions: <Subject 1> is defined.\n"
            "summary: [reference generation] Test.\nretention_analysis: Test.\n"
            "detailed_description: [Shot 1] Test.\noverall_soundscape: N/A\n"
            "non_diegetic_music: N/A"
        )
        self.assertTrue(MODULE._clean_llm_output(noisy).startswith("subject_definitions:"))

    def test_reference_image_metadata_is_preserved(self):
        result = self.compile({
            "references": [{
                "type": "picture", "role": "subject_identity", "image_filename": "person.png",
                "image_subfolder": "toyxyz_h3_references", "image_type": "input",
            }],
        })
        ref = result["project"]["references"][0]
        self.assertEqual(ref["image_filename"], "person.png")
        self.assertEqual(ref["image_subfolder"], "toyxyz_h3_references")

    def test_reference_video_metadata_is_preserved(self):
        result = self.compile({
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "Edit @clip."}],
            "references": [{
                "type": "video", "role": "video_editing", "alias": "clip", "duration": 3,
                "source_duration": 10, "trim_start": 2, "timeline_start": 1,
                "video_filename": "source.mp4", "video_subfolder": "toyxyz_h3_references",
            }],
        })
        ref = result["project"]["references"][0]
        self.assertEqual(ref["video_filename"], "source.mp4")
        self.assertEqual(ref["video_subfolder"], "toyxyz_h3_references")
        self.assertIn("pending duration-limited ordered-frame analysis", result["draft_video_prompt"])
        self.assertEqual(ref["source_duration"], 10)
        self.assertEqual(ref["trim_start"], 2)
        self.assertEqual(ref["timeline_start"], 1)
        self.assertIn("selected_source_duration_seconds: 3.00", result["draft_video_prompt"])
        self.assertIn("source_trim_start_seconds: 2.00", result["draft_video_prompt"])
        self.assertIn("target_timeline_start_seconds: 1.00", result["draft_video_prompt"])
        self.assertIn("VIDEO_TIMELINE_PLAN:", result["draft_video_prompt"])
        self.assertIn("<Video 1>: target 1.000-4.000", result["draft_video_prompt"])
        self.assertIn("uncovered_target_intervals: 0.000-1.000, 4.000-5.167", result["draft_video_prompt"])
        self.assertIn("never stretch, freeze, loop, or hold it across an uncovered interval", result["draft_video_prompt"])
        self.assertNotIn("\nsource_duration_seconds: 5.00", result["draft_video_prompt"])

    def test_reference_audio_upload_metadata_is_preserved(self):
        result = self.compile({
            "mode": "REF2VA",
            "references": [{
                "type": "audio", "alias": "music",
                "audio_filename": "source.wav", "audio_subfolder": "toyxyz_h3_references",
            }],
        })
        ref = result["project"]["references"][0]
        self.assertEqual(ref["audio_filename"], "source.wav")
        self.assertEqual(ref["audio_subfolder"], "toyxyz_h3_references")

    def test_video_reference_system_prompt_loads_only_selected_preset_modules(self):
        project, _warnings = MODULE.normalize_project({
            "mode": "REF2VA",
            "references": [
                {"type": "video", "role": "video_editing"},
                {"type": "video", "role": "camera"},
            ],
        })
        prompt = MODULE._video_reference_system_modules(project)
        self.assertIn("VIDEO REFERENCE CORE", prompt)
        self.assertIn("<Video 1>=video_editing", prompt)
        self.assertIn("<Video 2>=camera", prompt)
        self.assertIn("VIDEO PRESET: EDITING", prompt)
        self.assertIn("complete entity replacement", prompt)
        self.assertIn("every evidenced ACTION_TIMELINE interval", prompt)
        self.assertIn("VIDEO PRESET: CAMERA", prompt)
        self.assertNotIn("VIDEO PRESET: CONTINUATION", prompt)
        self.assertNotIn("VIDEO PRESET: MOTION", prompt)

    def test_audio_reference_system_prompt_loads_only_selected_preset_modules(self):
        project, _warnings = MODULE.normalize_project({
            "mode": "REF2VA",
            "references": [
                {"type": "audio", "role": "voice_delivery"},
                {"type": "audio", "role": "music_rhythm"},
            ],
        })
        prompt = MODULE._audio_reference_system_modules(project)
        self.assertIn("AUDIO REFERENCE CORE", prompt)
        self.assertIn("<Audio 1>=voice_delivery", prompt)
        self.assertIn("<Audio 2>=music_rhythm", prompt)
        self.assertIn("AUDIO PRESET: VOICE AND DELIVERY", prompt)
        self.assertIn("AUDIO PRESET: MUSIC AND RHYTHM", prompt)
        self.assertNotIn("AUDIO PRESET: FULL SIGNAL COPY", prompt)
        self.assertNotIn("AUDIO PRESET: DIALOGUE OR LYRICS REUSE", prompt)

    def test_audio_presets_map_to_locked_retention_and_task_types(self):
        expected = {
            "none": ("reference", "audio reference"),
            "full_signal_copy": ("fully_copy", "audio reuse"),
            "partial_signal_copy": ("partially_copy", "audio reuse"),
            "voice_delivery": ("reference", "audio reference"),
            "dialogue_lyrics": ("partially_copy", "audio reuse"),
            "sound_ambience": ("reference", "audio reference"),
            "music_rhythm": ("reference", "audio reference"),
        }
        for role, (marker, task_type) in expected.items():
            with self.subTest(role=role):
                project, _warnings = MODULE.normalize_project({
                    "mode": "REF2VA",
                    "references": [{
                        "type": "audio", "role": role,
                        "audio_filename": "source.wav",
                        "description": "Apply this audio relationship to the target.",
                    }],
                })
                model = MODULE._reference_model(project)
                self.assertEqual(model["label_plan"]["<Audio 1>"]["marker"], marker)
                self.assertIn(task_type, model["task_types"])

    def test_signal_reuse_requires_uploaded_audio(self):
        project, warnings = MODULE.normalize_project({
            "mode": "REF2VA",
            "references": [{"type": "audio", "role": "full_signal_copy"}],
        })
        errors, _warnings = MODULE.validate_project(project, warnings)
        self.assertTrue(any("requires an uploaded audio file" in error for error in errors))

    def test_video_analysis_does_not_make_source_appearance_mandatory_edit_continuity(self):
        prompt = MODULE._video_analysis_prompt("video_editing", 5.0, [0.0, 2.5, 4.95])
        self.assertIn("Keep source appearance only in SUBJECTS", prompt)
        self.assertIn("do not mark identity, body appearance, hair, clothing, or accessories", prompt)

    def test_video_analysis_uses_only_requested_leading_duration_and_ordered_frames(self):
        calls = {}

        class FakeSession:
            model_path = "qwen.gguf"
            mmproj_path = "vision.gguf"

            def analyze_images(self, image_paths, captions, prompt):
                calls["paths"] = list(image_paths)
                calls["captions"] = list(captions)
                calls["prompt"] = prompt
                return "<VIDEO_ANALYSIS>ordered temporal evidence</VIDEO_ANALYSIS>"

        def fake_extract(_video_path, duration, output_dir, start_time=0.0):
            calls["duration"] = duration
            calls["start_time"] = start_time
            paths = []
            for index in range(3):
                path = Path(output_dir, f"frame-{index:03d}.jpg")
                path.write_bytes(b"jpeg")
                paths.append(str(path))
            return paths, [0.0, 1.5, 2.95]

        with (
            mock.patch.object(MODULE, "_resolve_uploaded_video", return_value="source.mp4"),
            mock.patch.object(MODULE, "_probe_video_duration", return_value=3.0),
            mock.patch.object(MODULE, "_extract_video_analysis_frames", side_effect=fake_extract),
        ):
            result = MODULE.analyze_reference_video(
                {"filename": "source.mp4"}, "video_editing", 5.0,
                session=FakeSession(),
            )
        self.assertEqual(calls["duration"], 3.0)
        self.assertEqual(calls["captions"][0], "Frame 1 at 0.000 seconds.")
        self.assertEqual(calls["captions"][-1], "Frame 3 at 2.950 seconds.")
        self.assertIn("selected source interval 0.000-3.000 seconds", calls["prompt"])
        self.assertIn("scoped edit", calls["prompt"])
        self.assertEqual(result["analysis"], "ordered temporal evidence")
        self.assertEqual(result["analyzed_duration"], "3.000")
        self.assertEqual(result["frame_count"], "3")

    def test_video_output_resamples_to_24fps_and_exact_target_frame_count(self):
        from fractions import Fraction
        import torch

        source_video = mock.Mock()
        trimmed_video = mock.Mock()
        components = SimpleNamespace(
            images=torch.arange(155, dtype=torch.float32).reshape(155, 1, 1, 1),
            audio={
                "waveform": torch.ones((1, 48000 * 8), dtype=torch.float32),
                "sample_rate": 48000,
            },
            frame_rate=Fraction(30),
        )
        trimmed_video.get_components.return_value = components
        trimmed_video.get_bit_depth.return_value = 10
        source_video.as_trimmed.return_value = trimmed_video
        created_video = object()
        captured = {}

        def create_video(output_components, bit_depth=8):
            captured["components"] = output_components
            captured["bit_depth"] = bit_depth
            return created_video

        with (
            mock.patch.object(MODULE, "_resolve_uploaded_video", return_value="source.mp4"),
            mock.patch("comfy_api.latest.InputImpl.VideoFromFile", return_value=source_video) as loader,
            mock.patch("comfy_api.latest.InputImpl.VideoFromComponents", side_effect=create_video),
        ):
            output = MODULE._load_reference_video(
                {"video_filename": "source.mp4", "video_subfolder": ""},
                124,
            )
        self.assertIs(output, created_video)
        loader.assert_called_once_with("source.mp4")
        source_video.as_trimmed.assert_called_once_with(
            0.0, 124 / MODULE.MODEL_FPS, strict_duration=False,
        )
        self.assertEqual(captured["components"].images.shape[0], 124)
        self.assertEqual(captured["components"].images[-1].item(), 154)
        self.assertEqual(captured["components"].frame_rate, Fraction(24))
        self.assertEqual(captured["components"].audio["waveform"].shape[-1], 248000)
        self.assertEqual(captured["bit_depth"], 10)

    def test_short_video_is_not_padded_to_target_frame_count(self):
        from fractions import Fraction
        import torch

        source_video = mock.Mock()
        trimmed_video = mock.Mock()
        trimmed_video.get_components.return_value = SimpleNamespace(
            images=torch.zeros((60, 1, 1, 3), dtype=torch.float32),
            audio=None,
            frame_rate=Fraction(30),
        )
        trimmed_video.get_bit_depth.return_value = 8
        source_video.as_trimmed.return_value = trimmed_video
        captured = {}

        def create_video(output_components, bit_depth=8):
            captured["components"] = output_components
            return object()

        with (
            mock.patch.object(MODULE, "_resolve_uploaded_video", return_value="short.mp4"),
            mock.patch("comfy_api.latest.InputImpl.VideoFromFile", return_value=source_video),
            mock.patch("comfy_api.latest.InputImpl.VideoFromComponents", side_effect=create_video),
        ):
            MODULE._load_reference_video(
                {"video_filename": "short.mp4", "video_subfolder": ""}, 124,
            )
        self.assertEqual(captured["components"].images.shape[0], 48)
        self.assertEqual(captured["components"].frame_rate, Fraction(24))

    def test_video_output_uses_selected_source_interval(self):
        from fractions import Fraction
        import torch

        source_video = mock.Mock()
        trimmed_video = mock.Mock()
        trimmed_video.get_components.return_value = SimpleNamespace(
            images=torch.arange(60, dtype=torch.float32).reshape(60, 1, 1, 1),
            audio=None,
            frame_rate=Fraction(30),
        )
        trimmed_video.get_bit_depth.return_value = 8
        source_video.as_trimmed.return_value = trimmed_video
        captured = {}

        def create_video(output_components, bit_depth=8):
            captured["components"] = output_components
            return object()

        with (
            mock.patch.object(MODULE, "_resolve_uploaded_video", return_value="source.mp4"),
            mock.patch("comfy_api.latest.InputImpl.VideoFromFile", return_value=source_video),
            mock.patch("comfy_api.latest.InputImpl.VideoFromComponents", side_effect=create_video),
        ):
            MODULE._load_reference_video({
                "video_filename": "source.mp4", "video_subfolder": "",
                "trim_start": 1.25, "duration": 2.0,
            }, 124)
        source_video.as_trimmed.assert_called_once_with(1.25, 3.25, strict_duration=False)
        self.assertEqual(captured["components"].images.shape[0], 48)
        self.assertEqual(captured["components"].frame_rate, Fraction(24))

    def test_video_output_uses_only_timeline_visible_intersection(self):
        from fractions import Fraction
        import torch

        source_video = mock.Mock()
        trimmed_video = mock.Mock()
        trimmed_video.get_components.return_value = SimpleNamespace(
            images=torch.arange(120, dtype=torch.float32).reshape(120, 1, 1, 1),
            audio=None,
            frame_rate=Fraction(30),
        )
        trimmed_video.get_bit_depth.return_value = 8
        source_video.as_trimmed.return_value = trimmed_video
        captured = {}

        def create_video(output_components, bit_depth=8):
            captured["components"] = output_components
            return object()

        with (
            mock.patch.object(MODULE, "_resolve_uploaded_video", return_value="source.mp4"),
            mock.patch("comfy_api.latest.InputImpl.VideoFromFile", return_value=source_video),
            mock.patch("comfy_api.latest.InputImpl.VideoFromComponents", side_effect=create_video),
        ):
            MODULE._load_reference_video({
                "video_filename": "source.mp4", "video_subfolder": "",
                "trim_start": 1.0, "duration": 6.0, "timeline_start": -2.0,
            }, 124)

        source_video.as_trimmed.assert_called_once_with(3.0, 7.0, strict_duration=False)
        self.assertEqual(captured["components"].images.shape[0], 96)
        self.assertEqual(captured["components"].frame_rate, Fraction(24))

    def test_video_output_keeps_one_frame_trim_boundary_rounding(self):
        from fractions import Fraction
        import torch

        source_video = mock.Mock()
        trimmed_video = mock.Mock()
        trimmed_video.get_components.return_value = SimpleNamespace(
            images=torch.arange(26, dtype=torch.float32).reshape(26, 1, 1, 1),
            audio=None,
            frame_rate=Fraction(30),
        )
        trimmed_video.get_bit_depth.return_value = 8
        source_video.as_trimmed.return_value = trimmed_video
        captured = {}

        def create_video(output_components, bit_depth=8):
            captured["components"] = output_components
            return object()

        with (
            mock.patch.object(MODULE, "_resolve_uploaded_video", return_value="source.mp4"),
            mock.patch("comfy_api.latest.InputImpl.VideoFromFile", return_value=source_video),
            mock.patch("comfy_api.latest.InputImpl.VideoFromComponents", side_effect=create_video),
        ):
            MODULE._load_reference_video({
                "video_filename": "source.mp4", "video_subfolder": "",
                "trim_start": 0.0, "duration": 22 / 24,
            }, 124)

        self.assertEqual(captured["components"].images.shape[0], 22)
        self.assertEqual(captured["components"].frame_rate, Fraction(24))

    def test_prompt_generation_job_cancel_sets_event_and_calls_active_stopper(self):
        job_id = "cancel-test-job"
        stopped = []
        event = MODULE._begin_enhance_job(job_id)
        try:
            MODULE._set_enhance_stopper(job_id, lambda: stopped.append(True))
            self.assertTrue(MODULE._cancel_enhance_job(job_id))
            self.assertTrue(event.is_set())
            self.assertEqual(stopped, [True])
            self.assertEqual(MODULE._get_enhance_job(job_id)["stage"], "cancelled")
        finally:
            MODULE._finish_enhance_job(job_id)

    def test_audio_output_trims_to_the_same_target_duration(self):
        import torch

        sample_rate = 48000
        source = torch.ones((2, sample_rate * 8), dtype=torch.float32)
        with (
            mock.patch.object(MODULE, "_resolve_uploaded_audio", return_value="source.wav"),
            mock.patch("comfy_extras.nodes_audio.load", return_value=(source, sample_rate)),
        ):
            output = MODULE._load_reference_audio(
                {"audio_filename": "source.wav", "audio_subfolder": ""},
                124 / MODULE.MODEL_FPS,
            )
        expected_samples = round((124 / MODULE.MODEL_FPS) * sample_rate)
        self.assertEqual(output["sample_rate"], sample_rate)
        self.assertEqual(tuple(output["waveform"].shape), (1, 2, expected_samples))

    def test_short_audio_is_not_padded_past_its_available_samples(self):
        import torch

        sample_rate = 44100
        source = torch.ones((1, sample_rate * 2), dtype=torch.float32)
        with (
            mock.patch.object(MODULE, "_resolve_uploaded_audio", return_value="short.wav"),
            mock.patch("comfy_extras.nodes_audio.load", return_value=(source, sample_rate)),
        ):
            output = MODULE._load_reference_audio(
                {"audio_filename": "short.wav", "audio_subfolder": ""},
                124 / MODULE.MODEL_FPS,
            )
        self.assertEqual(output["waveform"].shape[-1], sample_rate * 2)

    def test_reference_analysis_cleanup_extracts_only_marked_text(self):
        noisy = "Loading model...\n<REFERENCE_ANALYSIS>observable English details</REFERENCE_ANALYSIS>\nExiting..."
        self.assertEqual(MODULE._clean_reference_analysis(noisy), "observable English details")

    def test_reference_analysis_cleanup_accepts_missing_closing_marker(self):
        noisy = "Loading model...\ncommands...\n<REFERENCE_ANALYSIS>\nobservable English details"
        self.assertEqual(MODULE._clean_reference_analysis(noisy), "observable English details")

    def test_webp_reference_is_temporarily_converted_for_vision_backend(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as root:
            source = Path(root, "reference.webp")
            Image.new("RGB", (8, 6), (240, 120, 30)).save(source, format="WEBP")
            with MODULE._vision_compatible_image(str(source)) as converted:
                converted_path = Path(converted)
                self.assertNotEqual(converted_path, source)
                self.assertEqual(converted_path.suffix.lower(), ".png")
                self.assertTrue(converted_path.is_file())
                with Image.open(converted_path) as prepared:
                    self.assertEqual(prepared.size, (8, 6))
            self.assertFalse(converted_path.exists())
            self.assertTrue(source.exists())

    def test_reference_analysis_requires_visual_medium_and_rendering_style_for_every_role(self):
        for role in ("reference", "subject_identity", "first_frame", "last_frame"):
            with self.subTest(role=role):
                prompt = MODULE._reference_analysis_prompt(role)
                self.assertIn("VISUAL_MEDIUM:", prompt)
                self.assertIn("exactly the eight labeled lines", prompt)
                self.assertIn("2D anime illustration", prompt)
                self.assertIn("3D CGI render", prompt)
                self.assertIn("physical collectible figurine photograph", prompt)
                self.assertIn("Keep style separate from identity facts", prompt)
                self.assertIn("never guess a production method", prompt)

    def test_reference_analysis_uses_image_and_mmproj(self):
        progress_events = []
        with tempfile.TemporaryDirectory() as root:
            Path(root, "image.png").write_bytes(b"image")
            fake_folder_paths = SimpleNamespace(get_input_directory=lambda: root)

            def fake_run(command, **kwargs):
                self.assertIn("--image", command)
                self.assertIn("--mmproj", command)
                self.assertIn("stable identity", command_prompt(command).lower())
                self.assertIn("Never infer or label nationality, ethnicity, race, age", command_prompt(command))
                self.assertIn("Do not use speculative alternatives joined by or", command_prompt(command))
                self.assertIn("crop boundaries, visible body range", command_prompt(command))
                self.assertIn("VISUAL_MEDIUM:", command_prompt(command))
                self.assertIn("POSE_SUPPORT_CONTACT:", command_prompt(command))
                self.assertIn("ACTION_RELEVANT_OBJECTS:", command_prompt(command))
                return SimpleNamespace(
                    returncode=0,
                    stdout="<REFERENCE_ANALYSIS>stable observable details</REFERENCE_ANALYSIS>",
                    stderr="",
                )

            with (
                mock.patch.dict(sys.modules, {"folder_paths": fake_folder_paths}),
                mock.patch.object(MODULE, "_resolve_image_model", return_value=("vision.gguf", "mmproj.gguf")),
                mock.patch.object(MODULE, "_find_llama_cli", return_value="llama-cli"),
                mock.patch.object(MODULE, "_find_llama_completion", return_value="llama-cli"),
                mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run),
            ):
                result = MODULE.analyze_reference_image(
                    {"filename": "image.png", "_analysis_index": 1, "_analysis_total": 2},
                    "subject_identity",
                    progress=lambda **values: progress_events.append(values),
                )
        self.assertEqual(result["analysis"], "stable observable details")
        self.assertEqual(progress_events[-1]["stage"], "reference_analysis")
        self.assertIn("1/2", progress_events[-1]["message"])
        self.assertIn("subject_identity", progress_events[-1]["message"])

    def test_qwen_reference_analysis_disables_thinking(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "image.png").write_bytes(b"image")
            fake_folder_paths = SimpleNamespace(get_input_directory=lambda: root)

            def fake_run(command, **kwargs):
                self.assertIn("--jinja", command)
                self.assertEqual(command[command.index("--reasoning") + 1], "off")
                self.assertEqual(
                    command[command.index("--chat-template-kwargs") + 1],
                    '{"enable_thinking":false}',
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout="<REFERENCE_ANALYSIS>Qwen vision details</REFERENCE_ANALYSIS>",
                    stderr="",
                )

            with (
                mock.patch.dict(sys.modules, {"folder_paths": fake_folder_paths}),
                mock.patch.object(MODULE, "_resolve_image_model", return_value=("qwen.gguf", "vision.gguf")),
                mock.patch.object(MODULE, "_find_llama_cli", return_value="llama-cli"),
                mock.patch.object(MODULE, "_find_llama_completion", return_value="llama-cli"),
                mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run),
            ):
                result = MODULE.analyze_reference_image(
                    {"filename": "image.png"}, "first_frame", MODULE.QWEN_IMAGE_MODEL_ID,
                )
        self.assertEqual(result["analysis"], "Qwen vision details")

    def test_enhance_project_runs_selected_model_with_chat_messages(self):
        def fake_run(command, **kwargs):
            self.assertIn("-no-cnv", command)
            self.assertIn("-st", command)
            self.assertIn("--special", command)
            self.assertNotIn("--chat-template-kwargs", command)
            rendered = command_prompt(command)
            system_text = rendered.split("<|im_end|>", 1)[0].split("\n", 1)[1]
            self.assertTrue(system_text.startswith("ACTIVE MODE: T2VA"))
            self.assertIn("Minimal detail needed to make the request renderable", system_text)
            self.assertIn("FINAL MODE LOCK — T2VA", system_text)
            self.assertNotIn("<H3_PROMPT>", system_text)
            self.assertIn("Return plain text with no wrapper", system_text)
            user_data = rendered.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
            self.assertTrue(user_data.startswith("INPUT DATA ONLY"))
            self.assertIn("mode: T2VA", user_data)
            self.assertIn("visual_action: A woman opens a door.", user_data)
            self.assertNotIn("--log-disable", command)
            self.assertNotIn("--reasoning", command)
            self.assertEqual(command[command.index("--top-k") + 1], "20")
            self.assertEqual(command[command.index("-n") + 1], "1800")
            return SimpleNamespace(returncode=0, stdout="<H3_PROMPT>\nintegrated_multimodal_description: [Shot 1] enhanced H3 prompt\n\noverall_soundscape: N/A\n\nnon_diegetic_music: N/A\n</H3_PROMPT>", stderr="")

        with (
            mock.patch.object(MODULE, "_resolve_enhance_model", return_value="model.gguf"),
            mock.patch.object(MODULE, "_find_llama_cli", return_value="llama-cli"),
            mock.patch.object(MODULE, "_find_llama_completion", return_value="llama-cli"),
            mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run),
        ):
            result = MODULE.enhance_project({
                "mode": "T2VA",
                "shots": [{"duration": 5, "visual_action": "A woman opens a door."}],
            }, "local:model.gguf")
        self.assertIn("[Shot 1] enhanced H3 prompt", result["enhanced_prompt"])

    def test_enhance_project_returns_fl2va_model_output_without_normalizing_it(self):
        def fake_run(command, **kwargs):
            output = (
                "Picture 1 anchors the visual state at 0.00 seconds, and Picture 2 anchors the end.\n\n"
                "integrated_multimodal_description: [Shot 1] The exact state in Picture 1 transitions "
                "continuously into Picture 2 while the static framing remains unchanged.\n\n"
                "overall_soundscape: N/A\n\nnon_diegetic_music: N/A"
            )
            return SimpleNamespace(returncode=0, stdout=f"<H3_PROMPT>{output}</H3_PROMPT>", stderr="")

        with (
            mock.patch.object(MODULE, "_resolve_enhance_model", return_value="model.gguf"),
            mock.patch.object(MODULE, "_find_llama_cli", return_value="llama-cli"),
            mock.patch.object(MODULE, "_find_llama_completion", return_value="llama-cli"),
            mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run),
        ):
            result = MODULE.enhance_project({
                "mode": "FL2VA",
                "requested_duration": 5,
                "shots": [{"duration": 5, "visual_action": "The character transforms continuously."}],
                "references": [
                    {"type": "picture", "role": "first_frame"},
                    {"type": "picture", "role": "last_frame"},
                ],
            }, "local:model.gguf")

        expected = (
            "How the reference pictures align with the target video — "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            "Picture 2 (from Shot 1) aligns with the 5.17-second mark of the target video."
        )
        self.assertTrue(result["enhanced_prompt"].startswith("Picture 1 anchors the visual state"))
        self.assertIn("Picture 2 anchors the end", result["enhanced_prompt"])

    def test_base_normalizer_moves_i2va_alignment_outside_main_field(self):
        malformed = (
            "integrated_multimodal_description: For the target video, at 0.00 seconds into the target "
            "video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
            "[Shot 1] The subject begins moving after the anchored frame.\n\n"
            "overall_soundscape: N/A\n\nnon_diegetic_music: N/A"
        )
        normalized = MODULE._normalize_base_enhanced_prompt(malformed, "I2VA", 5.17, 1)
        self.assertTrue(normalized.startswith(MODULE.I2VA_ALIGNMENT_INSTRUCTION + "\n\n"))
        self.assertIn(
            "integrated_multimodal_description: [Shot 1] The subject begins moving",
            normalized,
        )
        self.assertEqual(normalized.count(MODULE.I2VA_ALIGNMENT_INSTRUCTION), 1)
        self.assertEqual(MODULE._base_prompt_structure_issues(normalized, "I2VA", 5.17, 1), [])

    def test_i2va_semantic_validator_finds_unsupported_inference(self):
        prompt = (
            MODULE.I2VA_ALIGNMENT_INSTRUCTION
            + "\n\nintegrated_multimodal_description: [Shot 1] A young East Asian woman reaches "
            "below the frame for an unseen object, raises a revolver, and produces bone fragments.\n\n"
            "overall_soundscape: A hiss of smoke.\n\nnon_diegetic_music: N/A"
        )
        issues = MODULE._i2va_semantic_issues(
            prompt,
            "The subject raises a weapon while remaining in the visible frame.",
        )
        joined = " ".join(issues)
        self.assertIn("age", joined)
        self.assertIn("ethnicity", joined)
        self.assertIn("hidden or off-frame source", joined)
        self.assertIn("weapon type", joined)
        self.assertIn("graphic effects", joined)
        self.assertIn("smoke sound", joined)

    def test_i2va_validator_does_not_treat_vision_inference_as_user_authority(self):
        prompt = (
            MODULE.I2VA_ALIGNMENT_INSTRUCTION
            + "\n\nintegrated_multimodal_description: [Shot 1] A photorealistic medium shot frames "
            "a young woman and crops at her mid-thighs. Her glistening skin suggests body oil or water. "
            "She raises a handgun, then her head snaps sideways due to the impact and her body begins "
            "to slump.\n\noverall_soundscape: A gunshot and the soft thud of the woman's head snapping "
            "sideways.\n\nnon_diegetic_music: N/A"
        )
        issues = MODULE._i2va_semantic_issues(
            prompt,
            "[Shot 1] A woman raises a handgun.",
            "A young woman has glistening skin that may be covered in oil or water.",
        )
        joined = " ".join(issues)
        self.assertIn("age", joined)
        self.assertIn("speculative alternatives", joined)
        self.assertIn("observable facts", joined)
        self.assertIn("mid-thigh crop", joined)
        self.assertIn("physical consequences", joined)

    def test_i2va_validator_respects_explicit_korean_user_details(self):
        prompt = (
            MODULE.I2VA_ALIGNMENT_INSTRUCTION
            + "\n\nintegrated_multimodal_description: [Shot 1] A young woman moves as blood appears.\n\n"
            "overall_soundscape: N/A\n\nnon_diegetic_music: N/A"
        )
        issues = MODULE._i2va_semantic_issues(
            prompt,
            "[Shot 1] 젊은 여성이 움직이고 피가 나온다.",
        )
        joined = " ".join(issues)
        self.assertNotIn("age", joined)
        self.assertNotIn("graphic effects", joined)

    def test_enhance_project_does_not_retry_i2va_fidelity_violations(self):
        calls = []

        def fake_run(command, **kwargs):
            prompt = command_prompt(command)
            temperature = command[command.index("--temp") + 1]
            calls.append((prompt, temperature))
            if len(calls) == 1:
                output = (
                    "integrated_multimodal_description: For the target video, at 0.00 seconds into the "
                    "target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
                    "[Shot 1] A young woman reaches below the frame for an unseen revolver, then raises it.\n\n"
                    "overall_soundscape: N/A\n\nnon_diegetic_music: N/A"
                )
            else:
                self.assertIn("Correct only the listed I2VA fidelity violations", prompt)
                output = (
                    MODULE.I2VA_ALIGNMENT_INSTRUCTION
                    + "\n\nintegrated_multimodal_description: [Shot 1] The subject remains in the anchored "
                    "composition. The weapon first enters the visible frame, then rises smoothly.\n\n"
                    "overall_soundscape: N/A\n\nnon_diegetic_music: N/A"
                )
            return SimpleNamespace(
                returncode=0,
                stdout=f"<H3_PROMPT>{output}</H3_PROMPT>",
                stderr="",
            )

        with (
            mock.patch.object(MODULE, "_resolve_enhance_model", return_value="model.gguf"),
            mock.patch.object(MODULE, "_find_llama_cli", return_value="llama-cli"),
            mock.patch.object(MODULE, "_find_llama_completion", return_value="llama-cli"),
            mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run),
        ):
            result = MODULE.enhance_project({
                "mode": "I2VA",
                "shots": [{"duration": 5, "visual_action": "The subject raises a weapon."}],
                "references": [{"type": "picture", "role": "first_frame"}],
            }, "local:model.gguf")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], "0.22")
        self.assertIn("young", result["enhanced_prompt"].lower())
        self.assertIn("unseen", result["enhanced_prompt"].lower())

    def test_i2va_deterministic_cleanup_repairs_repeated_framing_and_reaction_errors(self):
        invalid = (
            MODULE.I2VA_ALIGNMENT_INSTRUCTION
            + "\n\nintegrated_multimodal_description: [Shot 1] A medium shot frames the subject and "
            "crops at her mid-thighs. She fires, and her head snaps sharply to her left side due to "
            "the impact as her body begins to slump.\n\noverall_soundscape: Ocean waves are punctuated "
            "by a gunshot and the soft thud of the woman's head snapping sideways.\n\n"
            "non_diegetic_music: N/A"
        )
        cleaned = MODULE._sanitize_i2va_semantics(
            invalid,
            "[Shot 1] The subject fires.",
        )
        self.assertIn("medium-full shot", cleaned)
        self.assertIn("She fires.", cleaned)
        self.assertIn("punctuated by a gunshot.", cleaned)
        self.assertNotIn("head snaps", cleaned.lower())
        self.assertNotIn("slump", cleaned.lower())
        self.assertNotIn("soft thud", cleaned.lower())
        self.assertEqual(
            MODULE._i2va_semantic_issues(cleaned, "[Shot 1] The subject fires."),
            [],
        )

    def test_i2va_deterministic_cleanup_removes_repeated_speculation(self):
        invalid = (
            MODULE.I2VA_ALIGNMENT_INSTRUCTION
            + "\n\nintegrated_multimodal_description: [Shot 1] Her skin has strong highlights, "
            "suggesting the application of body oil or water. The lighting is bright and direct, "
            "likely from the upper left, casting defined shadows. She remains centered.\n\n"
            "overall_soundscape: Ocean waves.\n\nnon_diegetic_music: N/A"
        )
        cleaned = MODULE._sanitize_i2va_semantics(invalid, "[Shot 1] She remains centered.")
        lowered = cleaned.lower()
        self.assertNotIn("suggesting", lowered)
        self.assertNotIn("likely", lowered)
        self.assertNotIn("oil or water", lowered)
        self.assertIn("Her skin has strong highlights.", cleaned)
        self.assertIn("The lighting is bright and direct.", cleaned)
        self.assertIn("She remains centered.", cleaned)
        self.assertEqual(
            MODULE._i2va_semantic_issues(cleaned, "[Shot 1] She remains centered."),
            [],
        )

    def test_enhance_project_does_not_apply_i2va_cleanup(self):
        calls = []
        invalid = (
            MODULE.I2VA_ALIGNMENT_INSTRUCTION
            + "\n\nintegrated_multimodal_description: [Shot 1] A medium shot frames the subject, "
            "cropping at her mid-thighs. She performs the requested action, and the impact causes "
            "her head to jerk sharply to her left side. Her body remains upright but sways slightly "
            "from the force of the movement.\n\noverall_soundscape: A sharp sound.\n\n"
            "non_diegetic_music: N/A"
        )

        def fake_run(command, **kwargs):
            calls.append(command_prompt(command))
            return SimpleNamespace(
                returncode=0,
                stdout=f"<H3_PROMPT>{invalid}</H3_PROMPT>",
                stderr="",
            )

        with (
            mock.patch.object(MODULE, "_resolve_enhance_model", return_value="model.gguf"),
            mock.patch.object(MODULE, "_find_llama_cli", return_value="llama-cli"),
            mock.patch.object(MODULE, "_find_llama_completion", return_value="llama-cli"),
            mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run),
        ):
            result = MODULE.enhance_project({
                "mode": "I2VA",
                "shots": [{"duration": 5, "visual_action": "The subject performs the requested action."}],
                "references": [{"type": "picture", "role": "first_frame"}],
            }, "local:model.gguf")

        self.assertEqual(len(calls), 1)
        self.assertIn("medium shot", result["enhanced_prompt"])
        self.assertIn("head to jerk", result["enhanced_prompt"].lower())
        self.assertIn("sways slightly", result["enhanced_prompt"].lower())

    def test_enhance_project_supplies_reference_image_analysis_as_explicit_context(self):
        analysis_calls = []

        def fake_analyze(image, role, model_id, progress=None):
            analysis_calls.append((image, role, model_id))
            return {
                "analysis": "A woman with long silver hair and a red coat.",
                "model_path": "vision.gguf",
                "mmproj_path": "mmproj.gguf",
            }

        def fake_run(command, **kwargs):
            rendered = command_prompt(command)
            user_prompt = rendered.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
            system_text = rendered.split("<|im_end|>", 1)[0].split("\n", 1)[1]
            self.assertTrue(system_text.startswith("ACTIVE MODE: REF2VA FULL-REFERENCE"))
            self.assertIn("FINAL MODE LOCK — REF2VA", system_text)
            self.assertIn("Define and use exactly these output labels in order: <Subject 1>", system_text)
            self.assertIn("never use `integrated_multimodal_description:`", system_text)
            self.assertEqual(command[command.index("--top-k") + 1], "20")
            self.assertEqual(command[command.index("-n") + 1], "3072")
            self.assertTrue(user_prompt.startswith("INPUT DATA ONLY"))
            self.assertIn("mode: REF2VA", user_prompt)
            self.assertIn("<Subject 1>\nsource: <Picture 1>", user_prompt)
            self.assertIn("role: subject_identity", user_prompt)
            self.assertIn("long silver hair and a red coat", user_prompt)
            self.assertNotIn("REFERENCE IMAGE ANALYSIS:", user_prompt)
            self.assertNotIn("RAW H3 PROMPT TO EXPAND:", user_prompt)
            self.assertNotIn("Stale analysis from an earlier role", user_prompt)
            return SimpleNamespace(
                returncode=0,
                stdout="<H3_PROMPT>subject_definitions:\n<Subject 1> is defined from <Picture 1>.\n\nsummary:\n[reference generation] The target video uses <Subject 1>.\n\nretention_analysis:\n<Subject 1>: fully_preserved - retained.\n\ndetailed_description:\n[Shot 1] <Subject 1> turns toward the camera with image evidence.\n\noverall_soundscape:\nQuiet room tone and soft movement sounds.\n\nnon_diegetic_music:\nN/A</H3_PROMPT>",
                stderr="",
            )

        with (
            mock.patch.object(MODULE, "_start_persistent_image_server", side_effect=RuntimeError("disabled in test")),
            mock.patch.object(MODULE, "analyze_reference_image", side_effect=fake_analyze),
            mock.patch.object(MODULE, "_resolve_enhance_model", return_value="model.gguf"),
            mock.patch.object(MODULE, "_find_llama_cli", return_value="llama-cli"),
            mock.patch.object(MODULE, "_find_llama_completion", return_value="llama-cli"),
            mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run),
        ):
            result = MODULE.enhance_project({
                "mode": "REF2VA",
                "shots": [{"duration": 5, "visual_action": "@hero turns toward the camera."}],
                "references": [{
                    "id": "hero-ref", "type": "picture", "role": "subject_identity", "alias": "@hero",
                    "description": "Stale analysis from an earlier role.",
                    "image_filename": "hero.png", "image_subfolder": "toyxyz_h3_references",
                }],
            }, "local:model.gguf", "qwen3.8-vision")
        self.assertIn(
            "<Subject 1> (appears in [Shot 1]): fully_preserved - retained.",
            result["enhanced_prompt"],
        )
        self.assertIn("[Shot 1] <Subject 1> turns toward the camera with image evidence", result["enhanced_prompt"])
        self.assertEqual(result["reference_analyses"][0]["id"], "hero-ref")
        self.assertEqual(result["reference_analyses"][0]["analysis"], "A woman with long silver hair and a red coat.")
        self.assertEqual(result["reference_analyses"][0]["filename"], "hero.png")
        self.assertEqual(len(analysis_calls), 1)
        self.assertEqual(analysis_calls[0][1:], ("subject_identity", "qwen3.8-vision"))

    def test_frontend_logs_complete_vision_analysis_after_enhancement(self):
        source = (MODULE_PATH.parent.parent / "web" / "minimax_h3_prompter.js").read_text(encoding="utf-8")
        self.assertIn("Vision analysis for ${label} [role=${role}]", source)
        self.assertIn('item.analysis || "No analysis text returned."', source)
        self.assertIn('"analysis"', source)

    def test_enhance_project_reuses_one_persistent_server_for_three_images(self):
        class FakeSession:
            def __init__(self):
                self.closed = 0

            def close(self):
                self.closed += 1

        session = FakeSession()
        analysis_calls = []

        def fake_server_analysis(image, role, active_session, progress=None):
            analysis_calls.append((image["filename"], role, active_session))
            return {"analysis": f"Observable traits from {image['filename']}."}

        labels = ("<Subject 1>", "<Subject 2>", "<Subject 3>")
        enhanced = (
            "subject_definitions:\n"
            "<Subject 1> is guidance derived from <Picture 1>.\n"
            "<Subject 2> is guidance derived from <Picture 2>.\n"
            "<Subject 3> is guidance derived from <Picture 3>.\n\n"
            "summary:\n[reference generation] The target uses <Subject 1>, <Subject 2>, and <Subject 3>.\n\n"
            "retention_analysis:\n"
            "<Subject 1>: weak_reference - broad similarity only.\n"
            "<Subject 2>: weak_reference - broad similarity only.\n"
            "<Subject 3>: weak_reference - broad similarity only.\n\n"
            "detailed_description:\n[Shot 1] <Subject 1>, <Subject 2>, and <Subject 3> move together.\n\n"
            "overall_soundscape:\nSoft movement sounds and room tone.\n\n"
            "non_diegetic_music:\nN/A"
        )

        def fake_run(command, **kwargs):
            return SimpleNamespace(returncode=0, stdout=f"<H3_PROMPT>{enhanced}</H3_PROMPT>", stderr="")

        with (
            mock.patch.object(MODULE, "_start_persistent_image_server", return_value=session) as start_server,
            mock.patch.object(MODULE, "_analyze_reference_image_with_server", side_effect=fake_server_analysis),
            mock.patch.object(MODULE, "analyze_reference_image") as cli_analysis,
            mock.patch.object(MODULE, "_resolve_enhance_model", return_value="model.gguf"),
            mock.patch.object(MODULE, "_find_llama_cli", return_value="llama-cli"),
            mock.patch.object(MODULE, "_find_llama_completion", return_value="llama-cli"),
            mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run),
        ):
            result = MODULE.enhance_project({
                "mode": "REF2VA",
                "shots": [{"duration": 5, "visual_action": "Three subjects move together."}],
                "references": [
                    {"id": f"ref-{index}", "type": "picture", "role": "reference",
                     "image_filename": f"image-{index}.png"}
                    for index in range(1, 4)
                ],
            }, "local:model.gguf")
        start_server.assert_called_once()
        self.assertEqual(len(analysis_calls), 3)
        self.assertTrue(all(call[2] is session for call in analysis_calls))
        cli_analysis.assert_not_called()
        self.assertEqual(session.closed, 1)
        self.assertTrue(all(label in result["enhanced_prompt"] for label in labels))

    def test_llama_server_image_request_uses_openai_multimodal_payload(self):
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": "<REFERENCE_ANALYSIS>Visible facts.</REFERENCE_ANALYSIS>"}}],
                }).encode("utf-8")

        session = MODULE._LlamaServerSession("llama-server", "model.gguf", "mmproj.gguf")
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test.png"
            image_path.write_bytes(b"not-a-real-png")
            with mock.patch.object(MODULE.urllib.request, "urlopen", return_value=FakeResponse()) as request:
                output = session.analyze_image(str(image_path), "Analyze this image.")
        payload = json.loads(request.call_args.args[0].data.decode("utf-8"))
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Analyze this image."})
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertIn("Visible facts", output)

    def test_enhance_project_does_not_correct_an_invented_second_shot(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command_prompt(command))
            if len(calls) == 1:
                output = "integrated_multimodal_description: [Shot 1] First. [Shot 2] At 00:04.500, invented cut.\n\noverall_soundscape: N/A\n\nnon_diegetic_music: N/A"
            else:
                self.assertIn("SHOT TOPOLOGY LOCK", calls[-1].upper())
                output = "integrated_multimodal_description: [Shot 1] First, with all detail retained in one continuous shot.\n\noverall_soundscape: N/A\n\nnon_diegetic_music: N/A"
            return SimpleNamespace(returncode=0, stdout=f"<H3_PROMPT>{output}</H3_PROMPT>", stderr="")

        with (
            mock.patch.object(MODULE, "_resolve_enhance_model", return_value="model.gguf"),
            mock.patch.object(MODULE, "_find_llama_cli", return_value="llama-cli"),
            mock.patch.object(MODULE, "_find_llama_completion", return_value="llama-cli"),
            mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run),
        ):
            result = MODULE.enhance_project({
                "mode": "T2VA",
                "shots": [{"duration": 5, "visual_action": "A woman walks along a beach."}],
            }, "local:model.gguf")
        self.assertEqual(len(calls), 1)
        self.assertEqual(MODULE._prompt_shot_numbers(result["enhanced_prompt"]), [1, 2])

    def test_enhance_project_does_not_correct_ref_weak_role_overreach(self):
        calls = []

        def fake_analyze(image, role, model_id, progress=None):
            return {
                "analysis": "An anime illustration of a blonde woman on a beach beside ocean waves.",
                "model_path": "vision.gguf", "mmproj_path": "mmproj.gguf",
            }

        def fake_run(command, **kwargs):
            prompt_text = command_prompt(command)
            calls.append(prompt_text)
            if len(calls) == 1:
                output = (
                    "subject_definitions:\n<Subject 1> is a blonde woman derived from <Picture 1>.\n\n"
                    "summary:\n[reference generation] Use <Subject 1> for character identity and styling guidance.\n\n"
                    "retention_analysis:\n<Subject 1>: weak_reference - retain blonde hair, a white bikini, "
                    "white sandals, and wet skin.\n\n"
                    "detailed_description:\nThe target uses an anime illustration style.\n"
                    "[Shot 1] <Subject 1> walks through a hotel corridor with ocean waves behind her.\n\n"
                    "overall_soundscape:\nFootsteps and room tone.\n\nnon_diegetic_music:\nN/A"
                )
            else:
                self.assertIn("weak_reference", prompt_text)
                self.assertIn("does not transfer source environment", prompt_text)
                self.assertIn("does not transfer source style", prompt_text)
                output = (
                    "subject_definitions:\n<Subject 1> is reusable appearance guidance derived from <Picture 1>.\n\n"
                    "summary:\n[reference generation] The target uses <Subject 1> as weak appearance guidance.\n\n"
                    "retention_analysis:\n<Subject 1>: weak_reference - use broad similarity in selected wardrobe traits.\n\n"
                    "detailed_description:\nThe target uses a clean digital-video presentation. "
                    "The source metadata mentions [Shot 1], while <Subject 1> walks steadily through "
                    "the requested hotel corridor.\n\n"
                    "overall_soundscape:\nSoft footsteps and quiet indoor room tone.\n\nnon_diegetic_music:\nN/A"
                )
            return SimpleNamespace(returncode=0, stdout=f"<H3_PROMPT>{output}</H3_PROMPT>", stderr="")

        with (
            mock.patch.object(MODULE, "_start_persistent_image_server", side_effect=RuntimeError("disabled in test")),
            mock.patch.object(MODULE, "analyze_reference_image", side_effect=fake_analyze),
            mock.patch.object(MODULE, "_resolve_enhance_model", return_value="model.gguf"),
            mock.patch.object(MODULE, "_find_llama_cli", return_value="llama-cli"),
            mock.patch.object(MODULE, "_find_llama_completion", return_value="llama-cli"),
            mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run),
        ):
            result = MODULE.enhance_project({
                "mode": "REF2VA",
                "shots": [{"duration": 5, "visual_action": "A woman walks through a hotel corridor."}],
                "references": [{
                    "type": "picture", "role": "reference",
                    "image_filename": "reference.png", "image_subfolder": "toyxyz_h3_references",
                }],
            }, "local:model.gguf")
        self.assertEqual(len(calls), 1)
        self.assertIn("styling guidance", result["enhanced_prompt"])
        self.assertIn("ocean waves", result["enhanced_prompt"])
        self.assertEqual(MODULE._prompt_shot_numbers(result["enhanced_prompt"]), [1])

    def test_enhance_project_does_not_repair_ref_summary_or_shot_header(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command_prompt(command))
            output = (
                "subject_definitions:\n<Subject 1> is reusable guidance derived from <Picture 1>.\n\n"
                "summary:\n[Reference Generation Task] The target uses <Subject 1> as weak appearance guidance.\n\n"
                "retention_analysis:\n<Subject 1>: weak_reference - use broad visual similarity only.\n\n"
                "detailed_description:\nThe target uses a clean digital-video presentation. "
                "<Subject 1> walks through the requested hotel corridor.\n\n"
                "overall_soundscape:\nSoft footsteps and quiet indoor room tone.\n\n"
                "non_diegetic_music:\nN/A"
            )
            return SimpleNamespace(returncode=0, stdout=f"<H3_PROMPT>{output}</H3_PROMPT>", stderr="")

        with (
            mock.patch.object(MODULE, "_resolve_enhance_model", return_value="model.gguf"),
            mock.patch.object(MODULE, "_find_llama_cli", return_value="llama-cli"),
            mock.patch.object(MODULE, "_find_llama_completion", return_value="llama-cli"),
            mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run),
        ):
            result = MODULE.enhance_project({
                "mode": "REF2VA",
                "shots": [{"duration": 5, "visual_action": "A person walks through a hotel corridor."}],
                "references": [{"type": "picture", "role": "reference"}],
            }, "local:model.gguf")
        self.assertEqual(len(calls), 1)
        self.assertIn("summary:\n[Reference Generation Task]", result["enhanced_prompt"])
        self.assertEqual(MODULE._prompt_shot_numbers(result["enhanced_prompt"]), [])

    def test_enhance_project_does_not_recover_missing_first_header(self):
        calls = []

        def output(weak_detail):
            retention = (
                "retain red hair, a grey jacket, black boots, and gold buttons"
                if weak_detail else "use broad similarity in selected appearance traits"
            )
            return (
                "subject_definitions:\n<Subject 1> is reusable guidance derived from <Picture 1>.\n\n"
                "summary:\n[reference generation] The target uses <Subject 1> as weak appearance guidance.\n\n"
                f"retention_analysis:\n<Subject 1>: weak_reference - {retention}.\n\n"
                "detailed_description:\nThe target uses a clean digital-video presentation. "
                "<Subject 1> walks through the corridor in the opening shot.\n"
                "[Shot 2] At 00:02.500, cut to a new shot. <Subject 1> stops near a door.\n\n"
                "overall_soundscape:\nSoft footsteps and quiet room tone.\n\n"
                "non_diegetic_music:\nN/A"
            )

        def fake_run(command, **kwargs):
            calls.append(command_prompt(command))
            generated = output(weak_detail=len(calls) == 1)
            return SimpleNamespace(returncode=0, stdout=f"<H3_PROMPT>{generated}</H3_PROMPT>", stderr="")

        with (
            mock.patch.object(MODULE, "_resolve_enhance_model", return_value="model.gguf"),
            mock.patch.object(MODULE, "_find_llama_cli", return_value="llama-cli"),
            mock.patch.object(MODULE, "_find_llama_completion", return_value="llama-cli"),
            mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run),
        ):
            result = MODULE.enhance_project({
                "mode": "REF2VA", "requested_duration": 5,
                "shots": [
                    {"duration": 2.5, "visual_action": "A person walks through a corridor."},
                    {"duration": 2.5, "visual_action": "The person stops near a door."},
                ],
                "references": [{"type": "picture", "role": "reference"}],
            }, "local:model.gguf")
        self.assertEqual(len(calls), 1)
        self.assertEqual(MODULE._prompt_shot_numbers(result["enhanced_prompt"]), [2])

    def test_shot_parser_ignores_inline_reference_to_existing_shot(self):
        prompt = (
            "integrated_multimodal_description: [Shot 1] Picture 1 comes from [Shot 1] and anchors the opening. "
            "[Shot 2] At 00:03.000, the camera cuts closer.\n\n"
            "overall_soundscape: N/A\n\nnon_diegetic_music: N/A"
        )
        self.assertEqual(MODULE._prompt_shot_numbers(prompt), [1, 2])

    def test_shot_parser_still_detects_duplicate_line_header(self):
        prompt = (
            "integrated_multimodal_description: [Shot 1] First block.\n"
            "[Shot 1] Duplicate block.\n\n"
            "overall_soundscape: N/A\n\nnon_diegetic_music: N/A"
        )
        self.assertEqual(MODULE._prompt_shot_numbers(prompt), [1, 1])

    def test_ref_structure_validator_rejects_source_label_leak_and_marker_drift(self):
        compiled = self.compile({
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "@hero walks through a corridor."}],
            "references": [{
                "type": "picture", "role": "subject_identity", "alias": "hero",
                "description": "A person in a red coat.",
            }],
        })
        plan = MODULE._reference_model(compiled["project"])["label_plan"]
        invalid = (
            "subject_definitions:\n<Subject 1> is derived from <Picture 1>.\n\n"
            "summary:\n[reference generation] Use <Picture 1>.\n\n"
            "retention_analysis:\n<Subject 1>: weak_reference - retain it.\n\n"
            "detailed_description:\n[Shot 1] <Picture 1> walks.\n\n"
            "overall_soundscape:\nFootsteps.\n\nnon_diegetic_music:\nN/A"
        )
        issues = MODULE._ref_prompt_structure_issues(invalid, plan)
        self.assertTrue(any("fully_preserved" in issue for issue in issues))
        self.assertTrue(any("source-only" in issue for issue in issues))
        self.assertTrue(any("summary must mention" in issue for issue in issues))
        self.assertTrue(any("detailed_description must apply" in issue for issue in issues))

    def test_ref_structure_validator_requires_subject_provenance(self):
        compiled = self.compile({
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "@hero walks."}],
            "references": [{"type": "picture", "role": "subject_identity", "alias": "hero"}],
        })
        plan = MODULE._reference_model(compiled["project"])["label_plan"]
        prompt = (
            "subject_definitions:\n<Subject 1> is a person in a red coat.\n\n"
            "summary:\n[reference generation] Use <Subject 1>.\n\n"
            "retention_analysis:\n<Subject 1>: fully_preserved - preserve the visible identity.\n\n"
            "detailed_description:\n[Shot 1] <Subject 1> walks.\n\n"
            "overall_soundscape:\nFootsteps.\n\nnon_diegetic_music:\nN/A"
        )
        issues = MODULE._ref_prompt_structure_issues(prompt, plan)
        self.assertTrue(any("must cite its source asset <Picture 1>" in issue for issue in issues))

    def test_ref_structure_validator_requires_subject_in_every_declared_shot(self):
        compiled = self.compile({
            "mode": "REF2VA",
            "shots": [
                {"duration": 2.5, "visual_action": "@hero waits."},
                {"duration": 2.5, "visual_action": "@hero watches the door."},
            ],
            "references": [{
                "type": "picture", "role": "subject_identity", "alias": "hero",
                "strength": "normal",
            }],
        })
        plan = MODULE._reference_model(compiled["project"])["label_plan"]
        prompt = (
            "subject_definitions:\n<Subject 1> is a person derived from <Picture 1>.\n\n"
            "summary:\n[reference generation] <Subject 1> waits and watches the door.\n\n"
            "retention_analysis:\n<Subject 1>: partially_preserved - preserve core identity.\n\n"
            "detailed_description:\n[Shot 1] <Subject 1> waits.\n"
            "[Shot 2] At 00:02.500, the camera shows the door.\n\n"
            "overall_soundscape:\nQuiet room tone.\n\nnon_diegetic_music:\nN/A"
        )
        issues = MODULE._ref_prompt_structure_issues(prompt, plan)
        self.assertTrue(any(
            "<Subject 1> is declared visible in [Shot 2]" in issue for issue in issues
        ))

    def test_ref_normalizer_restores_the_required_six_section_order(self):
        shuffled = (
            "summary:\n[reference generation] Use <Subject 1>.\n\n"
            "subject_definitions:\n<Subject 1> is derived from <Picture 1>.\n\n"
            "detailed_description:\n[Shot 1] <Subject 1> walks.\n\n"
            "retention_analysis:\n<Subject 1>: weak_reference - retain selected traits.\n\n"
            "non_diegetic_music:\nN/A\n\noverall_soundscape:\nFootsteps."
        )
        normalized = MODULE._normalize_ref_enhanced_prompt(shuffled)
        positions = [normalized.index(f"{field}:") for field in MODULE.REF_PROMPT_FIELDS]
        self.assertEqual(positions, sorted(positions))

    def test_ref_normalizer_locks_task_prefix_and_recovers_single_shot(self):
        prompt = (
            "subject_definitions:\n<Subject 1> is derived from <Picture 1>.\n\n"
            "summary:\n[invalid task] Use <Subject 1>.\n\n"
            "retention_analysis:\n<Subject 1>: weak_reference - broad similarity only.\n\n"
            "detailed_description:\nA clean digital-video style. The subject walks forward.\n\n"
            "overall_soundscape:\nFootsteps.\n\nnon_diegetic_music:\nN/A"
        )
        normalized = MODULE._normalize_ref_enhanced_prompt(
            prompt, ["reference generation"], [1],
        )
        self.assertIn("summary:\n[reference generation] Use <Subject 1>.", normalized)
        self.assertIn("A clean digital-video style.\n[Shot 1] The subject walks forward.", normalized)

    def test_ref_normalizer_does_not_confuse_inline_shot_reference_with_header(self):
        prompt = (
            "subject_definitions:\n<Subject 1> is derived from <Picture 1>.\n\n"
            "summary:\n[reference generation] Use <Subject 1>.\n\n"
            "retention_analysis:\n<Subject 1>: weak_reference - broad similarity only.\n\n"
            "detailed_description:\nA clean digital-video style. The source metadata mentions [Shot 1], "
            "and <Subject 1> walks forward.\n\n"
            "overall_soundscape:\nFootsteps.\n\nnon_diegetic_music:\nN/A"
        )
        normalized = MODULE._normalize_ref_enhanced_prompt(
            prompt, ["reference generation"], [1],
        )
        self.assertEqual(MODULE._prompt_shot_numbers(normalized), [1])

    def test_ref_normalizer_recovers_first_header_when_later_headers_exist(self):
        prompt = (
            "subject_definitions:\n<Subject 1> is derived from <Picture 1>.\n\n"
            "summary:\n[reference generation] Use <Subject 1>.\n\n"
            "retention_analysis:\n<Subject 1>: weak_reference - broad similarity only.\n\n"
            "detailed_description:\nA clean digital-video style. <Subject 1> walks in the opening shot.\n"
            "[Shot 2] At 00:02.500, cut to a new shot. <Subject 1> stops.\n\n"
            "overall_soundscape:\nFootsteps.\n\nnon_diegetic_music:\nN/A"
        )
        normalized = MODULE._normalize_ref_enhanced_prompt(
            prompt, ["reference generation"], [1, 2],
        )
        self.assertEqual(MODULE._prompt_shot_numbers(normalized), [1, 2])
        self.assertIn("A clean digital-video style.\n[Shot 1]", normalized)

    def test_ref_semantic_validator_rejects_inferred_demographics_and_empty_action_audio(self):
        prompt = (
            "subject_definitions:\n<Subject 1> is a young Japanese woman.\n\n"
            "summary:\n[reference generation] Use <Subject 1>.\n\n"
            "retention_analysis:\n<Subject 1>: weak_reference - retain selected traits.\n\n"
            "detailed_description:\n[Shot 1] <Subject 1> walks.\n\n"
            "overall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A"
        )
        project = {"user_request": "", "constraints": "", "verbatim_content": "", "references": [], "shots": [{
            "visual_action": "A person walks.", "dialogue": "", "visible_text": "", "diegetic_sound": "",
        }], "overall_soundscape": "", "non_diegetic_music": ""}
        issues = MODULE._ref_prompt_semantic_issues(prompt, project, "A person walks.")
        self.assertTrue(any("demographic" in issue for issue in issues))
        self.assertTrue(any("overall_soundscape" in issue for issue in issues))

    def test_ref_semantic_validator_enforces_weak_role_scope(self):
        prompt = (
            "subject_definitions:\n"
            "<Subject 1> is a woman with blonde hair and a white bikini derived from <Picture 1>.\n\n"
            "summary:\n[reference generation] The target uses <Subject 1> for character identity and styling guidance.\n\n"
            "retention_analysis:\n"
            "<Subject 1>: weak_reference - retain the blonde hair, blue eyes, white bikini, white sandals, and wet skin.\n\n"
            "detailed_description:\n"
            "The target uses an anime illustration style with a cool color palette.\n"
            "[Shot 1] <Subject 1> walks through a hotel corridor while ocean waves remain visible behind her.\n\n"
            "overall_soundscape:\nSoft footsteps and indoor room tone.\n\nnon_diegetic_music:\nN/A"
        )
        project = {
            "user_request": "", "constraints": "", "verbatim_content": "",
            "shots": [{
                "visual_action": "A woman walks through a hotel corridor.", "dialogue": "",
                "visible_text": "", "diegetic_sound": "",
            }],
            "references": [{
                "id": "ref-1", "type": "picture", "role": "reference", "alias": "",
                "description": "An anime illustration of a blonde woman on a beach beside ocean waves.",
                "duration": 0, "image_filename": "", "image_subfolder": "", "image_type": "input",
            }],
            "overall_soundscape": "", "non_diegetic_music": "",
        }
        issues = MODULE._ref_prompt_semantic_issues(
            prompt, project, "A woman walks through a hotel corridor.",
        )
        self.assertTrue(any("exhaustive or identity-preserving" in issue for issue in issues))
        self.assertTrue(any("weak appearance guidance only" in issue for issue in issues))
        self.assertTrue(any("does not transfer source environment" in issue for issue in issues))
        self.assertTrue(any("does not transfer source style" in issue for issue in issues))

    def test_ref_subject_identity_allows_independent_target_style_declaration(self):
        prompt = (
            "subject_definitions:\n"
            "<Subject 1> is a woman in a red coat derived from <Picture 1>.\n"
            "<Subject 2> is a man in a blue jacket derived from <Picture 2>.\n\n"
            "summary:\n[reference generation] The target uses <Subject 1> and <Subject 2> for subject identity.\n\n"
            "retention_analysis:\n"
            "<Subject 1>: fully_preserved - preserve the visible identity and clothing.\n"
            "<Subject 2>: fully_preserved - preserve the visible identity and clothing.\n\n"
            "detailed_description:\n"
            "The target video uses a cinematic presentation with natural lighting.\n"
            "[Shot 1] <Subject 1> and <Subject 2> walk through a hotel corridor.\n\n"
            "overall_soundscape:\nSoft footsteps and indoor room tone.\n\n"
            "non_diegetic_music:\nN/A"
        )
        project = {
            "user_request": "", "constraints": "", "verbatim_content": "",
            "shots": [{
                "visual_action": "Two people walk through a hotel corridor.", "dialogue": "",
                "visible_text": "", "diegetic_sound": "",
            }],
            "references": [
                {"id": "ref-1", "type": "picture", "role": "subject_identity", "alias": "",
                 "description": "A cinematic portrait with natural lighting.", "duration": 0},
                {"id": "ref-2", "type": "picture", "role": "subject_identity", "alias": "",
                 "description": "A cinematic illustration with natural lighting.", "duration": 0},
            ],
            "overall_soundscape": "", "non_diegetic_music": "",
        }
        issues = MODULE._ref_prompt_semantic_issues(
            prompt, project, "Two people walk through a hotel corridor.",
        )
        self.assertFalse(any("does not transfer source style" in issue for issue in issues))

    def test_ref_normal_subject_identity_rejects_explicit_source_style_transfer(self):
        prompt = (
            "subject_definitions:\n<Subject 1> is a woman derived from <Picture 1>.\n\n"
            "summary:\n[reference generation] The target uses <Subject 1> for subject identity.\n\n"
            "retention_analysis:\n"
            "<Subject 1>: fully_preserved - preserve the visible identity and clothing.\n\n"
            "detailed_description:\n"
            "The target video uses <Subject 1>'s cinematic style and natural lighting.\n"
            "[Shot 1] <Subject 1> walks through a hotel corridor.\n\n"
            "overall_soundscape:\nSoft footsteps and indoor room tone.\n\n"
            "non_diegetic_music:\nN/A"
        )
        project = {
            "user_request": "", "constraints": "", "verbatim_content": "",
            "shots": [{
                "visual_action": "A woman walks through a hotel corridor.", "dialogue": "",
                "visible_text": "", "diegetic_sound": "",
            }],
            "references": [{
                "id": "ref-1", "type": "picture", "role": "subject_identity", "alias": "",
                "strength": "normal",
                "description": "A cinematic portrait with natural lighting.", "duration": 0,
            }],
            "overall_soundscape": "", "non_diegetic_music": "",
        }
        issues = MODULE._ref_prompt_semantic_issues(
            prompt, project, "A woman walks through a hotel corridor.",
        )
        self.assertTrue(any("does not transfer source style" in issue for issue in issues))

    def test_ref_strong_subject_identity_allows_source_style_transfer(self):
        prompt = (
            "subject_definitions:\n<Subject 1> is a cel-shaded woman derived from <Picture 1>.\n\n"
            "summary:\n[reference generation] The target uses <Subject 1>.\n\n"
            "retention_analysis:\n"
            "<Subject 1>: fully_preserved - preserve identity and source rendering style.\n\n"
            "detailed_description:\n"
            "[Shot 1] <Subject 1> retains <Subject 1>'s cel-shaded rendering style while walking through a corridor.\n\n"
            "overall_soundscape:\nFootsteps.\n\nnon_diegetic_music:\nN/A"
        )
        project = {
            "user_request": "", "constraints": "", "verbatim_content": "",
            "shots": [{"visual_action": "A woman walks through a corridor."}],
            "references": [{
                "id": "ref-1", "type": "picture", "role": "subject_identity", "alias": "",
                "strength": "strong", "description": "A cel-shaded illustration.", "duration": 0,
            }],
        }
        issues = MODULE._ref_prompt_semantic_issues(
            prompt, project, "A woman walks through a corridor.",
        )
        self.assertFalse(any("does not transfer source style" in issue for issue in issues))

    def test_llm_system_prompt_includes_base_guide_rules(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [{"duration": 5, "visual_action": "Two people speak while walking."}],
        })
        llm_prompt = result["llm_prompt"]
        self.assertIn("Use a cut only when the configured next shot reveals new subject", llm_prompt)
        self.assertIn("Every later shot starts at its supplied timestamp", llm_prompt)
        self.assertIn("Place a synchronized physical sound beside its visible cause", llm_prompt)
        self.assertIn("Infer only concise ambience and physical sounds directly implied", llm_prompt)
        self.assertIn("BGM, soundtrack, or score", llm_prompt)
        self.assertIn("otherwise output N/A", llm_prompt)
        self.assertIn("without shot labels, timestamps", llm_prompt)
        self.assertIn("instrumentation, tempo, rhythm, and dynamics", llm_prompt)
        self.assertIn("Do not invent identities, demographics, backstory", llm_prompt)
        self.assertIn("show only the intermediate motion needed to make it physically legible", llm_prompt)
        self.assertIn("Return the final prompt only", llm_prompt)
        self.assertIn("PRIORITY\n1. Explicit user actions", llm_prompt)
        self.assertIn("Omit unsupported details instead of guessing", llm_prompt)
        self.assertIn("visible mouth movement synchronizes with the line", llm_prompt)
        self.assertIn("ends when the line ends", llm_prompt)
        self.assertIn("off-screen voiceover", llm_prompt)
        self.assertIn("Give speech a readable beat", llm_prompt)
        self.assertIn("unless the user explicitly requests simultaneity", llm_prompt)
        self.assertIn("SHOT_PLAN visual_action is the unified source", llm_prompt)
        self.assertIn("Write spoken dialogue as", llm_prompt)
        self.assertIn("never translate, paraphrase, duplicate, or add words", llm_prompt)
        self.assertIn("Visible text uses exact double-quoted characters", llm_prompt)
        self.assertIn("must be correct in the final output", llm_prompt)
        self.assertIn("Picture anchors are in-shot states, never cuts", llm_prompt)
        self.assertIn("Only SHOT_PLAN boundaries create cuts or transitions", llm_prompt)

    def test_ref_llm_system_prompt_includes_reference_guide_rules(self):
        result = self.compile({
            "version": 16,
            "mode": "REF2VA",
            "shots": [{"duration": 5, "visual_action": "@hero walks."}],
            "references": [{"type": "picture", "role": "subject_identity", "strength": "normal", "alias": "hero"}],
        })
        llm_prompt = result["llm_prompt"]
        self.assertIn("Do not define a standalone Picture unless it is a configured frame anchor", llm_prompt)
        self.assertIn("number each label type independently", llm_prompt)
        self.assertIn("<Subject N> (Sx)", llm_prompt)
        self.assertIn("Define each image-derived Subject in one line", llm_prompt)
        self.assertIn("RETENTION_LINE_PLAN:", llm_prompt)
        self.assertIn("<Subject 1> (appears in [Shot 1]): partially_preserved -", llm_prompt)
        self.assertNotIn("normal = partially_preserved", llm_prompt)
        self.assertIn("must be correct in the final output", llm_prompt)
        self.assertIn("ACTIVE MODE: REF2VA FULL-REFERENCE", llm_prompt)
        self.assertIn("FINAL MODE LOCK — REF2VA", llm_prompt)
        self.assertIn("never use `integrated_multimodal_description:`", llm_prompt)
        self.assertIn("Picture anchors are in-shot states, never cuts", llm_prompt)

    def test_base_mode_system_prompts_include_guide_transition_sequences(self):
        cases = {
            "T2VA": "concise chronological action",
            "I2VA": "actual opening state through action onset, necessary physical development, and a stable result",
            "FL2VA": "Picture 1 is the complete opening frame and Picture 2 is the complete frame reached only at the effective end",
            "L2VA": "Infer one plausible preceding state from the user's request",
        }
        references = {
            "T2VA": [],
            "I2VA": [{"type": "picture", "role": "first_frame"}],
            "FL2VA": [
                {"type": "picture", "role": "first_frame"},
                {"type": "picture", "role": "last_frame"},
            ],
            "L2VA": [{"type": "picture", "role": "last_frame"}],
        }
        for mode, phrase in cases.items():
            with self.subTest(mode=mode):
                result = self.compile({
                    "mode": mode,
                    "shots": [{"duration": 5, "visual_action": "A subject moves."}],
                    "references": references[mode],
                })
                self.assertIn(phrase, result["llm_prompt"])

    def test_i2va_system_prompt_locks_first_frame_and_physical_continuity(self):
        result = self.compile({
            "mode": "I2VA",
            "shots": [{"duration": 5, "visual_action": "The subject performs one action."}],
            "references": [{"type": "picture", "role": "first_frame"}],
        })
        llm_prompt = result["llm_prompt"]
        self.assertIn("Picture 1 is the complete literal frame at 0.00 seconds", llm_prompt)
        self.assertIn("observable visual medium and rendering style as part of the opening anchor", llm_prompt)
        self.assertIn("visible clothing and construction", llm_prompt)
        self.assertIn("pose, support and contact", llm_prompt)
        self.assertIn("Preserve every visible surface, foreground object, support, obstacle", llm_prompt)
        self.assertIn("Do not replace the evidenced setting with a generic room", llm_prompt)
        self.assertIn("actual opening state through action onset, necessary physical development, and a stable result", llm_prompt)
        self.assertIn("An initially absent object may enter the frame only through a physically plausible visible path", llm_prompt)
        self.assertIn("Never invent a target or aiming direction", llm_prompt)
        self.assertIn("The framing must contain the complete action path", llm_prompt)
        self.assertIn("use one slight pullback, tilt, pan, or tracking adjustment", llm_prompt)
        self.assertIn("no more than three connected beats", llm_prompt)
        self.assertIn("Do not add unrequested outcomes, injuries, reactions", llm_prompt)
        self.assertIn("110-160 English words", llm_prompt)
        self.assertIn("150-210 when contact objects or spatial staging require it", llm_prompt)

    def test_reference_analysis_captures_action_relevant_construction(self):
        prompt = MODULE._reference_analysis_prompt("first_frame")
        self.assertIn("fasteners, seams, layers, openings", prompt)
        self.assertIn("POSE_SUPPORT_CONTACT:", prompt)
        self.assertIn("ACTION_RELEVANT_OBJECTS:", prompt)
        self.assertIn("foreground obstacles", prompt)

    def test_shots_are_fitted_to_fixed_timeline_duration(self):
        result = self.compile({
            "mode": "T2VA",
            "requested_duration": 6,
            "shots": [
                {"duration": 5, "visual_action": "First action."},
                {"duration": 5, "visual_action": "Second action."},
            ],
        })
        durations = [shot["duration"] for shot in result["project"]["shots"]]
        self.assertAlmostEqual(sum(durations), 6.0)
        self.assertAlmostEqual(durations[0], 3.0)
        self.assertAlmostEqual(durations[1], 3.0)
        self.assertFalse(any("Shot durations total" in error for error in result["errors"]))

    def test_legacy_camera_controls_migrate_into_unified_visual_action(self):
        result = self.compile({
            "version": 14,
            "mode": "T2VA",
            "shots": [{
                "duration": 5,
                "visual_action": "A runner crosses the finish line.",
                "camera_framing": "Cowboy Shot",
                "camera_angle": "Low Angle Shot",
                "camera_motion": "Tracking Shot",
            }],
        })
        prompt = result["video_prompt"]
        shot = result["project"]["shots"][0]
        self.assertNotIn("camera_framing", shot)
        self.assertNotIn("camera_angle", shot)
        self.assertNotIn("camera_motion", shot)
        self.assertIn("The composition uses a cowboy shot", shot["visual_action"])
        self.assertIn("The camera uses a low angle", shot["visual_action"])
        self.assertIn("The camera tracks the moving subject", shot["visual_action"])
        self.assertIn("CAMERA_POLICY:", prompt)
        self.assertIn("choose framing that contains the largest required visible action and final state", prompt)
        self.assertIn("otherwise use one motivated reframe", prompt)
        self.assertNotIn("camera_framing:", prompt)
        self.assertNotIn("camera_angle:", prompt)
        self.assertNotIn("camera_motion:", prompt)

    def test_empty_shots_use_grammatical_continuity_text(self):
        result = self.compile({
            "mode": "FL2VA",
            "requested_duration": 5,
            "shots": [
                {"duration": 2.5},
                {"duration": 1.25},
                {"duration": 1.25},
            ],
            "references": [
                {"type": "picture", "role": "first_frame"},
                {"type": "picture", "role": "last_frame"},
            ],
        })
        description = result["video_prompt"]
        self.assertIn("[Shot 2]\ntime_range_seconds: 2.583-3.875", description)
        self.assertIn("[Shot 3]\ntime_range_seconds: 3.875-5.167", description)
        self.assertNotIn("transition:", description)

    def test_configured_shot_boundaries_are_defined_by_camera_policy(self):
        result = self.compile({
            "mode": "T2VA",
            "shots": [
                {"duration": 2.5, "visual_action": "A woman walks along the beach."},
                {"duration": 2.5, "visual_action": "The woman turns toward the camera."},
            ],
        })
        prompt = result["video_prompt"]
        self.assertIn("visual_action: The woman turns toward the camera.", prompt)
        self.assertIn("each configured shot after Shot 1 is an ordinary cut", prompt)
        self.assertNotIn("transition: cut", prompt)


if __name__ == "__main__":
    unittest.main()
