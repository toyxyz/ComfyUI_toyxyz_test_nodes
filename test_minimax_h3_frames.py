import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parent / "nodes" / "minimax_h3_frames.py"
SPEC = importlib.util.spec_from_file_location("toyxyz_minimax_h3_frames_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MiniMaxH3FramesTests(unittest.TestCase):
    def test_node_accepts_prompter_frames_bundle(self):
        inputs = MODULE.MiniMaxH3AddGuideFrames.INPUT_TYPES()["required"]
        self.assertEqual(inputs["frames"], ("MINIMAX_H3_FRAMES",))
        self.assertEqual(MODULE.MiniMaxH3AddGuideFrames.RETURN_TYPES, ("CONDITIONING",))
        self.assertEqual(MODULE.MiniMaxH3AddGuideFrames.CATEGORY, "model/conditioning/minimax")

    def test_bundle_entries_are_sorted_by_frame_and_keep_tie_order(self):
        first = object()
        second = object()
        third = object()
        entries = MODULE.MiniMaxH3AddGuideFrames._validate_frames_bundle({
            "type": "minimax_h3_frames",
            "frames": [
                {"image": third, "frame_idx": 80},
                {"image": first, "frame_idx": 12},
                {"image": second, "frame_idx": 12},
            ],
        })
        self.assertEqual(entries, [(12, 2, first), (12, 3, second), (80, 1, third)])

    def test_empty_or_foreign_bundle_is_rejected(self):
        for frames in ({}, {"type": "other", "frames": []}, {"type": "minimax_h3_frames", "frames": []}):
            with self.subTest(frames=frames):
                with self.assertRaises(ValueError):
                    MODULE.MiniMaxH3AddGuideFrames._validate_frames_bundle(frames)


if __name__ == "__main__":
    unittest.main()
