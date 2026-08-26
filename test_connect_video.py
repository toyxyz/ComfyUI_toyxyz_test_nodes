import importlib.util
import sys
import types
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


MODULE_PATH = Path(__file__).parent / "nodes" / "connect_video.py"
SPEC = importlib.util.spec_from_file_location("connect_video", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeVideo:
    def __init__(self, values, fps=24, audio=None, height=2, width=2):
        self.images = torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1, 1).repeat(1, height, width, 3)
        self.fps = Fraction(fps)
        self.audio = audio

    def get_frame_rate(self):
        return self.fps

    def get_components(self):
        return SimpleNamespace(images=self.images, audio=self.audio, frame_rate=self.fps)

    def get_bit_depth(self):
        return 8

    def get_color_space(self):
        return "sRGB"


class FakeVideoFromComponents:
    def __init__(self, components, bit_depth=8, color_space="sRGB"):
        self.components = components
        self.bit_depth = bit_depth
        self.color_space = color_space


class ConnectVideoTests(unittest.TestCase):
    def setUp(self):
        comfy_api = types.ModuleType("comfy_api")
        comfy_latest = types.ModuleType("comfy_api.latest")
        comfy_latest.InputImpl = SimpleNamespace(VideoFromComponents=FakeVideoFromComponents)
        comfy_latest.Types = SimpleNamespace(VideoComponents=lambda **kwargs: SimpleNamespace(**kwargs))
        self.modules = mock.patch.dict(sys.modules, {
            "comfy_api": comfy_api,
            "comfy_api.latest": comfy_latest,
        })
        self.modules.start()

    def tearDown(self):
        self.modules.stop()

    def test_schema_has_two_video_inputs_and_one_video_output(self):
        schema = MODULE.ConnectVideo.INPUT_TYPES()
        self.assertEqual(schema["required"]["video_1"][0], "VIDEO")
        self.assertEqual(schema["required"]["video_2"][0], "VIDEO")
        self.assertEqual(schema["required"]["smooth_transition"][0], "INT")
        self.assertEqual(schema["required"]["smooth_transition"][1]["default"], 0)
        self.assertEqual(MODULE.ConnectVideo.RETURN_TYPES, ("VIDEO",))

    def test_connects_frames_and_audio_in_order(self):
        audio_1 = {"waveform": torch.ones((1, 1, 20)), "sample_rate": 40}
        audio_2 = {"waveform": torch.full((1, 1, 20), 2.0), "sample_rate": 40}
        output, = MODULE.ConnectVideo().connect(
            FakeVideo([1, 2], fps=4, audio=audio_1),
            FakeVideo([3, 4], fps=4, audio=audio_2),
        )
        self.assertEqual(output.components.images[:, 0, 0, 0].tolist(), [1, 2, 3, 4])
        self.assertEqual(output.components.audio["waveform"].shape[-1], 40)
        self.assertTrue(torch.all(output.components.audio["waveform"][..., :20] == 1))
        self.assertTrue(torch.all(output.components.audio["waveform"][..., 20:] == 2))

    def test_missing_audio_is_filled_with_silence(self):
        audio_2 = {"waveform": torch.ones((1, 1, 20)), "sample_rate": 40}
        output, = MODULE.ConnectVideo().connect(
            FakeVideo([1, 2], fps=4),
            FakeVideo([3, 4], fps=4, audio=audio_2),
        )
        self.assertTrue(torch.all(output.components.audio["waveform"][..., :20] == 0))
        self.assertTrue(torch.all(output.components.audio["waveform"][..., 20:] == 1))

    def test_rejects_mismatched_fps(self):
        with self.assertRaisesRegex(ValueError, "FPS must match"):
            MODULE.ConnectVideo().connect(FakeVideo([1], fps=24), FakeVideo([2], fps=30))

    def test_rejects_mismatched_frame_dimensions(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            MODULE.ConnectVideo().connect(FakeVideo([1], width=2), FakeVideo([2], width=3))

    def test_smooth_transition_crossfades_video_and_audio(self):
        audio_1 = {"waveform": torch.ones((1, 1, 6)), "sample_rate": 4}
        audio_2 = {"waveform": torch.full((1, 1, 6), 3.0), "sample_rate": 4}
        output, = MODULE.ConnectVideo().connect(
            FakeVideo([1, 1, 1], fps=2, audio=audio_1),
            FakeVideo([3, 3, 3], fps=2, audio=audio_2),
            smooth_transition=2,
        )
        self.assertEqual(output.components.images.shape[0], 4)
        self.assertEqual(output.components.images[:, 0, 0, 0].tolist(), [1, 1, 3, 3])
        waveform = output.components.audio["waveform"]
        self.assertEqual(waveform.shape[-1], 8)
        self.assertEqual(waveform[0, 0, 2].item(), 1.0)
        self.assertEqual(waveform[0, 0, 5].item(), 3.0)

    def test_rejects_transition_longer_than_an_input(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            MODULE.ConnectVideo().connect(FakeVideo([1, 2]), FakeVideo([3]), smooth_transition=2)


if __name__ == "__main__":
    unittest.main()
