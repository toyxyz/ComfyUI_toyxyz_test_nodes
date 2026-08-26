import importlib.util
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = Path(__file__).parent / "nodes" / "cut_video.py"
SPEC = importlib.util.spec_from_file_location("cut_video", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeVideo:
    def __init__(self, frames=300, fps=30, audio=True):
        self.frames = frames
        self.fps = Fraction(fps)
        self.images = torch.arange(max(frames, 1), dtype=torch.float32).reshape(-1, 1, 1, 1)[:frames]
        self.audio = (
            {"waveform": torch.arange(max(frames * 10, 1), dtype=torch.float32).reshape(1, 1, -1), "sample_rate": fps * 10}
            if audio else None
        )
        self.trim_args = None

    def get_frame_count(self):
        return self.frames

    def get_frame_rate(self):
        return self.fps

    def get_bit_depth(self):
        return 8

    def get_color_space(self):
        return "sRGB"

    def get_components(self):
        return SimpleNamespace(images=self.images, audio=self.audio, frame_rate=self.fps)

    def as_trimmed(self, start, duration, strict_duration=True):
        self.trim_args = (start, duration, strict_duration)
        start_frame = round(start * float(self.fps))
        count = round(duration * float(self.fps))
        trimmed = FakeVideo(count, int(self.fps), audio=self.audio is not None)
        trimmed.images = self.images[start_frame:start_frame + count]
        if self.audio is not None:
            start_sample = round(start * self.audio["sample_rate"])
            end_sample = start_sample + round(duration * self.audio["sample_rate"])
            trimmed.audio = {**self.audio, "waveform": self.audio["waveform"][..., start_sample:end_sample]}
        return trimmed


class CutVideoTests(unittest.TestCase):
    def test_schema_uses_required_video_and_three_outputs(self):
        schema = MODULE.CutVideo.INPUT_TYPES()
        self.assertEqual(schema["required"]["video"][0], "VIDEO")
        self.assertEqual(schema["required"]["frame_count"][0], "INT")
        self.assertEqual(schema["required"]["frame_count"][1]["min"], -999999)
        self.assertEqual(schema["required"]["invert"][0], "BOOLEAN")
        self.assertNotIn("optional", schema)
        self.assertEqual(MODULE.CutVideo.RETURN_TYPES, ("VIDEO", "IMAGE", "AUDIO", "FLOAT"))

    def test_positive_count_trims_video_images_and_embedded_audio(self):
        video = FakeVideo(frames=300, fps=30)
        output_video, output_images, output_audio, fps = MODULE.CutVideo().cut(video, 124)
        self.assertEqual(video.trim_args, (0.0, 124 / 30, False))
        self.assertEqual(output_video.get_frame_count(), 124)
        self.assertEqual(output_images.shape[0], 124)
        self.assertEqual(output_audio["waveform"].shape[-1], 1240)
        self.assertEqual(fps, 30.0)

    def test_negative_video_trim_preserves_its_selected_embedded_audio(self):
        video = FakeVideo(frames=300, fps=30)
        output_video, output_images, output_audio, _fps = MODULE.CutVideo().cut(video, -22)
        self.assertEqual(video.trim_args, (278 / 30, 22 / 30, False))
        self.assertEqual(output_video.get_frame_count(), 22)
        self.assertEqual(output_images[0].item(), 278)
        self.assertEqual(output_audio["waveform"].shape[-1], 220)
        self.assertEqual(output_audio["waveform"][0, 0, 0].item(), 2780)

    def test_zero_keeps_complete_media(self):
        video = FakeVideo(frames=30, fps=30)
        output_video, output_images, output_audio, _fps = MODULE.CutVideo().cut(video, 0)
        self.assertIs(output_video, video)
        self.assertEqual(output_audio["waveform"].shape[-1], 300)
        self.assertEqual(output_images.shape[0], 30)

    def test_video_without_audio_returns_blank_audio(self):
        video = FakeVideo(frames=30, fps=30, audio=False)
        _output_video, _output_images, output_audio, _fps = MODULE.CutVideo().cut(video, 10)
        self.assertEqual(output_audio["waveform"].shape, (1, 1, 1))

    def test_invert_positive_excludes_frames_from_beginning(self):
        video = FakeVideo(frames=100, fps=25)
        output_video, output_images, output_audio, fps = MODULE.CutVideo().cut(video, 22, True)
        self.assertEqual(video.trim_args, (22 / 25, 78 / 25, False))
        self.assertEqual(output_video.get_frame_count(), 78)
        self.assertEqual(output_images[0].item(), 22)
        self.assertEqual(output_audio["waveform"][0, 0, 0].item(), 220)
        self.assertEqual(fps, 25.0)

    def test_invert_negative_excludes_frames_from_end(self):
        video = FakeVideo(frames=100, fps=25)
        output_video, output_images, output_audio, _fps = MODULE.CutVideo().cut(video, -22, True)
        self.assertEqual(video.trim_args, (0.0, 78 / 25, False))
        self.assertEqual(output_video.get_frame_count(), 78)
        self.assertEqual(output_images[-1].item(), 77)
        self.assertEqual(output_audio["waveform"].shape[-1], 780)

    def test_invert_rejects_excluding_every_frame(self):
        video = FakeVideo(frames=22, fps=24)
        with self.assertRaisesRegex(ValueError, "cannot exclude every frame"):
            MODULE.CutVideo().cut(video, 22, True)


if __name__ == "__main__":
    unittest.main()
