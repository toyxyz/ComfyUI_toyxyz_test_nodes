import torch

def _frame_selection(total_frames, frame_count):
    total_frames = max(0, int(total_frames))
    requested = int(frame_count)
    if total_frames == 0:
        return 0, 0
    if requested == 0 or abs(requested) >= total_frames:
        return 0, total_frames
    kept = abs(requested)
    return (0 if requested > 0 else total_frames - kept), kept


def _inverted_frame_selection(total_frames, frame_count):
    total_frames = max(0, int(total_frames))
    requested = int(frame_count)
    if total_frames == 0:
        return 0, 0
    if requested == 0:
        return 0, total_frames

    excluded = abs(requested)
    if excluded >= total_frames:
        raise ValueError(
            "Invert cannot exclude every frame. Use a frame_count smaller than the video frame count."
        )
    if requested > 0:
        return excluded, total_frames - excluded
    return 0, total_frames - excluded

def _blank_audio():
    return {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": 44100}


class CutVideo:
    """Trim a VIDEO and expose its aligned image and audio components."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (
                    "VIDEO",
                    {"tooltip": "Source VIDEO, including its embedded audio and frame rate."},
                ),
                "frame_count": (
                    "INT",
                    {
                        "default": 124,
                        "min": -999999,
                        "max": 999999,
                        "step": 1,
                        "display": "number",
                        "tooltip": (
                            "Positive: keep that many frames from the beginning. "
                            "Negative: keep that many frames from the end (-1 keeps the final frame). "
                            "Zero: keep the complete connected media."
                        ),
                    },
                ),
                "invert": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "enabled",
                        "label_off": "disabled",
                        "tooltip": (
                            "Disabled: positive keeps the first N frames and negative keeps the last N. "
                            "Enabled: positive excludes the first N frames and negative excludes the last N."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("video", "images", "audio", "fps")
    OUTPUT_TOOLTIPS = (
        "Trimmed VIDEO with synchronized audio.",
        "Trimmed IMAGE sequence.",
        "Audio trimmed to the selected frame time range.",
        "Original input VIDEO frame rate.",
    )
    FUNCTION = "cut"
    CATEGORY = "video/transform"
    DESCRIPTION = (
        "Trim a VIDEO with one signed frame count. Positive values keep the beginning; "
        "negative values keep the end. Embedded audio stays aligned with the selected frames."
    )

    def cut(self, video, frame_count, invert=False):
        total_frames = int(video.get_frame_count())
        if total_frames <= 0:
            raise ValueError("The input video must contain at least one frame.")

        fps = float(video.get_frame_rate())
        if fps <= 0:
            raise ValueError("The input video must have a positive frame rate.")

        selection = _inverted_frame_selection if invert else _frame_selection
        start_frame, kept_frames = selection(total_frames, frame_count)
        if start_frame == 0 and kept_frames == total_frames:
            output_video = video
        else:
            output_video = video.as_trimmed(
                start_frame / fps,
                kept_frames / fps,
                strict_duration=False,
            )
            if output_video is None:
                raise ValueError("The requested video range could not be trimmed.")

        components = output_video.get_components()
        output_audio = components.audio if components.audio is not None else _blank_audio()
        return output_video, components.images, output_audio, fps
