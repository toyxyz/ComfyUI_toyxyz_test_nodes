from fractions import Fraction

import torch


def _audio_for_video(audio, frame_count, fps, target_sample_rate, target_channels):
    sample_count = int(round((int(frame_count) / float(fps)) * int(target_sample_rate)))
    if audio is None or audio.get("waveform") is None:
        return torch.zeros((1, target_channels, sample_count), dtype=torch.float32)

    waveform = audio["waveform"]
    source_rate = int(audio["sample_rate"])
    if source_rate != target_sample_rate:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, source_rate, target_sample_rate)

    channels = int(waveform.shape[1])
    if channels == 1 and target_channels > 1:
        waveform = waveform.repeat(1, target_channels, 1)
    elif channels != target_channels:
        raise ValueError(
            f"Audio channel counts are incompatible: expected {target_channels}, got {channels}."
        )

    if waveform.shape[-1] < sample_count:
        waveform = torch.nn.functional.pad(waveform, (0, sample_count - waveform.shape[-1]))
    return waveform[..., :sample_count]


class ConnectVideo:
    """Append two VIDEO inputs while preserving synchronized audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_1": ("VIDEO", {"tooltip": "The first VIDEO in the connected result."}),
                "video_2": ("VIDEO", {"tooltip": "The VIDEO appended after video_1."}),
                "smooth_transition": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "step": 1,
                        "display": "number",
                        "tooltip": (
                            "Crossfade length in frames. 0 joins directly. A positive value overlaps "
                            "the end of video_1 with the beginning of video_2 while fading video_1 out."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    OUTPUT_TOOLTIPS = ("One VIDEO containing video_1 followed by video_2, including aligned audio.",)
    FUNCTION = "connect"
    CATEGORY = "video/transform"
    DESCRIPTION = "Connect two compatible VIDEO inputs, with an optional frame-based audiovisual crossfade."

    def connect(self, video_1, video_2, smooth_transition=0):
        from comfy_api.latest import InputImpl, Types

        fps_1 = float(video_1.get_frame_rate())
        fps_2 = float(video_2.get_frame_rate())
        if fps_1 <= 0 or fps_2 <= 0:
            raise ValueError("Both input videos must have a positive frame rate.")
        if abs(fps_1 - fps_2) > 0.001:
            raise ValueError(f"Video FPS must match: video_1={fps_1:g}, video_2={fps_2:g}.")

        components_1 = video_1.get_components()
        components_2 = video_2.get_components()
        images_1 = components_1.images
        images_2 = components_2.images
        if images_1.shape[1:] != images_2.shape[1:]:
            raise ValueError(
                "Video frame dimensions and channel counts must match: "
                f"video_1={tuple(images_1.shape[1:])}, video_2={tuple(images_2.shape[1:])}."
            )

        transition_frames = int(smooth_transition)
        if transition_frames < 0:
            raise ValueError("smooth_transition must be zero or greater.")
        if transition_frames > min(images_1.shape[0], images_2.shape[0]):
            raise ValueError(
                "smooth_transition cannot exceed the frame count of either input video: "
                f"requested={transition_frames}, video_1={images_1.shape[0]}, video_2={images_2.shape[0]}."
            )

        if transition_frames == 0:
            output_images = torch.cat((images_1, images_2), dim=0)
        else:
            video_weights = torch.linspace(
                1.0,
                0.0,
                transition_frames,
                dtype=images_1.dtype,
                device=images_1.device,
            ).reshape(-1, 1, 1, 1)
            overlap = (
                images_1[-transition_frames:] * video_weights
                + images_2[:transition_frames].to(images_1) * (1.0 - video_weights)
            )
            output_images = torch.cat(
                (images_1[:-transition_frames], overlap, images_2[transition_frames:].to(images_1)),
                dim=0,
            )
        audio_1 = components_1.audio
        audio_2 = components_2.audio
        output_audio = None
        if audio_1 is not None or audio_2 is not None:
            available = audio_1 if audio_1 is not None else audio_2
            sample_rate = int(available["sample_rate"])
            channels = max(
                int(audio_1["waveform"].shape[1]) if audio_1 is not None else 1,
                int(audio_2["waveform"].shape[1]) if audio_2 is not None else 1,
            )
            waveform_1 = _audio_for_video(audio_1, images_1.shape[0], fps_1, sample_rate, channels)
            waveform_2 = _audio_for_video(audio_2, images_2.shape[0], fps_2, sample_rate, channels)
            if transition_frames > 0:
                transition_samples = max(1, int(round((transition_frames / fps_1) * sample_rate)))
                transition_samples = min(
                    transition_samples,
                    waveform_1.shape[-1],
                    waveform_2.shape[-1],
                )
                audio_weights = torch.linspace(
                    1.0,
                    0.0,
                    transition_samples,
                    dtype=waveform_1.dtype,
                    device=waveform_1.device,
                ).reshape(1, 1, -1)
                audio_overlap = (
                    waveform_1[..., -transition_samples:] * audio_weights
                    + waveform_2[..., :transition_samples].to(waveform_1) * (1.0 - audio_weights)
                )
                output_waveform = torch.cat(
                    (
                        waveform_1[..., :-transition_samples],
                        audio_overlap,
                        waveform_2[..., transition_samples:].to(waveform_1),
                    ),
                    dim=-1,
                )
            else:
                output_waveform = torch.cat((waveform_1, waveform_2), dim=-1)
            output_audio = {
                "waveform": output_waveform,
                "sample_rate": sample_rate,
            }

        output_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=output_images,
                audio=output_audio,
                frame_rate=Fraction(fps_1).limit_denominator(100000),
            ),
            bit_depth=max(video_1.get_bit_depth(), video_2.get_bit_depth()),
            color_space=video_1.get_color_space(),
        )
        return (output_video,)
