"""Batch image-frame guides for the native MiniMax H3 conditioning pipeline."""


MINIMAX_H3_FRAMES_TYPE = "MINIMAX_H3_FRAMES"


class MiniMaxH3AddGuideFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "frames": (MINIMAX_H3_FRAMES_TYPE,),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "add_guides"
    CATEGORY = "model/conditioning/minimax"
    DESCRIPTION = (
        "Adds every image/frame_idx pair from a Minimax-H3-prompter frames bundle "
        "as native MiniMax H3 guide keyframes in one node."
    )

    @staticmethod
    def _validate_frames_bundle(frames):
        if not isinstance(frames, dict) or frames.get("type") != "minimax_h3_frames":
            raise ValueError("Add Guide for MiniMax H3 frames expects a frames bundle from Minimax-H3-prompter")
        entries = frames.get("frames")
        if not isinstance(entries, list) or not entries:
            raise ValueError("The MiniMax H3 frames bundle contains no uploaded Frame images")
        normalized = []
        for position, entry in enumerate(entries, 1):
            if not isinstance(entry, dict) or entry.get("image") is None:
                raise ValueError(f"Frame guide {position} has no image")
            try:
                frame_idx = int(entry.get("frame_idx"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Frame guide {position} has an invalid frame_idx") from exc
            normalized.append((frame_idx, position, entry["image"]))
        return sorted(normalized, key=lambda item: (item[0], item[1]))

    def add_guides(self, positive, latent, vae, frames):
        import node_helpers
        from comfy.ldm.minimax.model import FRAME_PER_TOKEN
        from comfy_extras.nodes_minimax_h3 import _resize

        samples = latent.get("samples") if isinstance(latent, dict) else None
        if (
            samples is None
            or not getattr(samples, "is_nested", False)
            or len(samples.tensors) != 2
            or samples.tensors[0].ndim != 5
            or samples.tensors[0].shape[1] != 24
        ):
            raise ValueError("Add Guide for MiniMax H3 frames expects a MiniMax H3 AV latent")

        video = samples.tensors[0]
        height = video.shape[3] * 16
        width = video.shape[4] * 16
        frame_count = sum(FRAME_PER_TOKEN[index % 5] for index in range(video.shape[2]))
        keyframes = list(positive[0][1].get("minimax_keyframes", []))

        for frame_idx, _position, image in self._validate_frames_bundle(frames):
            resolved_frame_index = frame_idx if frame_idx >= 0 else frame_count + frame_idx
            if resolved_frame_index < 0 or resolved_frame_index >= frame_count:
                raise ValueError(
                    f"frame_idx {frame_idx} is outside the video's {frame_count} frames"
                )
            resized = _resize(image[:1], width, height, "center")
            keyframes.append({
                "resolved_frame_index": resolved_frame_index,
                "latent": vae.encode(resized),
            })

        positive = node_helpers.conditioning_set_values(
            positive, {"minimax_keyframes": keyframes}
        )
        return (positive,)
