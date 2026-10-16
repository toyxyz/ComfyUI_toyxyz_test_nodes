"""Raster layer painting transport for the existing ComfyCouple region chain."""
import base64
import io
import json
import math
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
if __package__:
    from .pose_region_transport import REGION_PAINTER_TYPE, read_pose_regions, merge_source_layers, source_key
else:
    import importlib.util
    from pathlib import Path
    _spec = importlib.util.spec_from_file_location("pose_region_transport", Path(__file__).with_name("pose_region_transport.py"))
    _transport = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_transport)
    REGION_PAINTER_TYPE, read_pose_regions, merge_source_layers, source_key = (
        _transport.REGION_PAINTER_TYPE, _transport.read_pose_regions, _transport.merge_source_layers, _transport.source_key)

MAX_LAYERS = 32
MAX_LAYER_PIXELS = 64 * 1024 * 1024
BACKGROUND_MODES = ("region_and_background", "background_only")


class ComfyCoupleRegionPainter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "Use the text encoder belonging to your generation model."}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "paint_data": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {"region": ("region", {"tooltip": "Append painted regions to an existing region / base / background prompt chain."}),
                         "region_painter": (REGION_PAINTER_TYPE, {"tooltip": "Connect 3D Pose Editor's region painter output. Body masks stay protected; only added painting is editable. Source resolution is followed automatically."})},
        }

    RETURN_TYPES = ("IMAGE", "region")
    RETURN_NAMES = ("images", "region")
    FUNCTION = "paint"
    CATEGORY = "ToyxyzTestNodes"

    def paint(self, clip, width, height, paint_data="", region=None, region_painter=None):
        if not isinstance(width, int) or not isinstance(height, int) or not (64 <= width <= 4096 and 64 <= height <= 4096):
            raise ValueError("Painter resolution must be between 64 and 4096 pixels per axis.")
        if region is not None and not isinstance(region, list):
            raise ValueError("Painter region input must be a ComfyCouple region chain.")
        target = None
        for entry in region or []:
            if not isinstance(entry, dict):
                raise ValueError("Painter region chain entries must be dictionaries.")
            if "mask" not in entry:
                continue
            mask = entry["mask"]
            if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3) or any(size < 1 for size in mask.shape):
                raise ValueError("Existing region masks must have shape [H,W] or [B,H,W].")
            if target is not None and (mask.shape != target.shape or mask.device != target.device):
                raise ValueError("Existing region masks must have matching shapes and devices.")
            target = mask
        try:
            data = json.loads(paint_data) if paint_data else {"version": 1, "width": width, "height": height, "layers": []}
        except (ValueError, TypeError) as error:
            raise ValueError("Invalid painter data. Reload the workflow and try again.") from error
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("layers"), list):
            raise ValueError("Unsupported painter data format.")
        background = data.get("background", {"prompt": "", "mode": BACKGROUND_MODES[0]})
        if (not isinstance(background, dict) or not isinstance(background.get("prompt"), str)
                or background.get("mode") not in BACKGROUND_MODES):
            raise ValueError("Invalid painter background prompt or mode.")
        base = data.get("base", {"prompt": ""})
        if not isinstance(base, dict) or not isinstance(base.get("prompt"), str):
            raise ValueError("Invalid painter base prompt.")
        layers = data["layers"]
        # Saved preview caches are not a connection. A disconnected/headless
        # workflow must not render stale mannequin masks or their extra paint.
        source = read_pose_regions(region_painter)
        if region_painter is None:
            layers = [layer for layer in layers if not (isinstance(layer, dict) and layer.get("pose_source") is not None)]
        mask_width, mask_height = data.get("width"), data.get("height")
        if type(mask_width) is not int or type(mask_height) is not int or not 64 <= mask_width <= 4096 or not 64 <= mask_height <= 4096:
            raise ValueError("Invalid saved painter resolution.")
        if source is not None:
            width, height = source["width"], source["height"]
        layers, body_map = merge_source_layers(layers, source, mask_width, mask_height)
        if len(layers) > MAX_LAYERS or width * height * max(1, len(layers)) > MAX_LAYER_PIXELS:
            raise ValueError("Painter supports up to 32 layers and 64 million total layer pixels. Reduce layers or resolution.")
        # Never silently stretch stale captures to a different widget resolution.
        if source is None and (data.get("width") != width or data.get("height") != height):
            raise ValueError("Painter resolution changed. Wait for the canvas to resize and queue again.")
        decoded = []
        ids = set()
        for index, layer in enumerate(layers):
            if (not isinstance(layer, dict) or not isinstance(layer.get("id"), str)
                    or layer["id"] in ("__painter_background__", "__painter_base__") or layer["id"] in ids):
                raise ValueError("Painter layers need unique IDs.")
            ids.add(layer["id"])
            if layer.get("mask_layer_id") is not None and not isinstance(layer["mask_layer_id"], str):
                raise ValueError("Invalid layer Mask reference.")
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(layer.get("color", ""))):
                raise ValueError(f"Layer {index + 1} has an invalid color.")
            if not isinstance(layer.get("prompt", ""), str) or not isinstance(layer.get("visible", True), bool):
                raise ValueError(f"Layer {index + 1} has invalid prompt or visibility data.")
            try:
                strength = float(layer.get("strength", 1.0))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Layer {index + 1} has invalid strength.") from error
            if not math.isfinite(strength) or not 0 <= strength <= 10:
                raise ValueError(f"Layer {index + 1} strength must be between 0 and 10.")
            png = layer.get("mask", "")
            if not isinstance(png, str) or not png.startswith("data:image/png;base64,") or len(png) > 96 * 1024 * 1024:
                raise ValueError(f"Layer {index + 1} has an invalid mask capture.")
            try:
                raw = base64.b64decode(png.split(",", 1)[1], validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format != "PNG" or image.size != (mask_width, mask_height) or image.mode != "RGBA":
                        raise ValueError("Mask must be an RGBA PNG at the canvas resolution.")
                    if image.size != (width, height):
                        image = image.resize((width, height), Image.Resampling.BILINEAR)
                    alpha = np.array(image.getchannel("A"), dtype=np.float32) / 255.0
            except Exception as error:
                raise ValueError(f"Cannot read layer {index + 1} mask: {error}") from error
            color = np.array([int(layer["color"][offset:offset + 2], 16) for offset in (1, 3, 5)], dtype=np.float32) / 255.0
            decoded.append((layer, alpha, color, strength))

        # Preserve each body's seed within its own layer. Union it with editable
        # paint before normal back-to-front compositing: higher layers can cover
        # a lower body without deleting its source or saved overlay pixels.
        if source is not None:
            labels = source["labels"].numpy()
            protected = []
            for layer, alpha, color, strength in decoded:
                owner = layer.get("pose_source")
                if owner:
                    body = body_map[source_key(owner["source_id"], owner["body_id"])]
                    alpha = np.where(labels == body["label"], 1., alpha).astype(np.float32)
                protected.append((layer, alpha, color, strength))
            decoded = protected

        # Clip to effective pre-occlusion alpha, including hidden providers and
        # protected body seeds. Original PNGs remain untouched for unclipping.
        by_id = {layer["id"]: (layer, alpha) for layer, alpha, _, _ in decoded}
        effective, visiting = {}, set()

        def clipped(layer_id):
            if layer_id in effective:
                return effective[layer_id]
            if layer_id in visiting:
                raise ValueError("Layer Mask references cannot form a cycle or mask themselves.")
            visiting.add(layer_id)
            layer, alpha = by_id[layer_id]
            mask_id = layer.get("mask_layer_id")
            if mask_id in by_id:
                alpha = alpha * clipped(mask_id)
            visiting.remove(layer_id)
            effective[layer_id] = alpha
            return alpha

        for layer_id in by_id:
            clipped(layer_id)
        decoded = [(layer, effective[layer["id"]], color, strength) for layer, _, color, strength in decoded]
        image = np.zeros((height, width, 3), dtype=np.float32)
        uncovered = np.ones((height, width), dtype=np.float32)
        painted_regions = []
        # Saved layers are back -> front. Compute visible masks without changing
        # the original alpha PNGs, allowing reorder/erase to reveal lower layers.
        for index in range(len(decoded) - 1, -1, -1):
            layer, alpha, color, strength = decoded[index]
            if not layer.get("visible", True):
                continue
            mask = alpha * uncovered
            uncovered *= 1.0 - alpha
            image += mask[..., None] * color
            if not np.any(mask > 0):
                continue
            prompt = layer.get("prompt", "")
            if not prompt.strip():
                raise ValueError(f"Painted layer {index + 1} needs a prompt. Enter a prompt or hide the layer.")
            output_mask = torch.from_numpy(mask).unsqueeze(0)
            if target is not None:
                if target.numel() * max(1, len(layers)) > MAX_LAYER_PIXELS:
                    raise ValueError("Existing region resolution exceeds the painter's 64 million layer pixel limit.")
                if tuple(output_mask.shape[-2:]) != tuple(target.shape[-2:]):
                    output_mask = F.interpolate(output_mask.unsqueeze(1), size=target.shape[-2:], mode="area").squeeze(1)
                output_mask = output_mask.to(device=target.device)
                output_mask = output_mask[0] if target.ndim == 2 else output_mask.expand(target.shape[0], -1, -1).contiguous()
            painted_regions.append({"mask": output_mask, "strength": strength,
                                    "prompt_text": prompt, "clip": clip, "lora_hook": None})
        # Preserve the input chain and metadata, and match the existing Region
        # node's back -> front ordering. ComfyCouple Mask encodes text downstream.
        output_regions = list(region or []) + list(reversed(painted_regions))
        if base["prompt"].strip():
            # Same metadata as ComfyCoupleBasePrompt: shared text, not a mask.
            output_regions.append({"__comfycouple_chain_meta__": {
                "base_prompt_text": base["prompt"],
            }})
        if background["prompt"].strip():
            # Same chain metadata as ComfyCoupleBackgroundPrompt.process().
            # Mask derives uncovered background from ALL region masks; this
            # must not become an opaque black painting that covers the regions.
            output_regions.append({"__comfycouple_chain_meta__": {
                "background_prompt_text": background["prompt"],
                "background_prompt_mode": background["mode"],
            }})
        # Blank Base / Background prompts preserve incoming chain settings.
        return (torch.from_numpy(image).unsqueeze(0), output_regions)
