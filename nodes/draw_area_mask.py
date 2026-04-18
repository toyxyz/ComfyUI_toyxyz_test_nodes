import hashlib
import json
import math
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class DrawAreaMask:
    BOXES_STATE_FALLBACK = "[]"
    MAX_OUTPUTS = 32

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Canvas width for the generated masks.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Canvas height for the generated masks.",
                    },
                ),
                "boxes_state": (
                    "STRING",
                    {
                        "default": cls.BOXES_STATE_FALLBACK,
                        "multiline": False,
                        "tooltip": "Internal serialized box state.",
                    },
                ),
            },
        }

    RETURN_TYPES = tuple(["IMAGE"] + ["MASK"] * MAX_OUTPUTS)
    RETURN_NAMES = tuple(["canvas_image"] + [f"mask_{i}" for i in range(MAX_OUTPUTS)])
    FUNCTION = "draw_masks"
    CATEGORY = "ToyxyzTestNodes"

    @staticmethod
    def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
        h = h / 360.0
        s = s / 100.0
        l = l / 100.0

        if s == 0:
            r = g = b = l
        else:
            def hue_to_rgb(p, q, t):
                if t < 0:
                    t += 1
                if t > 1:
                    t -= 1
                if t < 1 / 6:
                    return p + (q - p) * 6 * t
                if t < 1 / 2:
                    return q
                if t < 2 / 3:
                    return p + (q - p) * (2 / 3 - t) * 6
                return p

            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1 / 3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1 / 3)

        return int(r * 255), int(g * 255), int(b * 255)

    @classmethod
    def _normalize_boxes(cls, raw_boxes: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        normalized = []

        for raw_box in raw_boxes:
            if not isinstance(raw_box, dict):
                continue

            try:
                x = float(raw_box.get("x", 0.0))
                y = float(raw_box.get("y", 0.0))
                width = float(raw_box.get("width", 0.0))
                height = float(raw_box.get("height", 0.0))
                hue = float(raw_box.get("hue", 0.0))
            except (TypeError, ValueError):
                continue

            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            x2 = max(x, min(1.0, x + width))
            y2 = max(y, min(1.0, y + height))
            width = x2 - x
            height = y2 - y

            if width <= 0.0 or height <= 0.0:
                continue

            normalized.append(
                {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "hue": hue % 360.0,
                }
            )

        return normalized[: cls.MAX_OUTPUTS]

    @classmethod
    def _load_boxes_from_state(cls, boxes_state: Any) -> List[Dict[str, float]]:
        if boxes_state is None:
            return []

        raw_boxes = boxes_state
        if isinstance(boxes_state, str):
            try:
                raw_boxes = json.loads(boxes_state)
            except json.JSONDecodeError:
                return []

        return cls._normalize_boxes(raw_boxes if isinstance(raw_boxes, list) else [])

    @staticmethod
    def _box_to_pixels(box: Dict[str, float], width: int, height: int) -> tuple[int, int, int, int]:
        left = max(0, min(width - 1, int(math.floor(box["x"] * width))))
        top = max(0, min(height - 1, int(math.floor(box["y"] * height))))
        right = max(left + 1, min(width, int(math.ceil((box["x"] + box["width"]) * width))))
        bottom = max(top + 1, min(height, int(math.ceil((box["y"] + box["height"]) * height))))
        return left, top, right, bottom

    @staticmethod
    def _blank_mask(width: int, height: int) -> torch.Tensor:
        return torch.zeros((1, height, width), dtype=torch.float32)

    @staticmethod
    def _blank_canvas_image(width: int, height: int) -> torch.Tensor:
        image = Image.new("RGB", (width, height), color="white")
        image_np = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np)[None,]

    @classmethod
    def _create_canvas_image(cls, width: int, height: int, boxes: List[Dict[str, float]]) -> torch.Tensor:
        image = Image.new("RGB", (width, height), color="white")
        overlay = Image.new("RGBA", (width, height), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except Exception:
            font = ImageFont.load_default()

        total_boxes = max(1, len(boxes))
        for index, box in enumerate(boxes):
            left, top, right, bottom = cls._box_to_pixels(box, width, height)
            hue = float(box.get("hue", ((index + 1) / total_boxes) * 360.0))
            r, g, b = cls._hsl_to_rgb(hue, 90, 65)
            fill_color = (r, g, b, 72)
            border_color = (r, g, b, 255)
            draw.rectangle([left, top, right - 1, bottom - 1], fill=fill_color, outline=border_color, width=3)

            label = str(index)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = left + (right - left - text_width) / 2
            text_y = top + (bottom - top - text_height) / 2
            draw.text((text_x - 1, text_y), label, fill=(0, 0, 0, 255), font=font)
            draw.text((text_x + 1, text_y), label, fill=(0, 0, 0, 255), font=font)
            draw.text((text_x, text_y - 1), label, fill=(0, 0, 0, 255), font=font)
            draw.text((text_x, text_y + 1), label, fill=(0, 0, 0, 255), font=font)
            draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=font)

        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np)[None,]

    def draw_masks(self, width, height, boxes_state):
        boxes = self._load_boxes_from_state(boxes_state)
        blank_mask = self._blank_mask(width, height)
        canvas_image = self._create_canvas_image(width, height, boxes) if boxes else self._blank_canvas_image(width, height)
        masks = []

        for box in boxes:
            left, top, right, bottom = self._box_to_pixels(box, width, height)
            mask = self._blank_mask(width, height)
            mask[:, top:bottom, left:right] = 1.0
            masks.append(mask)

        return tuple([canvas_image] + [masks[i] if i < len(masks) else blank_mask.clone() for i in range(self.MAX_OUTPUTS)])

    @classmethod
    def IS_CHANGED(cls, width, height, boxes_state):
        hasher = hashlib.sha256()
        hasher.update(str(width).encode("utf-8"))
        hasher.update(str(height).encode("utf-8"))
        boxes = cls._load_boxes_from_state(boxes_state)
        hasher.update(json.dumps(boxes, sort_keys=True).encode("utf-8"))
        return hasher.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, width, height, boxes_state=None):
        if int(width) < 16 or int(height) < 16:
            return "Width and height must be at least 16."
        return True
