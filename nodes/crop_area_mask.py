import hashlib
import json
import math
import os
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths


class CropAreaMask:
    BOXES_PROPERTY = "crop_area_boxes"
    BOXES_STATE_FALLBACK = "[]"

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (
                    sorted(files),
                    {
                        "image_upload": True,
                        "tooltip": "Load an image, then hold Ctrl and drag on the preview below to add crop boxes.",
                    },
                ),
                "boxes_state": ("STRING", {
                    "default": cls.BOXES_STATE_FALLBACK,
                    "multiline": False,
                    "tooltip": "Internal serialized crop box state.",
                }),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("crops", "masks", "source_image")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "crop_image"
    CATEGORY = "ToyxyzTestNodes"

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        image_np = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np)[None,]

    @classmethod
    def _extract_raw_boxes(cls, extra_pnginfo: Dict[str, Any] | None, unique_id: str | None) -> List[Dict[str, Any]]:
        if extra_pnginfo is None or unique_id is None:
            return []

        workflow = extra_pnginfo.get("workflow", {})
        nodes = workflow.get("nodes", [])
        for node in nodes:
            if str(node.get("id", "")) == str(unique_id):
                properties = node.get("properties", {})
                boxes = properties.get(cls.BOXES_PROPERTY, [])
                return boxes if isinstance(boxes, list) else []

        return []

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

        return normalized

    @classmethod
    def _load_boxes(cls, extra_pnginfo: Dict[str, Any] | None, unique_id: str | None) -> List[Dict[str, float]]:
        return cls._normalize_boxes(cls._extract_raw_boxes(extra_pnginfo, unique_id))

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

    def crop_image(self, image, boxes_state, extra_pnginfo=None, unique_id=None):
        boxes = self._load_boxes_from_state(boxes_state)
        if not boxes:
            boxes = self._load_boxes(extra_pnginfo, unique_id)
        if not boxes:
            raise ValueError("Crop area mask requires at least one crop box.")

        image_path = folder_paths.get_annotated_filepath(image)
        with Image.open(image_path) as image_file:
            image_file = ImageOps.exif_transpose(image_file)
            if image_file.mode == "I":
                image_file = image_file.point(lambda value: value * (1 / 255))
            source = image_file.convert("RGB")

        source_width, source_height = source.size
        source_tensor = self._pil_to_tensor(source)
        crop_tensors = []
        mask_tensors = []

        for box in boxes:
            left, top, right, bottom = self._box_to_pixels(box, source_width, source_height)
            crop = source.crop((left, top, right, bottom))
            crop_tensor = self._pil_to_tensor(crop)
            crop_tensors.append(crop_tensor)

            mask = torch.zeros((1, source_height, source_width), dtype=torch.float32)
            mask[:, top:bottom, left:right] = 1.0
            mask_tensors.append(mask)

        return (crop_tensors, mask_tensors, source_tensor)

    @classmethod
    def IS_CHANGED(cls, image, boxes_state, extra_pnginfo=None, unique_id=None):
        image_path = folder_paths.get_annotated_filepath(image)
        hasher = hashlib.sha256()

        with open(image_path, "rb") as image_file:
            hasher.update(image_file.read())

        boxes = cls._load_boxes_from_state(boxes_state)
        if not boxes:
            boxes = cls._load_boxes(extra_pnginfo, unique_id)
        hasher.update(json.dumps(boxes, sort_keys=True).encode("utf-8"))
        return hasher.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image, boxes_state=None):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True
