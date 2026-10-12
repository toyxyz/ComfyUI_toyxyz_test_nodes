"""Stable body-ID segmentation transport (never infer IDs from display colors)."""
import base64
import io
import re

import numpy as np
import torch
from PIL import Image

REGION_PAINTER_TYPE = "TOYXYZ_REGION_PAINTER"


def source_key(source_id, body_id):
    import json
    return json.dumps([source_id, body_id], ensure_ascii=False, separators=(",", ":"))


def read_pose_regions(payload):
    if payload is None:
        return None
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Invalid 3D Pose Editor region painter input.")
    if payload.get("error"):
        raise ValueError(payload["error"])
    width, height = payload.get("width"), payload.get("height")
    if (type(width) is not int or type(height) is not int or
            not 64 <= width <= 4096 or not 64 <= height <= 4096):
        raise ValueError("Invalid pose-region resolution.")
    source_id, bodies = payload.get("source_id"), payload.get("bodies")
    if not isinstance(source_id, str) or not 1 <= len(source_id) <= 256 or not isinstance(bodies, list) or len(bodies) > 256:
        raise ValueError("Invalid pose-region body list.")
    ids, labels, records = set(), set(), []
    for body in bodies:
        if (not isinstance(body, dict) or not isinstance(body.get("id"), str) or not 1 <= len(body["id"]) <= 80
                or body["id"] in ids or type(body.get("label")) is not int or not 1 <= body["label"] <= 256
                or body["label"] in labels or not re.fullmatch(r"#[0-9a-fA-F]{6}", str(body.get("color", "")))
                or not isinstance(body.get("name"), str) or len(body["name"]) > 160):
            raise ValueError("Invalid pose-region body identity or color.")
        ids.add(body["id"]); labels.add(body["label"])
        records.append({key: body[key] for key in ("id", "label", "color", "name")})
    encoded = payload.get("id_map")
    if not isinstance(encoded, str) or not encoded.startswith("data:image/png;base64,") or len(encoded) > 96 * 1024 * 1024:
        raise ValueError("Invalid pose-region ID map.")
    supplied = payload.get("labels")
    if supplied is not None:
        if not isinstance(supplied, torch.Tensor) or supplied.shape != (height, width) or supplied.dtype not in (torch.int32, torch.int64):
            raise ValueError("Invalid pose-region label tensor.")
        pixels = supplied.detach().cpu().numpy()
    else:
        try:
            raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "PNG" or image.size != (width, height) or image.mode not in ("RGB", "RGBA"):
                    raise ValueError("ID map must be a PNG at the source resolution.")
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint32)
            pixels = rgb[..., 0] + (rgb[..., 1] << 8) + (rgb[..., 2] << 16)
        except Exception as error:
            raise ValueError(f"Cannot decode pose-region ID map: {error}") from error
    if not np.isin(pixels, [0, *labels]).all():
        raise ValueError("Pose-region map contains unknown body IDs.")
    return {"version": 1, "source_id": source_id, "width": width, "height": height,
            "bodies": records, "id_map": encoded, "labels": torch.from_numpy(np.array(pixels, dtype=np.int32, copy=True))}


def blank_mask(width, height):
    out = io.BytesIO()
    Image.new("RGBA", (width, height)).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def merge_source_layers(layers, source, mask_width, mask_height):
    """Use current upstream bodies, but retain their authored overlay metadata."""
    if source is None:
        return layers, {}
    body_map = {source_key(source["source_id"], body["id"]): body for body in source["bodies"]}
    result, existing = [], set()
    for layer in layers:
        owner = layer.get("pose_source") if isinstance(layer, dict) else None
        if owner is None:
            result.append(layer)
            continue
        if not isinstance(owner, dict) or not isinstance(owner.get("source_id"), str) or not isinstance(owner.get("body_id"), str):
            raise ValueError("Invalid painter pose-source identity.")
        key = source_key(owner["source_id"], owner["body_id"])
        if key not in body_map:
            continue
        if key in existing:
            raise ValueError("Duplicate painter layers for the same pose body.")
        existing.add(key)
        result.append({**layer, "color": body_map[key]["color"]})
    empty = None
    for key, body in body_map.items():
        if key in existing:
            continue
        empty = empty or blank_mask(mask_width, mask_height)
        result.append({"id": "pose:" + key, "name": body["name"], "color": body["color"],
                       "prompt": "", "visible": True, "strength": 1., "mask": empty,
                       "pose_source": {"source_id": source["source_id"], "body_id": body["id"]}})
    return result, body_map
