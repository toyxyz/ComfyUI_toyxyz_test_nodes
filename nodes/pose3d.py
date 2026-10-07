"""Browser-rendered pose editor and local SMPL/SMPL-H template transport."""
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image

MODEL_ROOT = Path(__file__).resolve().parents[1] / "models" / "pose3d"

# COCO-18 RGB palette and ControlNet's 17 drawn connections. Joint colors and
# limb colors are separate index sequences; reversing a camera never swaps them.
OPENPOSE_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
                   (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
                   (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
                   (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
                   (255, 0, 170), (255, 0, 85)]
OPENPOSE_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8),
                 (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
                 (0, 14), (14, 16), (0, 15), (15, 17)]


def draw_openpose(payload, width, height):
    """Rasterize browser-projected model joints, without any pose detector.

    RGB values follow ControlNet/comfyui_controlnet_aux draw_bodypose: elliptic
    limbs at 60% brightness, followed by opaque colored joint disks. No lighting,
    tone mapping, interpolation or BGR-to-RGB channel swap is applied.
    """
    if not isinstance(payload, dict) or payload.get("version") not in (1, 2) or (payload.get("width"), payload.get("height")) != (width, height):
        raise ValueError("No current model-derived OpenPose data. Refresh the editor and queue again.")
    people = [payload] if payload["version"] == 1 else payload.get("people")
    if not isinstance(people, list) or len(people) > 256:
        raise ValueError("Invalid OpenPose body list.")
    all_points = []
    for person in people:
        points = person.get("keypoints") if isinstance(person, dict) else None
        if not isinstance(points, list) or len(points) != 18:
            raise ValueError("OpenPose requires exactly 18 COCO landmarks per body.")
        for point in points:
            if point is not None and (not isinstance(point, list) or len(point) != 2 or
                                      not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) and abs(v) < 1e7 for v in point)):
                raise ValueError("Invalid projected OpenPose landmark.")
        all_points.append(points)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    radius = max(1, round(4 * min(width, height) / 512))
    for points in all_points:
        for (start, end), color in zip(OPENPOSE_LIMBS, OPENPOSE_COLORS):
            a, b = points[start], points[end]
            if a is None or b is None:
                continue
            delta = np.array(b) - a
            length = float(np.linalg.norm(delta))
            if length < 1e-6:
                continue
            center = tuple(int(v) for v in (np.array(a) + b) / 2)
            angle = int(np.degrees(np.arctan2(delta[1], delta[0])))
            polygon = cv2.ellipse2Poly(center, (max(1, int(length / 2)), radius), angle, 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, tuple(int(c * .6) for c in color))
        for point, color in zip(points, OPENPOSE_COLORS):
            if point is not None:
                cv2.circle(canvas, tuple(int(v) for v in point), radius, color, thickness=-1)
    return torch.from_numpy(canvas.astype(np.float32) / 255).unsqueeze(0)


def decode_browser_pass(encoded, width, height, name):
    if not isinstance(encoded, str) or not encoded.startswith("data:image/png;base64,") or len(encoded) > 100_000_000:
        raise ValueError(f"No valid browser {name} capture. Refresh the editor and queue again.")
    raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(raw)) as source:
        if source.size != (width, height):
            raise ValueError(f"{name} size differs from node settings. Re-render before executing.")
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(rgb.copy()).unsqueeze(0)


def load_body(gender):
    if gender not in ("male", "female"):
        raise ValueError("Unknown body type.")
    path = MODEL_ROOT / f"{gender}.npz"
    if not path.is_file():
        return {"available": False, "message": f"Place {gender}.npz in models/pose3d to use an actual SMPL or SMPL-H body."}
    with np.load(path, allow_pickle=False) as data:
        vertices = np.asarray(data["v_template"], dtype=np.float32).reshape(-1, 3)
        weights = np.asarray(data["weights"], dtype=np.float32)
        joints = np.asarray(data["J_regressor"], dtype=np.float32) @ vertices
        tree = np.asarray(data["kintree_table"], dtype=np.int64)
        faces = np.asarray(data["f"], dtype=np.int32).reshape(-1, 3)
    if len(joints) not in (24, 52) or weights.shape != (len(vertices), len(joints)):
        raise ValueError("Expected a dense SMPL (24 joints) or SMPL-H (52 joints) NPZ template.")
    if not all(np.isfinite(a).all() for a in (vertices, weights, joints)):
        raise ValueError("Body contains non-finite data.")
    ids = {int(v): i for i, v in enumerate(tree[1])}
    parents = [-1] + [ids[int(v)] for v in tree[0, 1:]]
    indices = np.argsort(weights, axis=1)[:, -4:]
    skin = np.take_along_axis(weights, indices, axis=1)
    skin /= np.maximum(skin.sum(axis=1, keepdims=True), 1e-8)
    return {"available": True, "vertices": vertices.ravel().tolist(), "faces": faces.ravel().tolist(),
            "joints": joints.tolist(), "parents": parents, "indices": indices.ravel().tolist(),
            "weights": skin.ravel().tolist(), "kind": "SMPL-H" if len(joints) == 52 else "SMPL"}


class Pose3DEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
            "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
            "pose_data": ("STRING", {"default": "{}", "multiline": False}),
            "render_data": ("STRING", {"default": "", "multiline": False}),
        }, "optional": {"sam3d_body_model": ("SAM3D_BODY_MODEL", {"lazy": True})}}

    def check_lazy_status(self, **kwargs):
        # This socket is consumed by Get pose's separate mini-workflow, not render.
        return []

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "openpose", "depth", "normal")
    FUNCTION = "render_pose"
    CATEGORY = "ToyxyzTestNodes/Pose"

    def render_pose(self, width, height, pose_data, render_data, body=None, sam3d_body_model=None):
        # body is accepted only for old keyword-based API callers. The current
        # node has no body selector; saved model identity lives in pose_data.
        pose = json.loads(pose_data)
        if not isinstance(pose, dict) or pose.get("version") != 1:
            raise ValueError("Open the 3d pose editor and render the current body before executing.")
        if not isinstance(render_data, str) or len(render_data) > 300_000_000:
            raise ValueError("Invalid browser capture bundle.")
        try:
            bundle = json.loads(render_data)
        except ValueError as error:
            raise ValueError("Depth/normal captures are missing. Refresh the editor and queue again.") from error
        if not isinstance(bundle, dict) or bundle.get("version") != 1:
            raise ValueError("Invalid browser capture bundle. Refresh the editor and queue again.")
        image, depth, normal = (decode_browser_pass(bundle.get(name), width, height, name)
                                for name in ("render", "depth", "normal"))
        return (image, draw_openpose(pose.get("openpose"), width, height), depth, normal)


def register_routes():
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        return
    if not getattr(PromptServer, "instance", None):
        return

    @PromptServer.instance.routes.get("/toyxyz/pose3d/body")
    async def body(request):
        import asyncio
        try:
            return web.json_response(await asyncio.to_thread(load_body, request.query.get("gender", "female")))
        except (ValueError, KeyError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)


register_routes()
