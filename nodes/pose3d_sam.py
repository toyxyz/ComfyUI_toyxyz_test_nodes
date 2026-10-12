"""Single-image SAM 3D Body import, executed by ComfyUI's normal queue.

Reuse the installed core loader/predictor; never load a parallel inference engine.
Local TRS follows Meta MHR's parameter transform (Apache-2.0), in meters/xyzw.
"""
import numpy as np
import torch
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import threading
import urllib.request
import uuid

DEFAULT_MODEL = "sam_3d_body_dinov3_bf16.safetensors"
# Pin the official checkpoint and verify its published LFS size/SHA256.
DEFAULT_MODEL_URL = "https://huggingface.co/Comfy-Org/sam-3d-body/resolve/3d7eeedcb3fe5ab03235438b9e67c9b889eb97c1/detection/" + DEFAULT_MODEL
DEFAULT_MODEL_SIZE = 2830737652
DEFAULT_MODEL_SHA256 = "59fa45200c504c5b56625004d7d3385daf48c616613e88099e43bf83b3e249cf"
_download_lock = threading.Lock()
_status_lock = threading.Lock()
_download_status = {"state": "idle", "downloaded": 0, "total": DEFAULT_MODEL_SIZE}


def ensure_default_model(progress=None):
    """Only the known default can be fetched; existing configured models win."""
    import folder_paths
    with _download_lock:
        existing = folder_paths.get_full_path("detection", DEFAULT_MODEL)
        if existing:
            return existing
        target = Path(folder_paths.models_dir) / "detection" / DEFAULT_MODEL
        temp_path = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(target.parent).free < DEFAULT_MODEL_SIZE:
                raise OSError("Not enough free disk space for the SAM checkpoint (about 2.83 GB).")
            request = urllib.request.Request(DEFAULT_MODEL_URL, headers={"User-Agent": "ComfyUI-Toyxyz-Pose3D", "Accept-Encoding": "identity"})
            with urllib.request.urlopen(request, timeout=60) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != DEFAULT_MODEL_SIZE:
                    raise ValueError("The official SAM checkpoint has an unexpected file size.")
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".sam3d-", suffix=".download", delete=False) as output:
                    temp_path = Path(output.name)
                    digest, received = hashlib.sha256(), 0
                    while chunk := response.read(4 * 1024 * 1024):
                        received += len(chunk)
                        if received > DEFAULT_MODEL_SIZE:
                            raise ValueError("The SAM checkpoint exceeds the expected size.")
                        output.write(chunk); digest.update(chunk)
                        if progress:
                            progress(received, DEFAULT_MODEL_SIZE)
                    output.flush(); os.fsync(output.fileno())
            if received != DEFAULT_MODEL_SIZE or digest.hexdigest() != DEFAULT_MODEL_SHA256:
                raise ValueError("SAM checkpoint verification failed (size or SHA-256 mismatch).")
            # Publish atomically without overwriting a file created meanwhile.
            try:
                os.link(temp_path, target)
            except FileExistsError:
                pass
            return str(target)
        except Exception as exc:
            raise RuntimeError(f"SAM 3D Body download failed: {exc}. Check network/disk access and retry Get pose, or place {DEFAULT_MODEL} in ComfyUI/models/detection.") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def model_download_status():
    with _status_lock:
        return dict(_download_status)


def start_model_download():
    import folder_paths
    core_nodes()  # Do not download a large file for an unsupported ComfyUI.
    with _status_lock:
        if _download_status["state"] == "downloading":
            return dict(_download_status)
        if folder_paths.get_full_path("detection", DEFAULT_MODEL):
            _download_status.clear()
            _download_status.update(state="ready", downloaded=0, total=DEFAULT_MODEL_SIZE)
            return dict(_download_status)
        job_id = uuid.uuid4().hex
        _download_status.clear()
        _download_status.update(state="downloading", id=job_id, downloaded=0, total=DEFAULT_MODEL_SIZE)

        def update(received, total):
            with _status_lock:
                _download_status.update(downloaded=received, total=total)

        def worker():
            try:
                ensure_default_model(update)
                with _status_lock:
                    _download_status.update(state="ready", downloaded=DEFAULT_MODEL_SIZE)
            except Exception as exc:
                with _status_lock:
                    _download_status.update(state="error", error=str(exc))

        threading.Thread(target=worker, name="toyxyz-sam-model-download", daemon=True).start()
        return dict(_download_status)


def register_download_routes():
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        return
    if not getattr(PromptServer, "instance", None):
        return

    @PromptServer.instance.routes.post("/toyxyz/pose3d/sam-model")
    async def start(request):
        import asyncio
        try:
            return web.json_response(await asyncio.to_thread(start_model_download))
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.get("/toyxyz/pose3d/sam-model")
    async def status(request):
        return web.json_response(model_download_status())


def core_nodes():
    try:
        from comfy_extras.nodes_sam3d_body import SAM3DBody_Loader, SAM3DBody_Predict
    except ImportError as exc:
        raise RuntimeError("SAM 3D Body requires a ComfyUI version with SAM3DBody_Loader and SAM3DBody_Predict.") from exc
    return SAM3DBody_Loader, SAM3DBody_Predict


class Pose3DSAMLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        files = folder_paths.get_filename_list("detection")
        # Keep the downloadable default valid even if other detection files exist.
        return {"required": {"model_file": (list(dict.fromkeys([DEFAULT_MODEL, *files])), {"default": DEFAULT_MODEL})}}

    RETURN_TYPES = ("SAM3D_BODY_MODEL",)
    RETURN_NAMES = ("sam3d_body_model",)
    FUNCTION = "load_model"
    CATEGORY = "ToyxyzTestNodes/Pose"

    def load_model(self, model_file):
        import folder_paths
        loader, _ = core_nodes()
        if not folder_paths.get_full_path("detection", model_file):
            if model_file == DEFAULT_MODEL:
                ensure_default_model()
            else:
                raise FileNotFoundError(f"SAM 3D Body model missing: {model_file}. Place it in ComfyUI/models/detection (or a configured detection model path).")
        return (loader.execute(model_file)[0],)


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def import_payload(model, pose_data, request_id):
    """Export native local transforms, INCLUDING scale, without camera depth."""
    frames = pose_data.get("frames", [])
    if len(frames) != 1 or len(frames[0]) != 1:
        raise ValueError("Get pose requires exactly one predicted body from one image.")
    person = frames[0][0]
    shape = as_numpy(person["shape_params"]).reshape(-1)
    params = as_numpy(person["mhr_model_params"]).reshape(-1)
    if shape.shape != (45,) or params.shape != (204,) or not np.isfinite(shape).all() or not np.isfinite(params).all() or np.abs(shape).max() > 64:
        raise ValueError("Invalid SAM MHR body parameters.")
    rig = model.model.head_pose.mhr
    pt = as_numpy(rig.param_transform)
    offsets = as_numpy(rig.skel_joint_translation_offsets)
    pre = as_numpy(rig.skel_joint_prerotations)
    parents = as_numpy(rig.skel_joint_parents).astype(np.int32)
    if pt.shape != (889, 249) or offsets.shape != (127, 3) or pre.shape != (127, 4) or parents.shape != (127,):
        raise ValueError("Unsupported MHR rig: expected the editor's 127-joint LOD1 body.")
    jp = (pt @ np.concatenate([params, np.zeros(45, np.float32)])).reshape(127, 7)
    c, s = np.cos(jp[:, 3:6] * .5).T, np.sin(jp[:, 3:6] * .5).T
    cr, cp, cy = c
    sr, sp, sy = s
    q = np.stack([sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy], axis=1)
    # Hamilton product: pre-rotation * Euler rotation.
    xyz = pre[:, 3:4]*q[:, :3] + q[:, 3:4]*pre[:, :3] + np.cross(pre[:, :3], q[:, :3])
    w = pre[:, 3]*q[:, 3] - np.sum(pre[:, :3]*q[:, :3], axis=1)
    quaternion = np.column_stack([xyz, w])
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    position = (jp[:, :3] + offsets) * .01
    scale = np.exp2(jp[:, 6])
    if not all(np.isfinite(a).all() for a in (position, quaternion, scale)) or np.any(scale <= 1e-6) or np.any(scale > 1000):
        raise ValueError("Invalid predicted MHR transforms.")
    return {"version": 1, "request_id": request_id, "source": "MHR", "parents": parents.tolist(),
            "shape": {"schema": "MHR.identity45.v1", "coefficients": shape.tolist()},
            "joints": [{"position": p.tolist(), "quaternion": q.tolist(), "scale": [float(s)]*3}
                       for p, q, s in zip(position, quaternion, scale)]}


class Pose3DSAMImport:
    """Internal output sink for the Get pose mini-workflow (not image rendering)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "request_id": ("STRING",)},
                "optional": {"sam3d_body_model": ("SAM3D_BODY_MODEL",)},
                "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "get_pose"
    CATEGORY = "ToyxyzTestNodes/Pose/internal"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def get_pose(self, image, request_id, sam3d_body_model=None, unique_id=None):
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Get pose accepts one still image.")
        _, predict = core_nodes()
        if sam3d_body_model is None:
            sam3d_body_model = Pose3DSAMLoader().load_model(DEFAULT_MODEL)[0]
        # The installed core predictor creates some float32 decoder inputs even
        # when its native loader selects fp16 weights. Keep linear ops consistent
        # without patching core or mutating a shared/cached model's weights.
        # Do NOT autocast the whole prediction: MHR's sparse CUDA skin-corrective
        # multiplication needs fp32. Adapt only Linear inputs on this instance,
        # then remove every hook even on error; leave weights/core code intact.
        def linear_input(module, args):
            # DynamicVRAM stores native checkpoint weights (often BF16) while
            # Comfy casts them to the INPUT's compute dtype on demand. Casting
            # those inputs to storage dtype mixes DINO's rotary Q/K and V types.
            # Respect Comfy's weight-casting/patching path, including low-VRAM.
            if (getattr(module, "comfy_cast_weights", False) or hasattr(module, "_v") or
                    getattr(module, "weight_function", None) or getattr(module, "bias_function", None)):
                return None
            dtype = module.weight.dtype
            if args and isinstance(args[0], torch.Tensor) and args[0].dtype != dtype and dtype in (torch.float16, torch.bfloat16):
                return (args[0].to(dtype=dtype), *args[1:])
        hooks = []
        try:
            for module in sam3d_body_model.model.modules():
                if isinstance(module, torch.nn.Linear):
                    hooks.append(module.register_forward_pre_hook(linear_input))
            with torch.inference_mode():
                data = predict.execute(sam3d_body_model, image, run_hand_refinement=True, fov=0., batch_size=1)[0]
        finally:
            for hook in hooks:
                hook.remove()
        payload = import_payload(sam3d_body_model, data, request_id)
        return {"ui": {"toyxyz_sam_pose": [payload]}, "result": ()}


register_download_routes()
