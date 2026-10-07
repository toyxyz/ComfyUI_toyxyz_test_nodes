"""Single-image SAM 3D Body import, executed by ComfyUI's normal queue.

Reuse the installed core loader/predictor; never load a parallel inference engine.
Local TRS follows Meta MHR's parameter transform (Apache-2.0), in meters/xyzw.
"""
import numpy as np
import torch

DEFAULT_MODEL = "sam_3d_body_dinov3_bf16.safetensors"


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
        return {"required": {"model_file": (files or [DEFAULT_MODEL], {"default": DEFAULT_MODEL if DEFAULT_MODEL in files or not files else files[0]})}}

    RETURN_TYPES = ("SAM3D_BODY_MODEL",)
    RETURN_NAMES = ("sam3d_body_model",)
    FUNCTION = "load_model"
    CATEGORY = "ToyxyzTestNodes/Pose"

    def load_model(self, model_file):
        import folder_paths
        if not folder_paths.get_full_path("detection", model_file):
            raise FileNotFoundError(f"SAM 3D Body model missing: {model_file}. Place it in ComfyUI/models/detection (or a configured detection model path).")
        loader, _ = core_nodes()
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
