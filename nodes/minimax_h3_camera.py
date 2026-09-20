"""Independent keyframed proxy scene, deterministic camera prose and CPU renderer.

World: Y up, subject front +Z. Camera forward -Z, Euler order YXZ (degrees).
Orbit azimuth is unwrapped: positive initially travels camera-right at azimuth 0.
No Qwen/model loading or browser is needed to execute this node.
"""
import copy
import hashlib
import json
import logging
import math

import numpy as np

LOG = logging.getLogger(__name__)


def scene_render_signature(scene):
    render_scene = normalize_scene(scene)
    render_scene.pop("use_camera_prompt", None)  # Prompt routing does not change pixels.
    render_scene.pop("tracking", None)  # Prompt reference frame; proxy geometry is unchanged.
    render_scene.pop("refvid", None)  # Reference routing does not change the preview/render.
    render_scene["_renderer_version"] = "sphere-grid-white-face-t-opaque-floor-v3"
    if render_scene.get("show_grid", True):
        render_scene.pop("show_grid", None)  # Keep existing grid-on cache signatures.
    return hashlib.sha256(json.dumps(render_scene, sort_keys=True).encode("utf-8")).hexdigest()

RATIOS = {"1:1": 1, "2:3": 2/3, "3:2": 1.5, "3:4": .75,
          "4:3": 4/3, "9:16": 9/16, "16:9": 16/9, "21:9": 21/9}
SUBJECT_HEIGHT = 1.8
CAMERA = {"id": "camera", "name": "Camera", "mode": "free", "aim": "target",
          "target": "subject_1", "target_height": 1/SUBJECT_HEIGHT, "position": [0, 1.3, 4],
          "rotation": [0, 0, 0], "fov": 40, "azimuth": 0, "elevation": 0,
          "radius": 4, "keys": []}
SUBJECT = {"id": "subject_1", "name": "Subject 1", "shape": "human", "color": "#38a6c9",
           "position": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1], "keys": []}
DEFAULT_SCENE = {"version": 3, "requested_duration": 5, "frames": 124, "fps": 24, "aspect_ratio": "16:9",
                 "megapixels": .1, "use_camera_prompt": True, "refvid": True, "show_grid": True, "show_background_grid": True, "interpolation": "smooth", "camera": CAMERA,
                 "subjects": [SUBJECT]}
CHANNELS = ("position", "rotation", "scale", "fov", "azimuth", "elevation", "radius", "target_height")
SUBJECT_BOXES = (
    ((-.25, 0, -.15), (.25, 1.4, .15)),
    ((-.435, .60, -.11), (-.295, 1.35, .11)),
    ((.295, .60, -.11), (.435, 1.35, .11)),
)
# Visual orientation aid only: deliberately excluded from framing/target bounds.
HUMAN_FACE_BOXES = (
    ((-.12, 1.66, .175), (.12, 1.705, .22)),
    ((-.025, 1.49, .175), (.025, 1.68, .22)),
)


def number(value, default=0, low=-1e6, high=1e6):
    try:
        v = float(value)
        return max(low, min(high, v)) if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def subject_geometry(subject):
    if subject.get("shape") == "box":
        return {"height": 1, "boxes": (((-.5, 0, -.5), (.5, 1, .5)),), "spheres": ()}
    if subject.get("shape") == "sphere":
        return {"height": 1, "boxes": (), "spheres": (((0, .5, 0), .5),)}
    return {"height": SUBJECT_HEIGHT, "boxes": SUBJECT_BOXES, "spheres": (((0, 1.6, 0), .2),)}


def subject_bounds(subject):
    geometry = subject_geometry(subject)
    boxes = [*geometry["boxes"], *((np.array(center)-radius, np.array(center)+radius)
                                  for center, radius in geometry["spheres"])]
    return np.array([[x, y, z] for lo, hi in boxes for x in (lo[0], hi[0])
                     for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def align_frame_count(seconds):
    """Same 24fps / 17n+5 rule as minimax_h3_prompter.align_frame_count."""
    frames = max(5, int(round(seconds * 24)))
    return frames + (5 - frames % 17) % 17


def normalize_scene(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            LOG.warning("H3 Camera: unreadable scene data; using the default proxy scene.")
            value = {}
    value = value if isinstance(value, dict) else {}
    out = copy.deepcopy(DEFAULT_SCENE)
    outgoing_keys = number(value.get("version"), 0) < 3
    legacy = "requested_duration" not in value and "frames" in value
    old_fps = number(value.get("fps"), 24, 1, 120)
    seconds = number(value.get("frames"), 121, 2, 1441)/old_fps if legacy else 5
    out["requested_duration"] = number(value.get("requested_duration"), seconds, .1, 60)
    out["requested_duration"] = max(.1, min(60, out["requested_duration"]))
    out["frames"] = align_frame_count(out["requested_duration"])
    out["fps"] = 24
    out["use_camera_prompt"] = value.get("use_camera_prompt", True) is True
    out["tracking"] = value.get("tracking") if value.get("tracking") in ("auto", "follow", "world") else "auto"
    out["refvid"] = value.get("refvid", True) is True
    out["show_grid"] = value.get("show_grid", True) is True
    out["show_background_grid"] = value.get("show_background_grid", True) is True
    out["megapixels"] = number(value.get("megapixels"), .1, .01, 4)
    out["aspect_ratio"] = value.get("aspect_ratio") if value.get("aspect_ratio") in RATIOS else DEFAULT_SCENE["aspect_ratio"]
    out["interpolation"] = value.get("interpolation") if value.get("interpolation") in ("linear", "smooth", "hold") else "smooth"

    def entity(raw, default, ident):
        raw = raw if isinstance(raw, dict) else {}
        e = copy.deepcopy(default)
        e["id"], e["name"] = ident, str(raw.get("name", default["name"]))[:100]
        for k in CHANNELS:
            if k not in default:
                continue
            if isinstance(default[k], list):
                v = raw.get(k, default[k])
                e[k] = [number(v[i] if isinstance(v, list) and len(v) > i else None,
                               default[k][i], .01 if k == "scale" else -1e4, 1e4) for i in range(3)]
            else:
                e[k] = number(raw.get(k), default[k], .01 if k == "radius" else 1 if k == "fov" else -1e5,
                              175 if k == "fov" else 1e5)
        if ident == "camera":
            for k, choices in {"mode": ("free", "orbit"), "aim": ("free", "target")}.items():
                e[k] = raw.get(k) if raw.get(k) in choices else default[k]
            legacy_height = {"feet": 0, "body": 1, "head": 1.6}.get(str(raw.get("target_part", "body")), 1)/SUBJECT_HEIGHT
            e["target_height"] = number(raw.get("target_height"), legacy_height, 0, 1)
            e["target"] = str(raw.get("target", default["target"]))
        else:
            e["shape"] = raw.get("shape") if raw.get("shape") in ("human", "box", "sphere") else "human"
            c = str(raw.get("color", default["color"]))
            e["color"] = c if len(c) == 7 and c[0] == "#" and all(x in "0123456789abcdefABCDEF" for x in c[1:]) else default["color"]
        keys = {}
        for key in raw.get("keys", []) if isinstance(raw.get("keys", []), list) else []:
            if not isinstance(key, dict):
                continue
            source_frame = number(key.get("frame"), 0, 0, 1e6)
            f = max(0, min(out["frames"]-1, int(round(source_frame*24/old_fps)) if legacy else int(source_frame)))
            # Complete channel snapshots, no recursive keys.
            snap = {k: key.get(k, e[k]) for k in CHANNELS if k in e}
            normalized = entity({**snap, "keys": []}, default, ident)
            interpolation = key.get("interpolation") if key.get("interpolation") in ("linear", "smooth", "hold") else out["interpolation"]
            keys[f] = {"frame": f, "interpolation": interpolation, **{k: normalized[k] for k in CHANNELS if k in e}}
            if ident == "camera" and key.get("aim") in ("target", "free"):
                keys[f]["aim"] = key["aim"]
        e["keys"] = [keys[f] for f in sorted(keys)]
        # V3 owns incoming segments; migrate outgoing V1/V2 keys once without
        # altering their motion, including the implicit initial pose at frame 0.
        if outgoing_keys:
            modes = [k["interpolation"] for k in e["keys"]]
            for i, key in enumerate(e["keys"]):
                key["interpolation"] = modes[i-1] if i else out["interpolation"] if key["frame"] > 0 else modes[i]
        return e

    seen = {"camera"}
    subjects = value.get("subjects", [SUBJECT])
    out["subjects"] = []
    for i, raw in enumerate(subjects if isinstance(subjects, list) else [SUBJECT]):
        ident = str(raw.get("id", f"subject_{i+1}")) if isinstance(raw, dict) else f"subject_{i+1}"
        while ident in seen:
            ident += "_copy"
        seen.add(ident)
        out["subjects"].append(entity(raw, SUBJECT, ident))
    out["camera"] = entity(value.get("camera"), CAMERA, "camera")
    if out["camera"]["target"] and out["camera"]["target"] not in seen - {"camera"}:
        out["camera"]["target"] = out["subjects"][0]["id"] if out["subjects"] else ""
    return out


def resolution(scene):
    ratio = RATIOS[scene["aspect_ratio"]]
    pixels = scene["megapixels"] * 1e6
    return tuple(max(32, math.floor(math.sqrt(pixels * r) / 32 + .5)*32) for r in (ratio, 1/ratio))


def sample_entity(entity, frame, interpolation="smooth", *, before=False):
    keys = entity["keys"]
    if not keys:
        return {k: copy.deepcopy(v) for k, v in entity.items() if k != "keys"}
    if keys[0]["frame"] > 0:
        keys = [{"frame": 0, **{k: entity[k] for k in CHANNELS if k in entity}}, *keys]
    result = {k: copy.deepcopy(v) for k, v in entity.items() if k != "keys"}
    if "aim" in entity:
        result["aim"] = next((k["aim"] for k in reversed(entity["keys"])
                              if k.get("aim") in ("target", "free") and
                              (k["frame"] < frame or (not before and k["frame"] == frame) or frame == k["frame"] == 0)), entity["aim"])
    if frame <= keys[0]["frame"] or frame > keys[-1]["frame"] or (frame == keys[-1]["frame"] and not before):
        result.update(keys[0] if frame <= keys[0]["frame"] else keys[-1])
        return result
    i = next(j for j in range(len(keys)-1) if keys[j+1]["frame"] >= frame)
    a, b = keys[i:i+2]
    h = b["frame"]-a["frame"]
    t = (frame-a["frame"])/h
    mode = b.get("interpolation", interpolation)
    for k in CHANNELS:
        if k not in entity:
            continue
        x, y = np.asarray(a[k], dtype=float), np.asarray(b[k], dtype=float)
        if mode == "hold":
            v = y if t >= 1 and not before else x
        elif mode == "linear":
            v = x + (y-x)*t
        else:
            d = (y-x)/h
            def slope(index):
                if index <= 0 or index >= len(keys)-1:
                    return np.zeros_like(d)
                left, mid, right = keys[index-1:index+2]
                if mid.get("interpolation", interpolation) == "hold" or right.get("interpolation", interpolation) == "hold":
                    return np.zeros_like(d)
                dl = (np.asarray(mid[k])-np.asarray(left[k]))/(mid["frame"]-left["frame"])
                dr = (np.asarray(right[k])-np.asarray(mid[k]))/(right["frame"]-mid["frame"])
                # Monotone harmonic slope: zero only on reversing/stationary axes.
                with np.errstate(divide="ignore", invalid="ignore"):
                    return np.where(dl*dr > 0, 2*dl*dr/np.where(abs(dl+dr)>1e-12, dl+dr, 1), 0)
            v = (2*t**3-3*t*t+1)*x + (t**3-2*t*t+t)*h*slope(i) + (-2*t**3+3*t*t)*y + (t**3-t*t)*h*slope(i+1)
        result[k] = v.tolist()
    return result


def rotation_matrix(degrees):
    x, y, z = np.radians(degrees)
    cx, cy, cz, sx, sy, sz = math.cos(x), math.cos(y), math.cos(z), math.sin(x), math.sin(y), math.sin(z)
    return np.array([[cy*cz+sy*sx*sz, -cy*sz+sy*sx*cz, sy*cx],
                     [cx*sz, cx*cz, -sx], [-sy*cz+cy*sx*sz, sy*sz+cy*sx*cz, cy*cx]])


def unit(v):
    return v/max(float(np.linalg.norm(v)), 1e-12)


def evaluate(scene, frame, *, before=False):
    subjects = [sample_entity(s, frame, scene["interpolation"], before=before) for s in scene["subjects"]]
    c = sample_entity(scene["camera"], frame, scene["interpolation"], before=before)
    target = next((s for s in subjects if s["id"] == c["target"]), None)
    local = np.array([0, (subject_geometry(target)["height"] if target else SUBJECT_HEIGHT)*c["target_height"], 0.])
    anchor = (np.array(target["position"]) + rotation_matrix(target["rotation"]) @ (local*np.array(target["scale"]))) if target else np.array([0., 1., 0.])
    p = np.array(c["position"], dtype=float)
    if c["mode"] == "orbit":
        a, e = np.radians([c["azimuth"], c["elevation"]])
        p = anchor + c["radius"]*np.array([math.sin(a)*math.cos(e), math.sin(e), math.cos(a)*math.cos(e)])
    r = rotation_matrix(c["rotation"])
    if c["aim"] == "target":
        f = unit(anchor-p)
        if np.linalg.norm(f) < .5:
            f = np.array([0., 0., -1.])
        # Same lookAt pole perturbation as the JS math, not renderer-dependent.
        if abs(f[1]) > .999999:
            f = unit(f + np.array([0., 0., -1e-5]))
        right = unit(np.cross(f, [0, 1, 0]))
        up = unit(np.cross(right, f))
        roll = math.radians(c["rotation"][2])
        r = np.column_stack((right*math.cos(roll)+up*math.sin(roll),
                             up*math.cos(roll)-right*math.sin(roll), -f))
    return {"camera": c, "subjects": subjects, "position": p, "basis": r, "anchor": anchor,
            "show_grid": scene.get("show_grid", True),
            "show_background_grid": scene.get("show_background_grid", True)}


def project(points, state, aspect):
    local = (np.asarray(points)-state["position"]) @ state["basis"]
    depth = -local[..., 2]
    t = math.tan(math.radians(state["camera"]["fov"])/2)
    return np.stack((local[..., 0]/np.maximum(depth, 1e-6)/t/aspect,
                     local[..., 1]/np.maximum(depth, 1e-6)/t, depth), axis=-1)


def scene_framing(state, aspect):
    """Camera pose and projected framing, even without an aim target.

    Bounds are conservative, not silhouette visibility/occlusion or target-scene
    geometry. Never move the camera or infer a character mapping here.
    """
    basis = state["basis"]
    forward = -basis[:, 2]
    tilt = math.degrees(math.asin(float(np.clip(forward[1], -1, 1))))
    heading = math.degrees(math.atan2(-forward[0], -forward[2]))
    roll = math.degrees(math.atan2(basis[1, 0], basis[1, 1]))
    direction = ("looking horizontally" if abs(tilt) < .05 else
                 f"looking {'upward' if tilt > 0 else 'downward'} by {abs(tilt):.1f} degrees")
    x, y, z = state["position"]
    # Heading is undefined at a vertical pole. Do not turn a singular Euler
    # decomposition into a fake roll; the viewing direction remains well-defined.
    orientation = (f", world heading {heading:.1f} degrees, image roll {roll:+.1f} degrees"
                   if abs(forward[1]) < .9999 else ", nearly vertical lens direction")
    pose = (f"a view {direction}{orientation}, lens height {y:.2f} scene units, "
            f"camera position ({x:.2f}, {y:.2f}, {z:.2f}), "
            f"vertical field of view {state['camera']['fov']:.1f} degrees")
    entries = []
    for s in state["subjects"]:
        points = (subject_bounds(s)*np.array(s["scale"])) @ rotation_matrix(s["rotation"]).T + s["position"]
        q = project(points, state, aspect)
        # A framing target is an identifier, not an appearance/pose instruction.
        # Colors and shapes remain in the render, not in the camera-text channel.
        label = s["name"]
        if np.all(q[:, 2] <= .01):
            entries.append((0., s["id"], f"{label} behind the lens", None))
            continue
        if np.any(q[:, 2] <= .01):
            entries.append((0., s["id"], f"{label} intersects the lens plane; use the render for its crop", None))
            continue
        lo, hi = q[:, :2].min(axis=0), q[:, :2].max(axis=0)
        if np.any(lo > 1) or np.any(hi < -1):
            entries.append((0., s["id"], f"{label} outside the frame", None))
            continue
        clipped_lo, clipped_hi = np.maximum(lo, -1), np.minimum(hi, 1)
        center = (clipped_lo+clipped_hi)/2
        side = "center" if abs(center[0]) < .22 else "left" if center[0] < 0 else "right"
        vertical = "middle" if abs(center[1]) < .33 else "upper" if center[1] > 0 else "lower"
        height = (clipped_hi[1]-clipped_lo[1])*50
        cropped = np.any(lo < -1) or np.any(hi > 1)
        local = rotation_matrix(s["rotation"]).T @ (state["position"]-np.array(s["position"]))
        angle = abs(math.degrees(math.atan2(local[0], local[2])))
        view = "front" if angle < 22.5 else "rear" if angle > 157.5 else "side" if 67.5 <= angle <= 112.5 else "three-quarter"
        facing = f", camera-relative {view} view" if s.get("shape", "human") == "human" else ""
        text = (f"{label} at {side}/{vertical}, {'edge-cropped' if cropped else 'bounds inside frame'}, "
                f"projected bounds cover {height:.0f}% of frame height{facing}")
        area = float(np.prod(clipped_hi-clipped_lo))
        entries.append((area, s["id"], text, (clipped_lo, clipped_hi, q[:, 2].min(), q[:, 2].max(), label)))
    # Keep multi-object scenes bounded. Selected aim target first, then largest
    # visible bounds; identities stay stable, never renumber by screen position.
    entries.sort(key=lambda e: (e[1] != state["camera"]["target"], -e[0], e[1]))
    selected = entries[:4]
    layout = [e[2] for e in selected]
    overlaps = []
    for i, a in enumerate(selected):
        for b in selected[i+1:]:
            if a[3] is None or b[3] is None:
                continue
            al, ah, az0, az1, an = a[3]
            bl, bh, bz0, bz1, bn = b[3]
            if np.all(np.minimum(ah, bh)-np.maximum(al, bl) > .01):
                depth = (f"; {an} is nearer" if az1 < bz0 else
                         f"; {bn} is nearer" if bz1 < az0 else "")
                overlaps.append(f"projected bounding regions of {an} and {bn} overlap{depth}")
    layout.extend(overlaps[:2])
    if len(entries) > len(selected):
        layout.append(f"{len(entries)-len(selected)} additional framing targets not listed")
    if layout:
        pose += "; projected framing: " + "; ".join(layout)
    else:
        pose += "; no framing target"
    return pose


def framing(state, aspect):
    c = state["camera"]
    subject = next((s for s in state["subjects"] if s["id"] == c["target"]), None)
    if subject is None:
        return scene_framing(state, aspect), "no primary target"
    # Named anatomical landmarks: actual proxy projection, not a distance label.
    heights = np.array([0, .5, .9, 1.1, 1.4, 1.6, 1.8])
    pts = np.column_stack((np.zeros(7), heights, np.zeros(7)))
    pts = (pts*np.array(subject["scale"])) @ rotation_matrix(subject["rotation"]).T + subject["position"]
    q = project(pts, state, aspect)
    inside = (abs(q[:, 0]) <= 1) & (abs(q[:, 1]) <= 1) & (q[:, 2] > .01)
    name = subject["name"]
    corners = subject_bounds(subject)
    box = project((corners*np.array(subject["scale"])) @ rotation_matrix(subject["rotation"]).T + subject["position"], state, aspect)
    xs, ys = box[:, 0], box[:, 1]
    front = np.all(box[:, 2] > .01)
    outside = np.all(box[:, 2] <= .01) or (front and (xs.min() > 1 or xs.max() < -1 or ys.min() > 1 or ys.max() < -1))
    occupancy = float(max(np.ptp(xs), np.ptp(ys)))/2
    edges = [edge for clipped, edge in ((xs.min() < -1, "left"), (xs.max() > 1, "right"),
                                       (ys.min() < -1, "bottom"), (ys.max() > 1, "top")) if clipped]
    human = subject.get("shape", "human") == "human"
    size = "an extreme wide shot" if occupancy < .2 else "a wide shot" if occupancy < .55 else "a full shot"
    if outside:
        shot = f"a cropped or offscreen view of {name}"
    elif not front:
        # Projection across the lens plane is not a meaningful shot-size estimate.
        shot = f"a partial view of {name} intersecting the lens plane"
    elif not edges:
        shot = f"{size} containing {name} from head to feet" if human else f"{size} containing the complete object {name}"
    elif occupancy <= 1:
        # Small, off-centre subjects are not close-ups. Keep scale separate from
        # clipping, including the low-camera case where only the crown is outside.
        shot = f"{size} of {name}" if occupancy < .55 else f"a full-shot-scale view of {name}"
    elif not human:
        shot = f"a close-up showing only part of {name}"
    elif not inside.any():
        shot = f"a cropped or offscreen view of {name}"
    elif inside.all():
        shot = f"{size} containing {name} from head to feet"
    elif inside[0] and not inside[4]:
        shot = f"a close view of {name}'s feet and lower body, with the head out of frame"
    elif inside[-1] and inside[1]:
        shot = f"a medium full shot of {name} from the knees upward"
    elif inside[-1] and inside[2]:
        shot = f"a medium shot of {name} from the waist upward"
    elif inside[5] and not inside[2] and not inside[0]:
        shot = f"{'a close-up' if not inside[3] else 'a medium close-up'} of {name}'s head and upper body"
    else:
        shot = f"a partial body view of {name} at the lens's current aim"
    if front and not outside and edges:
        shot += f", with part of the {'subject' if human else 'object'} clipped at the {'/'.join(edges)} frame boundary"
    local = rotation_matrix(subject["rotation"]).T @ (state["position"]-np.array(subject["position"]))
    angle = math.degrees(math.atan2(local[0], local[2]))
    a = abs(angle)
    view = "front view" if a < 22.5 else "rear view" if a > 157.5 else "side-profile view" if 67.5 <= a <= 112.5 else "three-quarter view"
    forward = -state["basis"][:, 2]
    tilt = math.degrees(math.asin(float(np.clip(forward[1], -1, 1))))
    aim = "a horizontal viewing angle" if abs(tilt) < 3 else f"a {'low' if tilt > 0 else 'high'} angle looking {'upward' if tilt > 0 else 'downward'}"
    center = project([state["anchor"]], state, aspect)[0]
    if center[2] > .01 and abs(center[0]) < 1 and abs(center[1]) < 1:
        composition = "centered in the frame" if abs(center[0]) < .22 else f"toward the {'right' if center[0] > 0 else 'left'} of the frame"
        shot += f", {composition},"
    return f"{shot} in a {view} at {aim}", f"lens height {state['position'][1]:.2f}; distance {np.linalg.norm(state['position']-state['anchor']):.2f}"


def camera_view(state, aspect):
    text, _ = framing(state, aspect)
    if any(s["id"] == state["camera"]["target"] for s in state["subjects"]):
        text += "; measured scene view: " + scene_framing(state, aspect)
    return text


def camera_cut_frames(scene):
    """Actual optical discontinuities, including jumps of an orbit/aim target.

    Compare exact left limits, not epsilon samples of fast continuous motion.
    Identical Hold poses (including a wrapped full turn) do not create a cut.
    """
    times = sorted({k["frame"] for e in [scene["camera"], *scene["subjects"]] for k in e["keys"] if k["frame"] > 0})
    return [f for f in times if camera_key_cut(scene, f, evaluate(scene, f, before=True), evaluate(scene, f))]


def camera_key_cut(scene, frame, a, b):
    if not camera_pose_changed(a, b):
        return False
    if a["camera"]["aim"] == b["camera"]["aim"]:
        return True
    key = next((k for k in scene["camera"]["keys"] if k["frame"] == frame), {})
    if key.get("interpolation", scene["interpolation"]) == "hold":
        return True
    # Do not manufacture cuts for a discrete Aim switch alone.
    def inputs(s):
        return [*s["position"], *s["anchor"], *s["camera"]["rotation"], s["camera"]["fov"]]
    return not np.allclose(inputs(a), inputs(b), rtol=0, atol=1e-8)


def camera_pose_changed(a, b):
    return (not np.allclose(a["position"], b["position"], rtol=0, atol=1e-8)
            or not np.allclose(a["basis"], b["basis"], rtol=0, atol=1e-8)
            or abs(a["camera"]["fov"]-b["camera"]["fov"]) > 1e-8)


def key_interpolation(entity, frame, fallback):
    # Mode of the segment departing frame, owned by its next destination key.
    return next((k.get("interpolation", fallback) for k in entity["keys"] if k["frame"] > frame), fallback)


ORBIT_PROMPT_MIN_DEGREES = 25.0


def orbit_prompt_spans(scene, times, states, left_states):
    """Per-interval unwrapped turn of each uninterrupted, same-direction run.

    This controls prose only, never key interpolation or rendering. Subject keys
    can subdivide a camera move, so classify runs rather than individual pieces.
    A stationary interval, reversal, cut or Aim discontinuity starts a new run.
    """
    spans = [0.] * len(left_states)
    run, total, sign = [], 0., 0

    def finish():
        for index in run:
            spans[index] = total

    for i, (a, b, after) in enumerate(zip(states, left_states, states[1:])):
        da = b["camera"]["azimuth"]-a["camera"]["azimuth"] if a["camera"]["mode"] == "orbit" else 0.
        direction = 1 if da > 1e-8 else -1 if da < -1e-8 else 0
        if not direction or direction != sign:
            finish()
            run, total = [], 0.
        if direction:
            run.append(i)
            total += abs(da)
        sign = direction
        if camera_key_cut(scene, times[i+1], b, after) or b["camera"]["aim"] != after["camera"]["aim"]:
            finish()
            run, total, sign = [], 0., 0
    finish()
    return spans


def camera_prompt(scene):
    w, h = resolution(scene)
    times = sorted({0, scene["frames"]-1, *(k["frame"] for e in [scene["camera"], *scene["subjects"]] for k in e["keys"])})
    states = [evaluate(scene, t) for t in times]
    left_states = [evaluate(scene, t, before=True) for t in times[1:]]
    orbit_spans = orbit_prompt_spans(scene, times, states, left_states)
    opening = camera_view(states[0], w/h)
    lines = [f"[Shot 1] The camera opens on {opening}."]
    lines.append({
        'auto': 'Tracking reference: Auto; resolve from explicit user instructions, without inventing subject travel.',
        'follow': 'Tracking reference for all connected Moves: follow the selected target throughout each take, translating with its user-described travel WHILE performing the following compatible orbit, approach and height changes relative to it. A held relative camera pose continues following; never invent target locomotion.',
        'world': 'Tracking reference for all connected Moves: world-relative, without following target translation. Apply the following camera travel in scene space; held camera position stays world-fixed while the subject may move or leave frame.'
    }[scene.get('tracking', 'auto')] + ' Explicit user camera instructions override this setting. Measured coordinates below are proxy geometry, not a measured real-world target trajectory.')
    def target_description(state):
        camera = state["camera"]
        subject = next((s for s in state["subjects"] if s["id"] == camera["target"]), None)
        if not subject or (camera["aim"] != "target" and camera["mode"] != "orbit"):
            return ""
        role = "aim point" if camera["aim"] == "target" else "orbit pivot"
        return f"The {role} is at normalized height {camera['target_height']:.3f} on {subject['name']} (0 = base, 1 = top, along the subject's own height)."
    if target_description(states[0]):
        lines.append(target_description(states[0]))
    def stamp(frame):
        sec = frame/scene["fps"]
        return f"{int(sec//60):02d}:{sec%60:06.3f}"
    previous = {}
    shot_number = 1
    aim_warning = False
    has_travel = False
    for index, (fa, fb, a, b) in enumerate(zip(times, times[1:], states, states[1:])):
        after = b
        b = left_states[index]
        cut = camera_key_cut(scene, fb, b, after)
        mode = key_interpolation(scene["camera"], fa, scene["interpolation"])
        ca, cb = a["camera"], b["camera"]
        path, deltas = [], {}
        if ca["mode"] == "orbit" and ca["aim"] == "free" and abs(cb["target_height"]-ca["target_height"]) > 1e-6 and any(s["id"] == ca["target"] for s in a["subjects"]):
            path.append(f"shifts its orbit pivot along the subject's height from {ca['target_height']:.3f} to {cb['target_height']:.3f} (0 = base, 1 = top), preserving free lens rotation")
        displacement = b["position"]-a["position"]
        if ca["mode"] == "orbit":
            da = cb["azimuth"]-ca["azimuth"]
            de = cb["elevation"]-ca["elevation"]
            dr = cb["radius"]-ca["radius"]
            if abs(da) > .01:
                # Azimuth is world-relative; Free aim, roll and over-pole
                # elevation can reverse its relation to screen-right. Project
                # the orbit tangent onto the actual starting lens basis.
                az, el = np.radians([ca["azimuth"], ca["elevation"]])
                tangent = np.sign(da)*np.array([math.cos(az)*math.cos(el), 0., -math.sin(az)*math.cos(el)])
                lateral = float(np.dot(tangent, a["basis"][:, 0]))
                direction = f"toward the {'right' if lateral > 0 else 'left'} of its starting camera view " if abs(lateral) > 1e-6 else ""
                significant_orbit = orbit_spans[index] >= ORBIT_PROMPT_MIN_DEGREES-1e-8
                if significant_orbit:
                    path.append(f"physically travels {direction}along a {abs(da):.1f}-degree arc around {next((s['name'] for s in a['subjects'] if s['id'] == ca['target']), 'the target')}")
                else:
                    path.append(f"physically shifts {direction}with only a slight change in viewpoint")
                if not direction and significant_orbit:
                    path.append(f"follows {'increasing' if da > 0 else 'decreasing'} world-space azimuth rather than a screen-horizontal path")
                if significant_orbit and previous.get("orbital travel", 0)*da > 0:
                    path.append("continues in the same orbital direction as the preceding interval")
                axis = "orbital travel" if significant_orbit else "lateral travel" if direction else "viewpoint adjustment"
                # Reversals can cross the prose threshold; keep the actual axis
                # history rather than losing a reversal when its label changes.
                preceding_turn = next((previous[k] for k in ("orbital travel", "lateral travel", "viewpoint adjustment") if k in previous), 0)
                if preceding_turn:
                    previous[axis] = preceding_turn
                deltas[axis] = da
                if abs(da) >= 359.99:
                    path.append(f"preserves the full {abs(da):.1f}-degree turn rather than taking a shorter route")
            if abs(dr) > .001:
                path.append("pulls back" if dr > 0 else "pushes in")
                deltas["radial travel"] = dr
            if abs(de) > .01:
                path.append(f"{'ascends' if de > 0 else 'descends'} along the vertical orbit")
                deltas["vertical orbit"] = de
            if np.linalg.norm(b["anchor"]-a["anchor"]) > .001:
                path.append("follows the animated target point while preserving the keyed orbit offsets")
        else:
            # Separate physical height from horizontal travel. A tilted lens's
            # forward vector would otherwise count vertical translation twice.
            backward = a["basis"][:, 2].copy()
            backward[1] = 0
            if np.linalg.norm(backward) < 1e-6:
                yaw = math.radians(ca["rotation"][1])
                backward = np.array([math.sin(yaw), 0., math.cos(yaw)])
            backward = unit(backward)
            right = np.cross([0, 1, 0], backward)
            components = np.array([np.dot(displacement, right), displacement[1], np.dot(displacement, backward)])
            for val, axis, pos, neg in [(components[0], "lateral travel", "trucks right", "trucks left"),
                                       (components[2], "dolly travel", "pulls back", "pushes in"),
                                       (displacement[1], "height travel", "raises the lens", "lowers the lens")]:
                if abs(val) > .001:
                    path.append(pos if val > 0 else neg)
                    deltas[axis] = val
        # Aim is independently evaluated, including a stationary camera tracking a subject.
        f0, f1 = -a["basis"][:, 2], -b["basis"][:, 2]
        tilt = math.degrees(math.asin(float(np.clip(f1[1], -1, 1)))-math.asin(float(np.clip(f0[1], -1, 1))))
        if abs(tilt) > .1:
            path.append("tilts upward" if tilt > 0 else "tilts downward")
            deltas["tilt"] = tilt
        if ca["aim"] == "target":
            if abs(cb["target_height"]-ca["target_height"]) > 1e-6 and any(s["id"] == ca["target"] for s in a["subjects"]):
                path.append(f"shifts its aim point along the subject's height from {ca['target_height']:.3f} to {cb['target_height']:.3f} (0 = base, 1 = top), keeping the lens aimed at that moving point")
            else:
                path.append("keeps the lens aimed at the target point")
            roll = cb["rotation"][2]-ca["rotation"][2]
            if abs(roll) > .1:
                path.append(f"rolls the camera by {roll:+.1f} degrees around the lens axis")
                deltas["roll"] = roll
        else:
            for j, label in [(0, "pitch"), (1, "pan"), (2, "roll")]:
                d = cb["rotation"][j]-ca["rotation"][j]
                if abs(d) > .1:
                    path.append(f"rotates its {label} by {d:+.1f} degrees")
                    deltas[label] = d
        zoom = cb["fov"]-ca["fov"]
        if abs(zoom) > .01:
            path.append("zooms out by widening the lens field of view" if zoom > 0 else "zooms in by narrowing the lens field of view")
            deltas["lens zoom"] = zoom
        moving = np.linalg.norm(displacement) > .001 or any(abs(v) > .001 for v in deltas.values()) or abs(zoom) > .01
        has_travel = has_travel or moving or np.linalg.norm(f1-f0) >= .001
        reversed_axes = [k for k, v in deltas.items() if previous.get(k, 0)*v < 0]
        prefix = f"From {stamp(fa)} to {stamp(fb)}, without a cut, the same camera "
        if reversed_axes and mode == "smooth":
            prefix += "smoothly decelerates and reverses only its " + ", ".join(reversed_axes) + "; it "
        if not moving and np.linalg.norm(f1-f0) < .001:
            sentence = {
                'follow': 'tracks the target translation continuously, maintaining its relative position, orientation and lens while the target performs the user-requested action',
                'world': 'holds its world-space position, orientation and lens unchanged unless explicit user camera text overrides this setting',
                'auto': 'holds its keyed position, orientation and lens in the reference frame resolved from user camera instructions; requested tracking continues with unchanged relative offsets'
            }[scene.get('tracking', 'auto')]
        else:
            sentence = path[0] if len(path) == 1 else ", ".join(path[:-1]) + ", and " + path[-1]
        endpoint = camera_view(b, w/h)
        unchanged = not moving and np.linalg.norm(f1-f0) < .001
        lines.append(prefix + sentence + ("." if unchanged else f", reaching {endpoint} by {stamp(fb)}."))
        if moving:
            if mode == "linear":
                lines.append("The camera's keyed channels interpolate linearly during this interval, without an added pause.")
            elif mode == "smooth":
                lines.append("The camera's keyed channels follow a smooth curve, easing to zero on reversing axes and beside Hold boundaries; an instantaneous zero speed is not a stationary interval. Do not add a pause.")
            else:
                lines.append("The camera's own keyed channels remain held; any tracking follows only the moving target.")
            if abs(zoom) <= .01:
                lines.append("Keep the lens focal length fixed during this movement.")
        # Subject keys still drive projection, target tracking and camera cuts.
        # Their actions/poses/scaling are not camera instructions. Reference-role
        # transfer and explicit user action are handled independently downstream.
        previous = deltas
        if cut:
            shot_number += 1
            endpoint = camera_view(after, w/h)
            lines.append(f"[Shot {shot_number}] At {stamp(fb)}, the shot cuts to {endpoint}. The Hold key changes the camera view instantaneously to this new camera state; do not interpolate travel across this cut. Any following hold keeps this shot's new state, not the preceding shot's framing.")
            if target_description(after):
                lines.append(target_description(after))
            previous = {}
        if b["camera"]["aim"] != after["camera"]["aim"]:
            label = "target tracking" if after["camera"]["aim"] == "target" else "free rotation, releasing target tracking"
            lines.append(f"At {stamp(fb)}, the camera switches to {label}." + ("" if cut else " This Aim change stays within the same Shot."))
            if not cut and camera_pose_changed(b, after) and not aim_warning:
                LOG.warning("H3 Camera: Aim switches abruptly at frame %s; keeping the requested take without adding a cut.", fb)
                aim_warning = True
    if all(abs(s["camera"]["fov"]-states[0]["camera"]["fov"]) < .01 for s in states):
        lines.append("The lens focal length remains fixed throughout.")
    if has_travel:
        lines.append("Within each Shot, preserve the described continuous camera travel and timed stationary intervals; only the explicitly marked Hold cuts start a new Shot.")
    else:
        lines.append("Each Shot holds its keyed camera offsets in the resolved tracking reference frame; only explicitly marked Hold cuts change those offsets instantaneously. A held offset does not cancel Follow target or explicit user tracking; world-relative holds keep the camera in place. Explicit user camera text overrides this default throughout the take.")
    return " ".join(lines)


def shade_surface(normals, color):
    """View-space studio shading, matching subjectMaterial's GLSL in the UI."""
    n = normals/np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    key, fill = unit(np.array([-.45, .65, 1.])), unit(np.array([.7, .1, .5]))
    half_vector = unit(key + [0, 0, 1])
    light = .22 + .65*np.maximum(n @ key, 0) + .10*np.maximum(n @ fill, 0)
    highlight = .10*np.maximum(n @ half_vector, 0)**24
    linear = np.where(color <= .04045, color/12.92, ((color+.055)/1.055)**2.4)
    lit = linear*light[..., None] + highlight[..., None]
    return np.clip(np.where(lit <= .0031308, lit*12.92, 1.055*lit**(1/2.4)-.055), 0, 1)


def background_grid(rays, pixel_angle):
    """World-oriented sky sphere at infinity; no translation parallax.

    15-degree latitude/longitude lines, matching the preview shader. The sky
    follows camera position only, never rotation, and cannot occlude subjects.
    """
    direction = rays / np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-12)
    longitude = np.arctan2(direction[..., 0], direction[..., 2])
    latitude = np.arcsin(np.clip(direction[..., 1], -1, 1))
    horizontal = np.linalg.norm(direction[..., [0, 2]], axis=-1)
    distance = np.minimum(abs(np.sin(12*longitude))*horizontal/12, abs(np.sin(12*latitude))/12)
    aa = max(pixel_angle*.65, .00005)
    coverage = np.clip((.0012+aa-distance)/(2*aa), 0, 1)
    coverage = coverage*coverage*(3-2*coverage)
    # Avoid the meridians converging into a filled cap at the poles.
    coverage *= np.clip(horizontal/.03, 0, 1)
    return np.array([.055, .078, .098]) + coverage[..., None]*np.array([.16, .18, .20])


def render_frame(state, width, height):
    """Analytic sphere/box ray intersections. RGB proxy colors match the editor.

    No GPU/model memory needed. Work row chunks to avoid high-res ray buffers.
    """
    t = math.tan(math.radians(state["camera"]["fov"])/2)
    image = np.empty((height, width, 3), dtype=np.float32)
    for y0 in range(0, height, 64):
        y1 = min(height, y0+64)
        xx, yy = np.meshgrid((np.arange(width)+.5)/width*2-1, 1-(np.arange(y0, y1)+.5)/height*2)
        rays = np.stack((xx*t*width/height, yy*t, -np.ones_like(xx)), -1) @ state["basis"].T
        origin = state["position"]
        depth = np.full(xx.shape, np.inf)
        rgb = np.broadcast_to(np.array([.055, .078, .098]), (*xx.shape, 3)).copy()
        if state.get("show_background_grid", True):
            rgb = background_grid(rays, 2*t/height)
        if state.get("show_grid", True):
            with np.errstate(divide="ignore", invalid="ignore"):
                ground = -origin[1]/rays[..., 1]
            with np.errstate(invalid="ignore"):
                p = origin + rays*ground[..., None]
            floor = (ground > .01) & (ground < 500) & (abs(p[..., 0]) <= 20) & (abs(p[..., 2]) <= 20)
            grid = (abs(p[..., 0]-np.round(p[..., 0])) < .018) | (abs(p[..., 2]-np.round(p[..., 2])) < .018)
        for subject in state["subjects"]:
            r = rotation_matrix(subject["rotation"])
            scale = np.array(subject["scale"])
            o = (origin-np.array(subject["position"])) @ r / scale
            d = rays @ r / scale
            color = np.array([int(subject["color"][j:j+2], 16)/255 for j in (1, 3, 5)])
            def paint(valid, hit, local_normals, marker=False):
                # Inverse-transpose handles nonuniformly scaled/rotated subjects.
                normals = (local_normals/scale) @ r.T
                normals = np.where((normals*rays[valid]).sum(axis=-1, keepdims=True)>0, -normals, normals)
                rgb[valid] = 1.0 if marker else shade_surface(normals @ state["basis"], color)
                depth[valid] = hit[valid]

            geometry = subject_geometry(subject)
            boxes = [(lo, hi, False) for lo, hi in geometry["boxes"]]
            if subject.get("shape", "human") == "human":
                boxes.extend((lo, hi, True) for lo, hi in HUMAN_FACE_BOXES)
            for lower, upper, marker in boxes:
                lower, upper = np.array(lower), np.array(upper)
                with np.errstate(divide="ignore", invalid="ignore"):
                    lo = (lower-o)/d
                    hi = (upper-o)/d
                near = np.minimum(lo, hi).max(axis=-1)
                far = np.maximum(lo, hi).min(axis=-1)
                hit = np.where(near > .01, near, far)
                valid = (far >= near) & (hit > .01) & (hit < depth) & (hit < 500)
                point = o+d[valid]*hit[valid, None]
                relative = (point-(lower+upper)/2)/((upper-lower)/2)
                axis = np.argmax(abs(relative), axis=-1)
                normals = np.zeros_like(relative)
                normals[np.arange(len(axis)), axis] = np.sign(relative[np.arange(len(axis)), axis])
                paint(valid, hit, normals, marker)
            for center, radius in geometry["spheres"]:
                oc = o-np.array(center)
                aa = (d*d).sum(axis=-1)
                bb = (d*oc).sum(axis=-1)
                cc = (oc*oc).sum()-radius**2
                disc = bb*bb-aa*cc
                root = np.sqrt(np.maximum(0, disc))
                hit = (-bb-root)/aa
                hit = np.where(hit > .01, hit, (-bb+root)/aa)
                valid = (disc >= 0) & (hit > .01) & (hit < depth) & (hit < 500)
                paint(valid, hit, o+d[valid]*hit[valid, None]-center)
        # Opaque floor: hide the sky and any subject geometry behind the ground.
        if state.get("show_grid", True):
            visible_floor = floor & (ground < depth)
            rgb[visible_floor] = np.array([26, 33, 38])/255
            visible_grid = visible_floor & grid
            rgb[visible_grid] = np.array([59, 77, 87])/255
        image[y0:y1] = rgb
    return image


def proxy_subject_metadata(scene):
    """Bounded, projected source evidence; no raw animation channels or IMAGE copies."""
    frames = sorted({0, (scene["frames"]-1)//2, scene["frames"]-1, *[k["frame"] for entity in
                    [scene["camera"], *scene["subjects"]] for k in entity["keys"]]})
    if len(frames) > 8:
        frames = [frames[i] for i in np.linspace(0, len(frames)-1, 8).round().astype(int)]
    states = [(f, evaluate(scene, f)) for f in frames]
    w, h = resolution(scene)
    objects = []
    for index, subject in enumerate(scene["subjects"]):
        item = {k: copy.deepcopy(subject[k]) for k in ("id", "name", "shape", "color")}
        item["selector"] = f"proxy {index + 1}"
        item["samples"] = []
        for frame, state in states:
            s = state["subjects"][index]
            points = (subject_bounds(s)*np.array(s["scale"])) @ rotation_matrix(s["rotation"]).T + s["position"]
            q = project(points, state, w/h)
            row = {"frame": frame,
                   "world_position": [round(float(v), 3) for v in s['position']],
                   "rotation_degrees": [round(float(v), 2) for v in s['rotation']],
                   "scale": [round(float(v), 3) for v in s['scale']],
                   "depth_range": [round(float(q[:, 2].min()), 3), round(float(q[:, 2].max()), 3)]}
            if np.all(q[:, 2] <= .01):
                row["view"] = "behind camera"
            elif np.any(q[:, 2] <= .01):
                row["view"] = "near-plane intersection; screen bounds uncertain"
            else:
                lo, hi = q[:, :2].min(axis=0), q[:, :2].max(axis=0)
                x = float((lo[0]+hi[0])/2)
                row["center_x"] = round((x+1)/2, 3)
                row["screen_bounds"] = [round(float((lo[0]+1)/2), 3), round(float((1-hi[1])/2), 3),
                                        round(float((hi[0]+1)/2), 3), round(float((1-lo[1])/2), 3)]
                side = "left" if x < -.1 else "right" if x > .1 else "center"
                outside = lo[0] > 1 or hi[0] < -1 or lo[1] > 1 or hi[1] < -1
                partial = np.any(lo < -1) or np.any(hi > 1)
                row["view"] = ("offscreen " if outside else "partly cropped " if partial else "in-frame ") + side
            item["samples"].append(row)
        objects.append(item)
    return objects


class MinimaxH3Camera:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scene_data": ("STRING", {"default": json.dumps(DEFAULT_SCENE), "multiline": True})}}

    RETURN_TYPES = ("MINIMAX_H3_CAMERA", "IMAGE", "STRING")
    RETURN_NAMES = ("prompter_camera", "camera_render", "camera_prompt")
    FUNCTION = "compile"
    CATEGORY = "toyxyz/MiniMax H3"
    DESCRIPTION = "Keyframe a proxy camera and subjects. Outputs procedural camera prose and camera-view IMAGE frames, without loading Qwen."

    def compile(self, scene_data):
        import torch
        import comfy.utils
        import comfy.model_management
        scene = normalize_scene(scene_data)
        w, h = resolution(scene)
        count = scene["frames"]
        gib = count*w*h*12/1024**3
        LOG.info("H3 Camera: rendering %d frames at %dx%d (%.2f GiB IMAGE output).", count, w, h, gib)
        if __package__:
            from .h3_video_memory import allocate_images
        else:
            from h3_video_memory import allocate_images
        result, storage = allocate_images((count, h, w, 3))
        LOG.info("H3 Camera: %s float32 storage; no duplicate render buffer.", storage)
        progress = comfy.utils.ProgressBar(count)
        for f in range(count):
            comfy.model_management.throw_exception_if_processing_interrupted()
            result[f].copy_(torch.from_numpy(render_frame(evaluate(scene, f), w, h)))
            progress.update(1)
        prompt = camera_prompt(scene)
        bundle = {"type": "minimax_h3_camera", "version": 2, "frame_count": count,
                  "fps": scene["fps"], "render_signature": scene_render_signature(scene), "refvid": scene["refvid"]}
        if scene["refvid"]:
            bundle["images"] = result
        if scene["use_camera_prompt"]:
            bundle.update(use_camera_prompt=True, procedural_camera_prompt=prompt,
                          final_camera_view=camera_view(evaluate(scene, count-1), w/h),
                          cut_frames=camera_cut_frames(scene))
        return {"ui": {"camera_prompt": [prompt], "camera_info": [f"{w}×{h} · {count} frames · {scene['fps']:g} fps"]},
                "result": (bundle, result, prompt)}
