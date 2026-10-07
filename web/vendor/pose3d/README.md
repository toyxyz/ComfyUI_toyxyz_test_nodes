# 3D Pose Editor body assets

## Default: MHR (Apache-2.0)

`mhr_female.json.gz` and `mhr_male.json.gz` contain baked LOD1 meshes from
[Meta's Momentum Human Rig](https://github.com/facebookresearch/MHR), the human
representation used by SAM 3D Body. Each includes 18,439 vertices, 36,874
triangles, all 127 native bones, their parent-local bind transforms and all four
skin influences. Only anatomical control bones are shown as editor markers.
The MHR license is included as `LICENSE-MHR.txt`.

These are modified, fixed illustrative adult presets, not official sex-specific
MHR models. Their torso proportions were fitted in MHR's own identity space to
the previous adult presets; no MakeHuman mesh topology or rig is transferred.
Head and hand identity components are neutral in the default preset. The MHR
shape schema and preset coefficients are recorded in each asset.
`mhr_identity45.bin.gz` contains the 45 neutral bind-space displacement axes as
component-major little-endian float32 (45 × 18,439 × 3), in meters. It is about
2.8 MB compressed and supports the editor's body/head/hand shape controls.
It is derived from the Apache-2.0 MHR model, not the SAM inference checkpoint.
There is no male/female selector in the node. New nodes use one default preset;
the other asset remains available only to preserve older saved workflows.
Shape is mixed on the CPU into the rest-surface buffer before browser skinning;
normals are recomputed after edits. Native identity coefficients do not change
the rig's bone lengths. This is not a semantic height/weight or limb-length UI.

The bake uses the local `sam3dbody/assets/mhr_model.pt`. Face anchors use the
sparse MHR keypoint mapping from the existing SAM 3D Body checkpoint, rather
than image-based pose detection. This small mapping is a separate SAM-licensed
component; its terms are included in `LICENSE-SAM.txt` and attribution in
`NOTICE-MHR.txt`. The MHR mesh/rig itself remains Apache-2.0.
No SAM inference model, Python MHR dependency,
remote service or runtime download is required by the editor. Runtime assets
remain inside this custom node.

Rendering uses browser linear blend skinning with native procedural twist
relationships and two-sided opaque surfaces. MHR's pose-dependent neural
correctives and facial expression controls are not implemented, so extreme
poses may deform differently from the full MHR model.

## Implementation reference

The [VNCCS Pose Studio](https://github.com/AHEKOT/ComfyUI_VNCCS_Utils)
analytic IK, pose persistence, and browser capture architecture
informed the editor improvements. Its application code is MIT (MiuProject,
2025); no Pose Studio application module is bundled here.
