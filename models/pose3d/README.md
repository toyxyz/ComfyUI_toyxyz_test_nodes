# Local SMPL bodies

Legacy SMPL workflows may read `female.npz` and `male.npz` only from this folder.
New nodes use the bundled MHR body and have no body-type selector.
Provide your licensed SMPL or SMPL-H templates with numeric arrays:
`v_template` (N,3), `f` (F,3), `weights` (N,J), dense `J_regressor` (J,N),
and `kintree_table` (2,J). Pickled models are not loaded.

The editor uses the neutral template and linear blend skinning, not shape or
pose-corrective blendshapes. SMPL-H includes articulated fingers; SMPL alone
has rigid hands. Without these assets the editor uses bundled MHR LOD1 meshes
with 127 native bones and articulated fingers. MHR is SAM 3D Body's human
representation, not SMPL. See `web/vendor/pose3d/README.md` for provenance and
limitations. MakeHuman assets and workflow compatibility are no longer included.

Model files are user supplied and are not bundled or downloaded automatically.
See https://smpl.is.tue.mpg.de/ and https://mano.is.tue.mpg.de/ for models and terms.
