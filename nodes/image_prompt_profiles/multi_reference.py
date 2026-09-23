"""Source binding for multi-reference scene writing, independent of scene content."""

ANALYSIS = """MULTIPLE REFERENCES: These are separate sources, not panels of one shared scene.
Under each supplied image label, identify that source's subjects and bind each subject's appearance,
clothing colors/materials, accessories and held objects to that subject in that source. Do not propagate
one source's outfit, identity, pose or background to another. A multi-view design sheet shows multiple
views of one design, not an instruction to add multiple people. The user request describes desired
changes, not observed facts. Report observations first; never describe a transferred attribute as if
already present in the destination. Inspect all supplied sources, including those after the first."""

WRITER = """MULTI-REFERENCE SOURCE BINDING — user instructions remain highest priority.
Reference evidence contains separate numbered sources, NOT one common scene draft. Interpret 'image 1',
'이미지 1', 'image_1' and '<image1>' as the same supplied source number, likewise other numbers.
Internally resolve each requested source role: base scene, subject identity, clothing/object donor,
style or background. Bind every resulting subject to its own source appearance, outfit and held objects.
When adding a source person, carry that person's distinguishing outfit/accessories/objects unless the
user changes or excludes them; do not give everyone the base person's clothes or merge their identities.
An attribute transfer changes only the named recipient and attribute. A clothing donor supplies clothing,
not additional people or its backdrop. Replace the recipient's incompatible old clothing rather than
layering contradictory outfits. A requested new action overrides that source's static pose, not identity
or unrelated clothing. Preserve the user's coordinate frame and each action's actor and target.
Import backgrounds or other source elements only as requested or needed for the selected base scene.
When modifying an existing scene and adding source subjects to it, keep that base setting and composition:
subject donors contribute their selected subjects, not their floors, architecture, landscape or studio.
Do not blend donor backgrounds into the distance to make sources appear unified. A coherent scene places
the selected subjects in the base environment; it is not a collage of source settings unless requested.
Preserve the base canvas orientation rather than switching to a donor portrait's orientation.
Do not duplicate people from reference-sheet views. Enhancement may develop open details, never erase
source bindings. Before finalizing, check each requested source is used in its assigned role, each actor
has the correct appearance and possessions, and replacements remove incompatible prior attributes.
Write one self-contained resulting scene, explicitly describing the selected visible source features;
do not output image labels, a reference inventory, or editing instructions. If a referenced feature is
absent/uncertain, do not invent it or silently remap its source number."""
