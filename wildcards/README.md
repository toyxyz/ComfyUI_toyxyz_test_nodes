# Wildcards

Place UTF-8 `.txt` files here. Each non-empty line is one complete random option.
Lines may contain multiple tags, sentences, weights, or nested wildcards.

`cloth.txt` is called with `__cloth__`; `outfits/cloth.txt` with `__outfits/cloth__`.
`(__cloth__:1.5)` keeps the weight around the selected text.
Blank lines are skipped. Missing, empty, unreadable, or cyclic wildcards stay literal
instead of stopping generation. Each occurrence samples independently on each run.
Use Refresh in the node's Wildcards tab after adding or removing files.
In the prompt field, type `__` to search wildcard filenames inline; select a
suggestion to insert its `__name__` call.

To use another folder, set **Settings → toyxyz_test_nodes → Wildcards →
Wildcard folder path** to an existing absolute directory on the ComfyUI
computer, or click Browse to select one in a native Windows folder picker.
Leave the path blank to use this bundled folder. The sidebar and Folder
button follow the active directory. The setting is server-wide for this node
installation and is saved across restarts; it is not stored in workflows.
