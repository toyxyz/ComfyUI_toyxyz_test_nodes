# Danbooru tag snapshot for Anima

`danbooru-2026-09-24.csv` comes from
[HDiffusion/historical-danbooru-tag-counts](https://huggingface.co/datasets/HDiffusion/historical-danbooru-tag-counts),
revision `79e7d75fcef571b7c9049db4659d1cc9e3970ce9`, licensed Apache-2.0.

SHA-256: `1f64a73ac7e11b12d78d89eb5b9fc73525ee4347a182733f1db01b9cc5c85dcd`

This pinned snapshot contains 125,312 rows. It validates generated tag names
offline; it does not establish that every tag is understood by a given Anima
checkpoint. Refresh the snapshot and revision together when updating it.

`danbooru_wiki.sqlite3` is downloaded on first Wiki use from
`https://huggingface.co/toyxyz/backup_models/resolve/main/danbooru_wiki.sqlite3`
into this directory. It is not tracked in Git. Subsequent Wiki use is offline;
the file is opened read-only and is not modified by node execution. SHA-256:
`c17b31e0f6468d2a0d5094452085925643d8442a2433f798f74f26923758bac6`.
The snapshot is separate from the tag autocomplete CSV and does not update
automatically.
