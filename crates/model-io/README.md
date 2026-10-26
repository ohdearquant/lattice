# lattice-model-io

Model-file input and output shared by the Lattice inference crates: the model cache
directory (`LATTICE_MODEL_CACHE`, default `~/.lattice/models`), checksum-verified
model downloads (`download::ensure_model_files`, network access behind the `download`
feature and refused when `LATTICE_OFFLINE` is set), and the by-value reader for boolean
`LATTICE_*` environment switches. `lattice-inference` re-exports the download entry
point at its original path, `lattice_inference::download::ensure_model_files`.
