# lattice-weights

Weight storage and model-directory format detection for Lattice inference. Supports
safetensors loading for single-file and sharded checkpoints, f32 and f16 weight
representations, and Q3, Q4 and Q8 storage and conversion. Its only internal runtime
dependency is `lattice-inference-core`; `lattice-inference` retains the existing
`weights` and `model_format` API paths.
