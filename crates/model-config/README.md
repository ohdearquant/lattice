# lattice-model-config

Model configurations for Lattice inference. Each parser reads a checkpoint's `config.json`
through a bounded reader, requires every forward-relevant field, and rejects values outside
the bounds the engine supports, for the Qwen3.5 hybrid models, Gemma 4, EmbeddingGemma 2 and
BitNet b1.58. It also holds the Qwen3 embedding model configuration and the tensor names a
Qwen3.5 configuration requires of a checkpoint. `lattice-inference` re-exports these items at
their original paths.
