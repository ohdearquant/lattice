# lattice-model-config

Model configurations for Lattice inference. Each parser reads a checkpoint's `config.json`
through a bounded reader, requires every forward-relevant field, and rejects values outside
the bounds the engine supports, for the Qwen3.5 hybrid models, Gemma 4, EmbeddingGemma 2 and
BitNet b1.58. `lattice-inference` re-exports these types at their original paths.
