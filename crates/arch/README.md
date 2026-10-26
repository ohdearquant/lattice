# lattice-arch

The model interface for Lattice inference: the generation request and result types
(`GenerateConfig`, `GenerateOutput`, token log-probabilities), the stop-reason taxonomy,
sampling (`SamplingConfig`, `Sampler`), and the decode policy every decode loop runs. A
generation request can carry a grammar, so the crate depends on `lattice-grammar`.

Until the model families move into their own crates, it also holds their configurations.
Each parser reads a checkpoint's `config.json` through a bounded reader, requires every
forward-relevant field, and rejects values outside the bounds the engine supports, for the
Qwen3.5 hybrid models, Gemma 4, EmbeddingGemma 2 and BitNet b1.58. It also holds the Qwen3
embedding model configuration and the tensor names a Qwen3.5 configuration requires of a
checkpoint. `lattice-inference` re-exports these items at their original paths.
