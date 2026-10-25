# lattice-compute

Host-side compute for Lattice inference: rotary position embedding (RoPE) tables
(including the Gemma 4 text and vision inverse-frequency tables),
flat/paged KV caches with prefix reuse, LoRA adapter hooks and row application, and
the numerical diagnostics (metrics accumulators and block-influence pruning scores).
It also holds option-letter scoring over a full-vocabulary logit row and shared
attention row semantics.
CPU kernels, pooling and sampling move here in later releases.
`lattice-inference` re-exports the supported items at their original paths.

The crate depends on `lattice-inference-core` for the shared error type and on
`lattice-model-config` for the Qwen3.5 geometry used by the LoRA hook contract. Option
scoring adds a dependency on `lattice-tokenizer` for the `Tokenizer` trait.
