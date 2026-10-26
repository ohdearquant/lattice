# lattice-compute

Host-side compute for Lattice inference: rotary position embedding (RoPE) tables
(including the Gemma 4 text and vision inverse-frequency tables),
flat/paged KV caches with prefix reuse, LoRA adapter hooks and row application, and
the numerical diagnostics (metrics accumulators and block-influence pruning scores).
It also holds option-letter scoring over a full-vocabulary logit row, shared
attention row semantics, the CPU forward kernels, BERT-family pooling, and the CPU
attention kernels that build only on those primitives (grouped-query, single-token decode,
differential, causal flash, sigmoid output gating and native sparse attention), and the
GatedDeltaNet recurrent kernels.
Sampling moves here in a later release.
`lattice-inference` re-exports the supported items at their original paths.

The crate depends on `lattice-inference-core` for the shared error type and on
`lattice-model-config` for the Qwen3.5 geometry used by the LoRA hook contract. Option
scoring adds a dependency on `lattice-tokenizer` for the `Tokenizer` trait, and the CPU
F16 matmul one on `lattice-weights` for the F16 storage type.
