# lattice-compute

Host-side compute for Lattice inference: rotary position embedding (RoPE) tables,
flat/paged KV caches with prefix reuse, and LoRA adapter hooks and row application.
CPU kernels, attention, pooling and sampling move here in later releases.
`lattice-inference` re-exports the supported items at their original paths.

The crate depends on `lattice-inference-core` for the shared error type and on
`lattice-model-config` for the Qwen3.5 geometry used by the LoRA hook contract.
