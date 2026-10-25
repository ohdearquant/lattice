# lattice-compute

Host-side compute for Lattice inference. It currently holds the rotary position embedding
(RoPE) tables that the decoder forward passes apply to queries and keys; the CPU kernels,
attention, pooling, KV caches and sampling move here in later releases. `lattice-inference`
re-exports these items at their original paths.
