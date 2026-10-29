# lattice-backward

Reverse-mode gradients (vector-Jacobian products) for LoRA training on Lattice transformer layers:
linear and LoRA projections, RMSNorm, SwiGLU, RoPE, gated grouped-query attention and the
GatedDeltaNet recurrence. Each backward has a matching forward that keeps the activations it needs,
and `gradcheck` checks the whole chain against central finite differences on a tiny model.

The CPU kernels come from `lattice-compute` and the Qwen3.5 configuration from `lattice-arch`.
`lattice-inference` re-exports this crate at `lattice_inference::backward` and
`lattice_inference::attention::gdn_backward` under its `train-backward` feature for code written
before the split; those paths are compatibility re-exports and will be removed at 1.0.
