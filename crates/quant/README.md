# lattice-quant

Quantization primitives for Lattice inference: bounded Q4 checkpoint manifest
reading, Walsh-Hadamard transforms, seeded randomized and block Hadamard rotations,
and rotation absorption for weight matrices. Internal runtime dependencies are
`lattice-inference-core` and `lattice-weights`. `lattice-inference` retains the
existing `quant::q4_manifest`, `quant::quarot::hadamard` and
`quant::quarot::rotation` API paths.
