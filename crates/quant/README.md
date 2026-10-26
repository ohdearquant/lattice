# lattice-quant

Quantization for Lattice inference: bounded Q4 checkpoint manifest reading,
Walsh-Hadamard transforms, seeded randomized and block Hadamard rotations, rotation
absorption for weight matrices, and the offline QuaRot conversion of Qwen3.5
checkpoints (rotation plan, RMSNorm fusion, LM-head and LoRA handling, the
source-tensor reader and the online-rotation artifact descriptor). Internal runtime
dependencies are `lattice-inference-core`, `lattice-weights` and `lattice-arch`.
`lattice-inference` retains every existing `quant::q4_manifest` and
`quant::quarot::*` API path.
