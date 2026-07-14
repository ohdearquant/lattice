# lattice-fann documentation

This directory contains the long-form design reference for `lattice-fann`.
Crate rustdoc and public item documentation remain the API entry points; these
pages explain the algorithms, invariants, formats, and operational boundaries.

- [design.md](design.md) — crate architecture and how the builder, layers,
  network, training module, and optional GPU path fit together.
- [network.md](network.md) — dense-layer mechanics, preallocated activation
  buffers, activation behavior, SIMD dispatch, validation, and binary format.
- [training.md](training.md) — backpropagation, optimizer behavior, and
  gradient safeguards.
- [gpu.md](gpu.md) — optional GPU execution, resources, dispatch policy, and
  CPU fallback behavior.
