# lattice-embed extended documentation

These guides hold the design narrative behind the public Rust API. Start with the architecture
guide, then follow the subsystem relevant to the code or operational decision at hand.

| Guide                        | Covers                                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------------------- |
| [design.md](design.md)       | Crate architecture, component boundaries, cache identity, migration fit, and WebAssembly bindings |
| [simd.md](simd.md)           | SIMD dispatch, numerical behavior, and quantized-vector operations                                |
| [model.md](model.md)         | Supported models, model configuration, prompts, pooling, provenance, and MRL dimensions           |
| [backfill.md](backfill.md)   | Backfill coordination, routing policy, dual writes, and batch progression                         |
| [migration.md](migration.md) | Migration plan/state lifecycle, progress, and failure handling                                    |
| [service.md](service.md)     | Async service contract, native loading, cache wrapper, validation, and error handling             |

The crate-level Rust documentation is the short entry point; API-level contracts remain beside
their types and functions in `src/`.
