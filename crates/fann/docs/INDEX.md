# lattice-fann — extended docs

Deeper documentation for the crate, kept alongside the source so it stays current
as the code changes. The crate-level rustdoc (`src/lib.rs`) and `README.md` hold
the summary and public API surface; the files here hold the longer-form narrative.

- [design.md](design.md) — network/training architecture, the zero-allocation
  inference design, and the GPU backend (buffer pooling, circuit breaker,
  Apple Silicon tuning).
