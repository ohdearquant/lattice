# lattice-transport — extended docs

Deeper documentation for the crate, kept alongside the source so it stays current
as the code changes. The crate-level rustdoc (`src/lib.rs`) and `README.md` hold
the summary and public API surface; the files here hold the longer-form narrative.

- [design.md](design.md) — architecture layers, the log-domain invariant and why it
  matters, workspace preallocation, and the no-external-linear-algebra constraint.
