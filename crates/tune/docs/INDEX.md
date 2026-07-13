# lattice-tune — extended docs

Deeper documentation for the crate, kept alongside the source so it stays current
as the code changes. The crate-level rustdoc (`src/lib.rs`) and `README.md` hold
the summary and public API surface; the files here hold the longer-form narrative.

- [design.md](design.md) — the distillation/training/registry pipeline, walkthroughs
  for each stage, the `inference-hook` bridge, and the `Checkpoint` JSON schema.
- `adr/`, `ADR-*.md` — accepted architecture decisions for this crate (unchanged by
  this index; see each ADR for the alternatives considered and why).
