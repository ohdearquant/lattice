# lattice-embed — extended docs

Deeper documentation for the crate, kept alongside the source so it stays current
as the code changes. The crate-level rustdoc (`src/lib.rs`) and `README.md` hold
the summary and public API surface; the files here hold the longer-form narrative.

- [design.md](design.md) — module architecture and the model-migration/backfill
  workflow (`migration` + `backfill` modules): the state machine, the routing table,
  and the four-stage migration sequence.
