# ADR-019: Backfill and Migration Pipeline

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-embed

## Context

When the system switches from one embedding model to another (e.g., BGE-small → BGE-base
for higher quality), three operations must happen concurrently:

1. Existing stored embeddings must be re-embedded using the new model (backfill).
2. New documents written during the migration must be indexed in both models so neither
   index goes stale.
3. Queries must remain fast throughout — switching to the new model prematurely would
   produce results against an incomplete index.

Additionally, after cutover, a rollback window is needed: if the new model produces
unexpected quality regressions, the system must be able to revert without losing any
atoms created after cutover.

## Decision

The backfill subsystem is a two-layer design. `MigrationController` (in `src/migration/`)
implements a state machine. `BackfillCoordinator` (in `src/backfill/`) wraps it and adds
routing logic on top.

### Key Design Choices

**`MigrationController` state machine**

The underlying state machine transitions through:
`Planned → InProgress{processed, total, errors} → Completed | Paused | Failed | Cancelled`

Legal transitions are enforced at the type level — `start()`, `pause(reason)`, `resume()`,
`cancel()`, and `record_progress(count)` return `MigrationError::InvalidTransition` when
called in an inappropriate state. There is no direct `InProgress → Planned` back-edge:
once started, a migration either completes, fails, is paused (resumable), or is cancelled (terminal).

**`BackfillCoordinator` routing logic**

Routing decisions are functions of `MigrationState` and coordinator configuration,
not data in the atoms themselves. The routing matrix is:

| State                         | New doc route | Existing doc route | Query route                                |
| ----------------------------- | ------------- | ------------------ | ------------------------------------------ |
| Planned                       | Legacy        | Legacy             | Legacy                                     |
| InProgress (dual_write=true)  | DualWrite     | Legacy             | Legacy or Target (if progress ≥ threshold) |
| InProgress (dual_write=false) | Legacy        | Legacy             | Legacy                                     |
| Paused                        | Legacy        | Legacy             | Legacy                                     |
| Completed (in window)         | n/a           | n/a                | Target                                     |
| Completed (stable)            | Target        | Target             | Target                                     |
| Failed / Cancelled            | Legacy        | Legacy             | Legacy                                     |

`route_query()` switches to `EmbeddingRoute::Target` during `InProgress` once
`processed / total >= config.target_query_threshold` (default 0.8). This allows
query re-routing before full backfill completion.

**`EmbeddingRoutingConfig` for rollback-safe write routing**

`routing_config()` returns a struct with both `query_model` and `write_models` (a `Vec`).
During the rollback window (completed migration, within `rollback_window_secs`, default 24 hours),
`write_models = [source, target]` even though `query_model = target`. This means atoms
created after cutover exist in both indexes — instant rollback can switch `query_model` back
to `source` without data loss.

**Three `RoutingPhase` values**

`RoutingPhase::Stable` (no migration active), `Migrating` (active), `RollbackWindow`
(post-cutover dual-write). The phase is included in `EmbeddingRoutingConfig` so monitoring
systems can observe migration progress without inspecting the full coordinator state.

**`BackfillConfig` fields**

```rust
pub struct BackfillConfig {
    pub batch_size: usize,           // default: 100
    pub max_concurrent: usize,       // default: 4
    pub dual_write: bool,            // default: true
    pub target_query_threshold: f64, // default: 0.8
    pub rollback_window_secs: u64,   // default: 86400 (24 hours)
}
```

`next_batch_size()` returns `min(remaining, batch_size)`, returning 0 when not `InProgress`.
This gives a clean "are we done" signal to the caller's batch loop.

**Cutover timestamp via `Instant`**

When `record_batch()` causes `processed == total` and the state transitions to `Completed`,
`cutover_at = Some(Instant::now())` is recorded. `in_rollback_window()` checks
`cutover_at.elapsed() < rollback_window`. `Instant` is not serializable, so the rollback
window resets on process restart — this is intentional and accepted.

### Alternatives Considered

| Alternative                                     | Pros                                     | Cons                                                                                              | Why Not                                                         |
| ----------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Blue-green index swap                           | Atomic cutover; no dual-write complexity | Requires 2x peak storage during migration                                                         | Storage cost too high for large embedding indexes               |
| Shadow index (write to new only, no dual-write) | Simpler routing                          | New atoms created during migration are absent from legacy index; rollback requires re-backfilling | Silent data loss during rollback                                |
| Event-sourced routing (per-atom migration flag) | Precise, no false dual-writes            | Schema change required; per-atom storage overhead                                                 | Architectural change too large; coordinator state is sufficient |
| Migration as a database transaction             | ACID guarantees                          | Embedding indexes are not transactional                                                           | No transactional index API in scope                             |

## Consequences

### Positive

- `route_request(is_new_document)` and `route_query()` are `&self` methods — no locking required. The coordinator is read from multiple async tasks safely.
- The rollback window dual-write ensures zero data loss for the 24-hour post-cutover period with no changes to the write path after the rollback window closes.
- `record_error()` increments an error counter without changing state, so transient per-document embedding failures don't abort the migration.

### Negative

- The coordinator is not persistent. A process restart in the middle of a migration requires the caller to restore the coordinator state (processed count, total) from an external store before resuming. The `MigrationPlan` and `MigrationProgress` types are serializable (`serde`) for this purpose.
- `in_rollback_window()` uses `Instant`, which is process-local. After a restart, `cutover_at` is `None` and `in_rollback_window()` returns `false` immediately — the rollback window effectively ends on process restart.
- `max_concurrent` is stored in `BackfillConfig` but the coordinator itself does not enforce concurrency limits. The field exists as a hint to the caller's batch executor.

### Risks

- `DualWrite` routing during `InProgress` writes to both indexes for every new document. If the target model is slow (e.g., Qwen3-4B), write latency doubles. The caller should detect high-latency target embeddings and either pause the migration or switch `dual_write=false`.
- `target_query_threshold=0.8` means 20% of the index is still unbackfilled when queries switch to the new model. Near-duplicate documents for items in that 20% will be missed until backfill completes. For most workloads this is acceptable; latency-critical or high-recall use cases should set the threshold to 1.0.

## References

- [`crates/embed/src/backfill/coordinator.rs`](/Users/lion/projects/lattice/crates/embed/src/backfill/coordinator.rs) — `BackfillCoordinator` implementation
- [`crates/embed/src/backfill/types.rs`](/Users/lion/projects/lattice/crates/embed/src/backfill/types.rs) — `EmbeddingRoute`, `RoutingPhase`, `BackfillConfig`, `EmbeddingRoutingConfig`
- [`crates/embed/src/migration/controller.rs`](/Users/lion/projects/lattice/crates/embed/src/migration/controller.rs) — `MigrationController` state machine
- [`crates/embed/src/migration/types.rs`](/Users/lion/projects/lattice/crates/embed/src/migration/types.rs) — `MigrationPlan`, `MigrationProgress`, `MigrationState`
