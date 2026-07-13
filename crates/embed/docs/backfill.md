# Embedding backfill

## Purpose and boundary

Changing an embedding model changes the vector space used by an index. A source embedding
and a target embedding can differ in model family, dimension, tokenization, instruction
format, or all of those at once. They must therefore be treated as separate versions of the
same document data, not as vectors that can be mixed in one index.

The backfill subsystem makes that transition gradual. It has one narrow responsibility:
given the migration state, decide which model or models an application should use for reads
and writes, and report how much historical work remains. It does not fetch documents, invoke
an embedding service, persist vectors, create indexes, schedule workers, or checkpoint
application state. Those operations belong to the caller.

<code>BackfillCoordinator</code> wraps a
<code>MigrationController</code>. The controller owns lifecycle transitions and work
accounting; the coordinator adds embedding-specific routing, batch sizing, and a
post-cutover rollback window. This separation lets the migration state machine remain
independent of an application's storage and query implementation.

## The transition being protected

A safe model change has two distinct races:

1. Historical documents need target embeddings before their target index can be complete.
2. New documents can arrive while historical data is being re-embedded.

Backfill handles the second race with dual writes. While a migration is active, a new
document can be embedded with both source and target models, so the target index does not
fall behind while historical records are processed. A completed migration can keep doing
that for a bounded rollback period. The application must maintain distinct source and target
index namespaces; the coordinator selects models but does not provide that storage isolation.

The migration plan identifies the source and target models, total historical embedding
count, preferred plan batch size, identifier, and creation timestamp. The coordinator has
its own <code>BackfillConfig</code>, so its processing batch policy need not be identical to
the plan's descriptive batch size.

## Lifecycle ownership

Construct a coordinator with a plan and a configuration, then call <code>start</code>. That
delegates the <code>Planned → InProgress</code> transition to the controller. The coordinator
then exposes the same pause, resume, cancel, state, and progress operations.

The coordinator records the cutover time only when its own <code>record_batch</code> call
causes the underlying controller to become <code>Completed</code>. The timestamp uses
<code>Instant</code>, so it measures elapsed time in the current process and is not a
persisted wall-clock cutover record. Constructing a new coordinator starts with no cutover
timestamp, even if an external system has retained migration metadata.

<code>BackfillCoordinator</code> is a policy layer, not a worker pool. In particular,
<code>max_concurrent</code> is stored in configuration but this type does not start, limit,
or join concurrent batches. A caller that runs workers must apply the limit and make the
read-embed-write operation idempotent for its own storage system.

## Routing interfaces

The coordinator offers two levels of routing information:

- <code>route_request(is_new_document)</code> returns one
  <code>EmbeddingRoute</code> for a write-like embedding request.
- <code>route_query()</code> returns one <code>EmbeddingRoute</code> for a query.
- <code>routing_config()</code> returns the complete read model, write-model set, phase, and
  optional migration identifier. Use this form when the application must carry out dual
  writes or respect the rollback window.

An <code>EmbeddingRoute</code> is an instruction to the caller:

| Route | Required application behavior |
| --- | --- |
| <code>Legacy</code> | Use only the source model and source index. |
| <code>Target</code> | Use only the target model and target index. |
| <code>DualWrite</code> | Produce and persist both source and target embeddings. |

Returning <code>DualWrite</code> does not itself issue two embedding calls or commit two
index writes.

### Coarse request routing

<code>route_request</code> distinguishes new documents from existing documents. The
existing-document case represents ordinary re-embedding rather than the historical work that
the backfill worker is processing.

| Migration state | New document | Existing document |
| --- | --- | --- |
| Planned | Legacy | Legacy |
| InProgress, dual write enabled | DualWrite | Legacy |
| InProgress, dual write disabled | Legacy | Legacy |
| Completed | Target | Target |
| Paused, Failed, or Cancelled | Legacy | Legacy |

Disabling <code>dual_write</code> means newly indexed documents are not automatically added
to the target index while the migration is in progress. The caller must accept that coverage
gap or provide a separate reconciliation path.

This helper is intentionally coarser than <code>routing_config</code>: once state is
<code>Completed</code>, it returns <code>Target</code> even during the rollback window. A
write path that needs the source copy preserved during that window must use the full routing
configuration rather than treating this single route as a complete write plan.

### Query routing and the threshold

<code>route_query</code> chooses the target only after completion or while an active
migration has reached <code>target_query_threshold</code>. Its in-progress calculation is:

    processed / total

When <code>total</code> is zero, it treats progress as 1.0 and returns the target route.
Every non-active state, including <code>Paused</code> and <code>Failed</code>, returns the
legacy route regardless of any prior progress.

The threshold comparison uses the raw processed and total counts, not effective coverage
after permanent skips. If skips matter to an application's cutover policy, it must inspect
the migration progress itself and choose a threshold policy deliberately.

### Full read/write configuration

<code>EmbeddingRoutingConfig</code> is the authoritative representation for an application
that manages separate query and write paths:

| Controller state and timing | Query model | Write models | Phase | Migration ID |
| --- | --- | --- | --- | --- |
| Planned | source | source | Stable | absent |
| InProgress, dual write enabled | source | source, target | Migrating | present |
| InProgress, dual write disabled | source | source | Migrating | present |
| Completed, rollback window open | target | source, target | RollbackWindow | present |
| Completed, rollback window elapsed | target | target | Stable | absent |
| Paused, Failed, or Cancelled | source | source | Stable | absent |

The rollback-window row always writes both models; it does not depend on the
<code>dual_write</code> setting that applies during <code>InProgress</code>. That preserves
all documents created after query cutover in both indexes, allowing a caller to restore
source queries without losing those newly written records.

<code>RoutingPhase::Stable</code> describes the current routing arrangement, not necessarily
a successful completed migration. It is also returned for planned, paused, failed, and
cancelled states. Code that needs to know the actual lifecycle state must read
<code>state()</code>; a missing migration identifier likewise means there is no active
routing transition in this configuration.

## Batch accounting

The coordinator does not select documents. It tells a worker how many entries it may take
next:

    remaining = saturating_sub(saturating_sub(total, processed), skipped)
    next_batch_size = min(remaining, configured_batch_size)

The result is zero unless the controller is in <code>InProgress</code>. Saturating
subtraction prevents an underflow if a caller has supplied counts that exceed the original
budget. The formula matches the controller's completion threshold, whose effective total is
<code>total - skipped</code>.

A typical worker loop is:

~~~rust
coordinator.start()?;

loop {
    let batch_size = coordinator.next_batch_size();
    if batch_size == 0 {
        break;
    }

    // Load at most batch_size historical records, embed with the target model,
    // and commit their target-index entries before reporting the completed count.
    coordinator.record_batch(batch_size)?;
}
~~~

The sketch only shows coordinator calls. A production worker must use a durable work cursor
or equivalent claim mechanism; otherwise retries can embed or count the same record more
than once. It must report only records that have reached the application's intended durable
write boundary.

<code>record_batch(count)</code> adds <code>count</code> to
<code>backfilled_count</code> and delegates to the controller's
<code>record_progress</code>. It also records the cutover timestamp if that progress call
transitions from active to completed. The count is incremented before the delegated call can
return an invalid-transition error, so callers should invoke it only while active and should
not use it as a transactional acknowledgement of storage work.

The underlying migration state supports permanently skipped entries, and
<code>next_batch_size</code> accounts for their count. The public coordinator facade does
not expose a method to record an individual skip; applications that require skip reporting
must account for that limitation when integrating the controller and coordinator APIs.

## Rollback window

Completion is a query cutover, not immediate retirement of the source index. For
<code>rollback_window_secs</code> after the coordinator observes completion:

- queries use the target model;
- writes continue to source and target;
- the routing phase is <code>RollbackWindow</code>;
- the migration identifier remains available in the routing configuration.

After the elapsed interval, routing becomes stable target-only and the migration identifier
is omitted. A zero-second window expires immediately. Before completion, or if completion
was not observed through <code>record_batch</code>, <code>in_rollback_window()</code> is
false.

The coordinator only preserves a source copy during the window. It does not provide a
method to reverse index aliases, delete target data, or re-open the controller's completed
state. An application rollback procedure must supply those storage and serving actions.

## Configuration

<code>BackfillConfig::default()</code> uses the following policy:

| Field | Default | Meaning |
| --- | ---: | --- |
| <code>batch_size</code> | 100 | Maximum entries suggested by <code>next_batch_size</code>. |
| <code>max_concurrent</code> | 4 | Concurrency budget for the caller to enforce. |
| <code>dual_write</code> | true | Dual-write new documents while active. |
| <code>target_query_threshold</code> | 0.8 | Raw progress fraction at which <code>route_query</code> may choose target. |
| <code>rollback_window_secs</code> | 86,400 | Time to retain source writes after completed cutover. |

The configuration and full routing configuration are serializable with snake-case phase
names. A serialized routing configuration includes concrete model selections and should be
treated as a snapshot: recompute it from a live coordinator whenever the lifecycle state or
rollback timer may have changed.

## Operational checks

Before starting a transition, ensure that:

1. Source and target embeddings will be written to distinct, versioned index locations.
2. The target index has the schema and dimension required by the target model.
3. New-document writes honor <code>routing_config</code>, not just a single coarse route,
   when rollback is enabled.
4. The backfill worker persists a progress cursor outside this in-memory coordinator.
5. The query threshold reflects the application's tolerance for incomplete target coverage,
   including its policy for skipped entries.
6. Source indexes are retained at least through the intended rollback period and any
   external validation period.

The coordinator provides deterministic routing from its local state. Correct migration
outcomes still depend on the caller making embedding writes, index updates, cursor commits,
and query aliases agree with that routing decision.

## BackfillCoordinator::route_request

`route_request(is_new_document)` is the coordinator's coarse routing helper for a write-like
embedding request. `is_new_document` means the caller is indexing the document for the first
time. `false` denotes ordinary re-embedding and does not describe the historical backfill
worker's own work.

While the controller is `InProgress`, a new document returns `DualWrite` only when
`dual_write` is enabled; the caller must then create and persist source and target embeddings.
Every other active request returns `Legacy`. Planned, paused, failed, and cancelled migrations
also return `Legacy`, while completed migrations return `Target`. The helper never performs
the requested embedding or index writes.

This route deliberately omits the post-cutover source-write requirement. It returns `Target`
for a completed migration even during the rollback window, so applications that need to retain
the source copy must use `routing_config()` instead. That configuration is the complete
read/write plan because it exposes both write models and the `RollbackWindow` phase.

## BackfillCoordinator::record_batch

Call `record_batch(count)` only after the caller has completed the storage work it intends to
claim. The coordinator adds `count` to its independent `backfilled_count`, then delegates to
the migration controller's `record_progress(count)`. The counter update happens before the
delegated operation can return an invalid-transition error, so this method is not a
transactional acknowledgement and a failed call does not roll that counter back.

If the delegated progress call changes the controller from `InProgress` to `Completed`, the
coordinator records a process-local `Instant` as `cutover_at`. That timestamp enables the
rollback-window policy; it is not a durable or externally meaningful cutover record. Directly
changing lifecycle state through another path cannot set this timestamp, so the coordinator
only recognises cutover that it observed through this method.

## BackfillCoordinator::next_batch_size

`next_batch_size()` gives a worker an upper bound for the next historical batch; it does not
select or reserve documents. During `InProgress`, it computes:

    remaining = saturating_sub(saturating_sub(total, processed), skipped)
    next_batch_size = min(remaining, configured_batch_size)

It returns zero in every other lifecycle state. Saturating subtraction makes an oversupplied
processed or skipped count yield no further work rather than an underflow. The formula follows
the controller's effective-total completion rule, so a worker that honours it will not request
more entries than remain after permanent skips. It still needs a durable cursor or claim
mechanism to prevent concurrent workers or retries from processing the same document.
