# Embedding migration controller

## Scope

The migration module is the bookkeeping layer for a transition from one embedding model
version to another. A plan names a source model and a target model, and a controller tracks
the lifecycle and work budget for that one transition.

It deliberately does not embed text, write vectors, create an index, validate that the two
models are distinct, route reads or writes, or persist itself. The backfill coordinator uses
this controller to implement model routing; an application owns durable work cursors,
storage namespaces, retry policy, and deployment-level cutover actions.

This boundary matters because vector spaces are versioned data. A model swap can change
semantic geometry as well as vector dimension, so source and target embeddings must not be
mixed in an index merely because they describe the same documents. The controller records
that a transition exists; it cannot make an application's index layout safe by itself.

## Public data model

### Migration plan

<code>MigrationPlan</code> is the immutable input used to construct a controller:

| Field | Meaning |
| --- | --- |
| <code>id</code> | Caller-provided migration identifier. No uniqueness check is performed. |
| <code>source_model</code> | The model version being replaced. |
| <code>target_model</code> | The model version being introduced. |
| <code>total_embeddings</code> | Initial historical-work budget. |
| <code>batch_size</code> | Descriptive preferred processing batch size. The controller does not schedule batches. |
| <code>created_at</code> | Caller-supplied ISO 8601 timestamp string. It is stored without parsing or validation. |

The plan does not prove that source and target differ, that their dimensions are compatible
with any existing index, or that total_embeddings matches a live dataset. Perform those
checks before constructing the controller.

### Lifecycle state

<code>MigrationState</code> is serializable and non-exhaustive. Downstream code must include
a wildcard case when matching it so future states can be added without making integrations
invalid.

| State | Stored information | Interpretation |
| --- | --- | --- |
| <code>Planned</code> | none | Created but not started. |
| <code>InProgress</code> | processed, total, skipped | Active work budget. |
| <code>Paused</code> | processed, total, skipped, reason | Stopped intentionally; may resume. |
| <code>Failed</code> | processed, total, skipped, error | Stopped due to a failure; may resume. |
| <code>Completed</code> | processed, skipped, duration_secs | Work was reported complete. |
| <code>Cancelled</code> | processed, total, skipped | Abandoned by the caller. |

<code>Completed</code> and <code>Cancelled</code> are terminal according to
<code>is_terminal()</code>. A failure is not terminal: <code>is_resumable()</code> is true
for both paused and failed states. Only <code>InProgress</code> is active.

### Skips and progress reports

<code>SkipReason</code> records a permanent reason not to embed one entry:

- content exceeded its maximum allowed byte size;
- text used an invalid or unsupported encoding;
- the source content was deleted;
- an embedding API returned a non-retryable error; or
- a caller intentionally skipped the item.

<code>MigrationProgress</code> is a snapshot, not a mutable checkpoint. It includes the
migration identifier, cloned lifecycle state, skipped count, effective total and coverage,
active throughput, optional estimated remaining seconds, and a count of non-fatal errors.
<code>record_error</code> only increments that error count; it does not change lifecycle
state or retain an error message.

<code>MigrationError</code> currently reports an attempted invalid state transition, with
debug-formatted source and destination descriptions. It is also non-exhaustive, so callers
must not assume this is the only possible error category.

## State machine

The controller accepts only these lifecycle operations:

~~~text
Planned ── start ──> InProgress ── progress reaches effective total ──> Completed
   │                       │  │
   │ cancel                │  └── fail ──> Failed ── resume ─┐
   ▼                       │                                  │
Cancelled <── cancel ──────┼── pause ──> Paused ── resume ────┘
                           │
                           └── cancel ──> Cancelled
~~~

The allowed calls are:

| Method | Accepted source state | Result |
| --- | --- | --- |
| <code>start</code> | Planned | InProgress with zero processed and skipped counts. |
| <code>record_progress</code> | InProgress | Updates processed count or completes the migration. |
| <code>record_skip</code> | InProgress | Adds one permanent skip when capacity remains. |
| <code>pause</code> | InProgress | Paused, retaining all counts and a reason. |
| <code>fail</code> | InProgress | Failed, retaining all counts and an error string. |
| <code>resume</code> | Paused or Failed | InProgress, retaining all counts. |
| <code>cancel</code> | Planned, InProgress, Paused, or Failed | Cancelled, retaining counts available in that state. |

Starting twice, progressing while inactive, pausing an inactive migration, resuming planned
or terminal work, and cancelling a completed or already cancelled migration all return an
invalid-transition error. There is no transition out of <code>Completed</code> or
<code>Cancelled</code>.

## Work accounting

### Raw progress

For planned state, <code>progress()</code> returns 0.0. For completed state it returns 1.0.
For active, paused, failed, and cancelled states it calculates:

    processed / total

When total is zero, it returns 1.0 rather than dividing by zero. The controller preserves an
overshoot supplied to <code>record_progress</code>; it does not clamp processed to total.
Consequently, raw progress in a non-completed state can exceed 1.0 if a caller reports more
work than its original budget. Callers should keep their external work cursor consistent
with the plan instead of relying on this type to normalize counts.

### Effective total and coverage

Permanent skips reduce the number of entries that actually require an embedding:

    effective_total = saturating_sub(total, skipped)
    effective_coverage = processed / effective_total

Effective coverage is 1.0 when effective total is zero. Like raw progress, it retains an
overshoot rather than clamping the numerator. This measure is useful for worker accounting,
but it has a representation caveat: the completed state does not store the original total.
Its <code>total()</code> therefore returns zero, and a progress snapshot produced after
completion reports an effective total of zero. Preserve the plan separately if a completed
report must display the original or effective work budget.

### Completion rule

<code>record_progress(newly_processed)</code> updates:

    new_processed = processed + newly_processed

It completes when <code>new_processed >= total - skipped</code>. Completion stores the
reported processed count, skipped count, and elapsed duration. Reporting an oversized final
batch is therefore accepted and retains the oversized processed value.

<code>record_skip</code> never completes a migration by itself. It is accepted only while:

    processed + skipped < total

This guard prevents a duplicate or retried skip from exceeding the budget. If skips reduce
effective total to zero, call <code>record_progress(0)</code> to trigger the normal
completion rule. This is intentional: only the progress operation turns active work into
the completed state.

## Timing, throughput, and ETA

<code>start</code> records a process-local <code>Instant</code>. While the state is active,
throughput is:

    processed / elapsed_seconds_since_start

The estimated remaining time is calculated only when throughput is positive:

    (total - skipped - processed) / throughput

All subtractions used for remaining work saturate at zero. Paused, failed, completed,
cancelled, and planned snapshots report zero throughput and no ETA.

The timer is not paused when a migration enters <code>Paused</code> or <code>Failed</code>.
On resume, the original start time remains when one exists. The resulting rate is elapsed
wall time across the whole controller lifetime, including interruptions, rather than pure
active-worker time. Because <code>Instant</code> is in-memory, timing information cannot be
reconstructed from a serialized state alone.

## Serialization and version evolution

<code>MigrationState</code> uses snake-case variant names when serialized. Its state variants
also accept their former PascalCase names as deserialization aliases. This allows persisted
state written before the naming change to remain readable.

The <code>skipped</code> field defaults to zero when it is absent from serialized
in-progress, paused, failed, completed, or cancelled states. The skip-related fields on
<code>MigrationProgress</code> also have serde defaults, so older progress payloads may
deserialize without them. An absent value means zero at deserialization time; it does not
recover information that was never persisted.

The controller itself is not serializable. It owns an <code>Instant</code>, non-fatal error
count, and in-memory vector of skip reasons in addition to its plan and state. An application
that needs restart recovery must define how it persists the plan, state, work cursor, and any
skip audit data, then construct and drive a controller consistently with that external
checkpoint.

## Relationship to backfill routing

The controller knows nothing about source versus target index aliases. It can reach
<code>Completed</code> without changing which model handles a request. The backfill
coordinator is the layer that:

1. dual-writes newly created documents while historical records receive target embeddings;
2. chooses a target query route at a configured coverage threshold;
3. records a cutover moment when it observes completion; and
4. keeps source writes alive during a post-cutover rollback window.

This division prevents migration accounting from being coupled to one storage topology, but
it also means a successful controller state alone is not proof that target data is queryable.
Before using completion as a serving cutover, an application should verify target-index
coverage, model and dimension compatibility, durable persistence of processed work, and the
desired handling of permanently skipped entries.

See [backfill.md](backfill.md) for the embedding-specific routing and batch policy built on
top of this state machine.
