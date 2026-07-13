# Model and adapter registry

`lattice-tune`'s registry is an in-process catalog of versioned models and their
weight artifacts. It separates a model's descriptive record from bytes held by a
storage backend, supports lock-free catalog reads, and provides helpers for
shadow evaluation, rollback audit records, and atomic live-record replacement.

The registry does not load or execute a model. A `RegisteredModel` is metadata,
and `LiveModel` atomically replaces that metadata record. An application that
keeps instantiated inference models must coordinate its own model loading with a
registry update.

For the original choice of an in-process, lineage-aware registry and its
rejected alternatives, see [ADR-002](ADR-002-model-registry.md).

## Components and boundaries

```text
training output
      |
      v
RegisteredModel + weight bytes
      |
      v
ModelRegistry -----> StorageBackend
      |                    |
      |                    +-- in memory / filesystem / SQLite
      |
      +-- ShadowSession       (compare candidate with production)
      +-- RollbackController  (record a rollback)
      +-- LiveModel           (atomically replace a record)
```

`ModelRegistry` is the catalog and consistency boundary for the process in
which it is created. A storage backend persists artifacts, but constructing a
new registry does not hydrate its indexes from that backend. In particular,
creating a registry over an existing directory or SQLite database does not make
pre-existing rows or files visible through `get`, `list_*`, or checksum loading
until they have been registered in that registry instance.

The deployment helpers are deliberately separate:

- A `ShadowSession` collects and evaluates comparison observations. It never
  promotes a candidate.
- A `RollbackController` records a rollback decision. It never changes a
  `ModelRegistry` or a `LiveModel`.
- A `LiveModel` changes one `RegisteredModel` pointer. It neither persists the
  change nor updates registry status.

An application normally evaluates a candidate, promotes the chosen registry
record, swaps its serving record or loaded model, and records any rollback as
separate, explicit actions.

## The model record

### Identity, versioning, and lineage

Each `RegisteredModel` has a UUID `id` and a human-facing `(name, version)`
identity. The registry's duplicate key is the string `"{name}:{version}"`;
registration rejects an existing key while holding the writer lock. Validation
only requires both strings to be non-empty. A caller that uses filesystem or
SQLite storage must also supply path-safe names and versions, because those
backends make them directory components.

`version_tuple()` recognizes only exactly three dot-separated unsigned integer
components. It returns `None` for prefixes, suffixes, pre-release identifiers,
or any other non-`MAJOR.MINOR.PATCH` form. `get_latest` orders parseable
versions by that tuple and treats an unparsable version as `(0, 0, 0)`. Use
strict numeric three-part versions whenever automated latest-version selection
matters.

`parent_id` records the immediate source of a fine-tuned model and `children`
holds descendant IDs. The registry does not validate that a parent exists,
populate a parent's `children`, or cascade deletion. Applications that use
lineage must maintain those relations and account for orphaned references
themselves.

### Metadata and status

`ModelMetadata` records architecture, input and output dimensions, parameter
count, optional configuration and dataset identifiers, example count, training
metrics, validation and test accuracy, tags, and an extensible `extra` value.
With `serde`, `extra` is JSON; without it, it is a string. The `classifier`
constructor fills architecture and dimension/count fields, and builder methods
set architecture, dataset, metrics, and tags.

`training_metrics` retains the supplied metrics and sets `validation_accuracy`
only when `final_val_loss` is present; it then uses the most recent history
entry's validation accuracy, falling back to `0.0` if that entry has none.
Callers that need a different metric summary should set the public metadata
fields explicitly.

The status vocabulary is:

| Status | Meaning |
| --- | --- |
| `Pending` | Registered but not yet validated. |
| `Validated` | Validation has passed. |
| `Staged` | Prepared for deployment. |
| `Production` | The selected production version. |
| `Archived` | Superseded or retained for history. |
| `Deprecated` | Not for further use. |

The conventional lifecycle is `Pending -> Validated -> Staged -> Production`,
with old versions later archived or deprecated. This is convention, not a
state-machine enforcement layer: the status helper methods and `update_status`
can set any status, and `promote_to_production` does not first require the
target to be deployable. `is_deployable` reports true for `Validated`,
`Staged`, and `Production`; `is_active` reports false only for `Archived` and
`Deprecated`.

`RegisteredModel` also carries its registration and update timestamps, optional
registrant and description, and optional weight path, byte size, and SHA-256
hash. Builder methods that alter model content update `updated_at`; setting the
registrant does not. A model created with `new` starts as `Pending` with a new
UUID and both timestamps set to creation time.

## Registry operations and concurrency

`ModelRegistry` owns two immutable `ArcSwap` indexes:

- `UUID -> RegisteredModel` for direct lookup and enumeration.
- `"name:version" -> UUID` for lookup by model identity.

Read methods take snapshots without the writer lock and return owned cloned
records. A reader therefore never borrows internal map storage and may keep a
record after a writer publishes a newer map. `get_by_id`, `get`, `get_latest`,
`get_production`, the `list_*` methods, `len`, and `is_empty` are all lock-free
at the registry level.

Writes take one `write_lock`, clone the affected index map, modify the clone,
and atomically store the new `Arc`. The storage backend itself is behind a mutex
because saving and deletion require mutable backend access. This serializes
registry writers and prevents two writers from publishing updates derived from
the same old map; it does not make backend persistence and index publication a
cross-resource transaction.

### Registration and deletion ordering

`register(model, weights)` follows this order:

1. Validate the non-empty name and version.
2. Acquire the writer lock and reject a duplicate full name.
3. Ask the backend to save the bytes.
4. Compute the SHA-256 over those exact input bytes and attach the backend's
   returned relative path, size, and hash to the model record.
5. Publish the enriched record in both in-memory indexes.

If backend saving fails, neither index is changed. `register_metadata` follows
the same duplicate and index rules but creates no artifact or weight metadata.
The model passed to a backend is the pre-enrichment record; a backend that
persists metadata receives it before the outer registry attaches the returned
path, size, and checksum.

`delete(id)` first removes the record from a cloned map, then asks the backend
to delete its weight path (when one exists), and publishes the changed indexes
only after that operation succeeds. Thus a backend deletion error leaves the
in-memory indexes intact. The method only asks the backend to remove weights;
backend-specific side files and cross-resource cleanup have the behavior
described below.

`promote_to_production(id)` finds the selected record's name, changes every
production record with that name to `Staged`, changes the requested record to
`Production`, timestamps all changed records with the same `Utc::now()`, and
publishes one new model map. Models with other names are unaffected. This
creates one production status per name in the published registry snapshot, but
does not update a `LiveModel`, validate weights, or persist status changes
through the storage backend.

`update_status` similarly changes only the in-memory catalog. If a durable
deployment workflow needs status history, it must persist it outside the
registry or use backend-specific facilities.

### Checksum-bound weight loading

`register` records a SHA-256 checksum for its input weight bytes.
`load_weights` resolves the supplied model ID against the registry's canonical
snapshot before touching storage. It rejects a caller-provided clone whose
`weights_path` or `weights_hash` differs from the canonical record, preventing a
mutable public DTO from redirecting a load or selecting a replacement hash.

If the canonical record has a checksum, the loaded bytes must match it or the
method returns `TuneError::WeightIntegrityError`. If it has no hash, such as a
metadata-only or legacy record, `load_weights` returns the bytes without a
checksum comparison. That is the only unverified path and is explicit in the
canonical record.

Use `load_weights_verified` when verification is a caller requirement. It
rejects a model with no canonical `weights_hash`, then delegates to the same
canonical-record and SHA-256 checks. Both methods return `ModelNotFound` for a
record absent from the in-memory index and a storage error when no weight path
is recorded.

> **Integrity invariant:** never treat a hash on a caller-owned
> `RegisteredModel` clone as authority. The registry verifies against its own
> canonical snapshot, and a checksum mismatch is an error rather than a
> best-effort warning.

### Querying

`ModelQuery` starts from `list_all`, so it operates on owned snapshots rather
than a live cursor. It can require an exact name or status, a minimum validation
accuracy, every requested tag, and a result limit. A minimum-accuracy query
excludes records with no validation accuracy. Results keep the registry's map
iteration order; the builder does not sort them before applying `limit`.

## Shadow evaluation

Shadow evaluation compares a candidate UUID with a production UUID using
application-supplied observations. It is a decision aid for real traffic; the
registry does not run model inference, choose sampled requests, or verify that
either UUID currently denotes a registry record.

### Configuration

`ShadowConfig` contains four acceptance inputs:

| Field | Interpretation | Default | `quick()` | `strict()` |
| --- | --- | ---: | ---: | ---: |
| `sample_rate` | Fraction of traffic the caller intends to sample. | 0.10 | 0.20 | 0.10 |
| `min_samples` | Observations required before a terminal evaluation. | 1,000 | 100 | 10,000 |
| `min_agreement` | Minimum agreeing-output fraction. | 0.95 | 0.90 | 0.99 |
| `max_latency_increase_ms` | Largest allowed mean candidate-minus-production latency. | 50.0 | 100.0 | 20.0 |

`validate` requires `sample_rate` and `min_agreement` to lie in `[0, 1]`,
requires a nonzero sample count, and rejects a negative latency allowance.
`ShadowSession::new` does not call `validate`, and `sample_rate` is not applied
internally. Validate the configuration and make the traffic-sampling decision
before calling `record_sample`.

### State machine and calculation

Creating a session records its IDs and enters `Running` with zero samples.
While it is running, `record_sample(agreed, latency_diff_ms)` appends one
observation; positive latency means the candidate was slower. The session keeps
the raw `(bool, f64)` observations in memory. It neither caps their count nor
validates their latency values, so callers should provide meaningful finite
measurements.

At any point, `current_comparison` computes:

```text
agreement_rate = agreeing_observations / total_observations
latency_diff_ms = sum(candidate_latency - production_latency) / total_observations
```

An empty session reports zero for all three comparison fields. `progress` is
`sample_count / min_samples`, capped at one.

`evaluate` changes state only if the session is `Running` and it has at least
`min_samples` observations. It passes exactly at or above the agreement
threshold and exactly at or below the latency threshold. Otherwise it moves to
`Failed`, preserving the final comparison and producing one or both
human-readable reasons. Before the sample minimum, it returns `Running` and
makes no decision.

```text
Running -- enough samples and both criteria pass --> Passed
Running -- enough samples and either criterion fails -> Failed
Running -- cancel(reason) ------------------------> Cancelled
```

After `Passed`, `Failed`, or `Cancelled`, `record_sample` ignores subsequent
observations and `evaluate` leaves the terminal state untouched. `cancel` itself
sets `Cancelled` unconditionally, so it can replace an earlier terminal state.
`ShadowSession` is mutated through `&mut self`; synchronize it externally if
multiple tasks or threads feed observations.

Promotion remains a separate step:

```rust,no_run
use lattice_tune::registry::{ModelRegistry, ShadowSession};

fn promote_if_shadow_passes(
    registry: &ModelRegistry,
    session: &mut ShadowSession,
) -> lattice_tune::Result<()> {
    if session.passed() {
        registry.promote_to_production(&session.candidate_model_id)?;
    }
    Ok(())
}
```

The calling service is responsible for making the promotion decision durable,
putting the production artifact into service, and handling the behavior of
in-flight requests.

## Rollback records

`RollbackController` is a bounded, in-memory audit log. A `RollbackRecord`
stores its own UUID, the prior production model ID, the new production model
ID, a required reason, optional initiator, and a UTC timestamp.
`record_rollback` appends the record and returns a clone of it.

Records are oldest first. When appending would exceed `max_history`, the
controller removes exactly the oldest entry. A controller configured with zero
keeps no records: each appended record is immediately removed, although the
call still returns it. The default capacity is 100.

The controller exposes the whole ordered history, its most recent record, and
filters for records involving, from, or targeting a particular model. It does
not perform a registry promotion, weight swap, status update, or persistence.
Serialize records externally if audit history must survive a restart.

> **Lifecycle invariant:** recording a rollback is not the rollback itself.
> Coordinate the registry status change, serving-model change, and durable
> audit write in the application, and protect a shared controller with a
> `Mutex` or `RwLock` when it is mutated concurrently.

## Live-model replacement

`LiveModel` wraps one `ArcSwap<RegisteredModel>`. `load` returns an `Arc` to a
consistent current record without taking a lock. `swap` atomically publishes a
new `RegisteredModel` and returns an `Arc` to the old one. A snapshot obtained
before a swap stays valid and continues to expose the old record; snapshots
obtained after the swap see the new record.

This makes the handle useful for a serving layer that needs non-blocking
metadata selection during a hot reload. It does not deserialize weights,
coordinate a two-phase rollout, validate status, or ensure that an application
has replaced its separately held executable model. Readers that need several
fields from one version should call `load` once and read them from that single
snapshot rather than call `model_id` and `version` independently across a
possible swap.

```rust
use lattice_tune::registry::{LiveModel, RegisteredModel};

let live = LiveModel::new(RegisteredModel::new("classifier", "1.0.0"));
let before = live.load();
let replaced = live.swap(RegisteredModel::new("classifier", "2.0.0"));

assert_eq!(before.version, "1.0.0");
assert_eq!(replaced.version, "1.0.0");
assert_eq!(live.load().version, "2.0.0");
```

## Storage backends

`StorageBackend` has mutable `save` and `delete` operations plus read-only
`load`, `exists`, and `list`. It is `Send + Sync`, allowing a registry to hold a
boxed implementation behind its backend mutex. A custom backend receives a
`RegisteredModel` and a byte slice on save and must define its own persistence,
durability, and recovery guarantees.

### In-memory backend

`InMemoryStorage` keeps a `HashMap<String, Vec<u8>>` keyed by
`name/version/weights.bin`. It is intended for tests and process-local use.
Saving replaces any existing bytes at the same computed key, `load` clones the
stored bytes, deleting an absent key succeeds, and `list` returns map keys in
unspecified order. Unlike path-backed stores, it does not validate model names
or versions because the key is not a filesystem path.

### Filesystem backend

`FileSystemStorage::new(root)` creates the root directory. A save validates the
model name and version, then writes weights at:

```text
<root>/<name>/<version>/weights.bin
```

When the `serde` feature is enabled, it also writes a pretty JSON serialization
of the supplied model to `metadata.json` in the same directory. It returns the
relative `name/version/weights.bin` path, never the absolute path.

Loads and deletes reject null bytes, absolute paths, and any `..` component;
`exists` returns `false` rather than an error for such paths. `list` walks the
two-level name/version layout and returns only paths for a present `weights.bin`.
It tolerates unreadable entries by skipping them. Deletion removes the weight
file first and only attempts to remove its immediate parent if empty. A
`metadata.json` side file can therefore keep that directory in place after a
delete; the stale metadata is ignored by `list` because the weight file is gone.

### SQLite backend

With the `sqlite` feature, `SqliteStorage` stores registry fields in a SQLite
`models` table and writes artifact bytes beside the database under:

```text
<database-parent>/weights/<name>/<version>/weights.bin
```

The table has the model ID, name and version (unique as a pair), status,
metadata JSON, registration and update timestamps as epoch microseconds,
registrant and description, weight path/size/hash, and parent ID. It has name
and status indexes. The implementation serializes access to its SQLite
connection with a mutex.

Saving validates path components, writes the file, serializes the supplied
metadata, then inserts the row. If the insert fails, it removes the newly
written weight file before returning an error. The outer registry computes its
checksum after a backend save returns, so a backend row is built from the
pre-enrichment record supplied to `save`; applications that need durable
metadata should account for that ordering.

Deletion orders file removal before deleting the SQLite row. A file-removal
failure therefore preserves the row, but the two resources are not one
transaction: a later database deletion failure can leave a row for a file that
has already been removed. `exists` requires both a matching row and a present
file, while `list` reads non-null weight paths from SQLite. Reconcile partial
failures at the application level.

`SqliteStorage::in_memory` creates an in-memory database and a unique temporary
weights directory for tests. It constructs the table directly and does not run
the on-disk upgrade migrations or create the two indexes used by the on-disk
constructor.

### SQLite schema upgrades

The on-disk constructor creates the current table and indexes if needed, then
runs two fail-closed, idempotent migrations before exposing the storage:

1. It queries `PRAGMA table_info(models)` for a legacy `metadata_json` column.
   If it exists, `ALTER TABLE` renames it to `metadata`; if it is already absent
   the migration does nothing. Unexpected SQLite errors stop initialization.
2. It inspects the declared type of `registered_at`. If it is `INTEGER` (or the
   table is absent), timestamp migration is a no-op. Otherwise it runs a
   `BEGIN IMMEDIATE` batch that creates a correctly typed backup table, copies
   rows, drops the old table, renames the backup, recreates both indexes, and
   commits. Initialization returns an error rather than ignoring a migration
   failure.

The timestamp copy handles three legacy representations for both timestamp
columns:

| Stored value | Detection | Result |
| --- | --- | --- |
| Native integer | `typeof(value) = 'integer'` | Preserve it unchanged. |
| Parseable RFC 3339 / ISO 8601 text | `datetime(value) IS NOT NULL` | Convert `strftime('%s', value)` to microseconds by multiplying by 1,000,000. |
| Numeric text | Neither condition | Cast directly to `INTEGER`. |

The third case matters because SQLite's old `TEXT` affinity could convert
integer microseconds into digit strings. RFC 3339 conversion is based on epoch
seconds, so it discards any sub-second component in those legacy strings. The
backup-and-rename work is deliberately performed under the immediate SQLite
transaction; do not replace it with piecemeal schema changes that can expose a
half-converted table.

## Safetensors weight exchange

When the `safetensors` feature is enabled, `safetensors_io` reads and writes
flat `f32` tensors using the safetensors data format. The library format begins
with an eight-byte little-endian header length, followed by JSON tensor metadata
and the raw tensor-data region. It is a data-only format: parsing does not
deserialize executable objects. Format validation and bounds checking come from
the safetensors parser, while registry SHA-256 verification answers the separate
question of whether the expected artifact bytes were loaded.

`save_weights(weights, name)` produces one named, one-dimensional `F32` tensor
whose shape is `[weights.len()]`. It converts every value to little-endian
bytes. `load_weights(data, name)` requires that exact tensor name and `F32`
dtype, rejects byte lengths not divisible by four, and reconstructs a vector
from four-byte little-endian chunks.

The multi-tensor functions use the same one-dimensional representation for every
map entry:

- `save_tensors` serializes each `HashMap<String, Vec<f32>>` entry as a named
  `F32` vector.
- `load_tensors` iterates all names, skips tensors with a non-`F32` dtype, and
  rejects an `F32` tensor whose byte length is not divisible by four or whose
  byte-derived element count disagrees with the product of its declared shape.
- `validate` only deserializes the container. It establishes that the payload is
  well-formed but does not require a tensor name, dtype, shape, or registry
  checksum.

Use a single named tensor where a flat model-weight vector is the intended
contract. Use multiple tensors only when consumers agree that each tensor is a
flat `f32` vector; these helpers do not preserve arbitrary multidimensional
shapes in their returned `Vec<f32>` values.

> **Format invariant:** safetensors parsing prevents executable-payload
> deserialization, but it does not replace registry identity or SHA-256 checks.
> Validate the container before accepting external bytes and use
> `load_weights_verified` when a recorded artifact checksum is required.

