# ADR-034: Dataset Pipeline

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-tune

## Context

Training student networks requires feeding batches of `TrainingExample` records to the training loop in a controlled order. The pipeline must handle: (1) variable-length context windows (multiple context embeddings per example), (2) shuffling with optional deterministic seeding, (3) drop-last behavior for fixed-size batch training, (4) context size filtering and truncation, (5) efficient train/validation splitting, and (6) dataset statistics for monitoring class balance and embedding dimensions.

The `Dataset` struct must work identically for offline `TrainingLoop` training, JIT `JitAdapter` adaptation, and GPU `GpuTrainer` validation, so the API must be synchronous and side-effect-free from the training loop's perspective.

## Decision

`Dataset` holds `Vec<TrainingExample>` plus a `DatasetConfig` for batching parameters and a `Vec<usize>` index for shuffle order. Iteration is stateful via `current_idx`. Batching is exposed via `next_batch() -> Option<Batch>` and a `BatchIterator` adapter. Context size filtering happens at `Dataset::with_config` construction time.

### Key Design Choices

#### TrainingExample Structure

A `TrainingExample` contains:

- `id: Uuid` — unique identifier for traceability
- `context_embeddings: Vec<Vec<f32>>` — one embedding per context message, oldest-to-newest
- `message_embedding: Vec<f32>` — embedding of the current message to classify
- `labels: IntentLabels` — six soft probability values from the teacher
- `metadata: ExampleMetadata` — source ID, timestamps, teacher model, confidence

Embedding dimension consistency is validated in `TrainingExample::validate()`: all context embeddings must have the same dimension as `message_embedding`. This is a structural check (not enforced at construction time to avoid double-validation overhead).

#### IntentLabels (6 Classes)

`IntentLabels` is a flat struct (not a Vec) for the six intent classes: `continuation`, `topic_shift`, `explicit_query`, `person_lookup`, `health_check`, `task_status`. This choice:

- Enables named field access (`labels.continuation`) without index arithmetic
- Derives `Default` to all-zero
- `to_vec()` and `from_vec()` provide interop with the network output (`Vec<f32>`)
- `dominant() -> (&'static str, f32)` returns the highest-probability class for logging/statistics

`IntentLabels::NUM_CLASSES = 6` is a constant used in `DistillationStats::label_distribution` sizing.

`softmax_normalize(&mut self)` applies numerically stable softmax (max subtraction) in-place, returning an error if any input is non-finite. This is used by `DistillationPipeline` when `config.normalize_labels = true`.

#### Context Size Management

`DatasetConfig` exposes:

- `min_context_size: usize` (default 1) — filter out examples with fewer context messages
- `max_context_size: Option<usize>` (default None) — filter out examples with more context messages than this

Filtering is applied at `Dataset::with_config` construction time (not lazily). Examples that fail either bound are dropped from the dataset entirely. This is a deliberate design choice: it prevents the training loop from seeing inconsistent-sized context windows that might require padding logic.

Validation in `DatasetConfig::validate()` checks `batch_size > 0` and (if `max_context_size` is set) `max >= min`.

#### Shuffling (LCG)

`Dataset::shuffle_indices()` uses a Knuth multiplicative LCG (`6364136223846793005 * state + 1`) for Fisher-Yates. When `config.seed` is set, `state = seed`; when `None`, `state = SystemTime::now()` nanoseconds (with `42` as fallback if the system clock fails).

This avoids adding `rand` as a dependency to `lattice-tune`'s dataset module — the LCG is sufficient for shuffling training data where cryptographic quality is not required. The seeded path is used by `JitAdapter` (fixed seed `42`) and `TrainingLoop` (configurable seed).

#### Batch Structure

`Batch { examples: Vec<TrainingExample>, batch_idx: usize, total_batches: usize }`.

Helper methods on `Batch`:

- `message_embeddings() -> Vec<Vec<f32>>` — extracts all message embeddings as a 2D matrix
- `labels() -> Vec<Vec<f32>>` — extracts all label vectors via `IntentLabels::to_vec()`

These are used by `GpuTrainer::train_batch` which operates on flat matrix inputs.

`drop_last` behavior: if `config.drop_last = true` and the final batch would be smaller than `batch_size`, `next_batch()` returns `None` for that final batch. `num_batches()` returns `n / batch_size` when `drop_last` is true, `n.div_ceil(batch_size)` otherwise.

#### DatasetStats

`Dataset::stats()` computes in one pass:

- `num_examples`, `embedding_dim` (from first example)
- `avg_context_size`, `min_context_size`, `max_context_size`
- `label_distribution: Vec<usize>` — count of examples where each class is dominant (hardcoded length 6)

This is used in monitoring to detect class imbalance in the training data.

#### Train/Validation Split

`Dataset::split(train_ratio: f32) -> Result<(Dataset, Dataset)>` splits by index without shuffling (preserves order). Validation ratio is `1.0 - train_ratio`. Error if `train_ratio` outside `[0.0, 1.0]`.

`TrainingLoop::train` uses `config.val_split` to call `dataset.split(1.0 - val_split)` before configuring the training dataset.

#### ExampleMetadata

`ExampleMetadata` provides traceability: `source_id`, `timestamp`, `teacher_model`, `labeled_at`, `teacher_confidence`, `extra: Option<serde_json::Value>` (when `serde` feature enabled). Builder methods allow incremental construction (`with_source(...).teacher(...).labeled_at(...).confidence(...)`).

### Alternatives Considered

| Alternative                            | Pros                                   | Cons                                                                | Why Not                                                                                     |
| -------------------------------------- | -------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Iterator-based lazy loading            | Supports datasets larger than memory   | Requires file format, I/O layer, and error handling in the hot path | Student networks train on small distilled datasets (typically thousands, not millions)      |
| `ndarray` arrays for batches           | Contiguous memory, efficient batch ops | External dependency; `lattice-fann` uses `Vec<f32>` throughout      | Maintaining a separate array type adds conversion overhead with no benefit at current scale |
| `u8` encoded context size in the batch | Avoids variable-length Vec overhead    | Limits context to 255 messages                                      | Context size is not hot-path; clarity > micro-optimization                                  |
| Per-class `Vec<f32>` label storage     | Enables arbitrary number of classes    | Runtime index errors; no field names                                | Six fixed classes for the current scope; named fields catch bugs at compile time            |

## Consequences

### Positive

- `BatchIterator` implements `Iterator<Item=Batch>`, so `for batch in dataset.batches()` works idiomatically with all Rust iterator adapters
- Context size filtering at construction time ensures the training loop never sees a context-size mismatch
- `DatasetStats::label_distribution` enables immediate monitoring of class imbalance without external tooling
- `ExampleMetadata.teacher_model` provides full lineage for debugging: every training example records which teacher labeled it

### Negative

- `Dataset::split` does not shuffle before splitting. If examples were added in class order (all `continuation` first, then all `topic_shift`), the validation set may have a different class distribution from the training set. Callers must shuffle the dataset before splitting if this matters.
- `shuffle_indices` uses a simple LCG, not a cryptographically secure shuffle. This is sufficient for ML training but should not be relied upon for any security-sensitive ordering.
- `Batch::message_embeddings()` and `Batch::labels()` clone all embedding and label data into new `Vec`s. For large batches with high-dimensional embeddings, this is the most significant allocation in the training loop.

### Risks

- `label_distribution` in `DatasetStats::stats()` is hardcoded to length 6 (matching `IntentLabels::NUM_CLASSES`). If a future version adds intent classes, this code will silently produce wrong statistics unless updated simultaneously.
- `Dataset::stats()` takes the dominant class by argmax per example, not a soft distribution. For examples with near-uniform labels, the dominant class assignment is arbitrary and the distribution statistics will undercount the effective diversity of the training set.

## References

- `crates/tune/src/data/dataset.rs` — `Dataset`, `DatasetConfig`, `Batch`, `BatchIterator`, `DatasetStats`
- `crates/tune/src/data/example.rs` — `TrainingExample`, `IntentLabels`, `ExampleMetadata`
- `crates/tune/src/data/mod.rs` — module exports
- `crates/tune/src/train/loop/mod.rs` — `TrainingLoop::train` uses `Dataset::split` and `Dataset::batches`
- `crates/tune/src/train/jit.rs` — `JitAdapter::adapt_cpu` and `adapt_gpu` use `Dataset::next_batch`
