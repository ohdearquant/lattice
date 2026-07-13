# Training data

The `data` module defines the in-memory contract between distillation or other
data producers and the training APIs. It does not embed text, fetch records
from disk, or serialize a dataset by itself. Its job is narrower:

- represent one embedding-based classification example;
- preserve enough source and teacher metadata for traceability;
- enforce the structural checks that are available locally; and
- group examples into cloned batches for one in-memory epoch at a time.

The dataset-pipeline decision is recorded in the retired crate-local
[ADR-007](adr/ADR-007-dataset-pipeline.md), which links to the maintained
repository-wide ADR. This guide describes the current API behavior rather than
restating that decision.

## The example contract

A `TrainingExample` contains one current-message embedding, a chronological
window of context embeddings, six soft intent labels, and metadata:

```rust
pub struct TrainingExample {
    pub id: Uuid,
    pub context_embeddings: Vec<Vec<f32>>,
    pub message_embedding: Vec<f32>,
    pub labels: IntentLabels,
    pub metadata: ExampleMetadata,
}
```

The inner vectors in `context_embeddings` each represent one earlier message,
ordered oldest to newest. `message_embedding` represents the message to
classify. A new example receives a UUID and default metadata; `with_id` accepts
a caller-provided stable ID; `with_metadata` replaces the default metadata.

### Structural validation

`TrainingExample::validate` performs these checks:

1. The current message embedding must be nonempty.
2. Every context embedding must have exactly the same dimension as the current
   message embedding.
3. Each of the six label values must be in the closed interval `[0, 1]`.

It does **not** require a nonempty context window, a fixed global embedding
dimension, finite embedding values, or labels that sum to one. It also does
not validate metadata. Producers that need those stronger guarantees must
enforce them before adding examples to a dataset.

`embedding_dim()` reports the current-message vector length, and
`context_size()` reports the number of context vectors. The latter is what
dataset filtering uses; it does not inspect the embedding contents.

## Soft intent labels

`IntentLabels` is a six-field record. Its vector representation and class
order are fixed:

| Index | Field | Class name | Meaning |
| ---: | --- | --- | --- |
| 0 | `continuation` | `continuation` | Natural conversational continuation |
| 1 | `topic_shift` | `topic_shift` | A change of subject |
| 2 | `explicit_query` | `explicit_query` | Direct request or question |
| 3 | `person_lookup` | `person_lookup` | Looking up a person or contact |
| 4 | `health_check` | `health_check` | Health or wellness inquiry |
| 5 | `task_status` | `task_status` | Checking a task or todo status |

This order is used by `from_vec`, `to_vec`, `class_names`, batch label
matrices, distillation statistics, and dataset statistics. It is a
cross-module data invariant: do not reorder it or reinterpret a serialized
vector position without changing every producer and consumer.

The named constructors set just one field and leave the other five at zero:
`continuation(prob)`, `topic_shift(prob)`, `explicit_query(prob)`,
`person_lookup(prob)`, `health_check(prob)`, and `task_status(prob)`.
`from_vec` accepts any slice and fills unavailable positions with zero; values
after the sixth position are ignored.

### Validation, dominant class, and normalization

`validate()` rejects values outside `[0, 1]`, including non-finite values. It
does not require a normalized distribution. `dominant()` uses
`f32::total_cmp` over the vector and returns the winning class name and value;
`TrainingExample::dominant_intent()` delegates to it.

`softmax_normalize()` treats the six stored values as scores, not as already
normalized probabilities. It:

1. rejects any non-finite input;
2. subtracts the largest score for numerical stability;
3. exponentiates the shifted scores; and
4. divides each exponent by the total.

The resulting values are finite probabilities that sum approximately to one
for finite inputs. Calling it on values that already look like probabilities
changes their relative distribution; use it only when a score-to-distribution
conversion is intended. The distillation pipeline enables it by default; see
[distill.md](distill.md).

## Traceability metadata

`ExampleMetadata` is optional provenance attached to an example:

| Field | Meaning |
| --- | --- |
| `source_id` | Source conversation or session identifier |
| `timestamp` | UTC time of the original message |
| `teacher_model` | Teacher display name that generated labels |
| `labeled_at` | UTC time at which labeling occurred |
| `teacher_confidence` | Teacher confidence, conventionally 0–1 |
| `extra` | Additional application metadata |

With `serde` enabled, `extra` is `Option<serde_json::Value>`; without it,
`extra` is `Option<String>`. The constructors are intentionally additive:
`with_source` creates metadata with only a source ID, and the `teacher`,
`timestamp`, `labeled_at`, and `confidence` builders set their respective
fields.

The distillation conversion path uses the labeling result UUID as `source_id`,
records `TeacherConfig::display_name()`, records the current UTC time, and
carries the returned confidence. An application that needs the original raw
source ID should preserve it in its own metadata flow before conversion.

## Dataset configuration

`DatasetConfig` controls selection and iteration:

| Field | Default | Effect |
| --- | ---: | --- |
| `batch_size` | 32 | Number of examples in a full batch; must be nonzero. |
| `shuffle` | true | Whether `reset_epoch` permutes example indices. |
| `seed` | none | Fixed seed for a repeatable permutation; absent uses current system time when available. |
| `drop_last` | false | Whether a final partial batch is discarded. |
| `min_context_size` | 1 | Minimum number of context vectors accepted by `with_config`. |
| `max_context_size` | none | Maximum number accepted by `with_config`. |

Despite the field comment's shorthand, `max_context_size` does **not** truncate
a longer context sequence. `Dataset::with_config` filters out examples outside
the inclusive `min_context_size..=max_context_size` range. Configuration
validation rejects a zero batch size and a maximum lower than the minimum.

`DatasetConfig::with_batch_size` starts from defaults. The `shuffle`, `seed`,
and `drop_last` builders replace their corresponding values. There are no
builders for the two context-size limits, so configure them through public
fields before creating the dataset.

### When filtering happens

`Dataset::with_config(examples, config)` validates the configuration once and
filters the supplied vector by context size. It does not call
`TrainingExample::validate`. In contrast:

- `Dataset::new()` creates an empty dataset with default configuration;
- `Dataset::from_examples(...)` keeps every supplied example without filtering;
- `Dataset::add(...)` appends an example without validation or filtering; and
- `Dataset::set_config(...)` validates and replaces the stored configuration,
  but does not retroactively filter existing examples or reset iteration.

Configure and validate producer data before construction when you need the
dataset itself to be an eligibility gate.

## Batching and epoch state

`Dataset` owns the example vector plus an index permutation and a cursor. It
does not implement the standard `IntoIterator` contract; use `batches()` for a
fresh epoch or `next_batch()` to advance the current one manually.

```rust,ignore
use lattice_tune::data::{Dataset, DatasetConfig};

let mut dataset = Dataset::with_config(examples, DatasetConfig::with_batch_size(32))?;

for batch in dataset.batches() {
    // batch.examples owns cloned TrainingExample values
    consume(batch.message_embeddings(), batch.labels());
}
```

### Reset and shuffle

`reset_epoch()` sets the cursor to zero, rebuilds the index list as
`0..examples.len()`, and optionally shuffles it. `batches()` always calls
`reset_epoch()` before returning `BatchIterator`, so a new `batches()` call
abandons any partially consumed manual epoch.

The shuffle is Fisher–Yates driven by a simple linear congruential generator.
With `seed: Some(value)`, every reset starts from the same seed and therefore
produces the same order for unchanged examples. Without a seed, the current
UNIX-epoch nanoseconds seed the generator; if the system clock cannot be read,
the fallback is `42`. This is deterministic sampling machinery, not a
cryptographic randomizer.

### Batch boundaries

`num_batches()` returns:

```text
0                                  when the dataset is empty
floor(n / batch_size)              when drop_last is true
ceil(n / batch_size)               otherwise
```

`next_batch()` takes the next slice of permutation indices and clones the
corresponding examples. If `drop_last` is true and the remaining slice is
short, it advances the cursor to the end and returns `None`. Otherwise it
returns a `Batch` with a zero-based `batch_idx` and the epoch's
`total_batches`.

A `Batch` is an owned collection:

| Member or method | Behavior |
| --- | --- |
| `examples` | Cloned `TrainingExample` values selected for that batch |
| `batch_idx` / `total_batches` | Position and expected count within the epoch |
| `len()` / `is_empty()` | Inspect the owned example vector |
| `message_embeddings()` | Allocates a matrix of cloned message vectors |
| `labels()` | Allocates a matrix of six-element label vectors |

`Batch::from_examples` is a convenience constructor for a standalone batch. It
sets `total_batches` to one; an epoch-produced batch gets its actual epoch
total from `Dataset::next_batch`.

## Dataset inspection and partitioning

`len()`, `is_empty()`, `get(id)`, `get_idx(index)`, and `examples()` provide
read access. `examples_mut()` exposes the backing vector directly; callers
that change it are responsible for preserving valid examples and for resetting
the epoch before relying on a fresh index permutation.

### Statistics

`stats()` returns all-zero `DatasetStats` for an empty dataset. Otherwise it
uses the first example's message dimension as `embedding_dim`, calculates
minimum, maximum, and arithmetic mean context size, and produces six
`label_distribution` counters.

Each label-distribution counter represents the number of examples whose
dominant class is that index; it is not a sum or average of soft label values.
`stats()` does not validate examples first, so inconsistent dimensions can
yield a reported first dimension even when the collection is not fit for a
downstream training routine.

### Train/validation split

`split(train_ratio)` accepts an inclusive ratio in `0.0..=1.0`. It computes
the boundary as:

```text
split_index = (number_of_examples as f32 * train_ratio) as usize
```

and returns the prefix as training data and the suffix as validation data.
The split preserves current storage order: it does not shuffle or stratify by
label, context size, or metadata. Both returned datasets are constructed with
`Dataset::from_examples` and therefore have default configuration, fresh
indices, and cloned examples. Set configuration explicitly on each partition
when batch sizing, shuffling, or dropping partial batches matters.

## Producer checklist

Before training, a producer should:

1. create or preserve stable example IDs;
2. create every context and current-message embedding in the same dimension;
3. use the fixed `IntentLabels` vector order;
4. validate each `TrainingExample` if the data source is not already trusted;
5. choose context eligibility during `Dataset::with_config` construction;
6. pick a fixed `seed` if repeatable epoch order is needed; and
7. perform a deliberate, preferably shuffled or stratified, train/validation
   split before relying on validation metrics.

For teacher-generated labels and the conversion step that supplies these
examples, see [distill.md](distill.md). For the training loop that consumes
batches, see [train.md](train.md).
