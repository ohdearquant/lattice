# Training pipeline

`lattice-tune` exposes configuration, run orchestration, metrics, checkpoints, a
WGPU-backed trainer, and just-in-time (JIT) adaptation helpers. This document
describes their current behavior, including the boundaries that make a
configuration valid but do not yet make it an executable optimization run.

## Current implementation status

The public interfaces distinguish five relevant execution paths:

| Path                      | What works now                                                                                                                                  | What it does not do                                                                                   |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `TrainingLoop`            | Validates configuration, splits and batches data, invokes callbacks, records metrics, applies early stopping, and produces checkpoint metadata. | It simulates loss and accuracy and never forwards through or updates a network.                       |
| `GpuTrainer::validate`    | Runs the WGPU network forward, checks outputs for NaN/Infinity, computes loss, and reports top-class accuracy.                                  | It is evaluation only and does not update weights.                                                    |
| `GpuTrainer::train_batch` | Runs forward, finite-value checks, loss calculation, and placeholder backward preparation.                                                      | It always returns `TuneError::Training` at the optimizer update. No optimizer choice changes weights. |
| `JitAdapter::adapt_cpu`   | Batches examples, observes its timeout and patience rules, and returns a loss history.                                                          | Its per-step loss is simulated; it does not adapt a model.                                            |
| `JitAdapter::adapt_gpu`   | Uses the same batching and stop rules, when the `gpu` feature is enabled.                                                                       | It delegates to `GpuTrainer::train_batch`, so it currently fails at the optimizer update.             |

Treat these boundaries as part of the API contract. In particular, callers must
not interpret a successful forward-only validation result, a CPU-loop metric, or
a JIT CPU result as evidence that model parameters were trained.

## Training configuration

`TrainingConfig` is the root configuration passed to `TrainingLoop` and
`GpuTrainer`. Its builder methods replace the matching field and return the
updated configuration. Trainer constructors validate it before constructing a
trainer.

### Defaults and presets

| Setting             | `TrainingConfig::default()`                               | `quick()`       | `thorough()`                                              |
| ------------------- | --------------------------------------------------------- | --------------- | --------------------------------------------------------- |
| Epochs              | 100                                                       | 10              | 200                                                       |
| Batch size          | 32                                                        | 64              | 32                                                        |
| Optimizer           | default AdamW                                             | default AdamW   | AdamW, LR `0.0001`, decay `0.01`                          |
| Schedule            | cosine warmup: 100 steps, floor `1e-6`, period 100 epochs | same as default | cosine warmup: 500 steps, floor `1e-7`, period 200 epochs |
| Regularization      | default                                                   | default         | `strong()`                                                |
| Validation split    | 0.1                                                       | 0.0             | 0.15                                                      |
| Early stopping      | default validation-loss monitor                           | disabled        | validation-loss monitor, patience 20                      |
| Checkpoint interval | 10 epochs                                                 | 5 epochs        | 5 epochs                                                  |
| Logging interval    | 100 steps                                                 | 100 steps       | 50 steps                                                  |
| Mixed precision     | disabled                                                  | disabled        | enabled                                                   |
| Seed                | 42                                                        | 42              | 42                                                        |

The default optimizer is AdamW with learning rate `0.001`, momentum `0.9`,
Adam coefficients `beta1 = 0.9` and `beta2 = 0.999`, epsilon `1e-8`, and
weight decay `0.01`. The supported selection is SGD, SGD with momentum, Adam,
AdamW, or RMSprop. Optimizer configuration requires a finite positive learning
rate and epsilon; finite momentum in `[0, 1]`; finite Adam coefficients in
`[0, 1)`; and finite, non-negative weight decay.

Configuration validity also requires:

| Field                               | Requirement                                                   |
| ----------------------------------- | ------------------------------------------------------------- |
| `epochs`                            | Greater than zero.                                            |
| `batch_size`                        | In `1..=8192`; the upper bound prevents excessive allocation. |
| `val_split`                         | In the closed interval `[0, 1]`.                              |
| `accumulation_steps`                | Greater than zero.                                            |
| Optimizer, regularization, schedule | Each nested configuration must validate.                      |

The effective batch size reported by `effective_batch_size()` is
`batch_size.saturating_mul(accumulation_steps)`. It is a configuration-derived
value: neither current trainer consumes `accumulation_steps` to defer an
optimizer update.

`mixed_precision`, `log_interval`, and `checkpoint_dir` are stored in the
configuration. The current `TrainingLoop` does not implement mixed precision or
use `log_interval` itself; logging is callback-defined. A checkpoint directory
only enables periodic checkpoint callbacks—it does not create a file.

### Memory budgeting

`estimate_memory_mb(model_params, embedding_dim)` is a conservative planning
estimate, not an allocator reservation. It uses checked integer arithmetic and
returns `InvalidConfig` on overflow. Let `P` be the parameter count, `B` the
configured batch size, and `D` the embedding dimension. Its byte estimate is:

```text
model       = 4P
gradients   = 4P
optimizer   = 8P for Adam or AdamW
              4P for RMSprop or SGD with momentum
              0  for SGD
batch data  = 8BD
activations = 2 × batch data
total MB    = ceil(1.20 × total bytes / 1,048,576)
```

`check_memory_budget(required_mb)` returns
`TuneError::MemoryBudgetExceeded` only when `memory_budget_mb` is set and the
requested amount is greater than the budget. No trainer calls the estimate or
budget check automatically, so applications that require an admission check
must call both before construction or execution.

### Regularization

`RegularizationConfig` contains independent settings. The validation range does
not imply that every setting is applied by every path.

| Setting           |     Default | Validation                         | Current use                                                           |
| ----------------- | ----------: | ---------------------------------- | --------------------------------------------------------------------- |
| `dropout`         |         0.1 | Must be within `[0, 1]`.           | Stored only in the training configuration.                            |
| `label_smoothing` |         0.1 | Must be within `[0, 1]`.           | Used by `GpuTrainer` loss calculation.                                |
| `gradient_clip`   | `Some(1.0)` | If set, must be greater than zero. | Available through `apply_gradient_clip`; no current trainer calls it. |
| `mixup_alpha`     |      `None` | If set, must be greater than zero. | Stored only in the training configuration.                            |

The presets are `none()` (no dropout, no smoothing, no clipping, no mixup),
`light()` (dropout 0.05, smoothing 0.05, clipping 1.0), and `strong()`
(dropout 0.3, smoothing 0.2, clipping 0.5, mixup alpha 0.2).

Gradient clipping is global L2-norm clipping. For gradient vector `g`, it
computes `norm = sqrt(sum(g_i²))`, returns that original norm, and behaves as
follows:

```text
if norm > max_norm and norm > 0:
    g ← g × (max_norm / norm)
otherwise:
    g is unchanged
```

This preserves direction and makes the resulting norm exactly `max_norm` in the
clipping case. An empty or all-zero vector remains unchanged and returns zero.
`apply_gradient_clip` returns `None` when clipping is disabled and
`Some(original_norm)` when it is enabled, whether or not rescaling occurred.

### Learning-rate schedules

`LRSchedule::get_lr(base_lr, step, epoch)` evaluates from the immutable base
optimizer rate. Schedules do not mutate optimizer configuration. Let `b` be
`base_lr`, `s` be `step`, and `e` be `epoch`.

| Schedule                                               | Result                                                                                                              |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `Constant`                                             | `b`                                                                                                                 |
| `LinearWarmup { warmup_steps: W }`                     | `b × s / W` while `s < W`; otherwise `b`                                                                            |
| `StepDecay { step_size: K, gamma: γ }`                 | `b × γ^(floor(e / K))`                                                                                              |
| `ExponentialDecay { gamma: γ }`                        | `b × γ^e`                                                                                                           |
| `CosineAnnealing { min_lr: m, t_max: T }`              | `m + (b - m) × (1 + cos(π × ((e mod T) / T))) / 2`                                                                  |
| `CosineAnnealingWarmup { W, m, T }`                    | Linear warmup while `s < W`; otherwise the cosine formula above, using `e mod T`                                    |
| `OneCycle { max_lr: M, pct_start: p, total_steps: N }` | Rise linearly from `b` to `M` for `s/N < p`; then linearly descend from `M` to `0.01b` over the remaining fraction. |

The default is `CosineAnnealingWarmup { warmup_steps: 100, min_lr: 1e-6,
t_max: 100 }`. Warmup starts at zero when `step` is zero; the cosine phase uses
absolute epoch modulo `t_max`, not an epoch count offset by the warmup.

Validation prevents a zero warmup length where a warmup schedule needs one, a
zero decay period, or a zero OneCycle length. Decay factors must be finite and
positive; cosine floors must be finite and non-negative; OneCycle maximum rate
must be finite and positive; and its start fraction must be finite and strictly
inside `(0, 1)`. `get_lr` defensively returns the base rate for zero periods
even though validation rejects them. It does not clamp `step` to `total_steps`
for OneCycle, so callers should keep `step` inside the configured cycle when
they need the stated terminal rate.

### Early stopping

`EarlyStopping` monitors one of the metric names accepted by
`EpochMetrics::get_metric`: `train_loss`, `val_loss`, `train_accuracy`, or
`val_accuracy`. Its defaults monitor `val_loss` with patience 10, minimum delta
`1e-4`, and minimization. The helpers choose validation loss (minimize) or
validation accuracy (maximize).

For a minimization monitor, an epoch improves only when
`current < best - min_delta`; for maximization, it must satisfy
`current > best + min_delta`. An unknown monitor yields no metric and therefore
does not advance or reset the loop's patience counter.

The current `TrainingState` starts `best_metric` at positive infinity. That is
appropriate for a minimizing monitor, but a maximizing monitor never improves
against the initial value; `val_accuracy` monitoring will therefore exhaust
patience unless state is initialized or restored differently. This is a current
behavior to account for when selecting the monitor.

## CPU training-loop lifecycle

`TrainingLoop::new` validates its configuration, sets the state learning rate
to the optimizer base rate, creates empty metrics and callbacks, and clones the
optional early-stopping configuration. `train` rejects an empty dataset.

For a nonzero validation split, it calls `dataset.split(1.0 - val_split)` and
uses the returned training and validation datasets. With a zero split it clones
the original dataset as the training data and has no validation data. The
training dataset is configured with the selected batch size, shuffling enabled,
and either the configured seed or 42 when no seed is provided.

The lifecycle is:

1. Invoke `on_train_start(config)` on callbacks.
2. For each epoch, record the zero-based epoch in state, reset that epoch's
   step/loss/correct/total counters, then invoke `on_epoch_start(epoch)`.
3. For every batch, invoke `on_batch_start(batch_idx)`, run the simulated batch
   operation, then invoke `on_batch_end(batch_idx, loss)`.
4. When validation exists, calculate the simulated validation loss and
   accuracy. Build and append `EpochMetrics`, then invoke
   `on_epoch_end(epoch, metrics)`.
5. Evaluate early stopping. A stop breaks the epoch loop before that epoch's
   learning-rate update and periodic checkpoint callback.
6. Set the next learning rate from the schedule using the base optimizer rate,
   current global step, and current epoch.
7. If checkpointing is enabled and the epoch count is an interval multiple,
   construct an in-memory checkpoint and invoke `on_checkpoint`.
8. Invoke `on_train_end(metrics)` and return a clone of the aggregate metrics.

The simulated batch implementation increments both per-epoch and global step,
adds the batch size to examples seen, and uses the following synthetic values:

```text
loss          = 0.5 × exp(-0.01 × global_step) + 0.1
accuracy rate = 0.6 + 0.35 × (1 - exp(-0.005 × global_step))
correct       = trunc(batch_size × accuracy rate)
```

Validation returns 110% of epoch training loss and 95% of epoch training
accuracy. These values exercise reporting and stopping behavior only; neither
is produced from model predictions.

`checkpoint_interval` is not validated. When both a checkpoint directory is
configured and the interval is zero, the periodic modulo check panics. Use a
positive interval whenever checkpointing is enabled. Supplying a directory does
not write a checkpoint: only callbacks receive the object.

### Metrics and callbacks

`EpochMetrics` contains the zero-based epoch, train and optional validation
loss and accuracy, the learning rate recorded before the end-of-epoch schedule
update, duration, and examples seen in that epoch. Metric lookup recognizes the
four names used by early stopping.

`TrainingMetrics::add_epoch` appends history and updates:

- `best_val_loss` and `best_epoch` only for an epoch that has validation loss;
- final training and optional validation loss from the appended epoch;
- accumulated duration and `epochs_completed = epoch + 1`.

`train_loss_history` preserves one value per epoch. `val_loss_history` omits
epochs without validation. `is_overfitting(window)` needs at least two full
history windows and compares mean training losses and validation losses across
the earlier and most-recent windows. Its validation sums are still divided by
`window` even when individual epochs lack validation, so use it only with
validation metrics available for every epoch in the compared range.

Callbacks are mutable `Send + Sync` trait objects. The no-op callback accepts
every lifecycle event. `LoggingCallback` prints a summary at epoch end and a
batch loss when `batch_idx % log_interval == 0`. Its constructor accepts zero,
which makes the modulo expression panic on the first batch; use a positive
logging interval.

## Checkpoints and resumption

`Checkpoint::new(epoch, global_step, metrics)` constructs an in-memory record
with a UUID v4 identifier and current UTC timestamp. Its `weights` and
`optimizer_state` vectors start empty. Neither `Checkpoint` nor `TrainingLoop`
performs file creation, serialization, deserialization, or buffer restoration.

With the `serde` feature, the struct serializes as JSON. Applications that
populate and consume its byte vectors must preserve this byte-level contract:

| Field             | Required byte layout                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `weights`         | Little-endian `f32` values, layer by layer; for each layer, its row-major weight matrix then its biases; layers proceed input to output. |
| `optimizer_state` | Optimizer-specific vectors per parameter: one velocity vector for momentum SGD; first (`m`) and second (`v`) moment vectors for Adam.    |

`TrainingLoop::checkpoint()` captures only the current epoch, global step, and
cloned metrics in that record. `resume_from` restores exactly those three
values. It does not restore weights, optimizer bytes, learning rate,
early-stopping best metric, patience counter, callbacks, or the dataset
position. Moreover, a subsequent `train()` starts its `for epoch in
0..config.epochs` loop at zero and overwrites the restored current epoch.
A resumed loop therefore needs external model/optimizer restoration and is not
a continuation mechanism for epoch position or early-stopping state.

## GPU path

`GpuTrainer::new` creates a blocking `GpuContext`, wraps a `lattice_fann`
network in `GpuNetwork`, and allocates state for each network layer.
`with_context` uses an existing shared context instead. Both validate
`TrainingConfig` first and initialize the current learning rate to the optimizer
base rate.

For each layer with `W` weights and `B` biases, initialization creates storage
and copy-destination buffers for `W` weight gradients and `B` bias gradients.
It also allocates three optimizer-state buffers—Adam first moment, Adam second
moment, and SGD velocity—each sized for `W + B` `f32` values, regardless of the
optimizer selected. The state timestep starts at zero.

### Forward input, loss, and validation

Each training example becomes one network input. If it has context embeddings,
the trainer averages those vectors elementwise and then appends the message
embedding. This requires all context embeddings to have the first context
embedding's dimensionality; the current preparation code does not validate that
invariant before indexing.

The trainer calls synchronous GPU forward for every example and records only
the input and output as its placeholder activation set. It rejects any NaN or
infinite output before computing loss or accuracy.

For output values `p`, target values `y`, smoothing amount `a`, and
`C = output.len()`, the GPU loss calculation uses:

```text
y_smooth = (1 - a) × y + a / C
sample loss = -Σ y_smooth × ln(max(p, 1e-7))
batch loss = mean(sample loss)
```

The `1e-7` lower clamp applies only inside the logarithm. It prevents a
logarithm of zero but does not alter the output vector used for accuracy or the
placeholder backward derivative. `validate` accumulates each batch loss
weighted by batch length, divides by the number of examples, and computes
accuracy by comparing the index of the largest output to the index of the
largest target. Empty validation data returns `(0.0, 0.0)`.

### Backward and optimizer status

The current backward preparation first forms `output - target` using the
_unsmoothed_ target. It then ignores those values: the CPU fallback uploads the
constant gradient `0.01 / batch_size` for every weight and bias of every layer.
Consequently, even before optimizer dispatch, this is placeholder work rather
than an implementation of backpropagation for the loss above.

All optimizer selections deliberately fail:

| Optimizer         | Current result                                                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------------- |
| Adam              | Returns `TuneError::Training`; shader buffer bindings are absent.                                             |
| AdamW             | Returns the same error.                                                                                       |
| SGD with momentum | Returns the same error.                                                                                       |
| SGD               | Returns `TuneError::Training`; there is neither real gradient plumbing nor mutable network weight write-back. |
| RMSprop           | Returns `TuneError::Training`; it does not substitute a different algorithm.                                  |

The failure is intentional. A previous no-op or substitute update would have
made a training call look successful while leaving weights unchanged. Since the
optimizer returns before its post-update bookkeeping, optimizer timesteps are
not incremented and weights are not synchronized. `GpuTrainer::train_batch`
only increments its global step and evaluates the next learning rate after a
fully successful update, so failed calls cannot inflate step or schedule state.

Until this status changes, use `GpuTrainer::validate` only for forward
evaluation and handle `train_batch` errors as an expected capability boundary.

## JIT adaptation

`JitConfig` describes a short adaptation run. Its default is the
`LastNLayers(2)` strategy, 100 steps, learning rate `0.001`, frozen-layer
multiplier `0.1`, batch size 16, GPU requested, a 10-second timeout, patience
10, minimum improvement `1e-4`, and seed 42.

| Preset       | Strategy      | Steps |    LR | Batch | Timeout | Patience |
| ------------ | ------------- | ----: | ----: | ----: | ------: | -------: |
| `fast()`     | head only     |    50 |  0.01 |    32 |     5 s |        5 |
| `thorough()` | last 3 layers |   200 | 0.001 |    16 |    10 s |       20 |
| `few_shot()` | head only     |    30 | 0.005 |     4 |     3 s |        5 |

Validation requires positive `steps`, learning rate, batch size, and timeout.
It does not validate strategy parameters, `frozen_lr_multiplier`, `patience`,
or `min_delta`; callers must choose meaningful values for those fields.
`use_gpu` is also stored but not consulted to select a method: callers choose
explicitly between `adapt_cpu` and the feature-gated `adapt_gpu`.

### Strategies and helper output

The `freeze` helpers use `true` to mean a frozen layer and order layers from
input to output.

| Strategy           | `get_frozen_layers`                                              | `get_lr_multipliers`                                                                 |
| ------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `LastNLayers(n)`   | Freeze all except the final `min(n, total_layers)` layers.       | Frozen layers get `frozen_lr_multiplier`; final layers get 1.0.                      |
| `HeadOnly`         | Freeze all except the final layer; an empty model remains empty. | Frozen layers get the multiplier; the head gets 1.0.                                 |
| `GradualUnfreeze`  | All layers are marked trainable.                                 | Linear interpolation from the frozen multiplier on layer 0 to 1.0 on the last layer. |
| `LowRank { rank }` | Every original layer is frozen.                                  | Every layer gets the frozen multiplier.                                              |
| `Full`             | No layers are frozen.                                            | Every layer gets 1.0.                                                                |

The helpers describe selection only. In particular, the JIT adapter does not
construct low-rank matrices, apply per-layer multipliers, or otherwise connect
these strategy values to a model update path.

### Adaptation loop and result

Both CPU and GPU adaptation reject an empty example vector, create a dataset
with the configured batch size, shuffle it, and use the configured seed or 42.
When a batch iterator reaches the end, the adapter resets the dataset epoch and
continues with the next batch.

At the start of each requested step it checks `elapsed > timeout`; equality
does not stop the run. It records a completed step's loss, then resets patience
only when `loss < best_loss - min_delta`. Any other loss increments the counter;
when it reaches `patience`, the result is early-stopped. A zero patience value
therefore stops on the first non-improving completed step.

CPU adaptation uses the synthetic loss:

```text
1 / (1 + 0.1 × step) + abs(sin(0.1 × step) × 0.1)
```

GPU adaptation invokes `GpuTrainer::train_batch` and then, if a run could
finish, validates the same adaptation dataset to obtain final accuracy. The
current GPU optimizer limitation prevents it from reaching that point.

`JitResult` returns final loss (or positive infinity when no step completed),
optional final accuracy, completed-step count, elapsed seconds, early-stop and
timeout flags, and the complete loss history. `is_success()` means only that
the run did not time out and its final loss is finite; an early-stopped result
can be successful. `avg_loss()` returns zero for an empty history.

## Checkpoint byte layout

`Checkpoint` is an in-memory record; it does not serialize itself or verify
the contents of either byte vector. With the `serde` feature, the enclosing
record serializes as JSON. Producers and consumers of the two byte fields must
therefore agree on their binary contents independently of that JSON envelope.

`weights` contains little-endian `f32` values in network order: for each layer
from input to output, write its row-major weight matrix followed by its bias
vector. `optimizer_state` is optimizer-specific and is ordered per parameter:
momentum SGD uses one velocity vector, while Adam uses its first (`m`) and
second (`v`) moment vectors. The constructor leaves both fields empty, so a
checkpoint made by `TrainingLoop` contains metadata and metrics only unless a
caller populates those vectors.

Restoring through `TrainingLoop::resume_from` copies only epoch, global step,
and aggregate metrics. It cannot restore these byte vectors, learning rate,
early-stopping counters, callbacks, or dataset position. External persistence
code must restore model and optimizer state itself, and must not treat this API
as a complete continuation mechanism.

## GPU placeholder-gradient path

`GpuTrainer::train_batch` runs forward evaluation, rejects non-finite outputs,
computes loss, checks the loss, prepares placeholder gradients, and then calls
the optimizer dispatcher. The backward preparation first forms `output -
target` with the unsmoothed target, but the current CPU fallback does not use
those values to derive parameter gradients. For every layer, it uploads the
same `0.01 / batch_size` value for every weight and bias. The recorded
activation set contains only the input and output vectors.

This is explicitly scaffolding, not a gradient implementation. The optimizer
dispatcher then returns `TuneError::Training` for every optimizer choice until
network buffer binding and weight write-back exist. Because the error is
propagated before bookkeeping, a failed call leaves `global_step` and
`current_lr` unchanged. Only a completed forward, backward, and optimizer
update may advance the step; it then recomputes the learning rate from the base
rate using that new step and an approximate epoch of `global_step / 100`.
