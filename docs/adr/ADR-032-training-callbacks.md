# ADR-032: Training Callbacks

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-tune

## Context

The `TrainingLoop` orchestrator in `lattice-tune` needs to emit training events (epoch start/end, batch start/end, checkpoint) to external observers without coupling the loop to any specific logging or monitoring implementation. The training loop also needs to integrate early stopping and checkpoint logic, which involves per-epoch state decisions.

The design must: (1) allow multiple callbacks simultaneously, (2) support no-op callbacks for tests, (3) ship a default logging callback, and (4) not require callbacks to implement every hook.

## Decision

A `TrainingCallback` trait with default no-op implementations for all hooks. Callbacks are stored as `Vec<Box<dyn TrainingCallback>>` on `TrainingLoop`. Early stopping and checkpointing are handled directly in the training loop against `EarlyStopping` config and `TrainingState`, not via callbacks.

### Key Design Choices

#### Trait with Default No-Op Bodies

```rust
pub trait TrainingCallback: Send + Sync {
    fn on_train_start(&mut self, _config: &TrainingConfig) {}
    fn on_train_end(&mut self, _metrics: &TrainingMetrics) {}
    fn on_epoch_start(&mut self, _epoch: usize) {}
    fn on_epoch_end(&mut self, _epoch: usize, _metrics: &EpochMetrics) {}
    fn on_batch_start(&mut self, _batch_idx: usize) {}
    fn on_batch_end(&mut self, _batch_idx: usize, _loss: f32) {}
    fn on_checkpoint(&mut self, _checkpoint: &Checkpoint) {}
}
```

All methods have default no-op bodies, so implementations only override what they care about. `Send + Sync` is required because `TrainingLoop` may eventually be used from multiple threads.

#### NoOpCallback

`#[derive(Default)] pub struct NoOpCallback` — implements `TrainingCallback` with all defaults. Used in tests to satisfy the callback API without any observable behavior.

#### LoggingCallback

`LoggingCallback { log_interval: usize }` — logs per batch at `batch_idx % log_interval == 0` and per epoch with format:

```
Epoch N: train_loss: X.XXXX, train_acc: XX.XX% [, val_loss: X.XXXX] [X.Xs]
```

Validation info is included via `Option::map` only when `val_loss` is present. The logging interval prevents log flooding on small batch sizes.

#### EpochMetrics as Callback Argument

`on_epoch_end` receives `&EpochMetrics` rather than individual values. `EpochMetrics` contains: `epoch`, `train_loss`, `val_loss: Option<f32>`, `train_accuracy`, `val_accuracy: Option<f32>`, `learning_rate`, `duration_secs`, `examples_seen`. This single type is also used by `TrainingMetrics::add_epoch` and the early stopping check, ensuring all consumers see the same values.

`EpochMetrics::get_metric(name: &str) -> Option<f32>` supports early stopping by looking up metrics by string name (`"val_loss"`, `"val_accuracy"`, `"train_loss"`, `"train_accuracy"`).

#### Early Stopping (in TrainingLoop, not callbacks)

`EarlyStopping` config on `TrainingLoop`:

- `monitor: String` — which metric to track
- `patience: usize` — epochs without improvement before stopping
- `min_delta: f32` — minimum improvement magnitude (default `1e-4`)
- `mode_max: bool` — `false` for loss (lower is better), `true` for accuracy (higher is better)

Logic in `TrainingLoop::train`:

```
if es.is_improvement(metric_val, state.best_metric):
    state.best_metric = metric_val
    state.patience_counter = 0
else:
    state.patience_counter += 1
    if state.patience_counter >= es.patience:
        metrics.early_stopped = true
        break
```

This runs after `on_epoch_end` callbacks are fired, meaning callbacks see metrics from the epoch that triggered early stopping.

#### Checkpoint

`Checkpoint { epoch, global_step, metrics: TrainingMetrics }` is a snapshot value. `TrainingLoop` fires `on_checkpoint` every `checkpoint_interval` epochs when `checkpoint_dir` is set. The checkpoint struct is passed to all callbacks; actual serialization is the callback's responsibility.

`TrainingLoop::resume_from(&Checkpoint)` restores `state.epoch`, `state.global_step`, and `metrics` — allowing training to continue from a saved checkpoint without re-training.

### Alternatives Considered

| Alternative                                       | Pros                              | Cons                                                                                      | Why Not                                                                                                        |
| ------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Event enum dispatched via channel                 | Decoupled, async-friendly         | Channel overhead per event, requires receiver thread                                      | Synchronous callbacks are simpler for the current single-threaded training loop                                |
| Closure-based callbacks (`Vec<Box<dyn Fn(...)>>`) | Less boilerplate for simple cases | Cannot hold state across calls without closure capture; separate closure per event type   | Trait allows stateful callbacks (e.g., accumulating metrics) and groups all hooks for one callback in one impl |
| Early stopping as a callback                      | Consistent abstraction            | Requires loop to query callback for stop decision after each epoch — inversion of control | Loop controls training; stopping is a loop decision, not a side effect                                         |
| `tracing` spans per epoch/batch                   | Structured observability          | Adds `tracing` as a hard dependency to `lattice-fann`                                     | `tracing` is optional; callbacks let callers integrate their own observability                                 |

## Consequences

### Positive

- `TrainingLoop` code never imports a logger or monitoring library — all output is delegated to callbacks
- New callbacks can be added without modifying the training loop
- `NoOpCallback` prevents tests from needing `#[allow(unused)]` or conditional compilation
- `TrainingMetrics::is_overfitting(window)` provides an additional signal beyond early stopping for post-hoc analysis

### Negative

- `on_checkpoint` receives the `Checkpoint` but cannot write it to disk without access to `checkpoint_dir` — the loop sets `checkpoint_dir` in config but does not pass it to the callback. Callers must remember the directory separately or capture it in the closure.
- `Vec<Box<dyn TrainingCallback>>` prevents compile-time checking of callback combinations; two callbacks that both write to the same file would conflict silently.

### Risks

- `EarlyStopping::is_improvement` uses `>` with `min_delta` offset. If `min_delta` is too large relative to the loss scale (e.g., `min_delta = 0.1` for a loss that ranges `0.0–0.01`), early stopping will never trigger. The default `1e-4` should be safe for typical MSE losses.
- `TrainingMetrics::is_overfitting` uses a sliding window average, which smooths noise but can lag real overfitting signals. It is provided for informational use, not for automatic stopping.

## References

- `crates/tune/src/train/loop/callbacks.rs` — `TrainingCallback` trait, `NoOpCallback`, `LoggingCallback`
- `crates/tune/src/train/loop/mod.rs` — `TrainingLoop::train`, callback dispatch, early stopping integration
- `crates/tune/src/train/loop/metrics.rs` — `EpochMetrics`, `TrainingMetrics`, `get_metric`, `is_overfitting`
- `crates/tune/src/train/loop/checkpoint.rs` — `Checkpoint` struct
- `crates/tune/src/train/config/early_stopping.rs` — `EarlyStopping` config and `is_improvement`
