# ADR-006: JIT Adaptation

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-tune

## Context

The distillation workflow produces a student model trained offline on teacher-labeled examples. However, at runtime, the system may encounter distribution shifts — new users, new conversation patterns, domain changes — where the offline student is no longer well-calibrated. JIT (Just-In-Time) adaptation is a mechanism for rapid online fine-tuning of the student model using a small batch of recent examples, with a strict wall-clock time budget (5–10 seconds).

The design must: (1) complete adaptation within a time budget, (2) support multiple strategies for which parameters to update, (3) stop early if loss plateaus, (4) operate on CPU for small models and GPU (via `GpuTrainer`) when available, and (5) provide presets for common use cases without requiring configuration expertise.

## Context on "JIT" Naming

JIT here means "triggered at inference time" — not LLVM JIT or kernel JIT. The adaptation runs between inference requests on recent production examples to shift the model toward current distribution. This is distinct from offline training (which operates on a full dataset without time constraints) and from online learning (which updates weights per-sample during inference).

## Decision

`JitAdapter` holds a `JitConfig` and exposes `adapt_cpu` and `adapt_gpu` (feature `gpu`) methods. `JitStrategy` enum encodes which parameters to update. Timeout and patience-based early stopping are evaluated inside the adaptation loop. Three built-in presets: `fast()` (5s), `thorough()` (10s), `few_shot()` (3s, small datasets).

### Key Design Choices

#### JitStrategy Variants

| Strategy                      | Parameters Updated                                                | Use Case                                                    |
| ----------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------- |
| `LastNLayers(n)` (default: 2) | Last N layers trainable, earlier frozen                           | Standard adaptation with stability                          |
| `HeadOnly`                    | Only the classification head (last layer)                         | Fastest, minimum parameter update                           |
| `GradualUnfreeze`             | All layers, LR scales linearly from `frozen_lr_multiplier` to 1.0 | When earlier layers also need adaptation                    |
| `LowRank { rank }`            | All base weights frozen; low-rank delta matrices                  | LoRA-style adaptation (in-memory, not saved as safetensors) |
| `Full`                        | All parameters                                                    | Maximum adaptation, slowest                                 |

`freeze::get_frozen_layers(strategy, total_layers) -> Vec<bool>` returns a per-layer freeze mask. `freeze::get_lr_multipliers(strategy, total_layers, frozen_mult) -> Vec<f32>` returns per-layer learning rate scale factors. For `GradualUnfreeze`, multipliers are linearly interpolated from `frozen_lr_multiplier` at layer 0 to `1.0` at the last layer.

#### JitConfig Fields

| Field                  | Default          | Meaning                                             |
| ---------------------- | ---------------- | --------------------------------------------------- |
| `strategy`             | `LastNLayers(2)` | Which parameters to update                          |
| `steps`                | 100              | Maximum gradient steps                              |
| `learning_rate`        | 0.001            | Base LR                                             |
| `frozen_lr_multiplier` | 0.1              | LR scale for frozen layers (GradualUnfreeze)        |
| `batch_size`           | 16               | Batch size per step                                 |
| `use_gpu`              | true             | Use GPU if available                                |
| `timeout_secs`         | 10.0             | Wall clock budget                                   |
| `patience`             | 10               | Steps without min_delta improvement before stopping |
| `min_delta`            | 1e-4             | Minimum loss improvement to reset patience          |
| `seed`                 | Some(42)         | Reproducible dataset shuffling                      |

#### Timeout Enforcement

The training loop checks `start.elapsed() > timeout` before each step. If exceeded, `timed_out = true` and the loop breaks. The result `JitResult::is_success()` returns `true` as long as `!timed_out && final_loss.is_finite()` — a timeout is not treated as a failure, just a time-boxed result.

This enables the caller to use the partially-adapted model regardless of whether the full step budget was consumed.

#### Patience-Based Early Stopping

Inside the adaptation loop, after each step:

```
if loss < best_loss - min_delta:
    best_loss = loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        early_stopped = true
        break
```

`JitResult::early_stopped` and `timed_out` are both reported independently, allowing callers to diagnose why adaptation ended.

#### Dataset Integration

`adapt_cpu` and `adapt_gpu` both create a `Dataset` from the provided `Vec<TrainingExample>`, configure it with the JIT batch size and seed, then iterate via `dataset.next_batch()` / `dataset.reset_epoch()` for cycling. This reuses the same `Dataset` / `Batch` infrastructure as offline training, including shuffle with deterministic LCG when `seed` is set.

#### GPU Path (feature `gpu`)

`adapt_gpu(trainer: &mut GpuTrainer, examples)` delegates each step to `trainer.train_batch(&batch)` (returning loss as `f32`) and `trainer.validate(&dataset)` after the loop for final accuracy. This is the same `GpuTrainer` used by the offline `TrainingLoop`, so the GPU path gets full optimizer shader support (SGD momentum, Adam, AdamW).

#### Current CPU Training Step

`train_step_cpu` is a placeholder that returns a simulated decreasing loss: `1.0 / (1.0 + step * 0.1) + |sin(step * 0.1)| * 0.1`. The actual CPU training integration with `lattice-fann`'s `BackpropTrainer` is not yet wired (marked as placeholder in the comment). The interface contract and lifecycle are fully defined.

### Alternatives Considered

| Alternative                                | Pros                                        | Cons                                                             | Why Not                                                                                           |
| ------------------------------------------ | ------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Separate JIT inference path (shadow model) | No main model modification during inference | Requires maintaining two models simultaneously; complex rollback | The registry's shadow evaluation (`shadow.rs`) handles A/B testing; JIT updates the primary model |
| Continual learning (per-request gradient)  | Immediate adaptation per example            | Risk of catastrophic forgetting; inference latency impact        | JIT batches examples and runs offline between inference requests                                  |
| Federated adaptation                       | Privacy-preserving distributed adaptation   | Significant infrastructure overhead                              | Out of scope for local Lion deployment                                                            |
| Hard timeout (OS timer)                    | Guarantees wall time                        | Difficult to interrupt safely inside training step               | Soft timeout check before each step is safe (no mid-step interruption)                            |

## Consequences

### Positive

- `JitResult` contains `loss_history`, `steps_completed`, `time_secs`, `early_stopped`, `timed_out` — full diagnostic picture for monitoring adaptation health
- Three presets remove configuration burden for common cases
- `freeze::get_frozen_layers` is a pure function, testable independently of the training loop
- Timeout check before each step means adaptation always terminates within `timeout_secs + (one step duration)`

### Negative

- CPU training step is a placeholder — the actual `BackpropTrainer` integration is not wired. The CPU path currently returns synthetic loss and does not modify any weights.
- `JitStrategy::LowRank` defines the concept but the layer-level LoRA matrix management is not implemented (no in-memory delta matrices are created; `freeze` just marks all layers as frozen). Full LowRank JIT requires creating and training `(A, B)` delta pairs in memory.
- `adapt_cpu` and `adapt_gpu` duplicate the timeout + patience loop structure. A shared inner loop helper would reduce this duplication.

### Risks

- The seed `Some(42)` default in `JitConfig` means all JIT adaptation sessions use the same shuffle sequence unless overridden. For adversarial inputs that are always presented in the same order, this could lead to consistently biased adaptation. Callers using production data should set `seed = None` for entropy-seeded shuffling.
- If `dataset.next_batch()` returns `None` (empty dataset after `reset_epoch`), the `train_step_cpu` path returns a `TuneError::Training("Dataset is empty")`. The GPU path handles this identically. The `adapt_cpu`/`adapt_gpu` guard (`if examples.is_empty()`) prevents the most obvious case but does not guard against filtered-empty datasets (e.g., if all examples are invalid).

## References

- `crates/tune/src/train/jit.rs` — `JitAdapter`, `JitConfig`, `JitStrategy`, `JitResult`, `freeze` module
- `crates/tune/src/train/gpu/mod.rs` — `GpuTrainer` (referenced by `adapt_gpu`)
- `crates/tune/src/data/dataset.rs` — `Dataset`, `Batch`, cycling via `next_batch`/`reset_epoch`
- `crates/tune/src/registry/shadow.rs` — shadow evaluation (complementary to JIT for A/B model comparison)
