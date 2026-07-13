# LoRA Router Refit

`lora::router_update` updates the small gate that selects an adapter for a
request. It accepts completed-request feedback rather than a supervised label
set: a caller supplies the same context vector that was routed, the adapter
index the feedback concerns, and a preference signal. The result is a complete
FANN gate binary, ready to replace the previous gate on the next call.

This module is available only with the `mixture` feature.

## State and inputs

`update_router` operates on four persistent inputs plus a feedback batch:

| Input                | Role                                                   | Mutated by refit?               |
| -------------------- | ------------------------------------------------------ | ------------------------------- |
| `gate_bytes`         | Current FANN gate, serialized by `Network::to_bytes()` | No; decoded into a working gate |
| `events`             | Non-empty current feedback batch                       | No                              |
| `ReplayBuffer`       | Bounded FIFO history of known-good routes              | Yes                             |
| `DiagonalFisher`     | Per-parameter importance EMA and parameter anchor      | Yes                             |
| `RouterUpdateConfig` | RLOO, replay, and EWC settings                         | No                              |

The returned `RouterDelta::network_bytes` is a full `Network::to_bytes()`
payload, not a parameter diff. It can be passed back as `gate_bytes` to a later
refit or restored with `Network::from_bytes()`. `events_consumed` is always the
current batch length. `replay_accuracy` is the post-refit top-1 accuracy on the
entire, updated replay buffer, or `None` when that buffer is empty.

An event's `adapter_id` is retained for audit and tracing; gradient computation
uses only its context, adapter index, and signal. The context must be the
vector supplied to the gate when the request was made. Reconstructing a
different context at feedback time trains the router against a different
decision than the one being judged.

## Feedback polarity and RLOO

`PreferenceSignal` maps to a fixed signed reward:

| Signal             | Reward | Meaning for the named adapter                    |
| ------------------ | -----: | ------------------------------------------------ |
| `Positive`         | `+1.0` | Strong evidence it should be selected            |
| `ImplicitPositive` | `+0.5` | Weak evidence it should be selected              |
| `ImplicitNegative` | `-0.5` | Weak evidence it should be selected less often   |
| `Negative`         | `-1.0` | Strong evidence it should be selected less often |

The reward sign is the only polarity mechanism. Every event calls
`RlooTrainer::step(gate, context, action_idx, reward)`; there is no separate
negative-feedback branch. In the single-sample RLOO (`M = 1`) policy-gradient
update, the relevant logit gradient has the form:

```text
reward * (p - onehot(action))
```

Consequently, a positive reward increases the named adapter's routing
probability and a negative reward decreases it. Splitting positive and
negative events into separate update logic would risk reversing that invariant
or treating a negative signal as an alternative preferred adapter.

The RLOO trainer also receives the configured auxiliary load-balancing loss and
router z-loss. The former discourages all requests from collapsing onto one
adapter; the latter discourages unbounded router logits. Both coefficients may
be zero to disable their respective auxiliary term.

## Configuration defaults and bounds

`RouterUpdateConfig::default()` is intended as a conservative starting point:

| Setting               | Default | Accepted values    | Effect                                              |
| --------------------- | ------: | ------------------ | --------------------------------------------------- |
| `learning_rate`       |  `1e-3` | Finite and `> 0`   | SGD step magnitude                                  |
| `aux_loss_coeff`      |  `0.01` | Finite and `>= 0`  | Load-balancing penalty weight                       |
| `z_loss_coeff`        | `0.001` | Finite and `>= 0`  | Logit-growth penalty weight                         |
| `epochs`              |     `5` | Integer `> 0`      | Complete passes over new plus sampled replay data   |
| `ewc_lambda`          |   `0.1` | Finite and `>= 0`  | Reserved anchor-penalty coefficient; inactive in v1 |
| `replay_mix_fraction` |   `0.5` | Finite in `[0, 1]` | Fraction of stored replay entries sampled per refit |

The replay fraction is converted to a count with `ceil`, so any positive
fraction selects at least one entry when the buffer is non-empty. A fraction of
zero disables replay and a fraction of one selects the full buffer.

## Refit algorithm

The refit is deliberately ordered so invalid caller state is rejected before
the feedback loop mutates the gate, replay buffer, or Fisher state.

1. Reject an empty feedback batch, then deserialize the FANN gate.
2. Read the gate input dimension, output count, and flattened parameter count.
3. Validate every event as a batch: its context length must equal the gate's
   input dimension, its adapter index must be smaller than the output count,
   and every context value must be finite. One bad event rejects the complete
   batch.
4. Validate the configuration and the Fisher state. An empty Fisher is
   initialized with zero-valued importance and anchor vectors sized to the
   gate. A non-empty Fisher must already match the gate parameter count and
   contain valid values.
5. Construct `RlooTrainer` from the learning rate, auxiliary-loss coefficient,
   and z-loss coefficient.
6. Take a replay sample before changing the replay buffer. The sample size is
   `ceil(replay.len() * replay_mix_fraction)`, and entries are spaced through
   the buffer rather than taken only from its oldest end.
7. For each epoch, update first on every new event with its signed reward,
   then on every sampled replay entry with reward `+1.0`.
8. Snapshot the final parameters into the Fisher anchor.
9. Append only newly positive events to replay, evicting the oldest entry if
   capacity has been reached.
10. Measure top-1 replay accuracy with the updated gate, serialize it, and
    return the `RouterDelta`.

The replay sample is cloned before training so the buffer can be updated after
the epochs without invalidating borrowed entries. Its deterministic sampler
uses `ceil(len / n)` as an iteration stride, takes at most `n` entries, and
therefore spreads a small sample across the buffer's history. A zero requested
sample or an empty buffer produces no replay steps.

## One gradient step and Fisher projection

Each event or replay entry is processed with the following transaction:

1. Flatten the gate parameters in forward layer order: row-major weights,
   followed by biases, for each layer.
2. Let the RLOO trainer perform its ordinary SGD step.
3. Flatten the new parameters and calculate the raw update
   `delta = after - before`.
4. Approximate the gradient as `-delta / learning_rate`. The implementation
   uses `max(abs(learning_rate), 1e-10)` as the denominator guard; public
   validation still requires a finite, strictly positive learning rate.
5. Feed that gradient to the diagonal Fisher EMA. Each importance entry is a
   squared-gradient moving average:

   ```text
   F_i <- decay * F_i + (1 - decay) * gradient_i^2
   ```

6. Call `DiagonalFisher::project_delta` on the raw update. High-importance
   parameters have their updates damped toward zero; parameters whose Fisher
   importance is near zero pass through with little damping.
7. Restore the gate as `before + projected_delta`.

This is the v1 anti-forgetting mechanism. It protects parameters that the
recent Fisher history identifies as important while leaving lower-importance
degrees of freedom available to learn new routing feedback.

The parameter anchor is updated after the whole refit, but the projection path
does not use it. `RouterUpdateConfig::ewc_lambda` is validated and retained for
the alternative anchor-pullback EWC penalty (`penalty_gradient`), but it has no
effect on v1 gate bytes. Changing `ewc_lambda` alone must therefore not change
the result of an otherwise identical v1 refit.

## Replay semantics

Replay stores only positive signals. A positive event tells the system which
adapter should be selected, so it can safely reappear as a `+1.0` training
example. A negative event only says that one adapter should be selected less
often; it does not identify a replacement and is therefore never placed in
replay.

`ReplayBuffer` is a FIFO with a caller-supplied maximum size. Pushing when it
is full removes the oldest entry first. A zero-capacity buffer intentionally
accepts no entries. Refit adds new positive events only after all training
epochs finish, so current feedback is not replayed multiple extra times within
the same call.

Replay accuracy is an operational indicator, not a new optimization target.
For each stored positive route, the gate is run and its largest output index is
compared with the stored adapter index. A forward failure or an output that
cannot produce a comparable maximum counts as incorrect rather than aborting
the metric calculation.

## Validation contract

All of the following reject the call with `TuneError::Validation` before the
training loop begins:

| Value                                          | Required condition                                                                                                     |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `events`                                       | Non-empty                                                                                                              |
| Event context                                  | Exactly the gate input length; every value finite                                                                      |
| Event adapter index                            | Less than the gate output count                                                                                        |
| `learning_rate`                                | Finite and `> 0`                                                                                                       |
| `aux_loss_coeff`, `z_loss_coeff`, `ewc_lambda` | Finite and `>= 0`                                                                                                      |
| `replay_mix_fraction`                          | Finite and in `[0, 1]`                                                                                                 |
| `epochs`                                       | Greater than zero                                                                                                      |
| Fisher decay                                   | Finite and strictly in `(0, 1)`                                                                                        |
| Empty Fisher                                   | Auto-initialized to the gate parameter count                                                                           |
| Non-empty Fisher                               | Importance and anchor lengths equal parameter count; importance entries finite and non-negative; anchor entries finite |

Gate deserialization, RLOO stepping, Fisher observation, and anchor updates
that fail are surfaced as `TuneError::Training`. The validation boundary is
important because non-finite inputs or invalid Fisher state would otherwise
contaminate the gate and the persisted online-learning state.

## Example lifecycle

```rust,no_run
use lattice_fann::{Network, training::DiagonalFisher};
use lattice_tune::lora::router_update::{
    update_router, FeedbackEvent, PreferenceSignal, ReplayBuffer,
    RouterUpdateConfig,
};

let initial_gate: Network = /* construct a gate */ unimplemented!();
let mut gate_bytes = initial_gate.to_bytes();
let mut replay = ReplayBuffer::new(1_024);
let mut fisher = DiagonalFisher::new(0, 0.99)?; // Empty state auto-initializes.

let events = vec![FeedbackEvent {
    context_vector: vec![/* exact routing context */],
    preferred_adapter_idx: 2,
    adapter_id: "legal-domain".to_owned(),
    signal: PreferenceSignal::Positive,
}];

let delta = update_router(
    &gate_bytes,
    &events,
    &mut replay,
    &mut fisher,
    &RouterUpdateConfig::default(),
)?;
gate_bytes = delta.network_bytes;
# Ok::<(), lattice_tune::TuneError>(())
```

Persist `gate_bytes`, replay state, and Fisher state together when continuity
across process restarts matters. Restoring only the gate discards the
anti-forgetting history; restoring only the Fisher state with a different gate
will fail its parameter-count validation.

## `update_router`

`update_router` is an all-or-nothing refit. It takes a FANN gate payload, a
non-empty batch of feedback, mutable replay and Fisher state, and a
`RouterUpdateConfig`; it returns a complete replacement FANN payload rather
than a parameter patch. The caller must retain the returned bytes together
with the updated replay and Fisher values to preserve online-learning
continuity.

Before changing any state, the function deserializes the gate and validates
every event against its input and output dimensions, including finite context
values. It also validates the configuration and either initializes an empty
Fisher to the gate's parameter count or proves that an existing Fisher has
matching finite state. The refit then combines all current events with a
deterministic replay sample, performs the configured number of epochs,
anchors the Fisher at the final parameters, adds only current positive events
to replay, measures post-refit replay accuracy, and serializes the gate.

Both feedback polarities deliberately use the same `RlooTrainer::step` path.
The signed reward from `PreferenceSignal::reward()` carries the direction:
positive feedback raises the named adapter's score, while negative feedback
lowers it. A separate negative-feedback path would risk treating an adapter
known to be bad as though it named a replacement.

The function reports `TuneError::Validation` for bad event dimensions or
indices, non-finite contexts, invalid configuration, or invalid Fisher state.
Gate deserialization, policy-gradient, Fisher-observation, and anchor-update
failures are surfaced as `TuneError::Training`.

## `RouterUpdateConfig::ewc_lambda`

`ewc_lambda` belongs to the planned anchor-pullback (`penalty_gradient`) EWC
path. It is already validated and retained in configuration so callers can
persist a stable config shape, but the v1 refit uses only
`DiagonalFisher::project_delta`; that projection does not consult
`ewc_lambda`. Therefore changing only this value cannot change otherwise
identical v1 gate bytes. A zero value remains a useful declaration of intended
no-regularization behavior for a future penalty-gradient implementation.

## `ReplayBuffer`

Replay contains only positive `(context_vector, preferred_adapter_idx)`
observations. Positive feedback identifies an adapter to reinforce, so replay
can safely train it with `+1.0`; negative feedback identifies only an adapter
to suppress and cannot name a correct replacement. The buffer is a bounded
FIFO: a full buffer evicts its oldest entry before accepting a new one, while a
zero-capacity buffer retains nothing.

Sampling is deterministic and spread across the insertion history. For an
effective requested count `n`, the implementation uses `ceil(len / n)` as the
step and takes at most `n` entries. Samples are cloned before refit, and new
positive events are appended only after all refit epochs, so current feedback
does not receive extra replay updates in its initial call.

## `one_gradient_step`

Each update snapshots the gate's flattened parameters in forward layer order
(row-major weights followed by biases), lets RLOO apply its signed policy
gradient, and measures `raw_delta = after - before`. It approximates the
gradient as `-raw_delta / max(abs(learning_rate), 1e-10)` before observing it
in the diagonal Fisher EMA. `project_delta` then damps updates at parameters
with high accumulated squared-gradient importance, and the gate is restored
as `before + projected_delta`.

This projection is the v1 anti-forgetting mechanism. It protects parameters
that recent feedback identifies as important to earlier routing behavior while
leaving low-importance directions free to adapt. The reward sign is preserved
through every stage: a positive reward increases the selected output and a
negative reward decreases it.
