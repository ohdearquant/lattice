# Training

`lattice-fann` ships three training tools:

- [`BackpropTrainer`](#backprop--momentum) — the default, always-available trainer:
  full-batch or mini-batch gradient descent with momentum and weight decay.
- [`DiagonalFisher`](#elastic-weight-consolidation-ewc) (`training::ewc`, feature
  `online-router`) — an Elastic Weight Consolidation forgetting guard, composed
  alongside another trainer rather than used standalone.
- [`RlooTrainer`](#reinforce-with-leave-one-out-rloo) (`training::rloo`, feature
  `online-router`) — a REINFORCE policy-gradient trainer for selector-gate networks
  (e.g. adapter/expert routers), with an optional leave-one-out variance reducer.

Backprop and RLOO operate directly on `Network`/`Layer` buffers — there is no
separate training-graph representation. Their gradients are computed by
hand-rolled reverse-mode backprop over the layer stack, not by a general
autodiff engine. EWC instead remains deliberately independent of `Network` and
works on caller-provided flat parameter slices.

## Availability and shared types

The basic training surface is always present. `BackpropTrainer`,
`TrainingConfig`, `TrainingResult`, `Trainer`, and `GradientGuardStrategy` are
available without optional features. `DiagonalFisher`, `RlooConfig`, and
`RlooTrainer`, together with their `training::ewc` and `training::rloo`
modules, are compiled only with the `online-router` feature.

`TrainingConfig` is the configuration used by the `Trainer` implementation;
its builder methods simply replace the corresponding field and return the
configuration. The defaults are:

| Field            |  Default | Meaning                                                                           |
| ---------------- | -------: | --------------------------------------------------------------------------------- |
| `learning_rate`  |   `0.01` | Base step size before batch-size scaling.                                         |
| `momentum`       |    `0.9` | Coefficient applied to the prior weight and bias velocities.                      |
| `weight_decay`   | `0.0001` | L2 coefficient applied to weights, not biases.                                    |
| `max_epochs`     |   `1000` | Maximum completed epoch count.                                                    |
| `target_error`   |  `0.001` | Strict early-stop threshold for mean squared error.                               |
| `batch_size`     |     `32` | Samples accumulated before an update; `1` is SGD.                                 |
| `shuffle`        |   `true` | Shuffle sample indices before each epoch.                                         |
| `gradient_guard` |  `Error` | Response to a non-finite accumulated gradient.                                    |
| `seed`           |   `None` | Entropy-seeded shuffle RNG; `Some(seed)` makes its shuffle sequence reproducible. |

The builder does not normalize hyperparameters or impose ranges on learning
rate, momentum, weight decay, epoch count, or target error. Callers therefore
own choosing finite, meaningful values. `batch_size` is the exception:
`BackpropTrainer::train` rejects zero before training starts.

### Dataset validation and results

`Trainer::train` expects a nonempty input collection and exactly one target
vector per input vector. Each input must have `network.num_inputs()` values and
each target must have `network.num_outputs()` values; a mismatch returns a
`TrainingError` before velocities or gradient buffers are initialized.

`TrainingResult` records the final per-sample mean squared error,
`epochs_trained`, an `error_history` entry for every completed epoch, and the
`converged` flag. An epoch converges only when its average error is _strictly_
less than `target_error`. If `max_epochs` is zero, no epoch runs: the result is
non-converged, has an empty history, and reports `f32::MAX` as its final error.

## Backprop + momentum

`BackpropTrainer` (`training/backprop.rs`) implements standard SGD with momentum
and L2 weight decay, driven by a `TrainingConfig`:

```rust
use lattice_fann::{NetworkBuilder, Activation, BackpropTrainer, TrainingConfig, Trainer};

let mut network = NetworkBuilder::new()
    .input(2)
    .hidden(4, Activation::Tanh)
    .output(1, Activation::Tanh)
    .build()
    .unwrap();

let mut trainer = BackpropTrainer::new();
let config = TrainingConfig::new().learning_rate(0.5).max_epochs(1000);
let result = trainer.train(&mut network, &inputs, &targets, &config);
```

The trainer initializes fresh, zeroed velocity buffers at the start of every
`train` call. Momentum is consequently preserved across batches and epochs of
one call, but not across two separate calls on the same `BackpropTrainer`.

### Per-sample gradient computation

`compute_gradients` runs one forward pass per sample, then walks the layer stack
backward:

1. **Output-layer delta.** The implementation uses the derivative of the
   unscaled half-squared-error term, while it reports separately normalized MSE.
   For a Softmax output layer, the _full_ Jacobian is used (not the diagonal
   approximation — see [Activation reference](network.md#activation-reference)):
   `delta[i] = Σ_j (output[j] − target[j]) · J[j,i]` where `J[j,i] = output[j]·(δ_ij − output[i])`.
   For every other output activation, `delta[i] = (output[i] − target[i]) · f'(output[i])`.
   The reported sample MSE is `Σ_i (output[i] − target[i])² / output_width`; its
   formal derivative would additionally include `2 / output_width`.
2. **Hidden-layer deltas**, propagated backward layer by layer:
   `delta_i^(l) = f'(a_i^(l)) · Σ_j W_ji^(l+1) · delta_j^(l+1)`, reading the
   next layer's weight matrix in its row-major `[out * num_inputs + in]` layout.
3. **Weight/bias gradients**: `dW[i,j] = delta[i] · input[j]`, `dB[i] = delta[i]`,
   accumulated over every sample in the batch before the update step runs.

Both `BackpropTrainer::compute_gradients` and `RlooTrainer::backprop_and_apply`
(below) implement the _same_ backward recursion independently — the RLOO
trainer's comment block explicitly cites the `backprop.rs` line numbers it
mirrors. A bug found in one hidden-layer backward loop should be checked
against the other.

### Momentum + weight decay update

Once gradients are accumulated over a batch, `apply_gradients` runs classic
momentum SGD, one velocity buffer per weight and bias. Weight updates are:

```text
v ← momentum · v − lr · (grad + weight_decay · w)
w ← w + v
```

`lr` is `config.learning_rate / batch_size` — gradients are summed (not
averaged) over the batch, so the learning rate divides by the actual batch
size to keep the effective step size independent of batch size. Biases use the
same velocity recurrence but omit the `weight_decay · w` term.

The final batch of an epoch may contain fewer than `config.batch_size` samples;
the divisor is that final batch's actual size, not the configured maximum.

### Epoch flow and error accounting

For each epoch, the trainer optionally shuffles an index vector, clears the
reused gradient buffers for each batch, and sums the per-sample gradients. A
sample's error is the squared output difference summed over outputs and divided
by the output width. The epoch error is the sum of accepted sample errors
divided by the count of accepted samples.

This separation matters for `SkipBatch`: a rejected batch contributes neither
an update nor its error. If all batches are skipped, the denominator would be
zero and the epoch metric would be meaningless, so training returns
`NumericInstability` immediately. The trainer also rejects a completed epoch
whose average error is `NaN` or infinite instead of returning a misleading
result.

### Gradient guard strategies

Every batch's accumulated gradients are checked for NaN/Inf before the update
is applied (`GradientGuardStrategy`, in `training/gradient.rs`):

| Strategy          | Behavior                                                                               |
| ----------------- | -------------------------------------------------------------------------------------- |
| `Error` (default) | Abort the whole `train()` call with `NumericInstability`                               |
| `Sanitize`        | Zero out NaN/Inf entries and apply the update anyway; batch error is still counted     |
| `SkipBatch`       | Discard the batch's gradients _and_ its error contribution, continue to the next batch |

If every batch in an epoch is skipped under `SkipBatch`, `train()` returns
`NumericInstability` rather than a division-by-zero-derived `NaN` `final_error`
— `batch_count == 0` is a fail-loud condition, not silently reported as
"converged" or "error 0.0".

The guard examines accumulated weight and bias gradients before an update; it
does not make invalid configuration values or non-finite forward-pass errors
safe. In particular, `Sanitize` replaces only non-finite gradient entries with
zero. It cannot repair a non-finite error metric, which is still rejected at
the end of the epoch.

### Softmax hidden-layer limitation

`NetworkBuilder::build()` rejects Softmax on any hidden (non-output) layer at
construction time. Softmax normalizes over the whole layer's output simplex,
so its gradient is inherently a full Jacobian, not a per-element derivative;
allowing it on a hidden layer would require every caller of
`Activation::derivative` (a per-element `f32 -> f32` function) to instead
carry the whole-layer Jacobian, which the trainer doesn't implement for hidden
layers (see ADR-023). This is enforced once, at build time, so it can never
surface as a silent wrong-gradient bug during training.

## Elastic Weight Consolidation (EWC)

`training::ewc::DiagonalFisher` (feature `online-router`) prevents catastrophic
forgetting during online/continual learning by penalizing updates to parameters
that were important for a prior task. It operates on flat `&[f32]` parameter
slices with no `Network` dependency, so it is suitable for caller-controlled
training loops that can inspect gradients or updates before writing parameters.

References: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
networks", PNAS 2017; Chaudhry et al., "Efficient Lifelong Learning with
A-GEM", ICLR 2019 (the EWC++ online variant this module implements).

### State and lifecycle

`DiagonalFisher` contains two same-length vectors and an EMA decay:

| State    | Role                                                                       |
| -------- | -------------------------------------------------------------------------- |
| `values` | The diagonal Fisher estimate `F`, one importance value per parameter.      |
| `anchor` | The parameter snapshot `θ*` that a later task is discouraged from leaving. |
| `decay`  | The fraction of the previous importance retained by the next observation.  |

The caller owns the parameter layout because the guard intentionally has no
`Network` dependency. Use the same flat ordering for every gradient, parameter
snapshot, gradient buffer, and update slice passed to a guard. Construction
creates zeroed `values` and `anchor` vectors; `observe_gradient` records prior
task gradients, and `set_anchor` records the associated finished-task
parameters before training a new task. Then either add `penalty_gradient` to
the new task's gradient or run `project_delta` on a prospective update.

The public vectors should be treated as one unit. The constructor and setter
establish equal lengths, but modifying `values` or `anchor` independently can
break the correspondence on which the penalty relies. `observe_gradient` and
`penalty_gradient` validate their input/output slices against the Fisher length;
`set_anchor` validates against the current anchor length. `project_delta` is
the intentional exception: it processes the common prefix, allowing a guard to
cover a parameter sub-slice without an error.

### Diagonal Fisher information as an importance estimate

The full Fisher information matrix is `num_params × num_params` — infeasible to
track for anything but tiny networks. EWC++ instead tracks only the **diagonal**:
one scalar per parameter, updated online as an exponential moving average (EMA)
of squared gradients:

```text
F_i ← decay · F_i + (1 − decay) · g_i²
```

A large `F_i` means parameter `i`'s gradient has consistently had large
magnitude — i.e. the loss is sensitive to it, so it was "important" to
whatever task produced those gradients. `decay` must be strictly inside the
open interval `(0, 1)`: `decay = 1` freezes the estimate, while `decay > 1`
makes `(1 − decay)` negative and reverses the contribution of new
squared-gradient signal. `decay = 0` collapses to pure replacement with no
memory at all (the EMA degenerates to `F_i = g_i²` every step). Both ends, plus
non-finite `decay`, are rejected in `DiagonalFisher::new` before any allocation.

The constructor also checks its allocation request against the crate's safety
limit before allocating either vector. Gradient observations themselves are not
silently sanitized: callers should provide finite gradients and a finite
regularization coefficient so that the stored importance and subsequent update
remain finite.

### Anchor + penalty gradient

At a task boundary, `set_anchor(params)` snapshots the current parameter
vector `θ*` as the reference point for the _next_ task's training. The EWC
regularization loss is:

```text
L_ewc = (λ/2) · Σ_i F_i · (θ_i − θ*_i)²
```

`penalty_gradient` computes `∂L_ewc/∂θ_i = λ · F_i · (θ_i − θ*_i)` and
_adds_ it into the caller's gradient buffer (the caller is expected to combine
it with the task loss's own gradient and then subtract the combined vector in
its own descent step — this module never mutates network weights directly).
Because the gradient is proportional to `F_i`, parameters that were unimportant
to the prior task (`F_i ≈ 0`) are essentially unconstrained, while parameters
that were highly important are pulled back toward their anchor value in
proportion to both their importance and how far they've drifted.

`penalty_gradient` accumulates rather than overwrites `out`. Start with the
task-loss gradient (or another deliberate initial value), add the EWC term, and
perform the ordinary descent update only after both are present. This behavior
lets multiple regularizers share one gradient buffer, but it also means a stale
buffer will be counted again if the caller fails to clear it between updates.

### Null-space delta projection

`project_delta` offers a second, distinct use of the same Fisher estimate: given
a raw parameter update `delta` (e.g. a policy-gradient step, not necessarily one
this module computed), damp the components that touch high-Fisher parameters:

```text
scale_i = max(0, 1 − F_i / F_max)
delta_i ← delta_i · scale_i
```

Parameters at the maximum observed Fisher value are scaled to (near) zero —
their update is blocked entirely. Parameters with `F_i = 0` pass through
unchanged (`scale_i = 1`). This is a diagonal (per-parameter) approximation to
projecting the update into the null space of the Fisher matrix; a full
SVD-based null-space projection is more accurate but `O(n³)`, and is deferred
in favor of this `O(n)` approximation, which is judged adequate for online
continual learning where updates arrive one sample or small batch at a time.

**Degenerate-Fisher guard**: if no gradient has ever been observed
(`F_max < 1e-8`), `project_delta` returns immediately without modifying
`delta` — there's no importance signal yet, so treating it as identity avoids
a division by (near-)zero.

Projection does not use the anchor and is not equivalent to adding the penalty
gradient. It is a direct, relative suppression of an already computed update:
the largest Fisher entry receives scale zero, while lower-importance entries
receive a scale relative to that maximum. Choose one approach deliberately, or
combine them only when the resulting amount of protection is intended.

### API error contracts

`DiagonalFisher::new` rejects a non-finite or out-of-range `decay` with
`InvalidDistributionParams` and a too-large parameter count with
`ShapeTooLarge`. `observe_gradient`, `set_anchor`, and `penalty_gradient`
return `InputSizeMismatch` when their full-length slice arguments do not match
the guard. `project_delta` deliberately differs: it processes only the common
prefix so a guard can be applied to a parameter sub-slice.

## REINFORCE with Leave-One-Out (RLOO)

`training::rloo::RlooTrainer` (feature `online-router`) trains a _selector gate_
— a small `Network` whose job is to output logits over a set of discrete
choices (e.g. which adapter/expert to route to) — via policy gradient. It is
intentionally decoupled from EWC: `DiagonalFisher` can protect
caller-controlled parameter updates when forgetting protection is needed, while
this trainer implements only the policy-gradient update.

**Gate contract**: the gate network's _output_ layer activation must be
`Activation::Linear` (raw logits). Softmax is applied inside this module's loss
computation, not baked into the network, because the policy-gradient formulas
below need direct access to both the logits (for the log-sum-exp z-loss term)
and the softmax probabilities.

### Configuration and validation

`RlooConfig` has deliberately small, direct settings:

| Field            | Default | Effect                                               |
| ---------------- | ------: | ---------------------------------------------------- |
| `learning_rate`  | `0.001` | Plain-SGD step size for all gate weights and biases. |
| `aux_loss_coeff` |  `0.01` | Multiplier for the load-balance gradient.            |
| `z_loss_coeff`   | `0.001` | Multiplier for the router z-loss gradient.           |

The trainer does not use `TrainingConfig`: it updates one policy-gradient
example at a time, with neither momentum nor weight decay. `new` seeds its
sampling RNG from system entropy; `with_seed` fixes that stream for repeatable
multi-sample tests or experiments.

Both `step` and `rloo_step` check that `context` has the gate's input width,
that the action/preferred index is an output index, and that the output layer is
linear. `rloo_step` additionally requires `1 <= k <= num_outputs` and
`m_samples >= 1`. Before it allocates retained subsets, it bounds both
`m_samples` and `m_samples * k` against the crate allocation limit and uses a
checked multiply to turn an overflowing product into an error. The configuration
coefficients and `reward` itself are not range-checked, so finite, appropriate
values remain the caller's responsibility.

EWC is not injected into either RLOO update path. `DiagonalFisher` remains a
separate, flat-slice guard for integrations that control their own parameter
updates; `RlooTrainer` has no automatic EWC penalty or projection hook.

### Phase 1: single-sample REINFORCE (`step`, the active path)

Given a context vector, an `action_idx` (which output the reward is about), and
a scalar `reward` (`+1.0`/`−1.0` explicit or `+0.5`/`−0.5` implicit — magnitude
encodes confidence, sign encodes direction), `step` computes a single-sample
policy-gradient update:

1. Forward pass through the gate → `logits`, then `probs = softmax(logits)`.
2. Output-layer gradient combines three terms:
   ```text
   g[j] = reward · (p[j] − onehot(action)[j])                         (policy)
        + aux_coeff · (2/K) · p[j] · (p[j] − 1/K − c)                  (load-balance)
        + z_coeff   · 2 · LSE · p[j]                                    (z-loss)
   ```
   where `c = Σ p_i² − 1/K` and `LSE = log Σ exp(logits)`. Because the output
   activation is `Linear` (derivative 1), this _is_ the pre-activation error —
   no extra derivative multiply, unlike the general backprop path.
3. `backprop_and_apply` propagates this delta through hidden layers (mirroring
   `backprop.rs`'s hidden-layer recursion and plain-SGD apply, batch size 1,
   no momentum) and updates the gate's weights in place.

The returned scalar is the policy term only,
`−reward · log(max(p[action_idx], 1e-9))`, for logging. It does not include the
auxiliary losses and is not a full objective value. The probability floor keeps
the logging expression finite when a probability underflows; it does not alter
the gradient formula used for the update.

**Single code path for both reward polarities.** The policy term
`reward · (p[j] − onehot[j])` naturally flips sign with `reward`: a positive
reward pulls `p[action]` up (standard cross-entropy-style gradient toward the
action), a negative reward pushes it down. There is deliberately no
special-casing for negative reward — a regression test
(`rloo_negative_reward_decreases_action_score`) pins this polarity, because a
plausible-looking bug here (e.g. accidentally taking `reward.abs()`) would
silently convert "push this down" feedback into "pull this up" feedback.

**Auxiliary losses**, always added regardless of reward sign:

- _Load-balance loss_ `(1/K) Σ_i (p_i − 1/K)²` penalizes routing collapse
  (one action absorbing all probability mass) — pulls the distribution toward
  uniform.
- _Router z-loss_ `(log Σ exp(logits))²` penalizes logit magnitude explosion,
  which otherwise makes the softmax increasingly saturated/overconfident over
  training.

The public `load_balance_aux_loss` and `router_z_loss` functions expose these
scalar definitions for inspection or external logging. The load-balance helper
returns zero for an empty probability slice. The z-loss helper defines
log-sum-exp of an empty logits slice as zero, and therefore also returns zero.
For nonempty logits, the internal softmax and log-sum-exp routines subtract the
maximum logit before exponentiating to avoid overflow.

### Phase 2: multi-sample RLOO (`rloo_step`, not the default path)

`rloo_step` draws `m_samples` independent top-`k` subsets via Gumbel-max
sampling (perturb each logit with `Gumbel(0,1)` noise, take the top `k` by
perturbed value — the standard trick for sampling a `k`-subset without
replacement in proportion to the softmax distribution) and computes a reward
per sample: `+1.0` if `preferred_idx` is in the sampled subset, `−1.0`
otherwise. It then applies a **leave-one-out baseline** to reduce gradient
variance:

```text
baseline_m = (Σ R − R_m) / (M − 1)     (0 if M == 1)
advantage_m = R_m − baseline_m
g[j] += −(1/M) · advantage_m · (count(j ∈ subset_m) − k · p[j])
```

This is not the default training path — it exists for the bench harness that
compares its convergence against Phase 1. **Invariant for any future
activation of this path**: it only consumes preferred-known (positive) events,
so it must always be paired with the Phase-1 negative path; a positive-only
convergence run collapses to near-zero policy entropy (mass piles onto a
single output) because there's no signal pushing a wrongly-selected output
back down. This was verified empirically in the convergence bench, not just
reasoned about — treat it as a correctness requirement for any caller, not a
tuning knob.

Both `m_samples` and `m_samples * k` (the aggregate retained-subset storage)
are bounds-checked against `MAX_ALLOWED_ELEMENTS` before any allocation, since
both are caller-controlled and the product can overflow or drive an
unreasonably large `Vec::with_capacity`.

#### Sampling and update sequence

For each sample, the trainer draws one Gumbel perturbation per logit,

```text
u_i ~ Uniform(0, 1)
g_i = s_i - ln(-ln(u_i))
subset = indices of the largest k values of g
```

and assigns reward `+1` when `preferred_idx` appears in that subset or `−1`
otherwise. The uniform draw is clamped strictly inside the open interval before
the nested logarithms, which prevents `ln(0)` at an endpoint. Gumbel-top-k
therefore creates a top-`k` sample without replacement whose ordering is driven
by the policy logits.

The leave-one-out baseline for sample `m` excludes that sample's reward. For a
single draw it is defined as zero, avoiding a division by zero; for multiple
draws it makes the update respond to how the sample compared with its peers.
After accumulating the policy-gradient contributions, the trainer adds the same
load-balance and z-loss gradients used by Phase 1, backpropagates through each
hidden layer, and applies plain SGD to every weight and bias.

As with the ordinary backprop trainer, hidden-layer deltas read the next layer's
row-major weight matrix at `[output * num_inputs + input]`, multiply by the
next-layer deltas, and multiply by the current activation derivative. The RLOO
path computes this recurrence independently so the policy-loss output delta can
replace the normal MSE output delta.
