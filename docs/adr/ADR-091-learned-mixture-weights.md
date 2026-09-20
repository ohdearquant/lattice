# ADR-091: Learned mixture weights over a resident LoRA pool

**Status**: Accepted (2026-09-17)
**Date**: 2026-09-16
**Crate**: lattice-inference / lattice-fann / lattice-tune

## Context

ADR-079 records the composed adapter loop: a governed pool, a decode-time weighted blend, and a
learned selector that picks which adapters take part. The selector is learned; the mixture weights
are not. `AdapterRouter::route` computes one gate score per candidate adapter, partial-sorts to the
top `k`, sorts that prefix for determinism, and then assigns `1.0 / k` to every survivor
(`crates/inference/src/mixture.rs:205-232` at `19ec0cc606`). The scores it ranks by are discarded at
the same statement that would use them.

That is a deliberate decision with its reason written down, and this ADR exists to reverse it, so
the reason is quoted rather than paraphrased. From the type's own doc comment
(`crates/inference/src/mixture.rs:105-109`):

> Weights are constant and uniform (`1/k`). A learnable-softmax router can collapse to effectively
> one adapter under sparse, noisy reward, so the weights are fixed and only the selector is learned.

Two things have changed since that was written.

- The loop closes in-process. `AdapterRouter::reload` landed on 2026-09-12 (`7c48c8968f`) and swaps
  the gate atomically: it validates both dimensions and returns `RouterError::GateDimensionMismatch`
  without mutating `self`, so a refit that should not take effect costs one failed reload rather
  than an outage (`crates/inference/src/mixture.rs:130-145`). A refit-rejection policy is therefore
  cheap to express, which it was not when the weights were fixed.
- Several adapters are about to be resident and selectable per request in the serving path. While
  the pool was a training-time construct, the weight question was theoretical. It is not now.

Reachability, stated because it bounds the risk of the change rather than excusing it: `mixture` is
not in `crates/inference`'s default feature set, and the only callers of `route` in the tree are
`crates/tune/examples/prompt_router.rs` and `crates/tune/tests/router_loop_closure.rs`, both at
`k = 1`, where `1/k` is `1.0` and no weight policy is observable. There are no serving callers
today. This ADR changes a surface that nothing in production reaches yet, which is the moment to
change it.

## Decision

**1. Weights come from the scores the gate already produces.** `route` returns a softmax over the
selected adapters' gate scores at temperature `tau`, normalized to sum to 1 across the selected set.
Uniform `1/k` remains expressible and stays the default until the evidence gate below is passed:
`tau` large enough is uniform, and the uniform mode is named explicitly rather than reached by
choosing a large number.

Normalization is to 1 over the selected set, not over the pool, and not unnormalized. The blend
folds each effective weight into the B column blocks and loads the result at `scale = 1.0`, so the
applied delta is `sum_e w_e * (alpha_e / r_e) * B_e A_e`. If the weights do not sum to a constant,
the total adapter strength moves from request to request for reasons that have nothing to do with
which adapters were chosen, and every downstream quality reading inherits that as noise.

**2. A floor, and it drops rather than damps.** After normalization, any selected adapter whose
weight is below `epsilon` is removed from the mixture and the survivors are renormalized. It is not
blended at a weight near zero.

The reason is cost, and it is asymmetric. Per-token LoRA GEMV cost grows with `rank_total`, the sum
of the ranks in the blended adapter, which is why `generate_with_lora_mixture` documents a cost
model at all. An adapter carried at weight 0.001 pays its full rank in the decode of every token
and contributes an output change below the noise of the quantized base weights. Dropping it is not
an approximation of blending it; blending it is a way of paying for nothing. The returned vector is
therefore no longer guaranteed to have length `k`.

**3. Collapse guards, because this reverses a recorded decision rather than filling a blank.** The
original rationale is not withdrawn: a learned-softmax router really can collapse to one adapter
under sparse noisy reward. Three mechanisms bound it, and all three must be present for the learned
mode to be enabled.

- A floor on `tau`. A temperature free to fall toward zero is an argmax with extra steps.
- The load-balance and z-loss terms that already exist in the refit substrate
  (`crates/fann/src/training/rloo.rs:401` `load_balance_aux_loss`, `:418` `router_z_loss`,
  `z_loss_coeff` defaulting to 0.001 at `:33`) applied to the weight distribution, not only to the
  selection logits.
- An entropy floor on the produced weight vector, measured per refit round over the round's
  requests. A refit whose mean weight-entropy sits below the floor for `M` consecutive rounds is
  **rejected**: the new gate is not reloaded and serving continues on the previous one. This is the
  mechanism `reload`'s no-partial-mutation contract makes cheap, and it is the difference between
  detecting collapse and merely being able to describe it afterwards.

The entropy floor cannot tell collapse from a correctly sharp distribution: a router that has
genuinely learned to prefer one adapter produces the same low-entropy weight vector as one that has
collapsed under sparse reward. The floor is a guard on refits, not a guard on truth: it decides
which gate gets reloaded, not whether the resulting policy is good. The arbiter of that question is
Decision 5's held-out metric. The falsifier is named explicitly: a rejected refit whose held-out
metric was improving is the falsifier for the floor value, and that observation, not intuition
about the number, is what moves it.

#### Amendment, 2026-09-19 (Status: **Accepted**, 2026-09-19): the named guard was not the one running

The second bullet above cites `crates/fann/src/training/rloo.rs:401 load_balance_aux_loss` and
`:418 router_z_loss` as two of the three mandatory collapse guards. Two things were wrong with that,
and they are independent.

**The citation pointed at code nothing calls.** Measured at `2d5f7f9e60`: both functions have zero
callers anywhere outside their own `#[cfg(test)]` block. The objective reached production only as a
hand-inlined gradient, in two places (`step` and `rloo_step`, the second carrying the comment
"identical to step"). So there were three copies of one objective, and the one this ADR named was
the only one that did nothing. The consequence is the part worth recording: a maintainer who
followed this citation and corrected the loss at `:401` would have shipped a no-op and had every
reason to believe otherwise, while a maintainer who deleted those functions as unreferenced would
have left the guard fully in force.

**The objective taxed the behaviour it was meant to permit.** The inlined form is the gradient of
`(1/K) Σ_i (p_i − 1/K)²`, a pull toward uniform on a _single_ context. Load balance is a property of
traffic, not of any one decision, and the Switch / ST-MoE form `K · Σ_i f_i · P_i` is a batch
statistic for that reason. The per-context form instead penalizes a gate that is confidently and
correctly routing one request, which is the behaviour the policy gradient exists to learn. This ADR
already makes exactly this distinction about its neighbouring guard: "the entropy floor cannot tell
collapse from a correctly sharp distribution." The load-balance term had the same blindness, and
unlike the entropy floor, which only rejects a refit, it was in the gradient on every step.

**Amended decision.** The second bullet now reads: the load-balance and z-loss terms applied to the
weight distribution are `load_balance_aux_loss_batch` / `load_balance_aux_gradient` and
`router_z_loss` / `router_z_gradient`, and `step` and `rloo_step` **call** them rather than
re-deriving their gradients inline. `f` is an exponential moving average of which expert recent
decisions selected, held trainer-private, initialised uniform, with a fixed rate of `0.01` per
decision. It is trainer-private and not a config field because `RlooConfig` is `pub` with `pub`
fields and therefore externally constructible, which makes any added field a major-version break;
the rate only rescales a gradient whose coefficient is already exposed as `aux_loss_coeff`.

Two consequences stated rather than left to be discovered:

- **The guard is silent until observed routing drifts.** At uniform `f` the gradient is identically
  zero however sharp the gate is, so a fresh trainer applies no load-balance pressure at all. That
  is the intended semantics: there is no traffic yet to be imbalanced, and pressure derived from no
  observations is precisely what the superseded form applied.
- **The per-context function is deprecated, not deleted**, because removing a `pub` item is a
  major-version break. The deprecation note carries the warning that fixing the objective there
  changes no behaviour.

The one-copy property is now asserted rather than asked for.
`rloo::tests::step_applies_the_named_load_balance_guard` sets reward to zero and `z_loss_coeff` to
zero, leaving the load-balance guard as the only thing that can move a weight, and reddens if `step`
stops routing through the named function or stops folding decisions into the EMA.
`aux_gradient_matches_finite_difference_of_the_batch_loss` ties each stated loss to the gradient
actually applied, which is what keeps the scalar definition load-bearing instead of decorative: a
scalar nobody consumes is the defect this amendment corrects.

Falsifier for the amended form, named the way Decision 3 names its others: a closed-loop run in
which routing collapses onto one adapter while `f` reports balance. That would mean the EMA rate is
too slow to see the collapse it is meant to bound, and it is that observation, not intuition about
`0.01`, that moves the number.

#### Amendment, 2026-09-20 (Status: **Accepted**, 2026-09-20): "the weight distribution" scopes to the load-balance term only

The second bullet of Decision 3, as amended on 2026-09-19, asks for "the load-balance and z-loss
terms ... applied to the weight distribution, not only to the selection logits". Read as written it
puts both terms on the same vector, and that reading is wrong for one of them. The sentence is
amended to scope the phrase to the load-balance term, and the misreading is named here rather than
quietly avoided, because the plain reading of the superseded sentence is the one a maintainer
arrives at.

**The z-loss stays on the raw logits.** It is `log-sum-exp` of the pre-softmax scores, and it is
derived for values that are unbounded above: its whole purpose is to tax a gate that grows its
logits without bound, which is a thing only an unnormalised vector can do. A mixture weight vector
is normalised by construction, so `log-sum-exp` over it is confined to a sliver. For `k` adapters
whose weights sum to one, the value ranges from `log(k · e^(1/k))` at uniform to `log(e + k − 1)`
at one-hot: for `k = 4` that is `1.637` to `1.744`, a span of about `0.107`, and the span narrows
as `k` grows. A term whose entire dynamic range across the whole space it measures is a tenth of a
nat cannot bound anything. Applying it there would not be a stricter guard, it would be a guard
that reports success by arithmetic. `router_z_gradient(&logits, &probs)` therefore keeps its
current arguments.

**The load-balance term runs on the weight distribution, and specifically on `softmax(selected
logits / tau)` BEFORE Decision 2's drop and renormalisation.** The ordering is the substance of
this amendment, not a detail of it. An adapter dropped by the `epsilon` floor has no gradient path
through the drop, so a balance term computed on the post-floor vector cannot move a weight that has
already gone to zero. That is precisely the case the guard exists for: a router collapsing onto one
adapter drives the others under `epsilon`, and at that moment a post-floor term goes silent on the
exact event it was added to catch. Computed before the floor, the collapsing adapters are still
present in the distribution and still carry gradient.

`tau` enters this chain as the `1 / tau` scale on the logits, which is bounded below by guard (a).
Without that floor the scale is unbounded and the balance term's gradient can be made arbitrarily
small by sharpening, which is the argmax-with-extra-steps failure guard (a) exists to prevent. The
three mechanisms are load-bearing on each other, as Decision 3 says: this is one of the places that
is true rather than rhetorical.

**Batch form, per the 2026-09-19 amendment, which this one builds on rather than restates.** The
term is `load_balance_aux_gradient(route_freqs, weights)` with `route_freqs` the trainer-private
exponential moving average of which adapter recent decisions selected — not the deprecated
per-context pull toward uniform, which taxes a gate for confidently and correctly routing a single
request. Only the second argument changes: from the selection distribution over the full logit
vector to the tau-scaled weight distribution over the selected adapters.

**Guard (b) is a gradient term, not a monitor.** Decision 3 states that the three mechanisms _bound_
collapse, and a quantity that is only computed and reported bounds nothing. It belongs in the
objective the trainer already applies, beside the terms `step` and `rloo_step` fold into
`output_deltas` today.

**Amended decision.** The second bullet now reads: the load-balance term is applied to the weight
distribution, `softmax(selected logits / tau)` taken before Decision 2's `epsilon` drop and
renormalisation, through `load_balance_aux_gradient(route_freqs, weights)` with `route_freqs` as
defined by the 2026-09-19 amendment; the z-loss term is applied to the raw pre-softmax logits, and
`router_z_gradient(&logits, &probs)` is unchanged. Both remain gradient terms inside the trainer's
step, never separately reported quantities.

Consequence stated rather than left to be discovered: the two terms now take their arguments from
different stages of the same forward pass, the selection logits and the post-temperature weight
vector, so a future change that reorders selection and weighting silently changes what this guard
measures. A test should tie each argument to its stage, not merely to a vector of the right length.

Falsifier, named the way Decision 3 names its others: a closed-loop run in which routing collapses
onto one adapter while the load-balance gradient stays near zero throughout. That would mean the
pre-floor placement is not reaching the collapsing adapters after all, and it is that observation,
not the argument above, that moves the placement.

**4. Caller-supplied weights keep their magnitudes, and are floored without renormalisation.** The
normalisation in Decision 1 is a property of the learned path, not of the mixture in general. Raw
gate scores have no calibrated magnitude, so they have to become proportions before two requests can
be compared at all; a caller's scales are the instruction. A request naming one adapter at 0.5 means
half strength and gets half strength.

The floor still applies, because the cost argument in Decision 2 does not care who chose the weight:
an adapter below `epsilon` pays its full rank in the decode of every token and changes nothing. The
survivors are **not** renormalised after a drop, since scaling them up to recover the dropped mass
would move the caller's chosen strength by an amount that depends on what was dropped.

So the two paths differ in exactly one respect, and it is worth saying rather than leaving to be
inferred: learned weights are normalised because their magnitudes are arbitrary, caller weights are
not because their magnitudes are the request. Both are floored. The learned weights are a default,
not a policy the caller cannot escape.

Both paths report the dropped adapters and the applied weight vector, in the response or in
diagnostics. That is what makes a request naming an adapter that was **absent from the blend**
distinguishable from one naming an adapter that was **present at a small weight**. Those are
different facts about the caller's own request, and only a record of what was dropped and what
survived lets the caller tell them apart.

**5. Acceptance fixes the contract; the evidence gate flips the default.** Accepting this ADR fixes
the contract: weights drawn from the gate's own scores at temperature `tau` (Decision 1), the floor
that drops rather than damps (Decision 2), the three collapse guards (Decision 3), and refit
rejection. The mechanism code merges under that accepted contract, with the learned mode **off by
default** and `1/k` named as the default weight policy.

The evidence gate governs the **flip of that default**, not the status of this ADR. Enabling the
learned mode requires a closed-loop run over `N` feedback rounds on a small held-out task, with the
kill threshold and the decision rule registered before the run starts, not chosen from its output,
and the measurement host fixed and named in the run record so the pre/post comparison is not itself
a source of noise. The run reports:

- the held-out task metric before and after the rounds, as one table;
- the same metric on the base tasks, as a regression check, since a router that improves the target
  task by forgetting everything else has not improved anything;
- the rollback arm: unloading the adapters returns the base-model numbers.

A run that fails the threshold leaves the **default at `1/k`**; it does not leave this ADR at
`Proposed`. The mechanism code that the run exercises is already merged under this Accepted ADR, so
the closed-loop run needs no code that this ADR forbids merging. Only the flip to
learned-by-default waits on it.

## Alternatives considered

- **Keep `1/k`.** Cheapest, and it is the status quo this ADR is asked to change. It also leaves the
  gate computing a score it never uses, which is a standing invitation for someone to reintroduce
  the weight question without the guards above.
- **A separate learned weight head.** More parameters, a second thing to refit, and the same
  collapse exposure. Nothing available suggests it beats a temperature over the scores already
  computed at this pool size, and it can still be added later behind the same evidence gate.
- **Per-request weights only, no learning.** This is genuinely useful and it ships regardless, but
  it is a manual control. It does not answer the ask.

## Consequences

- `route`'s contract changes from "every weight is `1/k`, length `k`" to "weights sum to 1 over the
  selected set, length at most `k`". Callers that assumed the length are wrong at the moment the
  floor first drops an adapter, and both in-tree callers use `k = 1`, where the change is invisible.
  This is a behavioural break on a public method behind a non-default feature; the signature and
  `RouterError`'s variants are unchanged.
- The blend boundary is unchanged. Callers still fold `alpha / rank` into the effective weight
  before calling `blend_lora_layer_data`, which continues to load at `scale = 1.0`.
- Explicit per-request scales are not rescaled, so total adapter strength on that path stays the
  caller's arithmetic by design, including when their scales do not sum to 1. The only change for an
  explicit-scale caller is that an adapter whose weight falls below `epsilon` is absent from the
  blend rather than present at a weight that does nothing, and the survivors are left alone.
- ADR-079's COMPOSE row says selection policy is what ROUTE supplies next. This is that policy, and
  it lands as a new record rather than an edit to that row.

## Open, and deliberately not decided here

The numeric values of `tau`, `epsilon`, `M`, and the entropy floor. They are set from the run in
section 5. What this ADR fixes is that each exists, that the floor drops rather than damps, and that
a collapsing refit is rejected rather than reloaded.
