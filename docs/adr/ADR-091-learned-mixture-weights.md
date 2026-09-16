# ADR-091: Learned mixture weights over a resident LoRA pool

**Status**: Proposed
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

**4. Caller-supplied weights are unaffected.** A request that names adapters and scales explicitly
gets exactly those, with the same normalization and floor applied for the same reasons. The learned
weights are a default, not a policy the caller cannot escape.

**5. Evidence gate, before any dependent code merges.** The learned mode ships disabled. It is
enabled only after a closed-loop run over `N` feedback rounds on a small held-out task reports:

- the held-out task metric before and after the rounds, as one table;
- the same metric on the base tasks, as a regression check, since a router that improves the target
  task by forgetting everything else has not improved anything;
- the rollback arm: unloading the adapters returns the base-model numbers.

The kill threshold and the decision rule are registered before that run starts, not chosen from its
output. A run that fails the threshold leaves `1/k` in place and this ADR in `Proposed`.

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
- ADR-079's COMPOSE row says selection policy is what ROUTE supplies next. This is that policy, and
  it lands as a new record rather than an edit to that row.

## Open, and deliberately not decided here

The numeric values of `tau`, `epsilon`, `M`, and the entropy floor. They are set from the run in
section 5. What this ADR fixes is that each exists, that the floor drops rather than damps, and that
a collapsing refit is rejected rather than reloaded.
