# ADR-096: Producing a router artifact — schema lineage, bootstrap, the learned action, and admission

**Status**: Accepted (2026-09-21)
**Date**: 2026-09-21
**Crate**: lattice-inference, lattice-fann

## Context

ADR-095 decided what a router artifact is and how a server loads, reports and pins one. ADR-094
decided where the routing code lives and that the artifact's adapter-name list is authoritative over
residency. ADR-093 decided that the selection is fixed once per request and that every trained
column is routed. ADR-091 decided where weights come from once there is evidence to turn them on.

All four describe a gate that already exists. Nothing produces one.

That is not a gap in the reading. It is measurable, and the measurement is the reason this ADR is
separate from the four above:

- `router_state::write_artifact` has one definition and no caller outside its own test module
  (`crates/inference/src/router_state.rs`; the only other `write_artifact` in the crate belongs to
  `bin/lattice/prune_score.rs` and writes a prune plan, not a gate).
- `AdapterRouter::submit_refit` can replace a live gate from bytes a caller hands it. Its body
  contains no filesystem call and no artifact write: it reloads in memory and returns. A refit is
  therefore invisible after a restart.
- So the only way to obtain a servable artifact today is to write one by hand against the format.

Two further facts constrain every option below, and both were read at the serving head rather than
assumed:

- **The gate's output width is the adapter count, and `ServingRouter::new` refuses a disagreement.**
  Adapters load and unload at runtime. So a gate trained over two adapters, served next to three
  resident ones, reports `routable: false` and routes nothing.
- **The gate does not weight anything yet.** `WeightPolicy::default()` is `Uniform`, no serving path
  names another policy, and `k` is the column count, so every trained column is applied at `1/k`.
  This is ADR-091 decision 1 operating as written — uniform is the default until that ADR's evidence
  gate passes — and ADR-093 carries an amendment saying so. It means the serving path has no place
  to put a learned distribution even once one exists.

## Decision

### 1. The ordered adapter schema is immutable within a lineage, and changing it is an explicit versioned transition

A router artifact's schema is its ordered list of adapter identities together with the base-model
binding and the recorded input representation. **An ordinary refit preserves that schema exactly.**
Adding, removing, substituting or relabelling a column ends the lineage and begins a new one.

The reason is not that the network cannot be resized. It is that a column's meaning is not recoverable
from anything the network carries. A gate is weights and two widths; the correspondence between
column and adapter lives only in the artifact's name list. So:

- **Width is neither necessary nor sufficient to identify a schema change.** Replacing `[A, B]` with
  `[A, C]` keeps the width at two and changes what column one means. Reordering the names without
  moving the corresponding output rows changes every column's meaning at constant width.
- **A weight refit cannot perform a schema transition.** Training changes what the columns score, not
  how many there are or what they are called. `AdapterRouter::reload` already validates both
  dimensions before replacing a gate, which is the same refusal one level down.

A transition creates a new artifact with a new schema identity, **freshly initialised**, with any
optimiser, Fisher or replay state cleared. No output rows are transplanted from the old lineage. A
transplant is a migration algorithm with its own correctness argument — preserved logits do not
preserve normalised weights once a new column joins the denominator — and this ADR does not have
that argument. Data from the old lineage may be retained as audit evidence; it does not enter the new
lineage's training batch by position.

**A pinned version is never silently transitioned.** Pinning preserves the artifact's bytes, its
ordered schema, its policy and its status. If the adapters that schema names are not resident, the
server reports the mismatch and refuses routed requests; it does not reinterpret columns, mask the
extra adapter, or quietly fall back to the base model. That refusal is ADR-093's decision, unchanged.

One correction to how this was first framed, kept because it changes what to build: the mismatch is
**not** permanent. Unloading the extra adapter restores agreement with the existing artifact. The
transition exists so an operator can adopt a new set deliberately, not to rescue a state that has no
other exit.

### 2. Provisioning is an explicit, shipped command, and its output is labelled untrained

Add `lattice router init`. It takes an explicit ordered adapter specification, the base-model
binding, the embedding model, pooling and prompt rule, and a destination state directory. It measures
the input width through the same loader contract the serving path uses, so the width recorded in the
artifact is the width the server will produce.

Its output is a **deterministic untrained bootstrap**: a gate whose logits are zero, carrying the
named `Uniform` policy with learning disabled. That is legitimate to serve, because the operator
asked for that baseline, and it is not evidence of learning. `GET /v1/lora` distinguishes bootstrap
from learned, alongside the pin state and routability it already reports.

Three refusals, each because its permissive form fails toward looking successful:

- An empty or unreadable `--router-state` still refuses startup. ADR-095 decided that; minting an
  artifact on an empty directory would make a mistyped path look like a successful start.
- `router init` against an occupied directory refuses rather than overwriting.
- Initialisation succeeding does not enable learning. That is a separate explicit action.

A validated artifact trained elsewhere may be imported through the same schema and admission checks.
A scratch harness is not a production entry point.

### 3. The learned action is the whole mixture, not an adapter index

The existing selector updater requires a preferred adapter index and applies a discrete-action loss
over it. **A completion produced by a weighted mixture does not identify such an index.** Choosing
the largest-weight adapter, or rewarding every adapter equally, manufactures credit the feedback did
not supply. So the existing updater is not the production entry point for this policy, and moving it
between crates does not make it one.

The action to learn is the served mixture itself. Under explicitly enabled learning, the gate's raw
logits gain recorded exploration noise at a fixed, recorded scale; the mixture is formed from the
perturbed logits through the policy transformation and ADR-091's floor, and is frozen for the whole
completion. Only explicit completed-request feedback supplies a signal. Missing feedback is missing,
not zero: the estimand is utility among the population that gave explicit feedback, and saying so is
part of the decision.

**This is an explicit proposed amendment to ADR-091 decision 1**, stated here rather than taken as
implementation latitude: during enabled learning the logits are not solely the deterministic scores
the gate computes. The argument is identifiability. One scalar judgement of one deterministic mixture
contains neither a target column nor any observation of an alternative mixture. Recorded randomisation
supplies known variation and an attributable action probability without inventing either.

Each eligible feedback event carries its completion identity, the server-captured input vector, the
schema, the artifact and policy identity that produced it, the sampled logits, the pre-floor and
applied weights, and the signal. The server holds that provenance; a caller cannot supply a vector or
an index in its place. One idempotent judgement per completed request; conflicting repeats refuse.
An event that arrives after its policy epoch ends is counted stale and is not reused as an on-policy
sample for a different gate.

**Amended 2026-09-21: the policy, written out.** "Exploration noise at a fixed scale" and "an
attributable action probability" name a design without giving one, and the producer PR cannot be
written from them. Four things were missing, and the fourth is the one that makes the other three
easy.

The noise is Gaussian on the gate's raw logits, independent per adapter, at a scale recorded in the
policy identity: for a gate emitting logits `z` over `k` adapters, the served logits are
`z' = z + e` with `e ~ N(0, s^2 I)`. `s` is a property of the policy epoch, not of a request, so a
refit round has one value and events from a different value are the stale events decision 3 already
excludes.

The density that is scored is the density of `z'`, not of the weights. This is the step worth being
explicit about, because the obvious alternative does not exist: the applied mixture is
`w = floor(softmax(z'))`, and ADR-091's floor clamps a set of coordinates and renormalises the rest,
so the distribution of `w` puts positive mass on a lower-dimensional face and has no density with
respect to Lebesgue measure at all. Any estimator written against `p(w)` is therefore ill-defined
in exactly the region the floor is there to produce. Scoring `z'` avoids it outright:
`z' | z ~ N(z, s^2 I)` is a proper density everywhere, and it is the quantity the server already
records as the sampled logits.

The estimator follows from that choice, and the policy floor drops out of it. With
`log pi(z' | z) = -||z' - z||^2 / (2 s^2) + const`, the score is
`grad log pi = (z' - z) / s^2 * dz/dtheta`, which needs no derivative of `floor` or of `softmax`,
because both are downstream of the sampled action and affect only the reward earned. So there is
nothing to differentiate through the floor, which is what made the original wording read as a gap:
it was describing a gradient path that the design does not use.

The objective is expected reward under that policy, estimated on the round's eligible events with a
leave-one-out baseline over the round rather than a learned value function: for events
`i = 1..n` with rewards `r_i`, the update is the mean of `(r_i - mean of r_j for j != i) * grad
log pi(z_i' | z_i)`. The baseline is what the existing RLOO trainer already provides, it needs no
second network, and it is unbiased under the on-policy restriction decision 3 imposes. Reward is
the explicit signal's magnitude and nothing else; the implicit variants are refused at the wire
under ADR-095's amendment, so no half-magnitude term enters this objective. Retention
regularisation stays out of it, as below, and the population the estimate describes is still the
one that gave explicit feedback.

Retention regularisation is not in the initial objective. The existing wrapper validates its
regularisation strength and does not apply it; retention is measured at admission instead.

A refit is a round of those events, and two of the round's properties are not properties of any
sample in it.

**The baseline is the mean of the other rewards in the round.** A single judgement carries no
baseline: one completion rated helpful says the mixture was good, not that it was better than
whatever else the gate would have served, and a gradient taken against no baseline moves on the
reward's sign alone. A leave-one-out mean is independent of each sample's own action, so the
estimator stays unbiased. It does not control for how hard each context was, and a round whose
rewards vary more across contexts than across actions gets little variance reduction out of it.
That is a limit of the round, recorded here rather than tuned away.

**Every gradient in a round is taken at the gate that produced the actions.** Applying each sample
as the round proceeds would evaluate every gradient after the first at a gate that did not produce
its action, while the epoch rule above still counted those samples on-policy. The code's meaning of
on-policy and the feedback store's have to be the same one.

**A round whose rewards are all equal is a no-op, byte for byte**, and that includes a round of
one. Such a round carries no comparison. The load-balance and z-loss terms do not fill the gap, and
deliberately: they exist to shape a policy update, not to constitute one, and a gate that drifted
toward balance because a refit was attempted would change served behaviour with no feedback behind
it. Traffic history still records every sample, because those completions were served either way.

**The refit runs in the server, behind a feature that is off.** `router-learning` implies the
mixture gate and the trainer, and it is the only thing that pulls a trainer into the serving crate.
A default build is measured trainer-free rather than declared so: a check reads symbols out of the
artifact each build actually produced, over three feature sets, with the feature-on build as the
same-pass positive control, because an absence reported by a probe that has never been shown to
find a presence is not a finding. The refit is its own bounded task and never runs on the request
path; its candidate reaches the live gate only through the entropy guard and the promotion rule in
decision 5. Enabling it on a serving host happens only inside a registered qualification run.

The alternative shapes were a learner in the training crate, which needs a dependency edge from the
serving crate that does not exist, and an exported feedback stream with an admission endpoint,
which is new public API. The second is not refused: the round's objective is plain data over
recorded actions with no knowledge of storage or transport, so a learner that ever has to run off
the serving host costs an export surface and its own decision record, not a rewrite of the
objective.

### 4. The serving-side weight policy is a decision with an evidence gate, not a flag

Uniform `1/k` stays the default. Replacing it requires naming the policy, its temperature, and the
evidence that admits it, and ADR-091 decision 5 already requires closed-loop evidence with regression
and rollback arms before the default changes. This ADR adds only that the choice is recorded **in the
artifact**, so that loading the same gate bytes under two different policies cannot produce two
behaviours under one version label.

Evidence that a learned policy works must show the **actual request coefficients change**, not that
the network bytes changed. A gate whose scores move while the serving path applies `1/k` produces an
identical response, which is exactly the shape that reads as success.

### 5. A version is a complete policy, and admission is a positive result about the candidate

The hashed contract grows to cover the policy kind and its parameters, the guard configuration, the
schema identity, the parent identity, the training state, and the identity of the admission record.
This is a **new format revision**, not a redefinition of format 4: existing format-4 artifacts stay
readable as what they are, explicitly uniform, and do not acquire learner eligibility by defaulting
their missing fields.

Admission separates four things that are currently one: candidate construction, admission, durable
commit, and activation.

- The candidate's outputs are computed through the same policy transformation that will serve them,
  and its round statistics are derived from those outputs rather than from numbers a caller supplied.
- Admission requires a predeclared held-out metric, a base-task regression limit, a minimum evidence
  population, and an explicit pass rule. **Insufficient evidence is a rejection, not a pass.**
  Entropy, replay accuracy, and a historical judgement under the parent are none of them evidence
  about this candidate.
- Publication is create-only and staged outside the selectable namespace. Startup reads a committed
  head record, not the numerically largest file present, so a rejected candidate cannot become the
  served gate by surviving on disk across a restart. The plain write-then-scan the current writer
  uses cannot express that distinction.
- Activation follows the durable commit. An operation is not reported active before the in-memory
  gate has actually changed, and a crash between the two recovers the committed head.

**Two selection rules, one per format revision, and neither amends ADR-095.** ADR-095 decided that
startup serves the highest version counter present, resolved from the artifact filenames, and that
`--router-pin <version>` serves a named one instead. That rule stays exactly as written **for format
4**, because a format-4 directory has no head record and never acquires one: nothing writes it, so a
reader looking for one would find an absence it could not distinguish from a truncated write.

- **Format 4.** Unpinned startup scans the counters and serves the highest, and the reader compares
  the manifest's own counter against the filename-derived one. `--router-pin <version>` resolves by
  that same counter.
- **The new revision.** Unpinned startup reads the committed head record and serves what it names; a
  higher-numbered artifact present in the directory but not named by the head is a staged or rejected
  candidate and is never served. `--router-pin` resolves against the artifact's full version
  identity, and a pin **disables automatic promotion**, so a head advance while a pin is in force
  changes the served gate for nobody until the pin is lifted.
- **A directory holding both.** If a head record is present it governs unpinned selection, including
  when the artifact it names is older than a format-4 file sitting beside it: the head is a statement
  that a specific artifact was admitted, and a counter is not. A format-4 version in such a directory
  remains reachable **by pin**, under the format-4 rule above, which is what keeps a rollback target
  available across the revision boundary.

Stated this way the change is additive: no accepted decision is reinterpreted, and the rule that
applies is a property of the artifact being read rather than of the server reading it. The
alternative — amending ADR-095 so that every directory is expected to carry a head — was rejected
because it would make every existing state directory non-conforming the moment this ADR is accepted,
with no migration and no writer to produce what it would then require.

The bootstrap of decision 2 does not pass through the learned-candidate quality rule: it is checked
for structure and provenance and reported untrained. That exemption is written here so that it cannot
be widened by an implementation that finds an ordinary refit inconvenient.

### 6. The gate's output is per-adapter weights, and the contract it must satisfy already exists

Decision 3 says the learned action is the whole mixture rather than an adapter index. That settles
what the gate _decides_. It does not say what the gate _emits_, and the gap between those two is
where this chain currently ends: the serving path selects adapters by name and has no code that
turns a gate's output into a number.

This is worth stating precisely, because the missing piece is smaller than it looks and the
surrounding half is older than this chain. The application half shipped in #443, and its contract
is already weight-shaped:

```rust
pub fn blend_lora_layer_data(
    inputs: &[(&[LoraLayerData], f32)],
) -> Result<Vec<LoraLayerData>, InferenceError>
```

It takes `(adapter, weight)` pairs and blends them into a single rank-Σr adapter — `A_blend` the
vertical concatenation of the A matrices, `B_blend` the horizontal concatenation of the
weight-scaled B matrices. That construction is mathematically exact rather than an approximation,
and it needs no Metal kernel change, because adapter rank is a runtime `set_bytes` parameter and
the kernels are rank-agnostic.

So the contract this decision fixes is the one between the gate and that function.

**Shape.** One `f32` per adapter, positionally aligned to the artifact's ordered adapter schema of
decision 1. The schema is what makes the vector interpretable; a weight vector without its lineage
is a list of numbers with no referent, which is the failure decision 1 exists to prevent.

**Normalisation, and where it happens.** Weights are normalised to sum to 1 before they reach the
blend, and the serving-side floor of ADR-091 is applied _before_ that normalisation, not after.
The order matters and is not a detail: the floor clamps coordinates and the renormalisation that
follows is what keeps the applied mixture a convex combination. Applying the floor afterwards would
leave a vector that no longer sums to 1 and would silently rescale the blended adapter.

**Where they enter.** The normalised vector is zipped with the resident adapters named by the
artifact's schema and handed to `blend_lora_layer_data` as its `(adapter, weight)` pairs. Nothing
else in serving is permitted to scale a weight afterwards; a second scaling site is how two
correct-looking factors multiply into a wrong one.

**What a non-Metal build returns.** The real blend is `cfg(all(target_os = "macos", feature =
"metal-gpu"))`. Every other build links a stub that returns `Err` by design, so a caller fails
loudly rather than silently serving an unblended adapter. A gate on such a build must refuse at
startup with that reason named, in the same general-before-specific order the other routing
refusals already follow: a build that cannot blend cannot route, whatever its artifact says.

**Known bound, stated here rather than discovered later.** Weighted mixture is a Metal-only
capability today, and Metal serving requires a Q4 checkpoint while the embedder that produces the
gate's context vector rejects Q4 checkpoints unconditionally. Those two requirements are satisfied
by disjoint checkpoint classes, so the full path is not reachable on a single directory until the
embedder is loaded from a second one. This decision does not fix that; it records that the output
contract above is specified against a path whose other end is still blocked, so that a reader does
not mistake a specified contract for a reachable one.

**Falsifier.** The claim this decision makes is that a gate's output can reach the blend. It is
false until one test drives a gate output through the serving path into a weighted blend and
asserts the blended adapter differs from the one produced by uniform weights over the same
adapters. The uniform-weight arm is the control and is load-bearing: a blend that ignored its
weights entirely would satisfy every other assertion in that test. Until that test exists, decision
6 is a specification and nothing in the tree has been shown to meet it.

## What exists at this ADR's merge base, and what this chain builds

This ADR cites `router_state::write_artifact`, `ServingRouter::new` and a format-4 artifact. None of
them exists at the commit this document merges into. The only `write_artifact` under `crates` at
that base writes prune plans, and `AdapterRouter::reload` takes raw network bytes with no artifact
around them. A reader checking the compatibility and rollback decisions against the tree will find
nothing to check them against.

That is the intended order rather than a defect in the decisions, and it is written here because a
document that describes a contract in the present tense reads as a description of working code no
matter how the surrounding PRs are sequenced. The routing chain lands the ADRs first, so this ADR
is the specification the later PRs are built to satisfy: the router-state module, the artifact
format and its version lineage, and the serving façade arrive in the router-artifact and producer
PRs of this same chain. Until they do, every API name in this document is a name this chain is
obliged to create, not one a reader can open.

Decision 6 is the exception and is deliberately the other way round. `blend_lora_layer_data`
shipped in #443 and a reader can open it today; the contract in that decision is written
against an existing function rather than a promised one. That is why the gap it names is a
connection rather than a component, and it is called out here because the blanket statement
above would otherwise read as covering it.

The practical consequence is for whoever reviews the producer PR. The compatibility and rollback
decisions here are testable only against that PR's own tree, so the reviewer of this ADR is asked
to judge whether the contract is right, and the reviewer of the producer PR is asked whether the
code meets it. Splitting the question that way is deliberate; collapsing it is what produces a
contract nobody checked because everyone assumed the other reader had.

## Alternatives considered

**Grow, shrink or transplant output rows on a schema change.** Attractive because it avoids an
interruption. Rejected because it is a distinct migration algorithm with no evidence behind it: even
preserving old logits changes old normalised weights once a new column enters the denominator, and
the optimiser and guard state are indexed by column.

**A maximum-width gate with spare columns, or routing only the intersection of the artifact and
residency.** Rejected because both contradict the accepted exact-width and two-way set checks, and
both need masking, slot identity and renormalisation semantics that nothing here defines. Neither is
available as an implicit amendment.

**Freeze the adapter set for the lifetime of a state directory.** Simplest, and it fails the
capability this lane exists to deliver: adapters are loadable at runtime, so an operator who loads one
must have a defined path to a gate that knows about it.

**Mint an artifact automatically when the state directory is empty.** Rejected: it makes a mistyped
path indistinguishable from a successful start, which is the failure ADR-095 refuses at startup.

**Use the existing indexed updater on completion feedback.** Rejected: it requires a target column the
feedback does not contain, and every way of supplying one invents the comparison the reward was
supposed to provide.

**Admit a candidate on entropy or on replay accuracy.** Rejected: high entropy is compatible with a
uniform policy that has learned nothing, and replay accuracy scores the fitting population on the
wrong target.

## Consequences

A schema transition is an observable interruption: the new lineage starts untrained, so an operator
who adds an adapter trades routing quality for the new adapter's availability, deliberately and with
that stated up front.

The learning path cannot ship enabled. Its objective has a defined estimator and no measured variance
or utility; the sample sizes, exploration scale and thresholds are release dependencies, not defaults
an implementer may choose.

The format revision means artifacts written before it stay servable and stay uniform. That is the
intended asymmetry: an old artifact should not become eligible for a policy it never recorded.

## Open, and deliberately not decided here

- The numbers: exploration scale, evidence population, thresholds, and the wall budget for a refit.
  They come from the qualification experiment, and writing plausible defaults here would make the
  experiment's outcome look predetermined.
- Whether an off-policy estimator is worth its logged-density and support checks, which would let a
  late event train a newer gate.
- The second decoder's resident-memory cost, which ADR-094 also records as unmeasured.
- Adapter content identity. The accepted name-based checks detect a misconfigured server; they do not
  detect different bytes served under an unchanged name and path, and this ADR does not close that.
