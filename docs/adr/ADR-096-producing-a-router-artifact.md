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

Retention regularisation is not in the initial objective. The existing wrapper validates its
regularisation strength and does not apply it; retention is measured at admission instead.

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
