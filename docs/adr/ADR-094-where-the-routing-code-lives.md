# ADR-094: Where the routing code lives, and what feeds it

**Status**: Accepted (2026-09-21)
**Date**: 2026-09-20
**Crate**: lattice-inference / lattice-fann / lattice-tune

## Context

ADR-093 decided that mixture routing runs once per request. Nothing in the serving path calls a
router today: `crates/inference/src/serve/` contains no reference to `AdapterRouter`,
`update_router` or the `mixture` module at all. Wiring one in runs into three structural facts that
have to be settled before any of it is written, because each of them decides where code goes rather
than what it does.

**The router is behind a non-default feature and the serving path is not.** `pub mod mixture` is
gated by `#[cfg(feature = "mixture")]` at `crates/inference/src/lib.rs:100`, and `mixture` is not
in the crate's default set. The serving path is unconditional. So a call from `serve/` into the
router either drags a `cfg` onto the call site or the gate moves.

**The update primitive is in the wrong crate for the direction the call has to run.** `route` is in
`lattice-inference` (`mixture.rs:412`), but the thing that learns — `update_router`, `FeedbackEvent`,
`ReplayBuffer`, `DiagonalFisher`, `RouterUpdateConfig` — is in `lattice-tune`
(`crates/tune/src/lora/router_update.rs`, 1203 lines). The dependency runs `tune → inference`
(`crates/tune/Cargo.toml:57`, optional, under `inference-hook` and `train-backward`) and inference
declares no dependency on tune at all, against 14 declared path deps in its manifest. A feedback
endpoint in `serve/` calling `update_router` would invert that, which this repository bans outright.

**The context source is already in the serving crate.** `BertModel` is `lattice-inference`'s own
(`crates/inference/src/model/bert.rs:211`, re-exported at `lib.rs:191`), and
`serve/embeddings.rs` already loads and serves through it behind `--embedding-model`. Feeding
`route(context_vector, ...)` from an in-process embedding model needs no new crate reach.

## Decision

**1. The context vector comes from the in-process embedding model, with an explicit refusal when
none is loaded.** The gate was trained on that representation, so this reuses it as trained. When
no embedding model is configured, a request that would route refuses and says so; it does not fall
back to a default mixture silently, because a silent fallback makes "the router is enabled" and
"the router ran" indistinguishable at every later reading.

_Amended 2026-09-21._ Wiring decision 1 exposed a question this ADR did not settle: how a request
asks to be routed at all. `ChatRequest.lora` is `#[serde(default)] pub lora: Vec<LoraSelection>`,
documented as "omitted or empty selects the base model", and on a `Vec` under `serde(default)` an
absent field and `"lora": []` deserialize to the identical value. The wire could not express
"choose for me" as a third state.

The field becomes `Option<Vec<LoraSelection>>`, which is the pattern `ChatRequest.model` four lines
below it already uses, for the stated reason that an absent field and an explicit empty one "are
validated differently, so the distinction must survive deserialization". Three states, and each
means one thing:

- **absent** — route, when routing is enabled; the base model when it is not, which is exactly
  today's behaviour for every caller that omits the field.
- **`[]`** — the base model, pinned. Never routed.
- **an explicit list** — that list. Never routed.

`[]` stays pinned rather than becoming a second spelling of "route", and that is the whole reason
this is a decision rather than a detail. The cheap alternative was to treat empty as "route": it
needs no type change, and it would silently start returning adapter output to a client that sends
`"lora": []` today to pin the base model, at the moment an operator enables routing, with no
request change and no error anywhere. That is a behaviour change arriving a long way from its
cause, which is the failure shape decision 1 and ADR-095 decision 6 both refuse.

No second request field is needed to keep "routing was asked for" separable from "routing
happened": ADR-095 decision 2 already reports the selection a request actually used in that
request's response metadata, and it reports which of the three cases above occurred.

Every JSON body that parses today still parses. The change is Rust-side: absent stays absent, `[]`
stays `[]`, and only the type can now tell them apart.

_Amended again 2026-09-21, re-reading this decision against the tree before wiring it._ The
mechanism holds — `EmbeddingModel::embed_text` returns the vector, `AppState` already carries the
model as an `Option`, and `ApiError::EmbeddingModelNotLoaded` is the refusal this decision asks
for. What does not hold is the sentence "the gate was trained on that representation, so this
reuses it as trained". Nothing records which representation, and nothing checks it. Two gaps, and
they are not the same gap.

**The width is checkable and is currently checked in the wrong place.** `AdapterRouter::route`
compares the context vector's length against the gate's input width and returns
`InputSizeMismatch` — per request, at serve time. Both halves are in hand at startup:
`Network::num_inputs` on the loaded gate and `EmbeddingModel::dimensions`. So a gate trained on one
checkpoint's hidden size, loaded against another's, starts cleanly, reports `"enabled": true` from
`GET /v1/lora`, and fails every routed request. That is a behaviour change arriving a long way from
its cause, which is the shape this decision and ADR-095 decision 6 both refuse. **The comparison
moves to startup and refuses there**, naming both widths, on ADR-095 decision 6's argument: it
makes "routing is configured" and "routing can run" the same question, answered once, while the
operator can still act on it. The per-request check stays as the backstop it already is.

**The pooling strategy is not checkable at all today, and the width check cannot stand in for it.**
`embed_text` takes a `PoolingStrategy`, and the two variants — `MeanVisualTokens` and `LastToken` —
produce different vectors from the same text. Both produce vectors of the same length, because the
dimension is the checkpoint's hidden size either way. So a gate trained under one pooling and
served under the other passes the width check, passes every check in decision 2, and routes
confidently on a representation it was never trained on. There is no error anywhere, and the only
symptom is worse adapter selection, which is indistinguishable from a gate that simply did not
learn much.

This is decision 2's amendment one level out, and it takes decision 2's answer: **the artifact
records the representation it was trained on — the embedding model's identity, the pooling
strategy, and which text of the request was embedded — and the façade refuses when the server's
differs.** The same reasoning applies for the
same reason: an unwritten convention about how two sides encode a vector is unverifiable by
construction, so the only shape in which a mismatch is detectable at all is one where the artifact
carries its half of the key. A model identity is weaker evidence than the adapter names (two
checkpoints can share a name, and fine-tuning changes the representation without changing it), so
it is a refusal on mismatch and not a warrant of sameness on agreement — which is worth stating
rather than leaving for a later reader to discover.

**The shape both amendments share, stated once because the next one will have it too.** Making X
authoritative over Y does not remove an unverifiable pairing, it relocates it: X now has to
describe something, and the pairing between X and the thing it describes is new, unchecked, and
easy to miss precisely because the attention is on the X-to-Y relation that was just hardened.
Decision 2's amendment made the artifact's name list authoritative over the caller's adapter order,
and the first round of checks written for it compared that list against residency and nothing else
— an artifact naming three adapters around a five-column gate passed every one of them, while
`route` takes `available.len().min(scores.len())` columns and so drops the trailing ones in
silence, with valid weights and no error. The amendment above is the same step one level further
out: the artifact becomes authoritative over the representation, and only the half with a length is
checkable. Both are closed here, by comparing the name list's length against the gate's output
width and the recorded input width against the measured one. What is worth carrying forward is not
either fix but the question that finds them: after making an artifact, list or registry
authoritative for a correspondence, ask what that authority itself claims to describe, and what
compares the two.

Asked a third time, while wiring the serving path, it found a third member. The representation as
first written said how a text becomes a vector — which model, which pooling — and nothing about
WHICH text. A gate trained on the last user message and served the whole rendered conversation
agrees on the model, agrees on the pooling, and produces a vector of exactly the right width from
different content. It is the pooling argument again with the last checkable property removed, and
it is not a fourth incident: it is the same question returning an answer because it was asked.

So the representation carries the prompt-selection rule too, and the serving path names that rule
as a value rather than following it by convention, so the recorded name and the served rule can be
compared at all. There is one rule today, which is precisely when this is cheap: the check cannot
fail against an artifact this build wrote, only against one written by a build that had a second
rule — and that is the case with no symptom. Adding it later would invalidate every artifact
already written, which is the other reason the cost is lowest now.

Asked a fourth time, while wiring the serving path against real checkpoints, it found both a
second decoder and a fourth member.

**Served model and embedder cannot be the same directory on any server that can route.** Routing
applies adapters; applying needs the Metal backend; the Metal backend needs a Q4 directory. The
embeddings loader reads an f16 decoder and refuses a Q4 directory. So the configuration this
decision assumed — one in-process embedding model, loaded from the served checkpoint — does not
exist on a routing server, and before this amendment there was no second directory to name. The
serving binary gains `--embedding-model`, required by `--router-state` and refused at startup with
that reason rather than failing later inside the gate check. The load is fail-closed when the flag
is given and stays best-effort when it is not, because those are two different operator claims: one
names a directory and is owed an error, the other accepts whatever the served checkpoint offers.

**The cost is a second decoder resident in one process, and this ADR does not get to guess it.**
The f16 embedding checkpoint is a separate set of weights from the Q4 decode weights, held for the
life of the server. Slot, to be filled from a measurement rather than an estimate: resident-memory
delta between a server started with `--embedding-model` and the same server without it, same
checkpoint pair, measured on the quiet host, reported as RSS at steady state after the first
routed request. **TBD.** A number written here from the on-disk size of the checkpoint would be a
guess wearing a measurement's clothes, and the whole point of this family of amendments is that an
unverified claim is worse when it is precise.

**The identity is a NAME, and that bounds what it can detect.** `--embedding-model-id` overrides
the directory basename, mirroring `--model-id`. It detects a misconfigured server: an operator who
points the gate at a checkpoint they did not train it on, where the names differ, gets a refusal at
startup. It does not detect a substituted checkpoint. A different checkpoint of the same family and
the same hidden size, living under a directory of the same name, reads identical, and the config
carries only `model_type`, which is the family rather than the weights. Stating the non-detection
is the point: an identity check that is quietly believed to be a content check is the same class of
unverifiable pairing this decision keeps relocating, one level further along.

**The fourth member is the loader.** Model, pooling and prompt source say which weights, how the
hidden states are reduced, and which text goes in. None of them says how the bytes were READ. Two
loaders over one directory can produce different vectors of the same width from the same text, and
the width check cannot see it for the reason the pooling check could not: the dimension is the
checkpoint's hidden size either way. So the representation records `loader_format`, and the
serving check refuses a gate trained through a loader this build does not use. There is one value
today, which is exactly when it is cheap to record and impossible to add later without invalidating
every artifact already written. That is the third time that sentence has been the reason, which is
the tell that it is one rule and not three coincidences.

"One value today" is a statement about which builds can route, not about which loaders exist, and
the difference is worth writing down before it is load-bearing. A second loader is already in the
tree behind the SAME flag name: `lattice_serve --embedding-model` loads `BertModel` through
`BertModel::from_directory` and reduces with `BertPooling`, while `lattice serve` holds a
`serve::embeddings::EmbeddingModel` and reduces with `PoolingStrategy`. Same spelling, two loaders,
two pooling vocabularies, and vectors of the same width from the same text. The representation key
above is written against the second of those, and it is sound today for one reason only: the first
binary has no `--router-state`, so it cannot route and cannot serve a gate at all. That is the
whole of the protection, and it is a property of a missing flag rather than of the key.

_Superseded within a day of being written, and the correction is the more interesting half._ The
first version of this paragraph made the embedder type a member of the representation in its own
right, beside the model id, the pooling strategy, the prompt-selection rule and the loader format.
That member is withdrawn. `loader_format` already answers the question it was added to answer: it
is documented as how the bytes were READ, and the two embedders differ in exactly that respect —
the one reads an f16 decoder, the other reads a BERT encoder — so a server on the second records a
different loader format by construction. Two members that cannot disagree in any case either of
them can reach are one member with two names, and the failure mode of that shape is a later reader
updating one and not the other, leaving a key that reads complete and compares on the stale half.
A representation member also cannot be removed later without invalidating every artifact written
under it, so the redundancy would have been permanent.

What was actually missing was never a member. It is that `loader_format` has one value today for a
reason that is not about the key at all: the binary holding the other loader has no
`--router-state`, so it cannot route and cannot serve a gate. That is a property of a missing flag,
and it will stop being true.

So the arm is built on the field that already exists, and it has three parts, because two of them
are about the arm not being vacuous rather than about the check:

1. **Each binary names its OWN loader constant, and a test asserts the two constants differ.**
   Without this the cross-binary arm is vacuous by construction: an arm comparing a value against
   itself passes for every input and proves nothing, while reading exactly like a guard.
2. **The startup refusal names both values**, the artifact's and the binary's, so an operator can
   act on it without reading source.
3. **A test constructs an artifact recorded under the other binary's loader and asserts the refusal
   fires at startup, naming both.** The fixture has to come from the other binary's constant rather
   than from a literal, or it stops tracking the thing it is pinning the moment that constant moves.

All three are merge conditions on the router-artifact PR rather than later hardening.

Recording it moves the artifact format from 3 to 4, and the field is inside the content hash: a
representation member outside the hash is a member two artifacts can disagree on while claiming to
be the same artifact.

**2. The `mixture` gate moves off the call site.** The rule is that a `cfg` never lands on the
serving call. Either the router module stops being feature-gated, or the serving path acquires a
gate-free façade whose non-`mixture` build is a compiled-in refusal rather than an absent symbol.
The second is preferred: it keeps the feature's build-size argument intact while making the missing
configuration a runtime answer a caller can read, which is the same shape as decision 1's refusal.

The façade is a concrete type with a `cfg`-selected body, not a trait. One implementor is not a
trait's reason to exist, and a trait here would add a dispatch seam whose only caller is the one
this ADR is wiring.

_Amended 2026-09-21._ Building the façade surfaced the question this decision actually turns on,
and it is not the signature. `AdapterRouter::route` maps a gate output column to an adapter by
position: the selected column index is used directly as `available[idx]`, in both weight policies
(`mixture.rs:474` under `Uniform`, `mixture.rs:509` under `Softmax`), and the `AdapterId` string is
a label copied into the returned tuple, never matched against anything the gate holds. The gate is a bare fann `Network` — weights, input width, output width — and the
artifact carries no adapter identity, so nothing in the system can tell whether the caller's slice
is ordered the way the gate was trained.

That makes the correspondence between gate columns and adapters an unwritten contract owned
entirely by the caller, with no instrument that can check it. A gate trained with one adapter at
column 3 routes to whatever happens to sit at index 3 in the list the caller passes: valid weights,
no error, wrong adapter. It is latent rather than live only because there are no production callers
— the three that exist (tune's `prompt_router` example, `router_loop_closure`, the in-module tests)
each pass a fixed literal pool in the same scope, where position and name cannot disagree. Wiring
the serving path is the act that creates the first caller whose adapter set changes underneath it,
since an operator can load or unload an adapter between two requests.

**So the gate artifact gains the list of adapter names it was trained on, and the façade matches by
name and refuses when the resident set does not match.** This is chosen over pinning an ordering
convention in the caller for one reason that outweighs the rest: it is the only shape in which a
wrong pairing is detectable at all. An ordering convention is unverifiable by construction — there
is no assertion to write, because neither side holds both halves of the key. Refusing on mismatch
is also decision 1's shape reused: the missing configuration becomes a runtime answer a caller can
read, rather than a silent default. And it is the failure mode this ADR family already refuses
twice, in decision 1 and in ADR-095 decision 6 — a behaviour change arriving a long way from its
cause, here an adapter load quietly re-pointing every column of a gate nobody touched.

The cost is contained: the refit path already writes the artifact, so it gains a field, and the
residency registry already stores each adapter's name beside its `u32` id, so the façade's lookup
has somewhere to resolve against. Four things follow from that and are part of the decision
rather than implementation latitude:

- **The name list lives inside the versioned artifact, and inside its content hash.** Editing the
  list is therefore a new version, not a mutation of the current one, which is what makes a gate
  pinnable at all (ADR-095 decision 3). A name list stored beside the artifact would reintroduce
  the same unverifiable pairing one level up.
- **The refusal prints the artifact's list and the resident list side by side.** An operator's
  next question after "refused" is always "which adapter moved", and a refusal that does not
  answer it sends them to read two sources by hand.
- **Mismatch is refused in both directions**: a resident adapter absent from the artifact's list,
  and an artifact name absent from residency. They are the same unverifiable pairing seen from
  opposite sides, and refusing only one direction leaves the other silent. Differences of order
  between the two lists are resolved by name, never by position — position is the thing this
  amendment exists to stop trusting.
- **A duplicate name on either side is refused** (_added 2026-09-21, found while implementing_).
  Not a fifth case of the same shape: it is not reachable by set difference, because two sides can
  agree _as sets_ and still be unresolvable. Residency permits it — `(name, path)` identity reuses
  a resident id only on an exact match, so one name at a second path is a second resident adapter
  — and the gate's own trained list can repeat a name just as easily. Resolving that ambiguity in
  either direction is precisely the silent wrong-adapter selection this amendment exists to stop,
  so it refuses, with the same side-by-side lists. Its arm: two residents sharing a name must
  refuse even when the artifact's list matches as a set, which is what makes it a different test
  from the two below rather than a restatement of them.
- **The arm that carries it**: two adapter pools differing only in order must route identically,
  and a pool missing one of the artifact's names must refuse. The first fails against any
  implementation that kept a positional path; the second fails against one that treats an absent
  name as a zero column. Note that the two id types do not meet today — the router speaks
  `AdapterId = String` (`mixture.rs:46`) while the serving contract speaks `LoraSelection.id: u32` —
  and naming the trained set is what gives that translation a defined direction instead of an
  implied one.

On the façade's error type, which this decision left open and which needs no new invention: it
returns the existing `ApiError`. The pattern is already in the serving path — `lora_unsupported_backend`
and the `#[cfg(not(...))]` helper `adapter_unsupported_build` build an `ApiError` for a compiled-out
feature, and `lora_list` beside them is this decision's exact shape, two `cfg` blocks inside one
gate-free function. A new error enum would add a type whose only job is to be converted at the same
boundary. `RouterError` renders into the message through `Display`, on the same argument decision 3
used for its variant remap: a move should not change the text a user sees.

**3. `router_update.rs` moves from `lattice-tune` to `lattice-fann`, behind `online-router`.** Its
only tune-specific dependency is tune's error type; everything substantive it imports is already
`lattice_fann::{Network, training::{RlooConfig, RlooTrainer}}` (`router_update.rs:12-15`), and
`lattice-fann`'s `online-router` feature exists precisely to expose `RlooTrainer` and
`DiagonalFisher` (`crates/fann/Cargo.toml:16-17`). `lattice-inference` already depends on
`lattice-fann` under `mixture` (`crates/inference/Cargo.toml:70,123`), so the move makes the update
path reachable from the serving crate with the dependency direction unchanged. `lattice-tune`
re-exports the module so its two existing callers — `examples/prompt_router.rs` and
`tests/router_loop_closure.rs` — keep compiling unmodified, and that is the move's acceptance test.

## Alternatives considered

- **`cfg`-gate the serving call site.** Rejected, and this is the alternative the code invites: it
  is the smallest diff and it makes the serving path's behaviour depend on a build flag that
  nothing at the HTTP surface reports. A deployment with the feature off then answers routing
  requests as though routing were off by design.
- **Add `lattice-inference → lattice-tune`.** Rejected: it inverts a declared dependency direction
  and makes the serving crate carry the training stack.
- **Run the update out of process, over the replay buffer as a file.** Not rejected on the merits —
  it is a reasonable shape for a large training job — but it answers a different question than a
  feedback endpoint with a documented cadence does, and it is available later without any of the
  decisions here being wrong.
- **A second embedding model loaded for routing only.** Rejected: a second copy of a model the
  process already holds, and a second thing to configure, to answer a question the loaded one
  answers.
- **The served model's own prompt hidden state as the gate input.** This is the input a learned
  gate should end up on, and it removes both the second model and the per-request CPU embedding.
  It needs the gate retrained on that representation, so it is a follow-on rather than an
  alternative to decision 1, and it is recorded here so that decision 1 is read as a first wiring
  rather than as an endpoint.

## Consequences

- Decision 3 is a move, not a rewrite, so it is reversible by reverting one commit, and its
  acceptance is mechanical: `lattice-tune`'s existing example and test compile and pass unchanged.
- Decision 2's façade makes "no router configured" a response rather than a link error, which is
  what lets ADR-095's `GET` report the router's state honestly in a build without the feature.
- Decision 1 ties routing availability to the embedding model's presence. That coupling is real and
  is the reason the refusal is explicit: an operator who configures routing without an embedding
  model gets an error naming the missing piece, not a quietly unrouted service.

## Open, and deliberately not decided here

The numeric cost of the per-request embedding pass, which is unmeasured and belongs with the first
end-to-end rather than with this topology.
