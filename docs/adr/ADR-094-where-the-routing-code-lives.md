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
has somewhere to resolve against. Note that the two id types do not meet today — the router speaks
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
